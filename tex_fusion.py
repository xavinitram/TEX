"""Cross-node TEX fusion — splice an ordered chain of TEX programs into ONE.

A linked chain of TEX_Wrangle nodes (image-out -> image-in) is compiled and run
as a single program so only the terminal node cooks (the others never
materialize or cache). Each non-terminal stage's `@OUT` becomes a typed local;
the next stage's chain-input `@binding` reads that local. Every user identifier
(locals, `@` inputs, `$` params, user functions) is namespaced per stage with a
`_s{i}_` prefix, so independently-authored stages that happen to reuse the same
names never collide. The fused program is then compiled through the existing
pipeline (Lexer is skipped — we splice ASTs — but the TypeChecker + optimizer +
interpreter are reused unchanged), and is bit-equivalent to running the stages
sequentially (see benchmarks/fusion_splice_test.py).

The frontend (graphToPrompt) detects the fusable chain and hands the terminal a
serialized payload; the terminal calls compile_fused() instead of compiling its
own single `code`. When no payload is present, nothing here runs and the node
behaves exactly as before.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

from .tex_compiler.lexer import Lexer
from .tex_compiler.parser import Parser
from .tex_compiler.type_checker import TypeChecker, TypeCheckError
from .tex_compiler.optimizer import optimize
from .tex_compiler import ast_nodes as A
from .tex_runtime.stdlib import TEXStdlib
from .tex_runtime.interpreter import _collect_identifiers

# Names the splicer must NEVER prefix — they resolve globally, not per stage.
_BUILTINS = frozenset({
    "ix", "iy", "iw", "ih", "u", "v", "px", "py", "fi", "fn", "PI", "TAU", "E", "ic",
})
_STDLIB = frozenset(TEXStdlib.get_functions().keys())


class FusionError(Exception):
    """Raised when a chain can't be fused (e.g. an upstream node scatter-writes
    @OUT or has multiple outputs). The node surfaces it as a clean error — it
    must NOT silently run the terminal alone, since the upstream nodes were
    already collapsed out of the submitted prompt."""


def _seed_initializer(out_type):
    """A typed zero for the per-stage handoff local. Seeding matters when a stage
    writes @OUT only on some paths (inside an if/loop) or via a channel
    (`@OUT.r =`, which reads the base first) — the local must be defined first."""
    if out_type in ("float", "int"):
        return A.NumberLiteral(value=0.0, is_int=(out_type == "int"))
    if out_type in ("vec2", "vec3", "vec4"):
        n = int(out_type[-1])
        return A.VecConstructor(size=n, args=[A.NumberLiteral(value=0.0) for _ in range(n)])
    raise FusionError(
        f"This fused chain has a node whose @OUT is a {out_type}, which can't be "
        f"passed on to the next node — only image (vec2/vec3/vec4) and mask "
        f"(float) stages can be fused. Break the chain at that node.")


def _is_out_ref(node) -> bool:
    """True if `node` is a wire `@OUT` BindingRef."""
    return (node.__class__ is A.BindingRef and node.kind == "wire"
            and node.name == "OUT")


def _child_nodes(node):
    """Yield the direct AST children of `node` (descending into list fields)."""
    for f in dataclasses.fields(node):
        val = getattr(node, f.name)
        if isinstance(val, A.ASTNode):
            yield val
        elif isinstance(val, list):
            for x in val:
                if isinstance(x, A.ASTNode):
                    yield x


def _out_used_inside(node, enclosing, _inside=False) -> bool:
    """True if a wire `@OUT` appears anywhere inside a node whose type is in
    `enclosing`. Drives both fusion guards:
      - loops (`ForLoop`/`WhileLoop`): reading `@OUT` inside a loop returns the
        pre-loop value, which a plain handoff local wouldn't reproduce;
      - function defs (`FunctionDef`): a non-terminal `@OUT` write is rewritten
        to a top-level handoff local that isn't in scope inside a spliced fn.
    Either way such a stage can't be fused as a non-terminal."""
    if not isinstance(node, A.ASTNode):
        return False
    if _inside and _is_out_ref(node):
        return True
    nxt = _inside or node.__class__ in enclosing
    return any(_out_used_inside(c, enclosing, nxt) for c in _child_nodes(node))


def _out_used_in_loop(node) -> bool:
    return _out_used_inside(node, (A.ForLoop, A.WhileLoop))


def _out_used_in_function(node) -> bool:
    return _out_used_inside(node, (A.FunctionDef,))


def _out_target_binding(target):
    """The underlying @-binding written by an assignment target — handles
    `@OUT`, `@OUT.rgb`, and `@OUT[x,y]`. Returns the BindingRef, or None."""
    cls = target.__class__
    if cls is A.BindingRef:
        return target
    if cls is A.ChannelAccess and target.object.__class__ is A.BindingRef:
        return target.object
    if cls is A.BindingIndexAccess and target.binding.__class__ is A.BindingRef:
        return target.binding
    return None


def _references_out(node) -> bool:
    """True if a wire `@OUT` BindingRef appears anywhere in this subtree."""
    if not isinstance(node, A.ASTNode):
        return False
    if _is_out_ref(node):
        return True
    return any(_references_out(c) for c in _child_nodes(node))


def _seedless_out_index(statements):
    """Index of the first top-level statement that touches `@OUT`, IFF that touch
    is a plain unconditional `@OUT = expr` (bare BindingRef target, op `=`) whose
    RHS does not itself read `@OUT`. In that case the handoff local can be born
    directly from that write (a VarDecl) and the typed-zero seed is a dead store.
    Returns the index, or None if a seed is required."""
    for idx, s in enumerate(statements):
        if not _references_out(s):
            continue
        # First @OUT touch found — eligible only if it's a clean bare write.
        if (s.__class__ is A.Assignment and s.op is None
                and s.target.__class__ is A.BindingRef
                and s.target.kind == "wire" and s.target.name == "OUT"
                and not _references_out(s.value)):
            return idx
        return None
    return None


def _user_prefix(prefix: str) -> str:
    """The sub-namespace for user-authored names within a stage. Kept distinct
    from the bare `prefix` (reserved for compiler-generated handoff locals like
    out_local) so user names and synthetic names can never collide. The one and
    only place the `u_` convention is spelled."""
    return prefix + "u_"


def _transform(node, prefix, user_fns, chain_input, prev_out, out_local, is_terminal):
    """Recursively namespace one stage's AST and wire its boundaries in place.

    Non-terminal stages: `@OUT` reads/writes (including `@OUT.rgb =` and
    `@OUT += `) are redirected to the handoff local `out_local` (declared+seeded
    by compile_fused); the chain-input @binding reads the previous stage's local.
    Every other user identifier is prefixed so stages never collide. Returns the
    (possibly replaced) node."""
    if node is None:
        return None
    cls = node.__class__
    up = _user_prefix(prefix)

    if cls is A.BindingRef:
        if node.kind == "wire" and node.name == "OUT":
            # A non-terminal @OUT READ resolves to the handoff local; the terminal
            # keeps its real @OUT (the chain's actual output).
            return node if is_terminal else A.Identifier(loc=node.loc, name=out_local)
        if chain_input is not None and node.kind == "wire" and node.name == chain_input:
            return A.Identifier(loc=node.loc, name=prev_out)  # chain handoff
        node.name = up + node.name
        return node

    if cls is A.Identifier:
        if node.name not in _BUILTINS:
            node.name = up + node.name
        return node

    if cls is A.Assignment:
        node.value = _transform(node.value, prefix, user_fns, chain_input,
                                prev_out, out_local, is_terminal)
        base = _out_target_binding(node.target)
        if base is not None and base.kind == "wire" and base.name == "OUT" and not is_terminal:
            tgt = node.target
            if tgt.__class__ is A.BindingRef:                       # @OUT = / @OUT += expr
                return A.Assignment(loc=node.loc, op=node.op,
                                    target=A.Identifier(loc=tgt.loc, name=out_local),
                                    value=node.value)
            if tgt.__class__ is A.ChannelAccess:                    # @OUT.rgb = expr
                tgt.object = A.Identifier(loc=tgt.object.loc, name=out_local)
                return node
            # @OUT[x,y] = expr scatters into self.bindings (not env) — not fusable.
            raise FusionError(
                "This fused chain has a node that scatter-writes @OUT[x,y], which "
                "can't be passed on to the next node. Break the chain at that node.")
        node.target = _transform(node.target, prefix, user_fns, chain_input,
                                 prev_out, out_local, is_terminal)
        return node

    # Generic structural recursion over dataclass fields.
    for f in dataclasses.fields(node):
        val = getattr(node, f.name)
        if isinstance(val, A.ASTNode):
            setattr(node, f.name, _transform(val, prefix, user_fns, chain_input,
                                             prev_out, out_local, is_terminal))
        elif isinstance(val, list):
            setattr(node, f.name, [
                _transform(x, prefix, user_fns, chain_input, prev_out, out_local,
                           is_terminal) if isinstance(x, A.ASTNode) else x
                for x in val
            ])

    # Prefix declared names AFTER recursing into initializers/bodies.
    if cls in (A.VarDecl, A.ArrayDecl, A.ParamDecl):
        node.name = up + node.name
    elif cls is A.FunctionDef:
        node.name = up + node.name
        node.params = [(t, up + n) for (t, n) in node.params]
    elif cls is A.FunctionCall and node.name in user_fns:
        node.name = up + node.name
    return node


def _parse(code: str) -> A.Program:
    return Parser(Lexer(code).tokenize(), source=code).parse()


def compile_fused(stages: list[dict], infer_binding_type: Callable[[Any], Any]):
    """Splice + compile an ordered chain of stages into one program.

    stages: ordered upstream -> terminal. Each:
        {"code": str,
         "chain_input": str | None,   # the @binding fed by the previous stage
         "bindings": {name: value}}   # this stage's EXTERNAL inputs + params
                                      # (the chain_input is internal, not listed)

    Returns the same shape as TEXCache.compile_tex plus the merged bindings:
        (program, type_map, referenced, assigned, param_info, used_builtins,
         merged_bindings)
    The caller runs the interpreter on `program` with `merged_bindings`.
    """
    if len(stages) < 2:
        raise FusionError("fusion needs at least two stages")

    fused_stmts: list[A.ASTNode] = []
    merged_bindings: dict[str, Any] = {}
    prev_out: str | None = None
    prev_out_type = None  # TEXType of the previous stage's @OUT
    n = len(stages)

    for i, st in enumerate(stages):
        prefix = f"_s{i}_"
        is_terminal = (i == n - 1)
        chain_in = st.get("chain_input")
        ext = st.get("bindings", {})

        # Per-stage binding types for the standalone validation pass (un-prefixed).
        bt = {name: infer_binding_type(val) for name, val in ext.items()}
        if chain_in is not None:
            if prev_out_type is None:
                raise FusionError(f"stage {i} declares a chain input but has no upstream output")
            bt[chain_in] = prev_out_type

        prog = _parse(st["code"])
        # Standalone type-check: validates the stage (good error attribution) and
        # gives us @OUT's concrete type for the handoff local.
        checker = TypeChecker(binding_types=bt, source=st["code"])
        checker.check(prog)

        out_local = f"_s{i}_out"
        seed = None
        if not is_terminal:
            ot = checker.assigned_bindings.get("OUT")
            if ot is None:
                raise FusionError(
                    f"A fused TEX chain needs every upstream node to write @OUT, but "
                    f"stage {i} doesn't — it has nothing to pass to the next node.")
            extra = [k for k in checker.assigned_bindings if k != "OUT"]
            if extra:
                raise FusionError(
                    f"Upstream stage {i} writes {', '.join('@' + e for e in extra)} "
                    f"besides @OUT; only single-@OUT nodes fuse. Break the chain there.")
            if _out_used_in_loop(prog):
                raise FusionError(
                    f"Upstream stage {i} uses @OUT inside a loop; @OUT has special "
                    f"binding semantics there that fusion can't preserve. Break the "
                    f"chain at that node.")
            if _out_used_in_function(prog):
                raise FusionError(
                    f"Upstream stage {i} uses @OUT inside a user-defined function; "
                    f"the handoff local fusion redirects @OUT to isn't in scope there, "
                    f"so fusion can't preserve it. Break the chain at that node.")
            # If the first top-level @OUT touch is a plain unconditional `@OUT =`,
            # the handoff local is born from that write (lowered to a VarDecl below)
            # and the typed-zero seed would be a dead full-frame allocation. Only
            # the conditional / channel / compound case needs the seed.
            seedless_idx = _seedless_out_index(prog.statements)
            if seedless_idx is None:
                # Declare the handoff local once, at the stage's top level, seeded — so
                # conditional / channel writes to @OUT are always defined.
                seed = A.VarDecl(type_name=ot.value, name=out_local,
                                 initializer=_seed_initializer(ot.value), is_const=False)
        else:
            seedless_idx = None

        user_fns = {s.name for s in prog.statements if isinstance(s, A.FunctionDef)}
        _transform(prog, prefix, user_fns, chain_in, prev_out, out_local, is_terminal)
        if seedless_idx is not None:
            # _transform rewrote `@OUT = expr` to `Assignment(Identifier(out_local) = expr)`;
            # promote that first write to the declaration of the handoff local.
            written = prog.statements[seedless_idx]
            prog.statements[seedless_idx] = A.VarDecl(
                loc=written.loc, type_name=ot.value, name=out_local,
                initializer=written.value, is_const=False)
        if seed is not None:
            fused_stmts.append(seed)
        fused_stmts.extend(prog.statements)

        for name, val in ext.items():
            merged_bindings[_user_prefix(prefix) + name] = val
        prev_out = out_local
        prev_out_type = None if is_terminal else checker.assigned_bindings.get("OUT")

    fused = A.Program(statements=fused_stmts)
    binding_types = {name: infer_binding_type(val) for name, val in merged_bindings.items()}

    checker = TypeChecker(binding_types=binding_types, source="<fused chain>")
    try:
        type_map = checker.check(fused)
        fused = optimize(fused, type_map)
        # Re-type-check the optimized AST from scratch (mirrors TEXCache.compile_tex):
        # rebuilds a complete id()-keyed type_map covering optimizer-synthesized nodes.
        type_map = TypeChecker(binding_types=binding_types, source="<fused chain>",
                               strict_redeclare=False).check(fused)
    except TypeCheckError as e:
        # A fused chain that doesn't type-check as one program is unfusable —
        # surface it as a clean FusionError (the node's contract) instead of an
        # uncaught crash. The common trigger is an @OUT type hint (m@/img@/s@ )
        # on a non-terminal stage that fusion can't reproduce.
        raise FusionError(
            "This chain can't be fused into a single valid program (often an @OUT "
            "type hint like m@OUT/img@OUT on an upstream node, or mismatched channel "
            f"types across stages). Break the chain at that node. [{e}]") from e
    used_builtins = _collect_identifiers(fused)
    return (fused, type_map, checker.referenced_bindings, checker.assigned_bindings,
            checker.param_declarations, used_builtins, merged_bindings)


def prepare_fused(spec: dict, terminal_code: str, terminal_bindings: dict,
                  infer_binding_type: Callable[[Any], Any]):
    """Assemble + compile a fused chain from a frontend `_tex_chain` payload.

    spec (from graphToPrompt):
        {"stages": [{"code", "image_input", "params": {name: value}}, ...],  # upstream, source-first
         "terminal_image_input": str}    # the terminal's @binding that carries the
                                         # chain SOURCE (and that its code reads as the chain)

    terminal_bindings: the terminal node's own bindings (its $params and any
        extras) PLUS the source image under terminal_image_input; consumed here.

    Returns (program, type_map, referenced, assigned_bindings, param_info,
    used_builtins, merged_bindings) — drop-in for the node's interpreter call with
    output_names = sorted(assigned_bindings).
    """
    stages = spec.get("stages") or []
    if not stages:
        raise FusionError("fused chain payload has no upstream stages")
    # The terminal's chain-link socket carries the SOURCE image (the frontend
    # rewired it) AND is the @binding the terminal's code reads as the chain — one
    # key serves both roles (they are the same socket).
    chain_key = spec.get("terminal_image_input") or spec.get("source_key")
    bindings = dict(terminal_bindings)
    source = bindings.pop(chain_key, None)
    if source is None:
        raise FusionError(f"the fused chain's source image ('{chain_key}') reached "
                          f"the terminal node empty — re-run the upstream nodes.")

    fused_stages: list[dict] = []
    for idx, st in enumerate(stages):
        sb = dict(st.get("params") or {})
        if idx == 0:
            sb[st["image_input"]] = source        # first stage reads the real source image
            chain_in = None
        else:
            chain_in = st["image_input"]          # later upstream stages read the chain
        fused_stages.append({"code": st["code"], "chain_input": chain_in, "bindings": sb})

    fused_stages.append({"code": terminal_code,
                         "chain_input": chain_key,
                         "bindings": bindings})

    return compile_fused(fused_stages, infer_binding_type)
