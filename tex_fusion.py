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
import hashlib
import logging
from collections import OrderedDict
from typing import Any, Callable

logger = logging.getLogger("TEX.fusion")

from .tex_compiler.lexer import Lexer
from .tex_compiler.parser import Parser
from .tex_compiler.type_checker import TypeChecker, TypeCheckError
from .tex_compiler import ast_nodes as A
# STR-8: optimize + _collect_identifiers are no longer imported here — the shared
# post-parse pipeline now lives in TEXCache.compile_ast (called by compile_fused).

# Names the splicer must NEVER prefix — they resolve globally, not per stage.
_BUILTINS = frozenset({
    "ix", "iy", "iw", "ih", "u", "v", "px", "py", "fi", "fn", "PI", "TAU", "E", "ic",
})

# Memory LRU for compiled fused chains: chain key -> the 6-tuple compile_fused
# produces (program, type_map, referenced, assigned, param_info, used_builtins).
# The frontend rebuilds the _tex_chain payload on every queue, so without this
# memo every execution of a fused terminal re-parses and re-type-checks the
# whole chain. merged_bindings is NEVER cached — binding VALUES are rebuilt per
# execution (caching them would alias tensors and freeze param updates).
_FUSED_MEMO: "OrderedDict[tuple, tuple]" = OrderedDict()
_FUSED_MEMO_MAX = 32


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
    return any(_out_used_inside(c, enclosing, nxt) for c in A.iter_child_nodes(node))


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
    return any(_references_out(c) for c in A.iter_child_nodes(node))


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


def _tag_stage(node, stage: int) -> None:
    """Set loc.stage = stage on a node and all its descendants (Q-4)."""
    loc = getattr(node, "loc", None)
    if loc is not None and getattr(loc, "stage", None) is None:
        loc.stage = stage
    for ch in A.iter_child_nodes(node):
        _tag_stage(ch, stage)


def _user_prefix(prefix: str) -> str:
    """The sub-namespace for user-authored names within a stage. Kept distinct
    from the bare `prefix` (reserved for compiler-generated handoff locals like
    out_local) so user names and synthetic names can never collide. The one and
    only place the `u_` convention is spelled."""
    return prefix + "u_"


# ── CT-1: fused-chain disk persistence ────────────────────────────────
# The in-memory _FUSED_MEMO is lost on restart, forcing a full ~2.5 ms cold
# splice+compile per chain. Reuse TEXCache's disk tier (which stores the
# optimized AST and re-type-checks on load, respecting the id()-keyed type_map
# contract) keyed by a hash of the value-independent memo_key.

def _fused_memo_key(stages: list[dict], infer_binding_type: Callable) -> tuple:
    """The value-independent compile key for a stage list: per-stage (code,
    chain topology, sorted (name, binding-type)) in order. Q-3: the topology
    covers legacy `chain_input`, DAG `chain_inputs`, multi-@OUT `exports`, and
    `tap` — structurally different chains must key apart. prev_out types need no
    entry — they follow from earlier stages' (code, types)."""
    def _topology(st):
        ci = st.get("chain_inputs")
        ci_key = (tuple(sorted((b, tuple(src)) for b, src in ci.items()))
                  if ci is not None else st.get("chain_input"))
        return (ci_key,
                tuple(sorted(st.get("exports") or ())),
                bool(st.get("tap")))
    return tuple(
        (st["code"], _topology(st),
         tuple(sorted((n, infer_binding_type(v).value)
                      for n, v in (st.get("bindings") or {}).items())))
        for st in stages
    )


def _fused_fp(memo_key: tuple) -> str:
    """Stable filename-safe fingerprint over the fusion memo_key (tuples of
    str / None / sorted (name, type) tuples — repr is stable and canonical)."""
    return "fused_" + hashlib.sha256(repr(memo_key).encode()).hexdigest()[:24]


def _binding_types_from_memo_key(memo_key: tuple) -> dict:
    """Reconstruct the merged (prefixed) external-binding types from the memo_key
    — the same dict compile_fused builds from merged_bindings — needed to
    re-type-check the fused AST on disk load, without the binding VALUES."""
    from .tex_compiler.types import TEXType
    bt: dict[str, Any] = {}
    for i, (_code, _chain_in, sorted_bt) in enumerate(memo_key):
        pfx = _user_prefix(f"_s{i}_")
        for name, tv in sorted_bt:
            bt[pfx + name] = TEXType(tv)
    return bt


def _load_fused_from_disk(memo_key: tuple):
    """Load a persisted fused compile result (the 6-tuple), or None."""
    try:
        from .tex_cache import get_cache
        bt = _binding_types_from_memo_key(memo_key)
        return get_cache()._load_from_disk(_fused_fp(memo_key), bt)
    except Exception:
        return None


def _save_fused_to_disk(memo_key: tuple, fused_program, binding_types) -> None:
    try:
        from .tex_cache import get_cache
        get_cache()._save_to_disk(_fused_fp(memo_key), fused_program, binding_types)
    except Exception:
        pass  # best-effort; a miss just recompiles next restart


def _transform(node, prefix, user_fns, wire_map, redirect_map, passthrough):
    """Recursively namespace one stage's AST and wire its boundaries in place.

    `wire_map` (Q-3 DAG): {binding_name: upstream_handoff_local} — a wired input
    read resolves to the producing stage's handoff local (any earlier stage, not
    just i-1). `redirect_map` (Q-3 multi-@OUT): {binding_name: this_stage_local}
    for every output this stage exports downstream — `@OUT` and each extra
    exported `@binding` read/write (incl. `.rgb =`, `+=`) redirect to their
    handoff locals (declared+seeded by compile_fused). `passthrough`: binding
    names that are the chain's REAL outputs (the terminal's `@OUT` and any extra
    output bindings) — kept unprefixed. Every other user identifier is prefixed
    so stages never collide. Returns the (possibly replaced) node."""
    if node is None:
        return None
    cls = node.__class__
    up = _user_prefix(prefix)

    if cls is A.BindingRef:
        if node.kind == "wire":
            if node.name in passthrough:     # a real chain output — leave as @name
                return node
            local = redirect_map.get(node.name)
            if local is not None:            # reading back this stage's own output
                return A.Identifier(loc=node.loc, name=local)
            local = wire_map.get(node.name)
            if local is not None:            # reading an upstream handoff (DAG edge)
                return A.Identifier(loc=node.loc, name=local)
        node.name = up + node.name
        return node

    if cls is A.Identifier:
        if node.name not in _BUILTINS:
            node.name = up + node.name
        return node

    if cls is A.Assignment:
        node.value = _transform(node.value, prefix, user_fns, wire_map,
                                redirect_map, passthrough)
        base = _out_target_binding(node.target)
        if base is not None and base.kind == "wire" and base.name in redirect_map:
            local = redirect_map[base.name]
            tgt = node.target
            if tgt.__class__ is A.BindingRef:                       # @X = / @X += expr
                return A.Assignment(loc=node.loc, op=node.op,
                                    target=A.Identifier(loc=tgt.loc, name=local),
                                    value=node.value)
            if tgt.__class__ is A.ChannelAccess:                    # @X.rgb = expr
                tgt.object = A.Identifier(loc=tgt.object.loc, name=local)
                return node
            # @X[x,y] = expr scatters into self.bindings (not env) — not fusable.
            raise FusionError(
                "This fused chain has a node that scatter-writes @OUT[x,y], which "
                "can't be passed on to the next node. Break the chain at that node.")
        node.target = _transform(node.target, prefix, user_fns, wire_map,
                                 redirect_map, passthrough)
        return node

    # Generic structural recursion over dataclass fields.
    for f in dataclasses.fields(node):
        val = getattr(node, f.name)
        if isinstance(val, A.ASTNode):
            setattr(node, f.name, _transform(val, prefix, user_fns, wire_map,
                                             redirect_map, passthrough))
        elif isinstance(val, list):
            setattr(node, f.name, [
                _transform(x, prefix, user_fns, wire_map, redirect_map,
                           passthrough) if isinstance(x, A.ASTNode) else x
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

    # Memo key: exactly what compilation depends on — per-stage code, chain
    # input name, and STRUCTURAL binding types (infer_binding_type is value-
    # independent), in stage order. Binding values are deliberately absent:
    # a param tweak hits the memo, an RGB→RGBA swap correctly misses it.
    memo_key = _fused_memo_key(stages, infer_binding_type)
    def _merged_bindings():
        return {
            _user_prefix(f"_s{i}_") + name: val
            for i, st in enumerate(stages)
            for name, val in (st.get("bindings") or {}).items()
        }

    cached = _FUSED_MEMO.get(memo_key)
    if cached is not None:
        _FUSED_MEMO.move_to_end(memo_key)
        return (*cached, _merged_bindings())

    # CT-1: disk tier — a persisted result survives process restart, skipping the
    # full splice + double-typecheck + optimize below.
    disk = _load_fused_from_disk(memo_key)
    if disk is not None:
        _FUSED_MEMO[memo_key] = disk
        while len(_FUSED_MEMO) > _FUSED_MEMO_MAX:
            _FUSED_MEMO.popitem(last=False)
        return (*disk, _merged_bindings())

    fused_stmts: list[A.ASTNode] = []
    merged_bindings: dict[str, Any] = {}
    n = len(stages)
    # Q-3 DAG: (handoff_local, type) produced by each (stage, output). Keyed by
    # (stage_idx, out_name); `_s{i}_out` for @OUT (back-compat), `_s{i}_{name}`
    # for an extra export.
    produced: dict[tuple, tuple[str, Any]] = {}
    tap_exports: list[tuple[int, str]] = []  # (stage_idx, handoff_local) for @_tap_s{i}

    for i, st in enumerate(stages):
        prefix = f"_s{i}_"
        is_terminal = (i == n - 1)
        ext = st.get("bindings", {})

        # Resolve wired inputs. Legacy `chain_input: str` == a single edge from the
        # previous stage's @OUT; Q-3 `chain_inputs: {binding: [src_stage, out]}`
        # generalizes to arbitrary upstream edges (DAG) and named outputs (multi-@OUT).
        chain_inputs = st.get("chain_inputs")
        if chain_inputs is None:
            ci = st.get("chain_input")
            chain_inputs = {ci: [i - 1, "OUT"]} if ci is not None else {}

        # Per-stage binding types for the standalone validation pass (un-prefixed).
        bt = {name: infer_binding_type(val) for name, val in ext.items()}
        wire_map: dict[str, str] = {}
        for binding, (src_stage, src_out) in chain_inputs.items():
            src = produced.get((src_stage, src_out))
            if src is None:
                raise FusionError(
                    f"stage {i} wires @{binding} from stage {src_stage}'s "
                    f"@{src_out}, which isn't produced upstream")
            wire_map[binding], bt[binding] = src

        prog = _parse(st["code"])
        # Standalone type-check: validates the stage (good error attribution) and
        # gives each exported binding's concrete type for its handoff local.
        checker = TypeChecker(binding_types=bt, source=st["code"])
        checker.check(prog)

        # Which of this stage's assigned bindings are exported downstream.
        # Non-terminal: @OUT always (the primary handoff) + any declared extras
        # (Q-3 multi-@OUT). Terminal: nothing is redirected (real outputs stay).
        redirect_map: dict[str, str] = {}
        seeds: list[A.VarDecl] = []
        seedless_idx = None
        ot = checker.assigned_bindings.get("OUT")
        if not is_terminal:
            if ot is None:
                raise FusionError(
                    f"A fused TEX chain needs every upstream node to write @OUT, but "
                    f"stage {i} doesn't — it has nothing to pass to the next node.")
            declared_exports = list(st.get("exports") or [])
            unexpected = [k for k in checker.assigned_bindings
                          if k != "OUT" and k not in declared_exports]
            if unexpected:
                raise FusionError(
                    f"Upstream stage {i} writes {', '.join('@' + e for e in unexpected)} "
                    f"besides @OUT; only single-@OUT (or declared multi-output) nodes "
                    f"fuse. Break the chain there.")
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
            # @OUT handoff (with the seedless optimization for the plain `@OUT =` case).
            out_local = f"_s{i}_out"
            redirect_map["OUT"] = out_local
            produced[(i, "OUT")] = (out_local, ot)
            seedless_idx = _seedless_out_index(prog.statements)
            if seedless_idx is None:
                seeds.append(A.VarDecl(type_name=ot.value, name=out_local,
                                       initializer=_seed_initializer(ot.value), is_const=False))
            # Extra exported outputs (Q-3 multi-@OUT): always seeded handoff locals.
            for name in declared_exports:
                et = checker.assigned_bindings.get(name)
                if et is None:
                    raise FusionError(
                        f"stage {i} declares export @{name} but never writes it")
                loc = f"_s{i}_{name}"
                redirect_map[name] = loc
                produced[(i, name)] = (loc, et)
                seeds.append(A.VarDecl(type_name=et.value, name=loc,
                                       initializer=_seed_initializer(et.value), is_const=False))
            if st.get("tap"):
                tap_exports.append((i, out_local))

        # The terminal's assigned bindings are the chain's real outputs — keep
        # them unprefixed. Non-terminal outputs all route through redirect_map.
        passthrough = set(checker.assigned_bindings) if is_terminal else set()
        user_fns = {s.name for s in prog.statements if isinstance(s, A.FunctionDef)}
        _transform(prog, prefix, user_fns, wire_map, redirect_map, passthrough)
        if seedless_idx is not None:
            # _transform rewrote `@OUT = expr` to `Assignment(Identifier(out_local) = expr)`;
            # promote that first write to the declaration of the handoff local.
            written = prog.statements[seedless_idx]
            prog.statements[seedless_idx] = A.VarDecl(
                loc=written.loc, type_name=ot.value, name=redirect_map["OUT"],
                initializer=written.value, is_const=False)
        fused_stmts.extend(seeds)
        # Q-4: tag every node in this stage with its stage index so a fused-chain
        # runtime error can be attributed to the originating linked node.
        for s in prog.statements:
            _tag_stage(s, i)
        fused_stmts.extend(prog.statements)

        for name, val in ext.items():
            merged_bindings[_user_prefix(prefix) + name] = val

    # Q-3(c): observed-intermediate taps — expose a marked upstream handoff as a
    # terminal output @_tap_s{i} so a Preview can read it without breaking the
    # chain. Respect MAX_OUTPUTS (8, mirrors tex_node.TEXWrangleNode.MAX_OUTPUTS):
    # taps are optional conveniences, so DROP the excess (with a log) rather than
    # letting sorted(assigned) overflow the node's output slots and fail the cook.
    _MAX_FUSED_OUTPUTS = 8
    # Count outputs already assigned by the spliced stages (recurses into nested
    # if/for bodies, unlike a flat scan) — its binding-names half is exactly the
    # set of names that become node output slots.
    _, _existing_outputs = A.collect_assigned_vars(fused_stmts)
    _budget = _MAX_FUSED_OUTPUTS - len(_existing_outputs)
    _dropped = 0
    for i, out_local in tap_exports:
        if _budget <= 0:
            _dropped += 1
            continue
        fused_stmts.append(A.Assignment(
            op="=", target=A.BindingRef(kind="wire", name=f"_tap_s{i}"),
            value=A.Identifier(name=out_local)))
        _budget -= 1
    if _dropped:
        logger.warning("[TEX] fused chain has more preview taps than free output "
                       "slots; dropped %d tap(s) beyond MAX_OUTPUTS=%d.",
                       _dropped, _MAX_FUSED_OUTPUTS)

    fused = A.Program(statements=fused_stmts)
    binding_types = {name: infer_binding_type(val) for name, val in merged_bindings.items()}

    # STR-8: the post-parse pipeline (type-check → optimize → re-type-check → collect)
    # is the SAME as the normal path's; run it through the shared TEXCache.compile_ast
    # (fusion just supplies a spliced AST). Fusion keeps its own error translation so
    # tex_cache never has to import FusionError (which would create an import cycle).
    try:
        from .tex_cache import get_cache
        fused, type_map, refs, asg, params, used_builtins = get_cache().compile_ast(
            fused, binding_types, source="<fused chain>")
    except TypeCheckError as e:
        # A fused chain that doesn't type-check as one program is unfusable —
        # surface it as a clean FusionError (the node's contract) instead of an
        # uncaught crash. The common trigger is an @OUT type hint (m@/img@/s@ )
        # on a non-terminal stage that fusion can't reproduce.
        raise FusionError(
            "This chain can't be fused into a single valid program (often an @OUT "
            "type hint like m@OUT/img@OUT on an upstream node, or mismatched channel "
            f"types across stages). Break the chain at that node. [{e}]") from e
    result = (fused, type_map, refs, asg, params, used_builtins)
    # Cache only on success (a failing chain must re-raise on every run) and
    # only the compile artifacts — never the binding values.
    _FUSED_MEMO[memo_key] = result
    while len(_FUSED_MEMO) > _FUSED_MEMO_MAX:
        _FUSED_MEMO.popitem(last=False)
    _save_fused_to_disk(memo_key, fused, binding_types)  # CT-1 persist
    return (*result, merged_bindings)


def _stage_index_from_error(msg: str):
    """Best-effort: extract the offending stage index from a FusionError message
    (they render 'stage {i}'). Returns int or None."""
    import re
    m = re.search(r"stage (\d+)", msg)
    return int(m.group(1)) if m else None


def _count_fused_ops(program) -> int:
    """Cheap tensor-op estimate for the HUD (BinOp + FunctionCall nodes)."""
    n = 0
    stack = list(program.statements)
    while stack:
        node = stack.pop()
        cls = node.__class__
        if cls is A.BinOp or cls is A.FunctionCall:
            n += 1
        stack.extend(A.iter_child_nodes(node))
    return n


def chain_preflight(stages: list[dict],
                    infer_binding_type: Callable[[Any], Any]) -> dict:
    """Q-5: validate a drawn chain's fusability BEFORE queue time so the bubble
    can go red with the reason + offending stage immediately, instead of failing
    the whole queued prompt. Runs the real compile_fused (placeholder binding
    values are fine — infer_binding_type is value-shape-based) in try/except.

    Returns {ok, error, stage_of_error, stats}. Never raises."""
    try:
        prog, tm, refs, asg, params, used, merged = compile_fused(
            stages, infer_binding_type)
        return {
            "ok": True, "error": None, "stage_of_error": None,
            "stats": {
                "stages": len(stages),
                "statements": len(prog.statements),
                "tensor_ops": _count_fused_ops(prog),
                "outputs": sorted(asg.keys()),
            },
        }
    except FusionError as e:
        return {"ok": False, "error": str(e),
                "stage_of_error": _stage_index_from_error(str(e)), "stats": None}
    except TypeCheckError as e:
        return {"ok": False, "error": str(e),
                "stage_of_error": _stage_index_from_error(str(e)), "stats": None}
    except Exception as e:  # never let preflight hard-fail the UI
        return {"ok": False, "error": f"preflight error: {e}",
                "stage_of_error": None, "stats": None}


def preflight_from_spec(spec: dict, terminal_code: str,
                        infer_binding_type: Callable[[Any], Any]) -> dict:
    """Q-5 endpoint helper: build placeholder stages from a `_tex_chain` spec
    (1×8×8×3 zero image per wired input) and preflight them."""
    import torch
    placeholder = torch.zeros(1, 8, 8, 3)
    # Q-5 fix: seed a placeholder terminal binding for the chain-source key BEFORE
    # assembling. `_stages_from_spec` pops that key as the chain source and raises
    # if it is missing — passing {} here made EVERY chain preflight as not-fusable
    # (permanent false-red bubble). The per-stage placeholder loop below then fills
    # each stage's own wired image inputs.
    chain_key = spec.get("terminal_image_input")
    terminal_bindings = {chain_key: placeholder} if chain_key else {}
    stages = _stages_from_spec(spec, terminal_code, terminal_bindings)
    for st in stages:
        binds = st.get("bindings") or {}
        # Fill any wired image inputs / missing params with harmless placeholders.
        st["bindings"] = {k: (placeholder if _looks_image(v) else v)
                          for k, v in binds.items()}
    return chain_preflight(stages, infer_binding_type)


def _looks_image(v) -> bool:
    try:
        import torch
        return isinstance(v, torch.Tensor) and v.dim() >= 3
    except Exception:
        return False


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
    fused_stages = _stages_from_spec(spec, terminal_code, terminal_bindings)
    return compile_fused(fused_stages, infer_binding_type)


def _stages_from_spec(spec: dict, terminal_code: str, terminal_bindings: dict) -> list[dict]:
    """Assemble the ordered stage list (source-first) from a `_tex_chain` spec.
    Shared by prepare_fused and fused_fingerprint so both see the same key."""
    stages = spec.get("stages") or []
    if not stages:
        raise FusionError("fused chain payload has no upstream stages")
    # The terminal's chain-link socket carries the SOURCE image (the frontend
    # rewired it) AND is the @binding the terminal's code reads as the chain — one
    # key serves both roles (they are the same socket).
    chain_key = spec.get("terminal_image_input")
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
    return fused_stages


def fused_fingerprint(spec: dict, terminal_code: str, terminal_bindings: dict,
                      infer_binding_type: Callable) -> str | None:
    """Q-1: the value-independent fingerprint of a fused chain (same key the
    memo/disk cache use), so the fused program can be a CUDA-graph capture unit.
    Returns None if the spec can't be assembled."""
    try:
        stages = _stages_from_spec(spec, terminal_code, dict(terminal_bindings))
        return _fused_fp(_fused_memo_key(stages, infer_binding_type))
    except Exception:
        return None
