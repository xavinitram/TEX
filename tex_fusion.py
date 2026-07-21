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
import heapq
import logging
import re
from collections import OrderedDict
from typing import Any, Callable

# FUS-1: cap how large a single fused region can grow. A 50-node region compiles into
# ONE torch.compile program (longer trace, bigger capture) and materializes full-res
# intermediates (CACHE-6 hasn't landed) — Blender and torch.compile both cap their fuse
# units for the same reason. Past the cap the region is left unfused (the linear pass
# still fuses sub-chains); safe, just not one giant program.
_MAX_FUSED_REGION_STAGES = 16

logger = logging.getLogger("TEX.fusion")

from .tex_compiler.lexer import Lexer
from .tex_compiler.parser import Parser
from .tex_compiler.type_checker import TypeChecker, TypeCheckError
# ENG-4: the shared per-phase tuple + translator (compile_fused is the SECOND compile
# implementation — it validates each stage directly, not through the cache). TEXMultiError is
# also caught explicitly at the SPLICE below, where the policy is FusionError, not a compile error.
from .tex_compiler.diagnostics import (
    TEXCompileError, TEXMultiError, raw_compile_errors, compile_error_from)
from .tex_compiler import ast_nodes as A
# STR-8: optimize + _collect_identifiers are no longer imported here — the shared
# post-parse pipeline now lives in TEXCache.compile_ast (called by compile_fused).

# Names the splicer must NEVER prefix — they resolve globally, not per stage.
# ENG-7 adds frame/fps/time: the host's playhead is one value for the whole fused
# program, so a per-stage `_s0_u_frame` would be an undefined identifier at cook time.
_BUILTINS = frozenset({
    "ix", "iy", "iw", "ih", "u", "v", "px", "py", "fi", "fn", "PI", "TAU", "E", "ic",
    "frame", "fps", "time",
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


_USER_PREFIX_RE = re.compile(r"^_s\d+_u_")


def strip_user_prefix(name: str) -> str:
    """Recover a user's ORIGINAL binding name from its fused stage-renamed form
    (`_s0_u_amt` → `amt`); an unfused / already-bare name passes through unchanged. The one and
    only place the INVERSE of the `_s{i}_` + `_user_prefix` rename is spelled — so a consumer that
    needs the original name (the E6003 message; the DATA-1 tag match) calls this instead of
    hard-coding the pattern out of band (and Runtime-layer marshalling need not know it at all)."""
    return _USER_PREFIX_RE.sub("", name)


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
        # move_to_end can race a concurrent popitem: FUS-1's /detect_regions route
        # runs compile_fused on an aiohttp executor thread while a cook thread also
        # compiles a fused terminal, and a lost key would raise KeyError that fails
        # the whole prompt. GIL-atomic per op, but get-then-move_to_end is not one
        # op — tolerate the eviction (the value is still valid; only its LRU
        # recency is lost). ENG-9 will give the memo a real lock.
        try:
            _FUSED_MEMO.move_to_end(memo_key)
        except KeyError:
            pass
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

        # ENG-4: a per-stage compile error (a typo in ONE fused node's code) must surface as the
        # public TEXCompileError carrying the real diagnostic. The fused cook path never runs
        # through `_compile_or_raise`, so without this a raw Lexer/Parse/TypeCheckError skips the
        # node's `except TEXCompileError` and lands in the bug-report catch-all — losing the clean
        # TEX_DIAG the frontend parses. (A genuine FUSION problem — a missing @OUT, a bad wire — is
        # a FusionError, raised elsewhere; this is only the per-stage standalone compile.)
        try:
            prog = _parse(st["code"])
            # Standalone type-check: validates the stage (good error attribution) and
            # gives each exported binding's concrete type for its handoff local.
            checker = TypeChecker(binding_types=bt, source=st["code"])
            checker.check(prog)
        except raw_compile_errors() as e:
            raise compile_error_from(e, st["code"]) from e

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
        # C-fix: a TERMINAL that read-modify-writes one of its wired chain inputs
        # (`@b += …`, `@b = @b*2`, `@b.rgb = …`) both READS and ASSIGNS @b, so @b is
        # in `passthrough` (an output) — `_transform` keeps it as bare `@b` and never
        # resolves the READ to the upstream handoff (passthrough is matched before
        # wire_map). @b is not a merged binding either, so the read raises E6021 at
        # cook (a broken prompt invisible to preflight). Seed the output from the
        # upstream handoff BEFORE the terminal's statements — inserted POST-transform
        # so the handoff local isn't re-prefixed. Bit-exact to the unfused
        # read-modify-write; a write-first terminal just overwrites the seed.
        if is_terminal:
            rmw_seeds = [
                A.Assignment(op="=", target=A.BindingRef(kind="wire", name=name),
                             value=A.Identifier(name=wire_map[name]))
                for name in sorted(wire_map) if name in passthrough
            ]
            prog.statements[:0] = rmw_seeds
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
    # chain. Respect the engine's output ceiling: taps are optional conveniences, so
    # DROP the excess (with a log) rather than letting sorted(assigned) overflow the
    # host's output slots and fail the cook. Function-local import — tex_engine imports
    # tex_fusion at module level, so the edge only exists in this direction (ENG-1).
    from .tex_engine import MAX_OUTPUTS as _MAX_FUSED_OUTPUTS
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
    except (TypeCheckError, TEXMultiError) as e:
        # A fused chain that doesn't type-check as one program is unfusable — surface it as a
        # clean FusionError (the node's contract) instead of an uncaught crash. TEXMultiError
        # (the >=2-error case) is a SIBLING of TypeCheckError, not a subclass, so it must be
        # caught explicitly — else a spliced program with two type errors escapes to the node's
        # bug-report catch-all (ENG-4 removed the node's old `except TEXMultiError` net). Both
        # mean "the spliced program isn't one valid program"; the message is actionable and the
        # raw diagnostics would only point into the never-shown `<fused chain>` source anyway.
        # The common trigger is an @OUT type hint (m@/img@/s@) on a non-terminal stage.
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
    except (FusionError, TEXCompileError) as e:
        # Render the real message, not the catch-all's "preflight error:" wrapper. These are the
        # only two compile_fused can raise: a structural/splice problem (FusionError) or a
        # PER-STAGE compile failure (TEXCompileError since ENG-4 — the commonest unfusable cause;
        # note `stage_of_error` stays None for it, since a rendered diagnostic carries no "stage N").
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


# ── SCHED-1: the GraphSpec ───────────────────────────────────────────────────
#
# The `_tex_chain` payload started as a private handshake between TEX's own JS and its
# own node. It is now the INTERFACE any host uses to drive fusion: emit a GraphSpec,
# call detect_fusable_regions / prepare_fused, and DAG fusion works — the detector is
# pure Python (FUS-1) precisely so no host reimplements the legality rules. That makes
# the payload shape a contract, so it gets a version.
#
# GRAPHSPEC_SCHEMA is the version THIS build emits and accepts. It is a compatibility
# gate, not decoration: a host that pins an old TEX, or a stale workflow JSON carrying
# an embedded payload, must fail with a legible message instead of mis-splicing a graph.
#
# Bump it when the shape changes in a way an older reader would MISREAD. Adding an
# optional key that older readers ignore harmlessly does not qualify.
#
# Schema history:
#   1 — v0.15 linear chains; v0.21 added the optional DAG payload (`dag`, `chain_inputs`,
#       `source_stage`, `source_binding`, `terminal_chain_inputs`). Both read as schema 1:
#       the DAG keys are additive and a linear spec is byte-identical to v0.15's.
#   2 — v0.29 (FUS-1b) added `source_injections` (one external producer feeding >1 member).
#       A schema-1 reader would inject the source into only ONE member and mis-splice the
#       rest, so a multi-injection spec is stamped schema 2 and refused by older builds.
#       Single-injection specs stay schema 1 / byte-identical (no source_injections emitted).
GRAPHSPEC_SCHEMA = 2

# Absent `schema` means a pre-v0.22 emitter, which by definition emitted schema 1 — the
# shape is unchanged, so accepting it is correct, not lenient. Every workflow saved
# before v0.22 relies on this.
_GRAPHSPEC_DEFAULT_SCHEMA = 1


def _check_graphspec_schema(spec: dict) -> None:
    """SCHED-1: reject a GraphSpec this build cannot read. Never guesses — a spec from a
    NEWER TEX may have re-meant a key we still recognise, and silently splicing it would
    produce wrong pixels from a graph the user thinks is supported."""
    got = spec.get("schema", _GRAPHSPEC_DEFAULT_SCHEMA)
    if not isinstance(got, int) or got < 1:
        raise FusionError(
            f"this fused chain's GraphSpec has an invalid schema ({got!r}). The payload "
            f"is malformed — turn off TEX Fusion in settings, or re-create the link.")
    if got > GRAPHSPEC_SCHEMA:
        raise FusionError(
            f"this fused chain was built by a newer TEX (GraphSpec schema {got}; this "
            f"build reads {GRAPHSPEC_SCHEMA}). Update TEX, or turn off 'TEX Fusion: "
            f"Compile linked TEX nodes together' in settings to run the nodes unfused.")


def prepare_fused(spec: dict, terminal_code: str, terminal_bindings: dict,
                  infer_binding_type: Callable[[Any], Any]):
    """Assemble + compile a fused chain from a GraphSpec payload (SCHED-1).

    spec — the shape a host emits (`_tex_chain` on the ComfyUI node):

        {"schema": 1,                    # SCHED-1; optional, defaults to 1 (pre-v0.22)
         "stages": [                     # UPSTREAM stages, source-first; terminal excluded
             {"code": str,               #   the stage's TEX source
              "image_input": str,        #   linear only: the @binding carrying the chain
              "params": {name: value}},  #   the stage's own widget values
             ...],
         "terminal_image_input": str,    # the terminal's @binding carrying the chain SOURCE

         # --- optional DAG payload (FUS-1); absent => a linear chain ---
         "dag": True,
         "source_injections": [[stage, binding], ...],   # schema 2 (FUS-1b): one entry per
                                         #   external edge; OVERRIDES the two scalars below
         "source_stage": int,            # schema 1: which stage the external edge feeds
         "source_binding": str,          # ...and under which @binding
         "terminal_chain_inputs": {binding: [src_stage, out]},
         # ...and per-stage: "chain_inputs": {binding: [src_stage, out]}
        }

    terminal_bindings: the terminal node's own bindings (its $params and any
        extras) PLUS the source image under terminal_image_input; consumed here.

    Returns (program, type_map, referenced, assigned_bindings, param_info,
    used_builtins, merged_bindings) — drop-in for the engine's interpreter call with
    output_names = sorted(assigned_bindings).
    """
    fused_stages = _stages_from_spec(spec, terminal_code, terminal_bindings)
    return compile_fused(fused_stages, infer_binding_type)


def _stages_from_spec(spec: dict, terminal_code: str, terminal_bindings: dict) -> list[dict]:
    """Assemble the ordered stage list (source-first) from a GraphSpec.
    Shared by prepare_fused and fused_fingerprint so both see the same key."""
    _check_graphspec_schema(spec)
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

    # FUS-1: DAG payload (fan-out / diamonds). Each upstream stage carries its own
    # `chain_inputs` (which of its @-bindings read which earlier stage's @OUT), the
    # terminal carries `terminal_chain_inputs`, and the single external source is
    # injected into the stage/binding that reads it. The `terminal_image_input`
    # socket is only the SOURCE TRANSPORT here (the terminal's real reads are its
    # chain_inputs), so it is popped but not re-attached. Linear payloads (no `dag`)
    # take the unchanged path below and stay byte-identical (same memo/disk keys).
    if spec.get("dag"):
        # FUS-1b: one or more injection points, all fed the SAME transported source. A
        # single-injection (schema 1) spec carries no `source_injections` and falls back to
        # the scalar `source_stage`/`source_binding` — byte-identical to v0.21.
        raw_inj = spec.get("source_injections")
        if raw_inj:
            inj_points = [(int(s), b) for (s, b) in raw_inj]
        else:
            inj_points = [(spec.get("source_stage", 0), spec.get("source_binding"))]
        fused_stages = []
        for idx, st in enumerate(stages):
            sb = dict(st.get("params") or {})
            for s_stage, s_binding in inj_points:
                if idx == s_stage and s_binding:
                    sb[s_binding] = source
            entry = {"code": st["code"], "bindings": sb}
            ci = st.get("chain_inputs")
            if ci:
                entry["chain_inputs"] = _listify_chain_inputs(ci)
            fused_stages.append(entry)
        term = {"code": terminal_code, "bindings": bindings}
        # C14 — defensive only, UNREACHABLE via detect_fusable_regions: the detector
        # rejects any region whose external source feeds the terminal directly
        # (`dst == terminal`, the spatial-fragility guard), so no injection points past the
        # upstream stages here. Kept so a non-detector caller that hand-builds a "source
        # feeds the terminal" spec still injects the source rather than silently dropping it.
        for s_stage, s_binding in inj_points:
            if s_stage == len(stages) and s_binding:
                term["bindings"][s_binding] = source
        tci = spec.get("terminal_chain_inputs")
        if tci:
            term["chain_inputs"] = _listify_chain_inputs(tci)
        else:
            term["chain_input"] = chain_key       # terminal reads the source directly
        fused_stages.append(term)
        return fused_stages

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


# ── CACHE-6: fusion ↔ caching reconciliation (stage-list surgery) ─────────────
# A fused chain has no interior cut-points, so twiddling the LAST node's param recooks all N
# stages every tick — fusion makes interactivity WORSE exactly where it matters. CACHE-6's two
# levers cut the chain at a stage boundary k: cache the stage-(k-1) handoff (a stage-boundary
# TAP, keyed by the upstream SUB-CHAIN fingerprint below), then compile + cook only stages k..N
# reading that handoff (SUFFIX SPLICING) while a downstream knob is hot; recook the whole fused
# program on idle. The boundary MUST be the exact fp32 handoff or the FUS-3 oracle breaks — so
# the engine gates taps to fp32. v1 covers LINEAR chains (the interactive grade→blur→vignette
# case); a chain_inputs DAG is not suffix-split (a documented follow-up — the whole-chain recook
# stays correct, just not incremental).

def is_linear_stage_list(stages: list[dict]) -> bool:
    """True if `stages` is a plain source-first linear chain (no DAG `chain_inputs` on any
    stage) — the shape a suffix split rebases trivially. A DAG needs positional chain_inputs
    rebasing (deferred), so CACHE-6 v1 recooks it whole."""
    return bool(stages) and not any(st.get("chain_inputs") for st in stages)


def prefix_fingerprint(stages: list[dict], k: int, infer_binding_type: Callable) -> str:
    """The 'upstream sub-chain fingerprint' a stage-boundary tap keys on: the value-independent
    `_fused_fp` of the prefix `stages[:k]` (the stages producing the boundary). Derived, never
    stored (ENG-5: fingerprints are unstable across TEX versions). Value-independent — the
    param VALUES the boundary also depends on enter the tap's lineage key separately."""
    return _fused_fp(_fused_memo_key(stages[:k], infer_binding_type))


def suffix_stage_list(stages: list[dict], k: int, boundary_value) -> list[dict]:
    """Build the suffix `stages[k:]` as a standalone chain: old stage k, which read the chain on
    its `chain_input` binding, is rewritten to read the cached BOUNDARY as an external source
    (chain_input=None, boundary injected into its bindings) — exactly how stage 0 of a full chain
    reads the real source. The remaining stages are unchanged (they read within the suffix). The
    result cooks through the same compile_fused / single-program path as any chain. LINEAR only."""
    if not (1 <= k < len(stages)):
        raise FusionError(f"suffix cut-point {k} out of range for {len(stages)} stages")
    head = stages[k]
    chain_binding = head.get("chain_input")
    if chain_binding is None:
        raise FusionError("suffix head stage has no chain_input to rebind to the boundary")
    first = {"code": head["code"], "chain_input": None,
             "bindings": {**(head.get("bindings") or {}), chain_binding: boundary_value}}
    return [first] + [dict(st) for st in stages[k + 1:]]


# ── FUS-1: DAG-region detection (the host-agnostic fusion authority) ──────────
# compile_fused already consumes arbitrary DAG specs (chain_inputs / exports / tap,
# Q-3), but the production producer (_stages_from_spec) only ever emitted a LINEAR
# chain, and the frontend detector broke on any fan-out. detect_fusable_regions is
# the single pure-Python detector — any host (the ComfyUI frontend via the
# /detect_regions route, a standalone host, PORT-5) drives fusion by handing it the
# graph topology, so fusion legality is decided ONCE and can't drift per host.
#
# v0.21 scope: single-terminal regions over @OUT (slot-0) handoffs, fed by exactly ONE
# external image edge. A node folds iff EVERY consumer is an in-region TEX node
# (generalizes the linear "sole TEX consumer" rule).
#
# What that does and does NOT cover — the distinction is INTERNAL vs EXTERNAL fan-out:
#   covered:  a region member fans out to other members and they rejoin at the
#             terminal (`src -> A -> [B, C] -> D`). The branch point is inside R, so
#             the region still has one external edge: A's source.
#   NOT yet:  one EXTERNAL producer feeding two members (`Load -> [blur, sharpen] ->
#             merge`) — a common compositor shape. That's two external edges, so it
#             needs multi-injection (a source spec per edge, splicer + transport work),
#             not the single-source splice here. Left unfused: always safe, just not
#             collapsed. Same for a genuine two-source merge.
# Multi-output (exports) / preview-tap detection stays a later cut; the compile_fused
# machinery for them is untouched.

def _listify_chain_inputs(ci) -> dict:
    """chain_inputs with each [src_stage, out] edge as a JSON-safe list (the detector
    builds them as lists already, but region plans / specs pass through tuples too).
    One spelling, shared by the region assemblers and the DAG `_stages_from_spec`."""
    return {b: list(v) for b, v in (ci or {}).items()}


def _index_edges(edges):
    """(out_by_src, in_by_dst): src_id -> [edge], dst_id -> [edge]. Each edge is a
    dict {from, from_slot, to, to_binding}."""
    out_by_src, in_by_dst = {}, {}
    for e in edges:
        out_by_src.setdefault(e["from"], []).append(e)
        in_by_dst.setdefault(e["to"], []).append(e)
    return out_by_src, in_by_dst


def detect_fusable_regions(nodes: dict, edges: list) -> list:
    """`nodes`: {id: {"code_wired": bool}} for the non-muted TEX nodes ONLY.
    `edges`: image-carrying handoff edges {from, from_slot, to, to_binding} — the
    caller passes edges whose payload is a fusable IMAGE/MASK handoff; non-TEX
    producers/consumers appear as ids NOT in `nodes`. Returns a list of region
    plans (see _grow_region). Pure; never raises on well-formed input."""
    tex = set(nodes)
    out_by_src, in_by_dst = _index_edges(edges)

    def _consumers_internalizable(nid):
        """A non-terminal candidate: every out-edge goes to a TEX node via slot 0.
        An empty out-edge list => a sink (terminal), not a foldable member."""
        outs = out_by_src.get(nid, [])
        return bool(outs) and all(e["to"] in tex and e["from_slot"] == 0 for e in outs)

    regions = []
    for term in nodes:
        if _consumers_internalizable(term):
            continue  # internal to some region — not a terminal
        if nodes[term].get("code_wired"):
            continue  # C5: a wired-code terminal has no static code to fuse (it would
                      # serialize as "" and preflight an empty program — a false pass)
        plan = _grow_region(term, nodes, tex, out_by_src, in_by_dst)
        if plan is not None:
            regions.append(plan)
    return regions


def _grow_region(terminal, nodes, tex, out_by_src, in_by_dst):
    """Fixpoint-grow R(terminal) upstream, validate the single-external-source
    constraint, and emit the plan (or None if not fusably shaped in v0.21 scope):
        {"terminal", "order" (topo, source-first, terminal last), "delete",
         "source": {"origin","origin_slot","stage","binding"},
         "stages": [{"id","chain_inputs": {binding: [src_stage_idx, "OUT"]}}]}"""
    R = {terminal}
    changed = True
    while changed:
        changed = False
        frontier = set()
        for nid in R:
            for e in in_by_dst.get(nid, []):
                if e["from"] in tex and e["from"] not in R:
                    frontier.add(e["from"])
        for u in frontier:
            # A wired `code` or `$param` input can't fold: the serializer captures
            # only widget values, so folding would sever the dynamic wired value and
            # bake the stale widget (② silent-wrong). A zero-image-input generator
            # can't fold either: spatial-less, it would adopt the fused source's
            # resolution, diverging from its standalone cook (③ invariant #2).
            if nodes[u].get("code_wired") or nodes[u].get("param_wired"):
                continue
            if not in_by_dst.get(u):
                continue
            outs = out_by_src.get(u, [])
            if outs and all(e["to"] in R and e["from_slot"] == 0 for e in outs):
                R.add(u)
                changed = True
                if len(R) > _MAX_FUSED_REGION_STAGES:
                    # Early-out: stop growing the moment the region exceeds the cap
                    # (it will be rejected anyway). Otherwise a huge upstream cone
                    # feeding one sink is walked to fixpoint — O(cone × depth) —
                    # before the post-loop cap check rejects it, making detection
                    # pathological on a dense graph (every terminal re-walks its
                    # whole cone). Bounds each grow to O(cap × degree).
                    return None

    if len(R) < 2:
        return None

    # Region-EXTERNAL image inputs (edges into R from outside R).
    external = [(nid, e["to_binding"], e["from"], e["from_slot"], e.get("from_type"))
                for nid in R for e in in_by_dst.get(nid, []) if e["from"] not in R]
    if not external:
        return None    # a pure-generator region with no external source — leave unfused
    # FUS-1b: ONE external producer may feed >1 region member (`Load -> [blur, sharpen] ->
    # merge` is two external EDGES from a single producer). Group by producer socket: a
    # genuine two-PRODUCER merge still can't fuse (the splice carries one source), but a
    # single producer fanning in becomes a MULTI-INJECTION source — one injection per edge,
    # all fed the same transported tensor. v0.21 accepted only the len(external)==1 case.
    producers = {(o, s) for (_, _, o, s, _) in external}
    if len(producers) != 1:
        return None    # 0 or >1 distinct producers — unfused (unchanged 2-source behaviour)
    if any(dst == terminal for (dst, _, _, _, _) in external):
        # An external edge into the TERMINAL directly is the spatial-fragility case v0.21
        # rejected (a generator fused into a spatial program) — leave the region unfused.
        return None
    src_origin, src_slot = next(iter(producers))
    src_type = external[0][4]

    order = _topo_order(R, in_by_dst)
    if order is None or order[-1] != terminal:
        return None  # cycle, or the terminal is not R's unique sink
    idx = {nid: i for i, nid in enumerate(order)}

    stages = []
    for nid in order:
        chain_inputs = {e["to_binding"]: [idx[e["from"]], "OUT"]
                        for e in in_by_dst.get(nid, []) if e["from"] in R}
        stages.append({"id": nid, "chain_inputs": chain_inputs})

    # One injection per external edge, in a deterministic order (stage, then binding) so the
    # plan is stable. `injections` is the SOLE in-memory representation — the scalar
    # stage/binding pair exists only at the WIRE boundary (region_to_collapse_plan), where the
    # schema-1 compat story actually lives.
    injections = sorted(({"stage": idx[dst], "binding": binding}
                         for (dst, binding, _, _, _) in external),
                        key=lambda j: (j["stage"], j["binding"]))
    return {
        "terminal": terminal,
        "order": list(order),
        "delete": [nid for nid in order if nid != terminal],
        "source": {"origin": src_origin, "origin_slot": src_slot, "type": src_type,
                   "injections": injections},
        "stages": stages,
    }


def _topo_order(R, in_by_dst):
    """Kahn topo sort of the subgraph induced by R (producer -> consumer),
    source-first, id-repr tie-break for determinism. Returns None on a cycle."""
    succs = {nid: [] for nid in R}
    indeg = {nid: 0 for nid in R}
    for nid in R:
        preds = {e["from"] for e in in_by_dst.get(nid, []) if e["from"] in R}
        indeg[nid] = len(preds)
        for p in preds:
            succs[p].append(nid)
    # str-keyed min-heap: same deterministic order as a sorted ready-list, but
    # O((V+E) log V) instead of the re-sort-every-pop O(V^2 log V). ids are unique so
    # the str key never ties (the node itself is never compared).
    heap = [(str(n), n) for n in R if indeg[n] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, n = heapq.heappop(heap)
        order.append(n)
        for m in succs[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(heap, (str(m), m))
    return order if len(order) == len(R) else None


def region_to_stages(region: dict, node_code: dict, node_params: dict) -> list:
    """Turn a detect_fusable_regions plan into a compile_fused `stages` list (with
    DAG `chain_inputs`). `node_code`/`node_params` map node id -> code / {param:
    value}. The single external source is left OUT of the stage bindings here — the
    caller injects its value at region['source'] (stage index, binding). Terminal is
    the last stage (its assigned bindings are the real outputs)."""
    stages = []
    for st in region["stages"]:
        nid = st["id"]
        entry = {"code": node_code[nid], "bindings": dict(node_params.get(nid, {}))}
        if st["chain_inputs"]:
            entry["chain_inputs"] = _listify_chain_inputs(st["chain_inputs"])
        stages.append(entry)
    return stages


def region_to_collapse_plan(region: dict, node_code: dict, node_params: dict) -> dict:
    """Build a host-applyable collapse plan from a region: the DAG `_tex_chain`
    payload (everything except `terminal_image_input` — the physical socket the host
    picks to transport the source), plus which node survives (terminal), which are
    deleted, and where the source comes from. The host rewires the chosen terminal
    socket to `source_origin[:2]`, sets payload['terminal_image_input'] to that
    socket's binding name, attaches the payload to the terminal, and deletes the
    upstream nodes. Splitting it this way keeps DETECTION (drift-prone legality) in
    Python and only socket wiring in the host."""
    upstream = region["stages"][:-1]
    term_stage = region["stages"][-1]
    injections = region["source"]["injections"]
    multi = len(injections) > 1
    payload = {
        # FUS-1b: a SINGLE-injection region stays schema 1 / byte-identical (source_stage +
        # source_binding). A MULTI-injection region adds source_injections and bumps to
        # schema 2 — a schema-1 reader would inject the source into only one member and
        # mis-splice the rest, so it MUST be refused there, not silently degraded.
        "schema": GRAPHSPEC_SCHEMA if multi else 1,   # SCHED-1 (1 = the single-injection shape)
        "dag": True,
        "stages": [{"code": node_code[st["id"]],
                    "params": dict(node_params.get(st["id"], {}) or {}),
                    "chain_inputs": _listify_chain_inputs(st["chain_inputs"])}
                   for st in upstream],
        "terminal_chain_inputs": _listify_chain_inputs(term_stage["chain_inputs"]),
        "source_stage": injections[0]["stage"],   # index into `stages` (upstream)
        "source_binding": injections[0]["binding"],
    }
    if multi:
        payload["source_injections"] = [[j["stage"], j["binding"]] for j in injections]
    return {
        "terminal": region["terminal"],
        "delete": list(region["delete"]),
        "source_origin": [region["source"]["origin"], region["source"]["origin_slot"]],
        "payload": payload,
    }


def _region_is_linear(region: dict) -> bool:
    """A single-terminal region is non-linear (fan-out/diamond) iff it has more
    internal handoff edges than a path — a reconverging fan-out adds a fan-in at the
    terminal. A purely linear region is the JS linear-collapse pass's job (C17).

    FUS-1b: a multi-injection region (one external producer fanning into >1 member) is
    ALSO non-linear even though its INTERNAL handoffs look path-like — the fan-out lives on
    the external edges, not internal ones (`Load -> [blur, sharpen] -> merge` has one
    internal fan-in at merge, so the count alone reads linear). The linear pass can't
    collapse it, so it must reach the region path, not be skipped here."""
    if len(region.get("source", {}).get("injections") or []) > 1:
        return False
    stages = region.get("stages") or []
    internal = sum(len(st.get("chain_inputs") or {}) for st in stages)
    return internal <= len(stages) - 1


def _preflight_samples(source_type):
    """The tensor(s) a region's external source can plausibly BE at cook, given the
    producer's declared socket type. _region_compiles must splice+compile against all
    of them, since any one is what the host may hand the fused terminal.

    Families, per tex_marshalling.infer_binding_type:
      MASK   -> [B,H,W]                     -> FLOAT
      LATENT -> unwrapped to [B,H,W,C]      -> VEC4 when C==4 (SD/SDXL), but FLOAT when
                                               C==16 (SD3/Flux/Wan-class) — see below
      IMAGE  -> [B,H,W,C], C in (3,4)       -> VEC3 / VEC4
    Unknown/absent (an older JS transport, or an exotic socket type such as a litegraph
    wildcard) falls back to IMAGE, which is the pre-v0.21 behaviour.

    Why MASK matters even though scalars broadcast: TEX promotes a float through both
    operators and swizzles (`@in.rgb` on a float is a legal vec3), so most programs
    type-check either way. What an IMAGE-shaped guess actually costs is two things, and
    neither is a wrong pixel:
      * lost fusion — `@in.a` is legal on a float but not on a vec3, so the (3,4)
        preflight REJECTED mask regions that would have cooked fine;
      * a misleading error — vector-only calls (`length`, `normalize`) pass under
        VEC3/VEC4 and reject a FLOAT, so a mask-fed one false-PASSED, fused, and then
        died at cook as "couldn't fuse this chain, turn off TEX Fusion". Unfused it
        fails too (the program is simply invalid for a float), but with the honest
        `length() needs a vector, but argument 1 is float` at the offending node.

    Known coverage limits (stated, not silent). The socket type names a family, not a
    channel count, so two cases are deliberately preflighted at their DOMINANT shape and
    can still false-PASS a vector-only call:
      * a 1-channel (FLOAT) or 2-channel (VEC2) IMAGE — exotic;
      * a LATENT with C != 4 — NOT exotic: SD3/Flux/Wan-class latents are 16-channel and
        infer as FLOAT, yet are preflighted here as VEC4.
    Requiring both shapes to compile would fix the false-PASS by dropping every vector
    region off a *normal* image or 4-channel latent (`length(@lat)` doesn't type-check
    against a float) — a real loss to buy a better error message for a program that is
    already invalid either way. So each family stays at its dominant shape. The residue
    is exactly {vector-only call} x {non-dominant channel count}: never a wrong pixel,
    just "couldn't fuse this chain" where the true type error would read better."""
    import torch
    t = (source_type or "IMAGE").upper()
    if t == "MASK":
        return [torch.zeros(1, 8, 8)]
    if t == "LATENT":
        return [torch.zeros(1, 8, 8, 4)]
    return [torch.zeros(1, 8, 8, 3), torch.zeros(1, 8, 8, 4)]


def _region_compiles(region: dict, node_code: dict, node_params: dict) -> bool:
    """PREFLIGHT: does this region actually splice+compile? detect_fusable_regions
    only checks TOPOLOGY — it can't see the per-stage guards compile_fused enforces
    (@OUT inside a loop/user-function, a scatter-write @OUT[x,y], an undeclared extra
    output). A region that passes topology but trips one of those would, once its
    upstream nodes are deleted from the prompt, hard-fail the cook with a FusionError.
    Compiling it here drops it cleanly (host runs it unfused) AND warms the memo.

    C3: the real source SHAPE is unknown at detect time (the host sends no tensor
    shapes), and shape-sensitive stages (e.g. `vec4(@src, 1.0)` needs a vec3) compile
    against one binding type but not another — a fixed vec3 preflight would false-PASS
    a region whose source is a vec4 (or a mask) and then fails to cook. Preflight every
    shape the source's declared socket type can actually take (_preflight_samples); a
    region that isn't valid across all of them is left unfused (safe, minor coverage
    loss). Never raises."""
    try:
        from .tex_marshalling import infer_binding_type
        src = region["source"]
        injections = src["injections"]
        for sample in _preflight_samples(src.get("type")):
            stages = region_to_stages(region, node_code, node_params)
            for inj in injections:          # FUS-1b: the one source feeds every injection point
                stages[inj["stage"]]["bindings"][inj["binding"]] = sample
            compile_fused(stages, infer_binding_type)
        return True
    except Exception:
        return False


def detect_region_plans(graph: dict) -> list:
    """Route entry (FUS-1): given a serialized TEX subgraph
        {nodes: [{id, code, params, code_wired, param_wired}],
         edges: [{from, from_slot, to, to_binding}]}
    return host-applyable collapse plans (region_to_collapse_plan). Detection +
    preflight live here so every host performs the SAME fusion and legality can't
    drift per host. Never raises — a malformed graph yields [] (host runs unfused)."""
    try:
        gnodes = graph.get("nodes", [])
        node_code = {n["id"]: n.get("code", "") for n in gnodes}
        node_params = {n["id"]: (n.get("params") or {}) for n in gnodes}
        nodes = {n["id"]: {"code_wired": bool(n.get("code_wired")),
                           "param_wired": bool(n.get("param_wired"))} for n in gnodes}
        plans = []
        for reg in detect_fusable_regions(nodes, graph.get("edges", [])):
            if _region_is_linear(reg):
                continue  # C17: the JS linear-collapse pass owns linear chains
            if _region_compiles(reg, node_code, node_params):
                plans.append(region_to_collapse_plan(reg, node_code, node_params))
            # else: trips a compile_fused guard -> leave the region unfused (safe)
        return plans
    except Exception:
        logger.warning("[TEX] region detection failed; nothing fused this queue.",
                       exc_info=True)
        return []


# ── FUS-2: fused-chain lazy composition (the mechanism) ──────────────────────
# Walks a fused chain's stages terminal-first, folding each stage's params, to find
# which stages (and their external inputs) the fused program can actually reference.
# Over-approximate by construction (lazy_required_bindings already is; a stage is
# kept whenever any needed downstream reads its handoff), so a wrongly-dropped input
# would fail LOUD as E6003, never silent.
#
# WIRING DEFERRED (see tests/test_v021_phase1.py): in v0.21's single-external-source
# fusion scope the source is always the R1 shape anchor (never prunable) and every
# input traces to it, so hooking this into check_lazy_status/execute's E6003 gate
# would prune nothing yet add E6003-breakage risk. When multi-source regions land,
# ONE memoized result of this analysis must feed BOTH consumers (invariant #11's
# dual-consumer rule), exactly as tex_lazy does for the single-node case.

def fused_required_bindings(stages: list, source_stage: int | None = None,
                            source_binding: str | None = None):
    """`stages`: the SAME list compile_fused / region_to_stages produce — each
    {code, bindings, chain_inputs: {binding: [src_idx, "OUT"]}}. `source_stage` /
    `source_binding` name which stage reads the external chain source (from a region
    plan's `source`). Returns {source_needed, needed_stages, needed_names} or None
    (any stage's lazy analysis failed -> caller cooks everything). The foldable
    scalars come from each stage's `bindings` (the real contract — widget/wired param
    values live there), NOT a fabricated `params` key. Pure over-approximation."""
    from .tex_lazy import lazy_required_bindings
    n = len(stages)
    if n == 0:
        return None
    ref = []
    for st in stages:
        params = {k: v for k, v in (st.get("bindings") or {}).items()
                  if isinstance(v, (bool, int, float))}
        got = lazy_required_bindings(st.get("code", ""), params)
        if got is None:
            return None
        ref.append(got)
    needed = {n - 1}                        # the terminal always cooks
    changed = True
    while changed:
        changed = False
        for i in list(needed):
            for binding, edge in (stages[i].get("chain_inputs") or {}).items():
                src_idx = edge[0]
                if binding in ref[i] and src_idx not in needed:
                    needed.add(src_idx)
                    changed = True
    needed_names: set = set()
    for i in needed:
        needed_names |= set(ref[i])
    source_needed = (source_stage is not None and source_binding is not None
                     and 0 <= source_stage < n and source_stage in needed
                     and source_binding in ref[source_stage])
    return {"source_needed": source_needed, "needed_stages": needed,
            "needed_names": needed_names}
