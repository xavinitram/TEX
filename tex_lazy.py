"""
Lazy input cooking — static analysis of which inputs a TEX program can need.

Given the source and the widget parameter values, `lazy_required_bindings`
returns the set of @/$ names the program can reference at runtime, so
`check_lazy_status` (tex_node.py) can tell ComfyUI to skip cooking the
upstream subgraphs of everything else.

Tiers covered:
  T1  input wired but never referenced             -> not in the set
  T2  referenced only inside statically-dead flow  -> pruned by const-fold
  T3  usage gated on a $param widget value         -> param folded as a
      literal, the dead branch pruned (the sweet spot)
  T4  usage gated on another *input's* value       -> deliberately NOT here;
      wired scalar params get a limited form via check_lazy_status's
      iterative protocol (params cook first, then fold like T3).

What deliberately does NOT sever a dependency (correctness, verified):
  * `@A * 0.0`   — NaN*0 = NaN in IEEE, and the optimizer itself refuses
                   x*0->0 (shape-unsafe, optimizer.py). Only *both-literal*
                   BinOps fold, so @A survives — by construction.
  * `&&` / `||`  — the interpreter evaluates both sides (no short-circuit
                   on tensors); folding only reaches them via literal operands.
  * spatial ifs  — torch.where evaluates both branches; only conditions that
                   fold to a compile-time NumberLiteral are pruned, which is
                   exactly the class the interpreter short-circuits as scalars.

The analysis is syntactic and over-approximating: it may KEEP a binding that
is dead (missed optimisation, never a bug). If it ever wrongly DROPPED one,
the interpreter raises the loud "Input '@X' is not connected" error — never
silent wrong output.

Precision note: folding runs comparisons in Python floats, the runtime in
fp32 tensors. Substituted params are pre-rounded to fp32 to close that gap;
the residual window (a literal-vs-param straddling one fp32 ulp inside a
comparison) fails loud per the above, not silently.

Cache #14: a module-level LRU keyed on (code-hash, folded-param fp32 bits,
egress-profile key — see `_profile_key`).
Distinct key + lifecycle from the 13 existing caches (ARCHITECTURE.md), and
shared by check_lazy_status and execute() so the per-cook cost is a dict hit.
"""
from __future__ import annotations

import struct
from collections import OrderedDict

from .tex_compiler.ast_nodes import (
    ASTNode, BindingRef, NumberLiteral, IfElse, WhileLoop, ForLoop,
    FunctionDef, clone_tree, iter_child_nodes,
)
from .tex_compiler.optimizer import _propagate_literal_locals, _fold_all
from .tex_compiler.types import planes_wires_enabled

# Wire types that can carry a spatial tensor (participate in CF-6 consensus shape
# derivation). STRING/INT/FLOAT/BOOLEAN wires marshal to non-spatial values.
SPATIAL_WIRE_TYPES = frozenset({"IMAGE", "MASK", "LATENT", "*"})
# Wire types R1 accepts as a shape anchor (known [B,H,W,C] tensors post-marshal).
SHAPE_ANCHOR_TYPES = frozenset({"IMAGE", "MASK"})
# Wire types whose cooked value is a foldable scalar (T4-lite candidates).
SCALAR_WIRE_TYPES = frozenset({"INT", "FLOAT", "BOOLEAN"})

_MEMO_MAX = 256
_memo: "OrderedDict[tuple, frozenset | None]" = OrderedDict()

_PARSE_MEMO_MAX = 64
#: (source, profile key) -> the UNFOLDED `parse_and_split` AST. Handed out only as
#: `clone_tree` copies.
_parse_memo: "OrderedDict[tuple, object]" = OrderedDict()

#: COUNTS-44: `_memo` is keyed on `(code hash, param_values, profile)`, and `param_values`
#: moves on every widget scrub by construction (T3 genuinely depends on the values) — so
#: `_memo` itself misses every tick a scrub drives, same as it always has. But hashing the
#: SAME source on every one of those misses is the identical pattern PERF-44 already solved
#: for `tex_roi._walk` with `tex_cache.code_digest` — the value-INDEPENDENT SHA-256 memo
#: `TEXCache.fingerprint` keeps for itself (`_FINGERPRINT_MEMO`), shared rather than a second
#: one hand-rolled here. Imported where used (matches `tex_roi._walk`'s own call site) rather
#: than at module scope, so this module carries no load-time opinion about `tex_cache`.


# ── PERF-8: the egress-profile component of EVERY analysis memo key ───────────
#
# `tex_cache.parse_and_split` is a function of three things: the source, the binding types
# (every analysis caller here passes `{}`, and a caller passing a real map would have to key
# it), and ONE piece of process-global state — `planes_wires_enabled()`. While plane wires are
# on, `p@beauty.diffuse` stays a single dotted `BindingRef`, a plane read; while they are off
# the splitback puts it back to `ChannelAccess(@beauty, "diffuse")`, the swizzle it meant
# before planes existed. So one source has TWO parses, and every memo holding a value derived
# from one of them — the two parse memos, and the three answer memos above them (`_memo` here,
# `tex_roi._walk_memo` and `tex_roi._region_dep_memo`) — has to carry the flag or it serves one
# profile's answer under the other. The lexer mode is NOT in the key because the seam
# hard-codes it (`dotted_bindings=True`); if that ever becomes a caller's choice it belongs
# here beside the flag.
#
# ONE name for all five key sites, deliberately, and an ALIAS rather than a wrapper so the key
# costs exactly the one unavoidable read: a `planes_wires_enabled()` spelled per site is a
# per-site opportunity to forget one, which is how the blindness reached five keys at once.
#
# Under either shipped host this is a CONSTANT — the flag is set once before the first cook
# (the set-once posture documented in `tex_compiler/types.py`) and is deliberately absent from
# the program fingerprint — so every key gains a constant element, no hit rate moves and no
# answer moves. What it buys is a host that changes profile mid-process (an engine toggling a
# planes capability, an editor previewing both), which today is served a previous profile's
# reach for as long as the entry survives.
_profile_key = planes_wires_enabled


def _pristine_parse(code: str, memo: "OrderedDict", cap: int):
    """The `parse_and_split(code, {})` AST for `code` under the CURRENT egress profile, lexed
    and parsed at most once per `(source, profile)`.

    The body behind BOTH `_pristine_program`s — this module's and `tex_roi`'s — so the two
    memos cannot drift in key, in eviction order or in the "never hand out the entry" rule.
    They keep their own `memo`/`cap` so `clear_lazy_memo` and `clear_roi_memo` stay independent
    test hooks and the two caps stay separately tunable.

    The entry is the PRISTINE parse and is never handed out directly: every caller mutates
    (the `$param` substitution and the optimizer's fold both rewrite in place), so each takes
    its own `clone_tree` copy. Bounded LRU, oldest evicted first. A parse ERROR is not cached —
    the callers catch it and answer "keep everything".
    """
    key = (code, _profile_key())
    hit = memo.get(key)
    if hit is None:
        from .tex_cache import parse_and_split
        hit = parse_and_split(code, {})
        memo[key] = hit
        while len(memo) > cap:
            memo.popitem(last=False)
    else:
        memo.move_to_end(key)
    return hit


def _pristine_program(code: str):
    """The unfolded front-end AST for `code`, lexed and parsed AT MOST ONCE per source.

    `_memo` above is keyed on the parameter VALUES because the analysis genuinely depends on
    them (that is the whole of tier T3), so a moving widget misses it on every evaluation and
    paid a full `Lexer.tokenize` + `Parser.parse` for a program whose SOURCE had not changed.
    The parse is a function of the source alone, so it is cached here and each caller folds
    its own `clone_tree` copy. The substitution and the optimizer's fold both rewrite in
    place, which is why the entry is never handed out directly: a caller that mutated it
    would poison every later evaluation of that source with the previous value's literals.

    Same bounded-LRU discipline as `_memo`, with a smaller cap because an entry is a whole
    AST rather than a frozenset; `clear_lazy_memo` drops it with the rest. A parse ERROR is
    not cached — `lazy_required_bindings` catches it and answers None (keep everything).

    PERF-8: "per source" is per `(source, profile)` — the mechanics, and why the profile is in
    the key, live in `_pristine_parse` / `_profile_key`.

    DATA-6, invariant 11: through the one front end (`tex_cache.parse_and_split`) with NO
    binding types — see `lazy_required_bindings` for why every dotted read must split back
    to its base wire."""
    return _pristine_parse(code, _parse_memo, _PARSE_MEMO_MAX)


def _fp32(v: float) -> float:
    """Round a Python float to fp32 so folded comparisons match the runtime's
    fp32 tensors (numpy-free per the torch-only invariant)."""
    return struct.unpack("f", struct.pack("f", float(v)))[0]


def _substitute_params(node: ASTNode, subs: dict[str, NumberLiteral]) -> None:
    """Replace $param BindingRefs with their literal values, in place.

    Walks every child field; a BindingRef child whose kind is "param" and
    whose name is in *subs* is swapped for a fresh NumberLiteral.
    """
    for field_name in node.__dataclass_fields__:
        val = getattr(node, field_name, None)
        if isinstance(val, BindingRef):
            if val.kind == "param" and val.name in subs:
                lit = subs[val.name]
                setattr(node, field_name, NumberLiteral(
                    loc=val.loc, value=lit.value, is_int=lit.is_int))
        elif isinstance(val, ASTNode):
            _substitute_params(val, subs)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, BindingRef):
                    if item.kind == "param" and item.name in subs:
                        lit = subs[item.name]
                        val[i] = NumberLiteral(
                            loc=item.loc, value=lit.value, is_int=lit.is_int)
                elif isinstance(item, ASTNode):
                    _substitute_params(item, subs)


def _prune_static_flow(stmts: list) -> list:
    """Remove statically-dead control flow after const-folding.

    * IfElse with a NumberLiteral condition -> splice the taken body
      (`value > 0.5`, matching the interpreter's scalar short-circuit and the
      optimizer's ternary fold).
    * WhileLoop with a literal-false condition -> dropped.
    * ForLoop is left intact (zero-trip literal ranges are rare; keeping the
      body only over-approximates).
    Recurses into surviving bodies and FunctionDefs.
    """
    out: list = []
    for stmt in stmts:
        cls = stmt.__class__
        if cls is IfElse and isinstance(stmt.condition, NumberLiteral):
            taken = stmt.then_body if stmt.condition.value > 0.5 else stmt.else_body
            out.extend(_prune_static_flow(taken or []))
            continue
        if cls is WhileLoop and isinstance(stmt.condition, NumberLiteral) \
                and not stmt.condition.value > 0.5:
            continue
        if cls is IfElse:
            stmt.then_body = _prune_static_flow(stmt.then_body or [])
            stmt.else_body = _prune_static_flow(stmt.else_body or [])
        elif cls in (WhileLoop, ForLoop, FunctionDef):
            stmt.body = _prune_static_flow(stmt.body or [])
        out.append(stmt)
    return out


def _collect_binding_refs(stmts: list) -> frozenset[str]:
    """All @/$ names syntactically reachable in the surviving statements."""
    names: set[str] = set()
    stack = list(stmts)
    while stack:
        node = stack.pop()
        if node.__class__ is BindingRef:
            names.add(node.name)
        stack.extend(iter_child_nodes(node))
    return frozenset(names)


def _param_key(param_values: dict) -> tuple:
    """Stable memo key component: fp32 bit patterns for numerics, raw for the
    rest (bools fold as 0/1; strings never fold but distinguish programs)."""
    items = []
    for name in sorted(param_values):
        v = param_values[name]
        if isinstance(v, bool):
            items.append((name, "b", int(v)))
        elif isinstance(v, int):
            # type-tag int distinctly from float (doc 32): struct.pack("f", 1) and
            # struct.pack("f", 1.0) are identical bytes, so an un-tagged int 1 and float
            # 1.0 would share a memo entry. Harmless today (both fold to the same fp32
            # literal), but the tag keeps the key honest if the analysis ever distinguishes.
            items.append((name, "i", v))
        elif isinstance(v, float):
            items.append((name, "f", struct.pack("f", float(v))))
        else:
            items.append((name, "s", str(v)))
    return tuple(items)


def lazy_required_bindings(code: str,
                           param_values: dict | None = None,
                           ) -> frozenset[str] | None:
    """The set of @/$ names the program can reference at runtime given these
    widget values, or None when the analysis fails (caller keeps everything).
    Never raises. Over-approximates: extra names are a missed skip, never a bug.

    Only float/int/bool params fold (T3); strings and vectors stay symbolic,
    so conditions gated on them conservatively keep both branches.
    """
    param_values = param_values or {}
    # PERF-8: the profile is in the key because the ANSWER moves with it — `p@beauty.diffuse`
    # is one plane name under the engine profile and the wire `beauty` under ComfyUI's.
    from .tex_cache import code_digest
    key = (code_digest(code), _param_key(param_values), _profile_key())
    hit = _memo.get(key)
    if hit is not None or key in _memo:
        _memo.move_to_end(key)
        return hit
    try:
        # A private copy of the source's ONE parse: the analysis mutates its AST, and no
        # type info is needed (references are syntactic). The copy is what makes reusing
        # the parse safe — see `_pristine_program`.
        # DATA-6, invariant 11: through the one front end with NO binding types, so every
        # dotted `@image.r` is split back to a swizzle of its BASE wire before the set is
        # collected. A program whose only reads of a wire are dotted must still REQUEST that
        # wire — a set holding `image.r` where the host looks up `image` would under-
        # approximate (the wire skipped, the cook loud-failing E6003), and R1 only masks it
        # when the wire happens to be the first spatial one. Splitting everything is the
        # over-approximating side: a kept plane read would resolve to its base wire, whole.
        program = clone_tree(_pristine_program(code))
        subs = {
            name: NumberLiteral(value=_fp32(v), is_int=isinstance(v, (bool, int)))
            for name, v in param_values.items()
            if isinstance(v, (bool, int, float))
        }
        stmts = program.statements
        if subs:
            for stmt in stmts:
                _substitute_params(stmt, subs)
        # fold -> propagate -> fold: the first fold turns substituted-param
        # initializers into literals (`float k = $n * 2.0;`), propagation
        # spreads them, the second fold collapses the now-literal conditions.
        stmts = _fold_all(stmts)
        stmts = _propagate_literal_locals(stmts)
        stmts = _fold_all(stmts)
        stmts = _prune_static_flow(stmts)
        result: frozenset | None = _collect_binding_refs(stmts)
    except Exception:
        result = None
    _memo[key] = result
    if len(_memo) > _MEMO_MAX:
        _memo.popitem(last=False)
    return result


def clear_lazy_memo() -> None:
    """Test hook. Drops the answer memo and the source-keyed parse memo behind it. The
    source-digest memo behind THAT is `tex_cache`'s own (`code_digest`, shared with
    `tex_roi._walk`) and is cleared there, not here — mirrors `tex_roi.clear_roi_memo`,
    which draws the same line."""
    _memo.clear()
    _parse_memo.clear()
