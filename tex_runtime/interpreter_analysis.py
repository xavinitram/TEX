"""Interpreter static-program analysis — SPLIT-47 (v0.47.0, TRK-210).

Split mechanically out of `interpreter.py` (the STR-7/SPLIT-I pattern: every body below is
byte-identical to the code it replaced there — AGENTS.md §"Trades to REFUSE", mechanical
moves only, never an "improvement" mid-move). This module owns the AST-walking accessors
that answer two static questions about a `Program`, independent of any cook: which
wire-binding (`@A`) names it mentions anywhere (`_collect_binding_reads` and its cached
form), which of those sit at a registered NON-SPATIAL argument position (COLOR-1's
`_non_spatial_names_cached`), and which BUILTIN identifier names it references
(`_collect_identifiers`, used for lazy builtin-env construction) or Identifier/BindingRef
names an expression subtree touches (`_collect_expr_names`, UC-3's uniform-range guard).

`_consensus_extent` — the single owner of the cook-grid rule (CF-6) — stays in
`interpreter.py` itself, not here, even though it is this module's biggest caller: its exact
text is a mutation-check anchor (`tests/mutation_check.py`) pinned to `tex_runtime/interpreter.py`,
and AGENTS.md's module-size register already documents it as kept there on purpose. It reaches
this module's `_reads_and_non_spatial_cached` by a plain bare-name call, which resolves through
`interpreter.py`'s own module globals because `interpreter.py` re-exports every name this module
defines (see the re-export import at the bottom of `interpreter.py`'s top-level import block) —
the same ROUTE-45 shape SPLIT-E used, so a spy monkeypatched onto `interpreter.NAME`
(`tests/test_reg1d_consensus_extent_source_fastpath.py`'s `_spy_on_reads_walk`) still intercepts
the call. `interpreter.py`'s own `Interpreter` class also calls a few of these names as bare
globals (e.g. `vec_list_to_tensor`, unaffected — it stays put) and does not call any name this
module owns directly, but the re-export keeps every external `from .interpreter import NAME`
(`tex_cache.py`, `test_trk163_lut_tile_batch_collision.py`, ...) resolving unchanged.

This module deliberately does NOT import `interpreter.py` at module scope for anything —
`_BUILTIN_NAMES` (the one name `_collect_identifiers` needs from there) is imported lazily,
inside that function, exactly like `interpreter_spatial.py`/`interpreter_control_flow.py`/
`interpreter_binding.py` already do for their own back-references: a top-level import here
would be a load-time cycle the far end of which `interpreter.py`'s own top-level import of
*this* module (for the re-export) already occupies (FIX-OBSROUTE R1's follow-up documents the
exact failure mode: importing this module ALONE, first, in a fresh process, would reach
`interpreter.py`'s own body mid-import and ImportError on a name not yet defined there).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields as _dc_fields

from ..tex_compiler.ast_nodes import (
    ASTNode, Program, VarDecl, Assignment, IfElse, ForLoop, WhileLoop, ExprStatement,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, VecConstructor, CastExpr,
    ArrayDecl, ArrayIndexAccess, ArrayLiteral, MatConstructor,
    BindingIndexAccess, BindingSampleAccess,
    FunctionDef, ReturnStmt,
    iter_child_nodes,
)


def _collect_binding_reads_and_non_spatial(program: Program) -> tuple[frozenset[str], frozenset[str]]:
    """`(reads, non_spatial)` — wire-binding (`@A`) names the program mentions anywhere, and
    the subset of those names bound at a registered NON-SPATIAL argument position
    (`stdlib_registry.non_spatial_args_by_name`, e.g. `apply_lut3d`'s LUT argument).

    `reads` is over-inclusive on purpose: an assignment TARGET counts as a mention.
    Narrowing that would buy nothing — the only consumer is `_consensus_extent`, which
    looks these names up in the INPUT binding dict, and a name that is only ever written
    is an output, which is not in that dict when the grid is decided.

    `non_spatial` exists because a plain bound tensor RESOURCE (COLOR-1 ruling 5 — no new
    TEXType for it) can be structurally indistinguishable, by shape alone, from an ordinary
    `[B,H,W,C]` image binding to `_consensus_extent`'s shape scan: without this exclusion,
    binding one alongside a differently-shaped image lets its own leading dims leak into the
    cook's (B,H,W) grid via the `max()` consensus rule, corrupting an UNRELATED output's
    shape — the exact silent-wrong class `_consensus_extent`'s own docstring already exists
    to close for ordinary images. Detected structurally (the call shape, via the callee's
    OWN declared `non_spatial_args`), not by binding name, so any name works and any future
    function gets the same protection by declaring the field, no engine-side edit.

    ONE walk for both answers (`_collect_binding_reads` and `_non_spatial_names_cached` are
    the PUBLIC-shaped accessors over it, each preserving its own pre-existing return type —
    see `_reads_and_non_spatial_cached`, which is the ONE memo, `_READS_MEMO`, both go
    through). Walked with the generic `iter_child_nodes` rather than a hand-written
    per-class dispatch like `_collect_identifiers`'. That walk is field-driven, so a new
    ASTNode field is traversed instead of silently escaping — and the speed the
    hand-written version buys is not needed here, because `_consensus_extent` only pays
    for a MISS here when a program is first seen.
    """
    from .stdlib_registry import non_spatial_args_by_name
    non_spatial_positions = non_spatial_args_by_name()   # {fn name: (arg idx, ...)}
    reads: set[str] = set()
    non_spatial: set[str] = set()
    stack: list[ASTNode] = [program]
    while stack:
        node = stack.pop()
        if type(node) is BindingRef:
            if node.kind == "wire":
                reads.add(node.name)
            continue
        if type(node) is FunctionCall:
            positions = non_spatial_positions.get(node.name)
            if positions:
                for i in positions:
                    if i < len(node.args):
                        arg = node.args[i]
                        if type(arg) is BindingRef and arg.kind == "wire":
                            non_spatial.add(arg.name)
        stack.extend(iter_child_nodes(node))
    return frozenset(reads), frozenset(non_spatial)


def _collect_binding_reads(program: Program) -> frozenset[str]:
    """Wire-binding (`@A`) names the program mentions anywhere — the ORIGINAL, pre-COLOR-1
    public contract (a bare `frozenset[str]`), restored: `tests/test_v037_frontend_parity.py`
    calls this directly and subtracts other frozensets from its result, so its return type
    is load-bearing outside this module, not just an internal convenience. Delegates to
    `_collect_binding_reads_and_non_spatial` (the one AST walk) and returns only `reads`;
    `_non_spatial_names_cached` is the other accessor."""
    reads, _ = _collect_binding_reads_and_non_spatial(program)
    return reads


#: Memo for the walk above, mirroring `tex_memory._tile_safe_memo` (whose comment prices the
#: same shape of walk at ~22 us per CUDA cook — worth memoizing, and this one is worse: no
#: early exit, every node visited). The read/non-spatial answers are a pure function of the
#: AST, so they belong per PROGRAM, not per cook. Without this, the axis-disagreement gate
#: keeps the walk off most cooks but not all: an IMAGE `[4,H,W,3]` batch beside a single
#: `[1,H,W]` MASK disagrees on batch every cook, and that is an ordinary ComfyUI graph, not
#: a corner.
#:
#: Keyed by `id()` because `Program` is a slotted dataclass — no `__dict__` to hang an
#: attribute on and no `__weakref__` to key a WeakKeyDictionary with. The program itself is
#: held beside the answer and re-checked with `is`, which is what makes `id()` safe: a
#: recycled id belongs to a different object and misses. Holding it also pins the AST alive,
#: bounded here to the same order as `tex_cache`'s own program LRU.
_READS_MEMO: "OrderedDict[int, tuple[Program, frozenset[str], frozenset[str]]]" = OrderedDict()
_READS_MEMO_MAX = 128


def _reads_and_non_spatial_cached(program: Program) -> tuple[frozenset[str], frozenset[str]]:
    """`_collect_binding_reads_and_non_spatial`, memoized per program object — the ONE
    memo entry `_binding_reads_cached` and `_non_spatial_names_cached` both read, so a
    program pays for one walk regardless of which (or how many) accessor a caller uses.
    Internal: callers outside this module use one of those two, each of which preserves
    its OWN pre-existing return type (a bare `frozenset[str]`, never this tuple)."""
    from .lru_util import lru_get, lru_put   # lazy: keeps the engine's cold-import closure unchanged
    key = id(program)
    hit = lru_get(_READS_MEMO, key)
    if hit is not None and hit[0] is program:
        return hit[1], hit[2]
    reads, non_spatial = _collect_binding_reads_and_non_spatial(program)
    lru_put(_READS_MEMO, key, (program, reads, non_spatial), _READS_MEMO_MAX)
    return reads, non_spatial


def _binding_reads_cached(program: Program) -> frozenset[str]:
    """`_collect_binding_reads`, memoized per program object — the ORIGINAL, pre-COLOR-1
    public contract (a bare `frozenset[str]`), restored: this predates the non-spatial
    exclusion and callers outside this module may still expect exactly this shape."""
    return _reads_and_non_spatial_cached(program)[0]


def _non_spatial_names_cached(program: Program) -> frozenset[str]:
    """The COLOR-1 non-spatial-argument exclusion set (e.g. `apply_lut3d`'s LUT-binding
    name), memoized per program object — its own accessor over the SAME `_READS_MEMO`
    entry `_binding_reads_cached` populates, so `graphed._spatial_px` and
    `_consensus_extent` share one walk/one memo with `_collect_binding_reads`'s callers
    without either side's return type depending on the other's existence."""
    return _reads_and_non_spatial_cached(program)[1]


def _collect_identifiers(program: Program) -> frozenset[str]:
    """Collect all Identifier names referenced in a program (fast single-pass scan).

    Returns only names that match builtin variable names, for lazy construction.
    """
    from .interpreter import _BUILTIN_NAMES
    found: set[str] = set()
    # Use an explicit stack to avoid recursion overhead
    stack: list[ASTNode] = list(program.statements)
    while stack:
        node = stack.pop()
        cls = type(node)

        if cls is Identifier:
            if node.name in _BUILTIN_NAMES:
                found.add(node.name)
            continue

        # Statements
        if cls is VarDecl:
            if node.initializer:
                stack.append(node.initializer)
        elif cls is Assignment:
            stack.append(node.target)
            stack.append(node.value)
        elif cls is IfElse:
            stack.append(node.condition)
            stack.extend(node.then_body)
            stack.extend(node.else_body)
        elif cls is ForLoop:
            stack.append(node.init)
            stack.append(node.condition)
            stack.append(node.update)
            stack.extend(node.body)
        elif cls is WhileLoop:
            stack.append(node.condition)
            stack.extend(node.body)
        elif cls is ExprStatement:
            stack.append(node.expr)
        elif cls is ArrayDecl:
            if node.initializer:
                stack.append(node.initializer)
        # Expressions
        elif cls is BinOp:
            stack.append(node.left)
            stack.append(node.right)
        elif cls is UnaryOp:
            stack.append(node.operand)
        elif cls is TernaryOp:
            stack.append(node.condition)
            stack.append(node.true_expr)
            stack.append(node.false_expr)
        elif cls is FunctionCall:
            stack.extend(node.args)
        elif cls is VecConstructor:
            stack.extend(node.args)
        elif cls is MatConstructor:
            stack.extend(node.args)
        elif cls is CastExpr:
            stack.append(node.expr)
        elif cls is ChannelAccess:
            stack.append(node.object)
        elif cls is ArrayIndexAccess:
            stack.append(node.array)
            stack.append(node.index)
        elif cls is ArrayLiteral:
            stack.extend(node.elements)
        elif cls is BindingIndexAccess:
            stack.extend(node.args)
        elif cls is BindingSampleAccess:
            stack.extend(node.args)
        elif cls is FunctionDef:
            stack.extend(node.body)
        elif cls is ReturnStmt:
            if node.value:
                stack.append(node.value)
        # NumberLiteral, StringLiteral, BindingRef, BreakStmt, ContinueStmt, ParamDecl — skip

    return frozenset(found)


def _collect_expr_names(expr, idents: set, bindings: set, calls: set | None = None) -> None:
    """Single-pass generic AST walk collecting both Identifier names (into
    *idents*) and BindingRef names (into *bindings*) referenced in an expression, and,
    when *calls* is given, the names of the functions it calls.
    Used by UC-3's uniform-range guard, which needs both to reject a bound that
    reads a loop-var/env-var OR a binding the loop body reassigns."""
    cls = expr.__class__
    if cls is Identifier:
        idents.add(expr.name)
        return
    if cls is BindingRef:
        bindings.add(expr.name)
        return
    if calls is not None and cls is FunctionCall:
        calls.add(expr.name)
    for f in _dc_fields(expr):
        v = getattr(expr, f.name)
        if isinstance(v, ASTNode):
            _collect_expr_names(v, idents, bindings, calls)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, ASTNode):
                    _collect_expr_names(x, idents, bindings, calls)
