"""
ROI-2 — spatial footprint analysis (the spatial sibling of `tex_lazy`).

Where `tex_lazy` answers "which inputs can this program need," `tex_roi` answers
"for a given output pixel, which input pixels does it read." The answer per binding
lives on the lattice

    point  ⊑  halo(up, down, left, right)  ⊑  image

- `point`  — reads only the same input pixel (pure pointwise).
- `halo`   — reads a bounded neighbourhood; four non-negative pixel extents.
- `image`  — whole-image / data-dependent gather (the top; unbounded).

The analysis composes the existing pieces (roadmap ROI-2):
  * `$param` folding — reused from `tex_lazy` (`_substitute_params` + the optimizer
    fold/propagate), so `gauss_blur(@A, $sigma)` resolves its radius when the widget
    value is known; a radius that stays symbolic conservatively → `image`.
  * ROI-1's registry footprint + the **reach model** (`_call_reach`) — turns a call's
    `('halo', r)` / `('halo_arg', i[, mult])` descriptor into a pixel halo. The
    `halo_arg` multiplier is the fix for the trap the roadmap flagged: `gauss_blur`'s
    kernel radius is `ceil(3·sigma)`, not `sigma` — the descriptor carries `mult=3.0`.
  * affine offset extraction — reused from `codegen_stencil` — refines a constant-offset
    `fetch(@A, ix+3, iy)` / `@A[ix-1, iy]` from `image` to a bounded (but **non-narrowable**,
    absolute-coordinate) halo, for ROI-5/GRAPH-1 substrate.

Two consumers:
  * `binding_footprints(code, params)` — the per-binding footprint dict (the substrate),
    memoized like `tex_lazy._memo`. Never raises; over-approximates (a too-large footprint
    is a missed optimisation, never a wrong pixel — the invariant #11 discipline ported to
    the spatial lattice).
  * `roi_plan(code, params)` — the ROI-3 execution plan: is this program safe to cook on a
    sub-region, which bindings narrow to `ROI ⊕ halo`, which pass whole, and the single
    uniform cook halo `H`. Whitelist posture: anything unresolved → not executable → the
    engine cooks whole-frame.

See docs/roi-spatial-laziness.md for the execution model and why narrow-cook-crop is
bit-exact.
"""
from __future__ import annotations

import hashlib
import math
import operator
import os
from collections import OrderedDict
from dataclasses import dataclass

from .tex_compiler.lexer import Lexer
from .tex_compiler.parser import Parser
from .tex_compiler.ast_nodes import (
    BindingRef, NumberLiteral, ChannelAccess, FunctionCall, Assignment, Identifier,
    BindingIndexAccess, BindingSampleAccess, ArrayIndexAccess, VarDecl, FunctionDef,
    ForLoop, WhileLoop, iter_child_nodes,
)
from .tex_compiler.optimizer import _propagate_literal_locals, _fold_all
from .tex_lazy import _substitute_params, _fp32, _param_key
from .tex_runtime import codegen_stencil as _st


# ── The footprint lattice ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Footprint:
    """One binding's spatial access footprint. `kind` is 'point' | 'halo' | 'image'.
    For a halo, (up, down, left, right) are non-negative pixel extents beyond the output
    pixel. `narrowable` is False for a bounded-but-absolute-coordinate gather (an affine
    `fetch`/`sample`): the extent is known, but the executor cannot narrow it under one
    coordinate frame (the input is passed whole), so it does not contribute to the cook
    halo. A direct-tensor halo op (blur/morphology) is narrowable=True."""
    kind: str
    up: int = 0
    down: int = 0
    left: int = 0
    right: int = 0
    narrowable: bool = True

    @property
    def reach(self) -> int:
        """The largest single-side extent (0 for 'point')."""
        return max(self.up, self.down, self.left, self.right)


POINT = Footprint("point")
IMAGE = Footprint("image", narrowable=False)


def _halo(up, down, left, right, narrowable=True) -> Footprint:
    if up == down == left == right == 0:
        return POINT if narrowable else IMAGE
    return Footprint("halo", up, down, left, right, narrowable)


def _lub(a: Footprint, b: Footprint) -> Footprint:
    """Least upper bound on the lattice — the aggregate of two read sites of one binding."""
    if a.kind == "image" or b.kind == "image":
        return IMAGE
    if a.kind == "point":
        return b
    if b.kind == "point":
        return a
    return Footprint("halo", max(a.up, b.up), max(a.down, b.down),
                     max(a.left, b.left), max(a.right, b.right),
                     a.narrowable and b.narrowable)


# ── The reach model (per-function pixel reach from the ROI-1 descriptor) ───────

_FOOTMAP_CACHE: "dict | None" = None


def _footmap() -> dict:
    """`{fn_name: footprint_descriptor}` from the stdlib registry (lazy — REGISTRY is
    populated only when TEXStdlib's class body has run, exactly like tex_memory's
    derivation)."""
    global _FOOTMAP_CACHE
    if _FOOTMAP_CACHE is None:
        from .tex_runtime.stdlib_registry import REGISTRY
        _FOOTMAP_CACHE = {n: e.footprint for e in REGISTRY for n in e.names}
    return _FOOTMAP_CACHE


def _static_number(node) -> float | None:
    """The literal value of a (folded) NumberLiteral, else None."""
    return node.value if node.__class__ is NumberLiteral else None


def _call_reach(name: str, args: list):
    """Resolve a call's pixel reach from its ROI-1 footprint + arguments. Returns:
      * None        — not a spatial op (pointwise fn / unregistered) — reads at ctx.
      * int         — a narrowable-halo direct-tensor op (blur/morphology) of this radius.
      * 'unbounded' — a direct-tensor halo op whose radius is symbolic (a wired scalar): its
                      output is input-shaped, so it cannot be narrowed to an unknown halo and
                      blocks ROI (whole-frame fallback) UNLESS it sits inside a gather.
      * 'image'     — whole-image / temporal gather or reduction — reads the whole input.
    """
    fp = _footmap().get(name)
    if fp is None or fp == "point":
        return None
    if fp == "image":
        return "image"
    kind = fp[0]
    if kind == "halo":
        return int(math.ceil(fp[1]))                    # fixed pixel radius
    if kind == "halo_arg":
        i = fp[1]
        mult = fp[2] if len(fp) > 2 else 1.0            # the reach multiplier (gauss=3.0)
        v = _static_number(args[i]) if i < len(args) else None
        if v is None:
            return "unbounded"                          # symbolic radius → blocks narrowing
        return int(math.ceil(mult * abs(v)))
    if kind == "frame":
        return "image"                                  # spatially whole (temporal: ROI-6)
    return "image"


# ── Per-binding footprint accumulation ────────────────────────────────────────

class _Reads:
    """Per-binding read tally over one program walk."""
    __slots__ = ("narrow_reach", "has_narrow", "whole", "has_whole")

    def __init__(self):
        self.narrow_reach = 0            # max symmetric narrowable-halo reach
        self.has_narrow = False          # read pointwise or through a narrowable halo op
        self.whole = POINT               # accumulated non-narrowable/whole footprint
        self.has_whole = False           # read through a gather / reduction


def _entry(reads: dict, name: str) -> "_Reads":
    """The `_Reads` tally for `name`, created on first touch."""
    e = reads.get(name)
    if e is None:
        e = reads[name] = _Reads()
    return e


def _affine_gather_footprint(node) -> Footprint:
    """Refine a `fetch`/`sample`/`@A[..]`/`@A(..)` gather to a bounded NON-narrowable halo
    when its coordinates are constant offsets of the pixel/uv builtins, else `image`.
    Substrate only (the executor passes gathers whole regardless)."""
    off = _st._extract_fetch_offset(node)   # (binding, dx, dy, channels) | None
    if off is None:
        return IMAGE
    _b, dx, dy, _ch = off
    up = max(0, -dy)
    down = max(0, dy)
    left = max(0, -dx)
    right = max(0, dx)
    return _halo(up, down, left, right, narrowable=False)


def _binding_of(node) -> str | None:
    b = node.binding
    return b.name if isinstance(b, BindingRef) else None


def _scatter_target_base(tgt):
    """The base computed-coordinate access an assignment target scatters into
    (`BindingIndexAccess`/`BindingSampleAccess`), unwrapping any `ChannelAccess` /
    `ArrayIndexAccess` wrapper — `@OUT[x,y].r`, `@OUT[x,y].rgb`, `@OUT[x,y][0]` are all
    scatters — or None if the target is a plain name / swizzle write (not a scatter)."""
    cls = tgt.__class__
    if cls is BindingIndexAccess or cls is BindingSampleAccess:
        return tgt
    if cls is ChannelAccess:
        return _scatter_target_base(tgt.object)
    if cls is ArrayIndexAccess:
        return _scatter_target_base(tgt.array)
    return None


def _accumulate(node, ctx_halo, reads: dict, state: dict) -> None:
    """Walk the AST, tallying each wire binding's read mode. `ctx_halo` is the accumulated
    symmetric narrowable-halo radius from enclosing blur/morphology ops, or the sentinel
    'image' once inside a gather. `state['blocked']` is set when a construct the ROI executor
    cannot honour is reached (a direct-tensor halo op with a symbolic radius outside a
    gather)."""
    cls = node.__class__

    if cls is BindingRef:
        if node.kind == "wire":
            _record(reads, node.name, ctx_halo)
        return

    if cls is ChannelAccess:
        _accumulate(node.object, ctx_halo, reads, state)
        return

    if cls is Assignment:
        # The target is a WRITE, not a read (mirrors codegen_stencil._collect_ident_refs).
        # A computed-coordinate target (`@OUT[x,y]=…` / `@OUT(u,v)=…`) is a SCATTER — an
        # absolute write an ROI sub-region buffer can't land — so it blocks ROI execution
        # here (one walk, one definition of scatter), rather than in a separate `_has_scatter`
        # pass; its coordinate expressions are still reads worth tallying.
        _accumulate(node.value, ctx_halo, reads, state)
        scatter = _scatter_target_base(node.target)   # unwraps @OUT[x,y].r / [0] wrappers
        if scatter is not None:
            state["blocked"] = True
            for a in scatter.args:
                _accumulate(a, ctx_halo, reads, state)
        return

    if cls is FunctionCall:
        r = _call_reach(node.name, node.args)
        if r is None:                                   # pointwise fn — same ctx for all args
            for a in node.args:
                _accumulate(a, ctx_halo, reads, state)
            return
        if not node.args:
            # a spatial op (gather / reduction / halo) with NO image argument is degenerate
            # (a type error that never reaches the engine) — block defensively, since roi_plan
            # is a public API that must stay sound on malformed input.
            state["blocked"] = True
            return
        img, rest = node.args[0], node.args[1:]
        if r == "image":                                # gather / reduction
            # v1: ANY gather/reduction cooks whole-frame (a decoupled gather output grid is
            # ROI-5). Block on PRESENCE, not on attributing the image to a wire binding — a
            # gather over a local alias / a bindless generated image would otherwise escape
            # the gate and silently ROI-shrink (the whitelist posture).
            state["blocked"] = True
            _mark_whole(reads, img, node, state)        # still record the footprint (substrate)
        elif r == "unbounded":                          # direct-tensor op, symbolic radius
            if ctx_halo == "image":
                _accumulate(img, "image", reads, state)  # inside a gather → input is whole anyway
            else:
                state["blocked"] = True                  # cannot narrow to an unknown halo
                _accumulate(img, "image", reads, state)
        else:                                           # narrowable halo op — add r to image arg
            new_ctx = "image" if ctx_halo == "image" else ctx_halo + r
            if new_ctx != "image" and new_ctx > state["halo"]:
                # The cook halo is the max reach of ANY halo op — including one wrapping a
                # GENERATED expression (`erode(vec4(u,v,..),3)`), which reads neighbours of
                # values computed only over the cook region, so the region must still grow.
                state["halo"] = new_ctx
            _accumulate(img, new_ctx, reads, state)
        for a in rest:                                  # radius/coord args read at the outer ctx
            _accumulate(a, ctx_halo, reads, state)
        return

    if cls is BindingIndexAccess or cls is BindingSampleAccess:   # @A[..] / @A(..) gather
        state["blocked"] = True                         # a spatial gather → whole-frame (v1)
        _mark_whole(reads, node, node, state)
        for a in node.args:
            _accumulate(a, ctx_halo, reads, state)
        return

    for ch in iter_child_nodes(node):
        _accumulate(ch, ctx_halo, reads, state)


def _record(reads: dict, name: str, ctx_halo) -> None:
    """A pointwise / narrowable-halo read of `name` under `ctx_halo` (an int radius, or the
    'image' sentinel when the read sits inside a gather's image-argument expression)."""
    e = _entry(reads, name)
    if ctx_halo == "image":
        e.has_whole = True
        e.whole = _lub(e.whole, IMAGE)
    else:
        e.has_narrow = True
        if ctx_halo > e.narrow_reach:
            e.narrow_reach = ctx_halo


def _mark_whole(reads: dict, img_node, gather_node, state: dict) -> None:
    """A gather / reduction read: the image argument is passed whole. `img_node` may be a
    BindingRef (a bare `sample(@A, …)`), a ChannelAccess of one, or a
    BindingIndexAccess/BindingSampleAccess whose own binding is the image. When the image
    argument is itself an expression (`sample(gauss_blur(@A,2), u, v)`) every binding under
    it is read whole — recurse with the 'image' sentinel."""
    if img_node.__class__ is BindingRef:
        name = img_node.name if img_node.kind == "wire" else None
    elif img_node.__class__ is ChannelAccess and img_node.object.__class__ is BindingRef:
        name = img_node.object.name if img_node.object.kind == "wire" else None
    elif img_node.__class__ in (BindingIndexAccess, BindingSampleAccess):
        name = _binding_of(img_node)
    else:
        name = None
    if name is None:
        _accumulate(img_node, "image", reads, state)
        return
    # Substrate refinement: a constant-offset gather is a bounded (non-narrowable) halo.
    fp = _affine_gather_footprint(gather_node)
    e = _entry(reads, name)
    e.has_whole = True
    e.whole = _lub(e.whole, fp)


# ── Public analysis ───────────────────────────────────────────────────────────

_MEMO_MAX = 256
_walk_memo: "OrderedDict[tuple, tuple | None]" = OrderedDict()  # key -> (reads, blocked, halo, erased)


def _is_halo_call(n, fm) -> bool:
    if n.__class__ is not FunctionCall:
        return False
    fp = fm.get(n.name)
    return isinstance(fp, tuple) and len(fp) >= 1 and fp[0] in ("halo", "halo_arg")


def _subtree_has_halo(node, fm) -> bool:
    stack = [node]
    while stack:
        n = stack.pop()
        if _is_halo_call(n, fm):
            return True
        stack.extend(iter_child_nodes(n))
    return False


def _write_target_name(tgt, bindings_only: bool = False):
    """The base binding/variable name an assignment writes, or None if un-nameable.

    `bindings_only` restricts the answer to `@bindings`, skipping local variables — what a
    caller asking "which @names does this program WRITE" wants."""
    cls = tgt.__class__
    if cls is BindingRef:
        return tgt.name
    if cls is Identifier:
        return None if bindings_only else tgt.name
    if cls is ChannelAccess:
        return _write_target_name(tgt.object, bindings_only)
    if cls in (BindingIndexAccess, BindingSampleAccess):
        return tgt.binding.name if tgt.binding.__class__ is BindingRef else None
    if cls is ArrayIndexAccess:
        return _write_target_name(tgt.array, bindings_only)
    return None


def _collect_read_names(node, out) -> None:
    """Every Identifier / BindingRef name in a READ position (a plain assignment target is a
    write, not a read; a computed-coordinate target's index args ARE reads)."""
    cls = node.__class__
    if cls is Identifier or cls is BindingRef:
        out.add(node.name)
        return
    if cls is VarDecl:
        if node.initializer is not None:
            _collect_read_names(node.initializer, out)
        return
    if cls is Assignment:
        tgt = node.target
        if tgt.__class__ not in (Identifier, BindingRef, ChannelAccess):
            _collect_read_names(tgt, out)   # a computed index target — its args are reads
        _collect_read_names(node.value, out)
        return
    for ch in iter_child_nodes(node):
        _collect_read_names(ch, out)


def _has_ungrounded_halo(program) -> bool:
    """True if a narrowable-halo op (blur/morphology) can't have its reach composed by the
    single-expression-tree walk, so the cook halo would be under-sized (ROI-edge
    contamination) — those programs cook whole-frame (the whitelist posture, unknown →
    whole-image, never a shrunk ROI). Two ways this happens, both because `_accumulate` only
    tracks reach within one expression tree:

    (1) a halo op **inside** a VarDecl initializer / FunctionDef body / loop body; and
    (2) a halo **result that flows through a NAME** — a local var or intermediate `@binding`
        assigned a halo-containing value (via a VarDecl OR a bare/reassigning Assignment OR an
        `@T = …` intermediate output) that is then read elsewhere. The double blur
        `b = gauss_blur(@A,2); @OUT = gauss_blur(b,2)` reads `@A` ±12, not the ±6 the walk
        infers across the name boundary. Reading a name that holds a mere INPUT (no halo) is
        fine (`vec4 x = @A; gauss_blur(x,2)` stays executable) — only a name carrying a halo
        result blocks. Precise cross-name reach composition (inline non-literal locals) is
        ROI-5."""
    fm = _footmap()

    def _scan(node, ungrounded: bool) -> bool:
        if ungrounded and _is_halo_call(node, fm):
            return True
        cls = node.__class__
        if cls is VarDecl:
            return node.initializer is not None and _scan(node.initializer, True)
        if cls in (FunctionDef, ForLoop, WhileLoop):
            return any(_scan(ch, True) for ch in iter_child_nodes(node))
        return any(_scan(ch, ungrounded) for ch in iter_child_nodes(node))

    if any(_scan(s, False) for s in program.statements):
        return True

    # (2) a halo result assigned to a NAME that is read elsewhere — collected over the WHOLE
    # tree, not just top-level statements. An `if` body is NOT a case-(1) reach boundary (a
    # single grounded blur in a branch composes its reach fine), so a halo assigned to a name
    # *inside* an `if`/loop/function body — `if (c) { @T = gauss_blur(@A,2); @OUT =
    # gauss_blur(@T,2); }` — escapes case (1), yet still crosses the @T name boundary that the
    # single-expression walk can't compose across (true reach ±12, walk infers ±6). The read
    # side (`_collect_read_names`) already recurses into blocks, so the write side must too, or
    # the intersection misses the nested producer and the cook halo under-sizes (ROI-edge
    # contamination). A name carrying a mere INPUT (no halo) still never blocks.
    halo_named = set()
    stack = list(program.statements)
    while stack:
        n = stack.pop()
        cls = n.__class__
        if cls is VarDecl and n.initializer is not None and _subtree_has_halo(n.initializer, fm):
            halo_named.add(n.name)
        elif cls is Assignment and _subtree_has_halo(n.value, fm):
            tn = _write_target_name(n.target)
            if tn is None:
                return True   # un-nameable halo target → block conservatively
            halo_named.add(tn)
        stack.extend(iter_child_nodes(n))
    if not halo_named:
        return False
    reads = set()
    for s in program.statements:
        _collect_read_names(s, reads)
    return bool(halo_named & reads)


def _fold_program(code: str, param_values: dict):
    """Parse + `$param`-fold, reusing tex_lazy's substitution and the optimizer's
    fold/propagate so halo radii resolve to literals. Returns the folded Program (fresh
    parse — the analysis mutates its AST). Raises on a parse error (caller catches)."""
    program = Parser(Lexer(code).tokenize(), source=code).parse()
    subs = {
        name: NumberLiteral(value=_fp32(v), is_int=isinstance(v, (bool, int)))
        for name, v in param_values.items()
        if isinstance(v, (bool, int, float))
    }
    stmts = program.statements
    if subs:
        for stmt in stmts:
            _substitute_params(stmt, subs)
        stmts = _fold_all(stmts)
        stmts = _propagate_literal_locals(stmts)
        stmts = _fold_all(stmts)
        program.statements = stmts
    return program


def _referenced_at_bindings(code: str) -> frozenset:
    """Every `@name` the SOURCE mentions — lexed, so comments and strings can't spoof one.

    `_walk` accumulates its read set on the `$param`-FOLDED program. That is right for halo
    radii (a symbolic radius cannot be composed into a reach) and WRONG for the binding set,
    because the engine does not fold `$params`: they are runtime bindings and the compile cache
    is keyed on their TYPES, not their values. So a read that folding discarded is still
    evaluated, at FULL extent, by the program that actually runs.

    Measured: `@OUT = mix(@A, @B, $k);` with `k=0.0` folds to `@A`, so `@B` never reached the
    read set and `run_roi` passed it whole into a narrowed cook. The output came back as the
    whole frame while `cooked_roi` reported the window as served — and the shipped host blitted
    it, raising "expanded size (1) must match 64". Same for `lerp`, for a `$param`-guarded
    ternary, and for a blur inside either branch.

    ONE SCAN, shared: `tex_marshalling.sigil_names` does the tokenize for both this and ANIM-1's
    param-only set, memoized on the SOURCE. That matters here beyond tidiness — `_walk`'s memo
    is keyed on code + param VALUES, so before this an ROI param scrub re-tokenized the program
    on every frame."""
    from .tex_marshalling import sigil_names
    return sigil_names(code)[0]


def _walk(code: str, param_values: dict):
    """Parse + `$param`-fold + accumulate, memoized on `(code-hash, param bits)`. Returns
    `(reads, blocked, halo, fold_erased)` or None on ANY failure — the single shared engine behind
    `binding_footprints` and `roi_plan`, so the parse+walk runs once. `blocked` is True when
    the program cannot be cooked on a sub-region: a gather/reduction/scatter is present, a
    halo op has a symbolic radius, or a halo op is ungrounded (behind a local var / function /
    loop). The memo key is computed INSIDE the try, so a non-str code or an unsortable param
    dict falls to None (the 'never raises' contract) rather than escaping."""
    try:
        key = (hashlib.sha256(code.encode()).hexdigest(), _param_key(param_values))
    except Exception:
        return None
    hit = _walk_memo.get(key)
    if hit is not None or key in _walk_memo:
        _walk_memo.move_to_end(key)
        return hit
    try:
        program = _fold_program(code, param_values)
        reads: dict = {}
        state = {"blocked": False, "halo": 0}
        for stmt in program.statements:
            _accumulate(stmt, 0, reads, state)
        blocked = state["blocked"] or _has_ungrounded_halo(program)
        # Bindings the $param-fold ERASED from `reads` but the engine still evaluates. Write
        # targets are excluded — `@OUT` is not an input and narrowing it means nothing. Computed
        # here so it rides the same memo as the walk instead of re-lexing on every cook.
        written = {n for n in (_write_target_name(s.target, bindings_only=True)
                               for s in program.statements if isinstance(s, Assignment))
                   if n is not None}
        result = (reads, blocked, state["halo"],
                  _referenced_at_bindings(code) - written - set(reads))
    except Exception:
        result = None
    _walk_memo[key] = result
    if len(_walk_memo) > _MEMO_MAX:
        _walk_memo.popitem(last=False)
    return result


def binding_footprints(code: str, param_values: dict | None = None) -> dict | None:
    """`{wire_name: Footprint}` for the program given these widget values, or None when the
    analysis fails (caller treats every binding as whole-image). Never raises; over-
    approximates. `where`/`if` branches union automatically (the walk visits every branch).
    This is the ROI-2/5 SUBSTRATE — it reports true footprints (`sample(@A,…)` → A:image)
    regardless of whether the program is ROI-3-executable.

    KNOWN LIMIT (ROI-5): reach is composed only WITHIN one expression tree, so a footprint
    reached through a local-variable alias is UNDER-reported — `vec4 x=@A; @OUT=gauss_blur(x,2)`
    reports `A:point`, not `A:halo(6)` (the reach flows through `x`). ROI-3 execution is
    unaffected (it uses the program-wide cook halo and blocks halo-through-a-read-name); the
    caveat matters only for a future per-edge consumer, which should inline non-literal locals
    first."""
    walked = _walk(code, param_values or {})
    if walked is None:
        return None
    reads = walked[0]
    out = {}
    for name, e in reads.items():
        fp = POINT
        if e.has_narrow:
            fp = _halo(e.narrow_reach, e.narrow_reach, e.narrow_reach, e.narrow_reach)
        if e.has_whole:
            fp = _lub(fp, e.whole)
        out[name] = fp
    return out


@dataclass(frozen=True)
class RoiPlan:
    """ROI-3 execution plan. `executable` is False → the engine cooks whole-frame. When
    True: `narrow` names are sliced to `ROI ⊕ halo` (a zero-copy view) and `halo` is the
    single uniform cook margin (max narrowable reach anywhere in the program)."""
    executable: bool
    halo: int = 0
    narrow: frozenset = frozenset()


_NOT_EXECUTABLE = RoiPlan(False)


def roi_plan(code: str, param_values: dict | None = None) -> RoiPlan:
    """The ROI-3 plan for cooking this program on a sub-region. Not executable — cook
    whole-frame — when: the analysis fails; the program scatters (`@OUT[x,y]=`); a halo op
    has a symbolic radius; a halo op is ungrounded (behind a local var / function / loop, so
    its reach can't compose); or ANY gather / reduction is present
    (`sample`/`fetch`/`sample_*`/`img_*`/`@A[..]`/`@A(..)`).

    v1 scope (see docs/roi-spatial-laziness.md): the ROI cook narrows inputs to `ROI ⊕ halo`
    and cooks the cook-region grid. A gather sizes its output from the INPUT image, not the
    coordinate grid (`fn_sample`/`fn_fetch` build the grid from `img.shape`), so a whole-
    passed gather can't yield an ROI-sized output today — decoupling the gather output grid
    from its input, and a local-variable dataflow model, are ROI-5. So v1 executes exactly
    the point + top-level-grounded direct-tensor halo (blur / morphology) class — the
    dominant compositing ops (grade, blur, vignette, mask shrink/grow) — and everything else
    falls back to a whole-frame cook (correct, just not sub-region-lazy). `binding_footprints`
    still reports gather footprints as ROI-5 substrate. Never raises."""
    walked = _walk(code, param_values or {})
    if walked is None:
        return _NOT_EXECUTABLE
    reads, blocked, halo, fold_erased = walked
    if blocked:
        return _NOT_EXECUTABLE
    narrow = frozenset(name for name, e in reads.items() if e.has_narrow)
    # …plus every binding `$param`-folding removed from `reads` entirely. The plan is EXECUTABLE
    # here, which means no gather and no reduction (either sets `blocked`), so there is no such
    # thing as a legitimately whole-passed spatial binding — a name the fold dropped can only be
    # a pointwise or narrowable-halo read, and leaving it un-narrowed is what let a full-extent
    # operand back into a windowed cook. Only names absent from `reads` are added: a name that
    # IS there keeps its own has_narrow verdict rather than having one imposed on it.
    #
    # Safe even when the dropped read sits under a LARGER halo than the folded program's: the
    # fold is a semantic equivalence at these exact `$param` values (and the memo key includes
    # them), so the discarded branch cannot affect output VALUES — only its extent matters, and
    # narrowing it to the same window is what makes the extents agree.
    narrow |= fold_erased
    return RoiPlan(True, halo, narrow)


# ── ROI-6: temporal (frame-window) analysis ───────────────────────────────────

def _frame_ops(program):
    """Yield the frame-index argument node of every ('frame', i) footprint call
    (`fetch_frame`/`sample_frame`) — the cross-frame reads."""
    fm = _footmap()
    stack = list(program.statements)
    while stack:
        n = stack.pop()
        cls = n.__class__
        if cls is FunctionCall:
            fp = fm.get(n.name)
            if isinstance(fp, tuple) and fp and fp[0] == "frame":
                i = fp[1]
                if i < len(n.args):
                    yield n.args[i]
        elif (cls is BindingIndexAccess or cls is BindingSampleAccess) and len(n.args) == 3:
            # 3-arg cross-frame SUGAR: @A[ix,iy,frame] / @A(u,v,frame) route to
            # fetch_frame/sample_frame with the frame as the LAST arg (args[2]) — NOT args[1]
            # like the direct-call ('frame',1) footprint. _frame_ops missing these declared
            # cross-frame sugar per-frame (batch_sliceable=True) and corrupted every strip.
            yield n.args[2]
        stack.extend(iter_child_nodes(n))


def frame_window(code: str, param_values: dict | None = None):
    """ROI-6 substrate: the program's temporal footprint — the `(min_offset, max_offset)` of
    batch frames read relative to the current frame `fi` (the current frame, offset 0, is
    always in the window), or None ('whole batch') when a `fetch_frame`/`sample_frame` reads
    a frame index that isn't a simple `fi ± const` (a fixed frame, or a data-dependent one).
    `fetch_frame(@A, fi-1, …)` → `(-1, 0)`; a pure per-frame program → `(0, 0)`. Never raises."""
    param_values = param_values or {}
    try:
        program = _fold_program(code, param_values)
        lo = hi = 0
        for frame_arg in _frame_ops(program):
            off = _st._extract_pixel_offset(frame_arg, "fi")
            if off is None:
                return None                # unresolved frame index → whole batch
            lo, hi = min(lo, off), max(hi, off)
        return (lo, hi)
    except Exception:
        return None


def batch_sliceable(code: str, param_values: dict | None = None) -> bool:
    """ROI-6: True if the program has NO frame op at all, so its batch can be cooked in frame
    strips (`tex_memory.run_batch_strips`) and stitched — the batch-axis twin of ROI-3's
    whitelist posture. ANY `fetch_frame`/`sample_frame` (or 3-arg `@A[x,y,f]`/`@A(u,v,f)`
    sugar) is an ABSOLUTE frame-index gather into the batch: under a frame strip (a dim-0
    narrow) `fi` carries GLOBAL indices while the tensor is strip-local, so the op reads the
    wrong (clamped-to-strip) frame at EVERY offset — including offset 0 (the frozen-edge-frame
    the design doc warns of). So any frame op → whole-batch in v1 (the temporal analog of a
    spatial gather, deferred with the same absolute-index limitation). Spatial gathers/blurs
    are per-frame and do NOT block batch-slicing. Never raises."""
    param_values = param_values or {}
    try:
        program = _fold_program(code, param_values)
        return not any(True for _ in _frame_ops(program))
    except Exception:
        return False


def roi_exec_enabled(opt_in: bool | None = None) -> bool:
    """Whether the engine's auto-narrow ROI path may engage — v0.30's flip, **per cook**.

    The v0.30 flip is deliberately not a new global default. The arming decision belongs to
    the same call that supplies the window (`cook(roi=..., roi_exec=True)`), for the reason
    `prepare()`'s own `disown` docstring gives: a process-wide global "describes what a host
    set, not what THIS call will do" — and a host legitimately wants an ROI viewport cook and
    a whole-frame final render in one process, which one global cannot express. `disown` is
    the precedent: ownership rides the CALL.

      * `opt_in=True`  — arm ROI for this cook (the viewport channel).
      * `opt_in=False` — refuse regardless of the env (an explicit host kill switch).
      * `opt_in=None`  — fall back to `TEX_ROI_EXEC` (default off): the CI / nightly-oracle /
        rollback channel. A caller passing neither is byte-identical to v0.29.

    The oracle lane (ROI-4) also drives `tex_memory.run_roi` directly, so the mechanism is
    exercised regardless of this gate. Of the flip's two standing conditions (docs/
    roi-spatial-laziness.md, "The gate that would change the verdict"), (a) is met — the ROI-4
    differential lane now runs nightly — and (b), "a real host consumes `roi=`", is met by the
    host that passes `roi_exec=True`; the in-repo consumer is the v0.30 viewport in
    `examples/host_demo.py` plus `benchmarks/roi_scrub_bench.py`.
    """
    if opt_in is not None:
        # Coerced, not returned raw. `False` is documented above as an explicit KILL SWITCH, and
        # the truthiness of a string defeats that: `roi_exec="0"`, `"false"`, `"off"` — the
        # spellings a host reads out of a config file or an env var — are all truthy, so every
        # one of them ARMED the path they were written to disable. Strings get the same
        # vocabulary `TEX_ROI_EXEC` accepts; everything else is a plain truth test.
        if isinstance(opt_in, str):
            return opt_in.strip().lower() in ("1", "true", "yes", "on")
        return bool(opt_in)
    return os.environ.get("TEX_ROI_EXEC", "0") == "1"


def canonical_roi(roi) -> tuple:
    """The window as a fresh tuple of plain `int`s. Raises TypeError/ValueError if it is not.

    Two jobs, both about the window the ENGINE reports back. `CookResult.cooked_roi` used to be
    whatever object the caller passed, which meant (a) a list in, a list out, contradicting the
    documented 6-tuple, and worse (b) the host's own object handed back to it: a viewport that
    recycles one mutable rect buffer per frame — the obvious way to write one — mutates the
    rect it just cooked, and `cooked_roi` retroactively names a DIFFERENT window than the patch
    covers, so the next blit lands in the wrong place. Copying at the boundary ends both."""
    out = []
    for v in roi:
        if isinstance(v, bool):          # True is a bug in a coordinate slot, not a 1
            raise TypeError("roi entries must be ints, not bools")
        out.append(operator.index(v))
    return tuple(out)


def validate_roi(roi) -> str | None:
    """Is this `roi=` window WELL-FORMED? None when it is, else a short reason for the trace.

    Pure arithmetic, and only arithmetic (this module stays torch-free). Two things it
    deliberately does NOT answer, because each belongs to a caller that knows more:
      * whether the window's `(W, H)` matches the actual bindings — that is per-binding and
        per-dim, so `tex_memory.run_roi` (which holds the tensors and does the narrowing) owns
        it; a shared-size lookup here answered None for heterogeneous inputs and silently
        dropped the check;
      * whether a whole-frame window is worth arming — that is an fp16-economy POLICY, and it
        lives in `tex_engine.prepare`'s gate beside the fp32 clamp it protects. Baking it in
        here meant `run_roi` and the ROI-4 oracle inherited an optimization verdict when they
        only asked "is this window valid?".

    v0.30 makes `roi=` a production path, so an out-of-range or mis-sized window stops being a
    test-only concern. Unvalidated, two measured cases returned WRONG PIXELS silently rather
    than failing: an overhanging window (`roi=(28,28,10,10,32,32)` on a 32x32 input) came back
    4x4 instead of the requested 10x10, and a negative origin (`roi=(-4,0,8,8,32,32)`) produced
    a wrong-sized 8x4 through a negative crop offset in `tex_memory.run_roi`. Both are the
    silent-wrong-output class docs/roi-spatial-laziness.md calls the worst in the codebase.

    A bad window falls back to the whole frame (the whitelist posture — over-approximate,
    never guess), and the caller learns the window was refused from `CookResult.cooked_roi`
    being None.
    """
    if not (isinstance(roi, (tuple, list)) and len(roi) == 6):
        return "roi must be a 6-tuple (x0, y0, w, h, W, H)"
    # Any INTEGER, not just `int`. A host computing its viewport with numpy hands over
    # `numpy.int64`, which is not an `int` — an `isinstance` test refused those windows and
    # silently dropped the host to whole-frame cooks, i.e. the feature quietly did nothing for
    # exactly the callers most likely to use it. `__index__` is the language's own "this is an
    # integer" protocol: numpy/torch scalars satisfy it, floats and strings do not. `bool` does
    # too and is still refused — `True` as a coordinate is a bug, not a 1.
    try:
        roi = canonical_roi(roi)
    except (TypeError, ValueError):
        return "roi entries must be ints"
    x0, y0, w, h, W, H = roi
    if w <= 0 or h <= 0:
        return f"roi w/h must be positive (got {w}x{h})"
    if W <= 0 or H <= 0:
        return f"roi W/H must be positive (got {W}x{H})"
    if x0 < 0 or y0 < 0:
        return f"roi origin must be non-negative (got {x0},{y0})"
    if x0 + w > W or y0 + h > H:
        return f"roi window ({x0},{y0},{w},{h}) overhangs the {W}x{H} image"
    # The window's (W, H) must describe the image the bindings actually carry, or the narrow
    # slices the wrong extent (measured: a mismatch raises inside run_roi and re-cooks
    # whole-frame with a log line on EVERY frame of a pan).
    return None


# ── CACHE-9: chain-level window composition ───────────────────────────────────
# `roi_plan` answers "what margin does THIS program need". A host recooking a region through a
# CHAIN of stages needs the composition of those margins, and getting it wrong is silent: patch
# only the requested rect and every downstream halo op reads a ring of neighbours just outside
# the patch, which still holds pre-edit pixels. Measured in the demo host before it composed
# backwards: stage-5 sharpen wrong over 2157 px, stage-9 vignette over 3987 px, on ANY upstream
# edit. The composition lived only in `examples/host_demo.py`; this is it promoted, so a host
# gets the two inversions below for free rather than rediscovering them.

#: A reach that clamps to the whole frame. Not `inf` — these are pixel counts that get added to
#: coordinates, and an `inf` would poison the arithmetic rather than saturate it.
WHOLE_FRAME = 1 << 30


def stage_halo(code: str, param_values: dict | None = None) -> int:
    """The neighbour reach one stage reads, as a margin in pixels.

    THE INVERSION, and the whole reason this is a function rather than `roi_plan(...).halo`:
    a NON-EXECUTABLE plan reports `halo = 0`, which is the exact opposite of what it means. It
    means "there is a gather / an ungrounded halo / a scatter, so cook the whole frame" —
    unbounded reach, not zero reach. A consumer that trusts that `0` under-grows every upstream
    window and leaves precisely the stale ring this composition exists to prevent, i.e. it
    inverts the whitelist posture (unknown → whole image) into its most dangerous form.

    Never raises: `roi_plan` doesn't, and a reach question must always have a conservative
    answer."""
    plan = roi_plan(code, param_values or {})
    return int(plan.halo) if plan.executable else WHOLE_FRAME


def covers(valid, needed) -> bool:
    """Does the region `valid` contain `needed`? `None` means "the whole frame" — a canvas that
    has only ever been cooked whole is valid everywhere. Pure arithmetic."""
    if valid is None:
        return True
    if needed is None:
        return False
    vx, vy, vw, vh = (int(v) for v in tuple(valid)[:4])
    nx, ny, nw, nh = (int(v) for v in tuple(needed)[:4])
    return (vx <= nx and vy <= ny
            and nx + nw <= vx + vw and ny + nh <= vy + vh)


def chain_windows(halos, roi, dirty_from: int = 0, valid=None) -> "list | None":
    """The window each stage of a linear chain must cook so the FINAL window is correct.

    `halos[i]` is stage i's own reach (from `stage_halo`); `roi` is the 6-tuple window wanted
    out of the LAST stage. Returns one window per stage, `None` for the clean prefix below
    `dirty_from` (those stages are not cooking at all).

    Walk the suffix BACKWARDS, growing each window by its CONSUMER's halo and clamping to the
    frame — the same `ROI ⊕ halo` composition `run_roi` performs within one stage, lifted to
    the chain. Stage i is grown by `halos[i+1]` and not by its own, because the ring stage i
    must supply is the one its consumer will reach into.

    A `WHOLE_FRAME` reach anywhere saturates: that window clamps to the full frame, and so does
    every window above it, which is correct and is what makes this safe to call on chains it
    cannot narrow.

    `valid[i]` is the region canvas `i` is CURRENTLY correct over — `None` for "the whole
    frame", which is what a canvas cooked whole holds. Supplying it is not optional bookkeeping
    on a host that edits at more than one position:

        a region cook at `dirty_from=2` leaves canvases 2..n valid only over their composed
        windows. A LATER edit at `dirty_from=5` reads canvas 4 over a window derived from a
        different `roi` — and wherever that window escapes the region canvas 4 was patched
        over, it reads PRE-EDIT pixels. Measured through the documented `dirty_from` usage:
        2.17e-01 on the second, deeper edit.

    **Returns `None` when the plan cannot be served incrementally at all** — the host must cook
    the whole chain from the source. Note what does NOT work here, because the first fix for
    this tried it: widening the returned windows to the full frame is useless. The upstream
    canvas is not merely being read too narrowly, it is WRONG outside the earlier patch, and no
    window choice at a downstream stage can repair a stale input. The only correct remedy is to
    re-cook from far enough upstream, and `chain_windows` does not own `dirty_from` — so it says
    "not serviceable" and lets the caller do it. (A pixel-level test with a negative control is
    what caught this; the window-arithmetic test it replaced passed while the pixels were still
    wrong by 2.17e-01.)

    Pure arithmetic — this module stays torch-free."""
    n = len(halos)
    out = [None] * n
    if n == 0:
        return out
    out[n - 1] = canonical_roi(roi)
    start = max(0, dirty_from)
    for i in range(n - 2, start - 1, -1):
        x0, y0, w, h, W, H = out[i + 1]
        pad = int(halos[i + 1])
        nx0, ny0 = max(0, x0 - pad), max(0, y0 - pad)
        nx1, ny1 = min(W, x0 + w + pad), min(H, y0 + h + pad)
        out[i] = (nx0, ny0, nx1 - nx0, ny1 - ny0, W, H)
    if valid is not None and start > 0:
        # The dirty suffix reads canvas `start-1`, which is NOT cooking. Its window must lie
        # inside whatever region that canvas is actually correct over. `+ halos[start]` because
        # stage `start` reaches that far into its input.
        need = out[start]
        pad = int(halos[start]) if start < n else 0
        if need is not None:
            x0, y0, w, h, W, H = need
            grown = (max(0, x0 - pad), max(0, y0 - pad),
                     min(W, x0 + w + pad) - max(0, x0 - pad),
                     min(H, y0 + h + pad) - max(0, y0 - pad), W, H)
            upstream_valid = valid[start - 1] if start - 1 < len(valid) else None
            if not covers(upstream_valid, grown):
                return None        # not serviceable — cook the whole chain from the source
    return out


def clear_roi_memo() -> None:
    """Test hook (mirrors tex_lazy.clear_lazy_memo)."""
    _walk_memo.clear()
