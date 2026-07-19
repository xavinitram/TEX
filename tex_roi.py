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
_walk_memo: "OrderedDict[tuple, tuple | None]" = OrderedDict()  # key -> (reads, blocked, halo)


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


def _write_target_name(tgt):
    """The base binding/variable name an assignment writes, or None if un-nameable."""
    cls = tgt.__class__
    if cls is Identifier or cls is BindingRef:
        return tgt.name
    if cls is ChannelAccess:
        return _write_target_name(tgt.object)
    if cls in (BindingIndexAccess, BindingSampleAccess):
        return tgt.binding.name if tgt.binding.__class__ is BindingRef else None
    if cls is ArrayIndexAccess:
        return _write_target_name(tgt.array)
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


def _walk(code: str, param_values: dict):
    """Parse + `$param`-fold + accumulate, memoized on `(code-hash, param bits)`. Returns
    `(reads, blocked, halo)` or None on ANY failure — the single shared engine behind
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
        result = (reads, blocked, state["halo"])
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
    reads, blocked, halo = walked
    if blocked:
        return _NOT_EXECUTABLE
    narrow = frozenset(name for name, e in reads.items() if e.has_narrow)
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


def roi_exec_enabled() -> bool:
    """ROI-3 is FLAGGED OFF: the engine auto-narrow path engages only when `TEX_ROI_EXEC=1`.
    The oracle lane (ROI-4) drives `tex_memory.run_roi` directly, so it exercises the
    mechanism regardless of this flag; production flips it on once that lane is green and a
    host (a viewport) actually asks for a sub-region."""
    return os.environ.get("TEX_ROI_EXEC", "0") == "1"


def clear_roi_memo() -> None:
    """Test hook (mirrors tex_lazy.clear_lazy_memo)."""
    _walk_memo.clear()
