"""Interpreter tensor-value helpers — SPLIT-47 (v0.47.0, TRK-210).

Split mechanically out of `interpreter.py` (the STR-7/SPLIT-I pattern: every body below is
byte-identical to the code it replaced there — AGENTS.md §"Trades to REFUSE", mechanical
moves only, never an "improvement" mid-move). This module owns the shared, mostly-leaf
tensor/value utilities the tree-walking core and its sibling mixin modules
(`interpreter_binding.py`, `interpreter_control_flow.py`, `masked_flow.py`, `stdlib_math.py`)
all call: array-index clamping (`_safe_array_index`/`_const_index`/`_host_index`), the
uniform-range integer check (`_int_valued_scalar`), spatial-shape broadcasting
(`_ensure_spatial`/`_matvec`/`_broadcast_pair`/`_tensor_where`), and the CUDA H2D ingest
fence (`_record_ingest_event`, also called from `compiled.py`).

`vec_list_to_tensor` stays in `interpreter.py` itself, not here, even though it belongs to
the same family: its exact body (`t.dim() == 1 and t.shape[0] in (2, 3, 4): t = t.view(1, 1,
1, -1)`) is a mutation-check anchor (`tests/mutation_check.py`) pinned to
`tex_runtime/interpreter.py`.

This module deliberately does NOT import `interpreter.py` at module scope for anything — it
needs nothing from there (`_host_scalar`/`VEC_CHANNELS` come from `.stdlib`, a leaf module
both this file and `interpreter.py` already import directly), so it has no back-reference to
defer and no load-order cycle to avoid (FIX-OBSROUTE R1's fresh-import-first hazard does not
apply here). `interpreter.py` re-exports every name below at its own top level, so a bare
call from within `Interpreter`'s own methods (e.g. `_ensure_spatial(...)`, `_matvec(...)`)
resolves through `interpreter.py`'s module globals exactly as before the move, and every
external `from .interpreter import NAME` (`interpreter_binding.py`, `interpreter_control_flow.py`,
`masked_flow.py`, `stdlib_math.py`, `compiled.py`, ...) keeps resolving unchanged. The
`compiled._record_ingest_event is interpreter._record_ingest_event` identity
(`tests/test_v043_rider_b_ingest_merge.py`) still holds: both names are the SAME function
object, reached via two different re-export chains.
"""
from __future__ import annotations

import math

import torch

from ..tex_compiler.ast_nodes import NumberLiteral
from .stdlib import VEC_CHANNELS, _host_scalar


def _safe_array_index(idx: torch.Tensor, size: int) -> torch.Tensor:
    """Clamp a float index tensor to valid array bounds and convert to int64.

    Equivalent to ``torch.clamp(torch.floor(idx).long(), 0, size - 1)``.
    """
    return torch.clamp(torch.floor(idx).long(), 0, size - 1)


def _const_index(index_node, size: int) -> int | None:
    """Resolve a compile-time literal array index to a floor+clamped Python int
    (identical semantics to _safe_array_index), or None if the index isn't a
    NumberLiteral. Computed without a 0-dim device tensor or the .item() sync
    (a CUDA-graph capture blocker; UC-5)."""
    if index_node.__class__ is NumberLiteral:
        return max(0, min(int(math.floor(index_node.value)), size - 1))
    return None


def _host_index(index: torch.Tensor, size: int) -> int | None:
    """Floor+clamp a RUNTIME index using the host reading it carries (TRK-68), or None
    when it has none — the runtime-scalar counterpart to `_const_index`'s compile-time
    literal, and the same floor+clamp `_safe_array_index` does on the device
    (`torch.clamp(torch.floor(idx).long(), 0, size - 1)`), done on the host instead so a
    `$param` index does not drain the device on every evaluation. A missing tag (a
    genuinely per-cook computed index) answers None and the caller falls back to
    `idx.item()` exactly as before."""
    if index.__class__ is not torch.Tensor or index.dim() != 0:
        return None
    v = _host_scalar(index)
    if v is None:
        return None
    return max(0, min(int(math.floor(v)), size - 1))


def _int_valued_scalar(value) -> int | None:
    """The exact integer of a scalar bound when it is integer-valued; None for a
    fractional bound, a non-finite value, or a spatial/multi-element tensor.

    UC-3a: uniform-range resolution must only fire on integer-valued bounds —
    for those, floor/int/truncate all agree and the resolved Python range()
    matches the general per-iteration path for both int and float loop counters.
    A fractional bound falls back to the general path (correct fractional loop)."""
    if isinstance(value, torch.Tensor):
        if value.dim() != 0:
            return None
        # TRK-68: a `$param` loop bound is minted with a host reading; take it from
        # there instead of draining the device on every loop ENTRY (this runs once
        # per entry, not per iteration — the general per-iteration path below is
        # unaffected). A missing tag (a genuinely computed bound) still reads back.
        hv = _host_scalar(value)
        value = hv if hv is not None else value.item()
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f != math.floor(f):
        return None
    return int(f)


def _ensure_spatial(tensor: torch.Tensor, spatial_shape: tuple) -> torch.Tensor:
    """Expand a tensor to match a spatial shape [B, H, W] if needed.

    TRK-115: a `[B,H,W,1]` scalar-field binding (a 1-channel image — `C == 1` has no
    vec1 type, so `infer_binding_type` maps it to FLOAT the same as a `[B,H,W]` mask)
    passed the `shape[:len(spatial_shape)] == spatial_shape` check vacuously — its
    first 3 dims DO match — so it was handed back UNCHANGED, at rank 4, straight into a
    caller that assigns it into a `spatial_shape`-ranked slot (a vec-constructor
    component, an array element, a channel/index write). PyTorch then aligned the
    trailing dims of the mismatched ranks, lining `H` up against `W`, and raised.
    Every caller here wants exactly `spatial_shape`'s OWN rank back — none of them
    keeps a trailing extra axis — so squeezing it is within this function's existing
    contract, not a widening of it.

    Fixed HERE, at the point of use, rather than at ingest: an ingest-side squeeze
    would also change a plain passthrough's (`@OUT = @A;`) OUTPUT shape — that
    assignment never calls `_ensure_spatial` at all, so a `[B,H,W,1]` binding must
    keep egressing at rank 4 exactly as it always has (invariant 7). Codegen's
    generated code calls this SAME function (imported as `_es`), so both tiers pick
    this fix up identically (invariant 2) with no codegen-side change needed."""
    if not spatial_shape:
        return tensor
    if tensor.dim() == 0:
        return tensor.expand(spatial_shape)
    if tensor.dim() == len(spatial_shape) + 1 and tensor.shape[-1] == 1 \
            and tensor.shape[:len(spatial_shape)] == spatial_shape:
        return tensor.squeeze(-1)
    if tensor.shape[:len(spatial_shape)] == spatial_shape:
        return tensor
    # Try broadcasting
    try:
        return tensor.expand(spatial_shape)
    except RuntimeError:
        return tensor


def _matvec(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Batched matrix @ vector (last-dim contraction), device-tuned (P3).

    For TEX's tiny-matrix / huge-per-pixel-batch shape, `torch.matmul` on a 3x3 (or 4x4)
    against a [B,H,W] batch is launch/overhead-bound on CUDA -- the elementwise
    `(m * v.unsqueeze(-2)).sum(-1)` is 3.4-3.9x faster (mat3) there. On CPU matmul is ~7x
    faster, so keep it. Codegen emits the SAME device-gated expression (`_matvec_expr`),
    so interp<->codegen stays bit-exact on each device. The CUDA broadcast form differs
    from the CPU matmul form by <=1 fp32 ULP (2.4e-7) -- the identical cross-device class
    matmul already has, and 16000x below the 8-bit output quantum."""
    if m.is_cuda:
        return (m * v.unsqueeze(-2)).sum(-1)
    return torch.matmul(m, v.unsqueeze(-1)).squeeze(-1)


def _broadcast_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Broadcast two tensors to be compatible for element-wise operations.

    Handles the key case: scalar [B,H,W] op with vector [B,H,W,C]
    by expanding the scalar with unsqueeze(-1).
    Also pads channel dimensions when both are vectors with different channel counts.
    """
    ad, bd = a.dim(), b.dim()
    if ad == bd:
        # Same rank — pad channel dim when both are vectors (e.g. vec2 [B,H,W,2] + vec3 [B,H,W,3])
        if ad >= 1 and a.shape[-1] != b.shape[-1]:
            ac, bc = a.shape[-1], b.shape[-1]
            if not (ac in VEC_CHANNELS and bc in VEC_CHANNELS):
                return a, b
            if ac < bc:
                a = torch.nn.functional.pad(a, (0, bc - ac))
            else:
                b = torch.nn.functional.pad(b, (0, ac - bc))
        return a, b

    # A bare [N,N] matrix's axes are TRAILING (right-aligned), unlike a scalar
    # field whose spatial axes are LEADING. Handle every bare-matrix pairing
    # explicitly, building a common [<spatial>, N, N] — appending trailing
    # singletons to the matrix (as we do for a scalar) would mis-align it.
    _MAT = (3, 4)

    def _is_bare_mat(t: torch.Tensor) -> bool:
        return t.dim() == 2 and t.shape[-1] in _MAT and t.shape[-1] == t.shape[-2]

    if _is_bare_mat(a) or _is_bare_mat(b):
        mat, other, mat_is_a = (a, b, True) if _is_bare_mat(a) else (b, a, False)
        n = mat.shape[-1]
        if (other.dim() >= 4 and other.shape[-1] in _MAT
                and other.shape[-1] == other.shape[-2]):
            # bare matrix vs spatial matrix [...,N,N]: leading singletons on the bare one
            mat_e = mat.view(*((1,) * (other.dim() - 2)), n, n).expand_as(other)
            return (mat_e, other) if mat_is_a else (other, mat_e)
        # bare matrix vs scalar field [B,H,W]: matrix -> [1..,N,N], field -> [<sp>,1,1]
        sp = tuple(other.shape)
        mat_e = mat.view(*((1,) * len(sp)), n, n).expand(*sp, n, n)
        field_e = other.reshape(*sp, 1, 1).expand(*sp, n, n)
        return (mat_e, field_e) if mat_is_a else (field_e, mat_e)

    # Otherwise pad the lower-rank operand with TRAILING singletons and expand:
    # a scalar field [B,H,W] -> [B,H,W,1] against a vector [B,H,W,C], and a spatial
    # matrix [B,H,W,N,N] vs a scalar field [B,H,W] both land here correctly.
    if ad > bd:
        b = b.view(*b.shape, *((1,) * (ad - bd))).expand_as(a)
        return a, b
    else:
        a = a.view(*a.shape, *((1,) * (bd - ad))).expand_as(b)
        return a, b


def _tensor_where(cond: torch.Tensor, then_val: torch.Tensor, else_val: torch.Tensor) -> torch.Tensor:
    """torch.where with broadcasting support for mixed scalar/vector cases."""
    then_val, else_val = _broadcast_pair(then_val, else_val)

    # Expand condition to match value shapes (single view instead of while loop)
    cd, td = cond.dim(), then_val.dim()
    if cd < td:
        cond = cond.view(*cond.shape, *((1,) * (td - cd)))
        try:
            cond = cond.expand_as(then_val)
        except RuntimeError:
            pass

    return torch.where(cond, then_val, else_val)


# RT-b (v0.43): the single ingest-event fence helper. Was duplicated — this exact body in
# `compiled.py`, and an inline hand-rolled equivalent (detect-during-the-binding-loop,
# record-after-builtins) right here in `_execute_inner`. Moved here (compiled.py already
# imports names from this module, so this introduces no new import cycle) and both call
# shapes now call this one function. Same record-on-H2D detection, same `.synchronize()`
# fence, same stream — `compiled.py` calls it immediately after its own bindings are made
# contiguous (unchanged), and `_execute_inner` calls it once the binding loop has already
# enqueued every H2D copy (also unchanged) — recording an event any time after those
# copies are enqueued correctly bounds their completion, since a CUDA stream is FIFO; only
# recording BEFORE they are all enqueued would be wrong, and neither call site does that.
# `_execute_inner` keeps its own cheap `async_ingest` flag (the detect half of the old
# inline code) purely to GATE this call, so a cook with nothing pinned skips this helper's
# scan instead of re-walking every binding a second time; the helper itself is still the
# only place that does the detecting+recording, so there is one implementation, not two.
def _record_ingest_event(orig_bindings, dev) -> "torch.cuda.Event | None":
    """XPU (v0.20): when ingestion issued a non_blocking pinned→CUDA copy, record
    an event AT THE COPY POINT on the stream. The caller synchronizes it before
    returning the cook's output — closing the cross-node window where a
    (convention-violating) downstream in-place write to the shared pinned source
    could race the in-flight DMA. The wait covers only the copy (recorded before
    compute kernels queue), so it's ~free once the cook's Python work has run."""
    if getattr(dev, "type", None) != "cuda":
        return None
    try:
        for v in orig_bindings.values():
            if (isinstance(v, torch.Tensor) and v.device.type == "cpu"
                    and v.device != dev and v.is_pinned()):
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(dev))
                return ev
    except Exception:
        return None
    return None
