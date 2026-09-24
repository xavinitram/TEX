"""
TEX Standard Library — the shared substrate every `stdlib_<domain>` leaf imports.

Moved verbatim out of the one-file `stdlib.py` when it was split by domain: the safety
constants, the host-resolved-scalar helpers (PERF-2), the cook context (P0-D), the sampler /
grid / mip-pyramid / gaussian-kernel caches with their builders, and the `_to_tensor` family.
Nothing here registers a TEX function — the domain leaves do that, and `stdlib.py` composes
them into `TEXStdlib` and re-exports every name defined here, so `tex_runtime.stdlib.<name>`
still resolves for every reader. A leaf imports THIS module, never `stdlib`: the facade
imports the leaves, so a leaf importing the facade at load time would be a cycle.
"""
from __future__ import annotations
import math
import struct as _struct
import threading as _threading
from collections import OrderedDict as _OrderedDict
import torch
import logging

_texlog = logging.getLogger("TEX")

# pow() NaN detector state. A negative base with a fractional exponent has no
# real value (NaN). We check only a bounded number of pow evaluations
# process-wide, so steady-state execution never pays a per-call GPU sync.
_POW_NAN_STATE = {"checked": 0, "warned": False}

# Unified safety epsilon for division guards, domain clamping, and near-zero checks.
# Chosen to be well above float32 machine epsilon (~1.2e-7) while small enough
# to be invisible in image-processing contexts.
SAFE_EPSILON = 1e-8

# Dtype-aware zero-divisor guard: 1e-8 underflows to 0.0 in fp16 (min normal
# ~6.1e-5), which would defeat the where(divisor==0, eps, divisor) guard and
# yield NaN. Look up by divisor dtype, falling back to SAFE_EPSILON.
ZERO_GUARD_EPS = {torch.float16: 6.104e-5}

# Rec.709 luma coefficients for RGB → luminance conversion
LUMA_R, LUMA_G, LUMA_B = 0.2126, 0.7152, 0.0722

# Valid channel counts for vector types (vec2, vec3, vec4)
VEC_CHANNELS = frozenset((2, 3, 4))


def _has_channel_axis(t) -> bool:
    """True when length/distance/normalize should reduce over the last (channel)
    dim: a standard vec (last dim in {2,3,4}), or any 4D [B,H,W,C] tensor with
    C>1 (channels are always last for 4D) — but never a lower-rank scalar field /
    mask whose last dim is a spatial axis."""
    return (t.dim() >= 1 and t.shape[-1] in VEC_CHANNELS) or (t.dim() >= 4 and t.shape[-1] > 1)


# ── Host-resolved scalars (PERF-2) ────────────────────────────────────
# A few builtins need a PYTHON number rather than a tensor: a kernel radius sizes an
# allocation, an iteration count drives a `range`, a flag picks a branch. Reading one
# off a CUDA tensor costs a 4-byte device->host copy AND the stream synchronisation
# that copy implies, which drains a launch-bound pipeline — once per call, per cook.
#
# Most of those numbers never needed the device at all: a literal, a `$param` float/int
# and a folded constant all begin life as a Python number and are only turned into a
# 0-dim device tensor on the way in. `_tag_host_scalar` records the number ON the tensor
# at the moment that tensor is minted and `_host_scalar` hands it back, so the readback
# is skipped for exactly the value it would have returned. A tensor carrying no tag is a
# genuinely computed value (`gauss_blur(@A, $s * 2.0)`); reading THAT back is legitimate
# and stays.
#
# Two rules keep this behaviour-preserving rather than merely faster:
#   * The tag is the value `.item()` WOULD have returned — the number after the tensor's
#     dtype has rounded it (fp32 by default, fp16 under precision="fp16") — never the
#     un-rounded Python double. Kernel weights are computed from it, so skipping the
#     rounding would be a silent value divergence between the tiers (invariant #2).
#   * Only a mint site tags, and only with a value it is holding, so a tag can never go
#     stale: every tensor operation returns a NEW (untagged) object, and the tagged
#     populations (the interpreter's literal cache, its per-cook scalar bindings) are
#     never written in place. Every reader falls back to the readback it replaced, so a
#     missing tag is slower, never wrong.
_HOST_SCALAR_ATTR = "_tex_host_scalar"


def _dtype_rounded(value, dtype):
    """`value` as the Python double a 0-dim `dtype` tensor holding it would yield from
    `.item()`, computed entirely on the host. fp32 (the default working dtype) goes
    through `struct` — invariant #1 bans numpy and this is the same round-trip
    `tex_marshalling` uses for its fingerprint; any other dtype rounds through a CPU
    scalar tensor, which is a host allocation and still never touches the device.
    Returns None when the rounding cannot be established, which simply leaves the
    caller un-tagged."""
    if dtype is None or dtype is torch.float32:
        try:
            return _struct.unpack("<f", _struct.pack("<f", value))[0]
        except (OverflowError, ValueError, TypeError):
            return None
    try:
        return torch.scalar_tensor(value, dtype=dtype).item()
    except Exception:
        return None


def _tag_host_scalar(t: torch.Tensor, value, dtype=None) -> torch.Tensor:
    """Record `value`'s host reading on a freshly minted 0-dim tensor; returns `t`.

    The fp32 rounding is spelled out here rather than delegated because this runs once
    per scalar binding per cook: the mint sites are hot enough that the extra Python
    frame showed in the host-path frame counts."""
    dt = t.dtype if dtype is None else dtype
    if dt is None or dt is torch.float32:
        try:
            rounded = _struct.unpack("<f", _struct.pack("<f", value))[0]
        except (OverflowError, ValueError, TypeError):
            return t
    else:
        rounded = _dtype_rounded(value, dt)
    if rounded is not None:
        try:
            setattr(t, _HOST_SCALAR_ATTR, rounded)
        except AttributeError:            # a tensor class that refuses attributes
            pass
    return t


def _host_scalar(x):
    """What `.item()` would return for `x`, obtained WITHOUT a device->host copy, or
    None when the value genuinely only exists on the device (the caller then reads it
    back as before).

    A non-tensor answers None deliberately: each call site already has its own number
    conversion and they do NOT agree — `gauss_blur` routes a Python float through
    `_to_tensor` (so fp32-rounded), `bilateral_filter` through `float()` (so not) —
    and answering for them here would silently change one of them."""
    if x.__class__ is not torch.Tensor:
        return None
    v = getattr(x, _HOST_SCALAR_ATTR, None)
    if v is not None:
        return v
    if x.device.type == "cpu" and x.numel() == 1:
        return x.item()                   # host memory: no copy, no stream sync
    return None


_scalar_avg_warned = False


def _scalar_from_tensor(t: torch.Tensor, fn_name: str) -> float:
    """Extract a single scalar value from a tensor for string conversion.

    A 0-dim / single-element / all-equal (uniform) tensor has one well-defined
    value and yields it exactly. A genuinely multi-valued per-pixel field has no
    single value: rather than fail (which would break existing programs that
    summarise a field into a label, e.g. the string_format example), we fall
    back to the mean — but warn ONCE, because that averaged number corresponds
    to no actual element. The old behaviour did this silently; reduce the field
    explicitly (avg/min/max) or index one element to get a defined value.
    """
    if t.numel() == 1:
        v = _host_scalar(t)
        return v if v is not None else t.reshape(()).item()
    flat = t.reshape(-1)
    if bool(torch.all(flat == flat[0])):
        return flat[0].item()
    global _scalar_avg_warned
    if not _scalar_avg_warned:
        _scalar_avg_warned = True
        _texlog.warning(
            "%s() received a multi-valued %s tensor; averaging it to one number for the "
            "string. That value matches no single element — reduce the field first "
            "(e.g. an average/min/max) or index one element to make it explicit.",
            fn_name, tuple(t.shape),
        )
    return flat.float().mean().item()

# ── Sampler tensor cache ──────────────────────────────────────────────
# Caches reusable tensors for sampling functions keyed by (B, H, W, device).
# Avoids recreating batch index tensors and Lanczos tap offsets per call.
# Bounded via LRU eviction to prevent memory leaks in long sessions.
_sampler_cache: _OrderedDict[tuple, torch.Tensor] = _OrderedDict()
_SAMPLER_CACHE_MAX = 32

# ── BCHW permute helper ──────────────────────────────────────────────


def _get_bchw(img: torch.Tensor) -> torch.Tensor:
    """Return a non-contiguous BCHW view of a BHWC image tensor."""
    return img.permute(0, 3, 1, 2)


def _grid_sample_f32(inp: torch.Tensor, grid: torch.Tensor, **kwargs) -> torch.Tensor:
    """grid_sample reconciling dtype (M-3): the grid is always fp32, so if the
    image is fp16/bf16, sample in fp32 and cast the result back — grid_sample
    rejects mixed dtypes, and fp32 sampling is the only correct high-res option."""
    if inp.dtype != grid.dtype:
        out = torch.nn.functional.grid_sample(inp.to(grid.dtype), grid, **kwargs)
        return out.to(inp.dtype)
    return torch.nn.functional.grid_sample(inp, grid, **kwargs)


def _lerp_f32(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """torch.lerp reconciling dtype (M-3): lerp requires start/end/weight to share
    one dtype, but in fp16 mode image-data operands are fp16 while coordinate-
    derived weights stay fp32 — so `mix`/`lerp`/`fit`/`smin`/`smax`/`sample_mip`
    would raise 'expected dtype Half … but got dtype float'. Promote the three to
    their common (widest) dtype, compute, then cast back to the data operand's
    dtype so the fp16 memory contract is preserved. No-op on the all-fp32 path."""
    if a.dtype == b.dtype == t.dtype:
        return torch.lerp(a, b, t)
    common = torch.promote_types(torch.promote_types(a.dtype, b.dtype), t.dtype)
    return torch.lerp(a.to(common), b.to(common), t.to(common)).to(a.dtype)


def _expand_to_bhw(t: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
    """Expand a scalar (dim 0) or 2D [H, W] tensor to [B, H, W]."""
    d = t.dim()
    if d == 0:
        return t.expand(B, H, W)
    if d == 2:
        return t.unsqueeze(0).expand(B, H, W)
    return t  # already [B, H, W] or higher


def _provider_read(source, t, mode: str, a, b):
    """DATA-7: resolve `(source, t)` to ONE host frame, co-located with the coordinates.

    Returns `(frame[1,H,W,C], coord_a, coord_b)`. The frame comes from `tex_provider`'s
    pool — see `docs/frame-providers.md`; nothing here knows about files.

    DEVICE, and the cost it names. The frame lives wherever the provider put it, and the
    coordinates live on the cook device, so one of them has to move. Moving the FRAME is
    right: the coords are the cook's own grid and moving them would drag the result off the
    cook device. The pool caches the provider-device copy, so a CPU provider feeding a CUDA
    cook pays one H2D per CALL rather than per pixel — real, measured in the item's bench
    row, and avoidable by a provider that returns device tensors. Caching a second
    per-device copy is the deferred alternative (it doubles the pool's bytes for a win only
    a mismatched host sees).

    P0-D: the device comes from any coordinate that ARRIVED as a tensor, whatever its RANK.
    The first version tested `.dim()`, which looks like "is this a real grid?" and is not: the
    interpreter evaluates a numeric literal through `_literal_cache`, keyed by
    `(value, device_str, dtype)`, so a constant coordinate is already a 0-dim tensor ON THE
    COOK DEVICE. Testing rank threw that away and fell back to the frame's device, so a CPU
    provider in a CUDA cook raised a bare `RuntimeError` with no E-code the moment the result
    met anything else. Only a raw Python scalar — reachable from a direct API call, never from
    a cooked program — has no device, and then the frame's own device is the honest answer.
    """
    ta, tb = _to_tensor(a), _to_tensor(b)
    dev = None
    for orig, coerced in ((a, ta), (b, tb)):
        if isinstance(orig, torch.Tensor):
            dev = coerced.device
            break
    from ..tex_provider import materialize, _uniform_time
    key = source if isinstance(source, str) else str(source)
    img = materialize(key, _uniform_time(t, key), mode)
    # P0-C: the cast to the cook's working dtype happens HERE, not at the pool boundary, for
    # the reason the boundary's own comment gives: `_normalize` cannot see the cook's
    # precision, so forcing fp32 there doubled the pool and un-did `precision="fp16"`. The
    # pool keeps the source's own width; each read pays a cast only when the widths differ,
    # which for a matched host is never.
    #
    # Device and dtype move in ONE `.to()`. Two calls allocate two full frames for a
    # cross-device, cross-width read — 133 MB of avoidable allocation at 4K RGBA.
    dt = _uniform_dtype()
    if dev is None:
        dev = img.device
    if img.device != dev or (dt is not None and img.dtype != dt):
        img = img.to(device=dev, dtype=dt if dt is not None else img.dtype)
    if ta.device != dev:
        ta = ta.to(dev)
    if tb.device != dev:
        tb = tb.to(dev)
    return img, ta, tb


# P0-D (second half): the cook grid, published by the interpreter for the ONE case that
# cannot derive it from its arguments.
#
# `fetch_time("p", 0.0, 4, 4)` — constant coordinates — produced a [1,1,1,C] result while
# `vec4(0.5,0.5,0.5,1)` and `fetch(@A, 4, 4)` both produced the cook grid. It broadcasts
# correctly in an expression and is wrong the moment it IS the output: a 1×1 image where the
# user wrote a constant-colour frame. The in-batch twins have no such problem because their
# image argument IS the cook grid; a source frame's resolution is its own, so these two have
# nothing to size from.
#
# Thread-local, not a module global: DATA-4 leaves parallel hosts to guard their own caches,
# and a second cook thread must not inherit this one's grid. Set and restored by
# `Interpreter.execute`; absent (None) for any caller outside a cook, where [1,1,1,C] remains
# the honest broadcast-shaped answer.
_cook_ctx = _threading.local()


def set_cook_grid(grid, dtype=None, device=None, viewer=None):
    """Publish the cook's `(B,H,W)` grid, working dtype, device and (PM-11) viewer context.
    Returns an opaque token.

    Pass the token to `restore_cook_ctx` when the cook ends. Two functions rather than one
    that also accepts its own return value: cooks nest (a codegen invocation inside an
    interpreted fallback, a tiled strip loop), so the save/restore pair is real, and a single
    function would have to SNIFF whether its argument is a grid or a saved pair — which is
    guesswork in exactly the place P0-D was already caused by a type test standing in for an
    intent test.

    `device`/`viewer` ride the SAME seam (both `Interpreter.execute` and codegen's
    `_invoke_cg` already call this at the one place each tier publishes its cook state) rather
    than opening a second thread-local: `viewer_exposure`/`viewer_gamma` are the first stdlib
    builtins with no tensor argument to size a device from, and `viewer` (the host's
    `{"viewer_exposure": ..., "viewer_gamma": ...}` dict, or None) is the PM-11 VALUE — never
    part of any key, same discipline as ENG-7's `time_context`."""
    token = (getattr(_cook_ctx, "grid", None), getattr(_cook_ctx, "dtype", None),
             getattr(_cook_ctx, "device", None), getattr(_cook_ctx, "viewer", None))
    _cook_ctx.grid = grid
    _cook_ctx.dtype = dtype
    _cook_ctx.device = device
    _cook_ctx.viewer = viewer
    return token


def restore_cook_ctx(token) -> None:
    """Undo one `set_cook_grid`."""
    _cook_ctx.grid, _cook_ctx.dtype, _cook_ctx.device, _cook_ctx.viewer = token


def _uniform_grid():
    return getattr(_cook_ctx, "grid", None)


def _uniform_dtype():
    return getattr(_cook_ctx, "dtype", None)


def _cook_device():
    """PM-11: the cook device published at `set_cook_grid`, or None outside a cook."""
    return getattr(_cook_ctx, "device", None)


def _viewer_value(name: str, default: float) -> float:
    """PM-11: the host's viewer value for builtin `name` (e.g. "viewer_exposure"), or
    `default` when no `viewer_context` was supplied — the same no-op-by-absence contract
    `time_context.get(name, 0.0)` uses for `frame`/`fps`/`time` (invariant #7: a ComfyUI
    cook that never sets viewer_context sees the identity value, never a KeyError)."""
    ctx = getattr(_cook_ctx, "viewer", None)
    if ctx is None:
        return default
    return float(ctx.get(name, default))


# v042-graph: the CUDA-graph capture's per-replay static input buffers for host-context
# builtins (`viewer_exposure`/`viewer_gamma` today; any future `reads_host_context=True`
# builtin the same way). A SEPARATE thread-local attribute from `.viewer` above — never
# touched by `set_cook_grid`/`restore_cook_ctx` — because it names a buffer IDENTITY that
# `graphed.GraphedProgram` owns across the whole life of a captured key, not a per-cook
# VALUE: it is pushed once around the capture's warmup+record run (the only time the
# interpreter, and so a builtin, actually runs) and popped when that run ends. A later
# `.replay()` never re-enters the interpreter at all — it refreshes the SAME buffer
# tensors with `copy_()`/`fill_()` and replays the captured kernels, which read that
# memory directly, so nothing needs to be pushed again for a replay to see a new value.
#
# Absent (None) for every ordinary cook — including every codegen/interpreter tier cook
# that runs INSIDE a capture's warmup for a program the buffer doesn't cover — so the
# fallback in `fn_viewer_exposure`/`fn_viewer_gamma` (build a fresh tensor from
# `_viewer_value`) is exactly the pre-v042-graph behaviour and invariant 7 holds by
# construction: a program this mechanism never touches sees no new code path at all.
def _push_host_context_buffers(buffers: dict) -> "dict | None":
    """Install `buffers` ({name: persistent 0-dim tensor}) for the duration of one capture
    warmup/record run. Returns the previous value (normally None; nesting is defensive,
    not expected) for `_pop_host_context_buffers`."""
    prev = getattr(_cook_ctx, "host_context_buffers", None)
    _cook_ctx.host_context_buffers = buffers
    return prev


def _pop_host_context_buffers(prev) -> None:
    """Undo one `_push_host_context_buffers`."""
    _cook_ctx.host_context_buffers = prev


def _host_context_buffer(name: str) -> "torch.Tensor | None":
    """The GraphedProgram-owned persistent buffer for host-context builtin `name`, if one
    is installed right now (a capture's warmup/record run), else `None`. A builtin checks
    this FIRST and falls back to its ordinary per-call tensor construction — the same
    tensor object every call while installed, at a stable address, so whatever the graph
    tier bakes into a captured kernel launch is this buffer's memory, never a fresh
    allocation that a later replay (which never calls Python again) could not refresh."""
    buffers = getattr(_cook_ctx, "host_context_buffers", None)
    if buffers is None:
        return None
    return buffers.get(name)


# Pre-allocated grid buffer for sample() — avoids torch.stack allocation per call.
# Keyed by (B, H, W, device) → [B, H, W, 2] tensor.
# Bounded via LRU eviction (each entry is ~16 MB at 1080p).
_grid_buf: _OrderedDict[tuple, torch.Tensor] = _OrderedDict()
_GRID_BUF_MAX = 16


def _get_grid_buf(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Allocate a [B, H, W, 2] grid buffer, keeping the previous one alive.

    PERF TRAP — do not "fix" this into an actual reuse cache. The is_inference
    guard means the hit path never fires in production (everything runs under
    torch.inference_mode), so this always allocates — but storing the buffer in
    the dict keeps the PREVIOUS allocation alive until it is overwritten here,
    which prevents the allocator from returning the block to the OS between
    sample() calls. Measured on CPU at 512²: enabling reuse (or dropping the
    dict and allocating fresh) makes sample-heavy programs ~30% SLOWER
    (cross-region in-place writes / page-fault churn); this exact form is the
    fast one. On CUDA reuse measured neutral.
    """
    key = (B, H, W, device)
    buf = _grid_buf.get(key)
    if buf is not None and not buf.is_inference():
        return buf
    buf = torch.empty(B, H, W, 2, dtype=torch.float32, device=device)
    _grid_buf[key] = buf
    if len(_grid_buf) > _GRID_BUF_MAX:
        _grid_buf.popitem(last=False)
    return buf

# ── Mipmap pyramid cache ─────────────────────────────────────────────
# Caches mipmap pyramids keyed by (id, _version).  We store a reference
# to the source tensor so it won't be garbage-collected (which would let
# Python reuse the id for a new tensor, causing stale cache hits).
# OrderedDict gives LRU eviction to bound memory.
_mip_cache: _OrderedDict[int, tuple[tuple, torch.Tensor, list[torch.Tensor]]] = _OrderedDict()
_gauss_mip_cache: _OrderedDict[tuple, tuple[tuple, torch.Tensor, list[torch.Tensor]]] = _OrderedDict()
_gauss_kernel_cache: _OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = _OrderedDict()
_GAUSS_KERNEL_MAX_ENTRIES = 64  # max cached kernel pairs (tiny GPU tensors)
_MIP_MAX_ENTRIES = 8   # max cached pyramids (each holds multiple GPU tensors)
_MIP_MAX_LEVELS = 12   # cap: 4096 → 1px in 12 halvings


def _get_batch_index(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Get or create cached batch index tensor [B, H, W] for advanced indexing."""
    key = ("bidx", B, H, W, device)
    cached = _sampler_cache.get(key)
    if cached is not None:
        _sampler_cache.move_to_end(key)
        return cached
    t = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
    _sampler_cache[key] = t
    if len(_sampler_cache) > _SAMPLER_CACHE_MAX:
        _sampler_cache.popitem(last=False)
    return t


def _get_flat_batch_index(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Get or create the flattened [B*H*W] batch index for scatter writes.

    Materializing the contiguous flat copy of the expanded [B, H, W] index is
    the expensive part (a full int64 tensor), so it is what gets cached."""
    key = ("bidx_flat", B, H, W, device)
    cached = _sampler_cache.get(key)
    if cached is not None:
        _sampler_cache.move_to_end(key)
        return cached
    t = _get_batch_index(B, H, W, device).contiguous().reshape(-1)
    _sampler_cache[key] = t
    if len(_sampler_cache) > _SAMPLER_CACHE_MAX:
        _sampler_cache.popitem(last=False)
    return t


def _get_lanczos_taps(device: torch.device) -> torch.Tensor:
    """Get or create cached Lanczos-3 tap offset tensor [-2, -1, 0, 1, 2, 3]."""
    key = ("ltaps", device)
    cached = _sampler_cache.get(key)
    if cached is not None:
        _sampler_cache.move_to_end(key)
        return cached
    t = torch.arange(-2, 4, device=device, dtype=torch.float32)
    _sampler_cache[key] = t
    if len(_sampler_cache) > _SAMPLER_CACHE_MAX:
        _sampler_cache.popitem(last=False)
    return t


def _build_sample_grid(u: torch.Tensor, v: torch.Tensor,
                       B: int, H: int, W: int) -> torch.Tensor:
    """Convert [0,1] UV coords to a [-1,1] grid for ``grid_sample``.

    Handles scalar (0-dim), 2-dim [H,W], and 3-dim [B,H,W] inputs.
    For scalar inputs the grid is expanded to (B, H, W); pass H=1, W=1
    to get a single-point grid (useful for mip sampling).
    Returns a ``[B, H_out, W_out, 2]`` tensor suitable for ``grid_sample``.

    TRK-6: forced fp32 (invariant 4 — coordinate/spatial math is never
    `self._dtype`), never inherited from `u`/`v`. `.to(torch.float32)` is a
    no-op (returns the same tensor) when they already are, so the default
    fp32 path pays nothing. Under `precision="fp16"` a UV literal/variable
    takes the cook's own dtype (fp16), and without this the grid this
    function returns would too — `_grid_sample_f32`'s reconciliation only
    fires on a dtype MISMATCH between the image buffer and the grid, so a
    fp16 mip level sampled with a fp16 grid skipped it silently and sampled
    (and mis-addressed large-H rows on) raw fp16 coordinates."""
    u = u.to(torch.float32)
    v = v.to(torch.float32)
    grid_x = u * 2.0 - 1.0
    grid_y = v * 2.0 - 1.0

    gd = grid_x.dim()
    if gd == 0:
        grid_x = grid_x.reshape(1, 1, 1).expand(B, H, W)
        grid_y = grid_y.reshape(1, 1, 1).expand(B, H, W)
    elif gd == 2:
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    return torch.stack([grid_x, grid_y], dim=-1)  # [B, H_out, W_out, 2]


def _reduce_channels(t: torch.Tensor, op):
    """Reduce an array tensor along its element dimension.

    For vec arrays (dim 2 or 5, layout [..., N, C]) reduces along dim=-2.
    For scalar arrays (dim 1 or 4, layout [..., N]) reduces along dim=-1.
    *op* is called as ``op(tensor, dim)`` and must return the reduced tensor.
    """
    if t.dim() in (2, 5):
        return op(t, -2)
    return op(t, -1)



# -- Utility helpers (module-level) ------------------------------------

def _lanczos3(x: torch.Tensor) -> torch.Tensor:
    """Lanczos-3 kernel: sinc(x) * sinc(x/3) for |x| < 3, else 0.

    Uses torch.sinc (normalized: sinc(x) = sin(pi*x)/(pi*x)) which handles
    the x=0 singularity internally, reducing intermediate tensor allocations.
    """
    x = x.float()
    # torch.sinc uses the normalized definition: sinc(x) = sin(pi*x)/(pi*x)
    kernel = torch.sinc(x) * torch.sinc(x / 3.0)
    # Zero out beyond support radius of 3 (mask multiply avoids zeros_like alloc)
    kernel.mul_(torch.abs(x) < 3.0)
    return kernel


# -- Gaussian blur helpers (module-level) -----------------------------------

def _get_gauss_kernels(sigma: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Get or create cached horizontal and vertical 1D Gaussian kernels.

    Returns (kernel_h, kernel_v) as contiguous [1, 1, 1, K] and [1, 1, K, 1] tensors,
    ready for depthwise conv2d (expand to [C, 1, ...] before use).

    PERF-3: the key is the sigma the kernel is BUILT from, exactly. It used to be
    `round(sigma, 3)` while the build used the full value, so two sigmas agreeing to
    three decimals shared whichever kernel arrived first — including across a `ceil`
    step (1.9996 gives 6 taps either side, 2.0001 gives 7), which made the same program
    with the same bindings depend on what the process had blurred earlier. Quantising
    before the build agrees with itself too, but moves every un-quantised sigma's
    output; keying exactly moves none. The LRU below still bounds the entries.
    """
    key = (sigma, device)
    cached = _gauss_kernel_cache.get(key)
    if cached is not None:
        _gauss_kernel_cache.move_to_end(key)
        return cached
    radius = int(math.ceil(3.0 * sigma))
    size = 2 * radius + 1
    x = torch.arange(size, dtype=torch.float32, device=device) - radius
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel_h = kernel.view(1, 1, 1, size)
    kernel_v = kernel.view(1, 1, size, 1).contiguous()
    pair = (kernel_h, kernel_v)
    _gauss_kernel_cache[key] = pair
    if len(_gauss_kernel_cache) > _GAUSS_KERNEL_MAX_ENTRIES:
        _gauss_kernel_cache.popitem(last=False)
    return pair


# ASK-1: convolve's replicate-pad, chunked so a single F.pad call is never asked for
# more margin on an axis than that axis currently has. Replicate padding just repeats
# the border value, so padding an already-padded tensor with more replicate margin
# repeats the SAME original border value again — chunking changes nothing about the
# result (exact, deterministic, no mode change), only how many F.pad calls reach it.
# `gauss_blur`/`_morph` never needed this: their radius is bounded well under any
# image dimension in practice, but convolve's kernel is a full IMAGE binding, read
# whole, up to 257x257 — wider than a small ROI-cooked tile or a thumbnail input.
def _pad_replicate_chunked(x: torch.Tensor, pad_l: int, pad_r: int,
                            pad_t: int, pad_b: int) -> torch.Tensor:
    """Replicate-pad a [B,C,H,W] tensor by (left, right, top, bottom), splitting into
    multiple F.pad calls so no single call pads an axis by more than that axis's
    current size allows."""
    pad = torch.nn.functional.pad
    while pad_l > 0 or pad_r > 0:
        w = x.shape[-1]
        step_l = min(pad_l, max(1, w - 1))
        step_r = min(pad_r, max(1, w - 1))
        x = pad(x, (step_l, step_r, 0, 0), mode="replicate")
        pad_l -= step_l
        pad_r -= step_r
    while pad_t > 0 or pad_b > 0:
        h = x.shape[-2]
        step_t = min(pad_t, max(1, h - 1))
        step_b = min(pad_b, max(1, h - 1))
        x = pad(x, (0, 0, step_t, step_b), mode="replicate")
        pad_t -= step_t
        pad_b -= step_b
    return x


def _gauss_blur_bchw(
    img: torch.Tensor, sigma: float, downsample_2x: bool = False,
) -> torch.Tensor:
    """Apply separable Gaussian blur to a [B, C, H, W] tensor.

    Uses replicate padding and depthwise convolution (groups=C).
    When downsample_2x=True, uses strided convolution to fuse blur + 2× downsample
    into two passes instead of three (blur_h + blur_v + pool).
    """
    if sigma < 0.3:
        if downsample_2x:
            return torch.nn.functional.avg_pool2d(img, kernel_size=2, stride=2)
        return img
    C = img.shape[1]
    kernel_h, kernel_v = _get_gauss_kernels(sigma, img.device)
    radius = kernel_h.shape[-1] // 2
    # M-3: conv2d requires the kernel and input to share a dtype; the cached
    # gaussian kernels are fp32, so reconcile to the image dtype (a no-op on the
    # fp32 path, an fp16 cast under fp16 mode — consistent with fp16 image data).
    kh = kernel_h.to(img.dtype).expand(C, 1, 1, -1)
    kv = kernel_v.to(img.dtype).expand(C, 1, -1, 1)
    stride_w = 2 if downsample_2x else 1
    stride_h = 2 if downsample_2x else 1
    padded = torch.nn.functional.pad(img, (radius, radius, 0, 0), mode='replicate')
    result = torch.nn.functional.conv2d(padded, kh, stride=(1, stride_w), groups=C)
    padded = torch.nn.functional.pad(result, (0, 0, radius, radius), mode='replicate')
    result = torch.nn.functional.conv2d(padded, kv, stride=(stride_h, 1), groups=C)
    return result


# -- Mipmap helpers (module-level) -----------------------------------------

def _build_mip_pyramid(
    img: torch.Tensor,
    cache: _OrderedDict,
    key,
    pre_blur_fn=None,
    fused_blur_downsample_fn=None,
) -> list[torch.Tensor]:
    """Build or retrieve a cached mipmap pyramid for a [B, H, W, C] tensor.

    Returns a list of [B, C, H, W] tensors (channel-first for grid_sample).
    Level 0 is full resolution, each subsequent level is half the size.

    Args:
        cache: LRU OrderedDict to store the pyramid in.
        key: cache lookup key.
        pre_blur_fn: optional callable(bchw_tensor) → blurred bchw_tensor,
            applied before each downsample (e.g. Gaussian pre-blur).
        fused_blur_downsample_fn: optional callable(bchw_tensor) → blurred + 2× downsampled tensor.
            When provided and exact 2× downsample is possible, uses this instead of
            pre_blur_fn + avg_pool2d (saves one kernel launch per level).
    """
    cached = cache.get(key)
    if cached is not None and cached[0] == img.shape:
        cache.move_to_end(key)  # LRU touch
        return cached[2]

    B, H, W, C = img.shape
    level0 = _get_bchw(img)
    pyramid = [level0]

    current = level0
    max_levels = min(_MIP_MAX_LEVELS, int(math.log2(max(min(H, W), 1))))
    for _ in range(max_levels):
        _, _, ch, cw = current.shape
        if ch <= 1 or cw <= 1:
            break
        nh, nw = max(ch // 2, 1), max(cw // 2, 1)
        exact_2x = (ch == nh * 2 and cw == nw * 2)
        # Fused path: strided conv = blur + downsample in 2 ops instead of 3
        if exact_2x and fused_blur_downsample_fn is not None:
            current = fused_blur_downsample_fn(current)
        else:
            src = pre_blur_fn(current) if pre_blur_fn is not None else current
            if exact_2x:
                current = torch.nn.functional.avg_pool2d(src, kernel_size=2, stride=2)
            else:
                current = torch.nn.functional.interpolate(
                    src, size=(nh, nw), mode='area',
                )
        pyramid.append(current)

    # Store tensor ref to prevent GC (keeps id() stable); evict LRU
    cache[key] = (img.shape, img, pyramid)
    if len(cache) > _MIP_MAX_ENTRIES:
        cache.popitem(last=False)
    return pyramid


def _safe_version(t: torch.Tensor) -> int:
    """`t._version` for a normal tensor, 0 for an inference tensor.

    Reading `_version` on an inference tensor raises ("Inference tensors do not
    track version counter"). Inference tensors are immutable within their
    inference-mode region, so a constant is a sound cache-version stand-in.
    Without this, a plain fp32 CUDA `sample_mip` with a CPU-resident IMAGE
    binding fails outright (the binding is moved on-device inside inference mode,
    producing an inference tensor)."""
    return 0 if t.is_inference() else t._version


def _get_mip_pyramid(img: torch.Tensor) -> list[torch.Tensor]:
    """Area-downsample mipmap pyramid (cached)."""
    # Version-safe key: id() + version detects in-place mutations.
    return _build_mip_pyramid(img, _mip_cache, (id(img), _safe_version(img)))


def _get_mip_pyramid_gauss(img: torch.Tensor, sigma: float = 1.13) -> list[torch.Tensor]:
    """Gaussian-prefiltered mipmap pyramid (sigma=1.13, SIGMA_C ≈ 0.825, cached)."""
    # PERF-3, same rule as `_get_gauss_kernels`: key on the sigma the pyramid is BUILT
    # from. Today's only caller passes the default, so nothing moves — but a second
    # caller at a nearby sigma would have been served this one's pyramid.
    key = (id(img), _safe_version(img), sigma)
    return _build_mip_pyramid(
        img, _gauss_mip_cache, key,
        pre_blur_fn=lambda bchw: _gauss_blur_bchw(bchw, sigma),
        fused_blur_downsample_fn=lambda bchw: _gauss_blur_bchw(bchw, sigma, downsample_2x=True),
    )


def _sample_mip_level(
    level_bchw: torch.Tensor,
    grid: torch.Tensor,
    out_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Bilinear sample a single mip level. Returns [B, H_out, W_out, C].

    level_bchw: [B, C, Hl, Wl] — the mip level in channel-first layout.
    grid: [B, H_out, W_out, 2] — pre-stacked sampling grid in [-1, 1].
    out_size: if set and grid is identity UV, use F.interpolate (faster, no grid read).
    """
    if out_size is not None:
        Hl, Wl = level_bchw.shape[2], level_bchw.shape[3]
        if (Hl, Wl) == out_size:
            return level_bchw.permute(0, 2, 3, 1)
        result_bchw = torch.nn.functional.interpolate(
            level_bchw, size=out_size, mode='bilinear', align_corners=True,
        )
        return result_bchw.permute(0, 2, 3, 1)
    result_bchw = _grid_sample_f32(
        level_bchw, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    return result_bchw.permute(0, 2, 3, 1)  # [B, H_out, W_out, C]


def _sample_mip_trilinear(image, u_coord, v_coord, lod, pyramid_fn):
    """Trilinear mip sampling shared by sample_mip and sample_mip_gauss.

    pyramid_fn: callable(img) → list of [B, C, H, W] mip levels.
    """
    img = image if image.__class__ is torch.Tensor else _to_tensor(image)
    u = u_coord if u_coord.__class__ is torch.Tensor else _to_tensor(u_coord)
    v = v_coord if v_coord.__class__ is torch.Tensor else _to_tensor(v_coord)
    lod_t = lod if lod.__class__ is torch.Tensor else _to_tensor(lod)

    pyramid = pyramid_fn(img)
    max_level = len(pyramid) - 1
    # TRK-66/PERF-2: read the host value BEFORE the clamp below — `.clamp()` returns a
    # freshly computed tensor, so a `lod_t` minted from a literal / `$param` / folded
    # constant would otherwise lose its `_host_scalar` tag right here and pay a device
    # readback in the scalar fast path further down for no reason. The clamp itself is
    # host-cheap (min/max against an exact integer bound), so doing it on the host first
    # reproduces exactly what `lod_t.clamp(...).item()` would have returned. None when
    # `lod_t` has no host reading (a genuinely per-cook computed LOD, or a per-pixel
    # one) — the scalar fast path below then falls back to the tensor readback as before,
    # and the per-pixel general path always uses the tensor clamp regardless.
    lod_host = _host_scalar(lod_t) if lod_t.dim() == 0 else None
    if lod_host is not None:
        lod_host = min(max(lod_host, 0.0), float(max_level))
    lod_t = lod_t.clamp(0.0, float(max_level))

    B, H, W, C = img.shape

    # Detect identity UV (standard pixel grid) by checking corner values.
    # When identity, use F.interpolate instead of grid_sample (avoids grid read).
    identity_uv = False
    if u.dim() == 3 and u.shape == (B, H, W) and H > 1 and W > 1:
        # Identity UV: u[0,0,0]=0, u[0,0,W-1]=1, v[0,0,0]=0, v[0,H-1,0]=1.
        # Batch the four corner probes into ONE GPU->CPU sync instead of four
        # (each .item() forces a sync; this runs inside sampling loops).
        c0u, c1u, c0v, c1v = torch.stack(
            [u[0, 0, 0], u[0, 0, -1], v[0, 0, 0], v[0, -1, 0]]
        ).tolist()
        if (abs(c0u) < 1e-5 and abs(c1u - 1.0) < 1e-5
                and abs(c0v) < 1e-5 and abs(c1v - 1.0) < 1e-5):
            identity_uv = True

    if identity_uv:
        out_size = (H, W)
        grid = None  # not needed
    else:
        grid = _build_sample_grid(u, v, B, 1, 1)
        out_size = None

    # Fast path: scalar integer LOD → sample single level, no interpolation
    if lod_t.dim() == 0:
        lod_val = lod_host if lod_host is not None else lod_t.item()
        lod_floor = int(lod_val)
        frac = lod_val - lod_floor
        if frac < 1e-6:
            level = min(lod_floor, max_level)
            return _sample_mip_level(pyramid[level], grid, out_size)

    # General path: trilinear (bilinear per level + lerp between levels)
    lod_floor = torch.floor(lod_t)
    lod_frac = lod_t - lod_floor
    lo = lod_floor.long().clamp(0, max_level)
    hi = (lo + 1).clamp(0, max_level)

    if lo.dim() == 0:
        if lod_host is not None:
            lo_i = min(max(int(math.floor(lod_host)), 0), max_level)
            hi_i = min(lo_i + 1, max_level)
        else:
            lo_i = lo.item()
            hi_i = hi.item()
        s_lo = _sample_mip_level(pyramid[lo_i], grid, out_size)
        if lo_i == hi_i:
            return s_lo
        s_hi = _sample_mip_level(pyramid[hi_i], grid, out_size)
        return _lerp_f32(s_lo, s_hi, lod_frac)

    # Per-pixel LOD: gather-based blending (avoids N boolean mask allocations)
    lo_min = lo.min().item()
    hi_max = hi.max().item()
    n_levels = hi_max - lo_min + 1

    level_list = [_sample_mip_level(pyramid[lvl], grid, out_size)
                  for lvl in range(lo_min, hi_max + 1)]

    if n_levels == 1:
        return level_list[0]

    # Stack into [B, n_levels, H, W, C] and gather lo/hi samples
    stacked = torch.stack(level_list, dim=1)
    lo_local = (lo - lo_min).unsqueeze(1).unsqueeze(-1)  # [B, 1, H, W, 1]
    hi_local = (hi - lo_min).unsqueeze(1).unsqueeze(-1)
    expand_shape = list(stacked.shape)
    expand_shape[1] = 1
    s_lo = torch.gather(stacked, 1, lo_local.expand(expand_shape)).squeeze(1)
    s_hi = torch.gather(stacked, 1, hi_local.expand(expand_shape)).squeeze(1)
    frac_expanded = lod_frac.unsqueeze(-1) if lod_frac.dim() == 3 else lod_frac
    return _lerp_f32(s_lo, s_hi, frac_expanded)




def _to_tensor(x) -> torch.Tensor:
    """Ensure a value is a float torch.Tensor. Preserves an existing floating
    dtype (fp16/bf16/fp32) so the M-3 fp16 image-data mode isn't silently
    upcast to fp32; promotes int/bool tensors to float."""
    if x.__class__ is torch.Tensor:
        return x if x.is_floating_point() else x.float()
    return torch.scalar_tensor(float(x), dtype=torch.float32)


def _to_float(x) -> float:
    """Extract a Python float from a scalar (PERF-2: from the minted host value when
    the tensor carries one, so a radius / iteration count costs no device round trip)."""
    if isinstance(x, torch.Tensor):
        v = _host_scalar(x)
        return v if v is not None else x.item()
    return float(x)


def _is_scalar(x) -> bool:
    """Check if a value is a scalar (Python number or 0-dim tensor)."""
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, torch.Tensor):
        return x.dim() == 0
    return False
