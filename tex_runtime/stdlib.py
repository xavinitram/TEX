"""
TEX Standard Library — runtime implementations of built-in functions.

All functions operate on PyTorch tensors. Scalars are represented as
0-dim tensors or Python floats and get broadcast automatically by PyTorch.
"""
from __future__ import annotations
import hashlib
import math
import re
from collections import OrderedDict as _OrderedDict
import torch
import logging
from .stdlib_registry import stdlib
from . import guard_trace  # C4-ux: guarded-division near-singularity trace (leaf, no cycle)

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
        return t.reshape(()).item()
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
    """
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


class TEXStdlib:
    """Registry of built-in TEX functions."""

    @staticmethod
    def get_functions() -> dict[str, callable]:
        # REG-1: the name->impl map is the registry view — one line, no drift.
        # Each fn_* carries a co-located @stdlib("name") decorator (see
        # stdlib_registry); adding a function no longer edits this method.
        from .stdlib_registry import functions
        return functions()

    # -- Math functions -------------------------------------------------

    @stdlib("sin", sig='sin(x) \\u2192 float', category='Math', doc='Sine (radians).', ex='float s = sin(u * PI * 2.0);')
    @staticmethod
    def fn_sin(x) -> torch.Tensor:
        return torch.sin(_to_tensor(x))

    @stdlib("cos", sig='cos(x) \\u2192 float', category='Math', doc='Cosine (radians).', ex='float c = cos(v * PI);')
    @staticmethod
    def fn_cos(x) -> torch.Tensor:
        return torch.cos(_to_tensor(x))

    @stdlib("tan", sig='tan(x) \\u2192 float', category='Math', doc='Tangent (radians).', ex='float t = tan(u);')
    @staticmethod
    def fn_tan(x) -> torch.Tensor:
        return torch.tan(_to_tensor(x))

    @stdlib("asin", sig='asin(x) \\u2192 float', category='Math', doc='Arcsine. Returns radians.', ex='float angle = asin(0.5);')
    @staticmethod
    def fn_asin(x) -> torch.Tensor:
        return torch.asin(torch.clamp(_to_tensor(x), -1.0, 1.0))

    @stdlib("acos", sig='acos(x) \\u2192 float', category='Math', doc='Arccosine. Returns radians.', ex='float angle = acos(0.5);')
    @staticmethod
    def fn_acos(x) -> torch.Tensor:
        return torch.acos(torch.clamp(_to_tensor(x), -1.0, 1.0))

    @stdlib("atan", sig='atan(x) \\u2192 float', category='Math', doc='Arctangent. Returns radians.', ex='float angle = atan(1.0);')
    @staticmethod
    def fn_atan(x) -> torch.Tensor:
        return torch.atan(_to_tensor(x))

    @stdlib("atan2", sig='atan2(y, x) \\u2192 float', category='Math', doc='Two-argument arctangent. Returns radians.', ex='float angle = atan2(v - 0.5, u - 0.5);')
    @staticmethod
    def fn_atan2(y, x) -> torch.Tensor:
        return torch.atan2(_to_tensor(y), _to_tensor(x))

    @stdlib("sincos", sig='sincos(x) \\u2192 vec2', category='Math', doc='Returns vec2(sin(x), cos(x)). More efficient than separate sin/cos calls.', ex='vec2 sc = sincos(angle);\nfloat s = sc.x;\nfloat c = sc.y;')
    @staticmethod
    def fn_sincos(x) -> torch.Tensor:
        """Returns vec2(sin(x), cos(x)) — computes both in a single pass."""
        t = _to_tensor(x)
        return torch.stack([torch.sin(t), torch.cos(t)], dim=-1)

    @stdlib("sqrt", sig='sqrt(x) \\u2192 float', category='Math', doc='Square root.', ex='float s = sqrt(u * u + v * v);')
    @staticmethod
    def fn_sqrt(x) -> torch.Tensor:
        return torch.sqrt(torch.clamp(_to_tensor(x), min=0.0))

    @stdlib("pow", sig='pow(x, y) \\u2192 float', category='Math', doc='Raise x to the power y.', ex='float p = pow(u, 2.2);')
    @staticmethod
    def fn_pow(base, exp) -> torch.Tensor:
        b = _to_tensor(base)
        e = _to_tensor(exp)
        # An exp-log fast path (exp(log(b)*e)) is faster for spatial tensors but
        # silently destroys the sign of negative bases: pow(x, 2) on a signed /
        # centered coordinate (vignettes, radial gradients, SDFs) would return
        # ~0 instead of x*x. torch.pow is correct for negative bases with whole
        # exponents and matches the scalar path, so results are path-independent.
        out = torch.pow(b, e)
        # Make silent NaN visible: pow(-2, 0.5) and similar have no real answer.
        # Bounded so this never adds a per-call sync to steady-state execution.
        # CUDA-only exception: the .any() bool sync is a verified CUDA-graph
        # CAPTURE BLOCKER for the first 32 pow calls process-wide (capture failed
        # with 9 warm pow calls, succeeded with 36). The diagnostic is a
        # best-effort nicety, so skip it on CUDA and keep it CPU-only.
        _st = _POW_NAN_STATE
        if not _st["warned"] and _st["checked"] < 32 and not out.is_cuda:
            _st["checked"] += 1
            if torch.isnan(out).any() and not torch.isnan(b).any():
                _st["warned"] = True
                _texlog.warning(
                    "[TEX] pow() produced NaN: a negative base with a fractional exponent "
                    "(e.g. pow(-2, 0.5)) has no real answer, so those pixels are NaN. Use "
                    "spow(x, y) for a sign-preserving power, or pow(abs(x), y) for magnitude."
                )
        return out

    @stdlib("exp", sig='exp(x) \\u2192 float', category='Math', doc='e raised to the power x.', ex='float e = exp(-u * 5.0);')
    @staticmethod
    def fn_exp(x) -> torch.Tensor:
        return torch.exp(_to_tensor(x))

    @stdlib("log", sig='log(x) \\u2192 float', category='Math', doc='Natural logarithm (base e).', ex='float l = log(u + 1.0);')
    @staticmethod
    def fn_log(x) -> torch.Tensor:
        return torch.log(torch.clamp(_to_tensor(x), min=SAFE_EPSILON))

    @stdlib("abs", sig='abs(x) \\u2192 float', category='Math', doc='Absolute value.', ex='float a = abs(u - 0.5);')
    @staticmethod
    def fn_abs(x) -> torch.Tensor:
        return torch.abs(_to_tensor(x))

    @stdlib("sign", sig='sign(x) \\u2192 float', category='Math', doc='Returns -1, 0, or 1.', ex='float s = sign(u - 0.5);')
    @staticmethod
    def fn_sign(x) -> torch.Tensor:
        return torch.sign(_to_tensor(x))

    @stdlib("floor", sig='floor(x) \\u2192 float', category='Math', doc='Round down to nearest integer.', ex='float f = floor(u * 10.0);')
    @staticmethod
    def fn_floor(x) -> torch.Tensor:
        return torch.floor(_to_tensor(x))

    @stdlib("ceil", sig='ceil(x) \\u2192 float', category='Math', doc='Round up to nearest integer.', ex='float c = ceil(u * 10.0);')
    @staticmethod
    def fn_ceil(x) -> torch.Tensor:
        return torch.ceil(_to_tensor(x))

    @stdlib("round", sig='round(x) \\u2192 float', category='Math', doc='Round to nearest integer.', ex='float r = round(u * 10.0) / 10.0;')
    @staticmethod
    def fn_round(x) -> torch.Tensor:
        return torch.round(_to_tensor(x))

    @stdlib("trunc", sig='trunc(x) \\u2192 float', category='Math', doc='Truncate toward zero (drop fractional part).', ex='float t = trunc(u * 10.0);')
    @staticmethod
    def fn_trunc(x) -> torch.Tensor:
        return torch.trunc(_to_tensor(x))

    @stdlib("fract", sig='fract(x) \\u2192 float', category='Math', doc='Fractional part: x - floor(x).', ex='float f = fract(u * 5.0);')
    @staticmethod
    def fn_fract(x) -> torch.Tensor:
        t = _to_tensor(x)
        return t - torch.floor(t)

    @stdlib("mod", sig='mod(x, y) \\u2192 float', category='Math', doc='Modulo (remainder).', ex='float m = mod(u * 10.0, 1.0);')
    @staticmethod
    def fn_mod(a, b) -> torch.Tensor:
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        zero = b_t == 0
        guard_trace.note(zero)  # C4-ux (no-op unless armed)
        safe_b = torch.where(zero, ZERO_GUARD_EPS.get(b_t.dtype, SAFE_EPSILON), b_t)
        return torch.fmod(a_t, safe_b)

    @stdlib("log2", sig='log2(x) \\u2192 float', category='Math', doc='Logarithm base 2.', ex='float l = log2(256.0);')
    @staticmethod
    def fn_log2(x) -> torch.Tensor:
        return torch.log2(torch.clamp(_to_tensor(x), min=SAFE_EPSILON))

    @stdlib("log10", sig='log10(x) \\u2192 float', category='Math', doc='Logarithm base 10.', ex='float l = log10(1000.0);')
    @staticmethod
    def fn_log10(x) -> torch.Tensor:
        return torch.log10(torch.clamp(_to_tensor(x), min=SAFE_EPSILON))

    @stdlib("pow2", sig='pow2(x) \\u2192 float', category='Math', doc='2 raised to the power x.', ex='float p = pow2(8.0);')
    @staticmethod
    def fn_pow2(x) -> torch.Tensor:
        return torch.pow(2.0, _to_tensor(x))

    @stdlib("pow10", sig='pow10(x) \\u2192 float', category='Math', doc='10 raised to the power x.', ex='float p = pow10(3.0);')
    @staticmethod
    def fn_pow10(x) -> torch.Tensor:
        return torch.pow(10.0, _to_tensor(x))

    @stdlib("sinh", sig='sinh(x) \\u2192 float', category='Math', doc='Hyperbolic sine.', ex='float s = sinh(u);')
    @staticmethod
    def fn_sinh(x) -> torch.Tensor:
        return torch.sinh(_to_tensor(x))

    @stdlib("cosh", sig='cosh(x) \\u2192 float', category='Math', doc='Hyperbolic cosine.', ex='float c = cosh(u);')
    @staticmethod
    def fn_cosh(x) -> torch.Tensor:
        return torch.cosh(_to_tensor(x))

    @stdlib("tanh", sig='tanh(x) \\u2192 float', category='Math', doc='Hyperbolic tangent.', ex='float t = tanh(u * 2.0);')
    @staticmethod
    def fn_tanh(x) -> torch.Tensor:
        return torch.tanh(_to_tensor(x))

    @stdlib("hypot", sig='hypot(x, y) \\u2192 float', category='Math', doc='Hypotenuse: sqrt(x*x + y*y).', ex='float d = hypot(u - 0.5, v - 0.5);')
    @staticmethod
    def fn_hypot(x, y) -> torch.Tensor:
        return torch.hypot(_to_tensor(x), _to_tensor(y))

    @stdlib("isnan", sig='isnan(x) \\u2192 float', category='Math', doc='Returns 1.0 if x is NaN, 0.0 otherwise.', ex='float check = isnan(x);')
    @staticmethod
    def fn_isnan(x) -> torch.Tensor:
        return torch.isnan(_to_tensor(x)).float()

    @stdlib("isinf", sig='isinf(x) \\u2192 float', category='Math', doc='Returns 1.0 if x is infinite, 0.0 otherwise.', ex='float check = isinf(x);')
    @staticmethod
    def fn_isinf(x) -> torch.Tensor:
        return torch.isinf(_to_tensor(x)).float()

    @stdlib("degrees", sig='degrees(x) \\u2192 float', category='Math', doc='Convert radians to degrees.', ex='float d = degrees(PI);')
    @staticmethod
    def fn_degrees(x) -> torch.Tensor:
        return torch.rad2deg(_to_tensor(x))

    @stdlib("radians", sig='radians(x) \\u2192 float', category='Math', doc='Convert degrees to radians.', ex='float r = radians(180.0);')
    @staticmethod
    def fn_radians(x) -> torch.Tensor:
        return torch.deg2rad(_to_tensor(x))

    @stdlib("spow", sig='spow(x, y) \\u2192 float', category='Math', doc='Sign-preserving power. Safe for negative x.', ex='float s = spow(u - 0.5, 2.0);')
    @staticmethod
    def fn_spow(x, y) -> torch.Tensor:
        """Safe power — sign(x) * pow(abs(x), y). Avoids NaN on negative bases."""
        t = _to_tensor(x)
        yt = _to_tensor(y)
        abs_t = torch.abs(t)
        mask = abs_t < SAFE_EPSILON
        safe_abs = torch.clamp(abs_t, min=SAFE_EPSILON)
        return torch.where(mask, torch.zeros_like(t), torch.sign(t) * torch.pow(safe_abs, yt))

    @stdlib("sdiv", sig='sdiv(a, b) \\u2192 float', category='Math', doc='Safe divide. Returns 0 when b is zero.', ex='float d = sdiv(1.0, u);')
    @staticmethod
    def fn_sdiv(a, b) -> torch.Tensor:
        """Safe division — returns 0.0 where abs(b) < SAFE_EPSILON."""
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        mask = torch.abs(b_t) < SAFE_EPSILON
        guard_trace.note(mask)  # C4-ux (no-op unless armed)
        safe_b = torch.where(mask, torch.ones_like(b_t), b_t)
        return torch.where(mask, torch.zeros_like(a_t), a_t / safe_b)

    # -- Matrix operations ----------------------------------------------

    @stdlib("transpose", sig='transpose(m) \\u2192 mat', category='Matrix', doc='Transpose a matrix.', ex='mat3 mt = transpose(m);')
    @staticmethod
    def fn_transpose(m) -> torch.Tensor:
        return m.transpose(-1, -2)

    @stdlib("determinant", sig='determinant(m) \\u2192 float', category='Matrix', doc='Compute the determinant.', ex='float det = determinant(m);')
    @staticmethod
    def fn_determinant(m) -> torch.Tensor:
        return torch.linalg.det(m)

    @stdlib("inverse", sig='inverse(m) \\u2192 mat', category='Matrix', doc='Compute the matrix inverse.', ex='mat3 inv = inverse(m);')
    @staticmethod
    def fn_inverse(m) -> torch.Tensor:
        try:
            return torch.linalg.inv(m)
        except torch.linalg.LinAlgError as e:
            # Only a genuinely singular matrix gets the friendly explanation;
            # other failures (e.g. a CUDA OOM) keep their real cause.
            if "singular" in str(e).lower():
                raise ValueError(
                    "inverse() can't invert this matrix because it's singular "
                    "(its determinant is zero) — usually two rows/columns are identical, "
                    "or one is all zeros. Check determinant(m) first, or rebuild the matrix."
                ) from e
            raise

    # -- Clamping / interpolation ---------------------------------------

    @stdlib("min", sig='min(a, b) \\u2192 float', category='Interpolation', doc='Returns the smaller value.', ex='float m = min(u, 0.5);')
    @staticmethod
    def fn_min(a, b) -> torch.Tensor:
        return torch.minimum(_to_tensor(a), _to_tensor(b))

    @stdlib("max", sig='max(a, b) \\u2192 float', category='Interpolation', doc='Returns the larger value.', ex='float m = max(u, 0.0);')
    @staticmethod
    def fn_max(a, b) -> torch.Tensor:
        return torch.maximum(_to_tensor(a), _to_tensor(b))

    @stdlib("clamp", sig='clamp(x, lo, hi) \\u2192 float', category='Interpolation', doc='Clamp x to [lo, hi] range.', ex='float c = clamp(u * 2.0, 0.0, 1.0);')
    @staticmethod
    def fn_clamp(x, lo, hi) -> torch.Tensor:
        # Python-number bounds: the scalar torch.clamp overload (one kernel).
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return torch.clamp(_to_tensor(x), min=lo, max=hi)
        # 0-dim tensor bounds (interpreter literals live on-device), split by
        # device: on CUDA pass them through as tensor bounds — sync-free (each
        # .item() would flush the launch-bound pipeline); on CPU .item() is
        # nearly free and the scalar overload's kernel is measurably faster
        # than the broadcasting tensor overload.
        if _is_scalar(lo) and _is_scalar(hi):
            xt = _to_tensor(x)
            if xt.is_cuda:
                return torch.clamp(xt, min=_to_tensor(lo), max=_to_tensor(hi))
            return torch.clamp(xt, min=_to_float(lo), max=_to_float(hi))
        # Spatially-varying bounds
        return torch.minimum(torch.maximum(_to_tensor(x), _to_tensor(lo)), _to_tensor(hi))

    @stdlib("lerp", sig='lerp(a, b, t) \\u2192 float', category='Interpolation', aliases=("mix",), doc='Linear interpolation from a to b by t.', ex='@OUT = lerp(@A, @B, 0.5);')
    @staticmethod
    def fn_lerp(a, b, t) -> torch.Tensor:
        a_t, b_t, t_t = _to_tensor(a), _to_tensor(b), _to_tensor(t)
        # Auto-unsqueeze weight for channel broadcast: [B,H,W] weight with [B,H,W,C] values
        if t_t.dim() + 1 == a_t.dim():
            t_t = t_t.unsqueeze(-1)
        return _lerp_f32(a_t, b_t, t_t)

    @stdlib("fit", sig='fit(x, inLo, inHi, outLo, outHi) \\u2192 float', category='Interpolation', doc='Remap x from [inLo, inHi] to [outLo, outHi].', ex='float y = fit(u, 0.2, 0.8, 0.0, 1.0);')
    @staticmethod
    def fn_fit(val, old_min, old_max, new_min, new_max) -> torch.Tensor:
        """Remap val from [old_min, old_max] to [new_min, new_max]."""
        v = _to_tensor(val)
        o_min, o_max = _to_tensor(old_min), _to_tensor(old_max)
        n_min, n_max = _to_tensor(new_min), _to_tensor(new_max)
        t = (v - o_min) / (o_max - o_min + SAFE_EPSILON)
        return _lerp_f32(n_min, n_max, t)

    @stdlib("smoothstep", sig='smoothstep(lo, hi, x) \\u2192 float', category='Interpolation', doc='Smooth Hermite interpolation between lo and hi.', ex='float s = smoothstep(0.3, 0.7, u);')
    @staticmethod
    def fn_smoothstep(edge0, edge1, x) -> torch.Tensor:
        e0, e1, xv = _to_tensor(edge0), _to_tensor(edge1), _to_tensor(x)
        t = torch.clamp((xv - e0) / (e1 - e0 + SAFE_EPSILON), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @stdlib("step", sig='step(edge, x) \\u2192 float', category='Interpolation', doc='Returns 0 if x < edge, 1 otherwise.', ex='float s = step(0.5, u);')
    @staticmethod
    def fn_step(edge, x) -> torch.Tensor:
        return ((_to_tensor(x)) >= _to_tensor(edge)).float()

    # -- Vector operations ----------------------------------------------

    @stdlib("dot", sig='dot(a, b) \\u2192 float', category='Vector', doc='Dot product of two vectors.', ex='float d = dot(normal, lightDir);')
    @staticmethod
    def fn_dot(a, b) -> torch.Tensor:
        """Dot product over the channel (last) dim.

        einsum is ~4x faster than mul+sum on CPU but ~6-10x SLOWER on CUDA
        (measured), so pick per device. Numerically equivalent (~1e-7)."""
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        if a_t.is_cuda:
            return (a_t * b_t).sum(dim=-1)
        return torch.einsum('...c,...c->...', a_t, b_t)

    @stdlib("length", sig='length(v) \\u2192 float', category='Vector', doc='Length (magnitude) of a vector.', ex='float len = length(vec3(u, v, 0.0));')
    @staticmethod
    def fn_length(v) -> torch.Tensor:
        """Length (magnitude) of a vector, reduced over the channel (last) dim.

        A standard vec (last dim in {2,3,4}) reduces; additionally any 4D
        [B,H,W,C] tensor carries channels last, so C>1 reduces too (covers
        exotic channel counts). Lower-rank tensors (scalar fields / masks) have
        no channel dim and return abs — never reducing a mask's width axis.
        """
        t = _to_tensor(v)
        if _has_channel_axis(t):
            return torch.linalg.vector_norm(t, dim=-1)
        return torch.abs(t)

    @stdlib("distance", sig='distance(a, b) \\u2192 float', category='Vector', doc='Distance between two points.', ex='float d = distance(vec3(u,v,0), vec3(0.5,0.5,0));')
    @staticmethod
    def fn_distance(a, b) -> torch.Tensor:
        diff = _to_tensor(a) - _to_tensor(b)
        if _has_channel_axis(diff):
            return torch.linalg.vector_norm(diff, dim=-1)
        return torch.abs(diff)

    @stdlib("normalize", sig='normalize(v) \\u2192 vec', category='Vector', doc='Unit vector in the same direction.', ex='vec3 dir = normalize(vec3(u-0.5, v-0.5, 1.0));')
    @staticmethod
    def fn_normalize(v) -> torch.Tensor:
        # Pinned convention (the single oracle both backends use): a vector
        # (channel axis present) normalizes to unit length; a scalar / 1-channel
        # field normalizes to sign() — so sign(0)=0, and NOT x/abs(x) which is
        # NaN at 0. The type checker rejects normalize() on a non-vector
        # (E5003), and codegen routes scalar args here too, so interpreter and
        # codegen never diverge.
        t = _to_tensor(v)
        if _has_channel_axis(t):
            norm = torch.linalg.vector_norm(t, dim=-1, keepdim=True)
            return t / (norm + SAFE_EPSILON)
        return torch.sign(t)

    @stdlib("cross", sig='cross(a, b) \\u2192 vec3', category='Vector', doc='Cross product of two vec3 vectors.', ex='vec3 n = cross(tangent, bitangent);')
    @staticmethod
    def fn_cross(a, b) -> torch.Tensor:
        """Cross product. Only works on vec3 (last dim = 3)."""
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        # Take first 3 channels if vec4
        if a_t.shape[-1] == 4:
            a_t = a_t[..., :3]
        if b_t.shape[-1] == 4:
            b_t = b_t[..., :3]
        return torch.cross(a_t, b_t, dim=-1)

    @stdlib("reflect", sig='reflect(v, n) \\u2192 vec', category='Vector', doc='Reflect vector v around normal n.', ex='vec3 r = reflect(incoming, normal);')
    @staticmethod
    def fn_reflect(incident, normal) -> torch.Tensor:
        i, n = _to_tensor(incident), _to_tensor(normal)
        d = (i * n).sum(dim=-1, keepdim=True)
        return i - 2.0 * d * n

    # -- Color operations -----------------------------------------------

    @stdlib("luma", sig='luma(rgb) \\u2192 float', category='Color', doc='Perceptual luminance of an RGB color.', ex='float gray = luma(@image);')
    @staticmethod
    def fn_luma(color) -> torch.Tensor:
        """Compute luminance from RGB(A). Returns scalar per pixel."""
        c = _to_tensor(color)
        if c.dim() >= 1 and c.shape[-1] >= 3:
            return LUMA_R * c[..., 0] + LUMA_G * c[..., 1] + LUMA_B * c[..., 2]
        return c

    @stdlib("hsv2rgb", sig='hsv2rgb(hsv) \\u2192 vec3', category='Color', doc='Convert HSV color to RGB.', ex='vec3 rgb = hsv2rgb(vec3(u, 1.0, 1.0));')
    @staticmethod
    def fn_hsv2rgb(hsv) -> torch.Tensor:
        """Convert HSV to RGB. Expects vec3 [H, S, V] with H in [0, 1]."""
        c = _to_tensor(hsv)
        h = c[..., 0:1] * 6.0  # scale to [0, 6]
        s = c[..., 1:2]
        v = c[..., 2:3]

        i = torch.floor(h)
        f = h - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        i_mod = torch.fmod(i, 6.0)

        # Compute masks once (shared across all 3 channels)
        # instead of 5-deep nested torch.where (which repeats comparisons 3×)
        m0 = (i_mod == 0.0)
        m1 = (i_mod == 1.0)
        m2 = (i_mod == 2.0)
        m3 = (i_mod == 3.0)
        m4 = (i_mod == 4.0)

        # r: 0->v, 1->q, 2->p, 3->p, 4->t, 5->v  (default v)
        r = torch.where(m1, q, torch.where(m2 | m3, p, torch.where(m4, t, v)))
        # g: 0->t, 1->v, 2->v, 3->q, 4->p, 5->p  (default p)
        g = torch.where(m1 | m2, v, torch.where(m3, q, torch.where(m4, p,
            torch.where(m0, t, p))))
        # b: 0->p, 1->p, 2->t, 3->v, 4->v, 5->q  (default q)
        b = torch.where(m1, p, torch.where(m2, t, torch.where(m3 | m4, v,
            torch.where(m0, p, q))))

        result = torch.cat([r, g, b], dim=-1)
        # If input was vec4, preserve alpha
        if c.shape[-1] == 4:
            result = torch.cat([result, c[..., 3:4]], dim=-1)
        return result

    @stdlib("rgb2hsv", sig='rgb2hsv(rgb) \\u2192 vec3', category='Color', doc='Convert RGB color to HSV.', ex='vec3 hsv = rgb2hsv(@image);')
    @staticmethod
    def fn_rgb2hsv(rgb) -> torch.Tensor:
        """Convert RGB to HSV. Returns vec3 [H, S, V] with H in [0, 1]."""
        c = _to_tensor(rgb)
        r, g, b = c[..., 0:1], c[..., 1:2], c[..., 2:3]

        cmax = torch.maximum(torch.maximum(r, g), b)
        cmin = torch.minimum(torch.minimum(r, g), b)
        diff = cmax - cmin + SAFE_EPSILON

        # Hue
        h = torch.where(cmax == r, torch.fmod((g - b) / diff, 6.0),
            torch.where(cmax == g, (b - r) / diff + 2.0,
                                   (r - g) / diff + 4.0))
        h = h / 6.0  # normalize to [0, 1]
        h = torch.fmod(h + 1.0, 1.0)  # ensure positive

        # Saturation
        s = torch.where(cmax > SAFE_EPSILON, diff / cmax, cmax.new_zeros(()))

        # Value
        v = cmax

        result = torch.cat([h, s, v], dim=-1)
        if c.shape[-1] == 4:
            result = torch.cat([result, c[..., 3:4]], dim=-1)
        return result

    # -- Color management (SL-3): sRGB<->linear + OKLab -----------------
    # Blurring/blending in gamma space produces wrong halos; convert to
    # linear-light first. OKLab gives perceptually-uniform gradients/mixes.
    # Each is elementwise and preserves a vec4 alpha unchanged.

    @stdlib("srgb_to_linear", sig='srgb_to_linear(c) \\u2192 vec', category='Color', doc='Gamma-encoded sRGB → linear-light. Blur/blend in linear to avoid halos.', ex='vec3 lin = srgb_to_linear(@image.rgb);')
    @staticmethod
    def fn_srgb_to_linear(color) -> torch.Tensor:
        """sRGB EOTF: gamma-encoded sRGB -> linear-light (piecewise). vec4 alpha
        passes through. Compose before blur/blend, then linear_to_srgb after."""
        c = _to_tensor(color)
        has_alpha = c.dim() >= 1 and c.shape[-1] == 4
        rgb = c[..., 0:3] if has_alpha else c
        lin = torch.where(rgb <= 0.04045, rgb / 12.92,
                          ((rgb + 0.055) / 1.055).clamp(min=0.0) ** 2.4)
        return torch.cat([lin, c[..., 3:4]], dim=-1) if has_alpha else lin

    @stdlib("linear_to_srgb", sig='linear_to_srgb(c) \\u2192 vec', category='Color', doc='Linear-light → gamma-encoded sRGB (inverse of srgb_to_linear).', ex='@OUT = vec4(linear_to_srgb(lin), 1.0);')
    @staticmethod
    def fn_linear_to_srgb(color) -> torch.Tensor:
        """sRGB OETF: linear-light -> gamma-encoded sRGB (inverse of
        srgb_to_linear). vec4 alpha passes through."""
        c = _to_tensor(color)
        has_alpha = c.dim() >= 1 and c.shape[-1] == 4
        rgb = c[..., 0:3] if has_alpha else c
        srgb = torch.where(rgb <= 0.0031308, rgb * 12.92,
                           1.055 * rgb.clamp(min=0.0) ** (1.0 / 2.4) - 0.055)
        return torch.cat([srgb, c[..., 3:4]], dim=-1) if has_alpha else srgb

    @stdlib("oklab_from_rgb", sig='oklab_from_rgb(c) \\u2192 vec3', category='Color', doc='Linear RGB → OKLab. Mix/interpolate in OKLab for perceptually-even gradients.', ex='vec3 lab = oklab_from_rgb(srgb_to_linear(@image.rgb));')
    @staticmethod
    def fn_oklab_from_rgb(color) -> torch.Tensor:
        """Linear-light RGB -> OKLab (Ottosson). Mix/interpolate in OKLab then
        convert back for perceptually-even gradients. Expects LINEAR RGB — compose
        with srgb_to_linear for gamma-encoded images. vec4 alpha passes through."""
        c = _to_tensor(color)
        has_alpha = c.dim() >= 1 and c.shape[-1] == 4
        r, g, b = c[..., 0:1], c[..., 1:2], c[..., 2:3]
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        l_ = torch.sign(l) * torch.abs(l).pow(1.0 / 3.0)
        m_ = torch.sign(m) * torch.abs(m).pow(1.0 / 3.0)
        s_ = torch.sign(s) * torch.abs(s).pow(1.0 / 3.0)
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        lab = torch.cat([L, A, B], dim=-1)
        return torch.cat([lab, c[..., 3:4]], dim=-1) if has_alpha else lab

    @stdlib("oklab_to_rgb", sig='oklab_to_rgb(lab) \\u2192 vec3', category='Color', doc='OKLab → linear RGB (inverse of oklab_from_rgb).', ex='vec3 rgb = oklab_to_rgb(lab);')
    @staticmethod
    def fn_oklab_to_rgb(color) -> torch.Tensor:
        """OKLab -> linear-light RGB (inverse Ottosson). Compose with
        linear_to_srgb for a gamma-encoded result. vec4 alpha passes through."""
        c = _to_tensor(color)
        has_alpha = c.dim() >= 1 and c.shape[-1] == 4
        L, A, B = c[..., 0:1], c[..., 1:2], c[..., 2:3]
        l_ = L + 0.3963377774 * A + 0.2158037573 * B
        m_ = L - 0.1055613458 * A - 0.0638541728 * B
        s_ = L - 0.0894841775 * A - 1.2914855480 * B
        l, m, s = l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_
        r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        rgb = torch.cat([r, g, b], dim=-1)
        return torch.cat([rgb, c[..., 3:4]], dim=-1) if has_alpha else rgb

    # -- Compositing (SL-1): Porter-Duff on straight (un-premultiplied) vec4 --
    # ComfyUI IMAGE/MASK are un-premultiplied; over/under/atop take & return
    # straight-alpha vec4. premultiply/unpremultiply convert between conventions.

    @stdlib("premultiply", sig='premultiply(rgba) \\u2192 vec4', category='Color', doc='Straight → premultiplied alpha (rgb *= a).', ex='vec4 p = premultiply(@image);')
    @staticmethod
    def fn_premultiply(color) -> torch.Tensor:
        """Straight -> premultiplied alpha: rgb *= a (vec4)."""
        c = _to_tensor(color)
        a = c[..., 3:4]
        return torch.cat([c[..., 0:3] * a, a], dim=-1)

    @stdlib("unpremultiply", sig='unpremultiply(rgba) \\u2192 vec4', category='Color', doc='Premultiplied → straight alpha (rgb /= a).', ex='vec4 s = unpremultiply(p);')
    @staticmethod
    def fn_unpremultiply(color) -> torch.Tensor:
        """Premultiplied -> straight alpha: rgb /= a (vec4; safe at a=0, incl. fp16)."""
        c = _to_tensor(color)
        a = c[..., 3:4]
        return torch.cat([TEXStdlib._safe_div(c[..., 0:3], a), a], dim=-1)

    @stdlib("over", sig='over(fg, bg) \\u2192 vec4', category='Color', doc="Porter-Duff 'over': composite fg atop bg (straight-alpha RGBA).", ex='@OUT = over(@A, @B);')
    @staticmethod
    def fn_over(fg, bg) -> torch.Tensor:
        """Porter-Duff 'over': fg composited over bg (straight-alpha vec4)."""
        f = _to_tensor(fg)
        b = _to_tensor(bg)
        fa, ba = f[..., 3:4], b[..., 3:4]
        oa = fa + ba * (1.0 - fa)
        orgb = TEXStdlib._safe_div(f[..., 0:3] * fa + b[..., 0:3] * ba * (1.0 - fa), oa)
        return torch.cat([orgb, oa], dim=-1)

    @stdlib("under", sig='under(fg, bg) \\u2192 vec4', category='Color', doc='Composite fg under bg (= over(bg, fg)).', ex='@OUT = under(@A, @B);')
    @staticmethod
    def fn_under(fg, bg) -> torch.Tensor:
        """'under': fg under bg == over(bg, fg)."""
        return TEXStdlib.fn_over(bg, fg)

    @stdlib("atop", sig='atop(fg, bg) \\u2192 vec4', category='Color', doc="'atop': fg confined to bg's coverage.", ex='@OUT = atop(@A, @B);')
    @staticmethod
    def fn_atop(fg, bg) -> torch.Tensor:
        """'atop': fg atop bg — output confined to bg's coverage (out_a = bg.a)."""
        f = _to_tensor(fg)
        b = _to_tensor(bg)
        fa = f[..., 3:4]
        orgb = f[..., 0:3] * fa + b[..., 0:3] * (1.0 - fa)
        return torch.cat([orgb, b[..., 3:4]], dim=-1)

    # -- Blend modes (SL-2): per-channel, curated ~8 --------------------
    # Each op(base, blend) works on RGB channels; a vec4 base keeps its alpha.

    @staticmethod
    def _blend_rgb(base, blend, op):
        b = _to_tensor(base)
        s = _to_tensor(blend)
        n = min(b.shape[-1], 3)
        rgb = op(b[..., :n], s[..., :n])
        return torch.cat([rgb, b[..., 3:4]], dim=-1) if b.shape[-1] == 4 else rgb

    @staticmethod
    def _safe_div(num, denom):
        """num / denom with a DTYPE-AWARE, SIGN-PRESERVING zero floor on denom.

        The epsilon is dtype-aware because SAFE_EPSILON (1e-8) underflows to 0 in
        fp16 (ZERO_GUARD_EPS uses fp16's smallest normal there; fp32 keeps 1e-8).

        It floors the MAGNITUDE, not the signed value: `denom.clamp(min=eps)` would
        raise a small NEGATIVE denominator up to +eps — flipping the sign and
        blowing up the quotient (wrong for over/unpremultiply when an alpha goes
        out of [0,1], e.g. a mask subtraction dipping below zero). Here a
        below-threshold denominator is replaced by ±eps carrying denom's own sign."""
        eps = ZERO_GUARD_EPS.get(denom.dtype, SAFE_EPSILON)
        eps_t = torch.as_tensor(eps, dtype=denom.dtype, device=denom.device)
        below = denom.abs() < eps
        guard_trace.note(below)  # C4-ux (no-op unless armed)
        safe = torch.where(below, torch.copysign(eps_t, denom), denom)
        return num / safe

    @stdlib("screen", sig='screen(a, b) \\u2192 vec', category='Color', doc='Screen blend: 1 - (1-a)(1-b). Brightens.', ex='@OUT = vec4(screen(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_screen(base, blend) -> torch.Tensor:
        """1 - (1-a)(1-b)."""
        return TEXStdlib._blend_rgb(base, blend, lambda a, b: 1.0 - (1.0 - a) * (1.0 - b))

    @stdlib("overlay", sig='overlay(a, b) \\u2192 vec', category='Color', doc='Overlay blend (multiply/screen by base).', ex='@OUT = vec4(overlay(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_overlay(base, blend) -> torch.Tensor:
        """a<0.5 ? 2ab : 1-2(1-a)(1-b)."""
        return TEXStdlib._blend_rgb(base, blend, lambda a, b: torch.where(
            a < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b)))

    @stdlib("hard_light", sig='hard_light(a, b) \\u2192 vec', category='Color', doc='Hard-light blend (overlay with operands swapped).', ex='@OUT = vec4(hard_light(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_hard_light(base, blend) -> torch.Tensor:
        """overlay with the operands swapped."""
        return TEXStdlib._blend_rgb(base, blend, lambda a, b: torch.where(
            b < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b)))

    @stdlib("soft_light", sig='soft_light(a, b) \\u2192 vec', category='Color', doc='Soft-light blend (Pegtop, smooth).', ex='@OUT = vec4(soft_light(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_soft_light(base, blend) -> torch.Tensor:
        """Pegtop soft-light: (1-2b)a^2 + 2ab (smooth, no branch)."""
        return TEXStdlib._blend_rgb(base, blend,
                                    lambda a, b: (1.0 - 2.0 * b) * a * a + 2.0 * a * b)

    @stdlib("color_dodge", sig='color_dodge(a, b) \\u2192 vec', category='Color', doc='Color-dodge: brightens base by blend.', ex='@OUT = vec4(color_dodge(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_color_dodge(base, blend) -> torch.Tensor:
        """min(1, a / (1-b)); b>=1 -> 1."""
        return TEXStdlib._blend_rgb(base, blend, lambda a, b: torch.clamp(
            TEXStdlib._safe_div(a, 1.0 - b), max=1.0))

    @stdlib("color_burn", sig='color_burn(a, b) \\u2192 vec', category='Color', doc='Color-burn: darkens base by blend.', ex='@OUT = vec4(color_burn(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_color_burn(base, blend) -> torch.Tensor:
        """1 - min(1, (1-a)/b); b<=0 -> 0."""
        return TEXStdlib._blend_rgb(base, blend, lambda a, b: 1.0 - torch.clamp(
            TEXStdlib._safe_div(1.0 - a, b), max=1.0))

    @stdlib("linear_light", sig='linear_light(a, b) \\u2192 vec', category='Color', doc='Linear-light blend: clamp(a + 2b - 1).', ex='@OUT = vec4(linear_light(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_linear_light(base, blend) -> torch.Tensor:
        """clamp(a + 2b - 1, 0, 1)."""
        return TEXStdlib._blend_rgb(base, blend,
                                    lambda a, b: torch.clamp(a + 2.0 * b - 1.0, 0.0, 1.0))

    @stdlib("vivid_light", sig='vivid_light(a, b) \\u2192 vec', category='Color', doc='Vivid-light blend (burn/dodge by blend).', ex='@OUT = vec4(vivid_light(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_vivid_light(base, blend) -> torch.Tensor:
        """b<0.5 -> color_burn(a,2b); else color_dodge(a,2(b-0.5))."""
        def _op(a, b):
            burn = 1.0 - torch.clamp(TEXStdlib._safe_div(1.0 - a, 2.0 * b), max=1.0)
            dodge = torch.clamp(TEXStdlib._safe_div(a, 1.0 - 2.0 * (b - 0.5)), max=1.0)
            return torch.where(b < 0.5, burn, dodge)
        return TEXStdlib._blend_rgb(base, blend, _op)

    # -- Morphology (SL-4): erode / dilate ------------------------------
    # Iterative separable 3-window min/max, `radius` times. A square structuring
    # element is separable, and iterating a 3-window r times == a (2r+1)-window,
    # so this is O(1) extra memory in the radius (a 3-tensor transient per pass) —
    # avoiding the O((2r+1)^2) unfold blow-up at large radius/resolution. Replaces
    # the hand-rolled interpreted double loop that was radius-capped by the
    # 1024-iteration limit. Non-local (reads neighbours): excluded from tiling and
    # from CUDA-graph capture (the radius resolves via .item()).

    @staticmethod
    def _morph(image, radius, grow: bool):
        img = _to_tensor(image)
        r = max(0, min(int(_to_float(radius)), 256))
        if r == 0:
            return img
        squeeze = img.dim() == 3          # [B,H,W] mask -> add a channel
        x = (img.unsqueeze(-1) if squeeze else img).permute(0, 3, 1, 2)  # [B,C,H,W]
        op = torch.amax if grow else torch.amin
        pad = torch.nn.functional.pad
        for _ in range(r):
            xp = pad(x, (1, 1, 0, 0), mode="replicate")               # horizontal
            x = op(torch.stack([xp[..., :-2], xp[..., 1:-1], xp[..., 2:]]), dim=0)
            xp = pad(x, (0, 0, 1, 1), mode="replicate")               # vertical
            x = op(torch.stack([xp[..., :-2, :], xp[..., 1:-1, :], xp[..., 2:, :]]), dim=0)
        x = x.permute(0, 2, 3, 1)         # [B,H,W,C]
        return x.squeeze(-1) if squeeze else x

    @stdlib("erode", sig='erode(img, radius) \\u2192 vec', category='Sampling', sync=True, footprint=('halo_arg', 1), doc='Morphological erosion (local min over a (2r+1)² square). Shrinks bright regions.', ex='@OUT = erode(@mask, 3);')
    @staticmethod
    def fn_erode(image, radius) -> torch.Tensor:
        """Grayscale erosion (local min over a (2r+1)² square). Shrinks bright
        regions; the classic mask-shrink op."""
        return TEXStdlib._morph(image, radius, grow=False)

    @stdlib("dilate", sig='dilate(img, radius) \\u2192 vec', category='Sampling', sync=True, footprint=('halo_arg', 1), doc='Morphological dilation (local max). Grows bright regions.', ex='@OUT = dilate(@mask, 3);')
    @staticmethod
    def fn_dilate(image, radius) -> torch.Tensor:
        """Grayscale dilation (local max over a (2r+1)² square). Grows bright
        regions; the classic mask-grow op."""
        return TEXStdlib._morph(image, radius, grow=True)

    # -- Sampling -------------------------------------------------------

    @stdlib("sample", sig='sample(img, u, v) \\u2192 vec', category='Sampling', spatial=True, footprint='image', doc='Bilinear sample at normalized UV coordinates.', ex='@OUT = sample(@A, u + 0.01, v);')
    @staticmethod
    def fn_sample(image, u_coord, v_coord) -> torch.Tensor:
        """Sample an image at (u, v) coordinates using bilinear interpolation.

        Args:
            image: [B, H, W, C] tensor
            u_coord: float or [B, H, W] tensor — horizontal coordinate [0, 1]
            v_coord: float or [B, H, W] tensor — vertical coordinate [0, 1]

        Uses torch.nn.functional.grid_sample for fused C++ bilinear interpolation.
        """
        # Fast path: skip _to_tensor when inputs are already tensors
        img = image if image.__class__ is torch.Tensor else _to_tensor(image)
        u = u_coord if u_coord.__class__ is torch.Tensor else _to_tensor(u_coord)
        v = v_coord if v_coord.__class__ is torch.Tensor else _to_tensor(v_coord)

        B, H, W, C = img.shape

        # grid_sample expects [B, C, H, W] input
        img_bchw = _get_bchw(img)

        # Convert from [0, 1] UV to [-1, 1] grid coords (grid_sample convention)
        grid_x = u * 2.0 - 1.0
        grid_y = v * 2.0 - 1.0

        # Build grid: needs shape [B, H_out, W_out, 2]
        gd = grid_x.dim()
        if gd == 0:
            grid_x = grid_x.expand(B, H, W)
            grid_y = grid_y.expand(B, H, W)
        elif gd == 2:
            grid_x = grid_x.unsqueeze(0).expand(B, H, W)
            grid_y = grid_y.unsqueeze(0).expand(B, H, W)

        # Reuse pre-allocated grid buffer to avoid torch.stack allocation
        grid = _get_grid_buf(B, H, W, img.device)
        grid[..., 0] = grid_x
        grid[..., 1] = grid_y

        # Sample with bilinear interpolation (fused C++ kernel)
        result_bchw = _grid_sample_f32(
            img_bchw, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )

        # Back to [B, H, W, C]
        return result_bchw.permute(0, 2, 3, 1)

    @stdlib("fetch", sig='fetch(img, px, py) \\u2192 vec', category='Sampling', spatial=True, footprint='image', doc='Nearest-neighbor fetch at pixel coordinates.', ex='@OUT = fetch(@A, ix, iy);')
    @staticmethod
    def fn_fetch(image, px, py) -> torch.Tensor:
        """Fetch a pixel at integer coordinates (nearest-neighbor).

        Args:
            image: [B, H, W, C] tensor
            px: int/float or [B, H, W] tensor — horizontal pixel coordinate
            py: int/float or [B, H, W] tensor — vertical pixel coordinate

        Coordinates are clamped to valid range. Use with ix/iy built-ins
        for neighbor access patterns like fetch(@A, ix+1, iy).
        """
        # Fast path: skip _to_tensor when inputs are already float32 tensors
        img = image if image.__class__ is torch.Tensor else _to_tensor(image)
        px_t = px if px.__class__ is torch.Tensor else _to_tensor(px)
        py_t = py if py.__class__ is torch.Tensor else _to_tensor(py)

        B, H, W, C = img.shape

        # Clamp float then convert to int — faster than .long() then clamp
        px_i = px_t.clamp(0, W - 1).to(torch.int64)
        py_i = py_t.clamp(0, H - 1).to(torch.int64)

        # B=1 fast path: flat index is ~40% faster than 2D advanced indexing
        # for spatial-sized coordinate tensors (the common fetch() case).
        if B == 1:
            px_d = px_i.dim()
            py_d = py_i.dim()
            # Only use flat index when at least one coord is spatial (dim >= 2).
            # Scalar coords (dim 0) are rare and need special shape handling.
            if px_d >= 2 or py_d >= 2:
                # Expand BOTH coords to [H,W] (mirroring the B>=2 path below) so
                # the flat index always spans the full grid. Without this, a mixed
                # spatial+scalar fetch like @A[ix, ih-1.0] — where ix broadcasts as
                # [1,W] — collapses the H axis (wrong shape at B=1 only). expand is
                # a view, so the fast path stays fast.
                px_f = _expand_to_bhw(px_i, 1, H, W)[0]
                py_f = _expand_to_bhw(py_i, 1, H, W)[0]
                flat = py_f * W + px_f
                return torch.index_select(img.view(H * W, C), 0, flat.reshape(-1)).view(1, H, W, C)
            # Scalar coords: fall through to advanced indexing
            px_i = px_i.expand(1, H, W)
            py_i = py_i.expand(1, H, W)
            return img[0, py_i[0], px_i[0]].unsqueeze(0)

        # Multi-batch: expand coordinates to [B, H, W]
        px_i = _expand_to_bhw(px_i, B, H, W)
        py_i = _expand_to_bhw(py_i, B, H, W)

        return img[_get_batch_index(B, H, W, img.device), py_i, px_i]

    @stdlib("fetch_frame", sig='fetch_frame(img, frame, px, py) \\u2192 vec', category='Batch / Temporal', spatial=True, footprint=('frame', 1), doc='Nearest-neighbor fetch from a specific batch frame.', ex='@OUT = fetch_frame(@A, fi-1, ix, iy);')
    @staticmethod
    def fn_fetch_frame(image, frame, px, py) -> torch.Tensor:
        """Fetch a pixel from a specific frame at integer coordinates.

        Unlike fetch(), which reads each frame from itself (B-diagonal),
        fetch_frame() allows cross-frame access via the frame parameter.

        Args:
            image: [B, H, W, C] tensor
            frame: float or [B, H, W] tensor — target frame index (clamped to [0, B-1])
            px: int/float or [B, H, W] tensor — horizontal pixel coordinate
            py: int/float or [B, H, W] tensor — vertical pixel coordinate
        """
        img = _to_tensor(image)
        frame_t = _to_tensor(frame)
        px_t = _to_tensor(px)
        py_t = _to_tensor(py)

        B, H, W, C = img.shape

        # Round and clamp all indices
        f_idx = torch.clamp(torch.round(frame_t).long(), 0, B - 1)
        px_i = torch.clamp(torch.round(px_t).long(), 0, W - 1)
        py_i = torch.clamp(torch.round(py_t).long(), 0, H - 1)

        # Expand scalars/2D to [B, H, W]
        f_idx = _expand_to_bhw(f_idx, B, H, W)
        px_i = _expand_to_bhw(px_i, B, H, W)
        py_i = _expand_to_bhw(py_i, B, H, W)

        return img[f_idx, py_i, px_i]

    @stdlib("sample_frame", sig='sample_frame(img, frame, u, v) \\u2192 vec', category='Batch / Temporal', spatial=True, footprint=('frame', 1), doc='Bilinear sample from a specific batch frame.', ex='@OUT = sample_frame(@A, 0, u, v);')
    @staticmethod
    def fn_sample_frame(image, frame, u_coord, v_coord) -> torch.Tensor:
        """Sample from a specific frame using bilinear interpolation.

        Unlike sample(), which reads each frame from itself (B-diagonal),
        sample_frame() allows cross-frame access via the frame parameter.

        Args:
            image: [B, H, W, C] tensor
            frame: float or [B, H, W] tensor — target frame index (clamped to [0, B-1])
            u_coord: float or [B, H, W] tensor — horizontal coordinate [0, 1]
            v_coord: float or [B, H, W] tensor — vertical coordinate [0, 1]
        """
        img = _to_tensor(image)
        frame_t = _to_tensor(frame)
        u = _to_tensor(u_coord)
        v = _to_tensor(v_coord)

        B, H, W, C = img.shape

        # Resolve frame index
        f_idx = torch.clamp(torch.round(frame_t).long(), 0, B - 1)
        f_idx = _expand_to_bhw(f_idx, B, H, W)

        # Convert from [0,1] to pixel coordinates
        x = _expand_to_bhw(u * (W - 1), B, H, W)
        y = _expand_to_bhw(v * (H - 1), B, H, W)

        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)

        x0 = torch.floor(x).long()
        x1 = torch.clamp(x0 + 1, 0, W - 1)
        y0 = torch.floor(y).long()
        y1 = torch.clamp(y0 + 1, 0, H - 1)

        fx = (x - x0.float()).unsqueeze(-1)
        fy = (y - y0.float()).unsqueeze(-1)

        v00 = img[f_idx, y0, x0]
        v01 = img[f_idx, y0, x1]
        v10 = img[f_idx, y1, x0]
        v11 = img[f_idx, y1, x1]

        result = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + \
                 v10 * (1 - fx) * fy + v11 * fx * fy
        return result

    @stdlib("sample_cubic", sig='sample_cubic(img, u, v) \\u2192 vec', category='Sampling', spatial=True, footprint='image', doc='Bicubic (Catmull-Rom) sampling.', ex='@OUT = sample_cubic(@A, u, v);')
    @staticmethod
    def fn_sample_cubic(image, u_coord, v_coord) -> torch.Tensor:
        """Sample an image at (u, v) coordinates using bicubic (Catmull-Rom) interpolation.

        Args:
            image: [B, H, W, C] tensor
            u_coord: float or [B, H, W] tensor — horizontal coordinate [0, 1]
            v_coord: float or [B, H, W] tensor — vertical coordinate [0, 1]

        Uses torch.nn.functional.grid_sample with mode='bicubic' for
        high-quality upsampling with smoother gradients than bilinear.
        """
        img = _to_tensor(image)
        u = _to_tensor(u_coord)
        v = _to_tensor(v_coord)

        B, H, W, C = img.shape

        # grid_sample expects [B, C, H, W] input
        img_bchw = _get_bchw(img)

        grid = _build_sample_grid(u, v, B, H, W)

        # Sample with bicubic interpolation
        result_bchw = _grid_sample_f32(
            img_bchw, grid,
            mode='bicubic',
            padding_mode='border',
            align_corners=True,
        )

        # Back to [B, H, W, C]
        return result_bchw.permute(0, 2, 3, 1)

    @stdlib("sample_lanczos", sig='sample_lanczos(img, u, v) \\u2192 vec', category='Sampling', spatial=True, footprint='image', doc='Lanczos-3 high-quality sampling.', ex='@OUT = sample_lanczos(@A, u * 0.5, v * 0.5);')
    @staticmethod
    def fn_sample_lanczos(image, u_coord, v_coord) -> torch.Tensor:
        """Sample an image at (u, v) coordinates using Lanczos-3 interpolation.

        Args:
            image: [B, H, W, C] tensor
            u_coord: float or [B, H, W] tensor — horizontal coordinate [0, 1]
            v_coord: float or [B, H, W] tensor — vertical coordinate [0, 1]

        Lanczos-3 uses a 6×6 pixel neighborhood with sinc-based weights.
        Uses flat gather on a [B, H*W, C] view to avoid expanding 5-D index
        grids, then reshapes for weight application.
        """
        img = _to_tensor(image)
        u = _to_tensor(u_coord)
        v = _to_tensor(v_coord)

        B, H, W, C = img.shape
        dev = img.device

        # Convert from [0, 1] to pixel coordinates
        x = u * (W - 1)
        y = v * (H - 1)

        # Center pixel (integer part)
        x_floor = torch.floor(x)
        y_floor = torch.floor(y)

        # Fractional part
        fx = x - x_floor
        fy = y - y_floor

        # Expand scalars to spatial dims
        if fx.dim() == 0:
            fx = fx.expand(B, H, W)
            fy = fy.expand(B, H, W)
            x_floor = x_floor.expand(B, H, W)
            y_floor = y_floor.expand(B, H, W)
        elif fx.dim() == 2:
            fx = fx.unsqueeze(0).expand(B, H, W)
            fy = fy.unsqueeze(0).expand(B, H, W)
            x_floor = x_floor.unsqueeze(0).expand(B, H, W)
            y_floor = y_floor.unsqueeze(0).expand(B, H, W)

        # Tap offsets: -2, -1, 0, 1, 2, 3 (6 taps for Lanczos-3)
        taps = _get_lanczos_taps(dev)  # [6] — cached

        # Compute 1-D Lanczos weights for x and y (separable)
        wx = _lanczos3(fx.unsqueeze(-1) - taps)  # [B, H, W, 6]
        wy = _lanczos3(fy.unsqueeze(-1) - taps)  # [B, H, W, 6]

        # 2-D weights via outer product: [B, H, W, 6, 6]
        weights_2d = torch.einsum('...i,...j->...ij', wy, wx)

        # Normalize
        w_sum = weights_2d.sum(dim=(-2, -1), keepdim=True).clamp(min=SAFE_EPSILON)
        weights_2d = weights_2d / w_sum  # [B,H,W,6,6]

        # Pixel coordinates for all 36 taps, clamped to image bounds
        px_all = torch.clamp((x_floor.unsqueeze(-1) + taps).long(), 0, W - 1)  # [B,H,W,6]
        py_all = torch.clamp((y_floor.unsqueeze(-1) + taps).long(), 0, H - 1)  # [B,H,W,6]

        # Compute flat pixel indices: py * W + px → [B,H,W,6,6]
        # Use views to broadcast: py[...,6,1] * W + px[...,1,6] → [B,H,W,6,6]
        flat_idx = py_all.unsqueeze(-1) * W + px_all.unsqueeze(-2)  # [B,H,W,6y,6x]
        flat_idx = flat_idx.reshape(B, H * W * 36)  # [B, N]

        # Gather from flattened image: [B, H*W, C]
        img_flat = img.reshape(B, H * W, C)
        # Expand flat_idx for channel dim: [B, N, C]
        idx_exp = flat_idx.unsqueeze(-1).expand(-1, -1, C)
        pixels_flat = torch.gather(img_flat, 1, idx_exp)  # [B, N, C]

        # Reshape back: [B, H, W, 6, 6, C]
        pixels = pixels_flat.reshape(B, H, W, 6, 6, C)

        # Apply weights: [B,H,W,6,6,1] * [B,H,W,6,6,C] → sum → [B,H,W,C]
        result = (pixels * weights_2d.unsqueeze(-1)).sum(dim=(3, 4))

        return result

    @stdlib("sample_mip", sig='sample_mip(img, u, v, lod) \\u2192 vec', category='Sampling', spatial=True, sync=True, footprint='image', doc='Mipmap sampling with LOD. 0 = full res, 1 = half, etc. Trilinear between levels.', ex='@OUT = sample_mip(@A, u, v, 2.5);')
    @staticmethod
    def fn_sample_mip(image, u_coord, v_coord, lod) -> torch.Tensor:
        """Sample an image with mipmap filtering at an explicit level of detail.

        Args:
            image: [B, H, W, C] tensor
            u_coord: float or [B, H, W] tensor — horizontal coordinate [0, 1]
            v_coord: float or [B, H, W] tensor — vertical coordinate [0, 1]
            lod: float or [B, H, W] tensor — mip level (0 = full res, 1 = half, ...)

        Builds a mipmap pyramid on first call (cached per input tensor).
        Uses bilinear sampling within each level and linear interpolation
        between levels (trilinear). Fast path when LOD is a uniform integer:
        samples a single level with no interpolation.
        """
        return _sample_mip_trilinear(image, u_coord, v_coord, lod, _get_mip_pyramid)

    @stdlib("gauss_blur", sig='gauss_blur(img, sigma) \\u2192 vec', category='Sampling', spatial=True, sync=True, footprint=('halo_arg', 1), doc='Separable Gaussian blur. Kernel radius ≈ 3×sigma pixels. Replicate border padding.', ex='@OUT = gauss_blur(@A, 2.0);')
    @staticmethod
    def fn_gauss_blur(image, sigma) -> torch.Tensor:
        """Separable Gaussian blur.

        Args:
            image: [B, H, W, C] tensor
            sigma: float — standard deviation in pixels (kernel radius ≈ 3*sigma)

        Returns blurred [B, H, W, C] tensor with replicate border handling.
        """
        img = image if image.__class__ is torch.Tensor else _to_tensor(image)
        sigma_t = sigma if sigma.__class__ is torch.Tensor else _to_tensor(sigma)
        # .item() forces a GPU->CPU sync, but the Gaussian kernel radius is a
        # host-side Python int (radius ~= 3*sigma), so a scalar is unavoidable.
        # Prefer a constant sigma so this fires once rather than per element.
        sigma_val = max(sigma_t.item(), 0.0)
        if sigma_val < 0.3 or img.dim() < 4:
            return img
        bchw = _get_bchw(img)
        result = _gauss_blur_bchw(bchw, sigma_val)
        return result.permute(0, 2, 3, 1)

    @stdlib("bilateral_filter", sig='bilateral_filter(img, spatial_sigma, range_sigma) \\u2192 vec', category='Sampling', spatial=True, sync=True, footprint=('halo', 3), doc='Edge-preserving smoothing: blurs within regions but keeps edges. Window capped at 7×7.', ex='@OUT = bilateral_filter(@A, 1.5, 0.2);')
    @staticmethod
    def fn_bilateral_filter(image, sigma_s, sigma_r) -> torch.Tensor:
        """Edge-preserving bilateral filter using Tensor.unfold.

        Weights each neighbor by spatial Gaussian x range (color similarity)
        Gaussian. Radius is derived from sigma_s (3x sigma, capped at 3 → 7x7).
        Best for small kernels (3x3); for larger kernels, the loop-based
        approach in bilateral_approx.tex may be faster due to memory traffic.

        Args:
            image: [B, H, W, C] tensor
            sigma_s: float -- spatial sigma in pixels
            sigma_r: float -- range sigma (color similarity, 0.01-0.5 typical)
        """
        img = image if image.__class__ is torch.Tensor else _to_tensor(image)
        ss = sigma_s.item() if torch.is_tensor(sigma_s) else float(sigma_s)
        sr = sigma_r.item() if torch.is_tensor(sigma_r) else float(sigma_r)

        if img.dim() < 4 or ss < 0.3:
            return img

        B, H, W, C = img.shape
        radius = min(int(math.ceil(3.0 * ss)), 3)  # cap at 7x7 to limit memory (~500MB at 1080p)
        ksize = 2 * radius + 1

        # Convert to BCHW and pad
        bchw = _get_bchw(img)
        padded = torch.nn.functional.pad(bchw, (radius, radius, radius, radius), mode='replicate')

        # Extract all ksize×ksize patches: [B, C, H, W, kH, kW]
        patches = padded.unfold(2, ksize, 1).unfold(3, ksize, 1)

        # Center pixel: [B, C, H, W, 1, 1]
        center = bchw.unsqueeze(-1).unsqueeze(-1)

        # Spatial weights: precomputed [1, 1, 1, 1, kH, kW]
        inv_2ss = -0.5 / max(ss * ss, 1e-10)
        dy = torch.arange(ksize, device=img.device, dtype=torch.float32) - radius
        dx = dy.clone()
        d2 = dy.view(-1, 1) ** 2 + dx.view(1, -1) ** 2  # [kH, kW]
        w_spatial = torch.exp(d2 * inv_2ss).view(1, 1, 1, 1, ksize, ksize)

        # Range weights: per-pixel, based on color distance
        # diff: [B, C, H, W, kH, kW]
        diff = patches - center
        # Color distance squared, summed over channels: [B, 1, H, W, kH, kW]
        inv_2sr = -0.5 / max(sr * sr, 1e-10)
        cd2 = (diff * diff).sum(dim=1, keepdim=True)
        w_range = torch.exp(cd2 * inv_2sr)

        # Combined weight: [B, 1, H, W, kH, kW]
        w = w_spatial * w_range

        # Weighted sum: [B, C, H, W]
        numerator = (patches * w).sum(dim=(-2, -1))
        denominator = w.sum(dim=(-2, -1))
        result = numerator / denominator.clamp(min=1e-10)

        return result.permute(0, 2, 3, 1)  # back to BHWC

    @stdlib("sample_mip_gauss", sig='sample_mip_gauss(img, u, v, lod) \\u2192 vec', category='Sampling', spatial=True, sync=True, footprint='image', doc='Gaussian-prefiltered mipmap sampling. Smoother pyramid (sigma=1.13) gives ~5 dB better exponential blur accuracy vs sample_mip.', ex='@OUT = sample_mip_gauss(@A, u, v, 2.5);')
    @staticmethod
    def fn_sample_mip_gauss(image, u_coord, v_coord, lod) -> torch.Tensor:
        """Sample with Gaussian-prefiltered mipmap (sigma=1.13 pyramid).

        Same interface as sample_mip but uses a Gaussian pre-blur before each
        2x downsample, producing SIGMA_C ≈ 0.825. This gives ~5 dB better
        accuracy for exponential blur reconstruction vs the area-downsample pyramid.
        """
        return _sample_mip_trilinear(image, u_coord, v_coord, lod, _get_mip_pyramid_gauss)

    # -- Noise functions ------------------------------------------------
    # All noise functions support optional z for 3D:
    #   2 args = 2D, 3 args = 3D (for base noise)
    #   3 args = 2D with octaves, 4 args = 3D with octaves (for FBM family)

    @stdlib("perlin", sig='perlin(x, y) \\u2192 float', category='Noise', doc='2D Perlin noise. Returns value in [-1, 1].', ex='float n = perlin(u * 10.0, v * 10.0);')
    @staticmethod
    def fn_perlin(x, y, z=None) -> torch.Tensor:
        """Perlin noise. 2D when z omitted, 3D when z provided. Returns float in ~[-1, 1]."""
        if z is not None:
            return _perlin3d_fast(_to_tensor(x), _to_tensor(y), _to_tensor(z))
        return _perlin2d_fast(_to_tensor(x), _to_tensor(y))

    @stdlib("simplex", sig='simplex(x, y) \\u2192 float', category='Noise', doc='2D Simplex noise. Returns value in [-1, 1].', ex='float n = simplex(u * 8.0, v * 8.0);')
    @staticmethod
    def fn_simplex(x, y, z=None) -> torch.Tensor:
        """Simplex noise. 2D when z omitted, 3D falls back to Perlin. Returns float in ~[-1, 1]."""
        if z is not None:
            return _perlin3d_fast(_to_tensor(x), _to_tensor(y), _to_tensor(z))
        return _simplex2d(_to_tensor(x), _to_tensor(y))

    @stdlib("fbm", sig='fbm(x, y, octaves) \\u2192 float', category='Noise', sync=True, doc='Fractal Brownian Motion (multi-octave Perlin).', ex='float n = fbm(u * 4.0, v * 4.0, 6);')
    @staticmethod
    def fn_fbm(x, y, z_or_oct, octaves=None) -> torch.Tensor:
        """FBM noise. fbm(x,y,octaves) for 2D, fbm(x,y,z,octaves) for 3D."""
        if octaves is not None:
            return _fbm3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                          int(_to_float(octaves)))
        return _fbm2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))

    @stdlib("worley_f1", sig='worley_f1(x, y) \\u2192 float', category='Noise', doc='Worley (cellular) noise — distance to nearest cell center.', ex='float n = worley_f1(u * 5.0, v * 5.0);')
    @staticmethod
    def fn_worley_f1(x, y, z=None) -> torch.Tensor:
        """Worley F1 noise (nearest cell distance). Returns float in ~[0, 1]."""
        if z is not None:
            return _worley3d(_to_tensor(x), _to_tensor(y), _to_tensor(z), return_f2=False)
        return _worley2d(_to_tensor(x), _to_tensor(y), return_f2=False)

    @stdlib("worley_f2", sig='worley_f2(x, y) \\u2192 float', category='Noise', doc='Worley noise — distance to second-nearest cell center.', ex='float n = worley_f2(u * 5.0, v * 5.0);')
    @staticmethod
    def fn_worley_f2(x, y, z=None) -> torch.Tensor:
        """Worley F2 noise (2nd nearest cell distance). Returns float in ~[0, 1]."""
        if z is not None:
            return _worley3d(_to_tensor(x), _to_tensor(y), _to_tensor(z), return_f2=True)
        return _worley2d(_to_tensor(x), _to_tensor(y), return_f2=True)

    @stdlib("voronoi", sig='voronoi(x, y) \\u2192 float', category='Noise', doc='Voronoi cell ID noise. Returns a unique value per cell.', ex='float cell = voronoi(u * 8.0, v * 8.0);')
    @staticmethod
    def fn_voronoi(x, y, z=None) -> torch.Tensor:
        """Voronoi noise (alias for worley_f1)."""
        return TEXStdlib.fn_worley_f1(x, y, z)

    @stdlib("curl", sig='curl(x, y) \\u2192 vec2', category='Noise', doc='Curl of 2D noise field. Returns a divergence-free vector.', ex='vec2 c = curl(u * 5.0, v * 5.0);')
    @staticmethod
    def fn_curl(x, y, z=None) -> torch.Tensor:
        """Curl noise. 2D → vec2 (divergence-free), 3D → vec3."""
        if z is not None:
            return _curl3d(_to_tensor(x), _to_tensor(y), _to_tensor(z))
        return _curl2d(_to_tensor(x), _to_tensor(y))

    @stdlib("ridged", sig='ridged(x, y, octaves) \\u2192 float', category='Noise', sync=True, doc='Ridged multifractal — sharp ridges, good for mountains.', ex='float n = ridged(u * 4.0, v * 4.0, 6);')
    @staticmethod
    def fn_ridged(x, y, z_or_oct, octaves=None) -> torch.Tensor:
        """Ridged FBM. ridged(x,y,octaves) for 2D, ridged(x,y,z,octaves) for 3D."""
        if octaves is not None:
            return _ridged3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                             int(_to_float(octaves)))
        return _ridged2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))

    @stdlib("billow", sig='billow(x, y, octaves) \\u2192 float', category='Noise', sync=True, doc='Billowy noise — abs(fbm). Puffy cloud shapes.', ex='float n = billow(u * 4.0, v * 4.0, 6);')
    @staticmethod
    def fn_billow(x, y, z_or_oct, octaves=None) -> torch.Tensor:
        """Billow FBM. billow(x,y,octaves) for 2D, billow(x,y,z,octaves) for 3D."""
        if octaves is not None:
            return _billow3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                             int(_to_float(octaves)))
        return _billow2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))

    @stdlib("turbulence", sig='turbulence(x, y, octaves) \\u2192 float', category='Noise', sync=True, doc='Turbulence — sum of abs(noise) per octave. Veiny patterns.', ex='float n = turbulence(u * 4.0, v * 4.0, 6);')
    @staticmethod
    def fn_turbulence(x, y, z_or_oct, octaves=None) -> torch.Tensor:
        """Turbulence. turbulence(x,y,octaves) for 2D, turbulence(x,y,z,octaves) for 3D."""
        if octaves is not None:
            return _turbulence3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                                 int(_to_float(octaves)))
        return _turbulence2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))

    @stdlib("flow", sig='flow(x, y, angle) \\u2192 float', category='Noise', sync=True, doc='Flow noise — Perlin rotated by angle per octave. Avoids static patterns.', ex='float n = flow(u * 6.0, v * 6.0, fi * 0.1);')
    @staticmethod
    def fn_flow(x, y, z_or_time, time=None) -> torch.Tensor:
        """Flow noise. flow(x,y,time) for 2D, flow(x,y,z,time) for 3D."""
        if time is not None:
            return _flow3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_time),
                           _to_float(time))
        return _flow2d(_to_tensor(x), _to_tensor(y), _to_float(z_or_time))

    @stdlib("alligator", sig='alligator(x, y) \\u2192 float', category='Noise', sync=True, doc='Alligator noise — cellular crack patterns.', ex='float n = alligator(u * 5.0, v * 5.0);')
    @staticmethod
    def fn_alligator(x, y, z_or_oct=None, octaves=None) -> torch.Tensor:
        """Alligator noise. 2 args: 2D default octaves; 3 args: 2D custom octaves; 4 args: 3D."""
        if octaves is not None:
            return _alligator3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                                int(_to_float(octaves)))
        if z_or_oct is not None:
            return _alligator2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))
        return _alligator2d(_to_tensor(x), _to_tensor(y), 4)

    # -- SDF primitives -------------------------------------------------

    @stdlib("sdf_circle", sig='sdf_circle(px, py, radius) \\u2192 float', category='SDF & Smooth', doc='Signed distance to a circle centered at the origin (offset px/py to move it). Negative inside, positive outside.', ex='float d = sdf_circle(u - 0.5, v - 0.5, 0.3);')
    @staticmethod
    def fn_sdf_circle(px, py, radius) -> torch.Tensor:
        """Signed distance to a circle centered at origin."""
        return torch.hypot(_to_tensor(px), _to_tensor(py)) - _to_tensor(radius)

    @stdlib("sdf_box", sig='sdf_box(px, py, half_w, half_h) \\u2192 float', category='SDF & Smooth', doc='Signed distance to an axis-aligned box centered at the origin (half-extents half_w/half_h).', ex='float d = sdf_box(u - 0.5, v - 0.5, 0.2, 0.15);')
    @staticmethod
    def fn_sdf_box(px, py, half_w, half_h) -> torch.Tensor:
        """Signed distance to an axis-aligned box centered at origin."""
        dx = torch.abs(_to_tensor(px)) - _to_tensor(half_w)
        dy = torch.abs(_to_tensor(py)) - _to_tensor(half_h)
        dx_c = torch.clamp(dx, min=0.0)
        dy_c = torch.clamp(dy, min=0.0)
        outside = torch.sqrt(dx_c * dx_c + dy_c * dy_c)
        inside = torch.clamp(torch.max(dx, dy), max=0.0)
        return outside + inside

    @stdlib("sdf_line", sig='sdf_line(x, y, x1, y1, x2, y2) \\u2192 float', category='SDF & Smooth', doc='Distance to line segment.', ex='float d = sdf_line(u, v, 0.2, 0.2, 0.8, 0.8);')
    @staticmethod
    def fn_sdf_line(px, py, ax, ay, bx, by) -> torch.Tensor:
        """Unsigned distance to a line segment from (ax,ay) to (bx,by). Always >= 0."""
        px_t, py_t = _to_tensor(px), _to_tensor(py)
        ax_t, ay_t = _to_tensor(ax), _to_tensor(ay)
        bx_t, by_t = _to_tensor(bx), _to_tensor(by)
        pa_x = px_t - ax_t
        pa_y = py_t - ay_t
        ba_x = bx_t - ax_t
        ba_y = by_t - ay_t
        h = torch.clamp((pa_x * ba_x + pa_y * ba_y) / (ba_x * ba_x + ba_y * ba_y + SAFE_EPSILON), 0.0, 1.0)
        return torch.hypot(pa_x - ba_x * h, pa_y - ba_y * h)

    @stdlib("sdf_polygon", sig='sdf_polygon(px, py, radius, sides) \\u2192 float', category='SDF & Smooth', doc='Signed distance to a regular polygon (sides>=3) centered at the origin.', ex='float d = sdf_polygon(u - 0.5, v - 0.5, 0.3, 6);')
    @staticmethod
    def fn_sdf_polygon(px, py, radius, sides) -> torch.Tensor:
        """Signed distance to a regular polygon centered at origin."""
        px_t = _to_tensor(px)
        py_t = _to_tensor(py)
        r = _to_float(radius)
        n = max(int(_to_float(sides)), 3)
        an = math.pi / n  # half-angle of one segment
        cos_an = math.cos(an)
        full_an = 2.0 * an
        angle = torch.atan2(py_t, px_t)
        # Fold angle into one segment: [-an, an]
        sector = angle - full_an * torch.floor((angle + an) / full_an)
        dist = torch.sqrt(px_t * px_t + py_t * py_t) * torch.cos(sector) - r * cos_an
        return dist

    # -- Smooth min/max -------------------------------------------------

    @stdlib("smin", sig='smin(a, b, k) \\u2192 float|vec', category='SDF & Smooth', doc='Smooth minimum. Polynomial blending with radius k. Works on scalars and vectors.', ex='float d = smin(d1, d2, 0.1);')
    @staticmethod
    def fn_smin(a, b, k) -> torch.Tensor:
        """Polynomial smooth minimum with smoothing radius k."""
        a_t = _to_tensor(a)
        b_t = _to_tensor(b)
        k_t = _to_tensor(k)
        h = torch.clamp(0.5 + 0.5 * (b_t - a_t) / (k_t + SAFE_EPSILON), 0.0, 1.0)
        return _lerp_f32(b_t, a_t, h) - k_t * h * (1.0 - h)

    @stdlib("smax", sig='smax(a, b, k) \\u2192 float|vec', category='SDF & Smooth', doc='Smooth maximum. Polynomial blending with radius k. Works on scalars and vectors.', ex='float d = smax(d1, d2, 0.1);')
    @staticmethod
    def fn_smax(a, b, k) -> torch.Tensor:
        """Polynomial smooth maximum with smoothing radius k."""
        a_t = _to_tensor(a)
        b_t = _to_tensor(b)
        k_t = _to_tensor(k)
        h = torch.clamp(0.5 - 0.5 * (b_t - a_t) / (k_t + SAFE_EPSILON), 0.0, 1.0)
        return _lerp_f32(b_t, a_t, h) + k_t * h * (1.0 - h)

    # -- Gradient sampling -----------------------------------------------

    @stdlib("sample_grad", sig='sample_grad(img, u, v) \\u2192 vec2', category='Sampling', spatial=True, footprint='image', doc='Image gradient (Sobel) at UV. Returns vec2(dI/dx, dI/dy) of luminance.', ex='vec2 grad = sample_grad(@A, u, v);')
    @staticmethod
    def fn_sample_grad(image, u_coord, v_coord) -> torch.Tensor:
        """Sample the luminance gradient of an image at (u, v). Returns vec2 (dx, dy)."""
        img = _to_tensor(image)
        u = _to_tensor(u_coord)
        v = _to_tensor(v_coord)
        B, H, W, C = img.shape
        du = 1.0 / max(W - 1, 1)
        dv = 1.0 / max(H - 1, 1)
        s_right = TEXStdlib.fn_sample(img, u + du, v)
        s_left = TEXStdlib.fn_sample(img, u - du, v)
        s_down = TEXStdlib.fn_sample(img, u, v + dv)
        s_up = TEXStdlib.fn_sample(img, u, v - dv)
        # Luminance via Rec.709
        luma_r = TEXStdlib.fn_luma(s_right)
        luma_l = TEXStdlib.fn_luma(s_left)
        luma_d = TEXStdlib.fn_luma(s_down)
        luma_u = TEXStdlib.fn_luma(s_up)
        grad_x = (luma_r - luma_l) * 0.5
        grad_y = (luma_d - luma_u) * 0.5
        return torch.stack([grad_x, grad_y], dim=-1)

    # -- String functions -----------------------------------------------

    @stdlib("str", sig='str(x) \\u2192 string', category='Strings', doc='Convert a number to a string.', ex='string s = str(42);')
    @staticmethod
    def fn_str(x) -> str:
        """Convert number to string."""
        if isinstance(x, str):
            return x
        if isinstance(x, torch.Tensor):
            v = _scalar_from_tensor(x, "str")
            return str(int(v)) if v == int(v) else str(v)
        return str(x)

    @stdlib("len", sig='len(x) \\u2192 float', category='Strings', doc='Length of a string, array, or vec-array (element count).', ex='float n = len("hello");')
    @staticmethod
    def fn_len(s) -> torch.Tensor:
        """String length, array length, or vec array element count -> float tensor."""
        if isinstance(s, str):
            return torch.scalar_tensor(float(len(s)), dtype=torch.float32)
        if isinstance(s, list):
            return torch.scalar_tensor(float(len(s)), dtype=torch.float32)
        if isinstance(s, torch.Tensor):
            # Vec array [B,H,W,N,C] or [N,C]: element count is dim -2
            if s.dim() in (2, 5):
                return torch.scalar_tensor(float(s.shape[-2]), dtype=torch.float32)
            return torch.scalar_tensor(float(s.shape[-1]), dtype=torch.float32)
        raise ValueError("len() expects a string or array argument")

    @stdlib("replace", sig='replace(s, old, new) \\u2192 string', category='Strings', doc='Replace all occurrences of old with new.', ex='string r = replace(s, "foo", "bar");')
    @staticmethod
    def fn_replace(s, old, new, max_count=None) -> str:
        """Replace occurrences of old with new. Optional max_count limits replacements."""
        if not all(isinstance(x, str) for x in (s, old, new)):
            raise ValueError("replace() expects string arguments for s, old, new")
        if max_count is not None:
            n = int(max_count.item() if isinstance(max_count, torch.Tensor) else max_count)
            return s.replace(old, new, n)
        return s.replace(old, new)

    @stdlib("strip", sig='strip(s) \\u2192 string', category='Strings', doc='Remove leading/trailing whitespace.', ex='string clean = strip(s);')
    @staticmethod
    def fn_strip(s) -> str:
        """Trim leading/trailing whitespace."""
        if not isinstance(s, str):
            raise ValueError("strip() expects a string argument")
        return s.strip()

    @stdlib("lower", sig='lower(s) \\u2192 string', category='Strings', doc='Convert to lowercase.', ex='string lc = lower("Hello");')
    @staticmethod
    def fn_lower(s) -> str:
        """Convert to lowercase."""
        if not isinstance(s, str):
            raise ValueError("lower() expects a string argument")
        return s.lower()

    @stdlib("upper", sig='upper(s) \\u2192 string', category='Strings', doc='Convert to uppercase.', ex='string uc = upper("hello");')
    @staticmethod
    def fn_upper(s) -> str:
        """Convert to uppercase."""
        if not isinstance(s, str):
            raise ValueError("upper() expects a string argument")
        return s.upper()

    @stdlib("contains", sig='contains(s, sub) \\u2192 float', category='Strings', doc='Returns 1.0 if s contains sub, 0.0 otherwise.', ex='float has = contains(s, "test");')
    @staticmethod
    def fn_contains(s, sub) -> torch.Tensor:
        """Check if s contains sub. Returns 1.0 or 0.0."""
        if not (isinstance(s, str) and isinstance(sub, str)):
            raise ValueError("contains() expects two string arguments")
        return torch.scalar_tensor(1.0 if sub in s else 0.0, dtype=torch.float32)

    @stdlib("startswith", sig='startswith(s, prefix) \\u2192 float', category='Strings', doc='Returns 1.0 if s starts with prefix.', ex='float sw = startswith(s, "img_");')
    @staticmethod
    def fn_startswith(s, prefix) -> torch.Tensor:
        """Check if s starts with prefix. Returns 1.0 or 0.0."""
        if not (isinstance(s, str) and isinstance(prefix, str)):
            raise ValueError("startswith() expects two string arguments")
        return torch.scalar_tensor(1.0 if s.startswith(prefix) else 0.0, dtype=torch.float32)

    @stdlib("endswith", sig='endswith(s, suffix) \\u2192 float', category='Strings', doc='Returns 1.0 if s ends with suffix.', ex='float ew = endswith(s, ".png");')
    @staticmethod
    def fn_endswith(s, suffix) -> torch.Tensor:
        """Check if s ends with suffix. Returns 1.0 or 0.0."""
        if not (isinstance(s, str) and isinstance(suffix, str)):
            raise ValueError("endswith() expects two string arguments")
        return torch.scalar_tensor(1.0 if s.endswith(suffix) else 0.0, dtype=torch.float32)

    @stdlib("find", sig='find(s, sub) \\u2192 float', category='Strings', doc='Index of first occurrence, or -1.0 if not found.', ex='float idx = find(s, "world");')
    @staticmethod
    def fn_find(s, sub) -> torch.Tensor:
        """Find index of sub in s. Returns -1.0 if not found."""
        if not (isinstance(s, str) and isinstance(sub, str)):
            raise ValueError("find() expects two string arguments")
        return torch.scalar_tensor(float(s.find(sub)), dtype=torch.float32)

    @stdlib("substr", sig='substr(s, start, len?) \\u2192 string', category='Strings', doc='Extract a substring. len is optional.', ex='string sub = substr(s, 0, 5);')
    @staticmethod
    def fn_substr(s, start, length=None) -> str:
        """Extract substring. start is 0-based index."""
        if not isinstance(s, str):
            raise ValueError("substr() expects a string first argument")
        start_i = int(start.item() if isinstance(start, torch.Tensor) else start)
        if length is not None:
            len_i = int(length.item() if isinstance(length, torch.Tensor) else length)
            return s[start_i:start_i + len_i]
        return s[start_i:]

    @stdlib("to_int", sig='to_int(s) \\u2192 int', category='Strings', doc='Parse a string as an integer.', ex='int n = to_int("42");')
    @staticmethod
    def fn_to_int(s) -> torch.Tensor:
        """Parse integer from string."""
        if not isinstance(s, str):
            raise ValueError("to_int() expects a string argument")
        try:
            return torch.scalar_tensor(float(int(s.strip())), dtype=torch.float32)
        except ValueError:
            raise ValueError(f"to_int(): cannot parse '{s}' as integer")

    @stdlib("to_float", sig='to_float(s) \\u2192 float', category='Strings', doc='Parse a string as a float.', ex='float f = to_float("3.14");')
    @staticmethod
    def fn_to_float(s) -> torch.Tensor:
        """Parse float from string."""
        if not isinstance(s, str):
            raise ValueError("to_float() expects a string argument")
        try:
            return torch.scalar_tensor(float(s.strip()), dtype=torch.float32)
        except ValueError:
            raise ValueError(f"to_float(): cannot parse '{s}' as float")

    @stdlib("sanitize_filename", sig='sanitize_filename(s) \\u2192 string', category='Strings', doc='Remove unsafe characters for use in file paths.', ex='string safe = sanitize_filename(s);')
    @staticmethod
    def fn_sanitize_filename(s) -> str:
        """Remove characters illegal in filenames."""
        if not isinstance(s, str):
            raise ValueError("sanitize_filename() expects a string argument")
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', s)
        cleaned = cleaned.strip('. ')
        return cleaned if cleaned else "unnamed"

    @stdlib("split", sig='split(s, sep) \\u2192 string[]', category='Strings', doc='Split string into array by separator.', ex='string parts[4] = split(s, ",");')
    @staticmethod
    def fn_split(s, delimiter, max_splits=None) -> list:
        """Split string by delimiter. Returns a list of strings."""
        if not isinstance(s, str):
            raise ValueError("split() expects a string first argument")
        if not isinstance(delimiter, str):
            raise ValueError("split() delimiter must be a string")
        if max_splits is not None:
            n = int(max_splits.item() if isinstance(max_splits, torch.Tensor) else max_splits)
            return s.split(delimiter, n)
        return s.split(delimiter)

    @stdlib("lstrip", sig='lstrip(s) \\u2192 string', category='Strings', doc='Remove leading whitespace.', ex='string clean = lstrip(s);')
    @staticmethod
    def fn_lstrip(s) -> str:
        """Trim leading whitespace."""
        if not isinstance(s, str):
            raise ValueError("lstrip() expects a string argument")
        return s.lstrip()

    @stdlib("rstrip", sig='rstrip(s) \\u2192 string', category='Strings', doc='Remove trailing whitespace.', ex='string clean = rstrip(s);')
    @staticmethod
    def fn_rstrip(s) -> str:
        """Trim trailing whitespace."""
        if not isinstance(s, str):
            raise ValueError("rstrip() expects a string argument")
        return s.rstrip()

    @stdlib("pad_left", sig='pad_left(s, width, fill) \\u2192 string', category='Strings', doc='Pad string on the left to reach width.', ex='string n = pad_left(str(fi), 4, "0");')
    @staticmethod
    def fn_pad_left(s, width, char=None) -> str:
        """Pad string on the left to reach target width. Default pad char is space."""
        if not isinstance(s, str):
            raise ValueError("pad_left() expects a string first argument")
        w = int(width.item() if isinstance(width, torch.Tensor) else width)
        fill = " "
        if char is not None:
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError("pad_left() fill character must be a single character string")
            fill = char
        return s.rjust(w, fill)

    @stdlib("pad_right", sig='pad_right(s, width, fill) \\u2192 string', category='Strings', doc='Pad string on the right to reach width.', ex='string n = pad_right(s, 20, " ");')
    @staticmethod
    def fn_pad_right(s, width, char=None) -> str:
        """Pad string on the right to reach target width. Default pad char is space."""
        if not isinstance(s, str):
            raise ValueError("pad_right() expects a string first argument")
        w = int(width.item() if isinstance(width, torch.Tensor) else width)
        fill = " "
        if char is not None:
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError("pad_right() fill character must be a single character string")
            fill = char
        return s.ljust(w, fill)

    @stdlib("format", sig='format(fmt, ...) \\u2192 string', category='Strings', doc='Printf-style formatting. %d = int, %f = float, %s = string.', ex='string s = format("Frame %d of %d", fi, fn);')
    @staticmethod
    def fn_format(template, *args) -> str:
        """String interpolation. Replaces {} placeholders with arguments.
        format("frame_{}_v{}", 42, 3) produces "frame_42_v3".
        """
        if not isinstance(template, str):
            raise ValueError("format() expects a string template as first argument")
        # Convert tensor args to Python values for formatting
        converted = []
        for a in args:
            if isinstance(a, torch.Tensor):
                v = _scalar_from_tensor(a, "format")
                # Round to 6 significant digits to counteract float32 noise
                if v == int(v):
                    converted.append(int(v))
                else:
                    converted.append(float(f"{v:.6g}"))
            else:
                converted.append(a)
        try:
            return template.format(*converted)
        except KeyError as e:
            raise ValueError(
                f"format() only understands plain {{}} placeholders, but the template uses "
                f"a named one ({{{e.args[0]}}}). Replace it with {{}} and pass values in "
                f"order, e.g. format(\"hi {{}}\", x).") from e
        except IndexError as e:
            n_ph = template.count("{}")
            raise ValueError(
                f"format() template has {n_ph} {{}} placeholder{'s' if n_ph != 1 else ''}, "
                f"but {len(converted)} value{'s' if len(converted) != 1 else ''} "
                f"{'were' if len(converted) != 1 else 'was'} given.") from e

    @stdlib("repeat", sig='repeat(s, n) \\u2192 string', category='Strings', doc='Repeat a string N times.', ex='string bar = repeat("=", 10);')
    @staticmethod
    def fn_repeat(s, count) -> str:
        """Repeat a string N times."""
        if not isinstance(s, str):
            raise ValueError("repeat() expects a string first argument")
        n = int(count.item() if isinstance(count, torch.Tensor) else count)
        if n < 0:
            n = 0
        return s * n

    @stdlib("str_reverse", sig='str_reverse(s) \\u2192 string', category='Strings', doc='Reverse a string.', ex='string r = str_reverse("abc");')
    @staticmethod
    def fn_str_reverse(s) -> str:
        """Reverse a string."""
        if not isinstance(s, str):
            raise ValueError("str_reverse() expects a string argument")
        return s[::-1]

    @stdlib("count", sig='count(s, sub) \\u2192 float', category='Strings', doc='Count non-overlapping occurrences of sub in s.', ex='float n = count(s, "the");')
    @staticmethod
    def fn_count(s, sub) -> torch.Tensor:
        """Count non-overlapping occurrences of sub in s."""
        if not (isinstance(s, str) and isinstance(sub, str)):
            raise ValueError("count() expects two string arguments")
        return torch.scalar_tensor(float(s.count(sub)), dtype=torch.float32)

    @stdlib("matches", sig='matches(s, pattern) \\u2192 float', category='Strings', doc='Returns 1.0 if the whole string matches the regex pattern, else 0.0.', ex='float ok = matches(s, "[0-9]+");')
    @staticmethod
    def fn_matches(s, pattern) -> torch.Tensor:
        """Test if string matches a regex pattern. Returns 1.0 if the full string matches, 0.0 otherwise."""
        if not (isinstance(s, str) and isinstance(pattern, str)):
            raise ValueError("matches() expects two string arguments")
        try:
            return torch.scalar_tensor(1.0 if re.fullmatch(pattern, s) else 0.0, dtype=torch.float32)
        except re.error as e:
            raise ValueError(f"matches() invalid regex: {e}")

    @stdlib("hash", sig='hash(s) \\u2192 string', category='Strings', doc='Deterministic string hash (SHA-256 hex, first 16 chars).', ex='string h = hash("seed");')
    @staticmethod
    def fn_hash(s) -> str:
        """Deterministic hash of a string. Returns a stable string hash (SHA-256 hex, first 16 chars)."""
        if not isinstance(s, str):
            raise ValueError("hash() expects a string argument")
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    @stdlib("hash_float", sig='hash_float(s) \\u2192 float', category='Strings', doc='Deterministic hash of a string to a float in [0, 1).', ex='float r = hash_float("seed");')
    @staticmethod
    def fn_hash_float(s) -> torch.Tensor:
        """Deterministic hash of a string to a float in [0, 1).
        Useful for procedural seeding and per-pixel variation."""
        if not isinstance(s, str):
            raise ValueError("hash_float() expects a string argument")
        h = hashlib.sha256(s.encode("utf-8")).digest()
        # Use first 8 bytes as unsigned 64-bit integer, normalize to [0, 1)
        value = int.from_bytes(h[:8], "big") / (2**64)
        return torch.scalar_tensor(value, dtype=torch.float32)

    @stdlib("hash_int", sig='hash_int(s, max?) \\u2192 int', category='Strings', doc='Deterministic hash of a string to a non-negative int (optional exclusive max).', ex='int i = hash_int("seed", 100);')
    @staticmethod
    def fn_hash_int(s, max_val=None) -> torch.Tensor:
        """Deterministic hash of a string to a non-negative integer.
        If max_val is provided, result is in [0, max_val).
        Otherwise returns a large positive integer."""
        if not isinstance(s, str):
            raise ValueError("hash_int() expects a string first argument")
        h = hashlib.sha256(s.encode("utf-8")).digest()
        value = int.from_bytes(h[:8], "big")
        if max_val is not None:
            m = int(max_val.item() if isinstance(max_val, torch.Tensor) else max_val)
            if m > 0:
                value = value % m
                if m <= 2**24:
                    # Modulo already bounds the value within float32's exact-int
                    # range — don't clamp it down and break the [0, max_val) contract.
                    return torch.scalar_tensor(float(value), dtype=torch.float32)
        # No max_val (or a range beyond float32's exact-int range): clamp so the
        # value stays exactly representable as float32.
        value = min(value, 2**24 - 1)
        return torch.scalar_tensor(float(value), dtype=torch.float32)

    @stdlib("char_at", sig='char_at(s, idx) \\u2192 string', category='Strings', doc='Character at index (0-based).', ex='string c = char_at(s, 0);')
    @staticmethod
    def fn_char_at(s, index) -> str:
        """Get character at index. Returns empty string if out of bounds."""
        if not isinstance(s, str):
            raise ValueError("char_at() expects a string first argument")
        i = int(index.item() if isinstance(index, torch.Tensor) else index)
        if 0 <= i < len(s):
            return s[i]
        return ""

    # -- Array functions ------------------------------------------------

    @stdlib("sort", sig='sort(arr) \\u2192 array', category='Arrays', doc='Sort array elements in ascending order.', ex='sort(arr);')
    @staticmethod
    def fn_sort(arr):
        """Sort array elements in ascending order. Returns sorted copy."""
        if isinstance(arr, list):
            return sorted(arr)
        t = _to_tensor(arr)
        if t.dim() in (2, 5):  # vec array: sort along element dim per channel
            return torch.sort(t, dim=-2).values
        return torch.sort(t, dim=-1).values

    @stdlib("reverse", sig='reverse(arr) \\u2192 array', category='Arrays', doc='Reverse array element order.', ex='reverse(arr);')
    @staticmethod
    def fn_reverse(arr):
        """Reverse array elements. Returns reversed copy."""
        if isinstance(arr, list):
            return list(reversed(arr))
        t = _to_tensor(arr)
        if t.dim() in (2, 5):  # vec array
            return torch.flip(t, dims=[-2])
        return torch.flip(t, dims=[-1])

    @stdlib("arr_sum", sig='arr_sum(arr) \\u2192 float', category='Arrays', doc='Sum of all array elements.', ex='float total = arr_sum(arr);')
    @staticmethod
    def fn_arr_sum(arr) -> torch.Tensor:
        """Sum all elements of an array. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr).float(), lambda t, d: t.sum(dim=d))

    @stdlib("arr_min", sig='arr_min(arr) \\u2192 float', category='Arrays', doc='Minimum value in array.', ex='float lo = arr_min(arr);')
    @staticmethod
    def fn_arr_min(arr) -> torch.Tensor:
        """Minimum element of an array per channel. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr).float(), lambda t, d: t.min(dim=d).values)

    @stdlib("arr_max", sig='arr_max(arr) \\u2192 float', category='Arrays', doc='Maximum value in array.', ex='float hi = arr_max(arr);')
    @staticmethod
    def fn_arr_max(arr) -> torch.Tensor:
        """Maximum element of an array per channel. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr).float(), lambda t, d: t.max(dim=d).values)

    @stdlib("median", sig='median(arr) \\u2192 float', category='Arrays', doc='Median value of array.', ex='float mid = median(arr);')
    @staticmethod
    def fn_median(arr) -> torch.Tensor:
        """Median element of an array per channel. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr).float(), lambda t, d: torch.median(t, dim=d).values)

    @stdlib("arr_avg", sig='arr_avg(arr) \\u2192 float', category='Arrays', doc='Average of all array elements.', ex='float avg = arr_avg(arr);')
    @staticmethod
    def fn_arr_avg(arr) -> torch.Tensor:
        """Average of array elements per channel. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr).float(), lambda t, d: t.mean(dim=d))

    @stdlib("join", sig='join(arr, sep) \\u2192 string', category='Arrays', doc='Concatenate string array with separator.', ex='string csv = join(names, ", ");')
    @staticmethod
    def fn_join(arr, sep) -> str:
        """Concatenate string array elements with separator."""
        if not isinstance(arr, list):
            raise ValueError("join() expects a string array")
        if not isinstance(sep, str):
            raise ValueError("join() separator must be a string")
        return sep.join(str(s) for s in arr)

    # -- Image reduction functions ---------------------------------------

    @stdlib("img_sum", sig='img_sum(img) \\u2192 vec', category='Image Stats', footprint='image', doc='Per-channel sum of all pixel values.', ex='vec3 total = img_sum(@A);')
    @staticmethod
    def fn_img_sum(image) -> torch.Tensor:
        """Sum of all pixels per channel per frame. Returns broadcast-friendly shape."""
        img = _to_tensor(image)
        if img.dim() >= 3:
            # PR-LP4: accumulate + return fp32 — an fp16 sum overflows to inf at
            # >=1024^2 (5.2e5 > 65504). .float() is a no-op on fp32 (bit-identical,
            # and identically mirrored in codegen_stdfns), so never cast the result back.
            return img.float().sum(dim=(1, 2), keepdim=True)
        return img

    @stdlib("img_mean", sig='img_mean(img) \\u2192 vec', category='Image Stats', footprint='image', doc='Per-channel mean (average) of the image.', ex='vec3 avg = img_mean(@A);')
    @staticmethod
    def fn_img_mean(image) -> torch.Tensor:
        """Mean of all pixels per channel per frame."""
        img = _to_tensor(image)
        if img.dim() >= 3:
            return img.float().mean(dim=(1, 2), keepdim=True)  # PR-LP4: fp32 accumulate
        return img

    @stdlib("img_min", sig='img_min(img) \\u2192 vec', category='Image Stats', footprint='image', doc='Per-channel minimum across the entire image.', ex='vec3 lo = img_min(@A);')
    @staticmethod
    def fn_img_min(image) -> torch.Tensor:
        """Min pixel value per channel per frame."""
        img = _to_tensor(image)
        if img.dim() >= 3:
            return img.float().amin(dim=(1, 2), keepdim=True)  # PR-LP4: fp32 return
        return img

    @stdlib("img_max", sig='img_max(img) \\u2192 vec', category='Image Stats', footprint='image', doc='Per-channel maximum across the entire image.', ex='vec3 hi = img_max(@A);')
    @staticmethod
    def fn_img_max(image) -> torch.Tensor:
        """Max pixel value per channel per frame."""
        img = _to_tensor(image)
        if img.dim() >= 3:
            return img.float().amax(dim=(1, 2), keepdim=True)  # PR-LP4: fp32 return
        return img

    @stdlib("img_median", sig='img_median(img) \\u2192 vec', category='Image Stats', footprint='image', doc='Per-channel median of the image.', ex='vec3 mid = img_median(@A);')
    @staticmethod
    def fn_img_median(image) -> torch.Tensor:
        """Median pixel value per channel per frame."""
        img = _to_tensor(image).float()  # PR-LP4: reduce + return fp32 (fp16-safe)
        if img.dim() == 4:
            B, H, W, C = img.shape
            flat = img.reshape(B, H * W, C)
            return torch.median(flat, dim=1).values.unsqueeze(1).unsqueeze(1)
        if img.dim() == 3:
            B, H, W = img.shape
            flat = img.reshape(B, H * W)
            return torch.median(flat, dim=1).values.unsqueeze(1).unsqueeze(1)
        return img

    @stdlib("debug_print", sig='debug_print(label, value[, x, y]) \\u2192 value', category='Debugging', sync=True,
            doc="Probe a value at a pixel — records it for the node's "
            "HUD and returns the value unchanged (a print-style debug tap). Interpreter"
            "-only; a compiled tier falls back so the probe always fires.",
            ex='float g = debug_print("luma", luma(@A.rgb), 0, 0);')
    @staticmethod
    def fn_debug_print(label, value, x=0.0, y=0.0):
        """LX-5: value-at-pixel probe. Records value at (x,y) into the tier_trace probe
        list (folded into the ui= HUD payload) and returns `value` UNCHANGED so @OUT is
        bit-identical with or without the probe. torch-native readout, no numpy."""
        from . import tier_trace
        import math

        def _json_safe(v):
            # audit: a NaN/Inf probe would serialize as a bare NaN/Infinity token — invalid
            # JSON that breaks the ui= websocket frame. Map non-finite floats to None (null).
            if isinstance(v, list):
                return [_json_safe(x) for x in v]
            return None if isinstance(v, float) and not math.isfinite(v) else v

        try:
            xi = int(x.item()) if isinstance(x, torch.Tensor) else int(x)
            yi = int(y.item()) if isinstance(y, torch.Tensor) else int(y)
            if isinstance(value, torch.Tensor) and value.dim() >= 3:
                H, W = value.shape[1], value.shape[2]
                pv = value[0, min(max(yi, 0), H - 1), min(max(xi, 0), W - 1)]
                recorded = pv.detach().float().reshape(-1)[:4].tolist()
            elif isinstance(value, torch.Tensor):
                recorded = value.detach().float().reshape(-1)[:4].tolist()
            else:
                recorded = float(value)
            tier_trace.record_probe(label, _json_safe(recorded), xi, yi)
        except Exception:
            pass
        return value


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
    """
    key = (round(sigma, 3), device)
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
    sigma_q = round(sigma, 3)
    key = (id(img), _safe_version(img), sigma_q)
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
        lod_val = lod_t.item()
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


# -- Noise functions (extracted to noise.py) --------------------------------
from .noise import (
    _perlin2d_fast, _perlin3d_fast, _simplex2d,
    _fbm2d, _fbm3d,
    _worley2d, _worley3d,
    _curl2d, _curl3d,
    _ridged2d, _ridged3d,
    _billow2d, _billow3d,
    _turbulence2d, _turbulence3d,
    _flow2d, _flow3d,
    _alligator2d, _alligator3d,
)


def _to_tensor(x) -> torch.Tensor:
    """Ensure a value is a float torch.Tensor. Preserves an existing floating
    dtype (fp16/bf16/fp32) so the M-3 fp16 image-data mode isn't silently
    upcast to fp32; promotes int/bool tensors to float."""
    if x.__class__ is torch.Tensor:
        return x if x.is_floating_point() else x.float()
    return torch.scalar_tensor(float(x), dtype=torch.float32)


def _to_float(x) -> float:
    """Extract a Python float from a scalar."""
    if isinstance(x, torch.Tensor):
        return x.item()
    return float(x)


def _is_scalar(x) -> bool:
    """Check if a value is a scalar (Python number or 0-dim tensor)."""
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, torch.Tensor):
        return x.dim() == 0
    return False
