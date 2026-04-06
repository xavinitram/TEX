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

# Unified safety epsilon for division guards, domain clamping, and near-zero checks.
# Chosen to be well above float32 machine epsilon (~1.2e-7) while small enough
# to be invisible in image-processing contexts.
SAFE_EPSILON = 1e-8

# Rec.709 luma coefficients for RGB → luminance conversion
LUMA_R, LUMA_G, LUMA_B = 0.2126, 0.7152, 0.0722

# Valid channel counts for vector types (vec2, vec3, vec4)
VEC_CHANNELS = frozenset((2, 3, 4))

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


# Pre-allocated grid buffer for sample() — avoids torch.stack allocation per call.
# Keyed by (B, H, W, device) → [B, H, W, 2] tensor.
# Bounded via LRU eviction (each entry is ~16 MB at 1080p).
_grid_buf: _OrderedDict[tuple, torch.Tensor] = _OrderedDict()
_GRID_BUF_MAX = 16


def _get_grid_buf(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Get or allocate a reusable [B, H, W, 2] grid buffer."""
    key = (B, H, W, device)
    buf = _grid_buf.get(key)
    # Recreate if cached buffer was created under inference_mode
    # (inference tensors can't be written to outside inference_mode)
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
    key = ("bidx", B, H, W, str(device))
    cached = _sampler_cache.get(key)
    if cached is not None:
        return cached
    t = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
    _sampler_cache[key] = t
    if len(_sampler_cache) > _SAMPLER_CACHE_MAX:
        _sampler_cache.popitem(last=False)
    return t


def _get_lanczos_taps(device: torch.device) -> torch.Tensor:
    """Get or create cached Lanczos-3 tap offset tensor [-2, -1, 0, 1, 2, 3]."""
    key = ("ltaps", str(device))
    cached = _sampler_cache.get(key)
    if cached is not None:
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
        return {
            # Math
            "sin": TEXStdlib.fn_sin,
            "cos": TEXStdlib.fn_cos,
            "tan": TEXStdlib.fn_tan,
            "asin": TEXStdlib.fn_asin,
            "acos": TEXStdlib.fn_acos,
            "atan": TEXStdlib.fn_atan,
            "atan2": TEXStdlib.fn_atan2,
            "sqrt": TEXStdlib.fn_sqrt,
            "pow": TEXStdlib.fn_pow,
            "exp": TEXStdlib.fn_exp,
            "log": TEXStdlib.fn_log,
            "abs": TEXStdlib.fn_abs,
            "sign": TEXStdlib.fn_sign,
            "floor": TEXStdlib.fn_floor,
            "ceil": TEXStdlib.fn_ceil,
            "round": TEXStdlib.fn_round,
            "trunc": TEXStdlib.fn_trunc,
            "fract": TEXStdlib.fn_fract,
            "mod": TEXStdlib.fn_mod,

            "log2": TEXStdlib.fn_log2,
            "log10": TEXStdlib.fn_log10,
            "pow2": TEXStdlib.fn_pow2,
            "pow10": TEXStdlib.fn_pow10,
            "sinh": TEXStdlib.fn_sinh,
            "cosh": TEXStdlib.fn_cosh,
            "tanh": TEXStdlib.fn_tanh,
            "sincos": TEXStdlib.fn_sincos,
            "hypot": TEXStdlib.fn_hypot,
            "isnan": TEXStdlib.fn_isnan,
            "isinf": TEXStdlib.fn_isinf,
            "degrees": TEXStdlib.fn_degrees,
            "radians": TEXStdlib.fn_radians,
            "spow": TEXStdlib.fn_spow,
            "sdiv": TEXStdlib.fn_sdiv,

            # Clamping and interpolation
            "min": TEXStdlib.fn_min,
            "max": TEXStdlib.fn_max,
            "clamp": TEXStdlib.fn_clamp,
            "lerp": TEXStdlib.fn_lerp,
            "mix": TEXStdlib.fn_lerp,  # alias
            "fit": TEXStdlib.fn_fit,
            "smoothstep": TEXStdlib.fn_smoothstep,
            "step": TEXStdlib.fn_step,

            # Vector operations
            "dot": TEXStdlib.fn_dot,
            "length": TEXStdlib.fn_length,
            "distance": TEXStdlib.fn_distance,
            "normalize": TEXStdlib.fn_normalize,
            "cross": TEXStdlib.fn_cross,
            "reflect": TEXStdlib.fn_reflect,

            # Color operations
            "luma": TEXStdlib.fn_luma,
            "hsv2rgb": TEXStdlib.fn_hsv2rgb,
            "rgb2hsv": TEXStdlib.fn_rgb2hsv,

            # Sampling
            "sample": TEXStdlib.fn_sample,
            "fetch": TEXStdlib.fn_fetch,
            "sample_cubic": TEXStdlib.fn_sample_cubic,
            "sample_lanczos": TEXStdlib.fn_sample_lanczos,
            "sample_mip": TEXStdlib.fn_sample_mip,
            "sample_mip_gauss": TEXStdlib.fn_sample_mip_gauss,
            "gauss_blur": TEXStdlib.fn_gauss_blur,
            "fetch_frame": TEXStdlib.fn_fetch_frame,
            "sample_frame": TEXStdlib.fn_sample_frame,

            # Noise
            "perlin": TEXStdlib.fn_perlin,
            "simplex": TEXStdlib.fn_simplex,
            "fbm": TEXStdlib.fn_fbm,
            "worley_f1": TEXStdlib.fn_worley_f1,
            "worley_f2": TEXStdlib.fn_worley_f2,
            "voronoi": TEXStdlib.fn_voronoi,
            "curl": TEXStdlib.fn_curl,
            "ridged": TEXStdlib.fn_ridged,
            "billow": TEXStdlib.fn_billow,
            "turbulence": TEXStdlib.fn_turbulence,
            "flow": TEXStdlib.fn_flow,
            "alligator": TEXStdlib.fn_alligator,

            # SDF primitives
            "sdf_circle": TEXStdlib.fn_sdf_circle,
            "sdf_box": TEXStdlib.fn_sdf_box,
            "sdf_line": TEXStdlib.fn_sdf_line,
            "sdf_polygon": TEXStdlib.fn_sdf_polygon,

            # Smooth blending
            "smin": TEXStdlib.fn_smin,
            "smax": TEXStdlib.fn_smax,

            # Gradient sampling
            "sample_grad": TEXStdlib.fn_sample_grad,

            # String operations
            "str": TEXStdlib.fn_str,
            "len": TEXStdlib.fn_len,
            "replace": TEXStdlib.fn_replace,
            "strip": TEXStdlib.fn_strip,
            "lower": TEXStdlib.fn_lower,
            "upper": TEXStdlib.fn_upper,
            "contains": TEXStdlib.fn_contains,
            "startswith": TEXStdlib.fn_startswith,
            "endswith": TEXStdlib.fn_endswith,
            "find": TEXStdlib.fn_find,
            "substr": TEXStdlib.fn_substr,
            "to_int": TEXStdlib.fn_to_int,
            "to_float": TEXStdlib.fn_to_float,
            "sanitize_filename": TEXStdlib.fn_sanitize_filename,
            "split": TEXStdlib.fn_split,
            "lstrip": TEXStdlib.fn_lstrip,
            "rstrip": TEXStdlib.fn_rstrip,
            "pad_left": TEXStdlib.fn_pad_left,
            "pad_right": TEXStdlib.fn_pad_right,
            "format": TEXStdlib.fn_format,
            "repeat": TEXStdlib.fn_repeat,
            "str_reverse": TEXStdlib.fn_str_reverse,
            "count": TEXStdlib.fn_count,
            "matches": TEXStdlib.fn_matches,
            "hash": TEXStdlib.fn_hash,
            "hash_float": TEXStdlib.fn_hash_float,
            "hash_int": TEXStdlib.fn_hash_int,
            "char_at": TEXStdlib.fn_char_at,

            # Array operations
            "sort": TEXStdlib.fn_sort,
            "reverse": TEXStdlib.fn_reverse,
            "arr_sum": TEXStdlib.fn_arr_sum,
            "arr_min": TEXStdlib.fn_arr_min,
            "arr_max": TEXStdlib.fn_arr_max,
            "median": TEXStdlib.fn_median,
            "arr_avg": TEXStdlib.fn_arr_avg,
            "join": TEXStdlib.fn_join,

            # Matrix operations
            "transpose": TEXStdlib.fn_transpose,
            "determinant": TEXStdlib.fn_determinant,
            "inverse": TEXStdlib.fn_inverse,

            # Image reductions
            "img_sum": TEXStdlib.fn_img_sum,
            "img_mean": TEXStdlib.fn_img_mean,
            "img_min": TEXStdlib.fn_img_min,
            "img_max": TEXStdlib.fn_img_max,
            "img_median": TEXStdlib.fn_img_median,
        }

    # -- Math functions -------------------------------------------------

    @staticmethod
    def fn_sin(x):
        return torch.sin(_to_tensor(x))

    @staticmethod
    def fn_cos(x):
        return torch.cos(_to_tensor(x))

    @staticmethod
    def fn_tan(x):
        return torch.tan(_to_tensor(x))

    @staticmethod
    def fn_asin(x):
        return torch.asin(torch.clamp(_to_tensor(x), -1.0, 1.0))

    @staticmethod
    def fn_acos(x):
        return torch.acos(torch.clamp(_to_tensor(x), -1.0, 1.0))

    @staticmethod
    def fn_atan(x):
        return torch.atan(_to_tensor(x))

    @staticmethod
    def fn_atan2(y, x):
        return torch.atan2(_to_tensor(y), _to_tensor(x))

    @staticmethod
    def fn_sincos(x):
        """Returns vec2(sin(x), cos(x)) — computes both in a single pass."""
        t = _to_tensor(x)
        return torch.stack([torch.sin(t), torch.cos(t)], dim=-1)

    @staticmethod
    def fn_sqrt(x):
        return torch.sqrt(torch.clamp(_to_tensor(x), min=0.0))

    @staticmethod
    def fn_pow(base, exp):
        b = _to_tensor(base)
        e = _to_tensor(exp)
        # exp-log is ~3x faster than torch.pow for spatial tensors (common in image processing)
        if b.dim() >= 2:
            return torch.exp(torch.log(b.clamp(min=SAFE_EPSILON)) * e)
        return torch.pow(b, e)

    @staticmethod
    def fn_exp(x):
        return torch.exp(_to_tensor(x))

    @staticmethod
    def fn_log(x):
        return torch.log(torch.clamp(_to_tensor(x), min=SAFE_EPSILON))

    @staticmethod
    def fn_abs(x):
        return torch.abs(_to_tensor(x))

    @staticmethod
    def fn_sign(x):
        return torch.sign(_to_tensor(x))

    @staticmethod
    def fn_floor(x):
        return torch.floor(_to_tensor(x))

    @staticmethod
    def fn_ceil(x):
        return torch.ceil(_to_tensor(x))

    @staticmethod
    def fn_round(x):
        return torch.round(_to_tensor(x))

    @staticmethod
    def fn_trunc(x):
        return torch.trunc(_to_tensor(x))

    @staticmethod
    def fn_fract(x):
        t = _to_tensor(x)
        return t - torch.floor(t)

    @staticmethod
    def fn_mod(a, b):
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        safe_b = b_t + SAFE_EPSILON * (b_t == 0).float()
        return torch.fmod(a_t, safe_b)

    @staticmethod
    def fn_log2(x):
        return torch.log2(torch.clamp(_to_tensor(x), min=SAFE_EPSILON))

    @staticmethod
    def fn_log10(x):
        return torch.log10(torch.clamp(_to_tensor(x), min=SAFE_EPSILON))

    @staticmethod
    def fn_pow2(x):
        return torch.pow(2.0, _to_tensor(x))

    @staticmethod
    def fn_pow10(x):
        return torch.pow(10.0, _to_tensor(x))

    @staticmethod
    def fn_sinh(x):
        return torch.sinh(_to_tensor(x))

    @staticmethod
    def fn_cosh(x):
        return torch.cosh(_to_tensor(x))

    @staticmethod
    def fn_tanh(x):
        return torch.tanh(_to_tensor(x))

    @staticmethod
    def fn_hypot(x, y):
        return torch.hypot(_to_tensor(x), _to_tensor(y))

    @staticmethod
    def fn_isnan(x):
        return torch.isnan(_to_tensor(x)).float()

    @staticmethod
    def fn_isinf(x):
        return torch.isinf(_to_tensor(x)).float()

    @staticmethod
    def fn_degrees(x):
        return torch.rad2deg(_to_tensor(x))

    @staticmethod
    def fn_radians(x):
        return torch.deg2rad(_to_tensor(x))

    @staticmethod
    def fn_spow(x, y):
        """Safe power — sign(x) * pow(abs(x), y). Avoids NaN on negative bases."""
        t = _to_tensor(x)
        yt = _to_tensor(y)
        abs_t = torch.abs(t)
        mask = abs_t < SAFE_EPSILON
        safe_abs = torch.clamp(abs_t, min=SAFE_EPSILON)
        return torch.where(mask, torch.zeros_like(t), torch.sign(t) * torch.pow(safe_abs, yt))

    @staticmethod
    def fn_sdiv(a, b):
        """Safe division — returns 0.0 where abs(b) < SAFE_EPSILON."""
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        mask = torch.abs(b_t) < SAFE_EPSILON
        safe_b = torch.where(mask, torch.ones_like(b_t), b_t)
        return torch.where(mask, torch.zeros_like(a_t), a_t / safe_b)

    # -- Matrix operations ----------------------------------------------

    @staticmethod
    def fn_transpose(m):
        return m.transpose(-1, -2)

    @staticmethod
    def fn_determinant(m):
        return torch.linalg.det(m)

    @staticmethod
    def fn_inverse(m):
        return torch.linalg.inv(m)

    # -- Clamping / interpolation ---------------------------------------

    @staticmethod
    def fn_min(a, b):
        return torch.minimum(_to_tensor(a), _to_tensor(b))

    @staticmethod
    def fn_max(a, b):
        return torch.maximum(_to_tensor(a), _to_tensor(b))

    @staticmethod
    def fn_clamp(x, lo, hi):
        return torch.clamp(_to_tensor(x), min=_to_float(lo) if _is_scalar(lo) else None,
                          max=_to_float(hi) if _is_scalar(hi) else None) if _is_scalar(lo) and _is_scalar(hi) \
            else torch.minimum(torch.maximum(_to_tensor(x), _to_tensor(lo)), _to_tensor(hi))

    @staticmethod
    def fn_lerp(a, b, t):
        a_t, b_t, t_t = _to_tensor(a), _to_tensor(b), _to_tensor(t)
        # Auto-unsqueeze weight for channel broadcast: [B,H,W] weight with [B,H,W,C] values
        if t_t.dim() + 1 == a_t.dim():
            t_t = t_t.unsqueeze(-1)
        return torch.lerp(a_t, b_t, t_t)

    @staticmethod
    def fn_fit(val, old_min, old_max, new_min, new_max):
        """Remap val from [old_min, old_max] to [new_min, new_max]."""
        v = _to_tensor(val)
        o_min, o_max = _to_tensor(old_min), _to_tensor(old_max)
        n_min, n_max = _to_tensor(new_min), _to_tensor(new_max)
        t = (v - o_min) / (o_max - o_min + SAFE_EPSILON)
        return torch.lerp(n_min, n_max, t)

    @staticmethod
    def fn_smoothstep(edge0, edge1, x):
        e0, e1, xv = _to_tensor(edge0), _to_tensor(edge1), _to_tensor(x)
        t = torch.clamp((xv - e0) / (e1 - e0 + SAFE_EPSILON), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def fn_step(edge, x):
        return ((_to_tensor(x)) >= _to_tensor(edge)).float()

    # -- Vector operations ----------------------------------------------

    @staticmethod
    def fn_dot(a, b):
        """Dot product via einsum (4× faster than mul+sum on CPU)."""
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        return torch.einsum('...c,...c->...', a_t, b_t)

    @staticmethod
    def fn_length(v):
        """Length (magnitude) of a vector. Operates on last dimension."""
        t = _to_tensor(v)
        if t.dim() >= 1 and t.shape[-1] in VEC_CHANNELS:
            return torch.linalg.vector_norm(t, dim=-1)
        return torch.abs(t)

    @staticmethod
    def fn_distance(a, b):
        diff = _to_tensor(a) - _to_tensor(b)
        if diff.dim() >= 1 and diff.shape[-1] in VEC_CHANNELS:
            return torch.linalg.vector_norm(diff, dim=-1)
        return torch.abs(diff)

    @staticmethod
    def fn_normalize(v):
        t = _to_tensor(v)
        if t.dim() >= 1 and t.shape[-1] in VEC_CHANNELS:
            norm = torch.linalg.vector_norm(t, dim=-1, keepdim=True)
            return t / (norm + SAFE_EPSILON)
        return torch.sign(t)

    @staticmethod
    def fn_cross(a, b):
        """Cross product. Only works on vec3 (last dim = 3)."""
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        # Take first 3 channels if vec4
        if a_t.shape[-1] == 4:
            a_t = a_t[..., :3]
        if b_t.shape[-1] == 4:
            b_t = b_t[..., :3]
        return torch.cross(a_t, b_t, dim=-1)

    @staticmethod
    def fn_reflect(incident, normal):
        i, n = _to_tensor(incident), _to_tensor(normal)
        d = (i * n).sum(dim=-1, keepdim=True)
        return i - 2.0 * d * n

    # -- Color operations -----------------------------------------------

    @staticmethod
    def fn_luma(color):
        """Compute luminance from RGB(A). Returns scalar per pixel."""
        c = _to_tensor(color)
        if c.dim() >= 1 and c.shape[-1] >= 3:
            return LUMA_R * c[..., 0] + LUMA_G * c[..., 1] + LUMA_B * c[..., 2]
        return c

    @staticmethod
    def fn_hsv2rgb(hsv):
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
        m1 = (i_mod == 1.0)
        m2 = (i_mod == 2.0)
        m3 = (i_mod == 3.0)
        m4 = (i_mod == 4.0)

        # r: 0->v, 1->q, 2->p, 3->p, 4->t, 5->v  (default v)
        r = torch.where(m1, q, torch.where(m2 | m3, p, torch.where(m4, t, v)))
        # g: 0->t, 1->v, 2->v, 3->q, 4->p, 5->p  (default p)
        g = torch.where(m1 | m2, v, torch.where(m3, q, torch.where(m4, p,
            torch.where(i_mod == 0.0, t, p))))
        # b: 0->p, 1->p, 2->t, 3->v, 4->v, 5->q  (default q)
        b = torch.where(m1, p, torch.where(m2, t, torch.where(m3 | m4, v,
            torch.where(i_mod == 0.0, p, q))))

        result = torch.cat([r, g, b], dim=-1)
        # If input was vec4, preserve alpha
        if c.shape[-1] == 4:
            result = torch.cat([result, c[..., 3:4]], dim=-1)
        return result

    @staticmethod
    def fn_rgb2hsv(rgb):
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

    # -- Sampling -------------------------------------------------------

    @staticmethod
    def fn_sample(image, u_coord, v_coord):
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
        result_bchw = torch.nn.functional.grid_sample(
            img_bchw, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )

        # Back to [B, H, W, C]
        return result_bchw.permute(0, 2, 3, 1)

    @staticmethod
    def fn_fetch(image, px, py):
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
                px_f = px_i[0] if px_d == 3 else px_i
                py_f = py_i[0] if py_d == 3 else py_i
                flat = py_f * W + px_f
                out_shape = flat.shape
                return torch.index_select(img.view(H * W, C), 0, flat.reshape(-1)).view(1, *out_shape, C)
            # Scalar coords: fall through to advanced indexing
            px_i = px_i.expand(1, H, W)
            py_i = py_i.expand(1, H, W)
            return img[0, py_i[0], px_i[0]].unsqueeze(0)

        # Multi-batch: expand coordinates to [B, H, W]
        if px_i.dim() == 0:
            px_i = px_i.expand(B, H, W)
        elif px_i.dim() == 2:
            px_i = px_i.unsqueeze(0).expand(B, H, W)
        if py_i.dim() == 0:
            py_i = py_i.expand(B, H, W)
        elif py_i.dim() == 2:
            py_i = py_i.unsqueeze(0).expand(B, H, W)

        return img[_get_batch_index(B, H, W, img.device), py_i, px_i]

    @staticmethod
    def fn_fetch_frame(image, frame, px, py):
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
        if f_idx.dim() == 0:
            f_idx = f_idx.expand(B, H, W)
        if f_idx.dim() == 2:
            f_idx = f_idx.unsqueeze(0).expand(B, H, W)
        if px_i.dim() == 0:
            px_i = px_i.expand(B, H, W)
        if px_i.dim() == 2:
            px_i = px_i.unsqueeze(0).expand(B, H, W)
        if py_i.dim() == 0:
            py_i = py_i.expand(B, H, W)
        if py_i.dim() == 2:
            py_i = py_i.unsqueeze(0).expand(B, H, W)

        return img[f_idx, py_i, px_i]

    @staticmethod
    def fn_sample_frame(image, frame, u_coord, v_coord):
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

        # Expand scalars to spatial dims
        if f_idx.dim() == 0:
            f_idx = f_idx.expand(B, H, W)
        if f_idx.dim() == 2:
            f_idx = f_idx.unsqueeze(0).expand(B, H, W)

        # For cross-frame sampling, we need to gather from specific frames first
        # then apply grid_sample per-unique-frame or use advanced indexing.
        # Since frame indices can vary per-pixel, we fall back to manual bilinear
        # for the cross-frame case (grid_sample only handles per-batch).

        # Convert from [0,1] to pixel coordinates
        x = u * (W - 1)
        y = v * (H - 1)

        if x.dim() == 0:
            x = x.expand(B, H, W)
            y = y.expand(B, H, W)
        elif x.dim() == 2:
            x = x.unsqueeze(0).expand(B, H, W)
            y = y.unsqueeze(0).expand(B, H, W)

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

    @staticmethod
    def fn_sample_cubic(image, u_coord, v_coord):
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
        result_bchw = torch.nn.functional.grid_sample(
            img_bchw, grid,
            mode='bicubic',
            padding_mode='border',
            align_corners=True,
        )

        # Back to [B, H, W, C]
        return result_bchw.permute(0, 2, 3, 1)

    @staticmethod
    def fn_sample_lanczos(image, u_coord, v_coord):
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

    @staticmethod
    def fn_sample_mip(image, u_coord, v_coord, lod):
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

    @staticmethod
    def fn_gauss_blur(image, sigma):
        """Separable Gaussian blur.

        Args:
            image: [B, H, W, C] tensor
            sigma: float — standard deviation in pixels (kernel radius ≈ 3*sigma)

        Returns blurred [B, H, W, C] tensor with replicate border handling.
        """
        img = image if image.__class__ is torch.Tensor else _to_tensor(image)
        sigma_t = sigma if sigma.__class__ is torch.Tensor else _to_tensor(sigma)
        sigma_val = max(sigma_t.item(), 0.0)
        if sigma_val < 0.3 or img.dim() < 4:
            return img
        bchw = _get_bchw(img)
        result = _gauss_blur_bchw(bchw, sigma_val)
        return result.permute(0, 2, 3, 1)

    @staticmethod
    def fn_bilateral_filter(image, sigma_s, sigma_r):
        """Edge-preserving bilateral filter using Tensor.unfold.

        Weights each neighbor by spatial Gaussian x range (color similarity)
        Gaussian. Radius is derived from sigma_s (3x sigma, capped at 5).
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
        radius = min(int(math.ceil(3.0 * ss)), 5)
        ksize = 2 * radius + 1

        # Convert to BCHW and pad
        bchw = img.permute(0, 3, 1, 2)  # [B, C, H, W]
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

    @staticmethod
    def fn_sample_mip_gauss(image, u_coord, v_coord, lod):
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

    @staticmethod
    def fn_perlin(x, y, z=None):
        """Perlin noise. 2D when z omitted, 3D when z provided. Returns float in ~[-1, 1]."""
        if z is not None:
            return _perlin3d_fast(_to_tensor(x), _to_tensor(y), _to_tensor(z))
        return _perlin2d_fast(_to_tensor(x), _to_tensor(y))

    @staticmethod
    def fn_simplex(x, y, z=None):
        """Simplex noise. 2D when z omitted, 3D falls back to Perlin. Returns float in ~[-1, 1]."""
        if z is not None:
            return _perlin3d_fast(_to_tensor(x), _to_tensor(y), _to_tensor(z))
        return _simplex2d(_to_tensor(x), _to_tensor(y))

    @staticmethod
    def fn_fbm(x, y, z_or_oct, octaves=None):
        """FBM noise. fbm(x,y,octaves) for 2D, fbm(x,y,z,octaves) for 3D."""
        if octaves is not None:
            return _fbm3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                          int(_to_float(octaves)))
        return _fbm2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))

    @staticmethod
    def fn_worley_f1(x, y, z=None):
        """Worley F1 noise (nearest cell distance). Returns float in ~[0, 1]."""
        if z is not None:
            return _worley3d(_to_tensor(x), _to_tensor(y), _to_tensor(z), return_f2=False)
        return _worley2d(_to_tensor(x), _to_tensor(y), return_f2=False)

    @staticmethod
    def fn_worley_f2(x, y, z=None):
        """Worley F2 noise (2nd nearest cell distance). Returns float in ~[0, 1]."""
        if z is not None:
            return _worley3d(_to_tensor(x), _to_tensor(y), _to_tensor(z), return_f2=True)
        return _worley2d(_to_tensor(x), _to_tensor(y), return_f2=True)

    @staticmethod
    def fn_voronoi(x, y, z=None):
        """Voronoi noise (alias for worley_f1)."""
        return TEXStdlib.fn_worley_f1(x, y, z)

    @staticmethod
    def fn_curl(x, y, z=None):
        """Curl noise. 2D → vec2 (divergence-free), 3D → vec3."""
        if z is not None:
            return _curl3d(_to_tensor(x), _to_tensor(y), _to_tensor(z))
        return _curl2d(_to_tensor(x), _to_tensor(y))

    @staticmethod
    def fn_ridged(x, y, z_or_oct, octaves=None):
        """Ridged FBM. ridged(x,y,octaves) for 2D, ridged(x,y,z,octaves) for 3D."""
        if octaves is not None:
            return _ridged3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                             int(_to_float(octaves)))
        return _ridged2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))

    @staticmethod
    def fn_billow(x, y, z_or_oct, octaves=None):
        """Billow FBM. billow(x,y,octaves) for 2D, billow(x,y,z,octaves) for 3D."""
        if octaves is not None:
            return _billow3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                             int(_to_float(octaves)))
        return _billow2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))

    @staticmethod
    def fn_turbulence(x, y, z_or_oct, octaves=None):
        """Turbulence. turbulence(x,y,octaves) for 2D, turbulence(x,y,z,octaves) for 3D."""
        if octaves is not None:
            return _turbulence3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                                 int(_to_float(octaves)))
        return _turbulence2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))

    @staticmethod
    def fn_flow(x, y, z_or_time, time=None):
        """Flow noise. flow(x,y,time) for 2D, flow(x,y,z,time) for 3D."""
        if time is not None:
            return _flow3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_time),
                           _to_float(time))
        return _flow2d(_to_tensor(x), _to_tensor(y), _to_float(z_or_time))

    @staticmethod
    def fn_alligator(x, y, z_or_oct=None, octaves=None):
        """Alligator noise. 2 args: 2D default octaves; 3 args: 2D custom octaves; 4 args: 3D."""
        if octaves is not None:
            return _alligator3d(_to_tensor(x), _to_tensor(y), _to_tensor(z_or_oct),
                                int(_to_float(octaves)))
        if z_or_oct is not None:
            return _alligator2d(_to_tensor(x), _to_tensor(y), int(_to_float(z_or_oct)))
        return _alligator2d(_to_tensor(x), _to_tensor(y), 4)

    # -- SDF primitives -------------------------------------------------

    @staticmethod
    def fn_sdf_circle(px, py, radius):
        """Signed distance to a circle centered at origin."""
        return torch.hypot(_to_tensor(px), _to_tensor(py)) - _to_tensor(radius)

    @staticmethod
    def fn_sdf_box(px, py, half_w, half_h):
        """Signed distance to an axis-aligned box centered at origin."""
        dx = torch.abs(_to_tensor(px)) - _to_tensor(half_w)
        dy = torch.abs(_to_tensor(py)) - _to_tensor(half_h)
        dx_c = torch.clamp(dx, min=0.0)
        dy_c = torch.clamp(dy, min=0.0)
        outside = torch.sqrt(dx_c * dx_c + dy_c * dy_c)
        inside = torch.clamp(torch.max(dx, dy), max=0.0)
        return outside + inside

    @staticmethod
    def fn_sdf_line(px, py, ax, ay, bx, by):
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

    @staticmethod
    def fn_sdf_polygon(px, py, radius, sides):
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

    @staticmethod
    def fn_smin(a, b, k):
        """Polynomial smooth minimum with smoothing radius k."""
        a_t = _to_tensor(a)
        b_t = _to_tensor(b)
        k_t = _to_tensor(k)
        h = torch.clamp(0.5 + 0.5 * (b_t - a_t) / (k_t + SAFE_EPSILON), 0.0, 1.0)
        return torch.lerp(b_t, a_t, h) - k_t * h * (1.0 - h)

    @staticmethod
    def fn_smax(a, b, k):
        """Polynomial smooth maximum with smoothing radius k."""
        a_t = _to_tensor(a)
        b_t = _to_tensor(b)
        k_t = _to_tensor(k)
        h = torch.clamp(0.5 - 0.5 * (b_t - a_t) / (k_t + SAFE_EPSILON), 0.0, 1.0)
        return torch.lerp(b_t, a_t, h) + k_t * h * (1.0 - h)

    # -- Gradient sampling -----------------------------------------------

    @staticmethod
    def fn_sample_grad(image, u_coord, v_coord):
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

    @staticmethod
    def fn_str(x):
        """Convert number to string."""
        if isinstance(x, str):
            return x
        if isinstance(x, torch.Tensor):
            v = x.item() if x.dim() == 0 else x.float().mean().item()
            return str(int(v)) if v == int(v) else str(v)
        return str(x)

    @staticmethod
    def fn_len(s):
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

    @staticmethod
    def fn_replace(s, old, new, max_count=None):
        """Replace occurrences of old with new. Optional max_count limits replacements."""
        if not all(isinstance(x, str) for x in (s, old, new)):
            raise ValueError("replace() expects string arguments for s, old, new")
        if max_count is not None:
            n = int(max_count.item() if isinstance(max_count, torch.Tensor) else max_count)
            return s.replace(old, new, n)
        return s.replace(old, new)

    @staticmethod
    def fn_strip(s):
        """Trim leading/trailing whitespace."""
        if not isinstance(s, str):
            raise ValueError("strip() expects a string argument")
        return s.strip()

    @staticmethod
    def fn_lower(s):
        """Convert to lowercase."""
        if not isinstance(s, str):
            raise ValueError("lower() expects a string argument")
        return s.lower()

    @staticmethod
    def fn_upper(s):
        """Convert to uppercase."""
        if not isinstance(s, str):
            raise ValueError("upper() expects a string argument")
        return s.upper()

    @staticmethod
    def fn_contains(s, sub):
        """Check if s contains sub. Returns 1.0 or 0.0."""
        if not (isinstance(s, str) and isinstance(sub, str)):
            raise ValueError("contains() expects two string arguments")
        return torch.scalar_tensor(1.0 if sub in s else 0.0, dtype=torch.float32)

    @staticmethod
    def fn_startswith(s, prefix):
        """Check if s starts with prefix. Returns 1.0 or 0.0."""
        if not (isinstance(s, str) and isinstance(prefix, str)):
            raise ValueError("startswith() expects two string arguments")
        return torch.scalar_tensor(1.0 if s.startswith(prefix) else 0.0, dtype=torch.float32)

    @staticmethod
    def fn_endswith(s, suffix):
        """Check if s ends with suffix. Returns 1.0 or 0.0."""
        if not (isinstance(s, str) and isinstance(suffix, str)):
            raise ValueError("endswith() expects two string arguments")
        return torch.scalar_tensor(1.0 if s.endswith(suffix) else 0.0, dtype=torch.float32)

    @staticmethod
    def fn_find(s, sub):
        """Find index of sub in s. Returns -1.0 if not found."""
        if not (isinstance(s, str) and isinstance(sub, str)):
            raise ValueError("find() expects two string arguments")
        return torch.scalar_tensor(float(s.find(sub)), dtype=torch.float32)

    @staticmethod
    def fn_substr(s, start, length=None):
        """Extract substring. start is 0-based index."""
        if not isinstance(s, str):
            raise ValueError("substr() expects a string first argument")
        start_i = int(start.item() if isinstance(start, torch.Tensor) else start)
        if length is not None:
            len_i = int(length.item() if isinstance(length, torch.Tensor) else length)
            return s[start_i:start_i + len_i]
        return s[start_i:]

    @staticmethod
    def fn_to_int(s):
        """Parse integer from string."""
        if not isinstance(s, str):
            raise ValueError("to_int() expects a string argument")
        try:
            return torch.scalar_tensor(float(int(s.strip())), dtype=torch.float32)
        except ValueError:
            raise ValueError(f"to_int(): cannot parse '{s}' as integer")

    @staticmethod
    def fn_to_float(s):
        """Parse float from string."""
        if not isinstance(s, str):
            raise ValueError("to_float() expects a string argument")
        try:
            return torch.scalar_tensor(float(s.strip()), dtype=torch.float32)
        except ValueError:
            raise ValueError(f"to_float(): cannot parse '{s}' as float")

    @staticmethod
    def fn_sanitize_filename(s):
        """Remove characters illegal in filenames."""
        if not isinstance(s, str):
            raise ValueError("sanitize_filename() expects a string argument")
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', s)
        cleaned = cleaned.strip('. ')
        return cleaned if cleaned else "unnamed"

    @staticmethod
    def fn_split(s, delimiter, max_splits=None):
        """Split string by delimiter. Returns a list of strings."""
        if not isinstance(s, str):
            raise ValueError("split() expects a string first argument")
        if not isinstance(delimiter, str):
            raise ValueError("split() delimiter must be a string")
        if max_splits is not None:
            n = int(max_splits.item() if isinstance(max_splits, torch.Tensor) else max_splits)
            return s.split(delimiter, n)
        return s.split(delimiter)

    @staticmethod
    def fn_lstrip(s):
        """Trim leading whitespace."""
        if not isinstance(s, str):
            raise ValueError("lstrip() expects a string argument")
        return s.lstrip()

    @staticmethod
    def fn_rstrip(s):
        """Trim trailing whitespace."""
        if not isinstance(s, str):
            raise ValueError("rstrip() expects a string argument")
        return s.rstrip()

    @staticmethod
    def fn_pad_left(s, width, char=None):
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

    @staticmethod
    def fn_pad_right(s, width, char=None):
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

    @staticmethod
    def fn_format(template, *args):
        """String interpolation. Replaces {} placeholders with arguments.
        format("frame_{}_v{}", 42, 3) produces "frame_42_v3".
        """
        if not isinstance(template, str):
            raise ValueError("format() expects a string template as first argument")
        # Convert tensor args to Python values for formatting
        converted = []
        for a in args:
            if isinstance(a, torch.Tensor):
                v = a.item() if a.dim() == 0 else a.float().mean().item()
                # Round to 6 significant digits to counteract float32 noise
                if v == int(v):
                    converted.append(int(v))
                else:
                    converted.append(float(f"{v:.6g}"))
            else:
                converted.append(a)
        try:
            return template.format(*converted)
        except (IndexError, KeyError) as e:
            raise ValueError(f"format() error: {e}")

    @staticmethod
    def fn_repeat(s, count):
        """Repeat a string N times."""
        if not isinstance(s, str):
            raise ValueError("repeat() expects a string first argument")
        n = int(count.item() if isinstance(count, torch.Tensor) else count)
        if n < 0:
            n = 0
        return s * n

    @staticmethod
    def fn_str_reverse(s):
        """Reverse a string."""
        if not isinstance(s, str):
            raise ValueError("str_reverse() expects a string argument")
        return s[::-1]

    @staticmethod
    def fn_count(s, sub):
        """Count non-overlapping occurrences of sub in s."""
        if not (isinstance(s, str) and isinstance(sub, str)):
            raise ValueError("count() expects two string arguments")
        return torch.scalar_tensor(float(s.count(sub)), dtype=torch.float32)

    @staticmethod
    def fn_matches(s, pattern):
        """Test if string matches a regex pattern. Returns 1.0 if the full string matches, 0.0 otherwise."""
        if not (isinstance(s, str) and isinstance(pattern, str)):
            raise ValueError("matches() expects two string arguments")
        try:
            return torch.scalar_tensor(1.0 if re.fullmatch(pattern, s) else 0.0, dtype=torch.float32)
        except re.error as e:
            raise ValueError(f"matches() invalid regex: {e}")

    @staticmethod
    def fn_hash(s):
        """Deterministic hash of a string. Returns a stable string hash (SHA-256 hex, first 16 chars)."""
        if not isinstance(s, str):
            raise ValueError("hash() expects a string argument")
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def fn_hash_float(s):
        """Deterministic hash of a string to a float in [0, 1).
        Useful for procedural seeding and per-pixel variation."""
        if not isinstance(s, str):
            raise ValueError("hash_float() expects a string argument")
        h = hashlib.sha256(s.encode("utf-8")).digest()
        # Use first 8 bytes as unsigned 64-bit integer, normalize to [0, 1)
        value = int.from_bytes(h[:8], "big") / (2**64)
        return torch.scalar_tensor(value, dtype=torch.float32)

    @staticmethod
    def fn_hash_int(s, max_val=None):
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
        # Clamp to float32 safe integer range
        value = min(value, 2**24 - 1)
        return torch.scalar_tensor(float(value), dtype=torch.float32)

    @staticmethod
    def fn_char_at(s, index):
        """Get character at index. Returns empty string if out of bounds."""
        if not isinstance(s, str):
            raise ValueError("char_at() expects a string first argument")
        i = int(index.item() if isinstance(index, torch.Tensor) else index)
        if 0 <= i < len(s):
            return s[i]
        return ""

    # -- Array functions ------------------------------------------------

    @staticmethod
    def fn_sort(arr):
        """Sort array elements in ascending order. Returns sorted copy."""
        if isinstance(arr, list):
            return sorted(arr)
        t = _to_tensor(arr)
        if t.dim() in (2, 5):  # vec array: sort along element dim per channel
            return torch.sort(t, dim=-2).values
        return torch.sort(t, dim=-1).values

    @staticmethod
    def fn_reverse(arr):
        """Reverse array elements. Returns reversed copy."""
        if isinstance(arr, list):
            return list(reversed(arr))
        t = _to_tensor(arr)
        if t.dim() in (2, 5):  # vec array
            return torch.flip(t, dims=[-2])
        return torch.flip(t, dims=[-1])

    @staticmethod
    def fn_arr_sum(arr):
        """Sum all elements of an array. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr), lambda t, d: t.sum(dim=d))

    @staticmethod
    def fn_arr_min(arr):
        """Minimum element of an array per channel. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr), lambda t, d: t.min(dim=d).values)

    @staticmethod
    def fn_arr_max(arr):
        """Maximum element of an array per channel. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr), lambda t, d: t.max(dim=d).values)

    @staticmethod
    def fn_median(arr):
        """Median element of an array per channel. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr), lambda t, d: torch.median(t, dim=d).values)

    @staticmethod
    def fn_arr_avg(arr):
        """Average of array elements per channel. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr), lambda t, d: t.mean(dim=d))

    @staticmethod
    def fn_join(arr, sep):
        """Concatenate string array elements with separator."""
        if not isinstance(arr, list):
            raise ValueError("join() expects a string array")
        if not isinstance(sep, str):
            raise ValueError("join() separator must be a string")
        return sep.join(str(s) for s in arr)

    # -- Image reduction functions ---------------------------------------

    @staticmethod
    def fn_img_sum(image):
        """Sum of all pixels per channel per frame. Returns broadcast-friendly shape."""
        img = _to_tensor(image)
        if img.dim() >= 3:
            return img.sum(dim=(1, 2), keepdim=True)
        return img

    @staticmethod
    def fn_img_mean(image):
        """Mean of all pixels per channel per frame."""
        img = _to_tensor(image)
        if img.dim() >= 3:
            return img.mean(dim=(1, 2), keepdim=True)
        return img

    @staticmethod
    def fn_img_min(image):
        """Min pixel value per channel per frame."""
        img = _to_tensor(image)
        if img.dim() >= 3:
            return img.amin(dim=(1, 2), keepdim=True)
        return img

    @staticmethod
    def fn_img_max(image):
        """Max pixel value per channel per frame."""
        img = _to_tensor(image)
        if img.dim() >= 3:
            return img.amax(dim=(1, 2), keepdim=True)
        return img

    @staticmethod
    def fn_img_median(image):
        """Median pixel value per channel per frame."""
        img = _to_tensor(image)
        if img.dim() == 4:
            B, H, W, C = img.shape
            flat = img.reshape(B, H * W, C)
            return torch.median(flat, dim=1).values.unsqueeze(1).unsqueeze(1)
        if img.dim() == 3:
            B, H, W = img.shape
            flat = img.reshape(B, H * W)
            return torch.median(flat, dim=1).values.unsqueeze(1).unsqueeze(1)
        return img


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
    key = (round(sigma, 3), str(device))
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
    kh = kernel_h.expand(C, 1, 1, -1)
    kv = kernel_v.expand(C, 1, -1, 1)
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


def _get_mip_pyramid(img: torch.Tensor) -> list[torch.Tensor]:
    """Area-downsample mipmap pyramid (cached)."""
    # Version-safe key: id() + _version detects in-place mutations.
    return _build_mip_pyramid(img, _mip_cache, (id(img), img._version))


def _get_mip_pyramid_gauss(img: torch.Tensor, sigma: float = 1.13) -> list[torch.Tensor]:
    """Gaussian-prefiltered mipmap pyramid (sigma=1.13, SIGMA_C ≈ 0.825, cached)."""
    sigma_q = round(sigma, 3)
    key = (id(img), img._version, sigma_q)
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
    result_bchw = torch.nn.functional.grid_sample(
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
        # Identity UV: u[0,0,0]=0, u[0,0,W-1]=1, v[0,0,0]=0, v[0,H-1,0]=1
        if (abs(u[0, 0, 0].item()) < 1e-5 and abs(u[0, 0, -1].item() - 1.0) < 1e-5
                and abs(v[0, 0, 0].item()) < 1e-5 and abs(v[0, -1, 0].item() - 1.0) < 1e-5):
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
        return torch.lerp(s_lo, s_hi, lod_frac)

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
    return torch.lerp(s_lo, s_hi, frac_expanded)


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
    """Ensure a value is a torch.Tensor (skip recast if already float32)."""
    if x.__class__ is torch.Tensor:
        return x if x.dtype == torch.float32 else x.float()
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
