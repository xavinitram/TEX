"""
TEX Standard Library — runtime implementations of built-in functions.

All functions operate on PyTorch tensors. Scalars are represented as
0-dim tensors or Python floats and get broadcast automatically by PyTorch.
"""
from __future__ import annotations
import hashlib
import math
import re
import torch

# Unified safety epsilon for division guards, domain clamping, and near-zero checks.
# Chosen to be well above float32 machine epsilon (~1.2e-7) while small enough
# to be invisible in image-processing contexts.
SAFE_EPSILON = 1e-8

# ── Sampler tensor cache ──────────────────────────────────────────────
# Caches reusable tensors for sampling functions keyed by (B, H, W, device).
# Avoids recreating batch index tensors and Lanczos tap offsets per call.
_sampler_cache: dict[tuple, torch.Tensor] = {}


def _get_batch_index(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Get or create cached batch index tensor [B, H, W] for advanced indexing."""
    key = ("bidx", B, H, W, str(device))
    cached = _sampler_cache.get(key)
    if cached is not None:
        return cached
    t = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)
    _sampler_cache[key] = t
    return t


def _get_lanczos_taps(device: torch.device) -> torch.Tensor:
    """Get or create cached Lanczos-3 tap offset tensor [-2, -1, 0, 1, 2, 3]."""
    key = ("ltaps", str(device))
    cached = _sampler_cache.get(key)
    if cached is not None:
        return cached
    t = torch.arange(-2, 4, device=device, dtype=torch.float32)
    _sampler_cache[key] = t
    return t


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
            "fract": TEXStdlib.fn_fract,
            "mod": TEXStdlib.fn_mod,

            "log2": TEXStdlib.fn_log2,
            "log10": TEXStdlib.fn_log10,
            "pow2": TEXStdlib.fn_pow2,
            "pow10": TEXStdlib.fn_pow10,
            "sinh": TEXStdlib.fn_sinh,
            "cosh": TEXStdlib.fn_cosh,
            "tanh": TEXStdlib.fn_tanh,
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
            "fetch_frame": TEXStdlib.fn_fetch_frame,
            "sample_frame": TEXStdlib.fn_sample_frame,

            # Noise
            "perlin": TEXStdlib.fn_perlin,
            "simplex": TEXStdlib.fn_simplex,
            "fbm": TEXStdlib.fn_fbm,

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
    def fn_sqrt(x):
        return torch.sqrt(torch.clamp(_to_tensor(x), min=0.0))

    @staticmethod
    def fn_pow(base, exp):
        return torch.pow(_to_tensor(base), _to_tensor(exp))

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
        safe_abs = torch.where(mask, torch.full_like(abs_t, SAFE_EPSILON), abs_t)
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
        """Dot product. Sums over the last dimension (channels)."""
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        return (a_t * b_t).sum(dim=-1)

    @staticmethod
    def fn_length(v):
        """Length (magnitude) of a vector. Operates on last dimension."""
        t = _to_tensor(v)
        if t.dim() >= 1 and t.shape[-1] in (3, 4):
            return torch.linalg.vector_norm(t, dim=-1)
        return torch.abs(t)

    @staticmethod
    def fn_distance(a, b):
        diff = _to_tensor(a) - _to_tensor(b)
        if diff.dim() >= 1 and diff.shape[-1] in (3, 4):
            return torch.linalg.vector_norm(diff, dim=-1)
        return torch.abs(diff)

    @staticmethod
    def fn_normalize(v):
        t = _to_tensor(v)
        if t.dim() >= 1 and t.shape[-1] in (3, 4):
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
            return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]
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

        # Use torch.where for branchless selection
        r = torch.where(i_mod == 0, v, torch.where(i_mod == 1, q, torch.where(
            i_mod == 2, p, torch.where(i_mod == 3, p, torch.where(i_mod == 4, t, v)))))
        g = torch.where(i_mod == 0, t, torch.where(i_mod == 1, v, torch.where(
            i_mod == 2, v, torch.where(i_mod == 3, q, torch.where(i_mod == 4, p, p)))))
        b = torch.where(i_mod == 0, p, torch.where(i_mod == 1, p, torch.where(
            i_mod == 2, t, torch.where(i_mod == 3, v, torch.where(i_mod == 4, v, q)))))

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
        s = torch.where(cmax > SAFE_EPSILON, diff / cmax, torch.zeros_like(cmax))

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
        img = _to_tensor(image)
        u = _to_tensor(u_coord)
        v = _to_tensor(v_coord)

        B, H, W, C = img.shape

        # grid_sample expects [B, C, H, W] input
        img_bchw = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # Convert from [0, 1] UV to [-1, 1] grid coords (grid_sample convention)
        grid_x = u * 2.0 - 1.0
        grid_y = v * 2.0 - 1.0

        # Build grid: needs shape [B, H_out, W_out, 2]
        if grid_x.dim() == 0:
            grid_x = grid_x.expand(B, H, W)
            grid_y = grid_y.expand(B, H, W)
        elif grid_x.dim() == 2:
            grid_x = grid_x.unsqueeze(0).expand(B, H, W)
            grid_y = grid_y.unsqueeze(0).expand(B, H, W)

        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]

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
        img = _to_tensor(image)
        px_t = _to_tensor(px)
        py_t = _to_tensor(py)

        B, H, W, C = img.shape

        # Convert to integer indices — skip round() if already long/int dtype
        if px_t.dtype in (torch.int32, torch.int64):
            px_i = torch.clamp(px_t.long(), 0, W - 1)
        else:
            px_i = torch.clamp(px_t.long(), 0, W - 1)
        if py_t.dtype in (torch.int32, torch.int64):
            py_i = torch.clamp(py_t.long(), 0, H - 1)
        else:
            py_i = torch.clamp(py_t.long(), 0, H - 1)

        # Expand scalars to spatial dims
        if px_i.dim() == 0:
            px_i = px_i.expand(B, H, W)
        if py_i.dim() == 0:
            py_i = py_i.expand(B, H, W)
        if px_i.dim() == 2:
            px_i = px_i.unsqueeze(0).expand(B, H, W)
        if py_i.dim() == 2:
            py_i = py_i.unsqueeze(0).expand(B, H, W)

        # Fast path for B=1 (common case) — avoid creating batch index tensor
        if B == 1:
            return img[0, py_i[0], px_i[0]].unsqueeze(0)

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
        img_bchw = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # Convert from [0, 1] UV to [-1, 1] grid coords (grid_sample convention)
        grid_x = u * 2.0 - 1.0
        grid_y = v * 2.0 - 1.0

        # Build grid: needs shape [B, H_out, W_out, 2]
        if grid_x.dim() == 0:
            # Scalar coords -> sample single point, expand to spatial grid
            grid_x = grid_x.expand(B, H, W)
            grid_y = grid_y.expand(B, H, W)
        elif grid_x.dim() == 2:
            # [H, W] -> [B, H, W]
            grid_x = grid_x.unsqueeze(0).expand(B, H, W)
            grid_y = grid_y.unsqueeze(0).expand(B, H, W)

        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]

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

        Lanczos-3 uses a 6x6 pixel neighborhood with sinc-based weights
        for the sharpest reconstruction with minimal ringing.

        Fully vectorized: all 36 taps are gathered and weighted in batched
        tensor ops with no Python loops.
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
        x_center = torch.floor(x)
        y_center = torch.floor(y)

        # Fractional part
        fx = x - x_center
        fy = y - y_center

        # Expand scalars to spatial dims
        if fx.dim() == 0:
            fx = fx.expand(B, H, W)
            fy = fy.expand(B, H, W)
            x_center = x_center.expand(B, H, W)
            y_center = y_center.expand(B, H, W)
        elif fx.dim() == 2:
            fx = fx.unsqueeze(0).expand(B, H, W)
            fy = fy.unsqueeze(0).expand(B, H, W)
            x_center = x_center.unsqueeze(0).expand(B, H, W)
            y_center = y_center.unsqueeze(0).expand(B, H, W)

        # Tap offsets: -2, -1, 0, 1, 2, 3 (6 taps for Lanczos-3)
        taps = _get_lanczos_taps(dev)  # [6] — cached

        # Compute 1-D Lanczos weights for x and y (separable)
        # fx/fy: [B, H, W] → [B, H, W, 1] - taps: [6] → weights: [B, H, W, 6]
        wx = _lanczos3(fx.unsqueeze(-1) - taps)  # [B, H, W, 6]
        wy = _lanczos3(fy.unsqueeze(-1) - taps)  # [B, H, W, 6]

        # 2-D weights via outer product: [B, H, W, 6, 6]
        weights_2d = wy.unsqueeze(-1) * wx.unsqueeze(-2)  # [B,H,W,6(y),6(x)]

        # Pixel coordinates for all 36 taps, clamped to image bounds
        # x_center: [B,H,W] → [B,H,W,1] + taps → [B,H,W,6]
        px_all = torch.clamp((x_center.unsqueeze(-1) + taps).long(), 0, W - 1)  # [B,H,W,6]
        py_all = torch.clamp((y_center.unsqueeze(-1) + taps).long(), 0, H - 1)  # [B,H,W,6]

        # Gather all 36 neighbor pixels in one advanced-indexing op
        # Build full index grids: [B,H,W,6(y),6(x)]
        b_idx = _get_batch_index(B, H, W, dev).unsqueeze(-1).unsqueeze(-1).expand(B, H, W, 6, 6)
        py_grid = py_all.unsqueeze(-1).expand(B, H, W, 6, 6)  # y taps along dim 3
        px_grid = px_all.unsqueeze(-2).expand(B, H, W, 6, 6)  # x taps along dim 4

        pixels = img[b_idx, py_grid, px_grid]  # [B, H, W, 6, 6, C]

        # Apply weights and sum: [B,H,W,6,6,1] * [B,H,W,6,6,C] → sum → [B,H,W,C]
        w = weights_2d.unsqueeze(-1)  # [B, H, W, 6, 6, 1]
        result = (pixels * w).sum(dim=(3, 4))  # [B, H, W, C]
        weight_sum = w.sum(dim=(3, 4))  # [B, H, W, 1]

        # Normalize (avoids edge darkening from clamped kernels)
        return result / (weight_sum + SAFE_EPSILON)

    # -- Noise functions ------------------------------------------------

    @staticmethod
    def fn_perlin(x, y):
        """2D Perlin noise. Returns float in [-1, 1]."""
        return _perlin2d(_to_tensor(x), _to_tensor(y))

    @staticmethod
    def fn_simplex(x, y):
        """2D Simplex noise. Returns float in [-1, 1]."""
        return _simplex2d(_to_tensor(x), _to_tensor(y))

    @staticmethod
    def fn_fbm(x, y, octaves):
        """Fractional Brownian Motion using Perlin noise."""
        oct_int = int(_to_float(octaves))
        return _fbm2d(_to_tensor(x), _to_tensor(y), oct_int)

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
            return torch.tensor(float(len(s)), dtype=torch.float32)
        if isinstance(s, list):
            return torch.tensor(float(len(s)), dtype=torch.float32)
        if isinstance(s, torch.Tensor):
            # Vec array [B,H,W,N,C] or [N,C]: element count is dim -2
            if s.dim() in (2, 5):
                return torch.tensor(float(s.shape[-2]), dtype=torch.float32)
            return torch.tensor(float(s.shape[-1]), dtype=torch.float32)
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
        return torch.tensor(1.0 if sub in s else 0.0, dtype=torch.float32)

    @staticmethod
    def fn_startswith(s, prefix):
        """Check if s starts with prefix. Returns 1.0 or 0.0."""
        if not (isinstance(s, str) and isinstance(prefix, str)):
            raise ValueError("startswith() expects two string arguments")
        return torch.tensor(1.0 if s.startswith(prefix) else 0.0, dtype=torch.float32)

    @staticmethod
    def fn_endswith(s, suffix):
        """Check if s ends with suffix. Returns 1.0 or 0.0."""
        if not (isinstance(s, str) and isinstance(suffix, str)):
            raise ValueError("endswith() expects two string arguments")
        return torch.tensor(1.0 if s.endswith(suffix) else 0.0, dtype=torch.float32)

    @staticmethod
    def fn_find(s, sub):
        """Find index of sub in s. Returns -1.0 if not found."""
        if not (isinstance(s, str) and isinstance(sub, str)):
            raise ValueError("find() expects two string arguments")
        return torch.tensor(float(s.find(sub)), dtype=torch.float32)

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
            return torch.tensor(float(int(s.strip())), dtype=torch.float32)
        except ValueError:
            raise ValueError(f"to_int(): cannot parse '{s}' as integer")

    @staticmethod
    def fn_to_float(s):
        """Parse float from string."""
        if not isinstance(s, str):
            raise ValueError("to_float() expects a string argument")
        try:
            return torch.tensor(float(s.strip()), dtype=torch.float32)
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
        return torch.tensor(float(s.count(sub)), dtype=torch.float32)

    @staticmethod
    def fn_matches(s, pattern):
        """Test if string matches a regex pattern. Returns 1.0 if the full string matches, 0.0 otherwise."""
        if not (isinstance(s, str) and isinstance(pattern, str)):
            raise ValueError("matches() expects two string arguments")
        try:
            return torch.tensor(1.0 if re.fullmatch(pattern, s) else 0.0, dtype=torch.float32)
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
        return torch.tensor(value, dtype=torch.float32)

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
        return torch.tensor(float(value), dtype=torch.float32)

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
        t = _to_tensor(arr)
        if t.dim() in (2, 5):  # vec array → sum along element dim, keep channels
            return t.sum(dim=-2)
        return t.sum(dim=-1)

    @staticmethod
    def fn_arr_min(arr):
        """Minimum element of an array per channel. Returns scalar (or vec) per pixel."""
        t = _to_tensor(arr)
        if t.dim() in (2, 5):
            return t.min(dim=-2).values
        return t.min(dim=-1).values

    @staticmethod
    def fn_arr_max(arr):
        """Maximum element of an array per channel. Returns scalar (or vec) per pixel."""
        t = _to_tensor(arr)
        if t.dim() in (2, 5):
            return t.max(dim=-2).values
        return t.max(dim=-1).values

    @staticmethod
    def fn_median(arr):
        """Median element of an array per channel. Returns scalar (or vec) per pixel."""
        t = _to_tensor(arr)
        if t.dim() in (2, 5):
            return torch.median(t, dim=-2).values
        return torch.median(t, dim=-1).values

    @staticmethod
    def fn_arr_avg(arr):
        """Average of array elements per channel. Returns scalar (or vec) per pixel."""
        t = _to_tensor(arr)
        if t.dim() in (2, 5):
            return t.mean(dim=-2)
        return t.mean(dim=-1)

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
        if img.dim() == 4:  # [B, H, W, C]
            return img.sum(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        if img.dim() == 3:  # [B, H, W] mask
            return img.sum(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        return img

    @staticmethod
    def fn_img_mean(image):
        """Mean of all pixels per channel per frame."""
        img = _to_tensor(image)
        if img.dim() == 4:
            return img.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        if img.dim() == 3:
            return img.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        return img

    @staticmethod
    def fn_img_min(image):
        """Min pixel value per channel per frame."""
        img = _to_tensor(image)
        if img.dim() == 4:
            return img.amin(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        if img.dim() == 3:
            return img.amin(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        return img

    @staticmethod
    def fn_img_max(image):
        """Max pixel value per channel per frame."""
        img = _to_tensor(image)
        if img.dim() == 4:
            return img.amax(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        if img.dim() == 3:
            return img.amax(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
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
    """Lanczos-3 kernel: sinc(x) * sinc(x/3) for |x| < 3, else 0."""
    x = x.float()
    ax = torch.abs(x)
    # sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1
    pix = math.pi * x
    pix3 = pix / 3.0
    sinc_x = torch.where(ax < SAFE_EPSILON, torch.ones_like(x), torch.sin(pix) / (pix + SAFE_EPSILON))
    sinc_x3 = torch.where(ax < SAFE_EPSILON, torch.ones_like(x), torch.sin(pix3) / (pix3 + SAFE_EPSILON))
    kernel = sinc_x * sinc_x3
    # Zero out beyond support radius of 3
    return torch.where(ax < 3.0, kernel, torch.zeros_like(kernel))


# -- Noise helpers (module-level) --------------------------------------

# Classic 256-entry Perlin permutation table
_PERM = torch.tensor([
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
    140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
    247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
    57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
    74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
    60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
    65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,
    200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,
    52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
    207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,
    119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
    129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
    218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,
    81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,
    184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,
    222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
], dtype=torch.int64)

# Doubled for overflow-free indexing
_PERM2 = torch.cat([_PERM, _PERM])

# 8 gradient directions for 2D Perlin noise (unit circle at 45-degree intervals)
_GRAD2 = torch.tensor([
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    [0.7071, 0.7071], [-0.7071, 0.7071],
    [0.7071, -0.7071], [-0.7071, -0.7071],
], dtype=torch.float32)

# 12 gradient directions for 2D Simplex noise
_GRAD2_SIMPLEX = torch.tensor([
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
    [1.0, 0.5], [-1.0, 0.5], [1.0, -0.5], [-1.0, -0.5],
], dtype=torch.float32)

# Simplex skew/unskew constants
_SKEW_2D = 0.5 * (math.sqrt(3.0) - 1.0)      # ~0.3660254
_UNSKEW_2D = (3.0 - math.sqrt(3.0)) / 6.0     # ~0.2113249


def _fade(t: torch.Tensor) -> torch.Tensor:
    """Quintic smoothstep: 6t^5 - 15t^4 + 10t^3. Ensures C2 continuity."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


_noise_tables_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

def _get_noise_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get device-local noise lookup tables (cached)."""
    key = str(device)
    cached = _noise_tables_cache.get(key)
    if cached is not None:
        return cached
    tables = (_PERM2.to(device), _GRAD2.to(device), _GRAD2_SIMPLEX.to(device))
    _noise_tables_cache[key] = tables
    return tables


def _noise_hash(xi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
    """Hash 2D integer coordinates via permutation table lookup."""
    perm = _get_noise_tables(xi.device)[0]
    idx_x = (xi % 256).clamp(0, 255)
    inner = perm[idx_x]
    idx_xy = ((inner + yi % 256) % 512).clamp(0, 511)
    return perm[idx_xy]


def _perlin_grad2(hash_val: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Gradient dot product for 2D Perlin noise using 8 gradient directions."""
    grad = _get_noise_tables(dx.device)[1]
    idx = (hash_val % 8).long()
    g = grad[idx]
    return g[..., 0] * dx + g[..., 1] * dy


def _perlin2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """2D Perlin noise. Returns float tensor in approximately [-1, 1]."""
    xi = torch.floor(x).long()
    yi = torch.floor(y).long()

    xf = x - torch.floor(x)
    yf = y - torch.floor(y)

    u = _fade(xf)
    v = _fade(yf)

    # Hash the 4 corner lattice points
    h00 = _noise_hash(xi, yi)
    h10 = _noise_hash(xi + 1, yi)
    h01 = _noise_hash(xi, yi + 1)
    h11 = _noise_hash(xi + 1, yi + 1)

    # Gradient dot products at each corner
    g00 = _perlin_grad2(h00, xf, yf)
    g10 = _perlin_grad2(h10, xf - 1.0, yf)
    g01 = _perlin_grad2(h01, xf, yf - 1.0)
    g11 = _perlin_grad2(h11, xf - 1.0, yf - 1.0)

    # Bilinear interpolation with fade curves
    lerp_x0 = torch.lerp(g00, g10, u)
    lerp_x1 = torch.lerp(g01, g11, u)
    return torch.lerp(lerp_x0, lerp_x1, v)


def _simplex2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """2D Simplex noise. Returns float tensor in approximately [-1, 1]."""
    grad = _get_noise_tables(x.device)[2]

    # Skew input space to determine simplex cell
    s = (x + y) * _SKEW_2D
    i = torch.floor(x + s).long()
    j = torch.floor(y + s).long()

    # Unskew back to (x, y) space
    t = (i + j).float() * _UNSKEW_2D
    X0 = i.float() - t
    Y0 = j.float() - t

    # Distances from cell origin
    x0 = x - X0
    y0 = y - Y0

    # Determine which simplex triangle we're in
    i1 = (x0 > y0).long()
    j1 = 1 - i1

    # Offsets for middle and last corners
    x1 = x0 - i1.float() + _UNSKEW_2D
    y1 = y0 - j1.float() + _UNSKEW_2D
    x2 = x0 - 1.0 + 2.0 * _UNSKEW_2D
    y2 = y0 - 1.0 + 2.0 * _UNSKEW_2D

    # Hash corners and get gradient indices
    h0 = _noise_hash(i, j)
    h1 = _noise_hash(i + i1, j + j1)
    h2 = _noise_hash(i + 1, j + 1)

    gi0 = (h0 % 12).long()
    gi1 = (h1 % 12).long()
    gi2 = (h2 % 12).long()

    # Corner contributions: radial falloff (0.5 - d²)⁴ × dot(gradient, offset)
    t0 = torch.clamp(0.5 - x0 * x0 - y0 * y0, min=0.0)
    t0 = t0 * t0 * t0 * t0
    g0 = grad[gi0]
    n0 = t0 * (g0[..., 0] * x0 + g0[..., 1] * y0)

    t1 = torch.clamp(0.5 - x1 * x1 - y1 * y1, min=0.0)
    t1 = t1 * t1 * t1 * t1
    g1 = grad[gi1]
    n1 = t1 * (g1[..., 0] * x1 + g1[..., 1] * y1)

    t2 = torch.clamp(0.5 - x2 * x2 - y2 * y2, min=0.0)
    t2 = t2 * t2 * t2 * t2
    g2 = grad[gi2]
    n2 = t2 * (g2[..., 0] * x2 + g2[..., 1] * y2)

    # Scale to ~[-1, 1]
    return 70.0 * (n0 + n1 + n2)


def _fbm2d(x: torch.Tensor, y: torch.Tensor, octaves: int) -> torch.Tensor:
    """Fractional Brownian Motion using Perlin noise.
    Persistence=0.5, lacunarity=2.0. Octaves clamped to 1-10."""
    octaves = max(1, min(octaves, 10))
    result = torch.zeros_like(x)
    amplitude = 1.0
    frequency = 1.0
    max_amp = 0.0

    for _ in range(octaves):
        result.add_(_perlin2d(x * frequency, y * frequency), alpha=amplitude)
        max_amp += amplitude
        amplitude *= 0.5
        frequency *= 2.0

    return result.div_(max_amp)


def _to_tensor(x) -> torch.Tensor:
    """Ensure a value is a torch.Tensor (skip recast if already float32)."""
    if x.__class__ is torch.Tensor:
        return x if x.dtype == torch.float32 else x.float()
    return torch.tensor(float(x), dtype=torch.float32)


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
