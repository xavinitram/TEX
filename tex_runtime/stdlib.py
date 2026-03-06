"""
TEX Standard Library — runtime implementations of built-in functions.

All functions operate on PyTorch tensors. Scalars are represented as
0-dim tensors or Python floats and get broadcast automatically by PyTorch.
"""
from __future__ import annotations
import math
import re
import torch


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

            # Array operations
            "sort": TEXStdlib.fn_sort,
            "reverse": TEXStdlib.fn_reverse,
            "arr_sum": TEXStdlib.fn_arr_sum,
            "arr_min": TEXStdlib.fn_arr_min,
            "arr_max": TEXStdlib.fn_arr_max,
            "median": TEXStdlib.fn_median,
            "arr_avg": TEXStdlib.fn_arr_avg,
            "join": TEXStdlib.fn_join,

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
        return torch.asin(_to_tensor(x))

    @staticmethod
    def fn_acos(x):
        return torch.acos(_to_tensor(x))

    @staticmethod
    def fn_atan(x):
        return torch.atan(_to_tensor(x))

    @staticmethod
    def fn_atan2(y, x):
        return torch.atan2(_to_tensor(y), _to_tensor(x))

    @staticmethod
    def fn_sqrt(x):
        return torch.sqrt(_to_tensor(x))

    @staticmethod
    def fn_pow(base, exp):
        return torch.pow(_to_tensor(base), _to_tensor(exp))

    @staticmethod
    def fn_exp(x):
        return torch.exp(_to_tensor(x))

    @staticmethod
    def fn_log(x):
        return torch.log(_to_tensor(x))

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
        return torch.fmod(_to_tensor(a), _to_tensor(b))

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
        return a_t + (b_t - a_t) * t_t

    @staticmethod
    def fn_fit(val, old_min, old_max, new_min, new_max):
        """Remap val from [old_min, old_max] to [new_min, new_max]."""
        v = _to_tensor(val)
        o_min, o_max = _to_tensor(old_min), _to_tensor(old_max)
        n_min, n_max = _to_tensor(new_min), _to_tensor(new_max)
        t = (v - o_min) / (o_max - o_min + 1e-8)
        return n_min + t * (n_max - n_min)

    @staticmethod
    def fn_smoothstep(edge0, edge1, x):
        e0, e1, xv = _to_tensor(edge0), _to_tensor(edge1), _to_tensor(x)
        t = torch.clamp((xv - e0) / (e1 - e0 + 1e-8), 0.0, 1.0)
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
            return torch.sqrt((t * t).sum(dim=-1))
        return torch.abs(t)

    @staticmethod
    def fn_distance(a, b):
        diff = _to_tensor(a) - _to_tensor(b)
        if diff.dim() >= 1 and diff.shape[-1] in (3, 4):
            return torch.sqrt((diff * diff).sum(dim=-1))
        return torch.abs(diff)

    @staticmethod
    def fn_normalize(v):
        t = _to_tensor(v)
        if t.dim() >= 1 and t.shape[-1] in (3, 4):
            norm = torch.sqrt((t * t).sum(dim=-1, keepdim=True))
            return t / (norm + 1e-8)
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
        diff = cmax - cmin + 1e-8

        # Hue
        h = torch.where(cmax == r, torch.fmod((g - b) / diff, 6.0),
            torch.where(cmax == g, (b - r) / diff + 2.0,
                                   (r - g) / diff + 4.0))
        h = h / 6.0  # normalize to [0, 1]
        h = torch.fmod(h + 1.0, 1.0)  # ensure positive

        # Saturation
        s = torch.where(cmax > 1e-8, diff / cmax, torch.zeros_like(cmax))

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
        """
        img = _to_tensor(image)
        u = _to_tensor(u_coord)
        v = _to_tensor(v_coord)

        B, H, W, C = img.shape

        # Convert from [0,1] to pixel coordinates
        x = u * (W - 1)
        y = v * (H - 1)

        # Expand scalars to spatial dims
        if x.dim() == 0:
            x = x.expand(B, H, W)
            y = y.expand(B, H, W)
        elif x.dim() == 2:
            x = x.unsqueeze(0).expand(B, H, W)
            y = y.unsqueeze(0).expand(B, H, W)

        # Clamp to valid range
        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)

        # Bilinear interpolation
        x0 = torch.floor(x).long()
        x1 = torch.clamp(x0 + 1, 0, W - 1)
        y0 = torch.floor(y).long()
        y1 = torch.clamp(y0 + 1, 0, H - 1)

        fx = (x - x0.float()).unsqueeze(-1)
        fy = (y - y0.float()).unsqueeze(-1)

        b_idx = torch.arange(B, device=img.device).view(B, 1, 1).expand(B, H, W)

        v00 = img[b_idx, y0, x0]
        v01 = img[b_idx, y0, x1]
        v10 = img[b_idx, y1, x0]
        v11 = img[b_idx, y1, x1]

        result = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + \
                 v10 * (1 - fx) * fy + v11 * fx * fy
        return result

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

        # Round to nearest integer and clamp
        px_i = torch.clamp(torch.round(px_t).long(), 0, W - 1)
        py_i = torch.clamp(torch.round(py_t).long(), 0, H - 1)

        # Build batch index — expand scalars to spatial dims
        if px_i.dim() == 0:
            px_i = px_i.expand(B, H, W)
        if py_i.dim() == 0:
            py_i = py_i.expand(B, H, W)
        if px_i.dim() == 2:
            # [H, W] -> [B, H, W]
            px_i = px_i.unsqueeze(0).expand(B, H, W)
        if py_i.dim() == 2:
            py_i = py_i.unsqueeze(0).expand(B, H, W)

        b_idx = torch.arange(B, device=img.device).view(B, 1, 1).expand(B, H, W)

        return img[b_idx, py_i, px_i]

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

        # Convert from [0,1] to pixel coordinates
        x = u * (W - 1)
        y = v * (H - 1)

        # Expand scalars to spatial dims
        if f_idx.dim() == 0:
            f_idx = f_idx.expand(B, H, W)
        if f_idx.dim() == 2:
            f_idx = f_idx.unsqueeze(0).expand(B, H, W)
        if x.dim() == 0:
            x = x.expand(B, H, W)
            y = y.expand(B, H, W)
        elif x.dim() == 2:
            x = x.unsqueeze(0).expand(B, H, W)
            y = y.unsqueeze(0).expand(B, H, W)

        # Clamp to valid range
        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)

        # Bilinear interpolation
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
        """
        img = _to_tensor(image)
        u = _to_tensor(u_coord)
        v = _to_tensor(v_coord)

        B, H, W, C = img.shape

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

        b_idx = torch.arange(B, device=img.device).view(B, 1, 1).expand(B, H, W)

        # Lanczos-3: 6 taps centered around the sample point (-2..+3)
        # Separable: compute x-weights and y-weights independently, then combine
        result = torch.zeros(B, H, W, C, device=img.device)
        weight_sum = torch.zeros(B, H, W, 1, device=img.device)

        for j in range(-2, 4):  # 6 vertical taps
            wy = _lanczos3(fy - j)  # [B, H, W]

            for i in range(-2, 4):  # 6 horizontal taps
                wx = _lanczos3(fx - i)  # [B, H, W]

                # Pixel coordinates, clamped to bounds
                px = torch.clamp((x_center + i).long(), 0, W - 1)
                py = torch.clamp((y_center + j).long(), 0, H - 1)

                # Fetch pixel
                pixel = img[b_idx, py, px]  # [B, H, W, C]

                # Combined weight
                w = (wx * wy).unsqueeze(-1)  # [B, H, W, 1]
                result = result + pixel * w
                weight_sum = weight_sum + w

        # Normalize (avoids edge darkening from clamped kernels)
        return result / (weight_sum + 1e-8)

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
    def fn_replace(s, old, new):
        """Replace all occurrences of old with new."""
        if not all(isinstance(x, str) for x in (s, old, new)):
            raise ValueError("replace() expects three string arguments")
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
    sinc_x = torch.where(ax < 1e-7, torch.ones_like(x), torch.sin(pix) / (pix + 1e-8))
    sinc_x3 = torch.where(ax < 1e-7, torch.ones_like(x), torch.sin(pix3) / (pix3 + 1e-8))
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


def _noise_hash(xi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
    """Hash 2D integer coordinates via permutation table lookup."""
    perm = _PERM2.to(xi.device)
    idx_x = (xi % 256).clamp(0, 255)
    inner = perm[idx_x]
    idx_xy = ((inner + yi % 256) % 512).clamp(0, 511)
    return perm[idx_xy]


def _perlin_grad2(hash_val: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Gradient dot product for 2D Perlin noise using 8 gradient directions."""
    grad = _GRAD2.to(dx.device)
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
    lerp_x0 = g00 + u * (g10 - g00)
    lerp_x1 = g01 + u * (g11 - g01)
    return lerp_x0 + v * (lerp_x1 - lerp_x0)


def _simplex2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """2D Simplex noise. Returns float tensor in approximately [-1, 1]."""
    grad = _GRAD2_SIMPLEX.to(x.device)

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
        result = result + amplitude * _perlin2d(x * frequency, y * frequency)
        max_amp += amplitude
        amplitude *= 0.5
        frequency *= 2.0

    return result / max_amp


def _to_tensor(x) -> torch.Tensor:
    """Ensure a value is a torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x.float()
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
