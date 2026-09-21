"""
TEX Standard Library — distance-field builtins: SDF primitives, smooth min/max, gradient sampling (one domain leaf of `stdlib.py`).

The `TEXStdlib` class-body section(s) SDF primitives, Smooth min/max, Gradient sampling moved here verbatim, onto the `_StdlibSdf`
mixin. `stdlib.py` composes the leaves' mixins into `TEXStdlib` in the class body's original
section order, which is the REG-1 registration order (`help_entries()`, the generated
reference and the help panel all read it) — so import this leaf THROUGH `stdlib`, not
directly, unless registering only this domain is what you want.
"""
from __future__ import annotations
import math
import torch
from .stdlib_registry import stdlib
from .stdlib_core import (
    SAFE_EPSILON,
    _lerp_f32,
    _to_float,
    _to_tensor,
)

# `TEXStdlib` is the class `stdlib.py` composes from every leaf. A leaf cannot import it at
# load time (the facade imports the leaves), so the facade BINDS it into this namespace the
# moment the class exists; the `TEXStdlib.fn_*(...)` delegations below then resolve at call
# time exactly as they did inside the one-file class. The spelling is load-bearing:
# `stdlib_registry._impl_looks_fragile` reads the literal `TEXStdlib.fn_*(` from the source
# to follow one level of delegation, so it must not be rewritten to the mixin's name.
TEXStdlib = None


class _StdlibSdf:
    """distance-field builtins: SDF primitives, smooth min/max, gradient sampling: the `fn_*` methods `stdlib.py` mixes into `TEXStdlib`."""

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
