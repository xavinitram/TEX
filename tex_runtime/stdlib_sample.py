"""
TEX Standard Library — sampling, fetching and neighbourhood-filter builtins (one domain leaf of `stdlib.py`).

The `TEXStdlib` class-body section(s) Morphology (SL-4), Sampling (fetch/sample/mip, blur, bilateral, convolve, patch_dist) moved here verbatim, onto the `_StdlibSample`
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
    _build_sample_grid,
    _expand_to_bhw,
    _gauss_blur_bchw,
    _get_batch_index,
    _get_bchw,
    _get_grid_buf,
    _get_lanczos_taps,
    _get_mip_pyramid,
    _get_mip_pyramid_gauss,
    _grid_sample_f32,
    _host_scalar,
    _lanczos3,
    _pad_replicate_chunked,
    _provider_read,
    _sample_mip_trilinear,
    _to_float,
    _to_tensor,
    _uniform_grid,
)

# `TEXStdlib` is the class `stdlib.py` composes from every leaf. A leaf cannot import it at
# load time (the facade imports the leaves), so the facade BINDS it into this namespace the
# moment the class exists; the `TEXStdlib.fn_*(...)` delegations below then resolve at call
# time exactly as they did inside the one-file class. The spelling is load-bearing:
# `stdlib_registry._impl_looks_fragile` reads the literal `TEXStdlib.fn_*(` from the source
# to follow one level of delegation, so it must not be rewritten to the mixin's name.
TEXStdlib = None


class _StdlibSample:
    """sampling, fetching and neighbourhood-filter builtins: the `fn_*` methods `stdlib.py` mixes into `TEXStdlib`."""

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

    @stdlib("fetch_time", sig='fetch_time(source, t, px, py) \\u2192 vec', category='Batch / Temporal', spatial=True, sync=True, footprint='image', doc='Nearest-neighbour read of a HOST source at time t (DATA-7). Needs a registered FrameProvider.', ex='@OUT = fetch_time("plate", time, ix, iy);')
    @staticmethod
    def fn_fetch_time(source, t, px, py) -> torch.Tensor:
        """DATA-7: fetch a pixel from a host source frame at time `t`.

        The out-of-batch twin of `fetch_frame`. `fetch_frame` indexes INSIDE the marshalled
        batch; this reaches a frame the batch does not contain, through the `tex_provider`
        host seam — TEX never opens a file, the host decodes, and the pool remembers.

        Args:
            source: string source key the host's provider understands
            t: the SOURCE's own time. Must be uniform across the grid (E7003) — see
               `_uniform_time`'s note on why a per-pixel retime is refused rather than capped.
            px, py: pixel coordinates into the SOURCE's grid, which need not match the cook's.
        """
        img, px_t, py_t = _provider_read(source, t, "fetch", px, py)
        _b, Hs, Ws, _c = img.shape
        px_i = torch.clamp(torch.round(px_t).long(), 0, Ws - 1)
        py_i = torch.clamp(torch.round(py_t).long(), 0, Hs - 1)
        shape = torch.broadcast_shapes(px_i.shape, py_i.shape) or _uniform_grid() or (1, 1, 1)
        return img[0][py_i.expand(shape), px_i.expand(shape)]

    @stdlib("sample_time", sig='sample_time(source, t, u, v) \\u2192 vec', category='Batch / Temporal', spatial=True, sync=True, footprint='image', doc='Bilinear sample of a HOST source at time t (DATA-7). Needs a registered FrameProvider.', ex='@OUT = sample_time("plate", time - 0.5, u, v);')
    @staticmethod
    def fn_sample_time(source, t, u_coord, v_coord) -> torch.Tensor:
        """DATA-7: bilinear sample of a host source frame at time `t`.

        The provider MAY interpolate between the two frames bracketing `t` — that is the
        difference from `fetch_time`, and it is the provider's to make, which is why the two
        modes are cached separately. The bilinear math here mirrors `fn_sample_frame`'s
        expression for expression, so the two agree where they overlap.
        """
        img, u, v = _provider_read(source, t, "sample", u_coord, v_coord)
        _b, Hs, Ws, _c = img.shape
        shape = torch.broadcast_shapes(u.shape, v.shape) or _uniform_grid() or (1, 1, 1)
        x = torch.clamp(u * (Ws - 1), 0, Ws - 1).expand(shape)
        y = torch.clamp(v * (Hs - 1), 0, Hs - 1).expand(shape)

        x0 = torch.floor(x).long()
        x1 = torch.clamp(x0 + 1, 0, Ws - 1)
        y0 = torch.floor(y).long()
        y1 = torch.clamp(y0 + 1, 0, Hs - 1)

        fx = (x - x0.to(x.dtype)).unsqueeze(-1)
        fy = (y - y0.to(y.dtype)).unsqueeze(-1)

        plane = img[0]
        v00 = plane[y0, x0]
        v01 = plane[y0, x1]
        v10 = plane[y1, x0]
        v11 = plane[y1, x1]
        return v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + \
               v10 * (1 - fx) * fy + v11 * fx * fy

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

    @stdlib("gauss_blur", sig='gauss_blur(img, sigma) \\u2192 vec', category='Sampling', spatial=True, sync=True, footprint=('halo_arg', 1, 3.0), doc='Separable Gaussian blur. Kernel radius ≈ 3×sigma pixels. Replicate border padding.', ex='@OUT = gauss_blur(@A, 2.0);')
    @staticmethod
    def fn_gauss_blur(image, sigma) -> torch.Tensor:
        """Separable Gaussian blur.

        Args:
            image: [B, H, W, C] tensor
            sigma: float — standard deviation in pixels (kernel radius ≈ 3*sigma)

        Returns blurred [B, H, W, C] tensor with replicate border handling.
        """
        img = image if image.__class__ is torch.Tensor else _to_tensor(image)
        # The Gaussian kernel radius is a host-side Python int (radius ~= 3*sigma), so
        # a SCALAR is unavoidable here — a device round trip is not. `_host_scalar`
        # answers from the number a literal / `$param` / folded constant was minted
        # from; only a sigma genuinely computed on the device is read back (PERF-2).
        sigma_val = _host_scalar(sigma)
        if sigma_val is None:
            sigma_t = sigma if sigma.__class__ is torch.Tensor else _to_tensor(sigma)
            sigma_val = sigma_t.item()
        sigma_val = max(sigma_val, 0.0)
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
        # Both sigmas size the window / the weights host-side; PERF-2 resolves them
        # from the minted host value where there is one, and reads back where there
        # is not (see `_host_scalar`).
        ss = _host_scalar(sigma_s)
        if ss is None:
            ss = sigma_s.item() if torch.is_tensor(sigma_s) else float(sigma_s)
        sr = _host_scalar(sigma_r)
        if sr is None:
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

    # ASK-1: native convolution. `kernel` is a second IMAGE/MASK BINDING, read whole —
    # not an ARRAY literal (an array is expanded to one full frame per tap by the
    # interpreter, `interpreter.py:1611-1617`) and not a mat3/mat4 (capped at 4x4,
    # `DEVELOPMENT.md:165`). footprint='image', because `('halo_arg', kernel)` cannot be
    # built here: `tex_roi._call_reach` resolves a halo_arg only from a folded
    # NumberLiteral, so a kernel BINDING can only ever resolve 'unbounded' — never a
    # narrowable radius — and the variant that WOULD resolve accumulates the kernel into
    # the outer halo ctx, so a narrowed ROI slices the kernel binding itself (wrong pixels
    # the moment ROI narrows). Recorded in DEVELOPMENT.md §"Rejected design decisions".
    @stdlib("convolve", sig='convolve(img, kernel[, normalize]) \\u2192 vec', category='Sampling',
            spatial=True, sync=True, footprint='image',
            doc='General image-kernel convolution (the kernel is flipped, not correlated). '
                'kernel is a second IMAGE/MASK binding, read whole; kernel size in [1,257]. Its '
                'channel count broadcasts (1 plane -> every image channel) or weights per '
                'channel (== image channels, depthwise). normalize=1 (default) divides by the '
                'per-channel kernel sum; 0 returns the raw weighted sum. Replicate border '
                'padding.',
            ex='@OUT = convolve(@A, @kernel);')
    @staticmethod
    def fn_convolve(image, kernel, normalize=1) -> torch.Tensor:
        """General depthwise image-kernel convolution.

        Args:
            image: [B, H, W, C] tensor, or [B, H, W] mask.
            kernel: a second IMAGE/MASK tensor (batch must be 1), read whole. Its
                channel count Ck must be 1 (one weight plane, broadcast to every image
                channel) or equal the image's channel count C (depthwise: one weight
                plane per channel) — anything else raises. kH, kW must be in [1, 257].
            normalize: 1 (default, truthy) divides the result by the kernel's
                per-channel sum (safe-divide guarded near zero); 0 (falsy) returns the
                raw weighted sum. Resolved host-side (one `.item()` sync), like
                gauss_blur's sigma.

        TRUE convolution — the kernel is flipped before the tap, not correlated — the
        semantics a stock "Convolve" tool's own program comment documents; a
        correlation would silently mirror any asymmetric kernel. Replicate border
        padding (matches sample/gauss_blur/erode/dilate), chunked so an oversized
        kernel never asks a single F.pad call for more margin than an axis has. Raises
        (never clamps) on a kernel batch > 1, an out-of-range kernel size, or a
        channel-count mismatch.
        """
        img = image if image.__class__ is torch.Tensor else _to_tensor(image)
        ker = kernel if kernel.__class__ is torch.Tensor else _to_tensor(kernel)

        squeeze = img.dim() == 3                              # [B,H,W] mask -> add a channel
        x = _get_bchw(img.unsqueeze(-1) if squeeze else img)  # [B,C,H,W]
        C = x.shape[1]

        k4 = ker.unsqueeze(-1) if ker.dim() == 3 else ker      # [Bk,kH,kW,Ck]
        if k4.dim() != 4:
            raise ValueError(f"convolve(): kernel must be an image or mask, got rank {ker.dim()}")
        Bk, kH, kW, Ck = k4.shape
        if Bk != 1:
            raise ValueError(f"convolve(): kernel batch must be 1, got {Bk}")
        if not (1 <= kH <= 257 and 1 <= kW <= 257):
            raise ValueError(f"convolve(): kernel size {kH}x{kW} out of range [1,257]")
        if kH * kW > 66049:
            raise ValueError(f"convolve(): kernel area {kH * kW} exceeds the 257² (66049) cap")
        if Ck not in (1, C):
            raise ValueError(f"convolve(): kernel channel count {Ck} must be 1 or match the "
                              f"image's {C}")

        # TRUE convolution: flip the kernel. [Ck,1,kH,kW] -> (broadcast Ck==1 -> C) -> [C,1,kH,kW].
        w = torch.flip(k4[0].permute(2, 0, 1).unsqueeze(1), dims=[-2, -1])
        if Ck == 1 and C > 1:
            w = w.expand(C, 1, kH, kW)
        # M-3 reconcile: conv2d requires kernel and input to share a dtype; the kernel is
        # reconciled TO the image, never the reverse (image data follows self._dtype).
        w = w.to(x.dtype)

        pad_l, pad_r = kW // 2, kW - 1 - kW // 2
        pad_t, pad_b = kH // 2, kH - 1 - kH // 2
        padded = _pad_replicate_chunked(x, pad_l, pad_r, pad_t, pad_b)
        out = torch.nn.functional.conv2d(padded, w, groups=C)

        norm_val = _host_scalar(normalize)
        if norm_val is None:
            norm_t = normalize if normalize.__class__ is torch.Tensor else _to_tensor(normalize)
            norm_val = norm_t.item()
        if norm_val != 0:
            ksum = w.sum(dim=(1, 2, 3)).view(1, C, 1, 1)      # per-channel kernel sum
            out = TEXStdlib._safe_div(out, ksum)

        result = out.permute(0, 2, 3, 1)                       # [B,H,W,C]
        return result.squeeze(-1) if squeeze else result

    @staticmethod
    def _uniform_scalar_or_raise(x, argname: str) -> int:
        """The single uniform value `x` names, as an int, or a raise stating how many
        distinct values it saw.

        ASK-13: `patch_dist`'s dx/dy/radius resolve host-side (one `.item()`-shaped
        check each), exactly like gauss_blur's sigma or convolve's normalize flag
        above. Unlike those, a per-pixel value here would be a data-dependent gather
        with no honest cost bound, so it is refused rather than silently meaning one
        of its values. Same
        refusal SHAPE as `tex_provider._uniform_time`'s E7003 — numel==1 / all-equal /
        else raise naming the distinct count — restated for a stdlib argument instead
        of a host `t`. Not the same error FAMILY: E7xxx is `tex_provider.py`'s host-I/O
        class; this is a plain stdlib argument check, raised the same way fn_convolve's
        kernel-shape ValueErrors are above, so both tiers (which call this identical
        function) raise identically instead of one of them hitting a raw torch
        RuntimeError from an ambiguous `.item()`.
        """
        if not isinstance(x, torch.Tensor):
            return int(x)
        if x.numel() == 1:
            v = _host_scalar(x)
            return int(v if v is not None else x.reshape(()).item())
        flat = x.reshape(-1)
        if bool(torch.all(flat == flat[0])):
            return int(flat[0].item())
        n = int(torch.unique(flat).numel())
        raise ValueError(
            f"patch_dist(): '{argname}' must be uniform across the grid (one value "
            f"per cook), but saw {n} distinct values. Hoist it out of the pixel grid "
            f"(a $param or a scalar expression) — a per-pixel offset is an unbounded "
            f"data-dependent gather with no honest cost bound."
        )

    # ASK-13: patch-distance primitive. `dx`/`dy` are integer PIXEL offsets, not a
    # vec2 and not UV — pixels because the UV tap-step is off by one pixel in W and
    # is itself a separate ask; two scalars because fn_fetch's own
    # (img, px, py) order already fixes x-then-y here. footprint='image': the true
    # reach is radius + max(|dx|,|dy|) — TWO arguments — and the ROI-1 descriptor
    # grammar reads exactly one; `('halo_arg', ...)` would under-pad by the offset the
    # moment ROI narrows or halo-tiles a program that uses this call. Recorded in
    # DEVELOPMENT.md §"Rejected design decisions".
    @stdlib("patch_dist", sig='patch_dist(img, dx, dy, radius) \\u2192 float', category='Sampling',
            spatial=True, sync=True, footprint='image',
            doc='Mean squared difference between the patch at this pixel and the patch at (dx, dy) pixels away. The non-local-means core.',
            ex='float d = patch_dist(@A.rgb, 3, -2, 1);')
    @staticmethod
    def fn_patch_dist(image, dx, dy, radius) -> torch.Tensor:
        """Per-pixel mean squared difference between the (2r+1)^2 patch centred at
        this pixel and the patch centred at this pixel + (dx, dy) pixels, averaged
        over the patch AND over the channels of `image` as passed.

        Args:
            image: [B, H, W, C] tensor, or [B, H, W] mask.
            dx, dy: integer pixel offsets. Must be uniform across the grid (one
                value per cook) — see `_uniform_scalar_or_raise`; a per-pixel
                offset raises rather than silently meaning one of its values.
            radius: patch half-size; uniform-or-raise like dx/dy, then clamped to
                [0, 32] (mirrors `_morph`'s defensive clamp — the real range is 1-3).

        Replicate border padding throughout (matches sample/gauss_blur/erode/dilate/
        convolve), via `_pad_replicate_chunked` for both the (dx, dy) shift and the
        box-mean pad — never a single unchunked F.pad call. The shift is a pad+narrow
        VIEW (no gather, no index tensors); the box mean is separable with the
        summation order pinned (ascending offset, W pass then H pass) — explicit
        slice-adds, never cumsum (integral-image cancellation) or a ones-kernel
        conv2d (unpinned algorithm selection). No codegen emitter: codegen's general
        function-call fallback calls this identical callable, so interp and codegen
        are bit-exact by construction (invariant #2), not by parallel review.
        """
        img = _to_tensor(image)
        r = TEXStdlib._uniform_scalar_or_raise(radius, "radius")
        sx = TEXStdlib._uniform_scalar_or_raise(dx, "dx")
        sy = TEXStdlib._uniform_scalar_or_raise(dy, "dy")
        r = min(max(r, 0), 32)

        squeeze = img.dim() == 3                              # [B,H,W] mask -> add a channel
        x = _get_bchw(img.unsqueeze(-1) if squeeze else img)  # [B,C,H,W]
        H, W = x.shape[-2], x.shape[-1]

        # Shifted field: pad by |sx|/|sy| then NARROW to a view reading pixel+(dx,dy),
        # replicate-clamped at the border — O(1) extra, no gather, no index tensors.
        # Bound the offset actually used to (extent-1+r) per axis, sign preserved: with
        # replicate edges, any |shift| beyond that already reads only the edge-replicated
        # value (the true reach the ROI-1 footprint comment above already names), so this
        # clamp cannot change a single output value — only how large the pad allocates.
        bx, by = (W - 1) + r, (H - 1) + r
        sx, sy = max(-bx, min(sx, bx)), max(-by, min(sy, by))
        ax, ay = abs(sx), abs(sy)
        padded = _pad_replicate_chunked(x, ax, ax, ay, ay)
        x_shift = padded.narrow(-1, ax + sx, W).narrow(-2, ay + sy, H)

        # Channel mean FIRST (the box sum below then runs on one plane, not C).
        d2 = ((x - x_shift) ** 2).mean(dim=1)                  # [B,H,W]
        if r == 0:
            return d2                                          # a 1x1 patch IS this

        # Separable box mean over the (2r+1)^2 patch, ascending-offset summation.
        d2c = d2.unsqueeze(1)                                  # [B,1,H,W]
        pad_w = _pad_replicate_chunked(d2c, r, r, 0, 0)
        acc = pad_w[..., 0:W]
        for k in range(1, 2 * r + 1):
            acc = acc + pad_w[..., k:k + W]
        pad_h = _pad_replicate_chunked(acc, 0, 0, r, r)
        acc2 = pad_h[..., 0:H, :]
        for k in range(1, 2 * r + 1):
            acc2 = acc2 + pad_h[..., k:k + H, :]
        return (acc2 / float((2 * r + 1) ** 2)).squeeze(1)     # [B,H,W]

    @stdlib("sample_mip_gauss", sig='sample_mip_gauss(img, u, v, lod) \\u2192 vec', category='Sampling', spatial=True, sync=True, footprint='image', doc='Gaussian-prefiltered mipmap sampling. Smoother pyramid (sigma=1.13) gives ~5 dB better exponential blur accuracy vs sample_mip.', ex='@OUT = sample_mip_gauss(@A, u, v, 2.5);')
    @staticmethod
    def fn_sample_mip_gauss(image, u_coord, v_coord, lod) -> torch.Tensor:
        """Sample with Gaussian-prefiltered mipmap (sigma=1.13 pyramid).

        Same interface as sample_mip but uses a Gaussian pre-blur before each
        2x downsample, producing SIGMA_C ≈ 0.825. This gives ~5 dB better
        accuracy for exponential blur reconstruction vs the area-downsample pyramid.
        """
        return _sample_mip_trilinear(image, u_coord, v_coord, lod, _get_mip_pyramid_gauss)
