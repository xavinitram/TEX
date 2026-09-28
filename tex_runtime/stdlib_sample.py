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
    GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA,
    _build_sample_grid,
    _dtype_rounded,
    _expand_to_bhw,
    _gauss_blur_bchw,
    _gauss_blur_auto,
    _require_finite_arg,
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
    poll_cook_cancel,
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
    # RADIUS-50a/MORPH-50: no clamp — the author's rule ("all blurs and erodes should
    # support arbitrarily large radiuses") ruled a silent `min(r, 256)` a bug, not a
    # safety valve. Hybrid dispatch (RADIUS-50a-design.md D1, option 1, recorded
    # 2026-09-27): `r` at or below `_MORPH_VANHERK_CROSSOVER` runs the ORIGINAL
    # iterative separable 3-window loop below, byte-for-byte, so the default path's
    # cost is unchanged (invariant 7 — this is the radius band every existing program
    # actually uses). Above the crossover, `_morph_vanherk` below runs a van
    # Herk/Gil-Werman separable running extremum: `O(N)` per line, independent of `r`,
    # with a whole-line reduction shortcut once the window already reaches every
    # pixel on that line (see `_running_extreme_1d`) — so an absurd `r` (millions)
    # costs no more than `r == 8192` does: the shortcut fires long before either
    # dimension's block machinery would grow past `O(image size)`. That is also why no
    # limit is kept at all (the author's fallback, "or a limit ~8192px", is the OTHER
    # branch of the ruling — this lane took the unconditional one because van Herk
    # makes it both correct and fast, per RADIUS-50a-design.md §1's AUTHOR DECISIONS).
    # min is exact via `torch.cummin`, not `-max(-x)` — one fewer negate per pass, same
    # bit pattern (both are plain comparison reductions; there is no summation order to
    # diverge on either way, which is the same reasoning gauss_blur's own bit-exactness
    # note already gives). A square structuring element is separable either way.
    # Non-local (reads neighbours): excluded from tiling and from CUDA-graph capture
    # (the radius resolves via .item()) — `footprint=('halo_arg', 1)` is unchanged.
    _MORPH_VANHERK_CROSSOVER = 10  # measured crossover, this implementation and box.

    @staticmethod
    def _morph(image, radius, grow: bool):
        img = _to_tensor(image)
        radius_val = _to_float(radius)
        _require_finite_arg("erode" if not grow else "dilate", "radius", radius_val)  # A7
        r = max(0, int(radius_val))
        if r == 0:
            return img
        squeeze = img.dim() == 3          # [B,H,W] mask -> add a channel
        x = (img.unsqueeze(-1) if squeeze else img).permute(0, 3, 1, 2)  # [B,C,H,W]
        if r <= TEXStdlib._MORPH_VANHERK_CROSSOVER:
            x = TEXStdlib._morph_iterative(x, r, grow)
        else:
            x = TEXStdlib._morph_vanherk(x, r, grow)
        x = x.permute(0, 2, 3, 1)         # [B,H,W,C]
        return x.squeeze(-1) if squeeze else x

    @staticmethod
    def _morph_iterative(x, r: int, grow: bool):
        """The original small-`r` path, unmodified: iterating a 3-window `r` times
        equals a single (2r+1)-window (dilation/erosion by a flat SE is associative),
        so this stays `O(1)` extra memory in `r` — a 3-tensor transient per pass."""
        op = torch.amax if grow else torch.amin
        pad = torch.nn.functional.pad
        for _ in range(r):
            # PACE-47d (Gap 1): `erode`/`dilate` are footprint=halo_arg (registry-derived
            # heavy) and genuinely multi-pass at radius > 1 (one horizontal+vertical pass
            # per unit of radius, up to the crossover) -- the same "no internal poll
            # opportunity" gap PACE-47c left open is closed here the same way gauss_blur's
            # own is: a forced record between passes, so a cancel fired mid-loop can't
            # leave more than one iteration's device work unbounded regardless of stride.
            poll_cook_cancel(heavy=True)
            xp = pad(x, (1, 1, 0, 0), mode="replicate")               # horizontal
            x = op(torch.stack([xp[..., :-2], xp[..., 1:-1], xp[..., 2:]]), dim=0)
            xp = pad(x, (0, 0, 1, 1), mode="replicate")               # vertical
            x = op(torch.stack([xp[..., :-2, :], xp[..., 1:-1, :], xp[..., 2:, :]]), dim=0)
        return x

    @staticmethod
    def _morph_vanherk(x, r: int, grow: bool):
        """The large-`r` path: two separable 1-D running-extremum passes (van
        Herk/Gil-Werman), width then height. Each pass is `O(N)` in that dimension's
        length, independent of `r` — see `_running_extreme_1d`."""
        poll_cook_cancel(heavy=True)
        x = TEXStdlib._running_extreme_1d(x, r, dim=-1, grow=grow)    # horizontal (W)
        poll_cook_cancel(heavy=True)
        x = TEXStdlib._running_extreme_1d(x, r, dim=-2, grow=grow)    # vertical (H)
        return x

    @staticmethod
    def _running_extreme_1d(x: torch.Tensor, r: int, dim: int, grow: bool) -> torch.Tensor:
        """Sliding-window max (`grow=True`) or min over a `(2r+1)`-wide window along
        `dim`, replicate-boundary, exact for every `r >= 1` — the van Herk/Gil-Werman
        algorithm: pad by `r` (replicate), split into blocks of width `w = 2r+1`,
        take a forward-cumulative extremum `g` and a backward-cumulative extremum `h`
        within each block, then `out[i] = combine(h[i], g[i+w-1])` — because a
        `w`-wide window starting at padded index `i` spans at most two `w`-sized
        blocks. `O(N)` total work, independent of `w`.

        Whole-line shortcut: once `r >= N - 1`, EVERY output position's window
        already reaches both ends of the (replicate-padded) line, so the honest
        answer is the line's own extremum, broadcast — no block machinery, no
        growing-with-`r` memory. This is what keeps an arbitrarily large `r` bounded
        to `O(image size)`: the block algorithm below is only ever asked to build
        blocks up to about `4r` wide when `r < N - 1`, i.e. never past `O(N)`.
        """
        N = x.shape[dim]
        if r >= N - 1:
            reduced = x.amax(dim=dim, keepdim=True) if grow else x.amin(dim=dim, keepdim=True)
            return reduced.expand(x.shape).contiguous()

        w = 2 * r + 1
        pad = torch.nn.functional.pad
        x_last = x.movedim(dim, -1).contiguous()
        lead_shape = x_last.shape[:-1]

        # `F.pad(..., mode="replicate")` on a 4D input pads the LAST TWO dims only
        # (a torch constraint, not a choice here) -- `x_last` is still 4D after
        # `movedim` (it only permutes axes), so pad with an explicit 4-tuple and
        # leave the second-to-last dim's pair at 0.
        xp = pad(x_last, (r, r, 0, 0), mode="replicate")      # length N + 2r
        padded_len = N + 2 * r
        nblocks = -(-padded_len // w)                         # ceil div
        tail = nblocks * w - padded_len
        if tail:
            xp = pad(xp, (0, tail, 0, 0), mode="replicate")

        blocks = xp.reshape(*lead_shape, nblocks, w)
        if grow:
            g = torch.cummax(blocks, dim=-1).values
            h = torch.cummax(blocks.flip(-1), dim=-1).values.flip(-1)
        else:
            g = torch.cummin(blocks, dim=-1).values
            h = torch.cummin(blocks.flip(-1), dim=-1).values.flip(-1)

        g_flat = g.reshape(*lead_shape, nblocks * w)
        h_flat = h.reshape(*lead_shape, nblocks * w)
        h_slice = h_flat[..., :N]
        g_slice = g_flat[..., w - 1:w - 1 + N]
        out = torch.maximum(h_slice, g_slice) if grow else torch.minimum(h_slice, g_slice)
        return out.movedim(-1, dim)

    @stdlib("erode", sig='erode(img, radius) \\u2192 vec', category='Sampling', sync=True, footprint=('halo_arg', 1), pixel_args=(1,), doc='Morphological erosion (local min over a (2r+1)² square). Shrinks bright regions.', ex='@OUT = erode(@mask, 3);')
    @staticmethod
    def fn_erode(image, radius) -> torch.Tensor:
        """Grayscale erosion (local min over a (2r+1)² square). Shrinks bright
        regions; the classic mask-shrink op."""
        return TEXStdlib._morph(image, radius, grow=False)

    @stdlib("dilate", sig='dilate(img, radius) \\u2192 vec', category='Sampling', sync=True, footprint=('halo_arg', 1), pixel_args=(1,), doc='Morphological dilation (local max). Grows bright regions.', ex='@OUT = dilate(@mask, 3);')
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
        # FIX-PACE P4: an entry poll, unconditionally -- `_build_mip_pyramid`'s own
        # internal per-level poll (PACE-47c) sits INSIDE its build loop, which a WARM
        # cache hit returns before ever reaching (a live "drag a param on a static image"
        # session hits the warm cache every tick). Without this, a warm sample_mip call
        # touches the pacing pool not at all, even though this builtin's own registry
        # entry stays deliberately excluded from footprint-derived heaviness (it is
        # multi-pass, not halo-shaped) — see `pacing_heavy.py`'s docstring. `heavy=True`:
        # this call is device-expensive on both the cold AND warm path (a grid_sample per
        # mip level either way), so it must bypass stride economization like the other
        # Gap-1 builtins, not just get a plain poll.
        poll_cook_cancel(heavy=True)
        return _sample_mip_trilinear(image, u_coord, v_coord, lod, _get_mip_pyramid)

    # A1 (v0.50 Phase C): the footprint's 4th element, `GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA`,
    # tells `tex_roi._reach_of` the exact point past which this builtin stops being a
    # narrowable-halo op and switches to the downscale-pyramid approximation
    # (`_gauss_blur_pyramid_approx`) — whose resample grid is anchored to a crop's own
    # edges, not the frame's absolute coordinates, so a narrowed (non-saturating) window
    # would otherwise silently diverge from a whole-frame cook (B1/B2's finding). Past the
    # threshold, `_reach_of` answers 'unbounded' — the same decline a symbolic sigma
    # already gets — so the planner (ROI, tiling/OOM strips, `cook_stage_dag`) falls back
    # to a whole-frame cook instead of narrowing onto a wrong phase.
    # A7 (v0.50 Phase C, B4#4): the doc= string below now discloses the pyramid
    # approximation past `GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA` -- this is the single
    # source for Function-Reference.md, tex_help.json, and js/tex_extension.js's
    # TEX_HELP_DATA (all three regenerated from it), so a TEX author reading in-editor
    # help or the generated reference can now learn this the same way bilateral_
    # filter's own doc= already discloses its detail-transfer approximation.
    @stdlib("gauss_blur", sig='gauss_blur(img, sigma) \\u2192 vec', category='Sampling', spatial=True, sync=True, footprint=('halo_arg', 1, 3.0, GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA), pixel_args=(1,), doc='Separable Gaussian blur. Kernel radius ≈ 3×sigma pixels. Replicate border padding. Exact within a measured sigma; a bounded-cost downscale approximation runs past it.', ex='@OUT = gauss_blur(@A, 2.0);')
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
        _require_finite_arg("gauss_blur", "sigma", sigma_val)  # A7: friendly diagnostic, not a raw int() crash
        sigma_val = max(sigma_val, 0.0)
        if sigma_val < 0.3 or img.dim() < 4:
            return img
        bchw = _get_bchw(img)
        # GAUSSPYR-50: `_gauss_blur_auto` dispatches on sigma alone (an engine policy,
        # not a new argument) — exact and bit-identical at/below
        # GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA, an O(image size) downscale-pyramid
        # approximation above it. See stdlib_core.py's own comment on the constant.
        result = _gauss_blur_auto(bchw, sigma_val)
        return result.permute(0, 2, 3, 1)

    # BILAT-50: the old `min(ceil(3*ss), 3)` silently clamped every spatial_sigma past
    # ~1.0 to whatever a 7x7 window gives -- the same silent-wrong class the erode/dilate
    # 256 clamp was. The window now genuinely grows with `ss` (no clamp): exact wherever
    # memory-feasible (today's own 7x7 math is untouched below, and `_bilateral_exact_bchw`
    # extends the SAME exact math, row-tiled to stay memory-bounded, up to
    # `_BILATERAL_EXACT_RADIUS_MAX`). Past that, BILAT-51 measured `_bilateral_detail_
    # transfer_bchw` (BILAT-50's own downscale+residual stand-in) against the TRUE exact
    # filter at the SAME spatial_sigma (not a fixed boundary anchor) and found it scores
    # BELOW the disclosed band on realistic content (SSIMULACRA2 as low as -11.6 at
    # ss=32) -- worse, once measured correctly, than even the v0.49 fixed-7x7 clamp it
    # replaced (18.5-34.6 over the same range). A separable (per-axis) bilateral pass
    # (`_bilateral_separable_bchw`) measured 87.8-93.3 across every corpus at ss=8.5/16/32
    # -- it wins outright, so it is now the past-threshold default up to
    # `_BILATERAL_SEPARABLE_RADIUS_MAX`. It is NOT O(image size) like detail-transfer,
    # though -- its cost is O(image size * radius), the same class D1 rejected for
    # erode/dilate's "just raise the cap" option -- so detail-transfer stays as the
    # ultimate fallback past `_BILATERAL_SEPARABLE_RADIUS_MAX`, keeping the "no call
    # ever costs more than O(image size) at truly extreme spatial_sigma" guarantee this
    # codebase holds for every other builtin's own past-threshold tier.
    _BILATERAL_EXACT_RADIUS_MAX = 24  # ksize=49 (spatial_sigma up to ~8.0). A prior design
    # pass measured the UNTILED exact filter needing ~60GB at this exact radius on
    # a canvas far smaller than 1080p (an O(r^2) memory blow-up, not a resolution-specific
    # fluke) -- row-tiling keeps the identical math memory-bounded at any resolution up to
    # here; past it the O(r^2) per-pixel tap count makes even a tiled exact pass too slow
    # for a builtin on the default path, and the detail-transfer path (O(image size),
    # independent of sigma) takes over.
    # A1 (v0.50 Phase C): the `spatial_sigma` value past which `fn_bilateral_filter`
    # itself switches to the detail-transfer downscale approximation (`radius =
    # ceil(3*ss) > _BILATERAL_EXACT_RADIUS_MAX` <=> `ss > _BILATERAL_EXACT_RADIUS_MAX /
    # 3.0`) -- named here so the footprint declaration below and the dispatch in
    # `fn_bilateral_filter` read the same boundary from one place, not two matching
    # literals. See the footprint's own comment for why this matters to the planner.
    _BILATERAL_APPROX_THRESHOLD_SS = _BILATERAL_EXACT_RADIUS_MAX / 3.0
    # BILAT-51 round 3: bounded by MEASURED evidence on BOTH axes, not cost alone.
    # This ask's own SSIMULACRA2 sweep (docs/resolution-scale.md) only measured separable
    # beating both the v0.49 clamp and BILAT-50's detail-transfer through spatial_sigma=32
    # (radius=96) -- the first-shipped ceiling of 256 extended the regime past that into a
    # radius this ask never scored for accuracy. Round 3's own timing sweep then found the
    # cost at radius=256 reaches ~106s on a 4K CPU cook (this box, "best of 5") -- both the
    # untested-quality concern and the measured worst-case cost point the same way: down, to
    # the last radius this ask actually measured. At radius=96 the worst observed cost is a
    # few tens of seconds (4K CPU) / a few seconds (4K CUDA, RTX 2080 SUPER) -- see
    # docs/resolution-scale.md's own time/memory table for the full sweep. A genuinely huge
    # `spatial_sigma` past this falls back to `_bilateral_detail_transfer_bchw` (flat
    # O(image size), independent of sigma) rather than pay an ever-growing per-call cost,
    # matching the guarantee every OTHER builtin's own past-threshold tier already holds
    # (D1's van Herk, D2's gauss pyramid): no call ever costs more than O(image size) once
    # sigma is large enough to be past the tier that scales with it.
    _BILATERAL_SEPARABLE_RADIUS_MAX = 96
    _BILATERAL_TILE_BUDGET_ELEMS = 8_000_000  # ~32MB per fp32 intermediate tensor,
    # independent of image resolution: the row-tile height shrinks as `ksize` grows so
    # every per-tile intermediate (`patches`/`diff`/`w`, each [B,C,tile_h,W,ksize,ksize])
    # stays under this many elements regardless of the image's true H.

    @staticmethod
    def _bilateral_spatial_weights(ss, radius, device):
        """The precomputed spatial-Gaussian weight tensor [1,1,1,1,ksize,ksize] shared by
        every exact bilateral pass (tiled or not) at this (ss, radius)."""
        ksize = 2 * radius + 1
        inv_2ss = -0.5 / max(ss * ss, 1e-10)
        dy = torch.arange(ksize, device=device, dtype=torch.float32) - radius
        dx = dy.clone()
        d2 = dy.view(-1, 1) ** 2 + dx.view(1, -1) ** 2  # [kH, kW]
        return torch.exp(d2 * inv_2ss).view(1, 1, 1, 1, ksize, ksize), ksize

    @staticmethod
    def _sum_kernel_taps_fixed_order(t):
        """FIX-501 F3: sum a tensor's trailing (kH, kW) axes via a FIXED, sequential
        row-major accumulation -- one elementwise add per tap -- instead of
        `torch.sum(dim=(-2,-1))`. `torch.sum` over these axes is free to pick a
        different CUDA reduction algorithm depending on the tensor's OVERALL shape
        (the leading batch/spatial dims), even though every output element's own
        kH*kW taps are unchanged -- this is exactly what made a windowed cook and the
        matching crop of a whole-frame cook of `bilateral_filter`'s exact tier diverge
        by a few ULPs on CUDA (an embedding host's own finding): `_bilateral_exact_bchw`'s row
        tiling picks its tile height from the input's OWN width, so a narrower window
        and the wider whole frame hand this reduction differently-shaped tensors for
        the SAME output pixels. Elementwise add has no such size dependence -- every
        tap is combined in the same order regardless of what surrounds it."""
        kH, kW = t.shape[-2], t.shape[-1]
        acc = t[..., 0, 0]
        for j in range(1, kW):
            acc = acc + t[..., 0, j]
        for i in range(1, kH):
            for j in range(kW):
                acc = acc + t[..., i, j]
        return acc

    @staticmethod
    def _bilateral_weighted_avg(patches, center, w_spatial, sr, deterministic=False):
        """The exact bilateral core (today's own weighted-average formula, shared by the
        tiled exact pass and the detail-transfer path's reduced-scale call): `patches`
        [B,C,h,w,kH,kW] already unfolded, `center` [B,C,h,w,1,1] the un-unfolded pixel.
        `deterministic` (FIX-501 F3, default off): route the two kernel-window sums
        through `_sum_kernel_taps_fixed_order` instead of `torch.sum(dim=(-2,-1))`, so
        the result cannot depend on the surrounding tile's shape. Off by default so
        `radius<=3`'s existing bit-identity pin (`test_bilat50_radius.py`) and its
        untiled default-path perf stay untouched; `_bilateral_exact_bchw` turns it on
        only for `radius>3`, the row-tiled regime the bug lives in."""
        diff = patches - center
        inv_2sr = -0.5 / max(sr * sr, 1e-10)
        cd2 = (diff * diff).sum(dim=1, keepdim=True)
        w_range = torch.exp(cd2 * inv_2sr)
        w = w_spatial * w_range
        if deterministic:
            numerator = TEXStdlib._sum_kernel_taps_fixed_order(patches * w)
            denominator = TEXStdlib._sum_kernel_taps_fixed_order(w)
        else:
            numerator = (patches * w).sum(dim=(-2, -1))
            denominator = w.sum(dim=(-2, -1))
        return numerator / denominator.clamp(min=1e-10)

    @staticmethod
    def _bilateral_exact_bchw(bchw, ss, sr, radius):
        """The exact bilateral filter at any radius, row-tiled to keep peak memory
        bounded (`_BILATERAL_TILE_BUDGET_ELEMS`) independent of image resolution. Same
        math as today's original 7x7 pass (`_bilateral_weighted_avg`) -- tiling changes
        nothing about any one output pixel's own computation, since each pixel's window
        is entirely contained within its own row tile (the tile is padded by `radius` on
        each side before unfolding)."""
        B, C, H, W = bchw.shape
        w_spatial, ksize = TEXStdlib._bilateral_spatial_weights(ss, radius, bchw.device)
        padded = torch.nn.functional.pad(bchw, (radius, radius, radius, radius), mode='replicate')
        tile_h = max(1, TEXStdlib._BILATERAL_TILE_BUDGET_ELEMS // max(1, W * ksize * ksize))
        if tile_h >= H:
            tile_h = H
        # FIX-501 F3: `tile_h` above is derived from THIS call's own `W` -- a windowed
        # crop and the whole frame it is drawn from generally have different widths, so
        # they tile at different heights for the SAME output pixels. Past radius 3 (the
        # only regime this row-tiling loop actually exercises with more than one
        # possible tile shape), route the kernel-window sum through the shape-
        # independent fixed-order accumulator instead of `torch.sum`, so the windowed
        # and whole-frame cooks agree on CUDA regardless of how each one tiled.
        # `radius<=3` keeps `deterministic=False` -- untouched, still bit-identical to
        # v0.50.0 (test_bilat50_radius.py).
        deterministic = radius > 3
        outputs = []
        for y0 in range(0, H, tile_h):
            # CANCEL-44/PACE-47c idiom: a poll between tiles -- the multi-pass boundary
            # this loop introduces once a large radius tiles the work (bilateral_filter's
            # own entry poll, below, no longer covers every pass by itself).
            poll_cook_cancel(heavy=True)
            y1 = min(y0 + tile_h, H)
            padded_rows = padded[:, :, y0:y1 + 2 * radius, :]
            center_rows = bchw[:, :, y0:y1, :]
            patches = padded_rows.unfold(2, ksize, 1).unfold(3, ksize, 1)
            center = center_rows.unsqueeze(-1).unsqueeze(-1)
            outputs.append(TEXStdlib._bilateral_weighted_avg(
                patches, center, w_spatial, sr, deterministic=deterministic))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)

    @staticmethod
    def _bilateral_detail_transfer_bchw(bchw, ss, sr):
        """BILAT-50: past `_BILATERAL_EXACT_RADIUS_MAX`, downscale until
        the REDUCED spatial_sigma lands back inside today's own exact 7x7 window (<=1.0),
        run the exact filter there (always radius<=3 by construction -- no tiling needed),
        upsample the filtered result back to full resolution, and add back the full-
        resolution high-frequency detail the downscale discarded (`detail = original -
        upsample(downsample(original))`) -- a bounded-cost stand-in for a true joint
        bilateral upsample. Cost is O(image size), independent of spatial_sigma (measured
        flat, 1.0-15ms, across five orders of magnitude of sigma).
        """
        B, C, H, W = bchw.shape
        factor = 1
        while ss / factor > 1.0:
            factor *= 2
        Hr, Wr = max(1, round(H / factor)), max(1, round(W / factor))
        poll_cook_cancel(heavy=True)
        downsampled = torch.nn.functional.interpolate(bchw, size=(Hr, Wr), mode='area')
        reduced_ss = ss / factor
        reduced_radius = int(math.ceil(3.0 * reduced_ss))  # always <=3 by construction
        w_spatial, ksize = TEXStdlib._bilateral_spatial_weights(reduced_ss, reduced_radius, bchw.device)
        padded = torch.nn.functional.pad(
            downsampled, (reduced_radius, reduced_radius, reduced_radius, reduced_radius),
            mode='replicate')
        patches = padded.unfold(2, ksize, 1).unfold(3, ksize, 1)
        center = downsampled.unsqueeze(-1).unsqueeze(-1)
        filtered_reduced = TEXStdlib._bilateral_weighted_avg(patches, center, w_spatial, sr)
        poll_cook_cancel(heavy=True)
        upsampled_filtered = torch.nn.functional.interpolate(
            filtered_reduced, size=(H, W), mode='bilinear', align_corners=False)
        upsampled_plain = torch.nn.functional.interpolate(
            downsampled, size=(H, W), mode='bilinear', align_corners=False)
        detail = bchw - upsampled_plain
        return upsampled_filtered + detail

    @staticmethod
    def _bilateral_separable_1d_pass(bchw, dim, radius, ss, sr, range_ref=None):
        """One 1-D bilateral pass along `dim` (2=H, 3=W): each of the
        `2*radius+1` taps along that single axis is weighted by the SAME
        spatial-Gaussian x range-Gaussian formula the exact 2-D filter uses
        (`_bilateral_weighted_avg`'s own math, taken one axis at a time), with
        replicate-boundary handling via a clamped `index_select` (an O(image
        size) tensor per tap, never an O(radius) x O(radius) unfolded patch,
        so memory stays flat regardless of radius -- unlike the exact tier,
        this never needs row-tiling). BILAT-51: measured to score far higher
        (SSIMULACRA2) on realistic content than the detail-transfer residual
        it replaces past the exact tier's own ceiling, at the cost of growing
        with `radius` rather than staying flat (see `_BILATERAL_SEPARABLE_
        RADIUS_MAX`'s own comment).

        BILAT8-51: `range_ref` (default `bchw`, i.e. unchanged behaviour) is the
        tensor the RANGE weight's difference is measured against -- the tap is
        still read from `bchw` (the pass's own input; this is what gets
        averaged), only the similarity test can be redirected. The composing
        pass (`_bilateral_separable_bchw`) points this at the ORIGINAL image
        for both the row and the column pass, fixing the diagnosed defect: the
        column pass's range weight used to compare the row-passed (already
        smoothed) values against each other, which is a weaker edge test than
        the true 2-D filter's (both axes' difference measured against the same
        real pixel) and is what let the two-pass filter bleed across a hard
        edge into a many-code spike at 1080p (day-plate corners)."""
        device = bchw.device
        if range_ref is None:
            range_ref = bchw
        n = bchw.shape[dim]
        d = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        w_spatial_1d = torch.exp(-0.5 * (d * d) / max(ss * ss, 1e-10))
        inv_2sr = -0.5 / max(sr * sr, 1e-10)
        idx_base = torch.arange(n, device=device)
        acc = torch.zeros_like(bchw)
        wsum_shape = list(bchw.shape)
        wsum_shape[1] = 1
        wsum = torch.zeros(wsum_shape, device=device, dtype=bchw.dtype)
        for i, off in enumerate(range(-radius, radius + 1)):
            idx = (idx_base + off).clamp(0, n - 1)
            tap = torch.index_select(bchw, dim, idx)
            tap_ref = torch.index_select(range_ref, dim, idx)
            diff = range_ref - tap_ref
            cd2 = (diff * diff).sum(dim=1, keepdim=True)
            w_range = torch.exp(cd2 * inv_2sr)
            w = w_spatial_1d[i] * w_range
            acc = acc + tap * w
            wsum = wsum + w
        return acc / wsum.clamp(min=1e-10)

    @staticmethod
    def _bilateral_separable_bchw(bchw, ss, sr, radius):
        """BILAT-51/BILAT8-51: a row pass then a column pass of the 1-D
        bilateral core (`_bilateral_separable_1d_pass`) -- a well-known
        cheaper-than-exact approximation (still not a true 2-D bilateral
        filter: a genuine corner where two differently-coloured hard edges
        cross is invisible to any single row-then-column or column-then-row
        decomposition, since neither 1-D pass ever tests a diagonal
        neighbour's colour). BILAT8-51 measured the diagnosed defect (the
        second pass's range weight compared against the first pass's OWN
        smoothed output) against several candidate fixes -- pointing BOTH
        passes' range weight at the original image, symmetrizing row/col
        order, and a 4-direction (row+col+2 diagonals) variant -- on a
        purpose-built hard-edge plate at spatial_sigma 8.5-10 (the host's own
        priority range) and range_sigma 0.05/0.2/0.5/1.0. Only pointing the
        range weight at the original image never regressed any measured cell
        (fewer >=1-code pixels at every combination, same or lower max code);
        the order-symmetrized and 4-direction variants scored WORSE at
        range_sigma>=0.5 near a hard bright edge, so they were dropped.
        Adopted here. This does NOT clear the <=1-code bar near a hard bright
        edge/corner at range_sigma>=0.5 -- max code there is unchanged (up to
        ~20-30 at 1080p) because the residual is the corner effect above, a
        structural limit of any O(image size * radius) separable/directional-
        sum scheme, not a pass-order or range-reference choice. See
        `docs/resolution-scale.md`'s bilateral section for the frontier."""
        poll_cook_cancel(heavy=True)
        row_passed = TEXStdlib._bilateral_separable_1d_pass(bchw, 3, radius, ss, sr, range_ref=bchw)
        poll_cook_cancel(heavy=True)
        return TEXStdlib._bilateral_separable_1d_pass(row_passed, 2, radius, ss, sr, range_ref=bchw)

    # A1 (v0.50 Phase C): the footprint's 4th element, `_BILATERAL_APPROX_THRESHOLD_SS`,
    # tells `tex_roi._reach_of` the exact `spatial_sigma` past which this builtin
    # switches to the detail-transfer downscale approximation
    # (`_bilateral_detail_transfer_bchw`) -- whose resample grid is anchored to a crop's
    # own edges, not the frame's absolute coordinates, so a narrowed (non-saturating)
    # window would otherwise silently diverge from a whole-frame cook (B1/B2's finding,
    # the larger of the two: up to 0.0265 maxdiff on a realistic image). Past the
    # threshold, `_reach_of` answers 'unbounded' -- the same decline a symbolic
    # spatial_sigma already gets -- so the planner falls back to a whole-frame cook.
    # A4 (v0.50 Phase C, R3#5): the reach multiplier is 3.0, matching the exact/tiled-
    # exact tiers' own true reach (`radius = ceil(3*ss)`) -- the mult used to be 8.0,
    # picked to conservatively cover the detail-transfer tier's chain reach too from one
    # static number, but A1's `approx_above` decline now handles that tier by refusing to
    # narrow at all, so this mult only ever needs to describe the (unchanged) exact
    # tiers' real reach. The old 8.0 over-padded every ROI-narrowed or tiled cook below
    # the threshold by ~2.67x more halo than the math needs (R3#5's own measurement).
    @stdlib("bilateral_filter", sig='bilateral_filter(img, spatial_sigma, range_sigma) \\u2192 vec', category='Sampling', spatial=True, sync=True, footprint=('halo_arg', 1, 3.0, _BILATERAL_APPROX_THRESHOLD_SS), pixel_args=(1,), doc='Edge-preserving smoothing: blurs within regions but keeps edges. Exact within a measured window; a bounded-cost approximation runs past it.', ex='@OUT = bilateral_filter(@A, 1.5, 0.2);')
    @staticmethod
    def fn_bilateral_filter(image, sigma_s, sigma_r) -> torch.Tensor:
        """Edge-preserving bilateral filter using Tensor.unfold.

        Weights each neighbor by spatial Gaussian x range (color similarity)
        Gaussian. Radius is derived from sigma_s (3x sigma) -- exact (row-tiled,
        memory-bounded) up to `_BILATERAL_EXACT_RADIUS_MAX`, then a downscale +
        detail-transfer approximation (BILAT-50) past it. Best for small kernels
        (3x3); for larger kernels, the loop-based approach in bilateral_approx.tex
        may be faster due to memory traffic.

        Args:
            image: [B, H, W, C] tensor
            sigma_s: float -- spatial sigma in pixels
            sigma_r: float -- range sigma (color similarity, 0.01-0.5 typical)
        """
        # PACE-47d (Gap 1): an entry poll -- the closest equivalent to a between-pass
        # record for whichever regime below actually runs (each regime below now also
        # polls its OWN internal pass boundaries once there is more than one pass).
        poll_cook_cancel(heavy=True)
        img = image if image.__class__ is torch.Tensor else _to_tensor(image)
        # Both sigmas size the window / the weights host-side; PERF-2 resolves them
        # from the minted host value where there is one, and reads back where there
        # is not (see `_host_scalar`).
        #
        # TRK-69: a BARE Python float (never a shipped tier's own path — the
        # interpreter and codegen both mint every scalar into a tensor first, so this
        # branch is a direct-caller-only corner) used to keep the raw double here
        # while `gauss_blur`'s equivalent fallback fp32-rounds through a minted
        # tensor's `.item()`. fp32-rounding here too — via the same `_dtype_rounded`
        # the mint sites use to compute a tag — makes the two agree on what a number
        # means without minting a tensor just to round one. No default-path pixel
        # moves: every value either tier ever hands this builtin already carries a
        # host reading or is a real tensor, so `ss`/`sr` are unchanged on both.
        ss = _host_scalar(sigma_s)
        if ss is None:
            if torch.is_tensor(sigma_s):
                ss = sigma_s.item()
            else:
                raw = float(sigma_s)
                rounded = _dtype_rounded(raw, torch.float32)
                ss = raw if rounded is None else rounded
        sr = _host_scalar(sigma_r)
        if sr is None:
            if torch.is_tensor(sigma_r):
                sr = sigma_r.item()
            else:
                raw = float(sigma_r)
                rounded = _dtype_rounded(raw, torch.float32)
                sr = raw if rounded is None else rounded

        _require_finite_arg("bilateral_filter", "spatial_sigma", ss)  # A7: friendly diagnostic
        if img.dim() < 4 or ss < 0.3:
            return img

        B, H, W, C = img.shape
        radius = int(math.ceil(3.0 * ss))  # BILAT-50: no clamp -- the true window
        bchw = _get_bchw(img)

        # A5 (v0.50 Phase C, R2#1): `_bilateral_exact_bchw`'s row-tiling degenerates to a
        # single untiled pass whenever `tile_h >= H`, which is always true at a small
        # `ksize` (small radius) -- so it is correct, and bit-identical (proven,
        # `tests/test_bilat50_radius.py::test_bilat50_a5_exact_bchw_matches_inline_and_
        # degenerates_to_one_tile`), for `radius<=3` too.
        # BILAT-51: three regimes now -- exact (unchanged, bit-identical for
        # radius<=_BILATERAL_EXACT_RADIUS_MAX), separable (measured to beat both the
        # v0.49 clamp and BILAT-50's detail-transfer on realistic content, up to
        # _BILATERAL_SEPARABLE_RADIUS_MAX), then detail-transfer again as the flat-cost
        # fallback for spatial_sigma large enough that even the separable pass's O(radius)
        # cost would be excessive.
        if radius <= TEXStdlib._BILATERAL_EXACT_RADIUS_MAX:
            result = TEXStdlib._bilateral_exact_bchw(bchw, ss, sr, radius)
        elif radius <= TEXStdlib._BILATERAL_SEPARABLE_RADIUS_MAX:
            result = TEXStdlib._bilateral_separable_bchw(bchw, ss, sr, radius)
        else:
            result = TEXStdlib._bilateral_detail_transfer_bchw(bchw, ss, sr)

        return result.permute(0, 2, 3, 1)  # back to BHWC

    # ASK-1: native convolution. `kernel` is a second IMAGE/MASK BINDING, read whole —
    # not an ARRAY literal (an array is expanded to one full frame per tap by the
    # interpreter, `interpreter.py:1611-1617`) and not a mat3/mat4 (capped at 4x4,
    # `DEVELOPMENT.md:165`). footprint='image' (arg 0, the image itself), because
    # `('halo_arg', kernel)` cannot be built here: `tex_roi._call_reach` resolves a
    # halo_arg only from a folded NumberLiteral, so a kernel BINDING can only ever resolve
    # 'unbounded' — never a narrowable radius. REACH-48 (TIERS-48-design.md SS B.2 point 2)
    # closes the OTHER half of the DEVELOPMENT.md rejected-decision entry this cites: the
    # kernel argument is now given its OWN declared reach, `arg_footprint=((1, 'image'),)`
    # — "this argument is read whole", the exact vocabulary widening that entry named as
    # its reopening condition — so `tex_roi`'s per-binding footprint (and any future
    # per-argument consumer, e.g. a join-shaped `chain_windows` walk) reports @kernel as
    # 'image' instead of silently defaulting it to the outer/pointwise context.
    @stdlib("convolve", sig='convolve(img, kernel[, normalize]) \\u2192 vec', category='Sampling',
            spatial=True, sync=True, footprint='image', arg_footprint=((1, 'image'),),
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
        # FIX-PACE P4: same warm-cache entry poll as fn_sample_mip above -- see its comment.
        poll_cook_cancel(heavy=True)
        return _sample_mip_trilinear(image, u_coord, v_coord, lod, _get_mip_pyramid_gauss)
