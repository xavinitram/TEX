"""BILATX-51 -- the exact bilateral tier for `3 < radius <= _BILATERAL_EXACT_RADIUS_MAX`
now runs `_bilateral_exact_taploop_bchw` (one elementwise pass per tap) instead of
`_bilateral_exact_bchw`'s row-tiled `unfold` -- same math, far lower peak memory and wall
time at radius past the old ceiling (author's own measurement: r30 196s -> 2.9s on a
1920x1080 CUDA frame). The ceiling itself moved 24 -> 40 (`_BILATERAL_EXACT_RADIUS_MAX =
40`), and the footprint's `approx_above` threshold (`_BILATERAL_APPROX_THRESHOLD_SS`,
derived from the same constant) moves with it, so a windowed/tiled cook keeps narrowing
through the new ceiling and declines past it, same as before.

`radius<=3` is untouched (`_bilateral_exact_bchw`'s own untiled degeneracy, A5's proof,
still pinned by `test_bilat50_radius.py`). `_bilateral_exact_bchw` itself is UNCHANGED
math-wise and keeps a caller (radius<=3), so it stays in the product; only its channel-
count tile-budget fix (BILAT8-51's own finding: the per-tile budget must divide by B*C,
not just W*ksize^2) was carried over, since a future caller could still exercise its
tiling loop at a larger radius.
"""
from __future__ import annotations

import math

import torch

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib
from TEX_Wrangle.tex_runtime.stdlib_core import _get_bchw, _dtype_rounded
from TEX_Wrangle.tex_runtime import stdlib_registry as R
from TEX_Wrangle import tex_roi as _R
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_compiler.ast_nodes import NumberLiteral

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]


def _rounded_ss(ss):
    """`fn_bilateral_filter`'s own TRK-69 fp32-rounding of a bare Python float
    `spatial_sigma` (see its docstring/comment) -- this test's reference calls must use
    the SAME rounded value `fn_bilateral_filter` will actually compute with, or a value
    like 40.0/3.0 (not exactly representable in float32) silently compares two DIFFERENT
    sigmas instead of the same one through two code paths."""
    rounded = _dtype_rounded(float(ss), torch.float32)
    return ss if rounded is None else rounded


def _radius_for(ss):
    """The radius `fn_bilateral_filter` will actually dispatch on for a bare Python
    float `ss` -- `ceil(3*rounded_ss)`, NOT `ceil(3*raw_ss)`: fp32-rounding a value like
    4.0/3.0 can push it just over an integer boundary (1.3333333333... rounds to
    1.3333333730697632 in float32, whose 3x is just past 4.0, giving radius=5, not 4)."""
    return int(math.ceil(3.0 * _rounded_ss(ss)))


# ── 1. Regime-dispatch boundaries: 3, 40, 96 ─────────────────────────────────────────────

def test_bilatx51_exact_radius_max_is_40(r: SubTestResult):
    print("\n--- BILATX-51: _BILATERAL_EXACT_RADIUS_MAX moved 24 -> 40 (author's decision, "
          "2026-09-28) ---")
    if TEXStdlib._BILATERAL_EXACT_RADIUS_MAX != 40:
        r.fail("exact ceiling", f"expected 40, got {TEXStdlib._BILATERAL_EXACT_RADIUS_MAX}")
        return
    r.ok("_BILATERAL_EXACT_RADIUS_MAX == 40")


def test_bilatx51_dispatch_boundary_radius_3_unchanged(r: SubTestResult):
    print("\n--- BILATX-51: radius<=3 still dispatches to the OLD untiled "
          "_bilateral_exact_bchw path (bit-identical, untouched) ---")
    img = make_img(1, 20, 20, 3, seed=511)
    bchw = _get_bchw(img)
    ss, sr = 1.0, 0.2  # radius = 3
    via_fn = TEXStdlib.fn_bilateral_filter(img.clone(), ss, sr)
    direct = TEXStdlib._bilateral_exact_bchw(bchw.clone(), ss, sr, 3).permute(0, 2, 3, 1)
    if not torch.equal(via_fn, direct):
        r.fail("radius=3 dispatch", "fn_bilateral_filter(radius=3) is not torch.equal to "
               "_bilateral_exact_bchw -- BILATX-51 must not touch this boundary")
        return
    r.ok("radius=3 still routes through _bilateral_exact_bchw, torch.equal")


def test_bilatx51_dispatch_boundary_radius_4_is_taploop(r: SubTestResult):
    print("\n--- BILATX-51: radius=4 (just past 3) now dispatches to the NEW "
          "_bilateral_exact_taploop_bchw, not the old tiled path ---")
    img = make_img(1, 20, 20, 3, seed=512)
    bchw = _get_bchw(img)
    ss, sr = 1.3333332333333332, _rounded_ss(0.2)  # radius = 4 (NOT 4.0/3.0 -- see
    # _radius_for's own docstring: that fraction fp32-rounds just past the radius=4/5
    # boundary; this literal is chosen to land on radius=4 after the same rounding)
    via_fn = TEXStdlib.fn_bilateral_filter(img.clone(), ss, sr)
    taploop = TEXStdlib._bilateral_exact_taploop_bchw(bchw.clone(), ss, sr, 4).permute(0, 2, 3, 1)
    if not torch.equal(via_fn, taploop):
        r.fail("radius=4 dispatch", "fn_bilateral_filter(radius=4) is not torch.equal to "
               "_bilateral_exact_taploop_bchw")
        return
    r.ok("radius=4 routes through _bilateral_exact_taploop_bchw, torch.equal")


def test_bilatx51_dispatch_boundary_radius_40_is_taploop_41_is_separable(r: SubTestResult):
    print("\n--- BILATX-51: radius=40 (the new ceiling) is still tap-loop exact; radius=41 "
          "(just past it) falls to the separable tier ---")
    img = make_img(1, 16, 16, 3, seed=513)
    bchw = _get_bchw(img)
    sr = _rounded_ss(0.2)
    ss_40 = _rounded_ss(40.0 / 3.0)
    via_40 = TEXStdlib.fn_bilateral_filter(img.clone(), ss_40, sr)
    direct_40 = TEXStdlib._bilateral_exact_taploop_bchw(bchw.clone(), ss_40, sr, 40).permute(0, 2, 3, 1)
    if not torch.equal(via_40, direct_40):
        r.fail("radius=40 still taploop", "fn_bilateral_filter(radius=40) is not torch.equal "
               "to _bilateral_exact_taploop_bchw -- the ceiling moved")
        return
    ss_41 = _rounded_ss(13.666666466666667)  # radius=41 after fp32 rounding (41.0/3.0
    # itself rounds to radius=42 -- see radius=4's own note above); rounded ONCE here
    # and reused for both calls below, the same way ss_40 above is -- rounding an
    # already-rounded fp32 value is idempotent, so via_fn's OWN internal rounding then
    # matches this reference call exactly instead of rounding a second, different value.
    via_41 = TEXStdlib.fn_bilateral_filter(img.clone(), ss_41, sr)
    direct_41 = TEXStdlib._bilateral_separable_bchw(bchw.clone(), ss_41, sr, 41).permute(0, 2, 3, 1)
    if not torch.equal(via_41, direct_41):
        r.fail("radius=41 falls to separable", "fn_bilateral_filter(radius=41) is not "
               "torch.equal to _bilateral_separable_bchw -- the dispatch boundary is wrong")
        return
    r.ok("radius=40 -> tap-loop exact; radius=41 -> separable -- the new ceiling is exactly "
         "where the author's decision put it")


def test_bilatx51_dispatch_boundary_96_and_97_unchanged(r: SubTestResult):
    print("\n--- BILATX-51: the separable/detail-transfer crossover at radius 96/97 is "
          "UNCHANGED by this ask (only the exact ceiling moved) ---")
    img = make_img(1, 16, 16, 3, seed=514)
    bchw = _get_bchw(img)
    sr = _rounded_ss(0.2)
    ss_96 = _rounded_ss(96.0 / 3.0)
    via_96 = TEXStdlib.fn_bilateral_filter(img.clone(), ss_96, sr)
    direct_96 = TEXStdlib._bilateral_separable_bchw(bchw.clone(), ss_96, sr, 96).permute(0, 2, 3, 1)
    if not torch.equal(via_96, direct_96):
        r.fail("radius=96 still separable", "dispatch boundary at 96 moved unexpectedly")
        return
    ss_97 = _rounded_ss(97.0 / 3.0)
    via_97 = TEXStdlib.fn_bilateral_filter(img.clone(), ss_97, sr)
    direct_97 = TEXStdlib._bilateral_detail_transfer_bchw(bchw.clone(), ss_97, sr).permute(0, 2, 3, 1)
    if not torch.equal(via_97, direct_97):
        r.fail("radius=97 falls to detail-transfer", "dispatch boundary at 97 moved "
               "unexpectedly")
        return
    r.ok("radius=96/97 crossover unchanged (96 separable, 97 detail-transfer)")


# ── 2. Accuracy: r<=3 torch.equal; 3<r<=24 within 1e-5 of the OLD tiled exact (kept as a
#    test-local reference, never product code); 25<=r<=40 within 1e-5 of a brute reference ─

def _old_tiled_exact_reference(bchw, ss, sr, radius):
    """A byte-for-byte copy of the pre-BILATX-51 `_bilateral_exact_bchw` row-tiling loop,
    kept here ONLY as this test's own before/after reference -- never called by product
    code. Identical to the still-shipped `_bilateral_exact_bchw`, since BILATX-51 did not
    change its math, only which radii still call it; kept as a separate copy so this test
    does not silently start comparing a function against itself if a future change ever
    touches `_bilateral_exact_bchw`."""
    B, C, H, W = bchw.shape
    w_spatial, ksize = TEXStdlib._bilateral_spatial_weights(ss, radius, bchw.device)
    padded = torch.nn.functional.pad(bchw, (radius, radius, radius, radius), mode='replicate')
    tile_h = max(1, TEXStdlib._BILATERAL_TILE_BUDGET_ELEMS // max(1, B * C * W * ksize * ksize))
    if tile_h >= H:
        tile_h = H
    deterministic = radius > 3
    outputs = []
    for y0 in range(0, H, tile_h):
        y1 = min(y0 + tile_h, H)
        padded_rows = padded[:, :, y0:y1 + 2 * radius, :]
        center_rows = bchw[:, :, y0:y1, :]
        patches = padded_rows.unfold(2, ksize, 1).unfold(3, ksize, 1)
        center = center_rows.unsqueeze(-1).unsqueeze(-1)
        outputs.append(TEXStdlib._bilateral_weighted_avg(
            patches, center, w_spatial, sr, deterministic=deterministic))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)


def _brute_reference(bchw, ss, sr, radius):
    """An independent, deliberately naive (single untiled unfold, no fixed-order
    accumulator) exact bilateral reference, used ONLY for the r25-40 test row so that
    row does not simply compare the new tap-loop against the old tiled path's own
    reference copy above."""
    B, C, H, W = bchw.shape
    ksize = 2 * radius + 1
    padded = torch.nn.functional.pad(bchw, (radius, radius, radius, radius), mode='replicate')
    patches = padded.unfold(2, ksize, 1).unfold(3, ksize, 1)
    center = bchw.unsqueeze(-1).unsqueeze(-1)
    inv_2ss = -0.5 / max(ss * ss, 1e-10)
    dy = torch.arange(ksize, dtype=torch.float32, device=bchw.device) - radius
    dx = dy.clone()
    d2 = dy.view(-1, 1) ** 2 + dx.view(1, -1) ** 2
    w_spatial = torch.exp(d2 * inv_2ss).view(1, 1, 1, 1, ksize, ksize)
    diff = patches - center
    inv_2sr = -0.5 / max(sr * sr, 1e-10)
    cd2 = (diff * diff).sum(dim=1, keepdim=True)
    w_range = torch.exp(cd2 * inv_2sr)
    w = w_spatial * w_range
    numerator = (patches * w).sum(dim=(-2, -1))
    denominator = w.sum(dim=(-2, -1))
    return numerator / denominator.clamp(min=1e-10)


def test_bilatx51_radius_le_3_torch_equal_to_before(r: SubTestResult):
    print("\n--- BILATX-51 task b: radius<=3 stays torch.equal to before (untouched path) "
          "---")
    for dev in _DEVICES:
        img = make_img(1, 22, 22, 3, seed=520).to(dev)
        for ss in (0.3, 0.6, 1.0):
            for sr in (0.1, 0.3):
                got = TEXStdlib.fn_bilateral_filter(img.clone(), ss, sr)
                radius = int(math.ceil(3.0 * ss))
                want = _old_tiled_exact_reference(_get_bchw(img.clone()), ss, sr, radius).permute(0, 2, 3, 1)
                if not torch.equal(got, want):
                    r.fail(f"radius<=3 equality [{dev}] ss={ss} sr={sr}", "torch.equal False")
                    return
    r.ok(f"radius<=3 torch.equal to the pre-BILATX-51 reference on {_DEVICES}")


def test_bilatx51_radius_4_to_24_within_1e5_of_old_tiled_exact(r: SubTestResult):
    print("\n--- BILATX-51 task b: 3<radius<=24 within 1e-5 of the OLD tiled exact "
          "(test-local reference only) ---")
    band = 1e-5
    worst = 0.0
    for dev in _DEVICES:
        img = make_img(1, 40, 44, 3, seed=521).to(dev)
        bchw = _get_bchw(img)
        radius_to_ss = {4: 1.3333332333333332, 10: 10 / 3.0, 18: 18 / 3.0, 24: 24 / 3.0}
        for radius in (4, 10, 18, 24):
            ss = _rounded_ss(radius_to_ss[radius])
            sr = _rounded_ss(0.2)
            new = TEXStdlib.fn_bilateral_filter(img.clone(), ss, sr)
            old = _old_tiled_exact_reference(bchw.clone(), ss, sr, radius).permute(0, 2, 3, 1)
            md = (new.float() - old.float()).abs().max().item()
            worst = max(worst, md)
            if md > band:
                r.fail(f"[{dev}] radius={radius}", f"maxdiff {md:.3e} exceeds {band}")
                return
    r.ok(f"3<radius<=24: worst maxdiff {worst:.3e} vs the old tiled exact reference, within "
         f"{band}, on {_DEVICES}")


def test_bilatx51_radius_25_to_40_within_1e5_of_brute_reference(r: SubTestResult):
    print("\n--- BILATX-51 task b: 25<=radius<=40 (the newly-raised band) is exact within "
          "1e-5 of an independent brute reference on a small image ---")
    band = 1e-5
    worst = 0.0
    for dev in _DEVICES:
        img = make_img(1, 24, 28, 3, seed=522).to(dev)
        bchw = _get_bchw(img)
        for radius in (25, 30, 40):
            ss = _rounded_ss(radius / 3.0)
            sr = _rounded_ss(0.2)
            new = TEXStdlib.fn_bilateral_filter(img.clone(), ss, sr)
            brute = _brute_reference(bchw.clone(), ss, sr, radius).permute(0, 2, 3, 1)
            md = (new.float() - brute.float()).abs().max().item()
            worst = max(worst, md)
            if md > band:
                r.fail(f"[{dev}] radius={radius}", f"maxdiff {md:.3e} exceeds {band}")
                return
    r.ok(f"25<=radius<=40: worst maxdiff {worst:.3e} vs an independent brute reference, "
         f"within {band}, on {_DEVICES}")


# ── 3. Bounded peak memory (CUDA if available) ──────────────────────────────────────────

def test_bilatx51_taploop_peak_memory_is_image_sized_not_ksize_squared(r: SubTestResult):
    print("\n--- BILATX-51: the tap-loop's largest single allocation stays a small "
          "multiple of the IMAGE's own size, never growing with ksize^2 like the old "
          "tiled unfold would at this radius ---")
    img = make_img(1, 96, 96, 3, seed=523)
    bchw = _get_bchw(img)
    orig_pad = torch.nn.functional.pad
    peak = {"numel": 0}

    def _spy_pad(t, *a, **kw):
        out = orig_pad(t, *a, **kw)
        peak["numel"] = max(peak["numel"], out.numel())
        return out

    torch.nn.functional.pad = _spy_pad
    try:
        for radius in (10, 24, 40):
            ss = radius / 3.0
            peak["numel"] = 0
            TEXStdlib._bilateral_exact_taploop_bchw(bchw.clone(), ss, 0.2, radius)
            # The padded frame is the largest tensor this path ever allocates; it must
            # stay within a small, RADIUS-INDEPENDENT multiple of the image's own size
            # (the padding grows the frame by 2*radius per side, a modest constant
            # factor at these radii on a 96x96 image -- nothing like ksize^2).
            budget = 8 * img.numel()
            if peak["numel"] > budget:
                r.fail(f"peak memory radius={radius}",
                       f"largest padded allocation {peak['numel']} exceeds budget {budget}")
                return
        r.ok("largest single allocation (the replicate-padded frame) stays within a small, "
             "radius-independent multiple of the image's own size across radius=10/24/40")
    finally:
        torch.nn.functional.pad = orig_pad


def test_bilatx51_taploop_peak_cuda_memory_bounded(r: SubTestResult):
    if not _CUDA:
        r.skip("BILATX-51 CUDA peak memory", "no CUDA device present on this box")
        return
    print("\n--- BILATX-51: CUDA peak memory at radius=40 stays bounded (does not blow up "
          "like the old tiled unfold's O(image size * ksize^2) intermediate would) ---")
    device = torch.device("cuda")
    img = make_img(1, 128, 128, 3, seed=524).to(device)
    bchw = _get_bchw(img)
    torch.cuda.reset_peak_memory_stats(device)
    TEXStdlib._bilateral_exact_taploop_bchw(bchw.clone(), 40.0 / 3.0, 0.2, 40)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    # A generous budget: the old tiled path's own untiled equivalent at radius=40 on a
    # canvas this size would need [B,C,H,W,81,81] in fp32 -- 1*3*128*128*81*81*4 bytes
    # ~= 15.5 GB. The tap-loop's per-tap tensors are all [B,C,H,W]-shaped (a few MB
    # here); a few hundred MB of budget is generous headroom, not a tight pin.
    budget_mb = 512
    if peak_mb > budget_mb:
        r.fail("CUDA peak memory", f"{peak_mb:.1f} MiB exceeds the {budget_mb} MiB budget")
        return
    r.ok(f"CUDA peak memory at radius=40, 128x128: {peak_mb:.1f} MiB (budget {budget_mb} MiB)")


# ── 4. Footprint / approx_above threshold moved with the new ceiling ───────────────────

def test_bilatx51_approx_threshold_moved_to_40_over_3(r: SubTestResult):
    print("\n--- BILATX-51: _BILATERAL_APPROX_THRESHOLD_SS is derived from the new "
          "_BILATERAL_EXACT_RADIUS_MAX (40), not the old 24 ---")
    expected = 40.0 / 3.0
    got = TEXStdlib._BILATERAL_APPROX_THRESHOLD_SS
    if abs(got - expected) > 1e-9:
        r.fail("approx threshold", f"expected {expected}, got {got}")
        return
    r.ok(f"_BILATERAL_APPROX_THRESHOLD_SS == {got} (== _BILATERAL_EXACT_RADIUS_MAX / 3.0)")


def test_bilatx51_windowed_cook_narrows_through_radius_40_declines_past_it(r: SubTestResult):
    print("\n--- BILATX-51: a windowed/tiled cook keeps narrowing through the new ceiling "
          "(radius=40) and declines just past it (radius=41), matching the footprint's "
          "moved approx_above threshold ---")
    W, H = 60, 52
    roi = (12, 10, 20, 18, W, H)

    def _windowed_vs_whole(ss):
        torch.manual_seed(530)
        image = torch.rand(1, H, W, 3)
        code = f"@OUT = bilateral_filter(@A, {ss}, 0.2);"
        _R.clear_roi_memo()
        full = tex_engine.cook(code, {"A": image.clone()}, device_mode="cpu").outputs["OUT"]
        _R.clear_roi_memo()
        res = tex_engine.cook(code, {"A": image.clone()}, device_mode="cpu",
                               roi=roi, roi_exec=True)
        win = res.outputs["OUT"]
        x0, y0, w, h, _, _ = roi
        crop = full[:, y0:y0 + h, x0:x0 + w]
        return res.cooked_roi, win, crop

    try:
        cooked_roi, win, crop = _windowed_vs_whole(40.0 / 3.0)  # radius=40, still narrows
        if cooked_roi != roi:
            r.fail("radius=40 window", f"window declined unexpectedly: cooked_roi={cooked_roi}")
            return
        if not torch.equal(win, crop):
            md = (win.float() - crop.float()).abs().max().item()
            r.fail("radius=40 window identity", f"maxdiff {md:.4e}")
            return

        cooked_roi2, win2, crop2 = _windowed_vs_whole(41.0 / 3.0)  # radius=41, must decline
        if cooked_roi2 is not None:
            r.fail("radius=41 window", f"window narrowed unexpectedly past the new "
                   f"threshold: cooked_roi={cooked_roi2}")
            return
        full_shape = (1, H, W, 3)
        x0, y0, w, h, _, _ = roi
        if tuple(win2.shape) != full_shape or not torch.equal(
                win2[:, y0:y0 + h, x0:x0 + w], crop2):
            r.fail("radius=41 window decline", "served output does not match a whole-frame "
                   "decline")
            return
        r.ok("radius=40 (new ceiling) narrows and is torch.equal to the whole-frame crop; "
             "radius=41 (just past it) declines the window outright -- the approx_above "
             "threshold moved exactly with _BILATERAL_EXACT_RADIUS_MAX")
    finally:
        _R.clear_roi_memo()


def test_bilatx51_footprint_never_underpads_through_new_ceiling(r: SubTestResult):
    print("\n--- BILATX-51: the declared halo_arg reach never under-estimates the true "
          "(3*spatial_sigma) reach at any spatial_sigma up to and past the new ceiling ---")
    for ss in (0.5, 1.5, 6.0, 13.0, 13.34, 20.0):
        declared = _R._call_reach("bilateral_filter", [None, NumberLiteral(value=ss)])
        true_reach = int(math.ceil(3.0 * ss))
        if isinstance(declared, str):
            continue  # 'unbounded'/'image' -- always safe
        if declared < true_reach:
            r.fail(f"under-padded halo ss={ss}",
                   f"declared reach {declared} < true reach {true_reach}")
            return
    r.ok("declared halo_arg reach never under-estimates the true reach through and past "
         "the new ceiling")
