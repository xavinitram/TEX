"""GAUSS8-51 — closing the display-8 bar for `gauss_blur` past sigma 256.

THE BAR: the shipped pyramid approximation and the exact reference, both mapped through
the ACES RRT + sRGB ODT and rounded to 8 bits, must agree on every pixel within 1 code
(worst channel) -- see `docs/resolution-scale.md`'s "display-8 bar" section for the full
measured table. The first reading found up to +12 codes of mean SIGNED shift on a
scattered-highlight plate at sigma=1024, cap=8 — a bias, not noise.

THE MECHANISM (proved below, red-first). `_gauss_blur_pyramid_approx` downsamples with
one 2-D `area` reduction, then blurs the reduced image with `_gauss_blur_bchw`'s own
replicate padding — which repeats the REDUCED image's own edge pixel. That edge pixel
is a `factor`x`factor` 2-D block average taken INWARD from the border in both axes at
once, so it has already mixed interior content into what a correct replicate pad should
treat as a pure boundary constant. The exact convolution never does this: its pad always
repeats the single un-mixed border row/column. Once `sigma` is large relative to the
image (routine here — this path only runs past the threshold, and the kernel's 3-sigma
reach often exceeds the whole frame), the exact result is ITSELF dominated by that one
replicated row, so any mismatch in what gets replicated becomes a systematic, image-wide
bias — exactly the shape of the measured +12-code shift.

THE FIX (`_gauss_blur_bchw_edge_pad`, `tex_runtime/stdlib_core.py`): downsample the
border strip ONLY along the axis parallel to it (never mixing in the perpendicular,
into-the-image direction the 2-D `area` reduction does), and pad the residual blur with
THAT instead of the reduced image's own edge. `GAUSS_BLUR_PYRAMID_QUALITY_CAP` was also
raised 8.0 -> 96.0 (author-approved) — at cap=8.0 the reduced level shrinks fast enough
that even a correctly-seeded pad still lets a several-pixel-wide reduced image's border
dominate at high sigma; cap=32.0 still missed on a bright plate (night x16, sigma=260),
which `test_gauss8_bright_plate_holds_the_bar` pins; cap=64.0 still missed on a thin strip
(`test_gauss8_thin_strip_holds_the_bar`); cap=96.0 holds the bar across the
sigmas and exposures this file (and `docs/resolution-scale.md`'s table) cover.

The v0.51 pyramid review evaluated replacing this caller-supplied-edge-pad
mechanism with a single replicate-pad of the full-resolution image BEFORE the one
`area` reduction — NOT ADOPTED: it fails precisely where the reduced grid collapses to
one coarse pixel (a side <= `factor`), where there is no longer a distinct "border" to
seed a plain replicate pad with (24 of 240 sweep cells above the 1-code bar, worst 53,
vs. 2 for this shipped mechanism). See `_gauss_blur_bchw_edge_pad`'s own
docstring for the full evidence. `_replicate_pad_h_conv`/`_replicate_pad_v_conv` were
collapsed into one dim-parametrized `_replicate_pad_conv` helper instead (v0.51).

Fast rows only: 128x128 plates (vs. the 1080x1080 readings in
`docs/resolution-scale.md`'s own table), sigma scaled to stay in the same "well past
threshold" regime the full-size table exercises.
"""
from __future__ import annotations

import torch

from helpers import *
from helpers import load_display8_harness
from TEX_Wrangle.tex_runtime import stdlib_core as _sc
from TEX_Wrangle.tex_runtime.stdlib_core import (
    GAUSS_BLUR_PYRAMID_QUALITY_CAP as QUALITY_CAP,
    _gauss_blur_bchw,
    _gauss_blur_bchw_edge_pad,
    _gauss_blur_pyramid_approx,
    _get_gauss_kernels,
)

_d8 = load_display8_harness()


def _naive_pyramid_approx(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """The PRE-fix mechanism, reproduced inline so the bug can be proved red-first
    without reverting the shipped code: one 2-D `area` reduction, then the residual
    blur padded with the REDUCED image's own (block-mixed) edge — `_gauss_blur_bchw`
    unmodified, exactly what `_gauss_blur_pyramid_approx` called before this fix."""
    out_h, out_w = img.shape[-2], img.shape[-1]
    factor = 1
    while sigma / factor > QUALITY_CAP:
        factor *= 2
    reduced_h, reduced_w = max(1, round(out_h / factor)), max(1, round(out_w / factor))
    reduced = torch.nn.functional.interpolate(img, size=(reduced_h, reduced_w), mode="area")
    blurred = _gauss_blur_bchw(reduced, sigma / factor)
    return torch.nn.functional.interpolate(blurred, size=(out_h, out_w), mode="bilinear", align_corners=False)


def test_gauss8_mechanism_red_first_naive_pad_is_worse(r: SubTestResult):
    print("\n--- GAUSS8-51: red-first — the naive (block-mixed-edge) pad is measurably "
          "worse than the fix against the same exact reference ---")
    torch.manual_seed(5)
    H = W = 128
    img = _d8.plate_night(H, W, torch.device("cpu"))
    sigma = 1024.0
    exact = _gauss_blur_bchw(img, sigma)
    naive = _naive_pyramid_approx(img, sigma)
    fixed = _gauss_blur_pyramid_approx(img, sigma)
    naive_stats = _d8.code_diff_stats(_d8.aces_srgb8(naive), _d8.aces_srgb8(exact))
    fixed_stats = _d8.code_diff_stats(_d8.aces_srgb8(fixed), _d8.aces_srgb8(exact))
    if not (abs(naive_stats["mean_signed"]) > abs(fixed_stats["mean_signed"])):
        r.fail("gauss8 mechanism red-first",
               f"expected the naive pad's |mean_signed| to exceed the fix's; "
               f"naive={naive_stats}, fixed={fixed_stats}")
        return
    if fixed_stats["max"] > naive_stats["max"] and naive_stats["max"] <= 1:
        r.fail("gauss8 mechanism red-first",
               f"fix should not be worse than the naive pad when the naive pad already "
               f"meets the bar; naive={naive_stats}, fixed={fixed_stats}")
        return
    r.ok(f"naive block-mixed-edge pad: mean_signed={naive_stats['mean_signed']:+.3f}, "
         f"max={naive_stats['max']} vs. the fix: mean_signed={fixed_stats['mean_signed']:+.3f}, "
         f"max={fixed_stats['max']} -- the fix's bias is smaller, proving the mechanism")


def test_gauss8_display_bar_fast_rows(r: SubTestResult):
    print("\n--- GAUSS8-51: the display-8 bar (<=1 code) holds at a scaled-down size, "
          "day and night plates, sigma past the threshold ---")
    H = W = 128
    plates = {"day": _d8.plate_day(H, W, torch.device("cpu")),
              "night": _d8.plate_night(H, W, torch.device("cpu"))}
    for sigma in (260.0, 512.0, 1024.0, 2048.0):
        for pname, p in plates.items():
            exact = _gauss_blur_bchw(p, sigma)
            approx = _gauss_blur_pyramid_approx(p, sigma)
            stats = _d8.code_diff_stats(_d8.aces_srgb8(approx), _d8.aces_srgb8(exact))
            if stats["max"] > 1:
                r.fail(f"gauss8 display bar sigma={sigma} plate={pname}",
                       f"worst-channel code diff {stats['max']} exceeds the 1-code bar "
                       f"(stats={stats})")
                continue
            r.ok(f"sigma={int(sigma)} {pname}: max code diff {stats['max']} <= 1 "
                 f"(mean_signed={stats['mean_signed']:+.3f})")


def test_gauss8_bright_plate_holds_the_bar(r: SubTestResult):
    print("\n--- GAUSS8-51: the bar holds on a highlight-heavy plate (night x16) just past "
          "the threshold, where a cap of 32 read 2 codes ---")
    H = W = 384
    p = _d8.plate_night(H, W, torch.device("cpu")) * 16.0
    for sigma in (260.0, 400.0):
        exact = _gauss_blur_bchw(p, sigma)
        approx = _gauss_blur_pyramid_approx(p, sigma)
        stats = _d8.code_diff_stats(_d8.aces_srgb8(approx), _d8.aces_srgb8(exact))
        if stats["max"] > 1:
            r.fail(f"gauss8 bright plate sigma={sigma}",
                   f"worst-channel code diff {stats['max']} exceeds the 1-code bar (stats={stats})")
            continue
        r.ok(f"night x16 sigma={int(sigma)}: max code diff {stats['max']} <= 1")


def test_gauss8_non_multiple_frame_size_stays_registered(r: SubTestResult):
    print("\n--- GAUSS8-51: a frame side that is not a multiple of the downscale factor "
          "must not stretch the reduced grid (830 = 8*103.75 read 4 codes before) ---")
    big = _d8.plate_night(832, 832, torch.device("cpu"), seed=3) * 16.0
    for side in (826, 830, 832):
        p = big[:, :, :side, :side].contiguous()
        exact = _gauss_blur_bchw(p, 260.0)
        approx = _gauss_blur_pyramid_approx(p, 260.0)
        if approx.shape != p.shape:
            r.fail(f"gauss8 registration side={side}", f"shape {tuple(approx.shape)} != {tuple(p.shape)}")
            continue
        stats = _d8.code_diff_stats(_d8.aces_srgb8(approx), _d8.aces_srgb8(exact))
        if stats["max"] > 1:
            r.fail(f"gauss8 registration side={side}",
                   f"worst-channel code diff {stats['max']} exceeds the 1-code bar (stats={stats})")
            continue
        r.ok(f"side={side}: max code diff {stats['max']} <= 1")


def test_gauss8_border_interpolates_not_clamps(r: SubTestResult):
    print("\n--- GAUSS8-51: the outermost half coarse pixel interpolates toward the border "
          "instead of clamping to the first coarse sample (1084, night x16, read 2 codes) ---")
    p = (_d8.plate_night(1100, 1100, torch.device("cpu"), seed=3) * 16.0)[:, :, :1084, :1084].contiguous()
    exact = _gauss_blur_bchw(p, 260.0)
    approx = _gauss_blur_pyramid_approx(p, 260.0)
    stats = _d8.code_diff_stats(_d8.aces_srgb8(approx), _d8.aces_srgb8(exact))
    if stats["max"] > 1:
        r.fail("gauss8 border interpolation",
               f"worst-channel code diff {stats['max']} exceeds the 1-code bar (stats={stats})")
        return
    r.ok(f"1084 night x16 sigma=260: max code diff {stats['max']} <= 1")


def test_gauss8_thin_strip_holds_the_bar(r: SubTestResult):
    print("\n--- GAUSS8-51: a thin strip (100 rows) keeps a light's blurred peak concentrated; "
          "a cap of 64 read 2 codes here ---")
    p = _d8.plate_night(100, 1097, torch.device("cpu"), seed=2) * 16.0
    for sigma in (260.0, 300.0):
        exact = _gauss_blur_bchw(p, sigma)
        approx = _gauss_blur_pyramid_approx(p, sigma)
        diff = (_d8.aces_srgb8(approx).double() - _d8.aces_srgb8(exact).double()).abs().amax(dim=-1)
        maxcode = int(diff.max())
        if maxcode > 1:
            r.fail(f"gauss8 thin strip sigma={sigma}", f"worst-channel code diff {maxcode} exceeds the 1-code bar")
            continue
        r.ok(f"100x1097 night x16 sigma={int(sigma)}: max code diff {maxcode} <= 1")


def test_gauss8_below_threshold_still_bitexact(r: SubTestResult):
    print("\n--- GAUSS8-51: invariant 7 — the exact path below the threshold is untouched "
          "by the edge-pad fix or the raised cap ---")
    torch.manual_seed(9)
    img = torch.rand(1, 3, 10, 10)
    for sigma in (0.0, 1.0, 64.0, 200.0, 256.0):
        auto_out = _sc._gauss_blur_auto(img, sigma)
        exact_out = _gauss_blur_bchw(img, sigma)
        if not torch.equal(auto_out, exact_out):
            r.fail(f"gauss8 bitexact sigma={sigma}", "the below-threshold path changed")
            return
    r.ok("_gauss_blur_auto stays torch.equal() to _gauss_blur_bchw for every sigma <= threshold")


def test_gauss8_edge_pad_matches_kernel_of_plain_replicate_when_uniform(r: SubTestResult):
    print("\n--- GAUSS8-51: _gauss_blur_bchw_edge_pad reduces to plain replicate-pad "
          "behaviour on a spatially uniform image (no edge/interior mismatch possible) ---")
    torch.manual_seed(1)
    img = torch.full((1, 3, 12, 12), 0.4)
    sigma = 3.0
    plain = _gauss_blur_bchw(img, sigma)
    left = img[:, :, :, 0:1]
    right = img[:, :, :, -1:]
    top = img[:, :, 0:1, :]
    bottom = img[:, :, -1:, :]
    corners = (img[:, :, 0:1, 0:1], img[:, :, 0:1, -1:], img[:, :, -1:, 0:1], img[:, :, -1:, -1:])
    edge_padded = _gauss_blur_bchw_edge_pad(img, sigma, left, right, top, bottom, corners)
    if not torch.allclose(plain, edge_padded, atol=1e-5):
        r.fail("gauss8 edge pad uniform equivalence",
               f"max abs diff {(plain - edge_padded).abs().max().item():.6f} on a uniform "
               f"image, where the true edge and the block-averaged edge are identical")
        return
    r.ok("edge-pad blur matches plain replicate-pad blur exactly when there is no "
         "edge/interior content to mismatch")


# ── v0.51: fast rows for the shape families the display-8 bar and
# `docs/resolution-scale.md`'s tables never covered -- batch>1, 1/4 channels, fp16 and
# tiny frames. The [1,3,H,W] fp32 square-plate bar above says nothing about any of
# these; a v0.51 review note flagged the gap and a probe confirmed it by probe (no crash,
# but no accuracy check existed for any of them). ─────────────────────────────────

def test_gauss8_fast_rows_batch_gt1(r: SubTestResult):
    print("\n--- batch>1 -- two independent plates in one call, each batch index "
          "checked against the display-8 bar separately ---")
    H = W = 96
    day = _d8.plate_day(H, W, torch.device("cpu"))
    night = _d8.plate_night(H, W, torch.device("cpu"))
    batched = torch.cat([day, night], dim=0)  # [2, 3, H, W]
    for sigma in (512.0, 1024.0):
        exact = _gauss_blur_bchw(batched, sigma)
        approx = _gauss_blur_pyramid_approx(batched, sigma)
        for b, name in enumerate(("day", "night")):
            stats = _d8.code_diff_stats(_d8.aces_srgb8(approx[b:b + 1]), _d8.aces_srgb8(exact[b:b + 1]))
            if stats["max"] > 1:
                r.fail(f"gauss8 batch>1 sigma={sigma} batch={b}({name})",
                       f"worst-channel code diff {stats['max']} exceeds the 1-code bar (stats={stats})")
                continue
            r.ok(f"sigma={int(sigma)} batch={b}({name}): max code diff {stats['max']} <= 1")


def test_gauss8_fast_rows_channel_counts(r: SubTestResult):
    print("\n--- 1 and 4 channels -- the ACES display transform is RGB-only, so this "
          "checks the pyramid path directly against the exact reference in linear space, "
          "against the same 0.05-0.10 max-abs band docs/resolution-scale.md's R1 promise "
          "already accepts for this builtin family (there is no display-8 pipeline for a "
          "non-3-channel image) ---")
    torch.manual_seed(11)
    H = W = 96
    for C in (1, 4):
        img = torch.rand(1, C, H, W) * 4.0 + 0.01
        for sigma in (512.0, 1024.0):
            exact = _gauss_blur_bchw(img, sigma)
            approx = _gauss_blur_pyramid_approx(img, sigma)
            maxdiff = (exact - approx).abs().max().item()
            bound = 0.10 * img.max().item()
            if maxdiff > bound:
                r.fail(f"gauss8 channels={C} sigma={sigma}",
                       f"max abs diff {maxdiff:.4f} exceeds the stated {bound:.4f} bound")
                continue
            r.ok(f"channels={C} sigma={int(sigma)}: max abs diff {maxdiff:.4f} <= {bound:.4f}")


def test_gauss8_fast_rows_fp16(r: SubTestResult):
    print("\n--- fp16 -- a STATED fp16 bound of <= 2 codes (vs. the fp32 bar's 1 code), "
          "since the reduce/blur/upsample chain itself runs in fp16 and the kernel cast in "
          "`_gauss_blur_bchw` (M-3) rounds to fp16, not because this diff changed anything "
          "about the mechanism -- the exact reference stays fp32 (there is no separate fp16 "
          "'exact' answer to compare against) ---")
    H = W = 96
    plates = {"day": _d8.plate_day(H, W, torch.device("cpu")), "night": _d8.plate_night(H, W, torch.device("cpu"))}
    for sigma in (512.0, 1024.0):
        for pname, p in plates.items():
            exact = _gauss_blur_bchw(p, sigma)
            approx16 = _gauss_blur_pyramid_approx(p.half(), sigma).float()
            stats = _d8.code_diff_stats(_d8.aces_srgb8(approx16), _d8.aces_srgb8(exact))
            if stats["max"] > 2:
                r.fail(f"gauss8 fp16 sigma={sigma} plate={pname}",
                       f"worst-channel code diff {stats['max']} exceeds the stated fp16 bound of 2 "
                       f"(stats={stats})")
                continue
            r.ok(f"fp16 sigma={int(sigma)} {pname}: max code diff {stats['max']} <= 2 (stated fp16 bound)")


def test_gauss8_fast_rows_tiny_frames(r: SubTestResult):
    print("\n--- tiny frames -- 1x1, 2x2 and elongated 1xN/Nx1 frames, where the "
          "pyramid's own factor almost always exceeds every side (a flat-field reduction, "
          "never padded/extended). Uses a direct worst-channel-code diff, not "
          "`code_diff_stats`'s centre-half crop -- that crop (`h//4:3*h//4`) is EMPTY "
          "for H or W < 4, which is the whole point of this row ---")
    torch.manual_seed(13)
    sigma = 1024.0
    for H, W in ((1, 1), (2, 2), (1, 64), (64, 1), (3, 5)):
        img = torch.rand(1, 3, H, W)
        exact = _gauss_blur_bchw(img, sigma)
        approx = _gauss_blur_pyramid_approx(img, sigma)
        if approx.shape != img.shape:
            r.fail(f"gauss8 tiny shape {H}x{W}", f"got {tuple(approx.shape)}, expected {tuple(img.shape)}")
            continue
        if not torch.isfinite(approx).all():
            r.fail(f"gauss8 tiny finite {H}x{W}", "non-finite output")
            continue
        diff = (_d8.aces_srgb8(approx).double() - _d8.aces_srgb8(exact).double()).abs().amax(dim=-1)
        maxcode = int(diff.max())
        if maxcode > 1:
            r.fail(f"gauss8 tiny bar {H}x{W}",
                   f"worst-channel code diff {maxcode} exceeds the 1-code bar")
            continue
        r.ok(f"{H}x{W}: max code diff {maxcode} <= 1")
