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
raised 8.0 -> 32.0 (author-approved) — at cap=8.0 the reduced level shrinks fast enough
that even a correctly-seeded pad still lets a several-pixel-wide reduced image's border
dominate at high sigma; cap=32.0 keeps the reduced level wide enough for the fix to hold
the bar across the sigmas this file (and `docs/resolution-scale.md`'s table) cover.

Fast rows only: 128x128 plates (vs. the 1080x1080 readings in
`docs/resolution-scale.md`'s own table), sigma scaled to stay in the same "well past
threshold" regime the full-size table exercises.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from helpers import *
from TEX_Wrangle.tex_runtime import stdlib_core as _sc
from TEX_Wrangle.tex_runtime.stdlib_core import (
    GAUSS_BLUR_PYRAMID_QUALITY_CAP as QUALITY_CAP,
    _gauss_blur_bchw,
    _gauss_blur_bchw_edge_pad,
    _gauss_blur_pyramid_approx,
    _get_gauss_kernels,
)

_PKG = Path(__file__).resolve().parent.parent  # TEX_Wrangle/
_TOOL = _PKG / "tools" / "display8.py"


def _load_display8():
    spec = importlib.util.spec_from_file_location("_display8_bar_harness", _TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_d8 = _load_display8()


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
