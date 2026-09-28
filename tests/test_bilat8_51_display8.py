"""BILAT8-51 fast rows against the shared display-8 harness (`tools/display8.py`,
cherry-picked from GAUSS8-51's `e7796e9`, common brief).

Fast tier only: a small plate (64x64, not 1080p) with the SAME spatial_sigma /
radius the 1080p sweep uses (radius does not need to shrink with the plate --
cost is O(image size x radius), so shrinking the image alone already buys the
speedup, and keeps the separable code path's own regime, radius in
(_BILATERAL_EXACT_RADIUS_MAX, _BILATERAL_SEPARABLE_RADIUS_MAX], genuinely
exercised). The 1080p table itself lives in `docs/resolution-scale.md`
(the BILAT8-51 write-up) -- this file is the ratchet, not the evidence.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

from helpers import *
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib

_PKG = Path(__file__).resolve().parent.parent
_TOOL = _PKG / "tools" / "display8.py"


def _load_display8():
    spec = importlib.util.spec_from_file_location("_display8_harness_bilat8", _TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_d8 = _load_display8()
aces_srgb8, code_diff_stats, plate_day, plate_night = (
    _d8.aces_srgb8, _d8.code_diff_stats, _d8.plate_day, _d8.plate_night,
)


def _old_separable_bchw(bchw, ss, sr, radius):
    """Pre-BILAT8-51 math: the column pass's range weight compared the row
    pass's OWN output against itself -- kept here only as the fast tier's own
    before/after baseline (a byte-for-byte copy, same as
    test_bilat50_radius.py's copy of the same math)."""
    row_passed = TEXStdlib._bilateral_separable_1d_pass(bchw, 3, radius, ss, sr, range_ref=bchw)
    return TEXStdlib._bilateral_separable_1d_pass(row_passed, 2, radius, ss, sr, range_ref=row_passed)


def test_bilat8_51_display8_fast_rows_do_not_regress(r: SubTestResult):
    print("\n--- BILAT8-51 fast tier: a630685's range-vs-original fix never scores more "
          "changed/>=2/max codes than the pre-fix separable path against the exact filter, "
          "on the shared harness's day/night plates at a small size, ss=8.5/16 ---")
    H = W = 64
    device = torch.device("cpu")
    plates = {"day": plate_day(H, W, device), "night": plate_night(H, W, device)}
    for ss in (8.5, 16.0):
        radius = int(math.ceil(3 * ss))
        for pname, p in plates.items():
            exact = TEXStdlib._bilateral_exact_bchw(p, ss, 0.2, radius)
            e_exact = aces_srgb8(exact)
            fixed = TEXStdlib._bilateral_separable_bchw(p.clone(), ss, 0.2, radius)
            old = _old_separable_bchw(p.clone(), ss, 0.2, radius)
            s_fixed = code_diff_stats(aces_srgb8(fixed), e_exact)
            s_old = code_diff_stats(aces_srgb8(old), e_exact)
            # A tiny (<=1-pixel) tolerance on the >=2-code fraction: at 64x64 a single
            # pixel is ~0.024% of the frame, and the night plate's 40 tiny practical
            # lights make single-pixel code flips at this scale expected sampling noise,
            # not the hard-edge regression this row exists to catch. `max` gets none.
            tol = 1.5 / (H * W)
            if s_fixed["ge2"] > s_old["ge2"] + tol or s_fixed["max"] > s_old["max"]:
                r.fail(f"{pname} ss={ss}", f"fixed {s_fixed} regresses past pre-fix {s_old}")
                return
    r.ok("a630685's separable fix is never worse (>=2-code fraction, max code, within a "
         "1-pixel sampling-noise tolerance) than the pre-fix path on the shared harness's "
         "own plates at ss=8.5/16, radius 26/48")


def test_bilat8_51_display8_fast_rows_exact_tier_untouched(r: SubTestResult):
    print("\n--- BILAT8-51 fast tier: the exact MATH (radius<=24) is bit-identical on the "
          "shared harness's plates -- BILAT8-51 touched only the separable path, never this "
          "regime. BILATX-51 later gave 3<radius<=40 a different IMPLEMENTATION of the same "
          "math (`_bilateral_exact_taploop_bchw`, dispatched by `fn_bilateral_filter`), so this "
          "row now checks the MATH (both functions) rather than assuming a specific one is "
          "still wired -- see test_bilatx51_taploop.py for the dispatch-level pin ---")
    H = W = 32
    device = torch.device("cpu")
    p = plate_day(H, W, device)
    ss, sr = 6.0, 0.2  # radius = 18, inside the exact tier
    bchw = p
    img = bchw.permute(0, 2, 3, 1)
    via_fn = TEXStdlib.fn_bilateral_filter(img.clone(), ss, sr)
    direct = TEXStdlib._bilateral_exact_bchw(bchw.clone(), ss, sr, 18).permute(0, 2, 3, 1)
    if not torch.equal(via_fn, direct):
        r.fail("exact tier", "fn_bilateral_filter's radius=18 output is not torch.equal to "
               "_bilateral_exact_bchw's own math -- BILAT8-51 must not have touched this "
               "regime's math (BILATX-51 may have changed which function implements it)")
        return
    r.ok("radius=18 (inside the exact-math tier) stays bit-identical to _bilateral_exact_bchw's "
         "own math on the shared harness's own plate -- BILAT8-51's fix is confined to the "
         "separable path")
