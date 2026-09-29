"""FIX-APPROX A1 (v0.50 Phase C) — a genuinely narrowed (non-saturating) window of
`gauss_blur`/`bilateral_filter` past their approximation threshold no longer silently
diverges from a whole-frame cook.

Both builtins' downscale approximations (`gauss_blur`'s pyramid past
`GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA`; `bilateral_filter`'s detail-transfer past
`_BILATERAL_APPROX_THRESHOLD_SS`, 40/3, past which the exact tier gives way) resample starting at the CROP's own (0,0), not the
frame's absolute coordinates. A window whose halo has not saturated to the whole frame
therefore used to sample on a different phase than a whole-frame cook of the same
program (B1/B2's bug hunt: gauss_blur maxdiff ~8e-5, bilateral_filter maxdiff up to
0.0265 -- a visible divergence, not a rounding footnote).

The fix (`tex_roi._reach_of`'s `halo_arg` branch, the footprint's new 4th element
`approx_above`): once a call's folded argument crosses the builtin's own approximation
threshold, `_reach_of` answers `'unbounded'` -- the SAME decline a symbolic (non-foldable)
radius already got -- so the planner (ROI, tiling/OOM strips, `cook_stage_dag`, since all
three funnel through this one function) falls back to a whole-frame cook instead of
narrowing onto a wrong phase. Below each threshold, nothing changes (invariant 7): the
window still narrows and still matches a whole-frame crop exactly.
"""
from __future__ import annotations

import torch

from helpers import *  # noqa: F401,F403
from helpers import windowed_vs_whole
from TEX_Wrangle.tex_runtime.stdlib_core import GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib as _Sample  # populates REGISTRY


def _windowed_vs_whole(code, image, roi):
    return windowed_vs_whole(code, image, roi, whole=True)


def _make_frame(H, W, seed):
    torch.manual_seed(seed)
    return torch.rand(1, H, W, 3)


# ── gauss_blur ────────────────────────────────────────────────────────────────────────

def test_fixapprox_a1_gauss_blur_past_threshold_declines_and_matches_whole_frame(r: SubTestResult):
    print("\n--- A1: gauss_blur past the pyramid threshold, a genuinely narrowed (non-"
          "saturating) window declines and matches a whole-frame cook ---")
    H = W = 4320
    image = _make_frame(H, W, seed=101)
    sigma = GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA + 4.0  # 260.0: just past the threshold
    code = f"@OUT = gauss_blur(@A, {sigma});"
    # A window well inside the interior, with a halo (ceil(3*260)=780) that does NOT
    # saturate to the 4320-tall frame -- exactly B1's non-saturating repro shape.
    roi = (1220, 1220, 2560, 2560, W, H)
    cooked_roi, win, full = _windowed_vs_whole(code, image, roi)
    if cooked_roi is not None:
        r.fail("gauss_blur a1 decline", f"window narrowed instead of declining: cooked_roi={cooked_roi}")
        return
    if tuple(win.shape) != tuple(full.shape):
        r.fail("gauss_blur a1 decline", f"win shape {tuple(win.shape)} != whole-frame shape {tuple(full.shape)}")
        return
    if torch.equal(win, full):
        r.ok("gauss_blur past threshold declines the window (cooked_roi=None) and is "
             "torch.equal to an independent whole-frame cook")
    else:
        md = (win.float() - full.float()).abs().max().item()
        r.fail("gauss_blur a1 decline", f"declined-window output diverges from whole-frame, maxdiff={md:.4e}")


def test_fixapprox_a1_gauss_blur_below_threshold_still_narrows(r: SubTestResult):
    print("\n--- A1: gauss_blur BELOW the pyramid threshold is unaffected -- still narrows, "
          "still bit-exact vs a whole-frame crop (invariant 7) ---")
    H = W = 200
    image = _make_frame(H, W, seed=102)
    code = "@OUT = gauss_blur(@A, 4.0);"  # well below 256
    roi = (60, 60, 40, 40, W, H)
    cooked_roi, win, full = _windowed_vs_whole(code, image, roi)
    if cooked_roi != roi:
        r.fail("gauss_blur a1 below-threshold", f"window unexpectedly declined: cooked_roi={cooked_roi}")
        return
    x0, y0, w, h, _, _ = roi
    crop = full[:, y0:y0 + h, x0:x0 + w]
    if torch.equal(win, crop):
        r.ok("gauss_blur below threshold still narrows and matches the whole-frame crop exactly")
    else:
        md = (win.float() - crop.float()).abs().max().item()
        r.fail("gauss_blur a1 below-threshold", f"maxdiff={md:.4e}")


# ── bilateral_filter ─────────────────────────────────────────────────────────────────

def test_fixapprox_a1_bilateral_past_threshold_declines_and_matches_whole_frame(r: SubTestResult):
    print("\n--- A1: bilateral_filter past the detail-transfer threshold, a genuinely "
          "narrowed (non-saturating) window declines and matches a whole-frame cook ---")
    H = W = 4320
    image = _make_frame(H, W, seed=103)
    ss = _Sample._BILATERAL_APPROX_THRESHOLD_SS + 0.5  # 40/3 + 0.5: just past the threshold
    code = f"@OUT = bilateral_filter(@A, {ss}, 0.2);"
    roi = (1900, 1900, 1200, 1200, W, H)
    cooked_roi, win, full = _windowed_vs_whole(code, image, roi)
    if cooked_roi is not None:
        r.fail("bilateral a1 decline", f"window narrowed instead of declining: cooked_roi={cooked_roi}")
        return
    if tuple(win.shape) != tuple(full.shape):
        r.fail("bilateral a1 decline", f"win shape {tuple(win.shape)} != whole-frame shape {tuple(full.shape)}")
        return
    if torch.equal(win, full):
        r.ok("bilateral_filter past threshold declines the window (cooked_roi=None) and is "
             "torch.equal to an independent whole-frame cook")
    else:
        md = (win.float() - full.float()).abs().max().item()
        r.fail("bilateral a1 decline", f"declined-window output diverges from whole-frame, maxdiff={md:.4e}")


def test_fixapprox_a1_bilateral_below_threshold_still_narrows(r: SubTestResult):
    print("\n--- A1: bilateral_filter BELOW the detail-transfer threshold is unaffected -- "
          "still narrows, still bit-exact vs a whole-frame crop (invariant 7) ---")
    H = W = 200
    image = _make_frame(H, W, seed=104)
    code = "@OUT = bilateral_filter(@A, 3.0, 0.2);"  # well below 40/3 (the exact tier)
    roi = (60, 60, 40, 40, W, H)
    cooked_roi, win, full = _windowed_vs_whole(code, image, roi)
    if cooked_roi != roi:
        r.fail("bilateral a1 below-threshold", f"window unexpectedly declined: cooked_roi={cooked_roi}")
        return
    x0, y0, w, h, _, _ = roi
    crop = full[:, y0:y0 + h, x0:x0 + w]
    if torch.equal(win, crop):
        r.ok("bilateral_filter below threshold still narrows and matches the whole-frame crop exactly")
    else:
        md = (win.float() - crop.float()).abs().max().item()
        r.fail("bilateral a1 below-threshold", f"maxdiff={md:.4e}")
