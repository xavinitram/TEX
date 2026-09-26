"""SCALE-47b phase 4 — halo math scales, ceiling up, never down.

`SCALE-47-design.md` §1/§5 step 2: `stage_halo`/`roi_plan`'s margin derivation must multiply by
the cook's resolution scale, with the same "ceil, never round down" discipline `mult` (the
ROI-1 reach multiplier) already established -- under-padding a halo silently serves stale ring
pixels (invariant #5's whole reason for existing). `chain_windows` accepts the same `scale=`
kwarg for API symmetry (SCALE-47-design.md names all three call sites); its own composition
arithmetic needs no change because the `halos` it receives are already scale-adjusted by
whichever of the other two produced them.
"""
from helpers import *
from TEX_Wrangle import tex_roi as R
import math


def test_scale47b_stage_halo_scales_down(r: SubTestResult):
    print("\n--- SCALE-47b: stage_halo(scale=0.5) halves gauss_blur(8.0)'s margin ---")
    code = "@OUT = gauss_blur(@A, 8.0);"
    full = R.stage_halo(code)
    half = R.stage_halo(code, scale=0.5)
    if full != 24:
        r.fail("stage_halo baseline", f"expected 24 (ceil(3*8)) at scale=1.0, got {full}")
        return
    if half != 12:
        r.fail("stage_halo scaled", f"expected 12 (half of 24) at scale=0.5, got {half}")
        return
    r.ok(f"stage_halo: full={full}, scale=0.5 -> {half}")


def test_scale47b_stage_halo_ceils_never_rounds_down(r: SubTestResult):
    print("\n--- SCALE-47b: a fractional scaled margin rounds UP, never down ---")
    # radius 5 at scale 0.3 -> 1.5 -> must ceil to 2, never floor/round to 1.
    code = "@OUT = erode(@A, 5.0);"
    full = R.stage_halo(code)
    if full != 5:
        r.fail("erode baseline", f"expected halo=5, got {full}")
        return
    scaled = R.stage_halo(code, scale=0.3)
    want = math.ceil(5 * 0.3)
    if scaled != want:
        r.fail("erode ceil-up", f"expected ceil(5*0.3)={want}, got {scaled}")
        return
    if scaled < 5 * 0.3:
        r.fail("erode never-under-pad", f"{scaled} < {5*0.3} -- under-padded")
        return
    r.ok(f"erode(5.0) at scale=0.3: halo={scaled} (ceil of {5*0.3}, never floors)")


def test_scale47b_stage_halo_default_unaffected(r: SubTestResult):
    print("\n--- SCALE-47b: stage_halo() with no scale= is byte-identical (invariant #7) ---")
    code = "@OUT = gauss_blur(@A, 8.0);"
    a = R.stage_halo(code)
    b = R.stage_halo(code, scale=1.0)
    if a != b:
        r.fail("scale=1.0 parity", f"expected {a} == {b}")
        return
    r.ok(f"stage_halo() == stage_halo(scale=1.0) == {a}")


def test_scale47b_roi_plan_halo_scales(r: SubTestResult):
    print("\n--- SCALE-47b: roi_plan(scale=...).halo agrees with stage_halo(scale=...) ---")
    code = "@OUT = gauss_blur(@A, 8.0);"
    plan = R.roi_plan(code, scale=0.5)
    sh = R.stage_halo(code, scale=0.5)
    if not plan.executable:
        r.fail("roi_plan executable", "expected an executable plan for a plain gauss_blur")
        return
    if plan.halo != sh:
        r.fail("roi_plan/stage_halo agreement", f"roi_plan.halo={plan.halo} != stage_halo={sh}")
        return
    r.ok(f"roi_plan(scale=0.5).halo == stage_halo(scale=0.5) == {sh}")


def test_scale47b_chain_windows_accepts_scale_kwarg(r: SubTestResult):
    print("\n--- SCALE-47b: chain_windows accepts scale= (API symmetry, no behaviour change) ---")
    halos = [0, 4]
    roi = (0, 0, 8, 8, 32, 32)
    a = R.chain_windows(halos, roi)
    b = R.chain_windows(halos, roi, scale=1.0)
    if a != b:
        r.fail("chain_windows scale=1.0 parity", f"{a!r} != {b!r}")
        return
    r.ok("chain_windows(..., scale=1.0) matches the no-scale call exactly")
