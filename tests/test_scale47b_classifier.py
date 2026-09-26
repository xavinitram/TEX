"""SCALE-47b phase 6 — the conservative scale-safety classifier + override pragma + refusal.

`SCALE-47-design.md` §(a), AUTHOR DECISIONS #2: a program that reads pixel coordinates by hand
(`ix`/`iy`/`img_width`/`img_height` outside the whitelisted `fetch`/`@A[x,y]` call sites) does
NOT look like "the same picture, downscaled" under a resolution change, so a cook asking for a
non-1.0 scale on such a program must REFUSE rather than silently cook at full scale (R5: the
engine never chooses) or silently cook wrong (R1). The classifier over-approximates toward
UNSAFE (declines to prove safety rather than risk it) and an author override comment,
`//!tex scale: safe` / `//!tex scale: never`, parsed the same way as the existing `//!tex X.Y`
language pragma, can force the verdict either direction.
"""
from helpers import *
from TEX_Wrangle import tex_roi as R
from TEX_Wrangle import tex_engine


def test_scale47b_plain_blur_program_is_safe(r: SubTestResult):
    print("\n--- SCALE-47b: a plain gauss_blur/sample program is classified safe ---")
    code = "@OUT = gauss_blur(@A, 4.0);"
    if not R.scale_safe(code):
        r.fail("plain blur safety", "a plain gauss_blur program was classified unsafe")
        return
    r.ok("gauss_blur(@A, 4.0) is scale_safe")


def test_scale47b_hand_pixel_math_is_unsafe(r: SubTestResult):
    print("\n--- SCALE-47b: hand pixel arithmetic (ix/iy outside fetch) is classified unsafe ---")
    code = "@OUT = vec4(ix * 0.01, iy * 0.01, 0.0, 1.0);"
    if R.scale_safe(code):
        r.fail("hand pixel math safety",
               "a program reading ix/iy directly (not as a fetch coordinate) was classified safe")
        return
    r.ok("vec4(ix*0.01, iy*0.01, ...) is NOT scale_safe (ix/iy used outside a whitelisted call)")


def test_scale47b_img_width_is_unsafe(r: SubTestResult):
    print("\n--- SCALE-47b: img_width/img_height used arbitrarily is classified unsafe ---")
    code = "@OUT = vec4(1.0 / img_width, 1.0 / img_height, 0.0, 1.0);"
    if R.scale_safe(code):
        r.fail("img_width safety", "a program dividing by img_width/img_height was classified safe")
        return
    r.ok("1.0/img_width is NOT scale_safe")


def test_scale47b_fetch_with_ix_iy_is_safe(r: SubTestResult):
    print("\n--- SCALE-47b: fetch(@A, ix, iy) itself is the whitelisted, safe use of ix/iy ---")
    code = "@OUT = fetch(@A, ix, iy);"
    if not R.scale_safe(code):
        r.fail("fetch whitelist", "fetch(@A, ix, iy) -- the canonical whitelisted use -- "
               "was classified unsafe")
        return
    r.ok("fetch(@A, ix, iy) is scale_safe (ix/iy read the ACTUAL, possibly-scaled canvas)")


def test_scale47b_pragma_override_safe(r: SubTestResult):
    print("\n--- SCALE-47b: //!tex scale: safe overrides an unsafe verdict ---")
    code = "//!tex scale: safe\n@OUT = vec4(ix * 0.01, iy * 0.01, 0.0, 1.0);"
    if not R.scale_safe(code):
        r.fail("safe override", "the //!tex scale: safe pragma did not override the classifier")
        return
    r.ok("//!tex scale: safe forces a program the classifier can't prove safe to be treated as safe")


def test_scale47b_pragma_override_never(r: SubTestResult):
    print("\n--- SCALE-47b: //!tex scale: never overrides a safe verdict ---")
    code = "//!tex scale: never\n@OUT = gauss_blur(@A, 4.0);"
    if R.scale_safe(code):
        r.fail("never override", "the //!tex scale: never pragma did not override the classifier")
        return
    r.ok("//!tex scale: never forces a plain-looking program to be declared unsafe")


def test_scale47b_unsafe_program_refuses_nontrivial_scale(r: SubTestResult):
    print("\n--- SCALE-47b: scale != 1 on an unsafe program REFUSES (never silently full-scale) ---")
    A = make_img(1, 8, 8, 4)
    code = "@OUT = vec4(ix * 0.01, iy * 0.01, 0.0, 1.0);"
    raised = None
    try:
        tex_engine.prepare(code, {"A": A}, device_mode="cpu", scale=0.5)
    except Exception as e:
        raised = e
    if raised is None:
        r.fail("scale refusal", "prepare() did not raise for an unsafe program at scale=0.5")
        return
    refusal = getattr(raised, "tex_refusal", None)
    if refusal is None or refusal.code != "scale-unsafe":
        r.fail("scale refusal structure", f"expected .tex_refusal.code=='scale-unsafe', "
               f"got {refusal!r} on {type(raised).__name__}: {raised}")
        return
    r.ok(f"prepare(scale=0.5) on an unsafe program raised with tex_refusal.code={refusal.code!r}")


def test_scale47b_unsafe_program_scale_none_or_one_is_fine(r: SubTestResult):
    print("\n--- SCALE-47b: scale=None or scale=1.0 never refuses, even on an unsafe program ---")
    A = make_img(1, 8, 8, 4)
    code = "@OUT = vec4(ix * 0.01, iy * 0.01, 0.0, 1.0);"
    try:
        tex_engine.prepare(code, {"A": A}, device_mode="cpu")
        tex_engine.prepare(code, {"A": A.clone()}, device_mode="cpu", scale=1.0)
    except Exception as e:
        r.fail("no spurious refusal", f"unexpected raise on scale=None/1.0: {e}")
        return
    r.ok("an 'unsafe' program cooks normally at scale=None and scale=1.0 (R5: the engine "
         "never picks a scale, so a request that never asks for one never refuses)")
