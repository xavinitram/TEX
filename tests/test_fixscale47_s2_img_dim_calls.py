"""FIX-SCALE S2 (v0.47 Phase C, B2 finding 2) — `img_width`/`img_height` are ordinary stdlib
FUNCTION CALLS (`img_width(@A)`), never bare identifiers; the classifier's `_PIXEL_DIM_NAMES`
check lived only in the `Identifier` arm of `_scale_unsafe_walk`, a branch the parser can
never reach for these two names (it has no grammar production that emits a bare
`Identifier("img_width")`). A program that folds `img_width(@A)`/`img_height(@A)` into its
output — the exact hazard `docs/resolution-scale.md` documents as excluded — was silently
classified `safe=True`, so a coarse cook produced a value proportional to whatever canvas the
proxy happened to be: the silent-wrong-picture outcome R3 exists to prevent."""
from helpers import *
from TEX_Wrangle import tex_roi as R
from TEX_Wrangle import tex_engine


def test_s2_img_width_call_is_unsafe(r: SubTestResult):
    print("\n--- FIX-SCALE S2: img_width(@A) folded into the output is classified unsafe ---")
    code = "@OUT = vec4(img_width(@A) * 0.001, img_height(@A) * 0.001, 0.0, 1.0);"
    if R.scale_safe(code):
        r.fail("img_width call safety", "img_width(@A)/img_height(@A) folded into the output "
               "was classified safe -- the exact silent-wrong-picture hazard R3 exists to catch")
        return
    r.ok("img_width(@A)*0.001 is NOT scale_safe")


def test_s2_img_dim_call_behind_a_fetch_coordinate_is_still_unsafe(r: SubTestResult):
    print("\n--- FIX-SCALE S2: img_width(@A)/img_height(@A) inside a fetch coord arg is still unsafe ---")
    code = "@OUT = fetch(@A, int(img_width(@A) * 0.5), int(img_height(@A) * 0.5));"
    if R.scale_safe(code):
        r.fail("img_dim in fetch coord", "img_width/img_height used to COMPUTE a fetch "
               "coordinate was classified safe -- the whitelisted fetch position covers ix/iy "
               "themselves, never a call that reads the frame dimension")
        return
    r.ok("fetch(@A, int(img_width(@A)*0.5), ...) is NOT scale_safe")


def test_s2_plain_program_with_no_img_dim_call_is_unaffected(r: SubTestResult):
    print("\n--- FIX-SCALE S2: a program with no img_width/img_height call is unaffected ---")
    code = "@OUT = gauss_blur(@A, 4.0);"
    if not R.scale_safe(code):
        r.fail("no regression", "a plain gauss_blur program (no img_width/img_height call) "
               "was classified unsafe after the S2 fix")
        return
    r.ok("gauss_blur(@A, 4.0) is still scale_safe")


def test_s2_prepare_refuses_scale_on_img_width_call(r: SubTestResult):
    print("\n--- FIX-SCALE S2: prepare(scale=0.5) refuses a program that calls img_width ---")
    A = make_img(1, 8, 8, 4)
    code = "@OUT = vec4(img_width(@A) * 0.001, img_height(@A) * 0.001, 0.0, 1.0);"
    raised = None
    try:
        tex_engine.prepare(code, {"A": A}, device_mode="cpu", scale=0.5)
    except Exception as e:
        raised = e
    if raised is None:
        r.fail("prepare refusal", "prepare(scale=0.5) did not refuse a program that folds "
               "img_width(@A)/img_height(@A) into its output")
        return
    refusal = getattr(raised, "tex_refusal", None)
    if refusal is None or refusal.code != "scale-unsafe":
        r.fail("prepare refusal code", f"expected tex_refusal.code=='scale-unsafe', got "
               f"{refusal!r}")
        return
    r.ok(f"prepare(scale=0.5) refused with tex_refusal.code={refusal.code!r}")
