"""SCALE-47b phase 3 — runtime-scalar injection: the interpreter actually multiplies.

`SCALE-47-design.md` §2: scale multiplies the RESOLVED value of every `pixel_args=`-tagged
argument at the call site, as a runtime value (never an AST fold). Proven here on the
interpreter tier (the oracle, "oracle-first" per the design's phased plan): a `gauss_blur(@A,
sigma)` cook at `scale=0.5` must match a cook of the SAME program at `scale=None` with the
sigma argument pre-halved, within ordinary fp32 tolerance -- proving the multiplier is applied
to the right argument, not merely that *something* changed.

Also proves the tier-routing consequence: a scale-active cook forces the interpreter tier even
when an accelerated `compile_mode` was requested (no tier currently threads scale through
codegen/torch.compile/CUDA-graph capture), and that this is RECORDED (never a silent fallback) --
and that `scale=None` is completely unaffected (invariant #7 -- byte-identical to a call that
never mentions scale at all).
"""
from helpers import *
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import tier_trace as _tt


def _mk(seed_val=0.3):
    return make_img(1, 8, 8, 4).fill_(seed_val) if hasattr(make_img(1, 8, 8, 4), "fill_") else None


def test_scale47b_gauss_blur_sigma_scales(r: SubTestResult):
    print("\n--- SCALE-47b: gauss_blur's sigma is multiplied by scale (interpreter tier) ---")
    A = make_img(1, 16, 16, 4)
    code = "@OUT = gauss_blur(@A, 8.0);"
    scaled = tex_engine.cook(code, {"A": A.clone()}, device_mode="cpu", scale=0.5)
    code_half = "@OUT = gauss_blur(@A, 4.0);"
    half = tex_engine.cook(code_half, {"A": A.clone()}, device_mode="cpu")
    md = (scaled.outputs["OUT"].float() - half.outputs["OUT"].float()).abs().max().item()
    if md < 1e-5:
        r.ok(f"gauss_blur(@A, 8.0) at scale=0.5 == gauss_blur(@A, 4.0) at scale=None "
             f"(maxdiff {md:.2e})")
    else:
        r.fail("gauss_blur scale multiply", f"maxdiff {md:.2e} -- sigma was not halved")


def test_scale47b_erode_radius_scales(r: SubTestResult):
    print("\n--- SCALE-47b: erode's radius is multiplied by scale (interpreter tier) ---")
    A = make_img(1, 16, 16, 4)
    scaled = tex_engine.cook("@OUT = erode(@A, 4.0);", {"A": A.clone()},
                             device_mode="cpu", scale=0.5)
    half = tex_engine.cook("@OUT = erode(@A, 2.0);", {"A": A.clone()}, device_mode="cpu")
    if torch.equal(scaled.outputs["OUT"], half.outputs["OUT"]):
        r.ok("erode(@A, 4.0) at scale=0.5 == erode(@A, 2.0) at scale=None")
    else:
        md = (scaled.outputs["OUT"].float() - half.outputs["OUT"].float()).abs().max().item()
        r.fail("erode scale multiply", f"maxdiff {md:.2e} -- radius was not halved")


def test_scale47b_non_pixel_arg_untouched(r: SubTestResult):
    print("\n--- SCALE-47b: bilateral_filter's range_sigma is NOT scaled (only arg 1 is) ---")
    A = make_img(1, 12, 12, 4)
    a = tex_engine.cook("@OUT = bilateral_filter(@A, 6.0, 0.2);", {"A": A.clone()},
                        device_mode="cpu", scale=0.5)
    b = tex_engine.cook("@OUT = bilateral_filter(@A, 3.0, 0.2);", {"A": A.clone()},
                        device_mode="cpu")
    c = tex_engine.cook("@OUT = bilateral_filter(@A, 3.0, 0.1);", {"A": A.clone()},
                        device_mode="cpu")
    md_ab = (a.outputs["OUT"].float() - b.outputs["OUT"].float()).abs().max().item()
    md_bc = (b.outputs["OUT"].float() - c.outputs["OUT"].float()).abs().max().item()
    if md_ab < 1e-5 and md_bc > 1e-4:
        r.ok(f"spatial_sigma scaled (maxdiff {md_ab:.2e}), range_sigma untouched by scale "
             f"(halving it by hand moves the output by {md_bc:.2e})")
    else:
        r.fail("bilateral pixel_args selectivity",
               f"scaled-vs-manual-half maxdiff={md_ab:.2e} (want <1e-5); "
               f"manual range_sigma edit maxdiff={md_bc:.2e} (want >1e-4, sanity that it moves)")


def test_scale47b_scale_none_is_byte_identical(r: SubTestResult):
    print("\n--- SCALE-47b: scale=None cooks byte-identically to never passing scale= (invariant #7) ---")
    A = make_img(1, 10, 10, 4)
    code = "@OUT = gauss_blur(@A, 3.0);"
    a = tex_engine.cook(code, {"A": A.clone()}, device_mode="cpu")
    b = tex_engine.cook(code, {"A": A.clone()}, device_mode="cpu", scale=None)
    if torch.equal(a.outputs["OUT"], b.outputs["OUT"]):
        r.ok("cook(...) and cook(..., scale=None) are bit-identical")
    else:
        r.fail("invariant #7", "scale=None produced different pixels than omitting scale=")


def test_scale47b_forces_interpreter_tier_and_records_it(r: SubTestResult):
    print("\n--- SCALE-47b: a scale-active cook is routed to the interpreter, and it says so ---")
    A = make_img(1, 8, 8, 4)
    code = "@OUT = gauss_blur(@A, 2.0);"
    _tt.reset()
    plan = tex_engine.prepare(code, {"A": A}, device_mode="cpu",
                              compile_mode="torch_compile", scale=0.5)
    tex_engine.run(plan)
    rec = _tt.last()
    if rec is None or rec.tier != "interpreter":
        r.fail("scale tier bypass", f"expected tier='interpreter', got {rec!r}")
        return
    if not rec.reason or "scale" not in rec.reason.lower():
        r.fail("scale tier bypass reason", f"expected a reason naming scale, got {rec.reason!r}")
        return
    r.ok(f"compile_mode='torch_compile' + scale=0.5 recorded tier={rec.tier!r} "
         f"reason={rec.reason!r} -- never a silent fallback")
