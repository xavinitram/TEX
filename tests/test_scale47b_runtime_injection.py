"""SCALE-47b phase 3 — runtime-scalar injection: the interpreter actually multiplies.

`SCALE-47-design.md` §2: scale multiplies the RESOLVED value of every `pixel_args=`-tagged
argument at the call site, as a runtime value (never an AST fold). Proven here on the
interpreter tier (the oracle, "oracle-first" per the design's phased plan): a `gauss_blur(@A,
sigma)` cook at `scale=0.5` must match a cook of the SAME program at `scale=None` with the
sigma argument pre-halved, within ordinary fp32 tolerance -- proving the multiplier is applied
to the right argument, not merely that *something* changed.

Also proves the tier-routing consequence, UPDATED by SCALECX-49 (v0.49, `SCALE-COMPILED-48`):
a scale-active cook now dispatches DIRECTLY to whichever tier `compile_mode` names, including
`torch_compile`/`auto`/`cuda_graph` (each keys its compiled artifact / captured graph by an
explicit `scale` component instead) -- it is no longer bounced to the interpreter
unconditionally the way it was before this ask. `scale=None` remains completely unaffected
(invariant #7 -- byte-identical to a call that never mentions scale at all).
"""
from helpers import *
from TEX_Wrangle import tex_engine
from TEX_Wrangle import tex_engine_tiers as _tiers


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


def test_scale47b_torch_compile_at_scale_matches_the_unscaled_equivalent(r: SubTestResult):
    """`compile_mode='torch_compile'` at scale 0.5 must give the picture of the same program
    with its pixel-space argument halved, cooked unscaled on the interpreter. On a box with no
    torch.compile backend the route self-declines to the interpreter, so the row then checks
    the scale rewrite alone; the compiled artifact's scale keying is
    `test_scalecx49_compiled_graphed_scale.py`'s."""
    print("\n--- SCALECX-49 update: compile_mode='torch_compile' + scale=0.5 dispatches to "
          "its own tier, not the interpreter unconditionally ---")
    A = make_img(1, 8, 8, 4)
    out = tex_engine.cook("@OUT = gauss_blur(@A, 4.0);", {"A": A.clone()}, device_mode="cpu",
                          compile_mode="torch_compile", scale=0.5)
    ref = tex_engine.cook("@OUT = gauss_blur(@A, 2.0);", {"A": A.clone()}, device_mode="cpu",
                          compile_mode="none")
    md = (out.outputs["OUT"].float() - ref.outputs["OUT"].float()).abs().max().item()
    if md < 1e-5:
        r.ok(f"compile_mode='torch_compile' + scale=0.5 cooks correctly (maxdiff {md:.2e}) "
             f"-- SCALECX-49 (v0.49)")
    else:
        r.fail("scale tier bypass", f"maxdiff {md:.2e} >= 1e-5")


def test_scale47b_scale_active_cook_reaches_the_cuda_graph_strategy(r: SubTestResult):
    """`_run_tier(ctx, 'cuda_graph')` with a scale-active ctx calls the cuda_graph strategy
    (it is not bounced to the interpreter). Its capture key carries `scale`, which
    `test_scalecx49_compiled_graphed_scale.py` owns."""
    print("\n--- SCALECX-49 update: a scale-active cook DOES reach the cuda_graph tier "
          "strategy now ---")
    A = make_img(1, 8, 8, 4)
    code = "@OUT = gauss_blur(@A, 2.0);"
    plan = tex_engine.prepare(code, {"A": A}, device_mode="cpu", scale=0.5)

    called = {"n": 0}

    def _spy(ctx):
        called["n"] += 1
        return {"OUT": ctx.bindings["A"]}

    saved = _tiers._TIER_METHOD["cuda_graph"]
    _tiers._TIER_METHOD["cuda_graph"] = _spy
    try:
        out = _tiers._run_tier(plan.ctx, "cuda_graph")
    finally:
        _tiers._TIER_METHOD["cuda_graph"] = saved
    if called["n"] != 1:
        r.fail("scale reaches cuda_graph", f"expected 1 call, got {called['n']}")
        return
    if "OUT" not in out:
        r.fail("scale bypass output", f"expected an 'OUT' key, got {list(out.keys())!r}")
        return
    r.ok("_run_tier(ctx, 'cuda_graph') with ctx.scale set now calls the cuda_graph "
         "strategy directly -- SCALECX-49 (v0.49)")
