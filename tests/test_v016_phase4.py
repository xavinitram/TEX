"""
v0.16 Phase 4 — language / DX / stdlib regression tests.

SL-3 color management (sRGB<->linear, OKLab), SL-1 premult compositing,
SL-2 blend modes, SL-4 morphology.
"""
from helpers import *
from failure_harness import run_tier, max_diff, assert_tier_equiv


def _out(code, bindings):
    return run_tier(code, bindings, "interp")["OUT"]


def test_sl3_color_management(r: SubTestResult):
    print("\n--- SL-3: color management (sRGB<->linear, OKLab) ---")
    img = make_img(1, 8, 8, 3)

    # sRGB round-trip: linear_to_srgb(srgb_to_linear(x)) ~= x.
    try:
        rt = _out("@OUT = vec4(linear_to_srgb(srgb_to_linear(@A.rgb)), 1.0);", {"A": img})
        md = (rt[..., 0:3] - img).abs().max().item()
        assert md < 1e-4, f"sRGB round-trip drift {md:.3e}"
        # Known value: srgb_to_linear(0.5) = 0.21404.
        v = _out("@OUT = vec4(srgb_to_linear(vec3(0.5)), 1.0);", {"A": img})[0, 0, 0, 0].item()
        assert abs(v - 0.21404) < 1e-3, f"srgb_to_linear(0.5)={v}"
        r.ok(f"sRGB<->linear round-trips (drift {md:.1e}) + known value 0.214")
    except Exception as e:
        r.fail("SL-3 sRGB", f"{type(e).__name__}: {e}")

    # OKLab round-trip: oklab_to_rgb(oklab_from_rgb(x)) ~= x.
    try:
        rt = _out("@OUT = vec4(oklab_to_rgb(oklab_from_rgb(@A.rgb)), 1.0);", {"A": img})
        md = (rt[..., 0:3] - img).abs().max().item()
        assert md < 1e-4, f"OKLab round-trip drift {md:.3e}"
        # Known value: oklab_from_rgb(linear white) = (L~1, a~0, b~0).
        w = _out("@OUT = vec4(oklab_from_rgb(vec3(1.0)), 1.0);", {"A": img})[0, 0, 0]
        assert abs(w[0].item() - 1.0) < 2e-3 and abs(w[1].item()) < 2e-3 and abs(w[2].item()) < 2e-3, \
            f"OKLab(white)={w[:3].tolist()}"
        r.ok(f"OKLab round-trips (drift {md:.1e}) + white -> (1,0,0)")
    except Exception as e:
        r.fail("SL-3 OKLab", f"{type(e).__name__}: {e}")

    # vec4 alpha passes through unchanged.
    try:
        rgba = make_img(1, 8, 8, 4)
        o = _out("@OUT = srgb_to_linear(@A);", {"A": rgba})
        amd = (o[..., 3] - rgba[..., 3]).abs().max().item()
        assert amd < 1e-6, f"alpha not preserved ({amd})"
        r.ok("vec4 alpha passes through color-management unchanged")
    except Exception as e:
        r.fail("SL-3 alpha", f"{type(e).__name__}: {e}")

    # Codegen parity.
    assert_tier_equiv(r, "srgb_to_linear", "@OUT = vec4(srgb_to_linear(@A.rgb), 1.0);",
                      {"A": img}, tiers=("codegen",), tol=1e-5)
    assert_tier_equiv(r, "oklab_from_rgb", "@OUT = vec4(oklab_from_rgb(@A.rgb), 1.0);",
                      {"A": img}, tiers=("codegen",), tol=1e-5)


def _px(t):
    return t[0, 0, 0].tolist()


def test_sl1_compositing(r: SubTestResult):
    print("\n--- SL-1: premultiplied-alpha compositing ---")
    img = make_img(1, 8, 8, 4)
    try:
        # Opaque fg over bg -> fg.
        o = _px(_out("@OUT = over(vec4(1.0,0.0,0.0,1.0), vec4(0.0,1.0,0.0,1.0));", {"A": img}))
        assert max(abs(o[i] - v) for i, v in enumerate([1, 0, 0, 1])) < 1e-5, f"opaque over: {o}"
        # Transparent fg over bg -> bg.
        o = _px(_out("@OUT = over(vec4(1.0,0.0,0.0,0.0), vec4(0.0,1.0,0.0,1.0));", {"A": img}))
        assert max(abs(o[i] - v) for i, v in enumerate([0, 1, 0, 1])) < 1e-5, f"transparent over: {o}"
        # Half-alpha fg over opaque bg -> 0.5*fg + 0.5*bg.
        o = _px(_out("@OUT = over(vec4(1.0,1.0,1.0,0.5), vec4(0.0,0.0,0.0,1.0));", {"A": img}))
        assert max(abs(o[i] - v) for i, v in enumerate([0.5, 0.5, 0.5, 1.0])) < 1e-5, f"half over: {o}"
        r.ok("over: opaque/transparent/half-alpha all correct")
    except Exception as e:
        r.fail("SL-1 over", f"{type(e).__name__}: {e}")

    try:
        # premultiply then unpremultiply round-trips.
        o = _px(_out("@OUT = unpremultiply(premultiply(vec4(1.0,0.5,0.25,0.5)));", {"A": img}))
        assert max(abs(o[i] - v) for i, v in enumerate([1.0, 0.5, 0.25, 0.5])) < 1e-4, f"premult rt: {o}"
        # under(a,b) == over(b,a).
        u = _px(_out("@OUT = under(vec4(1.0,0.0,0.0,0.5), vec4(0.0,1.0,0.0,1.0));", {"A": img}))
        ov = _px(_out("@OUT = over(vec4(0.0,1.0,0.0,1.0), vec4(1.0,0.0,0.0,0.5));", {"A": img}))
        assert max(abs(u[i] - ov[i]) for i in range(4)) < 1e-6, "under != over(swapped)"
        r.ok("premultiply round-trip + under==over(swap)")
    except Exception as e:
        r.fail("SL-1 premult/under", f"{type(e).__name__}: {e}")

    assert_tier_equiv(r, "over", "@OUT = over(@A, @B);",
                      {"A": img, "B": make_img(1, 8, 8, 4, seed=9)}, tiers=("codegen",), tol=1e-5)


def test_sl2_blend_modes(r: SubTestResult):
    print("\n--- SL-2: blend modes ---")
    img = make_img(1, 8, 8, 3)
    checks = {
        "screen(vec3(0.5), vec3(0.5))": 0.75,
        "overlay(vec3(0.25), vec3(0.5))": 0.25,
        "hard_light(vec3(0.5), vec3(0.25))": 0.25,
        "color_dodge(vec3(0.5), vec3(0.5))": 1.0,
        "linear_light(vec3(0.5), vec3(0.5))": 0.5,
    }
    try:
        for expr, want in checks.items():
            got = _out(f"@OUT = vec4({expr}, 1.0);", {"A": img})[0, 0, 0, 0].item()
            assert abs(got - want) < 1e-4, f"{expr} = {got}, want {want}"
        r.ok(f"blend-mode known values correct ({len(checks)} modes)")
    except Exception as e:
        r.fail("SL-2 values", f"{type(e).__name__}: {e}")

    # Full family runs + codegen parity.
    for op in ("screen", "overlay", "soft_light", "color_burn", "vivid_light"):
        assert_tier_equiv(r, op, f"@OUT = vec4({op}(@A.rgb, @B.rgb), 1.0);",
                          {"A": img, "B": make_img(1, 8, 8, 3, seed=5)}, tiers=("codegen",), tol=1e-5)


def test_sl2_fp16_divide_guard(r: SubTestResult):
    print("\n--- SL-2 fix: fp16 divide guard (no NaN on black/white) ---")
    # SAFE_EPSILON=1e-8 underflows to 0.0 in fp16, so the divide guard no-ops and
    # the divide blends NaN on a black-base / white-blend pixel. The dtype-aware
    # ZERO_GUARD_EPS floor (fp16's smallest normal) fixes it; fp32 keeps 1e-8.
    from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib
    triggers = [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.5, 0.999)]
    try:
        for name in ("fn_color_dodge", "fn_color_burn", "fn_vivid_light"):
            fn = getattr(TEXStdlib, name)
            for av, bv in triggers:
                a = torch.full((1, 2, 2, 3), av, dtype=torch.float16)
                b = torch.full((1, 2, 2, 3), bv, dtype=torch.float16)
                out = fn(a, b)
                assert torch.isfinite(out).all(), f"{name}({av},{bv}) fp16 -> NaN/inf"
        r.ok("color_dodge/color_burn/vivid_light finite under fp16 (was NaN)")
    except Exception as e:
        r.fail("SL-2 fp16 divide guard", f"{type(e).__name__}: {e}")

    # Same divide-guard class in SL-1 compositing: over() divides by the output
    # alpha and unpremultiply() by alpha — both NaN'd under fp16 on a fully
    # transparent pixel (a NORMAL compositing input). Now routed through _safe_div.
    try:
        zf16 = torch.zeros(1, 2, 2, 4, dtype=torch.float16)          # transparent RGBA
        assert torch.isfinite(TEXStdlib.fn_over(zf16, zf16)).all(), "over fp16 NaN"
        assert torch.isfinite(TEXStdlib.fn_unpremultiply(zf16)).all(), "unpremultiply fp16 NaN"
        r.ok("over/unpremultiply finite under fp16 on transparent pixels")
    except Exception as e:
        r.fail("SL-1 fp16 divide guard", f"{type(e).__name__}: {e}")

    # SIGN-PRESERVING floor: a NEGATIVE denominator must keep its sign. The first
    # _safe_div used denom.clamp(min=eps), which raised a small negative up to +eps
    # → sign flip + magnitude blow-up (over/unpremultiply gave +9e7 for an
    # out-of-[0,1] / negative alpha on the DEFAULT fp32 path). Regression guard.
    try:
        D = TEXStdlib._safe_div
        for num, den in [(1.0, 0.5), (1.0, -0.5), (1.0, 1e-12), (1.0, -1e-12), (1.0, -1.0)]:
            got = D(torch.tensor([[num]]), torch.tensor([[den]])).item()
            assert torch.isfinite(torch.tensor(got)) and ((got > 0) == (den > 0)), \
                f"_safe_div sign flip: {num}/{den} -> {got}"
        # unpremultiply with a negative alpha stays sign-correct (was +9e7).
        c = torch.tensor([[[[0.9, 0.9, 0.9, -1e-4]]]])
        v = TEXStdlib.fn_unpremultiply(c)[0, 0, 0, 0].item()
        assert v < 0 and abs(v + 9000.0) < 1.0, f"unpremultiply(neg alpha) = {v}, want ~-9000"
        r.ok("_safe_div preserves denominator sign (over/unpremultiply neg-alpha correct)")
    except Exception as e:
        r.fail("safe_div sign preservation", f"{type(e).__name__}: {e}")

    # fp32 (default) is unaffected — ZERO_GUARD_EPS falls back to SAFE_EPSILON, so
    # results are finite and codegen-identical (parity elsewhere is the real guard).
    try:
        for name in ("fn_color_dodge", "fn_color_burn", "fn_vivid_light"):
            fn = getattr(TEXStdlib, name)
            for av, bv in triggers:
                out = fn(torch.full((1, 2, 2, 3), av), torch.full((1, 2, 2, 3), bv))
                assert torch.isfinite(out).all(), f"{name}({av},{bv}) fp32 -> NaN/inf"
        r.ok("fp32 path stays finite (default unaffected by the dtype-aware eps)")
    except Exception as e:
        r.fail("SL-2 fp32 regression", f"{type(e).__name__}: {e}")


def test_sl4_morphology(r: SubTestResult):
    print("\n--- SL-4: morphology (erode / dilate) ---")
    img = torch.zeros(1, 9, 9, 3)
    img[0, 3:6, 3:6, :] = 1.0   # a 3x3 bright center
    try:
        d = _out("@OUT = vec4(dilate(@A, 1), 1.0);", {"A": img})
        bright_d = int((d[0, :, :, 0] > 0.5).sum().item())
        assert bright_d == 25, f"dilate(1): {bright_d} bright px, want 25 (5x5)"
        e = _out("@OUT = vec4(erode(@A, 1), 1.0);", {"A": img})
        bright_e = int((e[0, :, :, 0] > 0.5).sum().item())
        assert bright_e == 1, f"erode(1): {bright_e} bright px, want 1"
        z = _out("@OUT = vec4(erode(@A, 0), 1.0);", {"A": img})
        assert (z[..., 0:3] - img).abs().max().item() < 1e-6, "radius 0 not identity"
        # dilate radius 2 grows the 3x3 to 7x7 = 49.
        d2 = _out("@OUT = vec4(dilate(@A, 2), 1.0);", {"A": img})
        assert int((d2[0, :, :, 0] > 0.5).sum().item()) == 49, "dilate(2) != 7x7"
        r.ok("erode/dilate: 3x3 -> erode 1px, dilate(1) 25px, dilate(2) 49px, r=0 identity")
    except Exception as e:
        r.fail("SL-4 morphology", f"{type(e).__name__}: {e}")

    # Works on a single-channel mask [B,H,W] too.
    try:
        mask = torch.zeros(1, 9, 9)
        mask[0, 4, 4] = 1.0
        dm = _out("@OUT = dilate(@A, 1);", {"A": mask})
        assert int((dm[0] > 0.5).sum().item()) == 9, "mask dilate(1) of 1px != 3x3"
        r.ok("erode/dilate operate on a [B,H,W] mask")
    except Exception as e:
        r.fail("SL-4 mask", f"{type(e).__name__}: {e}")

    assert_tier_equiv(r, "dilate", "@OUT = vec4(dilate(@A.rgb, 2), 1.0);",
                      {"A": make_img(1, 12, 12, 3)}, tiers=("codegen",), tol=1e-5)


def test_lx8_const_arrays(r: SubTestResult):
    print("\n--- LX-8: const arrays ---")
    img = make_img(1, 4, 4, 3)
    try:
        o = compile_and_run(
            "const float lut[3] = {1.0, 2.0, 3.0}; @OUT = vec4(lut[0], lut[1], lut[2], 1.0);",
            {"A": img})
        assert abs(o[0, 0, 0, 0].item() - 1.0) < 1e-6 and abs(o[0, 0, 0, 1].item() - 2.0) < 1e-6 \
            and abs(o[0, 0, 0, 2].item() - 3.0) < 1e-6, "float const LUT wrong"
        o2 = compile_and_run(
            "const vec3 pal[2] = {vec3(1.0,0.0,0.0), vec3(0.0,1.0,0.0)}; @OUT = vec4(pal[1], 1.0);",
            {"A": img})
        assert abs(o2[0, 0, 0, 1].item() - 1.0) < 1e-6, "vec3 const LUT wrong"
        r.ok("const arrays usable as LUTs (float + vec3), was a parser rejection")
    except Exception as e:
        r.fail("LX-8 usage", f"{type(e).__name__}: {e}")

    for code, why in [
        ("const float lut[3] = {1.0,2.0,3.0}; lut[0] = 9.0; @OUT = vec4(lut[0]);", "element write"),
        ("const float lut[3]; @OUT = vec4(lut[0]);", "no initializer"),
    ]:
        try:
            compile_and_run(code, {"A": img})
            r.fail(f"LX-8 reject {why}", "not rejected")
        except Exception:
            r.ok(f"const array {why} correctly rejected")

    # A plain (non-const) array still works and is writable (regression guard).
    try:
        o = compile_and_run("float a[3] = {1.0,2.0,3.0}; a[0] = 9.0; @OUT = vec4(a[0]);", {"A": img})
        assert abs(o[0, 0, 0, 0].item() - 9.0) < 1e-6
        r.ok("non-const arrays remain writable")
    except Exception as e:
        r.fail("LX-8 non-const regression", f"{type(e).__name__}: {e}")


def test_lx9_self_swizzle_write(r: SubTestResult):
    print("\n--- LX-9: self-referential swizzle-write is snapshot-safe ---")
    # `col.rg = col.gr` must snapshot the RHS before writing (result (2,1,3)),
    # NOT alias it (which would give the wrong (2,2,3)). Audited: the interpreter
    # is already safe; this pins it so a future refactor can't reintroduce aliasing.
    img = make_img(1, 2, 2, 3)
    cases = [
        ("vec3 c = vec3(1.0,2.0,3.0); c.rg = c.gr; @OUT = vec4(c, 1.0);", [2.0, 1.0, 3.0]),
        ("vec3 c = vec3(1.0,2.0,3.0); c.rgb = c.bgr; @OUT = vec4(c, 1.0);", [3.0, 2.0, 1.0]),
        ("vec4 c = vec4(1.0,2.0,3.0,4.0); c.rgba = c.abgr; @OUT = c;", [4.0, 3.0, 2.0, 1.0]),
    ]
    try:
        for code, want in cases:
            o = compile_and_run(code, {"A": img})
            got = [o[0, 0, 0, i].item() for i in range(len(want))]
            assert max(abs(got[i] - want[i]) for i in range(len(want))) < 1e-6, \
                f"self-swizzle aliased: {code} -> {got}, want {want}"
        r.ok(f"self-referential swizzle-writes snapshot correctly ({len(cases)} patterns)")
    except Exception as e:
        r.fail("LX-9 self-swizzle", f"{type(e).__name__}: {e}")
