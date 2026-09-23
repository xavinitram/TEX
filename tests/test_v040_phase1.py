"""v0.40 COLOR-1 — colour becomes a language citizen.

Lane A: Rec.709 transfer (rec709_to_linear/linear_to_rec709) + ACEScg<->linear matrix
(acescg_to_linear/linear_to_acescg), beside the existing sRGB/OKLab pair in
`tex_runtime/stdlib_color.py`. Named per-space functions (author ruling #2), no
PyOpenColorIO import anywhere (ruling #1). Each new name is RESERVED (a minor breaking
change, per AGENTS.md's stdlib recipe) — see the E3011 rows below.

Lanes B/D append their own rows to this same phase file as they land.
"""
from helpers import *
from failure_harness import run_tier, max_diff, assert_tier_equiv


def _out(code, bindings):
    return run_tier(code, bindings, "interp")["OUT"]


def test_color1_rec709_transfer(r: SubTestResult):
    print("\n--- COLOR-1 lane A: Rec.709 transfer (BT.709 EOTF/OETF) ---")
    img = make_img(1, 8, 8, 3)

    # Round-trip: linear_to_rec709(rec709_to_linear(x)) ~= x.
    try:
        rt = _out("@OUT = vec4(linear_to_rec709(rec709_to_linear(@A.rgb)), 1.0);", {"A": img})
        md = (rt[..., 0:3] - img).abs().max().item()
        assert md < 1e-4, f"Rec.709 round-trip drift {md:.3e}"
        r.ok(f"Rec.709 transfer round-trips (drift {md:.1e})")
    except Exception as e:
        r.fail("COLOR-1 rec709 round-trip", f"{type(e).__name__}: {e}")

    # Rec.709 is NOT sRGB: the two curves diverge at a known midpoint.
    try:
        v_srgb = _out("@OUT = vec4(srgb_to_linear(vec3(0.5)), 1.0);", {"A": img})[0, 0, 0, 0].item()
        v_709 = _out("@OUT = vec4(rec709_to_linear(vec3(0.5)), 1.0);", {"A": img})[0, 0, 0, 0].item()
        assert abs(v_srgb - v_709) > 1e-3, f"rec709_to_linear(0.5) collapsed onto sRGB's curve ({v_709} vs {v_srgb})"
        r.ok(f"rec709_to_linear(0.5)={v_709:.5f} distinct from srgb_to_linear(0.5)={v_srgb:.5f}")
    except Exception as e:
        r.fail("COLOR-1 rec709 vs srgb", f"{type(e).__name__}: {e}")

    # vec4 alpha passes through unchanged.
    try:
        rgba = make_img(1, 8, 8, 4)
        o = _out("@OUT = rec709_to_linear(@A);", {"A": rgba})
        amd = (o[..., 3] - rgba[..., 3]).abs().max().item()
        assert amd < 1e-6, f"alpha not preserved ({amd})"
        r.ok("vec4 alpha passes through Rec.709 transfer unchanged")
    except Exception as e:
        r.fail("COLOR-1 rec709 alpha", f"{type(e).__name__}: {e}")

    # Codegen parity (invariant #2): the generic `_fns[name]` dispatch calls the SAME
    # callable the interpreter uses, so this is a bit-exactness check, not a tolerance one.
    assert_tier_equiv(r, "rec709_to_linear", "@OUT = vec4(rec709_to_linear(@A.rgb), 1.0);",
                      {"A": img}, tiers=("codegen",), tol=1e-5)
    assert_tier_equiv(r, "linear_to_rec709", "@OUT = vec4(linear_to_rec709(@A.rgb), 1.0);",
                      {"A": img}, tiers=("codegen",), tol=1e-5)


def test_color1_acescg_matrix(r: SubTestResult):
    print("\n--- COLOR-1 lane A: ACEScg<->linear-Rec.709 3x3 matrix ---")
    img = make_img(1, 8, 8, 3)

    # Round-trip: linear_to_acescg(acescg_to_linear(x)) ~= x (the inverse is the EXACT
    # matrix inverse of the forward one, so this should be float-precision clean).
    try:
        rt = _out("@OUT = vec4(linear_to_acescg(acescg_to_linear(@A.rgb)), 1.0);", {"A": img})
        md = (rt[..., 0:3] - img).abs().max().item()
        assert md < 1e-4, f"ACEScg round-trip drift {md:.3e}"
        r.ok(f"ACEScg<->linear round-trips (drift {md:.1e})")
    except Exception as e:
        r.fail("COLOR-1 acescg round-trip", f"{type(e).__name__}: {e}")

    # Known value: a neutral (r=g=b) input stays neutral through a primary-only change
    # of basis with no white-point mismatch in the row sums (rows sum to ~1.0).
    try:
        w = _out("@OUT = vec4(acescg_to_linear(vec3(0.5)), 1.0);", {"A": img})[0, 0, 0]
        assert (w[:3] - 0.5).abs().max().item() < 1e-3, f"acescg_to_linear(grey)={w[:3].tolist()}"
        r.ok("acescg_to_linear(0.5,0.5,0.5) stays neutral (rows sum to ~1.0)")
    except Exception as e:
        r.fail("COLOR-1 acescg neutral", f"{type(e).__name__}: {e}")

    # vec4 alpha passes through unchanged.
    try:
        rgba = make_img(1, 8, 8, 4)
        o = _out("@OUT = acescg_to_linear(@A);", {"A": rgba})
        amd = (o[..., 3] - rgba[..., 3]).abs().max().item()
        assert amd < 1e-6, f"alpha not preserved ({amd})"
        r.ok("vec4 alpha passes through ACEScg matrix unchanged")
    except Exception as e:
        r.fail("COLOR-1 acescg alpha", f"{type(e).__name__}: {e}")

    assert_tier_equiv(r, "acescg_to_linear", "@OUT = vec4(acescg_to_linear(@A.rgb), 1.0);",
                      {"A": img}, tiers=("codegen",), tol=1e-5)
    assert_tier_equiv(r, "linear_to_acescg", "@OUT = vec4(linear_to_acescg(@A.rgb), 1.0);",
                      {"A": img}, tiers=("codegen",), tol=1e-5)


def test_color1_reserved_names_e3011(r: SubTestResult):
    print("\n--- COLOR-1 lane A: the four new names are RESERVED (E3011) ---")
    for name in ("rec709_to_linear", "linear_to_rec709", "acescg_to_linear", "linear_to_acescg"):
        try:
            raised = None
            try:
                check_code(f"float {name}(float x){{ return x; }}\n@OUT = vec4(0.0);")
            except Exception as e:
                raised = e
            assert raised is not None, f"redefining {name} as a user function did not raise"
            code = getattr(getattr(raised, "diagnostic", None), "code", None)
            assert code == "E3011", f"{name}: wrong error code: {code!r} (raised={raised!r})"
            r.ok(f"`{name}` is refused as E3011 (reserved builtin)")
        except Exception as e:
            r.fail(f"COLOR-1 reserved {name}", f"{type(e).__name__}: {e}")
