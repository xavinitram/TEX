"""v0.40 COLOR-1 — colour becomes a language citizen.

Lane A: Rec.709 transfer (rec709_to_linear/linear_to_rec709) + ACEScg<->linear matrix
(acescg_to_linear/linear_to_acescg), beside the existing sRGB/OKLab pair in
`tex_runtime/stdlib_color.py`. Named per-space functions (author ruling #2), no
PyOpenColorIO import anywhere (ruling #1). Each new name is RESERVED (a minor breaking
change, per AGENTS.md's stdlib recipe) — see the E3011 rows below.

Lanes B/D append their own rows to this same phase file as they land.
"""
import os

from helpers import *
from failure_harness import run_tier, max_diff, assert_tier_equiv, compile_program
from TEX_Wrangle.tex_io import lut as lutio


def _out(code, bindings):
    return run_tier(code, bindings, "interp")["OUT"]


def _identity_lut(n=5):
    """An [n,n,n,3] identity LUT: grid[b_idx,g_idx,r_idx] = (r,g,b)/(n-1). Trilinear
    interpolation of a linear function is EXACT regardless of grid resolution, so this
    gives a float-precision-clean "apply_lut3d is a no-op" reference at any n."""
    idx = torch.linspace(0.0, 1.0, n)
    bb, gg, rr = torch.meshgrid(idx, idx, idx, indexing="ij")
    return torch.stack([rr, gg, bb], dim=-1)


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


def test_color1_apply_lut3d(r: SubTestResult):
    print("\n--- COLOR-1 lane B: apply_lut3d (trilinear 3D LUT lookup) ---")
    img = make_img(1, 8, 8, 3)
    lut = _identity_lut(5)

    # An identity LUT is a no-op (float-precision clean — see _identity_lut's docstring).
    try:
        out = _out("@OUT = vec4(apply_lut3d(@A.rgb, @LUT), 1.0);", {"A": img, "LUT": lut})
        md = (out[..., 0:3] - img).abs().max().item()
        assert md < 1e-5, f"identity LUT was not a no-op (maxdiff {md:.3e})"
        r.ok(f"identity LUT is a no-op (maxdiff {md:.1e})")
    except Exception as e:
        r.fail("COLOR-1 apply_lut3d identity", f"{type(e).__name__}: {e}")

    # vec4 alpha passes through unchanged.
    try:
        rgba = make_img(1, 8, 8, 4)
        o = _out("@OUT = apply_lut3d(@A, @LUT);", {"A": rgba, "LUT": lut})
        amd = (o[..., 3] - rgba[..., 3]).abs().max().item()
        assert amd < 1e-6, f"alpha not preserved ({amd})"
        r.ok("vec4 alpha passes through apply_lut3d unchanged")
    except Exception as e:
        r.fail("COLOR-1 apply_lut3d alpha", f"{type(e).__name__}: {e}")

    # CF-6 regression: binding a LUT shaped differently from the driving image must NOT
    # perturb the cook's (B,H,W) grid (the _consensus_extent fix this lane needed — see
    # interpreter.py's _collect_binding_reads / apply_lut3d's non_spatial_args=(1,)).
    # Before the fix this raised inside
    # codegen (a torch.stack shape mismatch) and silently fell back to the interpreter;
    # both tiers are checked here so a regression can't hide behind the fallback again.
    try:
        odd_lut = _identity_lut(4)          # N=4, deliberately != @A's batch/H/W (1/8/8)
        base = run_tier("@OUT = vec4(apply_lut3d(@A.rgb, @LUT), 1.0);",
                        {"A": img, "LUT": odd_lut}, "interp")
        assert base["OUT"].shape[:3] == img.shape[:3], \
            f"interp: cook grid corrupted by the LUT's own shape: {base['OUT'].shape}"
        got = run_tier("@OUT = vec4(apply_lut3d(@A.rgb, @LUT), 1.0);",
                       {"A": img, "LUT": odd_lut}, "codegen")
        assert got["OUT"].shape[:3] == img.shape[:3], \
            f"codegen: cook grid corrupted by the LUT's own shape: {got['OUT'].shape}"
        assert max_diff(base, got) < 1e-4, "interp/codegen diverge with a mismatched LUT shape"
        r.ok(f"a [4,4,4,3] LUT beside a [1,8,8,3] image leaves the cook grid at {img.shape[:3]} "
             f"on both tiers")
    except Exception as e:
        r.fail("COLOR-1 apply_lut3d consensus-extent", f"{type(e).__name__}: {e}")

    assert_tier_equiv(r, "apply_lut3d", "@OUT = vec4(apply_lut3d(@A.rgb, @LUT), 1.0);",
                      {"A": img, "LUT": lut}, tiers=("codegen",), tol=1e-5)


def test_color1_apply_lut3d_pressure_paths(r: SubTestResult):
    print("\n--- COLOR-1 lane B: apply_lut3d under tiling/batch-strip/graph-worthiness pressure ---")
    # A LUT whose N (33, a real .cube size) matches NEITHER the image's H/W (2160/3840) nor
    # its batch (2) — so a shape-collision would be visible immediately, not by luck.
    from TEX_Wrangle.tex_runtime.interpreter import Interpreter
    from TEX_Wrangle.tex_memory import run_tiled, run_batch_strips, shared_tile_height, shared_batch_size
    from TEX_Wrangle.tex_runtime.graphed import _spatial_px

    lut = _identity_lut(33)
    img = make_img(2, 2160, 3840, 3, seed=1)
    code = "@OUT = vec4(apply_lut3d(@A.rgb, @LUT), 1.0);"
    bindings = {"A": img, "LUT": lut}
    prog, tm, outs = compile_program(code, bindings)
    interp = Interpreter()
    baseline = interp.execute(prog, dict(bindings), tm, device="cpu",
                              output_names=outs, precision="fp32")

    # (1) M-4 tiling (tex_engine._tile_plan's executor, run_tiled): shared_tile_height must
    # see the 2160 vs 33 disagreement and DECLINE — never silently narrow the LUT's own N as
    # if it were image rows. Forcing n_strips=4 bypasses the VRAM-pressure gate directly.
    try:
        # The LUT's own N (33) disagrees with the image's H (2160): shared_tile_height must
        # DECLINE (return None) — same "heterogeneous inputs can't be co-tiled" refusal it
        # already gives two differently-sized images — never silently narrow the LUT's own
        # axis 1 as if it were image rows.
        assert shared_tile_height(bindings) is None, \
            "shared_tile_height saw past the LUT/image height mismatch instead of declining"
        tiled = run_tiled(interp, prog, dict(bindings), tm, "cpu", 0, outs, None, "fp32", 4)
        md = (tiled["OUT"] - baseline["OUT"]).abs().max().item()
        assert md < 1e-6, f"run_tiled diverged from the untiled cook (maxdiff {md:.3e})"
        assert tiled["OUT"].shape == baseline["OUT"].shape
        r.ok(f"run_tiled(n=4) with a 33^3 LUT beside a 2160x3840 image declines to tile "
             f"(shared_tile_height=None) and matches the untiled cook exactly "
             f"(maxdiff {md:.1e})")
    except Exception as e:
        r.fail("COLOR-1 apply_lut3d tiling", f"{type(e).__name__}: {e}")

    # (2) ROI-6 batch-strip executor (dormant — "no engine caller yet" per its own docstring,
    # but callable directly and worth pinning before it gets one): shared_batch_size must see
    # B=2 vs the LUT's 33 disagree and decline.
    try:
        # Same refusal on the batch axis: the image's B=2 disagrees with the LUT's own
        # leading dim (33), so shared_batch_size must decline too.
        assert shared_batch_size(bindings) is None, \
            "shared_batch_size saw past the LUT/image batch mismatch instead of declining"
        bstrips = run_batch_strips(interp, prog, dict(bindings), tm, "cpu", 0, outs, None,
                                   "fp32", 2)
        md = (bstrips["OUT"] - baseline["OUT"]).abs().max().item()
        assert md < 1e-6, f"run_batch_strips diverged from the untiled cook (maxdiff {md:.3e})"
        r.ok(f"run_batch_strips(n=2) declines (shared_batch_size=None) and matches the "
             f"untiled cook exactly (maxdiff {md:.1e})")
    except Exception as e:
        r.fail("COLOR-1 apply_lut3d batch-strip", f"{type(e).__name__}: {e}")

    # (3) roi_plan/stage_halo/batch_sliceable (tex_roi.py) take `code: str` only — no tensor
    # shapes ever reach them, so a LUT binding cannot perturb them. Confirmed by signature,
    # not by a runtime probe (there is nothing shape-shaped to probe).
    import inspect
    from TEX_Wrangle import tex_roi
    for fname in ("roi_plan", "stage_halo", "batch_sliceable"):
        params = list(inspect.signature(getattr(tex_roi, fname)).parameters)
        assert params[0] == "code", f"{fname}'s first parameter is {params[0]!r}, not 'code'"
    r.ok("roi_plan/stage_halo/batch_sliceable take source code only — no binding shapes, "
         "so no LUT-shape exposure is possible there")

    # (4) graph-capture worthiness (graphed._spatial_px): before this lane's fix, dict
    # ITERATION ORDER decided whether the gate saw the LUT's tiny N*N or the real frame's
    # H*W — a wrong (order-dependent) PERF decision, never a wrong pixel (the actual capture
    # buffers size from _consensus_extent, already LUT-safe). Pin both orders now agreeing.
    try:
        a_first = _spatial_px({"A": img, "LUT": lut}, prog)
        lut_first = _spatial_px({"LUT": lut, "A": img}, prog)
        assert a_first == lut_first == img.shape[1] * img.shape[2], \
            f"order-dependent: A-first={a_first}, LUT-first={lut_first}, expected {img.shape[1]*img.shape[2]}"
        r.ok(f"graphed._spatial_px reads the real frame's H*W ({a_first}) regardless of "
             f"binding dict order")
    except Exception as e:
        r.fail("COLOR-1 apply_lut3d graph worthiness", f"{type(e).__name__}: {e}")

    # (5) fingerprint / precision: no runtime probe needed — read, not measured. fingerprint()
    # keys on `binding_types` (TEXType, e.g. IMAGE-shaped VEC3), never on tensor shape, so a
    # LUT's size cannot perturb PROGRAM-fingerprint/graph-cache-key reuse (tex_cache.py:387).
    # No engine site casts a raw tensor BINDING to fp16 wholesale for precision="fp16"/"auto"
    # (checked: interpreter._PRECISION_DTYPES sets only the INTERNAL compute dtype;
    # compiled._contiguous_bindings casts only an anomalous INTEGER dim>=3 tensor to fp32,
    # never float->float16) — fn_apply_lut3d forces its OWN grid_sample compute to fp32
    # regardless, so a LUT's storage precision is exactly as protected as any other function's
    # own inputs, by the same mechanism every other Color function already relies on.
    r.ok("fingerprint keys on binding TYPES not shapes; no engine-wide fp16 binding cast "
         "exists for apply_lut3d's LUT argument to be caught by (read, not measured — see "
         "the hand-back's Premises-verified section for the exact lines checked)")


def test_color1_lut_io(r: SubTestResult):
    print("\n--- COLOR-1 lane B: tex_io.lut (.cube / .spi1d reader) ---")

    # A minimal, valid 2^3 .cube identity LUT round-trips into the documented [N,N,N,3]
    # axis order (R fastest-varying, then G, then B — grid[b_idx,g_idx,r_idx]).
    try:
        cube_text = (
            'TITLE "id2"\nLUT_3D_SIZE 2\n'
            "0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1.0 1.0 0.0\n"
            "0.0 0.0 1.0\n1.0 0.0 1.0\n0.0 1.0 1.0\n1.0 1.0 1.0\n"
        )
        c = lutio.read_cube(cube_text.encode("utf-8"))
        assert c.grid.shape == (2, 2, 2, 3), f"wrong grid shape: {c.grid.shape}"
        assert c.title == "id2", f"wrong title: {c.title!r}"
        assert c.grid[0, 0, 0].tolist() == [0.0, 0.0, 0.0]
        assert c.grid[1, 1, 1].tolist() == [1.0, 1.0, 1.0]
        assert c.grid[0, 1, 0].tolist() == [0.0, 1.0, 0.0]   # g_idx=1, r_idx=0, b_idx=0
        r.ok("read_cube: 2^3 identity LUT decodes to the documented axis order")
    except Exception as e:
        r.fail("COLOR-1 read_cube basic", f"{type(e).__name__}: {e}")

    # Error paths: LutError, never a raw ValueError/IndexError.
    try:
        bad_cases = [
            ("LUT_1D_SIZE 4\n0\n0.5\n0.5\n1\n", "1D table"),
            ("LUT_3D_SIZE 2\n0 0 0\n", "short row count"),
            ("LUT_3D_SIZE 2\nDOMAIN_MAX 2.0 2.0 2.0\n" + "0 0 0\n" * 8, "non-default domain"),
        ]
        for text, desc in bad_cases:
            try:
                lutio.read_cube(text.encode("utf-8"))
                raise AssertionError(f"{desc}: read_cube did not raise")
            except lutio.LutError:
                pass
        r.ok("read_cube raises LutError for a 1D table / short body / non-default domain")
    except Exception as e:
        r.fail("COLOR-1 read_cube errors", f"{type(e).__name__}: {e}")

    # .spi1d: Version 1, Components 1, a 4-entry ramp.
    try:
        spi_text = "Version 1\nFrom 0.0 1.0\nLength 4\nComponents 1\n{\n0.0\n0.25\n0.75\n1.0\n}\n"
        s = lutio.read_spi1d(spi_text.encode("utf-8"))
        assert s.values.tolist() == [0.0, 0.25, 0.75, 1.0], f"wrong values: {s.values.tolist()}"
        assert s.domain == (0.0, 1.0), f"wrong domain: {s.domain}"
        r.ok("read_spi1d: a 4-entry 1-component ramp decodes correctly")
    except Exception as e:
        r.fail("COLOR-1 read_spi1d basic", f"{type(e).__name__}: {e}")

    # read_cube/read_spi1d also accept a real path (not just in-memory bytes).
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "id.cube")
            with open(p, "w", encoding="utf-8") as f:
                f.write("LUT_3D_SIZE 2\n" + "\n".join(
                    f"{r_} {g_} {b_}" for b_ in (0.0, 1.0) for g_ in (0.0, 1.0)
                    for r_ in (0.0, 1.0)) + "\n")
            c = lutio.read_cube(p)
            assert c.grid.shape == (2, 2, 2, 3)
        r.ok("read_cube accepts a real file path")
    except Exception as e:
        r.fail("COLOR-1 read_cube path", f"{type(e).__name__}: {e}")


def test_color1_colorspaces_extended(r: SubTestResult):
    print("\n--- COLOR-1 lane D: COLORSPACES grows alongside lane A's functions ---")
    from TEX_Wrangle.tex_marshalling import COLORSPACES, BufferMeta
    from TEX_Wrangle import tex_packing

    # The tag vocabulary grew by exactly the two spaces lane A's new functions produce/
    # consume (PREC-2's rule: never a speculative tag with nothing to set it).
    try:
        assert set(COLORSPACES) == {"srgb", "linear", "oklab", "rec709", "acescg", "unknown"}, \
            f"unexpected COLORSPACES: {COLORSPACES}"
        BufferMeta(colorspace="rec709")     # validates without raising
        BufferMeta(colorspace="acescg")
        try:
            BufferMeta(colorspace="aces2065-1")
            raise AssertionError("an unrelated ACES space was silently accepted")
        except ValueError:
            pass
        r.ok("COLORSPACES == {srgb, linear, oklab, rec709, acescg, unknown}; both new "
             "tags validate, an unrelated one still doesn't")
    except Exception as e:
        r.fail("COLOR-1 COLORSPACES", f"{type(e).__name__}: {e}")

    # The boundary this lane must NOT cross: choose_storage's kind-based colour/data split
    # is unaffected by a colorspace tag — it isn't even a parameter, so there's nothing to
    # pass. A colorspace tag stays advisory (DATA-1's "tags only, never transforms" rule).
    try:
        t = torch.rand(1, 4, 4, 3, dtype=torch.float32)
        same = (tex_packing.choose_storage(t, quality=tex_packing.PREVIEW, kind="IMAGE")
                == tex_packing.choose_storage(t, quality=tex_packing.PREVIEW, kind="IMAGE"))
        assert same, "choose_storage isn't even pure across identical calls"
        import inspect
        assert "colorspace" not in inspect.signature(tex_packing.choose_storage).parameters, \
            "choose_storage grew a colorspace parameter — DATA-1's tag must stay advisory-only"
        r.ok("choose_storage takes no colorspace parameter (kind alone decides eligibility)")
    except Exception as e:
        r.fail("COLOR-1 choose_storage boundary", f"{type(e).__name__}: {e}")


def test_color1_reserved_names_e3011(r: SubTestResult):
    print("\n--- COLOR-1: the five new names are RESERVED (E3011) ---")
    for name in ("rec709_to_linear", "linear_to_rec709", "acescg_to_linear", "linear_to_acescg",
                 "apply_lut3d"):
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
