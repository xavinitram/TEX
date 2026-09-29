"""TRK-165 — the same non-spatial blind spot TRK-163 closed for `shared_tile_height`/
`shared_batch_size` (a registered non-spatial binding, e.g. `apply_lut3d`'s LUT, whose own
leading dim can coincidentally collide with a real spatial size) also existed, structurally,
in `shared_tile_width` and in the two `tex_tiling.py` planners (`_tile_plan`, `_halo_tile_plan`)
that called `shared_tile_height` bare, with no `non_spatial` argument at all.

Unlike TRK-163, this was never exploitable into a wrong picture: `shared_tile_width`'s only
callers (`tex_memory.run_tiled_halo`, `tex_tiling._halo_tile_plan`) and `_tile_plan`/
`_halo_tile_plan` themselves only ever DECIDE off the shared size — none of them slices a
binding by shape match the way `run_tiled`/`run_batch_strips` did — so the failure mode here
is confined to the DECISION: a coincidental collision could make `_shared_dim_size` see two
distinct candidate sizes where there is really only one real spatial size, and over-refuse
(return None) a cook that could otherwise plan just fine, or skew which tensor a planner picks
as its spatial anchor for `estimate_peak_bytes`. Both are planning-quality issues, never a
wrong output.

Fixed by threading the same `non_spatial` exclusion (already backed by `_shared_dim_size`,
shared by every one of these callers) into `shared_tile_width`, `run_tiled_halo`'s two shared-
size calls, and `_tile_plan`/`_halo_tile_plan`'s calls to `shared_tile_height`/
`shared_tile_width` — the exact symmetry gap TRK-163 left open.

ComfyUI-invisible because: the default empty `non_spatial` set (a program binding no
non-spatial argument, i.e. every ordinary image/mask/latent graph) makes every touched
function compute exactly what it did before; only a program binding a non-spatial resource
(a LUT) beside a coincidentally-sized image is affected, and where it was affected before it
either read correctly by luck (the coincidence test below) or over-refused a plannable cook
(the mismatch test below) — never mis-sized nor mis-narrowed anything, so the fix only ever
LOOSENS an over-tight decision, it never changes a served pixel.
"""
from helpers import *

from TEX_Wrangle import tex_memory, tex_roi
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, _non_spatial_names_cached
from failure_harness import compile_program, clone_bindings, max_diff

_CUDA = torch.cuda.is_available()

_HALO_CODE = "@OUT = vec4(apply_lut3d(gauss_blur(@A.rgb, 2.0), @LUT), 1.0);"


def test_trk165_shared_tile_width_takes_non_spatial(r: SubTestResult):
    """`shared_tile_width` now carries the same `non_spatial` parameter
    `shared_tile_height`/`shared_batch_size` gained under TRK-163, with the same default-empty,
    unchanged-for-everyone-else contract."""
    print("\n--- TRK-165: shared_tile_width(bindings, non_spatial) ---")
    # (a) Coincidence: the LUT's own N equals the real image's W. Bare and excluded read the
    # SAME answer here (this is the benign half of the coincidence — no divergence to detect,
    # but the exclusion must not perturb a correct read).
    img = make_img(1, 16, 33, 3, seed=1)
    lut_coincident = torch.rand(33, 33, 33, 3)
    bindings = {"A": img, "LUT": lut_coincident}
    code = _HALO_CODE
    try:
        prog, tm, outs = compile_program(code, bindings)
        non_spatial = _non_spatial_names_cached(prog)
        assert non_spatial == {"LUT"}, f"expected {{'LUT'}}, got {non_spatial}"
        bare = tex_memory.shared_tile_width(bindings)
        excl = tex_memory.shared_tile_width(bindings, non_spatial)
        assert bare == 33 and excl == 33, f"bare={bare}, excluded={excl}, expected 33 both"
        r.ok(f"coincidental N==W ({bare}): bare and non_spatial-excluded reads agree")
    except Exception as e:
        r.fail("TRK-165 width coincidence", f"{type(e).__name__}: {e}")

    # (b) Genuine mismatch: the LUT's own N (20) differs from the real image's W (33). The
    # bare call sees two distinct dim-2 candidates (33 from @A, 20 from @LUT) and over-refuses
    # (None) a cook whose only REAL spatial binding is perfectly uniform. Excluding the LUT
    # restores the correct answer — this is the concrete decision-quality bug the symmetry
    # gap left open (never a wrong picture; at worst a cook that could plan declined to).
    lut_mismatch = torch.rand(20, 20, 20, 3)
    bindings2 = {"A": img, "LUT": lut_mismatch}
    try:
        prog2, tm2, outs2 = compile_program(code, bindings2)
        non_spatial2 = _non_spatial_names_cached(prog2)
        bare2 = tex_memory.shared_tile_width(bindings2)
        excl2 = tex_memory.shared_tile_width(bindings2, non_spatial2)
        assert bare2 is None, f"expected the bare call to over-refuse (None), got {bare2}"
        assert excl2 == 33, f"expected the excluded call to read the real W (33), got {excl2}"
        r.ok("a mismatched LUT (N=20) over-refuses shared_tile_width bare (None) but the "
             "non_spatial-excluded call correctly reads the real image's W (33)")
    except Exception as e:
        r.fail("TRK-165 width mismatch", f"{type(e).__name__}: {e}")

    # Control: a genuinely heterogeneous pair of REAL spatial bindings must still decline
    # regardless of non_spatial — this fix must not weaken the ordinary refusal.
    img2 = make_img(1, 16, 64, 3, seed=2)
    h = tex_memory.shared_tile_width({"A": img, "B": img2})
    r.ok("two genuinely different-width images still decline (None)") if h is None \
        else r.fail("TRK-165 control heterogeneous", f"expected None, got {h}")


def test_trk165_run_tiled_halo_matches_unpressured_with_width_collision(r: SubTestResult):
    """Forced halo-tiling (`run_tiled_halo`, n=2..4) of a blur+LUT program, with the LUT's own
    N coincidentally equal to the real image's W, must equal the unpressured cook on both
    tiers — `run_tiled_halo` now derives and passes `non_spatial` into its own
    `shared_tile_height`/`shared_tile_width` calls the same way `run_tiled` already does."""
    print("\n--- TRK-165: run_tiled_halo with a width-colliding LUT matches the unpressured "
          "cook (both tiers) ---")
    torch.manual_seed(7)
    img = torch.rand(1, 16, 33, 4)          # W == LUT's own N == 33
    lut = torch.rand(33, 33, 33, 3)
    bindings = {"A": img, "LUT": lut}
    code = _HALO_CODE
    try:
        base_interp = run_tier_ref(code, bindings, "interp")
        base_codegen = run_tier_ref(code, bindings, "codegen")
    except Exception as e:
        r.fail("TRK-165 halo baseline", f"{type(e).__name__}: {e}")
        return

    try:
        prog, tm, outs = compile_program(code, bindings)
        interp = Interpreter()
        plan = tex_roi.roi_plan(code, {})
        assert plan.executable and plan.halo > 0, "expected a halo-tileable plan"
        for n in (2, 3, 4):
            out = tex_memory.run_tiled_halo(interp, prog, clone_bindings(bindings), tm, "cpu",
                                            0, outs, None, "fp32", n, list(plan.narrow), plan.halo)
            md_i = max_diff(out, base_interp)
            md_c = max_diff(out, base_codegen)
            assert md_i < 1e-5, f"n_strips={n}: diverges from interp baseline (maxdiff {md_i:.3e})"
            assert md_c < 1e-5, f"n_strips={n}: diverges from codegen baseline (maxdiff {md_c:.3e})"
        r.ok("run_tiled_halo(n=2,3,4) with a width-colliding LUT matches the unpressured cook "
             "on both tiers")
    except Exception as e:
        r.fail("TRK-165 halo tiling", f"{type(e).__name__}: {e}")


def run_tier_ref(code, bindings, tier):
    """Local alias so this file reads standalone next to test_trk163's `_baselines`."""
    from failure_harness import run_tier
    return run_tier(code, bindings, tier)


def test_trk165_planners_take_non_spatial(r: SubTestResult):
    """`tex_tiling._tile_plan`/`_halo_tile_plan` derive and pass the `non_spatial` exclusion into
    their `shared_tile_height`/`shared_tile_width` calls. Without it a LUT whose own dims
    differ from the image's reads as a second candidate size and the planner refuses before it
    ever sizes the cook. The planners are gated on a `cuda` device NAME, so this passes
    "cuda:0" with CPU tensors and a stub for `estimate_peak_bytes` (the TRK-166 technique):
    reaching the stub proves the shared size resolved to the one real image size."""
    print("\n--- TRK-165: _tile_plan/_halo_tile_plan thread non_spatial through ---")
    from TEX_Wrangle import tex_tiling
    calls = []
    real = tex_memory.estimate_peak_bytes

    def _spy(program, spatial, dtype_bytes, fingerprint=None):
        calls.append(spatial)
        return 1024

    def _plan(fn, *args):
        calls.clear()
        tex_memory.estimate_peak_bytes = _spy
        try:
            fn(*args)
        finally:
            tex_memory.estimate_peak_bytes = real
        return list(calls)

    code = "@OUT = vec4(apply_lut3d(@A.rgb, @LUT), 1.0);"       # tile-safe (pointwise + LUT)
    bindings = {"LUT": torch.rand(5, 5, 5, 3), "A": make_img(2, 8, 12, 3, seed=3)}
    try:
        prog, tm, outs = compile_program(code, bindings)
        assert _non_spatial_names_cached(prog) == {"LUT"}
        # Control: without the exclusion the two leading dims (5 vs 8) disagree.
        assert tex_memory.shared_tile_height(bindings) is None
        got = _plan(tex_tiling._tile_plan, prog, bindings, "cuda:0", 0, 4, "trk165_tile",
                    None, code, None)
        assert got == [(2, 8, 12)], f"_tile_plan sized {got}, expected the real image's (2, 8, 12)"
        r.ok("_tile_plan excludes the mismatched LUT and sizes the cook off the real image")
    except Exception as e:
        r.fail("TRK-165 _tile_plan", f"{type(e).__name__}: {e}")

    bindings2 = {"LUT": torch.rand(5, 5, 5, 3), "A": make_img(2, 16, 24, 3, seed=4)}
    try:
        prog2, tm2, outs2 = compile_program(_HALO_CODE, bindings2)
        assert _non_spatial_names_cached(prog2) == {"LUT"}
        assert tex_memory.shared_tile_height(bindings2) is None
        assert tex_memory.shared_tile_width(bindings2) is None
        got2 = _plan(tex_tiling._halo_tile_plan, prog2, _HALO_CODE, bindings2, "cuda:0", 0, 4,
                     "trk165_halo", None, "fp32")
        assert got2 == [(2, 16, 24)], (
            f"_halo_tile_plan sized {got2}, expected the real image's (2, 16, 24)")
        r.ok("_halo_tile_plan excludes the mismatched LUT from both shared sizes")
    except Exception as e:
        r.fail("TRK-165 _halo_tile_plan", f"{type(e).__name__}: {e}")
