"""TRK-163 (COLOR-1-R1) — a `[N,N,N,3]` LUT beside an image whose H or B coincidentally
equals N is not recognised as heterogeneous by `tex_memory.shared_tile_height` /
`shared_batch_size`, so `run_tiled` / `run_batch_strips` could slice the LUT.

`_shared_dim_size` (backing both) built its shared-size candidate set from EVERY
dim-qualifying tensor binding, with no notion of which bindings are genuinely spatial.
When the ONLY other spatial-shaped binding is a LUT whose own leading dim (N) happens to
equal the real image's H (or B), the LUT's size does not add a second distinct
candidate — it collapses into the SAME one — so the "heterogeneous inputs can't be
co-strided" refusal never fires, `run_tiled`/`run_batch_strips` decide the axis IS shared,
and their own per-binding narrow loop then matches (and wrongly slices) the LUT by shape,
not by role. This is exactly the class `graphed._spatial_px` and `interpreter._consensus_extent`
already close via the SAME `non_spatial_args`-derived exclusion set
(`interpreter._non_spatial_names_cached`) — routed here too by this fix.

Fixed by threading that same exclusion set into `_shared_dim_size` (so a non-spatial
binding never enters the shared-size candidate set) AND into the per-strip narrow loops
of `run_tiled`/`run_batch_strips` (so a non-spatial binding is passed WHOLE even when its
own shape happens to match the decided H_total/B_total).

ComfyUI-invisible because: a program that never calls a `non_spatial_args`-declaring
stdlib function (e.g. `apply_lut3d`) gets an empty exclusion set from
`_non_spatial_names_cached`, so `_shared_dim_size`/`run_tiled`/`run_batch_strips` compute
exactly what they did before for every ordinary image/mask/latent graph — only a program
binding a non-spatial resource (a LUT) is affected, and it used to silently corrupt that
resource under tiling pressure; now it does not.
"""
from helpers import *

from TEX_Wrangle import tex_memory
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from failure_harness import compile_program, run_tier, max_diff, clone_bindings

from test_v040_phase1 import _identity_lut


def _baselines(code, bindings):
    """The unpressured cook, both tiers — the SAME reference the pre-existing COLOR-1
    pressure-path test uses ("both tiers" per the tracker row: interp and codegen must
    already agree — invariant #2 — and a tiled/batch-split cook must match either)."""
    return run_tier(code, bindings, "interp"), run_tier(code, bindings, "codegen")


def test_trk163_lut_height_collision_under_forced_tiling(r: SubTestResult):
    """A 33^3 LUT beside a B=2, H=33, W=16 image: H coincidentally equals the LUT's own
    N. Forced tiling (n=4) must equal the unpressured cook on both tiers — the LUT must
    never be sliced just because its shape[1] happens to match H_total."""
    print("\n--- TRK-163: a 33^3 LUT beside an H=33 image under forced tiling (run_tiled) ---")
    lut = _identity_lut(33)
    img = make_img(2, 33, 16, 3, seed=2)          # H == LUT's N == 33, by construction
    code = "@OUT = vec4(apply_lut3d(@A.rgb, @LUT), 1.0);"
    bindings = {"A": img, "LUT": lut}

    try:
        base_interp, base_codegen = _baselines(code, bindings)
    except Exception as e:
        r.fail("TRK-163 tiling baseline", f"{type(e).__name__}: {e}")
        return

    try:
        prog, tm, outs = compile_program(code, bindings)
        interp = Interpreter()
        tiled = tex_memory.run_tiled(interp, prog, clone_bindings(bindings), tm, "cpu", 0,
                                     outs, None, "fp32", 4)
        md_i = max_diff(tiled, base_interp)
        md_c = max_diff(tiled, base_codegen)
        assert md_i < 1e-5, f"run_tiled diverges from the interp baseline (maxdiff {md_i:.3e})"
        assert md_c < 1e-5, f"run_tiled diverges from the codegen baseline (maxdiff {md_c:.3e})"
        assert tiled["OUT"].shape == base_interp["OUT"].shape
        r.ok(f"run_tiled(n=4) with a 33^3 LUT beside a [2,33,16,3] image (H==N==33) matches "
             f"the unpressured cook on both tiers (interp maxdiff {md_i:.1e}, "
             f"codegen maxdiff {md_c:.1e})")
    except Exception as e:
        r.fail("TRK-163 tiling", f"{type(e).__name__}: {e}")

    # The decision itself, directly: with the LUT correctly excluded, the shared height
    # comes from the real image alone and must read exactly its own H (33) — never None
    # (over-refusing) and never corrupted by the LUT's coincidental match.
    try:
        from TEX_Wrangle.tex_runtime.interpreter import _non_spatial_names_cached
        non_spatial = _non_spatial_names_cached(prog)
        assert non_spatial == {"LUT"}, f"expected {{'LUT'}}, got {non_spatial}"
        h = tex_memory.shared_tile_height(bindings, non_spatial)
        assert h == 33, f"shared_tile_height with the LUT excluded should read 33, got {h}"
        r.ok(f"shared_tile_height(bindings, non_spatial={{'LUT'}}) reads the image's own "
             f"H ({h}), unperturbed by the LUT's coincidental N")
    except Exception as e:
        r.fail("TRK-163 tiling decision", f"{type(e).__name__}: {e}")


def test_trk163_lut_batch_collision_under_forced_batch_strips(r: SubTestResult):
    """A 33^3 LUT beside a B=33, H=8, W=8 image: B coincidentally equals the LUT's own N.
    Forced batch-strips (n=4) must equal the unpressured cook on both tiers — the LUT
    must never be sliced just because its own leading dim happens to match B_total."""
    print("\n--- TRK-163: a 33^3 LUT beside a B=33 image under forced batch-strips "
          "(run_batch_strips) ---")
    lut = _identity_lut(33)
    img = make_img(33, 8, 8, 3, seed=3)           # B == LUT's N == 33, by construction
    code = "@OUT = vec4(apply_lut3d(@A.rgb, @LUT), 1.0);"
    bindings = {"A": img, "LUT": lut}

    try:
        base_interp, base_codegen = _baselines(code, bindings)
    except Exception as e:
        r.fail("TRK-163 batch-strip baseline", f"{type(e).__name__}: {e}")
        return

    try:
        prog, tm, outs = compile_program(code, bindings)
        interp = Interpreter()
        bstrips = tex_memory.run_batch_strips(interp, prog, clone_bindings(bindings), tm,
                                              "cpu", 0, outs, None, "fp32", 4)
        md_i = max_diff(bstrips, base_interp)
        md_c = max_diff(bstrips, base_codegen)
        assert md_i < 1e-5, f"run_batch_strips diverges from the interp baseline (maxdiff {md_i:.3e})"
        assert md_c < 1e-5, f"run_batch_strips diverges from the codegen baseline (maxdiff {md_c:.3e})"
        assert bstrips["OUT"].shape == base_interp["OUT"].shape
        r.ok(f"run_batch_strips(n=4) with a 33^3 LUT beside a [33,8,8,3] image (B==N==33) "
             f"matches the unpressured cook on both tiers (interp maxdiff {md_i:.1e}, "
             f"codegen maxdiff {md_c:.1e})")
    except Exception as e:
        r.fail("TRK-163 batch-strip", f"{type(e).__name__}: {e}")

    try:
        from TEX_Wrangle.tex_runtime.interpreter import _non_spatial_names_cached
        non_spatial = _non_spatial_names_cached(prog)
        b = tex_memory.shared_batch_size(bindings, non_spatial)
        assert b == 33, f"shared_batch_size with the LUT excluded should read 33, got {b}"
        r.ok(f"shared_batch_size(bindings, non_spatial={{'LUT'}}) reads the image's own "
             f"B ({b}), unperturbed by the LUT's coincidental N")
    except Exception as e:
        r.fail("TRK-163 batch-strip decision", f"{type(e).__name__}: {e}")


def test_trk163_bare_decision_unaffected_without_non_spatial(r: SubTestResult):
    """Control: the pre-existing bare-call contract (no `non_spatial` argument) is
    unchanged — every existing caller that does not know about non-spatial bindings
    (or a program that binds none) gets exactly the old answer. This is what keeps
    `test_color1_apply_lut3d_pressure_paths`'s own bare assertions
    (`shared_tile_height(bindings) is None` for a genuinely mismatched LUT) valid
    unmodified."""
    print("\n--- TRK-163 control: shared_tile_height/shared_batch_size default to the "
          "pre-existing (non_spatial-blind) behaviour ---")
    lut = _identity_lut(33)
    img_h = make_img(2, 33, 16, 3, seed=4)
    img_b = make_img(33, 8, 8, 3, seed=5)
    # Bare calls (no non_spatial) reproduce the OLD collapse: the coincidence still
    # reads as "shared" because nothing excludes the LUT — this is the exact hazard the
    # fix's non_spatial argument closes, preserved here as the default so no existing
    # bare caller changes behaviour.
    h = tex_memory.shared_tile_height({"A": img_h, "LUT": lut})
    b = tex_memory.shared_batch_size({"A": img_b, "LUT": lut})
    r.ok(f"bare shared_tile_height (no non_spatial) still reads {h} — unchanged default") \
        if h == 33 else r.fail("TRK-163 control height", f"expected 33, got {h}")
    r.ok(f"bare shared_batch_size (no non_spatial) still reads {b} — unchanged default") \
        if b == 33 else r.fail("TRK-163 control batch", f"expected 33, got {b}")

    # A genuinely heterogeneous pair (real disagreement, not a LUT coincidence) must
    # still decline regardless of non_spatial — the ordinary "two images of different
    # sizes" refusal this fix must not weaken.
    img2 = make_img(1, 64, 64, 3, seed=6)
    h2 = tex_memory.shared_tile_height({"A": img_h, "B": img2})
    r.ok("two genuinely different-height images still decline (None)") if h2 is None \
        else r.fail("TRK-163 control heterogeneous", f"expected None, got {h2}")
