"""TRK-166 — `tex_tiling._tile_plan`/`_halo_tile_plan` pick their spatial ANCHOR (the binding
`estimate_peak_bytes` sizes the peak-bytes estimate off) by scanning `bindings.values()` for the
first `dim() >= 3` tensor whose `shape[1] == H` — by VALUE, not by NAME. TRK-163/TRK-165 already
taught `shared_tile_height`/`shared_tile_width` (which derive `H`/`W` in the first place) to
exclude a registered non-spatial binding (a LUT) from the size scan, but the two anchor-selection
loops below that point were never taught the same exclusion: a non-spatial binding whose OWN
leading dim coincidentally equals the real image's H can still be picked as the anchor, handing
`estimate_peak_bytes` the LUT's own leading dim as "batch" instead of the real image's.

Advisory, unexploited by any shipped program (the tracker's own characterization): neither
planner slices a binding by shape match, only decides, so a wrong anchor can only mis-size a
memory/TDR estimate, never corrupt output. Both functions also refuse off any non-CUDA device
before reaching this code, so the fix is exercised here directly against the two loops (by
recording the `spatial` tuple each hands to `estimate_peak_bytes`, stubbed out so no real CUDA
query is needed) rather than through a device-gated end-to-end plan.

ComfyUI-invisible because: the fix only changes which binding is picked when a non-spatial
binding's own leading dim coincidentally collides with H — every ordinary image/mask/latent
graph (no non-spatial binding at all) computes exactly the same anchor as before.
"""
from helpers import *

from TEX_Wrangle import tex_memory, tex_tiling
from TEX_Wrangle.tex_runtime.interpreter import _non_spatial_names_cached
from failure_harness import compile_program

_LUT_CODE = "@OUT = vec4(apply_lut3d(@A.rgb, @LUT), 1.0);"                       # tile-safe
_HALO_CODE = "@OUT = vec4(apply_lut3d(gauss_blur(@A.rgb, 2.0), @LUT), 1.0);"     # halo-tileable


def _capture_spatial(monkey_target):
    """Replace `tex_memory.estimate_peak_bytes` with a recorder; returns (calls, restore)."""
    calls = []
    real = tex_memory.estimate_peak_bytes

    def _spy(program, spatial, dtype_bytes, fingerprint=None):
        calls.append(spatial)
        return 1024  # a fixed, cheap answer — the value never matters to this test

    tex_memory.estimate_peak_bytes = _spy

    def _restore():
        tex_memory.estimate_peak_bytes = real
    return calls, _restore


def test_trk166_tile_plan_anchor_by_name(r: SubTestResult):
    """`_tile_plan`'s anchor scan must skip a registered non-spatial binding even when it is
    the FIRST candidate in dict order and its own leading dim collides with H."""
    print("\n--- TRK-166: _tile_plan anchor selection is by name ---")
    img = make_img(4, 8, 8, 3, seed=11)          # real image: B=4, H=8, W=8
    lut = torch.rand(8, 8, 8, 3)                 # LUT's own leading dim (8) collides with H (8)
    bindings = {"LUT": lut, "A": img}             # LUT inserted FIRST — the order a buggy
                                                   # by-value scan would encounter it in
    try:
        prog, tm, outs = compile_program(_LUT_CODE, bindings)
        non_spatial = _non_spatial_names_cached(prog)
        assert non_spatial == {"LUT"}, f"expected {{'LUT'}}, got {non_spatial}"
        calls, restore = _capture_spatial(tex_tiling)
        try:
            tex_tiling._tile_plan(prog, bindings, "cuda:0", 0, 4, "trk166_test",
                                  free_hint=None, code=_LUT_CODE, binding_types=None)
        finally:
            restore()
        assert calls, "estimate_peak_bytes was never called — the anchor scan did not run"
        spatial = calls[0]
        assert spatial == (4, 8, 8), (
            f"anchor picked {spatial}, expected the real image's (4, 8, 8) — "
            f"a by-value scan would have picked the LUT and read (8, 8, 8)")
        r.ok(f"_tile_plan anchor = {spatial} (the real image, not the colliding LUT)")
    except Exception as e:
        r.fail("TRK-166 _tile_plan anchor", f"{type(e).__name__}: {e}")


def test_trk166_halo_tile_plan_anchor_by_name(r: SubTestResult):
    """`_halo_tile_plan`'s `batch` lookup (the `next(...)` over `bindings.values()`) must skip
    a registered non-spatial binding the same way, when it is first in dict order."""
    print("\n--- TRK-166: _halo_tile_plan anchor selection is by name ---")
    img = make_img(4, 16, 16, 3, seed=12)         # real image: B=4, H=16, W=16
    lut = torch.rand(16, 16, 16, 3)               # LUT's own leading dim (16) collides with H
    bindings = {"LUT": lut, "A": img}
    try:
        prog, tm, outs = compile_program(_HALO_CODE, bindings)
        non_spatial = _non_spatial_names_cached(prog)
        assert non_spatial == {"LUT"}, f"expected {{'LUT'}}, got {non_spatial}"
        calls, restore = _capture_spatial(tex_tiling)
        try:
            tex_tiling._halo_tile_plan(prog, _HALO_CODE, bindings, "cuda:0", 0, 4,
                                       "trk166_halo_test", None, "fp32", None)
        finally:
            restore()
        assert calls, "estimate_peak_bytes was never called — the anchor lookup did not run"
        spatial = calls[0]
        assert spatial == (4, 16, 16), (
            f"anchor picked {spatial}, expected the real image's (4, 16, 16) — "
            f"a by-value scan would have picked the LUT and read (16, 16, 16)")
        r.ok(f"_halo_tile_plan anchor = {spatial} (the real image, not the colliding LUT)")
    except Exception as e:
        r.fail("TRK-166 _halo_tile_plan anchor", f"{type(e).__name__}: {e}")


def test_trk166_control_no_non_spatial_binding_unchanged(r: SubTestResult):
    """Control: a program with no non-spatial binding at all picks the same (only) anchor as
    before — the fix must not change the ordinary case."""
    print("\n--- TRK-166: control, no non-spatial binding ---")
    img = make_img(2, 8, 8, 3, seed=13)
    bindings = {"A": img}
    code = "@OUT = vec4(@A.rgb, 1.0);"
    try:
        prog, tm, outs = compile_program(code, bindings)
        calls, restore = _capture_spatial(tex_tiling)
        try:
            tex_tiling._tile_plan(prog, bindings, "cuda:0", 0, 4, "trk166_ctrl",
                                  free_hint=None, code=code, binding_types=None)
        finally:
            restore()
        assert calls and calls[0] == (2, 8, 8), f"expected (2, 8, 8), got {calls}"
        r.ok(f"ordinary single-binding cook still anchors on {calls[0]}")
    except Exception as e:
        r.fail("TRK-166 control", f"{type(e).__name__}: {e}")
