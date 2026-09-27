"""FIX-PACE P4 (Phase C, B1/R4 blind spots) — heaviness must be REGISTRY-DERIVED for cost
too, not just for footprint, and `sample_mip`'s warm cache-hit path must get an entry poll.

Two confirmed blind spots (B1's "findings not chased further", R4's "confirmed, not
hypothetical" counter-examples):

1. `fbm`/`ridged`/`billow`/`turbulence` (multi-octave, runtime-variable cost) and
   `worley_f1`/`worley_f2`/`worley_id` (per-pixel cellular search) carry footprint='point'
   (the default) -- `heavy_builtin_names()` walks only the footprint tag, so a 32-octave
   `fbm` call below the 512x512 large-resolution threshold is classified neither heavy nor
   large-resolution and falls straight into stride economization the way PACE-47c's fix
   was built to bypass.

2. `sample_mip`/`sample_mip_gauss` are deliberately excluded from footprint-derived heavy
   (footprint='image', multi-pass, PACE-47c's poll sits INSIDE `_build_mip_pyramid`'s
   per-level loop) -- correct on a COLD build, but `_build_mip_pyramid` returns immediately
   on a warm cache hit, before that loop (and its poll) ever runs. A live "drag a lod/param
   on a static image" session hits the warm cache every tick and never touches the pacing
   pool at all through this builtin.

RED at `3d39da2`: `heavy_builtin_names()` does not include the noise family at all
(footprint alone can't express "runtime-variable cost"), and a warm `fn_sample_mip` call
never calls `poll_cook_cancel`/`paced_check`. GREEN once a registry-derived cost tag covers
the noise family and `fn_sample_mip`/`fn_sample_mip_gauss` carry their own entry poll.
"""
import torch

from helpers import SubTestResult, TEXStdlib, make_img
from TEX_Wrangle.tex_runtime import pacing_heavy as _heavy
from TEX_Wrangle.tex_runtime import pacing as _pace
from TEX_Wrangle.tex_runtime import stdlib_core as _sc


_NOISE_HEAVY_NAMES = {
    "fbm", "ridged", "billow", "turbulence",
    "worley_f1", "worley_f2", "worley_id", "voronoi",
}


def test_octave_and_cellular_noise_builtins_are_registry_heavy(r: SubTestResult):
    print("\n--- FIX-PACE P4: fbm/ridged/billow/turbulence/worley_*/voronoi are "
          "registry-derived heavy, not just halo-footprint builtins ---")
    names = _heavy.heavy_builtin_names()
    missing = _NOISE_HEAVY_NAMES - names
    if missing:
        r.fail("P4 noise heaviness coverage",
               f"missing from heavy_builtin_names(): {missing}")
    else:
        r.ok(f"all of {sorted(_NOISE_HEAVY_NAMES)} are registry-derived heavy")


def test_halo_footprint_builtins_still_heavy_after_the_new_tag(r: SubTestResult):
    """The pre-existing footprint-derived coverage (PACE-47d) must not regress when a
    second, independent registry tag is added alongside it."""
    print("\n--- FIX-PACE P4: gauss_blur/erode/dilate/bilateral_filter remain heavy ---")
    names = _heavy.heavy_builtin_names()
    expected = {"gauss_blur", "erode", "dilate", "bilateral_filter"}
    missing = expected - names
    if missing:
        r.fail("P4 footprint-heaviness regression", f"missing: {missing}")
    else:
        r.ok(f"all of {sorted(expected)} remain heavy")


def test_sample_mip_warm_cache_hit_still_polls(r: SubTestResult):
    print("\n--- FIX-PACE P4: fn_sample_mip polls on a WARM mip-pyramid cache hit, not "
          "only on the cold build ---")
    img = make_img(1, 32, 32, 4, seed=21)

    class _Tok:
        def __init__(self):
            self.n = 0

        def check(self):
            self.n += 1

    tok = _Tok()
    heavy_flags = []
    real_paced_check = _pace.paced_check

    def _spy(token, device, heavy=False):
        heavy_flags.append(heavy)
        return real_paced_check(token, device, heavy=heavy)

    _pace.paced_check = _spy
    token = _sc.set_cook_grid((1, 32, 32), torch.float32, device="cpu", cancel=tok)
    try:
        u = torch.full((1, 32, 32), 0.5)
        v = torch.full((1, 32, 32), 0.5)
        TEXStdlib.fn_sample_mip(img, u, v, 1.0)   # cold: builds the pyramid
        cold_true = any(heavy_flags)
        heavy_flags.clear()
        TEXStdlib.fn_sample_mip(img, u, v, 1.0)   # warm: SAME tensor object -> cache hit
        warm_true = any(heavy_flags)
    finally:
        _sc.restore_cook_ctx(token)
        _pace.paced_check = real_paced_check

    if cold_true and warm_true:
        r.ok("both the cold build and the warm cache hit produced a heavy=True poll")
    else:
        r.fail("P4 sample_mip warm entry poll",
               f"cold call polled heavy={cold_true}, warm (cache-hit) call polled "
               f"heavy={warm_true} -- the warm path must poll too")
