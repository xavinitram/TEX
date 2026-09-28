"""BILATX-51: a `scale=` cook of `bilateral_filter` stays on the exact tap-loop tier.

A host serves an interactive drag through the resolution-scale contract: a `scale=s` cook
scales `spatial_sigma` (arg 1) before the call, so the regime dispatch must see the SCALED
radius. spatial_sigma=10 (radius 30, exact) at scale=0.5 is spatial_sigma=5 (radius 15): still
the tap-loop exact tier, at about a sixteenth of the full-resolution cost, and still the
same picture downscaled within the pinned `bilateral_filter` band.
"""
from helpers import *
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib

_CODE = "@OUT = bilateral_filter(@A, 10.0, 0.2);"


def _plate(size):
    img = make_img(1, size, size, 4)
    y = torch.arange(size).view(size, 1).float()
    x = torch.arange(size).view(1, size).float()
    base = 0.5 + 0.4 * torch.sin(x / 7.0) * torch.cos(y / 11.0)
    edge = ((x > size // 3) & (y > size // 4)).float()
    for c in range(3):
        img[..., c] = base * (0.6 + 0.2 * c) + 0.8 * edge
    img[..., 3] = 1.0
    return img


def test_bilatx51_scaled_cook_dispatches_exact_taploop_at_scaled_radius(r: SubTestResult):
    print("\n--- BILATX-51: scale=0.5 of bilateral_filter(10.0) runs the tap-loop at radius 15 ---")
    seen = []
    original = TEXStdlib._bilateral_exact_taploop_bchw

    def spy(bchw, ss, sr, radius):
        seen.append(radius)
        return original(bchw, ss, sr, radius)

    TEXStdlib._bilateral_exact_taploop_bchw = staticmethod(spy)
    try:
        tex_engine.cook(_CODE, {"A": _plate(48)}, device_mode="cpu", scale=0.5)
        tex_engine.cook(_CODE, {"A": _plate(96)}, device_mode="cpu")
    finally:
        TEXStdlib._bilateral_exact_taploop_bchw = staticmethod(original)
    if seen != [15, 30]:
        r.fail("bilatx51 scaled dispatch", f"tap-loop radii seen {seen}, expected [15, 30]")
        return
    r.ok("scale=0.5 cook ran the exact tap-loop at radius 15; full cook at radius 30")


def test_bilatx51_scaled_cook_within_band(r: SubTestResult):
    print("\n--- BILATX-51: spatial_sigma=10 (now exact) scaled is the same picture, "
          "downscaled, within its band at every rung ---")
    import test_scale47b_r1_envelope as env
    for scale, band in ((0.5, 0.10), (0.25, 0.06), (0.125, 0.05)):
        md = env._worst_of_both_patterns(_CODE, scale)
        if md > band:
            r.fail(f"bilatx51 scaled band scale={scale}", f"maxdiff {md:.4f} exceeds band {band}")
            continue
        r.ok(f"scale={scale}: worst-of-both-patterns maxdiff {md:.4f} (band {band})")
