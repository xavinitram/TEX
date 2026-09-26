"""SCALE-47b phase 9 — the R1 envelope: "same picture, downscaled", measured not promised.

AUTHOR DECISION #1 (SCALE-47-design.md §2): bit-exactness is not on offer (a Gaussian kernel
discretizes `ceil(3*sigma*scale)` to a different integer radius per resolution; morphology's
structuring element is likewise integer-radius). The promise is the same SHAPE as invariant
#9's cross-device envelope: a `scale=s` cook, upsampled, compared against a `scale=None` cook
of the SAME program, downsampled to the same size -- a maxdiff BAND, pinned per builtin
family, not a claim of equality. A regression past the pinned band is a loud decision to
re-measure and re-band, never a silently tightened or loosened tolerance.
"""
from helpers import *
from TEX_Wrangle import tex_engine
import torch.nn.functional as _F


def _checker(n, size=32):
    """A high-frequency test pattern (blur/morphology are near-identity on anything smooth)."""
    img = make_img(n, size, size, 4)
    y = torch.arange(size).view(size, 1)
    x = torch.arange(size).view(1, size)
    pat = ((x + y) % 8 < 4).float()
    img[..., 0] = pat
    img[..., 1] = pat
    img[..., 2] = pat
    img[..., 3] = 1.0
    return img


def _downsample(img, size):
    bchw = img.permute(0, 3, 1, 2)
    out = _F.interpolate(bchw, size=(size, size), mode="bilinear", align_corners=False)
    return out.permute(0, 2, 3, 1)


def _envelope_case(code: str, full_size=32, half_size=16, band=0.20):
    full_img = _checker(1, full_size)
    half_img = _downsample(full_img, half_size)

    full_out = tex_engine.cook(code, {"A": full_img}, device_mode="cpu").outputs["OUT"]
    full_down = _downsample(full_out, half_size)

    coarse_out = tex_engine.cook(code, {"A": half_img.clone()}, device_mode="cpu",
                                 scale=0.5).outputs["OUT"]

    md = (full_down.float() - coarse_out.float()).abs().max().item()
    return md, md <= band


def test_scale47b_r1_envelope_gauss_blur(r: SubTestResult):
    print("\n--- SCALE-47b R1 envelope: gauss_blur(8.0) ---")
    md, ok = _envelope_case("@OUT = gauss_blur(@A, 8.0);", band=0.10)
    if ok:
        r.ok(f"gauss_blur: coarse-upsampled vs full-downsampled maxdiff {md:.3e} (band 0.10)")
    else:
        r.fail("gauss_blur R1 envelope", f"maxdiff {md:.3e} exceeds the pinned 0.10 band")


def test_scale47b_r1_envelope_erode(r: SubTestResult):
    print("\n--- SCALE-47b R1 envelope: erode(4.0) ---")
    md, ok = _envelope_case("@OUT = erode(@A, 4.0);", band=0.05)
    if ok:
        r.ok(f"erode: coarse-upsampled vs full-downsampled maxdiff {md:.3e} (band 0.05)")
    else:
        r.fail("erode R1 envelope", f"maxdiff {md:.3e} exceeds the pinned 0.05 band")


def test_scale47b_r1_envelope_dilate(r: SubTestResult):
    print("\n--- SCALE-47b R1 envelope: dilate(4.0) ---")
    md, ok = _envelope_case("@OUT = dilate(@A, 4.0);", band=0.05)
    if ok:
        r.ok(f"dilate: coarse-upsampled vs full-downsampled maxdiff {md:.3e} (band 0.05)")
    else:
        r.fail("dilate R1 envelope", f"maxdiff {md:.3e} exceeds the pinned 0.05 band")


def test_scale47b_r1_envelope_bilateral_filter(r: SubTestResult):
    print("\n--- SCALE-47b R1 envelope: bilateral_filter(6.0, 0.2) ---")
    md, ok = _envelope_case("@OUT = bilateral_filter(@A, 6.0, 0.2);", band=0.08)
    if ok:
        r.ok(f"bilateral_filter: coarse-upsampled vs full-downsampled maxdiff {md:.3e} "
             f"(band 0.08)")
    else:
        r.fail("bilateral_filter R1 envelope", f"maxdiff {md:.3e} exceeds the pinned 0.08 band")


def test_scale47b_r1_envelope_scale_one_is_exact(r: SubTestResult):
    print("\n--- SCALE-47b: scale=1.0 is the degenerate, EXACT case of the envelope (sanity) ---")
    full_img = _checker(1, 16)
    a = tex_engine.cook("@OUT = gauss_blur(@A, 3.0);", {"A": full_img.clone()},
                        device_mode="cpu").outputs["OUT"]
    b = tex_engine.cook("@OUT = gauss_blur(@A, 3.0);", {"A": full_img.clone()},
                        device_mode="cpu", scale=1.0).outputs["OUT"]
    if not torch.equal(a, b):
        r.fail("scale=1.0 exactness", "scale=1.0 diverged from scale=None on the same canvas")
        return
    r.ok("scale=1.0 on the SAME canvas is bit-exact (the degenerate, promised-exact case)")
