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


# ── FIX-SCALE S6: the envelope beyond 1/2 ────────────────────────────────────
# B2#6: the bands above were pinned ONLY at scale=0.5 on a 32x32 checker. The Bible's own
# ladder is {1/2, 1/4, 1/8} (SCALE-47-design.md (c), R5), and the approximation genuinely
# gets worse at the coarser rungs -- this is the integer-radius/kernel discretization the
# design doc already names, not a canvas-size artifact (measured near-identical at 256x256
# and 512x512; only the size=256 rung is pinned here to keep the suite fast). Measured on
# BOTH the adversarial 8-pixel-period checker above AND a realistic "smooth gradient plus a
# few hard-edged rectangles" image -- checker is not always the worse of the two (erode's
# 1/8 divergence is WORSE on the smooth+edges image, 0.43-0.48, than on the checker, 0.25).
#
# erode/dilate's radius is truncated to an int downstream (`stdlib_sample._morph`), unlike
# gauss_blur's continuous sigma (whose own `ceil` computes a kernel radius from it further
# down, `stdlib_core.py`'s `radius = int(math.ceil(3.0 * sigma))`) -- at scale=0.125, radius
# 4.0 truncates to a scaled radius of 0 (a total no-op) rather than gauss_blur's never-zero
# ceil. Measured whether switching `_morph` to ceil (matching gauss_blur's own convention)
# actually helps: it does NOT reliably -- on the checker pattern the maxdiff is IDENTICAL
# either way (0.25/0.75), and on the smooth+edges image ceil is worse for one family and
# better for the other (erode: 0.4761 trunc vs 0.5263 ceil; dilate: 0.5263 trunc vs 0.4761
# ceil, at 256^2). A radius=1 morphology pass and a radius=0 no-op are simply two different
# discretizations of a true radius of 0.5, neither closer to it in general -- not a rounding
# BUG with a fix, the same integer-structuring-element discretization limit the R1 promise
# already names, just far more visible at this rung. Left as `int()` (unchanged, so the
# scale=None/1.0 default path is untouched either way -- invariant #7); erode/dilate are
# documented as not recommended below scale=1/4 in docs/resolution-scale.md instead of
# pinning an unusable (>0.5, more than half the value range) band at 1/8.

_R1_FULL_SIZE = 256


def _smooth_edges(n, size):
    """A realistic comp-like pattern: a smooth low-frequency gradient (blur/morphology are
    near-identity on this alone) plus a few hard-edged rectangles -- unlike `_checker`
    above, most of the image is smooth; only isolated boundaries are sharp."""
    img = make_img(n, size, size, 4)
    y = torch.linspace(0, 1, size).view(size, 1).expand(size, size)
    x = torch.linspace(0, 1, size).view(1, size).expand(size, size)
    grad = 0.5 + 0.5 * torch.sin(x * 3.14159) * torch.cos(y * 3.14159)
    img[..., 0] = grad
    img[..., 1] = grad
    img[..., 2] = grad
    for (y0f, y1f, x0f, x1f, v) in [(0.1, 0.3, 0.1, 0.4, 1.0),
                                    (0.6, 0.9, 0.55, 0.95, 0.0),
                                    (0.35, 0.5, 0.6, 0.8, 0.2)]:
        y0, y1 = int(y0f * size), int(y1f * size)
        x0, x1 = int(x0f * size), int(x1f * size)
        img[0, y0:y1, x0:x1, 0:3] = v
    img[..., 3] = 1.0
    return img


def _envelope_case_scaled(code: str, pattern_fn, scale: float, full_size=_R1_FULL_SIZE):
    """The same measurement `_envelope_case` makes, generalized to an arbitrary rung and
    pattern generator instead of a hardcoded scale=0.5/32x32 checker."""
    half_size = round(full_size * scale)
    full_img = pattern_fn(1, full_size)
    half_img = _downsample(full_img, half_size)
    full_out = tex_engine.cook(code, {"A": full_img.clone()}, device_mode="cpu").outputs["OUT"]
    full_down = _downsample(full_out, half_size)
    coarse_out = tex_engine.cook(code, {"A": half_img.clone()}, device_mode="cpu",
                                 scale=scale).outputs["OUT"]
    return (full_down.float() - coarse_out.float()).abs().max().item()


def _worst_of_both_patterns(code: str, scale: float) -> float:
    """The band this rung pins must cover BOTH the adversarial checker and a realistic
    smooth+edges image -- whichever measures worse, since (measured) neither pattern is
    reliably the worse one across every family."""
    return max(_envelope_case_scaled(code, _checker, scale),
               _envelope_case_scaled(code, _smooth_edges, scale))


def _assert_rung(r: SubTestResult, label: str, code: str, scale: float, band: float):
    md = _worst_of_both_patterns(code, scale)
    if md <= band:
        r.ok(f"{label} at scale={scale}: worst-of-both-patterns maxdiff {md:.4f} (band {band})")
    else:
        r.fail(f"{label} R1 envelope at scale={scale}",
               f"maxdiff {md:.4f} exceeds the pinned {band} band")


def test_fixscale47_s6_gauss_blur_quarter(r: SubTestResult):
    print("\n--- FIX-SCALE S6: gauss_blur(8.0) at scale=1/4 ---")
    _assert_rung(r, "gauss_blur", "@OUT = gauss_blur(@A, 8.0);", 0.25, band=0.20)


def test_fixscale47_s6_gauss_blur_eighth(r: SubTestResult):
    print("\n--- FIX-SCALE S6: gauss_blur(8.0) at scale=1/8 ---")
    _assert_rung(r, "gauss_blur", "@OUT = gauss_blur(@A, 8.0);", 0.125, band=0.40)


def test_fixscale47_s6_bilateral_filter_quarter(r: SubTestResult):
    print("\n--- FIX-SCALE S6: bilateral_filter(6.0, 0.2) at scale=1/4 ---")
    _assert_rung(r, "bilateral_filter", "@OUT = bilateral_filter(@A, 6.0, 0.2);", 0.25,
                band=0.06)


def test_fixscale47_s6_bilateral_filter_eighth(r: SubTestResult):
    print("\n--- FIX-SCALE S6: bilateral_filter(6.0, 0.2) at scale=1/8 ---")
    _assert_rung(r, "bilateral_filter", "@OUT = bilateral_filter(@A, 6.0, 0.2);", 0.125,
                band=0.05)


def test_fixscale47_s6_erode_quarter(r: SubTestResult):
    print("\n--- FIX-SCALE S6: erode(4.0) at scale=1/4 (its coarsest RECOMMENDED rung) ---")
    _assert_rung(r, "erode", "@OUT = erode(@A, 4.0);", 0.25, band=0.30)


def test_fixscale47_s6_dilate_quarter(r: SubTestResult):
    print("\n--- FIX-SCALE S6: dilate(4.0) at scale=1/4 (its coarsest RECOMMENDED rung) ---")
    _assert_rung(r, "dilate", "@OUT = dilate(@A, 4.0);", 0.25, band=0.30)


def test_fixscale47_s6_erode_dilate_eighth_is_documented_unusable(r: SubTestResult):
    print("\n--- FIX-SCALE S6: erode/dilate at scale=1/8 stay past ANY sane band (documented, not hidden) ---")
    # Not a "must stay small" assertion -- the opposite: this rung is DOCUMENTED as not
    # recommended (docs/resolution-scale.md), and this row exists so a future change that
    # makes it usable is noticed (an unexpectedly LOW maxdiff here is worth a second look,
    # not silently accepted) rather than the row just being absent.
    erode_md = _worst_of_both_patterns("@OUT = erode(@A, 4.0);", 0.125)
    dilate_md = _worst_of_both_patterns("@OUT = dilate(@A, 4.0);", 0.125)
    if erode_md < 0.30 and dilate_md < 0.30:
        r.fail("erode/dilate 1/8 premise",
               f"expected the documented-unusable rung to still measure worse than 0.30 "
               f"(erode={erode_md:.4f}, dilate={dilate_md:.4f}) -- if this now holds, "
               f"docs/resolution-scale.md's 'not below 1/4' guidance may be stale")
        return
    r.ok(f"erode(1/8)={erode_md:.4f}, dilate(1/8)={dilate_md:.4f}: both still past a sane "
         f"band, matching the documented guidance not to use them below scale=1/4")


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
