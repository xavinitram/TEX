"""GAUSSPYR-50 (v0.50, RADIUS-50a D2) — `gauss_blur`'s engine policy for large sigma.

No new language argument: `gauss_blur(img, sigma)`'s signature and registry entry are
unchanged. `_gauss_blur_auto` (`tex_runtime/stdlib_core.py`) dispatches purely on the
already-resolved sigma value:

  - sigma <= GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA: today's exact separable conv
    (`_gauss_blur_bchw`), called UNCONDITIONALLY -- bit-identical to every release
    before this file existed. Proven below with `torch.equal` (not a tolerance) across
    a sigma sweep from 0 up to the threshold, CPU and CUDA.
  - sigma > GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA: `_gauss_blur_pyramid_approx`, an
    O(image size) downscale-pyramid approximation whose cost does not grow with sigma
    (a huge-sigma exact conv is "correct but unusable" -- RADIUS-50a-design.md
    measured single-call costs into the seconds at 4k past this range).

Both constants were picked by measurement (a fuzzer sweep over a checker and a
smooth-gradient-plus-hard-edged-rectangles corpus, at 1080p and 4k, CPU) -- see
`docs/resolution-scale.md`'s "gauss_blur past the exact threshold" section for the
measured bands this file's accuracy rows pin.
"""
from __future__ import annotations

import math

import torch

from helpers import *
from TEX_Wrangle.tex_runtime import stdlib_core as _sc
from TEX_Wrangle.tex_runtime.stdlib_core import (
    GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA as THRESHOLD,
    GAUSS_BLUR_PYRAMID_QUALITY_CAP as QUALITY_CAP,
    _gauss_blur_auto,
    _gauss_blur_bchw,
    _gauss_blur_pyramid_approx,
)

# Measured worst-case bands (this file's own fuzzer sweep, box (avg_pool2d) cascading
# pyramid, quality_cap=8.0, sigma up to 2000, 512x512 CPU corpora) with headroom for a
# different image size/shape than the sweep used. A regression past either band is a
# loud decision to re-measure and re-band (the same R1 promise `docs/resolution-scale.md`
# already makes for `scale=`), never a silently loosened tolerance.
_SMOOTH_EDGES_BAND = 0.16
_CHECKER_BAND = 0.17


def _checker(h, w, period=8):
    yy = torch.arange(h).view(h, 1)
    xx = torch.arange(w).view(1, w)
    val = (((yy // period) + (xx // period)) % 2).float()
    return val.view(1, h, w, 1).expand(1, h, w, 3).contiguous()


def _smooth_edges(h, w):
    yy = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
    xx = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    img = (0.5 * yy + 0.5 * xx).clone()
    img[h // 8: h // 4, w // 8: w // 3] = 1.0
    img[h // 2: h // 2 + h // 10, w // 2: w // 2 + w // 6] = 0.0
    img[h * 3 // 4: h * 3 // 4 + h // 12, w * 3 // 4: w * 3 // 4 + w // 8] = 1.0
    return img.clamp(0, 1).view(1, h, w, 1).expand(1, h, w, 3).contiguous()


# ── D2: bit-identical below/at the threshold ───────────────────────────────

_SWEEP_SIGMAS = (0.0, 0.1, 0.29, 0.3, 1.0, 2.5, 8.0, 32.0, 64.0, 128.0, 200.0, 255.99, 256.0)


def _run_bitexact_sweep(r: SubTestResult, device: str):
    torch.manual_seed(7)
    img = torch.rand(1, 6, 6, 3, device=device)
    for sigma in _SWEEP_SIGMAS:
        assert sigma <= THRESHOLD, "sweep row must stay at/below the threshold by construction"
        auto_out = _gauss_blur_auto(img, sigma)
        exact_out = _gauss_blur_bchw(img, sigma)
        if not torch.equal(auto_out, exact_out):
            r.fail(f"gausspyr50 bitexact sigma={sigma} device={device}",
                   "torch.equal() False -- the engine policy changed a below-threshold result")
            return
    r.ok(f"_gauss_blur_auto is torch.equal() to today's _gauss_blur_bchw for every sigma in "
         f"{_SWEEP_SIGMAS} (device={device}) -- exact path unchanged below the threshold")


def test_gausspyr50_bitexact_below_threshold_cpu(r: SubTestResult):
    print("\n--- GAUSSPYR-50: bit-identical sweep, sigma in [0, threshold], CPU ---")
    _run_bitexact_sweep(r, "cpu")


def test_gausspyr50_bitexact_below_threshold_cuda(r: SubTestResult):
    print("\n--- GAUSSPYR-50: bit-identical sweep, sigma in [0, threshold], CUDA ---")
    if not torch.cuda.is_available():
        r.skip("gausspyr50 bitexact cuda", "no CUDA device on this box")
        return
    _run_bitexact_sweep(r, "cuda")


def test_gausspyr50_fn_gauss_blur_end_to_end_below_threshold(r: SubTestResult):
    print("\n--- GAUSSPYR-50: fn_gauss_blur (the [B,H,W,C] entry point) unchanged below threshold ---")
    from TEX_Wrangle import tex_engine
    torch.manual_seed(11)
    img = make_img(1, 12, 12, 4)
    for sigma in (1.0, 16.0, 256.0):
        out = tex_engine.cook(f"@OUT = gauss_blur(@A, {sigma});", {"A": img.clone()},
                               device_mode="cpu").outputs["OUT"]
        bchw = img[..., :3].permute(0, 3, 1, 2)
        expect_rgb = _gauss_blur_bchw(bchw, sigma).permute(0, 2, 3, 1)
        if not torch.equal(out[..., :3], expect_rgb):
            r.fail(f"fn_gauss_blur e2e sigma={sigma}", "diverged from the exact conv below threshold")
            return
    r.ok("tex_engine.cook(gauss_blur(...)) matches the exact conv bit-for-bit at sigma<=threshold")


# ── D2: bounded above the threshold (accuracy band) ─────────────────────────

def test_gausspyr50_pyramid_accuracy_band(r: SubTestResult):
    print("\n--- GAUSSPYR-50: accuracy band above the threshold (checker + smooth+edges) ---")
    h = w = 256
    corpora = {"checker": _checker(h, w), "smooth+edges": _smooth_edges(h, w)}
    bands = {"checker": _CHECKER_BAND, "smooth+edges": _SMOOTH_EDGES_BAND}
    sigmas = (THRESHOLD + 1.0, 512.0, 1024.0, 2000.0)
    worst = {}
    for name, img in corpora.items():
        bchw = _sc._get_bchw(img)
        for sigma in sigmas:
            exact = _gauss_blur_bchw(bchw, sigma)
            approx = _gauss_blur_pyramid_approx(bchw, sigma)
            md = (exact - approx).abs().max().item()
            worst[name] = max(worst.get(name, 0.0), md)
            if md > bands[name]:
                r.fail(f"gausspyr50 accuracy band {name} sigma={sigma}",
                       f"maxdiff {md:.4f} exceeds the pinned {bands[name]} band")
                return
    r.ok(f"pyramid approximation stays within its pinned band for every measured sigma "
         f"(worst: checker={worst['checker']:.4f}<= {_CHECKER_BAND}, "
         f"smooth+edges={worst['smooth+edges']:.4f} <= {_SMOOTH_EDGES_BAND})")


def test_gausspyr50_pyramid_upsamples_to_input_size(r: SubTestResult):
    print("\n--- GAUSSPYR-50: pyramid output shape always matches the input, any sigma ---")
    img = torch.rand(1, 3, 17, 23)  # odd, non-power-of-2 -- exercises interpolate's own resize
    for sigma in (300.0, 4096.0, 1_000_000.0):
        out = _gauss_blur_pyramid_approx(img, sigma)
        if tuple(out.shape) != tuple(img.shape):
            r.fail(f"gausspyr50 shape sigma={sigma}", f"got {tuple(out.shape)}, expected {tuple(img.shape)}")
            return
    r.ok("pyramid output shape matches the [B,C,H,W] input exactly, including odd dimensions")


# ── D2: bounded time/memory at huge sigma (counts, not wall-clock) ──────────

def test_gausspyr50_huge_sigma_bounded_levels(r: SubTestResult):
    """A radius-2000-to-1e6 blur must not cost O(sigma) work. Asserted STRUCTURALLY,
    not by wall-clock (CI's coverage tracer can invert a wall-clock comparison,
    GATE-47/TRK-208):

    1. The number of halving (avg_pool2d) passes is bounded by the image size ALONE
       (`floor(log2(min(H,W)))`) -- never by sigma. This is the actual "not O(sigma)"
       claim: the same small bound holds whether sigma is 2000 or 1e6.
    2. On an image big enough for the cascade to reach `quality_cap` before the
       1-pixel floor (true for sigma=2000/8192 at the 4096x4096 size used below), the
       ONE real Gaussian blur the pyramid path performs always runs at a kernel-bounded
       sigma <= quality_cap -- the same tiny kernel regardless of the caller's sigma.
       For sigma so large relative to the image that the 1-pixel floor binds FIRST
       (sigma=1e6 here), the residual is bounded instead by `sigma / 2**levels_used`
       (a fixed, image-size-determined divisor) -- still a bound independent of any
       further growth in sigma, just a looser one; this row checks that weaker bound
       instead of a fixed constant.
    """
    print("\n--- GAUSSPYR-50: huge sigma stays bounded (level count + final kernel sigma) ---")
    pool_calls = {"n": 0}
    real_avg_pool2d = torch.nn.functional.avg_pool2d

    def _counting_avg_pool2d(*args, **kwargs):
        pool_calls["n"] += 1
        return real_avg_pool2d(*args, **kwargs)

    final_sigma_seen = {"v": None}
    real_gauss_blur_bchw = _sc._gauss_blur_bchw

    def _capturing_gauss_blur_bchw(img, sigma, downsample_2x=False):
        if not downsample_2x:
            final_sigma_seen["v"] = sigma
        return real_gauss_blur_bchw(img, sigma, downsample_2x=downsample_2x)

    h = w = 4096
    img = torch.rand(1, 1, h, w)  # single channel: this is a structural/counts probe, not accuracy
    max_levels = math.floor(math.log2(min(h, w))) + 1  # image-size-only bound, independent of sigma

    for sigma in (2000.0, 8192.0, 1_000_000.0):
        pool_calls["n"] = 0
        final_sigma_seen["v"] = None
        torch.nn.functional.avg_pool2d = _counting_avg_pool2d
        _sc._gauss_blur_bchw = _capturing_gauss_blur_bchw
        try:
            out = _gauss_blur_pyramid_approx(img, sigma)
        finally:
            torch.nn.functional.avg_pool2d = real_avg_pool2d
            _sc._gauss_blur_bchw = real_gauss_blur_bchw

        if pool_calls["n"] > max_levels:
            r.fail(f"gausspyr50 level bound sigma={sigma}",
                   f"{pool_calls['n']} halving passes, expected <= {max_levels} "
                   f"(bounded by image size alone, never by sigma)")
            return
        floor_bound = sigma / (2.0 ** pool_calls["n"])
        allowed = max(QUALITY_CAP, floor_bound) + 1e-6
        if final_sigma_seen["v"] is None or final_sigma_seen["v"] > allowed:
            r.fail(f"gausspyr50 final kernel bound sigma={sigma}",
                   f"the one real blur ran at sigma={final_sigma_seen['v']!r}, "
                   f"expected <= {allowed:.4f} (quality_cap, or the 1-pixel-floor bound "
                   f"sigma/2**{pool_calls['n']} when the image floor binds first)")
            return
        if not torch.isfinite(out).all():
            r.fail(f"gausspyr50 finite sigma={sigma}", "non-finite output")
            return
    r.ok(f"halving-pass count stays <= {max_levels} (image-size bound, never sigma-dependent) "
         f"and the one exact blur's kernel stays bounded, for sigma up to 1e6")


# ── Invariant 2: interp/codegen parity, both paths ──────────────────────────

def test_gausspyr50_codegen_parity_both_paths(r: SubTestResult):
    print("\n--- GAUSSPYR-50: interp==codegen for gauss_blur, below and above the threshold ---")
    torch.manual_seed(3)
    img = torch.rand(1, 4, 4, 3)
    cases = [
        ("exact path", "@OUT = gauss_blur(@A, 2.0);"),
        ("exact path, at the threshold", f"@OUT = gauss_blur(@A, {THRESHOLD});"),
        ("pyramid path", f"@OUT = gauss_blur(@A, {THRESHOLD + 50.0});"),
        ("pyramid path, huge sigma", "@OUT = gauss_blur(@A, 5000.0);"),
    ]
    for name, code in cases:
        assert_equiv(r, f"gausspyr50 {name}", code, {"A": img}, B=1, H=4, W=4)
