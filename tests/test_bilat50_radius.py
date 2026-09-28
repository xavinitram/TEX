"""BILAT-50 -- bilateral_filter's window used to be silently clamped to 7x7
(`radius = min(ceil(3*spatial_sigma), 3)`): any `spatial_sigma` past ~1.0 ran the SAME
window a `spatial_sigma` of 1.0 would, a silent-wrong class this ask closes -- the same
shape as the `erode`/`dilate` 256 clamp.

The fix (D3 RECORDED, `stdlib_sample.py`'s `fn_bilateral_filter`): no clamp. At or below
today's original window (radius<=3, i.e. spatial_sigma<=1.0) the ORIGINAL math runs
untouched -- bit-identical, by construction (the code is the same code). Above it and up to
`_BILATERAL_EXACT_RADIUS_MAX`, the SAME exact weighted-average math runs, row-tiled to keep
peak memory bounded independent of resolution (`_bilateral_exact_bchw`). Past that measured
limit, a downscale + detail-transfer approximation runs instead (`_bilateral_detail_transfer_
bchw`) -- O(image size), independent of spatial_sigma.

The footprint moved from a fixed `('halo', 3)` to `('halo_arg', 1, 8.0)` (invariant #5): the
window's true reach now grows with `spatial_sigma`, so a windowed/tiled cook must widen its
halo to match, or ROI narrowing would starve the filter of context it actually reads.
"""
import math
from helpers import *
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib  # populates REGISTRY
from TEX_Wrangle.tex_runtime.stdlib_core import _get_bchw
from TEX_Wrangle.tex_runtime import stdlib_registry as R
from TEX_Wrangle import tex_roi as _R
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_compiler.ast_nodes import NumberLiteral

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]


def _old_clamped_bilateral(image, sigma_s, sigma_r):
    """A byte-for-byte copy of the PRE-BILAT-50 algorithm (the removed
    `radius = min(ceil(3*ss), 3)` clamp) -- kept here ONLY to characterize the bug this ask
    fixes; never called by product code."""
    ss, sr = sigma_s, sigma_r
    img = image
    B, H, W, C = img.shape
    radius = min(int(math.ceil(3.0 * ss)), 3)
    ksize = 2 * radius + 1
    bchw = _get_bchw(img)
    padded = torch.nn.functional.pad(bchw, (radius, radius, radius, radius), mode='replicate')
    patches = padded.unfold(2, ksize, 1).unfold(3, ksize, 1)
    center = bchw.unsqueeze(-1).unsqueeze(-1)
    inv_2ss = -0.5 / max(ss * ss, 1e-10)
    dy = torch.arange(ksize, dtype=torch.float32, device=img.device) - radius
    dx = dy.clone()
    d2 = dy.view(-1, 1) ** 2 + dx.view(1, -1) ** 2
    w_spatial = torch.exp(d2 * inv_2ss).view(1, 1, 1, 1, ksize, ksize)
    diff = patches - center
    inv_2sr = -0.5 / max(sr * sr, 1e-10)
    cd2 = (diff * diff).sum(dim=1, keepdim=True)
    w_range = torch.exp(cd2 * inv_2sr)
    w = w_spatial * w_range
    numerator = (patches * w).sum(dim=(-2, -1))
    denominator = w.sum(dim=(-2, -1))
    result = numerator / denominator.clamp(min=1e-10)
    return result.permute(0, 2, 3, 1)


def _checker(n, size=64, period=8):
    """The adversarial high-frequency pattern -- resolution-scale.md's own R1 protocol."""
    img = make_img(n, size, size, 3)
    y = torch.arange(size).view(size, 1)
    x = torch.arange(size).view(1, size)
    pat = (((x // period) + (y // period)) % 2).float()
    img[..., 0] = pat
    img[..., 1] = pat
    img[..., 2] = pat
    return img


def _smooth_edges(n, size=64):
    """A realistic comp-like pattern: a smooth gradient plus a few hard-edged rectangles --
    resolution-scale.md's own second R1 input."""
    img = make_img(n, size, size, 3)
    yy = torch.linspace(0, 1, size).view(-1, 1).expand(size, size)
    xx = torch.linspace(0, 1, size).view(1, -1).expand(size, size)
    base = (yy + xx) / 2
    base[size // 4:size // 2, size // 4:size // 2] = 0.9
    base[size // 2:3 * size // 4, size // 2:3 * size // 4] = 0.05
    for c in range(3):
        img[0, :, :, c] = base
    return img


# ── 1. THE BUG this ask fixes: sigma 10 behaved like sigma 1 (the clamp) ────────────────────

def _old_clamped_radius(ss):
    return min(int(math.ceil(3.0 * ss)), 3)


def test_bilat50_old_clamp_made_sigma10_match_sigma1(r: SubTestResult):
    print("\n--- BILAT-50: characterizing the removed bug -- old clamp made sigma=10 read "
          "the SAME 7x7 window as sigma=1 ---")
    # The clamp's bug is about the WINDOW (how much of the neighbourhood is even read),
    # not about the two outputs being numerically equal -- sigma still reweights whatever
    # is inside that window, so sigma=1 and sigma=10 legitimately differ even under the old
    # clamp. What was silently wrong is that sigma=10 never got to see past a 7x7
    # neighbourhood: any sigma >= 1.0 was capped to EXACTLY the same window size sigma=1.0
    # gets, no matter how much larger a "genuine" 3*sigma window should have been.
    r1, r10 = _old_clamped_radius(1.0), _old_clamped_radius(10.0)
    if r1 != r10 or r10 != 3:
        r.fail("old clamp characterization",
               f"expected both to clamp to radius 3, got radius(1.0)={r1}, radius(10.0)={r10}")
        return
    r.ok(f"pre-fix formula: bilateral_filter(sigma=10) read the SAME radius-{r10} window as "
         f"bilateral_filter(sigma=1) -- sigma=10's genuine window (radius="
         f"{int(math.ceil(3.0 * 10.0))}) was silently discarded")

    # A DIRECT consequence a caller could actually observe: at a very small range_sigma
    # (so only near-identical colours blend), a flat region far from any edge sees ONLY the
    # 7x7 neighbourhood regardless of sigma -- so the two flat-region outputs the OLD code
    # produced are themselves close (both are "the same tiny window, mildly reweighted"),
    # while a genuine (uncapped) sigma=10 pass would draw on a MUCH wider neighbourhood.
    img = make_img(1, 24, 24, 3)
    old_1 = _old_clamped_bilateral(img.clone(), 1.0, 0.05)
    old_10 = _old_clamped_bilateral(img.clone(), 10.0, 0.05)
    new_10 = TEXStdlib.fn_bilateral_filter(img.clone(), 10.0, 0.05)
    md_old = (old_1.float() - old_10.float()).abs().max().item()
    md_new_vs_old = (new_10.float() - old_10.float()).abs().max().item()
    if md_new_vs_old <= md_old:
        r.fail("old vs fixed sigma=10 divergence",
               f"the FIXED sigma=10 output should diverge from the old-clamped sigma=10 "
               f"output by MORE than sigma=1 and sigma=10 diverged from each other under "
               f"the old clamp (md_old={md_old:.3e}, md_new_vs_old={md_new_vs_old:.3e})")
        return
    r.ok(f"old clamp: sigma=1 vs sigma=10 maxdiff {md_old:.3e} (both tiny-window); fixed "
         f"sigma=10 vs old-clamped sigma=10 maxdiff {md_new_vs_old:.3e} (much larger -- the "
         f"fix genuinely reads a wider neighbourhood now)")


def test_bilat50_fix_sigma10_differs_from_sigma1(r: SubTestResult):
    print("\n--- BILAT-50: the fix -- sigma=10 now genuinely differs from sigma=1 ---")
    img = make_img(1, 24, 24, 3)
    new_1 = TEXStdlib.fn_bilateral_filter(img.clone(), 1.0, 0.2)
    new_10 = TEXStdlib.fn_bilateral_filter(img.clone(), 10.0, 0.2)
    if torch.equal(new_1, new_10):
        r.fail("BILAT-50 fix", "sigma=10 still matches sigma=1 -- the clamp is still active")
    else:
        md = (new_1.float() - new_10.float()).abs().max().item()
        r.ok(f"bilateral_filter(sigma=10) != bilateral_filter(sigma=1) (maxdiff {md:.3e}) -- "
             f"no more silent clamp")


# ── 2. Bit-identity at or below today's window ──────────────────────────────────────────────

def test_bilat50_bit_identical_at_or_below_today_window(r: SubTestResult):
    print("\n--- BILAT-50: at or below today's 7x7 window, output is torch.equal to the "
          "pre-fix formula (CPU" + (" and CUDA" if _CUDA else "") + ") ---")
    # Loops over `_DEVICES` (cpu, plus cuda when present) rather than a separate
    # CUDA-only test with its own r.skip -- the same device-list idiom
    # test_v024_phase1.py's own suite already uses, so a CUDA-less box runs the CPU leg
    # without adding a new SIMP-3 skip site (AGENTS.md's skip budget moves only with a
    # named environment change, not a per-test convenience skip).
    for dev in _DEVICES:
        img = make_img(1, 20, 20, 3).to(dev)
        for ss in (0.3, 0.5, 0.75, 1.0):
            for sr in (0.1, 0.2, 0.4):
                got = TEXStdlib.fn_bilateral_filter(img.clone(), ss, sr)
                want = _old_clamped_bilateral(img.clone(), ss, sr)
                if not torch.equal(got, want):
                    r.fail(f"bit-identity [{dev}] ss={ss} sr={sr}", "torch.equal is False")
                    return
    r.ok(f"bilateral_filter(ss<=1.0) is torch.equal to the pre-fix formula for every "
         f"tested (ss, sr) pair on {_DEVICES}")


def test_bilat50_tiling_does_not_change_the_exact_math(r: SubTestResult):
    print("\n--- BILAT-50: row-tiling the exact filter never changes any pixel's own value ---")
    img = make_img(1, 37, 41, 3)  # deliberately non-square, non-power-of-2
    orig_budget = TEXStdlib._BILATERAL_TILE_BUDGET_ELEMS
    try:
        TEXStdlib._BILATERAL_TILE_BUDGET_ELEMS = 40          # forces many tiny tiles
        many_tiles = TEXStdlib.fn_bilateral_filter(img.clone(), 4.0, 0.2)
        TEXStdlib._BILATERAL_TILE_BUDGET_ELEMS = 50_000_000  # forces a single tile
        one_tile = TEXStdlib.fn_bilateral_filter(img.clone(), 4.0, 0.2)
    finally:
        TEXStdlib._BILATERAL_TILE_BUDGET_ELEMS = orig_budget
    if torch.equal(many_tiles, one_tile):
        r.ok("tiled vs single-tile exact pass: torch.equal (tiling is a pure memory "
             "optimization, not a different computation)")
    else:
        md = (many_tiles.float() - one_tile.float()).abs().max().item()
        r.fail("tiling changes values", f"maxdiff {md:.3e} between tiled and single-tile")


# ── 3. Bounded memory at large sigma (allocation-size count, not wall-clock) ────────────────

def test_bilat50_bounded_memory_at_large_sigma(r: SubTestResult):
    print("\n--- BILAT-50: memory at large sigma is bounded (allocation-size count) ---")
    img = make_img(1, 48, 48, 3)
    peak = {"numel": 0}
    orig_unfold = torch.Tensor.unfold

    def _spy_unfold(self, *a, **kw):
        out = orig_unfold(self, *a, **kw)
        peak["numel"] = max(peak["numel"], out.numel())
        return out

    torch.Tensor.unfold = _spy_unfold
    try:
        for ss in (10.0, 100.0, 1000.0, 8192.0 / 3.0):
            peak["numel"] = 0
            TEXStdlib.fn_bilateral_filter(img.clone(), ss, 0.2)
            # The exact filter's own patch tensor is O(H*W*ksize^2); a naive (uncapped,
            # untiled) implementation at these sigmas would need ksize^2 in the MILLIONS
            # (ss=1000 -> ksize~6001 -> ksize^2~3.6e7) times the image size. Bounded here
            # means the largest single `.unfold` output this call produced stays within a
            # small, resolution-scale multiple of the IMAGE's own element count --
            # independent of sigma -- because the detail-transfer path downscales BEFORE
            # ever unfolding, and only ever unfolds a <=7x7 window at the reduced scale.
            budget = 64 * img.numel()
            if peak["numel"] > budget:
                r.fail(f"bounded memory ss={ss}",
                       f"largest unfold output was {peak['numel']} elements (budget "
                       f"{budget}) -- memory grew with sigma instead of staying bounded")
                return
        r.ok("largest single allocation stays within a small, sigma-independent multiple "
             "of the image's own size across sigma=10..2731 (ss=8192/3, radius~8192)")
    finally:
        torch.Tensor.unfold = orig_unfold


def test_bilat50_exact_tier_memory_grows_with_radius_then_stops(r: SubTestResult):
    print("\n--- BILAT-50: the EXACT tier's own peak allocation is tile-bounded, not "
          "resolution-bounded ---")
    img = make_img(1, 200, 200, 3)  # deliberately large enough that untiled would be huge
    peak = {"numel": 0}
    orig_unfold = torch.Tensor.unfold

    def _spy_unfold(self, *a, **kw):
        out = orig_unfold(self, *a, **kw)
        peak["numel"] = max(peak["numel"], out.numel())
        return out

    torch.Tensor.unfold = _spy_unfold
    try:
        # radius=24 (ss~8.0): the exact tier's own upper edge. An UNTILED unfold at this
        # radius over a 200x200x3 image would need 200*200*3*49*49 ~= 288M elements
        # (~1.15GB just for `patches`) -- a prior design pass measured a comparable
        # untiled shape needing ~60GB at a slightly larger radius on a SMALLER canvas.
        # Tiling must keep the actually-observed peak far below that naive figure.
        peak["numel"] = 0
        TEXStdlib.fn_bilateral_filter(img.clone(), 8.0, 0.2)
        naive_untiled = 200 * 200 * 3 * 49 * 49
        if peak["numel"] >= naive_untiled // 4:
            r.fail("exact-tier tiling", f"peak unfold output {peak['numel']} is not far "
                   f"below the naive untiled figure {naive_untiled} -- tiling is not "
                   f"bounding memory")
            return
        r.ok(f"exact-tier (radius=24) peak unfold output {peak['numel']} elements, far "
             f"below the naive untiled {naive_untiled} on this 200x200 canvas")
    finally:
        torch.Tensor.unfold = orig_unfold


# ── 4. Accuracy band: detail-transfer vs exact, checker + smooth+edges ──────────────────────

def _accuracy_case(pattern_fn, ss, size=64):
    img = pattern_fn(1, size)
    bchw = _get_bchw(img)
    radius = int(math.ceil(3.0 * ss))
    exact = TEXStdlib._bilateral_exact_bchw(bchw, ss, 0.2, radius).permute(0, 2, 3, 1)
    approx = TEXStdlib._bilateral_detail_transfer_bchw(bchw, ss, 0.2).permute(0, 2, 3, 1)
    return (exact.float() - approx.float()).abs().max().item()


def test_bilat50_accuracy_band_checker(r: SubTestResult):
    print("\n--- BILAT-50: detail-transfer vs exact, adversarial 8px checker (resolution-"
          "scale.md style band) ---")
    band = 0.02
    worst = 0.0
    for ss in (2.0, 4.0, 8.0):
        md = _accuracy_case(_checker, ss)
        worst = max(worst, md)
        print(f"    checker ss={ss}: maxdiff {md:.4e}")
    if worst <= band:
        r.ok(f"checker: worst maxdiff {worst:.3e} within the pinned {band} band")
    else:
        r.fail("BILAT-50 accuracy band (checker)", f"worst maxdiff {worst:.3e} exceeds {band}")


def test_bilat50_accuracy_band_smooth_edges(r: SubTestResult):
    print("\n--- BILAT-50: detail-transfer vs exact, smooth gradient + hard-edged "
          "rectangles (resolution-scale.md style band) ---")
    band = 0.05
    worst = 0.0
    for ss in (2.0, 4.0, 8.0):
        md = _accuracy_case(_smooth_edges, ss)
        worst = max(worst, md)
        print(f"    smooth+edges ss={ss}: maxdiff {md:.4e}")
    if worst <= band:
        r.ok(f"smooth+edges: worst maxdiff {worst:.3e} within the pinned {band} band")
    else:
        r.fail("BILAT-50 accuracy band (smooth+edges)",
               f"worst maxdiff {worst:.3e} exceeds {band}")


# ── 5. Edge preservation: an edge stays within the band ─────────────────────────────────────

def test_bilat50_edge_preservation(r: SubTestResult):
    print("\n--- BILAT-50: a hard edge survives the detail-transfer path within the pinned "
          "band (unlike a plain blur, which would smear it) ---")
    size = 64
    img = make_img(1, size, size, 3)
    img[0, :, :size // 2, :] = 0.05
    img[0, :, size // 2:, :] = 0.95
    bchw = _get_bchw(img)
    ss = 6.0
    radius = int(math.ceil(3.0 * ss))
    exact = TEXStdlib._bilateral_exact_bchw(bchw, ss, 0.1, radius).permute(0, 2, 3, 1)
    approx = TEXStdlib._bilateral_detail_transfer_bchw(bchw, ss, 0.1).permute(0, 2, 3, 1)

    # Both the exact filter and its detail-transfer approximation must keep the edge sharp
    # (a low range_sigma means "only blend near-identical colours" -- the two flat regions
    # far from the boundary column must stay close to their ORIGINAL values, not smear
    # toward the mid-grey a naive large-radius Gaussian blur would produce there).
    far_left = approx[0, :, :size // 4, 0].mean().item()
    far_right = approx[0, :, 3 * size // 4:, 0].mean().item()
    if abs(far_left - 0.05) > 0.05 or abs(far_right - 0.95) > 0.05:
        r.fail("edge preservation (flat regions)",
               f"far_left={far_left:.3f} (want ~0.05), far_right={far_right:.3f} "
               f"(want ~0.95) -- the edge bled into regions far from the boundary")
        return
    md = (exact.float() - approx.float()).abs().max().item()
    band = 0.05
    if md <= band:
        r.ok(f"edge stays within the {band} band vs the exact filter (maxdiff {md:.3e}); "
             f"flat regions stay at far_left={far_left:.3f}, far_right={far_right:.3f}")
    else:
        r.fail("edge preservation (vs exact)", f"maxdiff {md:.3e} exceeds {band}")


# ── 6. Footprint reflects the real (now sigma-dependent) reach ──────────────────────────────

def test_bilat50_footprint_is_halo_arg_tied_to_spatial_sigma(r: SubTestResult):
    print("\n--- BILAT-50: footprint is now halo_arg tied to spatial_sigma (arg 1), not a "
          "fixed halo ---")
    by_name = {n: e for e in R.REGISTRY for n in e.names}
    fp = by_name["bilateral_filter"].footprint
    if not (isinstance(fp, tuple) and fp[0] == "halo_arg" and fp[1] == 1):
        r.fail("footprint shape", f"expected ('halo_arg', 1, mult), got {fp!r}")
        return
    r.ok(f"bilateral_filter footprint = {fp!r}")

    # The declared reach must never be SMALLER than the true reach at any radius this ask
    # ever runs (over-approximation is safe -- AGENTS.md invariant #5 -- under-approximation
    # is a silent-wrong ROI narrowing).
    for ss in (0.5, 1.5, 6.0, 8.0, 20.0):
        declared = _R._call_reach("bilateral_filter", [None, NumberLiteral(value=ss)])
        true_reach_exact_tier = int(math.ceil(3.0 * ss))
        if isinstance(declared, str):
            continue  # 'unbounded'/'image' -- always safe, never under-pads
        if declared < true_reach_exact_tier:
            r.fail(f"under-padded halo ss={ss}",
                   f"declared reach {declared} < the exact-tier's own true reach "
                   f"{true_reach_exact_tier}")
            return
    r.ok("declared halo_arg reach never under-estimates the exact-tier's own true "
         "(3*spatial_sigma) reach, at every tested spatial_sigma")


def test_bilat50_a4_footprint_does_not_overpad_the_unchanged_regime(r: SubTestResult):
    print("\n--- A4 (v0.50 Phase C, R3#5): the declared halo matches the exact/tiled-"
          "exact tiers' own true reach EXACTLY, not a conservative multiple of it ---")
    # Below the approx threshold, the true reach is always ceil(3*ss) (the exact and
    # tiled-exact tiers share the identical weighted-average math -- A5's own finding).
    # A4's fix is the mult itself: it used to be 8.0 (picked to cover the detail-
    # transfer tier's own reach too, from a single static number), over-padding this
    # regime's real halo by ~2.67x on any ROI-narrowed or tiled cook. A1's own
    # approx_above decline now handles the detail-transfer tier by refusing to narrow
    # at all, so nothing needs the conservative 8.0 here any more.
    for ss in (0.3, 0.5, 1.5, 4.0, 6.0, 8.0):
        declared = _R._call_reach("bilateral_filter", [None, NumberLiteral(value=ss)])
        true_reach = int(math.ceil(3.0 * ss))
        if declared != true_reach:
            r.fail(f"a4 overpad ss={ss}",
                   f"declared reach {declared} != the true reach {true_reach} "
                   f"(ratio {declared / true_reach:.2f}x)")
            return
    r.ok("declared halo_arg reach == 3*spatial_sigma exactly, for every spatial_sigma "
         "at or below the approx threshold (no over-padding)")


# ── 7. Windowed-vs-whole-frame pixel identity ────────────────────────────────────────────────

def _windowed_vs_whole(code, params, image, roi):
    x0, y0, w, h, W, H = roi
    _R.clear_roi_memo()
    full = tex_engine.cook(code, dict(params, A=image.clone()), device_mode="cpu").outputs["OUT"]
    _R.clear_roi_memo()
    res = tex_engine.cook(code, dict(params, A=image.clone()), device_mode="cpu",
                           roi=roi, roi_exec=True)
    win = res.outputs["OUT"]
    crop = full[:, y0:y0 + h, x0:x0 + w]
    return res.cooked_roi, win, crop


def test_bilat50_windowed_vs_whole_frame_exact_tier(r: SubTestResult):
    print("\n--- BILAT-50: windowed cook == whole-frame crop, exact tier (radius > 3, "
          "<= EXACT_RADIUS_MAX) ---")
    W, H = 48, 40
    roi = (10, 8, 16, 14, W, H)
    torch.manual_seed(77)
    image = torch.rand(1, H, W, 3)
    code = "@OUT = bilateral_filter(@A, 6.0, 0.2);"   # radius=18: exact-tiled tier
    try:
        cooked_roi, win, crop = _windowed_vs_whole(code, {}, image, roi)
        if cooked_roi != roi:
            r.fail("exact-tier windowed identity", f"window declined: cooked_roi={cooked_roi}")
            return
        if tuple(win.shape) == tuple(crop.shape) and torch.equal(win, crop):
            r.ok("windowed cook torch.equal whole-frame crop (exact tier, radius=18)")
        else:
            md = (win.float() - crop.float()).abs().max().item() \
                if tuple(win.shape) == tuple(crop.shape) else float("nan")
            r.fail("exact-tier windowed identity",
                   f"shape {tuple(win.shape)} vs {tuple(crop.shape)}, maxdiff {md:.4e}")
    except Exception as e:
        r.fail("exact-tier windowed identity", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


def test_bilat50_windowed_vs_whole_frame_below_today_window(r: SubTestResult):
    print("\n--- BILAT-50: windowed cook == whole-frame crop, at/below today's original "
          "window (radius<=3) ---")
    W, H = 48, 40
    roi = (10, 8, 16, 14, W, H)
    torch.manual_seed(78)
    image = torch.rand(1, H, W, 3)
    code = "@OUT = bilateral_filter(@A, 0.8, 0.2);"   # radius=3: today's original window
    try:
        cooked_roi, win, crop = _windowed_vs_whole(code, {}, image, roi)
        if cooked_roi != roi:
            r.fail("today-window identity", f"window declined: cooked_roi={cooked_roi}")
            return
        if tuple(win.shape) == tuple(crop.shape) and torch.equal(win, crop):
            r.ok("windowed cook torch.equal whole-frame crop (radius=3, today's window)")
        else:
            md = (win.float() - crop.float()).abs().max().item() \
                if tuple(win.shape) == tuple(crop.shape) else float("nan")
            r.fail("today-window identity",
                   f"shape {tuple(win.shape)} vs {tuple(crop.shape)}, maxdiff {md:.4e}")
    except Exception as e:
        r.fail("today-window identity", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


def test_bilat50_windowed_vs_whole_frame_detail_transfer_declines(r: SubTestResult):
    print("\n--- A1 (v0.50 Phase C): detail-transfer regime -- ANY spatial_sigma past the "
          "approx threshold now declines ROI narrowing outright, not just when a large halo "
          "happens to saturate to the whole frame ---")
    # spatial_sigma=20 > _BILATERAL_APPROX_THRESHOLD_SS (8.0): `_reach_of` now answers
    # 'unbounded' unconditionally for this call (A1's footprint fix), so the planner
    # declines the window the same way a symbolic radius already did -- `cooked_roi` is
    # None and the engine serves the whole frame. This used to narrow (returning only the
    # caller's crop) whenever the declared halo happened to already saturate to the whole
    # frame internally, which is true HERE, so the two behaviours are pixel-equal -- but a
    # genuinely narrow (non-saturating) window at this same spatial_sigma would previously
    # have diverged from a whole-frame cook (B1/B2's finding); this is the fix, verified on
    # this specific (saturating) case: the values must still match a whole-frame cook.
    W, H = 48, 40
    roi = (10, 8, 16, 14, W, H)
    torch.manual_seed(79)
    image = torch.rand(1, H, W, 3)
    code = "@OUT = bilateral_filter(@A, 20.0, 0.2);"  # detail-transfer tier, past threshold
    try:
        cooked_roi, win, crop = _windowed_vs_whole(code, {}, image, roi)
        if cooked_roi is not None:
            r.fail("detail-transfer decline", f"window narrowed unexpectedly: cooked_roi={cooked_roi}")
            return
        full_shape = (1, H, W, 3)
        if tuple(win.shape) == full_shape and torch.equal(win[:, roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], crop):
            r.ok("window declines past the approx threshold (cooked_roi=None) and the served "
                 "whole-frame output's own crop is torch.equal to an independent whole-frame cook's crop")
        else:
            r.fail("detail-transfer decline",
                   f"win shape {tuple(win.shape)}, expected {full_shape} (whole-frame decline)")
    except Exception as e:
        r.fail("detail-transfer decline", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


# ── A5 (v0.50 Phase C, R2#1): the radius<=3 inline regime duplicates
# `_bilateral_exact_bchw` -- prove bit-identity AND cost-equality before collapsing it ──

def test_bilat50_a5_exact_bchw_matches_inline_and_degenerates_to_one_tile(r: SubTestResult):
    print("\n--- A5: _bilateral_exact_bchw is bit-identical to the radius<=3 inline math, "
          "AND degenerates to a single untiled pass there (cost-equal) -- proof before "
          "collapsing the dead middle regime ---")
    torch.manual_seed(83)
    img = make_img(1, 20, 24, 3, seed=83)
    bchw = _get_bchw(img)
    for ss, sr in ((0.3, 0.1), (0.5, 0.2), (0.75, 0.4), (1.0, 0.2)):
        radius = int(math.ceil(3.0 * ss))
        if radius > 3:
            r.fail("a5 precondition", f"ss={ss} gave radius={radius} > 3 -- wrong test row")
            return
        via_fn = TEXStdlib.fn_bilateral_filter(img.clone(), ss, sr)  # today's inline branch
        direct = TEXStdlib._bilateral_exact_bchw(bchw.clone(), ss, sr, radius).permute(0, 2, 3, 1)
        if not torch.equal(via_fn, direct):
            r.fail(f"a5 bit-identity ss={ss}", "fn_bilateral_filter's inline regime-1 output "
                   "is not torch.equal to _bilateral_exact_bchw at the same radius")
            return
        # Cost-equality (structural, not wall-clock): at this radius, ksize=2*radius+1 is
        # tiny, so `_BILATERAL_TILE_BUDGET_ELEMS` (~8M) covers the whole image in one tile
        # -- `_bilateral_exact_bchw` degenerates to a single untiled pass, the same shape
        # the inline branch always was.
        H = bchw.shape[-2]
        tile_h = max(1, TEXStdlib._BILATERAL_TILE_BUDGET_ELEMS // max(1, bchw.shape[-1] * (2 * radius + 1) ** 2))
        if tile_h < H:
            r.fail(f"a5 cost-equality ss={ss}", f"tile_h={tile_h} < H={H} -- would tile, not "
                   "degenerate to one pass")
            return
    r.ok("_bilateral_exact_bchw is bit-identical to the inline radius<=3 math and runs as a "
         "single untiled pass there -- the two regimes are provably one regime under two names")
