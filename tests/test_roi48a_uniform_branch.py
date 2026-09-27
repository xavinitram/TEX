"""ROI-48A — `tex_roi.roi_plan` resolves a branch whose condition depends only on UNIFORM
(`$param`) values BEFORE judging reach, so a program whose not-taken arm has a
wide/unbounded footprint still serves a window (an embedding host's ranked ask 1).

The shape, as the host named it: `blur.tex`'s Exponential mip taps sit beside its
Gaussian branch — present in source, never reached when `$mode` picks Gaussian — and
before this fix `roi_plan` walked BOTH arms unconditionally, so the untaken arm's
`sample_mip_gauss` (a whole-image gather, `footprint='image'`) declined the window even
though nothing the cook actually runs reads past a bounded neighbourhood. `Blur`,
`FlowWarp` and `VectorBlur` (the host's three named stock routes) all follow the same
quality-mode shape: a fast/preview arm with a bounded, narrowable reach beside a full-
quality arm that gathers the whole image. TEX ships no host-facing stock library beyond
`stock/blur.textool` (a single-mode wrapper); the three programs below are representative
test-authored TEX source built to the exact structural shape the host described, not a
port of the host's own private tool definitions.

RED AT BASE (`06f81e7`, verified by hand before this fix landed — see the ROI-48A hand-
back for the transcript): every `roi_plan(...)` call below for the SAFE (mode=0) arm of
all three programs returned `executable=False` at base, because `_accumulate` walked the
untaken wide arm regardless of the (already-folded-to-a-literal) `$mode` condition.
GREEN after: `tex_roi._resolved_branch` reads only the branch a folded-literal condition
actually selects, so mode=0 (bounded) now serves a window and mode=1 (a genuine
whole-image gather) still correctly declines one — the fix narrows the FALSE POSITIVE,
it does not loosen the real refusal.

SOUND ON DOUBT: `test_roi48a_sound_on_doubt` pins that a per-pixel condition, or a call
that never resolves `$mode` at all, leaves the condition symbolic and both arms are still
walked exactly as before the fix — the whitelist posture (unknown -> whole image) is
unchanged for every case this fix does not have proof for.
"""
from helpers import *


# ── Representative programs (the host's three named routes) ──────────────────────────
#
# Each is `i$mode = 0` (0 = fast/preview, bounded reach; 1 = full quality, whole-image
# gather) wrapped in one `if`/`else`. The safe arm is a DIRECT-TENSOR halo op (gauss_blur /
# dilate — narrowable, ROI-1 `footprint=('halo'|'halo_arg', ...)`); the wide arm is a
# `sample`/`sample_mip_gauss` gather (`footprint='image'`, unbounded, un-narrowable by
# construction — v1 ROI correctly declines a window whenever this arm is the one that
# runs, exactly as documented in `roi_plan`'s own docstring).

BLUR_SRC = """
i$mode = 0;   // 0 = Gaussian (bounded), 1 = Exponential (mip-pyramid taps)
f$sigma = 2.0;
f$radius = 32.0;
if ($mode == 0) {
    @OUT = gauss_blur(@image, $sigma);
} else {
    float base = log2(max($radius, 1.0) / 0.825);
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 6; i++) {
        acc = acc + sample_mip_gauss(@image, u, v, base);
    }
    @OUT = acc;
}
"""
BLUR_PARAMS = {"mode": 0, "sigma": 2.0, "radius": 32.0}

FLOWWARP_SRC = """
i$mode = 0;   // 0 = preview (morphological spread, bounded), 1 = full warp (whole-image sample)
f$dx = 0.05;
f$dy = 0.02;
if ($mode == 0) {
    @OUT = dilate(@image, 2);
} else {
    @OUT = sample(@image, u + $dx, v + $dy).rgb;
}
"""
FLOWWARP_PARAMS = {"mode": 0, "dx": 0.05, "dy": 0.02}

VECTORBLUR_SRC = """
i$mode = 0;   // 0 = preview (fixed Gaussian, bounded), 1 = full quality (per-pixel sample loop)
f$strength = 6.0;
if ($mode == 0) {
    @OUT = gauss_blur(@image, 1.5);
} else {
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        float t = (float(i) - 3.5) / 3.5;
        sum = sum + sample(@image, u + t * $strength / iw, v).rgb;
    }
    @OUT = sum / 8.0;
}
"""
VECTORBLUR_PARAMS = {"mode": 0, "strength": 6.0}

_PROGRAMS = [
    ("Blur", BLUR_SRC, BLUR_PARAMS),
    ("FlowWarp", FLOWWARP_SRC, FLOWWARP_PARAMS),
    ("VectorBlur", VECTORBLUR_SRC, VECTORBLUR_PARAMS),
]


def test_roi48a_uniform_branch_unlocks_window(r: SubTestResult):
    print("\n--- ROI-48A: a uniform-param branch resolves before roi_plan judges reach ---")
    from TEX_Wrangle import tex_roi as _R
    try:
        for name, code, params in _PROGRAMS:
            _R.clear_roi_memo()
            safe_params = dict(params, mode=0)
            plan_safe = _R.roi_plan(code, safe_params)
            if plan_safe.executable:
                r.ok(f"{name}: mode=0 (bounded arm taken) serves a window "
                     f"(halo={plan_safe.halo})")
            else:
                r.fail(f"ROI-48A {name} mode=0", f"still declined: {plan_safe}")

            _R.clear_roi_memo()
            wide_params = dict(params, mode=1)
            plan_wide = _R.roi_plan(code, wide_params)
            if not plan_wide.executable:
                r.ok(f"{name}: mode=1 (whole-image gather taken) still correctly declines")
            else:
                r.fail(f"ROI-48A {name} mode=1 unsound",
                       f"a real whole-image gather must still decline: {plan_wide}")
    except Exception as e:
        r.fail("ROI-48A uniform-branch reach", f"{type(e).__name__}: {e}")
    finally:
        from TEX_Wrangle import tex_roi as _R2
        _R2.clear_roi_memo()


def test_roi48a_pixel_identity(r: SubTestResult):
    print("\n--- ROI-48A: windowed vs whole-frame is torch.equal, CPU and CUDA ---")
    from TEX_Wrangle import tex_engine, tex_roi as _R
    W, H = 48, 40
    roi = (10, 8, 16, 14, W, H)
    x0, y0, w, h, _, _ = roi
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    try:
        for name, code, params in _PROGRAMS:
            safe_params = dict(params, mode=0)
            for dev in devs:
                torch.manual_seed(2048)
                image = torch.rand(1, H, W, 3, device=dev)
                _R.clear_roi_memo()
                full = tex_engine.cook(code, dict(safe_params, image=image.clone()),
                                       device_mode=dev).outputs["OUT"]
                _R.clear_roi_memo()
                res = tex_engine.cook(code, dict(safe_params, image=image.clone()),
                                      device_mode=dev, roi=roi, roi_exec=True)
                win = res.outputs["OUT"]
                if res.cooked_roi != roi:
                    r.fail(f"ROI-48A {name} pixel identity ({dev})",
                           f"window declined: cooked_roi={res.cooked_roi}")
                    continue
                crop = full[:, y0:y0 + h, x0:x0 + w]
                if tuple(win.shape) == tuple(crop.shape) and torch.equal(win, crop):
                    r.ok(f"{name} ({dev}): windowed cook torch.equal whole-frame crop")
                else:
                    md = (win.float() - crop.float()).abs().max().item() \
                        if tuple(win.shape) == tuple(crop.shape) else float("nan")
                    r.fail(f"ROI-48A {name} pixel identity ({dev})",
                           f"shape {tuple(win.shape)} vs {tuple(crop.shape)}, maxdiff {md:.3e}")
    except Exception as e:
        r.fail("ROI-48A pixel identity", f"{type(e).__name__}: {e}")
    finally:
        from TEX_Wrangle import tex_roi as _R2
        _R2.clear_roi_memo()


def test_roi48a_sound_on_doubt(r: SubTestResult):
    print("\n--- ROI-48A: any doubt keeps today's conservative (both-arms) answer ---")
    from TEX_Wrangle import tex_roi as _R
    try:
        # (1) a genuinely PER-PIXEL condition never folds to a literal — both arms must
        # still be walked, so the presence of the wide arm still declines.
        per_pixel = """
        f$sigma = 2.0;
        if (@image.r > 0.5) {
            @OUT = gauss_blur(@image, $sigma);
        } else {
            @OUT = sample(@image, u, v).rgb;
        }
        """
        _R.clear_roi_memo()
        plan = _R.roi_plan(per_pixel, {"sigma": 2.0})
        if not plan.executable:
            r.ok("a per-pixel condition is never treated as uniform (still declines)")
        else:
            r.fail("ROI-48A soundness: per-pixel condition", f"wrongly resolved: {plan}")

        # (2) an UNRESOLVED $param (the caller never supplied `mode`) leaves the condition
        # symbolic too — `_substitute_params` only substitutes names it was given a value
        # for, so the comparison can't fold and both arms are walked.
        for name, code, params in _PROGRAMS:
            unresolved = {k: v for k, v in params.items() if k != "mode"}
            _R.clear_roi_memo()
            plan_u = _R.roi_plan(code, unresolved)
            if not plan_u.executable:
                r.ok(f"{name}: an unresolved $mode stays symbolic (still declines)")
            else:
                r.fail(f"ROI-48A soundness: {name} unresolved $mode",
                       f"wrongly resolved without a mode value: {plan_u}")
    except Exception as e:
        r.fail("ROI-48A soundness on doubt", f"{type(e).__name__}: {e}")
    finally:
        from TEX_Wrangle import tex_roi as _R2
        _R2.clear_roi_memo()
