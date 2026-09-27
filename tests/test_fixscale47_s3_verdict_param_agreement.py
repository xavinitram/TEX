"""FIX-SCALE S3 (v0.47 Phase C, B2 finding 3) — `tex_api.scale_verdict(source, param_values)`
and `tex_engine.prepare()`'s own refusal must read the SAME memoized answer for the SAME
effective cook, so a host's cheap pre-cook query is never a lie.

Before this fix: `prepare()` always called `tex_roi.scale_verdict(code)` with an implicit
`{}` -- never the caller's real `$param` values, even though `bindings` (the very dict
`prepare()` was handed) carries them. A conditional gated on a scalar `$param` folds away
its unsafe arm only when the real value is known, so `tex_api.scale_verdict(code,
{"mode": 1.0})` (safe=True, the real value) and `prepare(code, {"A":..., "mode": 1.0},
scale=0.5)` (refused anyway, folded on `{}`) could disagree for the identical cook."""
from helpers import *
from TEX_Wrangle import tex_api, tex_engine

_COND_CODE = "@OUT = ($mode > 0.5) ? gauss_blur(@A, 4.0) : vec4(ix * 0.01, 0.0, 0.0, 1.0);"


def test_s3_verdict_with_no_params_is_conservative(r: SubTestResult):
    print("\n--- FIX-SCALE S3: with no param values, both branches are walked -> unsafe ---")
    v = tex_api.scale_verdict(_COND_CODE)
    if v.safe:
        r.fail("no-params verdict", f"expected safe=False (both arms walked with no fold "
               f"information), got {v!r}")
        return
    r.ok(f"scale_verdict({_COND_CODE!r}) with no param_values -> {v!r}")


def test_s3_verdict_with_real_params_folds_to_safe_arm(r: SubTestResult):
    print("\n--- FIX-SCALE S3: with mode=1.0, the unsafe arm folds away -> safe ---")
    v = tex_api.scale_verdict(_COND_CODE, {"mode": 1.0})
    if not v.safe:
        r.fail("real-params verdict", f"expected safe=True (mode=1.0 folds to the "
               f"gauss_blur-only arm), got {v!r}")
        return
    r.ok(f"scale_verdict({_COND_CODE!r}, mode=1.0) -> {v!r}")


def test_s3_prepare_agrees_with_the_pre_cook_query_at_the_real_param_value(r: SubTestResult):
    print("\n--- FIX-SCALE S3: prepare() must not refuse when scale_verdict(code, real_params) said safe ---")
    A = make_img(1, 8, 8, 4)
    v = tex_api.scale_verdict(_COND_CODE, {"mode": 1.0})
    raised = None
    try:
        tex_engine.prepare(_COND_CODE, {"A": A, "mode": 1.0}, device_mode="cpu", scale=0.5)
    except Exception as e:
        raised = e
    cook_says_safe = raised is None
    if cook_says_safe != v.safe:
        r.fail("verdict/cook agreement",
               f"tex_api.scale_verdict(code, {{'mode': 1.0}}) said safe={v.safe}, but "
               f"prepare() with the SAME mode=1.0 binding "
               f"{'did not raise' if cook_says_safe else 'raised: ' + str(raised)}")
        return
    r.ok("prepare(scale=0.5) with mode=1.0 agrees with scale_verdict(code, {'mode': 1.0}): "
         "both say safe")


def test_s3_prepare_still_refuses_when_the_unsafe_arm_is_live(r: SubTestResult):
    print("\n--- FIX-SCALE S3: prepare() still refuses when the REAL param keeps the unsafe arm live ---")
    A = make_img(1, 8, 8, 4)
    v = tex_api.scale_verdict(_COND_CODE, {"mode": 0.0})
    raised = None
    try:
        tex_engine.prepare(_COND_CODE, {"A": A, "mode": 0.0}, device_mode="cpu", scale=0.5)
    except Exception as e:
        raised = e
    cook_says_safe = raised is None
    if cook_says_safe != v.safe:
        r.fail("verdict/cook agreement (unsafe arm)",
               f"tex_api.scale_verdict(code, {{'mode': 0.0}}) said safe={v.safe}, but "
               f"prepare() with the SAME mode=0.0 binding "
               f"{'did not raise' if cook_says_safe else 'raised: ' + str(raised)}")
        return
    if v.safe:
        r.fail("premise", "expected mode=0.0 to keep the unsafe (ix-reading) arm live")
        return
    r.ok("prepare(scale=0.5) with mode=0.0 agrees with scale_verdict(code, {'mode': 0.0}): "
         "both refuse")
