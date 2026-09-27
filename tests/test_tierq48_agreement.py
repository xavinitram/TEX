"""TIERQ-48 — the declared-fallback query agrees with the real cook path.

A public, side-effect-free query (`tex_engine_tiers.tier_verdict`, re-exported at
`tex_api.tier_verdict`) answers which tier a cook of a given `(program, compile_mode,
device, precision, roi, scale, params)` WILL run on, and why — before a host cooks
anything. This file is the RED-FIRST
agreement test the design doc names: for a matrix of inputs, the query's answer must
equal what `tex_engine.prepare()` — the real cook path's own planning step — actually
decides, read off its `CookPlan` (never off `tier_trace`, which only records FALLBACK
events and can carry a stale record from a prior cook when the current one takes the
plain "default" tier cleanly — see `tex_engine_tiers._record_codegen_defect_fallback`'s
own docstring for that gap).

The query is built to call the exact same `tex_roi` predicates `prepare()`'s own gate
calls, in the same order (see `tier_verdict`'s docstring) — so a disagreement here would
mean the query's mirror has drifted from the real gate, which is exactly the class of
bug this test exists to catch before a host ever sees it.
"""
import torch

from TEX_Wrangle import tex_engine
from TEX_Wrangle import tex_roi as _tex_roi
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_engine_tiers import (
    tier_verdict, select_tier, TIER_REASON_SCALE_UNSAFE, TIER_REASON_SCALE_ACTIVE,
    TIER_REASON_SCALE_ACTIVE_CODEGEN,
    ROI_REASON_TIER_NOT_DEFAULT, ROI_REASON_NOT_ARMED, ROI_REASON_ARMED,
    ROI_REASON_WHOLE_FRAME, ROI_REASON_SCALE_ACTIVE,
)

# An ROI-executable program (pointwise + one inline gauss_blur reading a $param) — the
# same shape `benchmarks/roi_scrub_bench.py`/`roi_codegen_ab_bench.py` already use, kept
# inline for the same reason their docstrings give (a named local blocks ROI-1 reach
# composition in v1).
_ROI_CODE = "@OUT = vec4(mix(@A.rgb, gauss_blur(@A, 2.0).rgb, $amount), 1.0);\n"
_PLAIN_CODE = "@OUT = @A * 2.0;\n"
# Reads img_width() outside a whitelisted fetch call — the classifier's own documented
# unsafe class (docs/resolution-scale.md "The classifier and the override comment").
_SCALE_UNSAFE_CODE = "@OUT = vec4(vec3(float(img_width()) * 0.001), 1.0);\n"

# SCALE-CG-48's own precondition shape (tests/test_scalecg48_codegen_scale.py): an
# exact-fetch box-blur stencil (UC-2 routes this to codegen) alongside an independent
# gauss_blur output — `//!tex scale: safe` vouches for the hand-written ix/iy pixel
# arithmetic the classifier over-approximates as unsafe (documented, sanctioned override).
_STENCIL_PLUS_BLUR_CODE = """//!tex scale: safe
i$radius = 2;
vec3 acc = vec3(0.0);
float cnt = 0.0;
for (int dy = -$radius; dy <= $radius; dy = dy + 1) {
    for (int dx = -$radius; dx <= $radius; dx = dx + 1) {
        acc = acc + fetch(@A, ix + dx, iy + dy).rgb;
        cnt = cnt + 1.0;
    }
}
@STENCIL = vec4(acc / cnt, 1.0);
@BLUR = gauss_blur(@A, 8.0);
"""
_STENCIL_PLUS_BLUR_BT = {"A": TEXType.VEC3, "radius": TEXType.INT,
                        "STENCIL": TEXType.VEC4, "BLUR": TEXType.VEC4}


def _real_plan(code, bindings, **kw):
    """Ground truth: what `tex_engine.prepare()` actually decided, read off its
    `CookPlan` — never `tier_trace` for the NON-scale case (see module docstring: it
    only records on fallback and can carry a stale prior-cook record)."""
    _tex_roi.clear_roi_memo()
    return tex_engine.prepare(code, dict(bindings), device_mode="cpu", **kw)


def _real_tier_and_roi_armed(code, bindings, **kw):
    plan = _real_plan(code, bindings, **kw)
    if plan.ctx.scale is not None:
        # SCALE-CG-48: codegen-vs-interpreter for a scale-active cook is decided INSIDE
        # `_dispatch_tier` (the UC-2 stencil gate on the compiled program), not visible on
        # `CookPlan` — actually run it and read `tier_trace`, which BOTH branches of that
        # decision explicitly record for a scale-active cook (unlike the general
        # "default tier succeeded quietly" gap this module's docstring names).
        from TEX_Wrangle.tex_runtime import tier_trace as _tt
        tex_engine._dispatch_tier(plan)
        tier = _tt.last().tier
    else:
        tier = plan.tier_id
    return tier, plan.ctx.roi is not None


def test_tierq48_agrees_plain_cook_no_roi_no_scale():
    A = torch.rand(1, 8, 8, 4)
    real_tier, real_roi_armed = _real_tier_and_roi_armed(_PLAIN_CODE, {"A": A})
    v = tier_verdict(_PLAIN_CODE, compile_mode="none", device="cpu")
    assert (v.tier, v.roi_armed) == (real_tier, real_roi_armed)
    assert v.tier == "default"


def test_tierq48_agrees_scale_active_forces_interpreter():
    A = torch.rand(1, 8, 8, 4)
    real_tier, real_roi_armed = _real_tier_and_roi_armed(_PLAIN_CODE, {"A": A}, scale=0.5)
    v = tier_verdict(_PLAIN_CODE, compile_mode="none", device="cpu", scale=0.5)
    assert (v.tier, v.roi_armed) == (real_tier, real_roi_armed)
    assert v.tier == "interpreter" and v.reason == TIER_REASON_SCALE_ACTIVE


def test_tierq48_agrees_scale_1_0_still_forces_interpreter():
    """`scale=1.0` is the byte-identical-VALUE case, but `_run_tier`'s bypass is on
    `is not None`, not on value — 1.0 still routes to the interpreter tier."""
    A = torch.rand(1, 8, 8, 4)
    real_tier, _ = _real_tier_and_roi_armed(_PLAIN_CODE, {"A": A}, scale=1.0)
    v = tier_verdict(_PLAIN_CODE, compile_mode="none", device="cpu", scale=1.0)
    assert v.tier == real_tier == "interpreter"


def test_tierq48_agrees_scale_active_stencil_route_uses_codegen():
    """SCALE-CG-48: a scale-active cook on the `"default"` tier routes to codegen
    instead of the interpreter when the UC-2 stencil gate would already choose it —
    the query must name `"codegen"` here, not fall back to the pre-SCALE-CG-48
    `"interpreter"` answer."""
    A = torch.rand(1, 32, 32, 3)
    real_tier, real_roi_armed = _real_tier_and_roi_armed(
        _STENCIL_PLUS_BLUR_CODE, {"A": A, "radius": 2}, scale=0.5)
    v = tier_verdict(_STENCIL_PLUS_BLUR_CODE, compile_mode="none", device="cpu",
                     scale=0.5, binding_types=_STENCIL_PLUS_BLUR_BT)
    assert (v.tier, v.roi_armed) == (real_tier, real_roi_armed)
    assert v.tier == "codegen" and v.reason == TIER_REASON_SCALE_ACTIVE_CODEGEN


def test_tierq48_scale_active_codegen_route_falls_back_when_it_cannot_compile():
    """When `code` cannot be compiled against the given (or omitted) `binding_types` —
    here, a binding whose declared type disagrees with how it is used — the query
    cannot check the UC-2 stencil gate and conservatively reports `"interpreter"`,
    documented as a pessimistic-but-never-wrong-pixel gap: the tier this reports is a
    tier that ALSO would have run correctly, just not necessarily the fastest one a
    real (successfully-compiled) cook would pick."""
    bad_bt = dict(_STENCIL_PLUS_BLUR_BT)
    bad_bt["A"] = TEXType.FLOAT       # disagrees with `.rgb` channel access on @A
    v = tier_verdict(_STENCIL_PLUS_BLUR_CODE, compile_mode="none", device="cpu",
                     scale=0.5, binding_types=bad_bt)
    assert v.tier == "interpreter" and v.reason == TIER_REASON_SCALE_ACTIVE


def test_tierq48_agrees_scale_unsafe_refuses():
    """A scale-unsafe program at a genuinely coarse scale REFUSES (raises) in the real
    cook path rather than running on any tier; the query reports `tier=None` with the
    stable refusal reason instead of guessing."""
    A = torch.rand(1, 8, 8, 4)
    raised = False
    try:
        _real_plan(_SCALE_UNSAFE_CODE, {"A": A}, scale=0.5)
    except RuntimeError as e:
        raised = hasattr(e, "tex_refusal")
    assert raised, "the real cook path must refuse a scale-unsafe program at scale=0.5"
    v = tier_verdict(_SCALE_UNSAFE_CODE, compile_mode="none", device="cpu", scale=0.5)
    assert v.tier is None and v.reason == TIER_REASON_SCALE_UNSAFE


def test_tierq48_agrees_roi_armed_on_default_tier():
    A = torch.rand(1, 1024, 1024, 4)
    roi = (10, 10, 256, 256, 1024, 1024)
    real_tier, real_roi_armed = _real_tier_and_roi_armed(
        _ROI_CODE, {"A": A, "amount": 0.4}, roi=roi, roi_exec=True)
    v = tier_verdict(_ROI_CODE, compile_mode="none", device="cpu", roi=roi,
                     roi_exec=True, param_values={"amount": 0.4})
    assert (v.tier, v.roi_armed) == (real_tier, real_roi_armed)
    assert real_roi_armed and v.roi_reason == ROI_REASON_ARMED


def test_tierq48_agrees_roi_declines_when_not_armed():
    A = torch.rand(1, 1024, 1024, 4)
    roi = (10, 10, 256, 256, 1024, 1024)
    real_tier, real_roi_armed = _real_tier_and_roi_armed(
        _ROI_CODE, {"A": A, "amount": 0.4}, roi=roi, roi_exec=False)
    v = tier_verdict(_ROI_CODE, compile_mode="none", device="cpu", roi=roi,
                     roi_exec=False, param_values={"amount": 0.4})
    assert (v.tier, v.roi_armed) == (real_tier, real_roi_armed)
    assert not real_roi_armed and v.roi_reason == ROI_REASON_NOT_ARMED


def test_tierq48_agrees_roi_declines_on_whole_frame_window():
    A = torch.rand(1, 64, 64, 4)
    roi = (0, 0, 64, 64, 64, 64)               # covers the whole frame — nothing to narrow
    real_tier, real_roi_armed = _real_tier_and_roi_armed(
        _ROI_CODE, {"A": A, "amount": 0.4}, roi=roi, roi_exec=True)
    v = tier_verdict(_ROI_CODE, compile_mode="none", device="cpu", roi=roi,
                     roi_exec=True, param_values={"amount": 0.4})
    assert (v.tier, v.roi_armed) == (real_tier, real_roi_armed)
    assert not real_roi_armed and v.roi_reason == ROI_REASON_WHOLE_FRAME


def test_tierq48_agrees_roi_declines_when_scale_also_active():
    A = torch.rand(1, 1024, 1024, 4)
    roi = (10, 10, 256, 256, 1024, 1024)
    real_tier, real_roi_armed = _real_tier_and_roi_armed(
        _ROI_CODE, {"A": A, "amount": 0.4}, roi=roi, roi_exec=True, scale=0.5)
    v = tier_verdict(_ROI_CODE, compile_mode="none", device="cpu", roi=roi,
                     roi_exec=True, scale=0.5, param_values={"amount": 0.4})
    assert (v.tier, v.roi_armed) == (real_tier, real_roi_armed)
    assert v.tier == "interpreter" and not real_roi_armed
    assert v.roi_reason == ROI_REASON_SCALE_ACTIVE


def test_tierq48_agrees_a_non_default_tier_never_arms_roi():
    """`select_tier` alone decides eligibility for torch_compile/auto/cuda_graph — no
    real device is needed to prove ROI never arms there (CPU-testable, per
    `select_tier`'s own docstring)."""
    roi = (10, 10, 256, 256, 1024, 1024)
    for compile_mode in ("torch_compile", "auto"):
        tier_id = select_tier(compile_mode, "cpu", False, False)
        assert tier_id == compile_mode
        v = tier_verdict(_ROI_CODE, compile_mode=compile_mode, device="cpu", roi=roi,
                         roi_exec=True, param_values={"amount": 0.4})
        assert v.tier == compile_mode
        assert not v.roi_armed and v.roi_reason == ROI_REASON_TIER_NOT_DEFAULT
    # cuda_graph: select_tier only string-checks the device (no real GPU required).
    tier_id = select_tier("cuda_graph", "cuda:0", False, False)
    assert tier_id == "cuda_graph"
    v = tier_verdict(_ROI_CODE, compile_mode="cuda_graph", device="cuda:0", roi=roi,
                     roi_exec=True, param_values={"amount": 0.4})
    assert v.tier == "cuda_graph"
    assert not v.roi_armed and v.roi_reason == ROI_REASON_TIER_NOT_DEFAULT


def test_tierq48_never_raises_on_a_malformed_roi():
    """Contract: `tier_verdict` never raises — a malformed window is a declined reason,
    not a `TypeError`/`ValueError` escaping to the caller."""
    v = tier_verdict(_ROI_CODE, compile_mode="none", device="cpu",
                     roi=("not", "a", "window"), roi_exec=True)
    assert v.tier == "default"
    assert not v.roi_armed
    assert v.roi_reason is not None and v.roi_reason.startswith("roi-declined")
