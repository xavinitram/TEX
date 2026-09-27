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
    TIER_REASON_SCALE_ACTIVE_CODEGEN, TIER_REASON_SCALE_ACTIVE_COMPILED,
    TIER_REASON_SELECTED, TIER_REASON_TORCH_COMPILE_GRAPH_BREAK,
    TIER_REASON_CUDA_GRAPH_NOT_CAPTURABLE,
    ROI_REASON_TIER_NOT_DEFAULT, ROI_REASON_NOT_ARMED, ROI_REASON_ARMED,
    ROI_REASON_WHOLE_FRAME, ROI_REASON_SCALE_ACTIVE,
)
# FIX-TIER T6 (R2#5): single-sourced from test_scalecg48_codegen_scale.py rather than a
# verbatim second copy -- same TEX source, same binding-types dict, kept in one file so a
# future edit to the shape (e.g. widening the stencil radius default) only has one place
# to happen.
from test_scalecg48_codegen_scale import (
    _STENCIL_PLUS_BLUR as _STENCIL_PLUS_BLUR_CODE,
    _STENCIL_PLUS_BLUR_BT,
)
# COMPILETRY-50 (D1): `_torch_compile_graph_break` now consults the per-fingerprint
# fall-through memo instead of predicting a graph-break unconditionally for every
# `_has_fn_calls` program -- the tests near `_GAUSS_BLUR_CODE` below drive that memo
# directly (via `TEXCache.fingerprint`, the same call `tier_verdict` makes internally)
# to cover all three verdict states truthfully.
from TEX_Wrangle.tex_cache import get_cache as _get_cache
from TEX_Wrangle.tex_runtime import fncalls_compile as _fncalls_compile

# An ROI-executable program (pointwise + one inline gauss_blur reading a $param) — the
# same shape `benchmarks/roi_scrub_bench.py`/`roi_codegen_ab_bench.py` already use, kept
# inline for the same reason their docstrings give (a named local blocks ROI-1 reach
# composition in v1).
_ROI_CODE = "@OUT = vec4(mix(@A.rgb, gauss_blur(@A, 2.0).rgb, $amount), 1.0);\n"
_PLAIN_CODE = "@OUT = @A * 2.0;\n"
# Reads img_width() outside a whitelisted fetch call — the classifier's own documented
# unsafe class (docs/resolution-scale.md "The classifier and the override comment").
_SCALE_UNSAFE_CODE = "@OUT = vec4(vec3(float(img_width()) * 0.001), 1.0);\n"


def _real_plan(code, bindings, **kw):
    """Ground truth: what `tex_engine.prepare()` actually decided, read off its
    `CookPlan` — never `tier_trace` for the NON-scale case (see module docstring: it
    only records on fallback and can carry a stale prior-cook record)."""
    _tex_roi.clear_roi_memo()
    return tex_engine.prepare(code, dict(bindings), device_mode="cpu", **kw)


def _real_tier_and_roi_armed(code, bindings, **kw):
    plan = _real_plan(code, bindings, **kw)
    # SCALECX-49: the codegen-vs-interpreter peek below is SPECIFIC to the "default" tier's
    # own internal UC-2 stencil decision, whose BOTH branches explicitly `tier_trace.record`
    # (SCALE-CG-48). `torch_compile`/`auto` have no equivalent "both branches record"
    # contract — a no-backend decline there falls to `_plain_execute` without ever calling
    # `tier_trace.record`, so `tier_trace.last()` after one of THOSE cooks can be stale (a
    # leftover from whatever this thread cooked last, or None). For those two (and
    # `cuda_graph`), `plan.tier_id` IS the ground truth `tier_verdict` claims to predict —
    # "which tier `_run_tier` dispatches to" — so only peek inside `tier_trace` for the
    # "default" tier's own internal choice.
    if plan.ctx.scale is not None and plan.tier_id == "default":
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
    `select_tier`'s own docstring). ROI armament (this test's actual subject) is
    unaffected by FIX-SCALECX X1's correction of WHICH tier the query names for the
    non-ROI question — `_ROI_CODE` itself calls `gauss_blur`, so `select_tier`'s
    `torch_compile`/`auto`/`cuda_graph` choice is each corrected exactly per X1's
    dedicated tests above; `roi_armed`/`roi_reason` are the invariant this test pins."""
    roi = (10, 10, 256, 256, 1024, 1024)
    # COMPILETRY-50 (D1): the tier/reason this test's OWN docstring says is not its
    # subject still needs a settled (known-bad) fall-through verdict to keep predicting
    # "codegen" here, exactly like the dedicated tests above -- pinned False for the
    # duration of this loop, restored after (`tier_verdict` passes no `binding_types`
    # here, so the fingerprint is against an empty one, matching what it looks up).
    fp = _get_cache().fingerprint(_ROI_CODE, {})
    _fncalls_compile.reset_for_test()
    _fncalls_compile._memo[fp] = False
    for compile_mode in ("torch_compile", "auto"):
        tier_id = select_tier(compile_mode, "cpu", False, False)
        assert tier_id == compile_mode
        v = tier_verdict(_ROI_CODE, compile_mode=compile_mode, device="cpu", roi=roi,
                         roi_exec=True, param_values={"amount": 0.4})
        assert v.tier == "codegen" and v.reason == TIER_REASON_TORCH_COMPILE_GRAPH_BREAK
        assert not v.roi_armed and v.roi_reason == ROI_REASON_TIER_NOT_DEFAULT
    _fncalls_compile.reset_for_test()
    # cuda_graph: select_tier only string-checks the device (no real GPU required).
    tier_id = select_tier("cuda_graph", "cuda:0", False, False)
    assert tier_id == "cuda_graph"
    v = tier_verdict(_ROI_CODE, compile_mode="cuda_graph", device="cuda:0", roi=roi,
                     roi_exec=True, param_values={"amount": 0.4})
    assert v.tier == "interpreter" and v.reason == TIER_REASON_CUDA_GRAPH_NOT_CAPTURABLE
    assert not v.roi_armed and v.roi_reason == ROI_REASON_TIER_NOT_DEFAULT


# ── SCALECX-49: torch_compile/auto/cuda_graph now honour scale ──────────────────────
#
# Before this ask, a scale-active cook on ANY of these three tiers was unconditionally
# forced to `"interpreter"` (`TIER_REASON_SCALE_ACTIVE`) by `_run_tier`'s own bypass — the
# query already reported that correctly. Now `_run_tier` dispatches straight to each
# tier's own strategy (`_run_torch_compile`/`_run_auto`/`_run_cuda_graph`, each keying its
# compiled artifact / captured graph by an explicit `scale` component), so the query must
# name THAT tier instead, with the new `TIER_REASON_SCALE_ACTIVE_COMPILED` reason.
def test_tierq48_agrees_scale_active_compiled_tier_runs_on_torch_compile():
    A = torch.rand(1, 8, 8, 4)
    real_tier, real_roi_armed = _real_tier_and_roi_armed(
        _PLAIN_CODE, {"A": A}, scale=0.5, compile_mode="torch_compile")
    v = tier_verdict(_PLAIN_CODE, compile_mode="torch_compile", device="cpu", scale=0.5)
    assert (v.tier, v.roi_armed) == (real_tier, real_roi_armed)
    assert v.tier == "torch_compile" and v.reason == TIER_REASON_SCALE_ACTIVE_COMPILED


def test_tierq48_agrees_scale_active_compiled_tier_runs_on_auto():
    A = torch.rand(1, 8, 8, 4)
    real_tier, real_roi_armed = _real_tier_and_roi_armed(
        _PLAIN_CODE, {"A": A}, scale=0.25, compile_mode="auto")
    v = tier_verdict(_PLAIN_CODE, compile_mode="auto", device="cpu", scale=0.25)
    assert (v.tier, v.roi_armed) == (real_tier, real_roi_armed)
    assert v.tier == "auto" and v.reason == TIER_REASON_SCALE_ACTIVE_COMPILED


def test_tierq48_agrees_scale_active_cuda_graph_reported_without_real_gpu():
    """`select_tier`'s `cuda_graph` branch is a pure device-string check (no real GPU
    needed, same fact `test_tierq48_agrees_a_non_default_tier_never_arms_roi` already
    leans on) — this only exercises the QUERY, not a live cook (a live `cuda_graph` cook
    needs real CUDA hardware, covered separately, CUDA-gated, in
    test_scalecx49_compiled_graphed_scale.py)."""
    tier_id = select_tier("cuda_graph", "cuda:0", False, False)
    assert tier_id == "cuda_graph"
    v = tier_verdict(_PLAIN_CODE, compile_mode="cuda_graph", device="cuda:0", scale=0.125)
    assert v.tier == "cuda_graph" and v.reason == TIER_REASON_SCALE_ACTIVE_COMPILED


def test_tierq48_agrees_roi_still_declines_on_scale_active_compiled_tier():
    """ROI must stay out of scope here too: a scale-active cook that ALSO requests `roi=`
    on `torch_compile` declines ROI for the SAME pre-existing reason as before this ask
    (`tier_id != "default"`) — SCALECX-49 makes the tier itself scale-aware, it does not
    touch ROI eligibility at all."""
    A = torch.rand(1, 8, 8, 4)
    roi = (1, 1, 4, 4, 8, 8)
    real_tier, real_roi_armed = _real_tier_and_roi_armed(
        _PLAIN_CODE, {"A": A}, scale=0.5, roi=roi, roi_exec=True,
        compile_mode="torch_compile")
    v = tier_verdict(_PLAIN_CODE, compile_mode="torch_compile", device="cpu", scale=0.5,
                     roi=roi, roi_exec=True)
    assert (v.tier, v.roi_armed) == (real_tier, real_roi_armed)
    assert v.tier == "torch_compile" and not v.roi_armed
    assert v.roi_reason == ROI_REASON_TIER_NOT_DEFAULT


# ── FIX-SCALECX X1 (B2#1): tier_verdict must predict the tier that ACTUALLY runs ────
#
# `select_tier`'s choice is not the end of the story for torch_compile/auto/cuda_graph: each
# can itself self-decline past that point (a graph-break -> no real Inductor backend for
# torch_compile/auto; a not-capturable program for cuda_graph) -- a decline `_run_tier` never
# sees as an exception, so the pre-fix query named a tier that never actually ran. Every one
# of today's four registered `pixel_args=` builtins (gauss_blur/erode/dilate/bilateral_filter)
# hits ONE of these two declines unconditionally, on ANY box, scale-active or not -- this is
# not a scale-specific bug, the SCALECX-49 audit (B2) just happened to be what found it.
_GAUSS_BLUR_CODE = "@OUT = gauss_blur(@A, 6.0);\n"
_GAUSS_BLUR_BT = {"A": TEXType.VEC4}


def _gauss_blur_fingerprint():
    """The exact fingerprint `tier_verdict`'s own `_compile_for_query` derives for
    `_GAUSS_BLUR_CODE`/`_GAUSS_BLUR_BT` -- so a test can drive `fncalls_compile`'s memo
    for precisely the fingerprint the query itself will look up."""
    return _get_cache().fingerprint(_GAUSS_BLUR_CODE, _GAUSS_BLUR_BT)


def test_tierq48_torch_compile_predicts_graph_break_for_a_known_bad_fingerprint():
    """COMPILETRY-50 (D1, item 3): once a REAL cook's one remembered fall-through attempt
    has exhausted every backend for this exact fingerprint (`fncalls_compile.verdict`
    resolves False), the query must keep predicting the codegen-only fallback -- it must
    not go stale the moment `_has_fn_calls` alone stops being the whole story."""
    fp = _gauss_blur_fingerprint()
    _fncalls_compile.reset_for_test()
    _fncalls_compile._memo[fp] = False
    try:
        v = tier_verdict(_GAUSS_BLUR_CODE, compile_mode="torch_compile", device="cpu",
                         binding_types=_GAUSS_BLUR_BT)
        assert v.tier == "codegen" and v.reason == TIER_REASON_TORCH_COMPILE_GRAPH_BREAK
    finally:
        _fncalls_compile.reset_for_test()


def test_tierq48_torch_compile_reports_declared_tier_for_an_unresolved_fingerprint():
    """COMPILETRY-50 (D1, item 3): a fingerprint that has never been through
    `_try_compile` (the common case -- `tier_verdict` itself never triggers an attempt,
    it only reads the memo) is UNKNOWN, not a known break -- gauss_blur/erode/dilate
    already measured clean by `torch.compile` on this box's own evidence, so reporting
    the declared tier unchanged is the more accurate of the two guesses. This is RED
    against the pre-COMPILETRY-50 code (which predicted "codegen" unconditionally, the
    exact assertion this test's predecessor pinned) and GREEN at head."""
    fp = _gauss_blur_fingerprint()
    _fncalls_compile.reset_for_test()
    assert _fncalls_compile.verdict(fp) is None   # never attempted -- the case under test
    v = tier_verdict(_GAUSS_BLUR_CODE, compile_mode="torch_compile", device="cpu",
                     binding_types=_GAUSS_BLUR_BT)
    assert v.tier == "torch_compile" and v.reason == TIER_REASON_SELECTED


def test_tierq48_torch_compile_reports_declared_tier_for_a_known_good_fingerprint():
    """COMPILETRY-50 (D1, item 3): once a real cook's one remembered attempt has PROVED
    this fingerprint compiles for real (`verdict is True`), the query must report the
    declared compiled tier, never "codegen" -- the truthfulness half item 3 names
    explicitly ("after a success, the compiled tier")."""
    fp = _gauss_blur_fingerprint()
    _fncalls_compile.reset_for_test()
    _fncalls_compile._memo[fp] = True
    try:
        v = tier_verdict(_GAUSS_BLUR_CODE, compile_mode="torch_compile", device="cpu",
                         binding_types=_GAUSS_BLUR_BT)
        assert v.tier == "torch_compile" and v.reason == TIER_REASON_SELECTED
    finally:
        _fncalls_compile.reset_for_test()


def test_tierq48_auto_predicts_graph_break_for_a_known_bad_fingerprint():
    fp = _gauss_blur_fingerprint()
    _fncalls_compile.reset_for_test()
    _fncalls_compile._memo[fp] = False
    try:
        v = tier_verdict(_GAUSS_BLUR_CODE, compile_mode="auto", device="cpu",
                         binding_types=_GAUSS_BLUR_BT)
        assert v.tier == "codegen" and v.reason == TIER_REASON_TORCH_COMPILE_GRAPH_BREAK
    finally:
        _fncalls_compile.reset_for_test()


def test_tierq48_cuda_graph_predicts_not_capturable_for_a_real_pixel_args_builtin():
    """Red at 32f6917: the query said "cuda_graph" here; gauss_blur is `sync=True` in
    `graphed._SYNC_STDLIB`, so `_capturable` always declines it -- CPU-provable (`_capturable`
    is a pure AST walk, no real CUDA device needed, same fact
    `test_tierq48_agrees_scale_active_cuda_graph_reported_without_real_gpu` already leans on)."""
    v = tier_verdict(_GAUSS_BLUR_CODE, compile_mode="cuda_graph", device="cuda:0",
                     binding_types=_GAUSS_BLUR_BT)
    assert v.tier == "interpreter" and v.reason == TIER_REASON_CUDA_GRAPH_NOT_CAPTURABLE


def test_tierq48_scale_active_torch_compile_predicts_graph_break_for_a_known_bad_fingerprint():
    """Same correction applies in the SCALE-ACTIVE branch too -- it must not drift from the
    scale-inactive branch above (`_real_compiled_dispatch` is the one shared call site). The
    fall-through memo is keyed the same regardless of scale (SCALECX-49: `scale` is not part
    of the compiled-callable cache key), so the same fingerprint applies."""
    fp = _gauss_blur_fingerprint()
    _fncalls_compile.reset_for_test()
    _fncalls_compile._memo[fp] = False
    try:
        v = tier_verdict(_GAUSS_BLUR_CODE, compile_mode="torch_compile", device="cpu",
                         scale=0.5, binding_types=_GAUSS_BLUR_BT)
        assert v.tier == "codegen" and v.reason == TIER_REASON_TORCH_COMPILE_GRAPH_BREAK
    finally:
        _fncalls_compile.reset_for_test()


def test_tierq48_scale_active_cuda_graph_predicts_not_capturable_for_a_real_pixel_args_builtin():
    v = tier_verdict(_GAUSS_BLUR_CODE, compile_mode="cuda_graph", device="cuda:0", scale=0.5,
                     binding_types=_GAUSS_BLUR_BT)
    assert v.tier == "interpreter" and v.reason == TIER_REASON_CUDA_GRAPH_NOT_CAPTURABLE


_GAUSS_BLUR_CODE_UNCOMPILABLE_AGAINST_BAD_BT = "@OUT = gauss_blur(@A.rgb, 6.0);\n"
_BAD_BT = {"A": TEXType.FLOAT}   # FLOAT has no `.rgb` swizzle -- fails to compile


def test_tierq48_torch_compile_reports_declared_tier_when_it_cannot_compile():
    """When `code` cannot be compiled against the given `binding_types` (here, a binding
    whose declared type disagrees with how it is used -- the same shape
    `test_tierq48_scale_active_codegen_route_falls_back_when_it_cannot_compile` uses above),
    the query cannot check the graph-break predicate and conservatively reports the
    DECLARED tier unchanged: never wrong about pixel correctness, only potentially
    optimistic about which tier is named."""
    v = tier_verdict(_GAUSS_BLUR_CODE_UNCOMPILABLE_AGAINST_BAD_BT,
                     compile_mode="torch_compile", device="cpu", binding_types=_BAD_BT)
    assert v.tier == "torch_compile" and v.reason == TIER_REASON_SELECTED


def test_tierq48_cuda_graph_reports_declared_tier_when_it_cannot_compile():
    v = tier_verdict(_GAUSS_BLUR_CODE_UNCOMPILABLE_AGAINST_BAD_BT,
                     compile_mode="cuda_graph", device="cuda:0", binding_types=_BAD_BT)
    assert v.tier == "cuda_graph" and v.reason == TIER_REASON_SELECTED


def test_tierq48_never_raises_on_a_malformed_roi():
    """Contract: `tier_verdict` never raises — a malformed window is a declined reason,
    not a `TypeError`/`ValueError` escaping to the caller."""
    v = tier_verdict(_ROI_CODE, compile_mode="none", device="cpu",
                     roi=("not", "a", "window"), roi_exec=True)
    assert v.tier == "default"
    assert not v.roi_armed
    assert v.roi_reason is not None and v.roi_reason.startswith("roi-declined")


# ── FIX-TIER T1: extend the agreement test to the THIRD former copy ──────────
#
# `tex_chain.cook_stage_list`'s own ROI gate used to be a third, independently-maintained
# hand-copy of this exact ladder (R1 finding 1) with its own ad hoc reason strings (R2
# finding 2), pinned against neither `prepare()` nor `tier_verdict`. Both now call the
# same `tex_roi.roi_eligibility` `tier_verdict` calls, so this row closes the actual gap
# the two findings named: nothing previously cross-checked `cook_stage_list`'s verdict
# against either of the other two.
def _cook_stage_list_roi_armed(code, bindings, *, roi, roi_exec=None, scale=None,
                              latent_channel_count=0):
    """Whether `cook_stage_list` actually narrowed to the window: observed by counting
    calls into `tex_memory.run_roi` (the only place a window is ever actually cooked),
    the same "did the real mechanism fire" signal `_real_tier_and_roi_armed` reads off
    `tier_trace` for the `tex_engine.prepare()` path — `cook_stage_list`'s own
    `tier_trace.record_roi` call always passes `cooked_roi=None` regardless of arming
    (a pre-existing quirk, unrelated to and unchanged by this ask), so that signal alone
    cannot distinguish armed from declined here."""
    from TEX_Wrangle import tex_chain as _tex_chain
    from TEX_Wrangle import tex_memory as _tex_memory
    _tex_roi.clear_roi_memo()
    calls = {"n": 0}
    real_run_roi = _tex_memory.run_roi

    def _counting_run_roi(*a, **kw):
        calls["n"] += 1
        return real_run_roi(*a, **kw)

    _tex_memory.run_roi = _counting_run_roi
    try:
        _tex_chain.cook_stage_list(
            [{"code": code, "bindings": dict(bindings)}], device="cpu", roi=roi,
            roi_exec=roi_exec, scale=scale, latent_channel_count=latent_channel_count)
    finally:
        _tex_memory.run_roi = real_run_roi
    return calls["n"] > 0


def test_tierq48_cook_stage_list_roi_armed_agrees_with_tier_verdict():
    A = torch.rand(1, 1024, 1024, 4)
    roi = (10, 10, 256, 256, 1024, 1024)
    real_armed = _cook_stage_list_roi_armed(_ROI_CODE, {"A": A, "amount": 0.4},
                                            roi=roi, roi_exec=True)
    v = tier_verdict(_ROI_CODE, compile_mode="none", device="cpu", roi=roi,
                     roi_exec=True, param_values={"amount": 0.4})
    assert v.roi_armed == real_armed
    assert real_armed and v.roi_reason == ROI_REASON_ARMED


def test_tierq48_cook_stage_list_roi_declines_when_not_armed_agrees_with_tier_verdict():
    A = torch.rand(1, 1024, 1024, 4)
    roi = (10, 10, 256, 256, 1024, 1024)
    real_armed = _cook_stage_list_roi_armed(_ROI_CODE, {"A": A, "amount": 0.4},
                                            roi=roi, roi_exec=False)
    v = tier_verdict(_ROI_CODE, compile_mode="none", device="cpu", roi=roi,
                     roi_exec=False, param_values={"amount": 0.4})
    assert v.roi_armed == real_armed == False
    assert v.roi_reason == ROI_REASON_NOT_ARMED


def test_tierq48_cook_stage_list_roi_reason_agrees_when_precision_and_executability_both_fail():
    """Red-first at `5ae6288`: `cook_stage_list`'s OLD ROI ladder checked
    `eff_precision != "fp32"` BEFORE ever computing `roi_plan(...).executable` —
    `tex_engine.prepare()`/`tier_verdict` check `roi_plan` first and only consult
    precision once the program is confirmed executable. Both orderings decline the SAME
    window (never a wrong-pixel divergence — `roi_armed` is `False` either way), but for
    a program that is BOTH non-fp32 AND not ROI-executable, the OLD `cook_stage_list`
    reported a precision-flavored reason while `tier_verdict` reported
    `ROI_REASON_NOT_EXECUTABLE` for the identical inputs — exactly the "third copy can
    silently disagree" risk R1/R2 named. `@A(u, v)` (`BindingSampleAccess`, a whole-image
    sample) is not ROI-executable regardless of precision."""
    import torch as _torch
    from TEX_Wrangle import tex_chain as _tex_chain
    from TEX_Wrangle.tex_runtime import tier_trace as _tt
    code = "@OUT = vec4(@A(u, v).rgb, 1.0);\n"
    A = _torch.rand(1, 64, 64, 4)
    roi = (5, 5, 20, 20, 64, 64)
    _tex_roi.clear_roi_memo()
    _tt.reset()
    _tex_chain.cook_stage_list([{"code": code, "bindings": {"A": A}}], device="cpu",
                              precision="fp16", roi=roi, roi_exec=True)
    _, real_reason_text = _tt.last_roi()
    v = tier_verdict(code, compile_mode="none", device="cpu", roi=roi, roi_exec=True,
                     precision="fp16")
    from TEX_Wrangle.tex_engine_tiers import ROI_REASON_NOT_EXECUTABLE
    assert v.roi_reason == ROI_REASON_NOT_EXECUTABLE
    assert real_reason_text is not None and "not ROI-executable" in real_reason_text, (
        f"cook_stage_list reported {real_reason_text!r} -- expected a not-executable "
        f"reason agreeing with tier_verdict's {v.roi_reason!r}, not a precision-flavored "
        "one (the pre-fix reason-ordering bug)")


def test_tierq48_cook_stage_list_roi_declines_on_latent_agrees_with_tier_verdict():
    """`cook_stage_list`'s `has_latent_input` proxy is `bool(latent_channel_count)` —
    `tier_verdict`'s own `has_latent_input=True` names the same condition."""
    A = torch.rand(1, 1024, 1024, 4)
    roi = (10, 10, 256, 256, 1024, 1024)
    real_armed = _cook_stage_list_roi_armed(_ROI_CODE, {"A": A, "amount": 0.4}, roi=roi,
                                            roi_exec=True, latent_channel_count=4)
    v = tier_verdict(_ROI_CODE, compile_mode="none", device="cpu", roi=roi,
                     roi_exec=True, param_values={"amount": 0.4}, has_latent_input=True)
    assert v.roi_armed == real_armed == False
    from TEX_Wrangle.tex_engine_tiers import ROI_REASON_LATENT
    assert v.roi_reason == ROI_REASON_LATENT
