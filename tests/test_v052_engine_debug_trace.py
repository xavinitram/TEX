"""The near-singularity diagnostic describes only the attempt that is returned, and is
reported as unknown (None) when the tier that served the cook has no guard hooks."""
import torch

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import guard_trace, tier_trace

_SING = "@OUT = vec4(vec3(sdiv(1.0, @A.r - @A.r)), 1.0);"


def _arm_with_a_stale_attempt():
    guard_trace.arm()
    guard_trace.note(torch.ones(1, 4, 4, dtype=torch.bool))
    assert guard_trace.count() == 16


def test_finiteness_recook_discards_the_fp16_attempts_guard_counts(monkeypatch):
    plan = tex_engine.prepare("@OUT = @A * 1.0;", {"A": torch.ones(1, 4, 4, 4)},
                              precision="fp16", device_mode="cpu")
    monkeypatch.setattr(tex_engine, "_run_tier",
                        lambda ctx, tier_id: {"OUT": torch.ones(1, 4, 4, 4)})
    _arm_with_a_stale_attempt()
    try:
        tex_engine._fp16_finiteness_net({"OUT": torch.full((1, 4, 4, 4), float("inf"))},
                                        True, plan.ctx, plan.tier_id, plan.auto_ckey)
        assert guard_trace.count() == 0 and guard_trace.mask() is None
        assert guard_trace.armed(), "the re-cook must stay armed"
    finally:
        guard_trace.disarm()


def test_oom_retry_discards_the_failed_attempts_guard_counts():
    plan = tex_engine.prepare("@OUT = @A * 1.0;", {"A": torch.ones(1, 4, 4, 4)},
                              device_mode="cpu")
    oom = torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
    _arm_with_a_stale_attempt()
    try:
        tex_engine._oom_retry(plan.ctx, oom, oom)
        assert guard_trace.count() == 0 and guard_trace.mask() is None
    finally:
        guard_trace.disarm()


def test_near_singularities_is_none_when_the_serving_tier_is_not_instrumented(monkeypatch):
    A = torch.rand(1, 8, 8, 3)
    res = tex_engine.cook(_SING, {"A": A}, device_mode="cpu", debug_nan_highlight=True)
    assert isinstance(res.near_singularities, int) and res.near_singularities > 0, \
        "the interpreter tier must keep counting"

    real = tex_engine._run_tier

    def _served_by_codegen(ctx, tier_id):
        out = real(ctx, tier_id)
        tier_trace.record("codegen")        # what the compiled tiers record on success
        return out
    monkeypatch.setattr(tex_engine, "_run_tier", _served_by_codegen)
    res = tex_engine.cook(_SING, {"A": A}, device_mode="cpu", debug_nan_highlight=True)
    assert res.near_singularities is None, res.near_singularities
    # and with the toggle off it is None as always
    res = tex_engine.cook(_SING, {"A": A}, device_mode="cpu")
    assert res.near_singularities is None
