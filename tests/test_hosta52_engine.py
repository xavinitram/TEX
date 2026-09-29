"""Engine cook-path fixes: the fp16 finiteness net's fp32 re-cook uses the OOM ladder, and a
fused chain's auto-precision decision is memoized like a single program's."""
import pytest
import torch

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import precision_policy


def test_fp32_recook_after_the_finiteness_net_gets_the_oom_ladder(monkeypatch):
    plan = tex_engine.prepare("@OUT = @A * 1.0;", {"A": torch.ones(1, 4, 4, 4)},
                              precision="fp16", device_mode="cpu")

    def _oom(ctx, tier_id):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
    monkeypatch.setattr(tex_engine, "_run_tier", _oom)
    bad = {"OUT": torch.full((1, 4, 4, 4), float("inf"))}
    with pytest.raises(torch.cuda.OutOfMemoryError) as ei:
        tex_engine._fp16_finiteness_net(bad, True, plan.ctx, plan.tier_id, plan.auto_ckey)
    assert getattr(ei.value, "tex_refusal", None) is not None, \
        "the re-cook's out-of-memory error must carry the engine refusal like any other cook's"


def test_fused_chain_auto_precision_is_decided_once(monkeypatch):
    calls = []
    real = precision_policy.resolve_auto_precision

    def _counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(precision_policy, "resolve_auto_precision", _counting)
    spec = {"stages": [{"code": "@OUT = @IN * 1.25;", "image_input": "IN", "params": {}}],
            "terminal_image_input": "IN"}
    term = "@OUT = @IN + 0.125;"
    src = torch.rand(1, 16, 16, 3)
    for _ in range(3):
        tex_engine.cook(term, {"IN": src}, chain_payload=spec, device_mode="cpu",
                        precision="auto")
    assert len(calls) == 1
