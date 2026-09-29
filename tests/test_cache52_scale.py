"""The OOM ladder refuses a scale-active cook; the scale walk sees iw / ih / px / py."""
from types import SimpleNamespace

import pytest
import torch

from TEX_Wrangle import tex_roi


def _ladder_ctx(scale):
    return SimpleNamespace(fused_chain=None, latent_channel_count=0, device="cuda:0",
                           scale=scale, bindings={"A": torch.zeros(1, 256, 64, 4)},
                           program=None, fp="f", code="@OUT = @A;", binding_types={},
                           type_map={}, output_names=["OUT"], used_builtins=set(),
                           eff_precision="fp32", time_context=None, cancel=None,
                           on_progress=None)


def test_oom_ladder_declines_a_scale_active_cook(monkeypatch):
    from TEX_Wrangle import tex_engine, tex_memory
    calls = []
    monkeypatch.setattr(tex_memory, "shared_tile_height", lambda b: 256)
    monkeypatch.setattr(tex_memory, "is_tile_safe_cached", lambda p, f: True)
    monkeypatch.setattr(tex_memory, "run_tiled", lambda *a, **k: calls.append(1) or {"OUT": 1})
    monkeypatch.setattr(tex_roi, "region_dependent_cached", lambda *a, **k: False)
    monkeypatch.setattr(tex_engine, "_drop_tex_caches_on_oom", lambda: None)
    err = RuntimeError("out of memory")
    assert tex_engine._oom_retry(_ladder_ctx(0.5), err, err) is None
    assert not calls
    assert tex_engine._oom_retry(_ladder_ctx(None), err, err) == {"OUT": 1}   # unscaled: unchanged


@pytest.mark.parametrize("name", ["iw", "ih", "px", "py"])
def test_scale_walk_flags_the_pixel_grid_identifiers(name):
    assert tex_roi.scale_safe(f"float w = {name} * 100.0; @OUT = vec4(w);") is False
