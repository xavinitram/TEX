"""A spilled result's environment identity carries the full torch tag and the cook-pipeline files."""
import tempfile
from pathlib import Path

import torch

from TEX_Wrangle import tex_results_keys as RK


def test_env_epoch_carries_full_torch_tag_and_engine_files(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "9.9.9+cu999")
    RK._ENV_EPOCH_CACHE.clear()
    try:
        assert "9.9.9+cu999" in RK.env_epoch()
        d = Path(tempfile.mkdtemp())
        f = d / "engine_stand_in.py"
        f.write_text("a = 1\n")
        monkeypatch.setattr(RK, "_RESULT_FILES", (f,))
        RK._RESULT_EPOCH.clear()
        RK._ENV_EPOCH_CACHE.clear()
        before = RK.env_epoch()
        f.write_text("a = 2\n")
        RK._RESULT_EPOCH.clear()
        RK._ENV_EPOCH_CACHE.clear()
        assert RK.env_epoch() != before
    finally:
        RK._RESULT_EPOCH.clear()
        RK._ENV_EPOCH_CACHE.clear()


def test_result_files_name_the_cook_pipeline():
    names = {p.name for p in RK._RESULT_FILES}
    assert {"tex_engine.py", "tex_roi.py", "tex_tiling.py", "tex_marshalling.py"} <= names
    assert all(p.exists() for p in RK._RESULT_FILES)
