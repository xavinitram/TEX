"""Frame-cache edges: window bounds, requalify of a non-frame, the disk-total reconcile racing a
spill, the per-device environment epoch, and the shared discard of an unservable spill file."""
import os
import tempfile

import torch

from helpers import *  # noqa: F401,F403
from TEX_Wrangle import tex_results, tex_results_keys as RK


def test_patch_region_refuses_a_window_outside_the_frame():
    c = tex_results.ResultCache()
    base = torch.rand(1, 32, 32, 4)
    c.put("base", base)
    patch = torch.zeros(1, 8, 8, 4)
    for win in ((28, 4, 8, 8, 32, 32), (4, 28, 8, 8, 32, 32),
                (-4, 4, 8, 8, 32, 32), (4, -4, 8, 8, 32, 32)):
        assert c.patch_region("out", patch, win, base_key="base") is None, win
    assert torch.equal(c.get("base"), base)
    ok = c.patch_region("out", patch, (24, 24, 8, 8, 32, 32), base_key="base")
    assert ok is not None and float(ok[0, 31, 31, 0]) == 0.0


def test_requalify_of_a_non_tensor_keeps_the_preview():
    c = tex_results.ResultCache()
    c.put("prev", torch.rand(1, 4, 4, 4))
    assert c.requalify("prev", "final", None) is False
    assert c.requalify("prev", "final", [[1.0]]) is False
    assert c.get("prev") is not None and c.get("final") is None
    assert c.requalified == 0


def test_disk_reconcile_does_not_overwrite_a_concurrent_spill_total(monkeypatch):
    d = tempfile.mkdtemp()
    c = tex_results.ResultCache(budget_mb=64, cache_dir=d)
    c._disk_bytes = None
    real = os.scandir

    def scan(path):
        it = real(path)
        with c._lock:                     # a spill lands its locked update mid-scan
            c._disk_mut += 1
            c._disk_bytes = 123
        return it
    monkeypatch.setattr(os, "scandir", scan)
    c._enforce_disk_budget()
    monkeypatch.undo()
    assert c._disk_bytes is None          # uncertain, never the stale scanned total


def test_disk_reconcile_records_the_scanned_total_when_nothing_raced():
    d = tempfile.mkdtemp()
    c = tex_results.ResultCache(budget_mb=64, cache_dir=d)
    c._disk_bytes = None
    c._enforce_disk_budget()
    assert c._disk_bytes == 0


def _fake_gpus(monkeypatch, current):
    monkeypatch.setattr(RK, "_ENV_EPOCH_CACHE", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: f"GPU{i}")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (8, i))


def test_env_epoch_follows_the_frames_device_not_the_ambient_one(monkeypatch):
    _fake_gpus(monkeypatch, current=0)
    assert "GPU1" in RK.env_epoch("cuda:1")
    assert "GPU1" in RK.env_epoch(torch.device("cuda", 1))
    assert "GPU0" in RK.env_epoch("cuda:0") and "GPU1" not in RK.env_epoch("cuda:0")
    assert "GPU0" in RK.env_epoch()                     # no device: the ambient one, as before
    assert "GPU" not in RK.env_epoch("cpu")
    assert RK.env_epoch("cuda:1") != RK.env_epoch("cuda:0")


def test_lineage_key_folds_the_named_device_identity(monkeypatch):
    _fake_gpus(monkeypatch, current=0)
    kw = dict(program_fp="fp", precision="fp32")
    a = RK.lineage_key(device="cuda:1", **kw)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(RK, "_ENV_EPOCH_CACHE", {})
    assert RK.lineage_key(device="cuda:1", **kw) == a   # the ambient device does not move it


def test_unservable_spill_file_is_removed_and_forgotten():
    d = tempfile.mkdtemp()
    c = tex_results.ResultCache(budget_mb=64, cache_dir=d)
    p = os.path.join(d, "x.frame")
    open(p, "wb").write(b"junk")
    c._disk_bytes, c._spilled = 4, {"k", "other"}
    before = c._disk_mut
    c._discard_frame_file(p, "k")
    assert not os.path.exists(p)
    assert c._disk_bytes is None and c._spilled == {"other"} and c._disk_mut == before + 1
    c._disk_bytes = 9
    c._discard_frame_file(p, "k")                       # already gone: nothing is invalidated
    assert c._disk_bytes == 9
