"""Compile-cache behaviour: shared LRU safety, best-effort touch, eviction, clear_all, and
that a warm disk hit answers exactly what the cold compile did."""
import os
import sys
import tempfile
import threading
from pathlib import Path

import torch

from helpers import *  # noqa: F401,F403
from TEX_Wrangle import tex_cache as TC
from TEX_Wrangle.tex_cache import TEXCache


def _dir():
    return Path(tempfile.mkdtemp())


def test_disk_hit_survives_a_failing_lru_touch(monkeypatch):
    d = _dir()
    bt = {"A": TEXType.VEC4}
    code = "@OUT = @A * 0.5;"
    cold = TEXCache(cache_dir=d).compile_tex(code, bt)
    fresh = TEXCache(cache_dir=d)
    fp = fresh.fingerprint(code, bt)

    def _deny(*a, **k):
        raise PermissionError("read-only cache dir")
    monkeypatch.setattr(os, "utime", _deny)
    hit = fresh._load_from_disk(fp, bt)
    assert hit is not None, "a failed atime touch must not discard a verified entry"
    assert (d / f"{fp}.pkl").exists()
    assert hit[2] == cold[2] and hit[3] == cold[3] and hit[4] == cold[4]


def test_cold_compile_and_disk_hit_execute_identically():
    d = _dir()
    bt = {"A": TEXType.VEC4}
    code = "float k = 0.25; @OUT = vec4(@A.rgb * k + 0.1, @A.a);"
    img = torch.rand(1, 8, 8, 4)
    outs = []
    for _ in range(2):        # second construction reads the .pkl the first one wrote
        prog, tm, refs, asg, params, used = TEXCache(cache_dir=d).compile_tex(code, bt)
        outs.append((Interpreter().execute(prog, {"A": img}, tm, device="cpu",
                                           output_names=sorted(asg))["OUT"], refs, asg, params))
    assert torch.equal(outs[0][0], outs[1][0])
    assert outs[0][1:] == outs[1][1:]


def test_pkl_payload_carries_only_what_the_loader_reads():
    from TEX_Wrangle.tex_recovery import load_verified
    d = _dir()
    bt = {"A": TEXType.VEC4}
    c = TEXCache(cache_dir=d)
    c.compile_tex("@OUT = @A;", bt)
    data = load_verified(next(d.glob("*.pkl")))
    assert set(data) == {"version", "program", "sets"}


def test_eviction_skips_an_entry_that_vanished_after_the_glob(monkeypatch):
    d = _dir()
    monkeypatch.setattr(TC, "_DISK_MAX_ENTRIES", 2)
    for i in range(4):
        (d / f"{i:02d}.pkl").write_bytes(b"x")
        os.utime(d / f"{i:02d}.pkl", (100 + i, 100 + i))
    c = TEXCache(cache_dir=d)
    real_glob = Path.glob

    def _glob_with_ghost(self, pat):
        out = list(real_glob(self, pat))
        if pat == "*.pkl":
            out.append(d / "ghost.pkl")          # unlinked by another process after the glob
        return out
    monkeypatch.setattr(Path, "glob", _glob_with_ghost)
    c._evict_disk_if_needed()
    left = sorted(p.name for p in d.glob("*.pkl") if p.exists())
    assert len(left) == 2, left


def test_clear_all_removes_the_persisted_verdict_files():
    d = _dir()
    names = ["autotier.json", "xfer.json", "warm_state.json", "warm_state.json.journal", "a.pkl"]
    for n in names:
        (d / n).write_text("x")
    TEXCache(cache_dir=d).clear_all()
    assert [n for n in names if (d / n).exists()] == []


def test_memory_tiers_survive_concurrent_get_and_evict(monkeypatch):
    monkeypatch.setattr(TC, "_MEMORY_MAX_ENTRIES", 4)
    monkeypatch.setattr(TC, "_CODEGEN_MEMORY_MAX_ENTRIES", 4)
    c = TEXCache(cache_dir=_dir())
    errors = []
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    def hammer(seed):
        try:
            for i in range(4000):
                fp = f"k{(i * 7 + seed) % 16}"
                c._memory_put(fp, (i,))
                c._codegen_memory_put(fp, i)
                c.get("x", {}, fp=fp)
                c.get_codegen_fn(fp)
        except BaseException as e:      # noqa: BLE001 - the point is to see any escape
            errors.append(e)

    try:
        ts = [threading.Thread(target=hammer, args=(s,)) for s in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        sys.setswitchinterval(old)
    assert errors == [], errors[:1]
