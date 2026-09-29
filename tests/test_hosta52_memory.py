"""tex_memory: peak estimate for inferred-size arrays, LRU memo race tolerance, weak arming,
the governor's O(1) stdlib reader, and the shared positive-MiB env parser."""
import gc
import logging
import weakref
from collections import OrderedDict

import pytest

from TEX_Wrangle import tex_memory as M
from TEX_Wrangle import tex_results
from TEX_Wrangle.tex_cache import parse_and_split


def _peak(src):
    prog = parse_and_split(src, {})
    return M.estimate_peak_bytes(prog, (1, 256, 256), 4)


def test_inferred_size_array_counts_like_a_declared_one():
    body = "vec3 c = vec3(0.0); c.r = a[1]; @OUT = vec4(c, 1.0);"
    inferred = _peak("float a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}; " + body)
    declared = _peak("float a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}; " + body)
    assert inferred == declared
    assert declared > _peak(body.replace("a[1]", "0.0"))


class _EvictedOnTouch(OrderedDict):
    def move_to_end(self, key, last=True):
        raise KeyError(key)          # another thread evicted the key after our get


def test_lru_hit_path_tolerates_a_concurrent_eviction(monkeypatch):
    prog = parse_and_split("@OUT = vec4(1.0);", {})
    memo = _EvictedOnTouch()
    monkeypatch.setattr(M, "_tile_safe_memo", memo)
    assert M.is_tile_safe_cached(prog, "fp-x") in (True, False)     # fills the memo
    assert M.is_tile_safe_cached(prog, "fp-x") in (True, False)     # the hit path
    pm = _EvictedOnTouch()
    monkeypatch.setattr(M, "_peak_static_memo", pm)
    assert len(M._estimate_peak_statics_cached(prog, "fp-y")) == 4
    assert len(M._estimate_peak_statics_cached(prog, "fp-y")) == 4


def test_governor_does_not_keep_a_dropped_result_cache_alive():
    cache = tex_results.ResultCache()
    M.register_result_cache(cache, name="hosta52-test")
    M.get_cache_registry().unregister("hosta52-test")
    ref = weakref.ref(cache)
    del cache
    gc.collect()
    assert ref() is None


def test_governor_stdlib_pool_reads_the_running_total(monkeypatch):
    walked = []
    monkeypatch.setattr(M, "_total_cache_bytes", lambda *a: walked.append(1) or 0)
    M.get_cache_registry().stats("cpu")
    assert walked == []


@pytest.mark.parametrize("fn,var", [(M.cache_budget_bytes, "TEX_CACHE_BUDGET_MB"),
                                     (M.governor_budget, "TEX_GOVERNOR_BUDGET_MB")])
def test_budget_env_overrides_share_one_positive_mib_rule(monkeypatch, caplog, fn, var):
    monkeypatch.setenv(var, "64")
    assert fn("cpu") == 64 * 1024 * 1024
    for bad in ("0", "-8", "1.5", "lots"):
        monkeypatch.setenv(var, bad)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="TEX"):
            got = fn("cpu")
        assert got > 0 and got != 64 * 1024 * 1024
        assert var in caplog.text
