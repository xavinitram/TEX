"""v0.52 sweep: frame-provider pool rows (version bumps, budget edge cases, uniform time,
prefetch windows)."""
import math
import sys
import threading

import pytest
import torch

from TEX_Wrangle import tex_cookqueue as Q, tex_provider as TP


class _Plain:
    """The smallest legal provider: fetch/sample only, no rate, no quantize_time."""
    provider_id = "s52-plain"

    def __init__(self, on_fetch=None):
        self.on_fetch = on_fetch
        self.fetches = 0

    def fetch_time(self, source_key, t):
        self.fetches += 1
        if self.on_fetch is not None:
            self.on_fetch()
        return torch.ones(1, 4, 4, 4)

    sample_time = fetch_time


@pytest.fixture(autouse=True)
def _clean():
    """A fresh media pool before and after: its counters are process-wide, and other files
    assert absolute values on them."""
    TP.reset_provider()
    TP._versions.clear()
    TP._media_cache = None
    yield
    TP.reset_provider()
    TP._versions.clear()
    TP._media_cache = None


def test_concurrent_bumps_are_not_lost():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        n_threads, n_bumps = 4, 3000

        def work():
            for _ in range(n_bumps):
                TP.bump_source_version("plate")

        threads = [threading.Thread(target=work) for _ in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    finally:
        sys.setswitchinterval(old)
    assert TP._versions["plate"] == n_threads * n_bumps


def test_a_frame_fetched_across_a_version_bump_is_not_pooled():
    prov = _Plain(on_fetch=lambda: TP.bump_source_version("plate"))
    TP.set_provider(prov)
    frame = TP.materialize("plate", 0.0, "fetch")
    assert frame is not None and prov.fetches == 1
    assert TP.get_media_cache().stats()["frames"] == 0
    prov.on_fetch = None
    TP.materialize("plate", 0.0, "fetch")
    assert TP.get_media_cache().stats()["frames"] == 1


def test_a_zero_budget_caches_nothing():
    prov = _Plain()
    TP.set_provider(prov)
    TP.materialize("plate", 0.0, "fetch")
    TP.set_media_budget_mb(0.0)
    assert TP.get_media_cache().stats()["frames"] == 0
    for _ in range(3):
        assert TP.materialize("plate", 1.0, "fetch") is not None
    st = TP.get_media_cache().stats()
    assert st["frames"] == 0 and st["bytes"] == 0
    assert TP.materialize("plate", 1.0, "fetch", speculative=True) is None
    assert prov.fetches == 1 + 3 + 1


def test_an_oversized_frame_does_not_flush_the_pool():
    cache = TP.MediaCache(budget_mb=1.0)
    small = torch.zeros(1024)                       # 4 KiB
    assert cache.put(("a",), small)
    assert not cache.put(("big",), torch.zeros(512 * 1024))      # 2 MiB > the whole budget
    assert cache.get(("a",)) is small
    assert cache.stats()["frames"] == 1


def test_uniform_time_refuses_an_empty_tensor_and_accepts_a_uniform_nan_grid():
    with pytest.raises(Exception) as ei:
        TP._uniform_time(torch.zeros(0), "plate")
    assert getattr(ei.value, "_code", "") == "E7003", ei.value
    assert math.isnan(TP._uniform_time(torch.full((4, 4), float("nan")), "plate"))
    assert TP._uniform_time(torch.full((4, 4), 2.5), "plate") == 2.5
    with pytest.raises(Exception) as ei:
        TP._uniform_time(torch.tensor([1.0, 2.0]), "plate")
    assert getattr(ei.value, "_code", "") == "E7003"


def test_declare_window_never_guesses_a_frame_spacing():
    q = Q.CookQueue(name="s52-window")
    try:
        q.install_policy(Q.SpeculativePolicy(min_confidence=0.0, min_value_ms=0.0,
                                             unknown_min_confidence=0.0, max_pending=16))
        TP.set_provider(_Plain())
        with pytest.raises(ValueError):
            TP.declare_window(q, "plate", 0.0, 10.0)
        jobs = TP.declare_window(q, "plate", 0.0, 3.0, step=1.0, confidence=0.9)
        assert len(jobs) == 4
        assert len(TP.declare_window(q, "plate", 2.0, 2.0)) == 1
        with pytest.raises(ValueError):
            TP.declare_window(q, "plate", 0.0, 3.0, step=0.0)
        q.drain(10)
    finally:
        q.close()
