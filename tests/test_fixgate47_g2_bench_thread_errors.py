"""FIX-GATE G2 (v0.47.0 Phase C, B4#5) -- a benchmark's background thread must never let a
genuine crash pass as a quiet, on-time finish.

`benchmarks/preempt_drain_bench.py::_reap_background_cook` and the writer/fetcher joins in
`benchmarks/io_playback_bench.py::measure_overlapped` raise a hard failure ONLY when the
background thread is still alive after its join timeout -- the "still hung" shape BENCH-47
was originally filed for. An exception OTHER than `CookCancelled` inside a `_bg()` closure
(a genuine bug in the cook path) terminates the thread almost immediately: Python prints
"Exception in thread ..." to stderr and the thread simply ends, so `is_alive()` reads
`False` well within the timeout and the hard-fail check finds nothing wrong -- the trial's
timing numbers are recorded as if the background cook ran to its expected point, with no
error, warning, or nonzero exit code anywhere. This is worse than a hang: a hang at least
stops the run. This pins that the exception is now captured and re-raised at the join
point, not swallowed.
"""
import importlib.util
import os
import sys
import threading
import time

import pytest

from helpers import SubTestResult

_BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "benchmarks")


def _load(modname: str, filename: str):
    """Load a `benchmarks/*.py` file by path, once per process -- same technique
    `benchmarks/artist_loops_bench.py._load` and `tests/helpers.py.load_counts_harness`
    already use for this not-a-package directory."""
    key = f"_fixgate47_g2_{modname}"
    mod = sys.modules.get(key)
    if mod is not None:
        return mod
    path = os.path.join(_BENCH_DIR, filename)
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


# ── preempt_drain_bench: _start_background_cook / _reap_background_cook ────────────

def test_fixgate_g2_reap_background_cook_surfaces_a_non_cancel_exception(r: SubTestResult):
    print("\n--- G2: a background cook that raises (not CookCancelled) must not vanish ---")
    try:
        pd_bench = _load("preempt_drain_bench", "preempt_drain_bench.py")
        error_box = {}

        def _bg():
            try:
                raise RuntimeError("simulated genuine cook bug, not a cancel")
            except pd_bench.CookCancelled:
                pass
            except Exception as exc:
                error_box["exc"] = exc

        th = pd_bench._start_background_cook(_bg)
        raised = None
        try:
            pd_bench._reap_background_cook(th, 5.0, "g2 test", error_box=error_box)
        except Exception as e:
            raised = e
        assert raised is not None, (
            "a background cook that raised a genuine exception (not CookCancelled) must "
            "surface at the reap point, not be silently treated as a clean, on-time finish")
        assert "RuntimeError" in repr(raised) or "simulated genuine cook bug" in str(raised), raised
        r.ok(f"a non-cancel exception in the background cook is now surfaced: {raised}")
    except Exception as e:
        r.fail("G2 reap surfaces crash", str(e))


def test_fixgate_g2_reap_background_cook_still_passes_on_a_clean_cancel(r: SubTestResult):
    """Behaviour-preserving: the everyday CookCancelled path (the whole point of a
    `_bg()` closure) must still be a clean pass, exactly as before."""
    print("\n--- G2: a clean CookCancelled finish still reaps without error ---")
    try:
        pd_bench = _load("preempt_drain_bench", "preempt_drain_bench.py")
        error_box = {}

        def _bg():
            try:
                raise pd_bench.CookCancelled("simulated normal cancel")
            except pd_bench.CookCancelled:
                pass
            except Exception as exc:
                error_box["exc"] = exc

        th = pd_bench._start_background_cook(_bg)
        pd_bench._reap_background_cook(th, 5.0, "g2 test clean", error_box=error_box)
        r.ok("a clean CookCancelled finish reaps without raising")
    except Exception as e:
        r.fail("G2 reap clean cancel", str(e))


def test_fixgate_g2_reap_background_cook_still_detects_a_real_hang(r: SubTestResult):
    """Behaviour-preserving: the ORIGINAL BENCH-47 shape (a thread still alive after its
    join timeout) must still hard-fail exactly as before -- this ask closes the sibling
    gap, it does not loosen the existing one."""
    print("\n--- G2: a genuinely hung background thread still hard-fails ---")
    try:
        pd_bench = _load("preempt_drain_bench", "preempt_drain_bench.py")
        stop = threading.Event()

        def _bg():
            stop.wait(timeout=30)   # simulate a hang well past the short join timeout below

        th = pd_bench._start_background_cook(_bg)
        try:
            raised = None
            try:
                pd_bench._reap_background_cook(th, 0.2, "g2 test hang")
            except Exception as e:
                raised = e
            assert raised is not None and isinstance(raised, pd_bench.BackgroundCookHungError), raised
            r.ok(f"a genuinely hung thread still raises BackgroundCookHungError: {raised}")
        finally:
            stop.set()
    except Exception as e:
        r.fail("G2 reap still detects a hang", str(e))


# ── io_playback_bench: measure_overlapped's writer thread ───────────────────────────

_PRIOR_BUDGET: dict = {}


def _disarm(ipb):
    """Undo `_arm`: drop the provider AND put the process-wide media cache budget back (it is
    process-global, and a budget left at 0 starves every later test that relies on the pool)."""
    ipb.tex_provider.reset_provider()
    if "bytes" in _PRIOR_BUDGET:
        ipb.tex_provider.set_media_budget_mb(_PRIOR_BUDGET["bytes"] / (1024 * 1024))


def _arm(ipb, res: int, device: str, in_stall: float):
    """Mirror `main()`'s own `arm()` closure (module-level provider setup + `_CODE`), the
    minimum this file's functions need to run outside the CLI -- `measure_overlapped`'s own
    `provider=` parameter is never read by its body (it calls the module-level
    `tex_provider.materialize("plate", ...)` directly), so the provider must be armed onto
    `tex_provider` itself, exactly like `main()` does."""
    if getattr(ipb, "_CODE", None) is None:
        ipb._CODE = ipb._code(1)   # one grade is enough; these tests are about the threads
    _PRIOR_BUDGET.setdefault("bytes", ipb.tex_provider.get_media_cache()._budget)
    ipb.tex_provider.reset_provider()
    p = ipb.tex_provider.SyntheticFrameProvider(res=res, rate=1.0, device=device,
                                                latency_s=in_stall)
    ipb.tex_provider.set_provider(p)
    ipb.tex_provider.set_media_budget_mb(0.0)
    return p


def test_fixgate_g2_measure_overlapped_surfaces_a_writer_crash(r: SubTestResult):
    print("\n--- G2: measure_overlapped surfaces a crashing writer, not a bogus timing ---")
    try:
        ipb = _load("io_playback_bench", "io_playback_bench.py")
        p = _arm(ipb, res=8, device="cpu", in_stall=0.0)

        calls = {"n": 0}
        orig_write = ipb._write

        def _crashing_write(handle, stall):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated writer bug")
            return orig_write(handle, stall)

        ipb._write = _crashing_write
        try:
            raised = None
            try:
                ipb.measure_overlapped(frames=5, res=8, device="cpu", in_stall=0.0,
                                       out_stall=0.0, provider=p, lookahead=2)
            except Exception as e:
                raised = e
            assert raised is not None, (
                "a writer thread that raises mid-run must surface as a hard failure, not "
                "a silently-corrupted elapsed-time number")
            assert "simulated writer bug" in repr(raised), raised
            r.ok(f"a crashing writer now surfaces at measure_overlapped: {raised}")
        finally:
            ipb._write = orig_write
            _disarm(ipb)
    except Exception as e:
        r.fail("G2 measure_overlapped surfaces writer crash", str(e))


@pytest.mark.timing
def test_fixgate_g2_measure_overlapped_still_works_cleanly(r: SubTestResult):
    """Behaviour-preserving: an ordinary, error-free run must still return a real
    (elapsed, queue) pair exactly as before. Marked `timing` (GATE-47/G1): it reads a
    wall-clock value and asserts on it, even though the bound is a generous sanity check,
    not a performance claim."""
    print("\n--- G2: measure_overlapped still works on a clean run ---")
    try:
        ipb = _load("io_playback_bench", "io_playback_bench.py")
        p = _arm(ipb, res=8, device="cpu", in_stall=0.0)
        t0 = time.perf_counter()
        elapsed, q = ipb.measure_overlapped(frames=5, res=8, device="cpu", in_stall=0.0,
                                            out_stall=0.0, provider=p, lookahead=2)
        assert elapsed > 0.0, elapsed
        assert time.perf_counter() - t0 < 30.0, "an error-free run should not take this long"
        r.ok(f"a clean run still returns elapsed={elapsed:.4f}s")
    except Exception as e:
        r.fail("G2 measure_overlapped clean run", str(e))
    finally:
        try:
            _disarm(ipb)
        except Exception:
            pass
