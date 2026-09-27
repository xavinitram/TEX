"""FIX-GATE A1/A2 (v0.47.0 Phase C) -- compile_capability_async() robustness.

A1: a background probe that raises (a broken Triton install, or an OSError walking an
ACL-restricted Program Files subtree) must not permanently disable "auto"'s capability
check for the rest of the process (B3#1). `compile_capability()` now treats a raising
probe as a definite-for-now False (logged once), while allowing exactly ONE retry on a
LATER call before caching the negative answer for good -- a transient probe failure gets
a second chance; a second consecutive failure is a real, stable answer, cached like any
other. `compile_capability_async()`'s own bookkeeping (`_capability_future`) is cleared
whenever a resolved probe was NOT cached, so the next call resubmits instead of being
stuck reading the same exhausted future forever.

A2: the probe's own background pool must never delay process exit (B3#2): CPython's
`concurrent.futures.thread` module unconditionally joins every `ThreadPoolExecutor`
worker thread at interpreter exit, in-flight work or not, and a `ThreadPoolExecutor`
gives no public way to make its worker daemonic. The probe now runs on a small
hand-rolled single-daemon-thread pool instead, so a slow in-flight probe (up to 30s,
`_probe_cpu_inductor`'s Windows vcvarsall subprocess) never holds up shutdown -- unlike
`_WARM_POOL` (`compiled.py`), which is deliberately NOT reused here: a warm-call job's
whole point is to finish and be cached, so blocking exit on it is intentional; a
capability probe's answer is disposable (recomputed next process) and gates nothing
durable, so nothing should ever wait on it.
"""
import subprocess
import sys
import time
from pathlib import Path

from helpers import SubTestResult

from TEX_Wrangle.tex_runtime import compiled_capability as _CC


# ── A1: a raising probe recovers via one retry, then is bounded ────────────────────

def test_fixgate_a1_probe_exception_recovers_with_one_retry(r: SubTestResult):
    print("\n--- A1: an exception in the probe becomes a logged False, with one retry ---")
    calls = {"cuda": 0}

    def raising_then_ok():
        calls["cuda"] += 1
        if calls["cuda"] <= 1:
            raise RuntimeError("simulated broken triton probe")
        return True, None   # the retry succeeds

    def ok_cpu():
        return False, "no compiler (test)"

    orig_cuda, orig_cpu = _CC._probe_cuda_inductor, _CC._probe_cpu_inductor
    _CC._probe_cuda_inductor = raising_then_ok
    _CC._probe_cpu_inductor = ok_cpu
    _CC._reset_capability_cache_for_test()
    try:
        first = _CC.compile_capability()
        assert first["cuda_inductor"] is False, first
        assert "cuda_inductor" in first["reason"], first
        assert calls["cuda"] == 1, calls

        second = _CC.compile_capability()
        assert calls["cuda"] == 2, (
            f"a probe that raised once must get exactly one retry on the next call, "
            f"got {calls['cuda']} calls")
        assert second["cuda_inductor"] is True, second

        _CC.compile_capability()
        assert calls["cuda"] == 2, (
            "once cached (by success or by exhausting the retry), compile_capability() "
            f"must never probe again, got {calls['cuda']} calls")
        r.ok("a raising probe recovers via its one retry and then stays cached")
    except Exception as e:
        r.fail("A1 one-retry recovery", str(e))
    finally:
        _CC._probe_cuda_inductor, _CC._probe_cpu_inductor = orig_cuda, orig_cpu
        _CC._reset_capability_cache_for_test()


def test_fixgate_a1_second_consecutive_exception_caches_false(r: SubTestResult):
    print("\n--- A1: two consecutive exceptions cache False for good (no infinite retry) ---")
    calls = {"cuda": 0}

    def always_raises():
        calls["cuda"] += 1
        raise RuntimeError("simulated persistently broken probe")

    def ok_cpu():
        return True, None

    orig_cuda, orig_cpu = _CC._probe_cuda_inductor, _CC._probe_cpu_inductor
    _CC._probe_cuda_inductor = always_raises
    _CC._probe_cpu_inductor = ok_cpu
    _CC._reset_capability_cache_for_test()
    try:
        _CC.compile_capability()          # first: transient, not cached
        cap = _CC.compile_capability()    # second: retry spent, now cached False
        assert calls["cuda"] == 2, calls
        for _ in range(5):
            cap = _CC.compile_capability()
        assert calls["cuda"] == 2, (
            f"a permanently-failing probe must stop being retried once its budget is "
            f"spent, got {calls['cuda']} calls")
        assert cap["cuda_inductor"] is False, cap
        r.ok("a persistently-raising probe is bounded to 2 attempts, then cached False")
    except Exception as e:
        r.fail("A1 bounded retry", str(e))
    finally:
        _CC._probe_cuda_inductor, _CC._probe_cpu_inductor = orig_cuda, orig_cpu
        _CC._reset_capability_cache_for_test()


def test_fixgate_a1_async_resubmits_after_a_transient_probe_exception(r: SubTestResult):
    print("\n--- A1: compile_capability_async() re-probes instead of being stuck at None ---")
    calls = {"cuda": 0}

    def raising_then_ok():
        calls["cuda"] += 1
        if calls["cuda"] <= 1:
            raise RuntimeError("simulated broken triton probe")
        return True, None

    def ok_cpu():
        return True, None

    orig_cuda, orig_cpu = _CC._probe_cuda_inductor, _CC._probe_cpu_inductor
    _CC._probe_cuda_inductor = raising_then_ok
    _CC._probe_cpu_inductor = ok_cpu
    _CC._reset_capability_cache_for_test()
    try:
        deadline = time.perf_counter() + 10.0
        result = None
        while time.perf_counter() < deadline:
            result = _CC.compile_capability_async()
            if result is not None and result["cuda_inductor"] is True:
                break
            time.sleep(0.02)
        assert result is not None and result["cuda_inductor"] is True, (
            f"compile_capability_async() never recovered from the transient probe "
            f"exception within its retry budget: {result}")
        r.ok("compile_capability_async() recovers from a transient probe exception")
    except Exception as e:
        r.fail("A1 async resubmit after transient failure", str(e))
    finally:
        _CC._probe_cuda_inductor, _CC._probe_cpu_inductor = orig_cuda, orig_cpu
        _CC._reset_capability_cache_for_test()


# ── A2: the probe pool never delays process exit ───────────────────────────────────

_A2_BASELINE_CHILD = r'''
import sys
sys.path.insert(0, sys.argv[1])                       # .../custom_nodes
from TEX_Wrangle.tex_runtime import compiled_capability as _CC  # pay the import cost only
'''

_A2_PROBED_CHILD = r'''
import sys, time
sys.path.insert(0, sys.argv[1])                       # .../custom_nodes
from TEX_Wrangle.tex_runtime import compiled_capability as _CC

def slow_probe():
    time.sleep(3.0)
    return True, None

_CC._probe_cpu_inductor = slow_probe
_CC._probe_cuda_inductor = lambda: (False, "no cuda (test)")
_CC._reset_capability_cache_for_test()
_CC.compile_capability_async()      # kicks off the background probe, returns None at once
# Deliberately exit right away without waiting for the probe: the whole point is whether
# the PROCESS exit itself is held up by the still-running probe thread.
'''


def test_fixgate_a2_process_exit_never_waits_on_an_in_flight_probe(r: SubTestResult):
    print("\n--- A2: a slow in-flight probe must never delay process exit ---")
    custom_nodes = str(Path(__file__).resolve().parents[2])

    def _run(script):
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, "-c", script, custom_nodes],
                              capture_output=True, timeout=30)
        return time.perf_counter() - t0, proc

    try:
        baseline_s, base_proc = _run(_A2_BASELINE_CHILD)
        if base_proc.returncode != 0:
            r.fail("A2 exit stall", f"baseline child exited {base_proc.returncode}: "
                                    f"{base_proc.stderr.decode(errors='replace')[-400:]}")
            return
        probed_s, probed_proc = _run(_A2_PROBED_CHILD)
        if probed_proc.returncode != 0:
            r.fail("A2 exit stall", f"probed child exited {probed_proc.returncode}: "
                                    f"{probed_proc.stderr.decode(errors='replace')[-400:]}")
            return
    except subprocess.TimeoutExpired:
        r.fail("A2 exit stall", "a child never exited within 30s")
        return

    delta = probed_s - baseline_s
    assert delta < 1.5, (
        f"a 3s in-flight probe added {delta:.2f}s to process exit (baseline {baseline_s:.2f}s, "
        f"probed {probed_s:.2f}s) -- a non-daemon probe thread is blocking interpreter "
        f"shutdown (B3#2)")
    r.ok(f"in-flight probe added only {delta:.2f}s to process exit "
         f"(baseline {baseline_s:.2f}s, probed {probed_s:.2f}s)")
