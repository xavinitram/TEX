"""
PREWARM-481 (v0.48.1 patch candidate) -- `prewarm_async()`'s background compile-warm step
shares the caller's GIL, and holds it for as long as the underlying compile work runs.

An embedding host reported (relayed 2026-09-27) that on a project load, its UI-thread
heartbeat stalled several seconds while `prewarm_async()` warmed a handful of cold programs on
a newer GPU architecture -- the SAME duration as `prewarm()`'s own blocking call, even though
`prewarm_async()`'s whole contract is "off the cook thread". Measured here (RTX 2080 SUPER
sm_75, truly-cold Triton/Inductor on-disk kernel caches): `prewarm()`'s OWN loop (parse/
typecheck, codegen emit + the `.cg` persist, the capturability verdict) is cheap and bounded
(tens of ms total across 5 programs) -- a multi-second cost lives entirely in the FIRE-AND-
FORGET background `torch.compile` step (`compiled._submit_bg_compile` -> `_COMPILE_POOL`, a
plain in-process daemon THREAD), whose actual execution `prewarm_async()`'s own
`PrewarmHandle.wait()` never waits for, and which shares this process's GIL with the caller
regardless.

This test does not need a real slow compile (flaky, and CI has no CUDA/Triton) to prove the
MECHANISM: it substitutes a DETERMINISTIC stand-in for "whatever the compile-mode's background
step actually does" -- a fixed-duration, GIL-holding busy loop (`sys.setswitchinterval` maxed
so the interpreter never voluntarily switches away from it, the deterministic worst case any
real CPU-bound compile phase could produce) -- and asks only: while `prewarm_async()` runs,
what is the largest gap a concurrent heartbeat thread ever sees? RED at v0.48.0 (the stand-in
runs on the shared in-process pool, so the heartbeat starves for the stand-in's own duration);
GREEN once the background step is dispatched somewhere that does not share this process's GIL.

Marked `@pytest.mark.timing` per the standing rule (CI does not deselect `timing`; any
wall-clock assert must carry the marker) -- the ASSERTION is timing-shaped (a millisecond
bound), even though the mechanism producing the two outcomes is fully deterministic, not a
wall-clock race.
"""
import sys
import time
import threading

import pytest

from helpers import *  # noqa: F401,F403  (TEXType, cold_engine_state)
from TEX_Wrangle import tex_api
from TEX_Wrangle.tex_runtime import compiled as C

# The bound this fix must guarantee. The stand-in below holds the GIL continuously for
# _STALL_S -- comfortably larger than any tick/scheduling noise -- so a max-gap comfortably
# BELOW it (not just "less than _STALL_S") is real evidence, not a coin flip: base measures a
# gap close to _STALL_S (the whole stand-in is one un-preemptable span); a fix that merely
# shaves it down without actually moving the work off-GIL would still land well above
# _BOUND_MS, so this cannot pass by accident.
_STALL_S = 0.35
_BOUND_MS = 120.0


def _five_programs():
    out = []
    for i in range(5):
        code = f"vec3 c=@A.rgb*1.3 - 0.{i}; c=clamp(c,0.0,1.0); @OUT=vec4(c,1.0);"
        out.append((code, {"A": TEXType.VEC3}))
    return out


def _gil_hog(duration_s: float):
    """A deterministic, GIL-holding stand-in for "whatever the real background compile step
    does" -- no blocking call inside it (no sleep, no I/O, no subprocess), so nothing here
    ever voluntarily releases the GIL; `setswitchinterval` maxed means the interpreter itself
    does not preempt it either. This is the worst case a real CPU-bound compile phase could
    produce, used here specifically because it is deterministic where a real compile is not."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(3600.0)
    try:
        t_end = time.perf_counter() + duration_s
        x = 0
        while time.perf_counter() < t_end:
            x += 1
        return x
    finally:
        sys.setswitchinterval(old)


@pytest.mark.timing
def test_prewarm481_heartbeat_bounded_during_prewarm_async(r: SubTestResult):
    print("\n--- PREWARM-481: prewarm_async()'s background step must not starve a "
          "concurrent heartbeat ---")
    with cold_engine_state():
        real_is_available = torch.cuda.is_available
        real_headroom = C._cuda_headroom_ok
        real_capture = C._capture_in_flight
        real_try_compile = C._try_compile
        # Try the subprocess boundary this ask adds -- present only at HEAD. Patched to a
        # fast, GIL-RELEASING stand-in (`time.sleep`, exactly like the real wait for a child
        # process to exit) so the test never actually spawns a subprocess (deterministic,
        # and avoids paying a real interpreter start + torch import inside it).
        try:
            from TEX_Wrangle.tex_runtime import prewarm_worker as _pw_sub
            real_warm_subprocess = _pw_sub.warm_in_subprocess
        except ImportError:
            _pw_sub = None
            real_warm_subprocess = None

        torch.cuda.is_available = lambda: True
        C._cuda_headroom_ok = lambda *a, **kw: True
        C._capture_in_flight = lambda: False
        C._try_compile = lambda *a, **kw: (_gil_hog(_STALL_S), None)[1]
        if _pw_sub is not None:
            def _fake_subprocess_warm(jobs, **kw):
                time.sleep(_STALL_S)
                return {"programs": len(jobs), "bg_compile": len(jobs), "error": None}
            _pw_sub.warm_in_subprocess = _fake_subprocess_warm

        ticks = []
        stop = threading.Event()

        def heartbeat():
            while not stop.is_set():
                ticks.append(time.perf_counter())
                time.sleep(0.002)

        hb = threading.Thread(target=heartbeat, name="test-ui-heartbeat", daemon=True)
        try:
            keys_before = set(C._bg_futures.keys())
            hb.start()
            programs = _five_programs()
            handle = tex_api.prewarm_async(programs, device="cuda", precision="fp32",
                                           compile_mode="auto")
            summary = handle.wait(timeout=30)

            # Drain any NEW background futures the (base-shaped) "thread" mechanism left
            # in-flight -- `PrewarmHandle.wait()` never waits for these, so the heartbeat must
            # still be running while they finish, or the stand-in's stall would go unmeasured
            # and this test would pass at base for the wrong reason (vacuously).
            new_keys = set(C._bg_futures.keys()) - keys_before
            for k in new_keys:
                fut = C._bg_futures.get(k)
                if fut is not None:
                    try:
                        fut.result(timeout=10)
                    except Exception:
                        pass
        finally:
            stop.set()
            hb.join(timeout=5)
            torch.cuda.is_available = real_is_available
            C._cuda_headroom_ok = real_headroom
            C._capture_in_flight = real_capture
            C._try_compile = real_try_compile
            if _pw_sub is not None:
                _pw_sub.warm_in_subprocess = real_warm_subprocess

        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        max_gap_ms = (max(gaps) * 1000) if gaps else 0.0

        try:
            assert summary["programs"] == 5 and summary["errors"] == 0, summary
            assert len(ticks) >= 5, f"heartbeat barely ran at all ({len(ticks)} ticks) -- " \
                                     f"repro itself is broken, not measuring anything"
            assert max_gap_ms <= _BOUND_MS, (
                f"prewarm_async()'s background compile-warm step starved a concurrent "
                f"heartbeat for {max_gap_ms:.1f} ms (bound {_BOUND_MS:.0f} ms) -- it is "
                f"still sharing this process's GIL with the caller for the FULL duration "
                f"of the (stand-in) compile-mode work, defeating prewarm_async()'s own "
                f"'off the cook thread' contract")
            r.ok(f"heartbeat max gap {max_gap_ms:.1f} ms <= {_BOUND_MS:.0f} ms bound; "
                 f"summary={summary}")
        except AssertionError as e:
            r.fail("PREWARM-481 heartbeat bound", str(e))
