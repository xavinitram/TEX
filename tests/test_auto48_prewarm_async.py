"""
AUTO-48 (v0.48) -- `tex_api.prewarm_async()`, a compile-ahead HOOK.

`prewarm()` (CACHE-3) already warms the codegen/compiled tiers for a list of programs, but
it runs on the CALLING thread -- an embedding host that calls it from its own cook worker
pays its full cost inline (measured on an embedding host's own edit-tick shape at ~11s on
a cold cache). This ask adds the non-blocking counterpart:
`prewarm_async()` submits the SAME `prewarm()` body to a single-worker daemon pool of its own
(`tex_runtime.compiled_capability._get_prewarm_pool()`), separate from the one
`compile_capability_async()` uses (`_get_capability_pool()`, AUTO-47) so a large warm-ahead
batch cannot starve a capability probe -- and hands back a `PrewarmHandle` to poll, wait on,
or cancel.
Mechanism only: nothing here decides WHEN a host should call it, and nothing here can change
what a real cook produces -- warming only populates the same caches an un-warmed cook would
populate on its own first cook (invariant 7).

Four things pinned here:
  1. COUNTS: a later real cook of an already-warmed program emits ZERO codegen (the disk/
     memory cache `prewarm_async()` populated is what a real cook then hits).
  2. CANCEL: `PrewarmHandle.cancel()` stops the job at its next per-program yield point
     (the same `_cancel_check` `prewarm()` already had, HOSTAUDIT-2) -- proved by STRUCTURE
     (a deterministic block/release rendezvous), never by a wall-clock race.
  3. EXCEPTION ISOLATION: a program that cannot compile does not abort the job (already
     true of `prewarm()`; re-proved through the async wrapper) NOR poison the shared pool
     for the next, unrelated `prewarm_async()` call, even if the whole job callable itself
     raises (a bug, not a bad program).
  4. PROCESS EXIT: a slow job in flight when the caller returns and the process ends does
     not stall exit (the pool's worker thread is daemonic, mirroring `_DaemonProbePool`'s
     own contract) -- proved via a real subprocess.
"""
import subprocess
import sys
import threading
import time

import pytest

from helpers import *  # noqa: F401,F403  (TEXType, make_img, SubTestResult, cold_engine_state)
from TEX_Wrangle import tex_api
from TEX_Wrangle.tex_runtime import compiled as C


def _five_programs():
    """Five DISTINCT programs (a different literal each) -- distinct fingerprints, so each
    is its own codegen-emission unit and the cancel test can tell "warmed" from "not
    reached". `binding_types` maps INPUT names only (`tex_api.compile`'s own contract,
    the pair `prewarm`/`prewarm_async` forward unchanged) -- `@OUT` is an output, never a
    binding, and the real engine's own binding-types map (derived from the caller's actual
    `bindings` dict at cook time) never carries it either; including it here would give
    `prewarm_async()` a DIFFERENT fingerprint than the real cook path computes and the
    zero-emission count below would fail for a test-authoring reason, not a real one."""
    out = []
    for i in range(5):
        code = f"vec3 c=@A.rgb; c = c*1.3 - 0.{i}; c = clamp(c, 0.0, 1.0); @OUT=vec4(c,1.0);"
        bt = {"A": TEXType.VEC3}
        out.append((code, bt))
    return out


# ── 1. counts: a warmed program's later real cook emits zero codegen ───────────────────

def test_auto48_warmed_program_cooks_with_zero_emission(r: SubTestResult):
    print("\n--- AUTO-48: prewarm_async() -> a later real cook emits 0 codegen ---")
    with cold_engine_state():
        code, bt = _five_programs()[0]
        handle = tex_api.prewarm_async([(code, bt)], device="cpu", precision="fp32",
                                       compile_mode="auto")
        summary = handle.wait(timeout=30)
        try:
            assert summary["programs"] == 1 and summary["errors"] == 0, summary
            assert summary["codegen"] == 1, f"codegen was not warmed: {summary}"

            calls = []
            orig = C._try_codegen
            C._try_codegen = lambda *a, **kw: (calls.append(1), orig(*a, **kw))[1]
            try:
                img = make_img(1, 8, 8, 3, seed=7)
                from TEX_Wrangle import tex_engine
                tex_engine.cook(code, {"A": img}, device_mode="cpu",
                                compile_mode="auto", precision="fp32")
            finally:
                C._try_codegen = orig

            assert len(calls) == 0, (
                f"a real cook of an ALREADY-WARMED program still emitted codegen "
                f"({len(calls)} call(s)) -- prewarm_async()'s cache population and the "
                f"real cook path disagree about the fingerprint")
            r.ok(f"warmed program cooked with 0 codegen emissions; summary={summary}")
        except AssertionError as e:
            r.fail("AUTO-48 warmed program zero-emission", str(e))


# ── 2. cancel: stops at the next per-program yield point, proved by structure ──────────

def test_auto48_cancel_stops_between_programs(r: SubTestResult):
    print("\n--- AUTO-48: PrewarmHandle.cancel() stops between programs (structural) ---")
    with cold_engine_state():
        started = threading.Event()
        release = threading.Event()
        orig = C._get_or_make_codegen_fn
        first_call = {"seen": False}

        def blocking_first(*a, **kw):
            # Block INSIDE program 0's own warm step so the main thread can order
            # cancel() to land strictly BETWEEN program 0 finishing and program 1's own
            # `_cancel_check` -- no wall-clock margin required either side.
            if not first_call["seen"]:
                first_call["seen"] = True
                started.set()
                release.wait(timeout=10)
            return orig(*a, **kw)

        C._get_or_make_codegen_fn = blocking_first
        try:
            handle = tex_api.prewarm_async(_five_programs(), device="cpu",
                                           precision="fp32", compile_mode="auto")
            assert started.wait(timeout=10), "program 0's warm step never started"
            cancelled_now = handle.cancel()
            release.set()   # let program 0 finish; its NEXT iteration observes cancel()
            summary = handle.wait(timeout=30)
        finally:
            C._get_or_make_codegen_fn = orig

        try:
            assert cancelled_now is True, "cancel() reported nothing to cancel"
            assert summary["programs"] == 1, (
                f"expected exactly program 0 to have been processed before the cancel "
                f"took effect, got {summary}")
            assert summary["cancelled"] == 4, f"expected 4 skipped, got {summary}"
            assert handle.cancel() is False, (
                "cancel() on an already-finished handle must report False (nothing to "
                "cancel), never re-signal a done job")
            r.ok(f"cancel() stopped the job after program 0: {summary}")
        except AssertionError as e:
            r.fail("AUTO-48 cancel stops between programs", str(e))


# ── 3. exception isolation: a bad program, and a job that raises outright ──────────────

def test_auto48_exception_isolation(r: SubTestResult):
    print("\n--- AUTO-48: exception isolation (bad program; a job that raises outright) ---")
    with cold_engine_state():
        try:
            # 3a. one bad program (a genuine parse error) among two good ones --
            # prewarm()'s own per-program try/except already covers this; re-proved
            # through the async wrapper. (An untyped @-binding is NOT an error by itself
            # -- `tex_api.check`'s own contract says an undeclared input resolves to
            # VEC4 -- so the bad program here has to be malformed SOURCE, not a thin
            # binding-types map.)
            good1 = _five_programs()[0]
            bad = ("@OUT = vec4(this is not @@ valid tex syntax;", {})
            good2 = _five_programs()[1]
            handle = tex_api.prewarm_async([good1, bad, good2], device="cpu",
                                           precision="fp32", compile_mode="auto")
            summary = handle.wait(timeout=30)
            assert summary["errors"] == 1, f"expected the bad program isolated: {summary}"
            assert summary["programs"] == 2, f"expected both good programs counted: {summary}"

            # 3b. the WHOLE job callable raises (a bug, not a bad program) -- poll()/wait()
            # must surface it, and the SHARED pool must still serve the next, unrelated
            # call afterward (the daemon worker's own "propagate to the future, never
            # crash the worker" contract, mirrored from _DaemonProbePool._run).
            orig_prewarm = tex_api.prewarm

            def raising_prewarm(*a, **kw):
                raise RuntimeError("AUTO-48 test: the job itself is broken")

            tex_api.prewarm = raising_prewarm
            try:
                handle2 = tex_api.prewarm_async([good1], device="cpu", precision="fp32")
                raised = False
                try:
                    handle2.wait(timeout=10)
                except RuntimeError as e:
                    raised = "AUTO-48 test" in str(e)
                assert raised, "a job that raises outright did not surface on wait()"
                assert handle2.done, "a resolved (even a raised) job must report done=True"
            finally:
                tex_api.prewarm = orig_prewarm

            # the pool is unharmed: an ordinary call right after still completes normally.
            handle3 = tex_api.prewarm_async([good2], device="cpu", precision="fp32")
            summary3 = handle3.wait(timeout=30)
            assert summary3["errors"] == 0 and summary3["programs"] == 1, summary3

            r.ok("bad program isolated; a job-level exception surfaced without "
                 "poisoning the shared pool for the next call")
        except AssertionError as e:
            r.fail("AUTO-48 exception isolation", str(e))


# ── 4. process exit: a slow job in flight does not stall it ────────────────────────────

@pytest.mark.slow
def test_auto48_process_exit_not_stalled():
    """A real subprocess: submit a deliberately slow warm-ahead job (a monkeypatched
    `prewarm()` that sleeps well past any reasonable exit budget), never wait on the
    handle, and let the interpreter shut down immediately. If the pool's worker thread
    were non-daemonic (a plain `concurrent.futures.ThreadPoolExecutor`, B3#2's own repro),
    `atexit` would join it and the process would hang for the job's full duration. Asserts
    the process exits in well under that duration."""
    import pathlib
    worktree_parent = pathlib.Path(__file__).resolve().parents[2]
    script = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(worktree_parent)!r})\n"
        "from TEX_Wrangle import tex_api\n"
        "def slow(*a, **kw):\n"
        "    time.sleep(20)\n"
        "    return {'programs': 0, 'codegen': 0, 'bg_compile': 0, 'capturable': 0, "
        "'errors': 0, 'cancelled': 0}\n"
        "tex_api.prewarm = slow\n"
        "tex_api.prewarm_async([('@OUT = vec4(1.0);', {})], device='cpu')\n"
        "print('SUBPROCESS_REACHED_EXIT')\n"
    )
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, timeout=15)
    dt = time.perf_counter() - t0
    assert "SUBPROCESS_REACHED_EXIT" in proc.stdout, proc.stdout + proc.stderr
    assert dt < 10.0, (
        f"process took {dt:.1f}s to exit with a 20s job in flight -- the pool's worker "
        f"thread is blocking process exit (should be daemonic)")
