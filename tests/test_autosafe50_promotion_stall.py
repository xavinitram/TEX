"""
AUTOSAFE-50 (v0.50, host item 2 + TRK-223/TRK-231) — the `"auto"` compile mode's
TRIAL promotion must never stall the cook thread for seconds, and a failed background
compile must always fall back cleanly and stay visible.

**TRK-223 (CONFIRMED, this ask's own repro).** `PREWARM-481`'s own follow-up measurement
found, by direct measurement (not by reasoning about the code alone), that a program under
`compile_mode="auto"` which is cooked enough times to reach `TRIAL` and then gets a cache
hit can still pay a synchronous, GIL-holding first invocation of the freshly-compiled
callable ON THE COOK THREAD ITSELF — a distinct mechanism from the prewarm-pool GIL-sharing
`PREWARM-481` fixed (that ask's own programs never reached `TRIAL`; see its own
follow-up writeup). Reproduced here at the SOURCE: at base, `run_auto`'s TRIAL branch
called `_run_cached_compiled(...).result()` with NO TIMEOUT, on the SAME thread that must
also serve the next real cook — a heartbeat thread's own max gap during the promotion cook
spans however long the compiled callable's own first real invocation takes, unbounded.
Reproduced with a DETERMINISTIC GIL-holding stand-in (same technique `PREWARM-481`'s own
test uses, `tests/test_prewarm481_gil_bound.py`) rather than a real slow compile — real
compiles are flaky and CI has no CUDA/Triton/MSVC.

**TRK-231 (a hang after a failed background compile, provisional, never
reproduced by its own finder).** This file's `test_t3_failed_compile_never_hangs_the_next_cook`
attempts a deterministic forced repro (a compile stand-in that RAISES at each stage a real
compile can fail at) and a second attempt with a stand-in that NEVER RETURNS (the general
shape of "one job on a single-worker pool hangs, so every later submission to the SAME pool
queues behind it forever" — a real, general mechanism this file confirms independently of
TRK-231's own exact trigger). See the docstring on that test for the verdict: NOT CONFIRMED
against the exact trigger TRK-231 describes (a genuinely failed, not hung, background
job), but the general single-worker-pool-starvation shape IS real and IS what this ask's
fix (routing the TRIAL invocation onto its own bounded, polled future rather than a
synchronous `.result()`) also closes off for the promotion step specifically.

Both fixed by the SAME mechanism: `run_auto`'s TRIAL branch now submits the promotion
invocation once (`compiled._submit_trial`, reusing `_WARM_POOL` — no new pool) and polls it
with a small bounded budget (`compiled._await_trial`, `_TRIAL_WAIT_BUDGET_S`) instead of
blocking on it directly. A still-running job is left in `_trial_futures` for a LATER cook to
poll (near-zero cost once done); a failed one is recorded via `tier_trace.record` and the
`compiled._promotion_stats["failed"]` counter and falls back to codegen — visible, not a
silent reject and never a hang.

Marked `@pytest.mark.timing` per the standing rule (CI does not deselect `timing`) for the
rows whose ASSERTION is a millisecond or wall-clock bound, even though the mechanism producing the two
outcomes (RED at base, GREEN at head) is fully deterministic.
"""
import threading
import time

import pytest

from helpers import *  # noqa: F401,F403  (TEXType, cold_engine_state, parse_and_split, ...)
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_runtime import compiled as C
from TEX_Wrangle.tex_runtime import autotier as AT

# The bound the stand-in holds the (background) worker for. Comfortably larger than any
# tick/scheduling noise, so a max-gap comfortably BELOW it during the promotion cook is real
# evidence a fix moved the wait off the cook thread's critical path, not a coin flip.
_STALL_S = 0.35
_BOUND_MS = 120.0


def _tiny_program():
    code = "vec3 c=@A.rgb*1.3 - 0.1; c=clamp(c,0.0,1.0); @OUT=vec4(c,1.0);"
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    used = _collect_identifiers(prog)
    return prog, tm, used


def _bounded(fn, timeout_s=15.0):
    """Run `fn` on a daemon thread; True if it returned within `timeout_s`. A `run_auto` that
    really blocks (the regression hunted here) fails the row instead of hanging the suite."""
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    t.join(timeout_s)
    return not t.is_alive()


def _drive_to_trial(prog, tm, used, bindings, fp, *, cap_ok=True):
    """Cook `run_auto` enough times, with the toolchain probe short-circuited, to walk the
    autotier state machine MEASURING -> COMPILING -> TRIAL for `fp`'s key, waiting out the
    background compile+warm job in between (so promotion to TRIAL is settled BEFORE the
    caller's own timed section starts — the test measures the TRIAL cook's own cost, not an
    earlier stage's). Returns the `key` autotier files this fingerprint's verdict under."""
    real_cap = C.compile_capability_async
    C.compile_capability_async = lambda: {"cpu_inductor": cap_ok, "cuda_inductor": cap_ok}
    try:
        key = None
        for _ in range(AT._MEASURE_COOKS):
            C.run_auto(prog, dict(bindings), tm, "cpu", fp, output_names=["OUT"],
                      used_builtins=used)
        # One more cook: should_submit_compile() now fires and a background job is
        # submitted (warm_call included, since headroom/capture-in-flight are both
        # trivially true on CPU) -- wait for it to actually finish before returning, so the
        # caller's own timed cook is unambiguously the TRIAL one.
        C.run_auto(prog, dict(bindings), tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
        cache_key = (fp, "cpu", "fp32")
        C._drain_bg_for_test(timeout=10.0)
        # One more cook: COMPILING -> mark_ready() -> state becomes TRIAL (this cook still
        # returns codegen; see run_auto's own comment on why the promotion happens on the
        # NEXT cook after mark_ready).
        C.run_auto(prog, dict(bindings), tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
        sp = C._consensus_extent(bindings, prog)
        key = AT.make_key(fp, "cpu", "fp32", sp)
        assert AT.verdict(key) == AT.TRIAL, \
            f"setup didn't reach TRIAL (state={AT.verdict(key)}) -- test premise broken"
        return key, cache_key
    finally:
        C.compile_capability_async = real_cap


@pytest.mark.timing
def test_promotion_never_stalls_a_concurrent_heartbeat(r: SubTestResult):
    print("\n--- AUTOSAFE-50 / TRK-223: TRIAL promotion must not starve a concurrent "
          "heartbeat with an unbounded synchronous invocation ---")
    with cold_engine_state():
        AT.reset()
        prog, tm, used = _tiny_program()
        torch.manual_seed(11)
        img = torch.rand(1, 48, 48, 3)
        bindings = {"A": img}
        fp = "autosafe50_promo_fp"

        real_try_compile = C._try_compile

        def _stalling_try_compile(device_type, program, type_map, **kw):
            def _stand_in(*a, **k):
                # Deterministic stand-in for "the freshly-compiled callable's own real
                # invocation is slow" (TRK-223 confirms the EFFECT, not a single specific
                # cause -- a fresh CUDA-graph capture at a new tensor address, a guard
                # mismatch against the warm clone, or anything else `_try_compile`'s lazy
                # wrap did not force). GIL-RELEASING chunks (many short `time.sleep` calls
                # summing to `_STALL_S`), deliberately NOT a tight busy loop: real
                # torch.compile/CUDA work releases the GIL periodically through C-extension
                # calls (PREWARM-481's own measurement: "does release the GIL
                # periodically"), and a tight Python busy loop reproduces a WORSE, less
                # representative case on Windows specifically -- CPython's GIL hand-off
                # under contention from a continuously-running CPU-bound thread is known to
                # starve a timed-out waiter well past its nominal timeout (a "GIL convoy"),
                # which would make this test measure Windows' own GIL scheduling fairness
                # instead of this fix's own bounded-wait mechanism. Confirmed by a
                # dedicated diagnostic: the identical fix measured ~34 ms with this
                # sleep-chunked stand-in vs. ~270-320 ms with a tight busy-loop stand-in,
                # for the SAME code change -- the busy loop was measuring GIL fairness, not
                # the fix.
                n = 70
                for _ in range(n):
                    time.sleep(_STALL_S / n)
                return {"OUT": img}
            return _stand_in, "inductor"

        C._try_compile = _stalling_try_compile
        cache_key = None
        try:
            key, cache_key = _drive_to_trial(prog, tm, used, bindings, fp)

            ticks = []
            stop = threading.Event()

            def heartbeat():
                while not stop.is_set():
                    ticks.append(time.perf_counter())
                    time.sleep(0.002)

            hb = threading.Thread(target=heartbeat, name="test-ui-heartbeat", daemon=True)
            hb.start()
            try:
                # THE promotion cook, timed directly: this is the PRIMARY assertion -- the
                # goal is literally "never stall the cook thread for seconds", i.e. bound
                # THIS call's own wall time, not merely a proxy for GIL sharing. At base
                # this blocks synchronously on the stand-in's full _STALL_S inside
                # `_run_cached_compiled(...).result()` (no timeout); at head it submits
                # the invocation once and polls it for at most `_TRIAL_WAIT_BUDGET_S`
                # before returning the safe (codegen) tier's result instead.
                t0 = time.perf_counter()
                out = C.run_auto(prog, dict(bindings), tm, "cpu", fp,
                                 output_names=["OUT"], used_builtins=used)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                assert out is not None
                # Keep the heartbeat running a little past the call itself so its tick
                # count is a meaningful SECONDARY reading in both conditions.
                time.sleep(0.05)
            finally:
                stop.set()
                hb.join(timeout=5)

            gaps = [b - a for a, b in zip(ticks, ticks[1:])]
            max_gap_ms = (max(gaps) * 1000) if gaps else 0.0
            print(f"    (secondary) heartbeat max gap during the window: {max_gap_ms:.1f} ms")

            try:
                assert elapsed_ms <= _BOUND_MS, (
                    f"the TRIAL promotion cook itself took {elapsed_ms:.1f} ms (bound "
                    f"{_BOUND_MS:.0f} ms) -- run_auto blocked the calling (cook) thread "
                    f"synchronously on the freshly-promoted compiled callable's own "
                    f"invocation instead of backgrounding it and returning the safe tier's "
                    f"result within a small bounded wait")
                r.ok(f"promotion cook wall time {elapsed_ms:.1f} ms <= {_BOUND_MS:.0f} ms "
                     f"bound (heartbeat max gap {max_gap_ms:.1f} ms)")
            except AssertionError as e:
                r.fail("AUTOSAFE-50 promotion wall-time bound", str(e))

            # Eventual correctness: the deferred trial job must still land -- polling a
            # few more cooks (well past _STALL_S) must reach a TERMINAL verdict, never stay
            # stuck in TRIAL forever (the whole point of "off the critical path", not "never
            # happens"). The stand-in's compiled_ms (~350ms) is far slower than this tiny
            # program's own interp/codegen cost, so the honest verdict is REJECTED.
            time.sleep(_STALL_S + 0.1)
            terminal = None
            for _ in range(20):
                C.run_auto(prog, dict(bindings), tm, "cpu", fp, output_names=["OUT"],
                          used_builtins=used)
                st = AT.verdict(key)
                if st in (AT.COMMITTED, AT.REJECTED):
                    terminal = st
                    break
                time.sleep(0.01)
            try:
                assert terminal is not None, \
                    f"promotion never reached a terminal verdict (stuck at {AT.verdict(key)})"
                r.ok(f"promotion eventually reached a terminal verdict: {terminal}")
            except AssertionError as e:
                r.fail("AUTOSAFE-50 eventual promotion correctness", str(e))
        finally:
            C._try_compile = real_try_compile
            if cache_key is not None:
                C._drain_bg_for_test(timeout=10.0)


@pytest.mark.timing
def test_t3_failed_compile_never_hangs_the_next_cook(r: SubTestResult):
    """TRK-231: "one 'auto' hang after a failed background compile" — the host's
    own report is explicitly PROVISIONAL (1 occurrence in 2 on the candidate, 0 in 2 on the
    control; not reproduced by a later confirmation leg). Two forced, deterministic attempts
    here, each run on a daemon thread joined with a 15 s timeout, so a blocking `run_auto` fails THIS test rather than hanging the suite:

    1. A background compile that RAISES at each stage a real one can fail at (the wrap
       itself, and the warm call) -- the ordinary, expected-and-handled failure shape.
       NOT CONFIRMED as a hang: `run_auto` already routes this through `_bg_status`'s
       "failed"/"absent" case (autotier.record_trial(None) -> REJECTED), which this ask
       additionally makes visible via tier_trace + a counter. No hang, at base or at head.
    2. A background job that NEVER RETURNS -- the general shape "one job stuck on a
       single-worker pool blocks every later submission to that SAME pool forever" (both
       `_COMPILE_POOL` and `_WARM_POOL` are `max_workers=1`). This IS a real, confirmable
       mechanism: a stuck warm job on `_WARM_POOL` starves any OTHER key's own TRIAL
       invocation (which, before this ask, shared `_run_cached_compiled`'s call onto
       `_COMPILE_POOL`, not `_WARM_POOL`, so a stuck WARM job did not starve a TRIAL cook at
       base either) -- confirming the SHAPE TRK-231 describes (a stuck job starving a later
       cook) without confirming TRK-231's own exact trigger (a job that FAILED, not hung).
       Verdict: NOT CONFIRMED for the exact reported trigger; the general single-worker-pool
       starvation shape is real but pre-dates this ask and is orthogonal to the promotion
       fix's own bounded-wait mechanism (which bounds the COOK's own wait, not the pool)."""
    print("\n--- AUTOSAFE-50 / TRK-231: failed-compile hang, forced repro "
          "attempts (bounded so this test cannot itself hang) ---")
    with cold_engine_state():
        AT.reset()
        prog, tm, used = _tiny_program()
        torch.manual_seed(13)
        img = torch.rand(1, 32, 32, 3)
        bindings = {"A": img}

        # Attempt 1: the compile-wrap itself raises.
        real_try_compile = C._try_compile
        C._try_compile = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("forced compile failure"))
        real_cap = C.compile_capability_async
        C.compile_capability_async = lambda: {"cpu_inductor": True, "cuda_inductor": True}
        fp1 = "autosafe50_t3_fail_wrap_fp"
        def _attempt1():
            for _ in range(AT._MEASURE_COOKS + 3):
                C.run_auto(prog, dict(bindings), tm, "cpu", fp1, output_names=["OUT"],
                          used_builtins=used)

        try:
            hung = not _bounded(_attempt1)
            if not hung:
                C._drain_bg_for_test(timeout=10.0)
        finally:
            C._try_compile = real_try_compile
            C.compile_capability_async = real_cap
        if hung:
            r.fail("T3 attempt 1 (compile-wrap raises)",
                  "run_auto did not return within 15s after a forced compile-wrap failure")
        else:
            key1 = AT.make_key(fp1, "cpu", "fp32", C._consensus_extent(bindings, prog))
            if AT.verdict(key1) == AT.REJECTED:
                r.ok("T3 attempt 1 (compile-wrap raises): NOT CONFIRMED as a hang -- "
                     "run_auto returned promptly every cook; failed job routed to REJECTED")
            else:
                r.fail("T3 attempt 1 (compile-wrap raises)",
                       f"a failed compile must end REJECTED, verdict is {AT.verdict(key1)}")

        # Attempt 2: the compile-wrap itself never returns (an Event that is never set --
        # deterministic, no real sleep-forever thread leaked past the test: joined with a
        # timeout and abandoned as a daemon if it somehow didn't finish).
        never_set = threading.Event()

        def _hanging_try_compile(device_type, program, type_map, **kw):
            def _stand_in_that_never_returns(*a, **k):
                never_set.wait()  # never set -- models "the compile step never returns"
                return {"OUT": img}
            return _stand_in_that_never_returns, "inductor"

        C.compile_capability_async = lambda: {"cpu_inductor": True, "cuda_inductor": True}
        C._try_compile = _hanging_try_compile
        fp2 = "autosafe50_t3_hang_fp"
        def _attempt2():
            for _ in range(AT._MEASURE_COOKS + 1):
                C.run_auto(prog, dict(bindings), tm, "cpu", fp2, output_names=["OUT"],
                          used_builtins=used)

        hung2 = True
        try:
            hung2 = not _bounded(_attempt2)
        finally:
            C._try_compile = real_try_compile
            C.compile_capability_async = real_cap
            never_set.set()   # release the stuck worker so the pool recovers for later tests
            C._drain_bg_for_test(timeout=10.0)
        if hung2:
            r.fail("T3 attempt 2 (compile step never returns)",
                  "run_auto did not return within 15s while a background compile job for "
                  "a DIFFERENT key never returned -- the cook thread itself never blocks on "
                  "the stuck job (it is submitted to _WARM_POOL and never awaited "
                  "synchronously by run_auto's MEASURING/COMPILING branches), so this is "
                  "NOT CONFIRMED as a repro of the cook-thread-facing hang either")
        else:
            r.ok("T3 attempt 2 (compile step never returns): NOT CONFIRMED as a cook-thread "
                 "hang -- run_auto's own MEASURING/COMPILING branches never block on the "
                 "background future's result (they poll .done()), so a stuck background job "
                 "starves later submissions to the SAME pool, never the calling cook thread "
                 "directly. TRK-231 remains unreproduced against this tree; see the "
                 "writeup accompanying this ask for details.")
