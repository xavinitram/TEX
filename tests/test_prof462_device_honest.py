"""
PROF-462 — PROF-1 becomes device-honest for a sampled CUDA cook, without ever blocking the
cook path on the device.

**The problem this answers.** `tex_runtime.profile.measure` used to bracket a sampled CUDA
cook with `torch.cuda.synchronize()` at entry and exit, and `Interpreter._exec_stmts_profiled`
synchronized at every stage boundary too (the CHANGELOG's "four device barriers per sampled
cook", declined once already with the recorded reopen condition "device events read once per
cook: a different mechanism, not a tuning of this one"). A `synchronize()` stalls the calling
thread until the device drains — the same class of cost PACE-462 spends a whole lane removing
from the pacing poll loop, and it lands on `plan_checkpoints`' only input (`profile.
stage_snapshot`) and on PROF-1's whole-cook cost table.

**The fix.** `measure` now records one timing-enabled `torch.cuda.Event` at entry and one at
exit (never touching `CookResult.done`, which PACE-45 creates without `enable_timing` and
which this leaves alone in identity and cost); `Interpreter._exec_stmts_profiled` records one
more per stage boundary into `profile.stage_event_sink()`. None of these block. The pair (plus
any stage events) is handed to a small bounded pending queue and folded into the EWMA tables
the next time anything reads the table — by checking only the LAST event's `query()`, since
CUDA completes events on one stream in the order they were recorded.

**What this file pins**, per the ask's brief, with fake (no-GPU-needed) events for the
mechanism and one real-CUDA row for the real thing:

  * a sampled CUDA cook's recorded cost is the DEVICE's ms (from `elapsed_time`), not the
    host's wall-clock ms — `test_prof462_records_device_ms_not_host_ms`;
  * a sample whose end event has not yet signalled is NOT folded until it has —
    `test_prof462_lazy_fold_defers_until_query_true`;
  * the pending queue is bounded even if nothing ever resolves —
    `test_prof462_bounded_pending_queue`;
  * the CPU path is byte-for-behaviour unchanged and never queues anything —
    `test_prof462_cpu_path_unchanged_and_queues_nothing`;
  * one real CUDA cook, sampled with stages, actually resolves to a positive device ms —
    `test_prof462_real_cuda_cook_resolves_device_ms` (skip off CUDA).
"""
import time

import pytest

from helpers import *  # noqa: F401,F403  (SubTestResult, torch, make_img)
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import profile as P
from TEX_Wrangle.tex_runtime import pacing as _pace
from TEX_Wrangle.tex_testkit import armed_profiler


#: A fake device tick is worth this many "device ms" — deliberately far from any real
#: wall-clock duration this test could produce, so a reading equal to it (rather than to the
#: host-side sleep) can only have come from the fake event's `elapsed_time`, never `perf_counter`.
_TICK_MS = 250.0


class _FakeEvent:
    """A stand-in for `torch.cuda.Event(enable_timing=True)`. `record()` stamps a global,
    monotonically increasing tick (mirroring real CUDA events completing, on one stream, in
    the order they were recorded); `query()` answers a class-level flag so a test can hold a
    sample "still running" and then flip it; `elapsed_time` is pure arithmetic on the ticks,
    never a wall clock. No real CUDA context is touched anywhere below (`torch.cuda.Event` is
    monkeypatched), the same discipline `test_fixobsroute46_pacing.py`'s `_DeviceSpy` uses."""
    _next_tick = [0]
    DONE = True   # class-level: every existing instance's query() reads this

    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing
        self._tick = None

    def record(self):
        self._tick = _FakeEvent._next_tick[0]
        _FakeEvent._next_tick[0] += 1

    def query(self):
        return _FakeEvent.DONE

    def elapsed_time(self, other) -> float:
        return (other._tick - self._tick) * _TICK_MS


class _FakeDeviceCtx:
    """A no-op stand-in for `torch.cuda.device(...)` (the O4 discipline `measure._new_event`
    and `record_stage_boundary` both use). On a CI-shape interpreter with CPU-only torch, the
    REAL `torch.cuda.device.__enter__` calls into CUDA-only C bindings and raises even with
    `torch.cuda.Event` itself monkeypatched -- exactly the gap `test_fixobsroute46_pacing.py`'s
    `_DeviceSpy` exists to close for `pacing.py`'s identical idiom. Needed here too."""

    def __init__(self, dev):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _EventPatch:
    """Swap `torch.cuda.Event`/`torch.cuda.device`/`torch.cuda.is_available` for fakes, for
    the duration of a block, always restoring them — `tests/run_all.py` runs the whole suite
    in one process, so a leaked patch would break every later test that touches CUDA."""

    def __enter__(self):
        self._real_event = torch.cuda.Event
        self._real_device = torch.cuda.device
        self._real_avail = torch.cuda.is_available
        torch.cuda.Event = _FakeEvent
        torch.cuda.device = _FakeDeviceCtx
        torch.cuda.is_available = lambda: True
        _FakeEvent.DONE = True
        _FakeEvent._next_tick[0] = 0
        return self

    def __exit__(self, *exc):
        torch.cuda.Event = self._real_event
        torch.cuda.device = self._real_device
        torch.cuda.is_available = self._real_avail
        _FakeEvent.DONE = True
        return False


class _DeviceSpy:
    """FIX-PROF F5: the same mechanism scaffold `test_fixobsroute46_pacing.py::_DeviceSpy`
    uses to drive `pacing.py`'s CUDA branch on ANY box, CUDA or not — patches
    `torch.cuda.device`, `is_available`, `current_device` and `Event` well enough for
    `pacing.record_on`'s own `_resolve_cuda_target` call to run for real. `current` is the
    FIXED value `torch.cuda.current_device()` reports, so a test can put a genuinely
    non-current index on one side and a genuinely current one on the other. `calls` records
    every device `torch.cuda.device(...)` was actually entered with -- empty means the
    OVERHEAD-462 skip fired (already-current device); non-empty means it did not."""

    def __init__(self, current=0):
        self.calls = []
        self.current = current
        self._real_available = torch.cuda.is_available
        self._real_device_ctx = torch.cuda.device
        self._real_event = torch.cuda.Event
        self._real_current_device = torch.cuda.current_device

    def __enter__(self):
        spy = self

        def _fake_current_device():
            return spy.current

        class _SpyDeviceCtx:
            def __init__(self, dev):
                spy.calls.append(dev)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        torch.cuda.is_available = lambda: True
        torch.cuda.current_device = _fake_current_device
        torch.cuda.device = _SpyDeviceCtx
        torch.cuda.Event = _FakeEvent
        _FakeEvent.DONE = True
        _FakeEvent._next_tick[0] = 0
        return self

    def __exit__(self, *exc):
        torch.cuda.is_available = self._real_available
        torch.cuda.current_device = self._real_current_device
        torch.cuda.device = self._real_device_ctx
        torch.cuda.Event = self._real_event
        return False


def test_prof462_records_device_ms_not_host_ms(r: SubTestResult):
    print("\n--- PROF-462: a sampled CUDA cook records the DEVICE's ms ---")
    with armed_profiler():
        with _EventPatch():
            key = P.make_key("prof462-device-ms", "cuda", "fp32")
            with P.measure(key, 64 * 64, device="cuda", stages=True):
                time.sleep(0.05)   # host-side work the wall clock WOULD see
            ms = P.predict(key, 64 * 64)
        # One entry event + one exit event -> exactly one tick delta -> _TICK_MS exactly.
        # 0.05s of real sleeping would read as ~50ms if this were wall-clock; _TICK_MS=250
        # is chosen far enough from that that the two can never be confused by coincidence.
        if ms is not None and abs(ms - _TICK_MS) < 1e-6:
            r.ok(f"predict() reports {ms:.1f} ms (the fake device tick), not ~50 ms (the "
                 f"real sleep) -- the recording came from elapsed_time, not perf_counter")
        else:
            r.fail("PROF-462 device ms", f"expected {_TICK_MS}, got {ms!r}")


def test_prof462_lazy_fold_defers_until_query_true(r: SubTestResult):
    print("\n--- PROF-462: an unresolved sample is not folded until its event signals ---")
    with armed_profiler():
        with _EventPatch():
            _FakeEvent.DONE = False   # the device "hasn't finished" this sample yet
            key = P.make_key("prof462-lazy-fold", "cuda", "fp32")
            with P.measure(key, 32 * 32, device="cuda", stages=False):
                pass
            still_none = P.predict(key, 32 * 32)
            pending_while_stuck = P._pending_count()
            _FakeEvent.DONE = True    # ... now it has
            resolved = P.predict(key, 32 * 32)
        ok = (still_none is None and pending_while_stuck == 1
              and resolved is not None and abs(resolved - _TICK_MS) < 1e-6)
        if ok:
            r.ok(f"unresolved: predict()={still_none!r}, {pending_while_stuck} queued; "
                 f"after query()->True: predict()={resolved:.1f} ms")
        else:
            r.fail("PROF-462 lazy fold",
                   f"still_none={still_none!r} pending={pending_while_stuck} "
                   f"resolved={resolved!r}")


def test_prof462_bounded_pending_queue(r: SubTestResult):
    print("\n--- PROF-462: the pending queue is bounded even if nothing ever resolves ---")
    P.reset()
    try:
        with _EventPatch():
            _FakeEvent.DONE = False   # nothing will ever be foldable in this test
            key = P.make_key("prof462-bounded", "cuda", "fp32")
            for i in range(P._PENDING_MAX * 3):
                start, end = _FakeEvent(True), _FakeEvent(True)
                start.record()
                end.record()
                P._queue_pending(key, 16 * 16, start, end, None)
            n = P._pending_count()
        if n == P._PENDING_MAX:
            r.ok(f"queued {P._PENDING_MAX * 3} never-resolving samples, "
                 f"{n} remain (bounded at _PENDING_MAX={P._PENDING_MAX})")
        else:
            r.fail("PROF-462 bounded queue",
                   f"expected exactly {P._PENDING_MAX} queued, got {n}")
    finally:
        P.reset()


def test_prof462_cpu_path_unchanged_and_queues_nothing(r: SubTestResult):
    print("\n--- PROF-462: the CPU path is unchanged and never queues anything ---")
    with armed_profiler():
        P.reset()
        A = make_img(1, 64, 64, 4, seed=4620)
        code = "@OUT = vec4(@A.rgb * 1.05 + vec3(0.01), 1.0);"
        for _ in range(P._WARMUP_SAMPLES + 1):
            tex_engine.cook(code, {"A": A}, device_mode="cpu")
        snap = P.snapshot()
        pending = P._pending_count()
        if snap and pending == 0:
            r.ok(f"a CPU cook recorded {len(snap)} key(s) synchronously, "
                 f"0 samples ever queued")
        else:
            r.fail("PROF-462 CPU path", f"snapshot={bool(snap)} pending={pending}")


@pytest.mark.timing
def test_prof462_real_cuda_cook_resolves_device_ms(r: SubTestResult):
    """The real thing, no mocks: a sampled CUDA cook's whole-cook cost resolves to a positive
    device ms via the lazy queue, polled (never synchronized) for up to 2s -- generous next to
    any single cook on this box, and this is a functional check, not a tight bound."""
    if not torch.cuda.is_available():
        r.skip("PROF-462 real CUDA cook", "no CUDA on this box")
        return
    with armed_profiler():
        P.reset()
        A = make_img(1, 256, 256, 4, seed=4621).cuda()
        code = "@OUT = gauss_blur(vec4(@A.rgb, 1.0), 6.0);"
        from TEX_Wrangle.tex_cache import get_cache
        from TEX_Wrangle.tex_compiler.types import TEXType
        fp = get_cache().fingerprint(code, {"A": TEXType.VEC4})
        key = P.make_key(fp, "cuda", "fp32")

        for _ in range(P._WARMUP_SAMPLES + 1):
            tex_engine.cook(code, {"A": A}, device_mode="cuda")

        deadline = time.perf_counter() + 2.0
        ms = None
        while time.perf_counter() < deadline:
            ms = P.predict(key, 256 * 256)
            if ms is not None:
                break
            time.sleep(0.01)
        torch.cuda.synchronize()   # invariant #6: settle the box before the NEXT test times

    if ms is not None and ms > 0.0:
        r.ok(f"a sampled CUDA cook resolved to {ms:.3f} ms of device time via the lazy queue")
    else:
        r.fail("PROF-462 real CUDA cook", f"predict() never resolved within 2s (got {ms!r})")


def test_fixprof_f1_capture_suspends_the_outer_stage_sink(r: SubTestResult):
    """FIX-PROF F1: `GraphedProgram.capture()` runs its warmup (3x) and graph-capture (1x)
    passes through separate `Interpreter.execute()` calls on the SAME thread as the outer
    cook. If an outer `profile.measure(stages=True)` is open around the statement that
    triggers capture, those nested executions must not see the outer cook's sink/event-list
    as their own — `capture()` now wraps its whole body in `profile.suspend_stage_tracking()`.

    No CUDA needed: the corruption (and the fix) is a THREAD-LOCAL aliasing question, not a
    timing one, so this drives the real, unmodified `capture()` method and stubs only the two
    things that are genuinely CUDA-hardware-shaped and orthogonal to the profiler bug --
    `_capture_inner` (the actual warmup/graph-capture work) and `torch.cuda.device` (the O4
    device-context guard `capture()` takes before calling it) -- the same style of stub
    `_restoring_cuda_stream`'s own docstring names ("a test drives `torch.cuda.device` mocked
    to abort before any real CUDA op"). `_restoring_cuda_stream` itself needs no stub: its
    `try/except` already tolerates a CUDA-absent `current_stream()` by design. The stub
    observes, from INSIDE the stand-in for the nested work, exactly what a real nested
    `Interpreter.execute()` would ask `profile.stage_sink()`/`stage_event_sink()` for."""
    import TEX_Wrangle.tex_runtime.graphed as G

    P.reset()
    observed = {}

    def _fake_capture_inner(self, *a, **kw):
        # What a nested Interpreter.execute() would see if it asked right now.
        observed["stage_sink_during"] = P.stage_sink()
        observed["stage_event_sink_during"] = P.stage_event_sink()
        return True

    real_capture_inner = G.GraphedProgram._capture_inner
    real_cuda_device = torch.cuda.device
    ok = restored_sink_ok = restored_events_ok = None
    try:
        G.GraphedProgram._capture_inner = _fake_capture_inner
        torch.cuda.device = _FakeDeviceCtx
        with armed_profiler() as Pmod:
            key = Pmod.make_key("fixprof-f1-mocked", "cuda", "fp32")
            with Pmod.measure(key, 8 * 8, device="cuda", stages=True):
                # The OUTER cook's own sink/event-list, as a real nested execute() would
                # see them right now, before capture() runs.
                outer_sink = P.stage_sink()
                outer_events = P.stage_event_sink()

                gp = G.GraphedProgram(("fixprof-f1-mocked-key", 0))
                ok = gp.capture(program=None, bindings={}, type_map=None,
                                device="cuda:0", latent_channel_count=0,
                                output_names=None, precision="fp32", used_builtins=None)

                # Right after capture() returns (still inside the outer `with`): is the
                # ambient sink/event-list back to these SAME outer objects?
                restored_sink_ok = P.stage_sink() is outer_sink
                restored_events_ok = P.stage_event_sink() is outer_events
    except Exception as e:
        r.fail("FIX-PROF F1 (mocked)", f"setup/capture raised: {e!r}")
        return
    finally:
        G.GraphedProgram._capture_inner = real_capture_inner
        torch.cuda.device = real_cuda_device
        P.reset()

    ok_all = (ok is True
              and observed.get("stage_sink_during") is None
              and observed.get("stage_event_sink_during") is None
              and restored_sink_ok and restored_events_ok)
    if ok_all:
        r.ok(f"capture()'s (stubbed) body observed a SUSPENDED sink/event-list "
             f"({observed!r}), and the outer measure block's own sink/event-list were "
             f"restored afterward -- no CUDA device needed for this check")
    else:
        r.fail("FIX-PROF F1 (mocked)",
               f"ok={ok!r} observed={observed!r} restored_sink_ok={restored_sink_ok!r} "
               f"restored_events_ok={restored_events_ok!r}")


def test_fixprof_f2_failed_boundary_drops_stage_split_not_a_neighbour(r: SubTestResult):
    """FIX-PROF F2: `record_stage_boundary`'s swallowed construction/`record()` failure used
    to append nothing, so the sequential fold (`prev.elapsed_time(ev)`, walking `events` in
    order) silently handed the failed stage's real cost to whichever stage closed NEXT --
    wrong in a different way than "missing". A 3-stage sample where stage 1's boundary fails
    must instead record its WHOLE-COOK time (already timed independently by the entry/exit
    events) with NO per-stage split at all for that one sample, rather than a split where
    stage 2's number silently contains stage 1's work too."""
    P.reset()

    class _FlakyEvent(_FakeEvent):
        """A `_FakeEvent` whose `record()` raises on demand, to simulate the transient CUDA
        event-creation/record failure `record_stage_boundary`'s docstring calls "vanishingly
        rare... not impossible"."""
        _fail_next = [False]

        def record(self):
            if _FlakyEvent._fail_next[0]:
                _FlakyEvent._fail_next[0] = False
                raise RuntimeError("simulated CUDA event failure")
            super().record()

    real_event, real_device, real_avail = torch.cuda.Event, torch.cuda.device, torch.cuda.is_available
    try:
        with armed_profiler() as Pmod:
            torch.cuda.Event = _FlakyEvent
            torch.cuda.device = _FakeDeviceCtx
            torch.cuda.is_available = lambda: True
            _FakeEvent.DONE = True
            _FakeEvent._next_tick[0] = 0
            _FlakyEvent._fail_next[0] = False

            key = Pmod.make_key("fixprof-f2", "cuda", "fp32")
            events = []
            start = _FlakyEvent(True)
            start.record()
            Pmod.record_stage_boundary(events, 0, "cuda")     # stage 0: ok
            _FlakyEvent._fail_next[0] = True
            Pmod.record_stage_boundary(events, 1, "cuda")     # stage 1: FAILS mid-cook
            Pmod.record_stage_boundary(events, 2, "cuda")     # stage 2: ok
            end = _FlakyEvent(True)
            end.record()
            Pmod._queue_pending(key, 8 * 8, start, end, events)

            whole = Pmod.predict(key, 8 * 8)
            stages = Pmod.stage_costs(key, 8 * 8)
    finally:
        torch.cuda.Event, torch.cuda.device, torch.cuda.is_available = real_event, real_device, real_avail
        P.reset()

    ok = (whole is not None and whole > 0.0 and stages == {})
    if ok:
        r.ok(f"a failed mid-cook boundary recorded whole-cook ms={whole:.1f} and dropped "
             f"the per-stage split entirely (stages={stages!r}), instead of silently "
             f"folding stage 1's cost into stage 2's number")
    else:
        r.fail("FIX-PROF F2",
               f"whole={whole!r} stages={stages!r} (expected whole > 0 and stages == {{}})")


def test_fixprof_f3_warmup_gate_counts_inflight_pending(r: SubTestResult):
    """FIX-PROF F3: `should_sample`'s warmup check reads `.samples`, which only advances at
    FOLD time (lazy since PROF-462). If several cooks of a brand-new key arrive faster than
    their device events resolve -- exactly the interactive-host burst PACE-462/PROF-462 target
    -- every one of them sees `.samples == 0` and gets sampled, because none of their
    predecessors' folds have landed yet. A burst that never resolves must still be bounded at
    `_WARMUP_SAMPLES`, not sample the whole burst."""
    P.reset()
    try:
        with armed_profiler() as Pmod:
            with _EventPatch():
                _FakeEvent.DONE = False       # nothing in this burst ever resolves
                key = Pmod.make_key("fixprof-f3", "cuda", "fp32")
                burst = Pmod._WARMUP_SAMPLES + 5
                sampled = []
                for _ in range(burst):
                    on = Pmod.should_sample(key, 16 * 16)
                    sampled.append(on)
                    if on:
                        start, end = _FakeEvent(True), _FakeEvent(True)
                        start.record()
                        end.record()
                        Pmod._queue_pending(key, 16 * 16, start, end, None)
                warmup_hits = sum(sampled)
    finally:
        P.reset()

    if warmup_hits == P._WARMUP_SAMPLES:
        r.ok(f"a {burst}-cook burst with nothing ever resolving sampled exactly "
             f"{warmup_hits} times (the warmup budget), not the whole burst")
    else:
        r.fail("FIX-PROF F3",
               f"expected exactly {P._WARMUP_SAMPLES} warmup hits over a burst of "
               f"{burst}, got {warmup_hits}: {sampled}")


def test_fixprof_f4_lock_is_plain_not_reentrant(r: SubTestResult):
    """FIX-PROF F4: `profile._LOCK` went from a plain `Lock` to an `RLock` only because the
    lazy-fold drain used to call the PUBLIC `record`/`record_stages` (each of which takes
    `_LOCK` itself) from INSIDE a block that already held `_LOCK`. That reentrancy is removed
    (the drain now runs before any caller takes `_LOCK`, never inside it), so `_LOCK` should be
    a plain, non-reentrant `Lock` again -- the module's own comment calls it "a few dict
    operations", a claim only true of a lock nothing ever has to reenter."""
    P.reset()
    reentrant_ok = None
    ms = None
    try:
        P._LOCK.acquire()
        try:
            reentrant_ok = P._LOCK.acquire(blocking=False)
            if reentrant_ok:
                P._LOCK.release()
        finally:
            P._LOCK.release()

        # Functional: the drain/fold path must still work correctly with a plain Lock -- a
        # sampled cook that queues and resolves normally proves `_drain_pending` never calls
        # `record`/`record_stages` while holding `_LOCK`.
        with armed_profiler() as Pmod:
            with _EventPatch():
                key = Pmod.make_key("fixprof-f4", "cuda", "fp32")
                with Pmod.measure(key, 8 * 8, device="cuda", stages=False):
                    pass
                ms = Pmod.predict(key, 8 * 8)
    finally:
        P.reset()

    if (reentrant_ok is False) and ms is not None:
        r.ok(f"_LOCK refuses re-entry (plain Lock, not RLock) and the fold path still "
             f"resolves normally (predict()={ms!r})")
    else:
        r.fail("FIX-PROF F4",
               f"reentrant_ok={reentrant_ok!r} predict()={ms!r} "
               f"(expected a non-reentrant lock and a working fold)")


def test_fixprof_f5_record_stage_boundary_shares_pacing_record_on(r: SubTestResult):
    """FIX-PROF F5: `record_stage_boundary` now records its event via `pacing.record_on`
    instead of its own `with torch.cuda.device(...):` -- so it must show the SAME
    device-context-skip mechanism `pacing.py`'s own poll points do: enter
    `torch.cuda.device(...)` for a genuinely non-current device, skip it entirely for an
    already-current one."""
    print("\n--- FIX-PROF F5: record_stage_boundary shares pacing.record_on ---")
    ok_noncurrent = ok_current = False
    try:
        with _DeviceSpy(current=0) as spy:
            events = []
            P.record_stage_boundary(events, 0, "cuda:1")
        ok_noncurrent = (spy.calls == ["cuda:1"] and len(events) == 1
                         and events[0][0] == 0 and events[0][1] is not None)
    except Exception as e:
        r.fail("FIX-PROF F5 record_stage_boundary (non-current)", str(e))
        return
    try:
        with _DeviceSpy(current=0) as spy2:
            events2 = []
            P.record_stage_boundary(events2, 0, "cuda:0")
        ok_current = (spy2.calls == [] and len(events2) == 1 and events2[0][1] is not None)
    except Exception as e:
        r.fail("FIX-PROF F5 record_stage_boundary (current)", str(e))
        return

    if ok_noncurrent and ok_current:
        r.ok("record_stage_boundary entered torch.cuda.device('cuda:1') for a non-current "
             "device and skipped it entirely for an already-current one -- via "
             "pacing.record_on, not its own with torch.cuda.device(...)")
    else:
        r.fail("FIX-PROF F5 record_stage_boundary",
               f"noncurrent_ok={ok_noncurrent!r} current_ok={ok_current!r}")


def test_fixprof_f5_new_event_shares_pacing_record_on(r: SubTestResult):
    """FIX-PROF F5: `measure._new_event` (entry + exit events) shares the same
    `pacing.record_on` seam -- same mechanism check as above, driven through the public
    `measure` object rather than calling the private helper directly."""
    print("\n--- FIX-PROF F5: measure._new_event shares pacing.record_on ---")
    ok_noncurrent = ok_current = False
    try:
        P.reset()
        with armed_profiler() as Pmod:
            with _DeviceSpy(current=0) as spy:
                key = Pmod.make_key("fixprof-f5-noncurrent", "cuda", "fp32")
                with Pmod.measure(key, 8 * 8, device="cuda:1", stages=False):
                    pass
            ok_noncurrent = (len(spy.calls) >= 1 and all(c == "cuda:1" for c in spy.calls))
    except Exception as e:
        r.fail("FIX-PROF F5 _new_event (non-current)", str(e))
        return
    finally:
        P.reset()

    try:
        P.reset()
        with armed_profiler() as Pmod:
            with _DeviceSpy(current=0) as spy2:
                key = Pmod.make_key("fixprof-f5-current", "cuda", "fp32")
                with Pmod.measure(key, 8 * 8, device="cuda:0", stages=False):
                    pass
            ok_current = (spy2.calls == [])
    except Exception as e:
        r.fail("FIX-PROF F5 _new_event (current)", str(e))
        return
    finally:
        P.reset()

    if ok_noncurrent and ok_current:
        r.ok("measure's entry/exit event recording entered torch.cuda.device('cuda:1') for "
             "a non-current device and skipped it entirely for an already-current one -- "
             "via pacing.record_on, not its own with torch.cuda.device(...)")
    else:
        r.fail("FIX-PROF F5 measure._new_event",
               f"noncurrent_ok={ok_noncurrent!r} current_ok={ok_current!r}")
