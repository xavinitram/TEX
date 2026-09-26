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


def test_fixprof_f1_capture_does_not_pollute_outer_stage_sink(r: SubTestResult):
    """FIX-PROF F1: `GraphedProgram.capture()` runs its warmup (3x) and graph-capture (1x)
    passes through separate `Interpreter.execute()` calls on the SAME thread as the outer
    cook. If an outer `profile.measure(stages=True)` is open around the statement that
    triggers capture (real shape: `tex_engine.run`'s dispatch, reproduced directly here via
    `run_graphed` under a `measure` block with no outer statements of its own), those 4 nested
    executions must record ZERO stage boundaries into the outer sample -- capture is not part
    of the cook being timed. Before the fix each nested execute's own single-stage boundary
    landed in the outer thread-local list (real GPU work, no mocks needed: the corruption is a
    thread-local aliasing bug, not a timing artifact)."""
    if not torch.cuda.is_available():
        r.skip("FIX-PROF F1", "no CUDA on this box")
        return
    import TEX_Wrangle.tex_runtime.graphed as G
    from TEX_Wrangle.tex_cache import parse_and_split

    G.clear_graph_cache()
    P.reset()
    try:
        with armed_profiler() as Pmod:
            bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
            code = "@OUT = vec4(sin(@A) * 0.5 + 0.5, 1.0);"
            prog = parse_and_split(code, bt)
            tm = TypeChecker(binding_types=bt, source=code).check(prog)
            used = _collect_identifiers(prog)
            img = torch.rand(1, 32, 32, 3, device="cuda")

            calls = []
            real_boundary = Pmod.record_stage_boundary

            def _spy(events, stage, device):
                calls.append(stage)
                return real_boundary(events, stage, device)

            Pmod.record_stage_boundary = _spy
            try:
                key = Pmod.make_key("fixprof-f1", "cuda", "fp32")
                with Pmod.measure(key, 32 * 32, device="cuda", stages=True):
                    out = G.run_graphed(prog, {"A": img}, tm, "cuda", "fixprof_f1_fp",
                                        output_names=["OUT"], used_builtins=used)
            finally:
                Pmod.record_stage_boundary = real_boundary
            torch.cuda.synchronize()   # invariant #6: settle the box before the next test times
        assert out is not None, "program was not captured (capturability gate rejected it)"
    except Exception as e:
        r.fail("FIX-PROF F1", f"setup/capture raised: {e!r}")
        return
    finally:
        G.clear_graph_cache()
        P.reset()

    if len(calls) == 0:
        r.ok("capture's 3 warmup + 1 graph-capture passes recorded 0 stage boundaries into "
             "the outer measure block (which itself ran no statements of its own)")
    else:
        r.fail("FIX-PROF F1",
               f"expected 0 stage boundaries from capture's nested execute() calls landing "
               f"in the outer sink, got {len(calls)}: {calls!r}")
