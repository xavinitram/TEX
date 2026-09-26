"""
PACE-462 (v0.46.2) — bounded look-ahead pacing. A patch on `tex_runtime/pacing.py`
answering an embedding host's finding: the original PACE-45 mechanism (wait one
poll-interval, every poll, via an `event.query()` + fixed-sleep loop) cost a paced
background render 2.0x (sm_120) / 1.67x (sm_75) of its unpaced run time -- unacceptable
for a host that wants to pace every cancellable background cook, not just an untrusted-tool
ceiling.

This file is the MECHANISM half: deterministic, no real CUDA device and no wall-clock,
using a fake `torch.cuda.Event`/`torch.cuda.device`/`torch.cuda.is_available` (the same
`_DeviceSpy` shape `test_fixobsroute46_pacing.py` already uses) so the ring's exact
bookkeeping -- when it waits, on which event, how many `torch.cuda.Event` objects it ever
constructs -- is provable without hardware. `benchmarks/preempt_drain_bench.py` and this
file's one CUDA row (`test_pace462_cuda_drained_bound`, `timing`, skipped off CUDA) cover
the real-device cost/latency claims.

Every row here is RED against the pre-PACE-462 `pacing.py` (base `27f260e`): that module has
no `pace_depth`, no `_resolve_depth`, no event ring, and waits on every single poll once a
previous event exists (an unconditional one-poll-interval wait, not a depth-gated one) --
so `_pace._resolve_depth` does not exist at all (AttributeError) and the ring/ construction-
count assertions below have nothing matching to read.
"""
import pytest

from TEX_Wrangle.tex_runtime import pacing as _pace


@pytest.fixture(autouse=True)
def _fresh_pacing_state():
    """`pacing._state` is thread-local and, by design (the ring persists across cooks to
    amortize event construction — see `reset()`'s docstring), never clears itself between
    cooks on one thread. Tests run on this SAME thread, one after another, so without this
    fixture one test's ring/counters would leak into the next. Isolate here rather than
    changing the module's own reset() contract."""
    _pace._state.__dict__.clear()
    yield
    _pace._state.__dict__.clear()


class _Token:
    """`pace_stride_ms=0` (stride DISABLED) by default: every depth/ring test in this file
    predates the stride gate and asserts "every poll records" — mocked `_FakeEvent` calls
    execute in nanoseconds of real wall-clock time, so with the module's nonzero default
    stride, a mocked test's second-and-later polls would land inside the stride window and
    never touch the ring at all, which is a real (and separately tested, see the STRIDE
    section below) behaviour, but not what these depth/ring rows are about. The two rows
    that ARE about striding construct their own `_Token` and override `pace_stride_ms`
    explicitly."""

    def __init__(self, pace=True, pace_depth=None, pace_stride_ms=0):
        self.pace = pace
        if pace_depth is not None:
            self.pace_depth = pace_depth
        if pace_stride_ms is not None:      # pass None to get NO pace_stride_ms attribute
            self.pace_stride_ms = pace_stride_ms
        self.checks = 0

    def check(self):
        self.checks += 1


class _FakeEvent:
    """Stands in for `torch.cuda.Event`: `record()`/`synchronize()` are both no-ops that
    only count how many times they were called, so a test can assert on ORDER and COUNT
    without any real device. `query()` is not exercised by the new mechanism (it uses a
    blocking `synchronize()`, not a poll loop) but is kept for shape-compatibility."""

    _live = 0  # class-wide construction counter, reset per test

    def __init__(self, blocking=False):
        self.blocking = blocking
        self.record_calls = 0
        self.sync_calls = 0
        type(self)._live += 1
        self._id = type(self)._live

    def record(self):
        self.record_calls += 1

    def synchronize(self):
        self.sync_calls += 1

    def query(self):
        return True


class _DeviceSpy:
    """Patches enough of `torch.cuda` to drive `pacing.py`'s CUDA branch on ANY box: a
    context-manager stand-in for `torch.cuda.device`, `is_available() -> True`,
    `current_device()` fixed to 0 (OVERHEAD-462: `reset()` now calls it unconditionally
    whenever `is_available` reads True, so it MUST be mocked here too — otherwise this file
    would raise on a genuinely CPU-only torch build despite the `is_available` mock), and
    `Event` bound to `_FakeEvent`. Mirrors `test_fixobsroute46_pacing.py::_DeviceSpy`. None
    of this file's rows depend on WHICH index reads current — they assert on `_FakeEvent`
    construction/record/sync counts, not on whether the (now-optional) `torch.cuda.device(...)`
    context manager was entered."""

    def __init__(self):
        self.device_calls = []
        self._real_available = None
        self._real_device_ctx = None
        self._real_event = None
        self._real_current_device = None

    def __enter__(self):
        import torch
        spy = self
        self._real_available = torch.cuda.is_available
        self._real_device_ctx = torch.cuda.device
        self._real_event = torch.cuda.Event
        self._real_current_device = torch.cuda.current_device

        class _Ctx:
            def __init__(self, dev):
                spy.device_calls.append(dev)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        torch.cuda.is_available = lambda: True
        torch.cuda.current_device = lambda: 0
        torch.cuda.device = _Ctx
        torch.cuda.Event = _FakeEvent
        _FakeEvent._live = 0
        return self

    def __exit__(self, *exc):
        import torch
        torch.cuda.is_available = self._real_available
        torch.cuda.current_device = self._real_current_device
        torch.cuda.device = self._real_device_ctx
        torch.cuda.Event = self._real_event
        return False


# ── Depth semantics: waits only once the ring is full, on the OLDEST event ────────

def test_depth_gates_the_wait_not_every_poll(r):
    """With `pace_depth=3`, the first 3 poll points must NOT wait (the device has fewer
    than 3 poll-intervals queued so far); the 4th and 5th must each wait exactly once,
    on the OLDEST outstanding event, matching the bounded-look-ahead design."""
    print("\n--- PACE-462: depth gates the wait, not every poll ---")
    with _DeviceSpy():
        tok = _Token(pace=True, pace_depth=3)
        _pace.reset(tok, "cuda")
        events_before = []
        for i in range(5):
            _pace.paced_check(tok, "cuda")
            events_before.append(_pace._state.ring[:])  # noqa: SLF001 (white-box by design)
        ring = _pace._state.ring  # noqa: SLF001
        waits = [ev.sync_calls for ev in ring if ev is not None]
        total_waits = sum(waits)
    if total_waits == 2:
        r.ok(f"exactly 2 waits over 5 polls at depth 3 (calls 4 and 5), got sync counts {waits}")
    else:
        r.fail("PACE-462 depth gate", f"expected 2 total synchronize() calls, got {total_waits} "
               f"(per-slot: {waits})")


def test_depth_one_waits_on_every_poll_after_the_first(r):
    """`pace_depth=1` is the degenerate case: the ring holds at most one outstanding event,
    so every poll after the first must wait -- the same shape the original PACE-45
    one-poll-interval mechanism had, just expressed as depth 1 of the new ring."""
    print("\n--- PACE-462: pace_depth=1 waits on every poll but the first ---")
    with _DeviceSpy():
        tok = _Token(pace=True, pace_depth=1)
        _pace.reset(tok, "cuda")
        for _ in range(4):
            _pace.paced_check(tok, "cuda")
        ring = _pace._state.ring  # noqa: SLF001
        total_waits = sum(ev.sync_calls for ev in ring if ev is not None)
    if total_waits == 3:
        r.ok("4 polls at depth 1 produced exactly 3 waits (all but the first)")
    else:
        r.fail("PACE-462 depth=1", f"expected 3 waits, got {total_waits}")


def test_wait_rechecks_token_immediately_after_synchronize(r):
    """Contract: 'before returning from a wait, token.check() again.' At depth 1, poll 2
    must call token.check() TWICE -- once at the top of the poll (catching an
    already-tripped token before touching the device) and once more right after the
    synchronize() completes."""
    print("\n--- PACE-462: token is re-checked immediately after a wait ---")
    with _DeviceSpy():
        tok = _Token(pace=True, pace_depth=1)
        _pace.reset(tok, "cuda")
        _pace.paced_check(tok, "cuda")   # poll 1: no wait yet -> 1 check
        checks_after_1 = tok.checks
        _pace.paced_check(tok, "cuda")   # poll 2: waits -> 2 checks (before + after wait)
        checks_after_2 = tok.checks
    if checks_after_1 == 1 and checks_after_2 - checks_after_1 == 2:
        r.ok(f"poll 1 -> {checks_after_1} check(s); poll 2 (waits) -> "
             f"{checks_after_2 - checks_after_1} check(s)")
    else:
        r.fail("PACE-462 re-check after wait",
               f"poll1 checks={checks_after_1}, poll2 delta={checks_after_2 - checks_after_1}")


# ── The cheap path: the ring is built once and re-record()ed, not rebuilt ─────────

def test_ring_reuses_events_once_warm(r):
    """Over many polls at a fixed depth, the total number of `torch.cuda.Event` objects
    ever CONSTRUCTED must equal `depth`, not the number of polls -- the ask's 'cheap path'
    contract (event reuse via a re-record()ed ring, not fresh allocation every poll)."""
    print("\n--- PACE-462: the ring constructs at most `depth` events total ---")
    with _DeviceSpy() as spy:
        tok = _Token(pace=True, pace_depth=4)
        _pace.reset(tok, "cuda")
        for _ in range(25):
            _pace.paced_check(tok, "cuda")
        constructed = _FakeEvent._live
    if constructed == 4:
        r.ok(f"25 polls at depth 4 constructed exactly {constructed} Event objects (reused after)")
    else:
        r.fail("PACE-462 ring reuse", f"expected exactly 4 Event objects constructed "
               f"across 25 polls, got {constructed}")


def test_ring_grows_across_cooks_and_never_shrinks(r):
    """`reset()` between cooks on the same thread must GROW the ring when a later cook asks
    for a bigger depth, reusing the events already built for the smaller depth (so the
    first `k` slots are not reconstructed), and never shrink it when a later cook asks for
    a smaller depth (so a subsequent bigger ask doesn't pay for those slots twice)."""
    print("\n--- PACE-462: the ring grows across cooks on one thread, never shrinks ---")
    with _DeviceSpy():
        tok_small = _Token(pace=True, pace_depth=2)
        _pace.reset(tok_small, "cuda")
        for _ in range(2):
            _pace.paced_check(tok_small, "cuda")
        ring_after_small = _pace._state.ring  # noqa: SLF001
        built_after_small = _FakeEvent._live
        slot0, slot1 = ring_after_small[0], ring_after_small[1]

        tok_big = _Token(pace=True, pace_depth=5)
        _pace.reset(tok_big, "cuda")
        for _ in range(5):
            _pace.paced_check(tok_big, "cuda")
        ring_after_big = _pace._state.ring  # noqa: SLF001
        built_after_big = _FakeEvent._live

        tok_small2 = _Token(pace=True, pace_depth=2)
        _pace.reset(tok_small2, "cuda")
        ring_after_shrink_request = _pace._state.ring  # noqa: SLF001
        built_after_shrink_request = _FakeEvent._live

    ok = (built_after_small == 2 and built_after_big == 5
          and len(ring_after_shrink_request) == 5
          and ring_after_big[0] is slot0 and ring_after_big[1] is slot1
          and built_after_shrink_request == 5)
    if ok:
        r.ok("ring grew 2->5 events (reusing the first two), then held at 5 for a smaller ask")
    else:
        r.fail("PACE-462 ring growth",
               f"built: small={built_after_small} big={built_after_big} "
               f"after_shrink_request={built_after_shrink_request}, "
               f"ring len after shrink request={len(ring_after_shrink_request)}, "
               f"slot reuse={ring_after_big[0] is slot0 and ring_after_big[1] is slot1}")


# ── pace_depth validation ──────────────────────────────────────────────────────────

def test_pace_depth_default_when_absent(r):
    """A token with no `pace_depth` attribute at all resolves to `_pace._DEFAULT_DEPTH`."""
    print("\n--- PACE-462: pace_depth absent resolves to the module default ---")
    tok = _Token(pace=True)
    resolved = _pace._resolve_depth(tok)  # noqa: SLF001 (white-box by design)
    if resolved == _pace._DEFAULT_DEPTH:  # noqa: SLF001
        r.ok(f"resolved default depth {resolved}")
    else:
        r.fail("PACE-462 default depth", f"expected {_pace._DEFAULT_DEPTH}, got {resolved}")


@pytest.mark.parametrize("bad_depth", [0, -1, 2.5, "3", True, False])
def test_pace_depth_rejects_non_positive_int(bad_depth):
    """`pace_depth` must be a plain positive `int`. `type(x) is int` (not `isinstance`)
    deliberately rejects `bool` -- `True == 1` in Python, and depth must never be read out
    of the `pace` flag's own truthiness by accident."""
    tok = _Token(pace=True, pace_depth=bad_depth)
    with pytest.raises(ValueError):
        _pace._resolve_depth(tok)  # noqa: SLF001


def test_pace_depth_accepts_positive_int(r):
    print("\n--- PACE-462: a valid pace_depth is honoured exactly ---")
    tok = _Token(pace=True, pace_depth=7)
    resolved = _pace._resolve_depth(tok)  # noqa: SLF001
    if resolved == 7:
        r.ok("pace_depth=7 resolved to 7")
    else:
        r.fail("PACE-462 valid depth", f"expected 7, got {resolved}")


def test_pace_depth_rejects_above_the_ceiling():
    """P4 (Phase C, B1#5): `pace_depth` had no upper bound -- an oversized value (confirmed
    accepted: `pace_depth=1_000_000_000`) grows the per-thread ring in one unbounded Python
    list allocation, several GB before a single `torch.cuda.Event` is even built. Must now
    raise rather than resolve cleanly."""
    tok = _Token(pace=True, pace_depth=1_000_000_000)
    with pytest.raises(ValueError):
        _pace._resolve_depth(tok)  # noqa: SLF001


def test_reset_raises_on_invalid_pace_depth(r):
    """The validation fires at `reset()` time (once per cook), not lazily inside the poll
    loop -- a bad `pace_depth` must fail fast, before any device work is queued."""
    print("\n--- PACE-462: reset() raises immediately on an invalid pace_depth ---")
    with _DeviceSpy():
        tok = _Token(pace=True, pace_depth=0)
        try:
            _pace.reset(tok, "cuda")
        except ValueError:
            r.ok("reset() raised ValueError for pace_depth=0")
            return
    r.fail("PACE-462 reset validation", "reset() did not raise for an invalid pace_depth")


# ── Unpaced / None / CPU paths stay exactly what they were ────────────────────────

def test_unpaced_never_touches_cuda_state(r):
    """A token with `pace=False`, or a CPU device, must never call into `torch.cuda` at
    all -- the unpaced cost is one `token.check()`, nothing else, regardless of any
    `pace_depth` the token happens to also carry."""
    print("\n--- PACE-462: unpaced polls never touch torch.cuda ---")
    with _DeviceSpy() as spy:
        tok = _Token(pace=False, pace_depth=99)
        _pace.reset(tok, "cpu")
        for _ in range(5):
            _pace.paced_check(tok, "cpu")
    if tok.checks == 5 and spy.device_calls == []:
        r.ok("5 unpaced polls: 5 token.check() calls, 0 torch.cuda.device(...) entries")
    else:
        r.fail("PACE-462 unpaced path",
               f"checks={tok.checks} (want 5), device_calls={spy.device_calls} (want [])")


def test_none_token_is_a_no_op(r):
    print("\n--- PACE-462: paced_check(None, ...) is a no-op ---")
    with _DeviceSpy() as spy:
        _pace.reset(None, "cuda")
        _pace.paced_check(None, "cuda")
    if spy.device_calls == []:
        r.ok("paced_check(None, ...) touched no torch.cuda state")
    else:
        r.fail("PACE-462 None token", f"expected no device calls, got {spy.device_calls}")


# ── One real-CUDA row: the drained-p95 bound, at a small pace_depth ───────────────
#
# There is no CPU witness for "how much device work is left behind after a pre-emption" --
# a CPU cook has no async queue at all. Mirrors test_pace45_pacing.py's heavy-chain repro
# shape and its measurement discipline (discard a cold leg, floor of two warm legs).

from helpers import torch, time, make_img  # noqa: E402  (after the mechanism-only tests)
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime.host import CookCancelled

_N_STATEMENTS = 20
_SIZE = 2048
_PROGRAM = "vec4 x = @A;\n" + "x = gauss_blur(x, 8.0);\n" * _N_STATEMENTS + "@OUT = x;\n"


def _bindings(seed):
    return {"A": make_img(1, _SIZE, _SIZE, 4, seed=seed).cuda()}


def _measure_full_runtime():
    def once(seed):
        t0 = time.perf_counter()
        tex_engine.cook(_PROGRAM, _bindings(seed), device_mode="cuda")
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    once(940)  # discard cold leg
    return min(once(941), once(942))


class _StatementTripToken:
    """PACE-462: trips deterministically once `trip_after` top-level statements have been
    DISPATCHED, via the SAME `on_progress("stmt", ...)` callback the interpreter/codegen
    tiers already report per statement -- immune to wall-clock/GPU-clock variance, unlike a
    background `threading.Timer` calibrated against a separately-measured runtime (an
    earlier version of this test used exactly that and was flaky on this box's laptop GPU,
    which clock-ramps under load: a since-boosted paced cook's own host-return time could
    legitimately land under a delay calibrated a few cooks earlier at a slower clock,
    racing the trip to the finish before it ever fired). Pass as BOTH `cancel=` and
    `on_progress=self.on_progress`."""
    def __init__(self, trip_after, pace_depth=None):
        self.pace = True
        if pace_depth is not None:
            self.pace_depth = pace_depth
        self._trip_after = trip_after
        self._count = 0
        self._tripped = False

    def on_progress(self, phase, frac):
        if phase == "stmt":
            self._count += 1
            if self._count >= self._trip_after:
                self._tripped = True

    def check(self):
        if self._tripped:
            raise CookCancelled("PACE-462 drained-bound repro: statement-count trip fired")


@pytest.mark.timing
def test_pace462_cuda_drained_bound(r):
    """With a small `pace_depth` (2), a pre-empted background cook must leave only a FEW
    poll-intervals of device work behind: the time from 'the token catches the trip' to
    'the device is fully drained' should be a small fraction of the uncancelled full
    runtime -- the whole point of bounding the look-ahead. Quiet-box informational,
    deselected by default (`-m 'not timing'`)."""
    if not torch.cuda.is_available():
        r.skip("PACE-462 drained bound", "no CUDA on this box")
        return

    full = _measure_full_runtime()
    per_statement = full / _N_STATEMENTS
    depth = 2
    trip_after = _N_STATEMENTS // 4
    tok = _StatementTripToken(trip_after, pace_depth=depth)

    t_trip_seen = None
    try:
        tex_engine.cook(_PROGRAM, _bindings(950), device_mode="cuda", cancel=tok,
                        on_progress=tok.on_progress)
        r.fail("PACE-462 drained bound", f"cook completed without raising (tripped after "
               f"statement {trip_after}/{_N_STATEMENTS})")
        return
    except CookCancelled:
        t_trip_seen = time.perf_counter()
    torch.cuda.synchronize()
    t_drained = time.perf_counter()

    drain_tail = t_drained - t_trip_seen
    # Generous, box-robust bound: a few poll-intervals' worth of device time, not the
    # whole remaining queue. `depth + 2` intervals covers the ring's own bound plus slack
    # for the statement straddling the trip and scheduling noise.
    bound = per_statement * (depth + 2)
    if drain_tail < max(bound, 0.05):
        r.ok(f"drain tail after the caught trip: {drain_tail * 1000:.1f}ms, under the "
             f"bound {max(bound, 0.05) * 1000:.1f}ms (per-statement {per_statement * 1000:.2f}ms, "
             f"full runtime {full * 1000:.0f}ms, depth {depth})")
    else:
        r.fail("PACE-462 drained bound", f"drain tail {drain_tail * 1000:.1f}ms exceeds bound "
               f"{max(bound, 0.05) * 1000:.1f}ms")


# ── The STRIDE: polls inside the window record nothing; the first poll past it records ──
#
# Deterministic, no wall-clock: a fake `_pace._time.perf_counter` under direct control, so
# "has `stride` seconds of HOST time passed since the ring last recorded" is provable
# without ever actually sleeping.

class _FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_stride_gates_the_ring_not_the_token_check(r):
    """With a stride of 10ms and a large depth (so the ring-full wait never fires in this
    test), a poll 1ms after the ring last recorded must touch NEITHER the ring nor
    `torch.cuda` beyond `token.check()`; a poll 20ms after must record."""
    print("\n--- PACE-462 stride: polls inside the window record nothing ---")
    clock = _FakeClock(0.0)
    real_perf_counter = _pace._time.perf_counter
    _pace._time.perf_counter = clock
    try:
        with _DeviceSpy() as spy:
            tok = _Token(pace=True, pace_depth=8)
            tok.pace_stride_ms = 10.0
            _pace.reset(tok, "cuda")

            _pace.paced_check(tok, "cuda")             # t=0.0: no prior record -> records
            constructed_after_first = _FakeEvent._live
            count_after_first = _pace._state.count  # noqa: SLF001

            clock.advance(0.001)                        # t=0.001: 1ms since last record
            _pace.paced_check(tok, "cuda")               # inside the 10ms stride -> skip
            constructed_after_inside = _FakeEvent._live
            count_after_inside = _pace._state.count  # noqa: SLF001

            clock.advance(0.019)                         # t=0.020: 19ms since last record
            _pace.paced_check(tok, "cuda")               # past the stride -> records
            count_after_past = _pace._state.count  # noqa: SLF001
    finally:
        _pace._time.perf_counter = real_perf_counter

    checks_total = tok.checks
    ok = (count_after_first == 1 and constructed_after_first == 1
          and count_after_inside == 1 and constructed_after_inside == 1
          and count_after_past == 2
          and checks_total == 3)
    if ok:
        r.ok(f"first poll recorded (count=1), the 1ms-later poll stayed inside the stride "
             f"window (still count=1, 0 new events), the 19ms-later poll recorded (count=2); "
             f"token.check() called {checks_total} times across all 3 polls")
    else:
        r.fail("PACE-462 stride gate",
               f"counts: first={count_after_first} constructed={constructed_after_first}, "
               f"inside={count_after_inside} constructed={constructed_after_inside}, "
               f"past={count_after_past}, token.checks={checks_total}")


def test_stride_zero_disables_the_gate(r):
    """`pace_stride_ms=0` must record at EVERY poll, exactly as depth-only pacing (no
    stride at all) did -- the explicit escape hatch back to the pre-stride mechanism."""
    print("\n--- PACE-462 stride: pace_stride_ms=0 records every poll ---")
    with _DeviceSpy():
        tok = _Token(pace=True, pace_depth=8)
        tok.pace_stride_ms = 0
        _pace.reset(tok, "cuda")
        for _ in range(5):
            _pace.paced_check(tok, "cuda")
        count = _pace._state.count  # noqa: SLF001
    if count == 5:
        r.ok("pace_stride_ms=0 recorded on all 5 polls (stride disabled)")
    else:
        r.fail("PACE-462 stride=0", f"expected count=5 (every poll recorded), got {count}")


def test_pace_stride_ms_default_when_absent(r):
    print("\n--- PACE-462: pace_stride_ms absent resolves to the module default ---")
    tok = _Token(pace=True, pace_stride_ms=None)  # None -> no pace_stride_ms attribute at all
    resolved = _pace._resolve_stride(tok)  # noqa: SLF001
    if resolved == _pace._DEFAULT_STRIDE_S:  # noqa: SLF001
        r.ok(f"resolved default stride {resolved}s")
    else:
        r.fail("PACE-462 default stride", f"expected {_pace._DEFAULT_STRIDE_S}, got {resolved}")


@pytest.mark.parametrize("bad_stride", [-1, -0.5, "3", True, False])
def test_pace_stride_ms_rejects_invalid(bad_stride):
    """`pace_stride_ms` must be a plain non-negative `int` or `float` -- `bool` rejected
    (`type(x) not in (int, float)`), negative rejected."""
    tok = _Token(pace=True)
    tok.pace_stride_ms = bad_stride
    with pytest.raises(ValueError):
        _pace._resolve_stride(tok)  # noqa: SLF001


@pytest.mark.parametrize("good_stride,expected_s", [(0, 0.0), (5, 0.005), (2.5, 0.0025)])
def test_pace_stride_ms_accepts_valid(good_stride, expected_s):
    tok = _Token(pace=True)
    tok.pace_stride_ms = good_stride
    resolved = _pace._resolve_stride(tok)  # noqa: SLF001
    assert abs(resolved - expected_s) < 1e-12


@pytest.mark.parametrize("bad_stride", [float("nan"), float("inf"), float("-inf")])
def test_pace_stride_ms_rejects_nonfinite(bad_stride):
    """P4 (Phase C, B1#4): `float('nan')` passed the old guard (`type(x) not in (...) or
    x < 0`) -- `type(nan) is float` and `nan < 0` is `False` (NaN compares False to
    everything), so it silently resolved as 'striding disabled' instead of raising like
    every other malformed value in this family. `inf`/`-inf` share the same non-finite
    class."""
    tok = _Token(pace=True)
    tok.pace_stride_ms = bad_stride
    with pytest.raises(ValueError):
        _pace._resolve_stride(tok)  # noqa: SLF001


# ── P1 (Phase C, B1#1): the ring is thread-local, not device-local ───────────────
#
# `_DeviceSpy` above fixes `torch.cuda.current_device()` to a constant -- none of its rows
# depend on the ambient device actually MOVING. This one does: it needs a `torch.cuda.
# device(...)` that genuinely changes what "ambient" means for its duration (the real
# semantics), and an Event that raises the real `cudaEventRecord` device-mismatch error when
# re-record()ed while a DIFFERENT device is ambient than the one it was first recorded on --
# the exact mechanism a same-thread, sequential, cross-device cook pair can hit.

class _DeviceRegistry:
    def __init__(self, initial=0):
        self.ambient = initial


class _CrossDeviceEvent:
    """Faithful to the real rule: a `torch.cuda.Event` binds to whichever device is ambient
    the FIRST time it is `.record()`ed; recording it again while a DIFFERENT device is
    ambient raises, exactly as real `cudaEventRecord` does (not a PyTorch-added check)."""

    def __init__(self, blocking=False, registry=None):
        self.blocking = blocking
        self._registry = registry
        self._bound_device = None
        self.record_calls = 0
        self.sync_calls = 0

    def record(self):
        self.record_calls += 1
        if self._bound_device is None:
            self._bound_device = self._registry.ambient
        elif self._bound_device != self._registry.ambient:
            raise RuntimeError(
                f"CUDA error: event device {self._bound_device} does not match "
                f"current device {self._registry.ambient}")

    def synchronize(self):
        self.sync_calls += 1

    def query(self):
        return True


class _CrossDeviceSpy:
    """Like `_DeviceSpy`, but `torch.cuda.device(dev)` genuinely moves a tracked ambient
    index for its duration and `current_device()` reads that SAME tracked value (not a fixed
    constant) -- so a test can prove/disprove a real device-mismatch crash across a
    same-thread device switch."""

    def __init__(self):
        self.registry = _DeviceRegistry(initial=0)
        self._real_available = None
        self._real_device_ctx = None
        self._real_event = None
        self._real_current_device = None

    def __enter__(self):
        import torch
        reg = self.registry

        class _Ctx:
            def __init__(self, dev):
                s = str(dev)
                self._idx = int(s.split(":")[1]) if ":" in s else reg.ambient
                self._prev = None

            def __enter__(self):
                self._prev = reg.ambient
                reg.ambient = self._idx
                return self

            def __exit__(self, *exc):
                reg.ambient = self._prev
                return False

        def _fake_event(blocking=False):
            return _CrossDeviceEvent(blocking=blocking, registry=reg)

        self._real_available = torch.cuda.is_available
        self._real_device_ctx = torch.cuda.device
        self._real_event = torch.cuda.Event
        self._real_current_device = torch.cuda.current_device
        torch.cuda.is_available = lambda: True
        torch.cuda.current_device = lambda: reg.ambient
        torch.cuda.device = _Ctx
        torch.cuda.Event = _fake_event
        return self

    def __exit__(self, *exc):
        import torch
        torch.cuda.is_available = self._real_available
        torch.cuda.current_device = self._real_current_device
        torch.cuda.device = self._real_device_ctx
        torch.cuda.Event = self._real_event
        return False


def test_p3_nested_set_cook_grid_restores_pacing_state(r):
    """B1#2: `set_cook_grid`'s own docstring says cooks nest (a codegen invocation inside an
    interpreted fallback, a tiled strip loop) -- `restore_cook_ctx` must give pacing's own
    per-thread bookkeeping back too, not just the four `_cook_ctx` fields it always
    restored. Outer resets with `pace_depth=4` and polls a few times; a NESTED
    `set_cook_grid`/`paced_check`/`restore_cook_ctx` sequence with `pace_depth=1` must not
    permanently overwrite the outer's depth once the inner cook's own restore runs."""
    from TEX_Wrangle.tex_runtime import stdlib_core as _sc
    print("\n--- P3: nested set_cook_grid restores pacing's own state ---")
    with _DeviceSpy():
        outer_tok = _Token(pace=True, pace_depth=4)
        outer_grid_token = _sc.set_cook_grid((1, 8, 8), device="cuda", cancel=outer_tok)
        _pace.paced_check(outer_tok, "cuda")
        _pace.paced_check(outer_tok, "cuda")
        _pace.paced_check(outer_tok, "cuda")
        depth_before_nesting = _pace._state.depth  # noqa: SLF001 (white-box by design)

        inner_tok = _Token(pace=True, pace_depth=1)
        inner_grid_token = _sc.set_cook_grid((1, 4, 4), device="cuda", cancel=inner_tok)
        _pace.paced_check(inner_tok, "cuda")
        _sc.restore_cook_ctx(inner_grid_token)

        depth_after_restore = _pace._state.depth  # noqa: SLF001
        _sc.restore_cook_ctx(outer_grid_token)

    if depth_after_restore == depth_before_nesting == 4:
        r.ok(f"outer's pace_depth (4) survived a nested set_cook_grid/restore_cook_ctx pair "
             f"(read back as {depth_after_restore} right after the inner cook's own restore)")
    else:
        r.fail("P3 nested pacing state", f"outer depth before nesting="
               f"{depth_before_nesting}, after inner's restore={depth_after_restore} "
               f"(expected 4 both times -- the inner cook's pace_depth=1 leaked into the "
               f"outer's bookkeeping)")


def test_p1_ring_is_not_reused_across_a_same_thread_device_switch(r):
    """B1#1: a cook on `cuda:0` that warms the ring, followed sequentially (same thread) by
    a cook on `cuda:1`, must never re-record() a device-0-bound event while device 1 is
    ambient. Pre-P1, the ring's only 'cold slot' test is `ev is None` -- a device switch
    does not clear it, so the second cook's very first poll reuses the first cook's
    device-0-bound event and crashes."""
    print("\n--- P1: the ring must not survive a same-thread device switch ---")
    with _CrossDeviceSpy():
        tok_a = _Token(pace=True, pace_depth=1)
        _pace.reset(tok_a, "cuda:0")
        _pace.paced_check(tok_a, "cuda:0")   # warms a device-0-bound event

        tok_b = _Token(pace=True, pace_depth=1)
        _pace.reset(tok_b, "cuda:1")
        try:
            _pace.paced_check(tok_b, "cuda:1")
        except RuntimeError as e:
            r.fail("P1 cross-device ring reuse",
                   f"a same-thread device switch crashed: {e}")
            return
    r.ok("cuda:1 cook after a cuda:0 cook on the same thread never touched a "
         "device-0-bound event")
