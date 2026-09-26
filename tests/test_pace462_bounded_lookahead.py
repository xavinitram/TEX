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
import threading

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
    def __init__(self, pace=True, pace_depth=None):
        self.pace = pace
        if pace_depth is not None:
            self.pace_depth = pace_depth
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


class _LiveToken:
    def __init__(self, pace=True, pace_depth=None):
        self.pace = pace
        if pace_depth is not None:
            self.pace_depth = pace_depth

    def check(self):
        pass


class _ThreadTripToken:
    def __init__(self, delay_s, pace_depth):
        self.pace = True
        self.pace_depth = pace_depth
        self._tripped = threading.Event()
        self._timer = threading.Timer(delay_s, self._tripped.set)
        self._timer.daemon = True
        self._timer.start()

    def check(self):
        if self._tripped.is_set():
            raise CookCancelled("PACE-462 drained-bound repro: token tripped")


def _measure_full_runtime():
    def once(seed):
        t0 = time.perf_counter()
        tex_engine.cook(_PROGRAM, _bindings(seed), device_mode="cuda")
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    once(940)  # discard cold leg
    return min(once(941), once(942))


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
    delay = full * 0.25
    tok = _ThreadTripToken(delay, pace_depth=depth)

    t_trip_seen = None
    try:
        tex_engine.cook(_PROGRAM, _bindings(950), device_mode="cuda", cancel=tok)
        r.fail("PACE-462 drained bound", "cook completed without raising")
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
