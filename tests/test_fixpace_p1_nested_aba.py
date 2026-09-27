"""FIX-PACE P1 (Phase C, B1 finding 1, HIGH) — the R1 peek-cache
(`_state.last_confirmed_done`) is an ABA hazard across a same-device NESTED cook.

`save_state()` shallow-copies `_state.__dict__`: `pool` is the SAME dict object, not a
copy. A same-device nested cook's own `reset()` hands the outer's then-outstanding event
back to `free`; the nested cook's own first poll can pop that very event and re-`record()`
it onto a NEW point in the stream. When the outer is restored, it gets back BOTH the
mutated pool (correct: the same object) AND its own stale `last_confirmed_done` -- which
still identity-matches the very event the nested cook just re-armed. The outer's next
economizing poll then short-circuits `tail is _state.last_confirmed_done` to True WITHOUT
ever calling `tail.query()` -- trusting a confirmation made before the nesting, for a
recording made after it.

RED at `3d39da2`: `restore_state` puts `last_confirmed_done` back verbatim, so the poll
after a same-device nested cook can skip recording even though the device has genuinely
NOT reached the tail's new point. GREEN once `restore_state` never restores a peek-cache
confirmation across a nested cook's own pool mutation.
"""
import pytest

from TEX_Wrangle.tex_runtime import pacing as _pace
from TEX_Wrangle.tex_testkit import DeviceSpy, FakeCudaEvent

_FakeEvent = FakeCudaEvent
_DeviceSpy = DeviceSpy


@pytest.fixture(autouse=True)
def _fresh_pacing_state():
    """Same isolation `test_pace462_bounded_lookahead.py` uses: `_state` is thread-local
    and persists across cooks on this thread by design, so tests must clear it themselves."""
    _pace._state.__dict__.clear()
    yield
    _pace._state.__dict__.clear()


class _Token:
    def __init__(self, pace=True, pace_depth=None, pace_stride_ms=None):
        self.pace = pace
        if pace_depth is not None:
            self.pace_depth = pace_depth
        if pace_stride_ms is not None:
            self.pace_stride_ms = pace_stride_ms
        self.checks = 0

    def check(self):
        self.checks += 1


def _outstanding_len():
    return len(_pace._state.pool["outstanding"])


def test_p1_nested_same_device_cook_never_lets_outer_trust_a_stale_confirmation(r):
    print("\n--- FIX-PACE P1: a same-device nested cook must not leave the outer cook "
          "trusting a pre-nesting peek confirmation ---")
    with _DeviceSpy():
        # A large stride so the outer's second poll lands INSIDE the stride window (the
        # economizing branch this bug lives in), and depth=2 so the outer's first record
        # never waits.
        outer = _Token(pace=True, pace_depth=2, pace_stride_ms=10_000)
        _pace.reset(outer, "cuda")

        # Poll 1: nothing outstanding yet -> records E1 unconditionally.
        _pace.paced_check(outer, "cuda")
        assert _outstanding_len() == 1, "setup: outer's first poll must record one event"

        # Poll 2: inside the stride window, device (mocked) reports done -> the peek
        # confirms E1 and caches it as `last_confirmed_done`.
        FakeCudaEvent.DONE = True
        _pace.paced_check(outer, "cuda")
        assert _pace._state.last_confirmed_done is not None, (
            "setup: the outer's second poll must have cached a confirmed-done tail")

        # The outer nests: save its state (a snapshot of the CURRENT last_confirmed_done,
        # and a REFERENCE to the same pool dict), then a same-device inner cook resets.
        snapshot = _pace.save_state()
        inner = _Token(pace=True, pace_depth=2, pace_stride_ms=0)
        _pace.reset(inner, "cuda")  # hands the outer's outstanding event back to `free`

        # The inner cook's own first poll pops that event (LIFO) and RE-records it onto a
        # new point in the stream.
        _pace.paced_check(inner, "cuda")

        # The inner cook ends; the outer is restored.
        _pace.restore_state(snapshot)

        # The device has NOT reached the tail's new (inner-recorded) point.
        FakeCudaEvent.DONE = False
        before = _outstanding_len()
        _pace.paced_check(outer, "cuda")
        after = _outstanding_len()

    # A real `query()` on the tail would read False (DONE=False) and fall through to the
    # depth-gated record/wait path, which RECORDS a new event (outstanding grows). The bug
    # short-circuits on stale identity and skips entirely (outstanding unchanged).
    if after > before:
        r.ok(f"outer's post-nesting poll recorded (outstanding {before} -> {after}) -- "
             f"the stale peek cache was not trusted")
    else:
        r.fail("P1 nested ABA",
               f"outer's post-nesting poll SKIPPED (outstanding stayed at {before}) even "
               f"though the mocked device reports NOT done -- a stale cross-cook identity "
               f"cache was trusted instead of calling query()")


def test_p1_restore_without_nesting_is_unaffected(r):
    """Sanity: a save/restore pair with NO nested activity in between must still let a
    genuinely-confirmed tail economize afterward -- P1's fix must cost a query() call only
    when a nested cook could plausibly have mutated the shared pool, not defeat the
    economization mechanism outright for every restore."""
    print("\n--- FIX-PACE P1: save/restore with no nested activity still economizes ---")
    with _DeviceSpy():
        tok = _Token(pace=True, pace_depth=2, pace_stride_ms=10_000)
        _pace.reset(tok, "cuda")
        _pace.paced_check(tok, "cuda")   # records E1

        FakeCudaEvent.DONE = True
        _pace.paced_check(tok, "cuda")   # confirms E1 via peek

        snapshot = _pace.save_state()
        _pace.restore_state(snapshot)    # no nested cook ran in between

        query_calls_before = _FakeEvent._live  # no NEW event should be built by a skip
        outstanding_before = _outstanding_len()
        _pace.paced_check(tok, "cuda")   # device still reports done
        outstanding_after = _outstanding_len()

    if outstanding_after == outstanding_before:
        r.ok("a restore with no intervening nested cook still economized correctly "
             "(query() answers True, poll skips, no new event built)")
    else:
        r.fail("P1 restore regression",
               f"outstanding grew ({outstanding_before} -> {outstanding_after}) on a "
               f"restore with no nested activity -- P1's fix must not force a record on "
               f"every restore, only refuse to TRUST a stale identity match")
