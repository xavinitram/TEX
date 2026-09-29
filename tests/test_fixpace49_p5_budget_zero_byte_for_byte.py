"""FIX-PACE49 P5 -- `pace_budget_ms=0`'s own docstring
(`_resolve_budget_ms`, `pacing.py`) calls it "the ONLY way to recover byte-for-byte
pre-PACE-49 economizing", but before this fix only the DECISION half honoured budget<=0
(`_pace49_cost_gate`'s first line): the MEASUREMENT half (`_pace49_attribute`/`_cost_feed`,
including acquiring `_COST_LOCK` and mutating `_COST_TABLE`) ran unconditionally whenever
`call_site_id is not None` and a poll got a fresh tail confirmation, REGARDLESS of
`budget_ms`. Confirmed by running: with `_state.budget_ms=0.0`, two
attributed events still populated a fresh table entry.

The fix: gate every PACE-49 attribution/anchor-bookkeeping branch in `paced_check` on
`_state.budget_ms > 0` too, alongside `call_site_id is not None` -- so `pace_budget_ms=0`
really does perform zero attribution work (no lock, no table mutation, no timing-anchor
bookkeeping), matching the docstring's claim exactly.

RED at base `32f6917` (after P1-P4 land): `_cost_feed` is still called (and `_COST_TABLE`
still grows) even with `pace_budget_ms=0`.
"""
import types
import contextlib

import pytest

from TEX_Wrangle.tex_runtime import pacing as _pace
from TEX_Wrangle.tex_testkit import DeviceSpy, FakeCudaEvent


@pytest.fixture(autouse=True)
def _fresh_pacing_state():
    _pace._state.__dict__.clear()
    _pace._COST_TABLE.clear()
    yield
    _pace._state.__dict__.clear()
    _pace._COST_TABLE.clear()


class _Token:
    def __init__(self, pace=True, pace_depth=8, pace_stride_ms=10.0, pace_budget_ms=0):
        self.pace = pace
        self.pace_depth = pace_depth
        self.pace_stride_ms = pace_stride_ms
        self.pace_budget_ms = pace_budget_ms

    def check(self):
        pass


class _FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@contextlib.contextmanager
def _clock_ctx():
    c = _FakeClock(0.0)
    real = _pace._time
    _pace._time = types.SimpleNamespace(perf_counter=c)
    try:
        yield c
    finally:
        _pace._time = real


def test_pace_budget_ms_zero_performs_zero_attribution_work(r):
    print("\n--- FIX-PACE49 P5: pace_budget_ms=0 feeds the cost table 0 times (no "
          "attribution at all, matching the docstring's byte-for-byte claim) ---")
    feeds = []
    real_feed = _pace._cost_feed
    _pace._cost_feed = lambda *a, **k: (feeds.append((a, k)), real_feed(*a, **k))[1]
    try:
        with DeviceSpy(), _clock_ctx() as clock:
            tok = _Token(pace_budget_ms=0)
            _pace.reset(tok, "cuda")
            stmt = object()
            _pace.paced_check(tok, "cuda", call_site_id="BUDGET0_STMT", call_site_anchor=stmt)
            clock.advance(20.0)
            _pace.paced_check(tok, "cuda", call_site_id="BUDGET0_STMT", call_site_anchor=stmt)
            clock.advance(0.001)
            _pace.paced_check(tok, "cuda", call_site_id="BUDGET0_STMT", call_site_anchor=stmt)
    finally:
        _pace._cost_feed = real_feed

    if feeds == [] and len(_pace._COST_TABLE) == 0:
        r.ok("pace_budget_ms=0 fed the cost table 0 times across a full record+confirm "
             "cycle -- byte-for-byte pre-PACE-49 attribution cost")
    else:
        r.fail("FIX-PACE49 P5 budget-zero attribution",
               f"expected 0 feeds and an empty table, got {len(feeds)} feed(s), table size "
               f"{len(_pace._COST_TABLE)}")


def test_pace_budget_ms_zero_still_economizes_exactly_as_before(r):
    """Regression guard: P5 must not change budget=0's own DECISION behaviour (it already
    always economizes, per `_pace49_cost_gate`'s first line) -- only remove the
    now-provably-unread measurement side effect."""
    print("\n--- FIX-PACE49 P5: pace_budget_ms=0 still economizes on the device-caught-up "
          "skip path (unchanged) ---")
    with DeviceSpy(), _clock_ctx() as clock:
        tok = _Token(pace_budget_ms=0)
        _pace.reset(tok, "cuda")
        stmt = object()
        _pace.paced_check(tok, "cuda", call_site_id="loop", call_site_anchor=stmt)  # E1
        clock.advance(0.001)
        _pace.paced_check(tok, "cuda", call_site_id="loop", call_site_anchor=stmt)  # tail
                                                                                      # done
                                                                                      # -> skip
        constructed = FakeCudaEvent._live

    if constructed == 1:
        r.ok("pace_budget_ms=0 still economized (1 event constructed), unchanged by P5")
    else:
        r.fail("FIX-PACE49 P5 decision regression",
               f"expected 1 constructed event, got {constructed}")
