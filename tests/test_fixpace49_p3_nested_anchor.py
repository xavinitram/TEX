"""FIX-PACE49 P3 -- the timing anchor (`pool["timed_prev"]`/
`pool["timed_site"]`/`pool["timed_anchor"]`) crosses a same-device NESTED cook boundary
uncorrected, the v0.47 peek-cache ABA class (`test_fixpace_p1_nested_aba.py`) reopened for
PACE-49's own new anchor.

`save_state()`/`restore_state()` snapshot/restore every SCALAR `_state` field, but the
timing anchor lives on the (shared, per-device) `pool` dict, not in `_state.__dict__` --
`reset()` already knows this is shared, mutable state a nested cook can touch (it clears
`last_confirmed_done` on restore for the identical reason), but did not extend that same
care to the timing anchor. A nested cook's own `reset()` clears `timed_prev`/`timed_site`/
`timed_anchor` and repopulates them from ITS OWN call sites; nothing restores the outer's
pre-nesting anchor when the nested cook ends, so the OUTER's next real interval gets credited
to whatever call site the INNER cook happened to leave behind.

RED at base `32f6917` (after FIX-PACE49 P1/P2 land): the outer's post-nesting real device
interval is folded into the INNER call site's cost-table entry instead of being discarded (or
credited to the outer) -- matching this fix's own confirmed-by-running repro exactly
("`OUTER_STMT_A`'s own entry never received anything" / the inner's entry absorbed the
outer's real interval).
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
    def __init__(self, pace=True, pace_depth=8, pace_stride_ms=10.0):
        self.pace = pace
        self.pace_depth = pace_depth
        self.pace_stride_ms = pace_stride_ms

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


def test_nested_same_device_cook_never_lets_the_outer_credit_the_inners_call_site(r):
    print("\n--- FIX-PACE49 P3: a same-device nested cook must not leave the outer's real "
          "interval credited to the inner's own call site ---")
    with DeviceSpy(), _clock_ctx() as clock:
        outer = _Token()
        _pace.reset(outer, "cuda")
        outer_stmt = object()
        idx, bkt = _pace._state.device_idx, _pace._state.px_bucket

        # Outer poll 1: records E1, seeds the anchor (timed_prev=E1, timed_site=OUTER).
        _pace.paced_check(outer, "cuda", call_site_id="OUTER", call_site_anchor=outer_stmt)

        # The outer nests: save its state, then a same-device inner cook resets (clearing
        # the SHARED pool's timing anchor) and runs its own correct, self-contained
        # attribution cycle.
        snapshot = _pace.save_state()
        inner = _Token()
        _pace.reset(inner, "cuda")
        inner_stmt = object()

        _pace.paced_check(inner, "cuda", call_site_id="INNER", call_site_anchor=inner_stmt)
        clock.advance(20.0)  # past the stride window -> records E_inner2 for real
        _pace.paced_check(inner, "cuda", call_site_id="INNER", call_site_anchor=inner_stmt)
        clock.advance(0.001)
        # Fresh confirm of E_inner2 -> attributes ONE real interval to INNER alone.
        _pace.paced_check(inner, "cuda", call_site_id="INNER", call_site_anchor=inner_stmt)
        est_inner_after_own_cycle = _pace._cost_lookup(("INNER", idx, bkt), inner_stmt)

        # The inner cook ends; the outer is restored.
        _pace.restore_state(snapshot)

        # The outer resumes and runs its own full record+confirm cycle.
        clock.advance(20.0)
        _pace.paced_check(outer, "cuda", call_site_id="OUTER", call_site_anchor=outer_stmt)
        clock.advance(0.001)
        _pace.paced_check(outer, "cuda", call_site_id="OUTER", call_site_anchor=outer_stmt)

        est_inner_final = _pace._cost_lookup(("INNER", idx, bkt), inner_stmt)

    if (est_inner_after_own_cycle is not None and est_inner_after_own_cycle[1] == 1
            and est_inner_final is not None and est_inner_final[1] == 1):
        r.ok(f"INNER's own entry stayed at 1 sample ({est_inner_final}) -- the outer's "
             f"post-nesting real interval was not folded into it")
    else:
        r.fail("FIX-PACE49 P3 nested anchor",
               f"INNER after its own cycle: {est_inner_after_own_cycle} (expected 1 sample); "
               f"INNER after the outer resumed and recorded again: {est_inner_final} "
               f"(expected STILL 1 sample -- if this grew to 2, the outer's own real "
               f"interval was misattributed to the inner's call site)")


def test_restore_without_nesting_still_attributes_correctly(r):
    """Sanity: a save/restore pair with no nested activity in between must still let the
    SAME call site's own attribution continue normally afterward -- P3's fix (clearing the
    anchor unconditionally on every restore, exactly like `last_confirmed_done` already
    does) costs at most ONE interval of lost attribution right after a restore (the first
    post-restore record has no `prev` to diff against, since the anchor was just cleared --
    the identical "extra query() call" cost `restore_state`'s own P1 fix already accepts for
    `last_confirmed_done`), never a wrong credit and never a permanently broken mechanism:
    the NEXT interval after that attributes normally."""
    print("\n--- FIX-PACE49 P3: save/restore with no nesting still attributes normally "
          "(after the one lost interval the clear itself costs) ---")
    with DeviceSpy(), _clock_ctx() as clock:
        tok = _Token()
        _pace.reset(tok, "cuda")
        stmt = object()
        idx, bkt = _pace._state.device_idx, _pace._state.px_bucket

        _pace.paced_check(tok, "cuda", call_site_id="A", call_site_anchor=stmt)  # records E1

        snapshot = _pace.save_state()
        _pace.restore_state(snapshot)  # no nested cook ran in between -- anchor cleared

        clock.advance(20.0)
        _pace.paced_check(tok, "cuda", call_site_id="A", call_site_anchor=stmt)  # E2: seeds
                                                                                   # a fresh
                                                                                   # anchor,
                                                                                   # nothing
                                                                                   # to
                                                                                   # attribute
                                                                                   # yet
        clock.advance(20.0)
        _pace.paced_check(tok, "cuda", call_site_id="A", call_site_anchor=stmt)  # E3, real
        clock.advance(0.001)
        _pace.paced_check(tok, "cuda", call_site_id="A", call_site_anchor=stmt)  # confirms
                                                                                   # E3 ->
                                                                                   # attributes
                                                                                   # (E2, E3)

        est = _pace._cost_lookup(("A", idx, bkt), stmt)

    if est is not None and est[1] == 1:
        r.ok(f"a restore with no intervening nested cook still attributed normally once "
             f"one full interval had formed: {est}")
    else:
        r.fail("FIX-PACE49 P3 restore regression",
               f"expected 1 sample once a full post-restore interval had formed, got {est}")
