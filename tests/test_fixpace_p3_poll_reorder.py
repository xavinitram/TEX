"""FIX-PACE P3 (Phase C, R3 finding 2, MED) — `paced_check`'s per-poll cost in the common
"device keeps up, economize" case grew because `pool["free"]`/`_state.depth` are resolved
UNCONDITIONALLY at the top of every poll, even though neither is read anywhere on the path
that decides to skip (they are only needed by the depth-gated record/wait path, reached
only once the economize check has already failed or striding is off/heavy/large-resolution).

Measured: moving that resolution to AFTER the economize check recovers a meaningful share
of the paced skip path's own added per-poll cost, with no behaviour change (reproduce with
`benchmarks/preempt_drain_bench.py --sweep`).

This is a STRUCTURAL red-first test (AGENTS.md prefers counts/structure to a wall-clock
assertion): a poll that takes the skip path must never touch `pool["free"]` at all. RED at
`3d39da2` (the tuple-unpack `outstanding, free = pool["outstanding"], pool["free"]` sits
before the stride/economize check, so `"free"` is fetched on every poll regardless of
whether the skip path is taken); GREEN once that resolution is deferred.
"""
import pytest

from TEX_Wrangle.tex_runtime import pacing as _pace
from TEX_Wrangle.tex_testkit import DeviceSpy, FakeCudaEvent

_FakeEvent = FakeCudaEvent
_DeviceSpy = DeviceSpy


@pytest.fixture(autouse=True)
def _fresh_pacing_state():
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


class _KeyLoggingPool(dict):
    """A `dict` that logs every key fetched via `__getitem__`, wrapping the SAME
    `outstanding`/`free` list objects a real pool dict holds (a shallow copy preserves
    object identity of the values, only the wrapping container is new) -- so swapping it
    in for `_state.pool` mid-test observes exactly which keys a poll touches without
    disturbing the pool's actual bookkeeping."""
    def __init__(self, *a, log, **kw):
        super().__init__(*a, **kw)
        self._log = log

    def __getitem__(self, key):
        self._log.append(key)
        return super().__getitem__(key)


def test_p3_skip_path_never_touches_the_free_list(r):
    print("\n--- FIX-PACE P3: a skip-path poll must not resolve pool['free'] at all ---")
    with _DeviceSpy():
        tok = _Token(pace=True, pace_depth=2, pace_stride_ms=10_000)
        _pace.reset(tok, "cuda")
        _pace.paced_check(tok, "cuda")   # poll 1: nothing outstanding -> records, no skip

        FakeCudaEvent.DONE = True        # the device has caught up
        log = []
        _pace._state.pool = _KeyLoggingPool(_pace._state.pool, log=log)

        _pace.paced_check(tok, "cuda")   # poll 2: inside the stride window, device done
                                          # -> must take the skip path

    if "free" not in log:
        r.ok(f"the skip-path poll only touched {sorted(set(log))} -- 'free' was never "
             f"fetched")
    else:
        r.fail("P3 skip-path free-list touch",
               f"pool['free'] was fetched during a poll that took the skip path "
               f"(keys touched: {log}) -- free/depth resolution must be deferred until "
               f"after the economize check fails")


def test_p3_fallthrough_path_still_touches_free_and_records(r):
    """Sanity: once the economize check FAILS (device behind), the poll must still reach
    the depth-gated record/wait path -- which does need `free` -- so P3's deferral must
    not accidentally skip that resolution when it is genuinely needed."""
    print("\n--- FIX-PACE P3: a fall-through poll (device behind) still resolves and uses "
          "pool['free'] ---")
    with _DeviceSpy():
        tok = _Token(pace=True, pace_depth=2, pace_stride_ms=10_000)
        _pace.reset(tok, "cuda")
        _pace.paced_check(tok, "cuda")   # poll 1: records E1

        FakeCudaEvent.DONE = False       # the device is BEHIND
        log = []
        _pace._state.pool = _KeyLoggingPool(_pace._state.pool, log=log)

        _pace.paced_check(tok, "cuda")   # poll 2: peek says NOT done -> falls through

    if "free" in log:
        r.ok(f"the fall-through poll correctly resolved pool['free'] (keys touched: "
             f"{sorted(set(log))})")
    else:
        r.fail("P3 fall-through regression",
               f"pool['free'] was never resolved on a poll that should have fallen "
               f"through to the depth-gated record path (keys touched: {log})")
