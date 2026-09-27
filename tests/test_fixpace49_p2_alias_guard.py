"""FIX-PACE49 P2 (R3-efficiency.md #4, R4-altitude.md #1, B3-pacing.md #3, B4-tests-docs.md
#4) -- PACE-49's `_COST_TABLE` is keyed by `id(stmt)` alone (`(call_site_id, device_index,
px_bucket)`), a raw CPython memory address, and OUTLIVES the `Program` it was measured from
(it is module-global, deliberately unbounded-by-Program-lifetime -- "a call site's own cost,
once measured on one cook, informs every later cook of the SAME statement"). `tex_cache`'s
128-entry Program LRU means a freed Program's statement objects can have their addresses
reused by an entirely unrelated, later Program's own statements. When that happens, the new
statement's very first poll would read (and blend into) a stale EWMA left by whatever old
statement happened to die at that address -- silently misclassifying it for
`_COST_WARMUP_SAMPLES` samples' worth of budget decisions.

`pacing_heavy.py`'s own `_HEAVY_STMT_MEMO` already names and guards this EXACT class of
object for the exact same reason ("because a recycled id belongs to a different object[; ]
the `is` check catches [it]") by storing the statement-list object alongside its `id()` and
re-checking `cached[0] is stmts` on every lookup. `_COST_TABLE` had no such check.

The fix: every `_COST_TABLE` entry also carries a strong reference to the object its
`id()`-derived key component actually names (`call_site_id` itself, now threaded through as
an `anchor` argument to `_cost_feed`/`_cost_lookup`/`_pace49_attribute`/`_pace49_cost_gate`,
and as a NEW `call_site_anchor` keyword on `paced_check` -- the interpreter passes the actual
`stmt` object, only when already paced per FIX-PACE49 P1's own gate). A lookup or feed whose
stored anchor does not `is`-match the caller's anchor is treated exactly like a brand-new,
never-seen key: no stale EWMA is blended in or read back.

This file exercises the mechanism directly and deterministically -- CPython's own allocator
timing (confirmed allocator/timing-dependent by B3-pacing.md #3: "did not land a collision on
this run") makes a REAL forced id() collision unreliable to assert on in CI; fabricating the
identical key collision the id-reuse scenario would produce (same numeric `id()`-shaped key,
two distinct Python objects) proves the identical mechanism without depending on allocator
luck -- the same style this test file's own sibling (`test_pace49_cost_budget.py`) already
uses throughout (seeding `_cost_feed` directly rather than waiting for a real device).

RED at base `32f6917`: neither `_cost_feed` nor `_cost_lookup` nor `_pace49_cost_gate` accepts
an `anchor` argument at all (`TypeError`), and `paced_check` has no `call_site_anchor`
keyword.
"""
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
    def __init__(self, pace=True, pace_depth=8, pace_stride_ms=10.0, pace_budget_ms=None):
        self.pace = pace
        self.pace_depth = pace_depth
        self.pace_stride_ms = pace_stride_ms
        if pace_budget_ms is not None:
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
    real = _pace._time.perf_counter
    _pace._time.perf_counter = c
    try:
        yield c
    finally:
        _pace._time.perf_counter = real


# ── unit level: _cost_feed/_cost_lookup's own anchor check ───────────────────────

def test_cost_feed_blends_normally_when_the_anchor_matches(r):
    print("\n--- FIX-PACE49 P2: same key, same anchor -- blends normally (unchanged) ---")
    key = ("slot", 0, 5)
    site = object()
    _pace._cost_feed(key, 10.0, site)
    _pace._cost_feed(key, 10.0, site)
    est = _pace._cost_lookup(key, site)
    if est is not None and est[1] == 2 and abs(est[0] - 10.0) < 1e-9:
        r.ok(f"same-anchor feeds blended normally: {est}")
    else:
        r.fail("FIX-PACE49 P2 same-anchor", f"expected (10.0, 2), got {est}")


def test_cost_lookup_with_a_different_anchor_never_returns_the_stale_entry(r):
    print("\n--- FIX-PACE49 P2: id() collision -- a different anchor reads as COLD, "
          "never the stale EWMA ---")
    key = ("collide-slot", 0, 5)   # the shared id()-derived key component both objects
                                     # collide on, exactly as a freed Program statement's
                                     # address reused by an unrelated later Program would
    stale_stmt = object()           # stands in for the freed statement that measured this
    for _ in range(_pace._COST_WARMUP_SAMPLES):
        _pace._cost_feed(key, 9999.0, stale_stmt)   # warm, huge -- would blow any budget
    est_stale = _pace._cost_lookup(key, stale_stmt)

    new_stmt = object()             # stands in for the unrelated statement that reused
                                     # the freed address
    est_for_new = _pace._cost_lookup(key, new_stmt)

    if (est_stale is not None and est_stale[1] == _pace._COST_WARMUP_SAMPLES
            and est_for_new is None):
        r.ok(f"stale anchor's own entry intact ({est_stale}); the aliased new anchor read "
             f"COLD (None), never the stale 9999.0 EWMA")
    else:
        r.fail("FIX-PACE49 P2 alias lookup", f"est_stale={est_stale} (expected warm), "
               f"est_for_new={est_for_new} (expected None)")


def test_cost_feed_after_an_alias_reseeds_rather_than_blends_into_the_stale_entry(r):
    print("\n--- FIX-PACE49 P2: feeding an aliased key reseeds fresh, never blends with "
          "the stranger's stale EWMA ---")
    key = ("collide-slot-2", 0, 5)
    stale_stmt = object()
    for _ in range(_pace._COST_WARMUP_SAMPLES):
        _pace._cost_feed(key, 9999.0, stale_stmt)

    new_stmt = object()
    _pace._cost_feed(key, 0.01, new_stmt)          # the new, unrelated statement's OWN
                                                     # first-ever real reading
    est_for_new = _pace._cost_lookup(key, new_stmt)
    est_for_stale = _pace._cost_lookup(key, stale_stmt)   # the old anchor's own slot was
                                                            # overwritten -- it no longer
                                                            # reads back at all

    if (est_for_new is not None and est_for_new[1] == 1
            and abs(est_for_new[0] - 0.01) < 1e-9 and est_for_stale is None):
        r.ok(f"new anchor's entry seeded fresh from 0.01ms alone: {est_for_new} (no trace "
             f"of the stranger's 9999.0 EWMA); the old anchor no longer resolves")
    else:
        r.fail("FIX-PACE49 P2 alias feed", f"est_for_new={est_for_new} (expected "
               f"(0.01, 1)), est_for_stale={est_for_stale} (expected None)")


# ── integration level: the gate itself must not inherit a stale/aliased verdict ──

def test_pace49_cost_gate_does_not_inherit_a_stale_aliased_verdict(r):
    print("\n--- FIX-PACE49 P2: _pace49_cost_gate ignores an aliased, unrelated warm "
          "entry sharing its id()-derived key ---")
    with DeviceSpy():
        tok = _Token(pace=True, pace_budget_ms=5.0)
        _pace.reset(tok, "cuda")
        idx, bkt = _pace._state.device_idx, _pace._state.px_bucket
        site_id = 424242   # a fabricated call_site_id -- stands in for a since-freed
                            # Program statement's id(), reused below by an unrelated one
        stmt_old = object()
        for _ in range(_pace._COST_WARMUP_SAMPLES):
            _pace._cost_feed((site_id, idx, bkt), 50.0, stmt_old)   # warm, way over budget
        gate_for_old = _pace._pace49_cost_gate(site_id, stmt_old)

        stmt_new = object()   # an unrelated statement whose id() collided with site_id
        gate_for_new = _pace._pace49_cost_gate(site_id, stmt_new)

    if gate_for_old is False and gate_for_new is True:
        r.ok("the old anchor's own over-budget verdict (False) is unaffected; the aliased "
             "new anchor reads as cold and economizes (True) instead of inheriting it")
    else:
        r.fail("FIX-PACE49 P2 gate alias", f"gate_for_old={gate_for_old} (expected False), "
               f"gate_for_new={gate_for_new} (expected True)")


# ── end-to-end through paced_check: the interpreter's own call shape ─────────────

def test_paced_check_attributes_using_the_anchor_not_only_the_raw_id(r):
    """The real call shape (`call_site_id=id(stmt)`, `call_site_anchor=stmt`): two DISTINCT
    statement objects that happen to share the same `id()`-derived `call_site_id` (simulating
    a post-eviction address reuse) must accumulate two INDEPENDENT cost-table entries, never
    one blended entry."""
    print("\n--- FIX-PACE49 P2: paced_check keys by (call_site_id, anchor) together, not "
          "call_site_id alone ---")
    with DeviceSpy(), _clock_ctx() as clock:
        tok = _Token(pace=True, pace_depth=8, pace_stride_ms=10.0)
        _pace.reset(tok, "cuda")
        shared_id = 99
        stmt_a, stmt_b = object(), object()

        _pace.paced_check(tok, "cuda", call_site_id=shared_id, call_site_anchor=stmt_a)
        clock.advance(20.0)
        _pace.paced_check(tok, "cuda", call_site_id=shared_id, call_site_anchor=stmt_a)
        clock.advance(0.001)
        _pace.paced_check(tok, "cuda", call_site_id=shared_id, call_site_anchor=stmt_a)

        idx, bkt = _pace._state.device_idx, _pace._state.px_bucket
        est_a = _pace._cost_lookup((shared_id, idx, bkt), stmt_a)
        est_b = _pace._cost_lookup((shared_id, idx, bkt), stmt_b)

    if est_a is not None and est_a[1] >= 1 and est_b is None:
        r.ok(f"stmt_a's own entry: {est_a}; stmt_b (a different anchor, same raw id) "
             f"reads COLD (None), not stmt_a's estimate")
    else:
        r.fail("FIX-PACE49 P2 paced_check alias", f"est_a={est_a} (expected warm), "
               f"est_b={est_b} (expected None)")
