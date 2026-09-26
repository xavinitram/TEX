"""FIX-OBSROUTE (v0.46 Phase C) — O1/O2/O3: the cook-observer bug hunt findings.

Companion to `test_observer46_cook_observer.py` (OBSERVER-46's own coverage), which never
drove the seam under warnings-as-errors and never reproduced reentrancy. Each test below
is red against the pre-fix `tex_runtime/cook_observer.py`:

  O1 [B2#1 HIGH]: `_dispatch`'s exception-report path called a bare `warnings.warn(...)`,
     which itself RAISES under warnings-as-errors (`simplefilter("error")`, strict pytest,
     `-W error`) — escaping `enter()` and aborting the cook that triggered a misbehaving
     callback, rather than merely losing the report.
  O2 [B2#2]: `enter()` dispatched BEFORE incrementing the per-thread depth counter, so a
     callback that itself cooks on the SAME thread (a reentrant observer) saw depth still
     at 0 for its own nested `enter()` and dispatched a second notification for what the
     seam's count-once contract considers one cook.
  O3 [R2#1 + R1#2 + B4#3]: `cook_observer.scope(entry)`, a context manager replacing the 6
     hand-rolled `_obs_active = bool(_callbacks); if _obs_active: enter(...)` /
     `finally: if _obs_active: leave()` blocks, with the same one-snapshot-per-outermost-
     call and zero-added-notification-when-unregistered guarantees.
"""
import threading
import time
import warnings

import pytest
import torch

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import cook_observer


def _img():
    return torch.ones(1, 4, 4, 3)


def _register(cb):
    handle = cook_observer.register(cb)

    class _Ctx:
        def __enter__(self):
            return handle

        def __exit__(self, *exc):
            cook_observer.unregister(handle)
            return False

    return _Ctx()


def test_o1_raising_callback_under_warnings_as_errors_does_not_abort_the_cook(r):
    """O1's exact repro: with `simplefilter("error")` armed (the strict-pytest / `-W error`
    shape B2#1 names), a callback that raises must still let the cook complete — the report
    itself must never be what kills it."""
    print("\n--- FIX-OBSROUTE O1: warnings-as-errors must not abort the cook ---")
    saved_warned = cook_observer._warned
    cook_observer._warned = False
    calls = []

    def bad_cb(entry, thread):
        calls.append(entry)
        raise RuntimeError("O1 deliberate test failure")

    try:
        with _register(bad_cb):
            img = _img()
            with warnings.catch_warnings():
                warnings.simplefilter("error")   # every warning is now an exception
                try:
                    res = tex_engine.cook("@OUT = @A * 1.5;", {"A": img}, device_mode="cpu")
                except Exception as e:
                    r.fail("FIX-OBSROUTE O1", f"cook() raised under warnings-as-errors: "
                           f"{type(e).__name__}: {e}")
                    return
            if res is None or "OUT" not in res.outputs:
                r.fail("FIX-OBSROUTE O1", "cook() did not return a normal result")
            elif calls != ["cook"]:
                r.fail("FIX-OBSROUTE O1", f"callback call log was {calls}")
            else:
                r.ok("a raising callback's report did not abort the cook under "
                     "simplefilter('error')")
    finally:
        cook_observer._warned = saved_warned


def test_o2_reentrant_callback_does_not_double_notify(r):
    """O2's exact repro. The callback itself triggers a cook on the SAME thread, from
    INSIDE `_dispatch` — i.e. still on the call stack of the outer `enter()`, which has not
    returned yet. Structurally that nested cook is exactly the "outermost call among the
    six" contract's definition of nested: one user-visible cook, recursing through the
    observer callback. It must therefore share the OUTER notification, not add its own.

    Pre-fix (dispatch-before-increment): the outer `enter()` captures `depth=0` in a local
    variable, THEN calls `_dispatch` (still with the thread-local depth attribute unset).
    The reentrant inner cook's own `enter()` reads that same unset depth as 0 too — the
    outer hasn't written `_local.depth = depth + 1` yet, because it is still paused inside
    the `_dispatch` call that triggered the callback — so the inner call ALSO dispatches,
    producing 2 notifications for what is one user-visible cook. Post-fix
    (increment-before-dispatch): the outer `enter()` writes `_local.depth = 1` BEFORE
    calling `_dispatch`, so the inner call reads depth=1, increments to 2, and correctly
    skips dispatching a second time — exactly 1 notification."""
    print("\n--- FIX-OBSROUTE O2: a reentrant callback must not double-notify ---")
    events = []
    depth_guard = threading.local()
    img = _img()

    def reentrant_cb(entry, thread):
        events.append(entry)
        # Recurse exactly once: the inner cook's own callback call must not recurse again.
        if not getattr(depth_guard, "inside", False):
            depth_guard.inside = True
            try:
                tex_engine.cook("@OUT = @A * 2.0;", {"A": img}, device_mode="cpu")
            finally:
                depth_guard.inside = False

    with _register(reentrant_cb):
        try:
            tex_engine.cook("@OUT = @A * 1.5;", {"A": img}, device_mode="cpu")
        except Exception as e:
            r.fail("FIX-OBSROUTE O2", f"{type(e).__name__}: {e}")
            return
    if events == ["cook"]:
        r.ok("a same-thread reentrant callback's own nested cook shared the outer "
             "notification — exactly one, not two")
    else:
        r.fail("FIX-OBSROUTE O2", f"expected exactly one ['cook'] notification, got {events}")


def test_o3_scope_context_manager_exists_and_behaves_like_enter_leave(r):
    """O3: `cook_observer.scope(entry)` is a context manager with the same collapse-to-
    outermost-call and register/unregister semantics as the hand-rolled blocks it replaces."""
    print("\n--- FIX-OBSROUTE O3: cook_observer.scope() context manager ---")
    if not hasattr(cook_observer, "scope"):
        r.fail("FIX-OBSROUTE O3", "cook_observer.scope does not exist")
        return
    events = []

    def cb(entry, thread):
        events.append(entry)

    with _register(cb):
        events.clear()
        with cook_observer.scope("probe_outer"):
            with cook_observer.scope("probe_inner"):
                pass
        if events == ["probe_outer"]:
            r.ok("nested scope() calls collapse to the outermost notification")
        else:
            r.fail("FIX-OBSROUTE O3 nesting", f"expected ['probe_outer'], got {events}")

        # An exception inside the scope still balances enter/leave (depth returns to 0).
        events.clear()
        try:
            with cook_observer.scope("probe_raises"):
                raise ValueError("boom")
        except ValueError:
            pass
        with cook_observer.scope("probe_after"):
            pass
        if events == ["probe_raises", "probe_after"]:
            r.ok("an exception inside scope() still balances depth (the next scope notifies)")
        else:
            r.fail("FIX-OBSROUTE O3 exception", f"expected two entries, got {events}")

    # Unregistered: scope() must add no notification at all.
    events.clear()
    with cook_observer.scope("probe_unregistered"):
        pass
    if events == []:
        r.ok("scope() with nothing registered notifies nobody")
    else:
        r.fail("FIX-OBSROUTE O3 unregistered", f"expected no events, got {events}")


@pytest.mark.timing
def test_o3_scope_zero_cost_when_unregistered(r):
    """O3's zero-cost requirement, proved by microbenchmark rather than asserted: with
    nothing registered, `with cook_observer.scope(entry): pass` must be negligible next to a
    real cook — this codebase's own convention for "zero cost when disabled" claims
    (see pacing.py / cook_observer.py's own docstrings, and R3's ~35ns per-call measurement
    for the analogous `import torch` question). Generous bound (10 microseconds) so this
    holds across box load; the point is "not a new bottleneck", not a tight pin.

    Marked `timing` (CI-461 audit): a wall-clock deadline claim, same class as the R3
    microbenchmark CI-461 replaced. Unlike R3's tight 2x ratio between two DIFFERENT code
    shapes (a function call vs. a bare bytecode -- exactly what a line tracer taxes
    unevenly), this is a single-shape absolute bound with over an order of magnitude of
    headroom (measured ~700ns/call under a `sys.settrace` no-op tracer standing in for
    coverage.py, against a 10,000ns bound), so it stays green under `--cov`; marked anyway
    for consistency and because the gate's own cheap/full tiers exclude `timing`."""
    print("\n--- FIX-OBSROUTE O3: scope() overhead is negligible when unregistered ---")
    assert not cook_observer._callbacks, "test premise: nothing registered here"
    N = 200_000
    t0 = time.perf_counter()
    for _ in range(N):
        with cook_observer.scope("bench"):
            pass
    elapsed = time.perf_counter() - t0
    per_call_ns = (elapsed / N) * 1e9
    bound_ns = 10_000.0   # 10us/call is enormously generous; this is a "not pathological" gate
    if per_call_ns < bound_ns:
        r.ok(f"scope() unregistered: {per_call_ns:.1f} ns/call over {N} calls "
             f"(bound {bound_ns:.0f} ns)")
    else:
        r.fail("FIX-OBSROUTE O3 zero-cost", f"{per_call_ns:.1f} ns/call over {N} calls "
               f"exceeds the {bound_ns:.0f} ns generous bound")
