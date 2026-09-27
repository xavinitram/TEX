"""FIX-GATE G3 (v0.47.0 Phase C, B4#6) -- an unscoped global monkeypatch handed to a
background thread must never outlive an unchecked join.

`tests/test_v0332_audit.py::test_v0332_a3_clear_does_not_orphan_a_frame_spilled_during_its_walk`
reassigns the process-GLOBAL `os.listdir` from a background thread (`clearer`), restoring it
only in THAT thread's own `finally` -- the main thread's only safety net was a bare
`t.join(20)` whose result was never checked. If `c.clear(disk=True)` ever takes longer than
20s on a loaded shared box (a normal condition on a box shared with other work), `os.listdir`
stays patched at the process level for however much longer the straggler thread runs --
into whatever test pytest schedules next in the same worker process, plausibly the
DOC-7d flake this ask's brief named. The fix (already landed in `test_v0332_audit.py`
alongside this test): check `t.is_alive()` after the join and force-restore the global
THERE, in the main thread, regardless of what the background thread does, and fail loudly
(attributing the problem to the test that caused it) instead of silently letting the patch
leak.

This file proves the GENERAL shape (a fast, deterministic proxy -- the real A3 test's join
timeout is a hard-coded 20s, too slow to exercise a genuine hang in a unit test) rather than
re-running the real one: a small stand-in reproduces the exact "unscoped patch + unchecked
join" pattern with short, controllable timeouts, and shows it is bounded now, both for a
thread that hangs and for one that runs on time.
"""
import threading

from helpers import SubTestResult


class _Target:
    """Stands in for the process-global `os` module: a single mutable attribute a
    background thread can monkeypatch, exactly like `clearer()` does to `os.listdir`."""
    def __init__(self, original):
        self.attr = original


def _unfixed_pattern(target, original, patched, hang_seconds, join_timeout):
    """The ORIGINAL A3 shape: the background thread restores the patch only in its own
    `finally`; the caller's `t.join(join_timeout)` result is never checked. Returns
    `target.attr` right after the join -- if the thread is still hung, this is the PATCHED
    value, proving the patch outlived the join with nothing raised."""
    def clearer():
        target.attr = patched
        try:
            threading.Event().wait(hang_seconds)   # stands in for a slow c.clear(disk=True)
        finally:
            target.attr = original

    t = threading.Thread(target=clearer)
    t.start()
    t.join(join_timeout)
    return target.attr, t


def _fixed_pattern(target, original, patched, hang_seconds, join_timeout):
    """The FIX-GATE G3 shape: after the join, force-restore in the MAIN thread regardless
    of the background thread's own state, and report whether it was still alive (the
    caller decides whether that is a hard failure, mirroring `r.fail(...)` in the real
    test)."""
    def clearer():
        target.attr = patched
        try:
            threading.Event().wait(hang_seconds)
        finally:
            target.attr = original

    t = threading.Thread(target=clearer)
    t.start()
    t.join(join_timeout)
    still_alive = t.is_alive()
    target.attr = original   # force it HERE, regardless of the thread's own state
    return target.attr, still_alive, t


def test_fixgate_g3_unfixed_pattern_leaks_the_patch_past_a_hung_join(r: SubTestResult):
    """Demonstrates the DEFECT shape in isolation (never applied to product code -- this is
    a synthetic stand-in, proving the class the fix addresses, not a repro that touches
    `os.listdir` itself)."""
    print("\n--- G3: the unfixed shape leaks a global patch past an unchecked join ---")
    try:
        target = _Target("original")
        attr_after_join, t = _unfixed_pattern(target, "original", "PATCHED",
                                              hang_seconds=2.0, join_timeout=0.1)
        try:
            assert attr_after_join == "PATCHED", (
                f"expected the unfixed pattern to leave the patch in place past an "
                f"unchecked, too-short join, got {attr_after_join!r}")
            assert t.is_alive(), "the stand-in thread should still be hung at this point"
            r.ok("confirmed: the unfixed shape leaves the global patched past a hung join, "
                 "with nothing raised to say so")
        finally:
            t.join(5.0)   # let the real background thread finish and restore, for hygiene
    except Exception as e:
        r.fail("G3 unfixed pattern leaks", str(e))


def test_fixgate_g3_fixed_pattern_never_leaks_past_a_hung_join(r: SubTestResult):
    print("\n--- G3: the fixed shape never leaves the patch in place, hung or not ---")
    try:
        target = _Target("original")
        attr_after_join, still_alive, t = _fixed_pattern(
            target, "original", "PATCHED", hang_seconds=2.0, join_timeout=0.1)
        try:
            assert attr_after_join == "original", (
                f"the fixed pattern must force-restore the global regardless of the "
                f"background thread's state, got {attr_after_join!r}")
            assert still_alive, "the stand-in thread should still be hung at this point"
            r.ok("the fixed shape restores the global immediately and reports the hang "
                 "(a caller can now fail loudly instead of silently leaking the patch)")
        finally:
            t.join(5.0)
    except Exception as e:
        r.fail("G3 fixed pattern never leaks", str(e))


def test_fixgate_g3_fixed_pattern_still_works_on_a_clean_run(r: SubTestResult):
    """Behaviour-preserving: a background thread that finishes well within its join timeout
    (the everyday case) must still leave the global restored and report no hang -- this ask
    closes the leak on the SLOW path, it does not change the fast one."""
    print("\n--- G3: the fixed shape is unchanged on an ordinary, on-time run ---")
    try:
        target = _Target("original")
        attr_after_join, still_alive, t = _fixed_pattern(
            target, "original", "PATCHED", hang_seconds=0.05, join_timeout=5.0)
        assert attr_after_join == "original", attr_after_join
        assert not still_alive, "an on-time thread must not read as still alive"
        r.ok("a clean, on-time background thread still restores the global and reports "
             "no hang")
    except Exception as e:
        r.fail("G3 fixed pattern clean run", str(e))
