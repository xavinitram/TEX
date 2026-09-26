"""FIX-REC (v0.46.2 Phase C) — R4 [LOW-MED]: bound the permanently-unreadable retry (B3#2).

`load_verified` returns `_UNREADABLE` on ANY `OSError` from open()/read(), and every caller
(`_restore`, `_load_from_disk`, `_load_codegen_from_disk`) treats it as "leave it, just a
miss" — correct for the TRANSIENT case RESTORE-462 (`ea8a1d6`) targets, but nothing
distinguished that from a file that is unreadable for a PERMANENT reason (a real disk I/O
error, a permission grant that never comes back): such an entry was retried, one real open()
syscall, on every single restore/load for that key, forever.

The fix: `load_verified` counts CONSECUTIVE `_UNREADABLE` verdicts per path; past
`_UNREADABLE_STREAK_LIMIT` it reports `_UNVERIFIED` instead — the SAME action every caller
already takes for "give up on this file" (delete + its own accounting) — while a streak that
sees even one successful open resets to zero, so the transient case never comes close.
"""
from helpers import *

import builtins

from TEX_Wrangle import tex_recovery as R
from TEX_Wrangle import tex_results


def _frame(res=32):
    return torch.rand(1, res, res, 4)


def test_r4_note_unreadable_escalates_at_the_limit_and_resets(r: SubTestResult):
    print("\n--- FIX-REC R4: _note_unreadable escalates at the limit; a clear resets it ---")
    try:
        path = "some/fake/path.frame"
        R._unreadable_streak.pop(path, None)
        escalated_at = None
        for i in range(1, R._UNREADABLE_STREAK_LIMIT + 2):
            if R._note_unreadable(path):
                escalated_at = i
                break
        ok1 = escalated_at == R._UNREADABLE_STREAK_LIMIT
        ok2 = path not in R._unreadable_streak       # escalation clears the entry
        if not (ok1 and ok2):
            r.fail("FIX-REC R4 escalation",
                   f"escalated_at={escalated_at} (want {R._UNREADABLE_STREAK_LIMIT}) "
                   f"entry_cleared={ok2}")
        else:
            r.ok(f"escalates on exactly the {R._UNREADABLE_STREAK_LIMIT}th consecutive call, "
                 "then clears its own entry")

        # Below the limit, a clear (a successful open) resets the count to zero.
        for _ in range(R._UNREADABLE_STREAK_LIMIT - 1):
            R._note_unreadable(path)
        R._clear_unreadable_streak(path)
        again = [R._note_unreadable(path) for _ in range(R._UNREADABLE_STREAK_LIMIT - 1)]
        R._clear_unreadable_streak(path)
        if any(again):
            r.fail("FIX-REC R4 reset", "escalated before the limit after a clear")
        else:
            r.ok("a clear resets the streak; a fresh run of (limit-1) calls never escalates")
    except Exception as e:
        r.fail("FIX-REC R4 escalation", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def test_r4_permanently_unreadable_frame_is_deleted_after_the_bound(r: SubTestResult):
    print("\n--- FIX-REC R4: a permanently-unreadable .frame is bounded, then deleted ---")
    real_open = builtins.open
    try:
        d = Path(tempfile.mkdtemp())
        c = tex_results.ResultCache(cache_dir=str(d), budget_mb=0)
        f = _frame()
        c.put("k", f)
        c.put("evictor", f)                      # forces "k" out to a signed, valid .frame
        path = c._disk_path("k")
        if not os.path.exists(path):
            r.fail("FIX-REC R4 setup", "setup did not spill 'k'")
            shutil.rmtree(d, ignore_errors=True)
            return
        target = os.path.abspath(str(path))

        def always_fails(file, *a, **kw):
            if os.path.abspath(os.fspath(file)) == target:
                raise PermissionError(13, "in use by another process, unconditionally")
            return real_open(file, *a, **kw)

        builtins.open = always_fails
        try:
            results = []
            for _ in range(R._UNREADABLE_STREAK_LIMIT):
                with c._lock:
                    c._spilled = None            # force a fresh stat + restore attempt each time
                frame, _o, _fen = c._restore("k")
                results.append((frame is None, os.path.exists(path)))
        finally:
            builtins.open = real_open

        below_limit_ok = all(miss and survived for miss, survived in results[:-1])
        last_miss, last_survived = results[-1]
        at_limit_ok = last_miss and not last_survived  # deleted on the LIMIT-th consecutive miss

        ok = below_limit_ok and at_limit_ok
        if ok:
            r.ok(f"the file survives {R._UNREADABLE_STREAK_LIMIT - 1} consecutive unreadable "
                 f"restores, then is deleted on the {R._UNREADABLE_STREAK_LIMIT}th")
        else:
            r.fail("FIX-REC R4 bound",
                   f"below_limit_ok={below_limit_ok} last=(miss={last_miss}, "
                   f"survived={last_survived}) results={results}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        builtins.open = real_open
        r.fail("FIX-REC R4 bound", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def test_r4_transient_failures_below_the_bound_never_delete(r: SubTestResult):
    """The non-regression half: `_UNREADABLE_STREAK_LIMIT - 1` consecutive transient failures,
    each followed by a successful restore, must never delete the file — the bound must not
    regress the case `ea8a1d6` fixed. Runs several such cycles to prove a streak that resets on
    success never creeps toward the limit across cycles."""
    print("\n--- FIX-REC R4: repeated transient (non-consecutive) failures never delete ---")
    real_open = builtins.open
    try:
        d = Path(tempfile.mkdtemp())
        c = tex_results.ResultCache(cache_dir=str(d), budget_mb=0)
        f = _frame()
        c.put("k", f)
        c.put("evictor", f)
        path = c._disk_path("k")
        target = os.path.abspath(str(path))

        for cycle in range(3):
            state = {"n": 0}
            fail_count = R._UNREADABLE_STREAK_LIMIT - 1

            def flaky(file, *a, **kw):
                if (os.path.abspath(os.fspath(file)) == target
                        and state["n"] < fail_count):
                    state["n"] += 1
                    raise PermissionError(13, "transient")
                return real_open(file, *a, **kw)

            builtins.open = flaky
            try:
                for _ in range(fail_count):
                    with c._lock:
                        c._spilled = None
                    frame, _o, _fen = c._restore("k")
                    if frame is not None or not os.path.exists(path):
                        r.fail("FIX-REC R4 transient cycle",
                               f"cycle {cycle}: unexpected hit or deletion mid-cycle")
                        builtins.open = real_open
                        shutil.rmtree(d, ignore_errors=True)
                        return
                # one successful restore resets the streak for the next cycle
                with c._lock:
                    c._spilled = None
                frame, _o, _fen = c._restore("k")
            finally:
                builtins.open = real_open
            ok = frame is not None and float((frame - f).abs().max()) < 1e-6
            if not ok:
                r.fail("FIX-REC R4 transient cycle", f"cycle {cycle}: retry did not serve the frame")
                shutil.rmtree(d, ignore_errors=True)
                return

        r.ok(f"{3} cycles of ({R._UNREADABLE_STREAK_LIMIT - 1} transient misses + 1 hit) "
             "never deleted the file")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        builtins.open = real_open
        r.fail("FIX-REC R4 transient cycle", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
