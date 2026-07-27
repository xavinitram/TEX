"""v0.31 — ENG-13: the engine recovery contract.

    An engine-process crash loses AT MOST THE IN-FLIGHT COOK.

Doc 39 §2 is right that "everything is already disk-backed" — the program cache,
`autotier.json`, `warm_state.json` and the CACHE-2 spill dir all persist, all four through
`tmp + os.replace`. Two things were missing, and they are what these tests pin:

  * A THROTTLE IS NOT A DURABILITY BOUNDARY. `warm_state` coalesced writes over 5 seconds and
    relied on `atexit` to flush the tail. `atexit` does not run on `os._exit`, a SIGKILL, or a
    hard crash — so up to five seconds of learning was lost, which is not "at most the
    in-flight cook". The journal moves durability off the throttle.
  * ORDERING. `persist()` must write the snapshot and THEN clear the journal, never the
    reverse: a crash in between replays already-snapshotted records (idempotent), where the
    reverse order drops them.

THE TEST THAT MATTERS is `test_v031_eng13_kill_the_process`: a real subprocess learns a
verdict and dies via `os._exit`, so no `atexit`, no flush, no clean shutdown of any kind —
and the parent asserts the verdict came back. Everything else here is a unit of that claim.

Shapes (roadmap §10.4): NEVER-SEVER ROWS for the crash matrix, CANARY for the write ordering.
"""
import json
import os
import subprocess
import sys

from helpers import *

from TEX_Wrangle import tex_recovery as R


def _tmpdir():
    return tempfile.mkdtemp(prefix="tex_eng13_")


# ── the durable atomic write ─────────────────────────────────────────────────

def test_v031_eng13_atomic_write(r: SubTestResult):
    print("\n--- v0.31 ENG-13: one durable atomic write for every persisted file ---")
    d = _tmpdir()
    try:
        p = os.path.join(d, "thing.json")
        # Bytes or a streaming callable — the two forms production uses. (A `str` form existed
        # briefly with no caller; it is gone, so the contract is what the docstring argues for.)
        r.ok("atomic_write takes bytes and a streaming callable") \
            if R.atomic_write(p, b"bytes") and open(p, "rb").read() == b"bytes" \
            and R.atomic_write(p, lambda f: f.write(b"streamed")) \
            and open(p, "rb").read() == b"streamed" else \
            r.fail("ENG-13 atomic_write", "round-trip failed")

        r.ok("atomic_write_json round-trips") \
            if R.atomic_write_json(p, {"a": [1, 2]}) and json.load(open(p)) == {"a": [1, 2]} else \
            r.fail("ENG-13 atomic_write", "json round-trip failed")

        # A failed write must not leave a .tmp behind — a stale one confuses the disk-budget
        # scans in both caches, which size the tier by walking the directory.
        bad = os.path.join(d, "no_such_subdir", "x.json")
        ok = not R.atomic_write(bad, b"x")
        leftovers = [f for f in os.listdir(d) if f.endswith(".tmp")]
        r.ok("a failed write returns False and leaves no .tmp") if ok and not leftovers else \
            r.fail("ENG-13 atomic_write", f"returned {not ok}, leftovers={leftovers}")

        # Every persisted engine file must reach it. A hardcoded list of the modules that
        # HAPPEN to have been converted cannot catch the next one, so sweep for the pattern
        # instead: any `os.replace` outside the allowlist is a sixth private copy. (The first
        # draft of this canary listed four modules and silently missed `xfer.py`, which is
        # exactly the failure mode a name-list lint has.)
        strays = lint_sources(r"os\.replace\(", allow={
            "tex_recovery.py",       # the implementation itself
            # Two writers that hold USER-AUTHORED data (saved snippets, published manifests) and
            # keep their own: `tex_tool.write_tool` must RAISE on failure where `atomic_write`
            # returns False, and both predate the consolidation. Recorded as a follow-up rather
            # than converted inside this release.
            "tex_snippets.py", "tex_tool.py",
        })
        r.ok("no engine module hand-rolls tmp + os.replace outside tex_recovery") \
            if not strays else \
            r.fail("ENG-13 reuse", f"private atomic-write copies still in {strays}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_v031_eng13_journal_survives_a_torn_tail(r: SubTestResult):
    """A crash mid-append leaves half a line. That must cost exactly that one record."""
    print("\n--- v0.31 ENG-13: a torn journal tail costs one record, not the file ---")
    d = _tmpdir()
    try:
        j = R.Journal(os.path.join(d, "snap.json"))
        for i in range(3):
            j.append({"i": i})
        with open(j.path, "a", encoding="utf-8") as f:
            f.write('{"i": 3, "cap": tru')          # the crash
        got = [rec["i"] for rec in j.replay()]
        r.ok(f"replay() recovers the intact records and skips the torn one: {got}") \
            if got == [0, 1, 2] else r.fail("ENG-13 journal", f"replayed {got}")

        # ONE BAD BYTE, not just a torn tail. A partial multi-byte write — exactly what the crash
        # this exists to survive produces — raised UnicodeDecodeError straight out of `replay()`
        # and discarded EVERY good record, turning a one-record loss into total loss (12/12).
        with open(j.path, "ab") as f:
            f.write(b"\xff\xfe not utf-8\n")
        j.append({"i": 9})
        got = [rec["i"] for rec in j.replay()]
        r.ok(f"a non-UTF-8 byte costs its own line only: {got}") if got == [0, 1, 2, 9] else \
            r.fail("ENG-13 journal", f"a bad byte cost {4 - len(got)} good record(s): {got}")

        # `drop_prefix` is what compaction needs and `clear()` is not — see the ordering test.
        j.drop_prefix(2)
        r.ok(f"drop_prefix(2) keeps the tail: {[x['i'] for x in j.replay()]}") \
            if [x["i"] for x in j.replay()] == [2, 9] else \
            r.fail("ENG-13 journal", f"drop_prefix left {j.replay()}")

        j.clear()
        r.ok("clear() removes the journal and replay() is then empty") \
            if not j.exists() and j.replay() == [] else \
            r.fail("ENG-13 journal", "clear() left something behind")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── warm state: journal now, compact later ───────────────────────────────────

def test_v031_eng13_warm_state_journals_each_verdict(r: SubTestResult):
    print("\n--- v0.31 ENG-13: a learned verdict is durable before the snapshot ---")
    from TEX_Wrangle.tex_runtime import warm_state as W, graphed as G
    with cold_engine_state() as cold:
        d = cold.dir

        # Verdict #1. `_last_persist` is 0, so this one is not throttled and compacts
        # immediately — snapshot written, journal cleared.
        G._capturable_memo["eng13-first"] = (True, 42)
        W.note_update("eng13-first")
        snap_path = W._path()
        j = W._journal()
        r.ok("the first verdict compacts straight into the snapshot") \
            if os.path.exists(snap_path) and not j.exists() else \
            r.fail("ENG-13 compaction",
                   f"snapshot={os.path.exists(snap_path)} journal_left={j.exists()}")

        # Verdict #2, learned INSIDE the throttle window. This is the whole risk: before
        # ENG-13 nothing at all was written here, and a crash in the next five seconds lost it.
        G._capturable_memo["eng13-windowed"] = (False, 7)
        W.note_update("eng13-windowed")
        recs = j.replay()
        r.ok(f"a verdict learned inside the throttle window is journalled ({len(recs)} record)") \
            if any(x.get("fp") == "eng13-windowed" for x in recs) else \
            r.fail("ENG-13 warm journal", f"journal held {recs}")
        snap = json.load(open(snap_path, encoding="utf-8"))
        r.ok("…and is NOT yet in the snapshot (so recovery below is the journal's doing)") \
            if "eng13-windowed" not in (snap.get("capturable") or {}) else \
            r.fail("ENG-13 warm journal", "the throttle did not hold — the test proves nothing")

        # A fresh load recovers BOTH: one from the snapshot, one from the journal.
        G._capturable_memo.clear()
        W._reset_for_test()
        W.load()
        got = (G._capturable_memo.get("eng13-first"), G._capturable_memo.get("eng13-windowed"))
        r.ok("load() merges the snapshot and the journal") \
            if got == ((True, 42), (False, 7)) else \
            r.fail("ENG-13 load", f"recovered {got}")

        # ORDERING CANARY: if the snapshot fails, the journal must SURVIVE. Otherwise a full
        # disk turns a recoverable crash into a silent total loss of warm state.
        W.persist(force=True)                   # re-arm the throttle (the reset above zeroed it)
        G._capturable_memo["eng13-third"] = (True, 3)
        W.note_update("eng13-third")            # still inside the window: journal only
        orig = R.atomic_write_json
        try:
            R.atomic_write_json = lambda *a, **k: False        # simulate a failed snapshot
            W.persist(force=True)
        finally:
            R.atomic_write_json = orig
        r.ok("a failed snapshot leaves the journal intact (snapshot-then-clear)") \
            if any(x.get("fp") == "eng13-third" for x in W._journal().replay()) else \
            r.fail("ENG-13 ordering", "the journal was cleared without a successful snapshot")

        # THE WRITE-WINDOW LOSS. A verdict learned WHILE the snapshot is being written appends to
        # the journal but is not in the snapshot, so clearing wholesale dropped it (2/5). The
        # compactor now counts what it supersedes first and drops only that many records.
        W.persist(force=True)                    # re-arm the throttle, clean slate
        G._capturable_memo["eng13-during"] = (True, 11)
        W.note_update("eng13-during")            # journalled, not yet snapshotted
        j2 = W._journal()
        real_write = R.atomic_write_json

        def slow_write(*a, **k):
            # Stand in for the OS: another verdict is learned mid-write.
            G._capturable_memo["eng13-raced"] = (False, 12)
            j2.append({"version": W._tag(), "fp": "eng13-raced", "cap": False, "ops": 12})
            return real_write(*a, **k)

        try:
            R.atomic_write_json = slow_write
            W.persist(force=True)
        finally:
            R.atomic_write_json = real_write
        survived = [x.get("fp") for x in j2.replay()]
        r.ok("a verdict learned DURING the snapshot survives compaction") \
            if "eng13-raced" in survived else \
            r.fail("ENG-13 write window",
                   f"the mid-write verdict was compacted away; journal holds {survived}")



# ── the kill-the-process test ────────────────────────────────────────────────

_CRASH_CHILD = r'''
import os, sys
sys.path.insert(0, sys.argv[1])                       # .../custom_nodes
os.environ["TEX_CACHE_DIR"] = sys.argv[2]
import torch
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import warm_state as W, graphed as G

W._reset_for_test()
W.ensure_loaded()
# A real cook, so the process has genuinely done engine work before it dies.
A = torch.rand(1, 32, 32, 4)
tex_engine.cook("@OUT = vec4(@A.rgb * 1.25, 1.0);", {"A": A}, device_mode="cpu")

# The first verdict is unthrottled and compacts into the snapshot straight away. That is NOT
# the interesting one — a snapshot would have saved it before ENG-13 too.
G._capturable_memo["crash-snapshotted"] = (True, 1)
W.note_update("crash-snapshotted")

# THIS is the one. Learned inside the 5-second throttle window, so the snapshot is not
# rewritten; pre-ENG-13 nothing at all reached the disk for it, and `atexit` was the only
# thing that would ever have flushed it. Then DIE: `os._exit` skips atexit, skips the
# interpreter shutdown, skips every flush the engine would normally get.
G._capturable_memo["crash-survivor"] = (True, 123)
W.note_update("crash-survivor")
os._exit(9)
'''


def test_v031_eng13_kill_the_process(r: SubTestResult):
    """The contract's own test. A child learns a verdict and is killed with `os._exit` —
    no atexit, no flush, no clean shutdown — and the parent must find the verdict on disk.

    Before ENG-13 this failed by construction: `note_update` only scheduled a throttled
    snapshot, and the 5-second window had not elapsed, so nothing had been written at all."""
    print("\n--- v0.31 ENG-13: an os._exit crash loses at most the in-flight cook ---")
    from TEX_Wrangle.tex_runtime import warm_state as W, graphed as G
    custom_nodes = str(Path(__file__).resolve().parents[2])
    with cold_engine_state() as cold:
        d = cold.dir
        try:
            proc = subprocess.run([sys.executable, "-c", _CRASH_CHILD, custom_nodes, d],
                                  capture_output=True, timeout=180)
        except subprocess.TimeoutExpired:
            r.fail("ENG-13 crash", "the child never finished")
            return
        if proc.returncode != 9:
            r.fail("ENG-13 crash", f"child exited {proc.returncode}: "
                                   f"{proc.stderr.decode(errors='replace')[-400:]}")
            return
        r.ok("the child died via os._exit(9) — no atexit, no flush")

        # The snapshot must NOT hold the second verdict: it was learned inside the throttle
        # window. This is what makes the recovery below attributable to the JOURNAL rather
        # than to a snapshot that happened to fire — without it the test proves nothing.
        snap_path = os.path.join(d, "warm_state.json")
        snapped = json.load(open(snap_path, encoding="utf-8")).get("capturable", {}) \
            if os.path.exists(snap_path) else {}
        r.ok("the snapshot holds the first verdict but not the windowed one") \
            if "crash-survivor" not in snapped else \
            r.fail("ENG-13 crash", "the snapshot already had it — the throttle did not hold")

        # The child wrote into `d`, which the fixture minted and already points
        # TEX_CACHE_DIR at, so `warm_state._path()` here resolves to the same directory.
        W.load()
        r.ok("the parent recovers the crashed child's verdict from the journal") \
            if G._capturable_memo.get("crash-survivor") == (True, 123) else \
            r.fail("ENG-13 crash", f"lost the verdict; memo={dict(G._capturable_memo)}")



# ── reattach ─────────────────────────────────────────────────────────────────

def test_v031_eng13_reattach(r: SubTestResult):
    """`session.reattach()`: recover in a LIVE process, without a restart."""
    print("\n--- v0.31 ENG-13: EngineSession.reattach() ---")
    from TEX_Wrangle import tex_results, tex_session
    from TEX_Wrangle.tex_runtime import warm_state as W, graphed as G
    with cold_engine_state() as cold:
        d = cold.dir

        # Another process (simulated) left a verdict and a spilled frame behind.
        G._capturable_memo["reattach-probe"] = (True, 5)
        W.note_update("reattach-probe")
        W.persist(force=True)

        cache = tex_results.ResultCache(budget_mb=0, cache_dir=os.path.join(d, "results"))
        frame = make_img(1, 16, 16, 4, seed=31)
        cache.put("k-reattach", frame, canvas={"shape": [1, 16, 16, 4]})
        cache.put("k-filler", make_img(1, 16, 16, 4, seed=32))   # forces the first to spill

        # Now "reattach" from a cold view of the same state.
        G._capturable_memo.clear()
        W._reset_for_test()
        fresh = tex_results.ResultCache(budget_mb=64, cache_dir=os.path.join(d, "results"))
        report = tex_session.default_session().reattach(result_cache=fresh)

        r.ok(f"reattach() reports what it restored: {report}") \
            if isinstance(report, dict) and not report["errors"] else \
            r.fail("ENG-13 reattach", f"{report}")
        r.ok("the capturability verdict is back") \
            if G._capturable_memo.get("reattach-probe") == (True, 5) else \
            r.fail("ENG-13 reattach", f"memo={dict(G._capturable_memo)}")
        r.ok(f"the disk tier is re-indexed ({report['frames']} frame(s), "
             f"{report['frame_bytes']} bytes)") \
            if report["frames"] >= 1 and report["frame_bytes"] > 0 else \
            r.fail("ENG-13 reattach", f"disk tier not indexed: {report}")

        got = fresh.get("k-reattach")
        r.ok("a frame spilled before the crash is still served after reattach") \
            if got is not None and torch.equal(got.cpu(), frame) else \
            r.fail("ENG-13 reattach", "the spilled frame did not come back intact")

