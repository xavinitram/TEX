"""tex_runtime/warm_state.py — CACHE-3: warm-tier persistence.

Generalizes the autotier.json pattern into `warm_state.json`: the warm decisions that die at
process exit and force a relaunch to re-discover everything from scratch —

  * graph-capturability verdicts (`graphed._capturable_memo`: fp -> (capturable, op_count)) —
    the result of the static AST capture-gate walk, a deterministic function of the program AST
    and the arch (both True and False persist), so a relaunch skips re-walking the gate.

  * COMPILETRY-50 (D1): the fn-calls-compile verdict (`fncalls_compile._memo`: fp -> bool) —
    whether a program whose codegen'd fn calls a non-inlined stdlib builtin was worth handing
    to `torch.compile` for real, replacing `compiled._try_compile`'s old blanket
    `_has_fn_calls` gate. Paid once per fingerprint per this file's whole design; both True
    and False persist for the same reason the capturability verdict does.

CUDA graphs themselves cannot serialize — we persist the DECISION, re-capture off the hot path
(LAT-1b's lesson). Deliberately NOT persisted: backend probes (`compiled._backend_status` — a
persisted positive is inert since `_select_backend` only skips a known-FALSE, and a persisted
False would harden a one-off failure into a permanent skip); the torch.compile blacklist; and the
runtime CUDA-graph capture blacklist. The last three mix a stable verdict with a transient
runtime/OOM crash, which must not become a permanent cross-launch demotion (the transient-failure
hygiene reason recorded in DEVELOPMENT.md's rejected-decisions). The capturability memo already
keeps the expensive capture path away from the programs that genuinely can't capture.

Version-tagged by the CACHE-4 VERDICT epoch × GPU identity (device name + torch): a tier-policy or
codegen change invalidates a stale verdict (CACHE-4's contract), and a warm_state written on a
different GPU is ignored rather than replayed wrong.
"""
import atexit
import json
import os
import threading
import time

_FILE = "warm_state.json"
_loaded = False
_path_cache: "str | None" = None
_tag_cache: "str | None" = None
_atexit_registered = False
_last_persist = 0.0
_PERSIST_THROTTLE_SEC = 5.0   # ordinary cooks accumulate warm state without a write per verdict

# W3 (FIX-WARM, B3#3): `persist()`'s ordering invariant (ENG-13: snapshot the live table
# BEFORE compacting the journal, and only if the snapshot succeeded) was enforced only by
# there having been exactly one caller at a time -- incidentally, never by a lock. Every
# `persist()` call in one process used to run on whatever single thread called `prewarm()`
# or a cook; `prewarm_async` (AUTO-48) is the first path that puts a real, concurrent,
# `force=True` caller into the same process as the cook thread's own
# `note_update()`/`persist(force=False)` calls. This lock makes the invariant enforced, not
# incidental: two concurrent callers now run the throttle-check-through-write body one at a
# time instead of interleaving on the shared `_last_persist`/`_path_cache`/`_tag_cache`
# globals and the snapshot/`drop_prefix` sequence. Held only across `persist()` itself (the
# miss path for an ordinary single-threaded cook, which never contends) -- never around
# `note_update`'s journal append, which stays lock-free and cheap on the hot per-verdict path.
_persist_lock = threading.Lock()


def _path(*, recheck: bool = False):
    """The snapshot path, memoized. `os.makedirs` measured 48.9 µs on this box and the cache
    dir cannot change within a process, so re-deriving it per call put ~100 µs of pure
    repetition on every learned verdict (`note_update` reaches it two to three times).

    `recheck=True` re-runs `makedirs` — the memo caches the RESULT of creating the
    directory, so if something removes it (a cleanup script, a test fixture, a user) every
    later write fails silently and forever, where the old per-call form self-healed. The
    write paths retry through this once before giving up."""
    global _path_cache
    if _path_cache is not None and not recheck:
        return _path_cache
    try:
        from ..tex_cache import get_cache
        d = get_cache()._cache_dir
        os.makedirs(d, exist_ok=True)
        _path_cache = os.path.join(str(d), _FILE)
    except Exception:
        return None
    return _path_cache


def _tag() -> str:
    """Version tag = the CACHE-4 VERDICT epoch × arch identity (device name + torch). Both halves
    are load-bearing: the verdict epoch (which nests the codegen + tier-policy files) means a
    change to graphed.py's capture gate or compiled.py's tiering INVALIDATES a persisted verdict
    (CACHE-4's contract — a tightened gate must not replay a stale `capturable=True`); the arch
    identity means a warm_state from another GPU/torch is ignored (these verdicts don't transfer
    across hardware). A warm_state.json is only adopted when BOTH match.

    Memoized: both halves are fixed for the process, and `_version_tag` calls
    `torch.cuda.get_device_name` every time (7.6 µs)."""
    global _tag_cache
    if _tag_cache is not None:
        return _tag_cache
    try:
        from ..tex_cache import verdict_epoch
        from .xfer import _version_tag
        _tag_cache = f"{verdict_epoch()}_{_version_tag()}"
    except Exception:
        return "0"
    return _tag_cache


def ensure_loaded() -> None:
    """Load persisted warm state into the live tables exactly once (a latch). Cheap to call on
    every warm-tier decision. Also registers a shutdown flush so verdicts learned inside the last
    throttle window survive process exit (the `note_update` throttle would otherwise drop a
    verdict first learned <5s before exit with no later update to trigger a write)."""
    global _loaded, _atexit_registered
    if not _atexit_registered:
        _atexit_registered = True
        try:
            atexit.register(lambda: persist(force=True))   # ENG-11 will call this explicitly too
        except Exception:
            pass
    if _loaded:
        return
    _loaded = True
    load()


def _journal():
    """ENG-13: the append-only sidecar. None when there is no cache dir to write beside."""
    p = _path()
    if not p:
        return None
    from ..tex_recovery import Journal
    return Journal(p)


def _persisted_stores():
    """K6 (v0.50.0 Phase C, R1#2): the (json_key, adopt_fn, snapshot_fn) triple for every
    memo this module's SNAPSHOT persists -- the ONE list `load()`'s snapshot-merge branch
    and `_snapshot()`'s own write side both iterate over now, instead of a parallel
    dict-comprehension / adopt-closure / `data.get(...)` loop hand-copied per store (the
    module's own header used to read as a to-be-continued list of exactly this
    duplication; a future third store is now one entry here).

    The ENG-13 JOURNAL's per-record branches (`load()`, below) stay separate: each
    store's journal record uses a DIFFERENT key name (`fp`/`cap`/`ops` vs `fnfp`/`ok`)
    specifically so an old journal line can never be misread as the wrong store's shape,
    and unifying that would change the on-disk journal FORMAT itself -- a bigger, riskier
    lift than this ask's "reuse... parameterize" scope covers (R1#2 named the
    snapshot/adopt duplication, not the journal schema). The adopt functions here ARE
    reused by the journal branch, so there is exactly one `adopt`/`adopt_fnc` pair
    defined anywhere in this module, not two."""
    from . import graphed
    from . import fncalls_compile as _fnc

    def _adopt_cap(fp, val):
        try:
            graphed._capturable_memo.setdefault(fp, (bool(val[0]), int(val[1])))
        except Exception:
            pass

    def _adopt_fnc(fp, ok):
        try:
            _fnc.adopt_persisted(fp, ok)
        except Exception:
            pass

    def _snap_cap():
        return {fp: [bool(v[0]), int(v[1])] for fp, v in graphed._capturable_memo.items()}

    def _snap_fnc():
        return {fp: bool(ok) for fp, ok in _fnc.snapshot_items().items()}

    return [("capturable", _adopt_cap, _snap_cap),
           ("fncalls_compile", _adopt_fnc, _snap_fnc)]


def load() -> None:
    """Merge persisted verdicts into the live graphed/compiled tables, snapshot first and then
    the ENG-13 journal on top. `setdefault` so a verdict already learned this session (fresher)
    always wins over either. Best-effort.

    The journal is replayed even when the snapshot is absent or version-stale: it carries its
    own `version` per record, so a crash before the FIRST snapshot still recovers, which is
    exactly the window a cold launch spends learning."""
    p = _path()
    tag = _tag()
    stores = _persisted_stores()
    # The journal's two record shapes ("fp"/"cap"/"ops" vs "fnfp"/"ok") correspond
    # positionally to the two stores _persisted_stores() lists, in the same fixed order
    # both this function and `_snapshot()` already rely on.
    _cap_key, adopt_cap, _snap_cap = stores[0]
    _fnc_key, adopt_fnc, _snap_fnc = stores[1]

    if p and os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == tag:
                for json_key, adopt_fn_i, _snap_fn_i in stores:
                    for fp, val in (data.get(json_key) or {}).items():
                        adopt_fn_i(fp, val)
        except Exception:
            pass
    j = _journal()
    if j is not None:
        # Guarded like the snapshot branch above. `replay()` guarantees well-formed JSON,
        # NOT a dict — a line that is validly `42` or `null` raised `AttributeError` out of
        # `load()`, and since `ensure_loaded` latches `_loaded` BEFORE calling here, the
        # failure was permanent for the process and silent (graphed swallows it). That
        # re-introduced, one level up, the "one bad line loses everything" failure
        # `replay()`'s own `errors="replace"` exists to prevent.
        try:
            for rec in j.replay():
                if not (isinstance(rec, dict) and rec.get("version") == tag):
                    continue
                if rec.get("fp"):
                    adopt_cap(rec["fp"], (rec.get("cap"), rec.get("ops", 0)))
                elif rec.get("fnfp"):
                    adopt_fnc(rec["fnfp"], rec.get("ok"))
        except Exception:
            pass


def _snapshot() -> dict:
    """What we persist — every terminal, cross-launch-stable verdict `_persisted_stores()`
    lists (today: graph-CAPTURABILITY, a pure function of the program AST + arch; and
    COMPILETRY-50's fn-calls-compile verdict) — see that function for the single list
    both this and `load()` iterate over.

    NOT persisted, deliberately: (1) backend probes — `_select_backend` treats a known-True the
    same as an unknown (it only skips a known-False), so persisting positives is inert, and a
    persisted False would harden a one-off runtime failure into a permanent skip; (2) the
    torch.compile blacklist and the runtime CUDA-graph capture blacklist — both mix a stable
    verdict with a transient runtime/OOM crash, which must not become a permanent cross-launch
    demotion (the transient-hygiene reason in DEVELOPMENT.md's rejected-decisions)."""
    out = {"version": _tag()}
    for json_key, _adopt_fn, snap_fn in _persisted_stores():
        out[json_key] = snap_fn()
    return out


def persist(*, force: bool = False) -> None:
    """Write the current warm state atomically and durably, then clear the journal it
    subsumes. Throttled so a burst of verdicts within a few seconds writes once; `force=True`
    (prewarm / shutdown) writes now.

    ORDERING (ENG-13): snapshot FIRST, clear the journal SECOND, and only if the snapshot
    succeeded. A crash in between replays records the snapshot already holds, which is a no-op
    (a capturability verdict is a pure function of the AST + arch, so re-adopting it cannot
    conflict); the reverse order would lose them outright.

    W3 (FIX-WARM): a THROTTLED call (the common `note_update` case — every verdict inside the
    window) still returns lock-free, exactly as before, so an ordinary single-threaded cook
    pays nothing new. Only a call that is actually about to write (a genuine `force=True`, or
    a `force=False` call whose throttle window has elapsed) takes `_persist_lock`, re-checks
    the throttle INSIDE the lock (a second caller that lost the race to the first may now
    find the window has just been refreshed and bail out too, rather than duplicate the
    write), and only then runs the snapshot-through-`drop_prefix` body — so two concurrent
    callers (a cook thread's `note_update` and a `prewarm_async` background job's
    `force=True`, the first same-process pairing this invariant ever had to survive)
    serialize instead of interleaving on the shared globals and that sequence."""
    global _last_persist
    # Unlocked fast path FIRST: the throttled call is the overwhelmingly common one and it
    # must cost nothing beyond what it already did (invariant 7 — no lock on this path).
    now = time.time()
    if not force and (now - _last_persist) < _PERSIST_THROTTLE_SEC:
        return
    with _persist_lock:
        # Re-check inside the lock: a caller that lost the race to acquire it may find
        # another thread already refreshed `_last_persist` while it waited, in which case
        # this (non-forced) call is now stale and must not duplicate the write.
        now = time.time()
        if not force and (now - _last_persist) < _PERSIST_THROTTLE_SEC:
            return
        p = _path()
        if not p:
            return
        from ..tex_recovery import atomic_write_json
        try:
            # Count what this snapshot is about to supersede BEFORE taking it, and afterwards
            # drop only that many records. A verdict learned WHILE the snapshot is being
            # written appends to the journal but is not in the snapshot, so clearing wholesale
            # loses it (reproduced 2/5). `drop_prefix` keeps the tail.
            j = _journal()
            superseded = j.count() if j is not None else 0
            # RE-READ before writing. The snapshot is a whole-table overwrite and the journal
            # is compacted by line count, so with two instances sharing a TEX_CACHE_DIR — the
            # case `atomic_write` and `reattach` both name as a design driver — a persist here
            # would erase verdicts a peer had already made durable (measured: two lost across
            # an `os._exit` with nothing in flight). `load()` adopts by `setdefault`, so the
            # local memo still wins for anything this session learned; the merge only ADDS.
            load()
            # The ONE caller that asks for durability: this is the snapshot the journal is
            # compacted against, so losing it to a machine crash would lose the compaction too.
            ok = atomic_write_json(p, _snapshot(), fsync=True)
            if not ok:
                p = _path(recheck=True) or p     # the cache dir may have been removed
                ok = atomic_write_json(p, _snapshot(), fsync=True)
            if ok:
                _last_persist = now
                if j is not None:
                    j.drop_prefix(superseded)
        except Exception:
            pass


def note_update(fp: str | None = None) -> None:
    """A persistable warm decision (a capturability verdict) just changed.

    Two writes with different jobs, and separating them is ENG-13's fix. The JOURNAL append
    makes the verdict durable NOW — a flushed line, microseconds, no disk round-trip — so a
    crash costs at most the cook in flight rather than up to `_PERSIST_THROTTLE_SEC` of
    learning that `atexit` never got to flush. The throttled SNAPSHOT then compacts, so
    ordinary cooks still accumulate warm state without a full rewrite per verdict.

    `fp` names the verdict just learned. Omitted (an older caller), only the snapshot path
    runs — correct, just not crash-tight for that one verdict."""
    if fp is not None:
        j = _journal()
        if j is not None:
            from . import graphed
            val = graphed._capturable_memo.get(fp)
            if val is not None:
                j.append({"version": _tag(), "fp": fp,
                          "cap": bool(val[0]), "ops": int(val[1])})
    persist(force=False)


def note_fncalls_update(fp: str | None = None) -> None:
    """COMPILETRY-50: `note_update`'s counterpart for the fn-calls-compile verdict
    (`fncalls_compile._memo`) -- same journal-then-throttled-snapshot shape, a distinct
    record key (`fnfp`/`ok` instead of `fp`/`cap`/`ops`) so `load()`'s replay can tell the
    two record kinds apart in one journal file."""
    if fp is not None:
        j = _journal()
        if j is not None:
            from . import fncalls_compile as _fnc
            ok = _fnc._memo.get(fp)
            if ok is not None:
                j.append({"version": _tag(), "fnfp": fp, "ok": bool(ok)})
    persist(force=False)


def reload() -> int:
    """ENG-13: drop the load latch and re-merge the snapshot + journal, returning how many NEW
    verdicts arrived across BOTH memos this file persists (capturability + COMPILETRY-50's
    fn-calls-compile verdict). The counterpart to `autotier.reload` and
    `ResultCache.reindex_disk`, so `tex_recovery.reattach` never touches `_loaded` or either
    memo directly."""
    global _loaded
    from . import graphed
    from . import fncalls_compile as _fnc
    _loaded = False
    before = len(graphed._capturable_memo) + len(_fnc._memo)
    load()
    after = len(graphed._capturable_memo) + len(_fnc._memo)
    return max(0, after - before)


def _reset_for_test() -> None:
    """Test hook: forget the load latch + persist throttle so a test can drive load/persist
    deterministically."""
    global _loaded, _last_persist, _path_cache, _tag_cache
    _loaded = False
    _last_persist = 0.0
    _path_cache = _tag_cache = None      # a test may have moved TEX_CACHE_DIR
