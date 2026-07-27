"""tex_runtime/warm_state.py — CACHE-3: warm-tier persistence.

Generalizes the autotier.json pattern into `warm_state.json`: the warm decisions that die at
process exit and force a relaunch to re-discover everything from scratch —

  * graph-capturability verdicts (`graphed._capturable_memo`: fp -> (capturable, op_count)) —
    the result of the static AST capture-gate walk, a deterministic function of the program AST
    and the arch (both True and False persist), so a relaunch skips re-walking the gate.

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
import time

_FILE = "warm_state.json"
_loaded = False
_path_cache: "str | None" = None
_tag_cache: "str | None" = None
_atexit_registered = False
_last_persist = 0.0
_PERSIST_THROTTLE_SEC = 5.0   # ordinary cooks accumulate warm state without a write per verdict


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


def load() -> None:
    """Merge persisted verdicts into the live graphed/compiled tables, snapshot first and then
    the ENG-13 journal on top. `setdefault` so a verdict already learned this session (fresher)
    always wins over either. Best-effort.

    The journal is replayed even when the snapshot is absent or version-stale: it carries its
    own `version` per record, so a crash before the FIRST snapshot still recovers, which is
    exactly the window a cold launch spends learning."""
    p = _path()
    tag = _tag()
    from . import graphed

    def adopt(fp, val):
        try:
            graphed._capturable_memo.setdefault(fp, (bool(val[0]), int(val[1])))
        except Exception:
            pass

    if p and os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == tag:
                for fp, val in (data.get("capturable") or {}).items():
                    adopt(fp, val)
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
                if isinstance(rec, dict) and rec.get("version") == tag and rec.get("fp"):
                    adopt(rec["fp"], (rec.get("cap"), rec.get("ops", 0)))
        except Exception:
            pass


def _snapshot() -> dict:
    """What we persist — the graph-CAPTURABILITY verdict only, a pure function of the program AST +
    arch (a while-loop is never capturable), so both True and False are stable to persist and skip
    re-walking the AST gate next launch.

    NOT persisted, deliberately: (1) backend probes — `_select_backend` treats a known-True the
    same as an unknown (it only skips a known-False), so persisting positives is inert, and a
    persisted False would harden a one-off runtime failure into a permanent skip; (2) the
    torch.compile blacklist and the runtime CUDA-graph capture blacklist — both mix a stable
    verdict with a transient runtime/OOM crash, which must not become a permanent cross-launch
    demotion (the transient-hygiene reason in DEVELOPMENT.md's rejected-decisions)."""
    from . import graphed
    cap = {fp: [bool(v[0]), int(v[1])] for fp, v in graphed._capturable_memo.items()}
    return {"version": _tag(), "capturable": cap}


def persist(*, force: bool = False) -> None:
    """Write the current warm state atomically and durably, then clear the journal it
    subsumes. Throttled so a burst of verdicts within a few seconds writes once; `force=True`
    (prewarm / shutdown) writes now.

    ORDERING (ENG-13): snapshot FIRST, clear the journal SECOND, and only if the snapshot
    succeeded. A crash in between replays records the snapshot already holds, which is a no-op
    (a capturability verdict is a pure function of the AST + arch, so re-adopting it cannot
    conflict); the reverse order would lose them outright."""
    global _last_persist
    # Throttle FIRST. `_path()` is memoized now, but the ordering is the point: the throttled
    # call is the common one (every `note_update` inside the window) and it must do nothing.
    now = time.time()
    if not force and (now - _last_persist) < _PERSIST_THROTTLE_SEC:
        return
    p = _path()
    if not p:
        return
    from ..tex_recovery import atomic_write_json
    try:
        # Count what this snapshot is about to supersede BEFORE taking it, and afterwards drop
        # only that many records. A verdict learned WHILE the snapshot is being written appends
        # to the journal but is not in the snapshot, so clearing wholesale loses it (reproduced
        # 2/5). `drop_prefix` keeps the tail.
        j = _journal()
        superseded = j.count() if j is not None else 0
        # RE-READ before writing. The snapshot is a whole-table overwrite and the journal is
        # compacted by line count, so with two instances sharing a TEX_CACHE_DIR — the case
        # `atomic_write` and `reattach` both name as a design driver — a persist here would
        # erase verdicts a peer had already made durable (measured: two lost across an
        # `os._exit` with nothing in flight). `load()` adopts by `setdefault`, so the local
        # memo still wins for anything this session learned; the merge only ADDS.
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


def reload() -> int:
    """ENG-13: drop the load latch and re-merge the snapshot + journal, returning how many NEW
    capturability verdicts arrived. The counterpart to `autotier.reload` and
    `ResultCache.reindex_disk`, so `tex_recovery.reattach` never touches `_loaded` or the memo."""
    global _loaded
    from . import graphed
    _loaded = False
    before = len(graphed._capturable_memo)
    load()
    return max(0, len(graphed._capturable_memo) - before)


def _reset_for_test() -> None:
    """Test hook: forget the load latch + persist throttle so a test can drive load/persist
    deterministically."""
    global _loaded, _last_persist, _path_cache, _tag_cache
    _loaded = False
    _last_persist = 0.0
    _path_cache = _tag_cache = None      # a test may have moved TEX_CACHE_DIR
