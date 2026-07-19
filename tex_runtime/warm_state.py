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
_atexit_registered = False
_last_persist = 0.0
_PERSIST_THROTTLE_SEC = 5.0   # ordinary cooks accumulate warm state without a write per verdict


def _path():
    try:
        from ..tex_cache import get_cache
        d = get_cache()._cache_dir
        os.makedirs(d, exist_ok=True)
        return os.path.join(str(d), _FILE)
    except Exception:
        return None


def _tag() -> str:
    """Version tag = the CACHE-4 VERDICT epoch × arch identity (device name + torch). Both halves
    are load-bearing: the verdict epoch (which nests the codegen + tier-policy files) means a
    change to graphed.py's capture gate or compiled.py's tiering INVALIDATES a persisted verdict
    (CACHE-4's contract — a tightened gate must not replay a stale `capturable=True`); the arch
    identity means a warm_state from another GPU/torch is ignored (these verdicts don't transfer
    across hardware). A warm_state.json is only adopted when BOTH match."""
    try:
        from ..tex_cache import verdict_epoch
        from .xfer import _version_tag
        return f"{verdict_epoch()}_{_version_tag()}"
    except Exception:
        return "0"


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


def load() -> None:
    """Merge persisted verdicts into the live graphed/compiled tables. `setdefault` so a verdict
    already learned this session (fresher) always wins over the persisted one. Best-effort."""
    p = _path()
    if not p or not os.path.exists(p):
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != _tag():
            return
        from . import graphed
        for fp, val in (data.get("capturable") or {}).items():
            try:
                graphed._capturable_memo.setdefault(fp, (bool(val[0]), int(val[1])))
            except Exception:
                pass
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
    """Write the current warm state atomically (tmp + os.replace). Throttled so a burst of
    verdicts within a few seconds writes once; `force=True` (prewarm / shutdown) writes now."""
    global _last_persist
    p = _path()
    if not p:
        return
    now = time.time()
    if not force and (now - _last_persist) < _PERSIST_THROTTLE_SEC:
        return
    try:
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_snapshot(), f)
        os.replace(tmp, p)
        _last_persist = now
    except Exception:
        pass


def note_update() -> None:
    """A persistable warm decision (capturability verdict) just changed. Persist on
    a throttle so ordinary cooks accumulate the warm state without a disk write per verdict."""
    persist(force=False)


def _reset_for_test() -> None:
    """Test hook: forget the load latch + persist throttle so a test can drive load/persist
    deterministically."""
    global _loaded, _last_persist
    _loaded = False
    _last_persist = 0.0
