"""
CC-2 — measured auto-tier decision engine.

Replaces the user-facing `compile_mode` gamble with a policy that MEASURES.
Every program starts on the interpreter; after a few cooks a background compile
is submitted (non-blocking — the cook never stalls the ~28 s compile). When the
compiled fn is ready, the NEXT cook runs it timed and the tier is committed only
if it beats the rolling interpreter median. Ground truth proves assumption-based
routing wrong (warm CPU inductor often loses to the interpreter on small
programs; the same backend wins on pow-heavy ones) — so the verdict is decided
per (program, device, precision, resolution-bucket), on THIS machine.

This module is the pure state machine + persistence. It takes timings as
explicit milliseconds so it is fully deterministic and unit-testable; the
orchestration (background submission, cuda-event timing) lives in compiled.py.

States per key:
  MEASURING  — <N interpreter cooks recorded; keep sampling.
  COMPILING  — a background compile was submitted (non-blocking).
  TRIAL      — compiled fn ready; the next cook runs it timed.
  COMMITTED  — compiled beat the interpreter median → route compiled.
  REJECTED   — compiled lost (or crashed) → route codegen-only (always-safe).
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict, deque

# Tunables ------------------------------------------------------------------
MEASURING = "measuring"
COMPILING = "compiling"
TRIAL = "trial"
COMMITTED = "committed"
REJECTED = "rejected"

_MEASURE_COOKS = 3        # interpreter samples before a compile is worth trying
_COMMIT_RATIO = 0.9       # compiled must be <0.9x the interp median to commit
_ROLL = 8                 # rolling window for medians (recent cooks only)
_STATE_MAX = 512          # bound the in-memory table


class _KeyState:
    __slots__ = ("state", "interp_ms", "compiled_ms", "submitted")

    def __init__(self):
        self.state = MEASURING
        self.interp_ms: deque = deque(maxlen=_ROLL)
        self.compiled_ms: deque = deque(maxlen=_ROLL)
        self.submitted = False


_STATE: "OrderedDict[tuple, _KeyState]" = OrderedDict()


def _median(xs) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def make_key(fingerprint: str, device_type: str, precision: str,
             spatial_shape) -> tuple:
    """Verdicts are resolution-dependent (the win/lose boundary moves with
    pixel count), so bucket by (H*W).bit_length()."""
    px = 1
    if spatial_shape:
        try:
            _b, h, w = spatial_shape
            px = int(h) * int(w)
        except Exception:
            px = 1
    return (fingerprint, device_type, precision, max(1, px).bit_length())


def _get(key: tuple) -> _KeyState:
    st = _STATE.get(key)
    if st is None:
        st = _KeyState()
        _STATE[key] = st
        while len(_STATE) > _STATE_MAX:
            _STATE.popitem(last=False)
    else:
        _STATE.move_to_end(key)
    return st


def verdict(key: tuple) -> str:
    return _get(key).state


def cook_ms(key: tuple) -> float | None:
    """SCHED-2: the effective median cook time (ms) recorded for `key`, or None if this
    program was never cooked on that (device_type, precision, resolution-bucket). The
    scheduler reads this as `cook_cost(node, device)`. Loads persisted verdicts first (a cold
    process has an empty in-memory table), then returns the compiled median for a COMMITTED
    program (the tier it actually runs) else the interpreter/codegen median — mirroring how a
    real cook routes. `0.0`/empty deques → None so the caller falls back to greedy rather than
    treating 'unmeasured' as 'free' (the trap that would place everything on one device)."""
    load()
    st = _STATE.get(key)
    if st is None:
        return None
    ms = _median(st.compiled_ms) if st.state == COMMITTED and st.compiled_ms else _median(st.interp_ms)
    return ms if ms > 0 else None


def record_interp(key: tuple, ms: float) -> None:
    _get(key).interp_ms.append(ms)


def should_submit_compile(key: tuple) -> bool:
    st = _get(key)
    return (st.state == MEASURING and not st.submitted
            and len(st.interp_ms) >= _MEASURE_COOKS)


def mark_submitted(key: tuple) -> None:
    st = _get(key)
    st.submitted = True
    st.state = COMPILING


def mark_ready(key: tuple) -> None:
    """The background compile finished and the fn is cached — trial it next."""
    st = _get(key)
    if st.state == COMPILING:
        st.state = TRIAL


def record_trial(key: tuple, compiled_ms: float | None) -> str:
    """Record a trial timing (or None on trial-fn crash → REJECTED) and decide.
    Commit only when the compiled median is a clear win over the interpreter."""
    st = _get(key)
    if compiled_ms is None:
        st.state = REJECTED
    else:
        st.compiled_ms.append(compiled_ms)
        im = _median(st.interp_ms)
        cm = _median(st.compiled_ms)
        # im == 0 (no interp samples) should not happen; be conservative.
        st.state = COMMITTED if (im > 0 and cm < _COMMIT_RATIO * im) else REJECTED
    _persist()
    return st.state


def reset() -> None:
    """Test hook: drop all in-memory verdicts."""
    _STATE.clear()


# ── Persistence: terminal verdicts survive restarts (with PC-2, a COMMITTED
#    program's artifact is already on disk → next session goes straight to the
#    compiled tier without re-trialling). Versioned by the compiler hash so a
#    code change invalidates stale verdicts.
_persist_path_cache = None
_loaded = False


def _persist_path() -> str | None:
    global _persist_path_cache
    if _persist_path_cache is not None:
        return _persist_path_cache
    try:
        from ..tex_cache import get_cache
        d = get_cache()._cache_dir
        os.makedirs(d, exist_ok=True)
        _persist_path_cache = os.path.join(str(d), "autotier.json")
    except Exception:
        _persist_path_cache = None
    return _persist_path_cache


def _version_tag() -> str:
    try:
        from ..tex_cache import verdict_epoch   # CACHE-4: verdicts gated by the verdict epoch
        import torch
        return f"{verdict_epoch()}_{torch.__version__.split('+')[0]}"
    except Exception:
        return "0"


def load() -> None:
    """Load terminal verdicts once, straight into `_STATE`. The persisted key is
    a JSON list of the tuple's components, so it round-trips losslessly — no
    separate string-keyed staging table is needed. Best-effort; never raises."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    p = _persist_path()
    if not p or not os.path.exists(p):
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != _version_tag():
            return
        for rec in data.get("verdicts", []):
            v = rec.get("verdict")
            if v not in (COMMITTED, REJECTED):
                continue
            # Terminal state; medians seed the window so a re-trial (if ever
            # forced) has context.
            st = _KeyState()
            st.state = v
            st.submitted = True
            if rec.get("interp_ms"):
                st.interp_ms.append(float(rec["interp_ms"]))
            if rec.get("compiled_ms"):
                st.compiled_ms.append(float(rec["compiled_ms"]))
            _STATE[tuple(rec["key"])] = st
    except Exception:
        pass


def _persist() -> None:
    p = _persist_path()
    if not p:
        return
    try:
        verdicts = [
            {"key": list(key), "verdict": st.state,
             "interp_ms": round(_median(st.interp_ms), 4),
             "compiled_ms": round(_median(st.compiled_ms), 4) if st.compiled_ms else None}
            for key, st in _STATE.items() if st.state in (COMMITTED, REJECTED)
        ]
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": _version_tag(), "verdicts": verdicts}, f)
        os.replace(tmp, p)
    except Exception:
        pass


def seed_from_disk(key: tuple) -> None:
    """Adopt any persisted terminal verdicts into the live table (so a restart
    skips MEASURING/TRIAL). Loads the whole file once — verdicts are terminal."""
    load()
