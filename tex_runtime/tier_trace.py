"""
TST-5 — tier-execution observability.

Records which acceleration tier actually served the last cook, and (on a
fallback) which tier declined and why. Replaces the `_show_once`-and-forget
logging and the fragile per-test `_plain_execute` monkeypatch: a tier that
*stops* engaging becomes a red test (`tier_trace.last().tier != 'codegen'`),
not a silent 3x slowdown — and the fuzzer/edge-matrix can assert "no unexpected
fallback" for free.

Thread-local because the auto-tier's background compile runs on a worker thread;
each cook's record lives on the thread that produced its result. Recording is
one attribute write per cook (never per-pixel) — perf-neutral.
"""
import collections
import threading

_local = threading.local()

# DBG-4: a small PROCESS-wide ring of recent tier decisions, so the `tex doctor` route
# (which runs on the server thread, not the cook thread) can report what actually ran.
_ring = collections.deque(maxlen=16)


class TierRecord:
    __slots__ = ("tier", "fallback_from", "reason")

    def __init__(self, tier, fallback_from=None, reason=None):
        self.tier = tier                    # which tier PRODUCED the result
        self.fallback_from = fallback_from  # the tier that declined, if any
        self.reason = reason                # why it declined (diagnostics)

    def __repr__(self):
        if self.fallback_from:
            return (f"<TierRecord {self.tier} (fell back from {self.fallback_from}"
                    f": {self.reason})>")
        return f"<TierRecord {self.tier}>"


def record(tier, fallback_from=None, reason=None):
    """Record the tier that served this cook. Called at the tier-DECISION sites
    (codegen success/fallback, graph replay); the interpreter primitive itself
    does not record, so a fallback shows as tier='interpreter'."""
    _local.last = TierRecord(tier, fallback_from, reason)
    _ring.append({"tier": tier, "fallback_from": fallback_from, "reason": reason})


def recent():
    """DBG-4: the recent process-wide tier decisions (newest last), for `tex doctor`."""
    return list(_ring)


def last():
    """The last cook's TierRecord on this thread, or None."""
    return getattr(_local, "last", None)


def record_precision(precision, reason=None):
    """PR-LP2: record the resolved precision for this cook (especially the auto-mode
    fp16/fp32 decision) + why. Surfaced by the DBG-1 HUD and asserted by the auto
    gate's determinism test — so the decision is never silent."""
    _local.precision = (precision, reason)


def last_precision():
    """(precision, reason) for the last cook on this thread, or None."""
    return getattr(_local, "precision", None)


def record_probe(label, value, x, y):
    """LX-5: append a debug_print value-at-pixel probe for this cook (drained by
    execute() into the same ui= payload as the tier facts)."""
    probes = getattr(_local, "probes", None)
    if probes is None:
        probes = _local.probes = []
    probes.append({"label": str(label), "value": value, "x": int(x), "y": int(y)})


def get_probes():
    """This cook's debug_print probes on this thread (a list, possibly empty)."""
    return getattr(_local, "probes", None) or []


def clear_probes():
    """Drop this thread's probes without touching the tier/precision record — used before
    an auto fp16->fp32 re-cook so the discarded cook's probes don't duplicate (audit)."""
    _local.probes = []


def reset():
    """Clear this thread's record (tests call this before a cook to detect a
    tier that silently didn't run at all)."""
    _local.last = None
    _local.precision = None
    _local.probes = []
