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

Noise-tier provenance (`tex_engine.prepare/cook(want_noise_tiers=True)` fills
`CookResult.noise_tiers`). `noise._TieredCache` swaps a builtin's jit.trace callable
for a torch.compile'd one on that key's fourth CALL — not cook — on the default path,
for the life of the process, and the two agree only to a recorded envelope. A region
recooked after the swap and composited over a frame cooked before it can carry a
seam no band bounds (worley's follows its coordinates; a threshold amplifies any).
So a host composites a patch onto a base only when BOTH records are dicts and EQUAL,
and cooks whole otherwise — which is always correct. The record:

  None  not requested; a tier strategy other than "default" (a compiled or captured
        tier runs noise where this thread's record cannot see it); or some label
        served more than one tier in the cook (a promotion landed mid-cook) —
        `last_noise_tiers()` says which
  {}    requested, default strategy, no tiered noise builtin called
  {label: tier}, one label per cache key ("simplex@cuda:0", "fbm/6@cpu"), tier one of
        "trace"             the jit.trace tier: every key's until its promotion, and for
                            good where no compiler toolchain is present
        "promoted"          the torch.compile tier the promotion installed
        "promotion_failed"  jit.trace, in the cook whose promotion attempt raised (the
                            exception is on `noise_compile_failures()`); later cooks
                            read "trace" again
        "eager"             the eager body: the trace could not be built, or a
                            signature never settled

Labels and tier words are for EQUALITY only, never parsed. Unrequested (every ComfyUI
cook) the whole cost is one thread-local attribute read per tiered noise call.
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


# P6: a small ring of noise torch.compile events, so the one-time compile pause (and any
# future shape recompile — P2's dynamic=True should keep this at one entry per program) is
# VISIBLE in `tex doctor` / the HUD instead of a mystery stall.
_noise_ring = collections.deque(maxlen=16)


def record_noise_compile(name, ms, kind="compile"):
    """P6: record that a noise fn (`name`) was torch.compiled, taking `ms` ms."""
    _noise_ring.append({"noise": name, "ms": round(float(ms), 1), "kind": kind})


def noise_compiles():
    """P6: recent noise compile events (newest last), for `tex doctor` / the HUD."""
    return list(_noise_ring)


# BRIEF-4: a third ring, deliberately separate from `_noise_ring` above. A failed
# torch.compile PROMOTION used to vanish into a bare `except Exception: pass` in
# `noise.py`'s `try_upgrade` — this makes it observable in `tex_doctor.capabilities()`
# instead. Kept off `_noise_ring` on purpose: that ring is rendered by the modal and
# counted per builtin by tests as compile EVENTS, and a failure must never read as one.
_noise_failure_ring = collections.deque(maxlen=16)


def record_noise_compile_failure(name, device_type, exc, key=None):
    """BRIEF-4: record that a noise fn's (`name`) torch.compile promotion failed on
    `device_type` (`"cpu"` / `"cuda"`), with the exception that was raised and swallowed.

    The one hook a failed promotion reaches, so the per-cook noise-tier record learns of it
    here too: when a host asked for this cook's record, `key`'s label reads
    "promotion_failed" for this cook (the ring entry above is unchanged)."""
    _noise_failure_ring.append({"noise": str(name), "device": str(device_type),
                                "error": f"{type(exc).__name__}: {exc}"})
    if key is not None and _noise_tiers.record is not None:
        note_noise_tier(name, key, _PROMOTION_FAILED)


def noise_compile_failures():
    """BRIEF-4: recent noise-promotion failures (newest last), for
    `tex_doctor.capabilities()`'s `noise_promotion@*` rows. Never touched by
    `noise_compiles()` — see the ring's own comment."""
    return list(_noise_failure_ring)


# The per-cook noise-tier record (contract: the module docstring). Its own thread-local, whose
# class attributes are the disarmed defaults, so the read `_TieredCache.call` pays on the default
# path is one attribute lookup — no getattr default, no exception, on any thread.
class _NoiseTiers(threading.local):
    record = None              # None = not asked (the default); else {label: {tier: None}} in call order
    last = (None, None)        # (record, why it is None) for this thread's last cook that asked


_noise_tiers = _NoiseTiers()
_PROMOTION_FAILED = "promotion_failed"


def arm_noise_tiers():
    """Start this thread's record for one cook. Called by `tex_engine.run` only when a host asked."""
    _noise_tiers.record = {}
    _noise_tiers.last = (None, None)


def note_noise_tier(name, key, tier):
    """File the `tier` that served (or, for a failed promotion, befell) a call to the tiered-noise
    cache `name` under `key`. A no-op unless armed; the label is the cache name, the key's other
    parts, and its device — every `_TieredCache` key ends in (or is) its device."""
    record = _noise_tiers.record
    if record is None:
        return
    *parts, device = key if isinstance(key, tuple) and key else (key,)
    label = "/".join([str(name), *map(str, parts)]) + f"@{device}"
    record.setdefault(label, {})[tier] = None


def take_noise_tiers(tier_id="default"):
    """Close this thread's record and return `CookResult.noise_tiers` for the cook that just ran
    on the tier strategy `tier_id` (see the module docstring). Always disarms."""
    record, _noise_tiers.record = _noise_tiers.record, None
    result, reason = {}, None
    if record is None:
        result, reason = None, "disarmed mid-cook (a nested prepare/run on this thread)"
    elif tier_id != "default":
        result, reason = None, f"the {tier_id!r} strategy runs noise where this record cannot see it"
    else:
        for label, tiers in record.items():
            served = [tier for tier in tiers if tier != _PROMOTION_FAILED]
            if len(served) != 1:
                result, reason = None, f"{label} served {', '.join(tiers)} in one cook"
                break
            failed = served[0] == "trace" and _PROMOTION_FAILED in tiers
            result[label] = _PROMOTION_FAILED if failed else served[0]
    _noise_tiers.last = (result, reason)
    return result


def last_noise_tiers():
    """(record, reason) for the most recent cook on this thread that ASKED for its noise tiers —
    a cook that did not ask leaves it alone — where the reason says why that record is None.
    `(None, None)` before any cook asked, and while an asking cook is still running."""
    return _noise_tiers.last


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


def record_roi(cooked_roi, reason=None):
    """ROI-3 / v0.30: record whether this cook narrowed to a sub-window, and why not.

    `cooked_roi` is the `(x0, y0, w, h, W, H)` actually cooked, or None when the cook ran
    whole-frame. v0.30 makes `roi=` a production path, so an armed-but-declined ROI must not
    be silent: a refused window (malformed), a non-executable program (a gather), and a
    fallback after a failed narrow each land here with a reason — the same
    "never-silent decision" discipline as `record_precision`."""
    _local.roi = (cooked_roi, reason)


def last_roi():
    """(cooked_roi, reason) for the last cook on this thread — `(None, None)` if never
    recorded, so a consumer can unpack unconditionally."""
    return getattr(_local, "roi", None) or (None, None)


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
    _local.roi = None          # ROI-3: else CookResult.cooked_roi could read a PRIOR cook's window
    if _noise_tiers.record is not None:   # a cook that raised while asked: disarm (unasked, a check)
        _noise_tiers.record = None
