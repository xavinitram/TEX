"""
C4-ux — near-singularity (guarded-division) diagnostic trace.

A guarded division (`sdiv`, `mod`, the alpha `_safe_div`) silently returns 0 / the
epsilon-guarded value where the denominator vanishes — so a `1/(x-x)` bug produces a
clean-looking black frame, not a magenta NaN. This trace makes that visible: while
ARMED it records how many elements hit the epsilon branch and accumulates a per-pixel
mask, so the node can paint those pixels **cyan** (distinct from DBG-3's magenta NaN)
and report the count.

**Zero-cost when disarmed** — the toggle-off contract (as DBG-3). `note()`'s first act
is a thread-local bool check; when the `debug_nan_highlight` toggle is off the trace is
never armed, so every guard hook is one predictable branch and returns. Because the
counter accumulates a GPU reduction per guard-fire, it is armed only WITH the toggle (not
always-on): an unconditional per-division reduction would regress the hot path — the exact
trap this repo's perf methodology forbids — and could not thread through the codegen tier
anyway. So the count ships in `tex_perf` when the diagnostic is on, and off is free.

Thread-local (armed state included) — the auto-tier's background compile runs on a worker
thread, so a debug cook on one thread must not arm (or leak into) another's guards.

Scope: the hooks live in the INTERPRETER stdlib guards (`sdiv`/`mod`/`_safe_div`). The
codegen/graph tiers inline their own guarded division and are not instrumented, so cyan
highlighting reflects the interpreter path — the default and the tier a user debugs on
(`compile_mode="none"`). Instrumenting generated code is a v0.20+ concern, not worth the
codegen complexity for a debug-only diagnostic.
"""
import threading

_local = threading.local()


def arm() -> None:
    """Begin recording guard-fires for this cook (resets the accumulators)."""
    _local.armed = True
    _local.count = 0
    _local.mask = None


def reset() -> None:
    """Zero the accumulators and stay armed: call when the engine discards a cook attempt
    and re-cooks (fp16 fallback, out-of-memory retry), so the count and mask describe only
    the attempt that is returned."""
    _local.count = 0
    _local.mask = None


def disarm() -> None:
    """Stop recording and release the accumulated mask (the engine reads it, and holds its
    own reference, before disarming); the count stays readable until the next `arm()`."""
    _local.armed = False
    _local.mask = None


def armed() -> bool:
    return getattr(_local, "armed", False)


def note(mask) -> None:
    """Record one guarded-division epsilon-branch `mask` (a boolean tensor, True where the
    guard fired). No-op unless armed. Never raises — a diagnostic must not break a cook."""
    if not getattr(_local, "armed", False):
        return
    try:
        fired = int(mask.sum())
        # Reduce to a per-pixel [..., H, W] boolean: collapse a channel-like trailing dim.
        pix = mask
        if pix.dim() >= 4 and pix.shape[-1] <= 4:
            pix = pix.any(dim=-1)
        if pix.dim() >= 2:   # a scalar/1-D guard is counted below but can't localise to pixels
            old = _local.mask
            _local.mask = pix if old is None or old.shape != pix.shape else (old | pix)
        _local.count += fired
    except Exception:
        pass


def count() -> int:
    return getattr(_local, "count", 0)


def mask():
    """The accumulated per-pixel guard-fire mask for this cook (or None)."""
    return getattr(_local, "mask", None)
