"""
PACE-45 — bounding how far the host may queue GPU work ahead of the device when a
cancel token asks for it.

**The regression this answers.** Every cancel-poll point (the interpreter's per-top-level-
statement `_cancel_check`, the cancel-aware codegen tier's in-body `_CK()` polls, the
stencil route's entry poll, and a naturally multi-pass builtin's `poll_cook_cancel` between
its own internal passes — the four SCHED-3/CANCEL-44 yield-point families) only ever fires
while the host thread is still QUEUEING kernels. On CUDA that queueing is asynchronous and
fast (tens of milliseconds even for a long chain of heavy statements), so every poll lands,
is answered, and the whole program is on the device's queue before a token that trips
DURING the real GPU work has any yield point left to reach — the device then drains for
however long the queued work actually takes, unobserved. Earlier releases got away with
this by accident: incidental device syncs (readbacks later removed as pure waste, TRK-66/67/68)
paced the host to the device as a side effect. Removing that waste also removed the pacing.

**The fix, opt-in.** A token that wants the bound sets a truthy `pace` attribute on itself
(`wants_pacing` below) — `getattr(token, "pace", False)` costs nothing measurable and every
token that predates this ask (a bare `.check()` double, the ComfyUI interrupt bridge) has no
such attribute, so it reads False and nothing about that token's cook moves: same polls, same
timing, same output. `cancel=None` never reaches this module's real branch at all. Made
opt-in rather than default-on-whenever-a-token-is-passed deliberately: a host's own cancel
token rides EVERY cook, interactive included (SCHED-3's whole point), so turning pacing on by
that alone would change the default path's timing for every existing caller, which invariant
7 forbids without a measurement proving the cost is zero. A host that wants the ceiling back
(the documented case: a per-tool cook budget) asks for it explicitly, per token.

**The mechanism.** At each poll point that opts in via `paced_check`, this module records a
CUDA event on the device's current stream, then — before the CALLER is allowed to queue any
further work — waits for the event recorded at the PREVIOUS poll point to complete. That
keeps the host at most one poll-interval of device time ahead of the device at any moment.
The wait itself is not `torch.cuda.Event.synchronize()` (a blocking call with no chance to
notice a trip mid-wait): it is a short `event.query()` loop with a small sleep, re-checking
the token on every iteration, so a token that trips WHILE the host is waiting raises
`CookCancelled` immediately rather than only after the wait (and the rest of the queue)
drains.

Thread-local (mirrors `stdlib_core._cook_ctx`): a second cook on another thread must not
share, or wait on, this cook's event chain."""
from __future__ import annotations

import threading as _threading
import time as _time

import torch

_state = _threading.local()

#: Sleep between `event.query()` polls while waiting for the previous poll point's GPU work
#: to finish. Short enough that a token tripping mid-wait is caught promptly (worst case one
#: sleep late); long enough that the wait is not a CPU-spinning busy loop.
_POLL_SLEEP_S = 0.001


def wants_pacing(token) -> bool:
    """The opt-in gate: a token paces host-side queue-ahead only when it says so itself, via
    a truthy `pace` attribute. `getattr` on a token with no such attribute — every caller
    before this ask — reads False, so its cook's polls and timing are exactly what they were."""
    return bool(getattr(token, "pace", False))


def _is_cuda(device) -> bool:
    """`device` names a CUDA device AND CUDA is actually usable — off CUDA (CPU, or a string
    torch can't parse) pacing must never engage, so both a bad device spelling and a stray
    CUDA string on a CPU-only box read as False rather than raising."""
    try:
        d = device if isinstance(device, torch.device) else torch.device(device)
    except Exception:
        return False
    return d.type == "cuda" and torch.cuda.is_available()


def reset() -> None:
    """Start a fresh poll sequence with no pacing history to inherit. Called once at the top
    of every cook via `stdlib_core.set_cook_grid` — the one seam every tier already uses to
    publish its own cook state — and once more at a route whose own first poll can fire
    before that seam does (the stencil route's entry check), so that poll never waits on a
    stale event left over from an unrelated, already-returned cook on this thread. Cheap:
    one attribute write."""
    _state.event = None


def paced_check(token, device) -> None:
    """One poll point. `token is None` is the untouched default path (a no-op, exactly
    `host._cancel_check`'s own body). A token that does not ask for pacing, or a cook that
    is not on CUDA, is the SAME body too — one `token.check()` — so the unpaced cost is
    identical to before this ask plus one cheap attribute read.

    Paced (a CUDA cook, a token with a truthy `pace`): waits for the event recorded at the
    PREVIOUS call to `paced_check` in this cook to complete before returning (letting the
    caller queue further GPU work), polling the token in a short `event.query()` loop rather
    than blocking on `synchronize()` — so a trip mid-wait raises `CookCancelled` promptly.
    Then records a fresh event for the NEXT poll point to wait on."""
    if token is None:
        return
    if not wants_pacing(token) or not _is_cuda(device):
        token.check()
        return
    prev = getattr(_state, "event", None)
    if prev is not None:
        while not prev.query():
            token.check()
            _time.sleep(_POLL_SLEEP_S)
    token.check()
    new_event = torch.cuda.Event()
    new_event.record()
    _state.event = new_event


def cook_done_event(device) -> "torch.cuda.Event | None":
    """A "GPU work done" fence: a CUDA event recorded on *device*'s current stream, marking
    this cook's LAST launch so far — `None` off CUDA. Recording an event is itself just
    another stream-ordered enqueue (like any kernel launch), so this costs nothing unless a
    caller later reads or synchronizes it."""
    if not _is_cuda(device):
        return None
    ev = torch.cuda.Event()
    ev.record()
    return ev
