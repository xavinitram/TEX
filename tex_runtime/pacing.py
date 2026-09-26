"""
PACE-45 — bounding how far the host may queue GPU work ahead of the device when a
cancel token asks for it. PACE-462 (v0.46.2) replaces the original one-poll-interval
mechanism with **bounded look-ahead pacing**, because the original mechanism's cost when
nothing pre-empts turned out not to be negligible: an embedding host measured a paced
background render running 2.0x (sm_120) / 1.67x (sm_75) slower than unpaced, entirely from
the fixed per-poll sleep in the old `event.query()` loop. That is not an acceptable
standing cost for a host that wants to pace EVERY cancellable background cook, not just an
untrusted-tool ceiling.

**The regression this answers (unchanged from PACE-45).** Every cancel-poll point (the
interpreter's per-top-level-statement `_cancel_check`, the cancel-aware codegen tier's
in-body `_CK()` polls, the stencil route's entry poll, and a naturally multi-pass builtin's
`poll_cook_cancel` between its own internal passes — the four SCHED-3/CANCEL-44 yield-point
families) only ever fires while the host thread is still QUEUEING kernels. On CUDA that
queueing is asynchronous and fast (tens of milliseconds even for a long chain of heavy
statements), so every poll lands, is answered, and the whole program is on the device's
queue before a token that trips DURING the real GPU work has any yield point left to reach
— the device then drains for however long the queued work actually takes, unobserved.

**The fix, still opt-in.** A token that wants the bound sets a truthy `pace` attribute on
itself (`wants_pacing` below) — `getattr(token, "pace", False)` costs nothing measurable and
every token that predates this ask (a bare `.check()` double, the ComfyUI interrupt bridge)
has no such attribute, so it reads False and nothing about that token's cook moves: same
polls, same timing, same output. `cancel=None` never reaches this module's real branch at
all.

**The PACE-462 mechanism: bounded look-ahead, not one-poll-interval.** Per thread, a FIFO of
CUDA events recorded at poll points. Only when more than `depth` events are already
outstanding does a poll point wait — and it waits on the OLDEST one — so the device always
has up to `depth` poll-intervals of work queued (no bubble when nothing pre-empts) and a
pre-empted cook leaves at most `depth` poll-intervals of work behind on the device. `depth`
defaults to a module constant chosen by measurement (`_DEFAULT_DEPTH`, see the benchmark
under `benchmarks/preempt_drain_bench.py`); a host that wants a different bound sets an int
`pace_depth` attribute (>=1) on its token — deliberately a SEPARATE attribute from `pace`
itself, because `True == 1` in Python and reading a depth out of the opt-in flag would
silently pin every caller to depth 1 the moment it opted in.

The wait, when the ring is full, is a real blocking `event.synchronize()` on an event
created with `blocking=True` — chosen over the original `event.query()` + sleep loop by
measurement (see the hand-back): a blocking wait releases the GIL while parked (another
Python thread keeps making progress) and costs nothing beyond the wait itself, where the
poll loop paid a fixed sleep on every single poll regardless of whether the device was
actually behind. `token.check()` is still called once per poll point BEFORE the ring is
touched (so an already-tripped token is caught before any device interaction), and again
immediately AFTER a wait completes (so a token that trips WHILE the host is parked in
`synchronize()` is caught as soon as the wait returns, not only on the next ordinary poll).

**The cheap path.** The ring is a per-thread list of preallocated `torch.cuda.Event`
objects, re-`record()`ed rather than recreated: once a cook has run long enough to fill the
ring (`depth` polls), every later poll only calls `.record()` on an already-existing event —
no allocation. `reset()` grows the ring if a later cook asks for a bigger `depth` than the
thread has ever needed, and never shrinks it, so warm-thread cost trends toward "event
recording only" the way the ask's target names it.

**OVERHEAD-462's default-path finding, and the second cut here.** Every CUDA cook —
paced or not — pays `reset()` + `cook_done_event()`, and the attribution pass measured
~18us/cook on the default (unpaced) path, dominated by `with torch.cuda.device(device):`
entry/exit (~5us, a `cudaGetDevice`/`cudaSetDevice` round trip) plus re-deriving "is this
CUDA, and which device index" from scratch in `cook_done_event` even though `reset()` — the
seam every cook already calls first — could answer it once. Fixed here by resolving
"is-CUDA / device index / is-this-already-the-ambient-device" exactly ONCE per cook, in
`reset()`, and having both `cook_done_event` and `paced_check`'s event-record step read
that cached answer instead of re-deriving it: `torch.cuda.device(...)` is entered only when
the cook's device is NOT already ambient-current (the overwhelming common case is a
single-GPU host, or any cook already dispatched on its own device, where recording with no
context switch at all produces a byte-identical event). A caller that reaches
`cook_done_event`/`paced_check` without a prior `reset()` on this thread (none exist in this
tree, but the contract matters — same spirit as O5's `reset()`-with-no-arguments fallback)
recomputes fresh rather than trusting a stale or absent cache. Measured (this box, direct
microbenchmark, quiet, see the hand-back): the old `cook_done_event()` alone cost ~7.4us/call;
the new `reset()`+`cook_done_event()` pair — the actual per-cook cost, since every cook pays
both — costs ~5.8us, a ~21% cut. Against a whole real cook (hundreds of us, end to end), that
delta is within the whole-cook noise floor, the same LAT-4 lesson: a microsecond-class win is
provable directly, never by timing the cook around it.

Thread-local (mirrors `stdlib_core._cook_ctx`): a second cook on another thread must not
share, or wait on, this cook's event ring."""
from __future__ import annotations

import threading as _threading

import torch

_state = _threading.local()

#: Bounded look-ahead depth used when a token opts into pacing (`pace=True`) without naming
#: its own `pace_depth`. Chosen by the PACE-462 K-sweep (`benchmarks/preempt_drain_bench.py`,
#: laptop sm_120, quiet box): depths 1-8 all read within noise of unpaced (-0.89%..+0.54%)
#: on the ask's target shape (a 100+ms heavy background chain), so cost does not discriminate
#: among them there -- the discriminator is the drained-p95 bound, which scales with depth
#: (measured p50/p95 ms: depth1 8.9/10.8, depth2 14.1/18.6, depth4 23.7/29.6, depth8 45.3/50.6).
#: Depth 2 keeps that bound tight (about 4-6% of the measured full runtime) while giving one
#: level of look-ahead margin over depth 1, which read as the worst case (of depths tested) on
#: an informal secondary check with many cheap per-statement kernels rather than this ask's
#: heavy-chain target shape. See the hand-back for the full per-depth table and the sm_75 gap
#: (unreachable this session).
_DEFAULT_DEPTH = 2


def wants_pacing(token) -> bool:
    """The opt-in gate: a token paces host-side queue-ahead only when it says so itself, via
    a truthy `pace` attribute. `getattr` on a token with no such attribute — every caller
    before this ask — reads False, so its cook's polls and timing are exactly what they were."""
    return bool(getattr(token, "pace", False))


def _resolve_depth(token) -> int:
    """The look-ahead depth for this cook: `token.pace_depth` if the token names one
    (validated: must be a plain positive `int` — deliberately `type(x) is int`, not
    `isinstance`, so a `bool` (`True == 1`) is rejected rather than silently accepted as
    depth 1), else `_DEFAULT_DEPTH`. Depth is never read out of `pace` itself."""
    depth = getattr(token, "pace_depth", None)
    if depth is None:
        return _DEFAULT_DEPTH
    if type(depth) is not int or depth < 1:
        raise ValueError(f"pace_depth must be a positive int, got {depth!r}")
    return depth


def _is_cuda(device) -> bool:
    """`device` names a CUDA device AND CUDA is actually usable — off CUDA (CPU, or a string
    torch can't parse) pacing must never engage, so both a bad device spelling and a stray
    CUDA string on a CPU-only box read as False rather than raising. Kept as a small,
    independently-usable predicate (some call sites only need the bool); `_resolve_cuda_target`
    below is the version `reset()` uses that also resolves the device INDEX in the same parse."""
    try:
        d = device if isinstance(device, torch.device) else torch.device(device)
    except Exception:
        return False
    return d.type == "cuda" and torch.cuda.is_available()


def _resolve_cuda_target(device):
    """One parse, answering everything a poll point or `cook_done_event` needs about
    *device*: `(is_cuda, index, is_current)`. `index` is the resolved CUDA device index when
    `is_cuda` (never `None` in that case — an unindexed `"cuda"` resolves to
    `torch.cuda.current_device()`, matching what `torch.cuda.device(device)` would have
    targeted anyway). `is_current` is whether that index is already the thread's ambient
    CUDA device, i.e. whether entering `torch.cuda.device(...)` would be a no-op round trip.
    Off CUDA, or on an unparseable device, returns `(False, None, False)` without ever
    calling `torch.cuda.current_device()` (a bad spelling or a CPU-only box must not pay for,
    or crash on, a CUDA query it will never use)."""
    if not _is_cuda(device):
        return False, None, False
    d = device if isinstance(device, torch.device) else torch.device(device)
    current = torch.cuda.current_device()
    idx = d.index if d.index is not None else current
    return True, idx, idx == current


def _record_on(ev, device, is_current) -> None:
    """Record *ev* on *device*'s stream. Skips the `torch.cuda.device(...)` context manager
    (a `cudaGetDevice`/`cudaSetDevice` round trip, ~5us — OVERHEAD-462) when *is_current*
    says the cook's device is already the thread's ambient CUDA device: recording with no
    context switch at all is byte-identical to switching to the device you are already on.
    O4 (v0.46, FIX-OBSROUTE) still applies in full when it is not: the same
    `torch.cuda.device(...)` discipline `graphed.py:571` uses to replay on the cook's own
    device, so a cook on a non-default CUDA device never paces (or fences) against an event
    recorded on the WRONG device."""
    if is_current:
        ev.record()
    else:
        with torch.cuda.device(device):
            ev.record()


def reset(token=None, device=None) -> None:
    """Start a fresh poll sequence with no pacing history to inherit. Called once at the top
    of every cook via `stdlib_core.set_cook_grid` — the one seam every tier already uses to
    publish its own cook state — and once more at a route whose own first poll can fire
    before that seam does (the stencil route's entry check), so that poll never waits on a
    stale event left over from an unrelated, already-returned cook on this thread.

    Resolves "does THIS cook want pacing?" exactly once, here — `wants_pacing(token) and
    is_cuda` — rather than re-deriving it at every single `paced_check` poll point for the
    cook's whole duration (O5, v0.46). `token`/`device` default to `None` (reads as "no
    pacing"), so a caller that predates O5 and still calls `reset()` with no arguments gets
    the exact same answer `paced_check` used to compute fresh each time on a bare/absent
    token.

    OVERHEAD-462: ALSO resolves, once, the answer `cook_done_event` and the event-record
    step need regardless of pacing — is this CUDA, which index, and is that index already
    ambient-current — and caches it (`_state.is_cuda`/`device_index`/`is_current`/
    `resolved_device`) for both to reuse instead of re-deriving it. This runs unconditionally
    (no longer short-circuited behind `wants_pacing`), because `cook_done_event` needs the
    answer on EVERY CUDA cook whether or not that cook paces, and it previously computed it
    itself from scratch every single time; resolving it here removes that duplicate work
    rather than adding new work. On a CPU cook this is one cheap `torch.device()` construct-
    and-compare (`_is_cuda`'s existing short-circuit — no `is_available()` call once the type
    check alone answers `False`), the same cost `cook_done_event` used to pay alone.

    PACE-462: also resolves this cook's look-ahead `depth` once, the same way `paced`
    already was. The event RING itself is NOT reset here — it is a per-thread resource that
    outlives any one cook, grown (never shrunk, never recreated) only when a cook asks for a
    bigger depth than this thread has ever needed, so a thread's Nth cook pays for event
    allocation at most once per ring slot, not once per cook. Only the ring's head/count
    (this cook's own poll sequence) start fresh."""
    is_cuda, idx, is_current = _resolve_cuda_target(device)
    _state.is_cuda = is_cuda
    _state.device_index = idx
    _state.is_current = is_current
    _state.resolved_device = device
    _state.paced = wants_pacing(token) and is_cuda
    if _state.paced:
        _state.depth = _resolve_depth(token)
        ring = getattr(_state, "ring", None)
        if ring is None:
            ring = []
        if len(ring) < _state.depth:
            ring = ring + [None] * (_state.depth - len(ring))
        _state.ring = ring
    _state.head = 0
    _state.count = 0


def paced_check(token, device) -> None:
    """One poll point. `token is None` is the untouched default path (a no-op, exactly
    `host._cancel_check`'s own body). A token that does not ask for pacing, or a cook that
    is not on CUDA, is the SAME body too — one `token.check()` — so the unpaced cost is
    identical to before this ask plus one cheap attribute read.

    Paced (a CUDA cook, a token with a truthy `pace`): polls the token first (an
    already-tripped token is caught before any device interaction), then, only if the
    look-ahead ring already holds `depth` outstanding events, blocks on the OLDEST one
    (`event.synchronize()`, a real blocking wait — not a busy `query()` loop) and polls the
    token again immediately after, so a trip that lands WHILE the host is parked in the wait
    is caught as soon as the wait returns rather than only on the next ordinary poll. Then
    records (or, warm, re-records) the ring's next slot as the NEW outstanding event for a
    future poll point to wait on — via `_record_on`, which skips the device context manager
    when this cook's device is already ambient-current (OVERHEAD-462, resolved once by
    `reset()` above).

    `_state.paced`/`_state.depth` are read off state resolved once by `reset()` (above)
    rather than recomputed here every call — `getattr(..., False)` covers a poll reached
    without a prior `reset()` on this thread (reads as unpaced, the old default)."""
    if token is None:
        return
    if not getattr(_state, "paced", False):
        token.check()
        return

    token.check()

    depth = getattr(_state, "depth", _DEFAULT_DEPTH)
    ring = getattr(_state, "ring", None)
    if ring is None or len(ring) < depth:
        ring = (ring or []) + [None] * (depth - len(ring or []))
        _state.ring = ring
    head = getattr(_state, "head", 0)
    count = getattr(_state, "count", 0)

    if count >= depth:
        # The ring already holds `depth` outstanding events: the device is up to `depth`
        # poll-intervals behind the host. Wait on the OLDEST before queuing anything newer,
        # so the host never gets more than `depth` intervals ahead.
        oldest = ring[head]
        oldest.synchronize()
        token.check()
        head = (head + 1) % depth
        count -= 1

    tail = (head + count) % depth
    ev = ring[tail]
    is_current = getattr(_state, "is_current", False)
    if ev is None:
        # Cold ring slot: this thread has never needed this many outstanding events before.
        # `blocking=True` so a wait on this event (above) releases the GIL.
        ev = torch.cuda.Event(blocking=True)
        ring[tail] = ev
    _record_on(ev, device, is_current)

    _state.head = head
    _state.count = count + 1


def cook_done_event(device) -> "torch.cuda.Event | None":
    """A "GPU work done" fence: a CUDA event recorded on *device*'s current stream, marking
    this cook's LAST launch so far — `None` off CUDA. Recording an event is itself just
    another stream-ordered enqueue (like any kernel launch), so this costs nothing unless a
    caller later reads or synchronizes it. Always a FRESH `torch.cuda.Event()` — callers may
    hold onto and synchronize `CookResult.done` well after this cook returns, so it is never
    drawn from `paced_check`'s reusable ring.

    OVERHEAD-462: reuses `reset()`'s cached is-CUDA/index/is-current answer for *this exact*
    `device` instead of re-deriving it (the duplicate-parse half of the fix), and
    `_record_on` skips `torch.cuda.device(...)` when the cook's device is already
    ambient-current (the context-manager-cost half — O4's non-current-device correctness is
    unchanged, see `_record_on`). Falls back to a fresh `_resolve_cuda_target` call — exactly
    what this function did before OVERHEAD-462 — when there is no cached answer for this
    device on this thread (no `reset()` call reached this cook, or it targeted a different
    device), so a caller that predates this cut, or a stale/foreign cache, still gets the
    correct answer rather than a wrong cached one."""
    if getattr(_state, "is_cuda", None) is not None and getattr(_state, "resolved_device", object()) == device:
        is_cuda, is_current = _state.is_cuda, _state.is_current
    else:
        is_cuda, _idx, is_current = _resolve_cuda_target(device)
    if not is_cuda:
        return None
    ev = torch.cuda.Event()
    _record_on(ev, device, is_current)
    return ev
