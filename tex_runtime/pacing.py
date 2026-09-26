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

The wait, when `depth` events are already outstanding, is a real blocking
`event.synchronize()` on an event created with `blocking=True` — chosen over the original
`event.query()` + sleep loop by measurement (see the hand-back): a blocking wait releases
the GIL while parked (another Python thread keeps making progress) and costs nothing beyond
the wait itself, where the poll loop paid a fixed sleep on every single poll regardless of
whether the device was actually behind. `token.check()` is still called once per poll point
BEFORE the pool is touched (so an already-tripped token is caught before any device
interaction), and again immediately AFTER a wait completes (so a token that trips WHILE the
host is parked in `synchronize()` is caught as soon as the wait returns, not only on the
next ordinary poll).

**The cheap path (Phase C/P5: a FIFO of outstanding events plus a small free pool, per
device — R2#1).** Each CUDA device index this thread has ever paced for gets its own pool:
an `outstanding` FIFO (events genuinely in flight, from THIS cook's own poll sequence — a
fresh cook always starts this empty, handing any prior cook's leftover outstanding events
straight to `free`) and a `free` list of already-built, already-waited `torch.cuda.Event`
objects ready to be re-`record()`ed. The event that gets waited-on is exactly the event
that becomes free to reuse next, so there is no fixed-size array to size, no head/count
pair, and no modulo arithmetic — `reset()` never has to "grow" anything; a pool simply
accumulates events the first time a thread's depth for that device asks for more of them,
and never discards one.

**The STRIDE, added after measuring that depth alone does not help every program shape.**
Depth bounds LOOK-AHEAD in units of poll-intervals, but a "poll-interval" is not a fixed
amount of device work — a chain of many CHEAP per-pixel statements puts a `torch.cuda.Event`
record/wait at EVERY one of them, and a real `event.synchronize()` call has a fixed cost
that is not free next to a kernel that itself takes only microseconds. (A program shaped
like an embedding host's own background render — a `gauss_blur` chain, real per-statement
device work, not a cheap-arithmetic chain — already read within noise WITHOUT striding at
all; the cheap-chain risk is a defensive bound against a class of programs, not evidence
that any specific host workload needs it. See the hand-back for the full measured
table across shapes.) The fix is to stop treating every poll as a candidate to record: a
poll only touches the pool (records/waits) once at least `stride` seconds of HOST time have
passed since it last recorded; every poll in between is `token.check()` alone, no CUDA call
at all. `stride` defaults to a module constant (`_DEFAULT_STRIDE_S`, chosen by measurement)
and a token may override it with a `pace_stride_ms` attribute (non-negative, FINITE int or
float; `0` disables striding, recording at every poll exactly as depth-only pacing did).
**The pre-emption bound becomes approximately `depth * max(stride, one statement's own
device time)`** for the common case of many poll points per unit of device work. The token
is still polled (`token.check()`) at literally every poll point regardless of the stride
gate, so cancellation latency is unaffected by striding; only the pool's record/wait
bookkeeping is throttled.

**The honest bound, stated plainly (Phase C, R4#3): look-ahead is counted in POLL POINTS,
not in device time directly.** All four cancel-poll-point families this module rides (the
interpreter's per-statement poll, the codegen tier's in-body polls, the stencil route's
entry poll, and a multi-pass builtin's between-pass poll — named above) were placed to
bound CANCELLATION latency, not queued device work, and those are different quantities — a
"poll-interval" is not a fixed amount of device work. The `depth * max(stride, ...)` bound above holds well for a
program with many poll points relative to its device work (the common shape this ask was
measured against: an interpreted chain of many statements, or a multi-pass builtin that
polls between its own passes). It is NOT a tight bound for a program with FEW, COARSE poll
points relative to its device work — a single expensive builtin pass between two polls, or
a codegen-tier cook compiled without `emit_cancel_polls` (whose only poll is at entry, before
the whole compiled program's device work is even queued). For those shapes, the device can
fall behind by however much work sits between two consecutive poll points, REGARDLESS of
`depth`/`stride` — the guarantee is only ever as tight as whichever poll-point family the
running program actually hits, not a property of this module alone.

**P2 (Phase C) — the default (unpaced) path stays exactly as cheap as before PACE-462.**
`reset()` short-circuits on `wants_pacing(token)` before touching CUDA/device state at all,
restoring the original PACE-45 shape Python's `and` gave for free; `cook_done_event` keeps
its OWN tiny per-thread memo, keyed on the raw `device` value IT is actually called with,
rather than trying (and, before this fix, failing — see the hand-back) to share a cache
with `reset()`'s differently-shaped raw value.

Thread-local (mirrors `stdlib_core._cook_ctx`): a second cook on another thread must not
share, or wait on, this cook's event pools. `stdlib_core.set_cook_grid`/`restore_cook_ctx`
save/restore this module's own state across a NESTED cook on the same thread (P3)."""
from __future__ import annotations

import math as _math
import threading as _threading
import time as _time
from collections import deque as _deque

import torch

_state = _threading.local()

#: Bounded look-ahead depth used when a token opts into pacing (`pace=True`) without naming
#: its own `pace_depth`. Chosen by the PACE-462 K-sweep (`benchmarks/preempt_drain_bench.py`,
#: laptop sm_120, quiet box): depths 1-8 all read within noise of unpaced, so cost does not
#: discriminate among them there — the discriminator is the drained-p95 bound, which scales
#: with depth. Depth 2 keeps that bound tight while giving one level of look-ahead margin
#: over depth 1. See the hand-back for the full measured per-depth table and the sm_75
#: reading (DOC-6: a dated measurement table belongs in an evidence document, not in a
#: module docstring a reader opens just for the contract — R2#7).
_DEFAULT_DEPTH = 2

#: Minimum HOST time (seconds) that must pass since the pool last recorded before a poll
#: point is allowed to touch it again. Chosen by measurement (laptop sm_120, quiet box)
#: across several program shapes at depth 1-2, from a heavy real-device-work chain (cost
#: within noise at every stride including 0) to chains of 200+ cheap per-pixel statements
#: (cost +40-44% with striding OFF, within a few percent at every nonzero stride tried).
#: See the hand-back for the full (stride x depth) table across shapes (R2#7).
_DEFAULT_STRIDE_S = 0.0005


def wants_pacing(token) -> bool:
    """The opt-in gate: a token paces host-side queue-ahead only when it says so itself, via
    a truthy `pace` attribute. `getattr` on a token with no such attribute — every caller
    before this ask — reads False, so its cook's polls and timing are exactly what they were."""
    return bool(getattr(token, "pace", False))


#: P4 (Phase C): the ceiling on `pace_depth`. Unbounded, an adversarial or misconfigured
#: token grows the per-thread ring by that many `None` slots in one Python list expression
#: BEFORE a single `torch.cuda.Event` is allocated — confirmed accepted and unbounded at
#: `pace_depth=1_000_000_000` (several GB of pure list overhead). 64 is generous next to the
#: K-sweep's own tried range (1-8) and the depth*stride pre-emption bound this ask exists to
#: keep small — a real host has no reason to want look-ahead in the dozens, let alone more.
_MAX_DEPTH = 64


def _resolve_token_attr(token, name, default, types, minimum, maximum=None):
    """Shared shape for validating one optional numeric token attribute (P4, R1#4/R2#4):
    absent (`None`, or no such attribute) resolves to *default*; present, it must be one of
    *types* — checked via the tree's own `isinstance(x, T) and not isinstance(x, bool)`
    idiom (AGENTS.md), which rejects `bool` even though it is an `int` subclass — finite
    (never NaN/inf; a NaN silently compares `False` to every bound check below it, so it
    would otherwise sail through as though non-negative) and within `[minimum, maximum]`
    (`maximum=None` means no ceiling)."""
    val = getattr(token, name, None)
    if val is None:
        return default
    if not isinstance(val, types) or isinstance(val, bool):
        raise ValueError(f"{name} must be one of {types}, got {val!r}")
    if isinstance(val, float) and not _math.isfinite(val):
        raise ValueError(f"{name} must be finite, got {val!r}")
    if val < minimum or (maximum is not None and val > maximum):
        bound = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ValueError(f"{name} must be {bound}, got {val!r}")
    return val


def _resolve_depth(token) -> int:
    """The look-ahead depth for this cook: `token.pace_depth` if the token names one
    (a plain positive `int`, `bool` rejected, capped at `_MAX_DEPTH` — P4), else
    `_DEFAULT_DEPTH`. Depth is never read out of `pace` itself."""
    return _resolve_token_attr(token, "pace_depth", _DEFAULT_DEPTH, (int,), 1, _MAX_DEPTH)


def _resolve_stride(token) -> float:
    """The stride, in SECONDS, for this cook: derived from `token.pace_stride_ms` if the
    token names one (a plain, FINITE `int` or `float` — P4 rejects NaN/inf, which used to
    pass the old `type(x) not in (...) or x < 0` guard silently, since a NaN compares
    `False` to every bound — `bool` rejected, non-negative; `0` disables striding outright,
    recording at every poll), else `_DEFAULT_STRIDE_S`."""
    stride_ms = _resolve_token_attr(token, "pace_stride_ms", _DEFAULT_STRIDE_S * 1000.0,
                                     (int, float), 0)
    return stride_ms / 1000.0


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


def record_on(event, device) -> None:
    """P7 (Phase C): the public seam — record *event* on *device*'s current stream the same
    way this module's own poll points do, skipping the `torch.cuda.device(...)` context
    manager (OVERHEAD-462) when *device* is already the thread's ambient-current CUDA
    device. For a caller elsewhere in `tex_runtime` that already knows its event belongs on
    a CUDA device and wants this module's device-context-skip discipline without
    duplicating it (`profile.py`'s own event-recording helpers are exactly this shape today
    — R1#2/R3#4 in the Phase C simplification/efficiency reviews) — a future caller, not
    this module's own `paced_check`/`cook_done_event`, which read `reset()`'s cached
    `is_current` directly rather than paying this function's own fresh resolve.

    Resolves *device*'s is-current answer FRESH, via `_resolve_cuda_target`, rather than
    trusting any cache of this module's own: a caller reaching this function may not have
    gone through `reset()` at all, or may be recording for a DIFFERENT device than the
    active cook's, so nothing here assumes a prior call happened on this thread. Like
    `_record_on` itself, this is for a CUDA event on a CUDA device — calling it for a
    non-CUDA device is the caller's error to avoid, the same contract every other CUDA-only
    call in this module already carries."""
    _, _, is_current = _resolve_cuda_target(device)
    _record_on(event, device, is_current)


def reset(token=None, device=None) -> None:
    """Start a fresh poll sequence with no pacing history to inherit. Called once at the top
    of every cook via `stdlib_core.set_cook_grid` — the one seam every tier already uses to
    publish its own cook state — and once more at a route whose own first poll can fire
    before that seam does (the stencil route's entry check), so that poll never waits on a
    stale event left over from an unrelated, already-returned cook on this thread.

    Resolves "does THIS cook want pacing?" exactly once, here, rather than re-deriving it at
    every single `paced_check` poll point for the cook's whole duration (O5, v0.46).
    `token`/`device` default to `None` (reads as "no pacing"), so a caller that predates O5
    and still calls `reset()` with no arguments gets the exact same answer `paced_check` used
    to compute fresh each time on a bare/absent token.

    P2 (Phase C): SHORT-CIRCUITS exactly like the pre-OVERHEAD-462 body —
    `wants_pacing(token) and is_cuda`, Python's `and` — for an unpaced token: no
    `_resolve_cuda_target` call, no CUDA/device touch of any kind, `_state.paced` set `False`
    and nothing else written. OVERHEAD-462 had made this resolution run UNCONDITIONALLY (to
    share it with `cook_done_event`), which measured 8.5x-30.7x SLOWER on the exact
    default (unpaced) path invariant 7 protects — the device/CUDA resolution is not free, and
    `cook_done_event` no longer needs anything this function resolves (it keeps its own
    memo, see below), so there is nothing left to share it FOR. A CPU cook or a token with no
    `pace` attribute (every caller before PACE-45) costs one `wants_pacing` attribute read,
    nothing more — the same one-line body this function had before OVERHEAD-462.

    Only when `wants_pacing(token)` is true does this resolve is-CUDA/index/is-current (once,
    here — PACE-45's original short-circuit, restored), then, only if the device really is
    CUDA, this cook's look-ahead `depth` and `stride`, and looks up (or creates) this
    thread's event pool for THIS device index — never a different one (P1/P5: pools are
    keyed by device index, so a same-thread cook that switches CUDA devices always gets a
    fresh, empty pool for the new index; it can never reuse a slot still bound to the old
    device's events, the cross-device crash B1 found). A pool is a per-(thread, device)
    resource that outlives any one cook — its FREE list only grows, never shrinks, for the
    SAME device index — so a thread's Nth cook on a device it has already paced for pays for
    event allocation at most once per pool slot, not once per cook.

    A fresh cook, even a REPEAT one on a warm pool, starts with zero OUTSTANDING events of
    its own: whatever the previous cook on this pool left mid-flight is handed back to the
    free list here (mirroring the old design's unconditional per-cook head=0/count=0 reset),
    so a fresh cook's first paced poll always records/waits regardless of `stride`, exactly
    as its first `depth` polls always record regardless of the pool being warm."""
    if not wants_pacing(token):
        _state.paced = False
        return
    is_cuda, idx, is_current = _resolve_cuda_target(device)
    _state.paced = is_cuda
    if not is_cuda:
        return
    _state.is_current = is_current
    _state.depth = _resolve_depth(token)
    _state.stride_s = _resolve_stride(token)
    pools = getattr(_state, "pools", None)
    if pools is None:
        pools = {}
        _state.pools = pools
    pool = pools.get(idx)
    if pool is None:
        # P1: a pool is per-(thread, DEVICE INDEX) — a same-thread cook that targets a
        # different CUDA device than any prior paced cook on this thread always gets its
        # own fresh pool, never a slot warmed for a foreign device.
        pool = {"outstanding": _deque(), "free": []}
        pools[idx] = pool
    elif pool["outstanding"]:
        # This cook owns none of the PREVIOUS cook's outstanding events (they were never
        # this cook's poll sequence to wait on) — hand them all back to the free list so
        # the underlying Event objects stay warm/reusable without carrying stale
        # bookkeeping across the cook boundary.
        pool["free"].extend(pool["outstanding"])
        pool["outstanding"].clear()
    _state.pool = pool
    _state.last_record_t = None


def paced_check(token, device) -> None:
    """One poll point. `token is None` is the untouched default path (a no-op, exactly
    `host._cancel_check`'s own body). A token that does not ask for pacing, or a cook that
    is not on CUDA, is the SAME body too — one `token.check()` — so the unpaced cost is
    identical to before this ask plus one cheap attribute read.

    Paced (a CUDA cook, a token with a truthy `pace`): polls the token first (an
    already-tripped token is caught before any device interaction) — ALWAYS, regardless of
    what follows, so cancellation latency never depends on the stride gate below. Then, if
    fewer than `stride` seconds of host time have passed since the pool last recorded, this
    poll is DONE: no pool access, no CUDA call at all beyond the `token.check()` already
    paid. That is the stride gate a chain of many cheap statements needs — recording a CUDA
    event at every one of them costs more than the statements themselves, measured (see the
    hand-back).

    Past the stride, the poll behaves exactly as depth-only PACE-462 did: only if this
    cook's device pool already holds `depth` OUTSTANDING events, blocks on the OLDEST one
    (`event.synchronize()`, a real blocking wait — not a busy `query()` loop) and polls the
    token again immediately after, so a trip that lands WHILE the host is parked in the wait
    is caught as soon as the wait returns rather than only on the next ordinary poll — then
    hands that now-free event back to the pool's FREE list (P5: a plain FIFO of outstanding
    events plus a small free pool, R2#1 — the event that gets waited-on is exactly the event
    that becomes free to reuse next, so no fixed-size array, no head/count, no modulo, and
    no `None`-hole cold-slot check are needed). Every bookkeeping mutation (the `popleft()`,
    the `free.append`, the final `outstanding.append`) happens either BEFORE the operation
    that could raise or strictly after it succeeds, so an exception between the wait and
    this poll's own end (a trip caught by the re-check, or `_record_on` itself raising)
    leaves `outstanding` describing exactly what is really outstanding — never a stale
    count (B1#6; this falls out of the deque form rather than needing its own fix).

    Records (or, warm, re-records from the free list) the new outstanding event via
    `_record_on`, which skips the device context manager when this cook's device is already
    ambient-current (OVERHEAD-462, resolved once by `reset()` above). `_state.stride_s`/
    `depth`/`pool`/`is_current`/`last_record_t` are read directly, not via `getattr(...,
    default)`: `reset()` is the sole writer of `_state.paced` and it always writes every one
    of these fields in the SAME call whenever it writes `paced = True` (R2#2), so once past
    the `getattr(_state, "paced", False)` check above — the one read that DOES need a
    default, because it is the only one that must tolerate a poll reached with no prior
    `reset()` on this thread — every field below is guaranteed present."""
    if token is None:
        return
    if not getattr(_state, "paced", False):
        token.check()
        return

    token.check()

    stride = _state.stride_s
    if stride > 0:
        last = _state.last_record_t
        if last is not None and (_time.perf_counter() - last) < stride:
            return  # inside the stride window: token already checked, nothing else to do

    pool = _state.pool
    outstanding, free = pool["outstanding"], pool["free"]
    depth = _state.depth

    if len(outstanding) >= depth:
        # The pool already holds `depth` outstanding events: the device is up to `depth`
        # poll-intervals behind the host. Wait on the OLDEST before queuing anything newer,
        # so the host never gets more than `depth` intervals ahead.
        oldest = outstanding.popleft()
        oldest.synchronize()
        token.check()
        free.append(oldest)

    # `blocking=True` so a wait on this event (above, some FUTURE poll) releases the GIL.
    ev = free.pop() if free else torch.cuda.Event(blocking=True)
    _record_on(ev, device, _state.is_current)
    outstanding.append(ev)

    _state.last_record_t = _time.perf_counter()


_UNSET = object()  #: cook_done_event's memo has never been written on this thread yet


def cook_done_event(device) -> "torch.cuda.Event | None":
    """A "GPU work done" fence: a CUDA event recorded on *device*'s current stream, marking
    this cook's LAST launch so far — `None` off CUDA. Recording an event is itself just
    another stream-ordered enqueue (like any kernel launch), so this costs nothing unless a
    caller later reads or synchronizes it. Always a FRESH `torch.cuda.Event()` — callers may
    hold onto and synchronize `CookResult.done` well after this cook returns, so it is never
    drawn from `paced_check`'s reusable ring.

    P2 (Phase C): keeps its OWN tiny per-thread memo, keyed on the RAW *device* value this
    function is actually called with — `tex_engine`'s `ctx.device`, always a plain `str`
    (`resolve_device() -> str`). This is deliberately NOT `reset()`'s state: `reset()` is
    resolved from a DIFFERENT raw value (whichever tier is executing canonicalizes its OWN
    `device` argument to a `torch.device`, e.g. `interpreter.py`'s `self.device`), and
    `torch.device(...) == "<same device>"` is `False` for every device, always — confirmed
    directly against real `torch`. OVERHEAD-462's original fix compared those two raw values
    and so never hit on its one real call site; this fix compares each of the two functions'
    OWN raw values against themselves instead; `reset()` and `cook_done_event` no longer
    share a cache at all; they share the underlying resolution helper. A host that cooks
    repeatedly on the same device string (every single-GPU host, and every fixed-device
    host) hits this memo from the SECOND cook onward. `is_current` is never cached — it is
    re-derived every call from a fresh, cheap `torch.cuda.current_device()` read against the
    memoized index, so a cache hit never goes stale about which device is ambient, only
    about whether *device* itself names CUDA and which index it resolves to."""
    cached_key = getattr(_state, "done_device_key", _UNSET)
    if cached_key is not _UNSET and cached_key == device:
        is_cuda, idx = _state.done_is_cuda, _state.done_idx
    else:
        is_cuda, idx, _is_current_at_resolve = _resolve_cuda_target(device)
        _state.done_device_key = device
        _state.done_is_cuda = is_cuda
        _state.done_idx = idx
    if not is_cuda:
        return None
    is_current = idx == torch.cuda.current_device()
    ev = torch.cuda.Event()
    _record_on(ev, device, is_current)
    return ev


def save_state() -> dict:
    """P3 (Phase C, B1#2): snapshot every per-thread pacing field, for a caller whose own
    save/restore pair must NEST — `stdlib_core.set_cook_grid`/`restore_cook_ctx`, whose own
    docstring says cooks nest (a codegen invocation inside an interpreted fallback, a tiled
    strip loop) and which already saves/restores its OWN four `_cook_ctx` fields for exactly
    that reason. Before this, `set_cook_grid` called `reset()` with no save at all: an inner
    cook's `reset()` unconditionally overwrote `_state`'s `paced`/`depth`/`stride_s`/`ring`/
    `head`/`count`, and `restore_cook_ctx` never knew pacing had state to give back — so a
    real (opt-in, CUDA) outer cook that reached a second, nested `set_cook_grid` before its
    own `finally: restore_cook_ctx` would permanently lose its own pacing bookkeeping to the
    inner cook's.

    A shallow copy of `_state.__dict__` is enough to restore every SCALAR field (`paced`,
    `depth`, `stride_s`, `is_current`, `last_record_t`, which `_pace.pool` an outer cook is
    using) exactly as it was — `reset()` never REPLACES `_state.pools` or any one device's
    pool dict, only mutates one in place, so the reference itself survives an inner cook's
    own `reset()` call unchanged. What this does NOT isolate: an inner cook that nests on
    the SAME device index as the outer shares that ONE pool's `outstanding`/`free` split
    with it (by the design's own "one warm pool per device" contract), so the inner cook's
    `reset()` still hands the outer's then-outstanding events back to `free` before the
    inner cook runs. The scalar bookkeeping this snapshot restores (depth/stride/etc.) is
    exactly what B1#2 asked for; a same-device nested pacer sharing event tracking with its
    parent is the "currently latent, no live caller reaches it" half of that finding, not
    something this snapshot claims to solve."""
    return dict(_state.__dict__)


def restore_state(snapshot: dict) -> None:
    """Undo one `save_state()` — puts every field back exactly as `save_state()` found it,
    including a field an inner cook's `reset()` added that the outer never had (cleared
    first, so nothing inner-only survives the restore)."""
    _state.__dict__.clear()
    _state.__dict__.update(snapshot)
