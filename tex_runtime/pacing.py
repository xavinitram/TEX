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

**The STRIDE, added after measuring that depth alone does not help every program shape.**
Depth bounds LOOK-AHEAD in units of poll-intervals, but a "poll-interval" is not a fixed
amount of device work — a chain of many CHEAP per-pixel statements puts a `torch.cuda.Event`
record/wait at EVERY one of them, and a real `event.synchronize()` call has a fixed cost
that is not free next to a kernel that itself takes only microseconds: measured 10-44%
overhead on such a chain (200+ statements of trivial arithmetic, 256-1024 px) at every depth
swept, with `stride` disabled. (A program shaped like an embedding host's own background
render — a `gauss_blur` chain, real per-statement device work, not the cheap-arithmetic
shape above — already read within noise WITHOUT striding at all; the cheap-chain risk is a
defensive bound against a class of programs, not evidence that any specific host workload
needs it. See the hand-back for the full (stride x depth) table across three shapes.) The
fix is to stop treating every poll as a candidate to record: a poll only touches the ring
(records/waits) once at least `stride` seconds of HOST time have passed since the ring last
recorded; every poll in between is `token.check()` alone, no CUDA call at all. `stride`
defaults to a module constant (`_DEFAULT_STRIDE_S`, chosen by measurement) and a token may
override it with a `pace_stride_ms` attribute (non-negative int or float; `0` disables
striding, recording at every poll exactly as depth-only pacing did). **The pre-emption bound
becomes approximately `depth * max(stride, one statement's own device time)`**: a
poll-interval is now a stride WINDOW, which may contain many cheap statements or, on a heavy
chain where one statement alone exceeds `stride`, exactly one — either way the device is
never more than `depth` such windows behind the host. The token is still polled
(`token.check()`) at literally every poll point regardless of the stride gate, so
cancellation latency is unaffected by striding; only the ring's record/wait bookkeeping is
throttled.

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

import math as _math
import threading as _threading
import time as _time

import torch

_state = _threading.local()

#: Bounded look-ahead depth used when a token opts into pacing (`pace=True`) without naming
#: its own `pace_depth`. Chosen by the PACE-462 K-sweep (`benchmarks/preempt_drain_bench.py`,
#: laptop sm_120, quiet box): depths 1-8 all read within noise of unpaced (-0.89%..+0.54%)
#: on the ask's target shape (a 100+ms heavy background chain), so cost does not discriminate
#: among them there -- the discriminator is the drained-p95 bound, which scales with depth
#: (measured p50/p95 ms: depth1 8.9/10.8, depth2 14.1/18.6, depth4 23.7/29.6, depth8 45.3/50.6).
#: Depth 2 keeps that bound tight (about 4-6% of the measured full runtime) while giving one
#: level of look-ahead margin over depth 1, which the later stride sweep (see
#: `_DEFAULT_STRIDE_S` below) also read as the worst case among depths tried on cheap-statement
#: program shapes. See the hand-back for the full per-depth table and the sm_75 gap (unreachable
#: this session).
_DEFAULT_DEPTH = 2

#: Minimum HOST time (seconds) that must pass since the ring last recorded before a poll
#: point is allowed to touch it again. Chosen by measurement (laptop sm_120, quiet box)
#: across four program shapes at depth 1-2: a heavy `gauss_blur` chain (this ask's own
#: benchmark shape, cost within noise at every stride incl. 0); two chains of 220+ cheap
#: per-pixel statements at 256^2 and 1024^2 (cost +40-44%/+1.7-10.5% with striding OFF,
#: +/-3%/+/-5% at every nonzero stride tried); and a `gauss_blur(3.0)` chain calibrated to
#: ~300ms at 1024^2 mirroring an embedding host's own background-render repro shape (already
#: within noise, striding on or off). 0.5 ms keeps every shape within a few percent of
#: unpaced. See the hand-back for the full (stride x depth) table.
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
    CUDA, this cook's look-ahead `depth` and `stride`, and grows the per-(thread, device)
    look-ahead ring if a bigger `depth` is asked for than this thread has ever needed for
    THIS device index (P1: a different device index drops the ring rather than reusing a
    foreign-device slot — see the comment at the drop below). The ring itself is a per-thread
    resource that outlives any one cook — never shrunk, never recreated for the SAME device —
    so a thread's Nth cook on a device it has already paced for pays for event allocation at
    most once per ring slot, not once per cook. Only the ring's head/count (this cook's own
    poll sequence) and `last_record_t` start fresh, so a fresh cook's first paced poll always
    records/waits regardless of `stride`, exactly as its first `depth` polls always record
    regardless of the ring being warm."""
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
    ring = getattr(_state, "ring", None)
    ring_device_index = getattr(_state, "ring_device_index", None)
    if ring is None or ring_device_index != idx:
        # P1: the ring is thread-local, not (thread, device)-local. A warm slot holds an
        # already-record()ed `torch.cuda.Event`, and a CUDA event binds to whichever
        # device is ambient the first time it is recorded — re-record()ing it while a
        # DIFFERENT device is ambient is a real `cudaEventRecord` device-mismatch crash,
        # not a PyTorch-added restriction. A same-thread cook that targets a different
        # CUDA device than the last paced cook on this thread must never reuse the old
        # ring's slots, so drop it and start fresh — the overwhelmingly common
        # single-GPU-host case takes this branch at most once (the thread's first paced
        # cook), never again.
        ring = []
        _state.ring_device_index = idx
    if len(ring) < _state.depth:
        ring = ring + [None] * (_state.depth - len(ring))
    _state.ring = ring
    _state.head = 0
    _state.count = 0
    _state.last_record_t = None


def paced_check(token, device) -> None:
    """One poll point. `token is None` is the untouched default path (a no-op, exactly
    `host._cancel_check`'s own body). A token that does not ask for pacing, or a cook that
    is not on CUDA, is the SAME body too — one `token.check()` — so the unpaced cost is
    identical to before this ask plus one cheap attribute read.

    Paced (a CUDA cook, a token with a truthy `pace`): polls the token first (an
    already-tripped token is caught before any device interaction) — ALWAYS, regardless of
    what follows, so cancellation latency never depends on the stride gate below. Then, if
    fewer than `stride` seconds of host time have passed since the ring last recorded, this
    poll is DONE: no ring access, no CUDA call at all beyond the `token.check()` already
    paid. That is the stride gate a chain of many cheap statements needs — recording a CUDA
    event at every one of them costs more than the statements themselves, measured (see the
    hand-back).

    Past the stride, the poll behaves exactly as depth-only PACE-462 did: only if the
    look-ahead ring already holds `depth` outstanding events, blocks on the OLDEST one
    (`event.synchronize()`, a real blocking wait — not a busy `query()` loop) and polls the
    token again immediately after, so a trip that lands WHILE the host is parked in the wait
    is caught as soon as the wait returns rather than only on the next ordinary poll. Then
    records (or, warm, re-records) the ring's next slot as the NEW outstanding event for a
    future poll point to wait on — via `_record_on`, which skips the device context manager
    when this cook's device is already ambient-current (OVERHEAD-462, resolved once by
    `reset()` above).

    `_state.paced`/`_state.depth`/`_state.stride_s` are read off state resolved once by
    `reset()` (above) rather than recomputed here every call — `getattr(..., False)` covers
    a poll reached without a prior `reset()` on this thread (reads as unpaced, the old
    default)."""
    if token is None:
        return
    if not getattr(_state, "paced", False):
        token.check()
        return

    token.check()

    stride = getattr(_state, "stride_s", _DEFAULT_STRIDE_S)
    if stride > 0:
        last = getattr(_state, "last_record_t", None)
        if last is not None and (_time.perf_counter() - last) < stride:
            return  # inside the stride window: token already checked, nothing else to do

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

    A shallow copy of `_state.__dict__` is enough: every value here is a plain scalar or the
    ring list/the per-device pool, and neither is ever mutated by REPLACING the object a
    caller's earlier snapshot points at — `reset()` only ever rebinds `_state.ring` to a NEW
    list when it grows, never mutates an old one a snapshot still references, so an outer's
    saved reference stays exactly what it was even if an inner cook's own `reset()` runs
    after this snapshot is taken."""
    return dict(_state.__dict__)


def restore_state(snapshot: dict) -> None:
    """Undo one `save_state()` — puts every field back exactly as `save_state()` found it,
    including a field an inner cook's `reset()` added that the outer never had (cleared
    first, so nothing inner-only survives the restore)."""
    _state.__dict__.clear()
    _state.__dict__.update(snapshot)
