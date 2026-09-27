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
`event.query()` + sleep loop, chosen by measurement: a blocking wait releases
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
that any specific host workload needs it — reproduce the per-shape reading with
`benchmarks/preempt_drain_bench.py --sweep`.) The fix is to stop treating every poll as a candidate to record: a
poll only touches the pool (records/waits) once at least `stride` seconds of HOST time have
passed since it last recorded; every poll in between is `token.check()` alone, no CUDA call
at all. `stride` defaults to a module constant (`_DEFAULT_STRIDE_S`, chosen by measurement)
and a token may override it with a `pace_stride_ms` attribute (non-negative, FINITE int or
float; `0` disables striding, recording at every poll exactly as depth-only pacing did).

**PACE-47: a poll inside the stride window is a candidate to skip, never a guarantee.**
The shape above — skip purely on HOST-elapsed time — was found to make `stride` leak into
the correctness bound rather than staying a pure cost knob: measured, on a box whose host
dispatch is fast relative to its own device compute (sm_75, paired with a
fast desktop CPU), a chain of few, device-expensive statements (`medium`/`heavy`) can have
SEVERAL statements' worth of host
dispatch complete inside one stride window — so the window's one record covers several
statements' worth of enqueued device work, and the pool's "outstanding" count under-counts
what is really queued. `depth` then bounds something smaller than `depth` poll-intervals of
device work, growing worse as `stride` grows (measured: heavy-chain drained p95 21.7ms at
stride 0 -> 237.8ms at stride 1.0ms on sm_75). The fix: a poll inside the stride window
peeks (non-blocking `event.query()`, never a wait) at the most recently recorded event
before deciding to skip. Skipping is honoured only when that event has already completed —
the device has drained past everything queued so far, so no backlog can be hiding — or when
nothing is outstanding yet. The moment the peek finds the device still behind, the stride
gate is NOT honoured for that poll: it falls through to the ordinary depth-gated
record/wait exactly as `stride=0` would. **The pre-emption bound is therefore `depth`
poll-intervals of real device work, independent of `stride`, at every stride value** — the
device can only ever fall `stride`-window's-worth of HOST time behind at all if it is
genuinely keeping up (in which case there is nothing to bound), because the instant it
falls behind, the very next poll notices via the peek and resumes recording. `stride` is
now purely how often the module is willing to pay `query()`'s own (small, non-blocking) cost
while the device keeps up — a cost knob, never a correctness one. The token is still polled
(`token.check()`) at literally every poll point regardless of the stride gate, so
cancellation latency is unaffected by striding; only the pool's record/wait/peek
bookkeeping is throttled.

**The honest bound, stated plainly (Phase C, R4#3): look-ahead is counted in POLL POINTS,
not in device time directly.** All four cancel-poll-point families this module rides (the
interpreter's per-statement poll, the codegen tier's in-body polls, the stencil route's
entry poll, and a multi-pass builtin's between-pass poll — named above) were placed to
bound CANCELLATION latency, not queued device work, and those are different quantities — a
"poll-interval" is not a fixed amount of device work. The `depth`-poll-intervals bound
above (independent of `stride`, since PACE-47) holds well for a program with many poll
points relative to its device work (the common shape this ask was measured against: an
interpreted chain of many statements, or a multi-pass builtin that polls between its own
passes). It is NOT a tight bound for a program with FEW, COARSE poll points relative to its
device work — a single expensive builtin pass between two polls, or a codegen-tier cook
compiled without `emit_cancel_polls` (whose only poll is at entry, before the whole
compiled program's device work is even queued). For those shapes, the device can fall
behind by however much work sits between two consecutive poll points, REGARDLESS of
`depth`/`stride` — the guarantee is only ever as tight as whichever poll-point family the
running program actually hits, not a property of this module alone.

**P2 (Phase C) — the default (unpaced) path stays exactly as cheap as before PACE-462.**
`reset()` short-circuits on `wants_pacing(token)` before touching CUDA/device state at all,
restoring the original PACE-45 shape Python's `and` gave for free; `cook_done_event` keeps
its OWN tiny per-thread memo, keyed on the raw `device` value IT is actually called with,
rather than trying (and, before this fix, failing) to share a cache
with `reset()`'s differently-shaped raw value.

Thread-local (mirrors `stdlib_core._cook_ctx`): a second cook on another thread must not
share, or wait on, this cook's event pools. `stdlib_core.set_cook_grid`/`restore_cook_ctx`
save/restore this module's own state across a NESTED cook on the same thread (P3).

PACE-49: a MEASURED per-call-site device-time budget, additive to `depth`. Every iteration
through PACE-47e answered "is the statement about to run expensive enough to skip the
completed-tail peek's own blind spot" with a NAME (a registry footprint/`heavy` tag) or a
CONSTANT (`_HEAVY_PIXEL_THRESHOLD`) -- both proxies, and R4-altitude (v0.47 Phase C) named
two real shapes neither proxy covers: a user `for`-loop whose body is per-iteration
expensive but below the pixel threshold (a loop's own top-level statement carries no
footprint of its own -- the tag lives on whatever it CALLS, if anything), and a noise
builtin (`fbm`/etc.) whose runtime `octaves` argument makes ONE call far more expensive
than its binary `heavy` tag communicates. This module already does exactly two CUDA calls
at a poll point -- a non-blocking `query()` peek or a blocking `synchronize()` wait -- and
the moment either one confirms an event complete is exactly the moment that event's own
elapsed device time becomes knowable for free via `elapsed_time()` (a host-side read of two
already-recorded timestamps, never a new device call). PACE-49 reads it there and folds it
into a small, bounded, per-call-site EWMA table (`_COST_TABLE` below -- module-global, not
thread-local, mirroring `profile.py`'s own `_blend`/`_STATE` shape, AUTHOR DECISION 1(a): a
tiny, self-contained duplicate rather than a shared leaf module or an inverted import,
because `profile.py` already imports `pacing` (F5) and the reverse would be circular).

Cold start (AUTHOR DECISION 4(a)): today's registry rule is the seed. A call site with no
estimate yet (unmeasured, or still under `_COST_WARMUP_SAMPLES`) never enters the ms-budget
decision at all -- it rides the EXISTING `heavy`/`stride_s==0` gate exactly as every prior
PACE-4x release did, so tick one of a program this mechanism has never seen is provably no
worse than today. Only once a call site is WARM does its measured `ewma_ms` get summed
against `pace_budget_ms` (a token attribute alongside `pace_depth`/`pace_stride_ms`, `0`
disabling the dimension -- the same escape-hatch shape `stride` already uses) to decide
whether the running total of estimated queued work since the pool's last REAL record has
grown too large to keep economizing, REGARDLESS of what the tail's own peek says -- closing
exactly the gap a MEASUREMENT can close and a proxy cannot (AUTHOR DECISION 3(a): additive
to `depth`, never replacing it).

Opted in per call site (AUTHOR DECISION 2(a)): the `call_site_id` keyword. A caller that
omits it (every pre-PACE-49 call site: the codegen tier's in-body `_CK` polls, a multi-pass
builtin's between-pass poll) pays NOTHING new -- no lock, no dict lookup, no attribution --
byte-for-byte the pre-PACE-49 behaviour. Only the interpreter's own per-top-level-statement
poll (both the plain and the profiled loop) passes one, using `id(stmt)` -- the exact
identity `pacing_heavy.heavy_stmt_ids` already keys its own memo on -- because that
already-computed identity is what makes the design's own for-loop and high-octave-noise
counter-examples resolvable: an interpreted TOP-LEVEL statement gets exactly one poll before
it runs, in a Program object this tree already caches and re-cooks repeatedly (an
interactive host's slider drag, a background preview) -- so a call site's OWN cost, once
measured on one cook, informs every later cook of the SAME statement, regardless of what it
calls or how many pixels it touches."""
from __future__ import annotations

import math as _math
import threading as _threading
import time as _time
from collections import OrderedDict as _OrderedDict
from collections import deque as _deque

import torch

_state = _threading.local()

#: Bounded look-ahead depth used when a token opts into pacing (`pace=True`) without naming
#: its own `pace_depth`. Chosen by measurement (`benchmarks/preempt_drain_bench.py`, its own
#: K-sweep): depth 2 keeps the drained-p95 bound tight while giving one level of look-ahead
#: margin over depth 1, at a cost within noise of unpaced across the depths tried. A build
#: that wants to change this is a decision to re-measure with that benchmark, not a
#: tolerance to widen by feel.
_DEFAULT_DEPTH = 2

#: Minimum HOST time (seconds) that must pass since the pool last recorded before a poll
#: point is allowed to touch it again -- see `paced_check`'s own docstring for what this
#: buys and what PACE-47's `heavy=`/peek mechanism guarantees regardless of its value.
#: **Superseded by `_DEFAULT_STRIDE_S` below**, kept only as the value every stride/depth
#: combination was FIRST measured against; the override below is the one actually in
#: effect. Chosen by measurement (`benchmarks/preempt_drain_bench.py --sweep`).
_DEFAULT_STRIDE_S = 0.0005


#: **Re-chosen, PACE-47d.** Once every paced poll route honours `heavy` (PACE-47c/47d close
#: the completed-tail blind spot on every route this tree has), `stride` is a pure cost
#: knob, never a correctness bound (see `paced_check`'s own docstring) -- this choice is a
#: latency-margin/cost trade-off, not a safety one. Chosen by measurement
#: (`benchmarks/preempt_drain_bench.py --sweep`, swept across stride x depth x program
#: shape): the largest tested-safe value was also the cheapest, so it is the new default.
#: Overrides the module's original default above rather than editing it in place, so a
#: build that re-measures can tell which reading it is replacing.
_DEFAULT_STRIDE_S = 0.004

#: PACE-47e: a cook whose own (B*H*W) pixel count is at or above this is treated as heavy
#: on EVERY poll, regardless of what any single statement calls — closing the resolution-
#: driven completed-tail blind spot PACE-47d's own footprint-only classification could not
#: see (a `footprint='point'` statement's own device time scales with pixel count, not
#: footprint). Chosen by measurement (`benchmarks/preempt_drain_bench.py --sweep`) against
#: the criterion "device time exceeds the poll's own record cost by a wide enough margin
#: that recording every statement is cheap", deliberately conservative rather than cut
#: close to that crossover: a faster device only ever raises its own true crossover, never
#: lowers it, so a threshold conservative on a slower reference device stays conservative
#: everywhere. Not re-tuned per-box: a single, hand-picked constant, like `_DEFAULT_DEPTH`/
#: `_DEFAULT_STRIDE_S` above.
_HEAVY_PIXEL_THRESHOLD = 512 * 512


#: PACE-49: the default per-poll millisecond budget (see the module docstring's own
#: section). Chosen to sit clearly ABOVE a genuinely cheap point-footprint statement's own
#: measured device cost at a sub-`_HEAVY_PIXEL_THRESHOLD` resolution (this file's own
#: `_HEAVY_PIXEL_THRESHOLD` comment cites ~0.67ms/2.65ms at 1024^2/2048^2 on the reference
#: sm_75 card -- already caught by the pixel threshold, so the budget only has to matter
#: BELOW it) and clearly BELOW the drain-p95 blow-up PACE-47e measured for the class of bug
#: this ask closes (tens of ms). Re-measure with `benchmarks/preempt_drain_bench.py --sweep`
#: before changing; `pace_budget_ms=0` disables the dimension entirely (mirrors `stride`'s
#: own escape hatch), which is the ONLY way to recover byte-for-byte pre-PACE-49 economizing
#: for a token that already sets `pace_stride_ms` explicitly.
_DEFAULT_BUDGET_MS = 8.0

#: PACE-49: measure every call site until it has this many real device-time samples before
#: its estimate is trusted over the registry-derived cold-start guess (AUTHOR DECISION
#: 4(a)) -- mirrors `profile.py`'s own `_WARMUP_SAMPLES` shape and reasoning (a cold
#: estimate is unrepresentative; a few real samples settle it).
_COST_WARMUP_SAMPLES = 3

#: PACE-49: the EWMA blend weight, identical to `profile.py`'s own `_ALPHA` -- duplicated,
#: not imported (AUTHOR DECISION 1(a): `profile.py` already imports this module, so the
#: reverse would be circular).
_COST_ALPHA = 0.35

#: PACE-49: bound on the per-call-site table's size (an LRU, oldest-evicted-first), the same
#: shape and order of magnitude as `profile.py`'s own `_STATE_MAX` -- a long session must
#: never grow this table without bound just because it keeps seeing new call sites (a
#: program edited/reloaded many times, or many distinct programs cooked in one process).
_COST_TABLE_MAX = 512

#: PACE-49: {(call_site_id, device_index, px_bucket): [ewma_ms, samples, anchor]}. `anchor`
#: (FIX-PACE49 P2) is the actual object `call_site_id`'s `id()` was taken from, held by a
#: strong reference and checked by identity on every read/write -- see `_cost_feed`'s own
#: docstring for why (this table's own `id()`-derived key can otherwise alias an unrelated,
#: later object once the original is freed and its address reused). Module-global
#: (NOT thread-local, unlike the rest of this module's `_state`): a call site's own device
#: cost is a property of the (program, device, resolution) triple, not of which worker
#: thread happened to poll it, so a program cooked across several threads (ENG-9's per-cook-
#: thread interpreters) shares one measured history rather than each thread relearning it
#: from cold. Guarded by `_COST_LOCK` below, entirely independent of `_state`'s own
#: thread-local, lock-free bookkeeping.
_COST_TABLE: "_OrderedDict[tuple, list]" = _OrderedDict()

#: Guards `_COST_TABLE`'s own STRUCTURAL mutations (`_cost_feed`'s `popitem`/insertion/
#: `move_to_end`) only. A plain `Lock`: every critical section here is a handful of dict
#: operations (mirrors `profile.py`'s own `_LOCK` reasoning), and this lock is never
#: acquired while any OTHER lock in this tree is held, so there is no ordering cycle to
#: deadlock on. FIX-PACE49 P4: the READ side (`_cost_lookup`, on the already-economized
#: paced skip path) deliberately does NOT take this lock -- see its own docstring.
_COST_LOCK = _threading.Lock()


def _cost_blend(prev: float, ms: float, n: int) -> float:
    """PACE-49: identical rule to `profile.py:_blend` (duplicated, not imported -- see the
    module docstring). A mean while the key is young (sample 2 is worth half, sample 3 a
    third...), an EWMA once `_COST_ALPHA` takes over -- sheds a cold first-sample outlier
    fast without needing dozens of samples to dilute it."""
    if n <= 1:
        return ms
    a = max(_COST_ALPHA, 1.0 / n)
    return a * ms + (1.0 - a) * prev


def _cost_feed(key: tuple, ms: float, anchor=None) -> None:
    """PACE-49: fold one real, retrospectively-attributed device-time reading for *key* into
    the table -- called only from a poll that has ALREADY paid for the CUDA call whose
    completion made *ms* knowable (see `_pace49_attribute`'s own docstring); never a new
    CUDA call itself. Negative/non-finite readings are dropped rather than poisoning the
    EWMA (mirrors `profile.record`'s own `ms is None or ms < 0` guard); `elapsed_time`
    between two real, completed CUDA events should never produce one, but a mocked or
    exotic event implementation is not this function's contract to trust blindly.

    FIX-PACE49 P2 (R3-efficiency.md #4, R4-altitude.md #1): *anchor*, when the caller has
    one, is the actual object *key*'s `id()`-derived component names (the interpreter's own
    `stmt`) -- checked by identity (`is`), the same "a recycled id belongs to a different
    object" guard `pacing_heavy._HEAVY_STMT_MEMO` already uses for the exact same class of
    object. `_COST_TABLE` is module-global and deliberately outlives any one `Program`
    (that is the whole point of the table), but `tex_cache`'s 128-entry Program LRU means a
    freed Program's own statements' addresses CAN be reused by an unrelated, later
    Program's statements -- without this check, that reused address would silently inherit
    a stale EWMA measured on a completely different statement. A caller that omits *anchor*
    (every direct table-seeding call in this tree's own tests, and any future non-AST-keyed
    caller) gets exactly today's key-only behaviour unchanged: two omitted (`None`) anchors
    never mismatch each other, so no bug is introduced for that caller. A mismatch is
    treated exactly like a brand-new key -- reseed fresh, discard whatever the stranger's
    entry held; the stale EWMA is simply never read back and never blended into again."""
    if ms is None or not _math.isfinite(ms) or ms < 0:
        return
    with _COST_LOCK:
        entry = _COST_TABLE.get(key)
        if entry is None or entry[2] is not anchor:
            entry = [0.0, 0, anchor]
            _COST_TABLE[key] = entry
            while len(_COST_TABLE) > _COST_TABLE_MAX:
                _COST_TABLE.popitem(last=False)
        _COST_TABLE.move_to_end(key)
        entry[1] += 1
        entry[0] = _cost_blend(entry[0], ms, entry[1])


def _cost_lookup(key: tuple, anchor=None):
    """PACE-49: `(ewma_ms, samples)` for *key*, or `None` if this call site has never been
    fed a real reading -- the caller (`_pace49_cost_gate`) treats `None` and
    "not yet warm" (`samples < _COST_WARMUP_SAMPLES`) identically: ride the registry-derived
    cold-start rule, never this table, until there is enough real evidence to trust it
    (AUTHOR DECISION 4(a)).

    FIX-PACE49 P2: `anchor` is checked by identity against the entry's own stored anchor
    (see `_cost_feed`'s docstring) -- a mismatch (an aliased `id()`-derived key whose entry
    was fed by a DIFFERENT, since-freed object) reads back as `None`, never the stranger's
    stale estimate.

    FIX-PACE49 P4 (R3-efficiency.md #4): deliberately LOCK-FREE -- this is the
    already-economized "device caught up" SKIP path (`_pace49_cost_gate`'s own caller in
    `paced_check`), the exact path PACE-47b's own P3 fix (v0.47) measured and removed a
    lock/eager-resolution cost from ("~50-75% of the paced skip path's own per-poll
    regression"); PACE-49 must not put a lock back onto it (measured: +145%, 105.2 ns/call
    locked vs. 43.0 ns/call unlocked). A dict `.get()` plus two list-index reads is safe to
    run WITHOUT `_COST_LOCK` under the GIL: no single Python attribute/item read can
    observe a torn write (`_cost_feed`'s own mutations -- `entry[1] += 1`/`entry[0] = ...`
    -- are each a separate, atomic single assignment), so the worst a concurrent writer can
    hand this read is last cook's `(ewma_ms, samples)` OR the freshest one, never a
    corrupted mix of unrelated fields -- the identical safety argument `paced_check`'s own
    `last_confirmed_done` identity cache already relies on for ITS lock-free read. Skips
    the LRU `move_to_end` touch this function used to do on every hit -- an eviction-order
    nicety, not a correctness requirement, and doing it here would need the very lock this
    fix removes; `_cost_feed` (the write side, still locked for its OWN structural
    mutations -- `popitem`/insertion) already touches order on every real record, which is
    the operation that actually matters for keeping a warm call site's entry alive."""
    entry = _COST_TABLE.get(key)
    if entry is None or entry[2] is not anchor:
        return None
    return entry[0], entry[1]


def _resolve_budget_ms(token) -> float:
    """PACE-49: the additive millisecond budget for this cook: `token.pace_budget_ms` if the
    token names one (a plain, FINITE, non-negative `int`/`float` -- `bool` rejected, the same
    `_resolve_token_attr` shape every other pacing knob uses), else `_DEFAULT_BUDGET_MS`. `0`
    disables the dimension outright -- this cook's polls fall back to depth/stride/heavy
    exactly as pre-PACE-49, the same escape hatch `pace_stride_ms=0` already offers for
    striding."""
    return _resolve_token_attr(token, "pace_budget_ms", _DEFAULT_BUDGET_MS, (int, float), 0)


def wants_pacing(token) -> bool:
    """The opt-in gate: a token paces host-side queue-ahead only when it says so itself, via
    a truthy `pace` attribute. `getattr` on a token with no such attribute — every caller
    before this ask — reads False, so its cook's polls and timing are exactly what they were."""
    return bool(getattr(token, "pace", False))


def is_paced() -> bool:
    """P2 (Phase C, R3#1): whether THIS cook actually engaged pacing — `reset()` resolves
    this once per cook (O5) into `_state.paced`; a caller with its own per-statement work
    to do ONLY when a poll will actually read it (a heavy-builtin classification walk,
    thrown away unread by `paced_check` below whenever this reads False) can check here
    FIRST and skip that work entirely, rather than computing it and having `paced_check`
    discard it. Cheaper than `paced_check`'s own `getattr(_state, "paced", False)` need
    not be duplicated by a caller — reading `_state.paced` directly would raise on a
    thread that never called `reset()`, so this keeps the same tolerant `getattr` default.
    A CPU cook, an unwired cancel, or the real ComfyUI default (a token with no `pace`
    attribute) all read False here, identically to how they read inside `paced_check`."""
    return getattr(_state, "paced", False)


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


def reset(token=None, device=None, spatial_shape=None) -> None:
    """Start a fresh poll sequence with no pacing history to inherit. Called once at the top
    of every cook via `stdlib_core.set_cook_grid` — the one seam every tier already uses to
    publish its own cook state — and once more at a route whose own first poll can fire
    before that seam does (the stencil route's entry check), so that poll never waits on a
    stale event left over from an unrelated, already-returned cook on this thread.

    PACE-47e: `spatial_shape` (the cook's own `(B, H, W)`, when the caller has it in hand —
    `set_cook_grid` always does) resolves, ONCE here, whether this cook's own PIXEL COUNT
    alone makes every statement "heavy" regardless of what builtin it calls. PACE-47d's
    `heavy` classification is footprint-derived (halo/halo_arg), which correctly bounds a
    device-EXPENSIVE builtin at any resolution, but says nothing about a `footprint='point'`
    statement's own device time, which scales with PIXELS, not footprint — a trivial
    per-pixel op at 1024^2 or 2048^2 measures ~0.67ms / ~2.65ms of real device time per
    statement on the reference sm_75 card (RTX 2080 SUPER), large enough that the
    completed-tail blind spot reopens at exactly the wider strides PACE-47d's own
    re-measurement picked as the new default: CONFIRMED (measured; reproduce with
    `benchmarks/preempt_drain_bench.py --sweep`) drain p95 blowing up 9-16x at
    `stride=4ms` for cheap1024/cheap2048, the same mechanism PACE-47c closed for HALO
    builtins, just triggered by resolution instead. See `_HEAVY_PIXEL_THRESHOLD`'s own
    comment for how the crossover was chosen.

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
    # PACE-49: resolved once per cook, same shape as depth/stride above. `device_idx`/
    # `px_bucket` complete this cook's own cost-table KEY (with a caller's `call_site_id`);
    # `running_cost_ms` is the ms-budget's own running total, always fresh per cook (never
    # inherited from a prior cook on this pool -- a stale carry-over could force a
    # fall-through this cook's own first poll never earned).
    _state.budget_ms = _resolve_budget_ms(token)
    _state.running_cost_ms = 0.0
    _state.device_idx = idx
    # PACE-47e / FIX-PACE P5 (R2#1): resolved ONCE per cook, from whatever spatial_shape
    # the caller has in hand (`set_cook_grid` always does; a caller with none, e.g. the
    # stencil-only route's own pre-`_invoke_cg` entry poll, resolves 0 pixels here --
    # conservative in the sense of "no worse than before this ask", and `_invoke_cg`'s own
    # `set_cook_grid` call resolves it correctly before that route's per-statement polls
    # run). A cook at or above `_HEAVY_PIXEL_THRESHOLD` reuses the module's OWN existing
    # "never economize" rule instead of a second, parallel field: `stride_s == 0.0`
    # already means "record/wait at every poll" (see its own docstring above) to every
    # consumer of `stride`, so a large-resolution cook simply FORCES that value here,
    # overriding whatever the token itself asked for -- one condition at the read site
    # (`paced_check`'s `if stride > 0 and not heavy:`) instead of two ANDed together.
    pixels = 0
    if spatial_shape is not None:
        try:
            pixels = 1
            for dim in spatial_shape:
                pixels *= int(dim)
        except (TypeError, ValueError):
            pixels = 0
    if pixels >= _HEAVY_PIXEL_THRESHOLD:
        _state.stride_s = 0.0
    # PACE-49: the cost table's own resolution axis, bucketed exactly like `profile.
    # bucket_of` (`px.bit_length()`) so a session drifting within one octave of resolution
    # keeps landing in the same bucket rather than never accumulating samples.
    _state.px_bucket = pixels.bit_length()
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
    #: PACE-47b (R1, the query() cost this ask's own fix added): the event object last
    #: CONFIRMED complete by a `query()` peek, so a run of consecutive economizing polls
    #: against the SAME unchanged tail event pays for one real `query()` call, not one per
    #: poll -- see `paced_check`'s own comment at the read site for why identity alone is
    #: safe here (invalidated on every real record, never stale across a re-`record()`).
    _state.last_confirmed_done = None
    # PACE-49: the pool's own timing anchor (the last event this mechanism has already
    # attributed FROM, and which call site's poll set it) is cleared on every reset, never
    # inherited across a cook boundary — mirroring `last_confirmed_done` just above and for
    # the identical reason (P1, `restore_state`'s own docstring): the pool object itself
    # persists and is shared with a same-device NESTED cook, whose own poll sequence can
    # pop/re-record the very event this anchor points at before this cook's next poll runs,
    # so a carried-over anchor could attribute a stale interval to the wrong call site.
    pool["timed_prev"] = None
    pool["timed_site"] = None
    pool["timed_anchor"] = None  # FIX-PACE49 P2


def paced_check(token, device, heavy: bool = False, call_site_id=None,
                 call_site_anchor=None) -> None:
    """One poll point. `token is None` is the untouched default path (a no-op, exactly
    `host._cancel_check`'s own body). A token that does not ask for pacing, or a cook that
    is not on CUDA, is the SAME body too — one `token.check()` — so the unpaced cost is
    identical to before this ask plus one cheap attribute read.

    **PACE-47c — `heavy`: the completed-tail blind spot both PACE-47 and PACE-47b's cache
    share, closed only where a caller can say so.** The peek (PACE-47) only ever answers
    "has the LAST RECORDED event completed" — it says nothing about how much work has been
    enqueued (or is ABOUT to be enqueued) SINCE that event was recorded. Once that answer is
    "yes", every further poll inside the SAME stride window trusts it and skips, no matter
    how many MORE statements get dispatched in between — a host fast enough (microseconds
    per enqueue) can walk a whole RUN of heavy, real-device-time statements (each ~milliseconds
    to tens-of-milliseconds of actual GPU work) past this poll while the tail happens to have
    already finished, and NONE of them get an outstanding event to be bounded by `depth` —
    exactly PACE-47's original defect, just gated behind "the tail must complete first",
    which is why it is RARE (needs that lucky/unlucky timing) rather than constant, and why
    only the tail (`drained_p95`) is affected, never the median. `heavy=True` closes this
    the only way it CAN be closed without knowing the future: a caller that knows the
    statement it is about to run is expensive (a halo/halo_arg-footprint or otherwise
    non-cheap builtin) says so, and this poll bypasses the stride economization ENTIRELY for
    that one call — falls straight through to the ordinary depth-gated record/wait below,
    exactly as `stride=0` would, regardless of the tail's own state or how little host time
    has passed. A HOST-TIME stride can never bound device work on its own (this defect is
    exactly that failure mode) and neither can a poll-COUNT cap (it cannot tell a run of
    cheap polls, which should keep economizing indefinitely, from a run that happens to
    include a heavy one) — only information about what is ABOUT to run can. Default `False`
    preserves every existing call site's behaviour byte-for-byte (this parameter is new;
    not every call site can pass it yet — a stdlib builtin's own internal poll can, since it
    knows what it is about to run; codegen's in-body `_CK` passes whatever the emitted
    source hard-codes per poll site instead, see PACE-47d).

    **PACE-49 — `call_site_id`: a MEASURED additive ms-budget, opted in per caller.** `None`
    (every call site before this ask, and codegen's/a multi-pass builtin's own poll today)
    means this whole mechanism is inert for this call — no lock, no table lookup, no
    attribution — byte-for-byte the pre-PACE-49 body. When a caller passes an identity (the
    interpreter's per-statement poll passes `id(stmt)`), two things change: (1) at a peek
    that FRESHLY confirms the tail complete (never from the PACE-47b cache — that interval is
    already banked), the real elapsed device time since this mechanism's own last attribution
    point is folded into *this call site's* bounded EWMA table entry (see the module
    docstring); (2) once that entry is WARM, its estimate is summed into this cook's own
    running "estimated ms queued since the last real record" and compared against
    `pace_budget_ms` — if the sum would exceed budget, the stride/peek economization is NOT
    honoured for this poll, regardless of what the tail's own peek said, and the poll falls
    through to the ordinary depth-gated record/wait exactly as `heavy=True` would. A cold or
    not-yet-warm call site never contributes to that sum and never itself forces a
    fall-through — it rides the SAME `heavy`/large-resolution rule this function already had
    (AUTHOR DECISION 4(a): cold start is today's behaviour, never worse).

    **FIX-PACE49 P2 — `call_site_anchor`: the object `call_site_id` actually names, kept
    alongside it for an identity check.** `_COST_TABLE` is module-global and deliberately
    outlives any one `Program` (the whole point of the table), but a freed Program's
    statement can have its `id()` reused by an unrelated, later Program's own statement
    (`tex_cache`'s bounded Program LRU) — without an anchor, that reused id would silently
    read back (and blend into) a stale EWMA measured on a completely different statement.
    The interpreter passes the actual `stmt` object here (only when already paced, per P1's
    own gate); every other caller (every one before this ask) omits it, and two omitted
    (`None`) anchors never mismatch each other, so nothing about an existing caller's
    behaviour changes. See `_cost_feed`'s own docstring for the mismatch-reseeds-fresh rule.

    Paced (a CUDA cook, a token with a truthy `pace`): polls the token first (an
    already-tripped token is caught before any device interaction) — ALWAYS, regardless of
    what follows, so cancellation latency never depends on the stride gate below. Then, if
    fewer than `stride` seconds of host time have passed since the pool last recorded, this
    poll ECONOMIZES only while the device is genuinely keeping up: it peeks at the most
    recently recorded event's own `query()` (non-blocking; never waits) and, if that event
    has already completed, this poll is DONE — no pool access beyond the peek, no
    `event.record()` — because the device has already drained past everything the host has
    queued so far, so skipping the bookkeeping cannot hide a growing backlog (PACE-47). If
    that event has NOT completed (the device is behind), the stride gate is not honoured
    for this poll: falls through to the depth-gated record/wait below exactly as if
    striding were off. **This is what makes `stride` a pure COST knob rather than a
    correctness parameter** (PACE-47, the sm_75 finding): the original gate skipped
    recording purely on HOST-elapsed time, so a host fast enough to dispatch several
    statements inside one stride window could let the device fall arbitrarily far
    behind `depth` poll-intervals without a single poll ever recording an event to notice
    — a fast-host/slow-device combination, measured (reproduce with
    `benchmarks/preempt_drain_bench.py --sweep`). Gating
    the skip on the device's OWN completion state instead means striding
    only ever economizes recording overhead when it is genuinely free to (the device has
    nothing outstanding to fall behind on); the moment it is not, this poll behaves exactly
    like stride=0 and the depth bound reasserts itself within one poll. That is the stride
    gate a chain of many cheap statements needs — recording a CUDA event at every one of
    them costs more than the statements themselves, measured; the
    `query()` peek itself is a non-blocking, already-cheap CUDA call, paid only for the
    polls that land inside a stride window with something still outstanding to peek at.

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

    # P3 (Phase C, R3#2): `pool["free"]`/`_state.depth` are resolved lazily now -- neither
    # is read anywhere on the economize-and-skip path just below (only `pool["outstanding"]`
    # is, for the tail peek), so a poll that skips must never pay for them. Measured ~50-75%
    # of the paced skip path's own per-poll regression was exactly this: the pre-PACE-47
    # ordering resolved `free`/`depth` unconditionally, before the stride gate had a chance
    # to decide the poll needs neither. `pool` (and, inside the peek, `outstanding`) is still
    # resolved eagerly -- the peek itself needs `outstanding` to find the tail.
    pool = _state.pool

    stride = _state.stride_s
    # PACE-47e / FIX-PACE P5 (R2#1): a large-resolution cook's own `stride_s` was already
    # forced to `0.0` by `reset()` above, so this gate needs only its original two
    # conditions -- `stride > 0` alone now also means "this cook's own resolution keeps it
    # out of economization", with no second, parallel field to AND in here. A compiled
    # codegen function is built ONCE and reused across every cook that shares its
    # fingerprint, potentially at DIFFERENT resolutions, so a per-statement heavy/cheap
    # classification decided at BUILD time (PACE-47d's own `_CK(True)`/`_CK()` emission)
    # could never be resolution-correct on its own -- only this per-COOK, run-time
    # `stride_s` resolution can be. A statement a caller marked cheap (halo-derived
    # `heavy=False`) still gets the depth-gated record/wait below when this cook's own
    # resolution alone makes it expensive, because `stride` reads `0` for it.
    if stride > 0 and not heavy:
        last = _state.last_record_t
        if last is not None and (_time.perf_counter() - last) < stride:
            # Inside the stride window: this is a candidate to skip, but ONLY while the
            # device is genuinely keeping up (PACE-47). Peek at the most recently recorded
            # event (non-blocking `query()`, never a wait): if it has already completed,
            # the device has drained past everything queued so far and skipping here
            # cannot hide a backlog -- economize as before. If nothing is outstanding to
            # peek at, there is nothing behind either, so the same skip is safe. If the
            # peeked event has NOT completed, the device is behind: do not honour the
            # stride gate for this poll -- fall through to the depth-gated record/wait
            # below exactly as stride=0 would, so the bound stays honest regardless of how
            # fast the host is dispatching relative to this stride.
            #
            # PACE-47b (R1): a real `query()` call is not free, and a chain of many CHEAP
            # statements can land dozens of consecutive polls inside one window with the
            # SAME tail event (nothing recorded => `outstanding[-1]` never changes) -- once
            # that event is confirmed complete, it stays complete forever until it is
            # `record()`ed again onto a new point, so re-querying the identical, unchanged
            # tail on every one of those polls re-derives an answer that cannot have
            # changed. `_state.last_confirmed_done` remembers WHICH event object the last
            # real `query()` call confirmed, checked by identity (`is`, not equality) --
            # cheap and exact, since a Python object identity can only match a prior
            # confirmation if it is the literal same, still-unrecorded-since event. Every
            # path that appends a freshly `record()`ed event clears this to `None` first,
            # so the cache can never survive a re-arm and answer for the wrong recording.
            outstanding = pool["outstanding"]
            tail = outstanding[-1] if outstanding else None
            was_cached = tail is not None and tail is _state.last_confirmed_done
            if tail is None or was_cached or tail.query():
                # PACE-49: a FRESH confirmation (never one served from the PACE-47b cache,
                # which answers for an interval this mechanism has already banked) is the
                # "moment an event's own elapsed device time becomes knowable for free" the
                # module docstring describes -- attribute it before deciding whether to
                # skip, so the budget check just below sees this call site's latest number.
                if call_site_id is not None and tail is not None and not was_cached:
                    _pace49_attribute(pool, tail, call_site_id, call_site_anchor)
                _state.last_confirmed_done = tail
                if call_site_id is None or _pace49_cost_gate(call_site_id, call_site_anchor):
                    return  # device caught up (and, if measured, within budget)
                # else: a warm call site's own measured cost pushed the running estimate
                # past `pace_budget_ms` -- do not honour the stride/peek skip for this poll;
                # fall through to the ordinary depth-gated record/wait below, exactly as
                # `heavy=True` would.

    # P3: reached only when the poll must actually record/wait (striding off, heavy,
    # large-resolution, the stride window elapsed, or the peek found the device behind) --
    # `outstanding` may already be resolved (the peek above), `free`/`depth` never are yet.
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
        # PACE-49: `synchronize()` above is itself a fresh completion confirmation, the same
        # "free" moment the peek's `query()` is -- attribute from it too, so a cook that
        # never economizes (heavy/large-resolution/stride=0) still measures its own call
        # sites rather than only ever riding the cold-start guess.
        if call_site_id is not None:
            _pace49_attribute(pool, oldest, call_site_id, call_site_anchor)

    # `blocking=True` so a wait on this event (above, some FUTURE poll) releases the GIL.
    # PACE-49: `enable_timing=True` too -- every pool event is now timing-capable, so an
    # `elapsed_time()` read is always available at whichever peek/wait next confirms it
    # complete, at no cost beyond the flag itself (only a PACED cook's own events; the
    # unpaced default path never constructs one).
    ev = free.pop() if free else torch.cuda.Event(blocking=True, enable_timing=True)
    _record_on(ev, device, _state.is_current)
    outstanding.append(ev)
    # PACE-49: seed the timing anchor from THIS record, but ONLY when there is none yet
    # (`reset()` cleared it, and no peek/wait has confirmed anything since) -- never
    # overwrite an EXISTING anchor here: this event has not itself completed yet, so
    # nothing can be attributed FROM it until some LATER poll confirms it (as `tail`,
    # through `_pace49_attribute`); overwriting the anchor now, unconfirmed, would only
    # throw away whatever interval the OLD anchor was still waiting to be diffed against.
    # Safe even though *ev* has not completed: CUDA completes events on one stream in the
    # order they were recorded, so *ev* is guaranteed complete by the time any event
    # recorded strictly after it is confirmed complete -- the same in-order argument
    # `profile.py`'s own PROF-462 comment makes for its lazy fold.
    if call_site_id is not None and pool.get("timed_prev") is None:
        pool["timed_prev"] = ev
        pool["timed_site"] = call_site_id
        pool["timed_anchor"] = call_site_anchor  # FIX-PACE49 P2
    # PACE-47b: invalidate the query() cache -- this event was JUST re-armed onto a new
    # point (or is brand new), so any earlier "confirmed done" answer (for this object or
    # any other) no longer describes what `outstanding[-1]` is now. The next economizing
    # poll must peek fresh.
    _state.last_confirmed_done = None
    # PACE-49: a real record just happened -- whatever was accumulating toward the budget
    # since the last one is now moot; the next window starts clean.
    _state.running_cost_ms = 0.0

    _state.last_record_t = _time.perf_counter()


def _pace49_attribute(pool: dict, tail, call_site_id, call_site_anchor=None) -> None:
    """PACE-49: fold the real device-time interval between `pool["timed_prev"]` (this
    mechanism's own last attribution anchor) and *tail* (an event JUST confirmed complete by
    a peek's `query()` or a wait's `synchronize()` -- never called otherwise, so
    `elapsed_time()` is always safe to read here) into `pool["timed_site"]`'s cost-table
    entry, then advances the anchor to *tail*/`call_site_id` for the NEXT interval. Credits
    the WHOLE interval to `timed_site` alone (the call site whose own poll started it) rather
    than trying to split it among several call sites that may have run in between -- the
    simplest attribution that is never wrong in the case this ask's own acceptance shapes
    exercise (one call site polls repeatedly; nothing else's poll intervenes), and merely
    coarse, not incorrect, if something else did (that other call site's own polls get their
    own, later, correctly-anchored intervals once IT triggers a real record).

    FIX-PACE49 P2: `call_site_anchor` is threaded straight through to `_cost_feed` as the
    identity-check anchor (see its own docstring) — `pool["timed_anchor"]` remembers WHICH
    object `pool["timed_site"]` (the id()-derived key component) actually names, exactly
    like `timed_site` itself is remembered, so the credit below is anchored to the right
    object even if `call_site_id`'s own raw value has since been reused by something else.

    Silently skipped (never raises) if there is no anchor yet (`timed_prev is None`, the
    pool's first attribution point since the last `reset()`), the anchor IS *tail* itself
    (nothing elapsed to attribute, `depth<=1` can hand the same event to both roles), or
    `elapsed_time()` raises (an exotic event backend) -- losing one interval's reading is the
    honest choice `profile._drain_pending` already makes for the same class of failure,
    never a reason to raise out of a poll point."""
    prev, prev_site, prev_anchor = (pool.get("timed_prev"), pool.get("timed_site"),
                                    pool.get("timed_anchor"))
    if prev is not None and prev is not tail and prev_site is not None:
        try:
            ms = prev.elapsed_time(tail)
        except Exception:
            ms = None
        if ms is not None:
            _cost_feed((prev_site, _state.device_idx, _state.px_bucket), ms, prev_anchor)
    pool["timed_prev"] = tail
    pool["timed_site"] = call_site_id
    pool["timed_anchor"] = call_site_anchor


def _pace49_cost_gate(call_site_id, call_site_anchor=None) -> bool:
    """PACE-49: `True` while it is still safe to honour the stride/peek skip for this poll --
    either because the ms-budget dimension is disabled (`pace_budget_ms<=0`, the escape
    hatch), this call site has no estimate yet or is not yet WARM (`samples <
    _COST_WARMUP_SAMPLES` -- rides the registry-derived `heavy`/large-resolution rule
    instead, AUTHOR DECISION 4(a)), or its warm estimate, added to this cook's own running
    total since the pool's last real record, still fits under `pace_budget_ms`. Mutates
    `_state.running_cost_ms` as a side effect exactly when it returns `True` on a warm
    estimate -- the accumulation IS the point (several distinct cheap-alone call sites can
    still sum past budget before any one of them would trip it alone); `False` means the
    caller must fall through to a real record, which itself resets the running total to
    zero (see `paced_check`)."""
    budget = _state.budget_ms
    if budget <= 0:
        return True
    est = _cost_lookup((call_site_id, _state.device_idx, _state.px_bucket), call_site_anchor)
    if est is None:
        return True
    ewma_ms, samples = est
    if samples < _COST_WARMUP_SAMPLES:
        return True
    total = _state.running_cost_ms + ewma_ms
    if total > budget:
        return False
    _state.running_cost_ms = total
    return True


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
    first, so nothing inner-only survives the restore).

    P1 (Phase C, B1#1): `last_confirmed_done` is NEVER restored from the snapshot — it is
    always cleared instead. `save_state()` copies `_state.__dict__` shallowly: `pool` is
    the SAME dict object, not a copy, so a same-device NESTED cook (its own `reset()`
    hands this pool's then-outstanding event to `free`; its own first poll can pop and
    re-`record()` that very event onto a new point in the stream) can mutate the pool this
    snapshot points at before this function ever runs. Restoring the snapshot's own
    `last_confirmed_done` verbatim would then hand the outer cook a cache that still
    identity-matches an event object whose recorded point has moved since the snapshot was
    taken — the peek's `tail is _state.last_confirmed_done` check (`paced_check`) cannot
    tell that apart from a genuine, still-valid confirmation, and would skip recording on
    identity alone, without ever calling the real `query()` that would say the device has
    NOT reached the tail's new point. A just-ended nested cook makes no promise it left
    this pool's bookkeeping the way it found it, so nothing restored through it can be
    trusted as a confirmation of the CURRENT state — the cache is exactly that, an
    optimization: reading `None` here costs at most one extra `query()` call on the first
    poll after a restore, never a wrong answer, whether or not a nested cook actually ran."""
    _state.__dict__.clear()
    _state.__dict__.update(snapshot)
    _state.last_confirmed_done = None
    # PACE-49: same reasoning as `last_confirmed_done` just above -- a same-device nested
    # cook may have done its own real records against this SHARED pool in between, real
    # queued device work the outer's own pre-nesting running total knows nothing about.
    # Restoring it verbatim could UNDER-count what is really outstanding; starting the
    # window clean costs at most one extra warm-estimate addition on the outer's very next
    # poll, never a wrong (too-permissive) budget decision.
    _state.running_cost_ms = 0.0
    # FIX-PACE49 P3 (B3-pacing.md #2): the timing anchor (`pool["timed_prev"]`/
    # `pool["timed_site"]`/`pool["timed_anchor"]`) is exactly the same class of shared,
    # pool-resident state as `last_confirmed_done` above, for the identical reason -- it
    # lives on the SHARED pool dict, not in `_state.__dict__`, so a same-device nested
    # cook's own attribution cycle can have overwritten it with ITS OWN call site's
    # identity before this restore ever runs. Left verbatim, the outer's next real
    # interval would be credited to whatever call site the inner cook happened to leave
    # behind (confirmed: B3-pacing.md's own repro). Clearing it costs at most one interval
    # of lost attribution on the outer's own next confirm (exactly like a fresh `reset()`'s
    # first poll never has a prior anchor either) -- never a wrong credit.
    pool = getattr(_state, "pool", None)
    if pool is not None:
        pool["timed_prev"] = None
        pool["timed_site"] = None
        pool["timed_anchor"] = None
