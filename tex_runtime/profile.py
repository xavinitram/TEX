"""
PROF-1 — the per-stage cost profiler.

**Why this exists at all.** "Effort-based caching" and "admit by predicted cost" both need a
number, and verification found the engine does not have one. Q-4 is fused-chain *error*
attribution (a stage-tagged SourceLoc), not cost. `autotier.cook_ms` is the only shipped cost
signal, and it is (a) whole-program, (b) only ever fed on the `compile_mode="auto"` path — the
DEFAULT ComfyUI cook (`compile_mode="none"` → the interpreter) records nothing. So without
PROF-1, "effort-based" has no measured effort and PRED-1's admission has nothing to admit on.

**What it stores.** An EWMA of cook cost, keyed by (program fingerprint, device type,
precision) and bucketed by resolution, plus a per-STAGE breakdown for fused programs. Two
consumers: PRED-1's admission (v0.31, this release) and CACHE-7's checkpoint placement
(v0.32 — where to cut a chain is a question about cumulative *stage* cost, which is why the
per-stage half is here and not deferred).

**Why an EWMA rather than autotier's median deque.** autotier is deciding a one-way verdict
(is the compiled tier faster?) and wants outlier resistance. PROF-1 is answering "how long
will this take *next* time" for a host whose resolution, canvas and hardware load all drift;
recency matters more than robustness, and an EWMA carries no per-key deque. The decay is over
SAMPLES, not wall time, deliberately: a program nobody has cooked for an hour has not become
slower, and time-decaying its estimate towards nothing would make PRED-1 mis-admit it.

**Sampling is the whole reason it can be armed at all.** A per-stage timer on CUDA needs
`torch.cuda.synchronize()` at every boundary (the standing benchmark rule — without it you
time kernel *launches*), and that sync is exactly the stall the profiler must not introduce.
So: the first `_WARMUP_SAMPLES` cooks of an unseen key are measured (a cold key gets a usable
number immediately), and after that one cook in `_SAMPLE_EVERY` is. A steady-state interactive
session therefore pays the sync on ~6% of cooks and reads a fresh number on all of them.

**INVARIANT #7 applies to the profiler itself** (doc 39 §8 says so). It is DISABLED by
default. Disabled, the engine's whole cost is one module-global boolean load and a branch,
once per cook and once per interpreter `execute` — no timers, no syncs, no dict traffic. A
host arms the in-engine sampler explicitly with `enable()`.

`CookQueue` deliberately does NOT call `enable()`. It already brackets every job it runs, so
it feeds `record()` directly from that bracket — which costs nothing, needs no sampling gate,
and cannot put a CUDA sync into a cook the queue does not own. The in-engine sampler is only
for the per-STAGE breakdown and for hosts cooking outside the queue.

NOT persisted across processes, deliberately. autotier persists because re-deriving a
compile verdict costs a background compile; a PROF-1 estimate costs `_WARMUP_SAMPLES` cooks
the host was going to run anyway. CACHE-7 may want cross-launch placement stability — that
is its design doc's call; `snapshot()` is the seam it would persist through.
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

# ── policy constants (explicit numbers a test can feed, per autotier's discipline) ──
_ALPHA = 0.35             # EWMA weight on the newest sample
_WARMUP_SAMPLES = 3       # measure every cook of an unseen key until it has this many
_SAMPLE_EVERY = 16        # then measure one cook in N
_STATE_MAX = 512          # bound the table (LRU), same order as autotier's
_PENDING_MAX = 64         # PROF-462: bounded lazy-fold queue (below)

_enabled = False


@dataclass
class _Bucket:
    """The cost of one program at one resolution bucket."""
    px: int                             # representative pixel count (the last one seen)
    ewma_ms: float = 0.0
    samples: int = 0
    stages: dict = field(default_factory=dict)   # stage index -> EWMA ms
    skips: int = 0

    def feed(self, ms: float) -> None:
        self.samples += 1
        self.ewma_ms = _blend(self.ewma_ms, ms, self.samples)

    def feed_stage(self, stage, ms: float) -> None:
        # Stage counts ride the bucket's own sample count: they are fed from the same cook,
        # so a separate per-stage counter could only ever drift from it.
        self.stages[stage] = _blend(self.stages.get(stage, 0.0), ms, self.samples)


def _blend(prev: float, ms: float, n: int) -> float:
    """Fold sample #`n` into a running estimate: a MEAN while the key is young, an EWMA once
    it is established.

    `alpha = max(_ALPHA, 1/n)` is the whole rule, and the early half of it is load-bearing.
    A key's FIRST cook is cold by construction — it pays the compile, the allocator growth and
    the first-touch of every cached kernel — and it is routinely 3-10x the steady state
    (measured: a 3-stage fused chain at 256^2 came in at 7.4/6.4/0.6 ms cold and
    0.68/2.27/0.62 ms warm). A fixed alpha anchored on that outlier needs dozens of samples to
    shed it, and at the 1-in-16 sampling rate that is hundreds of cooks — long enough that
    PRED-1 would rank a cheap stage above an expensive one for most of a session.

    A running mean instead means sample 2 is worth half and sample 3 a third, so the cold
    reading is diluted immediately; from sample 3 on `_ALPHA` takes over and the estimate
    tracks recent cooks, which is what an interactive host's drifting resolution needs."""
    if n <= 1:
        return ms
    return max(_ALPHA, 1.0 / n) * ms + (1.0 - max(_ALPHA, 1.0 / n)) * prev


#: {(program_fp, device_type, precision): {px_bucket: _Bucket}}
_STATE: "OrderedDict[tuple, dict]" = OrderedDict()

#: Guards `_STATE` and every `_Bucket` inside it. There are genuinely two mutators in the
#: SHIPPED configuration — `tex_engine.run` on the cook thread when the sampler is armed, and
#: `tex_cookqueue`'s worker feeding job timings even while it is DISARMED — so this is not a
#: hypothetical. Without it, `snapshot()` iterating while the worker inserts a key raises
#: `RuntimeError: OrderedDict mutated during iteration` (reproduced 3/3, 41-68 hits per 4 s).
#:
#: `enabled()` stays OUTSIDE the lock, deliberately: it is the one thing the default cook path
#: touches, and putting a lock acquisition there would tax every ComfyUI cook to protect a
#: table that cook never writes (invariant #7). The critical sections below are all a few dict
#: operations, so an uncontended acquire is the whole cost.
#: PROF-462: an `RLock`, not a plain `Lock` — `_drain_pending_locked` calls the public
#: `record`/`record_stages` (deliberately, so a host or benchmark counting calls to those two
#: names — `docs/host-path-counts.md`'s BENCH-2 harness does exactly this — still sees a fold
#: happen, however lazily) from inside a block that already holds this lock; a non-reentrant
#: lock would deadlock the very first sampled cook.
_LOCK = threading.RLock()


# ── arming ───────────────────────────────────────────────────────────────────
def enable() -> None:
    """Arm the in-engine sampler process-wide. A host calls this; nothing in TEX does by
    default — not even `CookQueue` — which is what keeps invariant #7 true."""
    global _enabled
    _enabled = True


def disable() -> None:
    global _enabled
    _enabled = False


def enabled() -> bool:
    """The one branch the default cook path pays. Kept a plain function over a module global
    so the engine's call site reads as intent rather than as a poke at a private."""
    return _enabled


# ── the per-stage sink (thread-local, on the tier_trace model) ───────────────
# Thread-local because ENG-9 gives every cook thread its own interpreter and the cook queue
# runs on a worker: a process-wide sink would mix two threads' stages into one program's
# breakdown. Carried out-of-band rather than as an interpreter parameter for the same reason
# tier_trace is: it keeps `execute()`'s signature — a surface with several external callers —
# out of the profiler's business.
_tls = threading.local()


def stage_sink() -> dict | None:
    """The dict the interpreter accumulates per-stage ms into for THIS cook on THIS thread,
    or None. Only ever read behind `enabled()`.

    CUDA sampled cooks do NOT feed ms into this dict (see `stage_event_sink` below) — it
    stays the CPU path's own synchronous accumulator, and `measure.__exit__` only calls
    `record_stages` from it when no event-mode start event exists (CPU, or CUDA event
    creation failed and `measure` fell back to the old synchronize+perf_counter form)."""
    return getattr(_tls, "stages", None)


def record_stage_boundary(events: list, stage, device) -> None:
    """Append one `(stage, event)` boundary to `events` (from `stage_event_sink()`), recording
    a fresh timing-enabled CUDA event under `with torch.cuda.device(device):` (the O4
    discipline `measure._new_event` also follows, matching `pacing.cook_done_event`) and never
    blocking. Lives here, not inline in the interpreter, so `Interpreter._exec_stmts_profiled`
    stays a few lines shorter — the only caller.

    On failure this appends `(stage, None)` rather than nothing. The fold walks `events`
    SEQUENTIALLY, using each entry as the previous one's boundary — silently dropping a failed
    boundary (the old behaviour) does not lose just that stage's attribution, it hands stage
    K's real cost to whichever stage happens to close next, which is wrong in a different way
    than "missing". A `None` in the list tells the fold this sample's per-stage breakdown is
    unreliable, so it can fall back to recording the sample's WHOLE-COOK time only — still
    losing this one sample's stage split, never the cook, and never someone else's number."""
    try:
        import torch
        with torch.cuda.device(device):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
        events.append((stage, ev))
    except Exception:
        events.append((stage, None))


def stage_event_sink() -> list | None:
    """The list `Interpreter._exec_stmts_profiled` appends `(stage, event)` pairs into for a
    CUDA-sampled cook with per-stage tracking, or None off that path (CPU, or the profiler is
    not currently inside a `measure` block that got a start event). Populated by
    `measure.__enter__` alongside `stage_sink`'s dict, on the same thread-local discipline
    (ENG-9: one interpreter per cook thread). PROF-462."""
    return getattr(_tls, "stage_events", None)


@contextlib.contextmanager
def suspend_stage_tracking():
    """Suspend this THREAD's per-stage sink/event-list for the duration of a block, restoring
    exactly what was there on exit.

    `stage_sink()`/`stage_event_sink()` are scoped to the OS thread, not to whichever `measure`
    call is logically "in progress" on it. `GraphedProgram.capture()` runs its warmup and
    capture passes through a SEPARATE `Interpreter.execute()` call, on the SAME thread, while
    the outer cook's own `measure(stages=True)` is still open — that nested `execute()` reads
    the same ambient thread-local and (because the profiler has no idea it is not "the" cook
    being measured) appends its own boundaries into the outer cook's sample, corrupting the
    per-stage breakdown that seeds a new key's EWMA at full weight. Pre-existing: the same
    thread-local aliasing already mixed a nested execute's boundaries into the outer sink
    before this module grew CUDA-event recording, it just wrote wall-clock ms into a shared
    dict instead of events into a shared list.

    Call this around any nested `Interpreter.execute()` that is not itself part of the cook a
    `measure` block is timing (today: capture's warmup + graph-capture passes). It is not on
    any per-statement or per-cook hot path, so cost here is not a concern."""
    prev_stages = getattr(_tls, "stages", None)
    prev_events = getattr(_tls, "stage_events", None)
    _tls.stages = None
    _tls.stage_events = None
    try:
        yield
    finally:
        _tls.stages = prev_stages
        _tls.stage_events = prev_events


# ── keys ─────────────────────────────────────────────────────────────────────
def make_key(program_fp, device_type: str, precision: str) -> tuple:
    """The program axis of a cost. Resolution is NOT in here — it is the bucket dimension,
    because predicting an unseen resolution from a seen one is the whole point (§predict)."""
    return (program_fp, device_type, precision)


def bucket_of(spatial) -> tuple:
    """(bucket, px) for a (B, H, W) shape OR a bare pixel count. Bucketed by `px.bit_length()`
    — the same octave granularity autotier commits verdicts at, so a session at jittering
    resolutions (a folder of near-1000px photos, a zoom drag within one octave) keeps landing
    in one bucket instead of never accumulating samples.

    The int form is what the engine passes: `CookPlan.cook_px` is already H*W of the first
    spatial binding, scanned once per cook, and re-deriving it from shapes here would repeat
    an O(#bindings) walk the plan exists to avoid."""
    px = 1
    if isinstance(spatial, int):
        px = max(1, spatial)
    elif spatial:
        try:
            _b, h, w = spatial
            px = max(1, int(h) * int(w))
        except Exception:
            px = 1
    return px.bit_length(), px


def _buckets(key: tuple, *, create: bool) -> dict | None:
    b = _STATE.get(key)
    if b is None:
        if not create:
            return None
        b = {}
        _STATE[key] = b
        while len(_STATE) > _STATE_MAX:
            _STATE.popitem(last=False)
    else:
        _STATE.move_to_end(key)
    return b


# ── the sampling gate ────────────────────────────────────────────────────────
def should_sample(key: tuple, spatial=None) -> bool:
    """Should THIS cook be measured? Warmup cooks of an unseen key always are; after that one
    in `_SAMPLE_EVERY`.

    Mutates (it advances the skip counter), which is why it is `should_sample` and not a
    property: it is a rate limiter, and a caller that asks twice for one cook would double-count.
    Returns False immediately when disarmed, so a caller can use it as the only gate.

    The warmup check counts `.samples` (which only advances at FOLD time — lazy since
    PROF-462) PLUS the still-in-flight `_pending` entries for this same (key, bucket): without
    the latter, a burst of same-key cooks arriving faster than their device events resolve
    would see `.samples == 0` on every one of them and sample the whole burst, since none of
    their predecessors' folds have landed yet. Counting in-flight samples toward the budget
    keeps warmup bounded at `_WARMUP_SAMPLES` regardless of how fast the burst outpaces the
    device — it never inflates the recorded `.samples` count itself, only this gate's view of
    how many are already "spoken for"."""
    if not _enabled:
        return False                     # the default path never reaches the lock
    with _LOCK:
        _drain_pending_locked()          # PROF-462: fold whatever device work has landed
        bkt, px = bucket_of(spatial)
        buckets = _buckets(key, create=True)
        st = buckets.get(bkt)
        if st is None:
            buckets[bkt] = _Bucket(px=px)
            return True
        in_flight = sum(1 for p in _pending
                        if p.key == key and bucket_of(p.spatial)[0] == bkt)
        if st.samples + in_flight < _WARMUP_SAMPLES:
            return True
        st.skips += 1
        if st.skips >= _SAMPLE_EVERY:
            st.skips = 0
            return True
        return False


# ── recording ────────────────────────────────────────────────────────────────
def record(key: tuple, ms: float, spatial=None) -> None:
    """Feed one WHOLE-COOK measurement.

    Records even when disarmed — a caller that already paid for a timing (the cook queue,
    which brackets every job anyway) should not have its measurement thrown away because the
    in-engine sampler happens to be off. The GATE is `should_sample`, not this."""
    if ms is None or ms < 0:
        return
    with _LOCK:
        _bucket(key, spatial).feed(float(ms))


def record_stages(key: tuple, stages: dict, spatial=None) -> None:
    """Feed one PER-STAGE breakdown. Deliberately does NOT also feed the whole-cook EWMA from
    `sum(stages)`: `measure` owns that number, the two nest around the same cook, and adding
    both would count every profiled cook twice."""
    if not stages:
        return
    with _LOCK:
        st = _bucket(key, spatial)
        for idx, sms in stages.items():
            st.feed_stage(idx, float(sms))


def _bucket(key: tuple, spatial) -> _Bucket:
    bkt, px = bucket_of(spatial)
    buckets = _buckets(key, create=True)
    st = buckets.get(bkt)
    if st is None:
        st = buckets[bkt] = _Bucket(px=px)
    st.px = px
    return st


# ── PROF-462: lazy, device-honest sampling for CUDA cooks ────────────────────
# `measure` used to fence a sampled CUDA cook with `torch.cuda.synchronize()` at entry and
# exit (plus two more inside `Interpreter._exec_stmts_profiled`, one per stage boundary) —
# the "four device barriers" this reopens (CHANGELOG "The profiler's four device barriers
# per sampled cook are declined ... What would reopen it is device events read once per
# cook: a different mechanism, not a tuning of this one."). A `synchronize()` stalls the
# calling thread until the device drains, which is exactly the cost an interactive host
# cannot afford to put on a background cook's poll loop (see `tex_runtime/pacing.py`).
#
# The replacement: `measure` records one timing-enabled event at entry and one at exit
# (never touching `CookResult.done`, which stays untimed — PACE-45's identity and cost are
# unchanged), and the interpreter records one more per stage boundary (`profile.stage_event_
# sink()`). None of these calls block. The pair is queued here and folded into the EWMA
# tables the next time anything reads or samples the table — `should_sample`, `predict`,
# `stage_costs`, `samples`, `settled`, `stage_snapshot` and `snapshot` all drain first — by
# checking only the LAST event's `query()`: CUDA completes events on one stream in the order
# they were recorded, so a signalled last event means every earlier event on that same cook
# is safe to read with `elapsed_time` too (invariant #6 still holds — a reading is only ever
# taken after its event has completed; deferring changes WHEN, never WHETHER).
@dataclass
class _Pending:
    key: tuple
    spatial: object
    start: "object"                 # torch.cuda.Event, timing-enabled
    end: "object"                   # torch.cuda.Event, timing-enabled
    stage_events: "list | None"     # [(stage, event)] in recorded order, or None


_pending: "deque[_Pending]" = deque()


def _drain_pending_locked() -> None:
    """Fold every queued sample whose device work has completed into the tables. Caller
    holds `_LOCK` (an `RLock` — see there): this calls the PUBLIC `record`/`record_stages`,
    not `_bucket(...).feed(...)` directly, so a host or benchmark that counts calls to those
    two names (`docs/host-path-counts.md`'s BENCH-2 harness) still sees the fold happen,
    lazily, exactly once per sample — the same contract those functions have always kept,
    just no longer paid for with a device barrier. NEVER calls `.synchronize()` — a sample
    whose end event has not yet signalled is left queued for the next drain, exactly like
    `_timed_deferred`'s (LAT-3, `tex_runtime/compiled.py`) same-shaped deferral."""
    if not _pending:
        return
    keep = deque()
    for samp in _pending:
        try:
            done = samp.end.query()
        except Exception:
            done = True    # a dead/invalidated event can never complete; drop it, not wait
        if not done:
            keep.append(samp)
            continue
        try:
            whole_ms = samp.start.elapsed_time(samp.end)
            record(samp.key, whole_ms, samp.spatial)
            if samp.stage_events:
                # A `None` event (`record_stage_boundary`'s failure marker) means one of
                # this sample's own boundaries never recorded — the sequential fold below
                # cannot isolate that stage's interval, so it would silently hand its cost
                # to a neighbour instead. Whole-cook time is already recorded above; drop
                # the per-stage split for this ONE sample rather than corrupt a stage.
                if not any(ev is None for _, ev in samp.stage_events):
                    prev = samp.start
                    stages = {}
                    for stage, ev in samp.stage_events:
                        stages[stage] = prev.elapsed_time(ev)
                        prev = ev
                    record_stages(samp.key, stages, samp.spatial)
        except Exception:
            pass            # a readback failure loses one sample; never raises to a caller
    _pending.clear()
    _pending.extend(keep)


def _queue_pending(key: tuple, spatial, start, end, stage_events) -> None:
    """Queue one sampled cook's device-timing events for lazy fold (see above). Bounded:
    the oldest entry is dropped, unresolved or not, once the queue would exceed
    `_PENDING_MAX` — a profiler that never blocks the cook path must also never grow
    without bound if a consumer stops reading the table."""
    with _LOCK:
        _drain_pending_locked()
        _pending.append(_Pending(key, spatial, start, end, stage_events))
        while len(_pending) > _PENDING_MAX:
            _pending.popleft()


def _pending_count() -> int:
    """Test hook: how many samples are queued for lazy fold, without draining them."""
    with _LOCK:
        return len(_pending)


# ── prediction ───────────────────────────────────────────────────────────────
def _resolve_bucket(key: tuple, spatial, *, need_stages: bool = False):
    """`(bucket, pixel_scale)` for this (key, resolution), or `(None, 1.0)`.

    An exact bucket hit scales by 1. A MISS falls back to the nearest measured bucket scaled by
    the pixel ratio — the fallback is what makes the profiler useful to a host asking about a
    frame at a resolution the session has not cooked yet.

    ONE resolver rather than one per accessor, because the alternative already produced a live
    bug: `stage_costs` grew this fallback and `samples` did not, so a caller asking both — which
    is exactly what CACHE-7's planner does — got good scaled costs together with a sample count
    of ZERO, refused to place, and the fallback was unreachable through the only consumer that
    wanted it. An estimate and the confidence in that estimate must answer for the SAME bucket,
    and the only way to guarantee that is to select the bucket once.

    HONEST APPROXIMATION: linear-in-pixels over-predicts small frames, because a cook has a
    fixed cost (dispatch, binding marshalling, the Python walk) that does not shrink with the
    frame — at 64² a TEX cook is almost entirely that fixed part. For ORDERING work the bias
    cancels across candidates; against an ABSOLUTE threshold it does not, which is why CACHE-7
    also checks a materialization floor. Caller holds `_LOCK`."""
    _drain_pending_locked()   # PROF-462: every reader sees device-honest, already-folded data
    buckets = _buckets(key, create=False)
    if not buckets:
        return None, 1.0
    bkt, px = bucket_of(spatial)
    st = buckets.get(bkt)
    if st is not None and st.samples and (st.stages or not need_stages):
        return st, 1.0
    # Nearest measured OCTAVE — the dict is keyed by `px.bit_length()`, so compare the keys
    # rather than re-deriving them from each value.
    _, best = min(((b_key, b) for b_key, b in buckets.items()
                   if b.samples and (b.stages or not need_stages)),
                  key=lambda kv: abs(kv[0] - bkt), default=(None, None))
    if best is None:
        return None, 1.0
    return best, ((px / best.px) if best.px else 1.0)


def predict(key: tuple, spatial=None) -> float | None:
    """Expected cook cost in ms, or None if this program has never been measured on that
    (device, precision). See `_resolve_bucket` for the cross-resolution fallback and its
    honest approximation."""
    with _LOCK:
        best, scale = _resolve_bucket(key, spatial)
        return None if best is None else best.ewma_ms * scale


def stage_costs(key: tuple, spatial=None) -> dict:
    """{stage_index: EWMA ms} for a fused program, or {} if never measured per stage. This is
    CACHE-7's input: a checkpoint goes where the CUMULATIVE cost crosses its threshold."""
    with _LOCK:
        best, scale = _resolve_bucket(key, spatial, need_stages=True)
        return {} if best is None else {k: v * scale for k, v in best.stages.items()}


def samples(key: tuple, spatial=None, *, need_stages: bool = False) -> int:
    """How many cooks back the estimate for this (key, resolution) — INCLUDING one served by
    the cross-bucket fallback, so a caller reading `stage_costs` and `samples` together is
    told about the same bucket.

    `need_stages` MUST match what the caller actually read. `_resolve_bucket` skips buckets
    with no per-stage breakdown when it is True, so the two flags select DIFFERENT buckets on
    the same key: a resolution measured whole-cook-only but never per-stage answers `samples`
    generously while `stage_costs` falls back to a distant bucket. See `stage_snapshot`."""
    with _LOCK:
        best, _ = _resolve_bucket(key, spatial, need_stages=need_stages)
        return best.samples if best is not None else 0


def settled(key: tuple, spatial=None, *, need: int = 12,
            need_stages: bool = False) -> bool:
    """Is this estimate old enough to make an irreversible decision on?

    PROF-1 owns the answer because PROF-1 owns the schedule that determines it: `_blend`'s
    `max(_ALPHA, 1/n)` rule, `_WARMUP_SAMPLES`, and `_SAMPLE_EVERY` are all private here, and a
    consumer hard-coding a threshold against them goes silently stale when they change.

    `need` defaults to the measured settling point: on a 3-stage chain whose stage 0 is a
    multiply and stage 1 a blur, 3 samples attribute stage 0 at ~10x its truth and INVERT the
    ranking; by 12 the estimate matches standalone cooks closely. At 3 warmup cooks plus 1-in-16
    sampling, 12 samples is roughly 150 cooks.

    Prefer `stage_snapshot` when you want costs AND confidence — it cannot disagree with itself."""
    return samples(key, spatial, need_stages=need_stages) >= int(need)


def stage_snapshot(key: tuple, spatial=None, *, need: int = 12) -> tuple:
    """`(stage_costs, settled)` resolved from ONE bucket selection.

    The reason this exists rather than two calls: asking `stage_costs` and `settled`
    separately reads the table TWICE and can land on two different buckets, because only the
    first filters to buckets that actually carry a per-stage breakdown. That is not
    hypothetical — an earlier fix gave `stage_costs` a cross-bucket fallback and left
    `samples` without one, so a planner received good scaled costs together with a sample
    count of zero and refused to place. Re-splitting it re-creates the same bug in the
    opposite direction: costs from a distant bucket, confidence from a near one, and a
    placement made on numbers whose trustworthiness was measured somewhere else."""
    with _LOCK:
        best, scale = _resolve_bucket(key, spatial, need_stages=True)
        if best is None:
            return {}, False
        return ({k: v * scale for k, v in best.stages.items()},
                best.samples >= int(need))


# ── lifecycle / introspection ────────────────────────────────────────────────
def reset() -> None:
    """Forget everything (a test hook, and what a host calls between projects)."""
    with _LOCK:
        _STATE.clear()
        _pending.clear()          # PROF-462: a stale pending sample must not outlive a reset


def snapshot() -> dict:
    """A JSON-able view: {"fp|device|precision": {bucket: {...}}}. The seam CACHE-7 would
    persist through, and what a host HUD reads."""
    out = {}
    with _LOCK:                          # a concurrent insert would raise mid-iteration
        _drain_pending_locked()          # PROF-462: a snapshot reads whatever has folded
        for (fp, dev, prec), buckets in _STATE.items():
            out[f"{fp}|{dev}|{prec}"] = {
                str(bkt): {"px": b.px, "ms": round(b.ewma_ms, 4), "samples": b.samples,
                           "stages": {str(k): round(v, 4) for k, v in b.stages.items()}}
                for bkt, b in buckets.items()}
    return out


class measure:
    """Time a cook and feed it to `key`, syncing CUDA around the region — optionally with the
    per-STAGE breakdown too.

    The explicit surface for a host cooking OUTSIDE the queue (`tex_engine.cook` directly), and
    the same object the engine's own hook uses. A no-op body when `should_sample` says no, so a
    caller may wrap every cook unconditionally:

        with profile.measure(key, spatial, device="cuda", stages=True):
            tex_engine.cook(...)

    ONE object rather than a timer plus a separate sink-armer, because the sampling decision
    has to be shared: `should_sample` advances a skip counter, so two objects asking it about
    the same cook would both double-count the rate and disagree with each other. Here there is
    one gate and nothing to keep in sync.

    The sink is re-entrant by save/restore rather than by clearing: the OOM ladder and the
    tiled paths call `execute()` repeatedly inside one cook, and an inner block that reset the
    sink to None would silently drop the outer cook's breakdown.

    THE REPRO METHOD, for whoever next times a change to `_sync`'s count here or in
    `Interpreter._exec_stmts_profiled` (TRK-131): a "no difference" reading is only evidence
    of safety if the benchmarked program's stdlib pool contains NO `sync=True` builtin.
    Such a builtin does its own internal host readback (an `.item()`), which is itself a
    device-completion barrier — so a repro built on one (e.g. `gauss_blur`) can read nearly
    identical per-stage numbers with a profiler sync removed not because removing it was
    safe, but because the builtin's own readback was silently supplying the barrier the
    profiler had stopped supplying. Check every member of a timing pool against
    `stdlib_registry.REGISTRY`'s `.sync` field (`e.sync` per entry — the same tag
    `graphed._SYNC_STDLIB` is hand-kept from) BEFORE quoting a "no difference" result from
    it; a pool with a sync-tagged member proves nothing about this mechanism either way.

    PROF-462: on CUDA this no longer calls `torch.cuda.synchronize()` at all. A timing-
    enabled event is recorded at entry and one more at exit (never `CookResult.done`, which
    PACE-45 creates without `enable_timing` and which this leaves untouched in identity and
    cost); the pair — plus, when `stages=True`, the per-stage boundary events
    `Interpreter._exec_stmts_profiled` records via `stage_event_sink()` — is handed to
    `_queue_pending` and read back lazily, never here. If event creation fails (no CUDA
    context available despite `device` claiming one), this falls back to the old
    synchronize-and-perf_counter form for that one cook, same as before PROF-462."""
    __slots__ = ("key", "spatial", "device", "sink", "_t0", "_on", "_prev",
                 "_start_ev", "_prev_events")

    def __init__(self, key: tuple, spatial=None, *, device=None, stages: bool = False):
        self.key = key
        self.spatial = spatial
        self.device = str(device or "")
        self._on = should_sample(key, spatial)
        self.sink: dict | None = {} if (self._on and stages) else None
        self._prev = None
        self._t0 = 0.0
        self._start_ev = None
        self._prev_events = None

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            try:
                import torch
                torch.cuda.synchronize()
            except Exception:
                pass

    def _new_event(self):
        """A `torch.cuda.Event(enable_timing=True)`, recorded now under `with torch.cuda.
        device(self.device):` — the O4 discipline (`pacing.cook_done_event`) that makes the
        event mark THIS measure's device rather than whatever happens to be ambient. None on
        any failure (no CUDA context / bad device); the caller falls back to the sync form."""
        if not self.device.startswith("cuda"):
            return None
        try:
            import torch
            with torch.cuda.device(self.device):
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
            return ev
        except Exception:
            return None

    def __enter__(self) -> "measure":
        if self._on:
            if self.sink is not None:
                self._prev = getattr(_tls, "stages", None)
                _tls.stages = self.sink
            self._start_ev = self._new_event()
            if self._start_ev is not None and self.sink is not None:
                self._prev_events = getattr(_tls, "stage_events", None)
                _tls.stage_events = []
            if self._start_ev is None:
                self._sync()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._on:
            return False
        if self.sink is not None:
            _tls.stages = self._prev
        events = None
        if self._start_ev is not None:
            events = getattr(_tls, "stage_events", None)
            _tls.stage_events = self._prev_events
        # A cook that raised (OOM, CookCancelled) took an unrepresentative amount of time —
        # recording it would poison the EWMA with a number no future cook will reproduce.
        if exc_type is None:
            if self._start_ev is not None:
                end_ev = self._new_event()
                if end_ev is not None:
                    try:
                        _queue_pending(self.key, self.spatial, self._start_ev, end_ev, events)
                        return False
                    except Exception:
                        pass
                # The end event broke after a successful start (very rare — no CUDA context
                # loss short of that). `self._t0` predates the entry sync this cook never
                # paid, so a wall-clock fallback here would understate cost; losing this one
                # sample is the honest choice, matching `_timed_deferred`'s own "skip it".
                return False
            self._sync()
            record(self.key, (time.perf_counter() - self._t0) * 1000.0, self.spatial)
            if self.sink:
                record_stages(self.key, self.sink, self.spatial)
        return False
