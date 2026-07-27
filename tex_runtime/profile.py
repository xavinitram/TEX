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

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

# ── policy constants (explicit numbers a test can feed, per autotier's discipline) ──
_ALPHA = 0.35             # EWMA weight on the newest sample
_WARMUP_SAMPLES = 3       # measure every cook of an unseen key until it has this many
_SAMPLE_EVERY = 16        # then measure one cook in N
_STATE_MAX = 512          # bound the table (LRU), same order as autotier's

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
_LOCK = threading.Lock()


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
    or None. Only ever read behind `enabled()`."""
    return getattr(_tls, "stages", None)


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
    Returns False immediately when disarmed, so a caller can use it as the only gate."""
    if not _enabled:
        return False                     # the default path never reaches the lock
    with _LOCK:
        bkt, px = bucket_of(spatial)
        buckets = _buckets(key, create=True)
        st = buckets.get(bkt)
        if st is None:
            buckets[bkt] = _Bucket(px=px)
            return True
        if st.samples < _WARMUP_SAMPLES:
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


# ── prediction ───────────────────────────────────────────────────────────────
def predict(key: tuple, spatial=None) -> float | None:
    """Expected cook cost in ms, or None if this program has never been measured on that
    (device, precision).

    An exact bucket hit returns its EWMA. A MISS falls back to the nearest measured bucket
    scaled by the pixel ratio — that fallback is what makes the profiler useful to PRED-1,
    which is usually asked about a frame at a resolution the session has not cooked yet.

    HONEST APPROXIMATION: linear-in-pixels over-predicts small frames, because a cook has a
    fixed cost (dispatch, binding marshalling, the Python walk) that does not shrink with the
    frame — at 64² a TEX cook is almost entirely that fixed part. It is used for ORDERING
    speculative work, where a consistent bias across candidates cancels, and never as a
    deadline. A second measured bucket would let this fit a slope+intercept instead; that is
    left for CACHE-7, whose placement decision is the one that would actually be wrong."""
    with _LOCK:
        buckets = _buckets(key, create=False)
        if not buckets:
            return None
        bkt, px = bucket_of(spatial)
        st = buckets.get(bkt)
        if st is not None and st.samples:
            return st.ewma_ms
        best = min((b for b in buckets.values() if b.samples),
                   key=lambda b: abs(b.px.bit_length() - bkt), default=None)
        if best is None:
            return None
        return best.ewma_ms * (px / best.px) if best.px else best.ewma_ms


def stage_costs(key: tuple, spatial=None) -> dict:
    """{stage_index: EWMA ms} for a fused program, or {} if never measured per stage. This is
    CACHE-7's input: a checkpoint goes where the CUMULATIVE cost crosses its threshold."""
    with _LOCK:
        buckets = _buckets(key, create=False)
        if not buckets:
            return {}
        st = buckets.get(bucket_of(spatial)[0])
        return dict(st.stages) if st is not None else {}


def samples(key: tuple, spatial=None) -> int:
    with _LOCK:
        buckets = _buckets(key, create=False)
        if not buckets:
            return 0
        st = buckets.get(bucket_of(spatial)[0])
        return st.samples if st is not None else 0


# ── lifecycle / introspection ────────────────────────────────────────────────
def reset() -> None:
    """Forget everything (a test hook, and what a host calls between projects)."""
    with _LOCK:
        _STATE.clear()


def snapshot() -> dict:
    """A JSON-able view: {"fp|device|precision": {bucket: {...}}}. The seam CACHE-7 would
    persist through, and what a host HUD reads."""
    out = {}
    with _LOCK:                          # a concurrent insert would raise mid-iteration
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
    sink to None would silently drop the outer cook's breakdown."""
    __slots__ = ("key", "spatial", "device", "sink", "_t0", "_on", "_prev")

    def __init__(self, key: tuple, spatial=None, *, device=None, stages: bool = False):
        self.key = key
        self.spatial = spatial
        self.device = str(device or "")
        self._on = should_sample(key, spatial)
        self.sink: dict | None = {} if (self._on and stages) else None
        self._prev = None
        self._t0 = 0.0

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            try:
                import torch
                torch.cuda.synchronize()
            except Exception:
                pass

    def __enter__(self) -> "measure":
        if self._on:
            if self.sink is not None:
                self._prev = getattr(_tls, "stages", None)
                _tls.stages = self.sink
            self._sync()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._on:
            return False
        if self.sink is not None:
            _tls.stages = self._prev
        # A cook that raised (OOM, CookCancelled) took an unrepresentative amount of time —
        # recording it would poison the EWMA with a number no future cook will reproduce.
        if exc_type is None:
            self._sync()
            record(self.key, (time.perf_counter() - self._t0) * 1000.0, self.spatial)
            if self.sink:
                record_stages(self.key, self.sink, self.spatial)
        return False
