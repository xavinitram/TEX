"""
DATA-7 — the host source protocol, and the governed pool behind it.

Design doc: `docs/frame-providers.md`. Two sentences of it are load-bearing here, so this
file reads alone:

  * **TEX never opens a file.** A `FrameProvider` is a host object that answers
    "give me `source_key` at time `t`" with a tensor. Decoding, colour, containers, frame
    rates and disk are all on the host's side of the seam. What TEX owns is temporal
    identity (which two times are the same frame), the cache, and the governor's view of
    the bytes.
  * **A provider is a pure function of `(source_key, quantized_t)`** for as long as its
    source version does not change. TEX never stats a file and never watches a directory;
    the host bumps the version when it re-exports, because the host is the one that knows.

The module is shaped after `tex_runtime/host.py` on purpose — a `Protocol`, a `Null`
default that is a real implementation rather than a stub, a process-wide memo, and a
setter for tests and for a non-ComfyUI host. A reader who knows `get_host_services()`
knows `get_provider()`. It is a separate module rather than a second protocol inside
`host.py` because `host.py` is the PORT-1 boundary — the one place `comfy.model_management`
is imported, grep-pinned forever — and a frame cache in that file makes the lint's subject
ambiguous. The name is `docs/roadmap.md` §9's own pencil for the v0.34 row.

INVARIANT #7: nothing here is constructed until a host registers a provider. The pool
registers into the governor at `CacheRegistry` construction (like the CUDA-graph pool) and
reports 0 bytes forever if no provider is ever armed, so an unarmed engine pays a dict
lookup in `stats()` and nothing else.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

import torch

# ── E7xxx: the host-I/O error family (R8) ────────────────────────────────────
#
# Severity is CLASS-dependent and a provider does not know its job's class, so nothing here
# decides how loud a failure is: these raise, and `tex_cookqueue` decides whether a raise
# reaches the host or dies into the refusals ledger (a SPECULATIVE prefetch that alarms is
# worse than one that quietly did not happen).

E_NO_PROVIDER = "E7001"      # no frame provider is registered
E_FETCH_FAILED = "E7002"     # the provider raised
E_NONUNIFORM_TIME = "E7003"  # a per-pixel / per-batch-element `t`
E_BAD_FRAME = "E7004"        # the provider returned something that is not a frame


def _raise(code: str, message: str, hint: str = ""):
    """Raise a TEX diagnostic from inside a stdlib call.

    The import is deferred to the error path because `tex_runtime.interpreter` imports the
    stdlib, which imports this module — a module-level import would be a cycle. Nothing on
    the success path pays for it.
    """
    from .tex_runtime.interpreter import InterpreterError
    raise InterpreterError(message, None, source="", code=code, hint=hint)


# ── the protocol ─────────────────────────────────────────────────────────────

class FrameProvider(Protocol):
    """The host source surface. Everything except the two fetch methods is optional — a
    provider that implements only `fetch_time`/`sample_time` is legal and gets the engine's
    defaults for the rest.

    `fetch_time` returns the frame at or nearest to `t`; `sample_time` MAY interpolate
    between the two frames bracketing it. TEX does not check that they differ — a provider
    that returns the same pixels from both is a legal nearest-neighbour provider — but it
    does cache them separately, because a provider is *allowed* to make them differ.

    Both return `[1,H,W,C]` or `[H,W,C]` at the SOURCE's own resolution, not the cook's.
    """
    #: Stable for the life of the session; keys the pool. Defaults to the class name.
    provider_id: str

    def fetch_time(self, source_key: str, t: float) -> torch.Tensor: ...
    def sample_time(self, source_key: str, t: float) -> torch.Tensor: ...

    #: Temporal identity: which two times are the same frame. Only the provider knows its
    #: own rate, so only the provider can answer. Optional — see `quantize_time` below.
    def quantize_time(self, source_key: str, t: float) -> float: ...

    #: Optional. A provider that watches its own sources reports here; one that does not
    #: gets the engine-side counter `bump_source_version()` drives.
    def source_version(self, source_key: str) -> int: ...


class NullFrameProvider:
    """No host source (the default): every fetch is E7001.

    A real object rather than `None` for the reason `NullHostServices` is: the call sites
    stay branch-free, and the refusal carries a message that says what to do about it
    instead of an AttributeError that says a field was missing.
    """
    provider_id = "null"

    def fetch_time(self, source_key: str, t: float) -> torch.Tensor:
        _raise(E_NO_PROVIDER,
               f"no frame provider is registered, so `{source_key}` cannot be read at "
               f"t={t:g}.",
               hint="A host supplies sources with tex_provider.set_provider(p). "
                    "fetch_time/sample_time read through that seam and nothing else — "
                    "TEX never opens a file itself.")

    def sample_time(self, source_key: str, t: float) -> torch.Tensor:
        return self.fetch_time(source_key, t)


class SyntheticFrameProvider:
    """A deterministic procedural source: no files, no decoding, reproducible everywhere.

    It ships (rather than living in `tests/`) for the same reason `NullHostServices` does —
    it is the reference implementation of the protocol, and a host wiring itself in wants
    something to check its plumbing against before its own decoder exists. Tests and
    benchmarks use it as the synthetic provider the DoD asks for.

    Pixel value at time `t` is `(t*step + channel_bias)` folded into [0,1], so a frame's
    identity is readable from any single pixel — which is what makes a "did I get the frame
    I asked for?" assertion a scalar comparison rather than a hash.
    """

    def __init__(self, *, res: int = 64, channels: int = 4, rate: float = 24.0,
                 device="cpu", provider_id: str = "synthetic", latency_s: float = 0.0):
        self.provider_id = provider_id
        self.res = int(res)
        self.channels = int(channels)
        self.rate = float(rate)
        self.device = device
        #: A deliberate per-fetch stall, for the IO-1 / PM-9 harnesses. `time.sleep`
        #: releases the GIL cleanly and a real decoder does not — see docs/async-io.md §6.
        self.latency_s = float(latency_s)
        self.fetches = 0

    def quantize_time(self, source_key: str, t: float) -> float:
        return quantize_at_rate(t, self.rate)

    def _frame(self, source_key: str, t: float, bias: float) -> torch.Tensor:
        if self.latency_s > 0.0:
            import time as _time
            _time.sleep(self.latency_s)
        self.fetches += 1
        h = w = self.res
        base = math.fmod(abs(float(t)) * 0.01 + bias + (len(source_key) * 0.003), 1.0)
        ramp = torch.linspace(0.0, 0.25, w, device=self.device).view(1, 1, w, 1)
        col = torch.linspace(0.0, 0.125, h, device=self.device).view(1, h, 1, 1)
        out = (base + ramp + col).expand(1, h, w, self.channels).contiguous()
        return out.clamp_(0.0, 1.0)

    def fetch_time(self, source_key: str, t: float) -> torch.Tensor:
        return self._frame(source_key, t, 0.0)

    def sample_time(self, source_key: str, t: float) -> torch.Tensor:
        # Distinguishable from fetch_time by construction, so the "the two modes cache
        # separately" pin is a value comparison and not an act of faith.
        return self._frame(source_key, t, 0.5)


def quantize_at_rate(t: float, rate: float) -> float:
    """`round(t*rate)/rate`, the default temporal identity. `rate <= 0` means "do not
    quantize" — a host with variable-rate media returns whatever its own index says.

    This is why 23.999999 and 24.0 do not double-cache: at rate 1 they are the same frame,
    and the engine quantizes ONCE, at the pool boundary, then passes the quantized value on
    to the provider. A provider therefore never sees a time it would itself have rounded —
    the pool key and the fetch argument are the same number by construction rather than by
    two implementations agreeing."""
    t = float(t)
    if not rate or rate <= 0 or not math.isfinite(t):
        return t
    return round(t * rate) / rate


# ── the process-wide provider (mirrors get_host_services) ────────────────────

_provider = None
_provider_lock = threading.Lock()


def get_provider():
    """The registered provider, or the Null default. Never None."""
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = NullFrameProvider()
    return _provider


def set_provider(provider) -> None:
    """Register the host's source provider. Passing None restores the Null default.

    Changing the provider drops the pool: entries are keyed by `provider_id`, so a second
    provider reusing the first's id would otherwise be served the first's pixels. Dropping
    is the conservative direction and costs a host nothing it can notice.
    """
    global _provider
    with _provider_lock:
        _provider = provider
    get_media_cache().clear()


def reset_provider() -> None:
    """Drop the registration (tests)."""
    set_provider(None)


def provider_id(provider=None) -> str:
    p = provider if provider is not None else get_provider()
    return str(getattr(p, "provider_id", None) or type(p).__name__)


# ── source versions: the invalidation channel ────────────────────────────────
#
# The host bumps because the host knows it re-exported. A provider MAY own the counter
# instead (a host with a real file watcher); the engine's own dict is the default for the
# common case, where a provider is a thin wrapper over an already-decoded sequence.

_versions: dict = {}


def source_version(source_key: str, provider=None) -> int:
    p = provider if provider is not None else get_provider()
    fn = getattr(p, "source_version", None)
    if fn is not None:
        try:
            return int(fn(source_key))
        except Exception:
            pass          # a provider that raises here falls back; it never fails a cook
    return int(_versions.get(source_key, 0))


def bump_source_version(source_key: str) -> int:
    """Declare that `source_key` changed on disk. Returns the new version.

    Two effects, and the second is not redundant: the version is part of the pool key, so
    old entries are already unreachable — but leaving them would let a re-export leak the
    whole previous version's bytes into the governor's accounting until pressure happened
    to evict them.
    """
    v = int(_versions.get(source_key, 0)) + 1
    _versions[source_key] = v
    get_media_cache().invalidate_source(source_key)
    return v


def source_flags(*source_keys) -> tuple:
    """Lineage-key flags stamping the versions of the sources a cook reads:
    `source_flags("plate", "matte") -> ("src=matte@1", "src=plate@3")`.

    **The host obligation this exists for, stated plainly.** `lineage_key` knows nothing
    about sources, so a cook that read `plate@v3` and a cook that read `plate@v4` mint the
    same result key and the second is served the first's pixels. Passing these through the
    existing `flags=` component fixes that with no key-shape change — a host that never
    calls this sees byte-identical keys to v0.33.

    Deriving the read-set automatically is the structural fix and it is deferred, with the
    reason recorded in DEVELOPMENT.md: the set is only known AFTER the cook (a source key
    may be a `$param`, an expression, or a loop variable), and an AST derivation would
    close the common case while silently missing the rest.
    """
    return tuple(sorted(f"src={k}@{source_version(k)}" for k in set(source_keys)))


# ── the media pool ───────────────────────────────────────────────────────────

@dataclass
class _MediaEntry:
    tensor: torch.Tensor
    nbytes: int
    dev_type: str
    #: The quantized time this frame IS. Kept so eviction can be playhead-aware — this is
    #: the one pool in the tree whose entries carry a time, so the `playhead` hint the
    #: governor has always passed can finally mean something here.
    t: float
    hits: int = 0


class MediaCache:
    """The DATA-7 source pool: LRU, byte-budgeted, per-device buckets.

    Exposes exactly the `governed_bytes` / `evict_bytes` pair `ResultCache` does, so it
    registers into `CacheRegistry` through the existing call and the governor learns no new
    concept.

    **In-memory only, and that is a decision rather than an omission.** CACHE-8's ladder
    exists because a cooked frame is expensive and has nowhere else to live. A source
    frame's disk tier is *the source file*, which the host already has and can already
    decode; spilling a decoded copy beside it buys a faster decode at the price of a second
    copy of the user's media on their disk.
    """

    def __init__(self, budget_mb: float = 512.0):
        #: Re-entrant for the reason ResultCache's is: `put` can evict, and eviction is
        #: bookkeeping that reads the same structures.
        self._lock = threading.RLock()
        self._entries: "OrderedDict[tuple, _MediaEntry]" = OrderedDict()
        self._bytes_by_dev: dict = {}
        self._budget = int(max(0.0, budget_mb) * 1024 * 1024)
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        #: Speculative inserts refused for backpressure (IO-1). Counted separately from
        #: evictions because they mean the opposite thing: nothing was lost, a guess was
        #: declined.
        self.refused = 0

    # ── reads ──
    def get(self, key: tuple):
        with self._lock:
            e = self._entries.get(key)
            if e is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            e.hits += 1
            self.hits += 1
            return e.tensor

    # ── writes ──
    def put(self, key: tuple, tensor: torch.Tensor, *, t: float = 0.0,
            speculative: bool = False) -> bool:
        """Insert a frame. Returns False if the insert was REFUSED.

        A speculative insert (a prefetch) into a pool that is already at budget is refused
        rather than admitted-then-evicting: a prefetch is a guess, and evicting a frame
        somebody actually asked for to make room for a guess inverts the whole policy.
        Refusal is cheap precisely because the host can fetch on demand — the frame is
        still on disk. That is IO-1's backpressure, and it is the whole of it.
        """
        nbytes = int(tensor.numel() * tensor.element_size())
        dev_type = tensor.device.type
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return True
            if speculative and self._budget and self._total() + nbytes > self._budget:
                self.refused += 1
                return False
            self._entries[key] = _MediaEntry(tensor, nbytes, dev_type, float(t))
            self._bytes_by_dev[dev_type] = self._bytes_by_dev.get(dev_type, 0) + nbytes
            if self._budget:
                self._enforce_locked()
            return True

    def _total(self) -> int:
        return sum(self._bytes_by_dev.values())

    def _drop_locked(self, key: tuple) -> int:
        e = self._entries.pop(key, None)
        if e is None:
            return 0
        left = self._bytes_by_dev.get(e.dev_type, 0) - e.nbytes
        # Clamped, not asserted: a negative bucket is the accounting drift that produced
        # the 16x over-eviction incident, and the governor reads these numbers.
        self._bytes_by_dev[e.dev_type] = max(0, left)
        return e.nbytes

    def _enforce_locked(self) -> int:
        freed = 0
        while self._total() > self._budget and self._entries:
            freed += self._drop_locked(next(iter(self._entries)))
            self.evictions += 1
        return freed

    def invalidate_source(self, source_key: str) -> int:
        """Drop every entry for one source, whatever its version or mode. Returns the count."""
        with self._lock:
            doomed = [k for k in self._entries if len(k) > 1 and k[1] == source_key]
            for k in doomed:
                self._drop_locked(k)
            return len(doomed)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes_by_dev.clear()

    # ── the governor surface (mirrors ResultCache) ──
    def governed_bytes(self, dev_type: str) -> int:
        with self._lock:
            return int(self._bytes_by_dev.get(dev_type, 0))

    def evict_bytes(self, need: int, *, dev_type: str = "cpu", playhead=None) -> int:
        """Free at least `need` bytes from `dev_type`'s bucket. Returns bytes freed.

        `playhead` finally does something here. Every other pool the governor drains holds
        entries with no notion of when they are for, so the hint has always been accepted
        and ignored; a media frame IS a time, so the eviction order is "furthest from where
        the user is looking, first" rather than plain LRU. A scrub backwards through a
        window therefore does not evict the frames it is about to re-read.
        """
        if need <= 0:
            return 0
        with self._lock:
            cands = [k for k, e in self._entries.items() if e.dev_type == dev_type]
            if playhead is not None:
                try:
                    ph = float(playhead)
                    cands.sort(key=lambda k: -abs(self._entries[k].t - ph))
                except (TypeError, ValueError):
                    pass          # an unusable hint degrades to LRU, never to an error
            freed = 0
            for k in cands:
                if freed >= need:
                    break
                freed += self._drop_locked(k)
                self.evictions += 1
            return freed

    def stats(self) -> dict:
        with self._lock:
            return {"frames": len(self._entries), "bytes": self._total(),
                    "bytes_by_dev": dict(self._bytes_by_dev),
                    "hits": self.hits, "misses": self.misses,
                    "evictions": self.evictions, "refused": self.refused,
                    "budget": self._budget}


_media_cache: "MediaCache | None" = None


def get_media_cache() -> MediaCache:
    """The process-wide media pool (created on first use)."""
    global _media_cache
    if _media_cache is None:
        with _provider_lock:
            if _media_cache is None:
                _media_cache = MediaCache()
    return _media_cache


def set_media_budget_mb(mb: float) -> None:
    c = get_media_cache()
    with c._lock:
        c._budget = int(max(0.0, mb) * 1024 * 1024)
        c._enforce_locked()


def stats() -> dict:
    """What a host HUD, `tex doctor` and the reattach report all read."""
    return {"provider": provider_id(), **get_media_cache().stats()}


# ── materialization: the path the stdlib functions call ──────────────────────

def _normalize(frame, source_key: str, t: float) -> torch.Tensor:
    """Coerce a provider's return into `[1,H,W,C]`, or refuse it by name."""
    if not isinstance(frame, torch.Tensor):
        _raise(E_BAD_FRAME,
               f"the frame provider returned {type(frame).__name__} for `{source_key}` at "
               f"t={t:g}; a frame must be a tensor.",
               hint="Return [1,H,W,C] or [H,W,C] from fetch_time/sample_time.")
    if frame.dim() == 3:
        frame = frame.unsqueeze(0)
    # C == 4 exactly, not 1..4. `fetch_time`/`sample_time` are typed VEC4 by
    # `stdlib_signatures`, and unlike `sample(@A,…)` there is no wire to take a channel
    # count from — so either the type checker states a truth or it states a hope. The host
    # owns decoding, which makes expanding a mono/RGB source to RGBA its job, and this
    # refusal names that. Per-source declared channel counts arrive with DATA-6 (v0.35).
    if frame.dim() != 4 or frame.shape[0] != 1 or frame.shape[-1] != 4:
        _raise(E_BAD_FRAME,
               f"the frame provider returned shape {tuple(frame.shape)} for `{source_key}` "
               f"at t={t:g}; a frame must be [1,H,W,4].",
               hint="A provider returns ONE RGBA frame per call. Batch axes belong to the "
                    "cook, not to the source; expand a mono or RGB source host-side, where "
                    "the decoding decisions already live.")
    return frame


def materialize(source_key: str, t: float, mode: str = "fetch", *,
                speculative: bool = False):
    """Get one source frame, through the pool. `mode` is 'fetch' or 'sample'.

    Returns the frame, or None when `speculative=True` and the pool refused the insert
    (backpressure — the caller is a prefetch and has nothing to report).
    """
    prov = get_provider()
    source_key = "" if source_key is None else str(source_key)
    try:
        qt = float(prov.quantize_time(source_key, t))    # type: ignore[attr-defined]
    except AttributeError:
        qt = float(t)          # a provider without a rate quantizes not at all
    except Exception:
        qt = float(t)

    cache = get_media_cache()
    # CACHE-6's precedent, restated: no source key -> no caching, ever. A host that cannot
    # name a source cannot promise anything about it.
    key = None
    if source_key:
        key = (provider_id(prov), source_key, mode, repr(qt), source_version(source_key, prov))
        hit = cache.get(key)
        if hit is not None:
            return hit

    fn = getattr(prov, "sample_time" if mode == "sample" else "fetch_time")
    try:
        frame = fn(source_key, qt)
    except Exception as e:
        # An InterpreterError from the Null provider (E7001) is already the right
        # diagnostic; anything else is the host's exception and gets named as such.
        if type(e).__name__ == "InterpreterError":
            raise
        _raise(E_FETCH_FAILED,
               f"the frame provider failed reading `{source_key}` at t={qt:g}: "
               f"{type(e).__name__}: {e}",
               hint="This is the host's provider raising, not a TEX error. "
                    "A SPECULATIVE prefetch records it in the refusals ledger instead.")
    frame = _normalize(frame, source_key, qt)
    if key is not None:
        if not cache.put(key, frame, t=qt, speculative=speculative):
            return None if speculative else frame
    return frame


def _uniform_time(t, source_key: str) -> float:
    """The single scalar `t` a call names, or E7003.

    A per-pixel `t` is the interesting generalization — it is a retime map — and it is
    unbounded I/O from one statement: as many source frames as there are distinct values,
    at cook resolution. There is no honest cap to pick, and the failure mode of guessing
    one (silently servicing 8 of 4096 requested frames) is wrong pixels. So it is refused,
    and the message says how many distinct values it saw.

    The same rule makes a per-BATCH-ELEMENT `t` (`time + fi/fps` over a B=100 batch) a
    refusal rather than a silent single-frame read. That case is bounded by B and is the
    natural next step; it is deferred with its gate in DEVELOPMENT.md.
    """
    if not isinstance(t, torch.Tensor):
        return float(t)
    if t.numel() == 1:
        return float(t.reshape(()).item())
    flat = t.reshape(-1)
    if bool(torch.all(flat == flat[0])):
        return float(flat[0].item())
    n = int(torch.unique(flat).numel())
    _raise(E_NONUNIFORM_TIME,
           f"fetch_time/sample_time need ONE time per cook, but `{source_key}` was asked "
           f"for {n} distinct times across the grid.",
           hint="Hoist the time out of the pixel grid (a $param, the `time` builtin, or a "
                "literal), or cook one frame per playhead. A per-pixel retime would fetch "
                "one source frame per distinct value.")


# ── IO-1: prefetch windows ───────────────────────────────────────────────────

def declare_window(queue, source_key: str, t0: float, t1: float, *,
                   confidence: float = 0.5, mode: str = "sample",
                   max_frames: int = 64) -> list:
    """Mint one SPECULATIVE prefetch job per quantized frame in `[t0, t1]`.

    A prefetch window is a bet like any other: priced, ordered and shed by the
    `SpeculativePolicy` already installed on the queue, so the window that arrives during a
    render loses to the render by rules the tree already has. Returns the submitted jobs
    (some may already be CANCELLED — the policy refuses at submit).

    `feeds_profile=False` is the PROF-1 pollution guard and it is deliberately explicit
    rather than "we happened not to pass a profile key": an I/O wait recorded as compute
    cost would poison CACHE-7 placement and PRED-1 admission in one stroke, and the host
    controls the `profile_key` argument.

    `max_frames` bounds a fat-fingered window. It is REPORTED, not silent: the return list
    is short and the caller can see it.
    """
    from .tex_cookqueue import SPECULATIVE, PREFETCH

    prov = get_provider()
    try:
        qt0 = float(prov.quantize_time(source_key, t0))   # type: ignore[attr-defined]
        qt1 = float(prov.quantize_time(source_key, t1))   # type: ignore[attr-defined]
        step = abs(qt1 - qt0) / max(1, max_frames - 1) if qt1 != qt0 else 1.0
        rate = float(getattr(prov, "rate", 0.0) or 0.0)
        step = (1.0 / rate) if rate > 0 else step
    except Exception:
        qt0, qt1, step = float(t0), float(t1), 1.0

    times, t = [], qt0
    direction = 1.0 if qt1 >= qt0 else -1.0
    while len(times) < max_frames and (t - qt1) * direction <= 1e-9:
        times.append(t)
        t += step * direction

    jobs = []
    for ts in times:
        def _prefetch(cancel, _t=ts):
            # Drop-on-landing: a shed or cancelled prefetch that is already inside the
            # provider's read cannot be stopped from outside. What TEX guarantees is that
            # its result is never installed, and this is where that is guaranteed.
            frame = materialize(source_key, _t, mode, speculative=True)
            if cancel is not None:
                cancel.check()
            return frame is not None
        jobs.append(queue.submit(_prefetch, klass=SPECULATIVE, reason=PREFETCH,
                                 confidence=confidence, feeds_profile=False))
    return jobs
