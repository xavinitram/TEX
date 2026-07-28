"""tex_results.py — results become first-class (v0.25 "Remember frames").

This module is the engine's answer to "TEX persists *programs* superbly and *results* not
at all." It has two halves, in dependency order:

  CACHE-1  lineage keys (env_epoch / lineage_key). Every cooked output can carry the key
           that produced it: H(program fp × param values × upstream result keys × frame ×
           device × precision/quality × env_epoch × flags [× canvas/ROI]). Device and
           precision are MANDATORY components — invariant #9's up-to-6.1e-2 CPU↔GPU envelope
           makes placement visible, so a cross-device / precision / env-epoch cook mints a
           DISTINCT key and is never served from a stale one. This is Nuke's op-hash; the
           fused memo key (tex_fusion._fused_fp) already proved the value-independent half —
           CACHE-1 adds the value-DEPENDENT half (params by value, tensor inputs by their
           upstream lineage key, NEVER by content-hashing pixels — the sampling hash has an
           admitted collision class that is fine for cache-BUSTing and wrong for cache-REUSE).

  CACHE-2  ResultCache — the engine frame cache (added below CACHE-1). RAM tier byte-budgeted
           through the tex_memory seam, disk spill staged through tex_marshalling's pinned
           helpers, keyed by CACHE-1, frames frozen per ENG-12 (tex_engine.freeze). The
           ComfyUI node does NOT enable it (the host already caches); it is armed by an
           engine host — so it ships measured, tested, and dormant, exactly as ROI-3 did.

Scope honesty (docs/results-caching.md): under ComfyUI CACHE-1's reach is TEX-internal edges
(fused-stage handoffs, CACHE-6) — full lineage arrives with GRAPH-1's version counters. Here
it is the persistence/disk identity a frame cache and a future disk spill are keyed by.
"""

import hashlib
import json


# ── CACHE-1: lineage keys ─────────────────────────────────────────────────────

# env_epoch is a pure function of (active CUDA device, torch, code epoch), so memoize it per
# device — it is folded into every result key. Keyed by torch.cuda.current_device() (-1 for CPU)
# so a multi-GPU host that switches devices between cooks gets each GPU's real identity.
_ENV_EPOCH_CACHE: dict = {}


def _code_epoch() -> str:
    """The compiler/codegen code identity a cached RESULT is only reproducible under: the CACHE-4
    CODEGEN_EPOCH (which nests AST_EPOCH, so ANY parse/typecheck/optimize OR codegen change bumps
    it). A code change can re-dispatch conv/bilateral kernels (~1 ulp) and move any pixel, so a
    spilled frame from a prior codegen epoch must not be served — folding this epoch into the
    result key mints a fresh key on every such change."""
    try:
        from .tex_cache import codegen_epoch
        return codegen_epoch()
    except Exception:
        return "0"


def env_epoch() -> str:
    """The execution-environment identity a cached result is only valid within: torch
    version + GPU identity (device name + compute capability) + the code epoch. Folding all
    three into every result key means a frame minted under one environment is never served
    under another — the silent cross-environment hit a result cache must not have. Mirrors
    and extends xfer._version_tag (device name + torch); adds compute capability + code epoch.
    Memoized PER active CUDA device (torch/GPU identity is fixed per device, but a heterogeneous
    multi-GPU host switches current_device between cooks — a single process-wide memo would freeze
    the epoch to whichever GPU was active at the first call and stamp a cuda:1 frame with cuda:0's
    identity)."""
    parts = []
    dev = -1
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
    except Exception:
        pass
    cached = _ENV_EPOCH_CACHE.get(dev)
    if cached is not None:
        return cached
    try:
        import torch
        parts.append(torch.__version__.split("+")[0])
        if dev >= 0:
            parts.append(torch.cuda.get_device_name(dev))
            cc = torch.cuda.get_device_capability(dev)
            parts.append(f"sm{cc[0]}{cc[1]}")
    except Exception:
        pass
    parts.append(_code_epoch())
    tag = "|".join(parts)
    _ENV_EPOCH_CACHE[dev] = tag
    return tag


def _canon_params(params) -> str:
    """Deterministic, collision-free encoding of a program's scalar/vector param values.
    Tensor values must NOT be here — a tensor input enters a lineage key by its upstream
    key, never its pixels. `default=repr` keeps a stray unexpected type from raising (it
    just keys conservatively); sort_keys makes name order irrelevant."""
    return json.dumps(params or {}, sort_keys=True, default=repr)


def _canon_time(tc) -> str:
    """Deterministic encoding of the ENG-7 host playhead. ALL playhead builtins move output
    pixels while being kept out of the program fingerprint (interpreter `_TIME_BUILTIN_NAMES` =
    frame/fps/time), so a result key must carry every one of them, by EXACT value — folding the
    whole normalized dict (not just `frame`) future-proofs a fourth builtin, and `repr(float)`
    keeps fractional/sub-frame playheads (motion blur, retime) distinct where `int(frame)` would
    collide them onto a stale frame."""
    if not tc:
        return "n"
    return json.dumps({k: repr(float(v)) for k, v in tc.items()}, sort_keys=True)


def lineage_key(*, program_fp, device, precision, params=None, upstream=(),
                frame=None, time_context=None, quality=None, flags=(), canvas=None) -> str:
    """CACHE-1: the content-addressable identity of a cooked RESULT (a hex SHA-256).

    Composes H(program_fp × params × upstream × frame × device × precision/quality ×
    env_epoch × flags × canvas). Structured, length-prefixed encoding (mirrors
    TEXCache.fingerprint) so no component can bleed into an adjacent one.

    program_fp   the value-independent program fingerprint (fp or fused_fp).
    device       MANDATORY. str(device); a cook on another device is a different result.
    precision    MANDATORY. the EFFECTIVE precision the cook ran at.
    params       the non-tensor binding values (widget $params); enter by value.
    upstream     the lineage keys of this cook's tensor inputs (empty under ComfyUI, where
                 there is no TEX-internal upstream edge yet — a GRAPH-1 host threads them).
    frame        a single host playhead frame, or None (a still). Keyed by exact value.
    time_context the FULL ENG-7 playhead dict {frame,fps,time,...}, or None — every builtin in
                 it moves pixels, so every one must key (a `time`- or `fps`-only animation is a
                 distinct result even at the same frame). The engine passes this; a caller with
                 only a frame number may pass `frame=` instead.
    quality      a preview/final quality tag (PREC-1), or None.
    flags        any extra keying flags (e.g. an output name for a per-output key).
    canvas       a canvas / ROI descriptor (W,H[,x0,y0,w,h]); two cooks at different canvas
                 sizes or ROIs are distinct results (keys carry it from day one).
    """
    if program_fp is None:
        raise ValueError("lineage_key needs a program fingerprint (fp or fused_fp)")
    if device is None or precision is None:
        raise ValueError("lineage_key: device and precision are MANDATORY key components "
                         "(invariant #9 — a cross-device/precision hit is never served)")
    h = hashlib.sha256()

    def feed(tag: str, s: str) -> None:
        b = f"{tag}={s}".encode()
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)

    feed("fp", str(program_fp))
    feed("dev", str(device))
    feed("prec", str(precision))
    feed("env", env_epoch())
    feed("par", _canon_params(params))
    feed("up", json.dumps([str(u) for u in upstream]))
    feed("frm", "n" if frame is None else repr(float(frame)))   # exact value, no int() collide
    feed("tc", _canon_time(time_context))                        # every playhead builtin keys
    feed("q", "n" if quality is None else str(quality))
    feed("flg", json.dumps(sorted(str(f) for f in flags)))
    # canvas is any JSON-able shape/ROI descriptor (a dict {"shape":[B,H,W,C],"roi":[...]}, or a
    # legacy (W,H) tuple) — the engine keys each output by its produced-frame shape, so a
    # different batch/canvas/ROI mints a distinct key.
    feed("cnv", "n" if canvas is None else json.dumps(canvas, sort_keys=True, default=list))
    return h.hexdigest()


# ── CACHE-2: the engine frame cache (ResultCache) ─────────────────────────────

import os
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass

#: B5a — the `.frame` spill-record format version. ABSENCE reads as v0 (raw), which is exactly
#: what every file written before v0.33.1 is, so both formats are readable for one release and
#: an upgrade is never a silent cold start.
#:
#: This field should have landed with v0.33, which CHANGED the record (adding `orig`, and
#: redefining `device` as the HOME device rather than the current one) without one. Compat held
#: only by accident: `rec.get("orig")` defaults to `None` = "stored as cooked", and for a
#: v0.32 frame `device` and `home` were necessarily equal. Neither accident survives the next
#: change, so the field goes in now, while the reader still knows what v0 meant.
#:
#:   v0  (absent)  {t, device, canvas, epoch}                  — v0.32 and earlier
#:   v1            + orig (storage repr), viewed (uint16 view)  — v0.33.1
#:   v2            + quality (the tier tag the frame was stored under) — v0.33.2
_FRAME_FORMAT = 2


class _DepthLock:
    """An `RLock` that knows how deep this thread is inside it.

    The one thing it adds is `depth`, and it exists because the drains need to answer "is a
    composite operation on this thread holding the lock around me?" — which IS re-entrancy
    depth. `patch_region` takes the lock across a nested `put` for atomicity, and that `put`'s
    drains must not run a disk write and a D2H under it (the 327-496 ms stall `_drain_spills`
    was created to remove).

    The first version of this was a thread-local flag `patch_region` set by hand in a
    `try/finally`. That made correctness a property of one method remembering to do it: a
    second composite — a batched put, a future `patch_region` that recurses — would silently
    reintroduce the stall, and no test would notice, because the drains still HAPPEN, just at
    the wrong time. Deriving it from the lock makes every future composite correct by
    construction, and a nested composite cannot un-defer its parent (a flag is a boolean where
    the state is a depth).
    """

    __slots__ = ("_lock", "_local")

    def __init__(self):
        self._lock = threading.RLock()
        self._local = threading.local()

    @property
    def depth(self) -> int:
        return getattr(self._local, "n", 0)

    def __enter__(self):
        self._lock.acquire()
        self._local.n = self.depth + 1
        return self

    def __exit__(self, *exc):
        self._local.n = self.depth - 1
        self._lock.release()
        return False


@dataclass(slots=True, eq=False)
class _Entry:
    """One cached frame and everything the tiers need to know about it.

    A MUTABLE dataclass, not a NamedTuple: the residency ladder swaps `tensor` and `device` in
    place on a live entry, and identity (`is not entry`) is how `_drain_demotes`/`_promote`
    re-check that the entry they copied is still the one in the table — hence `eq=False`.

    It used to be a 7-slot list read positionally, and the release's own mutation table pins
    the bug that shape makes possible: `_spill` persisting `entry[3]` (where the frame IS)
    instead of `entry[6]` (where it BELONGS) turns the residency ladder into a one-way trip to
    the CPU. Two of the seven slots are device strings; only the names tell them apart.

    tensor      the frozen master (ENG-12), contiguous, storage-owning
    stamp       `frame_version` at insert — a constant 0 for a frozen entry, so `verify_unmutated`
                is a live detector only for a NORMAL (host-supplied) one
    nbytes      the byte charge every budget and the governor read
    device      where the frame IS right now (demotion moves it)
    canvas      opaque caller metadata (CACHE-9 records the window it wrote)
    orig_dtype  the dtype the frame was COOKED at when stored reduced, else None (PREC-1)
    home        where the frame BELONGS — its cook device (CACHE-8)
    quality     the tier tag this frame was STORED under (v0.33.2). The cache needs it for two
                things the tag alone could not do: `patch_region` must not launder a preview
                base into a final-shaped result, and a requalify-on-idle pass has to be able to
                ENUMERATE preview entries. `orig_dtype` is not a substitute — an unpackable
                preview frame (out of fp16 range, or a MASK) stores full and would read as final.
    """
    tensor: object
    stamp: int
    nbytes: int
    device: str
    canvas: object
    orig_dtype: object
    home: str
    quality: object = None


#: torch dtype -> the `tex_io.STORAGE_DTYPES` name a spill record persists it under. Built from
#: tex_io's own table rather than `str(dtype).replace("torch.", "")`, and read back through a
#: dict rather than `getattr(torch, <a string that came off disk>)`.
_DTYPE_NAME: dict = {}
_NAME_DTYPE: dict = {}


def _dtype_tables():
    global _DTYPE_NAME, _NAME_DTYPE
    if not _DTYPE_NAME:
        from .tex_io import _STORAGE_TORCH
        _NAME_DTYPE = dict(_STORAGE_TORCH)
        _DTYPE_NAME = {dt: name for name, dt in _STORAGE_TORCH.items()}
    return _DTYPE_NAME, _NAME_DTYPE


def _dev_bucket(device) -> str:
    """The per-device accounting bucket for a device or device string. One spelling: the
    ternary was written out at each accounting site, and `_bytes_by_dev` only has meaning if
    every site agrees on which bucket an entry lands in. Non-CUDA accelerators (mps/xpu) bucket
    with cpu today — the governor arbitrates a CUDA pool and a host pool, and nothing else."""
    return "cuda" if str(device).startswith("cuda") else "cpu"


def _budget_bytes(env_name: str, default: int) -> int:
    v = os.environ.get(env_name)
    if v:
        try:
            return max(0, int(float(v) * (1 << 20)))
        except Exception:
            pass
    return default


#: MEMOIZED default budget (P0-8). `torch.cuda.is_available()` initializes the CUDA context on
#: first call — measured at ~20 ms — and `_default_ram_budget` runs in `ResultCache.__init__`.
#: `cook_checkpointed` constructs a cache per call in its all-miss prologue, so that 20 ms was
#: being paid per cook and was the dominant term in the 2.46x first-cook slowdown its docstring
#: claimed could not happen. The value is a pure function of the box, so computing it once is
#: not a cache — it is not recomputing a constant.
_DEFAULT_RAM_BUDGET: "int | None" = None


def _default_ram_budget() -> int:
    """A conservative default frame-RAM budget. A cook keeps its outputs on the cook device,
    so a CUDA frame cache lives in VRAM and must not crowd out the cook itself — hence a
    modest slice, host-overridable via TEX_RESULTS_BUDGET_MB. CACHE-5 will fold this into the
    global governor; until then the frame budget self-evicts.

    Memoized: see `_DEFAULT_RAM_BUDGET`."""
    global _DEFAULT_RAM_BUDGET
    if _DEFAULT_RAM_BUDGET is not None:
        return _DEFAULT_RAM_BUDGET
    val = 512 << 20
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            val = min(2 << 30, int(0.25 * total))
    except Exception:
        pass
    _DEFAULT_RAM_BUDGET = val
    return val


class ResultCache:
    """CACHE-2: a keyed store of cooked frames — RAM-tier byte-budgeted with a disk-spill
    tail, every entry frozen (ENG-12) and keyed by a CACHE-1 lineage key.

    A frame served from here equals a freshly cooked one bit-exact (the entry IS that tensor);
    the spill→restore round-trip is bit-exact; the RAM tier stays under budget by spilling its
    LRU victim to disk rather than dropping it. The ComfyUI node does not use it — it is armed
    by an engine host (see docs/results-caching.md).

    THREAD-SAFE as of CACHE-7 (v0.32). It used to say "not thread-safe by itself; a host that
    shares one across threads guards it", which was fair while every writer was the host's own
    cook. CACHE-7's phase-2 harvest makes the ENGINE a writer, on the SCHED-4 worker thread,
    while the host's interactive cook may be `get`-ing on the main thread — concurrent
    `move_to_end` and `popitem` on one OrderedDict, which is a corrupted LRU or a RuntimeError,
    not merely a stale read. Pushing that precondition onto every host would be unenforceable,
    so the guard lives here: an uncontended acquire measures 220 ns against a `put` of 1.13 ms
    (512²×4 fp32) — 0.02%. RE-ENTRANT because `get` → `_restore` → `put` is a real call chain.
    Never on the default ComfyUI path either way (invariant #7): `tex_node.py` has no reference
    to this module.

    **ONE LIVE INSTANCE PER SPILL DIRECTORY.** Thread-safety here is process-and-instance
    scoped: `_lock`, the generation counter and the per-key spill tickets (`_spill_seq` /
    `_spill_locks`) are attributes of THIS object. Two live caches pointing at the same
    `results/` dir therefore serialise nothing between them — their writes to one key can
    still land in either order, and one's `clear(disk=True)` can unlink the other's frames
    underneath it. What IS safe across instances is a single frame's file: `_atomic_pickle`
    is temp-and-rename, so a reader never observes a half-written record, which is why the
    ENG-13 reattach case (a NEW cache over a populated directory left by a dead process) is
    supported and pinned. Concurrent live sharers are not. A host that wants two views of one
    spill tier should share the object, not the path."""

    def __init__(self, *, budget_mb=None, disk_budget_mb=None, cache_dir=None):
        # THE RULE, stated once so the scope is checkable rather than a judgement call repeated
        # at every method: the lock covers STRUCTURE and BYTE ACCOUNTING — `_ram`, `_ram_bytes`,
        # `_disk_bytes`, the counters — and nothing else. Full-frame tensor copies and disk I/O
        # happen OUTSIDE it, with the entry pulled out first.
        #
        # That rule is what makes the lock affordable at all. Holding it across the work would
        # mean an interactive `get` waiting behind a CACHE-7 phase-2 `put` for a 1.1 ms memcpy
        # (512²×4) or a `torch.load` from the spill tier. `_remove` and `_enforce_ram_budget`
        # assume it is HELD; `_spill`/`_restore` deliberately run OUTSIDE it (they are the disk
        # I/O the rule exists to exclude) and are single-writer by drain-path convention.
        # RE-ENTRANT because `get` → `_restore` → `put` is a real call chain.
        self._lock = _DepthLock()
        self._ram: "OrderedDict[str, _Entry]" = OrderedDict()
        self._budget = (int(budget_mb * (1 << 20)) if budget_mb is not None
                        else _budget_bytes("TEX_RESULTS_BUDGET_MB", _default_ram_budget()))
        self._disk_budget = (int(disk_budget_mb * (1 << 20)) if disk_budget_mb is not None
                             else _budget_bytes("TEX_RESULTS_DISK_MB", 4 << 30))
        self._cache_dir = cache_dir
        self._spill_dir_cached: str | None = None
        # Running spill-dir byte total so the disk budget is O(1) per spill (one getsize) instead
        # of an O(entries) scandir every time. None = unknown (force a reconciling scan); set to
        # the scanned total after any census, invalidated (None) on any out-of-band removal.
        self._disk_bytes: int | None = None
        # Keys known to be on the spill tier, or None = NOT YET KNOWN. `_restore` opened with
        # an `os.path.exists` on every RAM miss — 19.1 us of syscall, and CACHE-7's deepest-first
        # serve probes up to one key per checkpoint per cook. A membership set removes that.
        #
        # It starts UNKNOWN, not empty. A `ResultCache` constructed over a spill dir a PREVIOUS
        # process populated already serves those frames — `get` falls through to `_restore`,
        # which reads by key and re-checks `env_epoch`; that is ENG-13's whole reattach story.
        # Initialising to `set()` would have asserted "this cache has spilled nothing", and a
        # fresh cache over a populated dir would have served NOTHING. Unknown falls back to the
        # stat and is filled by the first `_learn_spilled()` scan or by `reindex_disk`.
        self._spilled: "set | None" = None
        # Per-device byte totals, maintained rather than summed. `governed_bytes(dev_type)` was
        # O(entries) UNDER THE LOCK — 415 us at 2000 entries — and the CACHE-5 governor calls it
        # per arbitration, blocking every concurrent get/put for the duration.
        self._bytes_by_dev: dict = {"cuda": 0, "cpu": 0}
        # Victims evicted under the lock, spilled to disk OUTSIDE it. See `_drain_spills`.
        # A deque because this is a FIFO drain: `pop(0)` on a list is O(n).
        self._pending_spills: deque = deque()
        # CACHE-8 residency: the VRAM ceiling above which cold CUDA frames are DEMOTED to host
        # RAM instead of spilled to disk. None = off, which is v0.32's behaviour exactly — a
        # frame cache that has never been told a VRAM budget must not start moving frames
        # between devices because it was upgraded. GOV-1's presets and `set_vram_budget` arm it.
        self._vram_budget: "int | None" = None
        self._pending_demotes: deque = deque()
        # A1: keys popped off `_pending_demotes` whose copy has not yet been committed.
        # Between the pop and the commit a victim is in NEITHER the queue nor its final
        # device, so without this it is re-queuable and demotable twice — and two commits
        # of one cuda->cpu transfer drive `_bytes_by_dev['cuda']` negative permanently.
        self._demoting: set = set()
        # A5: bumped by `clear()`; a spill that started before the bump must not re-index.
        self._generation: int = 0
        # A1 (v0.33.2): per-KEY write sequence. `_generation` is global and only moves on
        # `clear()`, so it cannot order two in-flight spills of the SAME key — and those invert:
        # put(K,v1) evicts and starts spilling; put(K,v2) evicts and its spill LANDS; v1's write
        # then overwrites v2 on disk and `get(K)` restores v1. Measured: restored mean 1.0 where
        # the last put was 2.0. `_drain_spills`' "nothing to guard against here" was true only
        # for distinct keys.
        self._spill_seq: dict = {}
        # ...and a per-KEY write lock. The ticket alone cannot fix the ordering, only detect it
        # afterwards — and detecting afterwards means the loser has already overwritten the
        # winner's file, so the only safe response is to delete it and lose the frame. Holding
        # a lock per key across the write makes the check-and-write atomic, so the stale writer
        # bails BEFORE touching the file. Contention is per key and only between two in-flight
        # spills of the same key; readers and other keys never wait on it, and the module-wide
        # `_lock` is NOT held across the I/O (that rule is the whole point of the drain path).
        self._spill_locks: dict = {}
        # Non-zero while a `clear(disk=True)` walks the spill dir with the lock released.
        # A DEPTH COUNT, not a bool: a bool left True by an interrupt mid-walk would disable the
        # disk tier for the life of the process with nothing reporting it, and with two
        # concurrent clears the first to finish would clear it out from under the other's walk.
        self._purge_depth = 0
        self.hits = self.misses = self.spills = self.restores = self.evictions = 0
        # CACHE-8: how many frames the residency ladder moved, each direction. These are the
        # numbers that say whether the tier is doing anything, and `stats()` reports them —
        # a residency policy nobody can observe is a residency policy nobody can tune.
        self.demotions = self.promotions = 0

    @property
    def _ram_bytes(self) -> int:
        """DERIVED, never stored: the RAM total is by definition the sum of the per-device
        totals. It used to be a second counter incremented beside `_bytes_by_dev` at every
        mutation site — the exact shape of the drift `_remove` was created to end, one level
        up. Deriving it makes the two unable to disagree instead of testing that they don't."""
        return self._bytes_by_dev["cuda"] + self._bytes_by_dev["cpu"]

    # ── disk seam ──
    def _spill_dir(self):
        # Resolve + makedirs ONCE per cache (the path never changes), not per spill/restore:
        # _spill hit it twice (via _disk_path and _enforce_disk_budget) and _restore once, each a
        # get_cache() lookup + a makedirs syscall to the same unchanging dir.
        if self._spill_dir_cached is not None:
            return self._spill_dir_cached
        if self._cache_dir is not None:
            d = self._cache_dir
        else:
            from .tex_cache import get_cache
            d = get_cache()._cache_dir
        p = os.path.join(str(d), "results")
        os.makedirs(p, exist_ok=True)
        self._spill_dir_cached = p
        return p

    def _learn_spilled(self) -> None:
        """Populate the spilled-key set from the dir, once. Leaves it None on any error, which
        is the safe answer: unknown means `_restore` pays the stat and still finds the frame.

        THE THIRD unlocked directory walk in this class, and it needs the same witness the other
        two carry (`reindex_disk`, `clear`). The scandir runs with no lock, and `_spill` only
        records into `_spilled` when membership is ALREADY known — so a frame that spills during
        this walk is recorded nowhere, and asserting a definite set afterwards forgets it. That
        frame is then on disk, absent from the membership set, and `_restore` short-circuits on
        the set without stat-ing: unserveable forever, uncounted forever, and unreachable by the
        stale-epoch cleanup. Same defect as P0-6 and v0.33.2 A3, through the one door neither
        closed. If the spill counter moved across the walk, stay UNKNOWN — one stat per miss is
        the price, and this file never gives a definite wrong answer where it can give none."""
        try:
            with self._lock:
                spills_at_entry = self.spills
            found = set()
            with os.scandir(self._spill_dir()) as it:
                for e in it:
                    if e.name.endswith(".frame") and e.is_file():
                        found.add(e.name[:-len(".frame")])
            with self._lock:
                self._spilled = None if self.spills != spills_at_entry else found
        except OSError:
            with self._lock:            # the last unlocked write to `_spilled`; P0-6's rule
                self._spilled = None

    def _disk_path(self, key: str) -> str:
        return os.path.join(self._spill_dir(), f"{key}.frame")

    # ── core API ──
    def put(self, key: str, tensor, *, canvas=None, quality=None, storage=None,
            kind=None) -> None:
        """Store `tensor` under `key`, frozen per ENG-12 (an in-place write to a cached frame
        then raises instead of silently corrupting it). Re-inserting a key replaces it.

        PREC-1: `quality` is the caller's tier tag — the same string it passed to
        `lineage_key(quality=...)` to mint `key`. `tex_packing.PREVIEW` makes this frame
        eligible to be STORED at half precision (2x the frames in the same budget); anything
        else, INCLUDING the default None, stores exactly what was cooked. `storage="fp32"`
        pins a frame at full precision regardless. `kind` is the output KIND
        (`tex_marshalling.map_inferred_type` — IMAGE / MASK / LATENT / ...): the colour-vs-data
        split, taken from the seam that already classifies it. MASK and LATENT are never packed.

        Storage precision is not visible through this class: `get` returns the cooked dtype
        either way. It is visible in the BYTES, which is the point, and in the pixels to the
        tune of 4.9e-4 — 8x under the 8-bit display quantum and 125x under the CPU-vs-GPU
        envelope invariant #9 already ships. See tex_packing for the argument and
        `benchmarks/storage_precision_bench.py` for the measurements.
        """
        import torch
        if not isinstance(tensor, torch.Tensor):
            return
        from . import tex_packing
        want = tex_packing.choose_storage(tensor, quality=quality, storage=storage, kind=kind)
        self._admit(key, tex_packing.pack(tensor, want), canvas,
                    tensor.dtype if want is not None else None, quality=quality)

    def _admit(self, key: str, tensor, canvas, orig_dtype, home=None,
               gen=None, quality=None) -> "_Entry | None":
        """Insert an ALREADY-REPRESENTED frame: freeze it, account it, enforce the budgets.

        The shared body of `put` (which decides the representation first) and `_restore`
        (whose representation was decided when the frame was spilled, and must not be decided
        again — re-running the policy on a restored half frame would ask `choose_storage`
        about a tensor whose original dtype is already gone).

        `home` (CACHE-8) is the device this frame BELONGS on. A plain `put` has no opinion, so
        it defaults to wherever the tensor already is — except on a replace, where the prior
        entry's home wins: a host re-putting a key whose frame is currently demoted to RAM is
        stating new pixels, not a new residency, and losing the home there would strand the
        frame on the CPU with nothing left that remembers it was a CUDA frame.
        """
        from . import tex_engine
        # Everything down to `stamp` is per-THIS-tensor work touching no shared state, and
        # `frozen_copy` is a full-frame memcpy — so it stays outside the lock, per the rule on
        # __init__. Normalize layout at ingest so the RAM tier and a disk-restored copy of the
        # same key are layout-identical (the spill serializes a contiguous copy); .contiguous()
        # is a no-op on a channels-last cook output, the common case.
        frozen = tex_engine.freeze(tensor.contiguous())
        own_bytes = frozen.numel() * frozen.element_size()
        try:
            storage_bytes = frozen.untyped_storage().nbytes()
        except Exception:
            storage_bytes = own_bytes
        # A frozen contiguous SLICE of a larger buffer (e.g. a host caching output[i] of a batch)
        # shares — so reports AND pins alive — the whole parent's storage. Compact it to its own
        # buffer so the byte accounting is honest and the parent can be freed. The documented
        # whole-output path already owns its storage (own == storage), so it is untouched.
        if storage_bytes > own_bytes:
            frozen = tex_engine.frozen_copy(frozen)
            storage_bytes = own_bytes
        stamp = tex_engine.frame_version(frozen)
        with self._lock:
            # A2: the generation check belongs INSIDE this locked section, not before the call.
            # `_restore` does its file read unlocked (by design — the lock covers structure, not
            # I/O), so a `clear()` can land at ANY point up to here; checking before `_admit`
            # leaves the window between that check and this lock, which is exactly where the
            # second audit's repro pauses. `gen=None` means the caller has no opinion.
            if gen is not None and gen != self._generation:
                return None
            old = self._remove(key)                    # replace: drop any prior entry's accounting first
            # a fresh key lands at the OrderedDict tail (MRU); _enforce_ram_budget evicts from the front
            dev = str(frozen.device)
            prev_home = home if home is not None else (old.home if old is not None else dev)
            admitted = _Entry(frozen, stamp, storage_bytes, dev, canvas,
                              orig_dtype, prev_home, quality)
            self._ram[key] = admitted
            self._bytes_by_dev[_dev_bucket(dev)] += storage_bytes
            # Residency first, then the total budget. The order is the point of having two
            # ladders: demoting a cold CUDA frame to host RAM costs 10.8 ms at 4K and keeps it
            # servable, where spilling it costs 204 ms and makes the next hit a 36 ms disk read.
            # Relieving VRAM the cheap way BEFORE deciding what to evict entirely means a frame
            # only reaches the disk when host RAM is full too.
            self._enforce_residency()
            self._enforce_ram_budget()
        self._drain_demotes()         # D2H outside the lock (see the rule on __init__)
        self._drain_spills()          # disk I/O outside the lock (same rule)
        # A2: hand the ENTRY back. `orig_dtype` is immutable for an entry's life (only `tensor`
        # and `device` are ever mutated, by the residency ladder), so a caller that captured the
        # entry here can read the representation without a second lookup — and a second lookup
        # is exactly what could return `None` after a racing `clear()`.
        return admitted

    def get(self, key: str, *, copy: bool = True):
        """Return the cached frame for `key`, or None. By DEFAULT (copy=True) returns an OWNED
        COPY — copy-on-read — so a consumer's in-place write can never reach the stored master.
        This is the load-bearing safety guarantee, because a frozen (inference) frame is NOT
        write-proof on torch 2.12+: an in-place op LANDS the write and THEN raises, so the raise
        does not roll it back, and a shared frozen buffer handed straight back would be silently
        corrupted and re-served. Pass copy=False for a read-only consumer that promises not to
        mutate the result — the fast zero-copy path, mirroring to_dlpack(copy=False).

        A RAM hit is version-verified (ENG-12 stratum 2) before serving — a NORMAL (host-supplied)
        entry written through in place is dropped, never served (a frozen master's version is a
        constant 0, so this net catches only mutable entries; copy-on-read is what protects the
        frozen master). A RAM miss falls to the disk spill tier and re-admits a restored frame.

        PREC-1: an entry STORED at reduced precision is returned at the dtype it was cooked at,
        so storage precision is invisible here. That unpack is a copy, so `copy=False` cannot
        hand back the master for such an entry — it returns an owned upcast instead. `copy=False`
        promises the CALLER will not mutate; it never promised object identity, and the one place
        identity is asserted (test_v025_phase1) stores at full precision, where the master is
        still handed back untouched."""
        from . import tex_engine
        orig_dtype = None
        demoted = None
        with self._lock:
            frame = None
            entry = self._ram.get(key)
            if entry is not None:
                tensor, stamp = entry.tensor, entry.stamp
                if not tex_engine.verify_unmutated(tensor, stamp):
                    self._remove(key)                  # a mutable entry written through: never serve it
                else:
                    self._ram.move_to_end(key)
                    frame, orig_dtype = tensor, entry.orig_dtype
                    # CACHE-8: a hit on a demoted frame IS the reuse signal the residency
                    # policy promotes on. Noted here, acted on below — the H2D copy is exactly
                    # the full-frame work the lock rule keeps outside.
                    if entry.device != entry.home:
                        demoted = entry
        if demoted is not None:
            frame = self._promote(key, demoted)
        if frame is None:
            # OUTSIDE the lock: `_restore` is a file read plus an H2D copy — measured at
            # hundreds of ms for a large frame, and holding the lock across it stalls every
            # concurrent `get` and `put` behind a disk seek. It re-admits through `put`, which
            # takes the lock itself, so the insert is still serialized. Two threads racing the
            # same key both read and the second `put` simply replaces the first: a duplicated
            # read, never a corrupted table.
            # Returns (master, orig_dtype) together — see A2 on `_restore`. Never re-look-up.
            frame, orig_dtype = self._restore(key)
        with self._lock:
            if frame is None:
                self.misses += 1
                return None
            self.hits += 1
        # The clone is OUTSIDE the lock, deliberately. It is the expensive part of a hit (~3 ms
        # for a 1024²×4 frame), and holding the lock across it would make an interactive `get`
        # wait behind a CACHE-7 phase-2 `put` for the length of a memcpy. Safe because `frame`
        # is a local strong reference: a concurrent evict can drop the entry from `_ram`, but
        # the tensor cannot be freed while this name holds it, and a frozen master is never
        # written in place (that is what the freeze and the copy-on-read exist to guarantee).
        # clone() on a frozen master (outside inference_mode) yields a normal, mutable, owned copy.
        #
        # PREC-1: for a reduced-precision entry the upcast IS that copy — `.to(fp32)` on an fp16
        # master allocates and writes a fresh normal buffer exactly as `clone()` would, so the
        # unpack REPLACES the clone rather than adding to it (measured 0.04 ms vs 0.03 ms CPU at
        # 1024²). `copy=False` cannot avoid it, so it does not try: an unpack is a copy whether or
        # not one was asked for.
        if orig_dtype is not None:
            from . import tex_packing
            out = tex_packing.unpack(frame, orig_dtype)
            # `unpack` is the identity when there is nothing to convert, and the identity would
            # hand back the frozen MASTER — the one thing copy-on-read exists to prevent. An
            # entry with `orig_dtype` set should never be in that state, so this is a guard
            # against a future codec whose unpack is a no-op, not against today's two.
            return out if out is not frame else (frame.clone() if copy else frame)
        return frame.clone() if copy else frame

    def set_budget(self, mb) -> None:
        """Change the RAM byte budget and enforce it NOW.

        The public seam GOV-1's profiles set a frame budget through. `_budget` is otherwise a
        constructor argument, which would mean a preset could only reach caches created after
        it was chosen — an ordering trap for a host that builds its caches at startup and lets
        the user pick a profile afterwards. Enforcing immediately matters because the tightening
        direction is the one with a consequence: a shrunk budget that waits for the next `put`
        leaves the cache over its stated cap for as long as the session is idle, which is
        exactly when a user switches to `efficient` to get memory back."""
        with self._lock:
            self._budget = max(0, int(mb) * (1 << 20))
            self._enforce_ram_budget()
        self._drain_spills()      # `_enforce_ram_budget` only QUEUES the victims (see it)

    def patch_region(self, key: str, patch, window, *, base=None,
                     base_key: str | None = None, canvas=None,
                     quality=None, storage=None):
        """CACHE-9: write a cooked REGION into a frame and store the result under `key`.

        Returns the patched frame (owned, mutable), or None when there is no base to patch —
        in which case the caller must cook whole, which is always correct.

        COPY-ON-PATCH, never in place. A cached frame is frozen under ENG-12, and freezing is a
        TRIPWIRE, NOT A FENCE: on torch 2.12 an in-place op on an inference tensor LANDS the
        write and then raises, so the raise is not a rollback and a "protected" master would be
        silently corrupted and re-served. The copy here is not a new cost invented for CACHE-9 —
        it is `get`'s existing `copy=True` contract used correctly.

        The base is addressed by `base_key` (a cached frame) or handed over directly as `base`
        (a frame the caller already owns — a chain's previous stage, typically). Passing `base`
        avoids a redundant clone: the caller states it owns that buffer.

        PROVENANCE lives in the KEY, not in a version stamp. `ResultCache` entries do carry
        `frame_version`, but that is `t._version`, and `frame_version` returns a CONSTANT 0 for
        any inference (frozen) tensor — and `put` always freezes. So every entry is stamped 0
        and the stamp is structurally incapable of saying "this frame is a patched descendant of
        that one". CACHE-1 already has the slot: the caller mints `key` with the base's key in
        `upstream` and the window in `canvas`, which is a real provenance chain that survives a
        spill/restore and needs no new entry type. This method records the window it wrote so
        `stats`/debugging can see it; it does not mint keys, because minting is CACHE-1's job.

        `window` is a 6-tuple `(x0, y0, w, h, W, H)` — the same shape `CookResult.cooked_roi`
        reports. A caller MUST pass what the engine actually served (`cooked_roi`), not what it
        requested: a stage can DECLINE a window and return a whole frame, and pasting that into
        a w×h slice is a crash. `cooked_roi is None` means "declined" — `put` the frame instead.

        `quality`/`storage` (PREC-1) forward to the `put` of the patched result, and `quality`
        is RATCHETED DOWN BY THE BASE'S OWN TIER when the base is addressed by key: patching a
        preview-stored base yields a preview-stored result even if you pass nothing. Preview is
        viral, and this is the one seam where the cache can SEE the upstream rather than having
        to trust the host to apply the rule — a patch cannot be more faithful than the frame it
        is written into, so a final-shaped key over half-quantized pixels is the one outcome
        this must not produce. It only ratchets DOWN: a final base with no tag stays final.

        The rule cannot apply when you pass `base=` WITHOUT `base_key=` — a bare tensor carries
        no tier, and the cache will not guess one.

        And it is BEST-EFFORT when you pass `base=` WITH `base_key=`: the tag is read from the
        live entry, and supplying `base=` is precisely what skips the `get` that would restore
        a spilled one, so a base that has aged out to the disk tier propagates nothing. Reading
        it back would mean an unpickle of the whole frame to recover one string, which defeats
        the reason you passed `base=`. **If the tier matters and the base may not be resident,
        pass `quality=` explicitly** — that is the spelling with no failure mode.

        *(Reversed in v0.33.2. This paragraph used to say the tier was NOT inherited and that
        passing nothing stored at full precision "which is the safe direction". It was not: it
        stored a frame whose out-of-window bytes still carried the base's fp16 quantization
        under a key that claimed final — measured 1.89e-03, half the 8-bit display quantum.)*
        """
        import torch
        if not isinstance(patch, torch.Tensor):
            return None
        # ONE lock acquisition around the whole read-modify-write. Composing public `get` +
        # `put` is per-call safe and still not ATOMIC: two threads patching the same base both
        # read it, both write their own window, and the second `put` discards the first's — a
        # lost update, not a race the type system can catch. An earlier draft carried this
        # comment WITHOUT the `with` below and measured 200/200 lost updates. RLock, so the
        # nested `get`/`put` re-enter freely.
        with self._lock:
            out = self._patch_region_locked(key, patch, window, base, base_key, canvas,
                                            quality, storage)
        # The nested `put` deferred these — `_lock.depth` was non-zero inside the block above —
        # so they would not run a disk write and a D2H under the lock this method holds for
        # atomicity. They run here, released, exactly as they do after a plain `put`.
        self._drain_demotes()
        self._drain_spills()
        return out

    def _patch_region_locked(self, key, patch, window, base, base_key, canvas,
                             quality=None, storage=None):
        # A4 (v0.33.2): PREVIEW IS VIRAL, and this is the one seam where the cache itself can
        # SEE the upstream — `propagate_quality` is otherwise a rule the host has to apply,
        # because TEX cannot recover a tag from an opaque `upstream` SHA. Here the base is a
        # cache entry with a recorded tag, so laundering is preventable rather than merely
        # documented: patching an fp16-STORED base while passing `quality=None` produced a
        # result stored full-fp32 under a final-shaped key whose out-of-window bytes still
        # carried fp16 quantization (measured 1.89e-03) — silently mixed fidelity.
        from . import tex_packing
        frame = base if base is not None else self.get(base_key or key, copy=True)
        if frame is None:
            return None                        # nothing to patch — the caller cooks whole
        # The lookup runs AFTER the `get` above, and only for a base addressed BY KEY. Both
        # halves were wrong in the first draft of this fix, and both were measured:
        #
        #   * BEFORE the `get`, a base sitting on the disk tier is not in `_ram` yet, so no tag
        #     propagated and a spilled preview base was laundered exactly as before the fix.
        #     The `get` is what restores the entry — reading `_ram` first asks the question one
        #     step too early.
        #   * `self._ram.get(base_key or key)` on the EXPLICIT-`base` path resolves to `key`,
        #     the DESTINATION being overwritten, not the base at all. A stale preview entry
        #     under that key then forced every later patch of it to preview, permanently, since
        #     `propagate_quality` only ratchets downward. The documented host shape walks into
        #     this: `patch_region(f"s{i}", out, served, base=canvas[i])` re-patches one stable
        #     key per stage on every edit (`benchmarks/region_recook_bench.py`).
        #
        # So the tag comes from the entry the frame CAME FROM, mirroring the line above it: the
        # explicit `base_key` when given, else the key we just read, and NOTHING when the caller
        # handed over a bare tensor — a tensor carries no tier, and inventing one from whatever
        # happens to sit under the destination key is how the second bug above happened.
        tag_key = base_key if base_key is not None else (None if base is not None else key)
        if tag_key is not None:
            src = self._ram.get(tag_key)
            # ...and only when the base's BYTES are actually reduced (`orig_dtype is not None`).
            # The tag alone is not sufficient authority to pack, because `choose_storage` refuses
            # several preview-tagged frames and the entry keeps the tag anyway: a MASK or LATENT
            # (`kind` in DATA_KINDS — data planes are NEVER packed) and any frame outside the
            # fp16 range both store full while reading `quality='preview'`. `patch_region` has no
            # `kind` parameter, so its nested `put` passes `kind=None` and `choose_storage` would
            # treat the ratcheted tag as licence to pack — quantizing a mask the host had
            # explicitly protected, through a rule written to PREVENT fidelity loss. Gating on
            # the stored representation keeps the rule exactly as strong as its justification:
            # a patch is never more faithful than its base, and never less faithful either.
            if src is not None and src.orig_dtype is not None:
                quality = tex_packing.propagate_quality(quality, (src.quality,))
        x0, y0, w, h = (int(v) for v in tuple(window)[:4])
        if patch.shape[1:3] != (h, w):
            # The engine served a different extent than the window claims — a declined window
            # reported as served, or a caller passing its REQUEST instead of `cooked_roi`.
            # Refusing beats writing the wrong pixels into a frame that then looks authoritative.
            return None
        if frame.shape[1:3] != (int(tuple(window)[5]), int(tuple(window)[4])):
            return None                        # base is not the frame this window describes
        # A5/PROBE-11: the window describes the SPATIAL extent only, so a base and a patch that
        # disagree on batch or channels reached the assignment below and raised a RuntimeError
        # out of a method whose whole contract is "refuse by returning None". Refusing is not a
        # nicety here: `patch_region` is documented as the call a host makes when it is unsure
        # whether a region cook is serviceable, so it must never be the thing that raises.
        if frame.shape[0] != patch.shape[0] or frame.shape[3:] != patch.shape[3:]:
            return None
        # `base` may be a frozen buffer the caller merely holds a reference to (a cook output
        # arrives frozen), so clone unless we know it is ours. A `get(copy=True)` above already
        # returned an owned clone, which is why this only guards the explicit-`base` path.
        if base is not None:
            from . import tex_engine
            if tex_engine.is_frozen(frame):
                frame = frame.clone()
        frame[:, y0:y0 + h, x0:x0 + w] = patch
        self.put(key, frame, canvas=(canvas if canvas is not None
                                     else {"shape": list(frame.shape),
                                           "roi": [x0, y0, w, h]}),
                 quality=quality, storage=storage)
        return frame

    # ── CACHE-8: residency (VRAM -> host RAM -> disk) ──
    def set_vram_budget(self, mb) -> None:
        """Arm the residency tier: above `mb` megabytes of CUDA-resident frames, the coldest
        are moved to host RAM. `None` disarms it, which is v0.32's behaviour exactly.

        Why a SECOND budget rather than a smarter single one: the two are different resources
        with different prices. `_budget` caps how much the cache holds AT ALL and its overflow
        goes to disk; this caps how much of that sits in VRAM, and its overflow goes to a place
        that is still a cache hit. A frame cache on a CUDA host is competing with the cook
        itself for VRAM while host RAM sits empty beside it — one number cannot express that.
        """
        with self._lock:
            self._vram_budget = None if mb is None else max(0, int(mb) * (1 << 20))
            if self._vram_budget is None:
                # A5 (v0.33.2): DISARM MEANS OFF, INCLUDING WORK ALREADY DECIDED ON. Victims
                # queued under the old ceiling would otherwise keep draining after this returns,
                # so a cache the host had just switched off went on moving frames cross-device —
                # contradicting the line above it ("`None` disarms it, which is v0.32's
                # behaviour exactly"; v0.32 never moves a frame between devices). Measured on
                # CUDA: `demotions` went 0 -> 1 with `vram_budget_bytes=None`.
                # Only the residency tier queues these (`evict_bytes` demotes solely when the
                # budget is set), so dropping the queue cannot strand a governor request.
                self._pending_demotes.clear()
            self._enforce_residency()
        self._drain_demotes()

    def _enforce_residency(self) -> None:
        """Queue the coldest CUDA frames for demotion until VRAM is under budget.

        POLICY, stated plainly because it is the part worth arguing with: victims are chosen
        by LRU, and a demoted frame is PROMOTED back on its next hit. That is a recency policy
        being used where the report asks for an access-FREQUENCY one, and the justification is
        that a frame cache's access pattern is a playhead — a scrub touches near-frames most
        recently and most often, so the two orderings largely coincide. `stats()` reports the
        demotion/promotion counts precisely so this can be revisited with a measurement instead
        of an opinion; a frequency-weighted victim choice is a change to this function alone.

        Never demotes the MRU entry (`len > 1` and the front-first walk), for the same reason
        `_enforce_ram_budget` does not: the frame just cooked is the one about to be read.

        Caller holds `_lock`."""
        if self._vram_budget is None:
            return
        over = self._bytes_by_dev["cuda"] - self._vram_budget
        if over > 0:                                  # O(1) early-out, the common case
            self._queue_demotions(over)

    def _queue_demotions(self, want: int) -> int:
        """Queue the coldest CUDA entries until ~`want` bytes are accounted for; returns the
        bytes queued. Caller holds `_lock`.

        THE one place a demotion victim is chosen. `_enforce_residency` (which wants VRAM back
        under a ceiling) and `evict_bytes` (which wants a byte count back for the governor)
        differ only in where that number comes from — and keeping the walk in one place is what
        makes `_enforce_residency`'s promise true when it says a frequency-weighted victim
        choice is a change to one function.

        Bytes are charged when a victim is QUEUED, not when the drain frees them: the drain is
        unconditional from here, and counting an entry twice would demote the world.

        A1: the skip-set is the queue UNION the in-flight set. A victim `_drain_demotes` has
        already popped is no longer in `_pending_demotes` but its copy has not landed — it is
        still on CUDA, still matches the walk, and re-queuing it gets it demoted TWICE. Both
        drains then commit the same cuda→cpu byte transfer, `_bytes_by_dev["cuda"]` goes
        NEGATIVE and stays skewed, `governed_bytes()` feeds the CACHE-5 governor garbage, and
        `_enforce_residency` stops firing forever because `over` can no longer be positive.
        Measured on a natural two-thread race over one 64 MiB frame: `{'cuda': -67107840}`
        against an actual 1024, with `demotions=2` for a single frame."""
        queued = {k for k, _e in self._pending_demotes} | self._demoting
        got = 0
        # A7: the MRU entry is the frame just cooked — the one about to be read. Excluding it
        # by KEY is what the docstring always promised; the `len(self._ram) <= 1` guard it used
        # to rely on is dead code here, because this function never removes an entry, so on any
        # multi-entry cache the walk reached the newest frame whenever `want` covered the older
        # CUDA bytes. Deterministic at `set_vram_budget(0)`: two puts demoted BOTH, including
        # the one whose `put` had just returned, which is then promoted back on its next hit —
        # ~22 ms of pointless copies per cook at 4K.
        mru = next(reversed(self._ram), None) if self._ram else None
        for key in list(self._ram.keys()):            # oldest -> newest
            if got >= want:
                break
            if key == mru:
                continue
            entry = self._ram.get(key)
            if entry is None or _dev_bucket(entry.device) != "cuda" or key in queued:
                continue
            self._pending_demotes.append((key, entry))
            queued.add(key)
            got += entry.nbytes
        return got

    def _drain_demotes(self) -> None:
        """Perform the queued D2H copies and swap the host buffers in. Called by the PUBLIC
        methods, after they release the lock.

        Unlike a spill, a demotion must leave the frame SERVABLE throughout — so the entry is
        never removed. It stays in `_ram`, on CUDA, answering `get` with the right pixels until
        the host copy exists; only then is slot 0 swapped and the per-device accounting moved.
        A concurrent `get` during the copy therefore serves the VRAM master, which is correct
        and is why this is not a window anyone has to reason about.

        XPU-2 is deliberately used in its `retained=True` mode here. A demoted frame's host
        buffer is RETAINED — for as long as the entry lives — and a page-locked buffer of that
        lifetime is a slow leak of unswappable memory. Copying into pinned and then cloning to
        pageable to release the lock would cost a second full host memcpy of the frame,
        which is more than the asynchrony saves on a copy that has almost nothing to overlap
        with. The handle is still the seam: when v0.34's async-write path hands a demoted frame
        to a writer thread, it does so through this same object.

        A demotion that fails leaves the frame exactly where it was — over budget, and correct.
        The budget is a target; the pixels are not."""
        import torch
        from .tex_runtime.streams import egress
        if self._lock.depth:
            return                # a composite (patch_region) holds the lock; it drains after
        while True:
            with self._lock:
                if not self._pending_demotes:
                    return
                key, entry = self._pending_demotes.popleft()
                if self._ram.get(key) is not entry or _dev_bucket(entry.device) != "cuda":
                    continue                          # evicted, replaced, or already demoted
                # A1: OFF the queue but not yet committed — `_queue_demotions` must still see it
                # as spoken for, or it re-queues a frame that is still on CUDA and a second drain
                # commits the same byte transfer.
                self._demoting.add(key)
                src = entry.tensor
            try:
                # Born frozen: slot 0's contract is a FROZEN master (`_spill` reads it as one).
                with torch.inference_mode():
                    host = egress(src, retained=True).tensor()
            except Exception:
                with self._lock:
                    self._demoting.discard(key)
                continue                              # leave it in VRAM; correctness is unharmed
            with self._lock:
                self._demoting.discard(key)
                cur = self._ram.get(key)
                # A1: identity AND device, mirroring `_promote`. Identity alone is not enough —
                # two drains can hold the SAME entry object, both find it unchanged, and both
                # apply the transfer. The device is what says whether the move already happened,
                # and it is the field the transfer itself mutates, so re-reading it under the
                # lock is the check that cannot be raced.
                if cur is not entry or _dev_bucket(cur.device) != "cuda":
                    continue                          # it moved on while we copied: drop the copy
                if self._vram_budget is None:
                    # A5, and LOAD-BEARING rather than belt-and-braces: `set_vram_budget(None)`
                    # empties `_pending_demotes`, but a victim THIS drain already popped is in
                    # neither the queue nor the clear's reach — it is mid-`egress`, holding only
                    # a local reference. Same argument `clear()` spells for spills ("a victim
                    # `_drain_spills` ALREADY popped is not in either queue"). Delete this and
                    # the disarm still lets one frame cross devices with the tier off.
                    continue
                self._bytes_by_dev["cuda"] -= entry.nbytes
                self._bytes_by_dev["cpu"] += entry.nbytes
                cur.tensor = host
                cur.device = "cpu"
                self.demotions += 1

    def _promote(self, key: str, entry):
        """Move a demoted frame back to its home device and return the promoted master, or the
        entry's current tensor if it cannot be moved.

        Called from `get` on a hit, OUTSIDE the lock — it is an H2D copy (11.1 ms at 4K),
        exactly the class of work the lock rule excludes. The re-entry check under the lock is
        what makes that safe: if the entry changed while we copied, the copy is discarded."""
        # A5(d) WITHDRAWN in v0.33.2 — the lock-depth early-out that stood here is deferred, not
        # forgotten (DEVELOPMENT.md carries the row). It read:
        #
        #     if self._lock.depth: return entry.tensor
        #
        # and its argument was sound as far as it went: the drains refuse to run a full-frame
        # copy while a composite holds the lock, and `_promote` is an H2D (11.1 ms at 4K) that
        # `patch_region` reaches at depth 1. What it missed is that `_promote` is not a drain.
        # A drain DEFERS — the queue survives and `patch_region` runs it on release. This
        # DEGRADED: it silently handed back the host copy, and there is no `_pending_promotes`
        # to make good on it. Measured consequence, not a worry:
        #
        #     base demoted to host RAM (home=cuda:0) -> patch_region -> result device=cpu,
        #     result HOME=cpu, promotions=0
        #
        # `frame[...] = patch` accepts a CUDA source into a CPU destination (`copy_` is
        # cross-device), so nothing raises; `_admit` then records `home="cpu"` for the fresh
        # destination key because that is where the frame it was handed actually lives. The
        # patched frame has left the residency ladder permanently, and every stage downstream of
        # it inherits a CPU home — precisely the "one-way trip to the CPU" `_Entry.home` was
        # introduced to prevent. Trading a latency problem for a residency-correctness one is a
        # bad trade, and closing it properly needs a promote QUEUE plus home propagation through
        # `_patch_region_locked`'s nested put — two coupled changes that do not belong in a
        # patch release. So the H2D runs under the lock here exactly as it did in v0.33.1.
        import torch
        src, home = entry.tensor, entry.home
        try:
            with torch.inference_mode():
                dev = torch.device(home)
                moved = torch.empty(src.shape, dtype=src.dtype, device=dev)
                moved.copy_(src)
        except Exception:
            return src                                # stay on the CPU; a hit is still a hit
        with self._lock:
            cur = self._ram.get(key)
            if cur is not entry:
                # The entry was replaced or evicted while we copied. `moved` still holds THIS
                # entry's pixels at THIS entry's representation, which is what `get`'s captured
                # `orig_dtype` describes — handing back the new entry's tensor instead would
                # pair one frame's bytes with another frame's unpack.
                return moved
            if cur.device == home:
                return cur.tensor                     # another thread promoted it first
            self._bytes_by_dev[_dev_bucket(cur.device)] -= cur.nbytes
            self._bytes_by_dev[_dev_bucket(home)] += cur.nbytes
            cur.tensor = moved
            cur.device = home
            self.promotions += 1
            return moved

    def _drain_spills(self) -> None:
        """Write out entries evicted under the lock. Called by the PUBLIC methods, after they
        release it.

        The lock rule on `__init__` says disk I/O happens outside it. That was a claim, not a
        fact: `_enforce_ram_budget` -> `_spill` ran inside `put`'s lock and `_restore` inside
        `get`'s, so a concurrent `get` could block for the length of a pickle write plus a D2H
        copy — measured at 327-496 ms. Eviction still happens under the lock (it is pure
        bookkeeping); only the write is deferred, so the entry is out of `_ram` and unreachable
        by the time anyone waits on nothing.

        A frame whose spill fails is simply gone, which is the pre-existing contract — a miss
        recooks."""
        if self._lock.depth:
            return                # a composite (patch_region) holds the lock; it drains after
        while True:
            with self._lock:
                if not self._pending_spills:
                    return
                key, entry, seq = self._pending_spills.popleft()
            # Both producers (`_enforce_ram_budget`, `evict_bytes`) queue only entries `_remove`
            # returned non-None, so there is nothing to guard against here. The ticket travels
            # WITH the victim: claiming it below, after this lock is released, would order the
            # writes by which drain happens to start first rather than by which `put` evicted.
            self._spill(key, entry, seq)

    def _remove(self, key: str):
        """Pop `key` and undo ALL of its accounting. Returns the entry, or None.

        THE ONLY place an entry leaves `_ram`. It exists because the accounting was spelled at
        each removal site and one of them — `_enforce_ram_budget` — decremented `_ram_bytes`
        and forgot `_bytes_by_dev`. That is not a cosmetic drift: `governed_bytes(dev_type)` is
        what `arbitrate()` reads, so the per-device total ratcheted up on every eviction and the
        governor was told the cache held 16000 MB when it held 1984. It then evicted to the
        one-entry floor — a 16x over-eviction that destroys exactly the frame reuse CACHE-2 and
        CACHE-9 exist to provide, while *looking* like the governor landing on budget.

        Caller holds `_lock`."""
        entry = self._ram.pop(key, None)
        if entry is not None:
            self._bytes_by_dev[_dev_bucket(entry.device)] -= entry.nbytes
        return entry

    def _claim_spill_ticket(self, key: str) -> int:
        """Claim this key's next write ticket. CALLER HOLDS `_lock`, and that is the whole point.

        The ticket must be claimed where the EVICTION is decided, not where the write starts.
        v0.33.2 first claimed it at `_spill`'s own (separate) lock acquisition, and the comment
        there asserted "the later `put`'s spill always holds the higher number regardless of
        which write finishes first" — which was false, because `_drain_spills` pops the victim
        under `_lock`, RELEASES it, and only then calls `_spill`. Two acquisitions, so pop order
        did not imply ticket order. Measured: thread A pops (K, v1) and is descheduled in that
        gap; the main thread runs `put(K, v2)` and its whole spill, taking ticket 1; A resumes,
        takes ticket 2 with the OLDER pixels, passes every check, and overwrites the winner —
        `get(K)` then serves v1 forever, after both puts and both spills returned. That is the
        A1 defect surviving the A1 fix, and the A1 test could not see it because its gate sits
        inside `egress`, which `_spill` reaches AFTER the old claim site.

        Claimed here, the ticket is ordered by the `put` that caused the eviction, under the
        lock that serialized those puts — which is what the ordering claim needs to be true."""
        seq = self._spill_seq.get(key, 0) + 1
        self._spill_seq[key] = seq
        return seq

    def _enforce_ram_budget(self) -> None:
        """Spill LRU victims to disk until under the RAM byte budget. Never evicts the entry
        just inserted (it is newest / move_to_end'd), so a single frame larger than the whole
        budget still serves once — it simply gets spilled on the next insert."""
        while self._ram_bytes > self._budget and len(self._ram) > 1:
            old_key = next(iter(self._ram))
            entry = self._remove(old_key)
            self.evictions += 1
            # QUEUED, not written — the caller must `_drain_spills()` after releasing the lock.
            # Every public method that can reach here does: put, evict_bytes, set_budget.
            # The write ticket is claimed HERE, under this lock (see `_claim_spill_ticket`).
            self._pending_spills.append((old_key, entry, self._claim_spill_ticket(old_key)))

    # ── disk spill / restore ──
    def _spill(self, key: str, entry, seq: int) -> None:
        """Write an evicted frame to disk (best-effort), then atomically pickle it under results/.
        A failed spill just drops the frame — the cook reproduces it. (Page-locking is applied on
        the RESTORE side, not here: pinning is not preserved through pickle, so a pinned spill
        buffer would deserialize pageable anyway.)"""
        try:
            import torch
            with self._lock:
                gen = self._generation          # A5: the generation this write belongs to
                wlock = self._spill_locks.get(key)
                if wlock is None:
                    wlock = self._spill_locks[key] = threading.Lock()
            tensor = entry.tensor
            # A normal, contiguous CPU copy for serialization in ONE host-visible copy. A frozen
            # (inference) entry can't be pickled, and `.to("cpu")` on an already-CPU frame is a
            # no-op view — so a naive `.to("cpu").clone()` costs two host memcpys on CPU frames and
            # a redundant H2H clone on CUDA ones. Copy into a fresh normal (non-inference) buffer
            # instead: empty() outside inference_mode is mutable/picklable, and one copy_ does the
            # (D2H or H2H) move — the clone that only existed to strip the inference flag is gone.
            #
            # XPU-2: for a CUDA frame that copy is a D2H, and `egress` issues it asynchronously
            # so `_disk_path`/`getsize` overlap it. The fence is `handle.tensor()` below, before
            # a single byte is read — which is the whole contract, and the reason this is the
            # one D2H in the tree that is allowed to be non-blocking (a bare tensor has no way
            # to say "not yet"; a handle has nowhere to be read from that does not wait first).
            from .tex_runtime.streams import egress
            handle = egress(tensor)
            # PREC-1: `orig` travels with the frame. Without it a restored half frame would be
            # served AS half — the dtype change `get` exists to hide would leak out through the
            # disk tier only, i.e. intermittently, on eviction, which is the worst shape a bug
            # can have. It also means a preview frame occupies half the DISK too, for free.
            # `device` is the HOME device, not the one the frame happens to be sitting on: a
            # demoted CUDA frame that then spills must come back from disk as a CUDA frame, or
            # the residency ladder would quietly turn into a one-way trip to the CPU.
            path = self._disk_path(key)
            # A key can spill again after a restore (restore leaves the .frame on disk), so the
            # write may OVERWRITE an existing file — measure the old size first and apply the NET
            # delta, or the running total over-counts and forces needless reconciling scans.
            # This is also the work that now overlaps the in-flight D2H: two filesystem syscalls,
            # issued while the DMA engine copies.
            prev = 0
            if self._disk_bytes is not None:
                try:
                    prev = os.path.getsize(path)
                except OSError:
                    prev = 0
            cpu_t = handle.tensor()             # ← THE FENCE. Nothing above reads the bytes.
            # A4: torch 2.10 PICKLES `torch.uint16` at every protocol and then cannot LOAD it
            # ("UntypedStorage has no attribute 'dtype'"). So the spill "succeeded" — file
            # written, bytes charged, key indexed — while `_restore` could never read it back:
            # every uint16 frame was permanently lost on eviction, and the dead `.frame` leaked
            # the disk budget forever because even the stale-epoch cleanup is unreachable (the
            # load raises before the epoch is compared).
            #
            # Store the same bits as `int16` and re-view on restore. Bit-identical (verified),
            # costs nothing, and keeps uint16 frames spillable rather than declining them.
            viewed = None
            if cpu_t.dtype is torch.uint16:
                cpu_t, viewed = cpu_t.view(torch.int16), "uint16"
            rec = {"t": cpu_t, "fmt": _FRAME_FORMAT,
                   "device": entry.home, "canvas": entry.canvas, "epoch": env_epoch(),
                   "orig": _dtype_tables()[0].get(entry.orig_dtype), "viewed": viewed,
                   "quality": entry.quality}
            # A1: the check and the write are ONE critical section, per key. Outside it, a
            # stale writer that has already passed a check can still overwrite the winner.
            with wlock:
                with self._lock:
                    if self._spill_seq.get(key, 0) != seq or self._generation != gen:
                        return            # a newer spill of this key won; touch nothing
                if not _atomic_pickle(path, rec):
                    # `atomic_write` reports failure by RETURN, and this discarded it: the
                    # counters advanced, the key was indexed, and `_disk_bytes` was adjusted for
                    # a file that was never written — or, worse, for a key whose PREVIOUS
                    # `.frame` is still sitting there and now gets served as the current one.
                    # That is A1's failure mode reached through a full disk instead of a race.
                    return
            self.spills += 1
            # P0-6: the INDEX MUTATIONS GO UNDER THE LOCK. The file write above stays outside it
            # (that is the whole point of the drain path), but `_spilled` and `_disk_bytes` are
            # shared state and were being mutated with no lock at all — so `reindex_disk`, whose
            # scandir also runs unlocked by design, could scan, miss this key, and then REBIND
            # `_spilled` to the scanned set. The frame is then on disk, absent from the
            # membership set, and `_restore` short-circuits on that set without ever stat-ing:
            # unserveable, unreachable by the epoch cleanup, and `_disk_bytes` under-counts it
            # by exactly its size, forever. Demonstrated deterministically.
            #
            # `reindex_disk` additionally MERGES rather than rebinds (see it), so the two halves
            # of the fix close the window from both sides.
            with self._lock:
                if self._generation != gen:
                    # A5: `clear()` ran while this write was in flight. The frame the user
                    # cleared must not come back — drop the file we just wrote and index
                    # nothing. Done under the lock so the check and the decision are atomic
                    # against another `clear()`.
                    stale = True
                else:
                    stale = False
                    if self._spilled is not None:
                        self._spilled.add(key)   # unknown (None) stays unknown, not {this key}
                    if self._disk_bytes is not None:    # keep the running total current
                        try:
                            self._disk_bytes += os.path.getsize(path) - prev
                        except OSError:
                            self._disk_bytes = None     # lost track: force a reconciling scan
            if stale:
                try:
                    os.remove(path)
                except OSError:
                    pass
                return
            self._enforce_disk_budget()
        except Exception:
            pass

    def _restore(self, key: str):
        """Load a spilled frame back to its cook device and re-admit it to the RAM tier. The H2D
        restore rides the DMA engine (non_blocking) when the staged host copy is pinned."""
        try:
            import torch
            import pickle
            if self._spilled is None:
                # Usually one scandir, ONCE, instead of a stat per miss. Not guaranteed
                # any more: H4 makes a scan that raced a spill leave membership UNKNOWN
                # rather than definite-and-wrong, so under sustained spilling this can
                # re-walk on the next miss. Correctness first; the cost is a recorded
                # deferral (DEVELOPMENT.md) with `_spilled_since` as the way out.
                self._learn_spilled()
            if self._spilled is not None and key not in self._spilled:
                return None, None          # not on the tier — no syscall needed
            path = self._disk_path(key)
            if not os.path.exists(path):
                return None, None
            with self._lock:
                if self._purge_depth:
                    # H2: a destructive walk is in progress over this directory. `clear(disk=
                    # True)` bumps the generation under the lock and then unlinks UNLOCKED (N
                    # `os.remove`s must not block every `get`), so a restore starting now holds
                    # a generation that MATCHES and a file the walk has not reached yet.
                    #
                    # Declined HERE, where the window OPENS — not at the re-admit, where the
                    # first draft of this fix put it. That draft closed only the case where
                    # `_admit` also ran inside the walk; a restore that read its file inside the
                    # window and reached `_admit` after the walk finished found the flag already
                    # cleared and the generation still matching, and re-admitted the purged
                    # frame. Measured with the whole `clear()` returned. Refusing at capture also
                    # saves the doomed unpickle and H2D, and it makes the `_admit` clause
                    # unreachable — so that clause is deleted rather than left as a guard nothing
                    # can exercise.
                    return None, None
                # A2 (v0.33.2): the generation this READ belongs to. `clear(disk=True)` can run
                # while the file read below is in flight, and without this the restore re-admits
                # the cleared frame to RAM — and a later eviction re-SPILLS it to disk, so the
                # frame the user deleted comes back on both tiers. A5 closed this on the write
                # side; this is the same defect through the read-side door, and two independent
                # audits found it.
                #
                # Captured BELOW the membership short-circuit, not above it: a total miss is the
                # common case on this path (`cook_checkpointed` probes one key per cut against a
                # cache that usually has neither), and the whole point of the `_spilled` set is
                # that such a miss costs no syscall. Taking the lock to read an int we then throw
                # away put a 220 ns acquire back on it. Everything from here to `_admit`'s locked
                # re-check is still inside the window the check covers.
                gen = self._generation
            with open(path, "rb") as f:
                rec = pickle.load(f)
            if int(rec.get("fmt", 0) or 0) > _FRAME_FORMAT:
                # A5/PROBE-8: the `fmt` field was write-only — a record from a NEWER TEX was
                # decoded best-effort and served as pixels. A forward-compatible reader cannot
                # know what a future field means, so the only safe read of "newer than me" is
                # to decline. (Absence still reads as v0; that is the backward direction, which
                # IS decodable.) Left on disk, not deleted: the newer TEX that wrote it can.
                return None, None
            if rec.get("epoch") != env_epoch():     # a prior-environment frame: discard
                try:
                    os.remove(path)
                except OSError:
                    pass
                else:
                    # A6: UNDER THE LOCK. These are the same two index fields P0-6 moved into
                    # the lock inside `_spill`, reached through a door P0-6 did not close — and
                    # unsynchronised they lose to a racing `_spill`, whose locked `+=` can land
                    # after this `None` and leave `_disk_bytes` a definite WRONG value. The
                    # file's convention is that an uncertain total is `None`, never a confident
                    # lie; that only holds if the invalidation cannot be overwritten.
                    with self._lock:
                        self._disk_bytes = None      # out-of-band removal: invalidate the total
                        if self._spilled is not None:
                            self._spilled.discard(key)
                return None, None
            host = rec["t"]
            if rec.get("viewed") == "uint16":       # A4: undo the int16 storage view
                host = host.view(torch.uint16)
            dev = rec.get("device", "cpu")
            # Restore UNDER inference_mode so the device tensor is born frozen — then put()'s
            # freeze() is a no-op instead of a full-frame VRAM re-clone (which would also force a
            # sync of the non_blocking H2D, negating the pinned-DMA win this path exists for).
            with torch.inference_mode():
                if str(dev).startswith("cuda") and torch.cuda.is_available():
                    # Stage the pageable pickle tensor into a page-locked buffer (when worthwhile)
                    # so the H2D rides the DMA engine — the restore copy can then overlap a
                    # prefetching host's cook time (pillar 5). Falls back to a plain move.
                    from . import tex_marshalling as M
                    if M._pin_worthwhile(host):
                        try:
                            pinned = torch.empty(host.shape, dtype=host.dtype, pin_memory=True)
                            pinned.copy_(host)
                            tensor = pinned.to(dev, non_blocking=True)
                        except Exception:
                            tensor = host.to(dev)
                    else:
                        tensor = host.to(dev)
                else:
                    tensor = host.to(dev)
            self.restores += 1
            # `_admit`, not `put`: the representation was decided when this frame was spilled and
            # is recorded in the file. Re-running the policy would ask about a tensor whose
            # cooked dtype no longer exists to be asked about.
            # A NAME LOOKUP, not `getattr(torch, <string off disk>)`: the spill record is data,
            # and an unknown name resolves to "stored as cooked" rather than to whatever
            # attribute happens to answer to it.
            orig = rec.get("orig")
            # A2: `_admit` re-checks under ITS lock and refuses to insert if a `clear()` landed
            # while we were reading — the frame is gone by the user's instruction and a miss is
            # the correct answer.
            entry = self._admit(key, tensor, rec.get("canvas"),  # already frozen: no-op-freezes
                                _dtype_tables()[1].get(orig), home=dev, gen=gen,
                                quality=rec.get("quality"))
            if entry is None:
                return None, None
            # A2: the frame AND its representation, from the ONE locked re-admit. This used to
            # return only the tensor and let `get` re-look-up `orig_dtype` in a second lock
            # acquisition — a window in which a concurrent `clear()` or eviction drops the entry,
            # `orig_dtype` reads `None`, and an fp16-STORED frame is served at storage dtype to a
            # caller owed fp32. Reproduced without any patching, on the first iteration.
            # A2: the frame AND its representation, from the ONE locked re-admit. Re-looking
            # the entry up in a second acquisition is what let a `clear()` in between make
            # `orig_dtype` read None and serve fp16 to a caller owed fp32.
            return entry.tensor, entry.orig_dtype
        except Exception:
            return None, None

    def _enforce_disk_budget(self) -> None:
        """Cap the spill directory's total bytes, deleting oldest-first (mtime). A separate cap
        from the program-cache disk tiers (CACHE-0), because frames dwarf .pkl/.cg sidecars.

        Fast path: when the running `_disk_bytes` total is known and under budget, return in O(1)
        — the common case for a run of spills that never approaches the (multi-GB) cap. Only when
        the total is unknown or over budget do we `os.scandir` the dir (one syscall/entry for
        name+size+mtime), delete oldest-first, and reconcile `_disk_bytes` to the scanned total —
        so the whole-dir walk is O(entries) at most once per budget-crossing, not per spill."""
        if self._disk_bytes is not None and self._disk_bytes <= self._disk_budget:
            return
        try:
            files, total = [], 0
            with os.scandir(self._spill_dir()) as it:
                for e in it:
                    if not e.name.endswith(".frame"):
                        continue
                    st = e.stat()
                    files.append((e.path, st.st_mtime, st.st_size))
                    total += st.st_size
            if total <= self._disk_budget:
                self._disk_bytes = total        # reconcile: now known + under budget
                return
            for path, _mtime, size in sorted(files, key=lambda t: t[1]):
                try:
                    os.remove(path)
                    # This loop evicts by PATH (oldest first), so the membership set can no
                    # longer answer "was this key spilled" — drop to unknown and let `_restore`
                    # pay the stat again. Cheap correctness beats a fast wrong answer: a stale
                    # `True` costs one syscall, a stale ABSENCE silently loses a frame.
                    self._spilled = None
                    total -= size
                except OSError:
                    pass
                if total <= self._disk_budget:
                    break
            self._disk_bytes = total            # reconcile to the post-eviction total
        except Exception:
            self._disk_bytes = None             # scan failed: stay in the safe rescan-next-time state

    # ── CACHE-5: governor hooks (the frame cache folded into the global arbitration) ──
    def governed_bytes(self, dev_type: str | None = None) -> int:
        """RAM bytes this cache holds on `dev_type` ("cuda"/"cpu"), or all if None — the figure
        the CACHE-5 governor sums against the per-device VRAM/RAM budget. O(1) — it reads the
        maintained `_bytes_by_dev` counters rather than walking the table, because `arbitrate()`
        calls this under the lock and the walk was 415 us at 2000 entries."""
        with self._lock:
            if dev_type is None:
                return self._ram_bytes
            return self._bytes_by_dev["cuda" if dev_type == "cuda" else "cpu"]

    def evict_bytes(self, need: int, *, dev_type: str | None = None, playhead=None) -> int:
        """CACHE-5 evict hook: free ~`need` bytes on `dev_type`; returns bytes freed. Never drops
        the whole cache (leaves ≥1 entry so a just-served frame survives). `playhead` is accepted
        for a future far-from-playhead ordering; today the LRU front already approximates it (a
        scrub touches near-playhead frames most recently).

        CACHE-8: when the residency tier is ARMED and the governor is asking for VRAM, the
        coldest CUDA frames are DEMOTED to host RAM first. That frees exactly the bytes the
        governor asked for — VRAM ones — at 10.8 ms/frame at 4K instead of 204 ms, and the frame
        survives as a cache hit rather than becoming a 36 ms disk read. Only what demotion cannot
        cover is spilled. With the tier disarmed this is v0.32's behaviour, unchanged: "off"
        means the cache does not move frames between devices, not that it moves them silently."""
        with self._lock:
            if need <= 0:
                return 0
            freed = 0
            if dev_type == "cuda" and self._vram_budget is not None:
                freed += self._queue_demotions(need)
            # snapshot LRU order (oldest first) so mutation during iteration is safe. Entries
            # already queued for demotion are skipped: their bytes are counted once, and
            # spilling one would race the drain for the same frame.
            queued = {k for k, _e in self._pending_demotes} | self._demoting
            for key in list(self._ram.keys()):
                if freed >= need or len(self._ram) <= 1:
                    break
                entry = self._ram.get(key)
                if entry is None or key in queued:
                    continue
                if dev_type is not None and _dev_bucket(entry.device) != dev_type:
                    continue
                nbytes = entry.nbytes
                self._remove(key)
                self.evictions += 1
                self._pending_spills.append((key, entry, self._claim_spill_ticket(key)))
                freed += nbytes
        # BOTH drains outside the lock, in ladder order. A demotion is the cheaper rung and is
        # what the governor was really asking for on CUDA; the spill is the fallback for what it
        # could not cover. `freed` counts the demotions before the drain runs, deliberately: the
        # hook's contract is "how many bytes will you release", the drain is unconditional from
        # here, and making `arbitrate()` block on a full-frame D2H would put exactly the copy
        # inside the eviction loop that the lock rule exists to keep out of it.
        self._drain_demotes()         # VRAM -> host RAM: the frame stays a cache hit
        self._drain_spills()          # to disk (best-effort) — a miss just recooks
        return freed

    # ── introspection / lifecycle ──
    def sweep_temps(self) -> int:
        """Drop crash-orphaned `tex_recovery` temps from the spill dir. Called from
        `reindex_disk` (i.e. on ENG-13 recovery), which is exactly when a previous
        process's leftovers are known to be dead."""
        from .tex_recovery import sweep_temps as _sweep
        try:
            return _sweep(self._spill_dir())
        except Exception:
            return 0

    def reindex_disk(self) -> tuple:
        """ENG-13: re-walk the spill directory and adopt its size. Returns (frames, bytes).

        A `ResultCache` in a fresh process already SERVES frames another process spilled —
        `get` falls through to `_restore`, which reads by key and re-checks `env_epoch`. What
        it lacks is the byte accounting, and a `None` total forces a reconciling scan on the
        next spill, i.e. on a cook. This does that walk once, off the cook path, and is what
        `tex_recovery.reattach` calls instead of reaching in for `_spill_dir`, `_disk_bytes`
        and the `.frame` suffix from outside the class."""
        # The directory walk runs OUTSIDE the lock (see the rule on __init__): it is pure
        # filesystem work over a dir this cache owns, and on a multi-GB spill tier it would
        # otherwise block every `get` and `put` for the length of a scandir.
        self.sweep_temps()              # a crashed writer's leftovers are dead by definition
        # A3: the two facts that decide whether the scan's answer may be trusted as COMPLETE.
        # Read before the walk, compared after it.
        with self._lock:
            spills_at_entry = self.spills
            unknown_at_entry = self._spilled is None
        n = nbytes = 0
        total = None
        try:
            found = set()
            with os.scandir(self._spill_dir()) as it:
                for entry in it:
                    if entry.name.endswith(".frame") and entry.is_file():
                        n += 1
                        nbytes += entry.stat().st_size
                        found.add(entry.name[:-len(".frame")])
            total = nbytes
        except OSError:
            total = None                # lost track: let the next spill reconcile
        with self._lock:
            # P0-6: MERGE, never rebind. The scandir above runs unlocked by design, so a
            # concurrent `_spill` can land a frame between the scan and this block. Rebinding
            # `_spilled` to the scanned set would then FORGET that frame — and `_restore`
            # short-circuits on the membership set without stat-ing, so the file becomes
            # unserveable and unreachable by the epoch cleanup while `_disk_bytes` under-counts
            # it by exactly its size. The frame is on disk forever and nothing can see it.
            #
            # Merging is the conservative direction the whole `_spilled` design already uses: a
            # stale TRUE costs one syscall in `_restore`, a stale ABSENCE loses a frame. Union
            # with whatever the spiller recorded, and prefer `unknown` (None) over a definite
            # wrong answer when the scan itself failed.
            raced = self.spills != spills_at_entry
            # `self._spilled` can be dropped to None DURING the scan — `_enforce_disk_budget`
            # evicts by path and invalidates membership when it does — and the merge below
            # would then raise `TypeError: unsupported operand type(s) for -: 'NoneType'`.
            # Found by asking why the `raced` mutation survived: it is redundant for the two
            # cases already covered, and the case it is NOT redundant for was crashing.
            if total is None or unknown_at_entry or raced or self._spilled is None:
                # A3: the v0.33 fix merged `_spilled` with the scan, which closes the window
                # only for a cache whose membership is already a LEARNED SET. It does nothing
                # for `_spilled is None` — which is precisely `reindex_disk`'s own reason to
                # exist, the ENG-13 reattach of a fresh cache over a populated directory:
                # `_spill` skips recording when membership is unknown, so the merge finds
                # "nothing missed" and the tail rebinds to a definite set that EXCLUDES the
                # racing frame. That frame is then unserveable forever (`_restore`
                # short-circuits on the set without stat-ing) and its bytes are uncounted.
                #
                # So the rebind happens only when nothing could have been missed. The spill
                # counter is the witness: if it moved during the scan, or membership was
                # unknown when we started, stay UNKNOWN — one stat per miss is the price, and
                # the file's standing convention is that an uncertain answer is `None`, never
                # a definite wrong one.
                self._disk_bytes = None
                self._spilled = None
            else:
                missed = self._spilled - found
                self._spilled = found | missed
                self._disk_bytes = total if not missed else None
        return n, nbytes

    def stats(self) -> dict:
        with self._lock:
            return {"ram_entries": len(self._ram), "ram_bytes": self._ram_bytes,
                    "budget_bytes": self._budget, "hits": self.hits, "misses": self.misses,
                    "spills": self.spills, "restores": self.restores, "evictions": self.evictions,
                    # CACHE-8. `demoted` is what the residency tier is currently HOLDING off the
                    # GPU; the counters are what it has DONE. Both, because a tier that is
                    # working and a tier that is thrashing look identical in either one alone.
                    "vram_bytes": self._bytes_by_dev["cuda"],
                    "vram_budget_bytes": self._vram_budget,
                    "demotions": self.demotions, "promotions": self.promotions,
                    "demoted": sum(1 for e in self._ram.values() if e.device != e.home)}

    def clear(self, *, disk=False) -> None:
        with self._lock:
            self._ram.clear()
            self._bytes_by_dev = {"cuda": 0, "cpu": 0}      # `_ram_bytes` derives from this
            # Both drain queues hold entries that no longer exist. `_drain_demotes` re-checks
            # identity and would drop them anyway, but a queued spill is written unconditionally
            # — a cleared cache must not keep writing the frames it just forgot.
            self._pending_demotes.clear()
            self._pending_spills.clear()
            self._spill_seq.clear()       # A1: keys are gone; their write tickets go with them
            # `_spill_locks` is deliberately NOT cleared here, and the asymmetry beside the line
            # above is the kind that reads as an oversight, so: dropping the lock OBJECTS would
            # let an in-flight writer (holding a strong local reference to the old lock) and a
            # fresh post-clear writer of the same key hold TWO DIFFERENT locks and interleave.
            # The stale one has already passed its check and is inside `_atomic_pickle`, so the
            # order becomes write-A, write-B, then A's stale-cleanup `os.remove` deleting B's
            # file while B's key stays indexed — a frame that is present in `_spilled` and absent
            # from disk. The tickets are safe to drop because a missing ticket reads as 0, which
            # only makes a stale writer MORE likely to bail. Unbounded growth of this dict is a
            # known, recorded deferral (DEVELOPMENT.md); it is not fixable by clearing it here.
            # A5: a victim `_drain_spills` ALREADY popped is not in either queue, so clearing
            # them does not stop it. Its write lands after the unlink loop below, re-creating
            # `key.frame` and re-adding the key to the index — and a later `get` then serves the
            # frame the user just cleared. The generation is checked under the lock before the
            # post-write index add; a stale writer drops its own file instead.
            self._generation += 1
            if disk:
                self._purge_depth += 1      # restores decline until the walk below finishes
            # A3: the same witness `reindex_disk` uses, captured HERE rather than in a second
            # acquisition below. Reading it at the generation bump is strictly more conservative
            # — a spill landing between this block and the walk now also forces "unknown", which
            # costs one stat per miss and never a wrong definite answer.
            spills_at_entry = self.spills
        if not disk:
            return
        # The unlink loop runs OUTSIDE the lock (see the rule on __init__) — N os.remove calls
        # on a large spill tier are not something a concurrent `get` should wait behind.
        total = None
        try:
            d = self._spill_dir()
            all_removed = True
            for n in os.listdir(d):
                if n.endswith(".frame"):
                    try:
                        os.remove(os.path.join(d, n))
                    except OSError:
                        all_removed = False     # a survivor still occupies disk bytes
            # Only claim 0 when the dir is truly empty; else force a reconciling scan (mirrors
            # the file's convention: an uncertain total is None, never a wrong definite value).
            total = 0 if all_removed else None
        except Exception:
            total = None
        finally:
            # The depth MUST come back down on EVERY exit, which is why this is a `finally` and
            # not a line in the tail block below: the `except` above catches `Exception`, so a
            # `BaseException` — a Ctrl-C during a large unlink walk is the realistic one — would
            # skip the tail entirely and leave the count stuck. A stuck count declines every
            # restore for the life of the process, turning the disk tier into a silent recook
            # path with no counter or log saying so. Dropping it here rather than after the tail
            # is safe: the walk is finished, so a restore that slips in between finds no file.
            with self._lock:
                self._purge_depth -= 1
        with self._lock:
            # A3: the unlink loop above runs UNLOCKED (N os.remove calls must not block every
            # concurrent get), so a spill can land during it — legitimately, since it passes the
            # A5 generation check by design if it started after the bump. Asserting a definite
            # empty set then ORPHANS that frame: `_restore` short-circuits on the membership set
            # without stat-ing, so the file is unserveable and its bytes uncounted. Measured:
            # `_spilled=set()`, `_disk_bytes` 66085 -> 0, `get(K)` None with the file on disk.
            if self.spills != spills_at_entry:
                self._disk_bytes = None
                self._spilled = None        # unknown beats a confident wrong answer
            else:
                self._disk_bytes = total
                self._spilled = set() if total == 0 else None


def _atomic_pickle(path: str, data) -> bool:
    """Atomic pickle, so a second process sharing the results/ dir never observes a
    half-written frame. ENG-13 routes it through the shared `tex_recovery.atomic_write` — one
    temp-and-rename discipline spelled once for every persisted engine file rather than once
    per module.

    STREAMED, not blobbed: `pickle.dump(data, f)` writes through the file object, where
    `pickle.dumps` would materialize the whole frame as a second `bytes` copy first. This runs
    from `_enforce_ram_budget`, i.e. *precisely when the RAM budget is already exceeded* — the
    spill that exists to shed pressure must not transiently double it (measured 2.5× the frame
    at 1024²; ~100 MB extra at 4K).

    NOT fsynced, unlike the verdict files. A spilled frame is pure cache: losing one to a crash
    costs a recook, which the cache is built to do anyway, and the flush measured 14 ms per
    eviction on this box against 9.9 ms for the whole write. Durability is for state whose loss
    is expensive to re-derive; this is not that.

    Returns `atomic_write`'s verdict rather than swallowing it. `_spill` is the caller that
    keeps a running byte total and a membership set, so it is exactly the caller that must not
    charge for a write that did not happen."""
    import pickle
    from .tex_recovery import atomic_write
    return bool(atomic_write(path, lambda f: pickle.dump(data, f,
                                                         protocol=pickle.HIGHEST_PROTOCOL)))
