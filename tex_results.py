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


def _default_ram_budget() -> int:
    """A conservative default frame-RAM budget. A cook keeps its outputs on the cook device,
    so a CUDA frame cache lives in VRAM and must not crowd out the cook itself — hence a
    modest slice, host-overridable via TEX_RESULTS_BUDGET_MB. CACHE-5 will fold this into the
    global governor; until then the frame budget self-evicts."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            return min(2 << 30, int(0.25 * total))
    except Exception:
        pass
    return 512 << 20


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
    to this module."""

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
        self._lock = threading.RLock()
        self._ram: OrderedDict = OrderedDict()   # key -> [tensor, stamp, nbytes, device, canvas]
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
        self.hits = self.misses = self.spills = self.restores = self.evictions = 0

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
        is the safe answer: unknown means `_restore` pays the stat and still finds the frame."""
        try:
            found = set()
            with os.scandir(self._spill_dir()) as it:
                for e in it:
                    if e.name.endswith(".frame") and e.is_file():
                        found.add(e.name[:-len(".frame")])
            self._spilled = found
        except OSError:
            self._spilled = None

    def _disk_path(self, key: str) -> str:
        return os.path.join(self._spill_dir(), f"{key}.frame")

    # ── core API ──
    def put(self, key: str, tensor, *, canvas=None) -> None:
        """Store `tensor` under `key`, frozen per ENG-12 (an in-place write to a cached frame
        then raises instead of silently corrupting it). Re-inserting a key replaces it."""
        import torch
        if not isinstance(tensor, torch.Tensor):
            return
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
            self._remove(key)                          # replace: drop any prior entry's accounting first
            # a fresh key lands at the OrderedDict tail (MRU); _enforce_ram_budget evicts from the front
            dev = str(frozen.device)
            self._ram[key] = [frozen, stamp, storage_bytes, dev, canvas]
            self._bytes_by_dev[_dev_bucket(dev)] += storage_bytes
            self._enforce_ram_budget()
        self._drain_spills()          # disk I/O outside the lock (see the rule on __init__)

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
        frozen master). A RAM miss falls to the disk spill tier and re-admits a restored frame."""
        from . import tex_engine
        with self._lock:
            frame = None
            entry = self._ram.get(key)
            if entry is not None:
                tensor, stamp = entry[0], entry[1]
                if not tex_engine.verify_unmutated(tensor, stamp):
                    self._remove(key)                  # a mutable entry written through: never serve it
                else:
                    self._ram.move_to_end(key)
                    frame = tensor
        if frame is None:
            # OUTSIDE the lock: `_restore` is a file read plus an H2D copy — measured at
            # hundreds of ms for a large frame, and holding the lock across it stalls every
            # concurrent `get` and `put` behind a disk seek. It re-admits through `put`, which
            # takes the lock itself, so the insert is still serialized. Two threads racing the
            # same key both read and the second `put` simply replaces the first: a duplicated
            # read, never a corrupted table.
            frame = self._restore(key)                 # re-admits to RAM; returns the master, or None
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
                     base_key: str | None = None, canvas=None):
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
            return self._patch_region_locked(key, patch, window, base, base_key, canvas)

    def _patch_region_locked(self, key, patch, window, base, base_key, canvas):
        frame = base if base is not None else self.get(base_key or key, copy=True)
        if frame is None:
            return None                        # nothing to patch — the caller cooks whole
        x0, y0, w, h = (int(v) for v in tuple(window)[:4])
        if patch.shape[1:3] != (h, w):
            # The engine served a different extent than the window claims — a declined window
            # reported as served, or a caller passing its REQUEST instead of `cooked_roi`.
            # Refusing beats writing the wrong pixels into a frame that then looks authoritative.
            return None
        if frame.shape[1:3] != (int(tuple(window)[5]), int(tuple(window)[4])):
            return None                        # base is not the frame this window describes
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
                                           "roi": [x0, y0, w, h]}))
        return frame

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
        while True:
            with self._lock:
                if not self._pending_spills:
                    return
                key, entry = self._pending_spills.popleft()
            # Both producers (`_enforce_ram_budget`, `evict_bytes`) queue only entries `_remove`
            # returned non-None, so there is nothing to guard against here.
            self._spill(key, entry)

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
            self._bytes_by_dev[_dev_bucket(entry[3])] -= entry[2]
        return entry

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
            self._pending_spills.append((old_key, entry))

    # ── disk spill / restore ──
    def _spill(self, key: str, entry) -> None:
        """Write an evicted frame to disk (best-effort), then atomically pickle it under results/.
        A failed spill just drops the frame — the cook reproduces it. (Page-locking is applied on
        the RESTORE side, not here: pinning is not preserved through pickle, so a pinned spill
        buffer would deserialize pageable anyway.)"""
        try:
            import torch
            tensor = entry[0]
            # A normal, contiguous CPU copy for serialization in ONE host-visible copy. A frozen
            # (inference) entry can't be pickled, and `.to("cpu")` on an already-CPU frame is a
            # no-op view — so a naive `.to("cpu").clone()` costs two host memcpys on CPU frames and
            # a redundant H2H clone on CUDA ones. Copy into a fresh normal (non-inference) buffer
            # instead: empty() outside inference_mode is mutable/picklable, and one copy_ does the
            # (D2H or H2H) move — the clone that only existed to strip the inference flag is gone.
            cpu_t = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
            cpu_t.copy_(tensor)
            rec = {"t": cpu_t, "device": entry[3], "canvas": entry[4], "epoch": env_epoch()}
            path = self._disk_path(key)
            # A key can spill again after a restore (restore leaves the .frame on disk), so the
            # write may OVERWRITE an existing file — measure the old size first and apply the NET
            # delta, or the running total over-counts and forces needless reconciling scans.
            prev = 0
            if self._disk_bytes is not None:
                try:
                    prev = os.path.getsize(path)
                except OSError:
                    prev = 0
            _atomic_pickle(path, rec)
            self.spills += 1
            if self._spilled is not None:
                self._spilled.add(key)      # still unknown (None) stays unknown, not {this key}
            if self._disk_bytes is not None:            # keep the running total current
                try:
                    self._disk_bytes += os.path.getsize(path) - prev
                except OSError:
                    self._disk_bytes = None             # lost track: force a reconciling scan
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
                self._learn_spilled()      # one scandir, once, instead of a stat per miss
            if self._spilled is not None and key not in self._spilled:
                return None                # not on the tier — no syscall needed
            path = self._disk_path(key)
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                rec = pickle.load(f)
            if rec.get("epoch") != env_epoch():     # a prior-environment frame: discard
                try:
                    os.remove(path)
                    self._disk_bytes = None          # out-of-band removal: invalidate the total
                    if self._spilled is not None:
                        self._spilled.discard(key)
                except OSError:
                    pass
                return None
            host = rec["t"]
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
            self.put(key, tensor, canvas=rec.get("canvas"))   # already frozen: put() no-op-freezes
            return self._ram[key][0] if key in self._ram else tensor
        except Exception:
            return None

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
        """CACHE-5 evict hook: spill LRU frames on `dev_type` to disk until ~`need` bytes are
        freed; returns bytes freed. Never drops the whole cache (leaves ≥1 entry so a just-served
        frame survives). `playhead` is accepted for a future far-from-playhead ordering; today the
        LRU front already approximates it (a scrub touches near-playhead frames most recently)."""
        with self._lock:
            if need <= 0:
                return 0
            freed = 0
            # snapshot LRU order (oldest first) so mutation during iteration is safe
            for key in list(self._ram.keys()):
                if freed >= need or len(self._ram) <= 1:
                    break
                entry = self._ram.get(key)
                if entry is None:
                    continue
                if dev_type is not None and _dev_bucket(entry[3]) != dev_type:
                    continue
                nbytes = entry[2]
                self._remove(key)
                self.evictions += 1
                self._pending_spills.append((key, entry))
                freed += nbytes
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
            self._disk_bytes = total
            # A fresh process over a populated dir starts at `unknown`; this walk is where it
            # learns the tier's real contents — precisely the ENG-13 reattach path.
            self._spilled = found if total is not None else None
        return n, nbytes

    def stats(self) -> dict:
        with self._lock:
            return {"ram_entries": len(self._ram), "ram_bytes": self._ram_bytes,
                    "budget_bytes": self._budget, "hits": self.hits, "misses": self.misses,
                    "spills": self.spills, "restores": self.restores, "evictions": self.evictions}

    def clear(self, *, disk=False) -> None:
        with self._lock:
            self._ram.clear()
            self._bytes_by_dev = {"cuda": 0, "cpu": 0}      # `_ram_bytes` derives from this
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
        with self._lock:
            self._disk_bytes = total
            self._spilled = set() if total == 0 else None


def _atomic_pickle(path: str, data) -> None:
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
    is expensive to re-derive; this is not that."""
    import pickle
    from .tex_recovery import atomic_write
    atomic_write(path, lambda f: pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL))
