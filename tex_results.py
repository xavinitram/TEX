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
from collections import OrderedDict


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
    by an engine host (see docs/results-caching.md). Not thread-safe by itself; a host that
    shares one across threads guards it (DATA-4 session contract)."""

    def __init__(self, *, budget_mb=None, disk_budget_mb=None, cache_dir=None):
        self._ram: OrderedDict = OrderedDict()   # key -> [tensor, stamp, nbytes, device, canvas]
        self._ram_bytes = 0
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
        self.hits = self.misses = self.spills = self.restores = self.evictions = 0

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
        # Normalize layout at ingest so the RAM tier and a disk-restored copy of the same key are
        # layout-identical (the spill serializes a contiguous copy); .contiguous() is a no-op on a
        # channels-last cook output, the common case.
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
        self._drop(key)                            # replace: drop any prior entry's accounting first
        # a fresh key lands at the OrderedDict tail (MRU); _enforce_ram_budget evicts from the front
        self._ram[key] = [frozen, stamp, storage_bytes, str(frozen.device), canvas]
        self._ram_bytes += storage_bytes
        self._enforce_ram_budget()

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
        frame = None
        entry = self._ram.get(key)
        if entry is not None:
            tensor, stamp = entry[0], entry[1]
            if not tex_engine.verify_unmutated(tensor, stamp):
                self._drop(key)                    # a mutable entry written through: never serve it
            else:
                self._ram.move_to_end(key)
                frame = tensor
        if frame is None:
            frame = self._restore(key)             # re-admits to RAM; returns the master, or None
        if frame is None:
            self.misses += 1
            return None
        self.hits += 1
        # clone() on a frozen master (outside inference_mode) yields a normal, mutable, owned copy.
        return frame.clone() if copy else frame

    def _drop(self, key: str) -> None:
        entry = self._ram.pop(key, None)
        if entry is not None:
            self._ram_bytes -= entry[2]

    def _enforce_ram_budget(self) -> None:
        """Spill LRU victims to disk until under the RAM byte budget. Never evicts the entry
        just inserted (it is newest / move_to_end'd), so a single frame larger than the whole
        budget still serves once — it simply gets spilled on the next insert."""
        while self._ram_bytes > self._budget and len(self._ram) > 1:
            old_key, entry = next(iter(self._ram.items()))
            self._ram.pop(old_key)
            self._ram_bytes -= entry[2]
            self.evictions += 1
            self._spill(old_key, entry)

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
            path = self._disk_path(key)
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                rec = pickle.load(f)
            if rec.get("epoch") != env_epoch():     # a prior-environment frame: discard
                try:
                    os.remove(path)
                    self._disk_bytes = None          # out-of-band removal: invalidate the total
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
                    total -= size
                except OSError:
                    pass
                if total <= self._disk_budget:
                    break
            self._disk_bytes = total            # reconcile to the post-eviction total
        except Exception:
            self._disk_bytes = None             # scan failed: stay in the safe rescan-next-time state

    # ── introspection / lifecycle ──
    def stats(self) -> dict:
        return {"ram_entries": len(self._ram), "ram_bytes": self._ram_bytes,
                "budget_bytes": self._budget, "hits": self.hits, "misses": self.misses,
                "spills": self.spills, "restores": self.restores, "evictions": self.evictions}

    def clear(self, *, disk=False) -> None:
        self._ram.clear()
        self._ram_bytes = 0
        if disk:
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
                self._disk_bytes = 0 if all_removed else None
            except Exception:
                self._disk_bytes = None


def _atomic_pickle(path: str, data) -> None:
    """Atomic pickle (temp + os.replace) — the frame-cache twin of TEXCache._atomic_pickle, so
    a second process sharing the results/ dir never observes a half-written frame."""
    import pickle
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
