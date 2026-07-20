"""
M-1/M-2 — memory cooperation for TEX.

`estimate_peak_bytes` gives a cheap, static upper-ish estimate of a cook's peak
transient VRAM so the node can preflight `comfy.model_management.free_memory`
and retry once on OOM instead of failing while GBs sit locked in resident
models. `free_tensor_caches` drops every module-level tensor cache TEX holds.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import torch

from .tex_runtime.host import (_report_progress as _progress,  # SCHED-3 best-effort progress sink
                               _cancel_check)                  # SCHED-3 yield-point poll (None-safe)
from .tex_compiler.ast_nodes import (
    ForLoop, WhileLoop, FunctionCall, ArrayDecl,
    BindingIndexAccess, BindingSampleAccess,
    iter_child_nodes as _iter_children,
)
from .tex_compiler.types import TEXType, TYPE_NAME_MAP

# M-4: stdlib functions whose output at a pixel depends on OTHER pixels (or the
# whole image), so a program that calls them cannot be split into horizontal
# strips. A whitelist posture: anything not here is treated as pixel-local.
#
# ROI-1: this set is now DERIVED from the stdlib registry's per-function `footprint`
# (invariant #5's single-source discipline), not a hand-kept literal — a footprint of
# anything but 'point' is non-local. Derived LAZILY: the registry is populated only
# when TEXStdlib's class body runs (a `tex_runtime.stdlib` import), which has always
# happened by the time any cook or test reaches this code. The historical public name
# `tex_memory._NON_LOCAL_FNS` stays a readable module attribute via `__getattr__`, so
# every existing consumer (is_tile_safe, TST-3) is unchanged.
_non_local_fns_cache: "frozenset | None" = None


def _non_local_fns() -> frozenset:
    global _non_local_fns_cache
    if _non_local_fns_cache is None:
        from .tex_runtime.stdlib_registry import non_local_names
        _non_local_fns_cache = non_local_names()
    return _non_local_fns_cache


def __getattr__(name):
    # PEP 562: expose the derived set under its historical public name without forcing
    # derivation at import time (REGISTRY is empty until stdlib.py's class body runs).
    if name == "_NON_LOCAL_FNS":
        return _non_local_fns()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_tile_safe(program) -> bool:
    """M-4: True if every pixel of the output depends only on the same pixel of
    the input (pointwise), so H can be split into strips. Excludes sample/fetch/
    blur/img-reduction calls and any binding index/sample access (a gather or a
    scatter `@OUT[x,y]=` — the latter parses as Assignment(target=BindingIndex))."""
    non_local = _non_local_fns()   # ROI-1: registry-derived, hoisted so the walk stays hot
    stack = list(program.statements)
    while stack:
        n = stack.pop()
        cls = n.__class__
        if cls is FunctionCall and n.name in non_local:
            return False
        if cls is BindingIndexAccess or cls is BindingSampleAccess:
            return False
        stack.extend(_iter_children(n))
    return True


# P4: is_tile_safe is a full AST walk (~22 us) run every CUDA cook in `_tile_plan`.
# Tile-safety is a STATIC property of the program, so memoize it on the same cook
# fingerprint `should_stencil_route` keys on (mirrors `_stencil_route_memo` exactly,
# cap included) -- the walk then runs once per program, not once per cook.
_TILE_SAFE_MEMO_MAX = 256
_tile_safe_memo: "OrderedDict[str, bool]" = OrderedDict()


def is_tile_safe_cached(program, fingerprint) -> bool:
    """`is_tile_safe(program)` memoized per cook fingerprint (P4). `fingerprint=None`
    (a fused chain, whose AST is spliced per-cook) falls through to the uncached walk.
    Stored values are always bool, so a `.get() is None` unambiguously means 'absent'
    (a cached False is served, not recomputed) -- identical to `should_stencil_route`."""
    if fingerprint is None:
        return is_tile_safe(program)
    v = _tile_safe_memo.get(fingerprint)
    if v is None:
        v = is_tile_safe(program)
        _tile_safe_memo[fingerprint] = v
        while len(_tile_safe_memo) > _TILE_SAFE_MEMO_MAX:
            _tile_safe_memo.popitem(last=False)
    else:
        _tile_safe_memo.move_to_end(fingerprint)
    return v

# stdlib calls whose peak is dominated by extra full-frame allocations.
_SAMPLE_FAMILY = frozenset({
    "sample", "sample_cubic", "sample_lanczos", "sample_frame",
    "fetch", "fetch_frame", "sample_grad",
})
_MIP_FAMILY = frozenset({"sample_mip", "sample_mip_gauss"})


def _array_element_floats(decl: ArrayDecl) -> int:
    """Per-pixel float count of one array element (vecN -> N, scalar -> 1)."""
    t = TYPE_NAME_MAP.get(decl.element_type_name, TEXType.FLOAT)
    return t.channels if getattr(t, "is_vector", False) else 1


# LAT-2: the four results of the AST walk below (has_loop_sample, has_mip, pointwise,
# array_floats) are a STATIC, value-independent property of the program — exactly like
# is_tile_safe / should_stencil_route — so memoize them on the same cook fingerprint those
# two key on. Only the trailing spatial/dtype arithmetic is per-call. On a default CUDA cook
# the walk ran up to twice/cook (once in _preflight_memory, once in _tile_plan); this makes
# it once per PROGRAM. Mirrors _tile_safe_memo exactly, cap included.
_PEAK_STATIC_MEMO_MAX = 256
_peak_static_memo: "OrderedDict[str, tuple]" = OrderedDict()


def _estimate_peak_statics(program) -> tuple:
    """The value-independent results of estimate_peak_bytes's AST walk:
    `(has_loop_sample, has_mip, pointwise, array_floats)`. A pure function of the program
    AST — nothing here depends on spatial_shape or dtype_bytes (those enter only in the
    per-call arithmetic estimate_peak_bytes does afterwards)."""
    has_loop_sample = False
    has_mip = False
    pointwise = True
    array_floats = 0

    # depth-tracked walk: mark sample-family calls that occur inside a loop.
    stack = [(s, 0) for s in program.statements]
    while stack:
        node, depth = stack.pop()
        cls = node.__class__
        if cls is FunctionCall:
            if node.name in _SAMPLE_FAMILY or node.name in _MIP_FAMILY:
                pointwise = False
                if depth > 0:
                    has_loop_sample = True
            if node.name in _MIP_FAMILY:
                has_mip = True
        elif cls is ArrayDecl:
            pointwise = False
            try:
                array_floats += int(node.size) * _array_element_floats(node)
            except Exception:
                pass
        elif cls in (ForLoop, WhileLoop):
            pointwise = False
        child_depth = depth + 1 if cls in (ForLoop, WhileLoop) else depth
        for ch in _iter_children(node):
            stack.append((ch, child_depth))

    return has_loop_sample, has_mip, pointwise, array_floats


def _estimate_peak_statics_cached(program, fingerprint) -> tuple:
    """`_estimate_peak_statics(program)` memoized per cook fingerprint (LAT-2). `fingerprint=
    None` (a fused chain, whose AST is spliced per-cook) falls through to the uncached walk —
    identical posture to `is_tile_safe_cached`. Stored values are always a 4-tuple, so a
    `.get() is None` unambiguously means 'absent'."""
    if fingerprint is None:
        return _estimate_peak_statics(program)
    v = _peak_static_memo.get(fingerprint)
    if v is None:
        v = _estimate_peak_statics(program)
        _peak_static_memo[fingerprint] = v
        while len(_peak_static_memo) > _PEAK_STATIC_MEMO_MAX:
            _peak_static_memo.popitem(last=False)
    else:
        _peak_static_memo.move_to_end(fingerprint)
    return v


def estimate_peak_bytes(program, spatial_shape, dtype_bytes: int = 4,
                        fingerprint=None) -> int:
    """Estimate a cook's peak transient bytes (M-1).

    Base unit is a [B,H,W,4] tensor. K scales the live-temp count: K=4 when a
    sample-family call appears inside a loop (measured godrays ~3.75), K=1 for a
    pointwise program, K=1.5 otherwise. Added terms: the static ArrayDecl bytes
    (a `vec4 arr[25]` at 2048² is ~1.6 GB — a ~17x underestimate without it) and
    ~1.33x a frame per cold mip pyramid when sample_mip-family calls are present.

    LAT-2: the AST walk that derives (has_loop_sample, has_mip, pointwise, array_floats) is
    static per program, so it's memoized on `fingerprint` (None → uncached walk). Only the
    spatial/dtype arithmetic below runs per call — which is why the two engine call sites
    pass their OWN `dtype_bytes` (preflight 4; tile-plan post-auto 2 or 4). That difference
    is intentional and must NOT be shared — only the walk is.
    """
    if not spatial_shape:
        return 0
    B, H, W = spatial_shape
    px = B * H * W
    frame4 = px * 4 * dtype_bytes  # a [B,H,W,4] tensor

    has_loop_sample, has_mip, pointwise, array_floats = \
        _estimate_peak_statics_cached(program, fingerprint)

    k = 4.0 if has_loop_sample else (1.0 if pointwise else 1.5)
    est = int(k * frame4)
    est += array_floats * px * dtype_bytes
    if has_mip:
        est += int(1.33 * frame4)  # cold pyramid build on the first cook
    return est


# ── M-2: byte-budgeted cache eviction ─────────────────────────────────
# The tensor caches evict by ENTRY COUNT, and the two mip caches pin the full
# source image (~341 MB per entry at 4K). This caps total residency at a byte
# budget by evicting oldest entries — the allocate-and-hold grid-buffer
# semantics (a measured perf trap) are untouched; only the eviction *trigger*
# changes from len>N to bytes>budget, and the most-recent entry is always kept.

_BUDGET_CACHES = None  # lazily bound: list of (OrderedDict, entry->tensors)


def _budget_caches():
    global _BUDGET_CACHES
    if _BUDGET_CACHES is None:
        from .tex_runtime import stdlib as sl
        _BUDGET_CACHES = [
            (sl._mip_cache, lambda e: [e[1], *e[2]]),         # (shape, img, pyramid)
            (sl._gauss_mip_cache, lambda e: [e[1], *e[2]]),
            (sl._grid_buf, lambda e: [e]),
            (sl._sampler_cache, lambda e: [e]),
            (sl._gauss_kernel_cache, lambda e: list(e)),
        ]
    return _BUDGET_CACHES


def _entry_bytes(entry, extract) -> int:
    total = 0
    try:
        for t in extract(entry):
            if isinstance(t, torch.Tensor):
                total += t.untyped_storage().nbytes()
    except Exception:
        pass
    return total


def _entry_dev_type(entry, extract):
    """MEM-4: the device.type of this cache entry's tensors (an entry's tensors share
    a device), or None if it holds none."""
    try:
        for t in extract(entry):
            if isinstance(t, torch.Tensor):
                return t.device.type
    except Exception:
        pass
    return None


def _total_cache_bytes(dev_type=None) -> int:
    """MEM-4: total tensor-cache bytes, optionally restricted to entries on `dev_type`.
    A CPU cook's 512 MB budget must not count (or evict) CUDA-resident mip entries, and
    a CUDA cook must not be throttled by CPU-resident ones — each device is accounted
    against its own budget."""
    total = 0
    for c, ex in _budget_caches():
        for e in c.values():
            if dev_type is None or _entry_dev_type(e, ex) == dev_type:
                total += _entry_bytes(e, ex)
    return total


def cache_budget_bytes(device) -> int:
    """VRAM/CPU byte budget for TEX's tensor caches. Env override
    TEX_CACHE_BUDGET_MB; else min(1 GB, 12.5% VRAM) on CUDA, 512 MB on CPU."""
    override = os.environ.get("TEX_CACHE_BUDGET_MB")
    if override:
        try:
            return int(override) * 1024 * 1024
        except ValueError:
            pass
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":
        try:
            total = torch.cuda.get_device_properties(dev.index or 0).total_memory
            return min(1024 * 1024 * 1024, total // 8)
        except Exception:
            return 1024 * 1024 * 1024
    return 512 * 1024 * 1024


def _entry_pinned(entry, extract, pinned) -> bool:
    """MEM-1: True if any storage in this cache entry is baked into a live captured
    graph (its `data_ptr()` is in `pinned`). Such an entry reclaims 0 bytes if
    evicted — the graph still holds it — so the evictor skips it."""
    if not pinned:
        return False
    try:
        for t in extract(entry):
            if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() in pinned:
                return True
    except Exception:
        pass
    return False


def _oldest_unpinned_key(cache, extract, pinned, dev_type=None):
    """The oldest key in `cache` whose entry isn't graph-pinned and (MEM-4) is on
    `dev_type`, never the most-recent entry OF THAT dev_type (kept so the current cook's
    working set survives). None if none qualify. (Audit: `newest` must be scoped to
    `dev_type` — a global newest let a CPU sweep evict the newest CUDA entry, defeating
    the per-device isolation MEM-4 adds.)"""
    newest = None
    for k in cache:  # oldest -> newest; the LAST matching entry is the newest of dev_type
        if dev_type is None or _entry_dev_type(cache[k], extract) == dev_type:
            newest = k
    for k in cache:
        if k == newest:
            continue
        if dev_type is not None and _entry_dev_type(cache[k], extract) != dev_type:
            continue
        if not _entry_pinned(cache[k], extract, pinned):
            return k
    return None


def enforce_cache_budget(device) -> None:
    """MEM-1: evict oldest UNPINNED cache entries (mip caches first — they pin full
    frames) until total tensor-cache bytes are under budget. An entry whose storage
    is baked into a live CUDA graph is skipped — evicting it frees 0 bytes and would
    only force a needless recapture. Captured graphs are left intact: the graph's own
    pinning (graphed.pinned_storages) is what makes freeing the unpinned entries safe,
    so the old blunt clear_graph_cache() teardown — which also reset the RNG-poison
    kill switch and blacklist, silently re-arming doomed captures — is gone.
    Best-effort; never raises."""
    try:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        dev_type = dev.type  # MEM-4: only account + evict entries on the cook's device
        budget = cache_budget_bytes(device)
        if _total_cache_bytes(dev_type) <= budget:
            return
        try:
            from .tex_runtime.graphed import pinned_storages
            pinned = pinned_storages()
        except Exception:
            pinned = set()
        for cache, extract in _budget_caches():
            while len(cache) > 1 and _total_cache_bytes(dev_type) > budget:
                victim = _oldest_unpinned_key(cache, extract, pinned, dev_type)
                if victim is None:
                    break  # every evictable same-device entry is pinned by a live graph
                del cache[victim]
            if _total_cache_bytes(dev_type) <= budget:
                break
    except Exception:
        pass


# ── CACHE-5: the global cache governor (CacheRegistry) ────────────────────────
# Four cache families hold per-device VRAM/RAM and NONE sees the others: the stdlib env-tensor
# pools (mip/grid/sampler), the CUDA-graph pool, and — armed by an engine host — the CACHE-2
# frame cache. On a 12 GB box their independent caps sum past 37.5% of VRAM before the cook's
# own transients. The governor is the ONE place that arbitrates them against a SINGLE budget.
#
# What it does NOT do:
#   * It is NOT the per-cook default path. `enforce_cache_budget` (above) still runs every cook
#     and is byte-identical to before (invariant #7). The governor is armed by a host that owns
#     a frame cache (the ROI-3 / CACHE-2 dormant-by-default posture) — under ComfyUI the host
#     caches results itself, so the frame cache (and thus the cross-pool pressure) never arms.
#   * It does NOT centralize keys or lifecycles — those stay per-cache (the "19 caches are
#     non-redundant" register). ONLY eviction ARBITRATION centralizes.
#   * It does NOT call `clear_graph_cache()`. That note in the register is stale: tex_memory has
#     used pin-and-skip since MEM-1, never a blunt graph teardown. The governor preserves that
#     exact graph-address safety — it skips graph-pinned storages (`pinned_storages()`), and to
#     free VRAM a live graph holds it tears graphs down with `free_graphs_only()` (which keeps
#     the capture blacklist + RNG-poison kill switch — `clear_graph_cache()` would re-arm doomed
#     captures and regress MEM-1). Disk tiers keep their OWN size caps (CACHE-0 / ResultCache
#     disk), never per-device VRAM arbitration.

def governor_budget(device) -> int:
    """The ONE coordinated VRAM/RAM budget the governor holds all arbitrated pools under — set
    BELOW the sum of the pools' independent caps (the point of CACHE-5). Env override
    TEX_GOVERNOR_BUDGET_MB; else ~40% of free VRAM on CUDA (a single pressure-responsive cap
    the stdlib/graph/frame pools share), 1 GB on CPU."""
    override = os.environ.get("TEX_GOVERNOR_BUDGET_MB")
    if override:
        try:
            return int(override) * 1024 * 1024
        except ValueError:
            pass
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":
        try:
            from .tex_runtime.host import get_host_services
            free = get_host_services().get_free_memory(dev)
            if free:
                return int(0.4 * free)
        except Exception:
            pass
        total = device_total_mem(dev)
        return int(0.4 * total) if total else 1024 * 1024 * 1024
    return 1024 * 1024 * 1024


def _evict_stdlib_bytes(dev_type, need: int, playhead=None) -> int:
    """Governor evict hook for the stdlib env-tensor pools: drop oldest UNPINNED entries on
    `dev_type` until ~`need` bytes are freed (graph-pinned entries skipped — MEM-1). Returns
    bytes freed. `playhead` is unused (these pools carry no frame identity)."""
    try:
        from .tex_runtime.graphed import pinned_storages
        pinned = pinned_storages()
    except Exception:
        pinned = set()
    freed = 0
    for cache, extract in _budget_caches():
        while len(cache) > 1 and freed < need:
            victim = _oldest_unpinned_key(cache, extract, pinned, dev_type)
            if victim is None:
                break
            freed += _entry_bytes(cache[victim], extract)
            del cache[victim]
    return freed


def _graph_pool_bytes(dev_type) -> int:
    """Bytes the CUDA-graph pool holds (CUDA only — graphs never live on CPU)."""
    if dev_type != "cuda":
        return 0
    try:
        from .tex_runtime import graphed
        return max(0, int(graphed._graph_bytes))
    except Exception:
        return 0


def _evict_graphs(dev_type, need: int, playhead=None) -> int:
    """Governor evict hook for the CUDA-graph pool. Graphs can't be partially freed (a replay
    reads baked addresses), so this is all-or-nothing: when VRAM is needed it tears every graph
    down via `free_graphs_only()` — preserving the blacklist + kill switch (NOT
    `clear_graph_cache`). Registered LAST (highest evict_order) because recapture is costly, so
    it only fires when the rebuildable stdlib/frame pools couldn't free enough."""
    if dev_type != "cuda" or need <= 0:
        return 0
    held = _graph_pool_bytes(dev_type)
    if held <= 0:
        return 0
    try:
        from .tex_runtime.graphed import free_graphs_only
        free_graphs_only()
        return held
    except Exception:
        return 0


class CacheRegistry:
    """CACHE-5: arbitrates the per-device VRAM/RAM cache pools against one budget. Pools register
    a `(bytes_fn, evict_fn, evict_order)`; `arbitrate()` frees across them, cheapest-to-rebuild
    first, until under budget — the single coordination point four independent budgets lacked.
    Not thread-safe by itself (the MUT caches are single-cook-thread; a parallel host guards it —
    DATA-4)."""

    def __init__(self):
        # name -> (bytes_fn(dev_type)->int, evict_fn(dev_type, need, playhead)->freed, evict_order)
        self._pools: "OrderedDict[str, tuple]" = OrderedDict()

    def register(self, name: str, bytes_fn, evict_fn, *, evict_order: int = 50) -> None:
        self._pools[name] = (bytes_fn, evict_fn, evict_order)

    def unregister(self, name: str) -> None:
        self._pools.pop(name, None)

    def total_bytes(self, dev_type: str) -> int:
        return sum(self.stats(dev_type).values())

    def stats(self, dev_type: str) -> dict:
        out = {}
        for name, (bytes_fn, _ev, _o) in self._pools.items():
            try:
                out[name] = int(bytes_fn(dev_type))
            except Exception:
                out[name] = 0
        return out

    def arbitrate(self, device, *, budget: int | None = None, playhead=None) -> int:
        """Evict across the registered pools until total ≤ budget (default: `governor_budget`).
        Pools are drained in ascending `evict_order` — rebuildable stdlib pyramids first, frame
        cache next, CUDA graphs last. The roadmap's "live-graph-key" priority hint is realized by
        THIS ordering plus the stdlib evictor's pin-skip (a graph-pinned storage is never freed),
        not a separate hint arg; `playhead` is the one passed hint (far-from-playhead frames first,
        when a pool carries the frame). Best-effort; never raises. Returns bytes freed."""
        try:
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            dev_type = dev.type
            budget = budget if budget is not None else governor_budget(dev)
            total = self.total_bytes(dev_type)
            if total <= budget:
                return 0                      # cheap early-out (mirrors enforce_cache_budget)
            freed = 0
            for _name, (_bytes_fn, evict_fn, _order) in sorted(
                    self._pools.items(), key=lambda kv: kv[1][2]):
                if total - freed <= budget:
                    break
                need = (total - freed) - budget
                try:
                    freed += int(evict_fn(dev_type, need, playhead))
                except Exception:
                    pass
            return freed
        except Exception:
            return 0


_registry: "CacheRegistry | None" = None


def get_cache_registry() -> CacheRegistry:
    """The process-wide governor, with the always-present pools (stdlib env tensors, CUDA
    graphs) registered on first use. A host arms the frame cache with `register_result_cache`."""
    global _registry
    if _registry is None:
        reg = CacheRegistry()
        reg.register("stdlib", lambda dt: _total_cache_bytes(dt), _evict_stdlib_bytes,
                     evict_order=10)
        reg.register("graphs", _graph_pool_bytes, _evict_graphs, evict_order=90)
        _registry = reg
    return _registry


def register_result_cache(cache, *, name: str = "results", evict_order: int = 50) -> None:
    """Arm a CACHE-2 `ResultCache` into the governor (the frame cache is host-instantiated, so a
    host calls this when it creates one). Its RAM bytes then count toward the per-device budget
    and its LRU frames become evictable under pressure — the fold-in the CACHE-5 register
    promises. `cache.evict_bytes` supplies the eviction (playhead-aware when frames carry one)."""
    reg = get_cache_registry()
    reg.register(name,
                 lambda dt: cache.governed_bytes(dt),
                 lambda dt, need, playhead: cache.evict_bytes(need, dev_type=dt, playhead=playhead),
                 evict_order=evict_order)


# MEM-2 (audit B2): per-device last-seen spatial pixel count + cached total VRAM, so the
# allocator is queried ONLY after a resolution downshift — not on every cook.
_last_trim_px: dict = {}
_total_mem_cache: dict = {}


def device_total_mem(device) -> int | None:
    """LAT-2/MEM-2: total VRAM for a CUDA device, cached per index in the same
    `_total_mem_cache` that `trim_reserved_pool` populates. None on CPU or if the device
    query fails. Lets the M-1 preflight size its skip-gate against total VRAM without paying
    a per-cook driver query (get_device_properties is hit once per device, then cached)."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return None
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    total = _total_mem_cache.get(idx)
    if total is None:
        try:
            total = torch.cuda.get_device_properties(idx).total_memory
        except Exception:
            return None
        _total_mem_cache[idx] = total
    return total


def trim_reserved_pool(device, spatial_px: int = 0) -> None:
    """MEM-2: return stranded reserved VRAM after a big->small cook downshift. When the
    caching allocator holds far more reserved than is live (measured: a 512->4096->512
    sequence stranded 1551 MB), a threshold-gated `empty_cache()` reclaims it (measured
    3.4 ms, +0.09 ms next-cook tax).

    B2 fix: the allocator query is gated on a **resolution downshift** (spatial_px < the
    previous cook's) — the only case the reclaim targets. Same-size / upshift steady state
    pays NOTHING (the old always-on `memory_reserved`+`memory_allocated`+`get_device_
    properties` query was a measured +48% tax on tiny fp32 cooks). CUDA-only; total VRAM is
    cached per device; `TEX_NO_POOL_TRIM=1` disables. Safe with live captured graphs."""
    if os.environ.get("TEX_NO_POOL_TRIM") == "1":
        return
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    prev = _last_trim_px.get(idx, 0)
    if spatial_px:
        _last_trim_px[idx] = spatial_px
    # only a genuine downshift warrants an allocator query
    if not spatial_px or spatial_px >= prev:
        return
    try:
        reserved = torch.cuda.memory_reserved(idx)
        allocated = torch.cuda.memory_allocated(idx)
        total = _total_mem_cache.get(idx)
        if total is None:
            total = torch.cuda.get_device_properties(idx).total_memory
            _total_mem_cache[idx] = total
        if reserved - allocated > max(1024 * 1024 * 1024, total // 8):
            torch.cuda.empty_cache()
    except Exception:
        pass


def _shared_dim_size(bindings, dim: int, min_ndim: int) -> int | None:
    """The single size shared by every tensor binding on axis `dim` that isn't a broadcast
    singleton (size 1), or None when zero or >1 distinct sizes qualify (heterogeneous inputs
    can't be co-strided). Backs both `shared_tile_height` (dim 1) and `shared_batch_size`
    (dim 0)."""
    sizes = {v.shape[dim] for v in bindings.values()
             if isinstance(v, torch.Tensor) and v.dim() >= min_ndim and v.shape[dim] > 1}
    return next(iter(sizes)) if len(sizes) == 1 else None


def _cook_whole(interp, program, bindings, type_map, device, latent_channel_count,
                output_names, used_builtins, precision, time_context,
                cancel=None, on_progress=None) -> dict:
    """The whole-frame (untiled / un-ROI'd / un-strided) cook — the shared fallback body for
    `run_tiled` / `run_roi` / `run_batch_strips`. SCHED-3: forwards the cancel token / progress
    sink so a whole-frame fallback stays as abortable as the tiled path it replaced."""
    return interp.execute(program, bindings, type_map, device=device,
                          latent_channel_count=latent_channel_count,
                          output_names=output_names, used_builtins=used_builtins,
                          precision=precision, time_context=time_context,
                          cancel=cancel, on_progress=on_progress)


def shared_tile_height(bindings) -> int | None:
    """M-4: the single image height shared by every spatial (dim>=3) binding that
    isn't a broadcast singleton (shape[1]==1), or None when zero or >1 distinct
    heights qualify (heterogeneous inputs can't be co-tiled). One source of truth
    for both the tile PLAN (`_tile_plan`) and the tile EXECUTOR (`run_tiled`)."""
    return _shared_dim_size(bindings, 1, 3)


def run_tiled(interp, program, bindings, type_map, device, latent_channel_count,
              output_names, used_builtins, precision, n_strips: int,
              time_context: dict | None = None, cancel=None, on_progress=None) -> dict:
    """M-4: execute a tile-safe program in `n_strips` horizontal strips, keeping
    peak transient to ~1/n_strips of the full-image cook. Spatial bindings are
    narrowed (zero-copy views); outputs are preallocated once and strips copy_'d
    in (avoids a full-size torch.cat transient). Coordinates stay seam-exact via
    the interpreter's tile=(y0, H_total).

    `time_context` (ENG-7) must be forwarded to every strip: the interpreter reads the
    playhead off per-execute state, so a strip cooked without it silently reads zeros —
    a frozen animation on exactly the big cooks that need tiling.

    SCHED-3: `cancel`/`on_progress` ride the same channel — checked at each strip boundary
    (a natural yield point) and forwarded into every strip's cook, so a big tiled cook is
    the MOST abortable path, not the least."""
    def _untiled():
        return _cook_whole(interp, program, bindings, type_map, device, latent_channel_count,
                           output_names, used_builtins, precision, time_context,
                           cancel, on_progress)

    # M-4 safety: tiling narrows dim 1 (the image HEIGHT for [B,H,W,C] IMAGE /
    # [B,H,W] MASK). Refuse — and cook untiled — when that axis isn't a shared
    # height: (a) a LATENT is present ([B,C,H,W] → dim 1 is CHANNELS, so narrowing
    # would slice channels), or (b) spatial bindings disagree on height (a second
    # full-frame input can't be co-tiled; a broadcast H==1 input is fine and passes
    # through un-narrowed). Either would silently corrupt or shape-mismatch.
    if latent_channel_count:
        return _untiled()
    H_total = shared_tile_height(bindings)
    if H_total is None:
        return _untiled()
    bounds = [(i * H_total) // n_strips for i in range(n_strips)] + [H_total]
    outputs: dict = {}
    for k in range(n_strips):
        y0, y1 = bounds[k], bounds[k + 1]
        if y1 <= y0:
            continue
        _cancel_check(cancel)               # SCHED-3 yield F: abort a stale cook between strips
        strip_bindings = {}
        for name, v in bindings.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == H_total:
                strip_bindings[name] = v.narrow(1, y0, y1 - y0)
            else:
                strip_bindings[name] = v
        # SCHED-3: forward the CANCEL token (finer per-statement abort inside the strip) but
        # NOT on_progress — the strip loop owns the progress axis; letting each strip also emit
        # per-statement fractions would reset 0→1 every strip and read as noise to a host.
        res = interp.execute(program, strip_bindings, type_map, device=device,
                             latent_channel_count=latent_channel_count,
                             output_names=output_names, used_builtins=used_builtins,
                             precision=precision, tile=(y0, H_total),
                             time_context=time_context, cancel=cancel)
        for name, strip_out in res.items():
            if isinstance(strip_out, torch.Tensor) and strip_out.dim() >= 3:
                buf = outputs.get(name)
                if buf is None:
                    full_shape = list(strip_out.shape)
                    full_shape[1] = H_total
                    buf = torch.empty(full_shape, dtype=strip_out.dtype,
                                      device=strip_out.device)
                    outputs[name] = buf
                buf[:, y0:y1] = strip_out
            elif name not in outputs:
                outputs[name] = strip_out  # scalar/string: any strip suffices
        if on_progress is not None:         # SCHED-3: report per-strip fraction
            _progress(on_progress, "strip", (k + 1) / n_strips)
    return outputs


def shared_batch_size(bindings) -> int | None:
    """ROI-6: the single batch size (dim 0) shared by every batched binding that isn't a
    broadcast singleton (shape[0]==1), or None when zero or >1 distinct sizes qualify. The
    batch-axis twin of `shared_tile_height`."""
    return _shared_dim_size(bindings, 0, 1)


def run_batch_strips(interp, program, bindings, type_map, device, latent_channel_count,
                     output_names, used_builtins, precision, n_strips: int,
                     time_context: dict | None = None, cancel=None, on_progress=None) -> dict:
    """ROI-6: cook a PER-FRAME-INDEPENDENT program's batch in `n_strips` frame-strips (narrow
    dim 0), bounding peak transient to ~1/n_strips of the full-batch cook, and stitch — the
    batch-axis twin of `run_tiled`. The caller guarantees `tex_roi.batch_sliceable` (no
    cross-frame read); `fi`/`fn` stay seam-exact via `batch_slice=(f0, B_total)`. Unlike
    `run_tiled` this narrows the BATCH axis (dim 0), so a LATENT ([B,C,H,W]) batch-strips
    safely (its batch is also dim 0). Falls back to a whole-batch cook when the batch isn't a
    single shared size (heterogeneous / broadcast-only inputs).

    ROI-6 GROUNDWORK: this has NO engine caller yet — the mechanism + analysis
    (`tex_roi.batch_sliceable`) ship so a host / a future OOM-ladder path can drive it; wiring
    it into the auto-tiling memory-pressure path is the near-term follow-up."""
    def _whole():
        return _cook_whole(interp, program, bindings, type_map, device, latent_channel_count,
                           output_names, used_builtins, precision, time_context,
                           cancel, on_progress)

    B_total = shared_batch_size(bindings)
    if B_total is None or B_total < 2:
        return _whole()
    n_strips = max(1, min(n_strips, B_total))
    bounds = [(i * B_total) // n_strips for i in range(n_strips)] + [B_total]
    # A batch-BROADCAST companion output (shape[0]==1 — e.g. `@MASK = @B` passing through a
    # [1,H,W] input) is frame-independent: it must be returned once as [1,...], matching the
    # whole-batch cook, NOT stitched into a [B_total,...] buffer. A per-frame output has
    # shape[0]==(f1-f0) at every strip (including size-1 strips). The two are only
    # distinguishable at a MULTI-frame strip (there a broadcast output is 1 while a per-frame
    # output is >1); if the caller asked for so many strips that every strip is size 1, they
    # are indistinguishable — cook whole-batch (no memory win over frame-by-frame anyway).
    if max(bounds[k + 1] - bounds[k] for k in range(n_strips)) < 2:
        return _whole()

    batched: dict = {}     # name -> list[(f0, f1, tensor)] — a per-frame slice, stitched
    broadcast: dict = {}   # name -> tensor — frame-independent / scalar / string (first wins)
    for k in range(n_strips):
        f0, f1 = bounds[k], bounds[k + 1]
        if f1 <= f0:
            continue
        _cancel_check(cancel)               # SCHED-3 yield F: abort between batch strips
        strip_bindings = {}
        for name, v in bindings.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == B_total:
                strip_bindings[name] = v.narrow(0, f0, f1 - f0)
            else:
                strip_bindings[name] = v          # broadcast singleton / scalar passes through
        # SCHED-3: forward CANCEL (per-statement abort within the strip) but not on_progress —
        # the batch-strip loop owns the progress axis (mirrors run_tiled).
        res = interp.execute(program, strip_bindings, type_map, device=device,
                             latent_channel_count=latent_channel_count,
                             output_names=output_names, used_builtins=used_builtins,
                             precision=precision, batch_slice=(f0, B_total),
                             time_context=time_context, cancel=cancel)
        for name, so in res.items():
            if isinstance(so, torch.Tensor) and so.dim() >= 1 and so.shape[0] == (f1 - f0):
                batched.setdefault(name, []).append((f0, f1, so))
            else:
                broadcast.setdefault(name, so)    # shape[0]==1 at a multi-frame strip, scalar, string
        if on_progress is not None:               # SCHED-3: report per-batch-strip fraction
            _progress(on_progress, "strip", (k + 1) / n_strips)
    outputs: dict = dict(broadcast)
    for name, parts in batched.items():
        if name in broadcast:
            continue                              # also seen broadcast at a multi-frame strip → store once
        first = parts[0][2]
        full_shape = list(first.shape)
        full_shape[0] = B_total
        buf = torch.empty(full_shape, dtype=first.dtype, device=first.device)
        for f0, f1, so in parts:
            buf[f0:f1] = so
        outputs[name] = buf
    return outputs


def run_roi(interp, program, bindings, type_map, device, latent_channel_count,
            output_names, used_builtins, precision, roi, narrow_names, halo: int,
            time_context: dict | None = None, cancel=None, on_progress=None) -> dict:
    """ROI-3: cook only the output window `roi=(x0, y0, w, h, W, H)` of a full W×H image
    (the 2-D generalization of `run_tiled`'s strip). The cook region is `ROI ⊕ halo`, clamped
    to the image; `narrow_names` bindings are sliced to it (a zero-copy view, per spatial dim
    so a broadcast singleton is left intact), and every other spatial binding passes WHOLE —
    a gather reads it at absolute / full-normalized coordinates while the output stays the
    ROI. The interpreter cooks the cook-region grid seam-exact (`roi=`); each cook-region
    output is cropped back to the ROI sub-window.

    Bit-exact vs the full cook: an ROI-interior pixel of a direct-tensor halo op (blur/
    morphology) is `halo` pixels from the cook-region edge, so it reads the same real
    neighbours as the whole-image cook (a conv's interior is independent of tensor extent);
    a clamped edge replicate-pads at the true image edge either way. See
    docs/roi-spatial-laziness.md.

    LATENT ([B,C,H,W]) narrows the wrong axis, so it cooks whole-frame (the caller — the ROI
    plan — already excludes it; this is belt-and-braces)."""
    x0, y0, w, h, W, H = roi

    def _whole():
        return _cook_whole(interp, program, bindings, type_map, device, latent_channel_count,
                           output_names, used_builtins, precision, time_context,
                           cancel, on_progress)

    if latent_channel_count:
        return _whole()
    # No spatial binding to anchor the shape → a purely generative program (no image input).
    # The whole-frame cook of such a program has NO spatial grid (scalar mode), so cooking an
    # ROI would fabricate a gradient the whole cook never produces — cook whole-frame instead.
    if not any(isinstance(v, torch.Tensor) and v.dim() >= 3 for v in bindings.values()):
        return _whole()
    # Cook region = ROI ⊕ halo, clamped to the image.
    cx0, cy0 = max(0, x0 - halo), max(0, y0 - halo)
    cx1, cy1 = min(W, x0 + w + halo), min(H, y0 + h + halo)
    cw, ch = cx1 - cx0, cy1 - cy0
    if cw <= 0 or ch <= 0:
        return _whole()

    cook_bindings = {}
    for name, v in bindings.items():
        if name in narrow_names and isinstance(v, torch.Tensor) and v.dim() >= 3:
            # Narrow each spatial dim ONLY if it is full-size (a broadcast singleton
            # H==1 / W==1 is left to broadcast over the cook region, mirroring run_tiled).
            if v.shape[1] == H:
                v = v.narrow(1, cy0, ch)
            if v.shape[2] == W:
                v = v.narrow(2, cx0, cw)
        cook_bindings[name] = v

    res = interp.execute(program, cook_bindings, type_map, device=device,
                         latent_channel_count=latent_channel_count,
                         output_names=output_names, used_builtins=used_builtins,
                         precision=precision, roi=(cx0, cy0, cw, ch, W, H),
                         time_context=time_context,
                         cancel=cancel, on_progress=on_progress)

    # Crop each cook-region output to the ROI sub-window, PER SPATIAL DIM independently
    # (mirroring the per-dim narrow): an output can legitimately be full in one spatial dim
    # and broadcast (size 1) in the other — a horizontal/vertical gradient companion output,
    # a broadcast strip. An all-or-nothing `shape[1]==ch AND shape[2]==cw` crop would leave
    # such an output at the cook-region extent, uncropped and offset-wrong.
    lx, ly = x0 - cx0, y0 - cy0
    outputs = {}
    for name, val in res.items():
        if isinstance(val, torch.Tensor) and val.dim() >= 3:
            if val.shape[1] == ch:
                val = val[:, ly:ly + h]
            if val.shape[2] == cw:
                val = val[:, :, lx:lx + w]
        outputs[name] = val   # scalar / string / genuinely-broadcast dims pass through
    return outputs


def shared_tile_width(bindings) -> int | None:
    """ROI-5: the single image width shared by every spatial (dim>=3) binding that isn't a
    broadcast singleton (shape[2]==1), or None otherwise — the width twin of
    `shared_tile_height`, needed to size the full-image W a halo strip clamps against."""
    return _shared_dim_size(bindings, 2, 3)


def run_tiled_halo(interp, program, bindings, type_map, device, latent_channel_count,
                   output_names, used_builtins, precision, n_strips: int,
                   narrow_names, halo: int, time_context: dict | None = None,
                   cancel=None, on_progress=None) -> dict:
    """ROI-5: cook a HALO program (a bounded direct-tensor neighbourhood op — blur / erode /
    dilate) in `n_strips` horizontal strips, each grown by `halo` rows so an interior pixel
    reads the SAME neighbours it would in the whole-image cook. This is the seam-exact
    generalization of `run_tiled` to NON-local programs — the class `is_tile_safe` refuses and
    an 8K `gauss_blur` could not tile at all before. It is driven entirely over `run_roi`'s
    proven 2-D grow-cook-crop (a vertical strip is `run_roi` with `x0=0, w=W`), so the ROI-4
    differential oracle's bit-exactness carries over unchanged. Falls back to a whole-frame cook
    on any shape it can't strip (LATENT, no shared H, no anchor width).

    COST (inherent, not a defect): each strip cooks its `y0..y1` rows PLUS a `halo` margin on each
    side, so the `halo` rows adjacent to every interior seam are cooked in two strips — up to
    ~2× the interior work at the smallest useful strip (`4·halo` rows). That recompute IS the price
    of tiling an image too big to cook whole; the strip planner (`_halo_tile_plan`) only pays it
    under real memory pressure or the TDR cap, never on a cook that fits."""
    def _untiled():
        return _cook_whole(interp, program, bindings, type_map, device, latent_channel_count,
                           output_names, used_builtins, precision, time_context,
                           cancel, on_progress)

    if latent_channel_count:
        return _untiled()
    H_total = shared_tile_height(bindings)
    W_total = shared_tile_width(bindings)
    if H_total is None or W_total is None:
        return _untiled()
    bounds = [(i * H_total) // n_strips for i in range(n_strips)] + [H_total]
    outputs: dict = {}
    for k in range(n_strips):
        y0, y1 = bounds[k], bounds[k + 1]
        if y1 <= y0:
            continue
        _cancel_check(cancel)                   # SCHED-3 yield F: abort between halo strips
        # A full-width horizontal strip is exactly an ROI window (x0=0, w=W); run_roi grows it
        # by `halo` (clamped at the true image edges), cooks the seam-exact region grid, and
        # crops the halo back to [y0, y1). Cancel forwarded (per-statement abort within the
        # strip); on_progress suppressed — the strip loop owns the progress axis.
        strip = run_roi(interp, program, bindings, type_map, device, latent_channel_count,
                        output_names, used_builtins, precision,
                        (0, y0, W_total, y1 - y0, W_total, H_total), narrow_names, halo,
                        time_context, cancel=cancel)
        for name, so in strip.items():
            if isinstance(so, torch.Tensor) and so.dim() >= 3 and so.shape[1] == (y1 - y0):
                buf = outputs.get(name)
                if buf is None:
                    full_shape = list(so.shape)
                    full_shape[1] = H_total
                    buf = torch.empty(full_shape, dtype=so.dtype, device=so.device)
                    outputs[name] = buf
                buf[:, y0:y1] = so
            elif name not in outputs:
                outputs[name] = so              # scalar/string/broadcast: any strip suffices
        if on_progress is not None:
            _progress(on_progress, "strip", (k + 1) / n_strips)
    return outputs


def free_tensor_caches() -> None:
    """Drop every module-level tensor cache TEX holds (M-1/M-2 recovery)."""
    # MEM-1: free captured graphs FIRST — they bake addresses of the cache entries
    # cleared below, so tearing graphs down before their baked storages avoids any
    # stale replay. free_graphs_only() leaves the blacklist + RNG-poison kill switch
    # intact; clear_graph_cache() (which resets those) is now test-only.
    try:
        from .tex_runtime.graphed import free_graphs_only
        free_graphs_only()
    except Exception:
        pass
    try:
        from .tex_runtime import stdlib as _sl
        for c in (_sl._sampler_cache, _sl._grid_buf, _sl._mip_cache,
                  _sl._gauss_mip_cache, _sl._gauss_kernel_cache):
            c.clear()
    except Exception:
        pass
    try:
        from .tex_runtime import noise as _ns
        _ns._worley_offsets_cache.clear()
        _ns._worley3d_offsets_cache.clear()
    except Exception:
        pass
    try:
        # ENG-1: the interpreter singleton is the ENGINE's, not the node's. This moves
        # the documented `tex_node ↔ tex_memory` cycle to `tex_engine ↔ tex_memory` —
        # it does NOT remove it (tex_engine still calls back into tex_memory for
        # run_tiled / enforce_cache_budget). Still function-local, still load-bearing.
        from .tex_engine import _clear_all_interpreter_caches
        _clear_all_interpreter_caches()   # ENG-9: sweep every per-thread interpreter
        #                                   (_literal_cache + LAT-4 coordinate-builtin LRU)
    except Exception:
        pass
    try:
        from .tex_runtime.compiled import clear_plain_interp_caches
        clear_plain_interp_caches()    # C6: the persistent fallback interpreter's LRUs
    except Exception:
        pass
