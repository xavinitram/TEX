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

from .tex_compiler.ast_nodes import (
    ForLoop, WhileLoop, FunctionCall, ArrayDecl,
    BindingIndexAccess, BindingSampleAccess,
    iter_child_nodes as _iter_children,
)
from .tex_compiler.types import TEXType, TYPE_NAME_MAP

# M-4: stdlib functions whose output at a pixel depends on OTHER pixels (or the
# whole image), so a program that calls them cannot be split into horizontal
# strips. A whitelist posture: anything not here is treated as pixel-local.
_NON_LOCAL_FNS = frozenset({
    "sample", "sample_cubic", "sample_frame", "sample_grad", "sample_lanczos",
    "sample_mip", "sample_mip_gauss", "fetch", "fetch_frame",
    "gauss_blur", "bilateral_filter",
    "img_min", "img_max", "img_mean", "img_median", "img_sum",
    "erode", "dilate",   # SL-4: neighbourhood morphology, not pixel-local
})


def is_tile_safe(program) -> bool:
    """M-4: True if every pixel of the output depends only on the same pixel of
    the input (pointwise), so H can be split into strips. Excludes sample/fetch/
    blur/img-reduction calls and any binding index/sample access (a gather or a
    scatter `@OUT[x,y]=` — the latter parses as Assignment(target=BindingIndex))."""
    stack = list(program.statements)
    while stack:
        n = stack.pop()
        cls = n.__class__
        if cls is FunctionCall and n.name in _NON_LOCAL_FNS:
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


def estimate_peak_bytes(program, spatial_shape, dtype_bytes: int = 4) -> int:
    """Estimate a cook's peak transient bytes (M-1).

    Base unit is a [B,H,W,4] tensor. K scales the live-temp count: K=4 when a
    sample-family call appears inside a loop (measured godrays ~3.75), K=1 for a
    pointwise program, K=1.5 otherwise. Added terms: the static ArrayDecl bytes
    (a `vec4 arr[25]` at 2048² is ~1.6 GB — a ~17x underestimate without it) and
    ~1.33x a frame per cold mip pyramid when sample_mip-family calls are present.
    """
    if not spatial_shape:
        return 0
    B, H, W = spatial_shape
    px = B * H * W
    frame4 = px * 4 * dtype_bytes  # a [B,H,W,4] tensor

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


# MEM-2 (audit B2): per-device last-seen spatial pixel count + cached total VRAM, so the
# allocator is queried ONLY after a resolution downshift — not on every cook.
_last_trim_px: dict = {}
_total_mem_cache: dict = {}


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


def shared_tile_height(bindings) -> int | None:
    """M-4: the single image height shared by every spatial (dim>=3) binding that
    isn't a broadcast singleton (shape[1]==1), or None when zero or >1 distinct
    heights qualify (heterogeneous inputs can't be co-tiled). One source of truth
    for both the tile PLAN (`_tile_plan`) and the tile EXECUTOR (`run_tiled`)."""
    heights = {v.shape[1] for v in bindings.values()
               if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] > 1}
    return next(iter(heights)) if len(heights) == 1 else None


def run_tiled(interp, program, bindings, type_map, device, latent_channel_count,
              output_names, used_builtins, precision, n_strips: int) -> dict:
    """M-4: execute a tile-safe program in `n_strips` horizontal strips, keeping
    peak transient to ~1/n_strips of the full-image cook. Spatial bindings are
    narrowed (zero-copy views); outputs are preallocated once and strips copy_'d
    in (avoids a full-size torch.cat transient). Coordinates stay seam-exact via
    the interpreter's tile=(y0, H_total)."""
    def _untiled():
        return interp.execute(program, bindings, type_map, device=device,
                              latent_channel_count=latent_channel_count,
                              output_names=output_names, used_builtins=used_builtins,
                              precision=precision)

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
        strip_bindings = {}
        for name, v in bindings.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == H_total:
                strip_bindings[name] = v.narrow(1, y0, y1 - y0)
            else:
                strip_bindings[name] = v
        res = interp.execute(program, strip_bindings, type_map, device=device,
                             latent_channel_count=latent_channel_count,
                             output_names=output_names, used_builtins=used_builtins,
                             precision=precision, tile=(y0, H_total))
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
        from .tex_node import _get_interpreter
        interp = _get_interpreter()
        interp._literal_cache.clear()
        interp._builtins_cache_env.clear()
        interp._builtins_cache_key = None
    except Exception:
        pass
