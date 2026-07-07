"""
M-1/M-2 — memory cooperation for TEX.

`estimate_peak_bytes` gives a cheap, static upper-ish estimate of a cook's peak
transient VRAM so the node can preflight `comfy.model_management.free_memory`
and retry once on OOM instead of failing while GBs sit locked in resident
models. `free_tensor_caches` drops every module-level tensor cache TEX holds.
"""
from __future__ import annotations

import os

import torch

from .tex_compiler.ast_nodes import (
    ForLoop, WhileLoop, FunctionCall, ArrayDecl,
    BindingIndexAccess, BindingSampleAccess,
    iter_child_nodes as _iter_children,
)
from .tex_compiler.type_checker import TEXType, TYPE_NAME_MAP

# M-4: stdlib functions whose output at a pixel depends on OTHER pixels (or the
# whole image), so a program that calls them cannot be split into horizontal
# strips. A whitelist posture: anything not here is treated as pixel-local.
_NON_LOCAL_FNS = frozenset({
    "sample", "sample_cubic", "sample_frame", "sample_grad", "sample_lanczos",
    "sample_mip", "sample_mip_gauss", "fetch", "fetch_frame",
    "gauss_blur", "bilateral_filter",
    "img_min", "img_max", "img_mean", "img_median", "img_sum",
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


def _total_cache_bytes() -> int:
    return sum(_entry_bytes(e, ex) for c, ex in _budget_caches() for e in c.values())


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


def enforce_cache_budget(device) -> None:
    """Evict oldest cache entries (mip caches first — they pin full frames) until
    total tensor-cache bytes are under budget. Best-effort; never raises."""
    try:
        budget = cache_budget_bytes(device)
        if _total_cache_bytes() <= budget:
            return
        evicted = False
        for cache, _ in _budget_caches():
            while len(cache) > 1 and _total_cache_bytes() > budget:
                cache.popitem(last=False)
                evicted = True
            if _total_cache_bytes() <= budget:
                break
        if evicted:
            # UC-1 safety: a captured CUDA graph may bake the device address of a
            # warmup-populated stdlib cache entry (e.g. a sampler batch-index
            # tensor). Freeing that storage here would let a later replay read
            # reused/foreign memory. Tear down the graph cache so those programs
            # re-capture against live buffers instead of replaying stale ones.
            try:
                from .tex_runtime.graphed import clear_graph_cache
                clear_graph_cache()
            except Exception:
                pass
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
    try:
        from .tex_runtime.graphed import clear_graph_cache
        clear_graph_cache()
    except Exception:
        pass
