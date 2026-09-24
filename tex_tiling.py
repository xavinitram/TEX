"""
ENG-14 — `tex_tiling`: cook-fit planning. Does this cook fit whole, and if not, how
is it cut?

Moved here whole out of `tex_engine` (ENG-14) with no body changed: the cook-fit
planners (`_tile_plan`, `_halo_tile_plan`, `_preflight_memory`), ROI-5's per-strip TDR
time floor and its budget, and the scalar-param helper the halo planner and the OOM
ladder share.

The seam this makes explicit: **`tex_tiling` plans, `tex_memory` runs, `tex_roi`
decides what may be narrowed** — already the house pattern (`tex_roi.roi_plan` decides,
`tex_memory.run_roi` executes), with the third planner now on the analysis side of a
line the tree already draws. `_preflight_memory` belongs with the planners rather than
the executors because it is their shared pressure test: it and `_tile_plan` both consult
`estimate_peak_bytes` + `device_total_mem`, and the free-VRAM reading it buys is handed
forward as `_tile_plan`'s `free_hint`.

PERF-6 added the second half of that hand-off, for the cook where the preflight buys no
reading at all: `_free_foreign` records what the last LIVE reading implied about the bytes
on the device that torch's allocator does not own, and `_free_memory_for_plan` serves the
planners from it whenever that already settles their question. The decomposition, the margin
that keeps it from ever changing an answer, and the one thing it cannot see are written out
above `_free_foreign`.

This module is a LEAF: its only package import at load is the host seam
(`get_host_services`), which sits below it in the layer table, so `tex_engine` imports
it at load and re-exports every name. The `tex_memory` / `tex_roi` / `autotier` imports
inside the bodies stay FUNCTION-LOCAL exactly as they were — that edge is deliberate
(ARCHITECTURE.md's "two logical import cycles"), and nothing here hoists it. Pinned by
`tests/test_v027_phase1.py`, `tests/test_v035_hygiene.py` and
`tests/test_v036_region_dependence.py`.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from .tex_runtime.host import get_host_services, services_generation


# ── PERF-6: one live free-VRAM reading, reused by the cooks it already answers ──────────────

#: What the last LIVE host free-VRAM reading implies about the bytes on a CUDA device that
#: this process's torch allocator does NOT own — per device index. Written only where a live
#: query is actually paid; read by the planners so they need not pay for another.
#:
#: THE IDENTITY THIS RESTS ON. The free figure a host reports for a CUDA device is
#: `driver_free + (torch_reserved - torch_allocated)`: the driver's own number plus the blocks
#: torch has cached and will hand straight back out (`tex_runtime/host._cuda_free_memory`
#: computes exactly that, and the ComfyUI host's own answer has the same shape). Substituting
#: `driver_free == total - torch_reserved - foreign` collapses `torch_reserved` out of it:
#:
#:     free  ==  total - foreign - torch_allocated
#:
#: where `foreign` is every byte on the device held by anything other than this process's
#: torch caching allocator. So a reading splits into a part that moves with every cook
#: (`torch_allocated`, re-read on each call for ~5 us) and a part a cook cannot move at all:
#: allocating an output, releasing one, evicting a cache, capturing a graph — none of it
#: reaches the device except through torch's allocator. That is the answer to "could a stale
#: value inside one cook change a decision": no. The only quantity carried across cooks is
#: `foreign`, and no cook touches it.
#:
#: Each entry is `(host_generation, foreign_bytes)`. The generation is `tex_runtime.host`'s
#: swap counter: a host REPLACED under a warm memo is the one `foreign` change that is both
#: plausible and invisible (the CLI resolving its host, a test installing a fake), and it is
#: the only one this module can be told about, so it is.
_free_foreign: "dict[int, tuple[int, int]]" = {}

#: A memoized reading may answer only when it answers with DOUBLE the room the live number
#: would have had to grant: the planners' budget is a quarter of free, so `est * 8 <= bound`
#: is `est <= budget / 2`. Two consequences, and together they are the safety argument:
#:   * a memoized reading can only ever produce the answer the live one would have. It is
#:     served only while `est` is under half the budget, so `est <= budget` holds for BOTH
#:     numbers and both reach "no strips". Every strip count TEX plans still comes from a
#:     reading bought inside that same call — a pressured cook is never planned on an old one.
#:   * the one thing the decomposition cannot see is `foreign` MOVING (another process taking
#:     VRAM, or an allocation that bypasses torch's allocator). The margin means such a
#:     consumer must take more than HALF the free VRAM between one cook and the next to flip
#:     an answer, and it is self-correcting: the moment a cook is big enough for the margin to
#:     fail, a live query is paid and `foreign` is re-learned.
_FREE_MEMO_MARGIN = 8.0


def _cuda_index(device) -> "int | None":
    """The CUDA device index of *device*, or None when it is not a CUDA device."""
    try:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type != "cuda":
            return None
        return dev.index if dev.index is not None else torch.cuda.current_device()
    except Exception:
        return None


def _torch_allocated(idx: int) -> "int | None":
    """Bytes this process's torch caching allocator currently holds ALLOCATED on *idx*.

    ONE nested-dict snapshot, never `torch.cuda.memory_allocated()`: that convenience
    re-derives the whole statistics table and FLATTENS it at ~38 us a call, while the nested
    snapshot both are built on costs ~5 us — the same measurement
    `tex_runtime/host._allocator_slack` records, for the same reason. None on any shape
    surprise, which simply makes the caller buy a live reading.

    An EMPTY snapshot is 0, not None: torch hands back `{}` for a device whose caching
    allocator has never been used, and that is the honest answer (nothing is allocated), not a
    failure. Reading it as "unknown" would make every planner in a process that has not yet
    touched the device buy its own query — which is exactly the process the first cook of a
    session runs in."""
    try:
        if not torch.cuda.is_available():
            return None
        st = torch.cuda.memory_stats_as_nested_dict(device=idx)
    except Exception:
        return None
    try:
        return int(st["allocated_bytes"]["all"]["current"]) if st else 0
    except Exception:
        return None


def forget_free_memory(device=None) -> None:
    """Drop what the last live free-VRAM reading implied (one device, or every device).

    For something that changes the device from OUTSIDE torch's allocator: a host that answers
    a different number than it used to, a caller that knows VRAM was released behind torch's
    back. A host SWAP does not need it — `services_generation()` already invalidates every
    entry — and nothing a COOK does needs it either, since a cook moves only
    `torch_allocated`, which is re-read on every call."""
    if device is None:
        _free_foreign.clear()
        return
    idx = _cuda_index(device)
    if idx is not None:
        _free_foreign.pop(idx, None)


def _query_free_memory(device, host=None) -> "float | None":
    """Buy a LIVE free-VRAM reading from the host and record what it implies about the
    device's non-torch footprint, so the next planner may not have to buy one.

    Every site in this module that asks the host goes through here: a reading that is paid for
    and then thrown away is the thing PERF-6 exists to stop."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    free = (host if host is not None else get_host_services()).get_free_memory(dev)
    idx = _cuda_index(dev)
    if idx is None:
        return free
    try:
        from .tex_memory import device_total_mem
        total = device_total_mem(dev)
        alloc = _torch_allocated(idx)
        if free is not None and total and alloc is not None:
            _free_foreign[idx] = (services_generation(),
                                  int(total) - int(free) - alloc)
        else:
            _free_foreign.pop(idx, None)   # can't decompose it → don't pretend we can
    except Exception:
        _free_foreign.pop(idx, None)
    return free


def _free_memory_bound(device, est) -> "float | None":
    """Free VRAM derived from the last live reading without buying another one — returned
    ONLY when it already settles this plan's question (`_FREE_MEMO_MARGIN`), else None so the
    caller queries. `est` is the cook's estimated peak, in bytes."""
    idx = _cuda_index(device)
    entry = _free_foreign.get(idx) if idx is not None else None
    if entry is None or entry[0] != services_generation():
        return None
    foreign = entry[1]
    try:
        from .tex_memory import device_total_mem
        total = device_total_mem(device)
    except Exception:
        return None
    alloc = _torch_allocated(idx)
    if not total or alloc is None:
        return None
    bound = int(total) - foreign - alloc
    if bound > 0 and est * _FREE_MEMO_MARGIN <= bound:
        return float(bound)
    return None


def _free_memory_for_plan(device, free_hint, est) -> "float | None":
    """The free-VRAM figure a tile plan should divide: the M-1 preflight's reading if it has
    one (P1), else the memoized decomposition if it settles the question, else a live query."""
    if free_hint is not None:
        return free_hint
    free = _free_memory_bound(device, est)
    return free if free is not None else _query_free_memory(device)


# ── Memory planning ──────────────────────────────────────────────────────────

def _tile_plan(program, bindings: dict[str, Any], device,
               latent_channel_count: int = 0, dtype_bytes: int = 4,
               fingerprint: str | None = None,
               free_hint: float | None = None,
               code: str | None = None, binding_types: dict | None = None) -> int | None:
    """M-4: strip count if the cook should be tiled (tile-safe + under memory
    pressure), else None. cuda only; needs the host's free-memory query.
    MEM-3: dtype_bytes=2 in fp16 mode halves the peak estimate (a fp16 cook that
    fits shouldn't be tiled as if it were fp32).

    P1: `free_hint` is the free-VRAM reading prepare()'s M-1 preflight already bought. On
    a default CUDA cook these two sites were the ONLY callers and ran microseconds apart
    with nothing allocating between them, yet each paid for its own query — measured at
    2 x ~68 us, ~42% of a 345 us 256² cook, ~94% of it in the query rather than the
    estimator. The caller passes None when the reading is stale (the preflight asked the
    host to unload) or absent, and then we buy our own.

    PERF-6, and it corrects the paragraph above for the cook that actually ships: on a default
    whole-frame CUDA cook the preflight buys NOTHING — LAT-2's cheap path returns before the
    query — so `free_hint` is None on every stage that reaches here and P1's hand-off is inert
    exactly where it was written to help. Measured: `host.get_free_memory` seven times per
    ten-stage 1024² frame at 90-112 us a call, ~10% of the whole-frame tick, all of it spent
    concluding `est <= budget` and returning None. `_free_memory_for_plan` answers those from
    the last live reading (see `_free_foreign`) and buys a query only for a cook whose plan
    could depend on the number.

    The ESTIMATE is deliberately NOT hinted, though it looks like the same redundancy: the
    two sites call it with different `dtype_bytes`. The preflight runs before `auto`
    resolves, so it passes 4; here, post-resolution, an auto->fp16 cook passes 2 — reusing
    the preflight's number would hand this function a 2x-inflated peak and over-tile
    exactly the cooks `auto` accepted (measured 67108864 vs 33554432 at 2048²). It is also
    only ~6% of the cost.

    TRK-25: `code` is the raw source, carried only so the region-dependence gate at the end
    can see a `//!tex X.Y` pragma; `None` means "no pragma visible", the conservative read.
    `binding_types` is that same gate's other input, and is conservative when absent too."""
    if not str(device).startswith("cuda"):
        return None  # host.get_free_memory returns None off a host → no tiling
    # M-4 safety: never tile a LATENT ([B,C,H,W] — dim 1 is channels, not
    # height) or a cook whose spatial bindings disagree on height (they can't
    # be co-tiled). run_tiled re-checks, but planning here avoids a bogus
    # peak estimate off the wrong axis.
    if latent_channel_count:
        return None
    try:
        from .tex_memory import (is_tile_safe_cached, estimate_peak_bytes, shared_tile_height,
                                 _non_spatial_for)
        if not is_tile_safe_cached(program, fingerprint):  # P4: memoized per fingerprint
            return None
        # TRK-165: the same non_spatial exclusion TRK-163 threaded into shared_tile_height's
        # executor-side callers (`run_tiled`), now also fed here on the planning side — a
        # registered non-spatial binding (a LUT) whose own leading dim coincidentally equals
        # H must not skew this decision either. Decision-only: this function never slices a
        # binding itself.
        non_spatial = _non_spatial_for(program)
        H = shared_tile_height(bindings, non_spatial)
        if H is None:
            return None
        spatial = None
        for v in bindings.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == H:
                spatial = (v.shape[0], v.shape[1], v.shape[2])
                break
        if spatial is None:
            return None
        est = estimate_peak_bytes(program, spatial, dtype_bytes, fingerprint)  # P4/LAT-2: memoized walk
        free = _free_memory_for_plan(device, free_hint, est)
        if not free or est <= 0:
            return None
        budget = 0.25 * free
        if est <= budget:
            return None  # no pressure — don't pay the launch tax
        n = math.ceil(est / budget)
        max_strips = max(1, spatial[1] // 64)  # ≥64-row strip floor
        n = min(n, max_strips)
        if n < 2:
            return None
        # TRK-25, and deliberately the LAST question this function asks: a region-dependent
        # program computes something different in a strip than whole-frame, so whole-frame-or-
        # OOM is the correct answer and a wrong picture is not. Asking it HERE rather than
        # beside `is_tile_safe_cached` is invariant 7 — an unpressured cook never gets here.
        from . import tex_roi
        if tex_roi.region_dependent_cached(program, fingerprint, binding_types, code):
            return None
        return n
    except Exception:
        return None


# ROI-5/WDDM: a per-strip cook-time ceiling. A display GPU's driver resets (TDR) any kernel
# that runs past ~2 s, killing the cook — so the strip planner caps estimated per-STRIP time,
# not just bytes. ~1.8 s leaves headroom under the 2 s watchdog.
_TDR_BUDGET_MS = 1800.0


def _tdr_strip_floor(fingerprint, spatial, precision, device) -> int:
    """ROI-5: the minimum strip count that keeps each strip's estimated cook time under the ~2 s
    WDDM TDR watchdog, derived from autotier's persisted WHOLE-FRAME median (BlinkScript's driver
    timeouts are this failure, un-planned-for). Best-effort: 0 when the program was never measured
    on this device (the very first big cook can't be pre-timed — the reach of persisted medians)."""
    if not str(device).startswith("cuda"):
        return 0
    try:
        from .tex_runtime import autotier
        whole_ms = autotier.cook_ms(autotier.make_key(fingerprint, "cuda", precision, spatial))
        if whole_ms and whole_ms > _TDR_BUDGET_MS:
            return int(math.ceil(whole_ms / _TDR_BUDGET_MS))
    except Exception:
        pass
    return 0


def _scalar_params(bindings) -> dict:
    """The foldable scalar/bool/int params of a binding set — what `tex_roi`/`tex_lazy` fold to
    resolve halo radii and dead branches. Shared by the halo planner and the OOM ladder (the
    default `prepare()` param loop keeps its own inline copy to stay off a per-cook call frame)."""
    return {n: v for n, v in bindings.items() if isinstance(v, (bool, int, float))}


def _halo_tile_plan(program, code, bindings, device, latent_channel_count, dtype_bytes,
                    fingerprint, free_hint, precision, binding_types=None):
    """ROI-5: `(n_strips, narrow_names, halo)` when a NON-tile-safe program is HALO-tileable — a
    bounded direct-tensor neighbourhood op (blur / erode / dilate), which `tex_roi.roi_plan`
    reports executable with a positive cook halo — and either memory pressure OR the TDR time cap
    calls for strips; else None. This is the class `is_tile_safe` refuses, so it never overlaps
    the pointwise `_tile_plan`. cuda + non-latent only (fused chains are excluded by the caller).

    CHEAP-GATED for invariant #7: a small cook returns BEFORE `roi_plan`, `_scalar_params`, and the
    free-VRAM driver query. `est` is a memo hit (the M-1 preflight already walked it this cook),
    `device_total_mem` is cached, and `_tdr_strip_floor` is a dict lookup — so a program a small
    fraction of VRAM with no TDR-risk median can't need tiling and never pays the ~61 µs
    `get_free_memory`, exactly mirroring `_preflight_memory`'s `est < total // 8` skip. A
    program that gets PAST that gate now asks `_free_memory_for_plan` (PERF-6) rather than the
    host directly, which is the same question with the last live reading consulted first."""
    if latent_channel_count or not str(device).startswith("cuda"):
        return None
    try:
        # Hot-path bail FIRST: a POINTWISE (tile-safe) program reaches here whenever `_tile_plan`
        # found no pressure, but halo tiling is only for the NON-tile-safe blur/morphology class (a
        # tile-safe program's roi_plan has halo 0 → None below anyway). The memoized tile-safe check
        # lets a pointwise cook skip roi_plan/estimate/the dim scans AND the imports below entirely —
        # this runs on every default pointwise CUDA cook (invariant #7).
        from .tex_memory import is_tile_safe_cached
        if is_tile_safe_cached(program, fingerprint):
            return None
        from .tex_memory import (shared_tile_height, shared_tile_width, estimate_peak_bytes,
                                 device_total_mem, _non_spatial_for)
        # TRK-165: same exclusion as `_tile_plan` above — decision-only, this planner never
        # slices a binding itself either.
        non_spatial = _non_spatial_for(program)
        H = shared_tile_height(bindings, non_spatial)
        W = shared_tile_width(bindings, non_spatial)
        if H is None or W is None:
            return None
        # Batch of the height-H anchor image — guaranteed present, since `shared_tile_height`
        # derived `H` from exactly such a tensor (a StopIteration if that ever breaks is caught by
        # the outer guard → None). The cook is sized off the SHARED height/width (both skip
        # broadcast singletons), never the anchor's own width — a [B,H,1] companion bound first
        # would otherwise under-size `est` by a factor of W and silently zero the TDR px-bucket.
        batch = next(v.shape[0] for v in bindings.values()
                     if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == H)
        spatial = (batch, H, W)
        est = estimate_peak_bytes(program, spatial, dtype_bytes, fingerprint)   # memoized walk
        tdr_floor = _tdr_strip_floor(fingerprint, spatial, precision, device)   # dict lookup
        total = device_total_mem(device)                                       # cached, no query
        if tdr_floor < 2 and total and 0 < est < total // 8:
            return None    # not big enough for memory pressure, not TDR-risky → skip (no free query)
        from . import tex_roi
        plan = tex_roi.roi_plan(code, _scalar_params(bindings), binding_types)
        if not plan.executable or plan.halo <= 0 or not plan.narrow:
            return None
        halo = plan.halo
        # Strip floor: ≥ 4·halo rows/strip so the grown-halo overhead stays under ~50%.
        max_strips = max(1, H // max(64, 4 * halo))
        n_mem = 0
        free = _free_memory_for_plan(device, free_hint, est)   # PERF-6, as in `_tile_plan`
        if free and est > 0 and est > 0.25 * free:
            n_mem = math.ceil(est / (0.25 * free))
        n = min(max(n_mem, tdr_floor), max_strips)
        if n < 2:
            return None
        # TRK-25, at the same late point `_tile_plan` uses. What CLOSED this route is the
        # `roi_plan` call above: a region-dependent program is not executable, so the plan was
        # already refused. This is kept as a local guard, stating the refusal where it returns.
        if tex_roi.region_dependent_cached(program, fingerprint, binding_types, code):
            return None
        return (n, plan.narrow, halo)
    except Exception:
        return None


def _preflight_memory(program, bindings: dict[str, Any], device,
                      dtype_bytes: int = 4, fingerprint=None) -> float | None:
    """M-1: if the estimated cook peak exceeds free VRAM, free resident
    models first (best-effort; never raises). MEM-3: dtype_bytes=2 for an explicit
    fp16 cook (auto is still unresolved here, so it stays the conservative 4).

    Returns the free-VRAM reading it bought, as a `free_hint` for `_tile_plan` to reuse
    instead of buying a second query microseconds later (P1). Returns None whenever
    `_tile_plan` must query for itself — either the reading is unknown/uncomputed, OR the
    preflight just asked the host to unload models, which raises true free and makes the
    number STALE-LOW. Collapsing "unknown" and "stale" into a single None is exactly what
    the caller wants: both mean "don't trust this, re-read."

    LAT-2: `fingerprint` memoizes the peak-estimate AST walk (see estimate_peak_bytes), and
    the estimate then gates whether the free-VRAM query below runs at all."""
    host = get_host_services()  # PORT-1: a host with no free-memory answer → this no-ops
    try:
        # CF-6: the peak estimate must describe the grid the cook will actually use. Under
        # first-wins this site and the cook agreed by construction (both collapsed to the first
        # binding); the consensus moved the cook and would have left this mirror behind,
        # under-estimating peak bytes by up to H× on a disagreeing binding set — i.e. the
        # model-unload preflight quietly no-ops on exactly the cook that needed it.
        from .tex_runtime.interpreter import _consensus_extent
        spatial = _consensus_extent(bindings, program)
        if spatial is None:
            return None
        from .tex_memory import estimate_peak_bytes, device_total_mem
        est = estimate_peak_bytes(program, spatial, dtype_bytes, fingerprint)
        if est <= 0:
            return None
        # LAT-2: `host.get_free_memory` below is a live driver query measured at ~61 us — 92%
        # of a 256² prepare(), on EVERY CUDA cook. Its only effect is to unload models when
        # free < est + headroom, which cannot happen when the estimate is a small fraction of
        # total VRAM and free is gigabytes. So when est is under total//8 (mirrors
        # cache_budget_bytes's VRAM fraction), skip the query AND the pre-unload entirely:
        # the rare genuine-pressure case is still caught by ComfyUI's host OOM ladder
        # (unload_all_models + retry) and the engine's own ENG-2 strip-retry ladder.
        total = device_total_mem(device)
        if total and est < total // 8:
            return None
        dev_t = torch.device(device)
        # PERF-6: through the recorder, so the reading this cook pays for is also the one the
        # tile planners downstream may reuse instead of buying a second.
        free = _query_free_memory(dev_t, host)
        headroom = 128 * 1024 * 1024
        if free is not None and free < est + headroom:
            host.free_memory(est + 256 * 1024 * 1024, dev_t)
            # PERF-6: an unload may release bytes torch's allocator never owned, which is
            # precisely the `foreign` term the decomposition holds fixed — so void it.
            forget_free_memory(dev_t)
            return None            # stale-low after the unload — make _tile_plan re-query
        return free
    except Exception:
        return None
