"""
ENG-8 — measured host<->device transfer-cost model.

Two consumers need a real number for "how long does moving this tensor cost":
the placement scheduler (SCHED-2) needs `transfer_cost(edge)`, and the v0.21
3-stream pipeline's own go/no-go gate needs "transfer time ~= compute time"
(docs/xpu-transfer-scheduling.md). Nothing in the repo measured PCIe bandwidth —
the v0.20 numbers are hand-measured prose in a doc, not machine-consumable at
decision time. This module supplies the measurement.

Design (mirrors autotier's persistence machinery):
  * A once-per-process probe times H2D and D2H copies, for pinned and pageable
    host memory, at two sizes so a **latency + inverse-bandwidth** line can be
    fit (a pure bandwidth number makes tiny cross-device hops look free, which
    would mislead the scheduler). GPU timing wraps `torch.cuda.synchronize()`
    (invariant #6 — an unsynced copy measures only the enqueue).
  * The fitted (latency_ms, ms_per_byte) per (direction, pinned) is cached in
    memory and persisted to `.tex_cache/xfer.json`, keyed by device name +
    torch version, so a restart skips the probe. Best-effort throughout; a
    miss just re-probes.

This module is torch-only (invariant #1: no numpy) and imports no ComfyUI
surface — it is a pure `tex_runtime` cost oracle. Costs GUIDE placement; they
never gate correctness, so staleness is harmless.
"""
from __future__ import annotations

import json
import os

from .autotier import _median  # shared stats helper (pure, no autotier state)

# (latency_ms, ms_per_byte) per key "h2d_pinned" / "h2d_pageable" / "d2h_pinned"
# / "d2h_pageable". Empty until the first probe (or a disk load).
_MODEL: dict[str, tuple[float, float]] = {}
_probed = False

# Probe sizes (bytes). Small captures per-copy latency; large captures bandwidth.
_SMALL = 4 << 20      # 4 MiB
_LARGE = 64 << 20     # 64 MiB
_REPS = 3             # median-of-3


def _time_copy(dst, src, reps: int) -> float:
    """Median wall-ms of `dst.copy_(src, non_blocking=True)` with a synchronize
    fencing each timed copy (invariant #6). A throwaway warm copy first."""
    import time
    import torch
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return _median(samples)


def _fit_line(small_bytes: int, small_ms: float,
              large_bytes: int, large_ms: float) -> tuple[float, float]:
    """Two-point fit of ms = latency + ms_per_byte * nbytes, clamped so neither
    term goes negative (measurement noise on a laptop can invert the two points)."""
    dz = large_bytes - small_bytes
    slope = (large_ms - small_ms) / dz if dz > 0 else 0.0
    slope = max(0.0, slope)
    latency = small_ms - slope * small_bytes
    latency = max(0.0, latency)
    return (latency, slope)


def _probe() -> None:
    """Measure the four transfer lanes once. Never raises; leaves `_MODEL` empty
    on any failure (callers then fall back to a coarse default). C8/C9: `_probed`
    latches only on a DEFINITIVE outcome — no CUDA, or at least one good lane. A
    transient failure (e.g. a `cudaHostAlloc` OOM on a loaded box) or an all-
    degenerate measurement leaves it False, so a later transfer_ms re-measures once
    the bus is quiet instead of falling back to the coarse default forever."""
    global _probed
    try:
        import torch
        if not torch.cuda.is_available():
            _probed = True   # no CUDA: coarse fallback is the permanent answer
            return
        dev = torch.device("cuda")
        for pinned in (True, False):
            suffix = "pinned" if pinned else "pageable"
            small_h = torch.empty(_SMALL, dtype=torch.uint8, pin_memory=pinned)
            large_h = torch.empty(_LARGE, dtype=torch.uint8, pin_memory=pinned)
            small_d = torch.empty(_SMALL, dtype=torch.uint8, device=dev)
            large_d = torch.empty(_LARGE, dtype=torch.uint8, device=dev)
            # H2D: host -> device
            h2d = _fit_line(_SMALL, _time_copy(small_d, small_h, _REPS),
                            _LARGE, _time_copy(large_d, large_h, _REPS))
            # D2H: device -> host
            d2h = _fit_line(_SMALL, _time_copy(small_h, small_d, _REPS),
                            _LARGE, _time_copy(large_h, large_d, _REPS))
            # Store only NON-DEGENERATE lanes. Noise on a loaded box can invert the
            # two measured points (small_ms >= large_ms) -> slope clamps to 0 -> a flat
            # model where transfer_ms is constant for every size (and its monotonicity
            # test flakes). Drop such a lane so transfer_ms falls back to the coarse
            # positive-slope default instead of persisting a bad line.
            for lane_key, lane in ((f"h2d_{suffix}", h2d), (f"d2h_{suffix}", d2h)):
                if lane[1] > 0:
                    _MODEL[lane_key] = lane
        if _MODEL:
            _persist()
            _probed = True   # measured >=1 good lane -> done
        # else: every lane was degenerate (loaded bus) -> don't latch; retry later.
    except Exception:
        _MODEL.clear()      # transient (OOM etc.) -> _probed stays False -> retry


# Coarse fallback if the probe never ran / CUDA absent: (latency_ms, ms_per_byte),
# direction-independent — ~25 GB/s pinned vs ~12 GB/s pageable PCIe. Only keeps
# transfer_ms finite before a real probe; never gates anything.
_FALLBACK_PINNED = (0.02, 4.0e-8)
_FALLBACK_PAGEABLE = (0.03, 8.3e-8)


def transfer_ms(nbytes: int, pinned: bool, direction: str) -> float:
    """Estimated wall-ms to move `nbytes` across PCIe. `direction` is 'h2d' or
    'd2h'. Probes lazily on first call. Never raises."""
    ensure_probed()  # its own early-return makes a caller-side _probed check redundant
    key = f"{direction}_{'pinned' if pinned else 'pageable'}"
    latency, slope = _MODEL.get(key) or (_FALLBACK_PINNED if pinned else _FALLBACK_PAGEABLE)
    return latency + slope * max(0, int(nbytes))


def ensure_probed() -> None:
    """Probe once if not already done (loading the persisted model first)."""
    if _probed:
        return
    _load()
    if not _MODEL:
        _probe()


def model() -> dict:
    """The fitted lanes (for tests). Probes lazily (may cost ~100-300 ms once)."""
    ensure_probed()
    return dict(_MODEL)


def peek() -> dict:
    """The cached lanes WITHOUT triggering a probe — loads the persisted model if
    present but never measures. For cheap status surfaces (`tex doctor`), where a
    300 ms bandwidth probe on every report would be hostile. Empty = not yet
    measured this session."""
    if not _MODEL and not _probed:
        _load()
    return dict(_MODEL)


def reset() -> None:
    """Test hook: drop the in-memory model and force a re-probe next call."""
    global _probed
    _MODEL.clear()
    _probed = False


# ── Persistence (mirrors autotier: versioned by torch version + device, atomic
#    replace, best-effort). CUDA graphs / compiled artifacts aside, a bandwidth
#    number is machine-stable, so it survives restarts. ─────────────────────
def _persist_path() -> str | None:
    try:
        from ..tex_cache import get_cache
        d = get_cache()._cache_dir
        os.makedirs(d, exist_ok=True)
        return os.path.join(str(d), "xfer.json")
    except Exception:
        return None


def _version_tag() -> str:
    try:
        import torch
        # Name the device the probe ACTUALLY measured on — _probe uses an index-less
        # torch.device("cuda"), which resolves to current_device(), not necessarily 0.
        # On a multi-GPU box, tagging with device 0 would let a model measured on GPU 1
        # be loaded for GPU 0 (or vice versa) when the names happen to match.
        name = (torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available() else "cpu")
        return f"{name}_{torch.__version__.split('+')[0]}"
    except Exception:
        return "0"


def _load() -> None:
    global _probed
    p = _persist_path()
    if not p or not os.path.exists(p):
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != _version_tag():
            return
        lanes = data.get("lanes") or {}
        for k, v in lanes.items():
            _MODEL[k] = (float(v[0]), float(v[1]))
        if _MODEL:
            _probed = True   # a valid cached model means the probe already ran
    except Exception:
        _MODEL.clear()


def _persist() -> None:
    p = _persist_path()
    if not p or not _MODEL:
        return
    try:
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": _version_tag(),
                       "lanes": {k: [round(a, 8), b] for k, (a, b) in _MODEL.items()}}, f)
        os.replace(tmp, p)
    except Exception:
        pass
