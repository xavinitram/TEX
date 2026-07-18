"""
PORT-1 — the host-services seam.

TEX cooperates with its host (ComfyUI) for memory management — preflighting free VRAM,
asking the host to unload models before a big cook, detecting OOM, soft-emptying the
cache. But TEX must also run **host-agnostic**: a `tex run` CLI, the test suite, or a
future non-ComfyUI host. This module is the ONE place `comfy.model_management` is
imported; everything else talks to the `HostServices` protocol. A grep-lint
(`test_port1_import_lint`) pins the `comfy.*` import here forever, so the coupling can
never re-scatter.

`get_host_services()` returns `ComfyHostServices` when ComfyUI is importable, else the
no-op `NullHostServices` (which still detects a torch OOM so standalone runs recover).
"""
from typing import Protocol

import torch


def _torch_oom(e) -> bool:
    oom_t = getattr(torch.cuda, "OutOfMemoryError", None)
    return isinstance(oom_t, type) and isinstance(e, oom_t)


class HostServices(Protocol):
    """The host cooperation surface TEX depends on (all optional / best-effort)."""
    def get_free_memory(self, device) -> "float | None": ...
    def free_memory(self, amount, device) -> None: ...
    def is_oom(self, exc) -> bool: ...
    def soft_empty_cache(self) -> None: ...
    def get_user_dir(self) -> "str | None": ...   # LANG-5: per-user data dir, or None


def _allocator_slack(idx: int) -> int:
    """Bytes torch has RESERVED but not allocated — cached blocks it will hand back out.

    Read from ONE `memory_stats_as_nested_dict` snapshot rather than via
    `torch.cuda.memory_reserved()` + `memory_allocated()`. Those two look like the
    obvious call and are a trap: each re-derives the whole stats table and FLATTENS it
    into a dotted-key dict, measured at ~38 us apiece — 76 us on a cook that can be
    263 us. The nested snapshot they are both built on costs ~5 us once (measured,
    sm_120/torch 2.12). This sits in the preflight of every CUDA cook, so that
    difference is the whole cost of ENG-2.

    Returns 0 on any shape surprise: slack is a refinement to the driver's number, and
    a wrong refinement is worse than none."""
    try:
        st = torch.cuda.memory_stats_as_nested_dict(device=idx)
        res = st["reserved_bytes"]["all"]["current"]
        alloc = st["allocated_bytes"]["all"]["current"]
        return max(int(res) - int(alloc), 0)
    except Exception:
        return 0


def _cuda_free_memory(device) -> "float | None":
    """ENG-2: free VRAM on *device*, measured directly from the driver + torch's
    allocator. None off CUDA, or if anything is unavailable.

    `mem_get_info` alone UNDER-reports: torch's caching allocator holds freed blocks as
    `reserved`, which the driver counts as used but torch will happily hand back out. So
    the cached-but-unallocated slack is added — otherwise a standalone cook would look
    starved and tile itself pointlessly. This is the same accounting a host does."""
    try:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        driver_free, _total = torch.cuda.mem_get_info(idx)
        return float(driver_free + _allocator_slack(idx))
    except Exception:
        return None


class NullHostServices:
    """No host (CLI / tests / a standalone engine): no cooperation hooks, but it still
    recognises a torch OOM, and since ENG-2 it can answer "how much VRAM is free?" on
    its own.

    That answer is what makes the Null host a real host rather than a stub. Every memory
    decision TEX makes — the M-1 preflight, M-4 strip tiling, the OOM ladder — is gated
    on a free-memory number, and returning None disabled all of them: an 8K cook that
    tiled happily under ComfyUI simply OOMed under `tex_api`. It cannot ask anyone to
    unload models (there is nobody to ask), so `free_memory` stays a no-op — but it no
    longer has to pretend it doesn't know how much room it has."""
    def get_free_memory(self, device):
        return _cuda_free_memory(device)

    def free_memory(self, amount, device):
        pass  # nobody to ask: a standalone host owns no models to unload

    def is_oom(self, exc):
        return _torch_oom(exc)

    def soft_empty_cache(self):
        pass

    def get_user_dir(self):
        return None  # LANG-5: no host user dir standalone — the store falls back to the cache dir


class ComfyHostServices:
    """Delegates to `comfy.model_management` — the ONLY consumer of that import in TEX.
    Every call is best-effort (a host API that's missing or raises degrades to the Null
    behaviour) so a ComfyUI version bump can never hard-fail a cook."""
    def __init__(self, mm):
        self._mm = mm

    def get_free_memory(self, device):
        try:
            if hasattr(self._mm, "get_free_memory"):
                return self._mm.get_free_memory(device)
        except Exception:
            pass
        # ENG-2: a host whose API is missing/broken falls back to measuring the device
        # itself rather than to None — degrading to "I don't know" would silently disable
        # preflight, tiling and the OOM ladder on a box where the numbers are readable.
        return _cuda_free_memory(device)

    def free_memory(self, amount, device):
        try:
            if hasattr(self._mm, "free_memory"):
                self._mm.free_memory(amount, device)
        except Exception:
            pass

    def is_oom(self, exc):
        try:
            if hasattr(self._mm, "is_oom"):
                return bool(self._mm.is_oom(exc))
            if hasattr(self._mm, "OOM_EXCEPTION"):
                return isinstance(exc, self._mm.OOM_EXCEPTION)
        except Exception:
            pass
        return _torch_oom(exc)

    def soft_empty_cache(self):
        try:
            if hasattr(self._mm, "soft_empty_cache"):
                self._mm.soft_empty_cache()
        except Exception:
            pass

    def get_user_dir(self):
        # LANG-5: ComfyUI's per-user data root (multi-user aware; falls back to <comfy>/user).
        # folder_paths is a host module — imported here, inside the PORT-1 boundary, only.
        try:
            import folder_paths
            if hasattr(folder_paths, "get_user_directory"):
                return folder_paths.get_user_directory()
        except Exception:
            pass
        return None


_cached = None


def get_host_services():
    """The process-wide host services (cached): ComfyUI if importable, else Null."""
    global _cached
    if _cached is None:
        try:
            import comfy.model_management as mm  # the ONE import of it (PORT-1 lint)
            _cached = ComfyHostServices(mm)
        except Exception:
            _cached = NullHostServices()
    return _cached


def set_host_services(services) -> None:
    """Override the host services (tests, or a non-ComfyUI host wiring itself in)."""
    global _cached
    _cached = services


def reset_host_services() -> None:
    """Drop the cached resolution (tests)."""
    global _cached
    _cached = None
