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


class NullHostServices:
    """No host (CLI / tests / future host): unknown free memory, no cooperation hooks —
    but still recognise a torch OOM so a standalone run can drop caches and retry."""
    def get_free_memory(self, device):
        return None

    def free_memory(self, amount, device):
        pass

    def is_oom(self, exc):
        return _torch_oom(exc)

    def soft_empty_cache(self):
        pass


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
        return None

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
