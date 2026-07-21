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


# ── SCHED-3: cancellation + progress ──────────────────────────────────────────
# A cook is a ladder of tier attempts, strip loops, and per-statement tree-walks. A host
# driving an interactive viewer must be able to ABANDON a stale cook the instant a newer
# edit arrives (the most-used code path in any scrubbing UI) and to report progress. These
# two primitives ride the VALUE channel exactly like ENG-7's `time_context` — threaded
# host → prepare() → ExecContext → every tier + interp.execute — and are NEVER part of any
# fingerprint / cache / lineage key (a per-cook token identity in a key would poison every
# cache). They live here, the lowest-level seam every consumer (engine/memory/interpreter)
# already imports, so no module grows a circular import to reach the exception type.
#
# HONEST GRANULARITY (documented, not a bug): checks fire between tier attempts, before the
# fp16/OOM re-cooks, per strip, and per TOP-LEVEL interpreter statement. An in-flight torch
# kernel and a `for`-loop body are NOT preempted — the floor is per-statement / per-strip.
class CookCancelled(Exception):
    """Raised at a cook yield point when the host's CancelToken reports cancellation.

    A plain Exception, deliberately unrelated to any OOM type: `tex_engine.run`'s OOM ladder
    keys on `_oom_in_chain`, and a CookCancelled must propagate straight out, never be
    mistaken for a recoverable OOM and silently retried."""


class CancelToken(Protocol):
    """The host's cooperative-cancellation surface. `check()` returns normally to continue
    or raises `CookCancelled` to abort. A host wires this to its own interrupt flag (e.g.
    ComfyUI's `throw_exception_if_processing_interrupted`, a newer-edit epoch, a SIGINT)."""
    def check(self) -> None: ...


def _cancel_check(token) -> None:
    """Poll a cancel token if one was supplied (a no-op when None — the default path pays a
    single `is not None` per yield point). A non-CookCancelled raised by a misbehaving token
    is left to propagate: swallowing it would turn a broken host into an un-abortable cook."""
    if token is not None:
        token.check()


def _report_progress(cb, phase: str, frac: float) -> None:
    """Invoke an on_progress(phase, frac) callback if supplied, best-effort — a host's
    progress sink must never be able to fail a cook (mirrors the tier_trace posture)."""
    if cb is not None:
        try:
            cb(phase, frac)
        except Exception:
            pass


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
    # SCHED-3 bridge: hand the engine a CancelToken bound to the host's own interrupt (or None
    # if it has none), and re-surface that host's clean interrupt after a CookCancelled.
    def cancel_token(self) -> "CancelToken | None": ...
    def raise_if_interrupted(self) -> None: ...


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

    def cancel_token(self):
        return None  # SCHED-3: no host interrupt to bridge (CLI / standalone / tests)

    def raise_if_interrupted(self):
        pass  # nothing to re-surface: there was no host interrupt flag


class _ComfyInterruptToken:
    """SCHED-3 bridge: a `CancelToken` (the protocol above) bound to ComfyUI's process-wide
    interrupt flag. `check()` raises `CookCancelled` when the host has requested an interrupt.

    READ-ONLY by design — it polls `processing_interrupted()` and does NOT clear the flag, so
    ComfyUI's own executor still sees the interrupt (the node re-surfaces the clean host
    exception via `raise_if_interrupted()`). A missing / broken host API never fails a cook: an
    unreadable flag reads as 'not interrupted'. It honours the CancelToken contract (raise
    `CookCancelled`, not comfy's `InterruptProcessingException`), so every engine `except
    CookCancelled` guard fires on the ComfyUI path exactly as on the CLI/test path."""
    __slots__ = ("_mm",)

    def __init__(self, mm):
        self._mm = mm

    def check(self) -> None:
        try:
            hit = self._mm.processing_interrupted()
        except Exception:
            return
        if hit:
            raise CookCancelled()


class ComfyHostServices:
    """Delegates to `comfy.model_management` — the ONLY consumer of that import in TEX.
    Every call is best-effort (a host API that's missing or raises degrades to the Null
    behaviour) so a ComfyUI version bump can never hard-fail a cook."""
    def __init__(self, mm):
        self._mm = mm
        # SCHED-3: the interrupt token is stateless (it only reads the flag), so build it ONCE
        # and hand back the same instance every cook — the compositor loop calls cancel_token()
        # per frame, and a fresh alloc each time is pure waste.
        self._cancel_token_obj = _ComfyInterruptToken(mm)

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

    def cancel_token(self):
        # SCHED-3 bridge: hand the engine the (cached, stateless) token that reads ComfyUI's
        # interrupt flag. The node passes it as `tex_engine.prepare(cancel=...)`, so a Stop mid-
        # cook aborts at the next yield point (per-strip / per-statement / between tiers).
        return self._cancel_token_obj

    def raise_if_interrupted(self):
        # After a CookCancelled, re-surface the CLEAN host interrupt. The token was read-only,
        # so the flag is still set; `throw_exception_if_processing_interrupted` clears it and
        # raises comfy's InterruptProcessingException — a BaseException, so it flies past the
        # node's `except Exception` catch-all straight to ComfyUI's executor, which marks the
        # prompt interrupted rather than rendering a red node error.
        try:
            if hasattr(self._mm, "throw_exception_if_processing_interrupted"):
                self._mm.throw_exception_if_processing_interrupted()
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
