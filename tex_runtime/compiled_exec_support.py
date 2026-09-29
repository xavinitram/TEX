"""Compiled-tier per-cook execution support — SPLIT-47 (v0.47.0, TRK-210).

Split mechanically out of `compiled.py` (the STR-7/SPLIT-I pattern: every body below is
byte-identical to the code it replaced there — AGENTS.md §"Trades to REFUSE", mechanical
moves only, never an "improvement" mid-move). This module owns the support utilities the
"hot" compile/execute pipeline (`execute_compiled`, `run_auto`, `_try_compile`, all still in
`compiled.py`) calls on every cook, as opposed to the capability/routing DECISIONS
`compiled_capability.py` owns: one-time diagnostics/logging (`_show_once`,
`_maybe_triton_hint`, `_ensure_inductor_cache_dir`), the two timing wrappers
(`_timed`/`_timed_deferred`), binding preparation (`_contiguous_bindings`/`_bindings_nbytes`/
`_cuda_headroom_ok`), and the graph-capture-in-flight check (`_capture_in_flight`).

Every function here is a true leaf: none calls back into `compiled.py` at all (each of
`_ensure_inductor_cache_dir`/`_cuda_headroom_ok`/`_capture_in_flight`/`_contiguous_bindings`
already deferred its own third-party/sibling-module import inside the function body before
this split, for the same reason — cheap after the first call, and no load-time cost for a
caller that never reaches it). `compiled.py` imports this module at its own top level and
re-exports every name below, so a bare call from within `execute_compiled`/`run_auto`/
`_try_compile` resolves through `compiled.py`'s own module globals exactly as before the
move (the ROUTE-45 shape SPLIT-E used), and the register-documented stores that moved with
their functions (`compiled._deferred_ev`, excused `compiled._warnings_shown`) are re-pointed
to this module in ARCHITECTURE.md.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict as _OrderedDict
from typing import Any

import torch

logger = logging.getLogger("TEX")

# ── One-time diagnostics / logging ──────────────────────────────────────

_WARNINGS_SHOWN_CAP = 256

# One-time log messages (avoid spamming the console)
_warnings_shown: set[str] = set()


def _show_once(key: str, msg: str, level: str = "info"):
    """Log a message at most once per session."""
    if key not in _warnings_shown and len(_warnings_shown) < _WARNINGS_SHOWN_CAP:
        _warnings_shown.add(key)
        getattr(logger, level)(msg)


def _maybe_triton_hint(err_str_lower: str, device_type: str) -> None:
    """CC-1: surface the Triton-on-Windows community-wheel pin when a CUDA
    inductor compile fails for lack of Triton. Must be called from BOTH the
    torch.compile() WRAP except AND the first-CALL execution except: on the target
    config (torch 2.10, CUDA, no Triton) the wrap succeeds and TritonMissing only
    surfaces at first invocation, so a hint living only at the wrap is dead code."""
    if device_type == "cuda" and "triton" in err_str_lower:
        _tv = torch.__version__.split("+")[0]
        _pin = '"triton-windows<3.7"' if _tv.startswith("2.10") else "triton-windows"
        _show_once(
            "triton_hint",
            f"[TEX] CUDA torch.compile needs Triton. On Windows install the "
            f"community wheel matched to torch {_tv}:  pip install {_pin}  "
            f"(enable the Windows LongPathsEnabled registry key too). Falling "
            f"back to CUDA-graph / interpreter for now.",
            level="warning",
        )


_STALE_STORE_S = 24 * 3600.0


def _recently_written(path, within_s: float = _STALE_STORE_S) -> bool:
    """True when any file under `path` changed within `within_s` seconds. Stops at the first
    recent file, so a store that is in use is cheap to recognise."""
    import time
    cutoff = time.time() - within_s
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                if os.stat(os.path.join(root, name)).st_mtime >= cutoff:
                    return True
            except OSError:
                pass
    return False


def _ensure_inductor_cache_dir() -> None:
    """Point TorchInductor's on-disk cache at TEX's owned cache dir.

    `torch._inductor.config.cache_dir` does NOT exist on torch 2.10 — assigning
    it raises AttributeError, so the previous wiring silently left inductor
    writing to %TEMP% (lost to cleanup, invisible to clear_all). The supported,
    dynamically-read control is the TORCHINDUCTOR_CACHE_DIR env var; a pre-set
    value (ours from an earlier call, or a user/ComfyUI override) is respected,
    which also makes this idempotent.
    """
    if "TORCHINDUCTOR_CACHE_DIR" in os.environ:
        return
    try:
        from ..tex_cache import get_cache, codegen_epoch
        # Version the dir by the CACHE-4 codegen epoch + torch build so a codegen or torch upgrade
        # starts from a clean inductor/dynamo store (PC-2). The codegen epoch nests the AST epoch,
        # so an AST-file edit bumps it too — as it must, since emitted code changes when the AST
        # does. The parent torch_compile/ is still what clear_all() removes.
        ver = f"{codegen_epoch()}_{torch.__version__.split('+')[0].replace('.', '')}"
        parent = get_cache().torch_compile_cache_dir
        tc_dir = str(parent / ver)
        os.makedirs(tc_dir, exist_ok=True)
        # PC-1: a TEX or torch upgrade mints a new {ver} subdir; the old one
        # (30–60 MB/program of inductor artifacts) would otherwise accumulate
        # forever. Sweep sibling version dirs that don't match the current tag, except one
        # another live process (a second install on the same cache root) is still writing to.
        # Runs once per process (the env guard above makes this idempotent).
        try:
            import shutil
            for child in parent.iterdir():
                if child.is_dir() and child.name != ver and not _recently_written(child):
                    shutil.rmtree(child, ignore_errors=True)
        except Exception:
            pass
        # cl.exe fails with C1083 (and torch.compile can escalate to an uncaught
        # AssertionError during precompile attach) when the cache path approaches
        # ~185 chars. Warn on deep installs so the user can enable Windows long
        # paths (same registry fix triton-windows documents).
        if os.name == "nt" and len(tc_dir) > 130:
            _show_once("inductor_cache_longpath",
                       f"[TEX] torch.compile cache path is {len(tc_dir)} chars deep; if "
                       "compiles fail with 'fatal error C1083', enable Windows long-path "
                       "support (LongPathsEnabled registry key).",
                       level="warning")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = tc_dir
    except Exception:
        pass  # Not critical — torch falls back to its default cache location.


# ── Timing wrappers ──────────────────────────────────────────────────────

def _timed(fn, device_type: str):
    """Run fn() and return (result, elapsed_ms). CUDA uses an event pair whose
    end is synchronized (only that event, at the cook boundary — not a device
    barrier); CPU uses perf_counter. The synchronous form — used where the caller
    must have the ms THIS cook: the TRIAL cook (its ms decides commit/reject) and the
    one-shot verify window. Only the FREQUENT MEASURING path uses _timed_deferred."""
    import time as _time
    if device_type == "cuda":
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        except Exception:
            start = None
        if start is not None:
            # fn() runs EXACTLY once: after it has run, a timing-readback failure
            # falls back to the wall-clock captured at t0 rather than re-invoking the
            # (side-effecting codegen) fn — the earlier "except -> fall through -> fn()
            # again" shape double-executed it.
            t0 = _time.perf_counter()
            start.record()
            out = fn()
            end.record()
            try:
                end.synchronize()
                return out, start.elapsed_time(end)
            except Exception:
                return out, (_time.perf_counter() - t0) * 1000.0
    t0 = _time.perf_counter()
    out = fn()
    return out, (_time.perf_counter() - t0) * 1000.0


# LAT-3: deferred CUDA-event readback. A per-cook `end.synchronize()` (as _timed
# does) stalls the CPU on the GPU at EVERY measured cook — fine for a batch ComfyUI
# render, hostile to an interactive viewport that re-enters MEASURING on each code
# edit. The deferred form records this cook's event pair and reads back the PRIOR
# cook's pair only if it is ALREADY complete (a non-blocking end.query()), so the
# sync never lands on the interactive path. Median-based verdicts (autotier's deque)
# tolerate the resulting stale-by-one / occasionally-skipped samples. Invariant #6
# holds: a reading is still taken only after its event pair completes — deferral
# changes WHEN the read happens, never WHETHER it is fenced.
_deferred_ev: "_OrderedDict[Any, tuple]" = _OrderedDict()
_DEFERRED_EV_MAX = 256


def _timed_deferred(fn, device_type: str, slot):
    """Run fn(), return (result, ms_or_None). ms is the prior same-slot cook's
    elapsed time if its end event has completed (no sync), else None (skip this
    sample). CPU path stays synchronous perf_counter (no GPU sync to avoid)."""
    import time as _time
    if device_type == "cuda":
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        except Exception:
            start = None
        if start is not None:
            start.record()
            out = fn()                     # runs EXACTLY once (see _timed)
            try:
                end.record()
                # Store THIS cook's pair BEFORE reading the prior one: if the
                # readback below raises on a stale/invalidated prior pair, the fresh
                # pair has already replaced it, so the slot self-heals next cook
                # instead of re-reading the dead pair forever.
                prev = _deferred_ev.get(slot)
                _deferred_ev[slot] = (start, end)
                _deferred_ev.move_to_end(slot)
                while len(_deferred_ev) > _DEFERRED_EV_MAX:
                    _deferred_ev.popitem(last=False)
                ms = None
                if prev is not None and prev[1].query():   # prior end done → free read
                    ms = prev[0].elapsed_time(prev[1])
                return out, ms
            except Exception:
                return out, None           # fn ran; deferral failed → skip the sample
    t0 = _time.perf_counter()
    out = fn()
    return out, (_time.perf_counter() - t0) * 1000.0


# ── Binding preparation / headroom ───────────────────────────────────────

def _contiguous_bindings(bindings: dict, device: "torch.device | None" = None) -> dict:
    """Normalize tensor bindings for the codegen path.

    * Make non-contiguous tensors contiguous (inductor/codegen can fail on BHWC
      stride patterns).
    * M-5-INT: cast an anomalous INTEGER image-like tensor (dim>=3) to fp32. A
      wired int tensor binding (e.g. torch.ones(1,H,W,3,dtype=long)) builds an int
      fresh temp, and the M-5 `out=` reuse then emits torch.mul(int, fp32, out=int)
      → "result type Float can't be cast to Long", silently dropping codegen to
      the interpreter. Its TEX type is float (shape→VECn) and the output marshals
      to fp32 regardless — at zero hot-path cost (a one-time ingestion cast, vs a
      per-op runtime dtype branch on the dominant color-grade reuse pattern). The
      interpreter applies the SAME cast (interpreter.py binding loop) so the two
      tiers converge even for FLOAT/LATENT outputs and int64 values > 2^24. Scalar
      int params and int index arrays (dim<3) are left intact.
    * XPU co-location (when `device` is given): a binding on another device is
      moved to the compute device here — fused with the M-5 cast when both apply.
      Codegen assumes bindings sit on `_dev`; without this, a forced cross-device
      cook raised at the first mixed op and burned a codegen attempt before the
      interpreter fallback did the same move anyway. Same-device inputs are
      untouched (the `!=` guard), so chained same-device nodes stay zero-copy.

    Non-tensors pass through by reference.
    """
    from ..tex_marshalling import to_fp32_if_int_image
    def _norm(v):
        if not isinstance(v, torch.Tensor):
            return v
        v = to_fp32_if_int_image(v, device=device)   # M5-INT + co-location: single source
        return v if v.is_contiguous() else v.contiguous()
    return {k: _norm(v) for k, v in bindings.items()}


# C3 (v0.46 Phase C, R3#1 + B1#3): warm_call clones every binding at full resolution —
# hundreds of MB at 4K — invisible to `_cuda_headroom_ok`, which ran BEFORE the clone and
# checked only a flat 2 GB. A hard cap on top of folding the clone into the check below:
# above this size the warm is skipped (the artifact still commits via an ordinary TRIAL).
_WARM_CLONE_CAP_BYTES = 512 * 1024 * 1024  # 512 MB


def _bindings_nbytes(bindings) -> int:
    """Total byte size of every tensor binding — the projected cost of cloning ALL of
    them at full resolution (C3), which is what `run_auto`'s warm path does."""
    total = 0
    for v in bindings.values():
        if isinstance(v, torch.Tensor):
            total += v.element_size() * v.nelement()
    return total


def _cuda_headroom_ok(device, extra_bytes: int = 0) -> bool:
    """Only submit a background CUDA compile with comfortable VRAM headroom, so
    a compile never allocates while another node's inference needs the memory.
    HW-2 (audit): query the COOK's device index, not a bare "cuda" (device 0) —
    a cuda:1 cook mis-reads GPU 0's headroom otherwise. Single-GPU unaffected.

    `extra_bytes` (C3, default 0): a projected allocation about to be made (warm_call's
    binding clone) that headroom must also cover."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return True
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    need = 2 * 1024 * 1024 * 1024 + max(0, extra_bytes)  # >2 GB, plus any projected clone
    try:
        from .host import get_host_services  # PORT-1 seam
        free = get_host_services().get_free_memory(torch.device("cuda", idx))
        if free is None:
            raise RuntimeError("no host free-memory query")
        return free > need
    except Exception:
        try:
            with torch.cuda.device(idx):
                free, _total = torch.cuda.mem_get_info()
            return free > need
        except Exception:
            return True


def _capture_in_flight() -> bool:
    try:
        from .graphed import is_capturing
        return is_capturing()
    except Exception:
        return False


# A loop-iteration or call-depth limit is the program's own bug, not a compile failure. These are
# the phrases the interpreter (E6010/E6060) and the generated code raise it with.
_USER_LIMIT_PHRASES = ("maximum iteration limit", "would exceed 1024 iterations",
                       "iterations without finishing", "function call depth")


def _is_user_limit(exc: BaseException) -> bool:
    """True when `exc` is TEX's loop-iteration or call-depth limit, not a compile defect."""
    if getattr(exc, "code", None) in ("E6010", "E6060"):
        return True
    msg = str(exc).lower()
    return any(p in msg for p in _USER_LIMIT_PHRASES)


def _is_transient_failure(exc: BaseException) -> bool:
    """True for a failure that says nothing about the program: out of memory, or a dead or
    cancelled worker."""
    import concurrent.futures as _cf
    return (isinstance(exc, (MemoryError, _cf.CancelledError, _cf.BrokenExecutor))
            or type(exc).__name__ == "OutOfMemoryError"
            or "out of memory" in str(exc).lower())


def _settle_fncalls(fingerprint, device_type: str, precision: str, backend, exc=None) -> None:
    """Settle the remembered fn-calls compile attempt. A failure that is a loop limit, a
    transient one, or a missing toolchain is not a fact about the program, so it forgets the
    attempt instead of recording a persisted `False`."""
    from . import fncalls_compile
    if exc is not None:
        low = str(exc).lower()
        if (_is_user_limit(exc) or _is_transient_failure(exc)
                or "triton" in low or "cl.exe" in low or "cl is not found" in low):
            fncalls_compile.discard_attempt(fingerprint, device_type, precision)
            return
    fncalls_compile.resolve_attempt(fingerprint, device_type, precision, backend)
