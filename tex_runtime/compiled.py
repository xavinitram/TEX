"""
TEX Compiled Execution — optional torch.compile wrapper around the interpreter.

When codegen can compile a TEX program to a flat Python function (no dispatch
overhead), that function is wrapped with torch.compile for kernel fusion.
Otherwise, falls back to wrapping the tree-walking interpreter.

If compilation fails for any reason the plain interpreter is used instead —
execution never fails because of torch.compile.

Cache key: (code_fingerprint, device_type, precision).
torch.compile handles shape-based recompilation internally via guards.
"""
from __future__ import annotations

import concurrent.futures
import glob
import logging
import math
import os
import subprocess
import sys
from collections import OrderedDict as _OrderedDict
from typing import Any, Callable

import torch

from ..tex_compiler.ast_nodes import (BinOp, UnaryOp, TernaryOp, FunctionCall,
                                      VecConstructor, MatConstructor, CastExpr,
                                      IfElse, ForLoop, WhileLoop)
from .interpreter import (Interpreter, _collect_identifiers,
                          _SCALAR_BUILTIN_DEFAULTS)
from .codegen import (try_compile as _try_codegen, _invoke_cg,
                      _iter_child_nodes)
from .stdlib import TEXStdlib
from . import tier_trace  # leaf module (imports only threading) — no cycle

logger = logging.getLogger("TEX")

# Hard limit on for-loop iterations (must match interpreter.MAX_LOOP_ITERATIONS)
_MAX_LOOP_ITERATIONS = 1024


# ── MSVC environment setup (Windows) ─────────────────────────────────

_msvc_env_initialized = False


def _setup_msvc_env():
    """
    On Windows, locate vcvarsall.bat and inject the MSVC environment
    (INCLUDE, LIB, PATH) into the current process so that torch inductor
    can invoke cl.exe with proper headers and libraries.

    Called once before the first torch.compile attempt.  No-op on non-Windows
    or if the environment is already configured.
    """
    global _msvc_env_initialized
    if _msvc_env_initialized:
        return
    _msvc_env_initialized = True

    if sys.platform != "win32":
        return

    # Skip if INCLUDE is already populated (e.g. running from Developer Prompt)
    if os.environ.get("INCLUDE"):
        return

    # Search common locations for vcvarsall.bat
    search_patterns = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\**\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\**\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\**\Community\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\**\Community\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\**\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\**\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
    ]

    vcvarsall = None
    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            vcvarsall = matches[0]
            break

    if vcvarsall is None:
        return

    try:
        # Run vcvarsall and dump the resulting environment
        result = subprocess.run(
            ["cmd.exe", "/c", "call", vcvarsall, "x64", ">nul", "2>&1", "&&", "set"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return

        # Parse and inject relevant environment variables
        new_env: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                new_env[k] = v

        # Merge INCLUDE, LIB, LIBPATH, and extend PATH
        for var in ("INCLUDE", "LIB", "LIBPATH"):
            if var in new_env:
                os.environ[var] = new_env[var]

        if "PATH" in new_env:
            # Prepend MSVC paths so cl.exe is found first
            os.environ["PATH"] = new_env["PATH"]

        logger.info("[TEX] MSVC environment configured from %s", vcvarsall)
    except Exception as e:
        logger.debug("[TEX] Failed to setup MSVC environment: %s", e)


# ── Caches & state ───────────────────────────────────────────────────

# Compiled executors: (fingerprint, device_type, precision) -> (callable, backend)
# `backend` is the torch.compile backend that produced the callable, or None
# for the codegen-only adapter (no torch.compile involved). precision is in
# the key because the adapters bake it into their closures.
# Bounded to prevent memory leaks in long sessions (each entry holds a compiled
# function with captured closure state, ~100 KB+ per entry).
_compiled_cache: _OrderedDict[tuple[str, str, str], tuple[Callable, str | None]] = _OrderedDict()
# Cap compiled-function cache to limit process memory growth.
# Each torch.compile'd program adds ~30-60 MB of Inductor-compiled
# C++ kernels to process RSS. 16 entries ≈ 0.5-1 GB ceiling.
_COMPILED_CACHE_MAX = 16

# Fingerprints that crashed torch.compile — skip on subsequent calls.
# Bounded LRU (OrderedDict used as an ordered set) so a long session that hits
# many distinct failing programs does not grow this set without limit.
_compile_blacklist: "_OrderedDict[str, None]" = _OrderedDict()
_BLACKLIST_MAX = 256


def _blacklist_add(fp: str) -> None:
    """Record a fingerprint that crashed torch.compile, bounding total size."""
    _compile_blacklist[fp] = None
    _compile_blacklist.move_to_end(fp)
    while len(_compile_blacklist) > _BLACKLIST_MAX:
        _compile_blacklist.popitem(last=False)

# Track which (backend, device_type) pairs have been tested and whether they
# work. Missing key = untested, True = works, False = failed/unavailable.
# Keyed per device: inductor failing on a Triton-less CUDA setup says nothing
# about inductor on CPU (where MSVC may be perfectly usable), and vice versa.
_backend_status: dict[tuple[str, str], bool | None] = {}

# G (v0.20) — post-commit verification. `torch_compile` mode commits blind (unlike
# the measured `auto` tier), so a compiled artifact that dynamo keeps re-tracing
# (guard churn) would be served forever at ~100x the interpreter's cost. Each
# committed cache_key has its first _VERIFY_COOKS warm cooks timed; the window's
# best sample is then compared against ONE timed interpreter cook, and an artifact
# slower than _DEMOTE_RATIO x interpreter is evicted + blacklisted (tier honesty:
# never keep serving a tier that measures worse than the fallback it replaced).
# The window is px-scoped: cache_key has no resolution in it, so a size change
# mid-window resets the samples (a 512² sample must never be judged against a
# 2048² interpreter baseline). Only real torch.compile artifacts arm a window —
# the codegen-eager entry (backend None) has no dynamo to churn.
_verify_state: "_OrderedDict[tuple, dict]" = _OrderedDict()
_VERIFY_COOKS = 3
_DEMOTE_RATIO = 1.5
_VERIFY_STATE_MAX = 256


def _bindings_px(bindings) -> int:
    """Pixels per frame (H*W) of the first spatial binding, or 0."""
    for v in bindings.values():
        if isinstance(v, torch.Tensor) and v.dim() >= 3:
            return v.shape[1] * v.shape[2]
    return 0


def _canon_device(device) -> "torch.device":
    """torch.device with an index-less "cuda" resolved to the current device —
    single source with the interpreter's canonicalization, so codegen cache keys
    ("cuda" vs "cuda:0") can't split and cached env tensors can't be served for
    the wrong GPU after a current-device switch."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda" and dev.index is None:
        try:
            dev = torch.device("cuda", torch.cuda.current_device())
        except Exception:
            pass
    return dev


def _record_ingest_event(orig_bindings, dev) -> "torch.cuda.Event | None":
    """XPU (v0.20): when ingestion issued a non_blocking pinned→CUDA copy, record
    an event AT THE COPY POINT on the stream. The caller synchronizes it before
    returning the cook's output — closing the cross-node window where a
    (convention-violating) downstream in-place write to the shared pinned source
    could race the in-flight DMA. The wait covers only the copy (recorded before
    compute kernels queue), so it's ~free once the cook's Python work has run."""
    if getattr(dev, "type", None) != "cuda":
        return None
    try:
        for v in orig_bindings.values():
            if (isinstance(v, torch.Tensor) and v.device.type == "cpu"
                    and v.device != dev and v.is_pinned()):
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(dev))
                return ev
    except Exception:
        return None
    return None

# Routing-gate memo: fingerprint -> (op_count, loop_depth). Both are pure
# functions of the AST, which is keyed by the same (code, binding-types)
# fingerprint, so the two full AST walks run once per program instead of on
# every uncached frame. has_spatial is deliberately NOT memoized: it depends
# on binding VALUES (a [B,H,W] mask and a Python float scalar both fingerprint
# as FLOAT but need different routes).
_route_memo: "_OrderedDict[str, tuple[int, int]]" = _OrderedDict()
_ROUTE_MEMO_MAX = 256

# One-time log messages (avoid spamming the console)
_warnings_shown: set[str] = set()


def _precompile_ctx():
    """Context manager enabling dynamo's persistent precompile cache (PC-2),
    scoped so the process-global flag is only held around dynamo entry on the
    compile worker thread. No-op when unsupported."""
    try:
        import torch._dynamo.config as _dc
        if hasattr(_dc, "caching_precompile"):
            return _dc.patch(caching_precompile=True)
    except Exception:
        pass
    import contextlib
    return contextlib.nullcontext()


# Error signatures that mean a persisted precompile entry failed to ATTACH
# (stale/corrupt/shape- or version-mismatched) rather than the program being
# genuinely uncompilable. These must NOT blacklist the fingerprint — instead the
# stale dynamo store is cleared so the next run recompiles fresh (PC-2).
def _is_precompile_attach_failure(e: Exception) -> bool:
    name = type(e).__name__
    msg = str(e)
    if name == "AssertionError":
        return True  # guard miss after attach (incl. first-call shape != saved)
    if name == "NameError" and "__compiled_fn_" in msg:
        return True  # same-position source-body edit / stale bytecode
    if "Compile package was created with a different" in msg:
        return True  # torch/CUDA/GPU/triton version mismatch (SystemInfo check)
    return False


def _clear_dynamo_precompile_store() -> None:
    """Delete the persisted dynamo precompile subdir so a poisoned/stale entry
    can't crash every later session (PC-2 recovery)."""
    try:
        import shutil
        from pathlib import Path
        root = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if not root:
            return
        # PC-2 safety: only clear a TEX-OWNED store. A user (or another custom
        # node) may point TORCHINDUCTOR_CACHE_DIR at a shared dir — deleting its
        # dynamo/ would wipe every tool's precompile entries on one TEX failure.
        try:
            from ..tex_cache import get_cache
            owned = get_cache().torch_compile_cache_dir.resolve()
            rp = Path(root).resolve()
            if rp != owned and owned not in rp.parents:
                return  # foreign dir — leave it untouched
        except Exception:
            return  # can't prove ownership → don't delete
        d = os.path.join(root, "dynamo")
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


# UC-2: fingerprint -> whether the program should default-route to the codegen
# tier for stencil lowering (exact/fetch stencils only). Memoized: detection is
# a full AST walk, run once per program.
_stencil_route_memo: "_OrderedDict[str, bool]" = _OrderedDict()


def should_stencil_route(fingerprint: str, program: Any) -> bool:
    """UC-2 gate (memoized per fingerprint): route the default cook through the
    codegen tier when the program has an exact stencil the lowering accelerates."""
    v = _stencil_route_memo.get(fingerprint)
    if v is None:
        try:
            from .codegen import detect_stencil_route
            v = bool(detect_stencil_route(program))
        except Exception:
            v = False
        _stencil_route_memo[fingerprint] = v
        while len(_stencil_route_memo) > _ROUTE_MEMO_MAX:
            _stencil_route_memo.popitem(last=False)
    else:
        _stencil_route_memo.move_to_end(fingerprint)
    return v


def _get_or_make_codegen_fn(program: Any, type_map: dict | None,
                            fingerprint: str | None):
    """Return the codegen flat fn for a program, or None if unsupported.

    Unified accessor for both codegen consumers (_codegen_only_execute and
    _try_compile). The fingerprinted result — the materialized fn or an
    "unsupported" sentinel — is memoized and marshal-persisted by TEXCache
    (PC-3), so re-executions and process restarts skip emit+compile()+exec().
    cg_fn captures no device/precision/bindings (all passed per call), so the
    fingerprint alone is a sufficient key.
    """
    from ..tex_cache import get_cache, _CG_UNSUPPORTED
    cache = get_cache() if fingerprint is not None else None
    cg_fn = cache.get_codegen_fn(fingerprint) if cache is not None else None
    if cg_fn is None:  # not yet generated (or no fingerprint) — emit now
        try:
            cg_fn = _try_codegen(program, type_map, fingerprint)
        except Exception:
            cg_fn = None
        if cache is not None:
            cache.store_codegen_fn(fingerprint, cg_fn)
    return None if cg_fn is _CG_UNSUPPORTED else cg_fn

# Persistent single-thread worker for the torch.compile lifecycle.
# A long-lived worker provides the same dynamo-TLS isolation as a fresh pool
# per call (the entire compile+run stays off the main thread) without paying
# OS thread create/destroy on every frame of a batch/video workload.
_COMPILE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Minimum tensor-op count for torch.compile to be worthwhile.
# Below this threshold the fusion benefit cannot overcome tracing overhead.
_COMPILE_OP_THRESHOLD = 8

# Maximum nested loop depth for torch.compile to be beneficial.
# Deep nesting causes graph breaks and recompilation overhead that
# exceeds any fusion benefit. Programs above this use codegen-only.
_COMPILE_MAX_LOOP_DEPTH = 2


# ── Helpers ───────────────────────────────────────────────────────────

_WARNINGS_SHOWN_CAP = 256

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
        from ..tex_cache import get_cache, _CACHE_VERSION
        # Version the dir by cache-version + torch build so a TEX or torch
        # upgrade starts from a clean inductor/dynamo store (PC-2). The parent
        # torch_compile/ is still what clear_all() removes.
        ver = f"{_CACHE_VERSION}_{torch.__version__.split('+')[0].replace('.', '')}"
        parent = get_cache().torch_compile_cache_dir
        tc_dir = str(parent / ver)
        os.makedirs(tc_dir, exist_ok=True)
        # PC-1: a TEX or torch upgrade mints a new {ver} subdir; the old one
        # (30–60 MB/program of inductor artifacts) would otherwise accumulate
        # forever. Sweep sibling version dirs that don't match the current tag.
        # Runs once per process (the env guard above makes this idempotent).
        try:
            import shutil
            for child in parent.iterdir():
                if child.is_dir() and child.name != ver:
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


_OP_TYPES = (BinOp, UnaryOp, TernaryOp, FunctionCall,
             VecConstructor, MatConstructor, CastExpr)


def _count_tensor_ops(program: Any) -> int:
    """Count tensor operations in an AST to estimate torch.compile benefit.

    Counts BinOps, FunctionCalls, VecConstructors, TernaryOps, UnaryOps,
    MatConstructors, and CastExprs — the operations that produce tensor work.
    Traverses via the shared generic child iterator, so user-function bodies
    and array/matrix constructs are all covered.
    """
    count = 0
    stack = list(program.statements)
    while stack:
        node = stack.pop()
        if isinstance(node, _OP_TYPES):
            count += 1
        stack.extend(_iter_child_nodes(node))
    return count


def _max_loop_depth(program: Any) -> int:
    """Return the maximum nesting depth of for/while loops in the AST."""
    def _depth(stmts: list, current: int) -> int:
        mx = current
        for s in stmts:
            if isinstance(s, (ForLoop, WhileLoop)):
                mx = max(mx, _depth(s.body, current + 1))
            elif isinstance(s, IfElse):
                mx = max(mx, _depth(s.then_body, current))
                if s.else_body:
                    mx = max(mx, _depth(s.else_body, current))
        return mx
    return _depth(program.statements, 0)



def _select_backend(device_type: str) -> str | None:
    """
    Pick the best available torch.compile backend.

    Priority:
      GPU → inductor > cudagraphs > None
      CPU → inductor > None

    Backends already marked as failed on this device type are skipped.
    """
    candidates = []
    if device_type == "cuda":
        candidates = ["inductor", "cudagraphs"]
    else:
        candidates = ["inductor"]

    for backend in candidates:
        if _backend_status.get((backend, device_type)) is not False:
            return backend
    return None


# ── Public API ────────────────────────────────────────────────────────

def execute_compiled(
    program: Any,
    bindings: dict[str, Any],
    type_map: dict,
    device: str | torch.device,
    fingerprint: str,
    latent_channel_count: int = 0,
    output_names: list[str] | None = None,
    used_builtins: set[str] | None = None,
    precision: str = "fp32",
    time_context: dict | None = None,
) -> torch.Tensor | dict:
    """
    Execute a TEX program with optional torch.compile acceleration.

    Falls back to the plain interpreter on any failure.

    The ENTIRE torch.compile lifecycle (wrapping via torch.compile() AND
    execution of the compiled callable) runs in a disposable thread.
    This is critical because torch.compile/dynamo can corrupt the C++
    PythonDispatcherTLS when tracing fails — and that corruption is
    thread-local and unrecoverable.  By keeping the main thread clean,
    ComfyUI and subsequent TEX runs stay healthy even after failures.

    Args:
        program:              Parsed AST (Program node).
        bindings:             Mapping of @ binding names to tensor / scalar values.
        type_map:             AST node id -> TEXType (from type checker).
        device:               Target device ("cpu", "cuda", "cuda:0", …).
        fingerprint:          Cache key produced by TEXCache.fingerprint().
        latent_channel_count: Channel count when the input is a LATENT (0 for IMAGE/MASK).
        output_names:         List of named outputs for multi-output programs.
        used_builtins:        Set of coordinate builtins the program uses (env pruning).
        precision:            "fp32" (default) or "fp16" — see the M-3 fp16 contract.

    Returns:
        The @OUT tensor result, or a dict of {name: tensor} for multi-output.
    """
    device_obj = _canon_device(device)
    device_type = device_obj.type  # "cpu" or "cuda"
    cache_key = (fingerprint, device_type, precision)

    # ── Blacklist: skip programs that previously crashed torch.compile
    if fingerprint in _compile_blacklist:
        return _plain_execute(program, bindings, type_map, device,
                              latent_channel_count, output_names,
                              used_builtins=used_builtins, precision=precision,
                              time_context=time_context)

    # ── Program analysis gates (only on first compile, not cached reruns).
    # op_count/loop_depth are memoized per fingerprint so routes that never
    # enter _compiled_cache (codegen-only, below-threshold) don't re-walk the
    # AST every frame.
    if cache_key not in _compiled_cache:
        route = _route_memo.get(fingerprint)
        if route is None:
            route = (_count_tensor_ops(program), _max_loop_depth(program))
            _route_memo[fingerprint] = route
            while len(_route_memo) > _ROUTE_MEMO_MAX:
                _route_memo.popitem(last=False)
        else:
            _route_memo.move_to_end(fingerprint)
        op_count, loop_depth = route
        # Skip torch.compile for trivial programs (tracing overhead > benefit)
        if op_count < _COMPILE_OP_THRESHOLD:
            return _plain_execute(program, bindings, type_map, device,
                                  latent_channel_count, output_names,
                                  used_builtins=used_builtins, precision=precision,
                                  time_context=time_context)
        # Use codegen WITHOUT torch.compile for deeply nested loops
        # (graph breaks and recompilation make torch.compile slower)
        if loop_depth > _COMPILE_MAX_LOOP_DEPTH:
            return _codegen_only_execute(program, bindings, type_map, device,
                                         latent_channel_count, output_names,
                                         used_builtins=used_builtins, precision=precision,
                                         fingerprint=fingerprint,
                                         time_context=time_context)   # ENG-7
        # Use plain interpreter for programs without spatial tensor context
        # (procedural noise, etc.) — codegen env setup overhead exceeds
        # benefit when all operations are on scalar tensors
        has_spatial = any(isinstance(v, torch.Tensor) and v.dim() >= 3
                         for v in bindings.values())
        if not has_spatial:
            return _plain_execute(program, bindings, type_map, device,
                                  latent_channel_count, output_names,
                                  used_builtins=used_builtins, precision=precision,
                                  time_context=time_context)

    # Ensure tensor bindings are contiguous — Inductor's codegen can
    # fail on non-contiguous strides (e.g. BHWC images loaded with
    # stride patterns like [1090020, 2220, 3, 1]) — and co-located on
    # the compute device (XPU: one direct hop instead of raise+retry).
    contiguous_bindings = _contiguous_bindings(bindings, device_obj)
    ingest_event = _record_ingest_event(bindings, device_obj)
    cook_px = _bindings_px(bindings)

    # Run the ENTIRE torch.compile lifecycle in an ISOLATED THREAD.
    #
    # Both torch.compile() wrapping and compiled function execution
    # touch dynamo's C++ TLS state.  If either phase fails mid-trace,
    # the TLS on that thread is permanently corrupted — all tensor ops
    # fail with INTERNAL ASSERT FAILED.
    #
    # By running everything in a disposable thread, any TLS corruption
    # stays contained.  The main thread never touches dynamo at all.
    compile_error = None

    def _compile_and_run():
        """Compile (if needed) and execute — all on this worker thread."""
        nonlocal compile_error
        try:
            # caching_precompile persists dynamo entries under
            # TORCHINDUCTOR_CACHE_DIR so a warm restart skips most of the compile
            # (PC-2). Scoped to this worker so the process-global flag is never
            # held across another node's cook.
            with _precompile_ctx(), torch.inference_mode():
                # Get or create the compiled callable (on THIS thread)
                if cache_key not in _compiled_cache:
                    entry = _try_compile(device_type, program, type_map,
                                         used_builtins=used_builtins, precision=precision,
                                         fingerprint=fingerprint)
                    if entry is None:
                        # No backend available — run plain interpreter here
                        return _plain_execute(program, contiguous_bindings, type_map,
                                              device, latent_channel_count, output_names,
                                              used_builtins=used_builtins, precision=precision,
                                              time_context=time_context)
                    _compiled_cache[cache_key] = entry
                    # G: arm post-commit verification for this fresh artifact
                    # (samples collect on the NEXT cooks — the commit cook itself
                    # includes compile latency and would poison the verdict).
                    # Only real torch.compile artifacts arm: the codegen-eager
                    # entry (backend None) has no dynamo to guard-churn, and its
                    # baseline cook would be pure waste.
                    if entry[1] is not None:
                        _verify_state[cache_key] = {"px": cook_px, "samples": []}
                        while len(_verify_state) > _VERIFY_STATE_MAX:
                            _verify_state.popitem(last=False)
                    if len(_compiled_cache) > _COMPILED_CACHE_MAX:
                        # Evict the oldest. Do NOT torch._dynamo.reset() here:
                        # this runs on the disposable worker thread, and dynamo
                        # state is PROCESS-GLOBAL — resetting it would corrupt the
                        # very thread that invokes compiled_fn just below (and the
                        # calling thread). The bounded cache already caps growth.
                        _compiled_cache.popitem(last=False)
                    compiled_fn, _entry_backend = _compiled_cache[cache_key]
                    return compiled_fn(program, contiguous_bindings, type_map, device,
                                       latent_channel_count, output_names)

                compiled_fn, _entry_backend = _compiled_cache[cache_key]
                _compiled_cache.move_to_end(cache_key)
                verify = _verify_state.get(cache_key)
                if verify is not None and len(verify["samples"]) < _VERIFY_COOKS:
                    # G: verification window — time this warm compiled cook.
                    # A resolution change mid-window resets it (samples at one
                    # px must never be judged against a baseline at another).
                    if verify["px"] != cook_px:
                        verify["px"] = cook_px
                        verify["samples"] = []
                    # LAT-3: the verify window stays SYNCHRONOUS. It is one-shot and
                    # bounded (3 warm cooks post-commit), so its sync doesn't hurt
                    # interactivity — and deferring it would let a resolution-flip-heavy
                    # session never accumulate 3 samples at a stable px, so a genuinely
                    # slow artifact would escape demotion. Only the FREQUENT MEASURING
                    # path is deferred.
                    out, ms = _timed(
                        lambda: compiled_fn(program, contiguous_bindings, type_map, device,
                                            latent_channel_count, output_names),
                        device_type)
                    verify["samples"].append(ms)
                    return out
                return compiled_fn(program, contiguous_bindings, type_map, device,
                                   latent_channel_count, output_names)
        except Exception as e:
            compile_error = e
            return None

    # Submit to the persistent single-thread worker. Same TLS isolation as a
    # fresh pool (compile+run never touches the main thread) without per-call
    # thread create/destroy.
    future = _COMPILE_POOL.submit(_compile_and_run)
    result = future.result()

    if compile_error is not None:
        # A loop-iteration / recursion-depth limit is a USER bug, not a compile
        # failure: don't blacklist the program from compiling forever (it would
        # stay blacklisted even after the user fixes the loop). The interpreter
        # fallback below raises the clean E6010/E6060 diagnostic.
        _emsg = str(compile_error)
        is_user_limit = ("iteration" in _emsg or "iterations" in _emsg
                         or "call depth" in _emsg)
        _show_once(
            f"compile_exec_fail_{fingerprint[:12]}",
            f"[TEX] torch.compile execution failed, falling back to interpreter: {compile_error}",
            level="warning",
        )
        popped = _compiled_cache.pop(cache_key, None)
        _verify_state.pop(cache_key, None)   # G: window dies with the artifact
        failed_backend = popped[1] if popped is not None else None
        # torch.compile() wraps lazily, so a missing backend toolchain (Triton
        # on CUDA, cl.exe on Windows CPU) only surfaces HERE, at the first
        # execution of the compiled callable. That is a property of the
        # (backend, device) pair, not of the program: mark the backend
        # unavailable so the next run cascades (inductor -> cudagraphs on
        # CUDA) instead of blacklisting every new fingerprint.
        _lower = _emsg.lower()
        _errname = type(compile_error).__name__
        # CC-1: a missing backend toolchain surfaces at FIRST CALL. It may be a
        # BackendCompilerFailed OR a bare TritonMissing / RuntimeError naming
        # triton/cl.exe — either way it is a (backend, device) property, so mark
        # the backend down (→ cascade to cudagraphs/interpreter) rather than
        # blacklisting this one fingerprint.
        backend_unavailable = (
            failed_backend in ("inductor", "cudagraphs")
            and (_errname == "BackendCompilerFailed" or "triton" in _errname.lower())
            and ("triton" in _lower or "cl.exe" in _lower
                 or "cl is not found" in _lower)
        )
        # Surface the actionable Triton pin at the point the failure actually
        # occurs (the wrap-time hint in _try_compile is dead on this config).
        _maybe_triton_hint(_lower, device_type)
        precompile_attach_failed = _is_precompile_attach_failure(compile_error)
        if precompile_attach_failed:
            # A persisted precompile entry failed to attach (stale/corrupt/shape-
            # or version-mismatch). Clear the dynamo store and recompile fresh
            # next run — do NOT blacklist (the program is fine); PC-2 recovery.
            _clear_dynamo_precompile_store()
            _show_once("precompile_recover",
                       "[TEX] Cleared a stale torch.compile precompile entry; "
                       "will recompile on the next run.", level="warning")
        if backend_unavailable:
            _backend_status[(failed_backend, device_type)] = False
        elif not is_user_limit and not precompile_attach_failed:
            # BackendCompilerFailed can also be program-specific — those (and
            # all other runtime failures) blacklist the fingerprint only.
            _blacklist_add(fingerprint)
        # IMPORTANT: torch.compile / dynamo state is PROCESS-GLOBAL, not
        # thread-local.  Resetting it on a *disposable worker thread* corrupts
        # the calling thread's dynamo / code-cache state and garbles the
        # interpreter fallback that runs immediately after — surfacing as bogus
        # "Variable not defined" / "dictionary changed size during iteration"
        # errors (or a segfault) on perfectly valid TEX code.  Reset on THIS
        # (the calling) thread instead.
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        # Fall back on the PRISTINE bindings, not contiguous_bindings: the
        # compiled fn may have partially mutated the latter (added 'OUT', in-place
        # scatter) before raising mid-execution, which would corrupt the
        # interpreter recompute. Matches _codegen_only_execute's fallback.
        return _plain_execute(program, bindings, type_map, device,
                              latent_channel_count, output_names,
                              used_builtins=used_builtins, precision=precision,
                              time_context=time_context)

    # G (v0.20): post-commit verification verdict. Once the window is full,
    # time ONE interpreter cook of the same program and demote the compiled
    # artifact if its warm cooks are clearly slower (dynamo guard churn:
    # measured ~100x on interpreter-class guard failures). One-shot per
    # cache_key; the baseline cook uses the pristine bindings, and only fires
    # when the window's samples were taken at THIS cook's resolution.
    verify = _verify_state.get(cache_key)
    if (verify is not None and len(verify["samples"]) >= _VERIFY_COOKS
            and verify["px"] == cook_px):
        _verify_state.pop(cache_key, None)
        try:
            _, interp_ms = _timed(
                lambda: _plain_execute(program, dict(bindings), type_map, device,
                                       latent_channel_count, output_names,
                                       used_builtins=used_builtins, precision=precision,
                                       time_context=time_context),
                device_type)
            # min(), not median: the first post-commit cooks can include cudagraph
            # recording (reduce-overhead), which would overstate a good artifact.
            # A bad artifact (guard churn) is slow on EVERY cook, so min still
            # catches it decisively.
            compiled_ms = min(verify["samples"])
            if compiled_ms > interp_ms * _DEMOTE_RATIO:
                _compiled_cache.pop(cache_key, None)
                _blacklist_add(fingerprint)
                tier_trace.record("interpreter", fallback_from="torch_compile",
                                  reason=f"demoted: compiled {compiled_ms:.1f}ms > "
                                         f"{_DEMOTE_RATIO}x interp {interp_ms:.1f}ms")
                _show_once(
                    f"compile_demoted_{fingerprint[:12]}",
                    f"[TEX] torch.compile artifact measured {compiled_ms:.1f}ms vs "
                    f"interpreter {interp_ms:.1f}ms — demoted to interpreter for this "
                    f"program (honest tiering).",
                    level="warning",
                )
        except Exception:
            pass  # verification is best-effort; never fail a successful cook

    # XPU fence: the async pinned→CUDA ingest DMA must land before the cook
    # returns (see _record_ingest_event) — ~always already signaled by now.
    if ingest_event is not None:
        try:
            ingest_event.synchronize()
        except Exception:
            pass

    return result


# ── CC-2: measured auto-tier orchestration ────────────────────────────────
# Background compiles in flight, keyed by cache_key. The compile runs on the
# same TLS-isolated worker; the cook never blocks on future.result().
_bg_futures: dict = {}


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


def _cuda_headroom_ok(device) -> bool:
    """Only submit a background CUDA compile with comfortable VRAM headroom, so
    a compile never allocates while another node's inference needs the memory.
    HW-2 (audit): query the COOK's device index, not a bare "cuda" (device 0) —
    a cuda:1 cook mis-reads GPU 0's headroom otherwise. Single-GPU unaffected."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return True
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    try:
        from .host import get_host_services  # PORT-1 seam
        free = get_host_services().get_free_memory(torch.device("cuda", idx))
        if free is None:
            raise RuntimeError("no host free-memory query")
        return free > 2 * 1024 * 1024 * 1024  # >2 GB
    except Exception:
        try:
            with torch.cuda.device(idx):
                free, _total = torch.cuda.mem_get_info()
            return free > 2 * 1024 * 1024 * 1024
        except Exception:
            return True


def _capture_in_flight() -> bool:
    try:
        from .graphed import is_capturing
        return is_capturing()
    except Exception:
        return False


def _submit_bg_compile(cache_key, program, type_map, device_type,
                       used_builtins, precision, fingerprint) -> bool:
    """Submit a compile-only job (populates _compiled_cache) without blocking.
    Returns True if a job is now in flight (or already was)."""
    if cache_key in _compiled_cache or cache_key in _bg_futures:
        return True
    if fingerprint in _compile_blacklist:
        return False

    def _compile_only():
        try:
            with _precompile_ctx(), torch.inference_mode():
                if cache_key not in _compiled_cache:
                    entry = _try_compile(device_type, program, type_map,
                                         used_builtins=used_builtins,
                                         precision=precision, fingerprint=fingerprint)
                    if entry is None:
                        return "no_backend"
                    _compiled_cache[cache_key] = entry
                    if len(_compiled_cache) > _COMPILED_CACHE_MAX:
                        _compiled_cache.popitem(last=False)
            return "ok"
        except Exception:
            return "failed"

    try:
        _bg_futures[cache_key] = _COMPILE_POOL.submit(_compile_only)
        return True
    except Exception:
        return False


def _bg_status(cache_key) -> str:
    """'pending' | 'ready' | 'failed' | 'absent'."""
    fut = _bg_futures.get(cache_key)
    if fut is None:
        return "ready" if cache_key in _compiled_cache else "absent"
    if not fut.done():
        return "pending"
    _bg_futures.pop(cache_key, None)
    try:
        res = fut.result()
    except Exception:
        res = "failed"
    if res == "ok" and cache_key in _compiled_cache:
        return "ready"
    return "failed"


def _run_cached_compiled(cache_key, program, bindings, type_map, device,
                         latent_channel_count, output_names, device_type,
                         timed):
    """Run the cached compiled fn on the worker thread (dynamo-TLS isolation is
    load-bearing even for an already-built artifact — a guard failure can retrace
    and corrupt the calling thread's TLS). Returns (result, ms) when *timed*, or
    (result, None) otherwise — the COMMITTED tier skips timing so it never forces
    a per-cook CUDA sync. Returns (None, None) if the run crashed."""
    contiguous = _contiguous_bindings(bindings, _canon_device(device))

    def _worker():
        with torch.inference_mode():
            compiled_fn, _b = _compiled_cache[cache_key]
            call = lambda: compiled_fn(program, contiguous, type_map, device,
                                       latent_channel_count, output_names)
            return _timed(call, device_type) if timed else (call(), None)

    try:
        return _COMPILE_POOL.submit(_worker).result()
    except Exception:
        # Crash — demote (do not blacklist forever). Reset dynamo on the calling
        # thread; the caller routes to codegen-only.
        _compiled_cache.pop(cache_key, None)
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        return None, None


def run_auto(program, bindings, type_map, device, fingerprint,
             latent_channel_count: int = 0, output_names=None,
             used_builtins=None, precision: str = "fp32",
             time_context: dict | None = None):
    """CC-2 entry: measure the always-safe codegen baseline, background-compile,
    trial the compiled fn, and commit only on a measured win. Never blocks on
    the compile; never routes to a slower tier than codegen-only."""
    from . import autotier
    device_obj = torch.device(device)
    device_type = device_obj.type
    cache_key = (fingerprint, device_type, precision)

    sp = None
    for v in bindings.values():
        if isinstance(v, torch.Tensor) and v.dim() >= 3:
            sp = (v.shape[0], v.shape[1], v.shape[2])
            break
    key = autotier.make_key(fingerprint, device_type, precision, sp)
    autotier.seed_from_disk(key)
    state = autotier.verdict(key)

    def _codegen(bind):
        return _codegen_only_execute(program, bind, type_map, device,
                                     latent_channel_count, output_names,
                                     used_builtins=used_builtins,
                                     precision=precision, fingerprint=fingerprint,
                                     time_context=time_context)   # ENG-7

    # Terminal: rejected → always-safe codegen; committed → cached compiled.
    if state == autotier.REJECTED:
        return _codegen(bindings)
    if state == autotier.COMMITTED and cache_key in _compiled_cache:
        # Terminal — skip timing (no per-cook CUDA sync; the verdict is frozen).
        res, _ = _run_cached_compiled(cache_key, program, bindings, type_map,
                                      device, latent_channel_count,
                                      output_names, device_type, timed=False)
        if res is None:
            autotier.record_trial(key, None)  # demote to rejected
            return _codegen(bindings)
        return res
    # Committed but artifact gone (restart w/o PC-2 persistence, or evicted):
    # re-establish by trialling again this cook.
    if state == autotier.COMMITTED:
        autotier.mark_ready(key)
        state = autotier.TRIAL

    if state == autotier.TRIAL and cache_key in _compiled_cache:
        res, ms = _run_cached_compiled(cache_key, program, bindings, type_map,
                                       device, latent_channel_count,
                                       output_names, device_type, timed=True)
        if res is None:
            autotier.record_trial(key, None)
            return _codegen(bindings)
        autotier.record_trial(key, ms)
        return res

    # MEASURING or COMPILING (or TRIAL with a lost artifact): run the codegen
    # baseline, timed via the DEFERRED reader (LAT-3) so the measure window never
    # syncs on the interactive path; a None (prior sample not yet complete) just
    # skips this cook's sample — the deque converges over the next few cooks.
    # Slot is `key` itself — the SAME px-BUCKET granularity autotier commits its
    # verdict at (make_key buckets by px.bit_length). Keying on the EXACT shape `sp`
    # instead would mean a session at jittering resolutions (a folder of
    # near-1000px photos, a zoom-drag within one octave) never sees a prior same-slot
    # cook, so `interp_ms` never fills and the program never promotes to the
    # compiled tier — the deferred reader must group at the bucket autotier medians
    # over, not below it.
    res, ms = _timed_deferred(lambda: _codegen(bindings), device_type,
                              (key, "measure"))
    if ms is not None:
        autotier.record_interp(key, ms)
    if state == autotier.MEASURING and autotier.should_submit_compile(key):
        if _cuda_headroom_ok(device) and not _capture_in_flight():
            if _submit_bg_compile(cache_key, program, type_map, device_type,
                                  used_builtins, precision, fingerprint):
                autotier.mark_submitted(key)
    elif state == autotier.COMPILING:
        st = _bg_status(cache_key)
        if st == "ready":
            autotier.mark_ready(key)
        elif st in ("failed", "absent"):
            autotier.record_trial(key, None)  # compile failed → reject, stay on codegen
    return res


def _plain_execute(
    program: Any,
    bindings: dict[str, Any],
    type_map: dict,
    device: str | torch.device,
    latent_channel_count: int = 0,
    output_names: list[str] | None = None,
    used_builtins: set[str] | None = None,
    precision: str = "fp32",
    *,
    time_context: dict | None,
) -> torch.Tensor | dict:
    """Execute without torch.compile (standard tree-walking interpreter). Reuses ONE
    persistent interpreter (C6) so its coordinate-builtin LRU (LAT-4) actually persists
    across cooks on the interpreter-FALLBACK path — the no-backend / codegen-declined
    path is never cached, so a fresh `Interpreter()` per call rebuilt u/v/ix/iy every
    cook, exactly what LAT-4 set out to avoid. This instance is SEPARATE from the graph
    tier's dedicated interpreter (graphed.py), so a module-shared builtins cache never
    lets another cook evict a captured graph's baked builtins. Under ComfyUI's
    one-cook-at-a-time executor this reuse is race-free (same contract as the engine's
    _get_interpreter singleton — tex_engine, ENG-1).

    ENG-7: `time_context` MUST be forwarded. This is where a time-reading program
    actually lands — codegen declines it (see codegen.try_compile), and that decline
    happens INSIDE execute_compiled / run_auto / _codegen_only_execute, which return
    through here rather than raising, so the engine's own `_interp_fallback` (which does
    forward it) is never reached. Missing it froze the playhead at 0 on the SHIPPED
    DEFAULT path — `compile_mode="none"` routes an exact stencil through
    _codegen_only_execute. The closure that made threading this dangerous elsewhere
    (`_codegen_exec_eager`, built once per fingerprint) is unreachable for these
    programs, precisely because codegen declined them.

    Which is why it is KEYWORD-ONLY and REQUIRED, with no default. `None` is a legitimate
    value (no host playhead) and is indistinguishable, at the interpreter, from a caller
    that simply forgot — `tc.get("frame", 0.0)` reads 0.0 for both, cooks happily, and
    returns a still image. A default therefore cannot be safe here: it converts an
    omission into a frozen playhead with no error, no diagnostic, and a plausible picture.
    Requiring it moves the failure to the call, where it is a TypeError naming the
    parameter. v0.22 shipped a route (execute_compiled's deep-loop branch) that omitted
    it, past a suite that already had SEVEN routes pinned; pass `time_context=None`
    explicitly to mean "no playhead"."""
    return _get_plain_interp().execute(
        program, bindings, type_map, device=device,
        latent_channel_count=latent_channel_count, output_names=output_names,
        precision=precision, used_builtins=used_builtins, time_context=time_context)


_PLAIN_INTERP = None


def _get_plain_interp():
    global _PLAIN_INTERP
    if _PLAIN_INTERP is None:
        _PLAIN_INTERP = Interpreter()
    return _PLAIN_INTERP


def clear_plain_interp_caches():
    """Sweep the persistent fallback interpreter's tensor LRUs (C6) — for
    free_tensor_caches (memory pressure) + clear_compiled_cache (test isolation).
    No-op if it was never created."""
    if _PLAIN_INTERP is not None:
        _PLAIN_INTERP._builtins_lru.clear()
        _PLAIN_INTERP._literal_cache.clear()


# Cross-cook cache for codegen builtin tensors (coordinates + constants). The
# interpreter has cached these across executes for ages (its _builtins_lru);
# the codegen path was re-allocating every single cook — a fresh arange/expand/
# scalar-tensor per builtin per frame (a handful of tiny kernel launches on CUDA,
# allocs on CPU). Values are NEVER mutated downstream: generated code treats
# builtins as read-only operands (the language forbids assigning them, and the
# in-place/`out=` reuse analysis only targets fresh-materialized locals), and the
# env DICT is built fresh per cook so loop-scoped rebinds can't leak across cooks.
# Registered in graphed._build_keepalive so a captured graph's baked addresses
# always outlive eviction (MEM-1).
_ENV_TENSOR_CACHE: "_OrderedDict[tuple, torch.Tensor]" = _OrderedDict()
_ENV_TENSOR_CACHE_MAX = 256


def _env_cached(key: tuple, make) -> torch.Tensor:
    t = _ENV_TENSOR_CACHE.get(key)
    if t is None:
        t = make()
        _ENV_TENSOR_CACHE[key] = t
        while len(_ENV_TENSOR_CACHE) > _ENV_TENSOR_CACHE_MAX:
            _ENV_TENSOR_CACHE.popitem(last=False)
    else:
        _ENV_TENSOR_CACHE.move_to_end(key)
    return t


def _build_codegen_env(
    program: Any, bindings: dict[str, Any],
    device: torch.device, latent_channel_count: int,
    used_builtins: set[str] | None = None,
    precision: str = "fp32",
) -> tuple[dict, tuple | None, set[str]]:
    """Build the environment dict and spatial shape for codegen execution.

    Returns (env, spatial_shape, used_identifiers).
    Shared by _codegen_exec and _codegen_only_execute.

    used_builtins (when provided) is the precomputed identifier set; passing it
    avoids a full per-frame AST walk via _collect_identifiers.

    Builtin tensors come from the cross-cook _ENV_TENSOR_CACHE (keyed on shape/
    device/dtype); only the env dict itself is per-cook.
    """
    env: dict[str, Any] = {}
    sp = None
    for v in bindings.values():
        if isinstance(v, torch.Tensor) and v.dim() >= 3:
            sp = (v.shape[0], v.shape[1], v.shape[2])
            break

    used = used_builtins if used_builtins is not None else _collect_identifiers(program)
    # Single owner: the interpreter's precision->dtype map (both backends agree).
    dtype = Interpreter._PRECISION_DTYPES.get(precision, torch.float32)
    dev_key = str(device)

    if sp:
        B, H, W = sp
        if "ix" in used or "u" in used:
            ix = _env_cached(("ix", W, dev_key, dtype),
                             lambda: torch.arange(W, dtype=dtype, device=device).view(1, 1, W))
            if "ix" in used:
                env["ix"] = ix
            if "u" in used:
                env["u"] = _env_cached(("u", B, H, W, dev_key, dtype),
                                       lambda: (ix / max(W - 1, 1)).expand(B, H, W))
        if "iy" in used or "v" in used:
            iy = _env_cached(("iy", H, dev_key, dtype),
                             lambda: torch.arange(H, dtype=dtype, device=device).view(1, H, 1))
            if "iy" in used:
                env["iy"] = iy
            if "v" in used:
                env["v"] = _env_cached(("v", B, H, W, dev_key, dtype),
                                       lambda: (iy / max(H - 1, 1)).expand(B, H, W))
        if "iw" in used:
            env["iw"] = _env_cached(("iw", W, dev_key, dtype),
                                    lambda: torch.tensor(float(W), dtype=dtype, device=device))
        if "ih" in used:
            env["ih"] = _env_cached(("ih", H, dev_key, dtype),
                                    lambda: torch.tensor(float(H), dtype=dtype, device=device))
        if "px" in used:
            env["px"] = _env_cached(("px", W, dev_key, dtype),
                                    lambda: torch.tensor(1.0 / max(W, 1), dtype=dtype, device=device))
        if "py" in used:
            env["py"] = _env_cached(("py", H, dev_key, dtype),
                                    lambda: torch.tensor(1.0 / max(H, 1), dtype=dtype, device=device))
        if "fi" in used:
            env["fi"] = _env_cached(("fi", B, dev_key, dtype),
                                    lambda: torch.arange(B, dtype=dtype, device=device).view(B, 1, 1))
        if "fn" in used:
            env["fn"] = _env_cached(("fn", B, dev_key, dtype),
                                    lambda: torch.tensor(float(B), dtype=dtype, device=device))
    else:
        for name, val in _SCALAR_BUILTIN_DEFAULTS.items():
            if name in used:
                env[name] = _env_cached((name, "scalar", dev_key, None),
                                        lambda v=val: torch.tensor(v, device=device))

    if "PI" in used:
        env["PI"] = _env_cached(("PI", dev_key, dtype),
                                lambda: torch.tensor(math.pi, dtype=dtype, device=device))
    if "TAU" in used:
        env["TAU"] = _env_cached(("TAU", dev_key, dtype),
                                 lambda: torch.tensor(math.tau, dtype=dtype, device=device))
    if "E" in used:
        env["E"] = _env_cached(("E", dev_key, dtype),
                               lambda: torch.tensor(math.e, dtype=dtype, device=device))
    if "ic" in used:
        env["ic"] = _env_cached(("ic", latent_channel_count, dev_key, dtype),
                                lambda: torch.tensor(float(latent_channel_count), dtype=dtype, device=device))

    # ENG-7: no host-time builtins here BY DESIGN — codegen declines any program that
    # reads them (codegen.try_compile -> _Unsupported), so this env never needs to carry
    # a playhead. See the note in codegen for why; the short version is that both this
    # module's cached executor closure and _env_cached would freeze the value.
    return env, sp, used


def _codegen_only_execute(
    program: Any,
    bindings: dict[str, Any],
    type_map: dict,
    device: str | torch.device,
    latent_channel_count: int = 0,
    output_names: list[str] | None = None,
    used_builtins: set[str] | None = None,
    precision: str = "fp32",
    fingerprint: str | None = None,
    *,
    time_context: dict | None,
) -> torch.Tensor | dict:
    """Execute via codegen flat function WITHOUT torch.compile.

    For loop-heavy programs where torch.compile overhead exceeds benefit.
    Still faster than the tree-walking interpreter (no per-node dispatch).
    Falls back to plain interpreter if codegen fails.

    The generated function (or a None "unsupported" sentinel) is memoized per
    fingerprint so re-executions skip the per-frame emit+compile()+exec().

    `time_context` is KEYWORD-ONLY and REQUIRED for the reason spelled out on
    `_plain_execute` — a forgotten forward is a frozen playhead, not an error, and this
    function is where the codegen decline lands a time-reading program. Pass
    `time_context=None` to mean "no host playhead".
    """
    cg_fn = _get_or_make_codegen_fn(program, type_map, fingerprint)

    if cg_fn is None:
        # ENG-7: "unsupported" is a lie for a time-reading program — the emitter handles
        # them fine, and the decline is a CACHING policy (see codegen.try_compile). This
        # is the channel the user actually sees on the DEFAULT path: it reaches the HUD
        # tooltip via tier_trace, not the _show_once log line. Sending someone hunting for
        # an unsupported construct in a program that has none is the whole cost.
        from .codegen import _reads_time_builtin
        reason = ("reads frame/fps/time — the compiled tiers cache per program and would "
                  "replay a stale playhead (ENG-7)"
                  if _reads_time_builtin(program) else "unsupported")
        tier_trace.record("interpreter", fallback_from="codegen", reason=reason)
        return _plain_execute(program, bindings, type_map, device,
                              latent_channel_count, output_names,
                              used_builtins=used_builtins, precision=precision,
                              time_context=time_context)

    dev = _canon_device(device)

    # Ensure tensor bindings are contiguous (same as torch.compile path) and
    # co-located on the compute device (XPU: one direct hop instead of raise+retry)
    contiguous_bindings = _contiguous_bindings(bindings, dev)
    ingest_event = _record_ingest_event(bindings, dev)

    env, sp, _ = _build_codegen_env(program, contiguous_bindings, dev, latent_channel_count,
                                    used_builtins=used_builtins, precision=precision)
    stdlib_fns = TEXStdlib.get_functions()

    try:
        with torch.inference_mode():
            _invoke_cg(cg_fn, env, contiguous_bindings, stdlib_fns, dev, sp)
    except Exception as e:
        _show_once(
            "codegen_only_fallback",
            f"[TEX] Codegen-only execution failed, falling back to interpreter: {e}",
            level="warning",
        )
        tier_trace.record("interpreter", fallback_from="codegen", reason=str(e))
        return _plain_execute(program, bindings, type_map, device,
                              latent_channel_count, output_names,
                              used_builtins=used_builtins, precision=precision,
                              time_context=time_context)

    tier_trace.record("codegen")
    # XPU fence: async ingest DMA must land before the cook's outputs escape.
    if ingest_event is not None:
        try:
            ingest_event.synchronize()
        except Exception:
            pass
    if output_names is not None:
        return {name: contiguous_bindings[name] for name in output_names}
    return contiguous_bindings.get("OUT")


def _try_compile(
    device_type: str,
    program: Any = None, type_map: dict | None = None,
    used_builtins: set[str] | None = None,
    precision: str = "fp32",
    fingerprint: str | None = None,
) -> tuple[Callable, str | None] | None:
    """
    Attempt to create a torch.compile-d callable for a TEX program.

    Strategy:
      1. Try codegen first — compiles TEX AST to a flat Python function that
         torch.compile can actually fuse (no dispatch overhead).
      2. Fall back to wrapping the tree-walking interpreter (limited benefit
         but still catches some fusion opportunities).

    Cascades through backends until one works or all fail.
    Returns (callable, backend_name) — backend_name is None for the
    codegen-only adapter — or None if torch.compile is entirely unavailable.
    """
    # Ensure MSVC env is set up before first compile attempt (Windows)
    _setup_msvc_env()

    backend = _select_backend(device_type)
    if backend is None:
        _show_once(
            "no_backend",
            "[TEX] No torch.compile backend available — using plain interpreter.",
        )
        return None

    # ── Try codegen path first (flat function → much better for torch.compile).
    # Reuse the persisted/memoized codegen fn (PC-3) rather than re-emitting.
    cg_fn = None
    if program is not None and type_map is not None:
        cg_fn = _get_or_make_codegen_fn(program, type_map, fingerprint)

    if cg_fn is None:
        # No codegen — DON'T wrap the tree-walking interpreter in torch.compile.
        # Measured (v0.20): there is no configuration where it wins — CPU inductor
        # needs MSVC (always fails → blacklist), CUDA-without-Triton fails at first
        # call, and CUDA-WITH-Triton is the worst case: dynamo can't stably guard
        # the interpreter's dynamic dict-driven dispatch (CLASS_MATCH guards on
        # interpreter classes, unserializable under caching_precompile) and
        # re-traces every cook — ~100x slower than the plain interpreter it wraps.
        if program is not None:
            # ENG-7: distinguish "codegen can't express this" from "codegen is barred
            # from this". The time-builtin decline is a CACHING policy (every artifact
            # downstream of codegen is keyed by a fingerprint that does not move when the
            # playhead does), not a capability limit — telling a user with an animating
            # program that codegen "doesn't support" it would be false, and would send
            # them looking for the unsupported construct.
            from .codegen import _reads_time_builtin
            if _reads_time_builtin(program):
                _show_once("codegen_time_builtin",
                           "[TEX] This program reads frame/fps/time, so it runs on the "
                           "interpreter — the compiled tiers cache per program, and would "
                           "replay a stale playhead (ENG-7).")
            else:
                _show_once("codegen_fallback",
                           "[TEX] Codegen unsupported for this program — plain interpreter "
                           "(interpreter-wrapping torch.compile measured as never beneficial)")
        return None

    # Build an adapter that matches the execute_compiled calling convention
    # but delegates to the codegen-generated flat function.
    stdlib_fns = TEXStdlib.get_functions()

    # If codegen has stdlib function calls (graph breaks), skip torch.compile
    # and return the codegen adapter directly — torch.compile overhead exceeds benefit
    if getattr(cg_fn, '_has_fn_calls', False):
        def _codegen_exec_eager(program, bindings, type_map, device,
                                latent_channel_count=0, output_names=None):
            dev = _canon_device(device)
            env, sp, _ = _build_codegen_env(program, bindings, dev, latent_channel_count,
                                            used_builtins=used_builtins, precision=precision)
            _invoke_cg(cg_fn, env, bindings, stdlib_fns, dev, sp)
            if output_names is not None:
                return {name: bindings[name] for name in output_names}
            return bindings.get("OUT")

        _show_once("codegen_only_fn_calls",
                   "[TEX] Codegen has stdlib calls — using codegen-only (no torch.compile)")
        return _codegen_exec_eager, None

    try:
        # Point inductor's disk cache at TEX's owned, evictable location so
        # compiled kernels persist across process restarts.
        _ensure_inductor_cache_dir()

        # G' (v0.20): compile ONLY the flat generated _tex_fn — pure tensor ops.
        # Compiling the whole adapter put _build_codegen_env's AST-walking /
        # dict-building Python inside the traced region; dynamo guarded on
        # interpreter-module classes (CLASS_MATCH, unserializable under
        # caching_precompile) and re-traced EVERY cook — measured 125ms/cook vs
        # 0.9ms plain interpreter on for_10@1024² (sm_120, torch 2.12 + Triton).
        # With env construction eager and a stable tensor-only compile target,
        # dynamo's guards hold and warm cooks run the cached kernel.
        compiled_flat = torch.compile(
            cg_fn,
            backend=backend,
            mode="reduce-overhead" if device_type == "cuda" else "default",
            fullgraph=False,  # Allow graph breaks at for-loop .item() calls
        )

        # Under mode="reduce-overhead" (CUDA) inductor's cudagraph trees OWN the
        # output storage — static buffers reused by the next replay. Outputs that
        # escape to ComfyUI must be cloned out at the graph boundary (same
        # stage→replay→clone contract as the graphed tier), else the next cook
        # overwrites them ("accessing tensor output of CUDAGraphs that has been
        # overwritten" — reproduced on sm_120/torch 2.12).
        _clone_out = device_type == "cuda"

        def _codegen_exec(program, bindings, type_map, device,
                          latent_channel_count=0, output_names=None):
            dev = _canon_device(device)
            env, sp, _ = _build_codegen_env(program, bindings, dev, latent_channel_count,
                                            used_builtins=used_builtins, precision=precision)
            _invoke_cg(compiled_flat, env, bindings, stdlib_fns, dev, sp)
            if output_names is not None:
                if _clone_out:
                    return {name: bindings[name].clone()
                            if isinstance(bindings[name], torch.Tensor) else bindings[name]
                            for name in output_names}
                return {name: bindings[name] for name in output_names}
            out = bindings.get("OUT")
            if _clone_out and isinstance(out, torch.Tensor):
                out = out.clone()
            return out

        _backend_status[(backend, device_type)] = True
        _show_once(
            f"compile_ok_{backend}",
            f"[TEX] torch.compile using '{backend}' backend (flat codegen fn)",
        )
        return _codegen_exec, backend

    except Exception as e:
        _backend_status[(backend, device_type)] = False
        _show_once(
            f"compile_fail_{backend}",
            f"[TEX] torch.compile backend '{backend}' unavailable: {e}",
            level="warning",
        )

        # Windows-specific guidance for inductor
        err_str = str(e).lower()
        if backend == "inductor" and ("cl" in err_str or "compiler" in err_str):
            _show_once(
                "msvc_hint",
                "[TEX] Hint: Install 'Visual Studio Build Tools' with the "
                "'Desktop development with C++' workload for best torch.compile "
                "performance on Windows.  https://visualstudio.microsoft.com/visual-cpp-build-tools/",
            )
        # CC-1: CUDA inductor needs Triton, which base torch doesn't ship on
        # Windows (usually surfaces at first call, not here — see _maybe_triton_hint).
        if backend == "inductor":
            _maybe_triton_hint(err_str, device_type)

        # Recurse to try the next backend in the cascade (fingerprint kept so the
        # codegen-fn memo isn't re-emitted on the retry)
        return _try_compile(device_type, program, type_map,
                            used_builtins=used_builtins, precision=precision,
                            fingerprint=fingerprint)


def clear_compiled_cache():
    """Clear all cached compiled functions and memos (useful for testing)."""
    _compiled_cache.clear()
    _compile_blacklist.clear()
    _verify_state.clear()      # G: post-commit verification windows
    _deferred_ev.clear()       # LAT-3: pending deferred-readback event pairs
    _ENV_TENSOR_CACHE.clear()  # codegen builtin tensors (test isolation)
    clear_plain_interp_caches()  # C6: the persistent fallback interpreter's LRUs
    # Reset backend status so backends are re-probed
    _backend_status.clear()
    _route_memo.clear()
    _stencil_route_memo.clear()
    _warnings_shown.clear()
    try:  # P4: the is_tile_safe fingerprint memo (mirror of _stencil_route_memo)
        from ..tex_memory import _tile_safe_memo
        _tile_safe_memo.clear()
    except Exception:
        pass
    # The codegen memo now lives in TEXCache (PC-3); clear its memory tier too
    # for test isolation (disk sidecars are cleared by TEXCache.clear_all()).
    try:
        from ..tex_cache import get_cache
        get_cache()._codegen_memory.clear()
    except Exception:
        pass



