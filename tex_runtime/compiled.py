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

# Routing-gate memo: fingerprint -> (op_count, loop_depth). Both are pure
# functions of the AST, which is keyed by the same (code, binding-types)
# fingerprint, so the two full AST walks run once per program instead of on
# every uncached frame. has_spatial is deliberately NOT memoized: it depends
# on binding VALUES (a [B,H,W] mask and a Python float scalar both fingerprint
# as FLOAT but need different routes).
_route_memo: "_OrderedDict[str, tuple[int, int]]" = _OrderedDict()
_ROUTE_MEMO_MAX = 256

# Codegen-only flat functions: fingerprint -> cg_fn. A None value means
# "codegen unsupported for this program" — skip re-attempting emit+exec on
# every frame. cg_fn captures no device/precision/bindings (everything is
# passed per call), so the fingerprint alone is a sufficient key.
_cg_only_cache: "_OrderedDict[str, Callable | None]" = _OrderedDict()
_CG_ONLY_CACHE_MAX = 64

# One-time log messages (avoid spamming the console)
_warnings_shown: set[str] = set()

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
        program:      Parsed AST (Program node).
        bindings:     Mapping of @ binding names to tensor / scalar values.
        type_map:     AST node id -> TEXType (from type checker).
        device:       Target device ("cpu", "cuda", "cuda:0", …).
        fingerprint:  Cache key produced by TEXCache.fingerprint().
        output_names: List of named outputs for multi-output programs.

    Returns:
        The @OUT tensor result, or a dict of {name: tensor} for multi-output.
    """
    device_obj = torch.device(device)
    device_type = device_obj.type  # "cpu" or "cuda"
    cache_key = (fingerprint, device_type, precision)

    # ── Blacklist: skip programs that previously crashed torch.compile
    if fingerprint in _compile_blacklist:
        return _plain_execute(program, bindings, type_map, device,
                              latent_channel_count, output_names,
                              used_builtins=used_builtins, precision=precision)

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
                                  used_builtins=used_builtins, precision=precision)
        # Use codegen WITHOUT torch.compile for deeply nested loops
        # (graph breaks and recompilation make torch.compile slower)
        if loop_depth > _COMPILE_MAX_LOOP_DEPTH:
            return _codegen_only_execute(program, bindings, type_map, device,
                                         latent_channel_count, output_names,
                                         used_builtins=used_builtins, precision=precision,
                                         fingerprint=fingerprint)
        # Use plain interpreter for programs without spatial tensor context
        # (procedural noise, etc.) — codegen env setup overhead exceeds
        # benefit when all operations are on scalar tensors
        has_spatial = any(isinstance(v, torch.Tensor) and v.dim() >= 3
                         for v in bindings.values())
        if not has_spatial:
            return _plain_execute(program, bindings, type_map, device,
                                  latent_channel_count, output_names,
                                  used_builtins=used_builtins, precision=precision)

    # Ensure tensor bindings are contiguous — Inductor's codegen can
    # fail on non-contiguous strides (e.g. BHWC images loaded with
    # stride patterns like [1090020, 2220, 3, 1]).
    contiguous_bindings = {}
    for k, v in bindings.items():
        if isinstance(v, torch.Tensor) and not v.is_contiguous():
            contiguous_bindings[k] = v.contiguous()
        else:
            contiguous_bindings[k] = v

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
            with torch.inference_mode():
                # Get or create the compiled callable (on THIS thread)
                if cache_key not in _compiled_cache:
                    entry = _try_compile(device_type, program, type_map,
                                         used_builtins=used_builtins, precision=precision)
                    if entry is None:
                        # No backend available — run plain interpreter here
                        return _plain_execute(program, contiguous_bindings, type_map,
                                              device, latent_channel_count, output_names,
                                              used_builtins=used_builtins, precision=precision)
                    _compiled_cache[cache_key] = entry
                    if len(_compiled_cache) > _COMPILED_CACHE_MAX:
                        # Evict the oldest. Do NOT torch._dynamo.reset() here:
                        # this runs on the disposable worker thread, and dynamo
                        # state is PROCESS-GLOBAL — resetting it would corrupt the
                        # very thread that invokes compiled_fn just below (and the
                        # calling thread). The bounded cache already caps growth.
                        _compiled_cache.popitem(last=False)

                compiled_fn, _entry_backend = _compiled_cache[cache_key]
                _compiled_cache.move_to_end(cache_key)
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
        failed_backend = popped[1] if popped is not None else None
        # torch.compile() wraps lazily, so a missing backend toolchain (Triton
        # on CUDA, cl.exe on Windows CPU) only surfaces HERE, at the first
        # execution of the compiled callable. That is a property of the
        # (backend, device) pair, not of the program: mark the backend
        # unavailable so the next run cascades (inductor -> cudagraphs on
        # CUDA) instead of blacklisting every new fingerprint.
        _lower = _emsg.lower()
        backend_unavailable = (
            type(compile_error).__name__ == "BackendCompilerFailed"
            and failed_backend in ("inductor", "cudagraphs")
            and ("triton" in _lower or "cl.exe" in _lower
                 or "cl is not found" in _lower)
        )
        if backend_unavailable:
            _backend_status[(failed_backend, device_type)] = False
        elif not is_user_limit:
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
                              used_builtins=used_builtins, precision=precision)

    return result


def _plain_execute(
    program: Any,
    bindings: dict[str, Any],
    type_map: dict,
    device: str | torch.device,
    latent_channel_count: int = 0,
    output_names: list[str] | None = None,
    used_builtins: set[str] | None = None,
    precision: str = "fp32",
) -> torch.Tensor | dict:
    """Execute without torch.compile (standard tree-walking interpreter)."""
    interp = Interpreter()
    return interp.execute(program, bindings, type_map, device=device,
                          latent_channel_count=latent_channel_count,
                          output_names=output_names,
                          precision=precision,
                          used_builtins=used_builtins)


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

    if sp:
        B, H, W = sp
        if "ix" in used or "u" in used:
            ix = torch.arange(W, dtype=dtype, device=device).view(1, 1, W)
            if "ix" in used:
                env["ix"] = ix
            if "u" in used:
                env["u"] = (ix / max(W - 1, 1)).expand(B, H, W)
        if "iy" in used or "v" in used:
            iy = torch.arange(H, dtype=dtype, device=device).view(1, H, 1)
            if "iy" in used:
                env["iy"] = iy
            if "v" in used:
                env["v"] = (iy / max(H - 1, 1)).expand(B, H, W)
        if "iw" in used:
            env["iw"] = torch.tensor(float(W), dtype=dtype, device=device)
        if "ih" in used:
            env["ih"] = torch.tensor(float(H), dtype=dtype, device=device)
        if "px" in used:
            env["px"] = torch.tensor(1.0 / max(W, 1), dtype=dtype, device=device)
        if "py" in used:
            env["py"] = torch.tensor(1.0 / max(H, 1), dtype=dtype, device=device)
        if "fi" in used:
            env["fi"] = torch.arange(B, dtype=dtype, device=device).view(B, 1, 1)
        if "fn" in used:
            env["fn"] = torch.tensor(float(B), dtype=dtype, device=device)
    else:
        for name, val in _SCALAR_BUILTIN_DEFAULTS.items():
            if name in used:
                env[name] = torch.tensor(val, device=device)

    if "PI" in used:
        env["PI"] = torch.tensor(math.pi, dtype=dtype, device=device)
    if "TAU" in used:
        env["TAU"] = torch.tensor(math.tau, dtype=dtype, device=device)
    if "E" in used:
        env["E"] = torch.tensor(math.e, dtype=dtype, device=device)
    if "ic" in used:
        env["ic"] = torch.tensor(float(latent_channel_count), dtype=dtype, device=device)

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
) -> torch.Tensor | dict:
    """Execute via codegen flat function WITHOUT torch.compile.

    For loop-heavy programs where torch.compile overhead exceeds benefit.
    Still faster than the tree-walking interpreter (no per-node dispatch).
    Falls back to plain interpreter if codegen fails.

    The generated function (or a None "unsupported" sentinel) is memoized per
    fingerprint so re-executions skip the per-frame emit+compile()+exec().
    """
    if fingerprint is not None and fingerprint in _cg_only_cache:
        cg_fn = _cg_only_cache[fingerprint]
        _cg_only_cache.move_to_end(fingerprint)
    else:
        try:
            cg_fn = _try_codegen(program, type_map)
        except Exception:
            cg_fn = None
        if fingerprint is not None:
            _cg_only_cache[fingerprint] = cg_fn
            while len(_cg_only_cache) > _CG_ONLY_CACHE_MAX:
                _cg_only_cache.popitem(last=False)

    if cg_fn is None:
        return _plain_execute(program, bindings, type_map, device,
                              latent_channel_count, output_names,
                              used_builtins=used_builtins, precision=precision)

    dev = torch.device(device) if not isinstance(device, torch.device) else device

    # Ensure tensor bindings are contiguous (same as torch.compile path)
    contiguous_bindings = {}
    for k, v in bindings.items():
        if isinstance(v, torch.Tensor) and not v.is_contiguous():
            contiguous_bindings[k] = v.contiguous()
        else:
            contiguous_bindings[k] = v

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
        return _plain_execute(program, bindings, type_map, device,
                              latent_channel_count, output_names,
                              used_builtins=used_builtins, precision=precision)

    if output_names is not None:
        return {name: contiguous_bindings[name] for name in output_names}
    return contiguous_bindings.get("OUT")


def _try_compile(
    device_type: str,
    program: Any = None, type_map: dict | None = None,
    used_builtins: set[str] | None = None,
    precision: str = "fp32",
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

    # ── Try codegen path first (flat function → much better for torch.compile)
    cg_fn = None
    if program is not None and type_map is not None:
        try:
            cg_fn = _try_codegen(program, type_map)
        except Exception:
            cg_fn = None

    if cg_fn is not None:
        # Build an adapter that matches the execute_compiled calling convention
        # but delegates to the codegen-generated flat function.
        stdlib_fns = TEXStdlib.get_functions()

        def _codegen_exec(program, bindings, type_map, device,
                          latent_channel_count=0, output_names=None):
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            env, sp, _ = _build_codegen_env(program, bindings, dev, latent_channel_count,
                                            used_builtins=used_builtins, precision=precision)

            _invoke_cg(cg_fn, env, bindings, stdlib_fns, dev, sp)

            if output_names is not None:
                return {name: bindings[name] for name in output_names}
            return bindings.get("OUT")

        # If codegen has stdlib function calls (graph breaks), skip torch.compile
        # and return the codegen adapter directly — torch.compile overhead exceeds benefit
        if getattr(cg_fn, '_has_fn_calls', False):
            _show_once("codegen_only_fn_calls",
                       "[TEX] Codegen has stdlib calls — using codegen-only (no torch.compile)")
            return _codegen_exec, None

        target_fn = _codegen_exec
        _show_once("codegen_active", "[TEX] Using codegen path for torch.compile (flat function)")
    else:
        # Fall back to wrapping the interpreter — bake precision/used_builtins
        # into a closure (mirroring _codegen_exec): the compiled callable is
        # invoked with only the six positional args, which would otherwise
        # silently reset both to their defaults.
        def _interp_exec(program, bindings, type_map, device,
                         latent_channel_count=0, output_names=None):
            return _plain_execute(program, bindings, type_map, device,
                                  latent_channel_count, output_names,
                                  used_builtins=used_builtins, precision=precision)

        target_fn = _interp_exec
        if program is not None:
            _show_once("codegen_fallback",
                       "[TEX] Codegen unsupported for this program, wrapping interpreter")

    try:
        # Configure torch.compile cache directory if the cache is available
        try:
            from ..tex_cache import get_cache

            tc_dir = str(get_cache().torch_compile_cache_dir)
            os.makedirs(tc_dir, exist_ok=True)
            # Point inductor's disk cache here and enable FxGraph caching
            # to persist compiled kernels across process restarts
            torch._inductor.config.cache_dir = tc_dir  # type: ignore[attr-defined]
            torch._inductor.config.fx_graph_cache = True  # type: ignore[attr-defined]
        except Exception:
            pass  # Not critical — torch uses its default cache location

        compiled = torch.compile(
            target_fn,
            backend=backend,
            mode="reduce-overhead" if device_type == "cuda" else "default",
            fullgraph=False,  # Allow graph breaks at for-loop .item() calls
        )

        _backend_status[(backend, device_type)] = True
        _show_once(
            f"compile_ok_{backend}",
            f"[TEX] torch.compile using '{backend}' backend",
        )
        return compiled, backend

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

        # Recurse to try the next backend in the cascade
        return _try_compile(device_type, program, type_map,
                            used_builtins=used_builtins, precision=precision)


def clear_compiled_cache():
    """Clear all cached compiled functions and memos (useful for testing)."""
    _compiled_cache.clear()
    _compile_blacklist.clear()
    # Reset backend status so backends are re-probed
    _backend_status.clear()
    _route_memo.clear()
    _cg_only_cache.clear()
    _warnings_shown.clear()



