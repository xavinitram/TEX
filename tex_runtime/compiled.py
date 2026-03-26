"""
TEX Compiled Execution — optional torch.compile wrapper around the interpreter.

When codegen can compile a TEX program to a flat Python function (no dispatch
overhead), that function is wrapped with torch.compile for kernel fusion.
Otherwise, falls back to wrapping the tree-walking interpreter.

If compilation fails for any reason the plain interpreter is used instead —
execution never fails because of torch.compile.

Cache key: (code_fingerprint, device_type).
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
from typing import Any, Callable

import torch

from .interpreter import Interpreter, _ensure_spatial, _broadcast_pair, _collect_identifiers
from .codegen import try_compile as _try_codegen, _CgBreak, _CgContinue
from .stdlib import TEXStdlib, SAFE_EPSILON
from ..tex_compiler.type_checker import CHANNEL_MAP

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

# Compiled executors: (fingerprint, device_type) -> compiled callable
_compiled_cache: dict[tuple[str, str], Callable] = {}

# Fingerprints that crashed torch.compile — skip on subsequent calls
_compile_blacklist: set[str] = set()

# Track which backends have been tested and whether they work.
# None = untested, True = works, False = failed
_backend_status: dict[str, bool | None] = {
    "inductor": None,
    "cudagraphs": None,
}

# One-time log messages (avoid spamming the console)
_warnings_shown: set[str] = set()

# Minimum tensor-op count for torch.compile to be worthwhile.
# Below this threshold the fusion benefit cannot overcome tracing overhead.
_COMPILE_OP_THRESHOLD = 8


# ── Helpers ───────────────────────────────────────────────────────────

def _show_once(key: str, msg: str, level: str = "info"):
    """Log a message at most once per session."""
    if key not in _warnings_shown:
        _warnings_shown.add(key)
        getattr(logger, level)(msg)


def _count_tensor_ops(program: Any) -> int:
    """Count tensor operations in an AST to estimate torch.compile benefit.

    Uses the same traversal pattern as interpreter._collect_identifiers.
    Counts BinOps, FunctionCalls, VecConstructors, TernaryOps, UnaryOps,
    and CastExprs — the operations that produce tensor work.
    """
    from ..tex_compiler.ast_nodes import (
        BinOp, UnaryOp, TernaryOp, FunctionCall, VecConstructor,
        CastExpr, MatConstructor, VarDecl, Assignment, IfElse, ForLoop,
        WhileLoop, ExprStatement, ArrayDecl, ChannelAccess,
        ArrayIndexAccess, ArrayLiteral,
    )
    _OP_TYPES = (BinOp, UnaryOp, TernaryOp, FunctionCall,
                 VecConstructor, MatConstructor, CastExpr)
    count = 0
    stack = list(program.statements)
    while stack:
        node = stack.pop()
        cls = type(node)
        if isinstance(node, _OP_TYPES):
            count += 1
        # Traverse children (same pattern as _collect_identifiers)
        if cls is VarDecl:
            if node.initializer:
                stack.append(node.initializer)
        elif cls is Assignment:
            stack.append(node.target)
            stack.append(node.value)
        elif cls is IfElse:
            stack.append(node.condition)
            stack.extend(node.then_body)
            stack.extend(node.else_body)
        elif cls is ForLoop:
            stack.append(node.init)
            stack.append(node.condition)
            stack.append(node.update)
            stack.extend(node.body)
        elif cls is WhileLoop:
            stack.append(node.condition)
            stack.extend(node.body)
        elif cls is ExprStatement:
            stack.append(node.expr)
        elif cls is ArrayDecl:
            if node.initializer:
                stack.append(node.initializer)
        elif cls is BinOp:
            stack.append(node.left)
            stack.append(node.right)
        elif cls is UnaryOp:
            stack.append(node.operand)
        elif cls is TernaryOp:
            stack.append(node.condition)
            stack.append(node.true_expr)
            stack.append(node.false_expr)
        elif cls is FunctionCall:
            stack.extend(node.args)
        elif cls is VecConstructor:
            stack.extend(node.args)
        elif cls is MatConstructor:
            stack.extend(node.args)
        elif cls is CastExpr:
            stack.append(node.expr)
        elif cls is ChannelAccess:
            stack.append(node.object)
        elif cls is ArrayIndexAccess:
            stack.append(node.array)
            stack.append(node.index)
        elif cls is ArrayLiteral:
            stack.extend(node.elements)
    return count


def _select_backend(device_type: str) -> str | None:
    """
    Pick the best available torch.compile backend.

    Priority:
      GPU → inductor > cudagraphs > None
      CPU → inductor > None

    Backends already marked as failed are skipped.
    """
    candidates = []
    if device_type == "cuda":
        candidates = ["inductor", "cudagraphs"]
    else:
        candidates = ["inductor"]

    for backend in candidates:
        if _backend_status.get(backend) is not False:
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
    cache_key = (fingerprint, device_type)

    # ── Blacklist: skip programs that previously crashed torch.compile
    if fingerprint in _compile_blacklist:
        return _plain_execute(program, bindings, type_map, device,
                              latent_channel_count, output_names)

    # ── Complexity gate: skip torch.compile for trivial programs
    #    where tracing overhead exceeds any possible fusion benefit.
    if cache_key not in _compiled_cache:
        op_count = _count_tensor_ops(program)
        if op_count < _COMPILE_OP_THRESHOLD:
            return _plain_execute(program, bindings, type_map, device,
                                  latent_channel_count, output_names)

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
                    compiled_fn = _try_compile(cache_key, device_type, program, type_map)
                    if compiled_fn is None:
                        # No backend available — run plain interpreter here
                        return _plain_execute(program, contiguous_bindings, type_map,
                                              device, latent_channel_count, output_names)
                    _compiled_cache[cache_key] = compiled_fn

                compiled_fn = _compiled_cache[cache_key]
                return compiled_fn(program, contiguous_bindings, type_map, device,
                                   latent_channel_count, output_names)
        except Exception as e:
            compile_error = e
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_compile_and_run)
        result = future.result()

    if compile_error is not None:
        _show_once(
            f"compile_exec_fail_{fingerprint[:12]}",
            f"[TEX] torch.compile execution failed, falling back to interpreter: {compile_error}",
            level="warning",
        )
        _compiled_cache.pop(cache_key, None)
        _compile_blacklist.add(fingerprint)
        # dynamo.reset() also runs in a clean thread to avoid tainting main
        def _reset_dynamo():
            try:
                torch._dynamo.reset()
            except Exception:
                pass
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_reset_dynamo).result()
        return _plain_execute(program, bindings, type_map, device,
                              latent_channel_count, output_names)

    return result


def _plain_execute(
    program: Any,
    bindings: dict[str, Any],
    type_map: dict,
    device: str | torch.device,
    latent_channel_count: int = 0,
    output_names: list[str] | None = None,
) -> torch.Tensor | dict:
    """Execute without torch.compile (standard tree-walking interpreter)."""
    interp = Interpreter()
    return interp.execute(program, bindings, type_map, device=device,
                          latent_channel_count=latent_channel_count,
                          output_names=output_names)


def _try_compile(
    cache_key: tuple[str, str], device_type: str,
    program: Any = None, type_map: dict | None = None,
) -> Callable | None:
    """
    Attempt to create a torch.compile-d callable for a TEX program.

    Strategy:
      1. Try codegen first — compiles TEX AST to a flat Python function that
         torch.compile can actually fuse (no dispatch overhead).
      2. Fall back to wrapping the tree-walking interpreter (limited benefit
         but still catches some fusion opportunities).

    Cascades through backends until one works or all fail.
    Returns None if torch.compile is entirely unavailable.
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
            env = {}

            # Determine spatial shape from image bindings
            sp = None
            for v in bindings.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 3:
                    sp = (v.shape[0], v.shape[1], v.shape[2])
                    break

            # Create builtins (same logic as Interpreter._create_builtins)
            # u/v use expand() (view, no copy) for torch.stack compat in sampling
            used = _collect_identifiers(program)
            if sp:
                B, H, W = sp
                dtype = torch.float32
                if "ix" in used or "u" in used:
                    ix = torch.arange(W, dtype=dtype, device=dev).view(1, 1, W)
                    if "ix" in used:
                        env["ix"] = ix
                    if "u" in used:
                        env["u"] = (ix / max(W - 1, 1)).expand(B, H, W)
                if "iy" in used or "v" in used:
                    iy = torch.arange(H, dtype=dtype, device=dev).view(1, H, 1)
                    if "iy" in used:
                        env["iy"] = iy
                    if "v" in used:
                        env["v"] = (iy / max(H - 1, 1)).expand(B, H, W)
                if "iw" in used:
                    env["iw"] = torch.tensor(float(W), dtype=dtype, device=dev)
                if "ih" in used:
                    env["ih"] = torch.tensor(float(H), dtype=dtype, device=dev)
                if "fi" in used:
                    env["fi"] = torch.arange(B, dtype=dtype, device=dev).view(B, 1, 1)
                if "fn" in used:
                    env["fn"] = torch.tensor(float(B), dtype=dtype, device=dev)
            else:
                _scalar_defaults = {"ix": 0.0, "iy": 0.0, "u": 0.0, "v": 0.0,
                                    "iw": 1.0, "ih": 1.0, "fi": 0.0, "fn": 1.0}
                for name, val in _scalar_defaults.items():
                    if name in used:
                        env[name] = torch.tensor(val, device=dev)

            if "PI" in used:
                env["PI"] = torch.tensor(math.pi, dtype=torch.float32, device=dev)
            if "E" in used:
                env["E"] = torch.tensor(math.e, dtype=torch.float32, device=dev)
            if "ic" in used:
                env["ic"] = torch.tensor(float(latent_channel_count), dtype=torch.float32, device=dev)

            # Call the codegen-generated function
            cg_fn(env, bindings, stdlib_fns, dev, sp,
                   torch, _broadcast_pair, _ensure_spatial, torch.where,
                   math, SAFE_EPSILON, CHANNEL_MAP, _MAX_LOOP_ITERATIONS,
                   _CgBreak, _CgContinue)

            # Extract outputs
            if output_names is not None:
                return {name: bindings[name] for name in output_names}
            return bindings.get("OUT")

        target_fn = _codegen_exec
        _show_once("codegen_active", "[TEX] Using codegen path for torch.compile (flat function)")
    else:
        # Fall back to wrapping the interpreter directly
        target_fn = _plain_execute
        if cg_fn is None and program is not None:
            _show_once("codegen_fallback",
                       "[TEX] Codegen unsupported for this program, wrapping interpreter")

    try:
        # Configure torch.compile cache directory if the cache is available
        try:
            from ..tex_cache import get_cache

            tc_dir = str(get_cache().torch_compile_cache_dir)
            os.makedirs(tc_dir, exist_ok=True)
            # Point inductor's disk cache here
            torch._inductor.config.cache_dir = tc_dir  # type: ignore[attr-defined]
        except Exception:
            pass  # Not critical — torch uses its default cache location

        compiled = torch.compile(
            target_fn,
            backend=backend,
            mode="reduce-overhead" if device_type == "cuda" else "default",
            fullgraph=False,  # Allow graph breaks at for-loop .item() calls
        )

        _backend_status[backend] = True
        _show_once(
            f"compile_ok_{backend}",
            f"[TEX] torch.compile using '{backend}' backend",
        )
        return compiled

    except Exception as e:
        _backend_status[backend] = False
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
        return _try_compile(cache_key, device_type, program, type_map)


def clear_compiled_cache():
    """Clear all cached compiled functions (useful for testing)."""
    _compiled_cache.clear()
    _compile_blacklist.clear()
    # Reset backend status so they are re-probed
    for k in _backend_status:
        _backend_status[k] = None
    _warnings_shown.clear()



