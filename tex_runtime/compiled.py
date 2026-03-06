"""
TEX Compiled Execution — optional torch.compile wrapper around the interpreter.

Wraps Interpreter.execute() in a torch.compile-d callable with automatic
backend selection and graceful fallback.  If compilation fails for any
reason the plain tree-walking interpreter is used instead — execution
never fails because of torch.compile.

Cache key: (code_fingerprint, device_type).
torch.compile handles shape-based recompilation internally via guards.
"""
from __future__ import annotations

import glob
import logging
import os
import subprocess
import sys
from typing import Any, Callable

import torch

from .interpreter import Interpreter

logger = logging.getLogger("TEX")

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

# Track which backends have been tested and whether they work.
# None = untested, True = works, False = failed
_backend_status: dict[str, bool | None] = {
    "inductor": None,
    "cudagraphs": None,
}

# One-time log messages (avoid spamming the console)
_warnings_shown: set[str] = set()


# ── Helpers ───────────────────────────────────────────────────────────

def _show_once(key: str, msg: str, level: str = "info"):
    """Log a message at most once per session."""
    if key not in _warnings_shown:
        _warnings_shown.add(key)
        getattr(logger, level)(msg)


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
) -> torch.Tensor:
    """
    Execute a TEX program with optional torch.compile acceleration.

    Falls back to the plain interpreter on any failure.

    Args:
        program:     Parsed AST (Program node).
        bindings:    Mapping of @ binding names to tensor / scalar values.
        type_map:    AST node id -> TEXType (from type checker).
        device:      Target device ("cpu", "cuda", "cuda:0", …).
        fingerprint: Cache key produced by TEXCache.fingerprint().

    Returns:
        The @OUT tensor result.
    """
    device_obj = torch.device(device)
    device_type = device_obj.type  # "cpu" or "cuda"
    cache_key = (fingerprint, device_type)

    # Get or create the compiled callable
    if cache_key not in _compiled_cache:
        compiled_fn = _try_compile(cache_key, device_type)
        if compiled_fn is None:
            # torch.compile not available / all backends failed
            return _plain_execute(program, bindings, type_map, device, latent_channel_count)
        _compiled_cache[cache_key] = compiled_fn

    compiled_fn = _compiled_cache[cache_key]

    try:
        return compiled_fn(program, bindings, type_map, device, latent_channel_count)
    except Exception as e:
        _show_once(
            f"compile_exec_fail_{fingerprint[:12]}",
            f"[TEX] torch.compile execution failed, falling back to interpreter: {e}",
            level="warning",
        )
        # Remove the broken entry so next call retries or falls back
        _compiled_cache.pop(cache_key, None)
        return _plain_execute(program, bindings, type_map, device, latent_channel_count)


def _plain_execute(
    program: Any,
    bindings: dict[str, Any],
    type_map: dict,
    device: str | torch.device,
    latent_channel_count: int = 0,
) -> torch.Tensor:
    """Execute without torch.compile (standard tree-walking interpreter)."""
    interp = Interpreter()
    return interp.execute(program, bindings, type_map, device=device,
                          latent_channel_count=latent_channel_count)


def _try_compile(
    cache_key: tuple[str, str], device_type: str
) -> Callable | None:
    """
    Attempt to create a torch.compile-d interpreter wrapper.

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

    def _interp_fn(program, bindings, type_map, device, latent_channel_count=0):
        interp = Interpreter()
        return interp.execute(program, bindings, type_map, device=device,
                              latent_channel_count=latent_channel_count)

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
            _interp_fn,
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
        return _try_compile(cache_key, device_type)


def clear_compiled_cache():
    """Clear all cached compiled functions (useful for testing)."""
    _compiled_cache.clear()
    # Reset backend status so they are re-probed
    for k in _backend_status:
        _backend_status[k] = None
    _warnings_shown.clear()


def get_compiled_cache_size() -> int:
    """Return the number of cached compiled functions."""
    return len(_compiled_cache)
