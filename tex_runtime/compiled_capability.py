"""Compiled-tier capability probing — SPLIT-47 (v0.47.0, TRK-210).

Split mechanically out of `compiled.py` (the STR-7/SPLIT-I pattern: every body below is
byte-identical to the code it replaced there — AGENTS.md §"Trades to REFUSE", mechanical
moves only, never an "improvement" mid-move). This module owns the "should/can we even try
torch.compile" questions that run BEFORE any real compile attempt: a per-program cost
estimate (`_count_tensor_ops`/`_max_loop_depth`, used by `execute_compiled`'s routing gate),
which backend is worth trying on a device (`_select_backend`, reading `compiled._backend_status`
— the LEARNED, per-process record of what has actually worked), and the static, cheap,
never-by-compiling toolchain probe `compile_capability()` (CC-3) that `run_auto`'s per-cook
"auto" routing consults before it may even attempt a background compile.

`_backend_status` itself stays in `compiled.py` (it is a register-documented store,
`compiled._backend_status`, mutated from several call sites across that module) — this
module only reads it, lazily (`_select_backend`), to avoid a load-time cycle with
`compiled.py`, which imports this module at its own top level to re-export every name below.
`_setup_msvc_env` also stays in `compiled.py` (its own module-level `_msvc_env_initialized`
flag is read directly by `tests/test_v018_ux.py` as `compiled._msvc_env_initialized` for
change detection — moving the flag without moving the reader would make that check silently
decorative, so the whole MSVC domain was left in place); `_probe_cpu_inductor` reaches it the
same lazy way.

Neither back-reference is a load-time import: both are deferred imports inside the one
function that needs them, exactly like `interpreter_spatial.py`/`interpreter_control_flow.py`/
`interpreter_binding.py` already defer their own back-references into `interpreter.py` (see
FIX-OBSROUTE R1 for the exact hazard this avoids — importing this module ALONE, first, in a
fresh process must not reach `compiled.py`'s own body mid-import).
"""
from __future__ import annotations

import sys
from typing import Any

import torch

from ..tex_compiler.ast_nodes import (BinOp, UnaryOp, TernaryOp, FunctionCall,
                                      VecConstructor, MatConstructor, CastExpr,
                                      ForLoop, WhileLoop, IfElse)
from .codegen import _iter_child_nodes

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
    from .compiled import _backend_status
    candidates = []
    if device_type == "cuda":
        candidates = ["inductor", "cudagraphs"]
    else:
        candidates = ["inductor"]

    for backend in candidates:
        if _backend_status.get((backend, device_type)) is not False:
            return backend
    return None


# ── CC-3: compile_capability() — toolchain probe, never by compiling ──────────
#
# `_backend_status` (in `compiled.py`) answers "did backend B actually work THIS process" — it
# is LEARNED, seeded only by a real compile attempt. `compile_capability()` answers the
# question "auto"'s per-cook routing (`run_auto`, `compiled.py`) needs answered BEFORE it may
# even try: is inductor's toolchain prerequisite present at all, for CUDA and for CPU
# independently? It is static and cheap after its first call (see the two probes below),
# computed ONCE per process and cached, so calling it every cook (as the toolchain-aware
# "auto" path does) costs nothing after the first call.
#
# Deliberately NOT `tex_doctor._inductor_prereq`, which this mirrors: that function is
# read-only by contract (never calls `_setup_msvc_env`, a <=30s subprocess) because a
# doctor report must never have a side effect. `compile_capability()` is allowed the ONE
# one-time cost `_setup_msvc_env` pays (idempotent, memoized by `_msvc_env_initialized`),
# because its whole point is to convert "unknown" into a definite answer WITHOUT ever
# entering `_try_compile` — the toolchain-aware "auto" gate (CC-4) has no other way to
# tell "no compiler" from "haven't looked yet".
_capability_cache: dict | None = None


def _probe_cuda_inductor() -> tuple:
    """(ok, reason). Static: torch.cuda availability + `find_spec("triton")` — the same
    fact `_maybe_triton_hint` only ever surfaces AFTER a failed first call, read here
    before any compile is attempted."""
    import importlib.util
    if not torch.cuda.is_available():
        return False, "CUDA is not available (torch.cuda.is_available() is False)"
    if importlib.util.find_spec("triton") is None:
        return False, "Triton is not installed (torch.compile's inductor backend needs it on CUDA)"
    return True, None


def _probe_cpu_inductor() -> tuple:
    """(ok, reason). Non-Windows: a C compiler on PATH — inductor's own prerequisite,
    checked without invoking it. Windows: run the codebase's existing vcvarsall search
    (`_setup_msvc_env`, idempotent) then check PATH for `cl.exe`. `_setup_msvc_env` is
    what promotes INCLUDE/LIB/PATH into this process's own environment (or leaves them
    exactly as a Developer Command Prompt already set them) — a PATH-only check after it
    reads the definitive POST-search state via `shutil.which` alone, with no environment
    read of this function's own (PUB-1: a new `os.environ` site in a shipped file moves a
    pinned ratchet; `shutil.which`'s own internal PATH read is not one)."""
    import shutil
    if sys.platform != "win32":
        if shutil.which("cc") or shutil.which("gcc") or shutil.which("clang"):
            return True, None
        return False, "no C compiler (cc/gcc/clang) found on PATH"
    from .compiled import _setup_msvc_env
    _setup_msvc_env()
    if shutil.which("cl") is not None:
        return True, None
    return False, ("no MSVC (cl.exe) found on PATH after the vcvarsall search (this "
                   "process already ran it and found nothing)")


def compile_capability() -> dict:
    """Read-only, process-wide capability report for torch.compile's inductor backend —
    the answer "auto" (`run_auto`, below) needs BEFORE deciding whether to even attempt a
    background compile. Probed ONCE per process (cached; see
    `_reset_capability_cache_for_test`) by `importlib.util.find_spec("triton")` (CUDA) and
    the codebase's existing MSVC search (`_setup_msvc_env`, CPU on Windows) / a PATH check
    (CPU elsewhere) — never by compiling, so calling this can never pay a failed-compile tax.

    Returns::

        {"cuda_inductor": bool, "cpu_inductor": bool,
         "reason": {"<key>": str}}   # a "reason" entry exists only for a False key

    A `True` reading means the PREREQUISITE holds, not that a compile will succeed or win a
    trial — `_backend_status` (measured, this process) and autotier's own verdict (measured,
    per program) can still say no afterward. This function only removes the one failure mode
    that otherwise costs a real compile attempt to discover: no toolchain at all."""
    global _capability_cache
    if _capability_cache is None:
        cuda_ok, cuda_why = _probe_cuda_inductor()
        cpu_ok, cpu_why = _probe_cpu_inductor()
        reason = {}
        if not cuda_ok:
            reason["cuda_inductor"] = cuda_why
        if not cpu_ok:
            reason["cpu_inductor"] = cpu_why
        _capability_cache = {"cuda_inductor": cuda_ok, "cpu_inductor": cpu_ok, "reason": reason}
    return {"cuda_inductor": _capability_cache["cuda_inductor"],
            "cpu_inductor": _capability_cache["cpu_inductor"],
            "reason": dict(_capability_cache["reason"])}


def _reset_capability_cache_for_test() -> None:
    """Test hook: forget the memoized probe so a test can force a re-probe under a
    monkeypatched environment. Mirrors autotier's own `_reset_for_test` shape."""
    global _capability_cache
    _capability_cache = None
