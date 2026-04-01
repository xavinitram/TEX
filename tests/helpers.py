"""
TEX Test Suite — shared helpers and utilities.

Provides SubTestResult, compilation helpers, and test fixtures used by all test files.
Importable by both pytest and the standalone runner (run_all.py).
"""
from __future__ import annotations
import sys
import os
import traceback
import math
import re
import shutil
import tempfile
import time
import pickle
from pathlib import Path

# Add custom_nodes dir to path so package-relative imports work
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_custom_nodes_dir = os.path.dirname(_pkg_dir)
if _custom_nodes_dir not in sys.path:
    sys.path.insert(0, _custom_nodes_dir)

import torch
from TEX_Wrangle.tex_marshalling import prepare_output as _prepare_output, unwrap_latent as _unwrap_latent, infer_binding_type as _infer_binding_type, map_inferred_type as _map_inferred_type
from TEX_Wrangle.tex_compiler.lexer import Lexer, LexerError, TokenType
from TEX_Wrangle.tex_compiler.parser import Parser, ParseError
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker, TypeCheckError, TEXType, CHANNEL_MAP
from TEX_Wrangle.tex_compiler.diagnostics import TEXMultiError
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, InterpreterError
from TEX_Wrangle.tex_runtime.interpreter import _ensure_spatial, _broadcast_pair, _collect_identifiers
from TEX_Wrangle.tex_compiler.optimizer import optimize
from TEX_Wrangle.tex_compiler.type_checker import BINDING_HINT_TYPES
from TEX_Wrangle.tex_cache import TEXCache
from TEX_Wrangle.tex_runtime.compiled import execute_compiled, _plain_execute, clear_compiled_cache
from TEX_Wrangle.tex_runtime.codegen import try_compile, _CgBreak, _CgContinue
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib, SAFE_EPSILON
from TEX_Wrangle.tex_runtime.noise import _perlin2d_fast, _grad2d_dot, _lowbias32

# Export everything including underscore-prefixed names for `from helpers import *`
__all__ = [
    # Standard library
    "sys", "os", "traceback", "math", "re", "shutil", "tempfile", "time", "pickle", "Path",
    # Third-party
    "torch",
    # TEX imports (including underscore-prefixed)
    "_prepare_output", "_unwrap_latent", "_infer_binding_type", "_map_inferred_type",
    "Lexer", "LexerError", "TokenType",
    "Parser", "ParseError",
    "TypeChecker", "TypeCheckError", "TEXType", "CHANNEL_MAP",
    "TEXMultiError",
    "optimize", "BINDING_HINT_TYPES",
    "Interpreter", "InterpreterError",
    "_ensure_spatial", "_broadcast_pair", "_collect_identifiers",
    "TEXCache",
    "execute_compiled", "_plain_execute", "clear_compiled_cache",
    "try_compile", "_CgBreak", "_CgContinue",
    "TEXStdlib", "SAFE_EPSILON",
    "_perlin2d_fast", "_grad2d_dot", "_lowbias32",
    # Test helpers
    "SubTestResult", "compile_and_run", "compile_and_infer", "check_code",
    "run_both", "assert_equiv", "check_val", "make_img", "make_latent",
    "_MAX_LOOP_ITERATIONS",
]


# ── Test Result Accumulator ───────────────────────────────────────────

class SubTestResult:
    """Pass/fail accumulator for sub-tests. Works with both pytest fixture and standalone runner."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    @staticmethod
    def _safe_print(text: str):
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("ascii", errors="replace").decode("ascii"))

    def ok(self, name: str):
        self.passed += 1
        self._safe_print(f"  PASS  {name}")

    def fail(self, name: str, msg: str):
        self.failed += 1
        self.errors.append(f"{name}: {msg}")
        self._safe_print(f"  FAIL  {name}: {msg}")

    def summary(self):
        total = self.passed + self.failed
        self._safe_print(f"\n{'='*60}")
        self._safe_print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            self._safe_print(f"\nFailures:")
            for e in self.errors:
                self._safe_print(f"  - {e}")
        self._safe_print(f"{'='*60}")
        return self.failed == 0


# ── Compilation Helpers ───────────────────────────────────────────────

def compile_and_run(code: str, bindings: dict, device: str = "cpu",
                    latent_channel_count: int = 0,
                    out_type: TEXType = TEXType.VEC4) -> torch.Tensor | str | dict:
    """Full pipeline: Lex -> Parse -> TypeCheck -> Interpret. Returns @OUT or multi-output dict."""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source=code)
    program = parser.parse()

    binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}

    checker = TypeChecker(binding_types=binding_types, source=code)
    type_map = checker.check(program)
    output_names = sorted(checker.assigned_bindings.keys())

    if not output_names:
        raise InterpreterError(
            "TEX program has no outputs. Assign to @OUT or another @name."
        )

    interp = Interpreter()
    result = interp.execute(program, bindings, type_map, device=device,
                            latent_channel_count=latent_channel_count,
                            output_names=output_names)

    # Unwrap single-output for backward compat with existing tests
    if output_names == ["OUT"]:
        return result["OUT"]
    return result


def compile_and_infer(code: str, bindings: dict, device: str = "cpu",
                      latent_channel_count: int = 0) -> tuple:
    """Like compile_and_run but also returns checker.inferred_out_type."""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source=code)
    program = parser.parse()

    binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}

    checker = TypeChecker(binding_types=binding_types, source=code)
    type_map = checker.check(program)
    inferred = checker.inferred_out_type
    output_names = sorted(checker.assigned_bindings.keys())

    interp = Interpreter()
    result = interp.execute(program, bindings, type_map, device=device,
                            latent_channel_count=latent_channel_count,
                            output_names=output_names)
    return result["OUT"], inferred


def check_code(code: str, bindings: dict[str, TEXType] | None = None):
    """Lex/parse/type-check only (no execution). For testing errors and diagnostics."""
    tokens = Lexer(code).tokenize()
    prog = Parser(tokens, source=code).parse()
    bt = dict(bindings) if bindings else {}
    bt.setdefault("OUT", TEXType.VEC4)
    checker = TypeChecker(binding_types=bt, source=code)
    return checker.check(prog), checker


_MAX_LOOP_ITERATIONS = 1024
_STDLIB_FNS = TEXStdlib.get_functions()
_CPU_DEVICE = torch.device("cpu")


def run_both(code, bindings, B=1, H=4, W=4):
    """Run through BOTH interpreter and codegen paths. Returns (interp_result, cg_result_or_None)."""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source=code)
    program = parser.parse()
    binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}
    checker = TypeChecker(binding_types=binding_types, source=code)
    type_map = checker.check(program)
    output_names = sorted(checker.assigned_bindings.keys())

    # Interpreter path
    interp = Interpreter()
    interp_result = interp.execute(program, bindings, type_map, device="cpu",
                                    output_names=output_names)

    # Codegen path
    cg_fn = try_compile(program, type_map)
    if cg_fn is None:
        return interp_result, None

    stdlib_fns = _STDLIB_FNS
    dev = _CPU_DEVICE
    env = {}
    sp = None
    for v in bindings.values():
        if isinstance(v, torch.Tensor) and v.dim() >= 3:
            sp = (v.shape[0], v.shape[1], v.shape[2])
            break

    # Build builtins (matches compiled.py _codegen_exec logic)
    used = _collect_identifiers(program)
    if sp:
        B_sp, H_sp, W_sp = sp
        dtype = torch.float32
        if "ix" in used or "u" in used:
            ix = torch.arange(W_sp, dtype=dtype, device=dev).view(1, 1, W_sp)
            if "ix" in used:
                env["ix"] = ix
            if "u" in used:
                env["u"] = (ix / max(W_sp - 1, 1)).expand(B_sp, H_sp, W_sp)
        if "iy" in used or "v" in used:
            iy = torch.arange(H_sp, dtype=dtype, device=dev).view(1, H_sp, 1)
            if "iy" in used:
                env["iy"] = iy
            if "v" in used:
                env["v"] = (iy / max(H_sp - 1, 1)).expand(B_sp, H_sp, W_sp)
        if "iw" in used:
            env["iw"] = torch.tensor(float(W_sp), dtype=dtype, device=dev)
        if "ih" in used:
            env["ih"] = torch.tensor(float(H_sp), dtype=dtype, device=dev)
        if "px" in used:
            env["px"] = torch.tensor(1.0 / max(W_sp, 1), dtype=dtype, device=dev)
        if "py" in used:
            env["py"] = torch.tensor(1.0 / max(H_sp, 1), dtype=dtype, device=dev)
        if "fi" in used:
            env["fi"] = torch.arange(B_sp, dtype=dtype, device=dev).view(B_sp, 1, 1)
        if "fn" in used:
            env["fn"] = torch.tensor(float(B_sp), dtype=dtype, device=dev)
    if "PI" in used:
        env["PI"] = torch.tensor(math.pi, dtype=torch.float32, device=dev)
    if "TAU" in used:
        env["TAU"] = torch.tensor(math.tau, dtype=torch.float32, device=dev)
    if "E" in used:
        env["E"] = torch.tensor(math.e, dtype=torch.float32, device=dev)
    if "ic" in used:
        env["ic"] = torch.tensor(0.0, dtype=torch.float32, device=dev)

    # Make a copy of bindings so codegen doesn't mutate the originals
    cg_bindings = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                   for k, v in bindings.items()}

    cg_fn(env, cg_bindings, stdlib_fns, dev, sp,
          torch, _broadcast_pair, _ensure_spatial, torch.where,
          math, SAFE_EPSILON, CHANNEL_MAP, _MAX_LOOP_ITERATIONS,
          _CgBreak, _CgContinue)

    cg_result = {name: cg_bindings[name] for name in output_names}
    return interp_result, cg_result


def assert_equiv(r, name, code, bindings, B=1, H=4, W=4):
    """run_both() + assert outputs match within 1e-5. Reports to SubTestResult."""
    try:
        interp_res, cg_res = run_both(code, bindings, B, H, W)
        if cg_res is None:
            r.ok(f"codegen equiv: {name} (codegen unsupported, SKIPPED)")
            return
        for out_name in interp_res:
            interp_t = interp_res[out_name]
            cg_t = cg_res[out_name]
            if isinstance(interp_t, torch.Tensor) and isinstance(cg_t, torch.Tensor):
                max_diff = (interp_t.float() - cg_t.float()).abs().max().item()
                assert max_diff < 1e-5, f"Max diff={max_diff} for output '{out_name}'"
        r.ok(f"codegen equiv: {name}")
    except Exception as e:
        r.fail(f"codegen equiv: {name}", f"{e}")


def check_val(r, name, code, expected, bindings=None, atol=1e-3):
    """Compile, run, extract [0,0,0,0] scalar, compare to expected."""
    if bindings is None:
        torch.manual_seed(0)
        bindings = {"A": torch.rand(1, 2, 2, 3)}
    try:
        result = compile_and_run(code, bindings)
        val = result[0, 0, 0, 0].item()
        assert abs(val - expected) < atol, f"Got {val}, expected {expected}"
        r.ok(name)
    except Exception as e:
        r.fail(name, f"{e}\n{traceback.format_exc()}")


# ── Test Data Factories ───────────────────────────────────────────────

def make_img(B=1, H=8, W=8, C=3, seed=42) -> torch.Tensor:
    """Deterministic test image [B,H,W,C]."""
    torch.manual_seed(seed)
    return torch.rand(B, H, W, C)


def make_latent(B=1, C=4, H=4, W=4, seed=42) -> dict:
    """Fake LATENT dict with 'samples' key in [B,C,H,W] layout."""
    torch.manual_seed(seed)
    return {"samples": torch.rand(B, C, H, W)}
