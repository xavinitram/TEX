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
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker, TypeCheckError
from TEX_Wrangle.tex_compiler.types import TEXType, CHANNEL_MAP
from TEX_Wrangle.tex_compiler.diagnostics import TEXMultiError
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, InterpreterError
from TEX_Wrangle.tex_runtime.interpreter import _ensure_spatial, _broadcast_pair, _collect_identifiers
from TEX_Wrangle.tex_compiler.optimizer import optimize
from TEX_Wrangle.tex_compiler.type_checker import BINDING_HINT_TYPES
from TEX_Wrangle.tex_cache import TEXCache
from TEX_Wrangle.tex_runtime.compiled import execute_compiled, _plain_execute, clear_compiled_cache
from TEX_Wrangle.tex_runtime.codegen import try_compile, _CgBreak, _CgContinue, _invoke_cg
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
    "cold_engine_state", "lint_sources",
    "_MAX_LOOP_ITERATIONS",
]


# ── Test Result Accumulator ───────────────────────────────────────────

class SubTestResult:
    """Pass/fail accumulator for sub-tests. Works with both pytest fixture and standalone runner."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
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

    def skip(self, name: str, reason: str):
        """Record a sub-test that could not run in this environment (not a pass,
        not a failure). Used when a check needs a resource CI doesn't have — e.g.
        a CUDA device on CPU-only torch, or the separate-repo wiki/ checkout."""
        self.skipped += 1
        self._safe_print(f"  SKIP  {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed
        self._safe_print(f"\n{'='*60}")
        skip_note = f", {self.skipped} skipped" if self.skipped else ""
        self._safe_print(f"Results: {self.passed}/{total} passed, {self.failed} failed{skip_note}")
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

    # Route through _invoke_cg (the single owner of the positional calling
    # convention) so new codegen runtime helpers don't need updating here too.
    _invoke_cg(cg_fn, env, cg_bindings, stdlib_fns, dev, sp)

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


class cold_engine_state:
    """A scratch `TEX_CACHE_DIR` + a clean warm-state/verdict table for the duration of a block,
    restored exactly on the way out.

    Four v0.31 tests needed this and three had hand-rolled it, which is a real hazard rather
    than a tidiness one: `run_all.py` runs the whole suite in ONE process, in order, so a single
    missed restore leaks a deleted cache directory into every later test. One implementation,
    one teardown.

    It is also load-bearing for correctness in at least one place: a cache probe checks memory
    and then DISK, so a test asserting "this program compiles" has to start from a cache that
    has never seen it — otherwise it passes vacuously on the second run of the suite.

        with cold_engine_state():
            ...                       # a fresh cache dir; warm state and memo start empty

    `warm=True` (the default) also clears `graphed._capturable_memo`, `profile._STATE` and
    `warm_state`'s load latch + path/tag memos; pass False when only the program cache matters.

    PROF-1's cost table belongs in that list even though it is not "warm state" in the CACHE-3
    sense: it is process-global engine state this release adds, it is keyed by fingerprints that
    a scratch cache dir invalidates, and leaving it out is exactly the leak class this fixture
    was written to end."""

    def __init__(self, *, warm: bool = True):
        self.warm = warm
        self.dir = None

    def __enter__(self):
        from TEX_Wrangle import tex_cache
        from TEX_Wrangle.tex_runtime import graphed, warm_state, profile, autotier
        self._cache_mod, self._graphed, self._ws = tex_cache, graphed, warm_state
        self._prof, self._autotier = profile, autotier
        self.dir = tempfile.mkdtemp(prefix="tex_cold_")
        self._prev_env = os.environ.get("TEX_CACHE_DIR")
        self._prev_cache = tex_cache._cache_instance
        os.environ["TEX_CACHE_DIR"] = self.dir
        tex_cache._cache_instance = None
        if self.warm:
            self._prev_memo = dict(graphed._capturable_memo)
            graphed._capturable_memo.clear()
            warm_state._reset_for_test()
            profile.reset()
            autotier._reset_for_test()
        return self

    def __exit__(self, *exc):
        if self._prev_env is None:
            os.environ.pop("TEX_CACHE_DIR", None)
        else:
            os.environ["TEX_CACHE_DIR"] = self._prev_env
        self._cache_mod._cache_instance = self._prev_cache
        if self.warm:
            self._graphed._capturable_memo.clear()
            self._graphed._capturable_memo.update(self._prev_memo)
            self._ws._reset_for_test()
            self._prof.reset()
            self._autotier._reset_for_test()
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


def lint_sources(pattern, *, allow=(), flags=0) -> list:
    """Every package `.py` (excluding tests/) whose text matches `pattern`, as `"rel:line"`.

    The shared shape behind the source canaries — PORT-1's comfy-import lint, S-1's, ENG-13's
    stray-`os.replace` sweep and SCHED-4's invariant-#7 sweep. It walks with `rglob`, which is
    the point: the two v0.31 canaries had each hardcoded a couple of globs, so `tex_compiler/`
    was unswept by one and anything added in a new subpackage by both — and "the list missed a
    file" is the exact failure the ENG-13 canary was rewritten to stop having."""
    import re as _re
    rx = _re.compile(pattern, flags)
    out = []
    root = Path(_pkg_dir)
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in allow or rel.startswith("tests/") or "/tests/" in f"/{rel}":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in rx.finditer(text):
            out.append(f"{rel}:{text[:m.start()].count(chr(10)) + 1}")
    return out
