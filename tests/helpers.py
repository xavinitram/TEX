"""
TEX Test Suite — shared helpers and utilities.

Provides SubTestResult, compilation helpers, and test fixtures used by all test files.
Importable by both pytest and the standalone runner (run_all.py).

HOOK-4: `make_img`, `cold_engine_state` and `armed_profiler` — one named "state-isolation
kit" — now live in `TEX_Wrangle/tex_testkit.py` and are re-exported below, so an embedding
host can import them without loading this file off disk by path under a private name.
`from helpers import *` is unchanged: every name below still resolves exactly as it did
before the move.
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
from TEX_Wrangle.tex_runtime.interpreter import (_ensure_spatial, _broadcast_pair,
                                                 _collect_identifiers, _consensus_extent)
from TEX_Wrangle.tex_compiler.optimizer import optimize
from TEX_Wrangle.tex_compiler.type_checker import BINDING_HINT_TYPES
from TEX_Wrangle.tex_cache import TEXCache, parse_and_split
from TEX_Wrangle.tex_runtime.compiled import execute_compiled, _plain_execute, clear_compiled_cache
from TEX_Wrangle.tex_runtime.codegen import try_compile, _CgBreak, _CgContinue, _invoke_cg
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib, SAFE_EPSILON
from TEX_Wrangle.tex_runtime.noise import _perlin2d_fast, _grad2d_dot, _lowbias32
from TEX_Wrangle.tex_testkit import make_img, cold_engine_state, armed_profiler  # HOOK-4

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
    "make_gradient_frame", "devices",
    "cold_engine_state", "lint_sources", "armed_profiler",
    "_MAX_LOOP_ITERATIONS",
]
# `load_counts_harness` is deliberately NOT in `__all__`. HOOK-4 pins this list to the set it
# held at v0.35.0 (`tests/test_hook4_testkit.py::test_hook4_bare_star_import_yields_the_base
# _sha_set`) because `from helpers import *` is a surface an embedding host's own suite binds,
# and a name added here changes what that star yields for everyone. Its two callers import it
# by name, which needs no entry and asks nothing of anybody else.


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
    """Full pipeline: front end -> TypeCheck -> Interpret. Returns @OUT or multi-output dict.

    The front end is `tex_cache.parse_and_split` (lex + parse + the dotted-binding
    splitback), the SAME one the production seam uses, so this harness reads a swizzle's
    base wire exactly as the cook does."""
    binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}
    program = parse_and_split(code, binding_types)

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
    binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}
    program = parse_and_split(code, binding_types)

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
    """Front end + type-check only (no execution). For testing errors and diagnostics."""
    bt = dict(bindings) if bindings else {}
    bt.setdefault("OUT", TEXType.VEC4)
    prog = parse_and_split(code, bt)
    checker = TypeChecker(binding_types=bt, source=code)
    return checker.check(prog), checker


_MAX_LOOP_ITERATIONS = 1024
_STDLIB_FNS = TEXStdlib.get_functions()
_CPU_DEVICE = torch.device("cpu")


def run_both(code, bindings, B=1, H=4, W=4):
    """Run through BOTH interpreter and codegen paths. Returns (interp_result, cg_result_or_None)."""
    binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}
    program = parse_and_split(code, binding_types)
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
    # CF-6: the SAME derivation production uses. This helper kept a private first-wins loop —
    # a THIRD copy of the grid rule, in the very oracle that exists to catch the two tiers
    # disagreeing. An oracle that derives the grid its own way cannot see a grid bug.
    sp = _consensus_extent(bindings, program)

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

# `make_img` moved to `TEX_Wrangle/tex_testkit.py` (HOOK-4, part of the "state-isolation
# kit") and is imported near the top of this file for re-export; kept out of this section
# so there is exactly one implementation.

def make_latent(B=1, C=4, H=4, W=4, seed=42) -> dict:
    """Fake LATENT dict with 'samples' key in [B,C,H,W] layout."""
    torch.manual_seed(seed)
    return {"samples": torch.rand(B, C, H, W)}


def make_gradient_frame(res=64, c=4, device="cpu", scale=1.0) -> torch.Tensor:
    """A smooth [1,res,res,c] frame — NOT `make_img`'s white noise.

    Deliberately a second frame builder rather than a seed on the first. Anything that measures
    a STORAGE representation (PREC-1's fp16/uint16 error, CACHE-8's compression ratios) is
    misled by noise in both directions: white noise is incompressible, and its values are spread
    across the top binade almost everywhere, so it flatters no codec and represents no frame a
    compositor holds. `scale` pushes the range past 1.0 for the scene-linear rows.

    Lives here because three v0.33 files hand-rolled it, two of them with the arguments in a
    different order — which is exactly the shape that produces a silent wrong-device call at the
    fourth copy."""
    g = torch.linspace(0.0, 1.0, res, device=device)
    y, x = torch.meshgrid(g, g, indexing="ij")
    return (torch.stack([x, y, x * y, torch.ones_like(x)], dim=-1)[..., :c]
            .unsqueeze(0) * scale).contiguous()


def devices() -> list:
    """`["cpu"]`, plus `"cuda"` when there is a GPU. The repo's device-loop idiom, spelled once.

    A device LOOP, not a skip: a CUDA-less box runs every row it can rather than reporting a
    skip that hides which half was checked."""
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# `cold_engine_state` and `armed_profiler` moved to `TEX_Wrangle/tex_testkit.py` (HOOK-4,
# part of the "state-isolation kit") and are imported near the top of this file for
# re-export; kept out of this section so there is exactly one implementation of each.


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


def load_counts_harness():
    """Load `benchmarks/host_path_counts.py` by path, once per process.

    `benchmarks/` is not a package (it is `.comfyignore`d, like `tests/`), so there is no
    import name to use. It lives here rather than in each caller because the module owns the
    ONE implementation of the `sys.setprofile` frame filter — `path_prefixes` and
    `package_relpath` — and a second, hand-spelled copy of that filter is precisely what once
    made a counter read ZERO for every row without failing: the copy compared a `resolve()`d
    package directory against `co_filename`s that kept the junction spelling the modules were
    imported under. One implementation, loaded, cannot drift from itself.

    Loaded from the UNRESOLVED package directory on purpose, so the harness's own idea of
    where it lives matches the spelling the suite was invoked under."""
    import importlib.util
    mod = sys.modules.get("_bench2_host_path_counts")
    if mod is not None:
        return mod
    path = os.path.join(_pkg_dir, "benchmarks", "host_path_counts.py")
    spec = importlib.util.spec_from_file_location("_bench2_host_path_counts", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_bench2_host_path_counts"] = mod
    spec.loader.exec_module(mod)
    return mod
