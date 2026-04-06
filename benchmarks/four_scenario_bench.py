#!/usr/bin/env python3
"""
TEX 4-Scenario Benchmark
========================
Measures cook times across the 4 scenarios:
  1. Compile OFF, Cold Start  — fresh compile + interpreter
  2. Compile OFF, Warm Start  — cached AST + interpreter
  3. Compile ON,  Cold Start  — codegen + first run (no torch.compile)
  4. Compile ON,  Warm Start  — codegen + cached run

Usage:
    python benchmarks/four_scenario_bench.py [--size 512] [--runs 10]
    python benchmarks/four_scenario_bench.py --examples-only --size 512 --runs 10
    python benchmarks/four_scenario_bench.py --synthetic-only
    python benchmarks/four_scenario_bench.py --save results/v0.10.0_baseline.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

_bench_dir = Path(__file__).resolve().parent
_pkg_dir = _bench_dir.parent
sys.path.insert(0, str(_pkg_dir.parent))

import torch
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker, TEXType, CHANNEL_MAP
from TEX_Wrangle.tex_compiler.optimizer import optimize
from TEX_Wrangle.tex_runtime.interpreter import (
    Interpreter, _collect_identifiers, _ensure_spatial, _broadcast_pair,
)
from TEX_Wrangle.tex_runtime.codegen import try_compile as try_codegen, _CgBreak, _CgContinue
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib, SAFE_EPSILON

EXAMPLES_DIR = _pkg_dir / "examples"

# Derive category mapping from the canonical source in __init__.py
# (e.g., "Color/Auto Levels" → "Color") to avoid maintaining a second copy.
from TEX_Wrangle import _EXAMPLE_CATEGORIES as _INIT_CATEGORIES
_EXAMPLE_CATEGORIES = {stem: val.split("/")[0] for stem, val in _INIT_CATEGORIES.items()}

# ── Synthetic core programs ──────────────────────────────────────────────────

SYNTHETIC_PROGRAMS = [
    ("passthrough", "Synthetic", "@OUT = @A;"),
    ("math_chain", "Synthetic",
     "float x = sin(u * 6.28) * cos(v * 6.28); @OUT = vec4(x, x, x, 1.0);"),
    ("for_100", "Synthetic", """\
float acc = 0.0;
for (int i = 0; i < 100; i++) {
    acc = acc + sin(u * float(i) * 0.1) * 0.01;
}
@OUT = vec4(acc, acc, acc, 1.0);"""),
    ("noise_fbm", "Synthetic", """\
float freq = 4.0;
float n = fbm(u * freq, v * freq, 6);
@OUT = vec4(n, n, n, 1.0);"""),
    ("fetch_kernel", "Synthetic", """\
vec4 s = vec4(0.0);
for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
        s = s + fetch(@A, ix + float(dx), iy + float(dy));
    }
}
@OUT = s / 9.0;"""),
    ("sample_bilinear", "Synthetic", """\
float su = u + sin(v * 6.28) * 0.02;
float sv = v + cos(u * 6.28) * 0.02;
@OUT = sample(@A, su, sv);"""),
    ("color_grade", "Synthetic", """\
vec3 c = @A.rgb;
c = pow(c, vec3(1.0/2.2));
float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
c = lerp(vec3(lum), c, 1.2);
@OUT = vec4(c, @A.a);"""),
    ("vector_ops", "Synthetic", """\
vec3 a = vec3(u, v, 0.5);
vec3 b = vec3(1.0 - u, 1.0 - v, 0.5);
float d = distance(a, b);
vec3 n = normalize(a - b);
@OUT = vec4(n * d, 1.0);"""),
]


# ── Binding detection (adapted from run_benchmarks.py) ───────────────────────

def _detect_bindings(code: str) -> dict:
    """Detect what input bindings a TEX program references."""
    return {
        "needs_image": bool(re.search(r"@(?:image|A)\b", code)),
        "needs_ref": bool(re.search(r"@ref\b", code)),
        "needs_frames": bool(re.search(r"@frames\b", code)),
        "needs_latent_ab": bool(
            re.search(r"@latent_a\b", code) and re.search(r"@latent_b\b", code)
        ),
        "needs_latent": (
            bool(re.search(r"@latent\b", code))
            and not bool(re.search(r"@latent_[ab]\b", code))
        ),
        "needs_base_overlay_blend": bool(re.search(r"@base\b", code)),
        "needs_text": bool(re.search(r"@text\b", code)),
        "needs_mask": bool(re.search(r"@mask\b", code)),
    }



def _extract_param_defaults(code: str) -> dict:
    """Extract $param declarations and their default values."""
    defaults = {}
    for m in re.finditer(r"([fisbc]|v[234])\$(\w+)\s*=\s*([^;]+);", code):
        hint, name, raw = m.groups()
        raw = raw.strip().strip("\"'")
        try:
            if hint == "f":
                defaults[name] = float(raw)
            elif hint == "i":
                defaults[name] = int(raw)
            elif hint == "b":
                defaults[name] = 1 if raw in ("1", "true") else 0
            elif hint == "c":
                defaults[name] = raw  # hex string like "#FF8800"
            elif hint.startswith("v"):
                # vec2/vec3/vec4 — extract comma values from vec3(1,2,3) or "1,2,3"
                vm = re.match(r"vec\d\s*\(([^)]+)\)", raw)
                defaults[name] = vm.group(1).strip() if vm else raw
            else:
                defaults[name] = raw
        except (ValueError, AttributeError):
            defaults[name] = 0.5
    return defaults


def _is_string_only(code: str) -> bool:
    """Check if a program is string-only (no spatial output)."""
    det = _detect_bindings(code)
    has_spatial = any(v for k, v in det.items() if k != "needs_text")
    return not has_spatial and bool(re.search(r"string\b", code))


def _has_multi_output(code: str) -> bool:
    """Check if program uses named outputs (not @OUT)."""
    return (bool(re.search(r"@\w+\s*=", code))
            and not bool(re.search(r"@OUT\s*=", code)))


def _uses_spatial_builtins(code: str) -> bool:
    """Check if a program uses spatial built-in variables."""
    return bool(re.search(r"\b(?:u|v|ix|iy|iw|ih)\b", code))


def generate_bindings(code: str, B: int, H: int, W: int,
                      device: str = "cpu") -> dict:
    """Create synthetic input tensors for a program.

    Uses 2-pass type checking (same as test harness) to discover all referenced
    bindings and their types, then creates appropriate dummy tensors.
    """
    from TEX_Wrangle.tex_compiler.type_checker import BINDING_HINT_TYPES

    b = {}

    # Pass 1: Parse and type-check to discover all referenced bindings
    program = Parser(Lexer(code).tokenize(), source=code).parse()
    checker = TypeChecker(binding_types={}, source=code)
    checker.check(program)
    output_names = set(checker.assigned_bindings.keys())
    param_names = set(checker.param_declarations.keys())
    input_names = checker.referenced_bindings - output_names - param_names

    # Detect type hints and vec4 context from AST
    has_vec4 = "vec4" in code
    default_type = TEXType.VEC4 if has_vec4 else TEXType.VEC3
    C = 4 if has_vec4 else 3

    # Collect type hints from binding refs in the AST
    from TEX_Wrangle.tex_compiler.ast_nodes import BindingRef
    hints = {}
    stack = list(program.statements)
    while stack:
        node = stack.pop()
        if isinstance(node, BindingRef) and node.type_hint and node.kind == "wire":
            hints[node.name] = BINDING_HINT_TYPES.get(node.type_hint, default_type)
        for attr in dir(node):
            if attr.startswith("_"):
                continue
            val = getattr(node, attr, None)
            if isinstance(val, list):
                stack.extend(v for v in val if hasattr(v, "loc"))
            elif hasattr(val, "loc"):
                stack.append(val)

    # Create dummy bindings for each referenced input
    for bname in input_names:
        bt = hints.get(bname, default_type)
        if bt == TEXType.STRING:
            b[bname] = "hello"
        elif bt == TEXType.FLOAT:
            b[bname] = torch.rand(B, H, W, dtype=torch.float32, device=device)
        elif bt == TEXType.INT:
            b[bname] = torch.randint(0, 10, (B, H, W), device=device).float()
        elif bt == TEXType.VEC2:
            b[bname] = torch.rand(B, H, W, 2, dtype=torch.float32, device=device)
        elif bt == TEXType.VEC3:
            b[bname] = torch.rand(B, H, W, 3, dtype=torch.float32, device=device)
        else:
            b[bname] = torch.rand(B, H, W, C, dtype=torch.float32, device=device)

    # Add param defaults
    for pname, pval in _extract_param_defaults(code).items():
        b[pname] = pval

    # Also add param defaults from type checker
    for pname, pinfo in checker.param_declarations.items():
        if pname not in b:
            hint = pinfo.get("type", TEXType.FLOAT)
            if hint == TEXType.STRING:
                b[pname] = ""
            elif hint == TEXType.INT:
                b[pname] = 0
            elif hint == TEXType.FLOAT:
                b[pname] = 0.5
            else:
                b[pname] = 0.5

    # Ensure spatial context for procedural programs
    has_spatial = any(isinstance(v, torch.Tensor) and v.dim() >= 3
                      for v in b.values())
    if not has_spatial and _uses_spatial_builtins(code):
        b["ref"] = torch.rand(B, H, W, C, dtype=torch.float32, device=device)
    return b


def infer_types(bindings: dict) -> dict[str, TEXType]:
    """Infer TEXType for each binding value."""
    types = {}
    for name, val in bindings.items():
        if isinstance(val, str):
            types[name] = TEXType.STRING
        elif isinstance(val, torch.Tensor):
            if val.dim() == 4:
                c = val.shape[-1]
                types[name] = (TEXType.VEC4 if c == 4
                               else TEXType.VEC3 if c == 3
                               else TEXType.FLOAT)
            else:
                types[name] = TEXType.FLOAT
        elif isinstance(val, int):
            types[name] = TEXType.INT
        else:
            types[name] = TEXType.FLOAT
    return types


# ── Example loader ───────────────────────────────────────────────────────────

def load_examples() -> list[tuple[str, str, str]]:
    """Load all .tex files from examples/ as (name, category, code) tuples."""
    programs = []
    if not EXAMPLES_DIR.exists():
        print(f"  Warning: {EXAMPLES_DIR} not found")
        return programs
    for f in sorted(EXAMPLES_DIR.glob("*.tex")):
        code = f.read_text(encoding="utf-8")
        stem = f.stem
        category = _EXAMPLE_CATEGORIES.get(stem, "Other")
        programs.append((f"ex_{stem}", category, code))
    return programs


# ── Compilation & execution ──────────────────────────────────────────────────

def compile_program(code: str, btypes: dict):
    tokens = Lexer(code).tokenize()
    program = Parser(tokens, source=code).parse()
    checker = TypeChecker(binding_types=btypes, source=code)
    type_map = checker.check(program)
    program = optimize(program)
    used = _collect_identifiers(program)
    assigned = checker.assigned_bindings
    return program, type_map, assigned, used


_interp = Interpreter()
_stdlib_fns = TEXStdlib.get_functions()


def _get_output_names(assigned: dict) -> list[str] | None:
    """Return output names list for multi-output programs, or None for @OUT."""
    if assigned and "OUT" not in assigned:
        return list(assigned.keys())
    return None


def run_interpreter(prog, bindings, tm, used, code, output_names=None):
    """Execute via tree-walking interpreter."""
    return _interp.execute(
        prog, bindings, tm, device='cpu',
        used_builtins=used, source=code,
        output_names=output_names,
    )


def make_codegen_runner(cg_fn, bindings, used, sp):
    """Create a closure that runs the codegen function."""
    dev = torch.device('cpu')
    B, H, W = sp
    dtype = torch.float32

    # Mirror interpreter._create_builtins — keep in sync if builtins change
    base_env = {}
    if "ix" in used or "u" in used:
        ix = torch.arange(W, dtype=dtype, device=dev).view(1, 1, W)
        if "ix" in used: base_env["ix"] = ix
        if "u" in used: base_env["u"] = (ix / max(W - 1, 1)).expand(B, H, W)
    if "iy" in used or "v" in used:
        iy = torch.arange(H, dtype=dtype, device=dev).view(1, H, 1)
        if "iy" in used: base_env["iy"] = iy
        if "v" in used: base_env["v"] = (iy / max(H - 1, 1)).expand(B, H, W)
    if "iw" in used: base_env["iw"] = torch.scalar_tensor(float(W), dtype=dtype, device=dev)
    if "ih" in used: base_env["ih"] = torch.scalar_tensor(float(H), dtype=dtype, device=dev)
    if "px" in used: base_env["px"] = torch.scalar_tensor(1.0 / max(W, 1), dtype=dtype, device=dev)
    if "py" in used: base_env["py"] = torch.scalar_tensor(1.0 / max(H, 1), dtype=dtype, device=dev)
    if "fi" in used: base_env["fi"] = torch.arange(B, dtype=dtype, device=dev).view(B, 1, 1)
    if "fn" in used: base_env["fn"] = torch.scalar_tensor(float(B), dtype=dtype, device=dev)
    if "PI" in used: base_env["PI"] = torch.scalar_tensor(math.pi, dtype=dtype, device=dev)
    if "TAU" in used: base_env["TAU"] = torch.scalar_tensor(math.tau, dtype=dtype, device=dev)
    if "E" in used: base_env["E"] = torch.scalar_tensor(math.e, dtype=dtype, device=dev)
    if "ic" in used: base_env["ic"] = torch.scalar_tensor(4.0, dtype=dtype, device=dev)

    def runner():
        env = dict(base_env)
        local_bindings = dict(bindings)
        cg_fn(env, local_bindings, _stdlib_fns, dev, sp,
              torch, _broadcast_pair, _ensure_spatial, torch.where,
              math, SAFE_EPSILON, CHANNEL_MAP, 1024,
              _CgBreak, _CgContinue)
        return local_bindings

    return runner


# ── Scenario runner ──────────────────────────────────────────────────────────

def run_scenario(name: str, code: str, size: int, runs: int,
                 warmup: int = 3) -> dict:
    """Run a single program under all 4 scenarios. Returns timing dict."""
    B = 1
    H = W = size

    # Generate bindings and infer types
    bindings = generate_bindings(code, B, H, W)
    btypes = infer_types(bindings)

    # Check if string-only (skip spatial scenarios)
    string_only = _is_string_only(code)

    results = {}

    # ── Scenario 1: Compile OFF, Cold Start ──
    try:
        prog, tm, assigned, used = compile_program(code, btypes)
        output_names = _get_output_names(assigned)
    except Exception as e:
        return {"error": str(e)}

    try:
        for _ in range(warmup):
            prog_w, tm_w, assigned_w, used_w = compile_program(code, btypes)
            run_interpreter(prog_w, dict(bindings), tm_w, used_w, code,
                            _get_output_names(assigned_w))

        times = []
        for _ in range(runs):
            gc.collect()
            t0 = time.perf_counter()
            prog_c, tm_c, assigned_c, used_c = compile_program(code, btypes)
            run_interpreter(prog_c, dict(bindings), tm_c, used_c, code,
                            _get_output_names(assigned_c))
            times.append((time.perf_counter() - t0) * 1000)
        results["compile_off_cold"] = statistics.median(times)
    except Exception as e:
        return {"error": f"OFF Cold: {e}"}

    # ── Scenario 2: Compile OFF, Warm Start ──
    try:
        for _ in range(warmup):
            run_interpreter(prog, dict(bindings), tm, used, code, output_names)

        times = []
        for _ in range(runs):
            gc.collect()
            t0 = time.perf_counter()
            run_interpreter(prog, dict(bindings), tm, used, code, output_names)
            times.append((time.perf_counter() - t0) * 1000)
        results["compile_off_warm"] = statistics.median(times)
    except Exception as e:
        return {"error": f"OFF Warm: {e}"}

    # ── Scenario 3 & 4: Compile ON (codegen) ──
    try:
        cg_fn = try_codegen(prog, tm)
    except Exception:
        cg_fn = None

    if cg_fn is not None and not string_only:
        # Determine spatial shape from bindings
        sp = None
        for v in bindings.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 3:
                sp = (v.shape[0], v.shape[1], v.shape[2])
                break
        if sp is None:
            sp = (B, H, W)

        try:
            runner = make_codegen_runner(cg_fn, bindings, used, sp)
            # Verify codegen actually runs
            runner()

            # Scenario 3: Compile ON, Cold Start
            for _ in range(warmup):
                cg_fn_fresh = try_codegen(prog, tm)
                make_codegen_runner(cg_fn_fresh, bindings, used, sp)()

            times = []
            for _ in range(runs):
                gc.collect()
                t0 = time.perf_counter()
                cg_fn = try_codegen(prog, tm)
                make_codegen_runner(cg_fn, bindings, used, sp)()
                times.append((time.perf_counter() - t0) * 1000)
            results["compile_on_cold"] = statistics.median(times)

            # Scenario 4: Compile ON, Warm Start
            cg_fn = try_codegen(prog, tm)
            runner = make_codegen_runner(cg_fn, bindings, used, sp)
            for _ in range(warmup):
                runner()

            times = []
            for _ in range(runs):
                gc.collect()
                t0 = time.perf_counter()
                runner()
                times.append((time.perf_counter() - t0) * 1000)
            results["compile_on_warm"] = statistics.median(times)
        except Exception as e:
            import traceback
            results["compile_on_cold"] = None
            results["compile_on_warm"] = None
            results["codegen_error"] = f"{e}\n{traceback.format_exc()}"
    else:
        results["compile_on_cold"] = None
        results["compile_on_warm"] = None

    return results


# ── Output formatting ────────────────────────────────────────────────────────

def format_ms(val):
    if val is None:
        return "N/A"
    return f"{val:.2f}ms"


def print_program_row(name: str, res: dict, name_width: int = 28):
    if "error" in res:
        print(f"  {name:<{name_width}} {'ERROR':>10}  {res['error'][:50]}")
        return

    off_cold = format_ms(res.get("compile_off_cold"))
    off_warm = format_ms(res.get("compile_off_warm"))
    on_cold = format_ms(res.get("compile_on_cold"))
    on_warm = format_ms(res.get("compile_on_warm"))
    has_cg = "yes" if res.get("compile_on_cold") is not None else "no"
    cg_err = res.get("codegen_error", "")

    line = f"  {name:<{name_width}} {off_cold:>10} {off_warm:>10} {on_cold:>10} {on_warm:>10} {has_cg:>8}"
    if cg_err and has_cg == "no":
        # Show first line of codegen error
        first_line = cg_err.split('\n')[0][:40]
        line += f"  {first_line}"
    print(line)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="TEX 4-Scenario Benchmark")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--synthetic-only", action="store_true",
                    help="Only run the 8 synthetic programs")
    ap.add_argument("--examples-only", action="store_true",
                    help="Only run production example programs")
    ap.add_argument("--category", type=str, default=None,
                    help="Only run examples from this category")
    ap.add_argument("--save", type=str, default=None,
                    help="Save results to JSON file")
    args = ap.parse_args()

    size = args.size
    runs = args.runs
    warmup = args.warmup
    NW = 28  # name column width

    # Build program list: (name, category, code)
    programs: list[tuple[str, str, str]] = []

    if not args.examples_only:
        programs.extend(SYNTHETIC_PROGRAMS)

    if not args.synthetic_only:
        examples = load_examples()
        if args.category:
            examples = [(n, c, code) for n, c, code in examples
                        if c.lower() == args.category.lower()]
        programs.extend(examples)

    if not programs:
        print("No programs to benchmark.")
        return

    print(f"\nTEX 4-Scenario Benchmark — {size}x{size}, {runs} runs, {warmup} warmup")
    print(f"Programs: {len(programs)} total")
    print(f"{'='*96}")

    # Group by category (dict preserves insertion order in Python 3.7+)
    categories: dict[str, list] = {}
    for name, category, code in programs:
        categories.setdefault(category, []).append((name, code))

    grand_totals = {
        "compile_off_cold": 0.0, "compile_off_warm": 0.0,
        "compile_on_cold": 0.0, "compile_on_warm": 0.0,
    }
    grand_codegen = 0
    grand_total = 0
    grand_errors = 0
    all_results = {}

    for category, progs in categories.items():
        cat_totals = {
            "compile_off_cold": 0.0, "compile_off_warm": 0.0,
            "compile_on_cold": 0.0, "compile_on_warm": 0.0,
        }
        cat_codegen = 0
        cat_errors = 0

        print(f"\n-- {category} ({len(progs)} programs) --")
        print(f"  {'Program':<{NW}} {'OFF Cold':>10} {'OFF Warm':>10} {'ON Cold':>10} {'ON Warm':>10} {'Codegen':>8}")
        print(f"  {'-'*(NW + 58)}")

        for name, code in progs:
            gc.collect()
            try:
                res = run_scenario(name, code, size, runs, warmup)
            except (RuntimeError, MemoryError) as e:
                res = {"error": f"OOM: {e}"}
            all_results[name] = res

            print_program_row(name, res, NW)

            if "error" in res:
                cat_errors += 1
                continue

            for key in cat_totals:
                val = res.get(key)
                if val is not None:
                    cat_totals[key] += val
            if res.get("compile_on_cold") is not None:
                cat_codegen += 1

        # Category subtotal
        n_ok = len(progs) - cat_errors
        print(f"  {'-'*(NW + 58)}")
        print(f"  {'SUBTOTAL':<{NW}} "
              f"{format_ms(cat_totals['compile_off_cold']):>10} "
              f"{format_ms(cat_totals['compile_off_warm']):>10} "
              f"{format_ms(cat_totals['compile_on_cold']):>10} "
              f"{format_ms(cat_totals['compile_on_warm']):>10} "
              f"{cat_codegen}/{n_ok:>5}")

        for key in grand_totals:
            grand_totals[key] += cat_totals[key]
        grand_codegen += cat_codegen
        grand_total += len(progs)
        grand_errors += cat_errors

    # Grand total
    n_ok = grand_total - grand_errors
    print(f"\n{'='*96}")
    print(f"  {'GRAND TOTAL':<{NW}} "
          f"{format_ms(grand_totals['compile_off_cold']):>10} "
          f"{format_ms(grand_totals['compile_off_warm']):>10} "
          f"{format_ms(grand_totals['compile_on_cold']):>10} "
          f"{format_ms(grand_totals['compile_on_warm']):>10} "
          f"{grand_codegen}/{n_ok:>5}")
    if grand_errors:
        print(f"  ({grand_errors} programs had errors)")
    print()

    # Save results
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "meta": {
                "size": size,
                "runs": runs,
                "warmup": warmup,
                "torch_version": torch.__version__,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "num_programs": grand_total,
                "num_errors": grand_errors,
            },
            "totals": grand_totals,
            "programs": all_results,
        }
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
