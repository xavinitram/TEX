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
"""
from __future__ import annotations

import argparse
import gc
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
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker, TEXType
from TEX_Wrangle.tex_compiler.optimizer import optimize
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, _collect_identifiers
from TEX_Wrangle.tex_runtime.codegen import try_compile as try_codegen

# Representative programs covering different patterns
PROGRAMS = {
    "passthrough": ("vec4", "A", "@OUT = @A;"),
    "math_chain": ("vec4", "A", "float x = sin(u * 6.28) * cos(v * 6.28); @OUT = vec4(x, x, x, 1.0);"),
    "for_100": ("vec4", "A", """
float acc = 0.0;
for (int i = 0; i < 100; i++) {
    acc = acc + sin(u * float(i) * 0.1) * 0.01;
}
@OUT = vec4(acc, acc, acc, 1.0);
"""),
    "noise_fbm": ("vec4", "A", """
float freq = 4.0;
float n = fbm(u * freq, v * freq, 6);
@OUT = vec4(n, n, n, 1.0);
"""),
    "fetch_kernel": ("vec4", "A", """
vec4 s = vec4(0.0);
for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
        s = s + fetch(@A, ix + float(dx), iy + float(dy));
    }
}
@OUT = s / 9.0;
"""),
    "sample_bilinear": ("vec4", "A", """
float su = u + sin(v * 6.28) * 0.02;
float sv = v + cos(u * 6.28) * 0.02;
@OUT = sample(@A, su, sv);
"""),
    "color_grade": ("vec4", "A", """
vec3 c = @A.rgb;
c = pow(c, vec3(1.0/2.2));
float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
c = lerp(vec3(lum), c, 1.2);
@OUT = vec4(c, @A.a);
"""),
    "vector_ops": ("vec4", "A", """
vec3 a = vec3(u, v, 0.5);
vec3 b = vec3(1.0 - u, 1.0 - v, 0.5);
float d = distance(a, b);
vec3 n = normalize(a - b);
@OUT = vec4(n * d, 1.0);
"""),
}


def compile_program(code: str, btypes: dict):
    tokens = Lexer(code).tokenize()
    program = Parser(tokens, source=code).parse()
    checker = TypeChecker(binding_types=btypes, source=code)
    type_map = checker.check(program)
    program = optimize(program)
    used = _collect_identifiers(program)
    return program, type_map, checker.assigned_bindings, used


def run_scenario(name: str, code: str, btype: str, bind_name: str,
                 image: torch.Tensor, size: int, runs: int, warmup: int = 3):
    """Run a single program under all 4 scenarios."""
    btypes = {bind_name: TEXType.VEC4 if btype == "vec4" else TEXType.FLOAT}
    bindings = {bind_name: image}

    results = {}
    interp = Interpreter()

    # Scenario 1: Compile OFF, Cold Start (compile + interpret each time)
    for _ in range(warmup):
        prog, tm, assigned, used = compile_program(code, btypes)
        interp.execute(prog, bindings, tm, device='cpu', used_builtins=used, source=code)

    times = []
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        prog, tm, assigned, used = compile_program(code, btypes)
        interp.execute(prog, bindings, tm, device='cpu', used_builtins=used, source=code)
        times.append((time.perf_counter() - t0) * 1000)
    results["compile_off_cold"] = statistics.median(times)

    # Scenario 2: Compile OFF, Warm Start (reuse AST)
    prog, tm, assigned, used = compile_program(code, btypes)
    for _ in range(warmup):
        interp.execute(prog, bindings, tm, device='cpu', used_builtins=used, source=code)

    times = []
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        interp.execute(prog, bindings, tm, device='cpu', used_builtins=used, source=code)
        times.append((time.perf_counter() - t0) * 1000)
    results["compile_off_warm"] = statistics.median(times)

    # Scenario 3 & 4: Compile ON (codegen)
    try:
        cg_fn = try_codegen(prog, tm)
    except Exception:
        cg_fn = None

    if cg_fn is not None:
        from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib, SAFE_EPSILON
        from TEX_Wrangle.tex_runtime.interpreter import _ensure_spatial, _broadcast_pair
        from TEX_Wrangle.tex_runtime.codegen import _CgBreak, _CgContinue
        from TEX_Wrangle.tex_compiler.type_checker import CHANNEL_MAP
        import math

        stdlib_fns = TEXStdlib.get_functions()

        def run_codegen():
            dev = torch.device('cpu')
            env = {}
            sp = (image.shape[0], image.shape[1], image.shape[2])
            B, H, W = sp
            dtype = torch.float32
            if "ix" in used or "u" in used:
                ix = torch.arange(W, dtype=dtype, device=dev).view(1, 1, W)
                if "ix" in used: env["ix"] = ix
                if "u" in used: env["u"] = (ix / max(W - 1, 1)).expand(B, H, W)
            if "iy" in used or "v" in used:
                iy = torch.arange(H, dtype=dtype, device=dev).view(1, H, 1)
                if "iy" in used: env["iy"] = iy
                if "v" in used: env["v"] = (iy / max(H - 1, 1)).expand(B, H, W)
            if "iw" in used: env["iw"] = torch.tensor(float(W), dtype=dtype, device=dev)
            if "ih" in used: env["ih"] = torch.tensor(float(H), dtype=dtype, device=dev)
            if "PI" in used: env["PI"] = torch.tensor(math.pi, dtype=dtype, device=dev)
            if "E" in used: env["E"] = torch.tensor(math.e, dtype=dtype, device=dev)

            local_bindings = dict(bindings)
            cg_fn(env, local_bindings, stdlib_fns, dev, sp,
                  torch, _broadcast_pair, _ensure_spatial, torch.where,
                  math, SAFE_EPSILON, CHANNEL_MAP, 1024,
                  _CgBreak, _CgContinue)
            return local_bindings.get("OUT")

        # Scenario 3: Compile ON, Cold Start (codegen + first run)
        for _ in range(warmup):
            # Re-codegen each time to simulate cold
            cg_fn_fresh = try_codegen(prog, tm)
            run_codegen()

        times = []
        for _ in range(runs):
            gc.collect()
            t0 = time.perf_counter()
            cg_fn = try_codegen(prog, tm)
            run_codegen()
            times.append((time.perf_counter() - t0) * 1000)
        results["compile_on_cold"] = statistics.median(times)

        # Scenario 4: Compile ON, Warm Start (reuse codegen fn)
        cg_fn = try_codegen(prog, tm)
        for _ in range(warmup):
            run_codegen()

        times = []
        for _ in range(runs):
            gc.collect()
            t0 = time.perf_counter()
            run_codegen()
            times.append((time.perf_counter() - t0) * 1000)
        results["compile_on_warm"] = statistics.median(times)
    else:
        results["compile_on_cold"] = None
        results["compile_on_warm"] = None

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--runs", type=int, default=15)
    args = ap.parse_args()

    size = args.size
    runs = args.runs
    image = torch.rand(1, size, size, 4)

    print(f"TEX 4-Scenario Benchmark — {size}x{size}, {runs} runs per scenario")
    print(f"{'='*90}")
    print(f"{'Program':<20} {'OFF Cold':>10} {'OFF Warm':>10} {'ON Cold':>10} {'ON Warm':>10} {'Codegen':>8}")
    print(f"{'-'*90}")

    totals = {"compile_off_cold": 0, "compile_off_warm": 0,
              "compile_on_cold": 0, "compile_on_warm": 0}
    codegen_count = 0

    for name, (btype, bind, code) in PROGRAMS.items():
        res = run_scenario(name, code, btype, bind, image, size, runs)

        off_cold = f"{res['compile_off_cold']:.2f}ms"
        off_warm = f"{res['compile_off_warm']:.2f}ms"
        on_cold = f"{res['compile_on_cold']:.2f}ms" if res['compile_on_cold'] is not None else "N/A"
        on_warm = f"{res['compile_on_warm']:.2f}ms" if res['compile_on_warm'] is not None else "N/A"
        has_cg = "yes" if res['compile_on_cold'] is not None else "no"

        print(f"{name:<20} {off_cold:>10} {off_warm:>10} {on_cold:>10} {on_warm:>10} {has_cg:>8}")

        totals["compile_off_cold"] += res["compile_off_cold"]
        totals["compile_off_warm"] += res["compile_off_warm"]
        if res["compile_on_cold"] is not None:
            totals["compile_on_cold"] += res["compile_on_cold"]
            totals["compile_on_warm"] += res["compile_on_warm"]
            codegen_count += 1

    print(f"{'-'*90}")
    print(f"{'TOTAL':<20} {totals['compile_off_cold']:.2f}ms {totals['compile_off_warm']:.2f}ms"
          f"  {totals['compile_on_cold']:.2f}ms  {totals['compile_on_warm']:.2f}ms"
          f"  {codegen_count}/{len(PROGRAMS)}")


if __name__ == "__main__":
    main()
