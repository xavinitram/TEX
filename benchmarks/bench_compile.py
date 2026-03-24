#!/usr/bin/env python3
"""
TEX Compile-Mode Benchmark
===========================
Compares interpreter-only vs torch.compile execution across ALL TEX programs
(synthetic micro-benchmarks + real-world examples from examples/).

Measures 4 configurations per program:
  - compile_off  / cold   (TEX recompile + interpreter each run)
  - compile_off  / warm   (TEX compiled once, interpreter reuses AST)
  - compile_on   / cold   (torch.compile cache cleared each run)
  - compile_on   / warm   (torch.compile cached)

torch.compile measurements run in isolated subprocesses to survive
segfaults from PyTorch's TLS corruption bugs.

Usage:
    cd TEX_Wrangle
    python benchmarks/bench_compile.py
    python benchmarks/bench_compile.py --resolution 1024
    python benchmarks/bench_compile.py --resolution 2048 --runs 5
    python benchmarks/bench_compile.py --examples-only
    python benchmarks/bench_compile.py --synthetic-only
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# -- Path setup --------------------------------------------------------
_bench_dir = Path(__file__).resolve().parent
_pkg_dir = _bench_dir.parent
_custom_nodes_dir = _pkg_dir.parent
sys.path.insert(0, str(_custom_nodes_dir))

import torch

# Reuse programs, bindings, and type inference from run_benchmarks
from run_benchmarks import (
    SYNTHETIC_PROGRAMS,
    load_example_programs,
    generate_bindings,
    compile_program,
    run_interpreter,
    BenchmarkProgram,
    _infer_types,
)
from TEX_Wrangle.tex_runtime.compiled import (
    execute_compiled, clear_compiled_cache,
)

# -- Helpers -----------------------------------------------------------

WARMUP = 3
RUNS = 5


def _fingerprint(code: str, btypes: dict) -> str:
    h = hashlib.sha256(code.encode())
    for k in sorted(btypes.keys()):
        v = btypes[k]
        h.update(f"{k}:{v.value if hasattr(v, 'value') else v}".encode())
    return h.hexdigest()[:16]


# -- Measurement (in-process) -----------------------------------------

def measure_interpreter(prog: BenchmarkProgram, B, H, W, device,
                        cold=False, runs=RUNS):
    """Measure compile_off mode (plain interpreter)."""
    bindings = generate_bindings(prog, B, H, W, device)
    btypes = _infer_types(bindings)

    program, type_map, assigned, used = compile_program(prog.code, btypes)
    out_names = (list(assigned.keys())
                 if assigned and "OUT" not in assigned else None)

    # Warmup
    for _ in range(WARMUP):
        if cold:
            program, type_map, assigned, used = compile_program(prog.code, btypes)
            out_names = (list(assigned.keys())
                         if assigned and "OUT" not in assigned else None)
        run_interpreter(program, bindings, type_map, device,
                        out_names, used_builtins=used)

    times = []
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        if cold:
            program, type_map, assigned, used = compile_program(prog.code, btypes)
            out_names = (list(assigned.keys())
                         if assigned and "OUT" not in assigned else None)
        run_interpreter(program, bindings, type_map, device,
                        out_names, used_builtins=used)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def measure_compiled(prog: BenchmarkProgram, B, H, W, device,
                     cold=False, runs=RUNS):
    """Measure compile_on mode (torch.compile)."""
    bindings = generate_bindings(prog, B, H, W, device)
    btypes = _infer_types(bindings)
    fp = _fingerprint(prog.code, btypes)

    program, type_map, assigned, used = compile_program(prog.code, btypes)
    out_names = (list(assigned.keys())
                 if assigned and "OUT" not in assigned else None)

    # Warmup (one run to prime torch.compile)
    clear_compiled_cache()
    try:
        execute_compiled(program, bindings, type_map, device, fp,
                         output_names=out_names)
    except Exception:
        pass

    if not cold:
        for _ in range(WARMUP - 1):
            try:
                execute_compiled(program, bindings, type_map, device, fp,
                                 output_names=out_names)
            except Exception:
                pass

    times = []
    for _ in range(runs):
        gc.collect()
        if cold:
            clear_compiled_cache()
            program, type_map, assigned, used = compile_program(prog.code, btypes)
            out_names = (list(assigned.keys())
                         if assigned and "OUT" not in assigned else None)
        t0 = time.perf_counter()
        try:
            execute_compiled(program, bindings, type_map, device, fp,
                             output_names=out_names)
        except Exception:
            pass
        times.append((time.perf_counter() - t0) * 1000)
    return times


# -- Subprocess-isolated measurement ----------------------------------

_SUBPROCESS_SCRIPT = textwrap.dedent("""\
    import gc, hashlib, json, sys, time

    args = json.loads(sys.argv[1])
    # Paths injected by parent process
    sys.path.insert(0, args["custom_nodes_dir"])
    sys.path.insert(0, args["bench_dir"])

    import torch
    from run_benchmarks import (
        SYNTHETIC_PROGRAMS, load_example_programs, generate_bindings,
        compile_program, _infer_types,
    )
    from TEX_Wrangle.tex_runtime.compiled import (
        execute_compiled, clear_compiled_cache,
    )
    prog_name = args["prog_name"]
    B, H, W = args["B"], args["H"], args["W"]
    device = args["device"]
    cold = args["cold"]
    runs = args["runs"]
    warmup = args["warmup"]

    # Find program
    all_progs = list(SYNTHETIC_PROGRAMS) + list(load_example_programs())
    prog = next(p for p in all_progs if p.name == prog_name)

    bindings = generate_bindings(prog, B, H, W, device)
    btypes = _infer_types(bindings)

    h = hashlib.sha256(prog.code.encode())
    for k in sorted(btypes.keys()):
        v = btypes[k]
        h.update(f"{k}:{v.value if hasattr(v, 'value') else v}".encode())
    fp = h.hexdigest()[:16]

    program, type_map, assigned, used = compile_program(prog.code, btypes)
    out_names = (list(assigned.keys())
                 if assigned and "OUT" not in assigned else None)

    # Warmup
    clear_compiled_cache()
    try:
        execute_compiled(program, bindings, type_map, device, fp,
                         output_names=out_names)
    except Exception:
        pass

    if not cold:
        for _ in range(warmup - 1):
            try:
                execute_compiled(program, bindings, type_map, device, fp,
                                 output_names=out_names)
            except Exception:
                pass

    times = []
    for _ in range(runs):
        gc.collect()
        if cold:
            clear_compiled_cache()
            program, type_map, assigned, used = compile_program(prog.code, btypes)
            out_names = (list(assigned.keys())
                         if assigned and "OUT" not in assigned else None)
        t0 = time.perf_counter()
        try:
            execute_compiled(program, bindings, type_map, device, fp,
                             output_names=out_names)
        except Exception:
            pass
        times.append((time.perf_counter() - t0) * 1000)
    print(json.dumps(times))
""")


def measure_compiled_subprocess(prog: BenchmarkProgram, B, H, W, device,
                                cold=False, runs=RUNS) -> list[float] | None:
    """Run torch.compile measurement in a subprocess to survive segfaults."""
    args_json = json.dumps({
        "prog_name": prog.name,
        "B": B, "H": H, "W": W,
        "device": device,
        "cold": cold,
        "runs": runs,
        "warmup": WARMUP,
        "bench_dir": str(_bench_dir),
        "custom_nodes_dir": str(_custom_nodes_dir),
    })

    python = sys.executable
    timeout = max(120, runs * 30)  # generous timeout

    try:
        result = subprocess.run(
            [python, "-c", _SUBPROCESS_SCRIPT, args_json],
            capture_output=True, text=True,
            timeout=timeout,
            cwd=str(_bench_dir),
        )
        if result.returncode != 0:
            # Segfault (139) or other crash
            stderr_short = result.stderr.strip().split("\n")[-1][:80] if result.stderr else "unknown"
            return None
        # Parse times from stdout (last line of JSON)
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("["):
                return json.loads(line)
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# -- Formatting --------------------------------------------------------

def fmt_ms(times):
    """Format timing stats."""
    med = statistics.median(times)
    p5 = sorted(times)[max(0, len(times) // 20)]
    p95 = sorted(times)[min(len(times) - 1, len(times) * 19 // 20)]
    return f"{med:9.1f}ms  (p5={p5:.1f}, p95={p95:.1f})"


# -- Main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TEX compile-mode benchmark")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--examples-only", action="store_true",
                        help="Skip synthetic micro-benchmarks")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip real-world example programs")
    parser.add_argument("--no-subprocess", action="store_true",
                        help="Run torch.compile in-process (may crash)")
    args = parser.parse_args()

    num_runs = args.runs
    H = W = args.resolution
    B = 1
    device = args.device
    use_subprocess = not args.no_subprocess

    # Load programs
    programs: list[BenchmarkProgram] = []
    if not args.examples_only:
        programs.extend(SYNTHETIC_PROGRAMS)
    if not args.synthetic_only:
        programs.extend(load_example_programs())

    if not programs:
        print("No programs to benchmark.")
        return

    print(f"TEX Compile-Mode Benchmark")
    print(f"==========================")
    print(f"Resolution : {H}x{W}  |  Batch: {B}  |  Device: {device}")
    print(f"Warmup     : {WARMUP}  |  Runs: {num_runs}")
    print(f"Programs   : {len(programs)}")
    print(f"Subprocess : {'yes (crash-safe)' if use_subprocess else 'no'}")
    print(f"PyTorch    : {torch.__version__}")
    print()

    nw = max(len(p.name) for p in programs) + 2
    header = (f"{'Program':<{nw}} {'Mode':<12} {'Cache':<5}"
              f" {'Median':>11}    {'Range':>24}")
    sep = "-" * len(header)

    total = len(programs) * 4
    done = 0

    compiled_fn = (measure_compiled_subprocess if use_subprocess
                   else measure_compiled)

    for prog in programs:
        # Skip string-only programs at high res
        if prog.string_only and H > 256:
            done += 4
            continue

        print(f"\n-- {prog.name} ({prog.category})")
        print(header)
        print(sep)

        modes = [
            ("compile_off", "cold",  measure_interpreter, True),
            ("compile_off", "warm",  measure_interpreter, False),
            ("compile_on",  "cold",  compiled_fn,         True),
            ("compile_on",  "warm",  compiled_fn,         False),
        ]

        for mode_name, cache_name, measure_fn, is_cold in modes:
            done += 1
            try:
                t = measure_fn(prog, B, H, W, device, cold=is_cold,
                               runs=num_runs)
                if t is None:
                    print(f"{prog.name:<{nw}} {mode_name:<12} {cache_name:<5}"
                          f"    CRASH (subprocess died)"
                          f"  [{done}/{total}]")
                else:
                    print(f"{prog.name:<{nw}} {mode_name:<12} {cache_name:<5}"
                          f" {fmt_ms(t)}"
                          f"  [{done}/{total}]")
            except Exception as e:
                err = str(e)[:60]
                print(f"{prog.name:<{nw}} {mode_name:<12} {cache_name:<5}"
                      f"    ERROR: {err}"
                      f"  [{done}/{total}]")

    print(f"\n{'='*len(header)}")
    print(f"Done. {done} measurements across {len(programs)} programs.")


if __name__ == "__main__":
    main()
