#!/usr/bin/env python3
"""Empirical GPU profile: resolution-scaling (launch-bound vs compute-bound) +
kernel-launch count. All timings are torch.cuda.synchronize()-bracketed."""
import sys
from pathlib import Path
_b = Path(__file__).resolve().parent
sys.path.insert(0, str(_b.parent.parent)); sys.path.insert(0, str(_b))
import torch
from run_benchmarks import (SYNTHETIC_PROGRAMS, load_example_programs, generate_bindings,
                            compile_program, run_interpreter, _infer_types, gpu_time_ms)

assert torch.cuda.is_available(), "needs CUDA"
allp = {p.name: p for p in (list(SYNTHETIC_PROGRAMS) + list(load_example_programs()))}


def prep(prog, H, W):
    b = generate_bindings(prog, 1, H, W, "cuda"); bt = _infer_types(b)
    program, tm, asg, used = compile_program(prog.code, bt)
    on = list(asg.keys()) if asg and "OUT" not in asg else None
    return program, b, tm, on, used


def run(st):
    program, b, tm, on, used = st
    return run_interpreter(program, b, tm, "cuda", on, used_builtins=used)


def gpu_ms(prog, H, W, runs=40):
    st = prep(prog, H, W)
    return gpu_time_ms(lambda: run(st), runs=runs)


def kernel_count(prog, H, W):
    st = prep(prog, H, W)
    run(st); torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run(st); torch.cuda.synchronize()
    except Exception as e:
        return None, None
    n_kernels = 0
    gpu_us = 0.0
    for e in prof.key_averages():
        dt = getattr(e, "device_type", None)
        is_cuda = (str(dt).endswith("CUDA"))
        cu = getattr(e, "cuda_time_total", 0) or getattr(e, "device_time_total", 0) or 0
        if is_cuda or cu > 0:
            n_kernels += int(getattr(e, "count", 0) or 0)
            gpu_us += cu
    return n_kernels, gpu_us / 1000.0  # us total / 1000 = ms-ish (per profiled run)


PROGS = ["passthrough", "ex_grayscale", "ex_color_grade", "math_chain",
         "ex_vignette", "ex_gaussian_blur", "ex_edge_detect", "noise_fbm", "vector_ops"]
RES = [64, 256, 512, 1024, 2048]

print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}\n")
print(f"{'program':<18}" + "".join(f"{str(r)+'^2':>9}" for r in RES) +
      f"{'2048/64':>9}{'verdict':>16}")
print("-" * 92)
for name in PROGS:
    p = allp.get(name)
    if not p:
        continue
    t = {}
    for r in RES:
        try:
            t[r] = gpu_ms(p, r, r)
        except Exception as e:
            t[r] = None
    row = "".join((f"{t[r]:>9.3f}" if t.get(r) is not None else f"{'--':>9}") for r in RES)
    ratio = (t[2048] / t[64]) if (t.get(2048) and t.get(64)) else None
    # compute-bound would scale (2048/64)^2 = 1024x; launch-bound ~1x
    verdict = ("launch-bound" if ratio and ratio < 8 else
               "mixed" if ratio and ratio < 200 else
               "compute-bound" if ratio else "?")
    print(f"{name:<18}{row}{(ratio if ratio else 0):>8.1f}x{verdict:>16}")

print("\n(compute-bound would scale ~1024x from 64^2 to 2048^2; launch-bound ~1x)\n")
print("=== kernel launches per run (1024^2) ===")
for name in PROGS:
    p = allp.get(name)
    if not p:
        continue
    try:
        nk, gpu_ms_ = kernel_count(p, 1024, 1024)
        wall = gpu_ms(p, 1024, 1024)
        print(f"  {name:<18} kernels~{nk:>4}  gpu_busy~{gpu_ms_:>7.3f}ms  wall~{wall:>7.3f}ms"
              f"  launch_overhead~{max(0, wall - gpu_ms_):>6.3f}ms")
    except Exception as e:
        print(f"  {name:<18} (profiler failed: {str(e)[:40]})")
