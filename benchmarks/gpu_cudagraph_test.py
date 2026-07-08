#!/usr/bin/env python3
"""HYPOTHESIS: GPU is launch-bound; a CUDA graph of the codegen flat function
(captures the whole kernel sequence, replays with ~0 launch overhead, and needs
NO Triton) should be much faster than the interpreter. Measure interpreter vs
codegen-direct vs cuda-graph-replay, and verify correctness."""
import sys, math
from pathlib import Path
_b = Path(__file__).resolve().parent
sys.path.insert(0, str(_b.parent.parent)); sys.path.insert(0, str(_b))
import torch
from run_benchmarks import (SYNTHETIC_PROGRAMS, load_example_programs, generate_bindings,
                            compile_program, run_interpreter, _infer_types, gpu_time_ms)
from TEX_Wrangle.tex_runtime.codegen import try_compile as try_codegen, _CgBreak, _CgContinue
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib, SAFE_EPSILON
from TEX_Wrangle.tex_runtime.interpreter import _broadcast_pair, _ensure_spatial
from TEX_Wrangle.tex_runtime.compiled import _build_codegen_env, _MAX_LOOP_ITERATIONS
from TEX_Wrangle.tex_compiler.types import CHANNEL_MAP

assert torch.cuda.is_available()
allp = {p.name: p for p in (list(SYNTHETIC_PROGRAMS) + list(load_example_programs()))}
DEV = torch.device("cuda")
STD = TEXStdlib.get_functions()


def call_cg(cg_fn, env, bindings, sp):
    cg_fn(env, bindings, STD, DEV, sp, torch, _broadcast_pair, _ensure_spatial,
          torch.where, math, SAFE_EPSILON, CHANNEL_MAP, _MAX_LOOP_ITERATIONS,
          _CgBreak, _CgContinue)
    return bindings.get("OUT")


def bench(name, H=1024, W=1024):
    prog = allp[name]
    bindings = generate_bindings(prog, 1, H, W, "cuda")
    btypes = _infer_types(bindings)
    program, tm, asg, used = compile_program(prog.code, btypes)
    on = list(asg.keys()) if asg and "OUT" not in asg else None

    # 1) interpreter
    interp_ms = gpu_time_ms(lambda: run_interpreter(program, dict(bindings), tm, "cuda", on, used_builtins=used))
    ref = run_interpreter(program, dict(bindings), tm, "cuda", on, used_builtins=used)
    torch.cuda.synchronize()

    # 2) codegen flat fn (direct, no graph)
    cg_fn = try_codegen(program, tm)
    if cg_fn is None:
        return name, interp_ms, None, None, "no-codegen", None
    env, sp, _ = _build_codegen_env(program, bindings, DEV, 0)
    if sp is None:
        return name, interp_ms, None, None, "no-spatial", None

    def run_cg():
        return call_cg(cg_fn, dict(env), dict(bindings), sp)
    try:
        cg_ms = gpu_time_ms(run_cg)
        cg_out = run_cg(); torch.cuda.synchronize()
    except Exception as e:
        return name, interp_ms, None, None, f"cg-fail:{str(e)[:30]}", None

    # 3) CUDA graph capture of the codegen flat fn (static env + bindings)
    try:
        env_s = dict(env)
        bind_s = dict(bindings)
        # warmup on a side stream (required before capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                call_cg(cg_fn, env_s, bind_s, sp)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_static = call_cg(cg_fn, env_s, bind_s, sp)

        g.replay(); torch.cuda.synchronize()
        graph_out = out_static.clone()
        graph_ms = gpu_time_ms(lambda: g.replay())

        # correctness: graph replay vs interpreter
        diff_i = (graph_out - ref).abs().max().item()
        diff_c = (cg_out - ref).abs().max().item()
        return name, interp_ms, cg_ms, graph_ms, "ok", (diff_i, diff_c)
    except Exception as e:
        return name, interp_ms, cg_ms, None, f"graph-fail:{str(e)[:40]}", None


PROGS = ["ex_grayscale", "ex_color_grade", "ex_vignette", "math_chain",
         "ex_edge_detect", "vector_ops", "ex_brightness_contrast", "ex_hue_shift"]
print(f"GPU: {torch.cuda.get_device_name(0)}  (1024x1024)\n")
print(f"{'program':<22}{'interp':>9}{'codegen':>9}{'cudagraph':>10}{'graph_speedup':>14}  status / maxdiff")
print("-" * 92)
for name in PROGS:
    if name not in allp:
        continue
    try:
        nm, im, cm, gm, st, diffs = bench(name)
        sp = (im / gm) if gm else None
        cs = f"{cm:8.3f}" if cm else "     --"
        gs = f"{gm:9.4f}" if gm else "      --"
        spd = f"{sp:11.1f}x" if sp else "          --"
        dd = f" diff(g={diffs[0]:.1e}, c={diffs[1]:.1e})" if diffs else ""
        print(f"{nm:<22}{im:9.3f}{cs}{gs}{spd}  {st}{dd}")
    except Exception as e:
        print(f"{name:<22} ERROR: {str(e)[:60]}")
