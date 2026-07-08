#!/usr/bin/env python
"""
v0.16 lean bench — the per-item A/B measurement tool.

Times each representative program on the interpreter vs codegen (CPU) and vs the
CUDA-graph tier (GPU), sync-bracketed, WITHOUT torch.compile (which hangs on this
no-Triton box). Compiles once, times execution only. Fast enough to interleave
before/after each roadmap item.

Usage:
    python v016_bench.py --device cpu  --res 512,1024
    python v016_bench.py --device cuda --res 256,512,1024,2048
    python v016_bench.py --device cpu  --res 1024 --only vector_ops,math_chain --iters 40
"""
import argparse, math, os, statistics, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
# Locate the custom_nodes dir so `import TEX_Wrangle...` resolves, whether this
# lives in benchmarks/ (…/TEX_Wrangle/benchmarks) or an external scratch dir.
_CN = os.environ.get("TEX_CUSTOM_NODES")
if not _CN:
    _p = _HERE
    for _ in range(4):
        _p = os.path.dirname(_p)
        if os.path.isdir(os.path.join(_p, "TEX_Wrangle")):
            _CN = _p
            break
    _CN = _CN or r"G:\ComfyUI_Menu\comfyUI\custom_nodes"
if _CN not in sys.path:
    sys.path.insert(0, _CN)

import torch
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute
from TEX_Wrangle.tex_marshalling import infer_binding_type as _ibt

# Programs chosen along doc-22's decision axes. Codegen/graph WIN on the first
# three (pointwise/vector/math), LOSE on sample/noise, and passthrough is the
# 0-kernel trivial case (PF-2). Sources match run_benchmarks' synthetic set.
PROGRAMS = {
    "passthrough":    "@OUT = @A;",
    "math_chain":     "@OUT = vec4(sin(u) * cos(v) + 0.5);",
    "vector_ops": ("vec4 a = vec4(u, v, u * v, 1.0);\n"
                   "vec4 b = vec4(1.0 - u, 1.0 - v, 0.5, 1.0);\n"
                   "vec4 c = a * 0.7 + b * 0.3;\n"
                   "vec3 d = c.rgb * 2.0 - vec3(0.5, 0.5, 0.5);\n"
                   "@OUT = vec4(clamp(d.r,0.0,1.0), clamp(d.g,0.0,1.0), clamp(d.b,0.0,1.0), 1.0);"),
    "color_grade": ("vec3 c = @A.rgb;\n"
                    "c = pow(c, vec3(2.2));\n"
                    "c = c * 1.15 + vec3(0.02);\n"
                    "float l = dot(c, vec3(0.2126, 0.7152, 0.0722));\n"
                    "c = lerp(vec3(l), c, 1.3);\n"
                    "c = pow(clamp(c, 0.0, 1.0), vec3(1.0/2.2));\n"
                    "@OUT = vec4(c, 1.0);"),
    "sample_bilinear": "@OUT = sample(@A, u + sin(v * 6.28) * 0.01, v);",
    "noise_fbm":      "float n = fbm(u * 6.0, v * 6.0, 6); @OUT = vec4(n * 0.5 + 0.5);",
}


def compile_prog(code, bindings):
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    bt = {n: _ibt(v) for n, v in bindings.items()}
    ck = TypeChecker(binding_types=bt, source=code)
    tm = ck.check(prog)
    outs = sorted(ck.assigned_bindings.keys())
    return prog, tm, outs


def make_bindings(H, W, device):
    g = torch.Generator().manual_seed(42)
    img = torch.rand(1, H, W, 3, generator=g).to(device)
    ref = torch.rand(1, H, W, 4, generator=g).to(device)
    return {"A": img, "ref": ref}


def timeit(fn, iters, sync):
    for _ in range(3):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return statistics.median(ts)


def bench_one(name, code, H, W, device, iters):
    bindings = make_bindings(H, W, device)
    prog, tm, outs = compile_prog(code, bindings)
    sync = torch.cuda.synchronize if str(device).startswith("cuda") else None
    interp = Interpreter()

    def run_interp():
        interp.execute(prog, {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                               for k, v in bindings.items()},
                       tm, device=device, output_names=outs)
    ms_interp = timeit(run_interp, iters, sync)

    row = {"program": name, "res": max(H, W), "interp_ms": round(ms_interp, 3)}

    fp = f"bench_{name}_{H}x{W}"

    def run_codegen():
        _codegen_only_execute(prog, {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                     for k, v in bindings.items()},
                              tm, device, output_names=outs, fingerprint=fp)
    ms_cg = timeit(run_codegen, iters, sync)
    row["codegen_ms"] = round(ms_cg, 3)
    row["cg_speedup"] = round(ms_interp / ms_cg, 2) if ms_cg > 0 else None

    if str(device).startswith("cuda"):
        from TEX_Wrangle.tex_runtime.graphed import run_graphed, clear_graph_cache
        clear_graph_cache()
        gfp = f"benchg_{name}_{H}x{W}"
        # Prime a capture (first cook captures; subsequent replay).
        primed = run_graphed(prog, dict(bindings), tm, device, gfp, output_names=outs)
        if primed is None:
            row["graph_ms"] = None
            row["graph_speedup"] = "declined"
        else:
            def run_graph():
                run_graphed(prog, bindings, tm, device, gfp, output_names=outs)
            ms_g = timeit(run_graph, iters, sync)
            row["graph_ms"] = round(ms_g, 3)
            row["graph_speedup"] = round(ms_interp / ms_g, 2) if ms_g > 0 else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--res", default="512,1024")
    ap.add_argument("--only", default="")
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; use --device cpu"); sys.exit(1)

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(PROGRAMS)
    resolutions = [int(x) for x in args.res.split(",")]

    print(f"# v016_bench device={args.device} iters={args.iters}")
    is_cuda = args.device.startswith("cuda")
    hdr = f"{'program':16} {'res':>5} {'interp_ms':>10} {'codegen_ms':>11} {'cg_x':>6}"
    if is_cuda:
        hdr += f" {'graph_ms':>9} {'graph_x':>8}"
    print(hdr)
    for res in resolutions:
        for name in names:
            row = bench_one(name, PROGRAMS[name], res, res, args.device, args.iters)
            line = (f"{row['program']:16} {row['res']:>5} {row['interp_ms']:>10} "
                    f"{row['codegen_ms']:>11} {str(row['cg_speedup']):>6}")
            if is_cuda:
                line += f" {str(row.get('graph_ms')):>9} {str(row.get('graph_speedup')):>8}"
            print(line)


if __name__ == "__main__":
    main()
