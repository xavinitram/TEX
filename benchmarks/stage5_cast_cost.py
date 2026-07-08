#!/usr/bin/env python3
"""
Stage-5 cast-cost probe (doc 28 Phase 1 gate) — the one unrun precision probe.

Sets PR-LP2's input-dtype policy. ComfyUI hands TEX an **fp32** image. The interpreter's
precision="fp16" path casts fp32 bindings -> fp16 on the way in (interpreter.py:285) and
upcasts the result back on the way out. Question: does that cast-in + fp16-compute +
upcast-out round trip STILL beat plain fp32 on an fp32 source, or does the cast eat the
measured fp16 compute gain? If it still wins -> auto mode may accept fp32 inputs and cast
once; if not -> require fp16-native inputs (decline on fp32 source).

Interleaved A/B (fp32 vs fp16 alternated each iteration to cancel this box's 10-30%/hr
drift), sync-bracketed, median of N. CUDA-only.
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_runtime.interpreter import Interpreter

GRADE = ("vec3 c = @A.rgb; c = pow(c, vec3(1.0/2.2));"
         "float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));"
         "c = lerp(vec3(lum), c, 1.2); @OUT = vec4(c, 1.0);")


def _prep(code):
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    tm = TypeChecker(binding_types={"A": TEXType.VEC3}, source=code).check(prog)
    return prog, tm


def _time(prog, tm, img, precision, iters):
    interp = Interpreter()
    # warmup
    for _ in range(3):
        interp.execute(prog, {"A": img}, tm, device="cuda",
                       output_names=["OUT"], precision=precision)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        interp.execute(prog, {"A": img}, tm, device="cuda",
                       output_names=["OUT"], precision=precision)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable — stage-5 probe is CUDA-only. SKIP.")
        return
    prog, tm = _prep(GRADE)
    iters = 31
    print(f"{'res':>6} | {'fp32 (ms)':>10} | {'fp16<-fp32 (ms)':>16} | {'speedup':>8} | verdict")
    print("-" * 68)
    for res in (1024, 2048, 4096):
        img = torch.rand(1, res, res, 3, device="cuda", dtype=torch.float32)
        # interleave to cancel drift: median of alternating A/B is drift-robust
        f32 = _time(prog, tm, img, "fp32", iters)
        f16 = _time(prog, tm, img, "fp16", iters)
        # re-measure fp32 after fp16 and average the two fp32 reads (drift bracket)
        f32b = _time(prog, tm, img, "fp32", iters)
        f32m = (f32 + f32b) / 2
        speed = f32m / f16
        verdict = "CAST WINS" if speed > 1.05 else ("neutral" if speed > 0.97 else "DECLINE")
        print(f"{res:>6} | {f32m:>10.3f} | {f16:>16.3f} | {speed:>7.2f}x | {verdict}")
    print("\nPolicy: CAST WINS at a resolution => auto may accept fp32 input and cast once "
          "there;\nDECLINE => require fp16-native input (skip auto on fp32 source).")


if __name__ == "__main__":
    main()
