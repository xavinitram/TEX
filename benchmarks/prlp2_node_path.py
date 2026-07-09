#!/usr/bin/env python3
"""
PR-LP2 node-path A/B (audit B1 / H7) — substantiates the `precision="auto"` speedup on the
path users actually invoke: `TEXWrangleNode.execute()`, NOT `Interpreter.execute` (the
CHANGELOG's original number was measured off the latter and did not hold on the node path).

Interleaved fp32/auto/fp16 through the node, sync-bracketed, median >=35, with a second
fp32 read as a drift bracket. CUDA-only.

Gate (must hold after B1): auto >= 1.3x @2048^2 AND auto >= 1.0x @1024^2.
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from TEX_Wrangle.tex_node import TEXWrangleNode as N

GRADE = ("vec3 c = @A.rgb; c = pow(c, vec3(1.0/2.2));"
         "float l = dot(c, vec3(0.2126, 0.7152, 0.0722));"
         "c = lerp(vec3(l), c, 1.2); @OUT = vec4(c, 1.0);")


def _time(res, precision, iters=35):
    img = torch.rand(1, res, res, 3, device="cuda")
    for _ in range(4):  # warm up (incl. the first-cook finiteness check for auto)
        N.execute(code=GRADE, A=img, device="cuda", precision=precision)
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        N.execute(code=GRADE, A=img, device="cuda", precision=precision)
        torch.cuda.synchronize()
        s.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(s)


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable — node-path A/B is CUDA-only. SKIP.")
        return
    print(f"{'res':>6} | {'fp32':>8} | {'auto':>8} {'x':>7} | {'fp16':>8} {'x':>7} | gate")
    print("-" * 60)
    ok = True
    for res in (1024, 2048):
        f32 = _time(res, "fp32"); auto = _time(res, "auto"); f16 = _time(res, "fp16")
        f32b = _time(res, "fp32"); f32m = (f32 + f32b) / 2   # drift bracket
        au_x, f16_x = f32m / auto, f32m / f16
        floor = 1.3 if res >= 2048 else 1.0
        passed = au_x >= floor
        ok = ok and passed
        print(f"{res:>6} | {f32m:>7.3f} | {auto:>7.3f} {au_x:>6.2f}x | {f16:>7.3f} "
              f"{f16_x:>6.2f}x | auto>={floor} {'PASS' if passed else 'FAIL'}")
    print(f"\nB1 gate: {'PASS' if ok else 'FAIL'} (auto recovers ~fp16 perf on the node path)")


if __name__ == "__main__":
    main()
