#!/usr/bin/env python3
"""Empirical test of the TEX cross-node fusion HYPOTHESIS (research, not impl):
fusing a chain of TEX programs into one (each stage's @OUT -> a local, the next
stage's input @A -> that local) is (a) BIT-EQUIVALENT to running them
sequentially, and (b) faster (one compile, no intermediate materialization,
cross-stage optimization). This is the backend half of 'only the terminal cooks'.

The splice here is done at SOURCE level for the test (the production splicer would
do it on the AST, per ast_nodes.py: flat Program.statements, @OUT = Assignment to
BindingRef('OUT'), @A = BindingRef('A')). Either way the semantics are identical:
TEX already supports multi-statement programs with typed locals."""
import sys, time, re
from pathlib import Path
_b = Path(__file__).resolve().parent
sys.path.insert(0, str(_b.parent.parent)); sys.path.insert(0, str(_b))
import torch
from run_benchmarks import compile_program, run_interpreter, _infer_types

# 3 chained stages: each reads @A (vec4 image), writes @OUT (vec4).
STAGES = [
    "@OUT = clamp(@A * vec4(1.1, 1.0, 0.9, 1.0) + vec4(0.02, 0.0, 0.05, 0.0), 0.0, 1.0);",
    "@OUT = @A * @A * vec4(1.2, 1.2, 1.2, 1.0);",
    "@OUT = clamp(@A - vec4(0.05, 0.05, 0.05, 0.0), 0.0, 1.0);",
]


def fuse(stages):
    """Textual splice: stage k's @OUT -> local __sk; stage k+1's @A -> __s{k}.
    The first stage keeps @A (the real input); the last keeps @OUT."""
    parts = []
    for i, src in enumerate(stages):
        s = src
        if i > 0:                                   # downstream: input @A -> previous local
            s = re.sub(r"@A\b", f"__s{i-1}", s)
        if i < len(stages) - 1:                     # not last: @OUT -> a fresh typed local
            s = s.replace("@OUT", f"vec4 __s{i}", 1)
        parts.append(s)
    return "\n".join(parts)


def comp(code, t):
    b = {"A": t, "image": t}
    bt = _infer_types(b)
    program, tm, asg, used = compile_program(code, bt)
    on = list(asg.keys()) if asg and "OUT" not in asg else None
    return program, tm, on, used, b


def run_seq(t):  # sequential: 3 separate compile+interpret, materializing each intermediate
    cur = t
    for src in STAGES:
        program, tm, on, used, b = comp(src, cur)
        b = {"A": cur, "image": cur}
        cur = run_interpreter(program, b, tm, str(t.device).split(":")[0] if t.is_cuda else "cpu",
                              on, used_builtins=used)
    return cur


def run_fused(t, fused_code):  # one compile, one interpret, no intermediates
    program, tm, on, used, b = comp(fused_code, t)
    dev = "cuda" if t.is_cuda else "cpu"
    return run_interpreter(program, b, tm, dev, on, used_builtins=used)


def timeit(fn, device, runs=50):
    for _ in range(8):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


fused_code = fuse(STAGES)
print("=== fused program (textual splice of 3 stages) ===")
print(fused_code)
print()

for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
    H = W = 1024
    img = torch.rand(1, H, W, 4, device=device)
    seq = run_seq(img)
    fus = run_fused(img, fused_code)
    maxdiff = (seq - fus).abs().max().item()
    seq_ms = timeit(lambda: run_seq(img), device)
    fus_ms = timeit(lambda: run_fused(img, fused_code), device)
    # compile-only cost: 3 compiles vs 1
    cseq = timeit(lambda: [comp(s, img) for s in STAGES], device, runs=20)
    cfus = timeit(lambda: comp(fused_code, img), device, runs=20)
    print(f"[{device}] 1024^2  equivalence maxdiff={maxdiff:.2e}  "
          f"{'OK' if maxdiff < 1e-5 else 'MISMATCH!'}")
    print(f"    run:     sequential {seq_ms:7.3f} ms   fused {fus_ms:7.3f} ms   "
          f"speedup {seq_ms/fus_ms:.2f}x")
    print(f"    compile: 3x stages  {cseq:7.3f} ms   fused {cfus:7.3f} ms   "
          f"speedup {cseq/cfus:.2f}x")
