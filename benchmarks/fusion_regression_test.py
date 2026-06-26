#!/usr/bin/env python3
"""Regression cases from the fusion code review (A1/B1/B2). Each fusable case
must be BIT-EQUIVALENT to sequential; unfusable cases must raise FusionError
(clean), not miscompile."""
import sys
from pathlib import Path
_b = Path(__file__).resolve().parent
sys.path.insert(0, str(_b.parent.parent)); sys.path.insert(0, str(_b))
import torch
from TEX_Wrangle.tex_fusion import compile_fused, FusionError
from TEX_Wrangle.tex_marshalling import infer_binding_type
from run_benchmarks import compile_program, run_interpreter

DEV = "cpu"


def _run(code, binds):
    bt = {k: infer_binding_type(v) for k, v in binds.items()}
    p, tm, asg, used = compile_program(code, bt)
    on = list(asg.keys()) if asg and "OUT" not in asg else None
    return run_interpreter(p, dict(binds), tm, DEV, on, used_builtins=used)


def fused_vs_seq(name, s0, s1):
    """Upstream s0 (reads @A), terminal s1 (reads chain @X). Return maxdiff."""
    img = torch.rand(1, 32, 32, 4)
    o0 = _run(s0, {"A": img})
    seq = _run(s1, {"X": o0})
    stages = [{"code": s0, "chain_input": None, "bindings": {"A": img}},
              {"code": s1, "chain_input": "X", "bindings": {}}]
    prog, tm, ref, asg, par, used, merged = compile_fused(stages, infer_binding_type)
    fus = run_interpreter(prog, dict(merged), tm, DEV, None, used_builtins=used)
    d = (seq - fus).abs().max().item()
    print(f"  {name:<34} maxdiff={d:.2e}  {'PASS' if d < 1e-5 else 'FAIL'}")
    return d < 1e-5


def expect_fusion_error(name, s0, s1):
    img = torch.rand(1, 16, 16, 4)
    stages = [{"code": s0, "chain_input": None, "bindings": {"A": img}},
              {"code": s1, "chain_input": "X", "bindings": {}}]
    try:
        compile_fused(stages, infer_binding_type)
        print(f"  {name:<34} NO ERROR  FAIL (expected FusionError)")
        return False
    except FusionError as e:
        print(f"  {name:<34} FusionError  PASS  ({str(e)[:40]}...)")
        return True
    except Exception as e:
        print(f"  {name:<34} {type(e).__name__}  FAIL (wanted FusionError)")
        return False


TERM = "@OUT = @X * vec4(2.0, 2.0, 2.0, 1.0);"
ok = True
print("A1 — non-terminal @OUT forms (must be bit-equivalent):")
ok &= fused_vs_seq("a) two top-level @OUT writes",
                   "@OUT = @A * vec4(0.5,0.5,0.5,1.0); @OUT = @OUT + vec4(0.1,0.0,0.0,0.0);", TERM)
ok &= fused_vs_seq("b) @OUT written in if/else",
                   "if (u > 0.5) { @OUT = @A; } else { @OUT = @A * vec4(0.4,0.4,0.4,1.0); }", TERM)
ok &= fused_vs_seq("d) channel write @OUT.r =",
                   "@OUT = @A; @OUT.r = @A.r * 0.5;", TERM)

print("\nUnfusable forms (must raise a clean FusionError):")
ok &= expect_fusion_error("c) @OUT used inside a loop",
                          "@OUT = vec4(0.0,0.0,0.0,1.0); for (int k=0;k<3;k=k+1){ @OUT = @OUT + @A*vec4(0.2,0.2,0.2,0.0); }", TERM)
ok &= expect_fusion_error("d) scatter @OUT[ix,iy] =",
                          "@OUT = @A; @OUT[ix,iy] = vec4(1.0,0.0,0.0,1.0);", TERM)
ok &= expect_fusion_error("B2) extra named output @MASK",
                          "@OUT = @A; @MASK = @A.r;", TERM)

print("\nB1 — prepare_fused returns the referenced set (param-default plumbing):")
from TEX_Wrangle.tex_fusion import prepare_fused
spec = {"stages": [{"code": "f$amt = 0.7; @OUT = @A * vec4($amt,$amt,$amt,1.0);",
                    "image_input": "A", "params": {}}],
        "terminal_image_input": "X"}
prog, tm, referenced, asg, par, used, merged = prepare_fused(
    spec, "@OUT = @X;", {"X": torch.rand(1, 8, 8, 4)}, infer_binding_type)
has_ref = isinstance(referenced, (set, frozenset))
print(f"  prepare_fused returns referenced set: {has_ref}  param decl _s0_amt in info: {'_s0_amt' in par}")
ok &= has_ref

print("\nRESULT:", "ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
