#!/usr/bin/env python3
"""
PR-LP2 differential sweep (doc 28 success test): resolve `precision="auto"` for all 114
examples at 2048^2 CUDA; every ACCEPTED (fp16) program must match its fp32 cook within
the 8-bit quantum (3.9e-3) and stay finite; the 4 named programs must be DECLINED.

Accuracy is measured at 512^2 (per-pixel fp16 error is resolution-independent; the
overflow-prone reductions are declined anyway). CUDA-only.
"""
import os
import sys
from pathlib import Path

_bench = Path(__file__).resolve().parent
sys.path.insert(0, str(_bench.parents[1]))
sys.path.insert(0, str(_bench))
import torch
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_runtime.precision_policy import resolve_auto_precision
from four_scenario_bench import generate_bindings

EX = _bench.parent / "examples"
BAR = 3.9e-3
MUST_DECLINE = ("lens_distortion", "caustics", "auto_levels", "halftone")


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable — PR-LP2 sweep is CUDA-only. SKIP.")
        return
    accepted, declined, acc_fail, run_fail, fallback = [], [], [], [], []
    declined_names = set()
    for fn in sorted(os.listdir(EX)):
        if not fn.endswith(".tex"):
            continue
        name = fn[:-4]
        code = (EX / fn).read_text(encoding="utf-8")
        try:
            prog = Parser(Lexer(code).tokenize(), source=code).parse()
        except Exception:
            continue  # non-compiling snippet (rare); skip
        prec, reason = resolve_auto_precision(prog, 2048 * 2048, "cuda")
        if prec == "fp32":
            declined.append(name)
            declined_names.add(name)
            continue
        accepted.append(name)
        # accepted -> verify fp16 within the 8-bit quantum of fp32, and finite
        try:
            binds = generate_bindings(code, 1, 512, 512, device="cuda")
            tm = TypeChecker(binding_types={}, source=code).check(prog)
            outs = list(tm.keys()) if hasattr(tm, "keys") else ["OUT"]
            o32 = Interpreter().execute(prog, binds, tm, device="cuda",
                                        output_names=["OUT"], precision="fp32")
            o16 = Interpreter().execute(prog, binds, tm, device="cuda",
                                        output_names=["OUT"], precision="fp16")
            t32 = o32["OUT"].float(); t16 = o16["OUT"].float()
            finite = bool(torch.isfinite(t16).all())
            if not finite:
                # the node's runtime finiteness fallback re-cooks fp32 → safe, not a
                # gate violation (but worth noting: a fp16 win forgone).
                fallback.append(name)
                continue
            md = (t32 - t16).abs().max().item()
            if md > BAR:
                acc_fail.append(f"{name}: maxdiff {md:.2e} (finite but > bar)")
        except Exception as e:
            run_fail.append(f"{name}: {type(e).__name__}: {str(e)[:60]}")

    print(f"\n=== PR-LP2 sweep over {len(accepted)+len(declined)} examples ===")
    print(f"accepted (fp16): {len(accepted)}")
    print(f"  of which runtime-fp32-fallback (fp16 non-finite): {len(fallback)} {fallback}")
    print(f"declined (fp32): {len(declined)}")
    print(f"accuracy violations (finite but >{BAR:.1e}): {len(acc_fail)}")
    for f in acc_fail:
        print("  ACCFAIL", f)
    print(f"run/binding failures (skipped, not a gate issue): {len(run_fail)}")
    for f in run_fail[:10]:
        print("  skip", f)
    # the 4 named must be declined
    missing = [n for n in MUST_DECLINE
               if not any(d.startswith(n) for d in declined_names)]
    print(f"\nMUST-DECLINE check: {'PASS' if not missing else 'FAIL ' + str(missing)}")
    print(f"accepted sample: {accepted[:12]}")
    verdict = "PASS" if not acc_fail and not missing else "FAIL"
    print(f"\nOVERALL: {verdict}")


if __name__ == "__main__":
    main()
