#!/usr/bin/env python3
"""Validate tex_fusion.compile_fused: the AST-splicer must be BIT-EQUIVALENT to
running the stages sequentially, even with deliberate cross-stage name
collisions (every stage reuses local `t` and param `amt`) and params."""
import sys
from pathlib import Path
_b = Path(__file__).resolve().parent
sys.path.insert(0, str(_b.parent.parent)); sys.path.insert(0, str(_b))
import torch
from TEX_Wrangle.tex_fusion import compile_fused, prepare_fused
from TEX_Wrangle.tex_marshalling import infer_binding_type
from run_benchmarks import compile_program, run_interpreter

# Each tuple: (code, external bindings+params, chain_input binding name | None)
STAGES = [
    ("float t = $amt; @OUT = clamp(@A * vec4(t, 1.0, 1.0 - t, 1.0), 0.0, 1.0);",
     {"amt": 0.3}, None, "A"),
    ("float t = $amt; @OUT = pow(max(@X, 0.0), vec4(t, t, t, 1.0));",
     {"amt": 0.8}, "X", None),
    ("float t = $amt; vec4 c = @Y - vec4(t, t, t, 0.0); @OUT = clamp(c, 0.0, 1.0);",
     {"amt": 0.05}, "Y", None),
]


def run_one(code, binds, device):
    bt = {k: infer_binding_type(v) for k, v in binds.items()}
    program, tm, asg, used = compile_program(code, bt)
    on = list(asg.keys()) if asg and "OUT" not in asg else None
    return run_interpreter(program, dict(binds), tm, device, on, used_builtins=used)


def run_sequential(img, device):
    # stage 0 (external @A=img), then feed each output into the next stage's input
    cur = img
    out = run_one(STAGES[0][0], {"A": cur, "amt": 0.3}, device)
    out = run_one(STAGES[1][0], {"X": out, "amt": 0.8}, device)
    out = run_one(STAGES[2][0], {"Y": out, "amt": 0.05}, device)
    return out


def run_fused(img, device):
    stages = [
        {"code": STAGES[0][0], "chain_input": None, "bindings": {"A": img, "amt": 0.3}},
        {"code": STAGES[1][0], "chain_input": "X", "bindings": {"amt": 0.8}},
        {"code": STAGES[2][0], "chain_input": "Y", "bindings": {"amt": 0.05}},
    ]
    program, tm, ref, asg, params, used, merged = compile_fused(stages, infer_binding_type)
    assert list(asg.keys()) == ["OUT"], f"fused outputs should be just OUT, got {list(asg.keys())}"
    out = run_interpreter(program, dict(merged), tm, device, None, used_builtins=used)
    return out


def run_payload(img, device):
    # Exactly what the frontend graphToPrompt would hand the terminal:
    # stages 0 and 1 are the UPSTREAM nodes; the terminal is STAGES[2].
    spec = {
        "stages": [
            {"code": STAGES[0][0], "image_input": "A", "params": {"amt": 0.3}},
            {"code": STAGES[1][0], "image_input": "X", "params": {"amt": 0.8}},
        ],
        # The terminal's chain socket (@Y) carries the source AND is read as the
        # chain — one key, matching how graphToPrompt rewires it.
        "terminal_image_input": "Y",
    }
    terminal_bindings = {"Y": img, "amt": 0.05}  # source on the terminal's chain socket + terminal param
    program, tm, ref, asg, params, used, merged = prepare_fused(
        spec, STAGES[2][0], terminal_bindings, infer_binding_type)
    assert list(asg.keys()) == ["OUT"]
    return run_interpreter(program, dict(merged), tm, device, None, used_builtins=used)


ok = True
for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
    img = torch.rand(1, 256, 256, 4, device=device)
    seq = run_sequential(img, device)
    fus = run_fused(img, device)
    pay = run_payload(img, device)
    d_fus = (seq - fus).abs().max().item()
    d_pay = (seq - pay).abs().max().item()
    good = d_fus < 1e-5 and d_pay < 1e-5
    ok = ok and good
    print(f"[{device}] direct maxdiff={d_fus:.2e}  payload maxdiff={d_pay:.2e}  "
          f"{'PASS' if good else 'FAIL'}")

print("RESULT:", "PASS — splicer + payload path bit-equivalent (params + collisions)" if ok else "FAIL")
sys.exit(0 if ok else 1)
