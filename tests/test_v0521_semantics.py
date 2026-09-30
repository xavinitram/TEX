"""Tier-parity rows: scalar-loop arithmetic rounds like the interpreter's fp32.

Each row runs the interpreter (the oracle) and the codegen-only route on copies of the same
bindings; codegen must serve the program. Loop rows compare BIT-exactly: a scalar loop runs
the same fp32 operations on both tiers, so a looser tolerance would hide the drift these rows
exist to catch (a long accumulation drifts by 1e-3, and a comparison can flip a branch).
"""
import pytest
import torch

from TEX_Wrangle.tex_cache import get_cache, parse_and_split
from TEX_Wrangle.tex_marshalling import infer_binding_type
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_runtime.codegen import try_compile
from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute
from TEX_Wrangle.tex_runtime import tier_trace


def _clone(bindings):
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in bindings.items()}


def both_tiers(code, bindings):
    """(interpreter outputs, codegen outputs); codegen must serve the program."""
    bt = {n: infer_binding_type(v) for n, v in bindings.items()}
    program = parse_and_split(code, bt)
    program, tm, _refs, assigned, _params, used = get_cache().compile_ast(program, bt, source=code)
    outs = sorted(assigned.keys())
    ref = Interpreter().execute(program, _clone(bindings), tm, device="cpu", output_names=outs)
    assert try_compile(program, tm) is not None, "codegen declined the program"
    tier_trace.reset()
    got = _codegen_only_execute(program, _clone(bindings), tm, "cpu", output_names=outs,
                                used_builtins=used, fingerprint=None, time_context=None)
    rec = tier_trace.last()
    assert rec is not None and rec.tier == "codegen", (
        f"codegen did not serve: {None if rec is None else rec.reason}")
    return ref, got


def _out():
    return {"OUT": torch.zeros(1, 2, 3, 4)}


# ── Scalar-mode loops round every operation to fp32 ─────────────────────────────────────

@pytest.mark.parametrize("code", [
    # Ten `+= 0.1` in fp32 end just above 1.0; in double just below. The branch flips.
    "float f = 0.0; int n = 0; for (int i = 0; i < 20; i++) { f += 0.1; "
    "if (!(f <= 1.0)) { break; } n += 1; } @OUT = vec4(n, f, 0.0, 1.0);",
    # A long accumulation: fp32 ends at 99.99905, double at 100.0.
    "float s = 0.0; for (int i = 0; i < 1000; i++) { s += 0.1; } @OUT = vec4(s);",
    # Compound expressions round each step, not only the assignment.
    "float s = 0.0; float t = 1.0; for (int i = 0; i < 500; i++) { s += 0.1 * t; "
    "t = t * 0.999 + 0.001; } @OUT = vec4(s, t, 0.0, 1.0);",
    # Division, including the zero guard, and the math builtins.
    "float s = 0.0; for (int i = 0; i < 300; i++) { s += 1.0 / (i + 3.0) + i / 0.0 * 0.0; "
    "s += sqrt(i * 0.37) * 1e-3 + sin(i * 0.1) * 1e-3; } @OUT = vec4(s);",
    "float s = 0.0; for (int i = 0; i < 200; i++) { s += smoothstep(0.1, 0.9, i * 0.004) "
    "+ fract(i * 0.013) + lerp(0.2, 0.7, i * 0.003) + pow(1.001, i * 0.5); } @OUT = vec4(s);",
    # A value past fp32's range is inf on both tiers, never a finite double.
    "float s = 1e30; for (int i = 0; i < 12; i++) { s = s * 1e30 / 1e30; } @OUT = vec4(s);",
], ids=["branch-flip", "accumulate", "compound", "div-and-math", "stdlib", "fp32-overflow"])
def test_scalar_loop_rounds_like_the_interpreter(code):
    ref, got = both_tiers(code, _out())
    assert torch.equal(ref["OUT"], got["OUT"]), (
        f"tiers differ: interp {ref['OUT'].flatten()[:2].tolist()} "
        f"codegen {got['OUT'].flatten()[:2].tolist()}")


def test_scalar_loop_break_count_is_the_fp32_one():
    """In fp32, ten `f += 0.1` exceed 1.0, so the loop counts nine passes (double: ten)."""
    code = ("float f = 0.0; int n = 0; for (int i = 0; i < 20; i++) { f += 0.1; "
            "if (!(f <= 1.0)) { break; } n += 1; } @OUT = vec4(n);")
    ref, got = both_tiers(code, _out())
    assert ref["OUT"].flatten()[0].item() == 9.0
    assert got["OUT"].flatten()[0].item() == 9.0
