"""Semantic rows: channel access on a uniform scalar, scatter reductions, lexical user-function
scope, loop counters written in the body, scalar-to-vector widening, and codegen casts.

Each row runs the interpreter (the oracle) and the codegen-only route on copies of the same
bindings. Where codegen serves the program the two must agree within 1e-5 (invariant 2); a row
that expects codegen to decline says so, and the cook then runs the interpreter.
"""
import pytest
import torch

from TEX_Wrangle.tex_cache import get_cache, parse_and_split
from TEX_Wrangle.tex_marshalling import infer_binding_type
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_runtime.codegen import try_compile
from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute
from TEX_Wrangle.tex_runtime import tier_trace


def _img(B=1, H=4, W=5, C=4, seed=11):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, H, W, C, generator=g)


def _clone(bindings):
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in bindings.items()}


def run_tiers(code, bindings, *, codegen=True, atol=1e-5):
    """Interpreter outputs; with `codegen`, codegen must serve and agree with them."""
    bt = {n: infer_binding_type(v) for n, v in bindings.items()}
    program = parse_and_split(code, bt)
    program, tm, _refs, assigned, _params, used = get_cache().compile_ast(program, bt, source=code)
    outs = sorted(assigned.keys())
    ref = Interpreter().execute(program, _clone(bindings), tm, device="cpu", output_names=outs)
    fn = try_compile(program, tm)
    if not codegen:
        assert fn is None, "codegen now serves this program: make it a parity row"
        return ref
    assert fn is not None, "codegen declined the program"
    tier_trace.reset()
    got = _codegen_only_execute(program, _clone(bindings), tm, "cpu", output_names=outs,
                                used_builtins=used, fingerprint=None, time_context=None)
    rec = tier_trace.last()
    assert rec is not None and rec.tier == "codegen", (
        f"codegen did not serve: {None if rec is None else rec.reason}")
    for n, rv in ref.items():
        gv = got[n]
        assert tuple(rv.shape) == tuple(gv.shape), f"{n}: codegen {tuple(gv.shape)} != {tuple(rv.shape)}"
        assert torch.allclose(rv.float(), gv.float(), atol=atol, equal_nan=True), f"{n}: tiers differ"
    return ref


# ── `.r` / `.x` on a uniform (0-dim) scalar ─────────────────────────────────────────────
# A scalar has no channel axis, so `.r`/`.x` is the value itself and `a.r = v` is `a = v`,
# whatever the scalar's rank. Codegen hands every non-vector channel access to the interpreter.

@pytest.mark.parametrize("code,want", [
    ("float f = 0.0; f.x = 2.0; @OUT = vec4(f);", 2.0),
    ("int f = 3; @OUT = vec4(f.r);", 3.0),
    ("float f = 0.25; float g = f.x + f.r; @OUT = vec4(g);", 0.5),
    ("float f = 1.0; f.r = 0.5; f = f + 0.25; @OUT = vec4(f);", 0.75),
], ids=["write", "int-read", "read-twice", "write-then-update"])
def test_uniform_scalar_channel(code, want):
    ref = run_tiers(code, {"OUT": torch.zeros(1, 3, 4, 4)}, codegen=False)
    assert torch.allclose(ref["OUT"], torch.full((1, 3, 4, 4), want))


def test_uniform_scalar_channel_write_takes_a_field():
    A = _img()
    ref = run_tiers("float a = 0.5; a.r = @A.g; @OUT = vec4(a);", {"A": A}, codegen=False)
    assert torch.allclose(ref["OUT"], A[..., 1:2].expand(-1, -1, -1, 4))
