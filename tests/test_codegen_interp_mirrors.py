"""Codegen mirrors of interpreter-side rules: the fp16 epsilon guard of the safe math
builtins and the floor+clamp of a string-array element write.

Invariant 2: the interpreter is the oracle. Each row runs both tiers on copies of the same
bindings at the same precision and requires the codegen tier to have served.
"""
import pytest
import torch

from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_cache import get_cache
from TEX_Wrangle.tex_marshalling import infer_binding_type
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute
from TEX_Wrangle.tex_runtime import tier_trace


def _img():
    return torch.rand(1, 4, 4, 4, generator=torch.Generator().manual_seed(3))


def _parity(code, precision):
    # The cook hands both tiers bindings already in the working dtype.
    bindings = {"A": _img().half() if precision == "fp16" else _img()}
    bt = {n: infer_binding_type(v) for n, v in bindings.items()}
    program = Parser(Lexer(code).tokenize(), source=code).parse()
    program, tm, _refs, _assigned, _params, used = get_cache().compile_ast(program, bt, source=code)
    ref = Interpreter().execute(program, {k: v.clone() for k, v in bindings.items()}, tm,
                                device="cpu", output_names=["OUT"], precision=precision)["OUT"]
    tier_trace.reset()
    got = _codegen_only_execute(program, {k: v.clone() for k, v in bindings.items()}, tm, "cpu",
                                output_names=["OUT"], used_builtins=used, precision=precision,
                                fingerprint=None, time_context=None)["OUT"]
    rec = tier_trace.last()
    assert rec is not None and rec.tier == "codegen", f"codegen did not serve: {rec and rec.reason}"
    assert torch.equal(torch.isnan(ref), torch.isnan(got)), (ref, got)
    assert torch.equal(torch.isinf(ref), torch.isinf(got)), (ref, got)
    fin = torch.isfinite(ref)
    assert (ref[fin].float() - got[fin].float()).abs().max().item() <= 1e-3
    return ref


_EPS_ROWS = [
    "sdiv(@A.r, @A.g * 0.000001)",
    "spow(@A.g * 0.000001, -1.0)",
    "normalize(@A.rgb * 0.0).x",
    "log(@A.r * 0.0)",
    "log2(@A.r * 0.0)",
    "log10(@A.r * 0.0)",
    "fit(@A.r, 0.5, 0.5, 0.0, 1.0)",
    "smoothstep(0.5, 0.5, @A.r)",
]


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("expr", _EPS_ROWS)
def test_safe_math_epsilon_follows_the_working_dtype(expr, precision):
    ref = _parity(f"@OUT = vec3({expr});", precision)
    assert torch.isfinite(ref).all()


def test_string_array_write_floors_its_index():
    code = ('string a[3] = {"a", "bb", "ccc"};\n'
            'a[1.7] = "zzzzz";\n@OUT = vec3(float(len(a[1])), float(len(a[2])), 0.0);')
    ref = _parity(code, "fp32")
    assert ref[..., 0].unique().tolist() == [5.0] and ref[..., 1].unique().tolist() == [3.0]


@pytest.mark.parametrize("first,expr", [
    ("0.0 * @A.g", "-0.0 * @A.r"),
    ("vec3(0.0, 1.0, 2.0).x * @A.g", "vec3(-0.0, 1.0, 2.0).x * @A.r"),
], ids=["scalar", "vec"])
def test_signed_zero_literal_keeps_its_sign(first, expr):
    # The +0.0 literal appears first, so a constant cache keyed by value would hand the
    # -0.0 literal its tensor.
    _parity(f"@OUT = vec3(atan2(0.0, {first}), atan2(0.0, {expr}), 0.0);", "fp32")


def test_float_cast_of_an_fp16_value_is_float32():
    # 100000 is past fp16's range: the interpreter's float() promotes, so the product is finite.
    ref = _parity("@OUT = vec3(float(@A.r) * 100000.0);", "fp16")
    assert torch.isfinite(ref).all()
