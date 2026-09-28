"""Compiler fixes found by the v0.52 audit: each program is COOKED end to end and compared with
the interpreter run on the UNOPTIMIZED program (the oracle), so an optimizer that changes a
result is caught on both the engine tier (codegen) and the interpreter tier.

Rows: literal propagation must respect the declared type (`vec3 c = 0.5;` stays a vec3); CSE temp
names are unique per program, the reassignment guard sees a `for` header's writes, and the CSE
walkers share one traversal (a duplicate first seen inside a fetch argument); `pow(x, -1)` keeps
`pow`'s result at zero; a wrong-direction static step is not a zero-trip loop; an array operand
of `+`/`-`/`!`/cast is a diagnostic; a hex array size is a located parse error or a number."""
from helpers import *

import pytest

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_compiler.ast_nodes import try_extract_static_range

_BT = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}


def _img(value=None):
    if value is None:
        return make_img(1, 4, 4, 4, seed=7)
    return torch.full((1, 4, 4, 4), float(value))


def _oracle(code, A):
    return compile_and_run(code, {"A": A.clone()})


def _optimized_interp(code, A):
    """The optimized program on the interpreter tier (the engine re-checks after optimize)."""
    program = parse_and_split(code, _BT)
    checker = TypeChecker(binding_types=_BT, source=code)
    tm = checker.check(program)
    optimize(program, tm)
    checker = TypeChecker(binding_types=_BT, source=code)
    tm = checker.check(program)
    out = Interpreter().execute(program, {"A": A.clone()}, tm, device="cpu",
                                output_names=sorted(checker.assigned_bindings.keys()))
    return out["OUT"]


def _cooked(code, A):
    return tex_engine.cook(code, {"A": A.clone()}, device_mode="cpu").outputs["OUT"]


def _same(a, b):
    return a.shape == b.shape and torch.equal(torch.nan_to_num(a, nan=-7.0, posinf=1e38, neginf=-1e38),
                                              torch.nan_to_num(b, nan=-7.0, posinf=1e38, neginf=-1e38))


def _agree(code, A=None):
    A = _img() if A is None else A
    want = _oracle(code, A)
    for tier, got in (("engine", _cooked(code, A)), ("optimized interpreter", _optimized_interp(code, A))):
        assert _same(got, want), (
            f"{tier} differs from the unoptimized interpreter:\n got  {got.flatten()[:8]}\n want {want.flatten()[:8]}")
    return want


def test_literal_propagation_keeps_declared_vector_type():
    # `vec3 c = 0.5;` is a vec3 local: replacing its reads with the float literal makes the
    # re-type-check reject length()/distance() (needs a vector) for a program that ran before
    for code in ("vec3 c = 0.5;\n@OUT = vec4(length(c), 0.0, 0.0, 1.0) + @A * 0.0;\n",
                 "vec3 c = 0.5;\n@OUT = vec4(distance(c, @A.rgb), 0.0, 0.0, 1.0);\n",
                 "vec3 c = 0.5;\nvec3 d = c + vec3(0.25);\n@OUT = vec4(d, 1.0) + @A * 0.0;\n",
                 "float k = 0.5;\nint n = 3;\n@OUT = vec4(k, float(n), k * 2.0, 1.0) + @A * 0.0;\n"):
        _agree(code)


def test_cse_temp_names_are_unique_across_nested_blocks():
    code = ("float a = sin(u * 2.0) + 1.0;\n"
            "float b = 0.0;\n"
            "if (u > 0.4) {\n"
            "    float p = cos(v * 3.0) + 1.0;\n"
            "    float q = cos(v * 3.0) + 1.0;\n"
            "    b = p + q;\n"
            "}\n"
            "float c = sin(u * 2.0) + 1.0;\n"
            "@OUT = vec4(a, b, c, 1.0) + @A * 0.0;\n")
    _agree(code)


def test_cse_sees_for_header_writes():
    code = ("float k = 0.0;\n"
            "float a = sin(k * 2.0) + 1.0;\n"
            "for (k = 0.0; k < 3.0; k += 1.0) { a += 0.0; }\n"
            "float b = sin(k * 2.0) + 2.0;\n"
            "@OUT = vec4(a, b, k, 1.0) + @A * 0.0;\n")
    want = _agree(code)
    assert abs(want[0, 0, 0, 1].item() - (math.sin(6.0) + 2.0)) < 1e-5


def test_cse_finds_duplicate_first_seen_in_a_binding_argument():
    # the first occurrence of floor(u * 3.0) + 1.0 is inside an indexed fetch, then again in two
    # plain statements: the temp must be declared before the fetch that now reads it
    code = ("vec4 p = @A[int(floor(u * 3.0) + 1.0) % 4, 0];\n"
            "float g = floor(u * 3.0) + 1.0;\n"
            "float h = floor(u * 3.0) + 1.0;\n"
            "@OUT = vec4(p.r, g, h, 1.0);\n")
    _agree(code)
    code = ("vec4 p = @A(fract(v * 5.0 + 0.1) * 0.5, 0.5);\n"
            "float g = fract(v * 5.0 + 0.1) * 0.5;\n"
            "float h = fract(v * 5.0 + 0.1) * 0.5;\n"
            "@OUT = vec4(p.r, g, h, 1.0);\n")
    _agree(code)


def test_pow_minus_one_keeps_pow_result_at_zero():
    code = "@OUT = vec4(pow(@A.r, -1.0), pow(@A.g + 2.0, -1.0), 0.0, 1.0);\n"
    want = _agree(code, _img(0.0))
    assert torch.isinf(want[..., 0]).all()


def test_wrong_direction_static_step_is_not_a_static_range():
    # start above the bound: the general loop runs zero times (a static range(10, 5, -1) ran five)
    code = ("float s = 0.0;\nfor (int i = 10; i < 5; i = i - 1) { s += 1.0; }\n"
            "@OUT = vec4(s, 0.0, 0.0, 1.0) + @A * 0.0;\n")
    want = _agree(code)
    assert want[0, 0, 0, 0].item() == 0.0
    # a loop that can never end hits the iteration cap instead of silently running nothing
    inf = ("float s = 0.0;\nfor (int i = 0; i < 5; i = i - 1) { s += 1.0; }\n"
           "@OUT = vec4(s, 0.0, 0.0, 1.0) + @A * 0.0;\n")
    with pytest.raises(Exception, match="E6010|iteration"):
        _cooked(inf, _img())
    with pytest.raises(Exception, match="E6010|iteration"):
        _oracle(inf, _img())


@pytest.mark.parametrize("src", [
    "for (int i = 0; i < 5; i = i - 1) { }",
    "for (int i = 10; i < 5; i = i - 1) { }",
    "for (int i = 0; i < 5; i = i + -1) { }",
])
def test_static_range_refuses_negative_step(src):
    prog = parse_and_split("float s = 0.0; " + src + " @OUT = vec4(s);", _BT)
    loop = [s for s in prog.statements if s.__class__.__name__ == "ForLoop"][0]
    assert try_extract_static_range(loop) is None


@pytest.mark.parametrize("expr", ["a + 1.0", "-a", "!a", "float(a)", "a < 2.0", "a * a"])
def test_array_operand_is_a_diagnostic(expr):
    code = f"float a[3] = {{1.0, 2.0, 3.0}};\nfloat x = {expr};\n@OUT = vec4(x);\n"
    with pytest.raises((TypeCheckError, TEXMultiError)) as ei:
        check_code(code)
    assert "array" in str(ei.value).lower()


def test_array_index_still_types_as_its_element():
    check_code("float a[3] = {1.0, 2.0, 3.0};\nfloat x = a[1] + 1.0;\n@OUT = vec4(x);\n")


def test_hex_array_size_parses_or_is_a_located_error():
    code = "float a[0x4] = {1.0, 2.0, 3.0, 4.0};\n@OUT = vec4(a[3]) + @A * 0.0;\n"
    want = _agree(code)
    assert want[0, 0, 0, 0].item() == 4.0
    with pytest.raises(ParseError):
        parse_and_split("float a[0x0];\n@OUT = vec4(1.0);\n", _BT)
