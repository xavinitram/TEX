"""Language-front-end rows: what the checker accepts, both tiers must run the same way.

Every row compiles through the pipeline a cook runs (type-check, optimize, re-check) and runs
the interpreter and the codegen-only route on copies of the same bindings. Invariant 2: the
interpreter is the oracle, codegen matches it within 1e-5, and the codegen tier must have
served the answer itself.
"""
import pytest
import torch

from TEX_Wrangle.tex_cache import get_cache, parse_and_split
from TEX_Wrangle.tex_marshalling import infer_binding_type
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker, TypeCheckError
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_compiler.diagnostics import TEXMultiError
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_runtime.codegen import try_compile
from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute
from TEX_Wrangle.tex_runtime import tier_trace


def _img(B=1, H=4, W=4, C=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, H, W, C, generator=g)


def _clone(bindings):
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in bindings.items()}


def _compile(code, bindings):
    bt = {n: infer_binding_type(v) for n, v in bindings.items()}
    program = parse_and_split(code, bt)
    return get_cache().compile_ast(program, bt, source=code)


def both_tiers(code, bindings):
    """(interpreter outputs, codegen outputs); fails unless codegen served."""
    program, tm, _refs, assigned, _params, used = _compile(code, bindings)
    outs = sorted(assigned.keys())
    ref = Interpreter().execute(program, _clone(bindings), tm, device="cpu",
                                output_names=outs)
    assert try_compile(program, tm) is not None, "codegen declined the program"
    tier_trace.reset()
    got = _codegen_only_execute(program, _clone(bindings), tm, "cpu", output_names=outs,
                                used_builtins=used, fingerprint=None, time_context=None)
    rec = tier_trace.last()
    assert rec is not None and rec.tier == "codegen", (
        f"codegen did not serve: {None if rec is None else rec.reason}")
    return ref, got


def run_both(code, bindings, expect=None, name="OUT", atol=1e-5):
    """Run both tiers, require parity, and (optionally) compare OUT with `expect`."""
    ref, got = both_tiers(code, bindings)
    for n, rv in ref.items():
        gv = got[n]
        assert tuple(rv.shape) == tuple(gv.shape), f"{n}: codegen shape {tuple(gv.shape)} != {tuple(rv.shape)}"
        assert (rv.float() - gv.float()).abs().max().item() <= atol, f"{n}: tiers differ"
    if expect is not None:
        out = ref[name]
        exp = torch.as_tensor(expect, dtype=out.dtype).expand_as(out) if not torch.is_tensor(expect) else expect
        assert tuple(out.shape) == tuple(exp.shape), f"shape {tuple(out.shape)} != {tuple(exp.shape)}"
        assert torch.allclose(out, exp, atol=1e-5), f"got {out.flatten()[:8]}, expected {exp.flatten()[:8]}"
    return ref


def check_errors(code, bindings=None):
    """The checker's error codes for `code` (empty when it type-checks)."""
    bt = {"OUT": TEXType.VEC4, **(bindings or {})}
    prog = parse_and_split(code, bt)
    try:
        TypeChecker(binding_types=bt, source=code).check(prog)
    except TypeCheckError as e:
        return [e.code]
    except TEXMultiError as e:
        return [d.code for d in e.diagnostics]
    return []


# ── Implicit widening: scalar -> vecN, narrow vec -> wider vec ─────────────────────────────
# The checker accepts both (LANGUAGE.md section 4: a scalar broadcasts to every component).
# The value is widened where the declared width is known: declaration, assignment, return,
# and a user-function argument. vec2 -> vec3 pads 0; a widening to vec4 pads alpha 1.

def _a():
    return _img()


def test_scalar_decl_broadcasts_to_declared_vec():
    A = _a()
    r = A[..., 0]
    run_both("float x = @A.r; vec3 c = x; @OUT = vec4(c, 1.0);", {"A": A},
             torch.stack([r, r, r, torch.ones_like(r)], -1))


def test_uniform_scalar_decl_broadcasts():
    run_both("float k = @A.r * 0.0 + 0.5; vec3 c = 0.25; c = c + k * 0.0; @OUT = vec4(c.r, c.g, c.b, 1.0);",
             {"A": _a()}, torch.tensor([0.25, 0.25, 0.25, 1.0]).expand(1, 4, 4, 4))


def test_scalar_assignment_broadcasts():
    A = _a()
    r = A[..., 0]
    run_both("vec3 c = vec3(0.0); c = @A.r; @OUT = vec4(c, 1.0);", {"A": A},
             torch.stack([r, r, r, torch.ones_like(r)], -1))


def test_scalar_return_broadcasts_to_vec_return_type():
    A = _a()
    ref = run_both("vec3 gray(vec4 c){ return luma(c); } @OUT = vec4(gray(@A), 1.0);", {"A": A})
    out = ref["OUT"]
    assert tuple(out.shape) == (1, 4, 4, 4)
    assert torch.allclose(out[..., 0], out[..., 1]) and torch.allclose(out[..., 0], out[..., 2])
    assert torch.allclose(out[..., 3], torch.ones_like(out[..., 3]))


def test_uniform_scalar_return_broadcasts():
    run_both("vec3 f(){ return 0.5; } @OUT = vec4(f(), @A.a * 0.0 + 1.0);", {"A": _a()},
             torch.tensor([0.5, 0.5, 0.5, 1.0]).expand(1, 4, 4, 4))


def test_scalar_argument_broadcasts_to_vec_param():
    A = _a()
    r = A[..., 0]
    run_both("vec3 f(vec3 c){ return c * 2.0; } @OUT = vec4(f(@A.r), 1.0);", {"A": A},
             torch.stack([2 * r, 2 * r, 2 * r, torch.ones_like(r)], -1))


def test_vec3_argument_widens_to_vec4_param_with_alpha_one():
    A = _a()
    run_both("vec4 f(vec4 v){ return v; } @OUT = f(@A.rgb);", {"A": A},
             torch.cat([A[..., :3], torch.ones_like(A[..., :1])], -1))


def test_vec3_decl_widens_to_vec4_alpha_one():
    run_both("vec4 c = vec3(1.0, 2.0, 3.0); @OUT = vec4(c.a) + @A * 0.0;", {"A": _a()},
             torch.ones(1, 4, 4, 4))


def test_vec2_decl_widens_to_vec3_zero():
    run_both("vec3 c = vec2(1.0, 2.0) + @A.rg * 0.0; @OUT = vec4(c, 1.0);", {"A": _a()},
             torch.tensor([1.0, 2.0, 0.0, 1.0]).expand(1, 4, 4, 4))


def test_vec2_assignment_widens_to_vec4():
    run_both("vec4 c = vec4(9.0); c = vec2(1.0, 2.0) + @A.rg * 0.0; @OUT = c;", {"A": _a()},
             torch.tensor([1.0, 2.0, 0.0, 1.0]).expand(1, 4, 4, 4))


def test_vec_array_element_scalar_assignment():
    run_both("vec3 a[2]; a[1] = @A.r * 0.0 + 0.5; @OUT = vec4(a[1], 1.0);", {"A": _a()},
             torch.tensor([0.5, 0.5, 0.5, 1.0]).expand(1, 4, 4, 4))


def test_widening_is_idempotent_on_recheck():
    # The optimized AST is re-checked; a widened value must not be wrapped twice.
    code = "vec4 c = @A.rgb; @OUT = c;"
    program, tm, *_ = _compile(code, {"A": _a()})
    TypeChecker(binding_types={"A": TEXType.VEC4}, source=code,
                strict_redeclare=False).check(program)
    init = program.statements[0].initializer
    assert type(init).__name__ == "VecConstructor" and type(init.args[0]).__name__ != "VecConstructor"


# ── A per-pixel `if` merges variables a for-loop HEADER assigns ─────────────────────────────

def _half_mask():
    A = torch.zeros(1, 2, 2, 4)
    A[0, :, 1, 0] = 1.0            # r = 1 only at x = 1
    return A


@pytest.mark.parametrize("pragma", ["", "//!tex 0.25\n"])
def test_for_header_assignment_does_not_leak_out_of_per_pixel_if(pragma):
    code = pragma + ("int i = 0; if (@A.r > 0.5) { for (i = 0; i < 3; i++) { } } "
                     "@OUT = vec4(float(i), 0.0, 0.0, 1.0);")
    ref = run_both(code, {"A": _half_mask()})
    assert ref["OUT"][0, :, :, 0].tolist() == [[0.0, 3.0], [0.0, 3.0]]


@pytest.mark.parametrize("pragma", ["", "//!tex 0.25\n"])
def test_for_update_assignment_does_not_leak_out_of_per_pixel_if(pragma):
    code = pragma + ("float j = 0.0; if (@A.r > 0.5) { for (int k = 0; k < 3; j += 1.0) { k++; } } "
                     "@OUT = vec4(j, 0.0, 0.0, 1.0);")
    ref = run_both(code, {"A": _half_mask()})
    assert ref["OUT"][0, :, :, 0].tolist() == [[0.0, 3.0], [0.0, 3.0]]


# ── int / int is a float quotient, in the checker as at run time ───────────────────────────

def test_int_div_int_is_typed_float():
    # Both tiers and the constant folder divide as float; the checker agrees, so the
    # program is valid (or not) independently of whether the operands fold.
    assert check_errors("int a = 7; int b = 2; float x = a / b; @OUT = vec4(x);") == []
    assert check_errors("int a = 7; int b = 2; int x = a / b; @OUT = vec4(float(x));") == ["E3200"]
    assert check_errors("int x = 7 / 2; @OUT = vec4(float(x));") == ["E3200"]
    assert check_errors("int f(int a){ return a / 2; } @OUT = vec4(float(f(3)));") == ["E3013"]


def test_int_div_int_floored_by_int_cast():
    code = "int n = int(@A.r * 0.0 + 7.0); int c = int(n / 2); float q = n / 2; @OUT = vec4(float(c), q, 0.0, 1.0);"
    run_both(code, {"A": _a()}, torch.tensor([3.0, 3.5, 0.0, 1.0]).expand(1, 4, 4, 4))


def test_vec_array_literal_elements_widen():
    A = _a()
    run_both("vec4 a[2] = {@A.rgb, vec4(0.5)}; @OUT = a[0];", {"A": A},
             torch.cat([A[..., :3], torch.ones_like(A[..., :1])], -1))
    run_both("vec3 a[2] = {@A.r * 0.0 + 0.5, vec3(1.0, 2.0, 3.0)}; @OUT = vec4(a[0], 1.0);", {"A": A},
             torch.tensor([0.5, 0.5, 0.5, 1.0]).expand(1, 4, 4, 4))


# ── Parser: scatter increments accumulate; literals the fp32 runtime cannot hold ───────────

@pytest.mark.parametrize("stmt,want", [("@OUT[0,0]++;", 16.0), ("@OUT[0,0]--;", -16.0),
                                       ("@OUT[0,0] += 1.0;", 16.0)])
def test_scatter_increment_accumulates_like_compound_add(stmt, want):
    ref = run_both("@OUT = @A.r * 0.0; " + stmt, {"A": _a()})
    assert ref["OUT"][0, 0, 0].item() == want


def _parse_error_code(code):
    from TEX_Wrangle.tex_compiler.parser import ParseError
    try:
        parse_and_split(code, {"OUT": TEXType.VEC4})
    except ParseError as e:
        return e.code if hasattr(e, "code") else "?"
    except TEXMultiError as e:
        return e.diagnostics[0].code
    return None


@pytest.mark.parametrize("code", [
    "@OUT = vec4(1e39);", "float x = 1e999;", "@OUT = vec4(-3.5e38);",
    "f$k = 0.5 [max: 1e999];", "int x = " + "9" * 5000 + ";", "float a[" + "9" * 5000 + "];",
], ids=["1e39", "1e999", "neg3.5e38", "meta1e999", "int5000digits", "size5000digits"])
def test_out_of_range_literal_is_a_parse_error(code):
    assert _parse_error_code(code) == "E2000"


def test_in_range_literals_parse():
    for code in ("@OUT = vec4(3.4028234e38);", "float a[0x4]; @OUT = vec4(a[1]);",
                 "f$k = 0.5 [max: 1e30]; @OUT = vec4($k);"):
        assert _parse_error_code(code) is None, code


# ── Builtin signatures: result types and argument kinds match the runtime ──────────────────

_V3 = {"A": TEXType.VEC3}
_V4 = {"A": TEXType.VEC4}


@pytest.mark.parametrize("code,want", [
    ("int n = 2; int r = sqrt(n);", ["E3200"]),
    ("int n = 2; int r = sin(n);", ["E3200"]),
    ("int n = 2; int r = smoothstep(0, 4, n);", ["E3200"]),
    ("int n = 2; int r = fit(n, 0, 4, 0, 1);", ["E3200"]),
    ("int n = 2; int r = pow(n, 2);", ["E3200"]),
    ("int n = 2; float r = sqrt(n);", []),
    ("int a = 2; int b = 3; int r = min(a, b) + abs(a) + floor(b) + mod(b, a);", []),
], ids=["sqrt", "sin", "smoothstep", "fit", "pow", "float-ok", "int-rows-ok"])
def test_fractional_builtins_of_ints_are_float(code, want):
    assert check_errors(code + " @OUT = vec4(1.0);") == want


def test_isnan_of_a_vector_is_a_vector():
    assert check_errors("float b = isnan(@A); @OUT = vec4(b);", _V4) == ["E3200"]
    A = _a()
    A[0, 0, 0, 1] = float("nan")
    ref = run_both("vec4 b = isnan(@A) + isinf(@A); @OUT = b;", {"A": A})
    assert ref["OUT"][0, 0, 0].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_blend_keeps_the_base_width():
    A = _a()
    ref = run_both("vec4 c = screen(@A.rgb, @A); @OUT = c;", {"A": A})
    out = ref["OUT"]
    assert tuple(out.shape) == (1, 4, 4, 4)
    assert torch.allclose(out[..., 3], torch.ones_like(out[..., 3]))
    assert check_errors("vec4 c = screen(0.5, @A); @OUT = c;", _V4) == ["E5003"]


@pytest.mark.parametrize("code,bt", [
    ("vec3 a = vec3(1.0); vec4 b = vec4(0.5); @OUT = vec4(min(a, b));", {}),
    ("@OUT = lerp(@A, @B, 0.5);", {"A": TEXType.VEC3, "B": TEXType.VEC4}),
    ("@OUT = vec4(dot(@A.rgb, @A.rg));", _V4),
    ("vec2 r = rgb2hsv(@A.rg); @OUT = vec4(r, 0.0, 1.0);", _V4),
    ("@OUT = vec4(luma(@A.r));", _V4),
    ("@OUT = vec4(perlin(0.5, @A.rg).x);", _V4),
    ("@OUT = vec4(sdf_circle(@A.rg, 0.5, 0.2));", _V4),
    ("@OUT = vec4(arr_sum(@A.r));", _V4),
    ("@OUT = over(@A, @A);", _V3),
    ("@OUT = vec4(premultiply(@A), 1.0);", _V3),
], ids=["min-v3v4", "lerp-v3v4", "dot-v3v2", "rgb2hsv-v2", "luma-scalar", "perlin-vec",
        "sdf-vec", "arr_sum-scalar", "over-v3", "premultiply-v3"])
def test_builtin_argument_kinds_are_checked(code, bt):
    assert "E5003" in check_errors(code, bt)


def test_not_of_a_vector_is_a_vector():
    assert check_errors("float t = !@A.rgb; @OUT = vec4(t);", _V4) == ["E3200"]
    A = _a()
    ref = run_both("vec3 t = !(@A.rgb - @A.rgb); @OUT = vec4(t, 0.0);", {"A": A})
    assert ref["OUT"][0, 0, 0].tolist() == [1.0, 1.0, 1.0, 0.0]


# ── Checker rules the runtime relies on ─────────────────────────────────────────────────

@pytest.mark.parametrize("code,want", [
    ("mat3 a[2]; a[0] = mat3(2.0); @OUT = vec4(a[0] * vec3(1.0), 1.0);", "E3101"),
    ("f$a = 1.0 + 2.0; @OUT = vec4($a);", "E3200"),
    ("f$k = PI; @OUT = vec4($k);", "E3200"),
    ("PI = 3.0; @OUT = vec4(PI);", "E3204"),
    ("ix = 0.0; @OUT = vec4(ix);", "E3204"),
    ("u += 0.5; @OUT = vec4(u);", "E3204"),
    ("for (frame = 0.0; frame < 2.0; frame += 1.0) { } @OUT = vec4(1.0);", "E3204"),
    ("@A.r = vec3(1.0, 2.0, 3.0); @OUT = @A;", "E3200"),
    ('@A.r = "x"; @OUT = @A;', "E3200"),
    ("if (@A.r > 2.0) { float f(float x) { return x; } } @OUT = vec4(f(2.0));", "E5001"),
    ("float g = 2.0; float f(float x) { g = x; return x; } float r = f(7.0); @OUT = vec4(g + r);", "E3204"),
    ("vec4 c = vec4(0.0); float f(float x) { c.r = x; return x; } @OUT = c + f(1.0);", "E3204"),
    ("float a[2] = {0.0, 0.0}; float f(float x) { a[0] = x; return x; } @OUT = vec4(a[0] + f(1.0));", "E3204"),
    ("float f = 0.0; f.x = 2.0; @OUT = vec4(f);", "E3301"),
], ids=["mat-array", "param-expr", "param-name", "PI", "ix", "u+=", "for-frame", "chan-vec3",
        "chan-str", "fn-after-if", "fn-outer-var", "fn-outer-chan", "fn-outer-array",
        "scalar-chan-write"])
def test_checker_rejects_what_the_runtime_cannot_run(code, want):
    assert want in check_errors(code, _V4)


@pytest.mark.parametrize("code", [
    "float f(float x) { float g = x; g = g * 2.0; return g; } @OUT = vec4(f(1.0));",
    "float g = 1.0; float f(float x) { return x + g; } @OUT = vec4(f(1.0));",
    "if (@A.r > 0.5) { float u = 2.0; u = 3.0; } @OUT = vec4(u);",
    "f$k = -0.5; v3$t = vec3(-1.0, 0.5, 2); v3$s = 0.25; @OUT = vec4($k);",
    "@OUT.r = 0.5; @OUT.gb = vec2(0.1, 0.2); @OUT.a = 1.0;",
])
def test_checker_still_accepts(code):
    assert check_errors(code, _V4) == []


def test_param_defaults_negative_and_broadcast():
    code = "v3$t = vec3(-1.0, 0.5, 2); v3$s = 0.25; f$k = -2; @OUT = vec4($t, $k);"
    ck = TypeChecker(binding_types={"OUT": TEXType.VEC4}, source=code)
    ck.check(parse_and_split(code, {"OUT": TEXType.VEC4}))
    assert ck.param_declarations["t"]["default_value"] == [-1.0, 0.5, 2.0]
    assert ck.param_declarations["s"]["default_value"] == [0.25, 0.25, 0.25]
    assert ck.param_declarations["k"]["default_value"] == -2


def test_reading_a_rebound_input_uses_the_assigned_type():
    assert "E3301" in check_errors("@A = @A.r; @OUT = vec4(@A.g);", _V4)
    ref = run_both("@A = @A.r; @OUT = vec4(@A, 0.0, 0.0, 1.0);", {"A": _a()})
    assert tuple(ref["OUT"].shape) == (1, 4, 4, 4)


def test_function_defined_in_a_loop_survives_unrolling():
    # The optimizer unrolls the loop, copying the definition; the cook's re-check accepts it.
    run_both("float s = @A.r * 0.0; for (int i = 0; i < 2; i++) { float f(float x) { return x; } "
             "s += f(1.0); } @OUT = vec4(s);", {"A": _a()}, torch.full((1, 4, 4, 4), 2.0))
    run_both("float s = @A.r * 0.0; if (@A.r > -1.0) { float f(float x) { return x * 3.0; } s = f(1.0); } "
             "@OUT = vec4(s);", {"A": _a()}, torch.full((1, 4, 4, 4), 3.0))


@pytest.mark.parametrize("code,want", [
    ('float x = ("a") ? 1.0 : 0.0; @OUT = vec4(x);', "E3500"),
    ("float x = (@A.rgb) ? 1.0 : 0.0; @OUT = vec4(x);", "E3500"),
    ("mat3 m = mat3(1.0); mat3 n = mat3(2.0); mat3 k = (@A.r < 1.0) ? m : n; @OUT = vec4(1.0);", "E3400"),
    ('if ("a") { @OUT = vec4(1.0); }', "E3500"),
    ("mat3 m = mat3(1.0); if (m) { @OUT = vec4(1.0); }", "E3500"),
    ("float x = float(vec3(1.0, 2.0, 3.0) + @A.rgb); @OUT = vec4(x);", "E3200"),
    ("int i = int(mat3(2.0)); @OUT = vec4(float(i));", "E3200"),
], ids=["tern-str", "tern-vec", "tern-mat", "if-str", "if-mat", "float-vec", "int-mat"])
def test_conditions_and_casts_are_typed_as_they_run(code, want):
    assert want in check_errors(code, _V4)


def test_int_cast_of_a_vector_is_an_elementwise_vector():
    A = _a()
    ref = run_both("vec3 q = int(@A.rgb * 8.0) / 8.0; @OUT = vec4(q, 1.0);", {"A": A})
    assert torch.allclose(ref["OUT"][..., :3], torch.floor(A[..., :3] * 8.0) / 8.0)


# ── Optimizer: folds agree with the fp32 runtime and keep shapes ────────────────────────

def _optimized(code, bindings=None):
    program, *_ = _compile(code, bindings or {"A": _a()})
    return program


def _unfolded_equals_folded(expr_folded, expr_runtime):
    """OUT of a constant expression equals the same expression computed on a runtime value."""
    A = _a()
    folded = run_both(f"@OUT = vec4({expr_folded}) + @A * 0.0;", {"A": A})["OUT"]
    live = run_both(f"float z = @A.r * 0.0; @OUT = vec4({expr_runtime}) + @A * 0.0;", {"A": A})["OUT"]
    assert torch.equal(folded, live), (folded.flatten()[:1], live.flatten()[:1])


@pytest.mark.parametrize("folded,live", [
    ("clamp(0.5, 1.0, 0.0)", "clamp(0.5 + z, 1.0, 0.0)"),
    ("1000000.0 % -3.7", "(1000000.0 + z) % -3.7"),
    ("mod(1000000.0, -3.7)", "mod(1000000.0 + z, -3.7)"),
    ("0.1 + 0.2", "(0.1 + z) + 0.2"),
    ("16777217.0 * 3.0", "(16777217.0 + z) * 3.0"),
], ids=["clamp-inverted", "rem", "mod", "add", "big"])
def test_constant_fold_matches_the_runtime(folded, live):
    _unfolded_equals_folded(folded, live)


def test_fold_never_mints_a_non_finite_literal():
    ref = run_both("float x = 1e38 * 10.0; @OUT = vec4(isinf(x)) + @A * 0.0;", {"A": _a()})
    assert ref["OUT"].flatten()[0].item() == 1.0


@pytest.mark.parametrize("expr", ["pow(@A.rgb, 0.0)", "lerp(vec3(1.0), @A.rgb, 0.0)",
                                  "(1.0) ? vec3(1.0) : @A.rgb"], ids=["pow0", "lerp0", "ternary"])
def test_fold_keeps_a_dropped_operands_shape(expr):
    # A fold to a uniform 1.0 would make img_sum see one value, not the 16 pixels.
    code = f"vec3 s = img_sum({expr}); @OUT = vec4(s.r, s.g, s.b, 1.0);"
    program, tm, _r, _a2, _p, _u = _compile(code, {"A": _a()})
    out = Interpreter().execute(program, {"A": _a()}, tm, device="cpu", output_names=["OUT"])["OUT"]
    assert out.flatten()[:3].tolist() == [16.0, 16.0, 16.0]


def test_pow_square_does_not_duplicate_an_expensive_operand():
    prog = _optimized("@OUT = pow(gauss_blur(@A, 3.0), 2.0);")
    call = prog.statements[0].value
    assert type(call).__name__ == "FunctionCall" and call.name == "pow"
    prog = _optimized("@OUT = pow(@A * 0.5, 2.0);")
    assert type(prog.statements[0].value).__name__ == "BinOp"


def test_unroll_leaves_a_loop_that_writes_its_counter():
    # Unrolling used to substitute the counter into `i = i + 1`'s target (an assignment
    # to a literal). The loop is now left a loop; both tiers agree on it.
    prog = _optimized("float s = @A.r * 0.0; for (int i = 0; i < 4; i++) { s += 1.0; i = i + 1; } @OUT = vec4(s);")
    assert any(type(st).__name__ == "ForLoop" for st in prog.statements)
    run_both("float s = @A.r * 0.0; for (int i = 0; i < 4; i++) { s += 1.0; i = i + 1; } @OUT = vec4(s);",
             {"A": _a()})
