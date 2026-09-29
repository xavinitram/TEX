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
