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


# ── Scatter `*=` / `/=` reduce over colliding writers, like `+=` / `-=` ─────────────────
# Every source pixel that lands on a destination applies its factor once.

def _ones(H=4, W=4, C=4):
    # `@A` sets the grid (a pre-bound @OUT alone does not size it); `@A.r` is the column.
    A = torch.arange(W, dtype=torch.float32).view(1, 1, W, 1).expand(1, H, W, C).contiguous()
    return {"A": A, "OUT": torch.ones(1, H, W, C)}


@pytest.mark.parametrize("code,want00", [
    ("@OUT[0, 0] *= vec4(2.0);", 2.0 ** 16),
    ("@OUT[0, 0] /= vec4(2.0);", 2.0 ** -16),
    ("@OUT[0, 0] -= vec4(1.0);", 1.0 - 16),
    ("@OUT[0, 0] += vec4(1.0);", 1.0 + 16),
    ("@OUT[0, 0] *= vec4(select(@A.r < 2.0, 2.0, 1.0));", 2.0 ** 8),
    ("@OUT[0, 0] /= vec4(select(@A.r < 1.0, 0.5, 1.0));", 2.0 ** 4),
], ids=["mul", "div", "sub", "add", "mul-some", "div-some"])
def test_scatter_compound_reduces_every_writer(code, want00):
    ref = run_tiers(code, _ones())
    out = ref["OUT"]
    assert out[0, 0, 0].tolist() == pytest.approx([want00] * 4, rel=1e-6)
    assert torch.equal(out[0, 1:], torch.ones(3, 4, 4))


def test_scatter_mul_on_a_mask_reduces_every_writer():
    M = torch.full((1, 4, 4), 3.0)
    ref = run_tiers("@M[ix * 0.5, 0] *= 2.0; @OUT = vec4(@M);", {"M": M})
    m = ref["M"][0]
    # Columns 0 and 1 each receive the 8 sources of two columns; 2 and 3 of row 0 receive none.
    assert m[0].tolist() == [3.0 * 2 ** 8, 3.0 * 2 ** 8, 3.0, 3.0]
    assert torch.equal(m[1:], torch.full((3, 4), 3.0))


def test_scatter_mul_without_collisions_is_the_elementwise_product():
    A, O = _img(seed=3), _img(seed=4)
    ref = run_tiers("@OUT[ix, iy] *= @A; @P = @Q; @P[ix, iy] /= @A;",
                    {"A": A, "OUT": O.clone(), "Q": O.clone(), "P": torch.zeros_like(O)})
    assert torch.equal(ref["OUT"], O * A)
    assert torch.equal(ref["P"], O / A)


def test_scatter_div_by_zero_uses_one_guard_on_both_tiers():
    ref = run_tiers("@OUT[0, 0] /= vec4(0.0);", _ones(2, 2))
    assert torch.isfinite(ref["OUT"]).all()


def test_scatter_increment_accumulates():
    ref = run_tiers("@M[0, 0]++; @OUT = vec4(@M);",
                    {"M": torch.zeros(1, 2, 3), "A": torch.zeros(1, 2, 3, 4)})
    assert ref["M"][0, 0, 0].item() == 6.0


def test_scatter_mul_under_a_mask_counts_live_sources_only():
    code = ("//!tex 0.25\nfloat a = @A.r;\n"
            "if (a > 0.5) { @OUT[0, 0] *= vec4(2.0); }\n")
    A = _img(seed=5)
    ref = run_tiers(code, {"A": A, "OUT": torch.ones(1, 4, 5, 4)})
    n = int((A[..., 0] > 0.5).sum())
    assert ref["OUT"][0, 0, 0, 0].item() == pytest.approx(2.0 ** n)


# ── Names resolve lexically: a shadow ends with its block, and a function reads the ─────
# variables visible where it is defined, never a caller's same-named local.

def _u():
    return torch.arange(5, dtype=torch.float32).div(4).view(1, 1, 5).expand(1, 4, 5)


@pytest.mark.parametrize("code,want", [
    ("float g = 1.0; float f() { return g; } float r = 0.0;"
     " if (1.0) { float g = 5.0; r = f(); } @OUT = vec4(r);", 1.0),
    ("float g = 1.0; float f() { return g; } float h(float g) { return f(); }"
     " @OUT = vec4(h(9.0));", 1.0),
    ("float g = 1.0; float f() { return g; } float h() { float g = 4.0; return f() + g; }"
     " @OUT = vec4(h());", 5.0),
    ("float g = 1.0; float f() { return g; } g = 3.0; @OUT = vec4(f());", 3.0),
    ("float g = 1.0; if (1.0) { float g = 5.0; } @OUT = vec4(g);", 1.0),
    ("float s = 1.0; for (int i = 0; i < 3; i++) { float s = 2.0 + i; } @OUT = vec4(s);", 1.0),
    ("float s = 1.0; int n = 3; while (n > 0) { float s = 9.0; n = n - 1; } @OUT = vec4(s);", 1.0),
    ("float g = 2.0; if (1.0) { float g = g * 3.0; @OUT = vec4(g); }", 6.0),
    ("float i = 7.0; for (int i = 0; i < 2; i++) { } @OUT = vec4(i);", 7.0),
    ("float g = 1.0; if (@A.r > 0.5) { vec3 g = vec3(2.0); } @OUT = vec4(g);", 1.0),
    ("float g = 1.0; float f() { return g; } float h() { return f() + 1.0; } @OUT = vec4(h());", 2.0),
    ("//!tex 0.25\nfloat g = 1.0; float f() { return g; } float h() { return f() + 1.0; }"
     " @OUT = vec4(h());", 2.0),
    ("float k = 2.0; float f() { float s = 0.0; for (int i = 0; i < 3; i++) { s = s + k; }"
     " return s; } @OUT = vec4(f());", 6.0),
    ("float t = 0.0; for (int i = 1; i < 3; i++) { float w = i * 1.0;"
     " float f() { return w; } t = t + f(); } @OUT = vec4(t);", 3.0),
], ids=["caller-block-shadow", "caller-param-shadow", "caller-local-shadow", "by-reference",
        "block-shadow-ends", "loop-shadow-ends", "while-shadow-ends", "shadow-init-reads-outer",
        "loop-counter-shadow", "per-pixel-typed-shadow", "nested-no-arg-call",
        "masked-nested-no-arg-call", "scalar-loop-reads-outer", "fn-in-loop-reads-loop-local"])
def test_names_resolve_lexically(code, want):
    ref = run_tiers(code, {"A": _img()})
    assert torch.allclose(ref["OUT"], torch.full_like(ref["OUT"], want))


def test_a_block_local_does_not_replace_a_builtin():
    ref = run_tiers("if (1.0) { float u = 7.0; } @OUT = vec4(u);", {"A": _img()})
    assert torch.allclose(ref["OUT"][..., 0], _u())


def test_masked_call_reads_the_definition_scope():
    code = ("//!tex 0.25\nfloat g = 1.0; float f() { return g; } float r = 0.0;\n"
            "if (@A.r > 0.5) { float g = 5.0; r = f(); }\n@OUT = vec4(r);")
    A = _img()
    ref = run_tiers(code, {"A": A})
    assert torch.equal(ref["OUT"][..., 0], (A[..., 0] > 0.5).float())


# ── A loop whose body writes its own counter runs the passes C would ─────────────────────
# The static and uniform-range fast paths precompute the counter's values, so they must not
# serve a loop whose body changes the counter.

@pytest.mark.parametrize("code,want", [
    ("float c = 0.0; for (int i = 0; i < 4; i++) { i = i + 1; c = c + 1.0; } @OUT = vec4(c);", 2.0),
    ("float c = 0.0; for (int i = 0; i < 100; i++) { i++; c = c + 1.0; } @OUT = vec4(c);", 50.0),
    ("float c = 0.0; for (int i = 0; i < 9; i = i + 1) { c = c + i; i = i * 2; } @OUT = vec4(c);", 11.0),
    ("float c = 0.0; for (int i = 0; i < $n; i++) { i = i + 2; c = c + 1.0; } @OUT = vec4(c);", 3.0),
    ("//!tex 0.25\nfloat c = 0.0; for (int i = 0; i < 4; i++) { i = i + 1; c = c + 1.0; }"
     " @OUT = vec4(c);", 2.0),
    ("float c = 0.0; for (int i = 0; i < 3; i++) { for (int j = 0; j < 2; j++) { c = c + 1.0; } }"
     " @OUT = vec4(c);", 6.0),
], ids=["static", "static-long", "static-mul", "uniform-bound", "masked", "inner-own-counter"])
def test_loop_body_writing_its_counter(code, want):
    ref = run_tiers(code, {"A": _img(), "n": 8})
    assert torch.allclose(ref["OUT"], torch.full_like(ref["OUT"], want))


# ── A scalar field bound to a vec2/vec3 variable broadcasts; it is never column-sliced ──

@pytest.mark.parametrize("code,n", [
    ("vec3 c = luma(@A); @OUT = vec4(c, 1.0);", 3),
    ("vec2 c = @M; @OUT = vec4(c, 0.0, 1.0);", 2),
    ("vec3 c = vec3(0.0); c = @M; @OUT = vec4(c, 1.0);", 3),
    ("vec3 c = vec3(0.0); c += @M; @OUT = vec4(c, 1.0);", 3),
    ("vec3 f(float x) { return x; } vec3 c = f(@M); @OUT = vec4(c, 1.0);", 3),
], ids=["decl-luma", "decl-mask", "assign", "compound", "return"])
def test_scalar_field_widens_to_the_declared_vector(code, n):
    A = _img(W=6)
    M = A[..., 1].clone()
    ref = run_tiers(code, {"A": A, "M": M})
    field = (A[..., 0] * 0.2126 + A[..., 1] * 0.7152 + A[..., 2] * 0.0722) if "luma" in code else M
    assert tuple(ref["OUT"].shape) == (1, 4, 6, 4)
    for ch in range(n):
        assert torch.allclose(ref["OUT"][..., ch], field, atol=1e-5)


# ── Casts: codegen prints numbers as the interpreter does, NaN/Inf included ─────────────

@pytest.mark.parametrize("code,want", [
    ("float b = 3e38; float x = b * 10.0; string s = string(x); @OUT = vec4(len(s));", 3.0),
    ("float b = 3e38; float x = b * 10.0 - b * 10.0; string s = string(x); @OUT = vec4(len(s));", 3.0),
    ("float x = @A.r * 0.0 + 3e38 * 10.0; string s = string(x); @OUT = vec4(len(s));", 3.0),
    ("string s = string(1.0 / 3.0); @OUT = vec4(len(s));", 8.0),
    ("string s = string(2.0); @OUT = vec4(len(s));", 1.0),
    ("float c = 0.0; float b = 1.0; for (int i = 0; i < 3; i++) { b = b / 3.0;"
     " string s = string(b); c = c + len(s); } @OUT = vec4(c);", 24.0),
    ("float c = 0.0; float b = 3e37; for (int i = 0; i < 3; i++) { b = b * 10.0;"
     " string s = string(b); c = c + len(s); } @OUT = vec4(c);", 45.0),
    ("float c = 0.0; float b = 3e37; for (int i = 0; i < 3; i++) { b = b * 10.0;"
     " c = c + int(b) * 0.0 + float(b) * 0.0; } @OUT = vec4(c);", float("nan")),
    ("float c = 0.0; float b = 3e30; for (int i = 0; i < 40; i++) { b = b * 1e30;"
     " c = c + int(b) * 0.0; } @OUT = vec4(c);", float("nan")),
], ids=["inf", "nan", "inf-field", "third", "whole", "loop-third", "loop-inf", "loop-int-inf",
        "scalar-loop-int-inf"])
def test_cast_of_any_float(code, want):
    ref = run_tiers(code, {"A": _img()})
    assert torch.allclose(ref["OUT"], torch.full_like(ref["OUT"], want), equal_nan=True)
