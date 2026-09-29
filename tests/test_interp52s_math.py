"""Scalar-math builtins: a vec against a per-pixel field, a uniform vec against a field,
rank-3 fields that look like vecs, and the fp16 epsilon guards."""
import pytest
import torch

from helpers import *
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib as S
import test_lang_l5_codegen_masking as L5

PRAGMA = "//!tex 0.25\n"


def _img(H=4, W=5, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, H, W, 4, generator=g)


# -- a vec against a scalar field ----------------------------------------------------------

_MIXED = {
    "max(@A.rgb, @A.a)": lambda a, f: torch.maximum(a, f),
    "min(@A.rgb, @A.a)": lambda a, f: torch.minimum(a, f),
    "pow(@A.rgb, @A.a)": lambda a, f: torch.pow(a, f),
    "atan2(@A.rgb, @A.a)": lambda a, f: torch.atan2(a, f),
    "hypot(@A.rgb, @A.a)": lambda a, f: torch.hypot(a, f),
    "step(@A.a, @A.rgb)": lambda a, f: (a >= f).float(),
    "clamp(@A.rgb, @A.a * 0.1, @A.a)": lambda a, f: torch.minimum(torch.maximum(a, f * 0.1), f),
}


@pytest.mark.parametrize("expr", list(_MIXED))
def test_vec_with_field_builtin_broadcasts(expr):
    img = _img()
    rgb, f = img[..., :3], img[..., 3:4]
    out = compile_and_run(f"vec3 q = {expr}; @OUT = vec4(q, 1.0);", {"A": img})
    want = _MIXED[expr](rgb, f) if "step" not in expr else (rgb >= f).float()
    assert torch.allclose(out[..., :3], want, atol=1e-6)


def test_vec_with_field_smoothstep_fit_mod_sdiv_spow():
    img = _img()
    rgb, f = img[..., :3], img[..., 3:4]
    cases = {
        "smoothstep(@A.a * 0.1, @A.a, @A.rgb)": S.fn_smoothstep(f * 0.1, f, rgb),
        "fit(@A.rgb, 0.0, 1.0, @A.a, 1.0)": None,
        "mod(@A.rgb, @A.a + 0.5)": torch.fmod(rgb, f + 0.5),
        "sdiv(@A.rgb, @A.a)": rgb / f,
        "spow(@A.rgb, @A.a)": torch.pow(rgb, f),
    }
    for expr, want in cases.items():
        out = compile_and_run(f"vec3 q = {expr}; @OUT = vec4(q, 1.0);", {"A": img})
        assert out.shape == (1, 4, 5, 4), expr
        if want is not None:
            assert torch.allclose(out[..., :3], want, atol=1e-5), expr
        assert torch.isfinite(out).all(), expr


# -- a uniform vec against a field ---------------------------------------------------------

def test_sincos_of_a_uniform_multiplies_a_vec_field():
    img = _img()
    src = "vec2 sc = sincos(0.5); vec2 q = @A.rg * sc; @OUT = vec4(q.x, q.y, 0.0, 1.0);"
    iout, cout, names = L5.cook_both(PRAGMA + src, {"A": img}, masked=False)
    sin_c = torch.tensor([torch.sin(torch.tensor(0.5)).item(), torch.cos(torch.tensor(0.5)).item()])
    want = img[..., :2] * sin_c
    assert torch.allclose(iout["OUT"][..., :2], want, atol=1e-6)
    L5.assert_bitwise("sincos", iout, cout, names)


def test_sincos_of_a_uniform_scales_by_a_field():
    img = _img()
    src = "vec2 sc = sincos(0.5); vec2 q = sc * @A.a; @OUT = vec4(q.x, q.y, 0.0, 1.0);"
    out = compile_and_run(src, {"A": img})
    assert torch.allclose(out[..., 0], torch.sin(torch.tensor(0.5)) * img[..., 3], atol=1e-6)
    assert torch.allclose(out[..., 1], torch.cos(torch.tensor(0.5)) * img[..., 3], atol=1e-6)


# -- rank-3 fields are never vecs ----------------------------------------------------------

@pytest.mark.parametrize("W", [2, 3, 4])
def test_scalar_field_is_not_reduced_by_vector_builtins(W):
    f = torch.rand(1, 4, W) - 0.5
    assert torch.equal(S.fn_length(f), f.abs())
    assert torch.equal(S.fn_distance(f, torch.zeros_like(f)), f.abs())
    assert torch.equal(S.fn_normalize(f), torch.sign(f))


def test_real_vectors_still_reduce():
    v = torch.tensor([3.0, 4.0])
    assert S.fn_length(v).item() == pytest.approx(5.0)
    vf = torch.rand(1, 4, 5, 3)
    assert S.fn_length(vf).shape == (1, 4, 5)


# -- fp16 epsilon guards -------------------------------------------------------------------

H = torch.float16


def _h(*v):
    return torch.tensor(v, dtype=H)


def test_sdiv_by_zero_is_zero_in_fp16():
    out = S.fn_sdiv(_h(1.0, 2.0), _h(0.0, 4.0))
    assert out.tolist() == [0.0, 0.5]


def test_normalize_zero_vector_in_fp16_is_zero():
    z = torch.zeros(1, 2, 2, 3, dtype=H)
    assert torch.equal(S.fn_normalize(z), z)


def test_spow_zero_with_negative_exponent_in_fp16_is_zero():
    out = S.fn_spow(_h(0.0, 2.0), _h(-1.0, 1.0))
    assert out.tolist() == [0.0, 2.0]


@pytest.mark.parametrize("fn", ["fn_log", "fn_log2", "fn_log10"])
def test_log_of_zero_in_fp16_is_finite(fn):
    assert torch.isfinite(getattr(S, fn)(_h(0.0))).all()


def test_fit_and_smoothstep_with_equal_edges_in_fp16_are_not_nan():
    assert torch.isfinite(S.fn_fit(_h(0.5), _h(0.5), _h(0.5), _h(0.0), _h(1.0))).all()
    assert torch.isfinite(S.fn_smoothstep(_h(0.5), _h(0.5), _h(0.5))).all()


def test_fp32_guards_are_unchanged():
    assert S.fn_sdiv(torch.tensor(1.0), torch.tensor(0.0)).item() == 0.0
    assert S.fn_log(torch.tensor(0.0)).item() == pytest.approx(-18.420681, abs=1e-4)
    assert S.fn_normalize(torch.zeros(3)).tolist() == [0.0, 0.0, 0.0]
