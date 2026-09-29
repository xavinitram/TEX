"""Interpreter statement execution: in-place assignment, array element writes, scatter
writes and the memory-sweep LRUs."""
import pytest
import torch

from helpers import *
from TEX_Wrangle.tex_runtime.interpreter import InterpreterError
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_runtime.interpreter import Interpreter


def _img(H=4, W=5, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, H, W, 4, generator=g)


def _cook(src, bindings, precision="fp32", interp=None):
    bt = {n: _infer_binding_type(v) for n, v in bindings.items()}
    prog = parse_and_split(src, bt)
    ch = TypeChecker(binding_types=bt, source=src)
    tm = ch.check(prog)
    names = sorted(ch.assigned_bindings)
    res = (interp or Interpreter()).execute(prog, dict(bindings), tm, device="cpu",
                                            output_names=names, precision=precision)
    return res["OUT"] if names == ["OUT"] else res


# -- in-place fast path -------------------------------------------------------------------

def test_in_place_shape_mismatch_evaluates_the_operand_once(monkeypatch):
    calls = []
    orig = Interpreter._eval_function_call

    def counting(self, node):
        calls.append(node.name)
        return orig(self, node)

    monkeypatch.setattr(Interpreter, "_eval_function_call", counting)
    img = _img()
    # a vec accumulator meets a scalar field: the shapes differ, so the fast path bails
    out = _cook("vec3 c = vec3(0.0); c = c + sin(@A.r); @OUT = vec4(c, 1.0);", {"A": img})
    assert calls.count("sin") == 1
    assert torch.allclose(out[..., 0], torch.sin(img[..., 0]), atol=1e-6)
    calls.clear()
    out = _cook("vec3 c = vec3(1.0); c = sin(@A.r) * c; @OUT = vec4(c, 1.0);", {"A": img})
    assert calls.count("sin") == 1
    assert torch.allclose(out[..., 1], torch.sin(img[..., 0]), atol=1e-6)


def test_in_place_keeps_a_mismatched_operand_order():
    img = _img()
    out = _cook("vec3 c = vec3(2.0); c = c - @A.r; @OUT = vec4(c, 1.0);", {"A": img})
    assert torch.allclose(out[..., 0], 2.0 - img[..., 0], atol=1e-6)
    out = _cook("vec3 c = vec3(2.0); c = c / (@A.r + 1.0); @OUT = vec4(c, 1.0);", {"A": img})
    assert torch.allclose(out[..., 2], 2.0 / (img[..., 0] + 1.0), atol=1e-6)


def test_in_place_add_promotes_like_the_out_of_place_path_in_fp16():
    img = _img(W=8)
    src = "float x = @A.r; x = x + u; @OUT = vec4(x, 0.0, 0.0, 1.0);"
    out = _cook(src, {"A": img}, precision="fp16")[..., 0].float()
    x16 = img[..., 0].half().float()
    u = torch.arange(8, dtype=torch.float32).view(1, 1, 8) / 7.0
    assert torch.allclose(out, (x16 + u), atol=1e-6)      # fp32 sum, not fp16-rounded


# -- array element writes -----------------------------------------------------------------

def test_per_pixel_array_write_of_an_fp32_value_in_fp16():
    img = _img()
    src = "float a[3]; int k = int(@A.r * 2.9); a[k] = u; @OUT = vec4(a[0], a[1], a[2], 1.0);"
    out = _cook(src, {"A": img}, precision="fp16")
    assert out.shape == (1, 4, 5, 4)
    assert torch.isfinite(out.float()).all()


def test_string_array_write_with_non_finite_index_is_clamped():
    src = ('string s[] = {"a", "bb", "ccc"}; float k = @K; s[k] = "zz"; '
           '@OUT = vec4(float(len(s[0])), float(len(s[2])), 0.0, 1.0);')
    out = _cook(src, {"A": _img(), "K": float("nan")})
    assert out[0, 0, 0, 0].item() == 2.0 and out[0, 0, 0, 1].item() == 3.0
    src = ('string s[] = {"a", "bb", "ccc"}; float k = @K; s[k] = "zzzz"; '
           '@OUT = vec4(float(len(s[2])), 0.0, 0.0, 1.0);')
    assert _cook(src, {"A": _img(), "K": float("inf")})[0, 0, 0, 0].item() == 4.0


def test_string_array_write_floors_a_fractional_index_like_the_read():
    src = ('string s[] = {"a", "bb", "ccc"}; float k = 1.7; s[k] = "zzzz"; '
           '@OUT = vec4(float(len(s[1])), float(len(s[2])), 0.0, 1.0);')
    out = _cook(src, {"A": _img()})
    assert out[0, 0, 0, 0].item() == 4.0 and out[0, 0, 0, 1].item() == 3.0


@pytest.mark.parametrize("src", [
    "mat3 a[2]; a[0] = mat3(2.0); mat3 m = a[0]; @OUT = vec4(m * vec3(1.0, 1.0, 1.0), 1.0);",
    "mat4 a[2]; a[int(@A.r * 2.0)] = mat4(2.0); mat4 m = a[0]; @OUT = m * vec4(1.0, 1.0, 1.0, 1.0);",
])
def test_matrix_arrays_are_a_compile_error(src):
    """Arrays of matrices are not part of the language (E3101); the checker refuses them
    before the interpreter's own matrix-array guard can be reached."""
    with pytest.raises(Exception, match="Arrays of 'mat"):
        _cook(src, {"A": _img()})


# -- scatter write --------------------------------------------------------------------------

@pytest.mark.parametrize("W", [2, 3, 4])
def test_scatter_of_a_scalar_field_on_a_narrow_frame(W):
    img = _img(H=4, W=W)
    out = _cook("@OUT[ix, iy] = @A.r;", {"A": img})
    assert torch.allclose(out, img[..., 0], atol=1e-6)


# -- memory sweep vs a live cook ----------------------------------------------------------

def test_lru_sweep_between_lookup_and_touch_does_not_fail_the_cook():
    it = Interpreter()
    img = _img()
    _cook("@OUT = vec4(u, v, 0.0, 1.0);", {"A": img}, interp=it)      # fill the LRUs
    real = type(it._builtins_lru).move_to_end

    class Sweeping(type(it._builtins_lru)):
        def move_to_end(self, key, last=True):
            self.clear()                                              # the other thread's sweep
            return real(self, key, last)

    it._builtins_lru = Sweeping(it._builtins_lru)
    it._coord_ramp_lru = Sweeping(it._coord_ramp_lru)
    out = _cook("@OUT = vec4(u, v, 0.0, 1.0);", {"A": img}, interp=it)
    assert out.shape == (1, 4, 5, 4)


# -- loops -------------------------------------------------------------------------------

def test_static_range_too_long_for_ssize_t_is_the_loop_limit_error():
    src = "float s = 0.0; for (int i = 0; i < @N; i++) { s += 1.0; } @OUT = vec4(s, 0.0, 0.0, 1.0);"
    with pytest.raises(InterpreterError) as ei:
        _cook(src, {"A": _img(), "N": 1e19})
    assert ei.value.code == "E6010"


def test_while_needing_exactly_the_limit_runs():
    src = ("int n = 0; while (n < 1024) { n = n + 1; } "
           "@OUT = vec4(float(n) / 1024.0, 0.0, 0.0, 1.0);")
    assert _cook(src, {"A": _img()})[0, 0, 0, 0].item() == 1.0


def test_while_needing_one_more_than_the_limit_fails():
    src = "int n = 0; while (n < 1025) { n = n + 1; } @OUT = vec4(float(n), 0.0, 0.0, 1.0);"
    with pytest.raises(InterpreterError) as ei:
        _cook(src, {"A": _img()})
    assert ei.value.code == "E6010"


def test_general_for_needing_exactly_the_limit_runs():
    # `lim` is written in the body, so the bound is not uniform and the general path runs
    src = ("int lim = 1024; float s = 0.0; "
           "for (int i = 0; i < lim; i++) { lim = 1024; s += 1.0; } "
           "@OUT = vec4(s / 1024.0, 0.0, 0.0, 1.0);")
    assert _cook(src, {"A": _img()})[0, 0, 0, 0].item() == 1.0


def test_uniform_range_is_not_used_when_a_bound_calls_a_function_that_reads_a_written_var():
    src = """
float lim = 4.0;
float getlim() { return lim; }
float s = 0.0;
for (int i = 0; i < getlim(); i++) { lim = 2.0; s += 1.0; }
@OUT = vec4(s / 10.0, 0.0, 0.0, 1.0);
"""
    assert _cook(src, {"A": _img()})[0, 0, 0, 0].item() == pytest.approx(0.2)


# -- literals and per-cook state ---------------------------------------------------------

def test_signed_zero_literals_do_not_share_a_cache_entry():
    from TEX_Wrangle.tex_compiler.ast_nodes import NumberLiteral
    it = Interpreter()
    it.device = torch.device("cpu")
    it._device_str = "cpu"
    it._dtype = torch.float32
    pos = it._eval_number_literal(NumberLiteral(value=0.0))
    neg = it._eval_number_literal(NumberLiteral(value=-0.0))
    assert torch.signbit(neg).item() and not torch.signbit(pos).item()
    assert torch.signbit(it._eval_number_literal(NumberLiteral(value=-0.0))).item()


def test_interpreter_drops_a_cooks_tensors_when_it_ends():
    it = Interpreter()
    out = _cook("vec3 c = @A.rgb * 2.0; @OUT = vec4(c, 1.0);", {"A": _img()}, interp=it)
    assert it.env == {} and it.bindings == {}
    assert out.shape == (1, 4, 5, 4)
    with pytest.raises(Exception):
        _cook("float y = @P[0, 0]; @OUT = vec4(y, 0.0, 0.0, 1.0);", {"A": _img(), "P": 1.0},
              interp=it)
    assert it.env == {} and it.bindings == {}
