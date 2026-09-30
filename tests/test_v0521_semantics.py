"""Tier-parity rows: scalar-loop arithmetic rounds like the interpreter's fp32, and a
one-channel image binding `[B,H,W,1]` enters both tiers as the mask `[B,H,W]`.

Each row runs the interpreter (the oracle) and the codegen-only route on copies of the same
bindings; codegen must serve the program. Loop rows compare BIT-exactly: a scalar loop runs
the same fp32 operations on both tiers, so a looser tolerance would hide the drift these rows
exist to catch (a long accumulation drifts by 1e-3, and a comparison can flip a branch).
"""
import pytest
import torch

from TEX_Wrangle.tex_cache import get_cache, parse_and_split
from TEX_Wrangle.tex_marshalling import infer_binding_type, to_fp32_if_int_image
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


# ── A one-channel image binding enters as a plain mask ──────────────────────────────────

def _mask_bindings(H, W, rank4):
    g = torch.Generator().manual_seed(958)
    m = torch.rand(1, H, W, generator=g)
    a = torch.rand(1, H, W, 4, generator=g)
    return {"M": m.unsqueeze(-1) if rank4 else m, "A": a}


@pytest.mark.parametrize("H,W", [(4, 5), (4, 4)], ids=["H!=W", "H==W"])
@pytest.mark.parametrize("code", [
    "float f = u * @M; @OUT = f;",
    "@OUT = @M;",
    "@OUT = vec4(@A.rgb, @M);",
    "@OUT = @M * @A;",
    "float f = @M; if (f > 0.5) { f = 1.0 - f; } @OUT = f;",
], ids=["u*M", "passthrough", "vec-constructor", "mask*image", "per-pixel-if"])
def test_rank4_single_channel_binding_is_a_mask(code, H, W):
    """A `[B,H,W,1]` binding cooks exactly as the `[B,H,W]` mask does, on both tiers:
    same rank, same values. Before, codegen broadcast `u * @M` to `[B,H,W,W]` when H==W."""
    ref3, got3 = both_tiers(code, _mask_bindings(H, W, rank4=False))
    ref4, got4 = both_tiers(code, _mask_bindings(H, W, rank4=True))
    for name, want in ref3.items():
        for tier, res in (("interpreter", ref4), ("codegen", got4)):
            assert tuple(res[name].shape) == tuple(want.shape), (
                f"{tier}: {tuple(res[name].shape)} != {tuple(want.shape)}")
            assert torch.equal(res[name], want), f"{tier}: values differ"
        assert torch.equal(got3[name], want)


def test_ingest_squeezes_only_a_single_channel_image():
    m4 = torch.rand(1, 3, 3, 1)
    assert tuple(to_fp32_if_int_image(m4).shape) == (1, 3, 3)
    assert torch.equal(to_fp32_if_int_image(m4), m4[..., 0])
    for keep in (torch.rand(1, 3, 3, 3), torch.rand(1, 3, 3), torch.rand(3, 1)):
        assert to_fp32_if_int_image(keep) is keep
    # An integer one-channel image is cast and squeezed in the same step.
    i4 = torch.ones(1, 2, 2, 1, dtype=torch.int64)
    out = to_fp32_if_int_image(i4)
    assert out.dtype is torch.float32 and tuple(out.shape) == (1, 2, 2)


# ── The default-tier stencil route runs at the cook's precision ─────────────────────────

_BOX = ("vec3 acc = vec3(0.0); float cnt = 0.0;"
        "for (int dy = -2; dy <= 2; dy = dy + 1) { for (int dx = -2; dx <= 2; dx = dx + 1) {"
        "acc = acc + fetch(@A, ix + dx, iy + dy).rgb; cnt = cnt + 1.0; } }"
        "@OUT = vec4(acc / cnt, 1.0);")


def _stencil_cook(precision, device, monkeypatch):
    from TEX_Wrangle import tex_engine
    seen = []
    real = tex_engine._codegen_only_execute

    def spy(*a, **kw):
        seen.append(kw.get("precision", "fp32"))
        return real(*a, **kw)

    monkeypatch.setattr(tex_engine, "_codegen_only_execute", spy)
    g = torch.Generator().manual_seed(112)
    img = torch.rand(1, 40, 48, 3, generator=g).to(device)
    tier_trace.reset()
    res = tex_engine.cook(_BOX, {"A": img}, device_mode=device,
                          compile_mode="none", precision=precision)
    rec = tier_trace.last()
    assert rec is not None and rec.tier == "codegen", "the stencil route did not serve"
    return res.outputs["OUT"], seen


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_stencil_route_gets_the_cook_precision(precision, monkeypatch):
    _out, seen = _stencil_cook(precision, "cpu", monkeypatch)
    assert seen == [precision], f"stencil route ran at {seen}, the cook at {precision}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp16_stencil_route_parity_on_cuda(monkeypatch):
    """The fp16 stencil route on the GPU agrees with the interpreter's fp16 cook within the
    8-bit quantum (the fp16 contract), and with the fp32 cook within fp16's own rounding."""
    from TEX_Wrangle import tex_engine
    got, seen = _stencil_cook("fp16", "cuda", monkeypatch)
    assert seen == ["fp16"]
    g = torch.Generator().manual_seed(112)
    img = torch.rand(1, 40, 48, 3, generator=g).cuda()
    prep = tex_engine.prepare(_BOX, {"A": img}, device_mode="cuda",
                              compile_mode="none", precision="fp16")
    ref16 = Interpreter().execute(prep.ctx.program, dict(prep.ctx.bindings), prep.ctx.type_map,
                                  device="cuda", output_names=["OUT"], precision="fp16")["OUT"]
    ref32, _ = _stencil_cook("fp32", "cuda", monkeypatch)
    assert torch.isfinite(got).all()
    assert (got.float() - ref16.float()).abs().max().item() < 1.0 / 255.0
    assert (got.float() - ref32.float()).abs().max().item() < 1e-3
