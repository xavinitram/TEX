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


# ── A static-range loop counter reads back nothing on the device ───────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_loop_counter_carries_its_host_reading_on_cuda():
    """An array index and a kernel radius built from the counter of a loop too long to
    unroll read the counter's host value, not the device: no `.item()` per pass."""
    code = ("float w[24]; for (int i = 0; i < 24; i++) { w[i] = i * 0.01; } vec3 s = vec3(0.0);"
            "for (int i = 0; i < 24; i++) { s += @A.rgb * w[i]; }"
            "for (int i = 1; i < 11; i++) { s += gauss_blur(@A, i).rgb * 0.01; }"
            "@OUT = vec4(s, 1.0);")
    g = torch.Generator().manual_seed(928)
    img = torch.rand(1, 16, 16, 4, generator=g)
    bt = {"A": infer_binding_type(img)}
    program = parse_and_split(code, bt)
    program, tm, *_ = get_cache().compile_ast(program, bt, source=code)
    interp = Interpreter()
    ref = interp.execute(program, {"A": img}, tm, device="cpu", output_names=["OUT"])["OUT"]
    interp.execute(program, {"A": img.cuda()}, tm, device="cuda", output_names=["OUT"])
    real, reads = torch.Tensor.item, []

    def item(self):
        if self.device.type != "cpu":
            reads.append(tuple(self.shape))
        return real(self)

    torch.Tensor.item = item
    try:
        got = interp.execute(program, {"A": img.cuda()}, tm, device="cuda",
                             output_names=["OUT"])["OUT"]
    finally:
        torch.Tensor.item = real
    assert reads == [], f"{len(reads)} device reads in the cook"
    assert torch.allclose(got.cpu(), ref, atol=1e-4)


# ── sample_mip takes the untouched u/v builtins as the identity grid without a readback ─

def _mip(code, img, device):
    bt = {"A": infer_binding_type(img)}
    program = parse_and_split(code, bt)
    program, tm, _r, _assigned, _p, used = get_cache().compile_ast(program, bt, source=code)
    b = {"A": img.to(device)}
    ref = Interpreter().execute(program, dict(b), tm, device=device, output_names=["OUT"])["OUT"]
    tier_trace.reset()
    got = _codegen_only_execute(program, dict(b), tm, device, output_names=["OUT"],
                                used_builtins=used, fingerprint=None, time_context=None)["OUT"]
    assert tier_trace.last().tier == "codegen"
    return ref, got


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))])
def test_sample_mip_identity_builtins_match_the_probed_grid(device):
    """The marked builtins take the identity route; `u + 0.0` (unmarked, same values) is
    proved identity by the probe. Both tiers give the same picture either way."""
    img = torch.rand(1, 32, 40, 4, generator=torch.Generator().manual_seed(931))
    direct = _mip("@OUT = sample_mip(@A, u, v, 1.5);", img, device)
    probed = _mip("float uu = u + 0.0; float vv = v + 0.0; @OUT = sample_mip(@A, uu, vv, 1.5);",
                  img, device)
    for a, b in zip(direct, probed):
        assert torch.equal(a, b)
    assert torch.equal(direct[0], direct[1])


def test_identity_mark_is_only_on_the_full_extent_ramp():
    """Both tiers mark `u`/`v` over the whole image, and neither marks a window's ramp."""
    from TEX_Wrangle.tex_runtime.stdlib_core import _IDENTITY_RAMP_ATTR
    from TEX_Wrangle.tex_runtime.compiled import _build_codegen_env
    code = "@OUT = u; @V = v;"
    img = torch.rand(1, 6, 8, 4)
    bt = {"A": infer_binding_type(img)}
    program = parse_and_split(code, bt)
    program, tm, *_ = get_cache().compile_ast(program, bt, source=code)
    for roi, want_u in ((None, True), ((2, 0, 4, 6, 8, 6), False)):
        env, _sp, _ = _build_codegen_env(program, {"A": img}, torch.device("cpu"), 0,
                                         used_builtins={"u", "v"}, roi=roi)
        out = Interpreter().execute(program, {"A": img if roi is None else img[:, :, 2:6]}, tm,
                                    device="cpu", output_names=["OUT", "V"], roi=roi)
        for tier, uu, vv in (("codegen", env["u"], env["v"]),
                             ("interpreter", out["OUT"], out["V"])):
            assert getattr(uu, _IDENTITY_RAMP_ATTR, False) is want_u, (tier, roi)
            assert getattr(vv, _IDENTITY_RAMP_ATTR, False) is True, (tier, roi)  # full height


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sample_mip_on_the_builtin_grid_reads_nothing_back():
    code = "@OUT = sample_mip(@A, u, v, 1.0);"
    img = torch.rand(1, 32, 40, 4).cuda()
    bt = {"A": infer_binding_type(img)}
    program = parse_and_split(code, bt)
    program, tm, *_ = get_cache().compile_ast(program, bt, source=code)
    interp = Interpreter()
    interp.execute(program, {"A": img}, tm, device="cuda", output_names=["OUT"])
    real, reads = torch.Tensor.item, []

    def item(self):
        if self.device.type != "cpu":
            reads.append(tuple(self.shape))
        return real(self)

    torch.Tensor.item = item
    try:
        interp.execute(program, {"A": img}, tm, device="cuda", output_names=["OUT"])
    finally:
        torch.Tensor.item = real
    assert reads == [], f"{len(reads)} device reads"
