"""TRK-115 (DATA-6 L-C2 F1) — a `[B,H,W,1]` scalar-field binding raised where the
same field bound as `[B,H,W]` cooked fine.

`tex_marshalling.infer_binding_type` types a 4-D tensor with `C == 1` as FLOAT (TEX has no
vec1 type, so a one-channel image IS a scalar field), but nothing normalised its runtime
rank, so `@OUT = vec4(@A.rgb, @M);` with a `[1,4,4,1]` `@M` raised.

The binding now enters both tiers as the mask `[B,H,W]` at ingest
(`tex_marshalling.to_fp32_if_int_image`, the seam both tiers call), so every operation sees
one rank for a scalar field. A one-channel passthrough therefore cooks to `[B,H,W]`; every
egress (MASK, IMAGE, LATENT) maps both ranks to the same wire value. `_ensure_spatial` still
squeezes a `[B,H,W,1]` value against a `[B,H,W]` target, for values made mid-program.
"""
from helpers import *

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime.interpreter import _ensure_spatial

_CODE = "@OUT = vec4(@A.rgb, @M);"


def _fields(seed=115):
    torch.manual_seed(seed)
    a = torch.rand(1, 4, 4, 3)
    torch.manual_seed(seed + 1)
    m3 = torch.rand(1, 4, 4)
    return a, m3


def test_trk115_rank4_binding_does_not_raise(r: SubTestResult):
    """The verbatim TRK-115 repro: a [B,H,W,1] @M binding must not crash a real
    cook (the bug was a raise, not a wrong value)."""
    print("\n--- TRK-115: a [B,H,W,1] scalar-field binding does not raise ---")
    a, m3 = _fields()
    m4 = m3.unsqueeze(-1)
    assert _infer_binding_type(m4) == TEXType.FLOAT, \
        "a 4-D C==1 binding must still type FLOAT (unchanged policy)"
    try:
        res = tex_engine.cook(_CODE, {"A": a.clone(), "M": m4.clone()},
                              device_mode="cpu", compile_mode="none")
        r.ok(f"cooked without raising, OUT shape {tuple(res.outputs['OUT'].shape)}")
    except Exception as e:
        r.fail("a [B,H,W,1] @M binding cooks without raising", f"{type(e).__name__}: {e}")


def test_trk115_rank4_and_rank3_bindings_are_bit_exact(r: SubTestResult):
    """The two spellings of the same mask ([B,H,W,1] vs [B,H,W]) must cook to the
    SAME pixels — not merely both avoid crashing — on the interpreter tier AND
    on the torch_compile tier (invariant 2: whichever tier actually serves it)."""
    print("\n--- TRK-115: [B,H,W,1] and [B,H,W] bindings of the same field agree ---")
    a, m3 = _fields()
    m4 = m3.unsqueeze(-1)
    for mode in ("none", "torch_compile"):
        try:
            res_4d = tex_engine.cook(_CODE, {"A": a.clone(), "M": m4.clone()},
                                     device_mode="cpu", compile_mode=mode)
            res_3d = tex_engine.cook(_CODE, {"A": a.clone(), "M": m3.clone()},
                                     device_mode="cpu", compile_mode=mode)
        except Exception as e:
            r.fail(f"both spellings cook without raising (compile_mode={mode})",
                  f"{type(e).__name__}: {e}")
            continue
        diff = (res_4d.outputs["OUT"] - res_3d.outputs["OUT"]).abs().max().item()
        if diff != 0.0:
            r.fail(f"[B,H,W,1] and [B,H,W] bindings match bit-exactly (compile_mode={mode})",
                  f"max abs diff {diff}")
        else:
            r.ok(f"compile_mode={mode}: the two spellings are bit-exact")


def test_trk115_passthrough_is_the_mask(r: SubTestResult):
    """A one-channel passthrough `@OUT = @A;` cooks to the `[B,H,W]` mask, values unmoved."""
    print("\n--- TRK-115: a 1-channel passthrough cooks to the [B,H,W] mask ---")
    torch.manual_seed(115)
    m4 = torch.rand(1, 4, 4, 1)
    try:
        res = tex_engine.cook("@OUT = @A;", {"A": m4.clone()}, device_mode="cpu")
    except Exception as e:
        r.fail("a 1-channel passthrough still cooks", f"{type(e).__name__}: {e}")
        return
    out = res.outputs["OUT"]
    if tuple(out.shape) != (1, 4, 4):
        r.fail("a 1-channel passthrough cooks to [B,H,W]", f"got shape {tuple(out.shape)}")
    elif not torch.equal(out, m4[..., 0]):
        r.fail("a 1-channel passthrough's VALUES are unmoved", "value mismatch")
    else:
        r.ok("[B,H,W,1] passthrough cooks to the [B,H,W] mask, values unmoved")


def test_trk115_ensure_spatial_squeezes_only_at_point_of_use(r: SubTestResult):
    """Unit-level pin directly on the fixed function: `_ensure_spatial` squeezes
    a `[B,H,W,1]` tensor reconciled against a `[B,H,W]` target, and only that
    shape — a genuine multi-channel or already-matching tensor is untouched."""
    print("\n--- TRK-115: _ensure_spatial squeezes [B,H,W,1] against a [B,H,W] target ---")
    t = torch.rand(1, 8, 8, 1)
    out = _ensure_spatial(t, (1, 8, 8))
    if out.shape != (1, 8, 8):
        r.fail("a [1,8,8,1] tensor reconciled against (1,8,8) is squeezed",
              f"got shape {tuple(out.shape)}")
    elif not torch.equal(out, t.squeeze(-1)):
        r.fail("the squeeze preserves values", "value mismatch after squeeze")
    else:
        r.ok("squeezed to [1,8,8] at the point of use, values preserved")

    # Controls: nothing else about _ensure_spatial's existing behaviour moves.
    t_already = torch.rand(1, 8, 8)
    if not torch.equal(_ensure_spatial(t_already, (1, 8, 8)), t_already):
        r.fail("an already-matching [1,8,8] tensor is returned unchanged", "value/identity mismatch")
    else:
        r.ok("an already-matching [1,8,8] tensor is untouched (control)")

    t3 = torch.rand(1, 8, 8, 3)
    out3 = _ensure_spatial(t3, (1, 8, 8))
    if out3.shape != (1, 8, 8, 3) or not torch.equal(out3, t3):
        r.fail("a C>1 tensor reconciled against (1,8,8) is left untouched",
              f"got shape {tuple(out3.shape)}")
    else:
        r.ok("a 3-channel [1,8,8,3] tensor is untouched (control)")

    scalar = torch.tensor(0.5)
    out_scalar = _ensure_spatial(scalar, (1, 8, 8))
    if tuple(out_scalar.shape) != (1, 8, 8):
        r.fail("a 0-dim scalar still expands to the full spatial shape (control, unchanged path)",
              f"got shape {tuple(out_scalar.shape)}")
    else:
        r.ok("a 0-dim scalar still expands normally (control)")
