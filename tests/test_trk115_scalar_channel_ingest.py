"""TRK-115 (DATA-6 L-C2 F1) — a `[B,H,W,1]` scalar-field binding raises where the
same field bound as `[B,H,W]` cooks fine.

`tex_marshalling.infer_binding_type` types ANY 4-D tensor with `C == 1` as FLOAT
(`_spatial_channels_to_type` — TEX has no vec1 type, so a 1-channel image IS a
scalar field), but nothing normalised the tensor's RUNTIME rank to match: a raw
`[B,H,W,1]` binding stayed rank 4 all the way to every FLOAT consumer that
stacks components — the vec-constructor's flatten path (`interpreter.py`
`_eval_vec_constructor`) and `_ensure_spatial` — which all assume the `[B,H,W]`
rank a genuine MASK binding already carries. `@OUT = vec4(@A.rgb, @M);` with a
`[1,4,4,1]` `@M` therefore raised (`_ensure_spatial`'s
`tensor.shape[:len(spatial_shape)] == spatial_shape` check passes vacuously —
`[1,4,4,1][:3] == (1,4,4)` — so the `[1,4,4,1]` tensor is handed back UNCHANGED
into a `[1,4,4]` slot, and PyTorch's trailing-dim broadcast then lines the
binding's H up against the slot's W and raises), while the identical field
bound as `[1,4,4]` cooked normally. Reachable by any engine host handing over a
1-channel `[B,H,W,C]` image (the natural shape out of an EXR channel read); the
ComfyUI MASK wire is always `[B,H,W]`, so a ComfyUI cook never sees this shape.

Fixed at `tex_marshalling.to_fp32_if_int_image` — the single ingestion point
BOTH the interpreter's binding loop and codegen's `_contiguous_bindings` call —
by squeezing a `[B,H,W,1]` tensor to `[B,H,W]` before anything downstream sees
it, so the two spellings of the same field become the same tensor at ingest
and cook identically on every tier.

These rows drive `tex_engine.cook` directly (not the lower-level `run_both`
harness): `run_both`'s codegen leg builds its own bindings by hand and calls
`_invoke_cg` straight, bypassing `_contiguous_bindings` entirely — it does not
exercise the ingestion seam this fix lives at. `tex_engine.cook` is the one
entry point that always routes through the real ingestion for whichever tier
it dispatches to.

ComfyUI-invisible because: the ComfyUI MASK/IMAGE wires never hand TEX a
`[B,H,W,1]` tensor (a MASK is already rank 3; an IMAGE is never 1-channel), so
this squeeze never fires on any cook a ComfyUI user can produce — it only
normalises a shape that a non-ComfyUI engine host can construct.
"""
from helpers import *

from TEX_Wrangle import tex_engine

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


def test_trk115_ingest_squeezes_rank4_c1(r: SubTestResult):
    """Unit-level pin directly on the fixed function, so a future edit to
    `to_fp32_if_int_image` gets a fast, precise signal beside the end-to-end ones
    above."""
    print("\n--- TRK-115: to_fp32_if_int_image squeezes [B,H,W,1] to [B,H,W] ---")
    from TEX_Wrangle.tex_marshalling import to_fp32_if_int_image
    t = torch.rand(1, 8, 8, 1)
    out = to_fp32_if_int_image(t)
    if out.shape != (1, 8, 8):
        r.fail("a [1,8,8,1] tensor is squeezed to [1,8,8]", f"got shape {tuple(out.shape)}")
    elif not torch.equal(out, t.squeeze(-1)):
        r.fail("the squeeze preserves values", "value mismatch after squeeze")
    else:
        r.ok("squeezed to [1,8,8], values preserved")

    # Control: a genuine multi-channel image (C>1) must be untouched by this branch.
    t3 = torch.rand(1, 8, 8, 3)
    out3 = to_fp32_if_int_image(t3)
    if out3.shape != (1, 8, 8, 3) or not torch.equal(out3, t3):
        r.fail("a C>1 image binding is left untouched", f"got shape {tuple(out3.shape)}")
    else:
        r.ok("a 3-channel [1,8,8,3] binding is untouched (control)")
