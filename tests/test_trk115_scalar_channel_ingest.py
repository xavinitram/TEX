"""TRK-115 (DATA-6 L-C2 F1) — a `[B,H,W,1]` scalar-field binding raises where the
same field bound as `[B,H,W]` cooks fine.

`tex_marshalling.infer_binding_type` types ANY 4-D tensor with `C == 1` as FLOAT
(`_spatial_channels_to_type` — TEX has no vec1 type, so a 1-channel image IS a
scalar field), but nothing normalised the tensor's RUNTIME rank to match: a raw
`[B,H,W,1]` binding stayed rank 4 all the way to every FLOAT consumer that
stacks components — the vec-constructor's flatten path (`interpreter.py`
`_eval_vec_constructor`) and `_ensure_spatial`. `@OUT = vec4(@A.rgb, @M);` with a
`[1,4,4,1]` `@M` raised (`_ensure_spatial`'s
`tensor.shape[:len(spatial_shape)] == spatial_shape` check passes vacuously —
`[1,4,4,1][:3] == (1,4,4)` — so the `[1,4,4,1]` tensor is handed back UNCHANGED
into a `[1,4,4]` slot, and PyTorch's trailing-dim broadcast then lines the
binding's H up against the slot's W and raises), while the identical field
bound as `[1,4,4]` cooked normally.

**First fix attempt (superseded, see below):** squeezing `[B,H,W,1]` to
`[B,H,W]` at INGEST (`to_fp32_if_int_image`, the shared binding-normalisation
seam both tiers call). That closed the crash but changed a ComfyUI-VISIBLE
default-path result: a plain passthrough `@OUT = @A;` never goes through
`_ensure_spatial` at all — it just re-emits whatever rank the binding
arrived at — so squeezing at ingest also squeezed a 1-channel passthrough's
OUTPUT from `[B,H,W,1]` to `[B,H,W]`, caught by
`test_v028_phase1.py::test_root_channel_and_swizzle_fixes`'s "root
passthrough" row (invariant 7: the fix may only turn a former RAISE into a
cook, never move an output that already cooked correctly).

**Fixed instead at `interpreter._ensure_spatial`** — the point of USE, not
ingest: when a `[B,H,W,1]` tensor is reconciled against a `[B,H,W]` target
(every one of `_ensure_spatial`'s callers wants exactly that target rank
back — a vec-constructor component, an array element, a channel/index
write), it is squeezed there. A plain passthrough assignment never calls
`_ensure_spatial`, so its output rank is untouched. Codegen's generated code
calls this exact same function (imported into the compiled namespace as
`_es`), so both tiers pick the fix up identically with no codegen-side
change (invariant 2).

ComfyUI-invisible because: the ComfyUI MASK/IMAGE wires never hand TEX a
`[B,H,W,1]` tensor (a MASK is already rank 3; an IMAGE is never 1-channel),
so this fix never fires on any cook a ComfyUI user can produce; and for the
one shape it DOES fire on, every OTHER output this engine already produced
(including the 1-channel passthrough) is provably unmoved (see the control
row below).
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


def test_trk115_passthrough_keeps_its_rank4_shape(r: SubTestResult):
    """Control (the exact regression the ingest-side attempt introduced): a
    1-channel passthrough `@OUT = @A;` never reaches `_ensure_spatial` at all,
    so it must keep egressing at `[B,H,W,1]`, byte-identical to before this
    row existed — this fix may only turn a former raise into a cook, never
    move an output that already cooked correctly (invariant 7)."""
    print("\n--- TRK-115 control: a 1-channel passthrough keeps its [B,H,W,1] shape ---")
    torch.manual_seed(115)
    m4 = torch.rand(1, 4, 4, 1)
    try:
        res = tex_engine.cook("@OUT = @A;", {"A": m4.clone()}, device_mode="cpu")
    except Exception as e:
        r.fail("a 1-channel passthrough still cooks", f"{type(e).__name__}: {e}")
        return
    out = res.outputs["OUT"]
    if tuple(out.shape) != (1, 4, 4, 1):
        r.fail("a 1-channel passthrough keeps its [B,H,W,1] shape",
              f"got shape {tuple(out.shape)}")
    elif not torch.equal(out, m4):
        r.fail("a 1-channel passthrough's VALUES are unmoved", "value mismatch")
    else:
        r.ok("[B,H,W,1] passthrough shape and values are byte-identical, unmoved by this fix")


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
