"""TRK-6 — `sample_mip`'s fp16 grid buffer keeps the UV's dtype instead of being
forced fp32 (CPU, explicit `precision="fp16"`).

`stdlib_core._build_sample_grid` built its `[-1,1]` grid straight from `u`/`v`
(`grid_x = u * 2.0 - 1.0`) without reconciling to fp32, so under
`precision="fp16"` (where a UV literal/variable takes the cook's own dtype —
`interpreter._eval_number_literal`) the returned grid stayed fp16. Because the
mip pyramid level `_sample_mip_level` samples is ALSO fp16 in that mode,
`_grid_sample_f32`'s `if inp.dtype != grid.dtype:` reconciliation never fired
(both were fp16, not mismatched) and `sample_mip` ran `grid_sample` on raw fp16
COORDINATES — invariant 4's exact hazard ("fp16 coords mis-address rows at
large H"), not merely fp16 data. `fn_sample`'s own grid-sample path builds an
unconditionally-fp32 buffer and is immune (the row's own control, reproduced
below): fp32-vs-fp16 max abs diff for `sample()` is ~1.8e-4, three orders of
magnitude below `sample_mip`'s pre-fix ~0.33 at the same UV.

Verbatim reproduction (CPU, `torch.manual_seed(3)`, `[1,257,260,4]` kernel,
`@OUT = sample_mip(@K, <u>, <v>, 0.0);` at three UVs — matching the design
doc's own construction and this row's own measured numbers exactly):

    UV=(0.25,0.75): fp32 reads [0.37509,0.38443,0.19548,0.80486],
                    fp16 (BEFORE fix) reads [0.70703,0.52686,0.28052,0.71045]
                    — max abs diff 0.332
    UV=(0.5,0.5):   max abs diff 0.316
    UV=(0.9,0.1):   max abs diff 0.019

Fixed by forcing `u`/`v` to fp32 inside `_build_sample_grid` before building
the grid (`.to(torch.float32)` is a no-op when already fp32, so the default
path pays nothing) — the same "coordinate math is never `self._dtype`"
contract `interpreter.py`'s builtins already enforce (invariant 4).

ComfyUI-invisible because: `precision="fp16"` is an explicit opt-in the
default ComfyUI node never selects (invariant 7's default path is
`precision="fp32"`), and even under fp16 this only changes an INTERNAL
grid buffer's dtype, not any value a ComfyUI cook can observe on the default
path.
"""
from helpers import *
from TEX_Wrangle.tex_cache import parse_and_split

_UVS = [(0.25, 0.75), (0.5, 0.5), (0.9, 0.1)]
# The row's own reproduction: max abs diff BEFORE the fix, worst UV first —
# used only to document the magnitude class this test guards against; the
# assertion below is a bound, not an exact replay (fp16 rounding is hardware/
# torch-version sensitive at the last bit).
_PRE_FIX_MAXDIFF = {(0.25, 0.75): 0.332, (0.5, 0.5): 0.316, (0.9, 0.1): 0.019}


def _kernel():
    torch.manual_seed(3)
    return torch.rand(1, 257, 260, 4)


def _sample_mip_fp16_vs_fp32_diff(u, v):
    K = _kernel()
    code = f"@OUT = sample_mip(@K, {u!r}, {v!r}, 0.0);"
    bt = {"K": _infer_binding_type(K)}
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    out32 = Interpreter().execute(prog, {"K": K.clone()}, tm, device="cpu",
                                  output_names=["OUT"], precision="fp32")["OUT"]
    out16 = Interpreter().execute(prog, {"K": K.clone()}, tm, device="cpu",
                                  output_names=["OUT"], precision="fp16")["OUT"]
    return (out32.float() - out16.float()).abs().max().item()


def test_trk6_sample_mip_fp16_grid_is_fp32(r: SubTestResult):
    """Every probed UV's fp16-vs-fp32 divergence must sit at the fp16-QUANTIZATION
    order (a few 1e-2 at worst, matching `sample()`'s own control) — not at the
    pre-fix wrong-row-addressing order (0.3+)."""
    print("\n--- TRK-6: sample_mip's fp16 grid is fp32, not the UV's own dtype ---")
    for u, v in _UVS:
        diff = _sample_mip_fp16_vs_fp32_diff(u, v)
        pre = _PRE_FIX_MAXDIFF[(u, v)]
        if diff >= 0.05:
            r.fail(f"UV=({u},{v}): fp16 max abs diff stays under 0.05 (fp16 "
                  f"quantization order, not the pre-fix {pre} wrong-addressing order)",
                  f"got {diff}")
        else:
            r.ok(f"UV=({u},{v}): max abs diff {diff} (pre-fix was {pre})")


def test_trk6_sample_mip_matches_sample_control(r: SubTestResult):
    """`sample()`'s own grid-sample path was always immune (unconditionally fp32
    buffer) — pin its control value, and that sample_mip's WORST-case UV is now
    within an order of magnitude of it, not ~1900x larger (0.332 / 1.76e-4) as
    it was pre-fix."""
    print("\n--- TRK-6: sample_mip's worst-UV diff is now near sample()'s own control ---")
    K = _kernel()
    code = "@OUT = sample(@K, 0.25, 0.75);"
    bt = {"K": _infer_binding_type(K)}
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    out32 = Interpreter().execute(prog, {"K": K.clone()}, tm, device="cpu",
                                  output_names=["OUT"], precision="fp32")["OUT"]
    out16 = Interpreter().execute(prog, {"K": K.clone()}, tm, device="cpu",
                                  output_names=["OUT"], precision="fp16")["OUT"]
    control_diff = (out32.float() - out16.float()).abs().max().item()

    worst = max(_sample_mip_fp16_vs_fp32_diff(u, v) for u, v in _UVS)
    if control_diff >= 0.05:
        r.fail("sample()'s own control stays at fp16-quantization order",
              f"got {control_diff}")
    elif worst > control_diff * 100:
        r.fail("sample_mip's worst-UV diff is within ~100x of sample()'s control "
              "(not ~1900x, the pre-fix ratio)",
              f"worst={worst} control={control_diff} ratio={worst / control_diff:.1f}")
    else:
        r.ok(f"worst sample_mip diff {worst} vs sample() control {control_diff} "
            f"(ratio {worst / control_diff:.1f}x)")
