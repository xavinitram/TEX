"""FIX-ROI O1 (v0.48.0 Phase C, B1#1) -- a uniform-branch fold must agree with the runtime's
own fp32 tensor decision before `tex_roi._resolved_branch` (or `tex_lazy._prune_static_flow`)
is allowed to trust it and prune the untaken arm.

THE BUG (confirmed at base `5ae6288` by the v0.48.0 Phase C bug hunt, finding B1#1):
`tex_roi._fold_program` folds a `$param`-substituted `IfElse`/
`WhileLoop` condition through the optimizer's generic constant fold
(`tex_compiler.optimizer._fold_all` / `_eval_binop_const`), which evaluates every
intermediate in plain Python DOUBLE precision. The runtime evaluates the identical
expression as fp32 tensors (`interpreter_control_flow.py`'s `cond.item() > 0.5` on an
fp32 `torch.Tensor`). For a condition built from an ARITHMETIC COMBINATION of two or more
substituted params (not a single `$param` compared directly -- that case was already
protected, since the leaf alone is fp32-rounded before folding), the double-precision fold
and the fp32 tensor evaluation can pick DIFFERENT branches whenever the true value sits
within half an fp32 ulp of the boundary the comparison tests.

Repro values (both exact fp32 floats): a=0.45405644178390503, b=0.045943569391965866.
`a + b` in Python double is 0.5000000111758709 (`> 0.5` is True); `torch.float32(a) +
torch.float32(b)` is exactly 0.5 (`> 0.5` is False). Before this fix, `roi_plan` reported
`executable=True, halo=0` for a program using this condition to pick between a bounded
`gauss_blur` and a plain copy -- a windowed cook then crops the input with NO halo margin
and runs the blur that actually executes on a halo-starved crop: a silently WRONG picture
(measured at base: max abs diff 0.1509), not a decline.

THE FIX: `_fold_program` (tex_roi.py) now snapshots every IfElse/WhileLoop condition BEFORE
the generic fold (`tex_lazy._capture_pre_fold_conditions`), and reverts (`tex_lazy.
_revert_unverified_folds`) any literal the fold produced that a dedicated fp32-per-op
re-evaluation (`tex_lazy._fp32_eval_expr`/`_fp32_binop`) cannot confirm agrees with it --
"cannot confirm" and "disagrees" are treated identically (doubt -> the pre-v0.48 both-arms
walk). `tex_lazy.lazy_required_bindings` gets the identical snapshot-then-revert step ahead
of its own `_prune_static_flow` call, closing the same defect there (the brief's "check
tex_lazy._prune_static_flow for the same defect"): neither `_resolved_branch` nor
`_prune_static_flow` needed a single line changed -- by the time either sees the tree, every
surviving NumberLiteral condition is proof, not a guess.

RED AT BASE (`5ae6288`, verified by hand against a snapshot of that commit before this fix
landed): `test_o1_pixel_identity` below failed with `torch.equal: False` / max diff ~0.1509;
`test_o1_lazy_never_severs` failed with `@B` absent from `lazy_required_bindings`'s result.
"""
from helpers import *
from TEX_Wrangle import tex_engine, tex_roi as _R, tex_lazy as _L

# Both exact fp32 floats. Python double sum is 0.5000000111758709 (> 0.5); the fp32 sum is
# exactly 0.5 (not > 0.5) -- the two evaluations pick opposite branches.
_A = 0.45405644178390503
_B = 0.045943569391965866
_PARAMS = {"a": _A, "b": _B}

# then = plain copy (halo 0, "safe" if the fold's [wrong] double decision is trusted);
# else = gauss_blur (halo > 0) -- what the runtime ACTUALLY runs, since the fp32 sum is not
# > 0.5. A trusted-but-wrong fold reports halo=0 and a windowed cook starves the blur's halo.
_BLUR_CODE = """
f$a = 0.0;
f$b = 0.0;
if ($a + $b > 0.5) {
    @OUT = @image;
} else {
    @OUT = gauss_blur(@image, 4.0);
}
"""

# The same boundary condition, this time picking between two DIFFERENT bindings -- the
# shape that would SEVER a dependency (invariant #11) if the wrong arm were pruned: the
# runtime reads @B (the fp32 sum is not > 0.5), so a lazy analysis that trusted the double
# fold's `True` and pruned to the `then` arm would report `@B` as unneeded.
_SEVER_CODE = """
f$a = 0.0;
f$b = 0.0;
if ($a + $b > 0.5) {
    @OUT = @A;
} else {
    @OUT = @B;
}
"""


def test_o1_pixel_identity(r: SubTestResult):
    print("\n--- FIX-ROI O1: a double/fp32-boundary uniform branch never starves the halo ---")
    try:
        _R.clear_roi_memo()
        plan = _R.roi_plan(_BLUR_CODE, _PARAMS)
        if not plan.executable:
            r.fail("O1 roi_plan executable", f"unexpectedly not executable: {plan}")
            return
        if plan.halo <= 0:
            r.fail("O1 roi_plan halo",
                   f"halo={plan.halo} -- the fold still trusts the wrong (double-precision) "
                   f"branch decision; the runtime actually runs gauss_blur(@image, 4.0), "
                   f"which needs halo > 0")
            return
        r.ok(f"roi_plan reports halo={plan.halo} (both arms walked -- the fold refused to "
             f"trust an unverifiable boundary literal)")

        W, H = 48, 40
        roi = (10, 8, 16, 14, W, H)
        x0, y0, w, h, _, _ = roi
        torch.manual_seed(2048)
        image = torch.rand(1, H, W, 3)

        _R.clear_roi_memo()
        full = tex_engine.cook(_BLUR_CODE, dict(_PARAMS, image=image.clone()),
                                device_mode="cpu").outputs["OUT"]
        _R.clear_roi_memo()
        res = tex_engine.cook(_BLUR_CODE, dict(_PARAMS, image=image.clone()),
                               device_mode="cpu", roi=roi, roi_exec=True)
        win = res.outputs["OUT"]
        if res.cooked_roi != roi:
            r.fail("O1 pixel identity", f"window declined: cooked_roi={res.cooked_roi}")
            return
        crop = full[:, y0:y0 + h, x0:x0 + w]
        if tuple(win.shape) == tuple(crop.shape) and torch.equal(win, crop):
            r.ok("windowed cook torch.equal whole-frame crop at the exact fp32 boundary")
        else:
            md = (win.float() - crop.float()).abs().max().item() \
                if tuple(win.shape) == tuple(crop.shape) else float("nan")
            r.fail("O1 pixel identity", f"shape {tuple(win.shape)} vs {tuple(crop.shape)}, "
                   f"maxdiff {md:.4e}")
    except Exception as e:
        r.fail("O1 pixel identity", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


def test_o1_lazy_never_severs(r: SubTestResult):
    print("\n--- FIX-ROI O1: the lazy analysis never severs @B at the fp32 boundary ---")
    try:
        _L.clear_lazy_memo()
        req = _L.lazy_required_bindings(_SEVER_CODE, _PARAMS)
        if req is None:
            r.fail("O1 lazy_required_bindings", "analysis failed (returned None)")
            return
        if "B" not in req:
            r.fail("O1 lazy never-sever (invariant 11)",
                   f"@B missing from {sorted(req)} -- the double-precision fold picked the "
                   f"`then` arm and `_prune_static_flow` pruned away the `else` arm the "
                   f"runtime actually executes (the fp32 sum is not > 0.5)")
            return
        r.ok(f"@B survives the fp32-boundary fold: required bindings = {sorted(req)}")
    except Exception as e:
        r.fail("O1 lazy never-sever", f"{type(e).__name__}: {e}")
    finally:
        _L.clear_lazy_memo()


def test_o1_tiling_halo_plan_consumes_the_same_fix(r: SubTestResult):
    print("\n--- FIX-ROI O1: tex_tiling._halo_tile_plan's roi_plan call sees the same fix "
          "(CPU-only; not TEX_ROI_EXEC-gated) ---")
    # `_halo_tile_plan` (tex_tiling.py) is unconditionally CUDA-gated (it early-returns on any
    # non-"cuda" device string) and not otherwise reachable here under a held bench lease (a
    # CPU-only window), so this calls the EXACT function it delegates the halo decision to
    # (`tex_roi.roi_plan(code, _scalar_params(bindings), binding_types)`, tex_tiling.py:373)
    # with the identical argument shape, proving the fix reaches this call site too. A CUDA
    # memory-pressure/TDR confirmation is deferred to a lease-holding follow-up, per B1-roi.md
    # finding 1's own "not confirmed by running" note for this exact path.
    try:
        from TEX_Wrangle import tex_tiling as _T
        scalar_params = _T._scalar_params(_PARAMS)
        _R.clear_roi_memo()
        plan = _R.roi_plan(_BLUR_CODE, scalar_params, None)
        if not plan.executable or plan.halo <= 0 or not plan.narrow:
            r.fail("O1 tiling halo plan",
                   f"_halo_tile_plan would read {plan} -- expected executable with halo > 0 "
                   f"and a non-empty narrow set")
            return
        r.ok(f"_halo_tile_plan's own roi_plan call would see halo={plan.halo}, "
             f"narrow={sorted(plan.narrow)}")
    except Exception as e:
        r.fail("O1 tiling halo plan", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()
