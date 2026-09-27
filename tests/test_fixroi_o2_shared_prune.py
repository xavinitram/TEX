"""FIX-ROI O2 (v0.48.0 Phase C, R4-altitude finding 2) -- one pruning step at fold level,
shared by `tex_roi._accumulate` and `tex_roi._has_ungrounded_halo`, reusing
`tex_lazy._prune_static_flow` (proven sound by O1's fp32-verification pass) instead of the
duplicated per-node `_resolved_branch` helper -- and picking up the literal-false `WhileLoop`
case neither walker had an equivalent for.

BEFORE: `tex_roi.py` carried its own `_resolved_branch()` helper, independently re-derived
inside BOTH `_accumulate`'s `IfElse` case and `_has_ungrounded_halo`'s `_visit` -- two copies
of the same "does this condition resolve, and if so which body runs" predicate, and neither
had any equivalent for a `while (0) { <halo op> }` (a literal-false condition drops the whole
loop in `tex_lazy._prune_static_flow`, but nothing in `tex_roi.py` ever called it).

AFTER: `_resolved_branch` is deleted. `tex_roi._walk` prunes a private clone of the folded
program (`tex_lazy._prune_static_flow`) once, before handing it to `_accumulate` and
`_has_ungrounded_halo` -- so a dead `while` no longer blocks ROI reach either.

RED AT BASE (`5ae6288`, verified by hand against a snapshot of that commit before this fix
landed): `test_o2_dead_while_unlocks_window` below failed with the dead loop's huge halo
still counted (either `executable=False` or a much larger `halo` than the live branch
needs).
"""
from helpers import *
from TEX_Wrangle import tex_engine, tex_roi as _R

# A while(0) wrapping a huge halo op: the loop never runs (the fp32-verified `$k > 0.5`
# folds to False), so the cook is really just `@OUT = @image` -- halo 0, not the ~150px
# reach a naive both-arms/no-prune walk would report for the dead `gauss_blur(@OUT, 50.0)`.
_DEAD_WHILE_CODE = """
f$k = 0.0;
@OUT = @image;
while ($k > 0.5) {
    @OUT = gauss_blur(@OUT, 50.0);
}
"""
_PARAMS = {"k": 0.0}


def test_o2_resolved_branch_helper_is_gone(r: SubTestResult):
    print("\n--- FIX-ROI O2: the duplicated per-node helper is deleted ---")
    if hasattr(_R, "_resolved_branch"):
        r.fail("O2 dedup", "tex_roi._resolved_branch still exists -- the duplicated "
               "per-node pruning helper was not removed")
        return
    r.ok("tex_roi._resolved_branch no longer exists (pruning moved to fold level)")


def test_o2_dead_while_unlocks_window(r: SubTestResult):
    print("\n--- FIX-ROI O2: a literal-false while no longer blocks ROI reach ---")
    try:
        _R.clear_roi_memo()
        plan = _R.roi_plan(_DEAD_WHILE_CODE, _PARAMS)
        if not plan.executable:
            r.fail("O2 roi_plan executable",
                   f"unexpectedly not executable: {plan} -- a dead `while(0)` should not "
                   f"block reach")
            return
        if plan.halo != 0:
            r.fail("O2 roi_plan halo",
                   f"halo={plan.halo} -- the dead loop's gauss_blur(@OUT, 50.0) is still "
                   f"being counted even though the loop never runs")
            return
        r.ok(f"roi_plan reports halo=0, executable=True (the dead while body is pruned "
             f"before the halo walk)")

        W, H = 32, 24
        roi = (4, 4, 12, 10, W, H)
        x0, y0, w, h, _, _ = roi
        torch.manual_seed(4096)
        image = torch.rand(1, H, W, 3)

        _R.clear_roi_memo()
        full = tex_engine.cook(_DEAD_WHILE_CODE, dict(_PARAMS, image=image.clone()),
                                device_mode="cpu").outputs["OUT"]
        _R.clear_roi_memo()
        res = tex_engine.cook(_DEAD_WHILE_CODE, dict(_PARAMS, image=image.clone()),
                               device_mode="cpu", roi=roi, roi_exec=True)
        win = res.outputs["OUT"]
        if res.cooked_roi != roi:
            r.fail("O2 pixel identity", f"window declined: cooked_roi={res.cooked_roi}")
            return
        crop = full[:, y0:y0 + h, x0:x0 + w]
        if tuple(win.shape) == tuple(crop.shape) and torch.equal(win, crop):
            r.ok("windowed cook torch.equal whole-frame crop with the dead while pruned")
        else:
            md = (win.float() - crop.float()).abs().max().item() \
                if tuple(win.shape) == tuple(crop.shape) else float("nan")
            r.fail("O2 pixel identity", f"shape {tuple(win.shape)} vs {tuple(crop.shape)}, "
                   f"maxdiff {md:.4e}")
    except Exception as e:
        r.fail("O2 dead while", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()
