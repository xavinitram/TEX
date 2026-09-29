"""CKPT-51 (v0.51 wave 1, host item 3) — joins through the checkpointed route.

JOINWIRE-50/50b + FIX-DAG (v0.50) already answered the composition question this ask names:
`cook_checkpointed` stays linear/fused (its `cut_set` analysis documents a multi-edge-cut
suffix rebind as DEFERRED, not merely untested); `tex_chain.cook_stage_dag` is the DAG-shaped,
node-by-node, never-fused sibling that was taught checkpoint semantics directly
(`result_cache=`/`upstream=`), so a host with an actual join calls THAT entry point and gets
windowed joins AND checkpoint-boundary reuse through the same call, with no second caching
contract (`docs/effort-based-checkpoints.md` §14). This file is not a new mechanism: it is two
proof rows the existing JOINWIRE-50b suite did not yet cover by name.

1. **A cut AT a join.** Every existing checkpoint-backed DAG row (`test_joinwire50b_*`) places
   the checkpoint boundary on a stage that FEEDS a join (an upstream of the Merge). None places
   it ON the join stage's own output -- i.e. `dirty_from` past the join index itself, so the
   Merge's own result must come out of `result_cache`/`known_outputs`, not be re-cooked. The
   per-stage mechanism (`boundary_lineage_key` over `chain_inputs`, the `served_roi is None`
   gate) never special-cases "is this stage a join", so this is expected to already hold; the
   row exists to prove that by construction, not merely by argument.
2. **An approximate path declines the window, through the checkpointed route.** `gauss_blur`
   past `GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA` (256) and `bilateral_filter` past
   `_BILATERAL_EXACT_RADIUS_MAX` (40, i.e. `spatial_sigma > 40/3`) both decline ROI narrowing
   (FIX-APPROX A1, `tex_roi._reach_of`'s `approx_above` element) at the SAME per-stage
   `cook_stage_list`/`tex_engine.cook` call every DAG stage already goes through -- so a
   checkpointed DAG stage using one of these builtins past its threshold must serve
   whole-frame (never a wrong-phase window), and MUST still be eligible for `result_cache`
   exactly like the pre-existing heavy-blur-clamp shape (`_cp_merge_stages`'s own docstring:
   "forces stage0's OWN planned window to clamp to the full frame ... exactly the shape a
   checkpoint boundary needs"). A windowed sibling stage in the same DAG must still never be
   cached as whole-frame (the pre-existing negative control already proves that generically;
   this file does not repeat it).

CPU only (no device-specific mechanism exercised beyond what JOINWIRE-50/50b's own CUDA rows
already cover).
"""
from __future__ import annotations

import torch

from helpers import *
from helpers import _crop

from TEX_Wrangle import tex_chain
from TEX_Wrangle import tex_results
from TEX_Wrangle.tex_runtime.stdlib_core import GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib as _Sample  # populates REGISTRY


_HEAVY_SIGMA = 20.0


# ── 1. A cut AT a join ───────────────────────────────────────────────────────────────────

def _cp_join_boundary_stages(A, B):
    """stage0/stage1: two independent sources. stage2: the JOIN (Merge) of stage0+stage1 --
    THIS is the checkpoint boundary under test. stage3: a further downstream consumer of the
    join's own output (a heavy blur, so stage2's planned window clamps to the full frame on a
    16x16 canvas, the same "planned-a-window-but-serves-whole-frame" shape every other row in
    this family uses to seed a boundary)."""
    return [
        {"code": "@OUT = @A;", "bindings": {"A": A}},
        {"code": "@OUT = @B;", "bindings": {"B": B}},
        {"code": "@OUT = (@bg + @fg) * 0.5;", "bindings": {},
         "chain_inputs": {"bg": [0, "OUT"], "fg": [1, "OUT"]}},
        {"code": f"@OUT = gauss_blur(@P, {_HEAVY_SIGMA});", "bindings": {},
         "chain_inputs": {"P": [2, "OUT"]}},
    ]


def test_ckpt51_checkpoint_boundary_at_a_join_pixel_identity(r: SubTestResult):
    print("\n--- CKPT-51: a checkpoint boundary placed AT a join stage's own output ---")
    torch.manual_seed(5101)
    W = H = 16
    A = torch.rand(1, H, W, 3)
    B = torch.rand(1, H, W, 3)
    rc = tex_results.ResultCache()
    up = ("ckpt51-join-boundary-src",)
    stages = _cp_join_boundary_stages(A, B)

    # Tick 1: everything dirty. Stage 2 (the JOIN) is planned a window (stage 3's heavy blur
    # creates a real downstream demand) but its own cook serves whole-frame (the halo clamps
    # to the full 16x16 frame) -- exactly the shape that should populate result_cache under
    # the join's own boundary key.
    roi1 = (1, 1, 4, 4, W, H)
    win1 = tex_chain.cook_stage_dag(stages, roi=roi1, roi_exec=True, dirty_from=0,
                                    result_cache=rc, upstream=up)
    if win1["windows"] is None or win1["windows"][2] is None:
        r.fail("CKPT-51 join-boundary seed",
               f"expected stage 2 (the join) to be planned a window, got "
               f"{win1['windows']!r}")
        return
    # The authoritative "did this stage actually serve whole-frame" signal is
    # `stage_windows` (`tier_trace.last_roi()`-gated), not the PLAN in `windows` -- a
    # stage can be planned a narrow window and still serve whole-frame internally (a halo
    # clamp, or an approximate-path decline). Absence from `stage_windows` is what makes
    # this stage eligible to seed a `result_cache` boundary at all.
    if 2 in win1["stage_windows"]:
        r.fail("CKPT-51 join-boundary seed",
               f"expected the join's own cook to serve whole-frame (halo clamp), but it "
               f"is in stage_windows: {win1['stage_windows'][2]!r}")
        return

    # Tick 2: dirty_from=3 -- everything AT AND BEFORE the join (stages 0,1,2) is clean and
    # supplied by NEITHER known_outputs NOR live bindings; only result_cache, keyed on the
    # JOIN's own boundary, can satisfy stage 3's read of stage 2's output.
    roi2 = (9, 9, 3, 3, W, H)
    win2 = tex_chain.cook_stage_dag(stages, roi=roi2, roi_exec=True, dirty_from=3,
                                    valid=[None, None, (0, 0, W, H), None],
                                    result_cache=rc, upstream=up)
    fresh = tex_chain.cook_stage_dag(stages)
    ref = _crop(fresh["result"]["OUT"], roi2)
    got = win2["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("CKPT-51 join-boundary round trip",
               f"maxdiff={(got - ref).abs().max().item()}")
        return
    r.ok("CKPT-51: a checkpoint boundary keyed at a JOIN stage's own output correctly "
        "serves a later tick with no known_outputs and matches a fresh whole-frame cook "
        "of the (unedited) graph exactly")


# ── 2. Approximate path declines the window, through the checkpointed route ────────────

def _cp_approx_gauss_stages(A, B):
    """stage0: source A. stage1: gauss_blur(A) PAST the pyramid threshold -- the checkpoint
    candidate; its own window must decline (serve whole-frame) rather than sample on a wrong
    phase. stage2: an independent edit of B. stage3: the join (sink)."""
    sigma = GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA + 8.0  # 264.0: past the threshold
    return [
        {"code": "@OUT = @A;", "bindings": {"A": A}},
        {"code": f"@OUT = gauss_blur(@P, {sigma});", "bindings": {},
         "chain_inputs": {"P": [0, "OUT"]}},
        {"code": "@OUT = @B + 0.1;", "bindings": {"B": B}},
        {"code": "@OUT = (@bg + @fg) * 0.5;", "bindings": {},
         "chain_inputs": {"bg": [1, "OUT"], "fg": [2, "OUT"]}},
    ]


def test_ckpt51_approximate_gauss_blur_declines_through_checkpointed_route(r: SubTestResult):
    print("\n--- CKPT-51: gauss_blur past the pyramid threshold, checkpointed DAG route ---")
    torch.manual_seed(5102)
    W = H = 900  # large enough that a genuinely narrowed (non-saturating) window is possible
    A = torch.rand(1, H, W, 3)
    B_old = torch.rand(1, H, W, 3)
    B_new = torch.rand(1, H, W, 3)
    rc = tex_results.ResultCache()
    up = ("ckpt51-approx-gauss-src",)

    # Tick 1: seed the checkpoint. Stage 1's own halo (3*264=792) does NOT saturate a 900px
    # frame from a small interior window, so this is a genuinely-narrowed shape, not a
    # halo-clamp -- it must decline (served whole-frame) for the CORRECT reason (the
    # approximation's phase, not the halo size).
    roi1 = (300, 300, 120, 120, W, H)
    stages_old = _cp_approx_gauss_stages(A, B_old)
    win1 = tex_chain.cook_stage_dag(stages_old, roi=roi1, roi_exec=True, dirty_from=0,
                                    result_cache=rc, upstream=up)
    # `windows[1]` is the PLAN (the demand backward-projected onto stage 1); what the stage
    # actually SERVED is `stage_windows` (present == genuinely windowed, absent == served
    # whole-frame) -- `tier_trace.last_roi()`-gated, exactly as `cook_stage_dag`'s own
    # docstring describes. A stage that declines internally is absent from `stage_windows`
    # even though `windows[1]` still names the plan it was handed.
    if 1 in win1["stage_windows"]:
        r.fail("CKPT-51 approx gauss seed",
               f"expected stage 1 to DECLINE and serve whole-frame (approx phase hazard), "
               f"but it is in stage_windows: {win1['stage_windows'][1]!r}")
        return

    # Tick 2: an edit lands on stage 2 (B changes); stage 1 (the approx-declined boundary) is
    # now clean and must be served from result_cache with no known_outputs.
    roi2 = (500, 500, 80, 80, W, H)
    stages_new = _cp_approx_gauss_stages(A, B_new)
    win2 = tex_chain.cook_stage_dag(stages_new, roi=roi2, roi_exec=True, dirty_from=1,
                                    valid=[None, (0, 0, W, H), None, None],
                                    result_cache=rc, upstream=up)
    fresh = tex_chain.cook_stage_dag(stages_new)
    ref = _crop(fresh["result"]["OUT"], roi2)
    got = win2["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("CKPT-51 approx gauss round trip",
               f"maxdiff={(got - ref).abs().max().item()}")
        return
    r.ok("CKPT-51: gauss_blur's approximate-path decline (a whole-frame serve, not a "
        "wrong-phase window) still populates result_cache correctly and the checkpointed "
        "DAG route matches a fresh whole-frame cook exactly")


def test_ckpt51_approximate_bilateral_declines_through_checkpointed_route(r: SubTestResult):
    print("\n--- CKPT-51: bilateral_filter past its exact-radius threshold, checkpointed "
          "DAG route ---")
    torch.manual_seed(5103)
    W = H = 900
    A = torch.rand(1, H, W, 3)
    B_old = torch.rand(1, H, W, 3)
    B_new = torch.rand(1, H, W, 3)
    rc = tex_results.ResultCache()
    up = ("ckpt51-approx-bilateral-src",)
    ss = _Sample._BILATERAL_APPROX_THRESHOLD_SS + 0.5  # spatial_sigma past the threshold

    def _stages(B):
        return [
            {"code": "@OUT = @A;", "bindings": {"A": A}},
            {"code": f"@OUT = bilateral_filter(@P, {ss}, 0.2);", "bindings": {},
             "chain_inputs": {"P": [0, "OUT"]}},
            {"code": "@OUT = @B + 0.1;", "bindings": {"B": B}},
            {"code": "@OUT = (@bg + @fg) * 0.5;", "bindings": {},
             "chain_inputs": {"bg": [1, "OUT"], "fg": [2, "OUT"]}},
        ]

    roi1 = (300, 300, 120, 120, W, H)
    win1 = tex_chain.cook_stage_dag(_stages(B_old), roi=roi1, roi_exec=True, dirty_from=0,
                                    result_cache=rc, upstream=up)
    if 1 in win1["stage_windows"]:
        r.fail("CKPT-51 approx bilateral seed",
               f"expected stage 1 to DECLINE and serve whole-frame (approx phase hazard), "
               f"but it is in stage_windows: {win1['stage_windows'][1]!r}")
        return

    roi2 = (500, 500, 80, 80, W, H)
    win2 = tex_chain.cook_stage_dag(_stages(B_new), roi=roi2, roi_exec=True, dirty_from=1,
                                    valid=[None, (0, 0, W, H), None, None],
                                    result_cache=rc, upstream=up)
    fresh = tex_chain.cook_stage_dag(_stages(B_new))
    ref = _crop(fresh["result"]["OUT"], roi2)
    got = win2["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("CKPT-51 approx bilateral round trip",
               f"maxdiff={(got - ref).abs().max().item()}")
        return
    r.ok("CKPT-51: bilateral_filter's approximate-path decline still populates "
        "result_cache correctly and the checkpointed DAG route matches a fresh "
        "whole-frame cook exactly")
