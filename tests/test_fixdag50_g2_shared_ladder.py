"""FIX-DAG G2 (the reuse review) — `cook_stage_dag`'s outer ROI-eligibility gate now CALLS
`tex_roi.roi_eligibility` (the same shared ladder `cook_stage_list`/`tex_engine.prepare`/
`tex_engine_tiers.tier_verdict` already call) instead of hand-copying its five-condition
arithmetic a second time.

Red-first at base `e3ba26b`: `cook_stage_dag` never calls `roi_eligibility` at all (it has
its own byte-for-byte copy of the same five conditions) — monkeypatching `roi_eligibility`
to force a decline has NO EFFECT on whether `cook_stage_dag` windows, which is exactly the
"silently does not reach the DAG-cook path" risk R1#1 names. After this fix, forcing a
decline through the mock DOES stop `cook_stage_dag` from windowing, proving the gate is a
real call, not a parallel copy that merely agrees with it today.
"""
from __future__ import annotations

from helpers import *

from TEX_Wrangle import tex_chain
from TEX_Wrangle import tex_roi


def _merge_below_edit_stages(A, B):
    return [
        {"code": "@OUT = @A + 0.1;", "bindings": {"A": A}},
        {"code": "@OUT = gauss_blur(@B, 3.0);", "bindings": {"B": B}},
        {"code": "@OUT = (@bg + @fg) * 0.5;", "bindings": {},
         "chain_inputs": {"bg": [0, "OUT"], "fg": [1, "OUT"]}},
    ]


def test_fixdag50_g2_roi_eligibility_is_actually_called(r: SubTestResult):
    print("\n--- FIX-DAG G2: cook_stage_dag calls the shared roi_eligibility ---")
    torch.manual_seed(50200)
    W = H = 32
    A = torch.rand(1, H, W, 3)
    B = torch.rand(1, H, W, 3)
    stages = _merge_below_edit_stages(A, B)
    roi = (10, 10, 8, 8, W, H)

    # Only count a call this outer gate itself could plausibly have made (a neutral
    # `code=""`, per its own docstring) -- `cook_stage_list`'s OWN per-stage call
    # (unrelated, pre-existing, real per-stage source) also reaches `roi_eligibility`
    # whenever a window is actually served, so counting EVERY call would pass even
    # against the base hand-copied gate for the wrong reason.
    outer_calls = []
    orig = tex_roi.roi_eligibility

    def _counting(*a, **kw):
        if a and a[0] == "":
            outer_calls.append((a, kw))
        return orig(*a, **kw)

    tex_roi.roi_eligibility = _counting
    try:
        win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True)
    finally:
        tex_roi.roi_eligibility = orig

    if not outer_calls:
        r.fail("FIX-DAG G2 shared-ladder call",
               "cook_stage_dag(roi=...)'s OWN outer gate never called "
               "tex_roi.roi_eligibility with a neutral code=\"\" -- the outer gate is "
               "still a hand copy, not a call (a per-stage cook_stage_list call with a "
               "real source doesn't count; this checks the outer gate specifically)")
        return
    if win["windows"] is None:
        r.fail("FIX-DAG G2 setup", "expected a real window plan on this DAG/roi")
        return

    # roi=None: no eligibility question exists, so no call should be made either -- the
    # SAME invariant-7 "no wasted work when nothing is requested" the hand-copied gate
    # already had, preserved by keeping the call inside `if roi is not None:`.
    outer_calls.clear()
    tex_roi.roi_eligibility = _counting
    try:
        tex_chain.cook_stage_dag(stages)
    finally:
        tex_roi.roi_eligibility = orig
    if outer_calls:
        r.fail("FIX-DAG G2 invariant 7",
               f"cook_stage_dag(roi=None) called roi_eligibility {len(outer_calls)} "
               f"time(s), expected 0")
        return

    r.ok("FIX-DAG G2: cook_stage_dag(roi=...) calls tex_roi.roi_eligibility exactly the way "
        "cook_stage_list already does, and roi=None still makes no call at all")


def test_fixdag50_g2_forced_decline_through_the_mock_actually_declines(r: SubTestResult):
    """The load-bearing proof, by contradiction: force `roi_eligibility` to always decline,
    and confirm `cook_stage_dag` then windows NOTHING on a DAG/roi combination that would
    otherwise window every stage. If this gate were still a hand copy, the mock would have
    no effect and this test would catch that (it does, against a reverted tree)."""
    print("\n--- FIX-DAG G2: a forced roi_eligibility decline actually declines the DAG ---")
    torch.manual_seed(50201)
    W = H = 32
    A = torch.rand(1, H, W, 3)
    B = torch.rand(1, H, W, 3)
    stages = _merge_below_edit_stages(A, B)
    roi = (10, 10, 8, 8, W, H)

    # Sanity: this roi/DAG combination DOES window for real, unmocked.
    unmocked = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True)
    if unmocked["windows"] is None:
        r.fail("FIX-DAG G2 forced-decline setup",
               "expected this DAG/roi to window for real before mocking anything")
        return

    orig = tex_roi.roi_eligibility
    # A REAL `RoiEligibility` decline (not a bespoke stub) so this mock behaves consistently
    # wherever the shared ladder is legitimately called from — including `cook_stage_list`'s
    # OWN per-stage call, which a forced-decline mock also reaches once a window is planned
    # at all. A stub missing `.message`/`.plan` would crash there instead of cleanly
    # declining, which is a weaker (and less honest) red-first signal than this one.
    _declined = tex_roi.RoiEligibility(False, None, None, tex_roi.ROI_REASON_NOT_ARMED,
                                       "forced decline (FIX-DAG G2 test)")

    def _always_decline(*a, **kw):
        return _declined

    tex_roi.roi_eligibility = _always_decline
    try:
        forced = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True)
    finally:
        tex_roi.roi_eligibility = orig

    if forced["windows"] is not None:
        r.fail("FIX-DAG G2 forced decline",
               f"forcing roi_eligibility to decline had no effect -- cook_stage_dag still "
               f"planned windows={forced['windows']!r}; the outer gate is not really "
               f"consulting the shared function")
        return
    if forced["stages_windowed"] != 0 or forced["stages_whole"] != 3:
        r.fail("FIX-DAG G2 forced decline",
               f"expected a fully whole-frame fallback, got windowed="
               f"{forced['stages_windowed']} whole={forced['stages_whole']}")
        return
    ref = tex_chain.cook_stage_dag(stages)
    if not torch.equal(forced["result"]["OUT"], ref["result"]["OUT"]):
        r.fail("FIX-DAG G2 forced decline pixel identity",
               "a forced decline's whole-frame fallback did not match a bare whole-frame cook")
        return

    r.ok("FIX-DAG G2: forcing tex_roi.roi_eligibility to decline actually stops "
        "cook_stage_dag from windowing (a whole-frame fallback that still matches the "
        "whole-frame cook exactly) -- the outer gate is a real call, not a parallel copy")
