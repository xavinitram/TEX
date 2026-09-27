"""FIX-SCALE S8 (v0.47 Phase C, R1 finding 1) — `tex_roi.clear_roi_memo()` must empty
`_scale_verdict_memo` (SCALE-47b) too, the same as every other bounded-LRU store in this
module.

Before this fix: `clear_roi_memo()` cleared `_walk_memo`, `_region_dep_memo` and
`_parse_memo` -- not `_scale_verdict_memo`. This module's own established contract (40+
call sites across `tests/`/`benchmarks/`, and `test_perf8_memo_flag_key.py` asserting
verbatim "clear_roi_memo / clear_lazy_memo still empty all five stores") treats
`clear_roi_memo()` as THE single reset point for every ROI-module cache; a caller relying on
that contract to get a clean slate silently kept a stale `scale_verdict()` answer -- for a
predicate that gates whether a cook may run at a non-1.0 `scale` at all."""
from helpers import *
from TEX_Wrangle import tex_roi as R

_CODE = "@OUT = gauss_blur(@A, vec4(11.0).r);"  # unlikely to collide with another test's memo


def test_s8_clear_roi_memo_empties_scale_verdict_memo(r: SubTestResult):
    print("\n--- FIX-SCALE S8: clear_roi_memo() empties _scale_verdict_memo too ---")
    R.scale_verdict(_CODE)
    if len(R._scale_verdict_memo) == 0:
        r.fail("premise", "scale_verdict() did not populate _scale_verdict_memo -- test setup "
               "is broken, not the fix")
        return
    R.clear_roi_memo()
    if len(R._scale_verdict_memo) != 0:
        r.fail("clear_roi_memo scale_verdict_memo",
               f"expected _scale_verdict_memo empty after clear_roi_memo(), still holds "
               f"{len(R._scale_verdict_memo)} entr(y/ies)")
        return
    r.ok("clear_roi_memo() empties _scale_verdict_memo")


def test_s8_clear_roi_memo_still_empties_the_other_three_stores(r: SubTestResult):
    print("\n--- FIX-SCALE S8: clear_roi_memo() still empties _walk_memo/_region_dep_memo/_parse_memo ---")
    R.scale_safe(_CODE)          # walks _fold_program, populates _walk_memo/_parse_memo
    R.roi_plan(_CODE, {})        # populates _region_dep_memo (region_dependent_cached)
    R.clear_roi_memo()
    empties = {
        "_walk_memo": len(R._walk_memo),
        "_region_dep_memo": len(R._region_dep_memo),
        "_parse_memo": len(R._parse_memo),
    }
    non_empty = {k: v for k, v in empties.items() if v != 0}
    if non_empty:
        r.fail("no regression", f"expected all three empty after clear_roi_memo(), still "
               f"non-empty: {non_empty}")
        return
    r.ok("clear_roi_memo() still empties _walk_memo, _region_dep_memo and _parse_memo")


def test_s8_a_cleared_verdict_reanalyzes_on_next_call(r: SubTestResult):
    print("\n--- FIX-SCALE S8: a fresh scale_verdict() call after clear_roi_memo() re-analyzes ---")
    real_walk = R._scale_unsafe_walk
    calls = {"n": 0}

    def _counting_walk(node, in_coord_arg=False):
        calls["n"] += 1
        return real_walk(node, in_coord_arg=in_coord_arg)

    R._scale_unsafe_walk = _counting_walk
    try:
        R._scale_verdict_memo.clear()
        R.scale_verdict(_CODE)
        first_calls = calls["n"]
        if first_calls == 0:
            r.fail("counting setup", "the walk was never called on the cold lookup")
            return
        R.clear_roi_memo()
        R.scale_verdict(_CODE)
        if calls["n"] <= first_calls:
            r.fail("re-analysis after clear",
                   f"expected the walk to run again after clear_roi_memo() (a genuinely cold "
                   f"lookup), but call count stayed at {calls['n']} (was {first_calls})")
            return
        r.ok(f"walk calls went {first_calls} -> {calls['n']} across a clear_roi_memo(), "
             f"proving the memo entry was actually gone")
    finally:
        R._scale_unsafe_walk = real_walk
        R._scale_verdict_memo.clear()
