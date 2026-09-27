"""FIX-TIER T1 -- guard the one piece of `roi_eligibility`'s consolidation that stayed
duplicated on purpose: the `ROI_REASON_*` STRING LITERALS.

The decision LOGIC (the ordered ladder) is single-sourced in `tex_roi.roi_eligibility`;
`tex_engine_tiers.py` calls it directly. The literal string VALUES, however, are defined
in both modules -- `tex_engine_tiers.py` cannot import them from `tex_roi` at its own
module scope without pulling `tex_roi` into `tex_engine`'s cold-import closure (`tex_engine.py`
re-exports every `ROI_REASON_*` name at ITS OWN module scope, so a lazy resolution inside
`tex_engine_tiers.py` alone does not help --
`test_hostaudit1_tex_engine_import_module_count_ratchet` catches the growth either way).
This test is the guard that closes the residual drift risk that duplication reopens: if
the two copies ever disagree, this reds immediately, rather than silently, the next time
someone edits one copy and forgets the other."""
from helpers import *
from TEX_Wrangle import tex_roi as _tex_roi
from TEX_Wrangle import tex_engine_tiers as _tiers

_NAMES = [
    "ROI_REASON_TIER_NOT_DEFAULT", "ROI_REASON_FUSED_CHAIN", "ROI_REASON_LATENT",
    "ROI_REASON_SCALE_ACTIVE", "ROI_REASON_NOT_ARMED", "ROI_REASON_MALFORMED",
    "ROI_REASON_WHOLE_FRAME", "ROI_REASON_NOT_EXECUTABLE", "ROI_REASON_PRECISION",
    "ROI_REASON_ARMED",
]


def test_fixtier_t1_roi_reason_constants_stay_in_sync(r: SubTestResult):
    print("\n--- FIX-TIER T1: tex_engine_tiers.ROI_REASON_* == tex_roi.ROI_REASON_* ---")
    mismatches = [name for name in _NAMES
                 if getattr(_tex_roi, name) != getattr(_tiers, name)]
    if mismatches:
        r.fail("roi reason sync", f"these constants disagree between tex_roi.py and "
               f"tex_engine_tiers.py: {mismatches}")
        return
    r.ok(f"all {len(_NAMES)} ROI_REASON_* constants agree between tex_roi and "
         "tex_engine_tiers")
