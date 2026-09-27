"""FIX-SCALE S9 (v0.47 Phase C, B2 finding 7) — a KNOWN, documented, LEFT-AS-IS over-refusal:
`fetch(@A, int(ix), int(iy))` is classified unsafe even though it is a benign,
canvas-relative whole-pixel fetch behind a defensive type cast.

This is NOT a bug fix -- the ruling on this finding was "leave
conservative; document." `_scale_unsafe_walk`'s `in_coord_arg` whitelist flag is set fresh at
each FunctionCall/BindingIndexAccess node from that node's own identity, never inherited from
the caller's -- so a defensive `int(...)`/`float(...)` cast around an otherwise-whitelisted
fetch coordinate loses the whitelist. Confirmed direction: this can only ever LOSE the flag,
never wrongly GRANT one (over-refusal, never a false-safe / wrong picture). This test PINS
the current, intentional behavior (so a future change either direction is a deliberate,
noticed decision, not an accident) and documents the `//!tex scale: safe` workaround."""
from helpers import *
from TEX_Wrangle import tex_roi as R


def test_s9_fetch_with_int_cast_is_over_refused(r: SubTestResult):
    print("\n--- FIX-SCALE S9: fetch(@A, int(ix), int(iy)) is declared unsafe (over-refusal, documented) ---")
    code = "@OUT = fetch(@A, int(ix), int(iy));"
    if R.scale_safe(code):
        r.fail("premise", "fetch(@A, int(ix), int(iy)) is now classified safe -- if this is "
               "an intentional improvement, update this pinned test and "
               "docs/resolution-scale.md's S9 note together; do not leave them disagreeing")
        return
    r.ok("fetch(@A, int(ix), int(iy)) is over-refused, exactly as documented (conservative, "
         "never a false-safe)")


def test_s9_bare_fetch_without_the_cast_is_still_safe(r: SubTestResult):
    print("\n--- FIX-SCALE S9: fetch(@A, ix, iy) (no cast) is still classified safe (no regression) ---")
    code = "@OUT = fetch(@A, ix, iy);"
    if not R.scale_safe(code):
        r.fail("no regression", "fetch(@A, ix, iy) without the defensive cast was classified "
               "unsafe -- the S9 over-refusal must not have widened to the plain case too")
        return
    r.ok("fetch(@A, ix, iy) (the whitelisted idiom, no cast) is still scale_safe")


def test_s9_pragma_safe_is_the_documented_workaround(r: SubTestResult):
    print("\n--- FIX-SCALE S9: //!tex scale: safe is the documented workaround for this over-refusal ---")
    code = "//!tex scale: safe\n@OUT = fetch(@A, int(ix), int(iy));"
    if not R.scale_safe(code):
        r.fail("workaround", "the //!tex scale: safe override did not rescue a program that "
               "hits the S9 over-refusal")
        return
    r.ok("//!tex scale: safe overrides the over-refusal, as documented in "
         "docs/resolution-scale.md")
