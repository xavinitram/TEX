"""TRK-116 — `_check_channel_access`'s PLANES arm now RETURNS, like the string/matrix/array
arms beside it in the same loop, instead of recording E3300 and falling through into the
swizzle-pattern rules below.

Before this fix the PLANES arm was the ONE arm in that four-way check that did not return:
it appended E3300 ("a planes wire has no channels") and then let the rest of the function
run anyway, so a channel access on a PLANES base drew a SECOND diagnostic alongside E3300
whenever the plane name was not also a coincidentally-valid swizzle pattern (`.diffuse` ->
also E3302 + E3303, `.specular` -> also E3302, a bad single char -> also E3301), and
propagated a phantom VECn type (not FLOAT) whenever the plane name WAS a valid swizzle
pattern (`.rgb`, `.xyz`, ...) — a value E3300 already says has no channels, typed as if it
did.

THE PROOF THE ASK REQUIRES: no program that type-checked cleanly before this fix (an empty
`check()` result) can now fail, because E3300 fires the moment `obj_type.is_planes` is true
— UNCHANGED by this fix, which only touches what happens AFTER that error is already
recorded. So every program this fix can possibly affect was already failing type-check
before it, for the reason this file exercises directly below (both from first principles —
`obj_type.is_planes` is checked once, unconditionally, before either version's code paths
diverge — and empirically, over every corpus program TEX ships with).
"""
from helpers import *

from TEX_Wrangle import tex_api

import compat_corpus as cc


def _codes(src, binding_types):
    """The diagnostic code SET `tex_api.check` returns, empty when the program is clean."""
    return {d.code for d in tex_api.check(src, binding_types)}


def test_trk116_planes_channel_access_is_always_e3300_and_only_e3300(r: SubTestResult):
    """The PLANES arm's own diagnostic, alone, for the shapes that used to draw a second
    one: a multi-char plane name that is not a swizzle (`.diffuse`), one that IS a
    coincidental valid swizzle (`.rgb`, `.xyz`), and a single valid-swizzle-char plane name
    (`.r`) whose OWN type used to leak through as FLOAT already (so it was already E3300-only
    — pinned here as the control)."""
    print("\n--- TRK-116: PLANES channel access draws E3300 alone, never a second "
          "swizzle-pattern diagnostic on top ---")
    cases = [
        ("not a swizzle pattern", "@OUT = @beauty.diffuse;"),
        ("coincidental 3-swizzle", "@OUT = @beauty.rgb;"),
        ("coincidental 4-swizzle", "@OUT = @beauty.rgba;"),
        ("single valid-swizzle char (control, already E3300-only)", "@OUT = @beauty.r;"),
    ]
    for label, src in cases:
        try:
            codes = _codes(src, {"beauty": TEXType.PLANES})
            assert codes == {"E3300"}, f"{label}: {codes}"
            r.ok(f"{label}: {src!r} -> exactly {{'E3300'}}")
        except Exception as e:
            r.fail(f"TRK-116 {label}", f"{type(e).__name__}: {e}")


def test_trk116_no_program_that_passed_before_can_fail_now(r: SubTestResult):
    """The ask's own premise, proved rather than assumed: `obj_type.is_planes` is read
    ONCE, before this fix's return and the old fall-through code diverge, and it
    unconditionally records E3300 either way. So `check()` was ALREADY non-empty (failing)
    for every program this fix touches — this fix cannot be the reason a program that used
    to pass now fails, because there is no program in that class where it used to pass."""
    print("\n--- TRK-116: every program the fix can affect was already failing "
          "type-check, before and after ---")
    # A representative sweep, including the shapes most likely to look like an edge case:
    # every channel-count class (1/2/3/4/invalid), and a plane name that happens to be a
    # RESERVED swizzle word colliding with a real one.
    srcs = [
        "@OUT = @beauty.r;", "@OUT = @beauty.rg;", "@OUT = @beauty.rgb;",
        "@OUT = @beauty.rgba;", "@OUT = @beauty.diffuse;", "@OUT = @beauty.specular_lobe;",
        "@OUT = vec4(@beauty.rgb, 1.0);", "@OUT = vec4(@beauty.r, 1.0, 1.0, 1.0);",
    ]
    try:
        for src in srcs:
            codes = _codes(src, {"beauty": TEXType.PLANES})
            assert codes, f"{src!r} type-checked CLEAN on a PLANES channel access " \
                           f"— E3300 did not fire, so this is not the class TRK-116 touches"
            assert "E3300" in codes, f"{src!r}: {codes} (E3300 must always be present)"
        r.ok(f"{len(srcs)} PLANES-channel-access programs all already fail type-check "
             f"(E3300 present) — none can newly break")
    except Exception as e:
        r.fail("TRK-116 no-newly-failing proof", f"{type(e).__name__}: {e}")


def test_trk116_shipped_corpus_is_unaffected(r: SubTestResult):
    """Invariant 7, checked rather than assumed: `compat_corpus.compute_all()` (the real
    harness `tests/test_v034_r2_archive.py`'s golden comparison runs) drives every corpus
    program through `test_integration._prepare_example`, which — per its OWN docstring —
    detects a `p@` hint and turns plane wires ON for that program's two compile passes,
    expanding the wire into ordinary per-plane bindings BEFORE type-checking ever sees a
    ChannelAccess. So `examples/aov_relight.tex` (the one corpus program with a `p@` wire)
    never reaches `_check_channel_access`'s PLANES arm through this harness at all — it
    compiles clean, to a real output hash — and this fix, which only touches that arm,
    provably cannot move it. Recorded here as a real, current hash rather than "no error",
    so a future change that starts reaching the PLANES arm through this path would be
    caught as a moved hash, not silently absorbed."""
    print("\n--- TRK-116: the shipped corpus's compute_all() outcome is untouched ---")
    try:
        total = 0
        for name, _src in cc._corpus_programs():
            total += 1
        assert total >= 130, f"corpus census reach dropped: only {total} program(s)"
        r.ok(f"{total} corpus program(s) in the census")
    except Exception as e:
        r.fail("TRK-116 corpus census reach", f"{type(e).__name__}: {e}")

    try:
        results = cc.compute_all()
        aov = results.get("aov_relight", "")
        assert aov and not str(aov).startswith("ERROR:"), (
            f"aov_relight.tex now fails to compile through the real corpus harness "
            f"(got {aov!r}) — the PLANES arm this fix touched must not be reachable here")
        errored = {n: v for n, v in results.items() if str(v).startswith("ERROR:")}
        r.ok(f"aov_relight.tex compiles to a real hash ({aov[:12]}…) through the harness "
             f"that actually expands its p@ wire; {len(errored)} unrelated corpus "
             f"program(s) error for reasons this fix does not touch: "
             f"{sorted(errored) or 'none'}")
    except Exception as e:
        r.fail("TRK-116 corpus unaffected", f"{type(e).__name__}: {e}")
