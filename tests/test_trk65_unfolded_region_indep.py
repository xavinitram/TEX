"""TRK-65 — the region-dependence walk skips its per-fold cost when the UNFOLDED source
already proves the verdict independent of any `$param` value.

THE ROW. `tex_roi._walk` calls the UNCACHED `region_dependent(folded_program, ...)` on every
memo miss (`tex_roi.py::_walk`), and every param-value change is a miss by construction (the
walk's own memo key carries the param values, because the walk's OTHER outputs genuinely
depend on them). The "obvious" fix — route that call through `region_dependent_cached` on the
FOLDED program, keyed value-independently — was investigated and found UNSAFE: folding a
`$param` can make a loop/string-select/cast clause disappear for ONE valuation and not
another, so a value-independent cache over the fold would serve a stale verdict.

THE FIX HERE is the other, safe direction: `_unfolded_region_independent` (`tex_roi.py`) reads
`region_dependent` on the PRISTINE, UNFOLDED parse — which no `$param` can touch — and reuses
`region_dependent_cached`'s existing `_region_dep_memo` (a synthetic, namespaced fingerprint;
no new cache store). Constant-folding only EVALUATES and PRUNES; it never SYNTHESIZES a new
loop / string-select / cast that was not already a node in the unfolded source, so "the
unfolded source says False" implies "every fold of it says False too" — proved in the helper's
own docstring. `_walk` uses this to skip the per-fold `region_dependent` walk entirely when it
answers True, and falls through to the existing (unchanged, safe) per-value walk otherwise.

THIS FILE proves three things: (1) the fast path actually engages, and only once, across many
param valuations of a clause-free program; (2) a genuinely region-dependent program (TRK-25's
own repro) still declines correctly — the fast path never mis-engages; (3) a MUTATION that
breaks the safety argument (patching the helper to answer "independent" unconditionally) is
caught by a test whose program has a real, live per-pixel loop — i.e. this file's own oracle
would go silently wrong without the safeguard the docstring describes, which is exactly the
"test that would catch a stale verdict" the ask calls for.

PORTABILITY: CPU, no ComfyUI, no CUDA, no compiler, no numpy, no timing assertion.
"""
from helpers import *

from TEX_Wrangle import tex_api, tex_roi

# A clause-free program: no loop, no string-select, no scalar-to-string cast anywhere in the
# unfolded source. `mix(@A, @B, $k)` folds an arm away at the extremes, which is the classic
# TRK-64 counterexample for the WALK's other outputs — but region-dependence is untouched by
# it, because `mix` is none of the three clause shapes.
_CLAUSE_FREE = "@OUT = mix(@A, @B, $k);"

# TRK-25's own repro: a while loop whose trip count is PER-PIXEL (driven by `v`), which is
# real region-dependence — the split must decline for every param valuation.
_LOOP_REPRO = ("float x = v;\n"
               "float n = 0.0;\n"
               "while (x < 1.0) { x = x + 0.25; n = n + 1.0; }\n"
               "@OUT = vec4(n, n, n, 1.0);\n")


def test_trk65_fast_path_engages_once_across_many_valuations(r: SubTestResult):
    print("\n--- TRK-65: the fast path answers False without re-walking the fold ---")
    tex_roi.clear_roi_memo()
    calls = {"n": 0}
    orig = tex_api._ControlFlowLint.region_clauses

    def spy(self):
        calls["n"] += 1
        return orig(self)

    tex_api._ControlFlowLint.region_clauses = spy
    try:
        answers = []
        for k in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 0.3333333):
            walked = tex_roi._walk(_CLAUSE_FREE, {"k": k})
            assert walked is not None, f"walk failed for k={k}"
            answers.append(walked[4])
    finally:
        tex_api._ControlFlowLint.region_clauses = orig

    if any(a is not False for a in answers):
        r.fail("TRK-65 clause-free verdict",
               f"a clause-free program must never be region-dependent: {answers}")
    elif calls["n"] != 1:
        r.fail("TRK-65 fast path", f"expected exactly ONE region_clauses() walk across "
               f"{len(answers)} param valuations of a clause-free source, got {calls['n']} — "
               f"the fast path is not engaging (or is re-walking on every miss)")
    else:
        r.ok(f"{len(answers)} param valuations, all False, ONE underlying "
             f"_ControlFlowLint.region_clauses() walk")


def test_trk65_real_region_dependence_still_declines(r: SubTestResult):
    print("\n--- TRK-65: a genuinely region-dependent program is unaffected ---")
    tex_roi.clear_roi_memo()
    walked = tex_roi._walk(_LOOP_REPRO, {})
    if walked is None:
        r.fail("TRK-65 loop repro", "walk failed")
        return
    if walked[4] is not True:
        r.fail("TRK-65 loop repro", f"region_dep = {walked[4]!r}, expected True — the fast "
               f"path must never mis-classify a real per-pixel loop as independent")
        return
    r.ok("the per-pixel while-loop repro still declines (region_dep=True), unchanged")

    # roi_plan (the public surface `_tile_plan`/`batch_sliceable` gate on) must decline too.
    plan = tex_roi.roi_plan(_LOOP_REPRO, {})
    if plan.executable:
        r.fail("TRK-65 roi_plan", "roi_plan reports executable=True for a region-dependent "
               "program — the split would run and produce a wrong picture")
    else:
        r.ok("roi_plan declines the split for the region-dependent program")


def test_trk65_mutation_a_wrongly_independent_verdict_is_visible(r: SubTestResult):
    """RED-FIRST HALF, inverted: patch the helper to lie ("everything is independent") and
    require `_walk` to still decline the loop repro correctly — i.e. prove the mutation WOULD
    be visible if the safety property broke, by observing the wrong answer the lie produces
    directly. This is the "test that would catch a stale verdict" the ask requires: it exists
    to fail loudly the day someone weakens the helper's safety argument, not to pass quietly.
    """
    print("\n--- TRK-65 mutation guard: a broken fast path is CAUGHT, not silently trusted ---")
    tex_roi.clear_roi_memo()
    orig = tex_roi._unfolded_region_independent
    tex_roi._unfolded_region_independent = lambda code, binding_types: True
    try:
        walked = tex_roi._walk(_LOOP_REPRO, {})
    finally:
        tex_roi._unfolded_region_independent = orig
        tex_roi.clear_roi_memo()

    if walked is None:
        r.fail("TRK-65 mutation guard", "walk failed under the mutation")
        return
    if walked[4] is True:
        r.fail("TRK-65 mutation guard",
               "the mutated (always-independent) helper still produced the correct True "
               "verdict — this test's program does not exercise the fast path's guard at "
               "all, so it cannot prove the guard matters; strengthen the repro")
    else:
        r.ok(f"a wrongly-'independent' helper flips region_dep to {walked[4]!r} (should be "
             f"True) — the mutation is VISIBLE, which is exactly why the real helper must "
             f"never answer True for this program (and, per its docstring's proof, does not)")
