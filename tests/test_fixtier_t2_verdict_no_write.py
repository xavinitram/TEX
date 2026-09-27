"""FIX-TIER T2 (B2#1) — `tex_engine_tiers.tier_verdict` must be side-effect-free.

Its own docstring says "Side-effect-free: no compile, no cache write, no cook, no cache
pollution" -- matching `tex_api.scale_verdict`'s genuinely side-effect-free contract. B2's
bug hunt found this false for the precise (`binding_types` given) branch: a scale-active
`tier_id == "default"` verdict calls `_stencil_route_would_apply`, which compiled `code`
through `TEXCache.compile_tex` -- a cache MISS there unconditionally calls `.put()`,
writing a NEW entry into both the in-memory tier and disk (`_save_to_disk`, an
unconditional `.pkl` write, no flag to suppress it).

Red-first: point a fresh `TEXCache` (its own scratch `cache_dir`, nothing pre-populated)
at the module singleton, call `tier_verdict(..., scale=..., binding_types=...)` for a
program/precondition combination that has never been compiled, and assert neither the
disk directory nor the in-memory tier gained an entry.
"""
from helpers import *
from TEX_Wrangle import tex_cache as _tex_cache
from TEX_Wrangle.tex_cache import TEXCache
from TEX_Wrangle.tex_engine_tiers import tier_verdict

# Plain pointwise program -- ROI/tier irrelevant here, only that it is a program the fresh
# cache has never seen before (a fresh temp dir already guarantees that, but a distinctive
# body keeps this test's fingerprint from ever colliding with another test's).
_CODE = "@OUT = vec4(@A.rgb * 1.2345 + 0.0007, 1.0);\n"
_BT = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}


def _run_against_fresh_cache(fn):
    """Swap the module-level `TEXCache` singleton for a fresh, empty one rooted at a new
    temp dir for the duration of `fn(fresh_cache)`, then restore the original -- so this
    test never touches whatever cache state other tests in the same process left behind,
    and never leaks its own scratch dir into the suite's real one."""
    tmp = tempfile.mkdtemp(prefix="tierq48_t2_")
    fresh = TEXCache(cache_dir=Path(tmp))
    prev = _tex_cache._cache_instance
    _tex_cache._cache_instance = fresh
    try:
        fn(fresh, tmp)
    finally:
        _tex_cache._cache_instance = prev
        shutil.rmtree(tmp, ignore_errors=True)


def test_fixtier_t2_verdict_no_disk_write(r: SubTestResult):
    print("\n--- FIX-TIER T2: tier_verdict(scale=..., binding_types=...) writes no .pkl ---")

    def _check(fresh, tmp):
        before = list(Path(tmp).rglob("*.pkl"))
        if before:
            r.fail("precondition", f"fresh cache dir already had files: {before}")
            return
        verdict = tier_verdict(_CODE, scale=0.5, binding_types=_BT, roi_exec=True)
        after = list(Path(tmp).rglob("*.pkl"))
        if after:
            r.fail("no disk write", f"tier_verdict() left {len(after)} .pkl file(s) on "
                   f"disk for a program nobody cooked: {after}; verdict={verdict}")
            return
        r.ok(f"tier_verdict() left no .pkl on disk (verdict.tier={verdict.tier!r})")

    _run_against_fresh_cache(_check)


def test_fixtier_t2_verdict_no_memory_write(r: SubTestResult):
    print("\n--- FIX-TIER T2: tier_verdict(scale=..., binding_types=...) writes no memory entry ---")

    def _check(fresh, tmp):
        if fresh._memory:
            r.fail("precondition", "fresh cache already has memory entries")
            return
        tier_verdict(_CODE, scale=0.5, binding_types=_BT, roi_exec=True)
        if fresh._memory:
            r.fail("no memory write", f"tier_verdict() populated the memory cache with "
                   f"{len(fresh._memory)} entr(y/ies) for a program nobody cooked")
            return
        r.ok("tier_verdict() left the in-memory program cache empty")

    _run_against_fresh_cache(_check)
