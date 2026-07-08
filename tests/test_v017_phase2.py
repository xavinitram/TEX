"""
v0.17 Phase 2 — the single-source spine.

REG-1 registry parity · TST-3 taxonomy consistency.
"""
from helpers import *
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib
from TEX_Wrangle.tex_runtime import stdlib_registry as R
from TEX_Wrangle.tex_compiler.stdlib_signatures import FUNCTION_SIGNATURES


def test_reg1_registry_parity(r: SubTestResult):
    print("\n--- REG-1: single-source stdlib registry ---")
    gf = TEXStdlib.get_functions()

    # (a) get_functions() is now the registry view and still exposes every name.
    try:
        assert len(R.REGISTRY) > 0, "REGISTRY empty — was stdlib.py imported?"
        assert set(gf) == set(FUNCTION_SIGNATURES), (
            "registry names != FUNCTION_SIGNATURES: "
            f"{set(gf) ^ set(FUNCTION_SIGNATURES)}")
        r.ok(f"registry view exposes all {len(gf)} names, parity with FUNCTION_SIGNATURES")
    except Exception as e:
        r.fail("REG-1 name parity", f"{type(e).__name__}: {e}")

    # (b) Each view fn IS the class attribute — behaviour is byte-identical to the
    #     old hand-listed dict (no wrapper, no re-bind).
    try:
        bad = [n for n, f in gf.items() if f is not getattr(TEXStdlib, f.__name__)]
        assert not bad, f"view fn is not the class attr for: {bad[:8]}"
        r.ok("every registered fn IS its TEXStdlib attr (identity preserved)")
    except Exception as e:
        r.fail("REG-1 identity", f"{type(e).__name__}: {e}")

    # (c) Aliases resolve to the same impl (lerp/mix share fn_lerp).
    try:
        assert "mix" in gf and gf["mix"] is gf["lerp"], "lerp/mix alias broken"
        r.ok("alias lerp/mix resolves to one impl")
    except Exception as e:
        r.fail("REG-1 alias", f"{type(e).__name__}: {e}")

    # (d) The decorator is pure data attachment — no duplicate registrations.
    try:
        names = [n for e in R.REGISTRY for n in e.names]
        dupes = [n for n in names if names.count(n) > 1]
        assert not dupes, f"duplicate registrations: {sorted(set(dupes))}"
        r.ok(f"{len(R.REGISTRY)} entries, {len(names)} names, no duplicates")
    except Exception as e:
        r.fail("REG-1 no-dupes", f"{type(e).__name__}: {e}")


# `blur`/`sample_lod` live in _SYNC_STDLIB but have no impl — the graph tier guards
# these names defensively; documented here so the parity check stays honest.
_SYNC_PHANTOMS = frozenset({"blur", "sample_lod"})


def _looks_spatial(n):
    """Name-prefix heuristic for functions that read neighbouring pixels — the
    class that is silently WRONG (tiled into bad output) if left un-tagged."""
    return (n.startswith("sample") or n.startswith("fetch") or "blur" in n
            or n in ("erode", "dilate") or n.endswith("_filter"))


def test_tst3_taxonomy_consistency(r: SubTestResult):
    print("\n--- TST-3: stdlib taxonomy consistency (derive + check from registry) ---")
    from TEX_Wrangle.tex_runtime import graphed, codegen
    from TEX_Wrangle import tex_memory
    names = {n for e in R.REGISTRY for n in e.names}
    tagged = lambda attr: {n for e in R.REGISTRY for n in e.names if getattr(e, attr)}

    # (1) DERIVATION — the registry's tags reproduce each taxonomy set exactly, so
    #     the parallel hand-tables can be *derived* (and thus can't silently drift).
    try:
        derivation = [
            ("spatial",   tagged("spatial"),   set(codegen._SPATIAL_STDLIB),   set()),
            ("non_local", tagged("non_local"), set(tex_memory._NON_LOCAL_FNS), set()),
            ("sync",      tagged("sync"),      set(graphed._SYNC_STDLIB),      _SYNC_PHANTOMS),
        ]
        fails = [f"{tag}: derived {d ^ (a - ph)} differs"
                 for tag, d, a, ph in derivation if d != a - ph]
        assert not fails, "; ".join(fails)
        r.ok("registry tags derive _SPATIAL_STDLIB / _SYNC_STDLIB / _NON_LOCAL_FNS exactly")
    except Exception as e:
        r.fail("TST-3 derivation", f"{type(e).__name__}: {e}")

    # (2) MEMBERSHIP — every taxonomy-set member is a registered name (catches a set
    #     entry left behind by a deleted/renamed fn), bar the documented sync phantoms.
    try:
        membership = [
            ("_SPATIAL_STDLIB", set(codegen._SPATIAL_STDLIB),   set()),
            ("_SYNC_STDLIB",    set(graphed._SYNC_STDLIB),      _SYNC_PHANTOMS),
            ("_NON_LOCAL_FNS",  set(tex_memory._NON_LOCAL_FNS), set()),
            ("_MIP_FAMILY",     set(tex_memory._MIP_FAMILY),    set()),
        ]
        fails = [f"{label}: non-registry names {sorted(s - names - allowed)}"
                 for label, s, allowed in membership if s - names - allowed]
        assert not fails, "; ".join(fails)
        r.ok("every taxonomy-set member is a registered name (phantoms documented)")
    except Exception as e:
        r.fail("TST-3 membership", f"{type(e).__name__}: {e}")

    # (3) NO DEAD DISPATCH — every codegen _fn_dispatch key is a registered name.
    try:
        cg = codegen._CodeGen({})
        dead = sorted(set(cg._fn_dispatch) - names)
        assert not dead, f"codegen _fn_dispatch has non-registry keys: {dead}"
        r.ok(f"all {len(cg._fn_dispatch)} codegen dispatch keys are registered names")
    except Exception as e:
        r.fail("TST-3 dispatch", f"{type(e).__name__}: {e}")

    # (4) FORGOTTEN-TAG CATCH — a sample_*/fetch_*/blur/erode/dilate/*_filter fn MUST be
    #     non_local. This turns the single most dangerous LLM stdlib edit (adding a
    #     neighbour-reading fn without tagging it) from silent-wrong into a red test.
    try:
        untagged = sorted(n for e in R.REGISTRY for n in e.names
                          if _looks_spatial(n) and not e.non_local)
        assert not untagged, (f"spatial-named fns missing non_local=True (would tile "
                              f"WRONG): {untagged}")
        r.ok("every neighbour-reading-named fn is tagged non_local (tiling-safe)")
    except Exception as e:
        r.fail("TST-3 forgotten-tag", f"{type(e).__name__}: {e}")


def _repo_root():
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_doc4_reference(r: SubTestResult):
    print("\n--- DOC-4: Function-Reference.md generated + drift-checked ---")
    import os
    import sys
    sys.path.insert(0, os.path.join(_repo_root(), "tools"))
    try:
        import gen_function_reference as G
        md, documented, reg_names = G.generate()

        # (1) COVERAGE — every registered function has a TEX_HELP_DATA entry, so no
        #     function ships undocumented (the measured wiki-drift, now CI-caught).
        missing = sorted(reg_names - documented)
        assert not missing, f"functions with no TEX_HELP_DATA help entry: {missing}"
        r.ok(f"all {len(reg_names)} registered functions have an editor help entry")
    except Exception as e:
        r.fail("DOC-4 coverage", f"{type(e).__name__}: {e}")
        return

    # (2) DRIFT — the committed reference matches a fresh regen (it is a *view*, not a
    #     hand-doc, so it cannot silently fall out of date).
    try:
        ref = os.path.join(_repo_root(), "Function-Reference.md")
        committed = open(ref, encoding="utf-8").read().replace("\r\n", "\n")
        assert committed == md.replace("\r\n", "\n"), (
            "Function-Reference.md is stale — run `python tools/gen_function_reference.py`")
        r.ok("Function-Reference.md matches a fresh regen (view is current)")
    except Exception as e:
        r.fail("DOC-4 drift", f"{type(e).__name__}: {e}")


def test_doc5_examples_index(r: SubTestResult):
    print("\n--- DOC-5: examples/INDEX.md generated + header contract + soft coverage ---")
    import os
    import sys
    sys.path.insert(0, os.path.join(_repo_root(), "tools"))
    try:
        import gen_examples_index as G
        # generate() calls parse_examples(), which RAISES on any example whose first
        # line isn't `// Name — desc` — so a green run IS the header-format contract.
        md, rows, covered, uncovered = G.generate()
        assert len(rows) > 100, f"only parsed {len(rows)} example headers"
        r.ok(f"all {len(rows)} example headers match the `// Name — desc` contract")
    except Exception as e:
        r.fail("DOC-5 headers", f"{type(e).__name__}: {e}")
        return

    # DRIFT — the committed index matches a fresh regen.
    try:
        idx = os.path.join(_repo_root(), "examples", "INDEX.md")
        committed = open(idx, encoding="utf-8").read().replace("\r\n", "\n")
        assert committed == md.replace("\r\n", "\n"), (
            "examples/INDEX.md is stale — run `python tools/gen_examples_index.py`")
        r.ok("examples/INDEX.md matches a fresh regen")
    except Exception as e:
        r.fail("DOC-5 drift", f"{type(e).__name__}: {e}")

    # COVERAGE — SOFT: report, never fail (a gap is a nudge to add an example).
    total = len(covered) + len(uncovered)
    r.ok(f"[soft] {len(covered)}/{total} stdlib functions exercised by an example "
         f"({len(uncovered)} uncovered — see INDEX.md)")


# Modules already over the 2000 hard budget, each with a Phase-3 split planned
# (see AGENTS.md). Grandfathered — the ratchet fails only on a NEW crossing.
_LOC_HARD, _LOC_SOFT = 2000, 1500
_OVER_HARD_BASELINE = frozenset({
    "tex_runtime/codegen.py", "tex_runtime/stdlib.py", "tex_runtime/interpreter.py",
})


def test_reg2_loc_budget(r: SubTestResult):
    print("\n--- REG-2: module LOC budget (soft policy + ratchet) ---")
    import os
    root = _repo_root()
    loc = {}
    for sub in ("tex_compiler", "tex_runtime", ""):
        d = os.path.join(root, sub)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".py") and not fn.startswith("__"):
                rel = f"{sub}/{fn}" if sub else fn
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    loc[rel] = sum(1 for _ in f)

    over_hard = {m for m, n in loc.items() if n > _LOC_HARD}
    over_soft = {m for m, n in loc.items() if _LOC_SOFT < n <= _LOC_HARD}
    # SOFT report — informational, always passes.
    r.ok(f"[soft] {len(over_soft)} module(s) over soft ({_LOC_SOFT}), "
         f"{len(over_hard)} over hard ({_LOC_HARD}); planned splits in Phase 3")

    # RATCHET — a NEW module over the hard budget is a regression (the baseline is
    # grandfathered pending its planned split).
    try:
        new_over = sorted(over_hard - _OVER_HARD_BASELINE)
        assert not new_over, (f"module(s) newly over the {_LOC_HARD}-LOC hard budget "
                              f"(split by domain, or update the baseline with a plan): "
                              f"{[(m, loc[m]) for m in new_over]}")
        # also flag if a baseline module was split away (keep the baseline honest)
        stale = sorted(m for m in _OVER_HARD_BASELINE if m in loc and loc[m] <= _LOC_HARD)
        note = f" (baseline modules now under budget — prune them: {stale})" if stale else ""
        r.ok(f"ratchet holds — no new module crossed the hard LOC budget{note}")
    except Exception as e:
        r.fail("REG-2 ratchet", f"{type(e).__name__}: {e}")
