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
            or n in ("erode", "dilate", "convolve") or n.endswith("_filter")
            # ASK-13: patch_dist escapes all five prefixes above (as convolve did too,
            # before ASK-1 added it by name).
            or n.startswith("patch_")
            # ASK-4: the whole-image family (img_sum/mean/min/max/median, now also
            # img_width/img_height) reads the ENTIRE binding, not just a neighbourhood
            # — still not 'point'. Pins the five existing names too.
            or n.startswith("img_"))


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

    # (4) FORGOTTEN-TAG CATCH (ROI-1: now in footprint terms) — a sample_*/fetch_*/blur/
    #     erode/dilate/*_filter fn MUST carry a non-'point' footprint. This turns the
    #     single most dangerous LLM stdlib edit (adding a neighbour-reading fn without
    #     classifying its footprint) from silent-wrong-if-tiled into a red test.
    try:
        untagged = sorted(n for e in R.REGISTRY for n in e.names
                          if _looks_spatial(n) and e.footprint == "point")
        assert not untagged, (f"spatial-named fns left at the default 'point' footprint "
                              f"(would tile WRONG): {untagged}")
        r.ok("every neighbour-reading-named fn has a non-'point' footprint (tiling-safe)")
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

    # (3) LANG-4 FLIP — the function help DATA now lives in the registry (single source);
    #     the JS TEX_HELP_DATA sig for every function MIRRORS it, so neither can silently
    #     drift from the other. (Descriptions legitimately differ: the JS panel keeps its
    #     richer prose; only the migrated `sig` is pinned.)
    try:
        import TEX_Wrangle.tex_runtime.stdlib  # noqa: F401 (populate REGISTRY)
        from TEX_Wrangle.tex_runtime import stdlib_registry as R
        entries, _cats = G.parse_help()
        drift = []
        for e in R.REGISTRY:
            js = entries.get(e.name)
            if js is None:
                drift.append(f"{e.name}: no JS entry")
            elif js["sig"] != e.sig:
                drift.append(f"{e.name}: JS {js['sig']!r} != registry {e.sig!r}")
        assert not drift, "; ".join(drift[:6])
        r.ok(f"all {len(R.REGISTRY)} function sigs agree between the registry and JS TEX_HELP_DATA")
    except Exception as e:
        r.fail("LANG-4 JS↔registry sig", f"{type(e).__name__}: {e}")

    # (4) the registry-sourced help JSON (tex_help.json) is current.
    try:
        import gen_help_data as GH
        text = GH.render(GH.build())
        committed = open(os.path.join(_repo_root(), "tex_help.json"),
                         encoding="utf-8").read().replace("\r\n", "\n")
        assert committed == text.replace("\r\n", "\n"), \
            "tex_help.json is stale — run `python tools/gen_help_data.py`"
        r.ok("tex_help.json matches a fresh regen (registry-sourced help JSON is current)")
    except Exception as e:
        r.fail("LANG-4 help JSON drift", f"{type(e).__name__}: {e}")


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
# LIB-1 took `tex_runtime/stdlib.py`'s planned split — it is now the facade (126 LOC) over
# seven domain leaves, none of which is anywhere near the hard budget, so it is re-pinned
# DOWN out of this baseline per the ratchet's own stale-baseline rule below.
# SPLIT-I (v0.44 Phase A1) took `tex_runtime/interpreter.py`'s planned split — the
# spatial-context, control-flow and binding-write seams moved to
# `interpreter_spatial.py` / `interpreter_control_flow.py` / `interpreter_binding.py`
# (mixins `Interpreter` still inherits, STR-7's pattern), leaving it at 1978, so it is
# re-pinned DOWN out of this baseline too.
_LOC_HARD, _LOC_SOFT = 2000, 1500
_OVER_HARD_BASELINE = frozenset({
    "tex_runtime/codegen.py",
})

# ENG-14 — a named module's HEADROOM floor, asserted separately from the ratchet.
# The ratchet only reds ABOVE the hard budget, which is too late for a module that a
# planned feature has to grow: `tex_engine.py` sat at exactly 2000/2000, so the next
# line added to it was a red test and the split had to be done under that pressure.
# A floor is the ratchet run early — it reds while there is still room to act.
# The floor MOVES DOWN, never up: raising it is how a budget becomes decoration.
# `tex_results.py` joins the register at 1886/2000: 114 lines from the wall, NOT
# grandfathered, and with the CACHE family still live work — exactly the situation ENG-14
# was created to prevent, one release later and in a different module. 1958 is the smallest
# floor that passes today while keeping the 72 lines of room ENG-14 gave `tex_engine.py`
# (1628 under 1700). It buys 42 lines of warning before the hard budget, which is where a
# split gets planned instead of improvised. Same rule: it moves DOWN when the split lands.
# NEG-2 moved tex_engine.py's floor down for the second time: the CHAIN split took the
# file from 1673 to 1318, so its floor goes 1700 -> 1400 — the same ~85-line margin
# ENG-14 chose over its own post-split size (1613 under 1700), which is one median
# release of growth. Two floors now, moving down independently as their splits land.
# NEG-6 moved tex_results.py's floor down for the first time since NEG-1 set it. CACHE-11
# landed on `main` first and grew the file from NEG-1's 1903 to 1944 — 14 lines under the
# 1958 floor, the closest this register has come to a wall since ENG-14's own. The
# pre-specified CACHE-1 key-minting cut (`tex_results_keys.py`) then took it from 1944 to
# 1815, so its floor goes 1958 -> 1887 — the same 72-line margin this module's own NEG-1
# precedent chose, kept rather than re-picked, so the register's margins stay comparable
# across modules instead of drifting per lane. 72 lines of room again, not 14.
# SPLIT-R (v0.44) moved tex_results.py's floor down for the second time, sitting AT it —
# `tex_results.py` was 1887/1887 when this landed, i.e. NEG-6's own 72-line margin had
# already been spent by later work with no split of its own. The pre-specified CACHE-8
# residency-ladder cut (`set_vram_budget`/`_enforce_residency`/`_queue_demotions`/
# `_drain_demotes`/`_promote`, mixed back in via `_ResultCacheResidency`) took it from 1887
# to 1651, so its floor goes 1887 -> 1810 — a 159-line margin, wider than NEG-6's 72: a
# floor sitting exactly AT the wall is the situation ENG-14 was created to prevent, so this
# cut buys more warning than the last one rather than the same amount again.
_HEADROOM_FLOOR = {"tex_engine.py": 1400, "tex_results.py": 1810}


def _product_packages(root) -> list:
    """The package directories the REG-2 ratchet scans — DERIVED from the tree.

    This list used to be the literal `("tex_compiler", "tex_runtime", "")`, and a literal is
    a gate that cannot fire on a package nobody remembered to add to it: `tex_io/` was
    invisible to the ratchet from the day it was created, so a new module there could cross
    the hard budget without a word. Derived instead, from two facts already in the tree:

      * a product package is a top-level directory carrying an `__init__.py` (it is importable
        as part of the node), and
      * it is not one of the development directories `.comfyignore` keeps out of the shipped
        archive (PUB-1 restricts that file to plain `name/` patterns, so reading it is exact).

    A package added tomorrow is scanned tomorrow, with no edit here. The empty string is the
    package root itself.
    """
    import os
    ignored = set()
    try:
        with open(os.path.join(root, ".comfyignore"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ignored.add(line.rstrip("/"))
    except OSError:
        pass
    subs = [""]
    for name in sorted(os.listdir(root)):
        if name.startswith((".", "_")) or name in ignored:
            continue
        d = os.path.join(root, name)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "__init__.py")):
            subs.append(name)
    return subs


def test_reg2_loc_budget(r: SubTestResult):
    print("\n--- REG-2: module LOC budget (soft policy + ratchet) ---")
    import os
    root = _repo_root()
    subs = _product_packages(root)
    loc = {}
    for sub in subs:
        d = os.path.join(root, sub)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".py") and not fn.startswith("__"):
                rel = f"{sub}/{fn}" if sub else fn
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    loc[rel] = sum(1 for _ in f)
    r.ok(f"scanned {len(loc)} module(s) across "
         + ", ".join(repr(s) if s else "<root>" for s in subs))

    # The derivation must not silently scan NOTHING: every module this file pins by name has
    # to be one the scan can see, or the pin is decoration and the derivation is broken.
    try:
        blind = sorted(m for m in (set(_OVER_HARD_BASELINE) | set(_HEADROOM_FLOOR))
                       if m not in loc)
        assert not blind, (
            "module(s) pinned by this file are outside the scanned packages — either the "
            "package derivation broke or the module moved/was renamed: " + str(blind))
        r.ok("every pinned module is inside the scanned set")
    except Exception as e:
        r.fail("REG-2 scan coverage", f"{type(e).__name__}: {e}")

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
        # STALE — a grandfathered module that has been split back under the wall. This is
        # the PUB-1 ratchet's "re-pin DOWN" rule (tests/test_pub1_archive.py): a bound that
        # only ever loosens is decoration. Printing it, as this arm used to, left the
        # baseline free to keep claiming a module is over budget years after its split
        # landed — and the next reader grants that module a licence it no longer has.
        stale = sorted(m for m in _OVER_HARD_BASELINE if m in loc and loc[m] <= _LOC_HARD)
        assert not stale, (
            "baseline module(s) are now UNDER the hard budget — re-pin DOWN by removing "
            "the literal(s) from _OVER_HARD_BASELINE in this file: "
            + ", ".join(f"{m!r} ({loc[m]} <= {_LOC_HARD})" for m in stale))
        r.ok("ratchet holds — no new module crossed the hard LOC budget, "
             "and every grandfathered module is still over it")
    except Exception as e:
        r.fail("REG-2 ratchet", f"{type(e).__name__}: {e}")

    # HEADROOM FLOOR (ENG-14) — a named module must keep room for its next feature.
    try:
        over_floor = sorted((m, loc[m], f) for m, f in _HEADROOM_FLOOR.items()
                            if m in loc and loc[m] > f)
        assert not over_floor, (
            "module(s) over their ENG-14 headroom floor — split by domain rather than "
            f"raising the floor: {over_floor}")
        r.ok("headroom floors hold: "
             + ", ".join(f"{m} {loc.get(m, '?')}/{f}" for m, f in sorted(_HEADROOM_FLOOR.items())))
    except Exception as e:
        r.fail("REG-2 headroom floor", f"{type(e).__name__}: {e}")
