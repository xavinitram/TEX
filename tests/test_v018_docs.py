"""
v0.18.0 doc-integrity checks (DOC-7b/7c/7d/7e + REG-1b).

The entry-point map (AGENTS.md) must not lie to the next agent session. DOC-7b turns
the LOC-budget table into a drift canary: every `~LOC` figure it quotes is checked
against the real `wc -l`, so a stale number (the "~3700" vs 2731 the split left behind)
reds the suite instead of misleading a reader.

DOC-7c widens that canary. DOC-7b parsed only the three rows of the budget TABLE at a
20 % band, which let four things rot at once: a prose `~LOC` figure was never read at
all, a module named nowhere in the table could drift without limit, the band was wide
enough to hide a ~600-line move in the largest module, and the `_HEADROOM_FLOOR` /
`_OVER_HARD_BASELINE` registers could name a module the map never mentions — which is
exactly what happened to `tex_results.py`, whose floor lived in the test file for a
release before the map said the word. So: every `~LOC` figure, wherever it is written;
10 % band; and every module the LOC registers pin must be a module the map names.

DOC-7d and DOC-7e close the other two halves of the same problem. A register that is
checked by *counting* is not checked: `test_c6st_cache_count_agree` only proves two
documents quote the same integer, and five module-level memo stores landed while it was
green. DOC-7d enumerates instead — an AST census of every module-level mutable container
in the product packages, each of which must be named in ARCHITECTURE.md's register or
carry a row in `_NOT_A_CACHE` saying why it is not one. DOC-7e binds AGENTS.md's
escape-hatch register to README's environment-switch table, so the sentence AGENTS.md
already asserts ("Every one of them is listed in README.md") is machine-checked in both
directions.
"""
from helpers import *
import ast
import re

_PKG = Path(__file__).resolve().parent.parent


def _loc(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


#: DOC-7c's drift band. 20 % on `tex_runtime/codegen.py` is ±624 lines — wider than the
#: whole of `tex_chain.py` — so the old band could not have reported the STR-7 split it was
#: written to catch until the split was most of the way done. 10 % still absorbs a normal
#: release's growth without asking anyone to re-type a number every commit.
_LOC_DRIFT_BAND = 0.10

#: A product module path as the map spells it: `tex_engine.py` at the root, or a module of
#: one of the product packages. `tools/`, `tests/` and `benchmarks/` are not the map's subject.
_MODULE_RE = r"(?:tex_(?:compiler|runtime|io)/[\w]+\.py|tex_[\w]+\.py)"


def _agents_loc_claims(text: str) -> dict:
    """Every `~LOC` figure AGENTS.md states, table row or prose, as {path: stated}.

    Three spellings are in use and all three are load-bearing, because a figure that the
    parser cannot see is a figure nobody is checking:
      * the budget table            — ``| `tex_runtime/codegen.py` | ~3080 | …``
      * a parenthesised prose aside — ``` `tex_compiler/optimizer.py` (~1540) is over *soft* ```
      * a prose sentence            — ``` `tex_engine.py` is 1318 LOC ```
    """
    claims = {}
    for pat in (rf"\|\s*`({_MODULE_RE})`\s*\|\s*~?([\d,]+)\s*\|",
                rf"`({_MODULE_RE})`\s*\(~?([\d,]+)\)",
                rf"`({_MODULE_RE})`\s+is\s+~?([\d,]+)\s+LOC"):
        for rel, stated in re.findall(pat, text):
            claims[rel] = int(stated.replace(",", ""))
    return claims


def _agents_names_module(text: str, module: str) -> bool:
    """Does AGENTS.md name *module* — by its package-relative path or its bare basename?

    The two LOC registers spell their keys differently (`_OVER_HARD_BASELINE` carries
    `tex_runtime/codegen.py`, `_HEADROOM_FLOOR` carries `tex_engine.py`), and the map is
    equally free about it, so a match on either spelling counts.
    """
    base = module.rsplit("/", 1)[-1]
    return f"`{module}`" in text or f"`{base}`" in text


def test_doc7b_map_drift(r: SubTestResult):
    print("\n--- DOC-7b/7c: AGENTS.md module-LOC map-drift canary ---")
    agents = _PKG / "AGENTS.md"
    try:
        text = agents.read_text(encoding="utf-8")
    except Exception as e:
        r.fail("DOC-7b read AGENTS.md", str(e))
        return

    claims = _agents_loc_claims(text)
    if not claims:
        r.fail("DOC-7c claim parse", "no `module.py` + ~LOC figure found in AGENTS.md")
        return

    fails, checked = [], 0
    for rel, stated in sorted(claims.items()):
        f = _PKG / rel
        if not f.exists():
            fails.append(f"{rel}: given a ~LOC figure in AGENTS.md but the file is missing")
            continue
        actual = _loc(f)
        checked += 1
        drift = abs(actual - stated) / max(actual, 1)
        if drift > _LOC_DRIFT_BAND:
            fails.append(f"{rel}: AGENTS.md says ~{stated} but file is {actual} "
                         f"({drift*100:.1f}% drift > {_LOC_DRIFT_BAND*100:.0f}%)")

    # the STR-7 split shipped: its three modules must exist (so the stale
    # "cluster 1 done / next stencil+persist" language can never truthfully return).
    for mod in ("tex_runtime/codegen_stdfns.py", "tex_runtime/codegen_stencil.py",
                "tex_runtime/codegen_persist.py"):
        if not (_PKG / mod).exists():
            fails.append(f"{mod}: STR-7 split module missing — split status is a lie")

    # stale phrasings that the split made false — pin them out.
    for stale in ("cluster 1 done", "STR-7 in progress"):
        if stale in text:
            fails.append(f"stale phrase {stale!r} still in AGENTS.md (split shipped)")

    if fails:
        r.fail("DOC-7c map drift", "; ".join(fails))
    else:
        r.ok(f"AGENTS.md map is honest ({checked} LOC claim(s) within "
             f"{_LOC_DRIFT_BAND*100:.0f}%; split modules present)")


def test_doc7c_loc_registers_are_on_the_map(r: SubTestResult):
    print("\n--- DOC-7c: every module the LOC registers pin is named in AGENTS.md ---")
    # The REG-2 registers live in the TEST file; the map an implementer plans against is
    # AGENTS.md. When the two drift, a lane budgets against a wall that is not where the
    # gate put it — `tex_results.py` carried a 1958 floor for a release while AGENTS.md's
    # headroom paragraph still described a one-module register, and the only person who
    # noticed was forbidden from editing the file (NEG-1 F-3, reported and left open).
    # Binding them means the next floor cannot land without the map moving with it.
    try:
        text = (_PKG / "AGENTS.md").read_text(encoding="utf-8")
    except Exception as e:
        r.fail("DOC-7c read AGENTS.md", str(e))
        return
    try:
        import importlib
        phase2 = importlib.import_module("test_v017_phase2")
        floors = dict(phase2._HEADROOM_FLOOR)
        baseline = set(phase2._OVER_HARD_BASELINE)
    except Exception as e:
        r.fail("DOC-7c import registers", f"{type(e).__name__}: {e}")
        return

    unnamed = sorted(m for m in (set(floors) | baseline) if not _agents_names_module(text, m))
    if unnamed:
        r.fail("DOC-7c register on the map",
               "module(s) pinned by tests/test_v017_phase2.py that AGENTS.md never names "
               "(a budget an implementer cannot read is a budget they will plan past): "
               + ", ".join(unnamed))
    else:
        r.ok(f"all {len(floors) + len(baseline)} pinned module(s) are named in AGENTS.md")

    # The floor VALUES have to be on the map too, not just the module names: a floor that
    # moves down in the register and not in the prose re-creates exactly the NEG-1 F-3 gap
    # one release later. Scoped to the REG-2 section so an unrelated number cannot satisfy it.
    section = text.split("## Module size budget", 1)[-1].split("\n## ", 1)[0]
    missing = sorted(f"{m}={v}" for m, v in floors.items() if str(v) not in section)
    if missing:
        r.fail("DOC-7c floor values on the map",
               "AGENTS.md's module-size-budget section does not state the current floor(s) "
               "from tests/test_v017_phase2.py::_HEADROOM_FLOOR: " + ", ".join(missing))
    else:
        r.ok("every _HEADROOM_FLOOR value appears in AGENTS.md's module-size-budget section")


def test_c3ux_error_codes_resolve(r: SubTestResult):
    print("\n--- C3-ux: every source error code has a resolving anchor in Error-Codes.md ---")
    # Every rendered diagnostic links to wiki/Error-Codes#e<NNNN>; the page must have an
    # anchor for each code used in the source, and the generator must be in sync.
    import sys as _sys
    sys_path0 = str(_PKG)
    if sys_path0 not in _sys.path:
        _sys.path.insert(0, sys_path0)
    try:
        from tools import gen_error_codes as gen
    except Exception as e:
        r.fail("C3-ux import generator", f"{type(e).__name__}: {e}")
        return
    # The page this checks is the PACKAGE-ROOT copy (LANG-7), which ships with the node and
    # is what `tools/gen_error_codes.py --check` reads. It used to check `wiki/Error-Codes.md`
    # instead — a separate, gitignored checkout that is on no machine and in no CI clone, so
    # the check skipped everywhere and the drift it names was verified nowhere. The wiki copy
    # is the same bytes from the same generator, so it is checked too WHEN PRESENT rather than
    # being the only thing that can be checked.
    page = _PKG / "Error-Codes.md"
    if not page.exists():
        r.fail("C3-ux Error-Codes.md", "the shipped package-root Error-Codes.md is missing — "
                                       "run tools/gen_error_codes.py")
        return
    text = page.read_text(encoding="utf-8").lower()
    codes = gen.harvest_codes()
    missing = [c for c in codes if f"### {c.lower()}" not in text]
    if missing:
        r.fail("C3-ux anchors", f"{len(missing)} source codes have no anchor: {missing[:10]}")
        return
    # in-process staleness check (no subprocess): the rendered content must match the file
    rendered = gen.render(codes)
    wiki = _PKG / "wiki" / "Error-Codes.md"
    if wiki.exists() and wiki.read_text(encoding="utf-8") != rendered:
        r.fail("C3-ux wiki drift", "wiki/Error-Codes.md is stale against the shipped copy — "
                                   "regenerate (tools/gen_error_codes.py)")
    elif wiki.exists():
        r.ok("wiki/Error-Codes.md is present and matches the shipped copy")
    if rendered != page.read_text(encoding="utf-8"):
        r.fail("C3-ux drift", "Error-Codes.md is stale — regenerate (tools/gen_error_codes.py)")
    else:
        r.ok(f"all {len(codes)} source error codes resolve to an anchor; page in sync")


def test_c6st_cache_count_agree(r: SubTestResult):
    print("\n--- C6-st: AGENTS.md and ARCHITECTURE.md agree on the cache count ---")
    # doc 34/35 C6: the two docs disagreed (13 vs 14) and both missed _AUTO_DECISION.
    # Machine-check they quote the SAME number so this class of drift can't recur.
    def _count(fname):
        text = (_PKG / fname).read_text(encoding="utf-8")
        m = re.search(r"[Tt]he\s+(\d+)[ -]cache", text)
        return int(m.group(1)) if m else None
    a, c = _count("AGENTS.md"), _count("ARCHITECTURE.md")
    if a is None or c is None:
        r.fail("C6-st cache count", f"couldn't find the count (AGENTS={a}, ARCHITECTURE={c})")
    elif a != c:
        r.fail("C6-st cache count", f"AGENTS.md says {a} caches, ARCHITECTURE.md says {c} — reconcile")
    else:
        r.ok(f"cache count agrees across both docs ({a})")


def test_c5ux_no_render_overstatement(r: SubTestResult):
    print("\n--- C5-ux: docs don't overstate what actually renders on-node ---")
    # v0.18 shipped CHANGELOG/README claims that debug_print "surface[s] on the node"
    # and the HUD/doctor render, before the DOM render path existed (doc 33/34). Guard
    # the specific overstatements so they can't silently return; when Phase-3 lands the
    # real rendering, THESE strings stay retired (the true claim uses different words).
    banned = [
        "results surface on the node",
        "surfaces on the node",  # bare debug_print render claim without the ui= caveat
    ]
    hits = []
    for fname in ("CHANGELOG.md", "README.md"):
        text = (_PKG / fname).read_text(encoding="utf-8").lower()
        for phrase in banned:
            if phrase in text:
                hits.append(f"{fname}: '{phrase}'")
    if hits:
        r.fail("C5-ux overstatement guard",
               "; ".join(hits) + " — debug_print returns values via the ui payload; "
               "describe rendering honestly (see doc 35 C5-ux)")
    else:
        r.ok("no retired render-overstatement phrasings in CHANGELOG/README")


def test_reg1b_doc_ex_populated(r: SubTestResult):
    print("\n--- REG-1b: every shipped stdlib fn carries doc= and ex= ---")
    from TEX_Wrangle.tex_runtime import stdlib_registry as R
    from TEX_Wrangle.tex_runtime import stdlib  # noqa: populate REGISTRY
    empty_doc = [e.name for e in R.REGISTRY if not e.doc.strip()]
    empty_ex = [e.name for e in R.REGISTRY if not e.ex.strip()]
    if empty_doc or empty_ex:
        r.fail("REG-1b doc/ex coverage",
               f"{len(empty_doc)} empty doc {empty_doc[:8]}; "
               f"{len(empty_ex)} empty ex {empty_ex[:8]} — the registry is the single "
               "prose source; a shipped fn with no doc/ex is a drift")
    else:
        r.ok(f"all {len(R.REGISTRY)} registered fns carry non-empty doc= and ex=")
