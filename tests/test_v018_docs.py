"""
v0.18.0 doc-integrity checks (DOC-7b + REG-1b).

The entry-point map (AGENTS.md) must not lie to the next agent session. DOC-7b turns
the LOC-budget table into a drift canary: every `~LOC` figure it quotes is checked
against the real `wc -l`, so a stale number (the "~3700" vs 2731 the split left behind)
reds the suite instead of misleading a reader.
"""
from helpers import *
import re

_PKG = Path(__file__).resolve().parent.parent


def _loc(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def test_doc7b_map_drift(r: SubTestResult):
    print("\n--- DOC-7b: AGENTS.md module-LOC map-drift canary ---")
    agents = _PKG / "AGENTS.md"
    try:
        text = agents.read_text(encoding="utf-8")
    except Exception as e:
        r.fail("DOC-7b read AGENTS.md", str(e))
        return

    # rows of the budget table: | `tex_runtime/codegen.py` | ~2730 | ... |
    rows = re.findall(r"\|\s*`([\w./]+\.py)`\s*\|\s*~?([\d,]+)\s*\|", text)
    if not rows:
        r.fail("DOC-7b table parse", "no `module.py | ~LOC` rows found in AGENTS.md")
        return

    fails, checked = [], 0
    for rel, stated_raw in rows:
        stated = int(stated_raw.replace(",", ""))
        f = _PKG / rel
        if not f.exists():
            fails.append(f"{rel}: listed in AGENTS.md but file missing")
            continue
        actual = _loc(f)
        checked += 1
        drift = abs(actual - stated) / max(actual, 1)
        if drift > 0.20:
            fails.append(f"{rel}: AGENTS.md says ~{stated} but file is {actual} "
                         f"({drift*100:.0f}% drift > 20%)")

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
        r.fail("DOC-7b map drift", "; ".join(fails))
    else:
        r.ok(f"AGENTS.md map is honest ({checked} LOC rows within 20%; split modules present)")


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
    page = _PKG / "wiki" / "Error-Codes.md"
    if not page.exists():
        # wiki/ is a separate (gitignored) repo checkout — absent on a fresh CI
        # clone. The drift check is a dev-time guard; skip when the page isn't here
        # rather than failing CI for a file this repo intentionally doesn't track.
        r.skip("C3-ux Error-Codes.md", "wiki/ checkout absent (separate wiki repo); run tools/gen_error_codes.py in a wiki checkout")
        return
    text = page.read_text(encoding="utf-8").lower()
    codes = gen.harvest_codes()
    missing = [c for c in codes if f"### {c.lower()}" not in text]
    if missing:
        r.fail("C3-ux anchors", f"{len(missing)} source codes have no anchor: {missing[:10]}")
        return
    # in-process staleness check (no subprocess): the rendered content must match the file
    if gen.render(codes) != page.read_text(encoding="utf-8"):
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
