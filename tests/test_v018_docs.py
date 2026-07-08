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
