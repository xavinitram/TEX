"""SIMP-5 — the docs' `file.py:NNN` pointers are checked by a machine, not by a reader.

A line number in a document is the one claim no test could catch: the line moves, the
sentence does not, and the next reader plans against a pointer that now lands somewhere
else. The audit that opened this lane measured the cost — four of four spot-checked
pointers in a document written *that same round* were already wrong, and three pointers in
shipped docs landed on a blank line or past the end of a file.

`tools/check_citations.py` is the check; this file is what puts it in the gate. Two rows:

  1. `test_simp5_shipped_doc_citations` — the real tree. Every citation resolves to a line
     that exists and is not blank (an ERROR, always), and the count of citations whose
     enclosing symbol is not named in the citing sentence equals the pinned budget (a
     RATCHET that only moves down — the PUB-1 discipline applied to docs).

  2. `test_simp5_the_checker_is_not_inert` — the witness. Row 1 asserts a clean tree, so on
     its own it would pass just as happily against a checker that found nothing. This row
     builds a throwaway tree whose citations are wrong in each of the four ways that matter
     and proves the checker reds on every one, and passes the one citation that is right.

The tool lives in `tools/`, which `.comfyignore` excludes from the registry archive, so
nothing here moves PUB-1's shipped surface (`tests/test_pub1_archive.py` is the proof).
Stdlib only: no torch, no ComfyUI, no CUDA, so it runs identically on the CI lane.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

from helpers import SubTestResult

_PKG = Path(__file__).resolve().parent.parent          # TEX_Wrangle/
_TOOL = _PKG / "tools" / "check_citations.py"


def _load_tool():
    """Import `tools/check_citations.py` by path — `tools/` is not a package."""
    spec = importlib.util.spec_from_file_location("_simp5_check_citations", _TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_simp5_shipped_doc_citations(r: SubTestResult):
    print("\n--- SIMP-5: `file.py:NNN` citations in the shipped docs ---")
    if not _TOOL.is_file():
        r.fail("SIMP-5 tool present", f"{_TOOL} is missing")
        return
    cc = _load_tool()

    citations, stats = cc.check(str(_PKG))
    docs = cc.document_set(str(_PKG))
    print(f"    {stats['total']} citations across {len(docs)} documents: "
          f"{stats['ok']} anchored, {stats['module-level']} module-level, "
          f"{stats['warning']} unanchored, {stats['error']} dead")

    if stats["total"] == 0:
        # The scan set resolving to nothing would make every assertion below vacuous.
        r.fail("SIMP-5 scan set", f"no citations found at all in {len(docs)} documents — "
                                  f"the scanner or the document set is broken")
        return
    r.ok(f"SIMP-5 scanned {stats['total']} citations in {len(docs)} documents")

    dead = [c for c in citations if c.verdict == "error"]
    if dead:
        r.fail("SIMP-5 dead citations",
               "a citation pointing at a line that does not exist is a claim about code "
               "that is not there — re-point it at the symbol, never delete the claim:\n  "
               + "\n  ".join(f"{c.where()} `{c.text}` — {c.detail}" for c in dead))
    else:
        r.ok("SIMP-5 no dead citations (every cited line exists and is not blank)")

    unanchored = stats["warning"]
    budget = cc.WARNING_BUDGET
    if unanchored > budget:
        offenders = [c for c in citations if c.verdict == "warning"]
        r.fail("SIMP-5 unanchored budget",
               f"{unanchored} citations do not name their enclosing symbol, budget is "
               f"{budget}. Name the symbol in the citing sentence; do not raise the "
               f"budget:\n  "
               + "\n  ".join(f"{c.where()} `{c.text}` — {c.detail}" for c in offenders))
    elif unanchored < budget:
        r.fail("SIMP-5 unanchored budget",
               f"re-pin DOWN: tools/check_citations.py WARNING_BUDGET is {budget}, the tree "
               f"now has {unanchored}. The budget only moves down; move it.")
    else:
        r.ok(f"SIMP-5 unanchored citations {unanchored}/{budget} (ratchet holds)")


def test_simp5_the_checker_is_not_inert(r: SubTestResult):
    """The negative control: a checker that reported nothing would pass the row above."""
    print("\n--- SIMP-5: the citation checker's witness ---")
    if not _TOOL.is_file():
        r.fail("SIMP-5 tool present", f"{_TOOL} is missing")
        return
    cc = _load_tool()

    target = (
        "import os\n"                     # 1
        "\n"                              # 2
        "\n"                              # 3
        "def widget_count(n):\n"          # 4
        "    total = n + 1\n"             # 5
        "    return total\n"              # 6
        "\n"                              # 7
        "\n"                              # 8
        "def other_helper():\n"           # 9
        "    return 0\n"                  # 10
    )
    doc = (
        "# probe\n"
        "\n"
        "`widget_count` adds one (`sample_mod.py:5`).\n"                 # anchored -> ok
        "\n"
        "The running total is incremented here (`sample_mod.py:5`).\n"   # unanchored -> warning
        "\n"
        "This is where it happens (`sample_mod.py:400`).\n"              # past EOF -> error
        "\n"
        "And here (`sample_mod.py:7`).\n"                                # blank line -> error
        "\n"
        "And in a range (`sample_mod.py:9-400`).\n"                      # range past EOF -> error
        "\n"
        "Nothing of the sort (`no_such_module.py:12`).\n"                # missing file -> error
        "\n"
        "Imports live at the top (`sample_mod.py:1`).\n"                 # module scope -> not checked
        "\n"
        "```\n"
        "Traceback: sample_mod.py:999\n"                                 # fenced -> skipped
        "```\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "sample_mod.py").write_text(target, encoding="utf-8")
        (root / "docs" / "probe.md").write_text(doc, encoding="utf-8")

        citations, stats = cc.check(str(root))
        by_text = {}
        for c in citations:
            by_text.setdefault(c.text, []).append(c.verdict)

        expected = {
            "sample_mod.py:400": "error",        # past EOF
            "sample_mod.py:7": "error",          # blank line
            "sample_mod.py:9-400": "error",      # range end past EOF
            "no_such_module.py:12": "error",     # no such file
            "sample_mod.py:1": "module-level",   # nothing to name
        }
        for text, verdict in expected.items():
            got = by_text.get(text)
            if got and all(v == verdict for v in got):
                r.ok(f"SIMP-5 witness: `{text}` -> {verdict}")
            else:
                r.fail(f"SIMP-5 witness `{text}`",
                       f"expected {verdict}, got {got!r}")

        # The same line, cited twice: once naming `widget_count`, once not. Anything that
        # made the symbol rule vacuous (or unconditional) would break this pair.
        pair = by_text.get("sample_mod.py:5", [])
        if sorted(pair) == ["ok", "warning"]:
            r.ok("SIMP-5 witness: the symbol rule separates a named citation from an "
                 "unnamed one on the SAME line")
        else:
            r.fail("SIMP-5 witness symbol rule",
                   f"expected one 'ok' and one 'warning' for sample_mod.py:5, got {pair!r}")

        if "sample_mod.py:999" not in by_text:
            r.ok("SIMP-5 witness: a citation inside a fenced code block is not a claim")
        else:
            r.fail("SIMP-5 witness fence", "a fenced `sample_mod.py:999` was read as a citation")

        if stats["error"] == 4 and stats["warning"] == 1 and stats["ok"] == 1:
            r.ok("SIMP-5 witness: 4 errors / 1 warning / 1 anchored, as constructed")
        else:
            r.fail("SIMP-5 witness counts",
                   f"expected 4 errors, 1 warning, 1 anchored; got {stats!r}")

        # The other direction: repair the probe doc the way the convention says, and the
        # checker must go quiet. A check that cannot go green is as useless as one that
        # cannot go red.
        repaired = (
            "# probe\n"
            "\n"
            "`widget_count` adds one (`sample_mod.py:5`).\n"
            "\n"
            "`widget_count` increments the running total (`sample_mod.py:5`).\n"
            "\n"
            "`other_helper` returns zero (`sample_mod.py:9-10`).\n"
            "\n"
            "Imports live at the top (`sample_mod.py:1`).\n"
        )
        (root / "docs" / "probe.md").write_text(repaired, encoding="utf-8")
        cc._SPAN_CACHE.clear()
        _citations, stats2 = cc.check(str(root))
        if stats2["error"] == 0 and stats2["warning"] == 0 and stats2["ok"] == 3:
            r.ok("SIMP-5 witness: naming the symbol clears every finding")
        else:
            r.fail("SIMP-5 witness repair",
                   f"expected a clean read after the repair; got {stats2!r}")

    # And the CLI wrapper reports the same verdict through its exit code, since that is what
    # a gate actually reads.
    rc = cc.main(["--root", str(_PKG)])
    if rc == 0:
        r.ok("SIMP-5 tools/check_citations.py exits 0 on this tree")
    else:
        r.fail("SIMP-5 tool exit code", f"tools/check_citations.py exited {rc}, expected 0")


if __name__ == "__main__":
    _r = SubTestResult()
    test_simp5_shipped_doc_citations(_r)
    test_simp5_the_checker_is_not_inert(_r)
    print(f"\n{_r.passed} passed, {_r.failed} failed")
    sys.exit(1 if _r.failed else 0)
