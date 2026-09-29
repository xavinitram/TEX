"""DOCS-49 — an error-code range heading must name the phase that actually raises it.

Bug this catches: `tools/gen_error_codes.py`'s `_FAMILIES` table described `E4xxx` as
"Optimizer" and `E5xxx` as "Compile / cache" — plausible-sounding phase names that name no
module that actually constructs an `E4xxx`/`E5xxx` diagnostic. Both codes are raised only
from `tex_compiler/type_checker.py` (E4000: an unrecognized construct; E5001-E5003: a
function-signature error), matching what `DEVELOPMENT.md`'s own range table already said.
The generator's copy disagreed with the tree it claims to describe, and `Error-Codes.md`
inherited the wrong story because nothing checked the generator against the source it
harvests codes FROM (as opposed to checking the codes it lists are complete — that part
was already covered by `test_v018_docs.py::test_c3ux_error_codes_resolve`).

This does not hand-list "E4 should say type checker" — that would just be a second copy of
the same claim the generator got wrong, and a future re-shuffle of which module owns which
family would silently desync it again. Instead it re-derives, from the product source
itself, which module raises the plurality of a family's codes, maps that module to a
phase keyword via a small physical table (a module doesn't change what it does when a
range's prose changes), and asserts the keyword appears in the family's description in
all three places that state a phase for a code range: `tools/gen_error_codes.py`'s
`_FAMILIES`, `DEVELOPMENT.md`'s "Error Code Ranges" table, and `CONTRIBUTING.md`'s "Error
Codes & Diagnostics" table. That last part is also DOCS-49's check that the three range
tables — after adding `E0`/`E7`/`E9` rows to the two hand-written ones to match the
generator, which already had them — stay in agreement.

A family with no clear plurality winner (a tie between two modules whose keywords differ)
asserts nothing — `W7` is one, split evenly between `tex_api.py` and `tex_marshalling.py` —
so this stays a claim about what the source can actually prove, not a forced opinion.
"""
import collections
import importlib.util
import re
from pathlib import Path

from helpers import SubTestResult

# FIX-ROI49 Q6 (R1#2): the "what counts as product source" exclusion set and its os.walk
# harvest are a single source of truth, not a second hand-kept copy — importing them is
# exactly the drift this file's own docstring says the generator lacked one layer up
# ("nothing checked the generator against the source it harvests codes FROM"); a second,
# separately-maintained copy here would recreate that same bug class one level down.
from test_simp6_error_codes import _product_files

_PKG = Path(__file__).resolve().parent.parent

# A real *construction* site: `code="E1234"`, `code='E1234'`, or a dict literal's
# `"code": "E1234"` — deliberately narrower than "the code's digits appear in this file",
# so a comment or docstring that merely MENTIONS a code (there are several, e.g.
# `precision_policy.py`'s "the type checker resolves callees backward-only, E5001") is not
# mistaken for a site that raises it.
_CODE_SITE = re.compile(r'["\']?code["\']?\s*[:=]\s*["\']([EW]\d{4})["\']')

# module (package-relative, forward-slashed) -> the phase keyword a description must carry
# if that module turns out to be the plurality emitter for some family. This is a physical
# fact about what each module IS (matching DEVELOPMENT.md's own "Compilation Pipeline"
# section and module descriptions), not a claim about which family belongs to which phase.
_MODULE_KEYWORD = {
    "tex_compiler/lexer.py": "lex",
    "tex_compiler/parser.py": "pars",
    "tex_cache.py": "pars",  # re-surfaces a cached parse failure; parser's own phase
    "tex_compiler/type_checker.py": "type check",
    "tex_compiler/optimizer.py": "optimiz",
    "tex_runtime/interpreter.py": "runtime",
    "tex_runtime/interpreter_binding.py": "runtime",
    "tex_runtime/interpreter_control_flow.py": "runtime",
    "tex_runtime/masked_flow.py": "runtime",
    "tex_runtime/compiled.py": "runtime",
    "tex_engine.py": "runtime",
    "tex_marshalling.py": "host",
    "tex_tool.py": "tool",
    "tex_api.py": "internal",
    "tex_compiler/diagnostics.py": "internal",
}


def _plurality_keyword_by_family():
    """family ("E4", "W7", ...) -> keyword, for families with an unambiguous top emitter.

    Ties (including a family raised from only modules absent from `_MODULE_KEYWORD`) are
    left out rather than guessed.
    """
    hits = collections.defaultdict(lambda: collections.Counter())
    for path, rel in _product_files():
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for m in _CODE_SITE.finditer(text):
            code = m.group(1)
            fam = code[:2]
            kw = _MODULE_KEYWORD.get(rel)
            if kw is not None:
                hits[fam][kw] += 1
    out = {}
    for fam, counter in hits.items():
        ranked = counter.most_common()
        if len(ranked) == 1 or (len(ranked) > 1 and ranked[0][1] > ranked[1][1]):
            out[fam] = ranked[0][0]
    return out


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "docs49_gen_error_codes", _PKG / "tools" / "gen_error_codes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# FIX-ROI49 Q6 (R2#5): DEVELOPMENT.md's and CONTRIBUTING.md's range tables differ only in
# WHERE the table sits (the pair of split markers bounding the section); the parse itself —
# split section, regex out `| \`E4xxx\` | description |` rows — was the same function typed
# twice. One parametrized helper, called with each doc's own two literal markers.
def _range_table(text, start_anchor, end_marker):
    """{'E4': 'Type checker — unrecognized construct (catch-all)', ...} from the range table
    between `start_anchor` and `end_marker` (the `Phase`/description column, family prefix
    stripped to 2 chars)."""
    section = text.split(start_anchor, 1)[-1].split(end_marker, 1)[0]
    out = {}
    for m in re.finditer(r'\|\s*`([EW]\d)xxx`\s*\|\s*([^|]+?)\s*\|', section):
        out[m.group(1)] = m.group(2)
    return out


def _development_range_table(text):
    """DEVELOPMENT.md's '### Error Code Ranges' table."""
    return _range_table(text, "### Error Code Ranges", "\n### ")


def _contributing_range_table(text):
    """CONTRIBUTING.md's "Codes are grouped by stage:" table."""
    return _range_table(text, "Codes are grouped by stage:", "\n## ")


def test_docs49_range_heading_matches_emitting_phase(r: SubTestResult):
    print("\n--- DOCS-49: Error-Codes.md / DEVELOPMENT.md / CONTRIBUTING.md range headings "
          "match the phase that actually emits each family's codes ---")

    try:
        gen = _load_generator()
    except Exception as e:
        r.fail("DOCS-49 import generator", f"{type(e).__name__}: {e}")
        return

    plurality = _plurality_keyword_by_family()
    if not plurality:
        r.fail("DOCS-49 derivation", "no family had an unambiguous emitting module — "
                                     "the harvest or the keyword table is broken")
        return

    gen_desc = {fam: name for fam, name, _ in gen._FAMILIES}

    try:
        dev_text = (_PKG / "DEVELOPMENT.md").read_text(encoding="utf-8")
        dev_table = _development_range_table(dev_text)
    except Exception as e:
        r.fail("DOCS-49 read DEVELOPMENT.md", str(e))
        return

    try:
        contrib_text = (_PKG / "CONTRIBUTING.md").read_text(encoding="utf-8")
        contrib_table = _contributing_range_table(contrib_text)
    except Exception as e:
        r.fail("DOCS-49 read CONTRIBUTING.md", str(e))
        return

    checked = 0
    for fam, keyword in sorted(plurality.items()):
        sources = {
            "tools/gen_error_codes.py _FAMILIES": gen_desc.get(fam),
            "DEVELOPMENT.md Error Code Ranges": dev_table.get(fam),
            "CONTRIBUTING.md Error Codes & Diagnostics": contrib_table.get(fam),
        }
        for source_name, desc in sources.items():
            checked += 1
            if desc is None:
                r.fail(f"DOCS-49 {fam} in {source_name}",
                       f"no row/entry for `{fam}xxx` — the emitting module maps to "
                       f"phase keyword {keyword!r}, which is a source of drift here")
                continue
            if keyword.lower() not in desc.lower():
                r.fail(f"DOCS-49 {fam} phase in {source_name}",
                       f"says {desc!r}, but the codes are actually raised (by plurality) "
                       f"from a module whose phase keyword is {keyword!r}")
                continue
    if r.failed == 0:
        r.ok(f"{len(plurality)} families' range headings agree across all three sources "
             f"and match their emitting module ({checked} source-rows checked)")


def test_docs49_three_range_tables_list_the_same_families(r: SubTestResult):
    print("\n--- DOCS-49: DEVELOPMENT.md, CONTRIBUTING.md and the generator's _FAMILIES "
          "list exactly the same set of code-range families ---")
    try:
        gen = _load_generator()
        dev_table = _development_range_table(
            (_PKG / "DEVELOPMENT.md").read_text(encoding="utf-8"))
        contrib_table = _contributing_range_table(
            (_PKG / "CONTRIBUTING.md").read_text(encoding="utf-8"))
    except Exception as e:
        r.fail("DOCS-49 setup", f"{type(e).__name__}: {e}")
        return

    gen_families = {fam for fam, _, _ in gen._FAMILIES}
    dev_families = set(dev_table)
    contrib_families = set(contrib_table)

    if gen_families != dev_families:
        r.fail("DOCS-49 DEVELOPMENT.md vs generator",
               f"generator has {sorted(gen_families)}, DEVELOPMENT.md's table has "
               f"{sorted(dev_families)}")
        return
    if gen_families != contrib_families:
        r.fail("DOCS-49 CONTRIBUTING.md vs generator",
               f"generator has {sorted(gen_families)}, CONTRIBUTING.md's table has "
               f"{sorted(contrib_families)}")
        return
    r.ok(f"all three sources list the same {len(gen_families)} families: "
         f"{sorted(gen_families)}")
