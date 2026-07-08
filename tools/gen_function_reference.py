"""
DOC-4 — generate `Function-Reference.md` from the single sources of truth.

The reference is a *view*: function names/tags come from the REG-1 registry, arg
counts + return rules from `FUNCTION_SIGNATURES`, and prose (sig string / desc /
example) from the editor's `TEX_HELP_DATA`. Regenerating it (and the drift test in
tests/test_v017_phase2.py) keeps the human reference from drifting off the code —
the "docs one click away, always current" contract. Run:  python tools/gen_function_reference.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))  # custom_nodes on path

from TEX_Wrangle.tex_runtime import stdlib_registry as R  # noqa: E402
from TEX_Wrangle.tex_compiler.stdlib_signatures import FUNCTION_SIGNATURES  # noqa: E402

_FIELD = lambda k, line: (re.search(rf'\b{k}:\s*"((?:[^"\\]|\\.)*)"', line) or [None, None])[1]


def parse_help():
    """Return {name: {sig, desc, example}} parsed from TEX_HELP_DATA, plus the
    ordered list of (category, [names]) for grouping. One entry per line."""
    js = open(os.path.join(ROOT, "js", "tex_extension.js"), encoding="utf-8").read()
    block = js[js.index("const TEX_HELP_DATA"):js.index("\n];", js.index("const TEX_HELP_DATA")) + 3]
    entries, cats, cur = {}, [], None
    for line in block.splitlines():
        cm = re.search(r'\btitle:\s*"((?:[^"\\]|\\.)*)"', line)
        if cm:
            cur = cm.group(1)
            cats.append((cur, []))
            continue
        name = _FIELD("name", line)
        if name is not None and _FIELD("sig", line) is not None:
            entries[name] = {"sig": _FIELD("sig", line), "desc": _FIELD("desc", line) or "",
                             "example": _FIELD("example", line) or ""}
            if cats:
                cats[-1][1].append(name)
    return entries, cats


def _unescape(s):
    s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)
    s = s.replace("\\n", " ").replace('\\"', '"').replace("\\\\", "\\")
    return s.replace("|", r"\|")  # keep markdown table cells intact


def generate():
    entries, cats = parse_help()
    reg_names = {n for e in R.REGISTRY for n in e.names}
    tag = {n: e for e in R.REGISTRY for n in e.names}
    out = ["# TEX Function Reference",
           "",
           "> **Generated** by `tools/gen_function_reference.py` from `TEX_HELP_DATA` +",
           "> the REG-1 registry + `FUNCTION_SIGNATURES`. Do not edit by hand — edit the",
           "> source and regenerate. The drift test (`test_doc4_reference`) keeps it current.",
           ""]
    documented = set()
    for cat, catnames in cats:
        fn_rows = [n for n in catnames if n in reg_names]
        if not fn_rows:
            continue
        out.append(f"## {cat}")
        out.append("")
        out.append("| Function | Signature | Description | Tags |")
        out.append("|----------|-----------|-------------|------|")
        for n in fn_rows:
            documented.add(n)
            e = entries[n]
            tags = [t for t, on in (("spatial", tag[n].spatial), ("sync", tag[n].sync),
                                    ("non-local", tag[n].non_local)) if on]
            out.append(f"| `{n}` | `{_unescape(e['sig'])}` | {_unescape(e['desc'])} | "
                       f"{', '.join(tags) or '—'} |")
        out.append("")
    # functions with no help entry (drift) — list explicitly so the gap is visible
    missing = sorted(reg_names - documented)
    if missing:
        out.append("## Undocumented (no `TEX_HELP_DATA` entry)")
        out.append("")
        out.append("These registered functions lack an editor help entry — a drift the "
                   "DOC-4 test flags:")
        out.append("")
        for n in missing:
            out.append(f"- `{n}`")
        out.append("")
    return "\n".join(out).rstrip() + "\n", documented, reg_names


if __name__ == "__main__":
    md, documented, reg_names = generate()
    with open(os.path.join(ROOT, "Function-Reference.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote Function-Reference.md: {len(documented)}/{len(reg_names)} functions documented")
    miss = sorted(reg_names - documented)
    if miss:
        print("undocumented:", miss)
