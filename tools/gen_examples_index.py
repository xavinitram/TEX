"""
DOC-5 — generate `examples/INDEX.md` from the example headers + a soft coverage report.

Each `examples/*.tex` opens with a uniform `// Name — description` header; this
emits a linked index from those headers and a *soft* report of which registered
stdlib functions no example exercises ("examples are the curriculum"). The coverage
figure is informational — a gap is a nudge to add an example, not a build break.
Run:  python tools/gen_examples_index.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))

from TEX_Wrangle.tex_runtime import stdlib_registry as R  # noqa: E402

_HEADER = re.compile(r"^//\s*(.+?)\s*—\s*(.+?)\s*$")  # em-dash separator (enforced)


def parse_examples():
    """[(filename, name, desc)] sorted by filename; raises on a bad header."""
    ex_dir = os.path.join(ROOT, "examples")
    out = []
    for fn in sorted(os.listdir(ex_dir)):
        if not fn.endswith(".tex"):
            continue
        with open(os.path.join(ex_dir, fn), encoding="utf-8") as f:
            first = f.readline().rstrip("\n")
        m = _HEADER.match(first)
        if not m:
            raise ValueError(f"{fn}: first line is not `// Name — desc`: {first!r}")
        out.append((fn, m.group(1), m.group(2)))
    return out


def coverage():
    """(covered, uncovered) registry function-name sets vs all example bodies."""
    ex_dir = os.path.join(ROOT, "examples")
    blob = ""
    for fn in os.listdir(ex_dir):
        if fn.endswith(".tex"):
            blob += open(os.path.join(ex_dir, fn), encoding="utf-8").read() + "\n"
    reg = {n for e in R.REGISTRY for n in e.names}
    covered = {n for n in reg if re.search(rf"\b{re.escape(n)}\(", blob)}
    return covered, reg - covered


def generate():
    rows = parse_examples()
    covered, uncovered = coverage()
    total = len(covered) + len(uncovered)
    out = ["# TEX Examples Index",
           "",
           f"> **Generated** by `tools/gen_examples_index.py` from the `// Name — desc`",
           f"> header of each `examples/*.tex`. {len(rows)} examples. Do not edit by hand.",
           "",
           "| Example | Description |",
           "|---------|-------------|"]
    for fn, name, desc in rows:
        out.append(f"| [{name}](examples/{fn}) | {desc} |")
    out += ["",
            "## Function coverage (soft)",
            "",
            f"{len(covered)}/{total} registered stdlib functions are exercised by an "
            f"example. The {len(uncovered)} below are not — a nudge to add one, not a "
            f"build break:",
            ""]
    if uncovered:
        out.append("`" + "`, `".join(sorted(uncovered)) + "`")
    else:
        out.append("*(all functions have an example.)*")
    return "\n".join(out).rstrip() + "\n", rows, covered, uncovered


if __name__ == "__main__":
    md, rows, covered, uncovered = generate()
    with open(os.path.join(ROOT, "examples", "INDEX.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote examples/INDEX.md: {len(rows)} examples; "
          f"{len(covered)}/{len(covered) + len(uncovered)} functions exercised")
    if uncovered:
        print(f"uncovered ({len(uncovered)}):", ", ".join(sorted(uncovered)))
