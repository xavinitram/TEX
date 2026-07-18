"""
LANG-4 — generate `tex_help.json` from the stdlib registry (the single source).

Since v0.23 the function help data (signature, description, example, category, tags)
lives in the REG-1 registry, migrated out of the hand-kept JS `TEX_HELP_DATA`. This
emits a machine-readable help JSON from it, for the CLI (`tex help <fn>`), any future
editor/LSP, and offline docs. The editor's `TEX_HELP_DATA` function entries are now a
drift-pinned MIRROR of the registry (test_doc4_reference); the grammar/concept entries
(Wire Bindings, types, operators, …) remain hand-kept in the JS — they are not stdlib
functions and have no registry entry.

Run:  python tools/gen_help_data.py
"""
import json
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))  # custom_nodes on path

from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib  # noqa: E402,F401  (populates REGISTRY)
from TEX_Wrangle.tex_runtime import stdlib_registry as R  # noqa: E402

_OUT = os.path.join(ROOT, "tex_help.json")


def build():
    """Registry → {version, categories: [{name, entries:[{name, sig, desc, example,
    tags, aliases}]}]}, grouped by category in first-appearance (source) order."""
    cats = OrderedDict()
    for e in R.help_entries(decode=True):
        cats.setdefault(e["category"] or "Other", []).append({
            "name": e["name"], "sig": e["sig"], "desc": e["desc"],
            "example": e["example"], "tags": e["tags"], "aliases": e["aliases"],
        })
    return {
        "categories": [{"name": c, "entries": items} for c, items in cats.items()],
        "function_count": sum(len(v) for v in cats.values()),
    }


def render(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def main():
    text = render(build())
    check = "--check" in sys.argv
    if check:
        try:
            existing = open(_OUT, encoding="utf-8").read()
        except FileNotFoundError:
            print("tex_help.json missing — run tools/gen_help_data.py")
            return 1
        if existing.replace("\r\n", "\n") != text.replace("\r\n", "\n"):
            print("tex_help.json is stale — regenerate with tools/gen_help_data.py")
            return 1
        print("tex_help.json up to date")
        return 0
    with open(_OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {_OUT} ({build()['function_count']} functions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
