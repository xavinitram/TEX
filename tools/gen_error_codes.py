#!/usr/bin/env python
"""C3-ux — generate wiki/Error-Codes.md from the error codes used across the source.

Every rendered diagnostic links to `wiki/Error-Codes#e<NNNN>` (diagnostics.wiki_url_for_code).
Before v0.19 that page did not exist, so every error linked to a 404. This harvests every
`code="ENNNN"` used in the compiler/runtime, groups by family, and emits one anchored
`### ENNNN` subsection per code so the link always resolves. Messages are runtime f-strings
(not centrally tabled), so per-code prose is a short curated line by family; the value is a
complete, never-404 reference. Run: python tools/gen_error_codes.py [--check].

The companion drift test (test_v019_docs) asserts every source code has a heading here.
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_OUT = os.path.join(_PKG, "wiki", "Error-Codes.md")

# Family descriptions (the stable contract; the phase prefix of each code).
_FAMILIES = [
    ("E0", "Internal", "Internal errors — a compiler phase raised without a structured "
     "diagnostic, so TEX synthesized a fallback (E0000) rather than surface a bare message. "
     "Rare and not your fault; please file an issue with the program that triggered it."),
    ("E1", "Lexer", "Tokenization errors — a character or literal the scanner can't read "
     "(bad number, unterminated string/comment, stray symbol)."),
    ("E2", "Parser", "Grammar errors — a statement or expression that doesn't parse "
     "(missing `;`/`)`/`}`, a misplaced token, an unexpected keyword)."),
    ("E3", "Type checker", "Type errors — an operation on the wrong type "
     "(vec/scalar/matrix/string/array mismatch, bad swizzle, wrong argument type or arity, "
     "assigning to a built-in, indexing a non-array)."),
    ("E4", "Optimizer", "Errors surfaced while optimizing the checked AST "
     "(rare; usually indicates an internal invariant — please file an issue with the program)."),
    ("E5", "Compile / cache", "Errors compiling or loading a cached program "
     "(codegen fell back, a cache artifact was rejected)."),
    ("E6", "Runtime / node", "Errors while executing or wiring the node "
     "(an input `@X` isn't connected, a runtime value went non-finite, a fused-chain or "
     "lazy-input problem, an OOM the node re-raised for ComfyUI to handle)."),
    ("E9", "Tools", "Tool errors — building or preflighting a `.textool` bundle failed "
     "(a fused-tool graphspec is malformed, or its stages don't compile together). Check the "
     "tool's stages and manifest; the message names the stage that broke."),
    ("W7", "Warnings", "Non-fatal advisories (LANG-2). The program still compiles and "
     "runs; these flag likely mistakes — an unused variable or wired input, or a name "
     "that shadows a built-in or an outer-scope variable."),
]


def harvest_codes():
    """Return the sorted set of ENNNN/WNNNN codes used across the source (excluding tests)."""
    codes = set()
    pat = re.compile(r"\b[EW]\d{4}\b")
    for root, _dirs, files in os.walk(_PKG):
        if any(s in root for s in ("tests", "benchmarks", "__pycache__", ".tex_cache", "tools", "wiki")):
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(root, fn), encoding="utf-8") as f:
                for m in pat.finditer(f.read()):
                    codes.add(m.group(0))
    return sorted(codes)


def render(codes):
    out = [
        "# Error codes",
        "",
        "Every TEX diagnostic carries a stable `ENNNN` (error) or `WNNNN` (warning) code and "
        "links here. Codes are grouped "
        "by compiler/runtime phase (the first digit). The exact message is written for your "
        "specific program at the point of failure; this page explains the *class* and how to "
        "approach it. **This page is generated** (`tools/gen_error_codes.py`) — do not edit by hand.",
        "",
    ]
    by_family = {fam: [] for fam, _, _ in _FAMILIES}
    for c in codes:
        by_family.setdefault(c[:2], []).append(c)
    for fam, name, desc in _FAMILIES:
        fam_codes = by_family.get(fam, [])
        if not fam_codes:
            continue
        out.append(f"## {name} (`{fam}xxx`)")
        out.append("")
        out.append(desc)
        out.append("")
        for c in fam_codes:
            out.append(f"### {c}")
            out.append("")
            out.append(f"{name}. See the message shown with the code for the specific cause "
                       "and fix; the class is described above.")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


# LANG-7: a package-root copy that ships with the node (like Function-Reference.md), so the
# offline `/tex_wrangle/docs/Error-Codes` route + `TEX_DOCS_LOCAL` serve real content. The
# wiki/ copy (a separate gitignored checkout) is still written when that dir is present.
_ROOT_OUT = os.path.join(_PKG, "Error-Codes.md")


def main():
    check = "--check" in sys.argv
    codes = harvest_codes()
    content = render(codes)
    if check:
        try:
            existing = open(_ROOT_OUT, encoding="utf-8").read()
        except FileNotFoundError:
            print("Error-Codes.md missing — run tools/gen_error_codes.py")
            return 1
        if existing != content:
            print("Error-Codes.md is stale — regenerate with tools/gen_error_codes.py")
            return 1
        print(f"Error-Codes.md up to date ({len(codes)} codes)")
        return 0
    with open(_ROOT_OUT, "w", encoding="utf-8") as f:  # shipped, package-root
        f.write(content)
    outs = [_ROOT_OUT]
    if os.path.isdir(os.path.dirname(_OUT)):           # wiki/ checkout present -> also update it
        with open(_OUT, "w", encoding="utf-8") as f:
            f.write(content)
        outs.append(_OUT)
    print(f"wrote {', '.join(outs)} ({len(codes)} codes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
