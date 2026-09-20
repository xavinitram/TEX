"""SIMP-6 — every error code the product can emit is a code some test names.

An `E####`/`W####` is the public contract between TEX and whatever draws its diagnostics:
an editor gutter, a host's error panel, a CLI. A code nobody tests is a code the next
change is free to re-spell, re-severity or quietly stop emitting, and the only reader who
notices is the one whose editor stopped underlining. A census of the tree when this landed
found **88** distinct codes and **46** of them named by no test at all.

The ratchet below counts the codes no test names, pins the count, and lets it move DOWN
only. A code that gains a test lowers the pin (the row says so, by name); a NEW code that
arrives without a test pushes the count above the pin and reds, naming it.

The population is every `E####`/`W####` token the product's own source names —
deliberately WIDER than "every site that constructs a diagnostic". A code named only in a
constructor default or in a comment describing a contract is still a code the product
knows and a host may receive, and a wider population cannot be shrunk by moving a code
into a different syntactic position.

No product code is imported here: the file reads the tree as text. It needs no host, no
CUDA and no compiler.
"""
import collections
import os
import re

from helpers import SubTestResult

# ── The tree, as data ─────────────────────────────────────────────────

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_TESTS_DIR)
_THIS_FILE = os.path.basename(__file__)

# Everything that is not product code. A code named in a test, a benchmark or a tool is
# not the product emitting it.
_NOT_PRODUCT = {
    "tests", "benchmarks", "tools", "docs", "examples", "editor_build", "assets",
    "results", "results_test", "stock", "wiki", "js",
    ".git", ".github", ".tex_cache", "__pycache__",
}

_CODE = re.compile(r"\b([EW]\d{4})\b")


def _product_files():
    """(absolute path, slash-separated path relative to the package) for every product
    `.py` file. Slash-separated so a failure message reads the same on either OS."""
    out = []
    for root, dirs, files in os.walk(_PKG_DIR):
        rel = os.path.relpath(root, _PKG_DIR)
        top = "" if rel == "." else rel.split(os.sep)[0]
        if top in _NOT_PRODUCT:
            dirs[:] = []
            continue
        for fn in sorted(files):
            if fn.endswith(".py"):
                p = os.path.join(root, fn)
                out.append((p, os.path.relpath(p, _PKG_DIR).replace(os.sep, "/")))
    return sorted(out, key=lambda t: t[1])


def _codes_named_by_product():
    """code -> sorted list of "file:line" where the product's source names it."""
    seen = collections.defaultdict(list)
    for path, rel in _product_files():
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                for code in sorted(set(_CODE.findall(line))):
                    seen[code].append(f"{rel}:{i}")
    return seen


def _codes_named_by_tests():
    """code -> set of test file names that name it.

    THIS file is excluded from the population on purpose: the pinned backlog below spells
    out the codes it is counting, and a code written down as debt is bookkeeping, not
    coverage — counting it would let the ratchet be moved by writing a comment.
    """
    hits = collections.defaultdict(set)
    for fn in sorted(os.listdir(_TESTS_DIR)):
        if not fn.endswith(".py") or fn == _THIS_FILE:
            continue
        with open(os.path.join(_TESTS_DIR, fn), encoding="utf-8") as fh:
            src = fh.read()
        for code in set(_CODE.findall(src)):
            hits[code].add(fn)
    return hits


# ── The ratchet ────────────────────────────────────────────────────

# The codes that had no test when this pin was last moved — the debt, written down. The
# count is pinned beside it because the count is what the ratchet promises: it may go DOWN
# and never up. Raising either is not a fix; a code arriving without a test is the thing
# the row exists to catch, and the failure names it.
#
# Moving the pin is a two-line edit in ONE direction: drop the code that gained a test from
# the set, and lower the number to match. The row tells you both numbers when it reds.
_UNTESTED_PIN = 46
_UNTESTED_AT_PIN = frozenset("""
    E1000 E1002 E1006 E1008
    E2001 E2003 E2004 E2005 E2006 E2010 E2011 E2020
    E3000 E3002 E3010 E3012 E3013 E3014
    E3100 E3101 E3102 E3103 E3201 E3202 E3204
    E3400 E3401 E3402 E3500 E3600 E3700 E3800 E3900
    E4000
    E6000 E6001 E6002 E6004 E6005 E6006 E6030 E6040 E6050 E6051 E6060
    E9001
""".split())


def test_simp6_untested_error_codes_only_go_down(r: SubTestResult):
    """Pin the codes no test names; a new one reds, and retiring one moves the pin."""
    if len(_UNTESTED_AT_PIN) != _UNTESTED_PIN:
        r.fail("the pin disagrees with itself",
               f"_UNTESTED_PIN is {_UNTESTED_PIN} but _UNTESTED_AT_PIN holds "
               f"{len(_UNTESTED_AT_PIN)} code(s). They are one fact written twice; fix both.")
        return

    product = _codes_named_by_product()
    tested = _codes_named_by_tests()
    untested = sorted(c for c in product if c not in tested)

    fresh = sorted(set(untested) - _UNTESTED_AT_PIN)
    if fresh:
        r.fail(
            "error-code coverage ratchet",
            f"{len(fresh)} code(s) have no test naming them and are not in the pinned "
            f"backlog: {' '.join(fresh)}. "
            f"(All {len(untested)} untested now: {' '.join(untested)}.) "
            f"Add a test that triggers the code through the public surface and asserts "
            f"it. If the code genuinely cannot be triggered, say so where the lane's "
            f"evidence lives; do NOT widen the pin.")
        return

    retired = sorted(_UNTESTED_AT_PIN - set(untested))
    if retired:
        r.fail(
            "error-code coverage ratchet is stale",
            f"{len(retired)} code(s) in the pinned backlog now have a test: "
            f"{' '.join(retired)}. Move the pin: drop them from _UNTESTED_AT_PIN and set "
            f"_UNTESTED_PIN to {len(untested)}. A pin that no longer describes the tree "
            f"stops being a ratchet.")
        return

    r.ok(f"{len(untested)} of {len(product)} error codes have no test naming them, exactly "
         f"the pinned backlog: {' '.join(untested)}")
