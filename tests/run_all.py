#!/usr/bin/env python
"""Standalone TEX test runner — `python tests/run_all.py` (or `cd tests && python run_all.py`).

The wiring is DERIVED, not typed. This file used to be ~2 100 lines of hand-written
`from test_x import test_y` / `test_y(r)` pairs, one per row, appended to by hand every
time a test landed. Three costs came with that, all of them paid repeatedly:

* it was the one file every concurrent branch conflicted on, because every branch
  appended its block at the same place;
* a row that was written but never appended silently never ran (TST-7 exists because
  that happened), and at least one branch shipped a missed registration; and
* the list said nothing the tree did not already say — `tests/test_*.py` plus the
  runner calling convention IS the list.

So the list is now computed. Every top-level `def test_*(r, ...)` in `tests/test_*.py`
is a row (the convention `tests/conftest.py` also keys its `r` fixture on), and the rows
run in a deterministic order: `_ORDER_FIRST` modules first, in the order written there,
then every remaining module by file name, and within a file by definition line.

TST-7 (`tests/test_v017_phase1.py::test_tst7_runner_coverage`) is unchanged in purpose
and now guards the derivation itself: it censuses the tree independently and asserts that
`discover()` found every row, that it invented none, and that `_EXCLUDE` is not stale.

Adding a test therefore needs NO edit here. If a row must be skipped, it goes in
`_EXCLUDE` with a reason — an empty `_EXCLUDE` is the healthy state.
"""
# CACHE-0: point the disk cache at a scratch dir BEFORE any TEX import — get_cache()
# resolves the location once, on first call — so a test run never writes compiled
# artifacts into the shipping package's .tex_cache. setdefault: an outer harness that
# already chose a dir wins.
import ast as _ast
import glob as _glob
import importlib as _importlib
import os as _os
import sys as _sys
import tempfile as _tempfile

_os.environ.setdefault(
    "TEX_CACHE_DIR", _os.path.join(_tempfile.gettempdir(), "tex_test_cache"))

# The runner is invoked both as `python tests/run_all.py` (from the package root) and as
# `cd tests && python run_all.py` (AGENTS.md §"Test"), and is imported by TST-7. Resolve
# the test directory from this file, never from the working directory.
_TESTS_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _TESTS_DIR not in _sys.path:
    _sys.path.insert(0, _TESTS_DIR)

from helpers import SubTestResult


# Modules whose rows must run EARLY, in this order. Keep it short: a module belongs here
# only for a reason written beside it, and everything else is ordered by file name.
_ORDER_FIRST = (
    # The front end, cheapest first: every later row compiles a program through it, so a
    # broken lexer/parser/type-checker/interpreter should red in the first seconds of a
    # run rather than after the suite's slow half.
    "test_lexer",
    "test_parser",
    "test_type_checker",
    "test_interpreter",
    # A1-4: test_release_gate.py::test_scatter_determinism_band reads the value the PR-LP5
    # pin recorded (test_determinism_pin.LAST_CUDA_DET_VAR, module-level, None until the
    # pin runs), so the pin must run first. File-name order already puts
    # test_determinism_pin before test_release_gate; this row makes the dependency
    # explicit so a rename cannot break it silently.
    "test_determinism_pin",
)

# "module::function" rows that discovery finds and the runner must NOT call, each with a
# one-line reason. A row here is a claim that the name matches the runner convention by
# accident (a helper, a factory) — not a way to park a failing test. TST-7 reds on a
# stale entry, so an exclusion cannot outlive the name it names. Empty is correct.
_EXCLUDE: frozenset[str] = frozenset()


def _rtest_defs(path):
    """`(name, lineno)` for every TOP-LEVEL `def test_*(r, ...)` in one test module.

    Top-level only: a nested definition is not reachable as a module attribute, so the
    runner could not call it. TST-7 censuses at any depth and reds on the difference,
    which is how a nested (therefore dead) row is reported rather than skipped."""
    with open(path, encoding="utf-8") as f:
        tree = _ast.parse(f.read(), path)
    return [
        (node.name, node.lineno)
        for node in tree.body
        if (isinstance(node, _ast.FunctionDef) and node.name.startswith("test_")
            and node.args.args and node.args.args[0].arg == "r")
    ]


def discover(tests_dir: str = None):
    """Every runner row, in run order: a list of `(module_name, func_name, lineno)`.

    Order: `_ORDER_FIRST` modules first in the order written there, then the rest by file
    name; within a module, by definition line. Pure AST — nothing is imported here, so
    TST-7 can call this without paying 110 module imports."""
    tests_dir = tests_dir or _TESTS_DIR
    by_module = {}
    for path in _glob.glob(_os.path.join(tests_dir, "test_*.py")):
        module = _os.path.basename(path)[:-3]
        rows = sorted(_rtest_defs(path), key=lambda nl: (nl[1], nl[0]))
        if rows:
            by_module[module] = rows

    ordered = [m for m in _ORDER_FIRST if m in by_module]
    ordered += sorted(m for m in by_module if m not in _ORDER_FIRST)

    out = []
    for module in ordered:
        for name, lineno in by_module[module]:
            if f"{module}::{name}" in _EXCLUDE:
                continue
            out.append((module, name, lineno))
    return out


def main():
    # B7: a redirected stdout on Windows defaults to cp1252, and the ROI-4 banner in
    # test_v024_phase1 contains U+2261 — so `run_all.py > log.txt` died with a
    # UnicodeEncodeError partway through a green suite. Force UTF-8 on our own streams
    # rather than asking every caller to remember PYTHONIOENCODING (and rather than
    # removing the glyph, which would only move the trap to the next one someone types).
    for _s in (_sys.stdout, _sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 60)
    print("TEX Test Suite")
    print("=" * 60)

    r = SubTestResult()

    # A module is imported when its first row is reached, not up front: the old file
    # imported half the tree before the first row ran, which turned one unimportable
    # module into a run that produced no rows and no summary at all. An exception still
    # propagates — a test that raises instead of calling r.fail() is a defect in that
    # test, and hiding it behind a synthetic FAIL row would lose the traceback.
    loaded = {}
    for module_name, func_name, _lineno in discover():
        mod = loaded.get(module_name)
        if mod is None:
            mod = loaded[module_name] = _importlib.import_module(module_name)
        getattr(mod, func_name)(r)

    success = r.summary()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
