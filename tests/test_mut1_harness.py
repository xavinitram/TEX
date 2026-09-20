"""MUT-1 — the mutation harness's suite list is DERIVED from its rows, and cannot drift.

`tests/mutation_check.py` runs each mutation against a curated subset of the suite, in a
subprocess, once per row. The subset used to be a hand-written `import test_... as A, ...`
line, and it drifted twice: it ended at v0.33 when v0.34 shipped, and it ended at v0.35 when
v0.36 shipped. The second drift meant all three TRK-25 region-dependence rows reported
`*** SURVIVED *** (0 failing rows)` for three releases — not because the guards were weak, but
because the file holding every test that could kill them was never imported. A row that cannot
be killed is worse than no row: it reports a guarantee the tree does not have, in the voice of
a measurement.

The fix is the discipline invariant 5 already uses for `tex_memory._NON_LOCAL_FNS`: derive the
downstream set from the single source instead of keeping a parallel literal. Each mutation row
carries the suite(s) that kill it, and the runner's import list is the union over the rows being
swept. These rows pin that it stays that way:

  * the harness is importable WITHOUT running the sweep (a 20-minute import is why the harness
    itself was never under test — this row is the one that reds first on the pre-fix tree, by
    timing out);
  * every row names at least one killing suite, and every named suite exists on disk;
  * a row that names a missing suite, or names none at all, is REFUSED with a message naming
    the row — not swept with a verdict printed beside it;
  * the runner template contains no literal `import test_...`, so the list cannot be
    hand-written back in;
  * every row's declared suite is in the list the runner actually imports;
  * every row's ANCHOR still matches the file it names exactly once, so `source.replace(old,
    new)` actually introduces the bug — a row whose anchor has drifted to 0 matches sweeps
    the unmutated tree and prints a verdict about a bug that was never there, which is the
    same lie as a missing import list and went unseen for three releases.

The harness is exercised in a subprocess on purpose: importing it in-process would couple this
file's collection to whatever the harness does at import time, which is the failure being
pinned.
"""
import json
import os
import pathlib
import subprocess
import sys

from helpers import SubTestResult

_HERE = pathlib.Path(__file__).resolve().parent

# One probe, many assertions: the subprocess reports everything the rows below need, so the
# suite pays for one interpreter start rather than six.
_PROBE = r"""
import json, pathlib, sys
sys.path.insert(0, r"{here}")
import mutation_check as mc

_synthetic = [("a synthetic row", "tex_roi.py", "old", "new", ("test_no_numpy_ban",))]
_missing = [("a row naming a suite that is not there", "tex_roi.py", "old", "new",
             ("test_this_module_does_not_exist",))]
_none = [("a row naming no suite at all", "tex_roi.py", "old", "new")]

# Anchor liveness: how many times each row's `old` text occurs in the file it names, at the
# tree the harness would copy. `None` means the file itself is gone. Sources are read once.
_src = {{}}
_anchors = []
for _row in mc.MUTATIONS:
    _rel = _row[1]
    if _rel not in _src:
        _p = pathlib.Path(mc.SRC) / _rel
        _src[_rel] = _p.read_text(encoding="utf-8") if _p.is_file() else None
    _text = _src[_rel]
    _anchors.append([_row[0], _rel, None if _text is None else _text.count(_row[2])])

print("PROBE" + json.dumps({{
    "anchors": _anchors,
    "rows": [[row[0], row[1], list(row[4])] for row in mc.MUTATIONS],
    "modules": list(mc.suite_modules(mc.MUTATIONS)),
    "synthetic_modules": list(mc.suite_modules(_synthetic)),
    "synthetic_source": mc.runner_source(_synthetic, "TESTS", "PARENT"),
    "template": mc.RUNNER_TEMPLATE,
    "problems": mc.validate_rows(mc.MUTATIONS),
    "missing_suite_problems": mc.validate_rows(_missing),
    "no_suite_problems": mc.validate_rows(_none),
    "suite_files": sorted(p.stem for p in pathlib.Path(mc.TESTS_DIR).glob("test_*.py")),
}}))
"""

# Generous, and still an order of magnitude under the sweep it is guarding against: a harness
# that runs its rows at import time cannot finish this inside any budget a test may hold.
_IMPORT_BUDGET_S = 90

_probe_cache: "dict | None" = None
_probe_error: "str | None" = None


def _probe():
    """Import the harness in a child and bring back the facts. Cached for the file."""
    global _probe_cache, _probe_error
    if _probe_cache is not None or _probe_error is not None:
        return _probe_cache
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        out = subprocess.run([sys.executable, "-c", _PROBE.format(here=str(_HERE))],
                             capture_output=True, text=True, cwd=str(_HERE), env=env,
                             timeout=_IMPORT_BUDGET_S)
    except subprocess.TimeoutExpired:
        _probe_error = (f"importing tests/mutation_check.py did not finish in "
                        f"{_IMPORT_BUDGET_S}s - it is running its sweep at import time, so "
                        f"nothing about it can be tested")
        return None
    if out.returncode != 0:
        _probe_error = f"probe exited {out.returncode}: {out.stderr.strip()[-600:]}"
        return None
    for line in out.stdout.splitlines():
        if line.startswith("PROBE"):
            _probe_cache = json.loads(line[len("PROBE"):])
            return _probe_cache
    _probe_error = f"probe printed no PROBE line; stdout tail: {out.stdout.strip()[-400:]}"
    return None


def test_mut1_the_harness_imports_without_running_the_sweep(r: SubTestResult):
    print("\n--- MUT-1: mutation_check.py is importable (the sweep is behind __main__) ---")
    data = _probe()
    if data is None:
        r.fail("MUT-1 harness import", _probe_error)
        return
    r.ok(f"mutation_check imported in a child and reported {len(data['rows'])} rows")


def test_mut1_every_row_names_an_existing_killing_suite(r: SubTestResult):
    print("\n--- MUT-1: every mutation row names a killing suite that exists ---")
    data = _probe()
    if data is None:
        r.fail("MUT-1 row suites", _probe_error)
        return
    if data["problems"]:
        r.fail("MUT-1 row suites",
               f"{len(data['problems'])} row(s) with a broken suite column:\n  "
               + "\n  ".join(data["problems"]))
        return
    files = set(data["suite_files"])
    bad = []
    for label, _path, suites in data["rows"]:
        if not suites:
            bad.append(f"{label}: names no killing suite")
        for name in suites:
            if name not in files:
                bad.append(f"{label}: names {name!r}, which is not a tests/ module")
    if bad:
        r.fail("MUT-1 row suites", "\n  ".join(bad))
    else:
        r.ok(f"all {len(data['rows'])} rows name an existing killing suite")


def test_mut1_a_row_without_a_usable_suite_is_refused_by_name(r: SubTestResult):
    print("\n--- MUT-1: a row whose killing suite is not loadable reds, naming the row ---")
    data = _probe()
    if data is None:
        r.fail("MUT-1 refusal", _probe_error)
        return
    checks = (
        ("a suite that does not exist", data["missing_suite_problems"],
         "a row naming a suite that is not there"),
        ("no suite at all", data["no_suite_problems"], "a row naming no suite at all"),
    )
    for what, problems, label in checks:
        if not problems:
            r.fail("MUT-1 refusal", f"a row with {what} was accepted - the sweep would print "
                                    f"a verdict for a row that asserts nothing")
            return
        if not any(label in p for p in problems):
            r.fail("MUT-1 refusal", f"a row with {what} was refused, but the message does not "
                                    f"name the row: {problems}")
            return
    r.ok("both broken shapes are refused, and the message names the offending row")


def test_mut1_the_runner_import_list_is_derived_not_hand_written(r: SubTestResult):
    print("\n--- MUT-1: the runner's import list is derived from the rows ---")
    data = _probe()
    if data is None:
        r.fail("MUT-1 derivation", _probe_error)
        return
    template = data["template"]
    if "import test_" in template:
        r.fail("MUT-1 derivation",
               "the runner template contains a literal `import test_...` - a hand-kept list "
               "is exactly what drifted at v0.34 and again at v0.36")
        return
    if "FAILMOD" not in template or "FAILCOUNT" not in template:
        r.fail("MUT-1 derivation",
               "the runner must report FAILCOUNT and per-module FAILMOD lines; without the "
               "latter a KILLED verdict cannot be checked against the row's declared suite")
        return
    # The derivation is a function of the rows handed in, and of nothing else: a one-row sweep
    # imports that row's suite and NOT the other seventy rows' suites.
    if data["synthetic_modules"] != ["test_no_numpy_ban"]:
        r.fail("MUT-1 derivation",
               f"a one-row sweep derived {data['synthetic_modules']}, not that row's suite")
        return
    src = data["synthetic_source"]
    leaked = [m for m in data["modules"] if m in src]
    if leaked:
        r.fail("MUT-1 derivation",
               f"a one-row sweep's runner still imports unrelated suites: {leaked}")
        return
    if "test_no_numpy_ban" not in src:
        r.fail("MUT-1 derivation", "the derived runner does not import the row's own suite")
        return
    r.ok("the import list is a pure function of the rows being swept")


def test_mut1_every_rows_suite_is_loaded_by_the_runner(r: SubTestResult):
    print("\n--- MUT-1: the swept runner loads every row's declared killing suite ---")
    data = _probe()
    if data is None:
        r.fail("MUT-1 coverage", _probe_error)
        return
    modules = set(data["modules"])
    missing = [f"{label}: {name}" for label, _p, suites in data["rows"]
               for name in suites if name not in modules]
    if missing:
        r.fail("MUT-1 coverage",
               "row(s) whose killing suite the runner never imports - these rows assert "
               "nothing:\n  " + "\n  ".join(missing))
        return
    r.ok(f"all {len(data['rows'])} rows' suites are in the runner's {len(modules)}-module list")


def test_mut1_every_mutation_anchor_matches_exactly_once(r: SubTestResult):
    """Every row's `old` text is findable, and findable ONCE, in the file it names.

    The sweep applies a row by `source.replace(old, new)`. Zero matches replaces nothing, so
    the row runs the UNMUTATED tree and prints a verdict about a bug that was never
    introduced — a pass that asserted nothing, in the voice of a measurement. Two or more
    matches mutate a site the row never meant, so a KILLED verdict may belong to a different
    bug. Both are silent: the sweep lives outside the standalone runner, so only a hand-run
    over the whole file would see them, and one row had anchored 0x for three releases before
    anybody did. This row makes the anchors a pin the suite holds, in milliseconds and with
    no torch: it is a substring count over the sources the harness already names.
    """
    print("\n--- MUT-1: every mutation anchor still matches its file exactly once ---")
    data = _probe()
    if data is None:
        r.fail("MUT-1 anchor liveness", _probe_error)
        return
    stale = [f"{label} ({rel}): "
             + ("the file is gone" if n is None else f"{n} matches, expected exactly 1")
             for label, rel, n in data["anchors"] if n != 1]
    if stale:
        r.fail("MUT-1 anchor liveness",
               f"{len(stale)} of {len(data['anchors'])} row(s) cannot be applied, so the "
               f"sweep reports a guarantee it never tested:\n  " + "\n  ".join(stale))
        return
    r.ok(f"all {len(data['anchors'])} mutation anchors match their file exactly once")
