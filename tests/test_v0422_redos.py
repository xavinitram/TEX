"""
v0422-redos — CodeQL flagged `tools/gate.py`'s `_SUMMARY_RE` as an inefficient regular
expression: its second alternative used to be `(?:\\d+ \\w+,? ?)+ in [\\d.]+s`. `\\d+` and
`\\w+` both accept digits, and the trailing `,? ?` was optional on both sides, so on a tail
that never reaches `" in <secs>s"` the engine could re-split the same run of digits between
`\\d+` and `\\w+` in exponentially many ways before giving up — e.g. `"0 " + "000 " * 5000 +
"x"`. The fix (in `tools/gate.py`) restricts the count-word alternative to `[a-z]+`: every
real pytest count word (`passed`, `failed`, `errors`, `warnings`, `deselected`, `skipped`,
`xfailed`, `xpassed`) is purely alphabetic, so it shares no characters with `\\d+`, and the
repeated clauses are joined by a literal `", "` — the one separator pytest actually uses —
instead of an optional one. There is now exactly one way to parse a match, so there is
nothing left to backtrack over.

PORTABILITY: pure stdlib (`importlib.util`, `re`, `subprocess`, `json`). The ReDoS timing row
runs the regex match in a CHILD process with a hard `subprocess.run(timeout=...)` ceiling, so
a regression that reintroduces catastrophic backtracking fails this test by timing out
instead of hanging the whole suite.
"""
import importlib.util
import json
import pathlib
import subprocess
import sys

from helpers import SubTestResult

_PKG = pathlib.Path(__file__).resolve().parent.parent
_GATE_PY = _PKG / "tools" / "gate.py"

# Hard wall-clock bound for a single regex match, not a ratio against some other run: the
# real matches above take microseconds and the pathological one takes about the same once
# fixed, so anything under a second is generous rather than tight, and a genuine relapse
# into catastrophic backtracking overshoots it by orders of magnitude, not by a hair.
_HARD_BOUND_S = 1.0
# Ceiling on the whole child process, so a regression that reintroduces the exponential
# blowup fails this file by timing out in bounded time rather than hanging the suite.
_SUBPROCESS_TIMEOUT_S = 15


def _gate():
    """Load `tools/gate.py` by path, once per process (mirrors `test_simp1_gate.py`)."""
    mod = sys.modules.get("_v0422_gate")
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location("_v0422_gate", str(_GATE_PY))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_v0422_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


# Real pytest summary-line TAILS `_summary_of` must still pick the right line out of --
# collected from `tools/gate.py`'s own docstring, CLAUDE.md's "Last reading" table and real
# runs on this box (v0.35.0/v0.35.3/v0.36.1 readings) -- each paired with the exact line
# `_summary_of` must return, UNCHANGED by this fix.
_REAL_STDOUT_TAILS = [
    ("mixed failures + errors, no parens",
     "============================= test session starts ==============================\n"
     "collected 626 items\n...\n"
     "3 failed, 602 passed, 1 deselected, 58 warnings, 10 errors in 273.17s",
     "3 failed, 602 passed, 1 deselected, 58 warnings, 10 errors in 273.17s"),
    ("all passed, with a (h:mm:ss) tail after the seconds",
     "...\n1405 passed, 1 deselected, 40 warnings in 248.05s (0:04:08)",
     "1405 passed, 1 deselected, 40 warnings in 248.05s (0:04:08)"),
    ("banner-wrapped single clause",
     "...\n==== 31 passed in 13.13s ====",
     "31 passed in 13.13s"),
    ("errors present, no passed/failed clause at all",
     "...\n626 passed, 58 warnings, 4 errors in 232.75s",
     "626 passed, 58 warnings, 4 errors in 232.75s"),
    ("bare count, no trailing ' in Xs'",
     "...\n723 passed",
     "723 passed"),
    ("single error, singular word",
     "...\n1 error in 0.01s",
     "1 error in 0.01s"),
]


def test_v0422_gate_summary_real_shapes_unchanged(r: SubTestResult):
    print("\n--- v0422-redos: the rewritten _SUMMARY_RE still recognizes every real pytest "
          "summary shape ---")
    g = _gate()
    for label, stdout, expect in _REAL_STDOUT_TAILS:
        got = g._summary_of(stdout)
        if got == expect:
            r.ok(f"{label}: {got!r}")
        else:
            r.fail("v0422 gate summary shape", f"{label}: got {got!r}, expected {expect!r}")


def test_v0422_gate_summary_does_not_widen_to_prose(r: SubTestResult):
    """"no tests ran in 0.01s" starts with a word, not a digit; neither alternative, before or
    after this rewrite, is meant to match it. Pinned so the rewrite is not read as license to
    widen the pattern into swallowing prose lines that merely contain " in <n>s"."""
    print("\n--- v0422-redos: the rewrite did not widen _SUMMARY_RE to prose lines ---")
    g = _gate()
    got = g._summary_of("collecting ...\nno tests ran in 0.01s")
    if got == "(no pytest summary line)":
        r.ok("a prose line with no leading digit is still not treated as a summary")
    else:
        r.fail("v0422 gate summary false positive", f"got {got!r}")


# ---- ReDoS guard: run in a child process with a hard timeout, so a regression fails this
# ---- test by timing out in bounded time rather than hanging the suite. ----

_GATE_PROBE = r"""
import importlib.util, json, time

spec = importlib.util.spec_from_file_location("_v0422_gate_child", r"{gate_py}")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

bad = "0 " + "000 " * 5000 + "x"
t0 = time.perf_counter()
gate._SUMMARY_RE.match(bad)
elapsed = time.perf_counter() - t0

print("PROBE" + json.dumps({{"elapsed": elapsed}}))
"""


def test_v0422_gate_summary_redos_guard(r: SubTestResult):
    print("\n--- v0422-redos: _SUMMARY_RE on the CodeQL-flagged pathological input, hard-"
          "bounded in a subprocess ---")
    script = _GATE_PROBE.format(gate_py=str(_GATE_PY))
    try:
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                              timeout=_SUBPROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        r.fail("v0422 gate ReDoS guard",
               f"the child did not finish in {_SUBPROCESS_TIMEOUT_S}s -- _SUMMARY_RE is "
               f"catastrophically backtracking again")
        return
    if out.returncode != 0:
        r.fail("v0422 gate ReDoS guard", f"child exited {out.returncode}: "
                                          f"{out.stderr.strip()[-800:]}")
        return
    payload = None
    for line in out.stdout.splitlines():
        if line.startswith("PROBE"):
            payload = json.loads(line[len("PROBE"):])
            break
    if payload is None:
        r.fail("v0422 gate ReDoS guard",
               f"child printed no PROBE line; stdout tail: {out.stdout.strip()[-400:]}")
        return
    elapsed = payload["elapsed"]
    if elapsed < _HARD_BOUND_S:
        r.ok(f"\"0 \" + \"000 \" * 5000 + \"x\" matched in {elapsed:.6f}s "
             f"(bound {_HARD_BOUND_S}s)")
    else:
        r.fail("v0422 gate ReDoS guard",
               f"took {elapsed:.3f}s, over the {_HARD_BOUND_S}s hard bound")
