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

The sweep this ask also asked for (nested/adjacent quantifiers over overlapping character
classes elsewhere in `tools/`, `tests/`, `benchmarks/`, `tex_*`, `tex_runtime/`,
`tex_compiler/`) found one more real instance: `tests/test_v031_phase1.py`'s SCHED-4
invariant-#7 canary led its "from" branch with `\\.*[\\w.]*` — `[\\w.]*` alone already matches
any run of dots (a relative-import level) or dotted names, so the leading `\\.*` was a
redundant quantifier overlapping the very next one, the same shape as `_SUMMARY_RE`'s, just
adjacent rather than nested (a polynomial rather than exponential blowup — measured
quadratic, ~0.7s at a 16000-character dot run — but the same defect: two ways to assign the
same characters to the two quantifiers). Dropping `\\.*` there is a no-op for every string it
used to match — provably, since `\\.*` subseteq `[\\w.]*` — so it removes the ambiguity
without changing what the canary does or does not flag (a pre-existing, unrelated defect in
that same pattern — literal backspace bytes bracketing `tex_cookqueue` that make it match no
real import line at all — is reported separately; it is orthogonal to the regex's SHAPE).

Everything else `re.compile(` in the sweep (checked by hand, not re-typed here): single
quantified classes (`\\d+`, `[A-Za-z_][A-Za-z0-9_]*`, `[^{}]*`, `[^']+`, lookaround-anchored
single classes in `tools/check_citations.py` and `tests/test_simp3_no_machine_paths.py`), or
alternations of fixed literals / disjoint character classes (`tests/test_pub1_archive.py`,
`tests/test_simp3_skip_budget.py`'s `_SKIP_VOCAB`, `tests/test_v018_portability.py`,
`tests/test_v019_phase2.py`). None of them repeats a *group* (nested quantifier), and none
puts two quantified pieces back to back that accept overlapping characters, so none can be
driven into the same re-split ambiguity. `tools/gen_examples_index.py`'s
`^//\\s*(.+?)\\s*—\\s*(.+?)\\s*$` chains two lazy `.+?` around fixed anchors on a single
source line — not nested, and each anchor forces a single linear scan — so it is polynomial at
worst on line length, not the exponential shape CodeQL flagged here.

PORTABILITY: pure stdlib (`ast`, `importlib.util`, `re`, `subprocess`, `json`). Both ReDoS
timing rows run their regex match in a CHILD process with a hard `subprocess.run(timeout=...)`
ceiling, so a regression that reintroduces catastrophic backtracking fails the test by timing
out instead of hanging the whole suite.
"""
import ast
import importlib.util
import json
import pathlib
import subprocess
import sys

from helpers import SubTestResult

_PKG = pathlib.Path(__file__).resolve().parent.parent
_GATE_PY = _PKG / "tools" / "gate.py"
_SCHED4_FILE = pathlib.Path(__file__).resolve().parent / "test_v031_phase1.py"
_SCHED4_FN = "test_v031_sched4_off_the_default_path"

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
# collected from `tools/gate.py`'s own docstring, recorded gate readings and real
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


# ──────────────────────────────────────────────────────────────────────────────
# SCHED-4's canary pattern (tests/test_v031_phase1.py) — the adjacent-quantifier instance
# ──────────────────────────────────────────────────────────────────────────────

def _extract_sched4_pattern():
    """The SCHED-4 canary's own `lint_sources(...)` pattern and flags, read from the live
    file with `ast` rather than re-typed here — a hand transcription would have to carry the
    literal backspace bytes the source embeds around `tex_cookqueue` (a separate,
    pre-existing defect, reported out of band; orthogonal to whether the regex's SHAPE is
    ReDoS-safe), and reading it live means this row tracks the real pattern instead of a copy
    that can silently drift from it."""
    source = _SCHED4_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_SCHED4_FILE))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _SCHED4_FN:
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "lint_sources":
                    pattern = call.args[0].value
                    flags = 0
                    for kw in call.keywords:
                        if kw.arg == "flags" and isinstance(kw.value, ast.Attribute) \
                                and kw.value.attr == "MULTILINE":
                            import re as _re
                            flags = _re.MULTILINE
                    return pattern, flags
    raise AssertionError(f"{_SCHED4_FN} no longer calls lint_sources(...) in {_SCHED4_FILE}")


#: The pattern's SHAPE before this ask's fix, spelled out with its real bytes (including the
#: pre-existing backspace bytes around `tex_cookqueue`) so the row below can PROVE the fix
#: changed nothing about what the pattern matches — only removing the leading `\.*` that
#: overlapped the `[\w.]*` right after it.
_SCHED4_PATTERN_BEFORE = (
    "^[ \t]*(?:from[ \t]+\\.*[\\w.]*\x08tex_cookqueue\x08|"
    "import[ \t]+[\\w.]*\x08tex_cookqueue\x08)"
)

#: A representative corpus, short enough to be safe to match in-process even under the OLD,
#: ambiguous pattern (no pathological long runs here — those are exercised only in the
#: timeout-guarded subprocess row below).
_SCHED4_IMPORT_CORPUS = [
    "from tex_cookqueue import CookQueue",
    "from .tex_cookqueue import CookQueue",
    "from ..tex_cookqueue import CookQueue",
    "from pkg.tex_cookqueue import CookQueue",
    "from .pkg.sub.tex_cookqueue import CookQueue",
    "import tex_cookqueue",
    "import pkg.tex_cookqueue",
    "    from tex_cookqueue import CookQueue",
    "# a comment that only mentions tex_cookqueue",
    "from tex_engine import CookQueue",
    "",
]


def test_v0422_sched4_pattern_matching_is_unchanged(r: SubTestResult):
    print("\n--- v0422-redos: dropping the redundant \\.* left the SCHED-4 canary's "
          "matching behavior unchanged ---")
    import re
    try:
        pattern_after, flags_after = _extract_sched4_pattern()
    except Exception as e:
        r.fail("v0422 sched4 extraction", str(e))
        return
    before = re.compile(_SCHED4_PATTERN_BEFORE, re.MULTILINE)
    after = re.compile(pattern_after, flags_after)

    mismatches = []
    for line in _SCHED4_IMPORT_CORPUS:
        b, a = bool(before.search(line)), bool(after.search(line))
        if b != a:
            mismatches.append((line, b, a))
    if mismatches:
        r.fail("v0422 sched4 matching changed",
               f"{len(mismatches)} corpus line(s) differ (line, before, after): {mismatches}")
    else:
        r.ok(f"identical matches on {len(_SCHED4_IMPORT_CORPUS)} corpus lines "
             f"(before={[bool(before.search(l)) for l in _SCHED4_IMPORT_CORPUS]})")


_SCHED4_PROBE = r"""
import json, re, time

sched4_pattern = {sched4_pattern!r}
sched4_flags = {sched4_flags!r}
sched4_rx = re.compile(sched4_pattern, sched4_flags)
bad = "from " + "." * 20000

t0 = time.perf_counter()
sched4_rx.match(bad)
elapsed = time.perf_counter() - t0

print("PROBE" + json.dumps({{"elapsed": elapsed}}))
"""


def test_v0422_sched4_pattern_redos_guard(r: SubTestResult):
    print("\n--- v0422-redos: the SCHED-4 canary pattern on a long dot run, hard-bounded in "
          "a subprocess ---")
    try:
        pattern_after, flags_after = _extract_sched4_pattern()
    except Exception as e:
        r.fail("v0422 sched4 ReDoS guard", str(e))
        return
    script = _SCHED4_PROBE.format(sched4_pattern=pattern_after, sched4_flags=flags_after)
    try:
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                              timeout=_SUBPROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        r.fail("v0422 sched4 ReDoS guard",
               f"the child did not finish in {_SUBPROCESS_TIMEOUT_S}s -- the SCHED-4 pattern "
               f"is backtracking catastrophically again")
        return
    if out.returncode != 0:
        r.fail("v0422 sched4 ReDoS guard", f"child exited {out.returncode}: "
                                            f"{out.stderr.strip()[-800:]}")
        return
    payload = None
    for line in out.stdout.splitlines():
        if line.startswith("PROBE"):
            payload = json.loads(line[len("PROBE"):])
            break
    if payload is None:
        r.fail("v0422 sched4 ReDoS guard",
               f"child printed no PROBE line; stdout tail: {out.stdout.strip()[-400:]}")
        return
    elapsed = payload["elapsed"]
    if elapsed < _HARD_BOUND_S:
        r.ok(f"\"from \" + \".\" * 20000 matched in {elapsed:.6f}s (bound {_HARD_BOUND_S}s)")
    else:
        r.fail("v0422 sched4 ReDoS guard",
               f"took {elapsed:.3f}s, over the {_HARD_BOUND_S}s hard bound")
