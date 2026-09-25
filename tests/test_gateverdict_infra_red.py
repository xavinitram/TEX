"""GATE-VERDICT — `_run` may not report a clean pass for a leg that never proved anything.

`tools/gate.py::_run` (used by `run_cheap`, `run_ci_shape` and `run_canonical`) used to set
`leg.rc` from the subprocess and derive `leg.failures`/`leg.collected` from the junit report
alone (`_parse_junit`). `judge()` reads only `leg.failures`, never `leg.rc` -- so a leg whose
process died before pytest ever wrote a junit report (a `conftest.py` import error, which
exits pytest with a nonzero rc before `pytest.main()` runs) parsed as zero failures and
`judge()` printed `VERDICT GREEN` over a nonzero exit code. Reproduced directly: a checkout
with no `TEX_Wrangle`-named sibling makes `canonical_harness.py`'s
`import TEX_Wrangle.tex_node` raise `ModuleNotFoundError` before pytest starts, and the old
`_run` read that as a clean cheap-tier pass.

`run_counts` already carried the fix for exactly this class of bug (its own
`<counts:infra-rc…>` synthetic id, so no allowlist entry can name it and it always lands in
`judge()`'s `real` list). This file pins the same guard now shared by `_run` through
`_mark_infra_red`, for both ways a leg can prove nothing:

  1. a nonzero rc with an empty junit report (no file at all, or a file with no matching
     failure) -- the conftest-import-error shape;
  2. an rc of 0 that still collected zero tests -- a silent, empty "pass".

And the property that makes the fix safe to ride along everywhere `_run` is called: a normal
leg (rc 0, at least one collected test, no failures) is untouched.

PORTABILITY. Pure stdlib subprocesses (`sys.executable -c "..."`) standing in for pytest, so
this needs no torch, no CUDA, no ComfyUI and no real test collection -- it exercises `_run`'s
own bookkeeping, not any actual suite.
"""
import importlib.util
import io
import sys
import tempfile
from contextlib import redirect_stdout

from helpers import SubTestResult


def _gate():
    """Load `tools/gate.py` by path, once per process (mirrors `tests/test_simp1_gate.py`).

    `tools/` is not a package (`.comfyignore`d, like `tests/` and `benchmarks/`), so there is
    no import name -- loading the file keeps this honest about testing the tool the law tells
    an implementer to run, not a copy of its logic."""
    mod = sys.modules.get("_gateverdict_gate")
    if mod is not None:
        return mod
    import pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "gate.py"
    spec = importlib.util.spec_from_file_location("_gateverdict_gate", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gateverdict_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_gateverdict_run_reds_on_nonzero_rc_with_no_junit(r: SubTestResult):
    """The conftest-import-error shape: rc != 0, nothing parsed from a junit report at all."""
    print("\n--- GATE-VERDICT: _run reds on a nonzero rc that wrote no junit ---")
    g = _gate()
    with tempfile.TemporaryDirectory(prefix="tex-gate-infra-") as scratch:
        leg = g.Leg("fake-noJunit", "a fake leg standing in for a conftest import error")
        argv = [sys.executable, "-c", "import sys; sys.exit(1)"]
        g._run(leg, argv, scratch, {}, scratch, False)

        if leg.rc != 1:
            r.fail("GATE-VERDICT infra-red (no junit)", f"expected rc 1, got {leg.rc}")
        elif leg.failures != [f"<{leg.name}:infra-rc1>"]:
            r.fail("GATE-VERDICT infra-red (no junit)",
                   f"expected the synthetic infra id alone, got {leg.failures!r}")
        else:
            j = g.judge([leg], [], False)
            if j["verdict"] != "RED":
                r.fail("GATE-VERDICT infra-red (no junit)",
                       f"judge() must RED this, got {j['verdict']!r}")
            else:
                r.ok("a nonzero rc with no junit report lands as a real, un-allowlistable red")


def test_gateverdict_run_reds_on_zero_collected_at_rc_zero(r: SubTestResult):
    """The silent-empty-pass shape: rc == 0, but the leg collected nothing at all."""
    print("\n--- GATE-VERDICT: _run reds on rc 0 with zero collected ---")
    g = _gate()
    with tempfile.TemporaryDirectory(prefix="tex-gate-infra-") as scratch:
        leg = g.Leg("fake-zeroCollect", "a fake leg standing in for a leg that collected nothing")
        # Exits 0 and writes no junit report at all -- indistinguishable, from `_run`'s own
        # bookkeeping, from a junit report with zero <testcase> rows (both parse to an empty
        # `collected` set), which is the other half of this same shape.
        argv = [sys.executable, "-c", "pass"]
        g._run(leg, argv, scratch, {}, scratch, False)

        if leg.rc != 0:
            r.fail("GATE-VERDICT infra-red (zero collected)", f"expected rc 0, got {leg.rc}")
        elif leg.failures != [f"<{leg.name}:infra-rc0>"]:
            r.fail("GATE-VERDICT infra-red (zero collected)",
                   f"expected the synthetic infra id alone, got {leg.failures!r}")
        else:
            j = g.judge([leg], [], False)
            if j["verdict"] != "RED":
                r.fail("GATE-VERDICT infra-red (zero collected)",
                       f"judge() must RED this, got {j['verdict']!r}")
            else:
                r.ok("rc 0 with zero collected tests lands as a real, un-allowlistable red")


def test_gateverdict_run_leaves_a_normal_green_leg_untouched(r: SubTestResult):
    """The property that makes the fix safe everywhere `_run` is already called: a leg that
    actually ran and collected something, with no failures, must still read GREEN."""
    print("\n--- GATE-VERDICT: _run does not touch a normal green leg ---")
    g = _gate()
    with tempfile.TemporaryDirectory(prefix="tex-gate-infra-") as scratch:
        leg = g.Leg("fake-green", "a fake leg standing in for an ordinary passing run")
        # Writes a junit report with one passing <testcase> and exits 0 -- a real, if tiny,
        # clean pass. The script reads its own `--junit-xml=<path>` argument back out of
        # sys.argv, exactly as `_run` appends it.
        script = (
            "import sys\n"
            "path = [a.split('=', 1)[1] for a in sys.argv if a.startswith('--junit-xml=')][0]\n"
            "open(path, 'w', encoding='utf-8').write(\n"
            "    '<testsuites><testsuite>'\n"
            "    '<testcase classname=\"tests.fake\" name=\"test_ok\" file=\"tests/fake.py\">'\n"
            "    '</testcase></testsuite></testsuites>'\n"
            ")\n"
        )
        argv = [sys.executable, "-c", script]
        g._run(leg, argv, scratch, {}, scratch, False)

        if leg.rc != 0:
            r.fail("GATE-VERDICT green leg untouched", f"expected rc 0, got {leg.rc}")
        elif leg.failures:
            r.fail("GATE-VERDICT green leg untouched",
                   f"a leg that collected a passing test must have no failures, got "
                   f"{leg.failures!r}")
        elif not leg.collected:
            r.fail("GATE-VERDICT green leg untouched",
                   "the fixture's own passing test was not even collected -- fixture is broken")
        else:
            j = g.judge([leg], [], False)
            if j["verdict"] != "GREEN":
                r.fail("GATE-VERDICT green leg untouched",
                       f"a normal passing leg must judge GREEN, got {j['verdict']!r}")
            else:
                r.ok("a leg that ran and collected a passing test is left exactly as it was "
                     "and judges GREEN")


def test_gateverdict_refuses_when_not_importable_as_tex_wrangle(r: SubTestResult):
    """A checkout with no `TEX_Wrangle`-named sibling anywhere can't import a single test,
    not just the ones a gate leg is meant to catch -- every test dies at collection with a
    bare `ModuleNotFoundError`. gate.py refuses up front instead: rc 2 and a one-line
    message, before any leg's process is spawned."""
    print("\n--- GATE-VERDICT: gate.py refuses up front with no TEX_Wrangle-named sibling ---")
    g = _gate()
    with tempfile.TemporaryDirectory(prefix="tex-gate-noimport-") as scratch:
        original = g._PARENT
        g._PARENT = scratch      # a directory with no "TEX_Wrangle" child at all
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = g.main(["--tier", "cheap"])
        finally:
            g._PARENT = original
        out = buf.getvalue()
        if rc != 2:
            r.fail("GATE-VERDICT refusal", f"expected rc 2, got {rc}")
        elif "TEX_Wrangle" not in out or "run from a directory" not in out:
            r.fail("GATE-VERDICT refusal", f"expected the naming-convention refusal message, "
                                            f"got {out!r}")
        else:
            r.ok("no TEX_Wrangle-reachable sibling refuses up front with rc 2, no leg spawned")

    # And the ordinary case (this suite's own tree) must NOT refuse.
    if not g._importable_as_tex_wrangle():
        r.fail("GATE-VERDICT refusal", "this suite's own checkout must resolve as TEX_Wrangle")
    else:
        r.ok("this suite's own checkout resolves as TEX_Wrangle and is not refused")
