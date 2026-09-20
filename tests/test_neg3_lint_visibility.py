"""NEG-3 — a lint that crashes must not report "no problems".

`tex_api.control_flow_advisories` swallowed every exception into `[]` at both of its sites.
`[]` is also the answer for a clean program, so an internal failure — a bad import, a new AST
node the walker does not know, a typo in the analysis — rendered as an empty gutter in the
editor, and the node then cooked a program the editor had called clean. `check()` next door
had the answer already: on an internal failure it returns ONE synthetic `E0000` describing
it. The advisory now does the same, so the return type never changes and a host needs no new
branch to see the difference.

What must NOT change, and is pinned here in both directions: the three DECLINE cases. A
program that does not parse, a non-string source and an over-budget analysis are answers, not
crashes, and each still returns []. The budget case is load-bearing — `tex_roi` fails CLOSED
on the same program while this fails OPEN, and that asymmetry is deliberate.
"""
from helpers import SubTestResult

_RAISER_MSG = "probe: the lint fell over"


class _Boom(RuntimeError):
    pass


def _one_e0000(diags):
    assert isinstance(diags, list), type(diags)
    assert len(diags) == 1, [getattr(d, "code", d) for d in diags]
    d = diags[0]
    assert d.code == "E0000", d.code
    assert d.severity == "error", d.severity
    return d


def test_neg3_a_crashed_advisory_lint_says_so(r: SubTestResult):
    """Both sites: a raising front end and a raising analysis each yield one E0000."""
    from TEX_Wrangle import tex_api, tex_cache

    good = "for (int i = 0; i < int(u * 8.0); i = i + 1) { }\n@OUT = vec4(1.0);"

    # Sanity: the program really does produce an advisory, so an empty list below would be a
    # real regression and not a program with nothing to say.
    try:
        codes = [d.code for d in tex_api.control_flow_advisories(good, {})]
        assert "W7007" in codes, codes
        r.ok(f"the probe program really does draw an advisory when the lint works: {codes}")
    except Exception as e:
        r.fail("advisory probe program", f"{type(e).__name__}: {e}")
        return

    # Site 1 — the front end raises something that is NOT a compile error.
    real_parse = tex_cache.parse_and_split
    try:
        def _boom(*a, **k):
            raise _Boom(_RAISER_MSG)
        tex_cache.parse_and_split = _boom
        try:
            d = _one_e0000(tex_api.control_flow_advisories(good, {}))
        finally:
            tex_cache.parse_and_split = real_parse
        assert "_Boom" in d.message, d.message
        assert _RAISER_MSG in d.message, d.message
        assert "parse" in d.message, d.message
        r.ok("front-end crash: one E0000 naming the exception type, not an empty list")
    except Exception as e:
        tex_cache.parse_and_split = real_parse
        r.fail("crashed front end is visible", f"{type(e).__name__}: {e}")

    # Site 2 — the analysis itself raises.
    real_run = tex_api._ControlFlowLint.run
    try:
        def _boom_run(self):
            raise _Boom(_RAISER_MSG)
        tex_api._ControlFlowLint.run = _boom_run
        try:
            d = _one_e0000(tex_api.control_flow_advisories(good, {}))
        finally:
            tex_api._ControlFlowLint.run = real_run
        assert "_Boom" in d.message and _RAISER_MSG in d.message, d.message
        assert "analysis" in d.message, d.message
        r.ok("analysis crash: one E0000 naming the exception type, not an empty list")
    except Exception as e:
        tex_api._ControlFlowLint.run = real_run
        r.fail("crashed analysis is visible", f"{type(e).__name__}: {e}")

    try:
        # ...and it stays TOTAL: the contract is that this never raises, whatever happens.
        tex_api._ControlFlowLint.run = _boom_run
        try:
            for src in (good, "@OUT = vec4(1.0);", ""):
                tex_api.control_flow_advisories(src, {})
        finally:
            tex_api._ControlFlowLint.run = real_run
        r.ok("totality survives the change: a crashing lint still never raises")
    except Exception as e:
        tex_api._ControlFlowLint.run = real_run
        r.fail("advisory totality", f"{type(e).__name__}: {e}")


def test_neg3_the_declines_still_return_empty(r: SubTestResult):
    """A decline is not a crash. The three documented [] answers are unchanged."""
    from TEX_Wrangle import tex_api

    try:
        for bad in ("", "@OUT = ;", "if (u > 0.5) { break; ", "\x00\xff{{{", "for (;;) {}"):
            got = tex_api.control_flow_advisories(bad, {})
            assert got == [], (bad, [d.code for d in got])
        r.ok("a program that does not parse still returns [] (check() reports that error)")
    except Exception as e:
        r.fail("parse-failure decline", f"{type(e).__name__}: {e}")

    try:
        for junk in (None, 123, object(), b"@OUT = vec4(1.0);"):
            assert tex_api.control_flow_advisories(junk, {}) == [], junk
        r.ok("a non-string source still returns [] (junk input, not a crashed lint)")
    except Exception as e:
        r.fail("junk-input decline", f"{type(e).__name__}: {e}")

    try:
        # The budget decline, driven through the real budget rather than a patch: the same
        # program `tex_roi.region_dependent` fails CLOSED on must still fail OPEN here.
        real_run = tex_api._ControlFlowLint.run

        def _over_budget(self):
            raise tex_api._CFBudget()
        tex_api._ControlFlowLint.run = _over_budget
        try:
            got = tex_api.control_flow_advisories("@OUT = vec4(1.0);", {})
        finally:
            tex_api._ControlFlowLint.run = real_run
        assert got == [], [d.code for d in got]
        r.ok("an over-budget analysis still returns [] (a decline, not a failure)")
    except Exception as e:
        r.fail("budget decline", f"{type(e).__name__}: {e}")

    try:
        # And the neighbour whose shape this borrows still behaves the same way, so the two
        # internal-failure reports cannot drift apart into two different editor experiences.
        from TEX_Wrangle.tex_compiler import type_checker as _tc
        real_check = _tc.TypeChecker.check_collect

        def _boom(self, *a, **k):
            raise _Boom("probe: the checker fell over")
        _tc.TypeChecker.check_collect = _boom
        try:
            diags = tex_api.check("@OUT = vec4(1.0);", {})
        finally:
            _tc.TypeChecker.check_collect = real_check
        assert [d.code for d in diags] == ["E0000"], [d.code for d in diags]
        assert "internal error" in diags[0].message, diags[0].message
        r.ok("check() still answers an internal failure with the same one E0000")
    except Exception as e:
        _tc.TypeChecker.check_collect = real_check
        r.fail("check() internal-failure shape", f"{type(e).__name__}: {e}")
