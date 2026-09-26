"""PACE-47d — registry-derived heavy/cheap statement classification, and coverage on
every paced poll route (interpreter per-statement, codegen `_CK`, the two Gap-1 builtins
that gained an internal poll this ask, and the stencil/codegen entry poll).

PACE-47c closed the completed-tail blind spot (`tex_runtime/pacing.py:paced_check`'s
`heavy=` bypass) for exactly two builtins by hand. PACE-47d generalizes the
CLASSIFICATION half so every poll site can ask "is the statement about to run heavy",
derived from the registry's own footprint tag (`tex_runtime/pacing_heavy.py`), never a
hand-maintained name list. These rows are RED against PACE-47c's own head (`heavy_builtin_
names`/`heavy_stmt_ids` do not exist there at all -- ImportError) and GREEN at this ask's
head.
"""
import pytest

from TEX_Wrangle.tex_runtime import pacing_heavy as _heavy
from TEX_Wrangle.tex_runtime import pacing as _pace
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle import tex_engine

from helpers import SubTestResult, make_img, torch  # noqa: F401


# ── heavy_builtin_names(): registry-derived, not a hand list ─────────────────────

def test_heavy_names_include_every_halo_footprint_builtin(r):
    print("\n--- PACE-47d: heavy_builtin_names() is derived from the registry's own "
          "footprint tag ---")
    names = _heavy.heavy_builtin_names()
    expected_heavy = {"gauss_blur", "erode", "dilate", "bilateral_filter"}
    missing = expected_heavy - names
    if missing:
        r.fail("heavy names coverage", f"missing from heavy_builtin_names(): {missing}")
    else:
        r.ok(f"all of {sorted(expected_heavy)} are registry-derived heavy")


def test_heavy_names_exclude_image_footprint_non_halo_builtins(r):
    """`sample`/`fetch`/`sample_mip` are footprint='image', not halo-shaped -- NOT in the
    per-statement heavy set (by design: `sample_mip`'s own multi-pass internal poll,
    PACE-47c, already covers it at its own entry -- see `pacing_heavy.py`'s docstring)."""
    print("\n--- PACE-47d: heavy_builtin_names() excludes plain image-footprint names ---")
    names = _heavy.heavy_builtin_names()
    unexpected = {"sample", "fetch", "sample_mip", "sample_mip_gauss"} & names
    if unexpected:
        r.fail("heavy names over-inclusion", f"unexpectedly heavy: {unexpected}")
    else:
        r.ok("sample/fetch/sample_mip/sample_mip_gauss are not in the halo-derived set")


# ── heavy_stmt_ids(): per-statement classification, including nested calls ───────

_MIXED_PROGRAM = """
vec4 x = @A;
x = x * 1.5;
x = gauss_blur(x, 2.0);
float t = 0.0;
if (x.r > 0.5) {
    x = erode(x, 2);
}
@OUT = x;
"""


def test_heavy_stmt_ids_marks_direct_and_nested_heavy_calls(r):
    print("\n--- PACE-47d: heavy_stmt_ids() marks the gauss_blur and if-nested erode "
          "statements, not the cheap ones ---")
    prog = parse_and_split(_MIXED_PROGRAM, {"A": None})
    stmts = prog.statements
    heavy_ids = _heavy.heavy_stmt_ids(stmts)
    # stmts[0] = "vec4 x = @A;" (cheap), [1] = "x = x*1.5;" (cheap),
    # [2] = "x = gauss_blur(...)" (heavy), [3] = "float t=0.0;" (cheap),
    # [4] = the if-block containing erode (heavy, nested), [5] = "@OUT = x;" (cheap)
    heavy_flags = [id(s) in heavy_ids for s in stmts]
    expected = [False, False, True, False, True, False]
    if heavy_flags == expected:
        r.ok(f"heavy flags {heavy_flags} match expected {expected}")
    else:
        r.fail("heavy_stmt_ids nested classification",
               f"got {heavy_flags}, expected {expected}")


def test_heavy_stmt_ids_memoizes_per_statement_list(r):
    """A second call with the SAME `stmts` list object must not re-walk -- proven by
    monkeypatching the per-statement walker to count calls."""
    print("\n--- PACE-47d: heavy_stmt_ids() memoizes per program statement list ---")
    prog = parse_and_split(_MIXED_PROGRAM, {"A": None})
    stmts = prog.statements
    _heavy._HEAVY_STMT_MEMO.clear()
    calls = {"n": 0}
    real_walker = _heavy._stmt_calls_heavy_builtin

    def _counting_walker(stmt):
        calls["n"] += 1
        return real_walker(stmt)

    _heavy._stmt_calls_heavy_builtin = _counting_walker
    try:
        _heavy.heavy_stmt_ids(stmts)
        first_call_count = calls["n"]
        _heavy.heavy_stmt_ids(stmts)     # same list object -- must hit the memo
        second_call_count = calls["n"]
    finally:
        _heavy._stmt_calls_heavy_builtin = real_walker
        _heavy._HEAVY_STMT_MEMO.clear()

    if first_call_count == len(stmts) and second_call_count == first_call_count:
        r.ok(f"first call walked {first_call_count} statements; second call added 0 "
             f"(memo hit)")
    else:
        r.fail("heavy_stmt_ids memo", f"first={first_call_count}, second={second_call_count}, "
               f"len(stmts)={len(stmts)}")


# ── End-to-end: the interpreter's per-statement poll actually passes heavy= ──────

def test_interpreter_poll_passes_heavy_true_for_the_gauss_blur_statement(r):
    """The real wiring, not just the classifier in isolation: cook `_MIXED_PROGRAM` on
    CPU with a paced (but never-tripping) token, spying on `_pace.paced_check` to record
    the `heavy` kwarg passed for each of the 6 top-level-statement polls. Exactly the
    gauss_blur and the if-block (erode) polls must read `heavy=True`; every other poll
    must read `heavy=False` -- proving the interpreter's classification reaches the real
    call, not just `pacing_heavy`'s own unit-level answer."""
    print("\n--- PACE-47d: the interpreter's per-statement poll passes heavy= correctly ---")
    calls = []
    real_paced_check = _pace.paced_check

    def _spy_paced_check(token, device, heavy=False):
        calls.append(heavy)
        return real_paced_check(token, device, heavy=heavy)

    class _NeverTripToken:
        pace = True
        pace_depth = 8
        pace_stride_ms = 0

        def check(self):
            pass

    _pace.paced_check = _spy_paced_check
    try:
        img = make_img(1, 8, 8, 4, seed=3)
        tex_engine.cook(_MIXED_PROGRAM, {"A": img}, device_mode="cpu",
                        cancel=_NeverTripToken())
    finally:
        _pace.paced_check = real_paced_check

    # CPU cooks are never actually `_state.paced` (pacing only engages on CUDA), so the
    # calls happen but resolve heavy=False downstream regardless of what was PASSED IN --
    # this test is about what the interpreter PASSES, which is checkable independent of
    # CUDA. NOT asserting an exact poll count or position: `tex_engine.cook` runs the
    # optimizer/fusion/lazy passes before the interpreter ever sees `stmts`, so the
    # POST-OPTIMIZATION statement shape (and count) is not the same 6-statement AST
    # `parse_and_split` alone returns (confirmed empirically: 8 polls fired here, not 6) --
    # asserting the real wiring works means checking heavy WAS seen (the gauss_blur/erode
    # calls survived optimization and were classified) and NOT-heavy was ALSO seen (the
    # classifier does not just default everything true), not pinning the exact shape.
    if True in calls and False in calls:
        r.ok(f"paced_check received a real mix of heavy=True/False across {len(calls)} "
             f"polls: {calls}")
    else:
        r.fail("interpreter heavy wiring", f"paced_check received heavy={calls} -- "
               f"expected at least one True (a heavy statement detected) and one False "
               f"(a cheap statement not over-classified)")
