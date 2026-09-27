"""FIX-PACE P5 (Phase C, R2 simplification findings #1/#2/#4/#5) — four collapses with no
behaviour change:

1. (R2#1) `_state.large_resolution` is gone; a large-resolution cook forces `stride_s` to
   `0.0` in `reset()` instead of writing a second, parallel field (covered by
   `test_pace462_bounded_lookahead.py::test_pace47e_reset_forces_stride_zero_from_spatial_shape`,
   updated alongside this item — not repeated here).
2. (R2#2) `poll_cook_cancel_heavy()` is gone; every former caller now spells
   `poll_cook_cancel(heavy=True)`.
3. (R2#4) `Interpreter.execute()`'s heavy-id classification is computed ONCE per cook (when
   paced at all), not once per branch.
4. (R2#5) `tex_runtime/pacing.py` carries no pointer, by name, to a local evidence document
   this project keeps out of every push — a reader outside this checkout gains nothing
   from being told such a thing exists somewhere they cannot reach. The two-word compound
   this project uses for that document is spelled from PIECES below (never contiguous in
   this file's own source text), the same technique `test_lint1_no_local_only_path_refs.py`
   uses for its own fragments — otherwise this file would itself become an instance of the
   very leak it checks for.

RED at the FIX-PACE P4 head (before this item): `poll_cook_cancel_heavy` still exists,
`pacing.py` still carries that pointer ten times, and the interpreter's per-statement
branches still each classify independently.
"""
import re

from helpers import SubTestResult
from TEX_Wrangle.tex_runtime import stdlib_core as _sc
from TEX_Wrangle.tex_runtime import interpreter as _interp

#: Assembled at RUN time from pieces that are never contiguous in this file's own source
#: text — see the module docstring's point 4.
_LOCAL_DOC_WORD = "".join(("hand", "-", "back"))


def test_poll_cook_cancel_heavy_is_gone(r: SubTestResult):
    print("\n--- FIX-PACE P5 (R2#2): poll_cook_cancel_heavy no longer exists ---")
    if hasattr(_sc, "poll_cook_cancel_heavy"):
        r.fail("R2#2 poll_cook_cancel_heavy removal",
               "stdlib_core.poll_cook_cancel_heavy still exists -- expected it folded "
               "into poll_cook_cancel(heavy=True)")
    else:
        r.ok("stdlib_core.poll_cook_cancel_heavy is gone")


def test_gauss_blur_and_mip_pyramid_use_poll_cook_cancel_heavy_kwarg(r: SubTestResult):
    print("\n--- FIX-PACE P5 (R2#2): the two former callers now pass heavy=True by "
          "keyword ---")
    src = open(_sc.__file__, encoding="utf-8").read()
    calls = re.findall(r"poll_cook_cancel\(heavy=True\)", src)
    if len(calls) >= 2:
        r.ok(f"found {len(calls)} poll_cook_cancel(heavy=True) call sites "
             f"(_gauss_blur_bchw's inter-pass poll, _build_mip_pyramid's per-level poll)")
    else:
        r.fail("R2#2 call-site rewrite",
               f"expected >= 2 poll_cook_cancel(heavy=True) call sites in stdlib_core.py, "
               f"found {len(calls)}")


def test_pacing_module_carries_no_bare_local_doc_pointer(r: SubTestResult):
    """R2#5: `pacing.py` used to point at a local evidence document by name, ten times --
    a pointer to a local, unpushed document a reader of the published package can never
    open. None of this module's own prose or comments may reference it by that name."""
    print("\n--- FIX-PACE P5 (R2#5): pacing.py carries no bare local-doc-name pointer ---")
    import TEX_Wrangle.tex_runtime.pacing as _pace_mod
    src = open(_pace_mod.__file__, encoding="utf-8").read()
    needle = _LOCAL_DOC_WORD.replace("-", "")
    hits = [n for n, line in enumerate(src.splitlines(), 1)
            if _LOCAL_DOC_WORD in line.lower() or needle in line.lower()]
    if not hits:
        r.ok("no local-doc-name pointer remains in pacing.py")
    else:
        r.fail("R2#5 local-doc-name leak", f"still present at line(s) {hits}")


def test_heavy_ids_has_one_call_site_in_execute_not_two(r: SubTestResult):
    """R2#4: `execute()`'s unprofiled dispatch used to call `_heavy_stmt_ids(stmts)` once
    inside the `on_progress is None` branch and again, identically (the second site's own
    comment: "see the branch above"), inside the `else` branch -- two source call sites
    for the SAME memoized value, one per branch. `_exec_stmts_profiled` already had the
    simpler one-call-above-the-branch shape. A per-cook call COUNT can't tell these apart
    (only one branch runs per cook either way) -- this is a source-shape finding, so it is
    checked as one: the whole file should carry exactly 2 call sites total (`execute()`'s
    ONE hoisted call, plus `_exec_stmts_profiled`'s own), not 3."""
    print("\n--- FIX-PACE P5 (R2#4): _heavy_stmt_ids(stmts) has one call site in "
          "execute(), not one per branch ---")
    src = open(_interp.__file__, encoding="utf-8").read()
    sites = re.findall(r"_heavy_stmt_ids\(stmts\)", src)
    if len(sites) == 2:
        r.ok(f"exactly 2 call sites in interpreter.py (execute()'s hoisted call + "
             f"_exec_stmts_profiled's own), not 3")
    else:
        r.fail("R2#4 single classification site",
               f"found {len(sites)} call sites to _heavy_stmt_ids(stmts) in "
               f"interpreter.py, expected exactly 2")
