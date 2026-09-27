"""FIX-PACE P2 (Phase C, R3 finding 1) — heavy-classification must not be computed (and
thrown away) on a cook that never actually engages pacing.

`tex_runtime/pacing.py:paced_check` never reads its own `heavy` argument at all unless
`_state.paced` is true (an early return for `token is None`, another for an unpaced/CPU
cook) — but every one of the three call sites that pass `heavy=` computes it EAGERLY, as a
plain Python argument expression, before `paced_check` ever gets a chance to ignore it.
Confirmed real (not hypothetical): ComfyUI's own node always wires a cancel token
(`tex_node.py`), and that token has no `pace` attribute at all, so `pacing.wants_pacing()`
reads `False` for it and `_state.paced` is `False` for every real ComfyUI cook — this is
"cancel wired, pacing never engaged", the actual shipped default, not merely `cancel is
None`. These rows reproduce exactly that shape: a token with `.check()` but no `.pace`.

RED at `3d39da2` (the classifier IS invoked — the call sites gate only on `cancel is not
None` / nothing at all, never on whether pacing engaged); GREEN once each site is gated
behind a cheap `pacing.is_paced()` check.
"""
import torch

from helpers import SubTestResult, make_img
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import interpreter as _interp
from TEX_Wrangle.tex_runtime import compiled as _compiled
from TEX_Wrangle.tex_runtime import pacing as _pace
from failure_harness import compile_program, clone_bindings


class _UnpacedToken:
    """The real ComfyUI shape: a cancel token wired (not None) but with no `pace`
    attribute at all — `pacing.wants_pacing()` reads False, so pacing never engages."""
    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


_HEAVY_PROGRAM = """
vec4 x = @A;
x = x * 1.5;
x = gauss_blur(x, 2.0);
@OUT = x;
"""


def test_interpreter_default_progress_branch_never_classifies_when_unpaced(r: SubTestResult):
    print("\n--- FIX-PACE P2: interpreter's SCHED-3 (on_progress=None) branch skips "
          "heavy classification on an unpaced cook ---")
    calls = {"n": 0}
    real = _interp._heavy_stmt_ids

    def _counting(stmts):
        calls["n"] += 1
        return real(stmts)

    _interp._heavy_stmt_ids = _counting
    try:
        img = make_img(1, 8, 8, 4, seed=11)
        tex_engine.cook(_HEAVY_PROGRAM, {"A": img}, device_mode="cpu",
                         cancel=_UnpacedToken())
    finally:
        _interp._heavy_stmt_ids = real

    if calls["n"] == 0:
        r.ok("heavy_stmt_ids was never called on the unpaced (real ComfyUI) default path")
    else:
        r.fail("P2 gate (on_progress=None)",
               f"heavy_stmt_ids was called {calls['n']} time(s) though this cook never "
               f"engaged pacing (no `pace` attribute on the token)")


def test_interpreter_progress_wired_branch_never_classifies_when_unpaced(r: SubTestResult):
    print("\n--- FIX-PACE P2: interpreter's on_progress-wired branch skips heavy "
          "classification on an unpaced cook ---")
    calls = {"n": 0}
    real = _interp._heavy_stmt_ids

    def _counting(stmts):
        calls["n"] += 1
        return real(stmts)

    def _on_progress(phase, frac):
        pass

    _interp._heavy_stmt_ids = _counting
    try:
        img = make_img(1, 8, 8, 4, seed=12)
        tex_engine.cook(_HEAVY_PROGRAM, {"A": img}, device_mode="cpu",
                         cancel=_UnpacedToken(), on_progress=_on_progress)
    finally:
        _interp._heavy_stmt_ids = real

    if calls["n"] == 0:
        r.ok("heavy_stmt_ids was never called on the unpaced, progress-wired path")
    else:
        r.fail("P2 gate (on_progress wired)",
               f"heavy_stmt_ids was called {calls['n']} time(s) on an unpaced cook")


def test_profiled_branch_never_classifies_when_unpaced(r: SubTestResult):
    """`_exec_stmts_profiled` already gated its classification on `cancel is not None`
    alone — the same gap, just behind PROF-1's own `enabled()` guard. Arm the profiler so
    this branch is reached, and confirm it too skips classification on an unpaced cook."""
    print("\n--- FIX-PACE P2: the profiled statement walk also skips classification on "
          "an unpaced cook ---")
    from TEX_Wrangle.tex_runtime import profile as _prof
    calls = {"n": 0}
    real = _interp._heavy_stmt_ids

    def _counting(stmts):
        calls["n"] += 1
        return real(stmts)

    _interp._heavy_stmt_ids = _counting
    _prof.reset()
    _prof.enable()
    try:
        img = make_img(1, 8, 8, 4, seed=13)
        tex_engine.cook(_HEAVY_PROGRAM, {"A": img}, device_mode="cpu",
                         cancel=_UnpacedToken())
    finally:
        _interp._heavy_stmt_ids = real
        _prof.disable()
        _prof.reset()

    if calls["n"] == 0:
        r.ok("the profiled walk skipped heavy classification on an unpaced cook")
    else:
        r.fail("P2 gate (profiled branch)",
               f"heavy_stmt_ids was called {calls['n']} time(s) on an unpaced cook")


def test_compiled_entry_poll_never_classifies_when_unpaced(r: SubTestResult):
    """`compiled.py`'s codegen/stencil-tier entry poll (`_codegen_only_execute`) computes
    `_program_has_any_heavy_stmt(program)` gated only on `cancel is not None` — the same
    gap, at the tier's own single per-cook poll point."""
    print("\n--- FIX-PACE P2: compiled.py's entry poll skips classification on an "
          "unpaced cook ---")
    calls = {"n": 0}
    real = _compiled._program_has_any_heavy_stmt

    def _counting(program):
        calls["n"] += 1
        return real(program)

    code = "@OUT = gauss_blur(@A, 2.0);"
    img = make_img(1, 8, 8, 4, seed=14)
    bindings = {"A": img}
    prog, tm, outs = compile_program(code, bindings)

    _compiled._program_has_any_heavy_stmt = _counting
    try:
        _compiled._codegen_only_execute(
            prog, clone_bindings(bindings), tm, "cpu", output_names=outs,
            fingerprint="fixpace-p2-repro", time_context=None, cancel=_UnpacedToken())
    finally:
        _compiled._program_has_any_heavy_stmt = real

    if calls["n"] == 0:
        r.ok("_program_has_any_heavy_stmt was never called on the unpaced entry poll")
    else:
        r.fail("P2 gate (compiled.py entry poll)",
               f"_program_has_any_heavy_stmt was called {calls['n']} time(s) on an "
               f"unpaced cook")
