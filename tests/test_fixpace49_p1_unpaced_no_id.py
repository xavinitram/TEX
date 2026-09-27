"""FIX-PACE49 P1 -- an unpaced cook must
never compute `id(stmt)` at all, at any of the interpreter's three per-top-level-statement
poll call sites (the two loops in `Interpreter._execute_inner` and `_exec_stmts_profiled`).

Before this fix, `call_site_id=id(stmt))  # PACE-49` ran UNCONDITIONALLY on every statement
dispatch, paced or not -- measured +12.6 ns/call, 32% relative, on the
hottest of hot loops, for a value `paced_check` never reads once unpaced (it returns after
`token.check()` before ever looking at `call_site_id`). `heavy=_heavy_ids is not None and
id(stmt) in _heavy_ids` already short-circuits `id(stmt)` away when `_heavy_ids is None`
(unpaced) -- the caller-side `call_site_id=id(stmt))` argument did not share that gate.

Same shape as `test_fixpace_p2_gate_classification.py`'s existing four rows (a real,
`tex_engine.cook`-driven cook, an `_UnpacedToken` with `.check()` but no `.pace` attribute --
the actual shipped ComfyUI default), but counting `id()` calls on an `ASTNode` directly
(monkeypatching the module-global name `id` that `interpreter.py`'s `LOAD_GLOBAL` resolves
first, before the builtin) rather than a named classifier function -- `call_site_id`'s own
`id(stmt)` call has no dedicated wrapper to intercept otherwise.

RED at base `32f6917`: `id(stmt)` is called once per statement per poll site regardless of
whether pacing engaged.
"""
import torch

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_compiler.ast_nodes import ASTNode
from TEX_Wrangle.tex_runtime import interpreter as _interp

from helpers import make_img


class _UnpacedToken:
    """The real ComfyUI shape: a cancel token wired (not None) but with no `pace`
    attribute at all -- `pacing.wants_pacing()` reads False, so pacing never engages."""

    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1


_MULTI_STMT_PROGRAM = """
vec4 x = @A;
x = x * 1.5;
x = x + 0.25;
@OUT = x;
"""


def _count_ast_id_calls(cook_fn):
    calls = []
    real_id = id

    def _counting_id(obj):
        if isinstance(obj, ASTNode):
            calls.append(obj)
        return real_id(obj)

    _interp.id = _counting_id
    try:
        cook_fn()
    finally:
        del _interp.id
    return calls


def test_unpaced_default_progress_branch_never_calls_id_on_a_statement(r):
    print("\n--- FIX-PACE49 P1: unpaced default (on_progress=None) loop computes "
          "id(stmt) 0 times ---")
    img = make_img(1, 8, 8, 4, seed=21)

    def _cook():
        tex_engine.cook(_MULTI_STMT_PROGRAM, {"A": img}, device_mode="cpu",
                         cancel=_UnpacedToken())

    calls = _count_ast_id_calls(_cook)
    if len(calls) == 0:
        r.ok("id() was never called on an ASTNode by the unpaced default loop")
    else:
        r.fail("FIX-PACE49 P1 default loop", f"expected 0 id(stmt) calls, got {len(calls)}")


def test_unpaced_progress_wired_branch_never_calls_id_on_a_statement(r):
    print("\n--- FIX-PACE49 P1: unpaced progress-wired loop computes id(stmt) 0 times ---")
    img = make_img(1, 8, 8, 4, seed=22)

    def _on_progress(phase, frac):
        pass

    def _cook():
        tex_engine.cook(_MULTI_STMT_PROGRAM, {"A": img}, device_mode="cpu",
                         cancel=_UnpacedToken(), on_progress=_on_progress)

    calls = _count_ast_id_calls(_cook)
    if len(calls) == 0:
        r.ok("id() was never called on an ASTNode by the unpaced progress-wired loop")
    else:
        r.fail("FIX-PACE49 P1 progress loop", f"expected 0 id(stmt) calls, got {len(calls)}")


def test_unpaced_profiled_branch_never_calls_id_on_a_statement(r):
    """`_exec_stmts_profiled` is reached once PROF-1's own profiler is armed (`cancel is
    not None` alone used to be its own gate for classification, same P2-class gap)."""
    print("\n--- FIX-PACE49 P1: unpaced _exec_stmts_profiled computes id(stmt) 0 times ---")
    from TEX_Wrangle.tex_runtime import profile as _prof
    img = make_img(1, 8, 8, 4, seed=23)

    def _cook():
        _prof.reset()
        _prof.enable()
        try:
            tex_engine.cook(_MULTI_STMT_PROGRAM, {"A": img}, device_mode="cpu",
                             cancel=_UnpacedToken())
        finally:
            _prof.disable()
            _prof.reset()

    calls = _count_ast_id_calls(_cook)
    if len(calls) == 0:
        r.ok("id() was never called on an ASTNode by the unpaced profiled loop")
    else:
        r.fail("FIX-PACE49 P1 profiled loop", f"expected 0 id(stmt) calls, got {len(calls)}")


def test_paced_cook_still_calls_id_on_every_statement(r):
    """Regression guard the other direction: once `_pace.is_paced()` reads True, the
    interpreter must still classify and key by `id(stmt)` exactly as before -- this fix
    gates the UNPACED case only, never removes the paced one. `is_paced()` itself requires a
    real CUDA device to ever read True (`reset()`'s own `_state.paced = is_cuda`), so this
    unit-isolates the interpreter's OWN gating decision (which reads `_pace.is_paced()`, not
    the device) by monkeypatching it directly, matching this file's own repro shape."""
    print("\n--- FIX-PACE49 P1: is_paced()=True still computes id(stmt) per statement ---")
    real_is_paced = _interp._pace.is_paced
    _interp._pace.is_paced = lambda: True
    img = make_img(1, 8, 8, 4, seed=24)

    def _cook():
        tex_engine.cook(_MULTI_STMT_PROGRAM, {"A": img}, device_mode="cpu",
                         cancel=_UnpacedToken())

    try:
        calls = _count_ast_id_calls(_cook)
    finally:
        _interp._pace.is_paced = real_is_paced
    if len(calls) > 0:
        r.ok(f"with is_paced() forced True, id(stmt) was computed {len(calls)} time(s) -- "
             f"the gate is on pacing, not a blanket removal of id(stmt)")
    else:
        r.fail("FIX-PACE49 P1 paced regression",
               "id(stmt) was computed 0 times even with is_paced() forced True -- over-gated")
