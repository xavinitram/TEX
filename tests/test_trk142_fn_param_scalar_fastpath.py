"""TRK-142 (LANG-L2 F2) — codegen's scalar-loop fast path misclassifies a user
function's own PARAMETER as scalar.

`codegen.py`'s `_emit_for_loop` decides whether a loop body is scalar-only via
`_is_scalar_body` / `_is_scalar_node`, which (for an `Identifier`) consults
`_spatial_vars` / `_var_initializers` for a body-local's spatial-ness. A
function PARAMETER never gets an entry in either dict — only `VarDecl` and
`Assignment` populate them — so `_is_scalar_node` fell through to its final
`return True`, and `_init_is_spatial` (the sibling walk `VarDecl` uses to
decide whether a *local* initialized from the parameter is itself spatial)
fell through to its final `return False`. A parameter's spatial-ness depends
on how the function is CALLED, which is unknowable from the body alone, so
BOTH walks picked the wrong default for it.

Reproduction (verbatim from the row, `run_both` at `B=1,H=1,W=4`):

    float f(float a) {
        float acc = a;
        for (int i = 0; i < 3; i = i + 1) {
            if (acc > 5.0) { break; }
            acc = acc + 1.0;
        }
        return acc;
    }
    @OUT = vec4(f(@A.r), 0.0, 0.0, 1.0);

`@A.r` is a per-pixel (spatial) argument. Before the fix, codegen's scalar
fast path fires for `f`'s loop (believing `acc`, seeded from parameter `a`, is
scalar), emits a `.item()` conversion on the per-pixel tensor `_p_a` carries,
and raises `RuntimeError: a Tensor with 4 elements cannot be converted to
Scalar` — a crash, not a wrong value; the interpreter runs the same program
fine.

Fixed by seeding every parameter name into `_spatial_vars` for the duration
of that function's own body emission (`_emit_function_def`, saved/restored
exactly like `_local_vars`) — the SAME "possibly spatial" set a reassigned
local already uses, so both `_is_scalar_node` and `_init_is_spatial` now see
`a` (and anything assigned from it, like `acc`) as spatial by the existing
mechanism, with no new one to keep in sync. A false positive here only costs
the tensor path instead of the scalar one (never wrong, only maybe slower);
that is the safe direction under invariant 2.

ComfyUI-invisible because: this is strictly a decline-vs-crash fix inside
codegen's internal fast-path selection — the DEFAULT interpreter tier is
untouched, no program's assigned pixel VALUES move (the fuzzer/edge-matrix
corpus contains no user function whose own loop reads a parameter this way,
per the row's own "reach not measured" note — this test is the first one
that does), and a program that previously crashed the accelerated tier now
degrades to the *correct*, already-existing tensor-mode loop instead of a
different value.
"""
from helpers import *

_CODE = """
float f(float a) {
    float acc = a;
    for (int i = 0; i < 3; i = i + 1) {
        if (acc > 5.0) { break; }
        acc = acc + 1.0;
    }
    return acc;
}
@OUT = vec4(f(@A.r), 0.0, 0.0, 1.0);
"""


def test_trk142_fn_param_spatial_arg_matches_interpreter(r: SubTestResult):
    """Verbatim TRK-142 repro: codegen must not crash, and must agree with the
    interpreter bit-exactly (invariant 2) when `f` is called with a spatial arg."""
    print("\n--- TRK-142: a function parameter fed a spatial arg is not fast-pathed as scalar ---")
    bindings = {"A": make_img(1, 1, 4, 3, seed=142)}
    assert_equiv(r, "trk142_fn_param_spatial_arg", _CODE, bindings, B=1, H=1, W=4)


def test_trk142_codegen_actually_ran(r: SubTestResult):
    """`assert_equiv` reports OK on a codegen decline too (SKIPPED) — pin that this
    program is one codegen actually SERVES, so the row's crash (not a mere decline)
    is the thing under test, not silently bypassed."""
    print("\n--- TRK-142: codegen serves this program (does not merely decline it) ---")
    bindings = {"A": make_img(1, 1, 4, 3, seed=142)}
    try:
        _interp_res, cg_res = run_both(_CODE, bindings, B=1, H=1, W=4)
        if cg_res is None:
            r.fail("codegen serves the TRK-142 program", "try_compile declined it (cg_res is None)")
        else:
            r.ok("codegen served the program (crash, if any, is a real bug, not a decline)")
    except Exception as e:
        r.fail("codegen serves the TRK-142 program without raising", f"{type(e).__name__}: {e}")


def test_trk142_fn_param_scalar_arg_still_scalar_fast_path(r: SubTestResult):
    """Control: the SAME function called with a purely scalar (non-spatial) argument
    must still agree with the interpreter — the fix must not force every user-function
    call onto the slower tensor path when the argument genuinely isn't spatial."""
    print("\n--- TRK-142 control: a scalar argument still cooks correctly ---")
    code = """
    float f(float a) {
        float acc = a;
        for (int i = 0; i < 3; i = i + 1) {
            if (acc > 5.0) { break; }
            acc = acc + 1.0;
        }
        return acc;
    }
    @OUT = vec4(f(2.0), 0.0, 0.0, 1.0);
    """
    assert_equiv(r, "trk142_fn_param_scalar_arg_control", code, {}, B=1, H=1, W=4)
