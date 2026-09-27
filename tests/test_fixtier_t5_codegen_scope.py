"""FIX-TIER T5 (B2#2, doc) — the scale-on-codegen route is scoped to the UC-2 stencil
shape, not to "any program calling a pixel_args= builtin".

B2's bug hunt confirmed (by running) that a plain `gauss_blur(@A, 2.0)` call, with no
independent hand-written exact-fetch stencil loop, still records tier `"interpreter"`
at a non-1.0 scale -- unchanged from before SCALE-CG-48, because `_should_stencil_route`
recognizes only the UC-2 hand-written-loop shape, never "the program calls one of the
four `pixel_args=` builtins." Not a bug (B2 filed it as informational), but
`docs/resolution-scale.md`'s prose was easy to over-read as "gauss_blur now cooks faster
under scale" -- T5 restates the scope plainly there and in `TierVerdict`'s/
`TIER_REASON_SCALE_ACTIVE_CODEGEN`'s own docstrings (`tex_engine_tiers.py`). This test
locks the underlying (already-correct) behavior against a future regression: a plain
`pixel_args=` call, alone, must keep reporting the interpreter tier under scale, and
`tier_verdict` must agree with a real cook's own `tier_trace` record."""
from helpers import *
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import tier_trace as _tt
from TEX_Wrangle.tex_engine_tiers import (
    tier_verdict, TIER_REASON_SCALE_ACTIVE, TIER_REASON_SCALE_ACTIVE_CODEGEN,
)

# A plain pixel_args= call, nothing else -- no hand-written fetch/ix/iy loop anywhere, so
# `_should_stencil_route` cannot recognize a stencil here regardless of this call.
_PLAIN_BLUR_CODE = "@OUT = vec4(gauss_blur(@A, 2.0).rgb, 1.0);\n"
_PLAIN_BLUR_BT = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}


def test_fixtier_t5_plain_pixel_args_call_stays_interpreter_under_scale(r: SubTestResult):
    print("\n--- FIX-TIER T5: a bare gauss_blur() call is NOT routed to codegen by scale ---")
    verdict = tier_verdict(_PLAIN_BLUR_CODE, compile_mode="none", device="cpu",
                           scale=0.5, binding_types=_PLAIN_BLUR_BT)
    if verdict.tier != "interpreter" or verdict.reason != TIER_REASON_SCALE_ACTIVE:
        r.fail("query scope", f"expected tier='interpreter'/{TIER_REASON_SCALE_ACTIVE!r} "
               f"for a plain pixel_args= call with no stencil loop, got {verdict!r} -- the "
               "scale-on-codegen route must stay scoped to the UC-2 stencil shape")
        return
    r.ok(f"tier_verdict reports {verdict.tier!r}/{verdict.reason!r} for a bare gauss_blur() "
         "call at scale=0.5 -- the codegen route did not fire")

    _tt.reset()
    A = make_img(1, 16, 16, 3, seed=23)
    tex_engine.cook(_PLAIN_BLUR_CODE, {"A": A}, device_mode="cpu", compile_mode="none",
                    scale=0.5)
    rec = _tt.last()
    if rec is None or rec.tier != verdict.tier:
        r.fail("query/cook agreement", f"tier_verdict said {verdict.tier!r} but the real "
               f"cook recorded {getattr(rec, 'tier', None)!r}")
        return
    r.ok(f"a real cook of the same program agrees: tier_trace recorded tier={rec.tier!r}")
