"""FIX-SCALE S4 (v0.47 Phase C, B2 finding 4) — an EXPLICIT `precision="fp32"` on a coarse
cook must be honoured unchanged, exactly like any other explicit precision, never silently
promoted to `"auto"` the same as an unspecified caller.

Before this fix, `tex_engine.prepare(..., precision: str = "fp32", ...)` had no sentinel: the
literal default and an explicit caller value of `"fp32"` were the same Python string,
indistinguishable at the promotion site (`if precision == "fp32": precision = "auto"`). Both
the unspecified case and the explicit-fp32 case went through `resolve_auto_precision` — on
this CPU box `auto` happens to resolve back to `fp32` so the symptom was invisible here, but
on a CUDA host where `auto` can resolve to `fp16`, a caller who explicitly asked to KEEP fp32
under scale (a numerically sensitive program, or one opting out of the invariant-10 heuristic)
was silently downgraded anyway. `tier_trace.last_precision()` is the load-bearing signal: it is
only ever recorded inside the `precision == "auto"` branch, so `None` after a cook proves the
auto-resolution path never ran for it -- it is not just a matching label, it is proof the
mechanism itself was skipped."""
from helpers import *
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import tier_trace

_CODE = "@OUT = gauss_blur(@A, 4.0);"


def _cook_and_get_last_precision(precision_kwarg):
    A = make_img(1, 8, 8, 4)
    kwargs = {} if precision_kwarg is _UNSET else {"precision": precision_kwarg}
    tex_engine.cook(_CODE, {"A": A}, device_mode="cpu", scale=0.5, **kwargs)
    return tier_trace.last_precision()


_UNSET = object()


def test_s4_unspecified_precision_goes_through_auto(r: SubTestResult):
    print("\n--- FIX-SCALE S4: an UNSPECIFIED precision on a coarse cook resolves via auto ---")
    got = _cook_and_get_last_precision(_UNSET)
    if got is None:
        r.fail("unspecified precision", "tier_trace.last_precision() was None -- the auto "
               "path never ran for an unspecified precision, which should still be promoted")
        return
    r.ok(f"unspecified precision -> tier_trace.last_precision() = {got!r} (went through auto)")


def test_s4_explicit_fp32_never_touches_auto_resolution(r: SubTestResult):
    print("\n--- FIX-SCALE S4: an EXPLICIT precision='fp32' must NOT go through auto ---")
    got = _cook_and_get_last_precision("fp32")
    if got is not None:
        r.fail("explicit fp32 precision", f"expected tier_trace.last_precision() to be None "
               f"(the auto-resolution branch never runs for an explicit precision), got {got!r} "
               f"-- an explicit precision='fp32' was silently routed through auto-resolution, "
               f"identically to an unspecified one")
        return
    r.ok("explicit precision='fp32' never entered the auto-resolution branch")


def test_s4_explicit_fp32_cooks_at_fp32(r: SubTestResult):
    print("\n--- FIX-SCALE S4: an explicit precision='fp32' cook still reports fp32 ---")
    A = make_img(1, 8, 8, 4)
    res = tex_engine.cook(_CODE, {"A": A}, device_mode="cpu", scale=0.5, precision="fp32")
    if res.precision != "fp32":
        r.fail("explicit fp32 result", f"expected CookResult.precision == 'fp32', got "
               f"{res.precision!r}")
        return
    r.ok(f"explicit precision='fp32' cook reports CookResult.precision={res.precision!r}")


def test_s4_explicit_fp16_is_still_honoured(r: SubTestResult):
    print("\n--- FIX-SCALE S4: an explicit precision='fp16' is unaffected by the sentinel change ---")
    A = make_img(1, 8, 8, 4)
    res = tex_engine.cook(_CODE, {"A": A}, device_mode="cpu", scale=0.5, precision="fp16")
    if res.precision != "fp16":
        r.fail("explicit fp16 result", f"expected CookResult.precision == 'fp16', got "
               f"{res.precision!r}")
        return
    got = tier_trace.last_precision()
    if got is not None:
        r.fail("explicit fp16 trace", f"expected no auto-resolution trace for an explicit "
               f"precision, got {got!r}")
        return
    r.ok("explicit precision='fp16' cooks at fp16 and never entered auto-resolution")
