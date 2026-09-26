"""SCALE-47b phase 7 — a coarse cook defaults to precision="auto" (AUTHOR DECISION #3).

R6, adopted: reduced precision under scale's own R1 envelope must never be surfaced as a new
decision a host has to make -- the EXISTING invariant #10 accuracy net (`precision="auto"`)
already reasons about data amplification independent of canvas resolution, so a coarse cook
may simply inherit it. A caller that explicitly asks for something else on a coarse cook is
still honoured (this is a DEFAULT, not a forced override) -- a plain `prepare()` call is the
only case whose default value ("fp32", indistinguishable here from "I didn't think about it")
gets promoted.
"""
from helpers import *
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import tier_trace as _tt


def test_scale47b_coarse_cook_resolves_auto(r: SubTestResult):
    print("\n--- SCALE-47b: scale=0.5 with default precision resolves through the auto gate ---")
    A = make_img(1, 16, 16, 4)
    code = "@OUT = gauss_blur(@A, 4.0);"
    _tt.reset()
    tex_engine.cook(code, {"A": A}, device_mode="cpu", scale=0.5)
    rec = _tt.last_precision()
    if rec is None:
        r.fail("coarse auto default", "tier_trace.last_precision() is None -- the auto gate "
               "never ran, so scale did not default precision to 'auto'")
        return
    r.ok(f"a coarse (scale=0.5) default-precision cook resolved through auto: {rec!r}")


def test_scale47b_full_scale_cook_stays_plain_fp32(r: SubTestResult):
    print("\n--- SCALE-47b: scale=None (or 1.0) never defaults precision to auto (invariant #7) ---")
    A = make_img(1, 16, 16, 4)
    code = "@OUT = gauss_blur(@A, 4.0);"
    _tt.reset()
    tex_engine.cook(code, {"A": A.clone()}, device_mode="cpu")
    rec_none = _tt.last_precision()
    _tt.reset()
    tex_engine.cook(code, {"A": A.clone()}, device_mode="cpu", scale=1.0)
    rec_one = _tt.last_precision()
    if rec_none is not None or rec_one is not None:
        r.fail("no spurious auto", f"expected no auto resolution at scale=None/1.0, got "
               f"{rec_none!r} / {rec_one!r}")
        return
    r.ok("scale=None and scale=1.0 never enter the auto gate -- byte-identical to pre-SCALE-47b")


def test_scale47b_explicit_precision_on_coarse_cook_is_honoured(r: SubTestResult):
    print("\n--- SCALE-47b: an EXPLICIT precision= on a coarse cook is still honoured ---")
    A = make_img(1, 16, 16, 4)
    code = "@OUT = gauss_blur(@A, 4.0);"
    _tt.reset()
    res = tex_engine.cook(code, {"A": A}, device_mode="cpu", scale=0.5, precision="fp16")
    if res.precision != "fp16":
        r.fail("explicit precision honoured", f"expected precision='fp16', got {res.precision!r}")
        return
    r.ok(f"an explicit precision='fp16' on a coarse cook is respected, not overridden to auto "
         f"(effective precision: {res.precision!r})")
