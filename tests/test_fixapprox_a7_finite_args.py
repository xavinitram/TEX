"""FIX-APPROX A7 (v0.50 Phase C, B4#4) -- NaN/Inf radius/sigma on gauss_blur, erode,
dilate, and bilateral_filter raise a friendly TEX diagnostic (E6052), not a raw
`ValueError: cannot convert float NaN to integer` / `OverflowError: cannot convert
float infinity to integer` leaking straight from `int(math.ceil(...))`. Pre-existing
on all four builtins (confirmed unchanged at base `297903d`, per B4's own finding);
not a regression from v0.50's other radius/sigma work, but never guarded until now despite the theme
("arbitrarily large... radiuses/sigma") inviting exactly this boundary into scope.

`_require_finite_arg` (`tex_runtime/stdlib_core.py`) is the one shared check; each
builtin calls it BEFORE its own `int(math.ceil(...))`/`int(...)` conversion, so the
raw Python exception never has a chance to fire.
"""
from __future__ import annotations

import math

from helpers import *  # noqa: F401,F403
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime.interpreter import InterpreterError

_NAN_INF_CASES = [
    ("gauss_blur", "@OUT = gauss_blur(@A, $v);", "sigma"),
    ("erode", "@OUT = erode(@A, $v);", "radius"),
    ("dilate", "@OUT = dilate(@A, $v);", "radius"),
    ("bilateral_filter", "@OUT = bilateral_filter(@A, $v, 0.2);", "spatial_sigma"),
]


def test_fixapprox_a7_nan_inf_raise_e6052_not_a_raw_python_exception(r: SubTestResult):
    print("\n--- A7: NaN/Inf radius/sigma on all four radius builtins raises E6052, "
          "never a raw ValueError/OverflowError ---")
    img = make_img(1, 4, 4, 3, seed=91)
    all_ok = True
    for fn_name, code, arg_name in _NAN_INF_CASES:
        for bad, tag in ((float("nan"), "nan"), (float("inf"), "inf")):
            try:
                tex_engine.cook(code, {"A": img.clone(), "v": bad}, device_mode="cpu")
                r.fail(f"a7 {fn_name} {tag}", "cook did not raise at all")
                all_ok = False
                continue
            except InterpreterError as e:
                if e.code != "E6052":
                    r.fail(f"a7 {fn_name} {tag}", f"raised InterpreterError but code={e.code!r}, expected E6052")
                    all_ok = False
                elif fn_name not in str(e) or arg_name not in str(e):
                    r.fail(f"a7 {fn_name} {tag}", f"message does not name the function/arg: {e}")
                    all_ok = False
            except (ValueError, OverflowError) as e:
                r.fail(f"a7 {fn_name} {tag}",
                       f"raw {type(e).__name__} leaked out uncaught/unwrapped: {e}")
                all_ok = False
            except Exception as e:
                r.fail(f"a7 {fn_name} {tag}", f"unexpected {type(e).__name__}: {e}")
                all_ok = False
    if all_ok:
        r.ok("every (builtin, NaN/Inf) pair raises InterpreterError code=E6052 naming "
             "the function and the offending argument")


def test_fixapprox_a7_finite_values_are_unaffected(r: SubTestResult):
    print("\n--- A7: ordinary finite radius/sigma values are completely unaffected "
          "(invariant 7) ---")
    img = make_img(1, 8, 8, 3, seed=92)
    cases = [
        ("@OUT = gauss_blur(@A, 2.0);", {}),
        ("@OUT = erode(@A, 2);", {}),
        ("@OUT = dilate(@A, 2);", {}),
        ("@OUT = bilateral_filter(@A, 1.0, 0.2);", {}),
    ]
    for code, params in cases:
        try:
            out = tex_engine.cook(code, dict(params, A=img.clone()), device_mode="cpu").outputs["OUT"]
            if not torch.isfinite(out).all():
                r.fail("a7 finite regression", f"{code} produced a non-finite output")
                return
        except Exception as e:
            r.fail("a7 finite regression", f"{code} raised unexpectedly: {type(e).__name__}: {e}")
            return
    r.ok("every ordinary finite call still cooks normally")


def test_fixapprox_a7_gauss_blur_doc_discloses_the_approximation(r: SubTestResult):
    print("\n--- A7 (B4#4): gauss_blur's doc= string now discloses the pyramid "
          "approximation, matching bilateral_filter's own disclosure style ---")
    from TEX_Wrangle.tex_runtime import stdlib_registry as R
    by_name = {n: e for e in R.REGISTRY for n in e.names}
    doc = by_name["gauss_blur"].doc
    if "approx" in doc.lower() or "past it" in doc.lower():
        r.ok(f"gauss_blur doc= discloses the approximation: {doc!r}")
    else:
        r.fail("a7 doc disclosure", f"gauss_blur doc= does not mention the approximation: {doc!r}")
