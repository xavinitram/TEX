"""FIX-TIER T4 (R3#2) — the `scale==1.0` short-circuit for a `pixel_args=`-tagged
codegen call site moves to the EMITTED CALL SITE, so the interpreter-level `==` happens
inline and a `scale==1.0` cook (the overwhelmingly common one -- `scale=None` codegen
cooks, and any explicit `scale=1.0` cook) never pays a Python function-call frame into
`_scale_pixel_arg` (`_SCM`) at all.

R3#1's own measurement: `_scale_pixel_arg(value, 1.0)` costs ~36.4 ns/call vs. 27.7 ns/call
for a direct pass-through -- ~8.8 ns per call site per cook, on the four "dominant
compositing ops" (`gauss_blur`/`erode`/`dilate`/`bilateral_filter`). `_SCM` already
early-returns the SAME object at `scale==1.0` with no allocation, so this is a routing
change, not a behavior change: `args[i] = f"{args[i]} if _env['__tex_scale']==1.0 else
_SCM({args[i]}, _env['__tex_scale'])"` gets the identical value on that branch while
skipping the call frame, and still reaches `_SCM` (and its host-scalar-tag-preserving
multiply) unchanged for a genuine `scale != 1.0` cook.

Red-first, counts-based (GATE-47's own ratchet shape, never timing): monkeypatch
`codegen._scale_pixel_arg` with a call-counting wrapper BEFORE building a fresh program
(so the wrapper, not the original, is the one baked into the generated function's
`_SCM` global), cook it at `scale=1.0`, and assert the wrapper was never invoked -- then
cook the SAME compiled fn at a genuine scale and assert it WAS, proving the short-circuit
only ever skips the no-op case.
"""
from helpers import *
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import codegen as _cg
from TEX_Wrangle.tex_runtime import stdlib_core as _stdlib_core

# A fresh, distinctive program text so this file's compile never reuses another test's
# cached codegen fn (which would already have the REAL `_scale_pixel_arg` baked into its
# `_SCM` global from an earlier build) -- the UC-2 exact-fetch stencil shape (so
# `_should_stencil_route` fires) plus one independent `gauss_blur` call, the same
# combination `test_scalecg48_codegen_scale.py`'s `_STENCIL_PLUS_BLUR` uses, with a
# distinct sigma literal so its fingerprint differs.
_CODE = """//!tex scale: safe
i$radius = 2;
vec3 acc = vec3(0.0);
float cnt = 0.0;
for (int dy = -$radius; dy <= $radius; dy = dy + 1) {
    for (int dx = -$radius; dx <= $radius; dx = dx + 1) {
        acc = acc + fetch(@A, ix + dx, iy + dy).rgb;
        cnt = cnt + 1.0;
    }
}
@STENCIL = vec4(acc / cnt, 1.0);
@BLUR = vec4(gauss_blur(@A, 9.125).rgb, 1.0);
"""
_BT = {"A": TEXType.VEC3, "radius": TEXType.INT, "STENCIL": TEXType.VEC4, "BLUR": TEXType.VEC4}


def _bindings(res: int = 20):
    return {"A": make_img(1, res, res, 3, seed=17), "radius": 2}


def test_fixtier_t4_scm_never_called_at_scale_one(r: SubTestResult):
    print("\n--- FIX-TIER T4: _SCM is never invoked for a scale==1.0 pixel_args= call site ---")
    calls = {"n": 0}
    real = _stdlib_core._scale_pixel_arg

    def _counting(value, scale):
        calls["n"] += 1
        return real(value, scale)

    _cg._scale_pixel_arg = _counting
    try:
        out1 = tex_engine.cook(_CODE, _bindings(), device_mode="cpu", compile_mode="none",
                               scale=1.0)
        n_at_one = calls["n"]
        # Same compiled fn (fingerprint unchanged -- scale excluded), a genuine scale value:
        # the wrapper (and therefore the real multiply) MUST still be reached.
        out2 = tex_engine.cook(_CODE, dict(_bindings(), A=_bindings()["A"].clone()),
                               device_mode="cpu", compile_mode="none", scale=0.5)
    finally:
        _cg._scale_pixel_arg = real
    if n_at_one != 0:
        r.fail("no call at scale=1.0", f"_SCM's wrapper was invoked {n_at_one}x for a "
               f"scale==1.0 cook -- the short-circuit should have skipped the call entirely")
        return
    r.ok("scale=1.0 cook never called into _scale_pixel_arg (inline == short-circuits it)")
    if calls["n"] == 0:
        r.fail("still reachable at scale!=1.0", "_scale_pixel_arg's wrapper was never called "
               "for a genuine scale=0.5 cook of the SAME compiled function -- the "
               "short-circuit swallowed the real path, not just the no-op one")
        return
    md = (out1.outputs["BLUR"].float() - out2.outputs["BLUR"].float()).abs().max().item()
    if md < 1e-5:
        r.fail("sanity", f"scale=1.0 and scale=0.5 produced identical BLUR output "
               f"(maxdiff {md:.2e}) -- the multiply isn't varying with scale, so this test "
               "would pass vacuously")
        return
    r.ok(f"the SAME compiled fn called _scale_pixel_arg {calls['n']}x for scale=0.5 "
         f"(BLUR maxdiff vs scale=1.0: {md:.2e}) -- short-circuit is scale-selective, "
         "not a blanket skip")
