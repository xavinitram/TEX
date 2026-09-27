"""SCALE-CG-48 — resolution scale threaded through the codegen tier.

A `pixel_args=`-tagged builtin's pixel-unit argument now emits `arg * _env['__tex_scale']` in codegen
(`tex_runtime/codegen.py`), a runtime value read from the cook's own environment dict, never a
folded literal -- exactly why the SAME cached codegen fn (keyed only by the program fingerprint;
`scale` is deliberately excluded, `tex_runtime/compiled._get_or_make_codegen_fn`'s own docstring)
can serve every scale value without recompiling. `tex_engine_tiers._run_tier` now routes a
scale-active cook to codegen when `_should_stencil_route` (UC-2's own "would codegen otherwise be
chosen" test for the "default" tier) says yes; `torch_compile`/`auto`/`cuda_graph` remain forced
to the interpreter, unchanged (SCALE-COMPILED-48 is a v0.49+ item).

Red-first per the brief: (a) the SAME fingerprint serves two scale values without recompiling;
(b) codegen vs interpreter stay bit-exact under scale (the codegen-equivalence suite's own
`scale != 1.0` axis, CPU + CUDA); (c) the scale refusal/verdict contract is unchanged on the new
route (the classifier still runs, and still refuses, entirely upstream of tier dispatch).
"""
from helpers import *
from TEX_Wrangle import tex_engine
from TEX_Wrangle import tex_engine_tiers as _tiers
from TEX_Wrangle import tex_roi as _tex_roi
from TEX_Wrangle.tex_runtime import tier_trace as _tt
from TEX_Wrangle.tex_runtime import compiled as _compiled
from TEX_Wrangle.tex_runtime.codegen import detect_stencil_route
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_cache import parse_and_split


# The UC-2 exact-fetch box-blur (tests/test_v015_phase2.py / test_v044_cancel44.py's own
# `_BOX_BLUR`), reused verbatim as the "codegen would otherwise be chosen" shape, PLUS a second,
# independent top-level output calling a `pixel_args=`-tagged builtin (`gauss_blur`) directly on
# the raw binding -- `detect_stencil_route` walks every top-level statement (codegen_stencil.py),
# so the extra output does not disturb the stencil detection this whole file's routing rests on.
#
# `fetch(@A, ix + dx, iy + dy)` reads `ix`/`iy` through a BINARY EXPRESSION, not the bare
# whitelisted argument position `tex_roi.scale_safe`'s walk recognises (SCALE-47b's "known
# over-refusal", FIX-SCALE S9) -- the classifier over-approximates to UNSAFE here exactly as
# documented, so this file uses the sanctioned author-override pragma (`//!tex scale: safe`) to
# vouch for it, per `docs/resolution-scale.md`'s own "the classifier and the override comment".
# This is orthogonal to what this file tests (codegen ROUTING and EMISSION, not the classifier).
_STENCIL_PLUS_BLUR = """//!tex scale: safe
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
@BLUR = gauss_blur(@A, 8.0);
"""
_STENCIL_PLUS_BLUR_BT = {"A": TEXType.VEC3, "radius": TEXType.INT,
                         "STENCIL": TEXType.VEC4, "BLUR": TEXType.VEC4}


def _prog():
    prog = parse_and_split(_STENCIL_PLUS_BLUR, _STENCIL_PLUS_BLUR_BT)
    TypeChecker(binding_types=_STENCIL_PLUS_BLUR_BT, source=_STENCIL_PLUS_BLUR).check(prog)
    return prog


def test_scalecg48_precondition_stencil_plus_blur_shape(r: SubTestResult):
    """This file's whole premise: `_STENCIL_PLUS_BLUR` must still be the exact-fetch stencil
    shape UC-2 routes to codegen. If this stops being true every other test here is exercising
    the wrong path and its passes would be silently meaningless (mirrors CANCEL-44's own
    precondition test for the identical reason)."""
    if detect_stencil_route(_prog()):
        r.ok("_STENCIL_PLUS_BLUR is still the exact-fetch stencil shape UC-2 accelerates, "
             "with an independent gauss_blur output alongside it")
    else:
        r.fail("scalecg48 precondition", "_STENCIL_PLUS_BLUR no longer routes -- wrong test shape")


def _bindings(res: int, radius: int = 2, seed: int = 5):
    return {"A": make_img(1, res, res, 3, seed=seed), "radius": radius}


def test_scalecg48_scale_active_stencil_route_uses_codegen(r: SubTestResult):
    print("\n--- SCALE-CG-48: a scale-active cook on a stencil-routable program runs on "
          "codegen, not the interpreter ---")
    _tt.reset()
    plan = tex_engine.prepare(_STENCIL_PLUS_BLUR, _bindings(24), device_mode="cpu",
                              compile_mode="none", scale=0.5)
    tex_engine.run(plan)
    rec = _tt.last()
    if rec is None or rec.tier != "codegen":
        r.fail("scale codegen routing", f"expected tier='codegen', got {rec!r}")
        return
    r.ok(f"scale=0.5 on a stencil-routable program recorded tier={rec.tier!r} "
         "(SCALE-CG-48: codegen, not forced interpreter)")


def test_scalecg48_non_default_tier_still_forces_interpreter(r: SubTestResult):
    print("\n--- SCALE-CG-48: compiled/graphed tiers are UNCHANGED -- still forced interpreter ---")
    _tt.reset()
    plan = tex_engine.prepare(_STENCIL_PLUS_BLUR, _bindings(24), device_mode="cpu",
                              compile_mode="torch_compile", scale=0.5)
    tex_engine.run(plan)
    rec = _tt.last()
    if rec is None or rec.tier != "interpreter":
        r.fail("compiled tier scale bypass", f"expected tier='interpreter', got {rec!r}")
        return
    if not rec.reason or "scale" not in rec.reason.lower():
        r.fail("compiled tier scale bypass reason", f"reason did not name scale: {rec.reason!r}")
        return
    r.ok(f"compile_mode='torch_compile' + scale=0.5 still forces tier={rec.tier!r} "
         f"reason={rec.reason!r} -- SCALE-COMPILED-48 is a later ask, not this one")


def test_scalecg48_gauss_blur_sigma_scales_on_codegen_route(r: SubTestResult):
    print("\n--- SCALE-CG-48: gauss_blur's sigma is multiplied by scale on the codegen route ---")
    A = make_img(1, 24, 24, 3, seed=7)
    scaled = tex_engine.cook(_STENCIL_PLUS_BLUR, dict(_bindings(24), A=A.clone()),
                             device_mode="cpu", compile_mode="none", scale=0.5)
    half_code = _STENCIL_PLUS_BLUR.replace("gauss_blur(@A, 8.0)", "gauss_blur(@A, 4.0)")
    half = tex_engine.cook(half_code, dict(_bindings(24), A=A.clone()),
                          device_mode="cpu", compile_mode="none")
    md = (scaled.outputs["BLUR"].float() - half.outputs["BLUR"].float()).abs().max().item()
    if md < 1e-5:
        r.ok(f"gauss_blur(@A, 8.0) at scale=0.5 == gauss_blur(@A, 4.0) at scale=None, "
             f"both via codegen (maxdiff {md:.2e})")
    else:
        r.fail("codegen scale multiply", f"maxdiff {md:.2e} -- sigma was not halved on codegen")


def test_scalecg48_same_fingerprint_no_recompile_across_scale_values(r: SubTestResult):
    """Red-first (a): the SAME fingerprint serves two DIFFERENT scale values without
    recompiling -- a counts/structural test, never timing (GATE-47's own ratchet shape)."""
    print("\n--- SCALE-CG-48: one compile serves scale=0.5 AND scale=0.25 (same fingerprint) ---")
    calls = {"n": 0}
    real_try_codegen = _compiled._try_codegen

    def _counting_try_codegen(*a, **kw):
        calls["n"] += 1
        return real_try_codegen(*a, **kw)

    _compiled._try_codegen = _counting_try_codegen
    try:
        A = make_img(1, 20, 20, 3, seed=11)
        out1 = tex_engine.cook(_STENCIL_PLUS_BLUR, dict(_bindings(20), A=A.clone()),
                               device_mode="cpu", compile_mode="none", scale=0.5)
        out2 = tex_engine.cook(_STENCIL_PLUS_BLUR, dict(_bindings(20), A=A.clone()),
                               device_mode="cpu", compile_mode="none", scale=0.25)
    finally:
        _compiled._try_codegen = real_try_codegen

    if calls["n"] > 1:
        r.fail("no-recompile-across-scale", f"codegen emitted/compiled {calls['n']} times for "
               "two scale values of the SAME program -- expected at most 1 (fingerprint cache "
               "hit on the second)")
        return
    md = (out1.outputs["BLUR"].float() - out2.outputs["BLUR"].float()).abs().max().item()
    if md < 1e-5:
        r.fail("no-recompile-across-scale sanity",
               f"scale=0.5 and scale=0.25 produced the SAME BLUR pixels (maxdiff {md:.2e}) -- "
               "the multiply is not actually varying with scale, this test would pass vacuously")
        return
    r.ok(f"codegen emit+compile called {calls['n']}x total for scale=0.5 then scale=0.25 of the "
         f"SAME program (fingerprint cache reused), and the two scales produced DIFFERENT "
         f"pixels (maxdiff {md:.2e}) -- proving the reused fn still reads scale as a value")


def _cook_forcing_interpreter(code, bindings, *, scale, device_mode):
    """Force the plain-interpreter tier for comparison, bypassing the new codegen route,
    by making `_should_stencil_route` answer False for exactly this one call -- the
    cleanest way to get an interpreter-tier reading of the SAME scale-active cook without
    duplicating `_run_tier`'s own dispatch logic here. Patched on `tex_engine` itself (the
    REAL module `_tiers._tex_engine` lazily proxies to, per its own `_LazyTexEngine` --
    `__slots__` refuses a direct attribute set on the proxy)."""
    saved = tex_engine._should_stencil_route
    tex_engine._should_stencil_route = lambda *a, **kw: False
    try:
        return tex_engine.cook(code, bindings, device_mode=device_mode,
                               compile_mode="none", scale=scale)
    finally:
        tex_engine._should_stencil_route = saved


def _codegen_interp_scale_parity(r: SubTestResult, device_mode: str, scale: float, tag: str):
    A = make_img(1, 24, 24, 3, seed=3)
    cg = tex_engine.cook(_STENCIL_PLUS_BLUR, dict(_bindings(24), A=A.clone()),
                         device_mode=device_mode, compile_mode="none", scale=scale)
    interp = _cook_forcing_interpreter(_STENCIL_PLUS_BLUR, dict(_bindings(24), A=A.clone()),
                                       scale=scale, device_mode=device_mode)
    md_blur = (cg.outputs["BLUR"].float() - interp.outputs["BLUR"].float()).abs().max().item()
    md_stencil = (cg.outputs["STENCIL"].float()
                  - interp.outputs["STENCIL"].float()).abs().max().item()
    if md_blur < 1e-5 and md_stencil < 1e-5:
        r.ok(f"[{tag}] scale={scale}: codegen == interpreter (invariant #2), "
             f"BLUR maxdiff {md_blur:.2e}, STENCIL maxdiff {md_stencil:.2e}")
    else:
        r.fail(f"codegen/interpreter parity [{tag}] scale={scale}",
               f"BLUR maxdiff {md_blur:.2e}, STENCIL maxdiff {md_stencil:.2e} (want < 1e-5)")


def test_scalecg48_codegen_interp_parity(r: SubTestResult):
    """FIX-TIER T6 (R2#3): one parametrised loop over (device, scale) replacing six
    near-identical 3-line wrappers that differed only in that pair -- `r.ok`/`r.fail`
    already carry the per-case `[tag] scale=...` label, so nothing about per-case
    reporting is lost by looping instead of repeating the call six times."""
    print("\n--- SCALE-CG-48 (b): codegen == interpreter, CPU + CUDA, at three scales ---")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    if "cuda" not in devices:
        r.skip("cuda parity", "CUDA not available on this box -- covered on the laptop lease")
    for device in devices:
        for scale in (0.5, 0.25, 0.125):
            _codegen_interp_scale_parity(r, device, scale, device)


def test_scalecg48_scale_none_emits_no_new_bytes_without_pixel_args(r: SubTestResult):
    """Invariant #7: a program with NO `pixel_args=` call is untouched -- `__tex_scale` never
    appears in its emitted source at all, scale-active or not."""
    print("\n--- SCALE-CG-48: a stencil-only program (no gauss_blur) never emits __tex_scale ---")
    box_only = _STENCIL_PLUS_BLUR.split("@BLUR")[0] + "@STENCIL2 = @STENCIL;\n"
    # (kept minimal: just prove the marker string is absent from a program with no
    # pixel_args=-tagged call anywhere in it)
    from TEX_Wrangle.tex_runtime import codegen as _cg
    prog = parse_and_split(
        "i$radius = 2;\nvec3 acc = vec3(0.0);\nfloat cnt = 0.0;\n"
        "for (int dy = -$radius; dy <= $radius; dy = dy + 1) {\n"
        "  for (int dx = -$radius; dx <= $radius; dx = dx + 1) {\n"
        "    acc = acc + fetch(@A, ix + dx, iy + dy).rgb;\n"
        "    cnt = cnt + 1.0;\n"
        "  }\n"
        "}\n@STENCIL = vec4(acc / cnt, 1.0);\n",
        {"A": TEXType.VEC3, "radius": TEXType.INT, "STENCIL": TEXType.VEC4})
    tm = TypeChecker(binding_types={"A": TEXType.VEC3, "radius": TEXType.INT,
                                    "STENCIL": TEXType.VEC4}, source="").check(prog)
    fn = _cg.try_compile(prog, tm)
    if fn is None:
        r.fail("stencil-only compile", "the box-blur-only program failed to codegen at all")
        return
    src = getattr(fn, "_tex_src", "") or ""
    if "__tex_scale" in src:
        r.fail("invariant 7 leak", "'__tex_scale' appeared in a program with no pixel_args= "
               "call -- emission is no longer byte-identical for the untouched majority case")
        return
    r.ok("no pixel_args= call anywhere -> '__tex_scale' never appears in the emitted source")


def test_scalecg48_refusal_contract_unchanged_on_new_route(r: SubTestResult):
    """Red-first (c): a scale-unsafe program still REFUSES, and still does so entirely
    upstream of tier dispatch -- `require_scale_safe` is called inside `prepare()` before any
    tier is even selected, so SCALE-CG-48's new codegen branch (which only ever runs inside
    `_run_tier`, reached only by an ALREADY-ACCEPTED plan) cannot see an unsafe cook at all."""
    print("\n--- SCALE-CG-48 (c): the scale refusal contract is unchanged on the new route ---")
    unsafe_code = "@OUT = vec4(ix * 0.01, iy * 0.01, 0.0, 1.0);"
    if _tex_roi.scale_safe(unsafe_code):
        r.fail("refusal precondition", "unsafe_code was classified safe -- wrong test shape")
        return
    threw = False
    refusal_code = None
    try:
        tex_engine.prepare(unsafe_code, {}, device_mode="cpu", compile_mode="none", scale=0.5)
    except RuntimeError as e:
        threw = True
        refusal_code = getattr(getattr(e, "tex_refusal", None), "code", None)
    if not threw:
        r.fail("scale refusal", "prepare() did not raise for a scale-unsafe program at scale=0.5")
        return
    r.ok(f"prepare(..., scale=0.5) on a scale-unsafe program still raises "
         f"(refusal code={refusal_code!r}) -- unchanged by SCALE-CG-48's tier-dispatch-only edit")


def test_scalecg48_scale_verdict_query_unaffected(r: SubTestResult):
    """`tex_api.scale_verdict` reads the exact same memoized classifier `require_scale_safe`
    consults -- SCALE-CG-48 touches neither, so the two can never disagree, exactly as before."""
    print("\n--- SCALE-CG-48 (c): tex_api.scale_verdict still agrees with the real refusal ---")
    from TEX_Wrangle import tex_api
    verdict = tex_api.scale_verdict(_STENCIL_PLUS_BLUR, {"radius": 2})
    if not verdict.safe:
        r.fail("verdict agreement", "_STENCIL_PLUS_BLUR (a plain fetch stencil + gauss_blur, no "
               "hand pixel math) was classified unsafe -- verdict/refusal would now disagree")
        return
    # Must not raise -- same program, same param values, same verdict.
    _tex_roi.require_scale_safe(_STENCIL_PLUS_BLUR, 0.5, {"radius": 2})
    r.ok("scale_verdict().safe agrees with require_scale_safe() raising nothing, "
         "for the SAME program+params SCALE-CG-48's new codegen route now serves")
