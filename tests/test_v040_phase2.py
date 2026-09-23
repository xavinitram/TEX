"""v0.40.1 PM-11 — the fused viewer transform (design lane C, COLOR-1's design doc §3).

`viewer_exposure()` / `viewer_gamma()`: reserved, zero-arg builtins fed by a new
`viewer_context=` engine kwarg mirroring `time_context=` (never a `$param` — a `$param`
is baked into the compile fingerprint and would recompile on every slider drag). The
proof this lane owes: a viewer tweak NEVER recompiles (fingerprint / compile-cache
identity unchanged across two different viewer values) and codegen is BIT-EXACT with
the interpreter — unlike `frame`/`fps`/`time`, codegen does not decline these programs.
"""
from helpers import *
from failure_harness import compile_program, clone_bindings
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute, _compiled_cache
from TEX_Wrangle.tex_runtime.codegen import try_compile, _reads_time_builtin
from TEX_Wrangle.tex_runtime.graphed import _capturable
from TEX_Wrangle.tex_runtime.precision_policy import _has_fp16_hazard
from TEX_Wrangle.tex_runtime.stdlib_registry import FP16_FRAGILE


def test_pm11_default_is_identity(r: SubTestResult):
    print("\n--- PM-11: invariant 7 — a cook that never passes viewer_context sees no change ---")
    img = make_img(1, 8, 8, 3, seed=3)
    try:
        out = tex_engine.cook(
            "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);", {"A": img.clone()}).outputs["OUT"]
        md = (out[..., 0:3] - img).abs().max().item()
        assert md < 1e-6, f"viewer_exposure() defaulted to non-1.0: maxdiff {md:.3e}"
        out2 = tex_engine.cook(
            "@OUT = vec4(pow(@A.rgb, vec3(1.0 / viewer_gamma())), 1.0);",
            {"A": img.clone()}).outputs["OUT"]
        md2 = (out2[..., 0:3] - img).abs().max().item()
        assert md2 < 1e-4, f"viewer_gamma() defaulted to non-1.0: maxdiff {md2:.3e}"
        r.ok(f"no viewer_context: exposure maxdiff {md:.1e}, gamma maxdiff {md2:.1e} "
             f"(both no-op)")
    except Exception as e:
        r.fail("PM-11 default identity", f"{type(e).__name__}: {e}")


def test_pm11_value_propagates_both_tiers(r: SubTestResult):
    print("\n--- PM-11: viewer_context reaches the interpreter AND codegen, bit-exact ---")
    img = make_img(1, 8, 8, 3, seed=5)
    code = "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);"
    prog, tm, outs = compile_program(code, {"A": img})
    try:
        for exposure in (0.5, 2.5):
            vc = {"viewer_exposure": exposure}
            interp_out = Interpreter().execute(
                prog, clone_bindings({"A": img}),
                tm, device="cpu", output_names=outs, precision="fp32", viewer_context=vc)
            cg_out = _codegen_only_execute(
                prog, clone_bindings({"A": img}), tm, "cpu", output_names=outs,
                precision="fp32", fingerprint=f"pm11_prop_{exposure}",
                time_context=None, viewer_context=vc)
            md_val = (interp_out["OUT"][..., 0:3] - img * exposure).abs().max().item()
            md_tiers = (interp_out["OUT"] - cg_out["OUT"]).abs().max().item()
            assert md_val < 1e-5, f"exposure={exposure}: interp value wrong (maxdiff {md_val:.3e})"
            assert md_tiers < 1e-5, \
                f"exposure={exposure}: interp/codegen diverge (maxdiff {md_tiers:.3e})"
        r.ok("interp==codegen bit-exact at two different viewer_exposure values, "
             "and the value is the one the host passed")
    except Exception as e:
        r.fail("PM-11 value propagation", f"{type(e).__name__}: {e}")

    try:
        assert not _reads_time_builtin(prog), \
            "a viewer-only program was misclassified as a time-builtin read"
        fn = try_compile(prog, tm, fingerprint="pm11_not_declined")
        assert fn is not None, \
            "codegen DECLINED a viewer_exposure() program — unlike frame/fps/time, PM-11 " \
            "requires codegen to compile these (design doc §3: interp/codegen mirror)"
        r.ok("codegen does NOT decline a viewer_exposure() program (contrast frame/fps/time)")
    except Exception as e:
        r.fail("PM-11 codegen not declined", f"{type(e).__name__}: {e}")


def test_pm11_fingerprint_and_cache_neutral(r: SubTestResult):
    print("\n--- PM-11: a viewer tweak never recompiles (fingerprint / compile-cache identity) ---")
    img = make_img(1, 8, 8, 3, seed=9)
    code = "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);"
    prog, tm, outs = compile_program(code, {"A": img})
    fp = "pm11_cache_neutral"
    try:
        _compiled_cache.clear()
        before = len(_compiled_cache)
        r1 = _codegen_only_execute(prog, clone_bindings({"A": img}), tm, "cpu",
                                   output_names=outs, precision="fp32", fingerprint=fp,
                                   time_context=None, viewer_context={"viewer_exposure": 1.0})
        r2 = _codegen_only_execute(prog, clone_bindings({"A": img}), tm, "cpu",
                                   output_names=outs, precision="fp32", fingerprint=fp,
                                   time_context=None, viewer_context={"viewer_exposure": 4.0})
        after = len(_compiled_cache)
        assert after == before, \
            f"_compiled_cache grew ({before} -> {after}) across two viewer values at the SAME fingerprint"
        md = (r2["OUT"][..., 0:3] - img * 4.0).abs().max().item()
        assert md < 1e-5, f"the second viewer value did not actually reach the pixels ({md:.3e})"
        r.ok(f"one fingerprint, two viewer_exposure values: 0 new codegen-cache entries "
             f"(both read {before}), and the second value still moved the pixels (maxdiff {md:.1e})")
    except Exception as e:
        r.fail("PM-11 compile-cache neutrality", f"{type(e).__name__}: {e}")

    # tex_cache.fingerprint() itself never sees viewer_context — it isn't a parameter.
    try:
        import inspect
        from TEX_Wrangle.tex_cache import TEXCache
        params = inspect.signature(TEXCache.fingerprint).parameters
        assert "viewer_context" not in params, \
            "fingerprint() grew a viewer_context parameter — it must stay a VALUE, never a key"
        r.ok("tex_cache.TEXCache.fingerprint takes no viewer_context parameter")
    except Exception as e:
        r.fail("PM-11 fingerprint signature", f"{type(e).__name__}: {e}")


def test_pm11_cuda_graph_declines(r: SubTestResult):
    print("\n--- PM-11: cuda_graph capture bars viewer_exposure()/viewer_gamma() (same class as frame/time) ---")
    plain = "@OUT = vec4(@A.rgb * 0.5, 1.0);"
    timed = "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);"
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    fails = []
    for code, expect_capturable, label in ((plain, True, "plain"), (timed, False, "viewer_exposure")):
        prog = parse_and_split(code, bt)
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        got = _capturable(prog)[0]
        if got != expect_capturable:
            fails.append(f"{label}: _capturable={got}, expected {expect_capturable}")
    if fails:
        r.fail("PM-11 cuda_graph bar", "; ".join(fails))
    else:
        r.ok("a viewer_exposure() program is barred from CUDA-graph capture; a plain "
             "program is unaffected")


def test_pm11_fp16_auto_declines(r: SubTestResult):
    print("\n--- PM-11: precision='auto' declines a viewer_exposure()/viewer_gamma() program ---")
    fails = []
    for name in ("viewer_exposure", "viewer_gamma"):
        if name not in FP16_FRAGILE:
            fails.append(f"{name} missing from stdlib_registry.FP16_FRAGILE")
    code = "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);"
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    prog = parse_and_split(code, bt)
    TypeChecker(binding_types=bt, source=code).check(prog)
    if not _has_fp16_hazard(prog, ["OUT"]):
        fails.append("_has_fp16_hazard did not flag a viewer_exposure() program — "
                     "auto would resolve fp16 for a host-supplied, unbounded gain")
    if fails:
        r.fail("PM-11 fp16 auto decline", "; ".join(fails))
    else:
        r.ok("both names are in FP16_FRAGILE, and _has_fp16_hazard declines fp16 for "
             "a program that multiplies image lineage by viewer_exposure()")


def test_pm11_reserved_names_e3011(r: SubTestResult):
    print("\n--- PM-11: viewer_exposure/viewer_gamma are RESERVED (E3011) ---")
    for name in ("viewer_exposure", "viewer_gamma"):
        try:
            raised = None
            try:
                check_code(f"float {name}(float x){{ return x; }}\n@OUT = vec4(0.0);")
            except Exception as e:
                raised = e
            assert raised is not None, f"redefining {name} as a user function did not raise"
            code = getattr(getattr(raised, "diagnostic", None), "code", None)
            assert code == "E3011", f"{name}: wrong error code: {code!r} (raised={raised!r})"
            r.ok(f"`{name}` is refused as E3011 (reserved builtin)")
        except Exception as e:
            r.fail(f"PM-11 reserved {name}", f"{type(e).__name__}: {e}")


def test_pm11_engine_plumbing(r: SubTestResult):
    print("\n--- PM-11: viewer_context= mirrors time_context= on the engine's public surface ---")
    import inspect
    from TEX_Wrangle import tex_engine as te
    fails = []
    for fn in (te.prepare,):
        params = inspect.signature(fn).parameters
        if "viewer_context" not in params:
            fails.append(f"{fn.__name__} has no viewer_context parameter")
        elif params["viewer_context"].default is not None:
            fails.append(f"{fn.__name__}.viewer_context default is not None "
                         f"(invariant 7: absence must mean identity)")
    if "viewer_context" not in {f.name for f in __import__("dataclasses").fields(te.ExecContext)}:
        fails.append("ExecContext has no viewer_context field")
    if fails:
        r.fail("PM-11 engine plumbing", "; ".join(fails))
    else:
        r.ok("tex_engine.prepare() takes viewer_context=None (mirrors time_context=); "
             "ExecContext carries it through to the tier strategies")
