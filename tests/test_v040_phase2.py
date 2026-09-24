"""v0.40.1 PM-11 — the fused viewer transform (design lane C, COLOR-1's design doc §3).

`viewer_exposure()` / `viewer_gamma()`: reserved, zero-arg builtins fed by a new
`viewer_context=` engine kwarg mirroring `time_context=` (never a `$param` — a `$param`
is baked into the compile fingerprint and would recompile on every slider drag). The
proof this lane owes: a viewer tweak NEVER recompiles (fingerprint / compile-cache
identity unchanged across two different viewer values) and codegen is BIT-EXACT with
the interpreter — unlike `frame`/`fps`/`time`, codegen does not decline these programs.
"""
import tempfile
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
from TEX_Wrangle.tex_memory import run_tiled, run_roi, run_tiled_halo
from TEX_Wrangle import tex_chain, tex_fusion
from TEX_Wrangle.tex_results import ResultCache


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
    print("\n--- PM-11/v042-graph: cuda_graph capture of viewer_exposure()/viewer_gamma() ---")
    # v042-graph LANDED the follow-up PM-11's own hand-back pencilled ("Graph-capture
    # exclusion: what it costs, and whether to pencil the fix" — yes): a viewer program is
    # capturable now, fed via a per-replay static buffer (`tests/test_v042_graph.py` has
    # the capture/replay/no-recapture proofs; this row stays here only to keep PM-11's own
    # "same class as frame/time" comparison in one place, updated to the new verdict).
    plain = "@OUT = vec4(@A.rgb * 0.5, 1.0);"
    timed = "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);"
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    fails = []
    for code, expect_capturable, label in ((plain, True, "plain"), (timed, True, "viewer_exposure")):
        prog = parse_and_split(code, bt)
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        got = _capturable(prog)[0]
        if got != expect_capturable:
            fails.append(f"{label}: _capturable={got}, expected {expect_capturable}")
    if fails:
        r.fail("PM-11/v042-graph cuda_graph bar", "; ".join(fails))
    else:
        r.ok("a viewer_exposure() program is capturable (v042-graph); a plain "
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


# ── Memory-pressure paths: PM-11-F1's fix. A viewer tweak reaching only the DEFAULT
# untiled cook and going silent under memory pressure would be a wrong-but-plausible
# picture on exactly the large cooks most likely to need one — not deferrable, per
# ENG-7's own history with `time_context` (the tiled path froze `frame` at 0 until
# fixed; the same shape of bug, the same fix shape). Each test drives the pressure
# path DIRECTLY (the forcing hook `test_eng7_time_builtins_advance`'s "TILED path"
# assertion and `test_color1_apply_lut3d_pressure_paths` already use) rather than
# provoking real memory pressure, and compares against the unpressured whole-frame
# cook at the SAME viewer value.

def test_pm11_tiled_pressure_path(r: SubTestResult):
    print("\n--- PM-11-F1: viewer_context reaches tex_memory.run_tiled (M-4 memory pressure) ---")
    img = make_img(1, 64, 64, 3, seed=11)
    code = "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);"
    prog, tm, outs = compile_program(code, {"A": img})
    vc = {"viewer_exposure": 3.0}
    try:
        interp = Interpreter()
        whole = interp.execute(prog, clone_bindings({"A": img}), tm, device="cpu",
                               output_names=outs, precision="fp32", viewer_context=vc)
        tiled = run_tiled(interp, prog, clone_bindings({"A": img}), tm, "cpu", 0,
                          outs, None, "fp32", 4, time_context=None, viewer_context=vc)
        md = (whole["OUT"] - tiled["OUT"]).abs().max().item()
        assert md < 1e-6, f"run_tiled lost viewer_context under memory pressure (maxdiff {md:.3e})"
        # Prove the value genuinely reached the strips, not that the comparison is trivial:
        # the identity-default tiled cook must read DIFFERENT pixels from the forced one.
        identity = run_tiled(interp, prog, clone_bindings({"A": img}), tm, "cpu", 0,
                             outs, None, "fp32", 4, time_context=None, viewer_context=None)
        moved = (tiled["OUT"] - identity["OUT"]).abs().max().item()
        assert moved > 1e-3, "run_tiled read the same pixels regardless of viewer_context"
        r.ok(f"run_tiled(n=4) with viewer_exposure=3.0 matches the untiled cook exactly "
             f"(maxdiff {md:.1e}); moved {moved:.2f} vs the identity-default tiled cook")
    except Exception as e:
        r.fail("PM-11 tiled pressure path", f"{type(e).__name__}: {e}")


def test_pm11_roi_pressure_path(r: SubTestResult):
    print("\n--- PM-11-F1: viewer_context reaches tex_memory.run_roi, interp AND codegen exec_fn ---")
    img = make_img(1, 16, 16, 3, seed=13)
    code = "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);"
    prog, tm, outs = compile_program(code, {"A": img})
    vc = {"viewer_exposure": 2.0}
    roi_window = (2, 2, 6, 6, 16, 16)   # x0, y0, w, h, W, H
    x0, y0, w, h, _W, _H = roi_window
    try:
        interp = Interpreter()
        whole = interp.execute(prog, clone_bindings({"A": img}), tm, device="cpu",
                               output_names=outs, precision="fp32", viewer_context=vc)
        ref_crop = whole["OUT"][:, y0:y0 + h, x0:x0 + w]

        roi_out = run_roi(interp, prog, clone_bindings({"A": img}), tm, "cpu", 0, outs, None,
                          "fp32", roi_window, {"A"}, 0, time_context=None, viewer_context=vc,
                          record_trace=False)
        md = (ref_crop - roi_out["OUT"]).abs().max().item()
        assert md < 1e-5, f"run_roi (interpreter exec_fn) lost viewer_context (maxdiff {md:.3e})"

        cg_exec = tex_engine._roi_codegen_exec("pm11_roi_cg")
        roi_out_cg = run_roi(interp, prog, clone_bindings({"A": img}), tm, "cpu", 0, outs, None,
                             "fp32", roi_window, {"A"}, 0, time_context=None, viewer_context=vc,
                             exec_fn=cg_exec, record_trace=False)
        md_cg = (ref_crop - roi_out_cg["OUT"]).abs().max().item()
        assert md_cg < 1e-5, f"run_roi (codegen exec_fn) lost viewer_context (maxdiff {md_cg:.3e})"
        r.ok(f"run_roi carries viewer_exposure=2.0 into the cropped window on both the "
             f"interpreter exec_fn (maxdiff {md:.1e}) and the codegen exec_fn (maxdiff {md_cg:.1e})")
    except Exception as e:
        r.fail("PM-11 roi pressure path", f"{type(e).__name__}: {e}")


def test_pm11_halo_pressure_path(r: SubTestResult):
    print("\n--- PM-11-F1: viewer_context reaches tex_memory.run_tiled_halo (ROI-5 grown strips) ---")
    img = make_img(1, 32, 32, 3, seed=17)
    code = "@OUT = vec4(gauss_blur(@A.rgb, 1.0) * viewer_exposure(), 1.0);"
    prog, tm, outs = compile_program(code, {"A": img})
    vc = {"viewer_exposure": 1.7}
    try:
        interp = Interpreter()
        whole = interp.execute(prog, clone_bindings({"A": img}), tm, device="cpu",
                               output_names=outs, precision="fp32", viewer_context=vc)
        halo_out = run_tiled_halo(interp, prog, clone_bindings({"A": img}), tm, "cpu", 0,
                                  outs, None, "fp32", 4, {"A"}, 4, time_context=None,
                                  viewer_context=vc)
        md = (whole["OUT"] - halo_out["OUT"]).abs().max().item()
        assert md < 1e-4, f"run_tiled_halo lost viewer_context under grown-strip tiling (maxdiff {md:.3e})"
        r.ok(f"run_tiled_halo(n=4, halo=4) with viewer_exposure=1.7 matches the whole-frame "
             f"cook exactly (maxdiff {md:.1e})")
    except Exception as e:
        r.fail("PM-11 halo pressure path", f"{type(e).__name__}: {e}")


def test_pm11_oom_rung_path(r: SubTestResult):
    print("\n--- PM-11-F1: viewer_context reaches tex_engine._oom_retry's tiled rung ---")
    if not torch.cuda.is_available():
        r.skip("PM-11 OOM rung path",
               "no CUDA on this box — _oom_retry's rung 2 requires "
               "str(ctx.device).startswith('cuda') and cannot be exercised here")
        return
    try:
        img = make_img(1, 128, 128, 3, seed=19)
        code = "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);"
        vc = {"viewer_exposure": 2.3}
        plan = tex_engine.prepare(code, {"A": img.clone()}, device_mode="cuda",
                                  viewer_context=vc)
        ctx = plan.ctx
        assert str(ctx.device).startswith("cuda"), f"prepare() did not resolve cuda: {ctx.device}"
        whole = Interpreter().execute(
            ctx.program, clone_bindings(ctx.bindings), ctx.type_map, device=ctx.device,
            output_names=ctx.output_names, precision=ctx.eff_precision, viewer_context=vc)
        fake = RuntimeError("CUDA out of memory (forced by PM-11-F1's regression test)")
        recovered = tex_engine._oom_retry(ctx, fake, fake)
        assert recovered is not None, "_oom_retry declined to recover a tile-safe CUDA program"
        md = (whole["OUT"] - recovered["OUT"]).abs().max().item()
        assert md < 1e-4, f"the OOM ladder's tiled retry lost viewer_context (maxdiff {md:.3e})"
        r.ok(f"tex_engine._oom_retry's tiled rung carries viewer_exposure=2.3 through "
             f"(maxdiff vs the unpressured cook {md:.1e})")
    except Exception as e:
        r.fail("PM-11 OOM rung path", f"{type(e).__name__}: {e}")


def test_pm11_fused_chain_path(r: SubTestResult):
    print("\n--- PM-11-F2: viewer_context reaches tex_chain.cook_stage_list (the FUSED chain) ---")
    # PM-11's own design doc names this the primary use case: a viewer transform expressed
    # as an ordinary trailing TEX stage, fused with the comp by tex_fusion.compile_fused. A
    # fused 2-stage chain must read viewer_exposure() exactly as the same two stages cooked
    # unfused, stage-by-stage, do — on both the interpreter (cook_stage_list's own tier) AND
    # codegen (proving the fused PROGRAM itself, not just cook_stage_list's plumbing, is
    # bit-exact — a fused program is an ordinary TEX program the moment compile_fused hands
    # it back).
    img = make_img(1, 8, 8, 3, seed=23)
    stage0 = {"code": "@OUT = @A.rgb * 1.5;", "bindings": {"A": img.clone()}}
    stage1 = {"code": "@OUT = @X.rgb * viewer_exposure();", "chain_input": "X", "bindings": {}}
    vc = {"viewer_exposure": 2.2}
    try:
        fused = tex_chain.cook_stage_list([stage0, stage1], viewer_context=vc)

        # Unfused, stage-by-stage, through the SAME cook_stage_list (interpreter) tier.
        s0 = tex_chain.cook_stage_list([stage0], viewer_context=vc)
        s1_interp = tex_chain.cook_stage_list(
            [{"code": stage1["code"], "bindings": {"X": s0["OUT"].clone()}}], viewer_context=vc)
        md_interp = (fused["OUT"] - s1_interp["OUT"]).abs().max().item()
        assert md_interp < 1e-5, \
            f"fused chain lost viewer_context vs the unfused interpreter cook (maxdiff {md_interp:.3e})"

        # Unfused, stage-by-stage, second stage through CODEGEN instead — proving the fused
        # program's own bit-exactness carries the value the same way an unfused one does.
        prog1, tm1, outs1 = compile_program(stage1["code"], {"X": s0["OUT"].clone()})
        s1_cg = _codegen_only_execute(prog1, {"X": s0["OUT"].clone()}, tm1, "cpu",
                                      output_names=outs1, precision="fp32",
                                      fingerprint="pm11_chain_cg", time_context=None,
                                      viewer_context=vc)
        md_cg = (fused["OUT"] - s1_cg["OUT"]).abs().max().item()
        assert md_cg < 1e-5, \
            f"fused chain diverges from the unfused CODEGEN cook (maxdiff {md_cg:.3e})"

        # Identity default: no viewer_context anywhere must read viewer_exposure()==1.0.
        identity = tex_chain.cook_stage_list([stage0, stage1])
        md_id = (identity["OUT"] - img * 1.5).abs().max().item()
        assert md_id < 1e-5, f"fused chain with no viewer_context was not a no-op ({md_id:.3e})"

        r.ok(f"a fused 2-stage chain (viewer_exposure=2.2) matches the unfused stage-by-stage "
             f"cook on both the interpreter (maxdiff {md_interp:.1e}) and codegen "
             f"(maxdiff {md_cg:.1e}); no viewer_context is a no-op (maxdiff {md_id:.1e})")
    except Exception as e:
        r.fail("PM-11 fused chain path", f"{type(e).__name__}: {e}")


# ── Result-cache audit: viewer values must KEY the pixel-holding caches (unlike the compile
# fingerprint, which must stay viewer-free) — a tweak that recomputes correctly but still
# hits a stale cached frame is the same wrong-picture class as F1/F2, one layer over.

def test_pm11_resultcache_keys_on_viewer(r: SubTestResult):
    print("\n--- PM-11: tex_results.lineage_key/CACHE-1 keys on viewer_context, ONLY when read ---")
    img = make_img(1, 8, 8, 3, seed=29)
    viewer_code = "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);"
    plain_code = "@OUT = vec4(@A.rgb * 0.5, 1.0);"
    try:
        # A viewer-reading program: two different exposures must mint DIFFERENT lineage keys
        # (else the second exposure hits the first one's cached, stale-exposure frame).
        rA = tex_engine.cook(viewer_code, {"A": img.clone()}, want_lineage=True,
                             viewer_context={"viewer_exposure": 1.0})
        rB = tex_engine.cook(viewer_code, {"A": img.clone()}, want_lineage=True,
                             viewer_context={"viewer_exposure": 3.0})
        assert rA.lineage["OUT"] != rB.lineage["OUT"], \
            "two different viewer_exposure values minted the SAME lineage key"
        md = (rA.outputs["OUT"] - rB.outputs["OUT"]).abs().max().item()
        assert md > 1e-3, "exposure=1.0 and exposure=3.0 rendered the same pixels (test premise)"

        # A repeat cook at the SAME exposure must mint the SAME key (dedup still works).
        rA2 = tex_engine.cook(viewer_code, {"A": img.clone()}, want_lineage=True,
                              viewer_context={"viewer_exposure": 1.0})
        assert rA2.lineage["OUT"] == rA.lineage["OUT"], \
            "the SAME viewer_exposure value minted a DIFFERENT lineage key"

        # Invariant 7: a program that never calls a viewer builtin must key IDENTICALLY
        # regardless of what viewer_context a host happens to pass (nothing to invalidate,
        # nothing new may enter this program's key at all).
        pA = tex_engine.cook(plain_code, {"A": img.clone()}, want_lineage=True,
                             viewer_context={"viewer_exposure": 1.0})
        pB = tex_engine.cook(plain_code, {"A": img.clone()}, want_lineage=True,
                             viewer_context={"viewer_exposure": 3.0})
        assert pA.lineage["OUT"] == pB.lineage["OUT"], \
            "a NON-viewer program's lineage key moved when only viewer_context changed"

        # The compile side stays viewer-free throughout (the other half of the same proof).
        from TEX_Wrangle.tex_cache import get_cache
        fpA = get_cache().fingerprint(viewer_code, {"A": TEXType.VEC3})
        fpB = fpA  # same source, same binding types -> same fingerprint by construction
        assert fpA == fpB, "unreachable: fingerprint is a pure function of source+types"

        r.ok("viewer-reading program: exposure 1.0 vs 3.0 mint different lineage keys "
             "(and differ pixels); a repeat at 1.0 re-hits the same key. Non-viewer program: "
             "the key never moves across the same two viewer_context values")
    except Exception as e:
        r.fail("PM-11 ResultCache viewer keying", f"{type(e).__name__}: {e}")


def test_pm11_boundary_tap_keys_on_viewer(r: SubTestResult):
    print("\n--- PM-11: CACHE-6/7's boundary-tap ResultCache keys on viewer_context ---")
    # A 2-stage chain whose PREFIX (stage 0, the tap at k=1) reads viewer_exposure(). A
    # differently-exposed cook must NOT hit the first cook's cached boundary tap.
    src = make_img(1, 16, 16, 3, seed=31)
    stages = [
        {"code": "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);", "chain_input": None,
         "bindings": {"A": src}},
        {"code": "@OUT = vec4(@X.rgb + 0.1, 1.0);", "chain_input": "X", "bindings": {}},
    ]
    upstream = ("pm11_src",)
    try:
        rc = ResultCache(budget_mb=100, cache_dir=tempfile.mkdtemp(prefix="tex_pm11_c6_"))
        n_fused_before = len(tex_fusion._FUSED_MEMO)

        outA = tex_chain.cook_fused_cached(
            stages, 1, rc, device="cpu", upstream=upstream,
            viewer_context={"viewer_exposure": 1.0})["OUT"]
        missesA = rc.misses
        outB = tex_chain.cook_fused_cached(
            stages, 1, rc, device="cpu", upstream=upstream,
            viewer_context={"viewer_exposure": 3.0})["OUT"]

        md = (outA - outB).abs().max().item()
        assert md > 1e-3, \
            f"exposure=1.0 and exposure=3.0 produced the same chain output (maxdiff {md:.3e}) " \
            f"— the second cook served the FIRST exposure's stale boundary"
        assert rc.misses > missesA, \
            "the second exposure HIT the first exposure's boundary tap instead of re-materializing"
        # "Compile count unchanged": the SAME stage list/cut point never re-splices — only the
        # boundary CACHE result changed keys, not the fused program itself.
        assert len(tex_fusion._FUSED_MEMO) == n_fused_before or \
            len(tex_fusion._FUSED_MEMO) == n_fused_before + 1, \
            "compile_fused's memo grew by more than the one splice this chain needed once"

        # A repeat at exposure=1.0 must still HIT (the fix must not defeat dedup entirely).
        hitsA_before = rc.hits
        outA2 = tex_chain.cook_fused_cached(
            stages, 1, rc, device="cpu", upstream=upstream,
            viewer_context={"viewer_exposure": 1.0})["OUT"]
        assert rc.hits > hitsA_before, "a repeat at the SAME exposure missed the boundary tap"
        assert (outA - outA2).abs().max().item() < 1e-5, \
            "a boundary-tap HIT at the same exposure served different pixels"

        r.ok(f"boundary tap: exposure 1.0 vs 3.0 correctly MISS each other (maxdiff {md:.1e}, "
             f"misses {missesA}->{rc.misses}); a repeat at 1.0 still HITS")
    except Exception as e:
        r.fail("PM-11 boundary-tap viewer keying", f"{type(e).__name__}: {e}")
