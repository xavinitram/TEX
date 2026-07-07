"""
v0.15.0 Phase 5 regression tests — CC-2 measured auto-tier + Q-3 fusion widening.
"""
from helpers import *
from TEX_Wrangle.tex_runtime import autotier as AT
import TEX_Wrangle.tex_fusion as _FUS
from TEX_Wrangle.tex_node import _infer_binding_type


def _run_fused(stages):
    prog, tm, refs, asg, params, used, merged = _FUS.compile_fused(
        stages, _infer_binding_type)
    return Interpreter().execute(prog, dict(merged), tm, device="cpu",
                                 output_names=sorted(asg.keys()),
                                 used_builtins=used)


def _run_unfused(stages):
    """Ground truth: run each stage as a standalone program, threading outputs
    along the declared (chain_input | chain_inputs) edges."""
    outs = {}
    n = len(stages)
    for i, st in enumerate(stages):
        ext = dict(st.get("bindings") or {})
        ci = st.get("chain_inputs")
        if ci is None:
            c = st.get("chain_input")
            ci = {c: [i - 1, "OUT"]} if c is not None else {}
        binds = dict(ext)
        for b, (s, o) in ci.items():
            binds[b] = outs[(s, o)]
        bt = {k: _infer_binding_type(v) for k, v in binds.items()}
        prog = Parser(Lexer(st["code"]).tokenize(), source=st["code"]).parse()
        checker = TypeChecker(binding_types=bt, source=st["code"])
        tm = checker.check(prog)
        res = Interpreter().execute(prog, binds, tm, device="cpu",
                                    output_names=sorted(checker.assigned_bindings.keys()),
                                    used_builtins=_collect_identifiers(prog))
        for name, val in res.items():
            outs[(i, name)] = val
    return outs


def test_cc2_state_machine(r: SubTestResult):
    print("\n--- CC-2: auto-tier decision state machine ---")

    # res-bucket keying: different resolutions key apart, same bucket coalesces.
    try:
        AT.reset()
        k512 = AT.make_key("fp", "cuda", "fp32", (1, 512, 512))
        k513 = AT.make_key("fp", "cuda", "fp32", (1, 512, 513))  # ~same bucket
        k2048 = AT.make_key("fp", "cuda", "fp32", (1, 2048, 2048))
        assert k512 != k2048, "distinct resolutions must key apart"
        assert k512[3] == (512 * 512).bit_length()
        r.ok("res-bucket key = (H*W).bit_length()")
    except Exception as e:
        r.fail("res-bucket keying", str(e))

    # Commit path: compiled clearly beats the baseline → COMMITTED after trial.
    try:
        AT.reset()
        k = AT.make_key("commit", "cpu", "fp32", (1, 256, 256))
        assert AT.verdict(k) == AT.MEASURING
        for _ in range(2):
            AT.record_interp(k, 10.0)
            assert not AT.should_submit_compile(k)  # <3 samples
        AT.record_interp(k, 10.0)
        assert AT.should_submit_compile(k), "should submit after 3 samples"
        AT.mark_submitted(k)
        assert AT.verdict(k) == AT.COMPILING
        assert not AT.should_submit_compile(k)  # already submitted
        AT.mark_ready(k)
        assert AT.verdict(k) == AT.TRIAL
        AT.record_trial(k, 4.0)  # 4 < 0.9*10 → win
        assert AT.verdict(k) == AT.COMMITTED
        r.ok("MEASURING→COMPILING→TRIAL→COMMITTED on a measured win")
    except Exception as e:
        r.fail("commit path", str(e))

    # Reject path: compiled loses (warm-inductor-slower-than-interp case).
    try:
        AT.reset()
        k = AT.make_key("reject", "cpu", "fp32", (1, 64, 64))
        for _ in range(3):
            AT.record_interp(k, 5.0)
        AT.mark_submitted(k); AT.mark_ready(k)
        AT.record_trial(k, 6.0)  # slower than baseline → reject
        assert AT.verdict(k) == AT.REJECTED, AT.verdict(k)
        r.ok("compiled slower than baseline → REJECTED")
    except Exception as e:
        r.fail("reject path", str(e))

    # A marginal win (>0.9x) does NOT commit — the 10% margin is enforced.
    try:
        AT.reset()
        k = AT.make_key("margin", "cpu", "fp32", (1, 64, 64))
        for _ in range(3):
            AT.record_interp(k, 10.0)
        AT.mark_submitted(k); AT.mark_ready(k)
        AT.record_trial(k, 9.5)  # only 5% faster → not enough
        assert AT.verdict(k) == AT.REJECTED
        r.ok("marginal <10% win does not commit")
    except Exception as e:
        r.fail("commit margin", str(e))

    # Trial crash (compiled_ms=None) → REJECTED, never a hard error.
    try:
        AT.reset()
        k = AT.make_key("crash", "cpu", "fp32", (1, 64, 64))
        for _ in range(3):
            AT.record_interp(k, 8.0)
        AT.mark_submitted(k); AT.mark_ready(k)
        AT.record_trial(k, None)
        assert AT.verdict(k) == AT.REJECTED
        r.ok("trial crash → REJECTED (not blacklist-forever)")
    except Exception as e:
        r.fail("trial crash handling", str(e))

    # Persistence round-trips through a reversible (JSON-list) key: a committed
    # verdict survives a simulated restart (no string/tuple staging table).
    try:
        import os, tempfile
        AT.reset()
        saved_path = AT._persist_path_cache
        AT._persist_path_cache = os.path.join(tempfile.gettempdir(),
                                              "tex_autotier_roundtrip_test.json")
        AT._loaded = False
        try:
            k = AT.make_key("persist", "cuda", "fp32", (1, 512, 512))
            for _ in range(3):
                AT.record_interp(k, 10.0)
            AT.mark_submitted(k); AT.mark_ready(k)
            AT.record_trial(k, 4.0)  # commit → persists to disk
            AT.reset(); AT._loaded = False  # simulate restart
            AT.seed_from_disk(k)
            assert AT.verdict(k) == AT.COMMITTED, "verdict not restored from disk"
        finally:
            try:
                os.remove(AT._persist_path_cache)
            except OSError:
                pass
            AT._persist_path_cache = saved_path
            AT._loaded = False
        r.ok("terminal verdict persists + restores across restart")
    except Exception as e:
        r.fail("verdict persistence round-trip", str(e))


def test_cc2_no_stall_sim(r: SubTestResult):
    print("\n--- CC-2: 30-cook no-stall simulation ---")
    # Model the roadmap's batch/video sequence: the first ~4 cooks stay on the
    # baseline (no 28s stall), then a trial, then steady-state on the winner.
    try:
        AT.reset()
        k = AT.make_key("seq", "cpu", "fp32", (1, 512, 512))
        baseline_ms, compiled_ms = 12.0, 5.0
        routes = []
        compiled_ready = False
        for i in range(30):
            v = AT.verdict(k)
            if v == AT.COMMITTED:
                routes.append("compiled")
            elif v == AT.REJECTED:
                routes.append("codegen")
            elif v == AT.TRIAL:
                routes.append("trial")
                AT.record_trial(k, compiled_ms)
            else:  # MEASURING / COMPILING → baseline
                routes.append("baseline")
                AT.record_interp(k, baseline_ms)
                if v == AT.MEASURING and AT.should_submit_compile(k):
                    AT.mark_submitted(k)          # submit (non-blocking)
                elif v == AT.COMPILING:
                    if not compiled_ready:        # background compile finishes ~2 cooks later
                        compiled_ready = True
                    else:
                        AT.mark_ready(k)
        # No cook was ever a blocking compile; steady state is compiled.
        assert routes[-1] == "compiled", routes[-8:]
        assert routes.count("trial") == 1, "exactly one trial cook"
        assert routes[:3] == ["baseline"] * 3, "first cooks stay on baseline"
        n_compiled = routes.count("compiled")
        assert n_compiled >= 20, f"steady state not reached ({n_compiled} compiled)"
        r.ok(f"30-cook seq: 3 baseline → 1 trial → {n_compiled} compiled, no stall")
    except Exception as e:
        r.fail("30-cook no-stall sim", str(e))


def test_cc2_end_to_end(r: SubTestResult):
    print("\n--- CC-2: auto-tier end-to-end (correctness) ---")
    from TEX_Wrangle.tex_runtime.compiled import run_auto
    import TEX_Wrangle.tex_cache as tc

    # run_auto must produce interpreter-correct output on every cook regardless
    # of tier state (baseline / trial / committed / rejected).
    try:
        AT.reset()
        code = "vec3 c=@A.rgb; c = c*1.3 - 0.1; c = clamp(c, 0.0, 1.0); @OUT=vec4(c,1.0);"
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        used = _collect_identifiers(prog)
        torch.manual_seed(3)
        img = torch.rand(1, 96, 96, 3)
        ref = Interpreter().execute(prog, {"A": img}, tm, device="cpu",
                                    output_names=["OUT"])["OUT"]
        fp = tc.get_cache().fingerprint(code, bt)
        ok = True
        for _ in range(8):  # exercise several tier transitions
            out = run_auto(prog, {"A": img}, tm, "cpu", fp,
                           output_names=["OUT"], used_builtins=used)
            t = out["OUT"] if isinstance(out, dict) else out
            if (t - ref).abs().max().item() >= 1e-4:
                ok = False
                break
        assert ok, "run_auto output diverged from interpreter"
        r.ok("run_auto output interpreter-correct across tier transitions")
    except Exception as e:
        r.fail("run_auto end-to-end correctness", str(e))


def test_q3_fusion_widening(r: SubTestResult):
    print("\n--- Q-3: fusion coverage widening (DAG / multi-@OUT / tap) ---")
    torch.manual_seed(7)
    img = torch.rand(1, 8, 8, 3)

    # Backward-compat: a plain linear chain still fuses bit-exactly.
    try:
        _FUS.clear_fused_cache() if hasattr(_FUS, "clear_fused_cache") else None
        stages = [
            {"code": "@OUT = @A * 0.5 + 0.1;", "chain_input": None, "bindings": {"A": img}},
            {"code": "@OUT = @X * 2.0;", "chain_input": "X", "bindings": {}},
        ]
        f = _run_fused(stages)["OUT"]
        u = _run_unfused(stages)[(1, "OUT")]
        assert (f - u).abs().max().item() < 1e-6
        r.ok("linear chain bit-exact (back-compat)")
    except Exception as e:
        r.fail("linear chain back-compat", str(e))

    # (a) DAG diamond: stage 0 fans out to stages 1 & 2; terminal joins both.
    try:
        stages = [
            {"code": "@OUT = @A * 0.5 + 0.1;", "bindings": {"A": img}},
            {"code": "@OUT = @X * 2.0;", "chain_inputs": {"X": [0, "OUT"]}, "bindings": {}},
            {"code": "@OUT = @Y + 0.2;", "chain_inputs": {"Y": [0, "OUT"]}, "bindings": {}},
            {"code": "@OUT = @P * @Q;", "chain_inputs": {"P": [1, "OUT"], "Q": [2, "OUT"]},
             "bindings": {}},
        ]
        f = _run_fused(stages)["OUT"]
        u = _run_unfused(stages)[(3, "OUT")]
        assert (f - u).abs().max().item() < 1e-6, "diamond DAG diverged"
        r.ok("DAG diamond (fan-out + join) bit-exact vs unfused")
    except Exception as e:
        r.fail("DAG diamond fusion", str(e))

    # (b) multi-@OUT: stage 0 exports @OUT + @extra; terminal consumes both.
    try:
        stages = [
            {"code": "@OUT = @A * 0.5; @extra = @A * 2.0;", "exports": ["extra"],
             "bindings": {"A": img}},
            {"code": "@OUT = @P + @Q * 0.25;",
             "chain_inputs": {"P": [0, "OUT"], "Q": [0, "extra"]}, "bindings": {}},
        ]
        f = _run_fused(stages)["OUT"]
        u = _run_unfused(stages)[(1, "OUT")]
        assert (f - u).abs().max().item() < 1e-6, "multi-@OUT diverged"
        r.ok("multi-@OUT export (2 outputs from one stage) bit-exact")
    except Exception as e:
        r.fail("multi-@OUT fusion", str(e))

    # (c) tap: an intermediate marked tap is exposed as @_tap_s0 without breaking
    # the chain; equals that stage's standalone @OUT.
    try:
        stages = [
            {"code": "@OUT = @A * 0.5;", "tap": True, "bindings": {"A": img}},
            {"code": "@OUT = @X + 0.1;", "chain_input": "X", "bindings": {}},
        ]
        res = _run_fused(stages)
        u = _run_unfused(stages)
        assert "_tap_s0" in res, f"tap output missing (got {sorted(res)})"
        assert (res["OUT"] - u[(1, "OUT")]).abs().max().item() < 1e-6, "tap broke terminal"
        assert (res["_tap_s0"] - u[(0, "OUT")]).abs().max().item() < 1e-6, "tap != intermediate"
        r.ok("observed-intermediate tap exposes @_tap_s0 = intermediate, chain intact")
    except Exception as e:
        r.fail("tap-export fusion", str(e))

    # Guard still fires: an undeclared extra @OUT-sibling on an upstream stage
    # remains unfusable (only DECLARED exports are allowed).
    try:
        stages = [
            {"code": "@OUT = @A * 0.5; @sneaky = @A;", "bindings": {"A": img}},
            {"code": "@OUT = @X * 2.0;", "chain_input": "X", "bindings": {}},
        ]
        raised = False
        try:
            _run_fused(stages)
        except _FUS.FusionError:
            raised = True
        assert raised, "undeclared multi-write should still raise FusionError"
        r.ok("undeclared extra @-write still rejected (guard intact)")
    except Exception as e:
        r.fail("multi-write guard", str(e))


def test_q5_chain_preflight(r: SubTestResult):
    print("\n--- Q-5: chain preflight validation ---")
    from TEX_Wrangle.tex_fusion import chain_preflight
    img = torch.zeros(1, 8, 8, 3)

    # A fusable chain → ok + stats for the HUD.
    try:
        stages = [
            {"code": "@OUT = @A*0.5;", "chain_input": None, "bindings": {"A": img}},
            {"code": "@OUT = @X + 0.1;", "chain_input": "X", "bindings": {}},
        ]
        res = chain_preflight(stages, _infer_binding_type)
        assert res["ok"] is True and res["error"] is None
        assert res["stats"]["stages"] == 2 and res["stats"]["tensor_ops"] >= 1
        assert res["stats"]["outputs"] == ["OUT"]
        r.ok("fusable chain → ok=True with HUD stats")
    except Exception as e:
        r.fail("preflight fusable", str(e))

    # An unfusable upstream (missing @OUT) → red with the offending stage index.
    try:
        stages = [
            {"code": "float t = 1.0;", "chain_input": None, "bindings": {}},
            {"code": "@OUT = @X + 0.1;", "chain_input": "X", "bindings": {}},
        ]
        res = chain_preflight(stages, _infer_binding_type)
        assert res["ok"] is False and res["stage_of_error"] == 0
        assert "OUT" in res["error"]
        r.ok("unfusable (no @OUT) → ok=False, stage_of_error=0")
    except Exception as e:
        r.fail("preflight unfusable stage attribution", str(e))

    # A scatter-write upstream → red (the classic queue-time hard error, now
    # surfaced pre-queue). Never raises.
    try:
        stages = [
            {"code": "@OUT = @A; @OUT[ix,iy] = vec4(1.0);", "chain_input": None,
             "bindings": {"A": img}},
            {"code": "@OUT = @X + 0.1;", "chain_input": "X", "bindings": {}},
        ]
        res = chain_preflight(stages, _infer_binding_type)
        assert res["ok"] is False and "scatter" in res["error"]
        r.ok("scatter-write upstream → red bubble (no queue-time crash)")
    except Exception as e:
        r.fail("preflight scatter", str(e))


def test_q6_preview_downscale(r: SubTestResult):
    print("\n--- Q-6: low-res preview downscale (backend piece) ---")
    from TEX_Wrangle.tex_node import _downscale_for_preview

    # Downscale caps the max dimension while preserving aspect + channels.
    try:
        img = torch.rand(1, 1024, 512, 3)
        small = _downscale_for_preview(img, max_dim=256)
        assert small.shape[0] == 1 and small.shape[3] == 3
        assert max(small.shape[1], small.shape[2]) <= 256
        # aspect ratio preserved (2:1 → 256:128)
        assert small.shape[1] == 256 and small.shape[2] == 128
        r.ok("preview downscale caps max_dim, preserves aspect/channels")
    except Exception as e:
        r.fail("preview downscale", str(e))

    # Already-small images pass through untouched.
    try:
        img = torch.rand(1, 100, 80, 3)
        assert _downscale_for_preview(img, max_dim=256) is img
        r.ok("small image passes through unchanged")
    except Exception as e:
        r.fail("preview downscale passthrough", str(e))
