"""
v0.16 Phase 2 regression tests.

P2-UC4-NEG: constant-propagate NEGATIVE literal locals. `float k = -0.5;` parses
as UnaryOp('-', NumberLiteral), and the propagation pass runs before folding, so
the negative-tuning-constant class was previously missed.
"""
from helpers import *
from TEX_Wrangle.tex_compiler.optimizer import optimize, _const_literal_value
import TEX_Wrangle.tex_compiler.ast_nodes as A


def test_uc4_neg_const_prop(r: SubTestResult):
    print("\n--- P2-UC4-NEG: negative-literal const propagation ---")

    # The helper recognizes a negated literal (built by the real parser).
    try:
        prog = Parser(Lexer("float k = -0.5; @OUT = vec4(k);").tokenize(),
                      source="x").parse()
        kdecl = prog.statements[0]
        assert _const_literal_value(kdecl.initializer) == (-0.5, False), \
            "negated float literal not recognized"
        progi = Parser(Lexer("int n = -3; @OUT = vec4(float(n));").tokenize(),
                       source="x").parse()
        assert _const_literal_value(progi.statements[0].initializer) == (-3, True), \
            "negated int literal not recognized"
        # A plain identifier initializer is not a constant.
        progp = Parser(Lexer("float a = 1.0; float b = a; @OUT = vec4(b);").tokenize(),
                       source="x").parse()
        assert _const_literal_value(progp.statements[1].initializer) is None
        r.ok("_const_literal_value recognizes negated int/float literals")
    except Exception as e:
        r.fail("UC4-NEG helper", str(e))

    # End-to-end: a negative-tuning-constant program fully propagates — after
    # optimize() the locals are substituted and DCE'd, leaving no residual refs.
    try:
        code = "float k = -0.5; float g = -2.0; @OUT = vec4(@A.rgb * g + vec3(k), 1.0);"
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        optimize(prog, tm)
        names = set()
        stack = list(prog.statements)
        while stack:
            n = stack.pop()
            if isinstance(n, A.Identifier):
                names.add(n.name)
            stack.extend(A.iter_child_nodes(n))
        assert not (names & {"k", "g"}), f"locals not propagated: {names & {'k', 'g'}}"
        r.ok("negative-literal locals fully propagated (no residual refs)")
    except Exception as e:
        r.fail("UC4-NEG propagation", str(e))

    # Correctness is preserved: interpreter == codegen on the negative-literal
    # program (and matches the hand-computed value).
    from failure_harness import assert_tier_equiv
    assert_tier_equiv(r, "uc4_neg_equiv",
                      "float k = -0.5; float g = -2.0; @OUT = vec4(@A.rgb * g + vec3(k), 1.0);",
                      {"A": make_img(1, 8, 8, 3)}, tiers=("codegen",), tol=1e-6)


def test_m5_int_binding(r: SubTestResult):
    print("\n--- P2-M5-INT: int64 tensor binding + out= reuse ---")
    # The exact re-opened repro: a wired int64 image binding feeds the M-5 out=
    # reuse pattern `(@A + @A) * 0.5`, which emitted torch.mul(long, fp32, out=long)
    # → error → silent interpreter fallback (codegen/stencil speedup lost).
    import TEX_Wrangle.tex_runtime.compiled as C
    code = "@OUT = vec4((@A + @A) * 0.5, 1.0);"
    imgL = torch.ones(1, 8, 8, 3, dtype=torch.long) * 3   # int64 binding, value 3
    from failure_harness import compile_program

    # (1) Codegen no longer falls back to the interpreter: _plain_execute is not
    #     called (would signal a runtime error was caught).
    try:
        prog, tm, outs = compile_program(code, {"A": imgL})
        calls = {"n": 0}
        orig = C._plain_execute
        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        C._plain_execute = counting
        try:
            out_cg = C._codegen_only_execute(prog, {"A": imgL.clone()}, tm, "cpu",
                                             output_names=outs, fingerprint="m5int", time_context=None)
        finally:
            C._plain_execute = orig
        out_cg = out_cg["OUT"] if isinstance(out_cg, dict) else out_cg
        assert calls["n"] == 0, f"codegen still fell back to interpreter {calls['n']}x on int64 binding"
        r.ok("int64 binding: codegen runs the out= reuse (no interpreter fallback)")
    except Exception as e:
        r.fail("M5-INT no-fallback", f"{type(e).__name__}: {e}")

    # (2) And the codegen result is bit-equal to the interpreter (which promotes
    #     the int64 temp to float naturally): (3+3)*0.5 = 3.0.
    try:
        out_i = Interpreter().execute(prog, {"A": imgL.clone()}, tm, device="cpu",
                                      output_names=outs)["OUT"]
        md = (out_i.float() - out_cg.float()).abs().max().item()
        assert md < 1e-6, f"int64 codegen vs interp maxdiff {md:.3e}"
        assert abs(out_cg.float()[0, 0, 0, 0].item() - 3.0) < 1e-6, "wrong value"
        r.ok(f"int64 binding: codegen == interpreter (maxdiff {md:.1e}, value 3.0)")
    except Exception as e:
        r.fail("M5-INT equivalence", f"{type(e).__name__}: {e}")

    # (3) Cross-tier CONVERGENCE for the divergence class the v0.16 verification
    #     found: a passthrough / int64 value > 2^24 with a non-IMAGE (raw) output.
    #     The interpreter now casts int image-like bindings to fp32 too, so interp
    #     and codegen agree in BOTH dtype and value (was: interp int64 vs cg fp32).
    try:
        from failure_harness import run_tier, max_diff
        big = torch.full((1, 4, 4, 3), 2 ** 25 + 1, dtype=torch.long)  # not fp32-exact
        for c in ("@OUT = @A;", "@OUT = @A + @A;"):
            oi = run_tier(c, {"A": big}, "interp")
            oc = run_tier(c, {"A": big}, "codegen")
            assert oi["OUT"].dtype == oc["OUT"].dtype, \
                f"{c}: dtype diverges ({oi['OUT'].dtype} vs {oc['OUT'].dtype})"
            assert max_diff(oi, oc) < 1e-3, f"{c}: value diverges for int64 > 2^24"
        r.ok("int64 > 2^24 passthrough: interp == codegen (dtype + value)")
    except Exception as e:
        r.fail("M5-INT cross-tier convergence", f"{type(e).__name__}: {e}")


def test_m2cpu_and_m1_freeretry(r: SubTestResult):
    print("\n--- P1-M2-CPU + P1-M1-FREERETRY: memory citizenship ---")
    import os
    import TEX_Wrangle.tex_memory as M
    from TEX_Wrangle.tex_runtime import stdlib as SL

    # P1-M2-CPU: enforce_cache_budget now evicts on a CPU device too.
    saved = dict(SL._grid_buf)
    try:
        SL._grid_buf.clear()
        for i in range(6):
            SL._grid_buf[("m2cpu", i)] = torch.zeros(1, 256, 256, 4)  # ~1 MB each
        os.environ["TEX_CACHE_BUDGET_MB"] = "2"
        before = len(SL._grid_buf)
        M.enforce_cache_budget(torch.device("cpu"))   # CPU device, not cuda
        after = len(SL._grid_buf)
        assert after < before, f"CPU budget did not evict ({before}->{after})"
        assert M._total_cache_bytes() <= 2 * 1024 * 1024 or after == 1, "still over budget"
        r.ok(f"M-2-CPU: enforce_cache_budget evicts on CPU ({before}->{after} entries)")
    except Exception as e:
        r.fail("M-2-CPU eviction", f"{type(e).__name__}: {e}")
    finally:
        os.environ.pop("TEX_CACHE_BUDGET_MB", None)
        SL._grid_buf.clear()
        SL._grid_buf.update(saved)

    # P1-M1-FREERETRY: on OOM, TEX frees its own caches before re-raising the OOM
    # (so ComfyUI's unload_all_models + retry has that memory). Simulated OOM.
    import TEX_Wrangle.tex_node as NODE
    oom_t = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_t is None:
        r.ok("M-1-FREERETRY skipped (no OutOfMemoryError type)")
        return
    calls = {"n": 0}
    orig_free = M.free_tensor_caches
    from TEX_Wrangle import tex_engine as _E   # ENG-1: interpreter singleton moved
    interp = _E._get_interpreter()
    orig_exec = interp.execute
    try:
        M.free_tensor_caches = lambda: calls.__setitem__("n", calls["n"] + 1)

        def boom(*a, **k):
            raise oom_t("CUDA out of memory (simulated for M-1-FREERETRY)")
        interp.execute = boom
        raised_oom = False
        try:
            NODE.TEXWrangleNode().execute(code="@OUT = @A * 0.5;", device="cpu",
                                          compile_mode="none", A=torch.rand(1, 8, 8, 3))
        except BaseException as e:
            raised_oom = isinstance(e, oom_t)
        assert raised_oom, "OOM was not re-raised unwrapped (detection failed)"
        assert calls["n"] >= 1, "free_tensor_caches not called on OOM"
        r.ok("M-1-FREERETRY: TEX caches freed on OOM before re-raise")
    except Exception as e:
        r.fail("M-1-FREERETRY", f"{type(e).__name__}: {e}")
    finally:
        interp.execute = orig_exec
        M.free_tensor_caches = orig_free
