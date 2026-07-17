"""
v0.15.0 Phase 2 regression tests — cook speed.
Q-2 phase 1: purity-aware DCE + post-CSE re-run (optimizer).
"""
from helpers import *
from TEX_Wrangle.tex_compiler.ast_nodes import FunctionCall
from TEX_Wrangle.tex_runtime.codegen import _iter_child_nodes


def _count_calls(prog):
    n, stack = 0, list(prog.statements)
    while stack:
        x = stack.pop()
        if isinstance(x, FunctionCall):
            n += 1
        stack.extend(_iter_child_nodes(x))
    return n


def _optimize(code, bt):
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    before = _count_calls(prog)
    prog2 = optimize(prog, tm)
    return before, _count_calls(prog2)


def test_q2_purity_dce(r: SubTestResult):
    print("\n--- Q-2 phase 1: purity-aware DCE + post-CSE re-run ---")
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

    # (1) A dead VarDecl whose initializer is a pure call is now removed.
    try:
        before, after = _optimize("float x = sin(0.5); @OUT = @A;", bt)
        assert after == 0, f"dead pure call not removed ({before} -> {after})"
        r.ok("dead pure-call VarDecl removed")
    except Exception as e:
        r.fail("dead pure-call VarDecl removed", str(e))

    # (2) A builtin-derived duplicate deduplicates (CSE + DCE re-run).
    try:
        before, after = _optimize(
            "float m = sin(u*10.0); float n = sin(u*10.0); @OUT = @A * (m + n);", bt)
        assert after < before, f"builtin duplicate not deduped ({before} -> {after})"
        r.ok(f"builtin-derived duplicate deduped ({before}->{after} calls)")
    except Exception as e:
        r.fail("builtin-derived duplicate deduped", str(e))

    # (3) A dead call with an IMPURE arg (binding read) must be kept.
    try:
        before, after = _optimize("float x = luma(@A[ix, iy]); @OUT = @A;", bt)
        assert after == before == 1, f"impure-arg call wrongly removed ({before} -> {after})"
        r.ok("dead call with impure (binding-read) arg kept")
    except Exception as e:
        r.fail("dead call with impure (binding-read) arg kept", str(e))

    # (4) Output is bit-exact for the dedup case (correctness preserved).
    try:
        img = make_img(1, 8, 8, 3)
        code = "float m = sin(u*10.0); float n = sin(u*10.0); @OUT = vec4(@A * (m + n), 1.0);"
        assert_equiv(r, "Q-2 dedup bit-exact", code, {"A": img})
    except Exception as e:
        r.fail("Q-2 dedup bit-exact", str(e))

    # (5) A dead user-function call is NOT removed (user fns stay impure).
    try:
        before, after = _optimize(
            "float f(float z) { return z * 2.0; } float x = f(0.5); @OUT = @A;", bt)
        # f(0.5) call must survive (user functions are not on the purity whitelist)
        assert after >= 1, "user-function call wrongly treated as pure/removed"
        r.ok("dead user-function call kept (not whitelisted)")
    except Exception as e:
        r.fail("dead user-function call kept (not whitelisted)", str(e))


def test_uc4_const_prop(r: SubTestResult):
    print("\n--- UC-4: literal-local constant propagation ---")
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

    # A named identity constant lets the scalar pow fold away (pow(g,1.0/1.0)->g),
    # so only luma survives (FunctionCalls drop).
    try:
        before, after = _optimize("""
            float gamma = 1.0;
            float g = luma(@A);
            @OUT = vec4(vec3(pow(g, 1.0 / gamma)), 1.0);
        """, bt)
        assert after < before, f"pow not folded via const-prop ({before}->{after})"
        r.ok(f"identity constant folds pow ({before}->{after} calls)")
    except Exception as e:
        r.fail("identity constant folds pow", str(e))

    # Scope safety: a function param shadowing a top-level literal local must not
    # be corrupted by propagation.
    try:
        code = "float g = 2.0; float f(float g) { return g * 3.0; } @OUT = vec4(vec3(f(4.0) * 0.1), 1.0);"
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        prog = optimize(prog, tm)
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        img = make_img(1, 4, 4, 3)
        out = Interpreter().execute(prog, {"A": img}, tm, device="cpu", output_names=["OUT"])["OUT"]
        assert abs(out[0, 0, 0, 0].item() - 1.2) < 1e-5, "function param shadow corrupted"
        r.ok("function-param shadowing is safe")
    except Exception as e:
        r.fail("function-param shadowing is safe", str(e))

    # A reassigned local is NOT propagated.
    try:
        code = "float k = 1.0; k = k + luma(@A); @OUT = vec4(@A * k, 1.0);"
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        img = make_img(1, 4, 4, 3)
        # bit-exact vs interpreter after optimize (correctness under reassignment)
        assert_equiv(r, "UC-4 reassigned-local bit-exact", code, {"A": img})
    except Exception as e:
        r.fail("UC-4 reassigned-local bit-exact", str(e))


def test_uc1_cuda_graph(r: SubTestResult):
    print("\n--- UC-1: CUDA-graph replay engine ---")
    if not torch.cuda.is_available():
        r.ok("UC-1 cuda_graph (no GPU, SKIPPED)")
        return
    import TEX_Wrangle.tex_runtime.graphed as G
    from TEX_Wrangle.tex_runtime.compiled import _plain_execute

    def _compile(code, bt):
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        used = _collect_identifiers(prog)
        return prog, tm, used

    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

    # Gate: pointwise program capturable; while-loop and param-loop are not.
    try:
        prog, _, _ = _compile("@OUT = vec4(sin(@A) * 0.5 + 0.5, 1.0);", bt)
        assert G._capturable(prog)[0] is True   # (capturable, op_count)
        prog2, _, _ = _compile("float s=0.0; while (s < 3.0) { s = s + 1.0; } @OUT = @A * s;", bt)
        assert G._capturable(prog2)[0] is False
        prog3, _, _ = _compile(
            "i$radius=2; vec3 a=vec3(0.0); for (int d=-$radius; d<=$radius; d=d+1){ a=a+@A.rgb; } @OUT=vec4(a,1.0);",
            {"A": TEXType.VEC3, "radius": TEXType.INT, "OUT": TEXType.VEC4})
        assert G._capturable(prog3)[0] is False  # param loop → .item() at entry
        r.ok("capturability gate (pointwise yes; while/param-loop no)")
    except Exception as e:
        r.fail("capturability gate", str(e))

    # Capture + bit-exact replay vs interpreter.
    try:
        G.clear_graph_cache()
        code = "vec3 c = @A.rgb; float g = luma(c); @OUT = vec4(mix(c, vec3(g), 0.4) * 1.1, 1.0);"
        prog, tm, used = _compile(code, bt)
        img = torch.rand(1, 64, 64, 3, device="cuda")
        g = G.run_graphed(prog, {"A": img}, tm, "cuda", "t_uc1a",
                          output_names=["OUT"], used_builtins=used)
        it = _plain_execute(prog, {"A": img}, tm, "cuda", output_names=["OUT"], used_builtins=used, time_context=None)
        assert g is not None, "program was not captured"
        gt = g["OUT"] if isinstance(g, dict) else g
        itt = it["OUT"] if isinstance(it, dict) else it
        md = (gt.float() - itt.float()).abs().max().item()
        assert md < 1e-5, f"graph replay diverges (maxdiff {md})"
        r.ok(f"capture + bit-exact replay (maxdiff {md:.1e})")
    except Exception as e:
        r.fail("capture + bit-exact replay", str(e))

    # Param staging: a new param value is reflected after fill_ + replay.
    try:
        G.clear_graph_cache()
        code = "f$k = 0.5; @OUT = vec4(@A * $k, 1.0);"
        prog, tm, used = _compile(code, {"A": TEXType.VEC3, "k": TEXType.FLOAT, "OUT": TEXType.VEC4})
        img = torch.rand(1, 64, 64, 3, device="cuda")
        _ = G.run_graphed(prog, {"A": img, "k": 0.5}, tm, "cuda", "t_uc1b",
                          output_names=["OUT"], used_builtins=used)
        r2 = G.run_graphed(prog, {"A": img, "k": 0.25}, tm, "cuda", "t_uc1b",
                           output_names=["OUT"], used_builtins=used)
        out2 = (r2["OUT"] if isinstance(r2, dict) else r2)[..., :3]
        assert torch.allclose(out2, img * 0.25, atol=1e-6), "staged param value not applied"
        r.ok("param staging (fill_ + replay reflects new value)")
    except Exception as e:
        r.fail("param staging", str(e))

    # A gated (uncapturable) program returns None → caller falls back.
    try:
        G.clear_graph_cache()
        prog, tm, used = _compile("float s=0.0; while (s<2.0){s=s+1.0;} @OUT=@A*s;", bt)
        img = torch.rand(1, 32, 32, 3, device="cuda")
        g = G.run_graphed(prog, {"A": img}, tm, "cuda", "t_uc1c",
                          output_names=["OUT"], used_builtins=used)
        assert g is None, "uncapturable program should return None"
        r.ok("uncapturable program returns None (fallback signal)")
    except Exception as e:
        r.fail("uncapturable program returns None", str(e))
    finally:
        G.clear_graph_cache()


def test_q1_fused_capture(r: SubTestResult):
    print("\n--- Q-1: fused chain as CUDA-graph capture unit ---")
    if not torch.cuda.is_available():
        r.ok("Q-1 fused capture (no GPU, SKIPPED)")
        return
    import TEX_Wrangle.tex_fusion as FUS
    import TEX_Wrangle.tex_runtime.graphed as G

    torch.manual_seed(1)
    img = torch.rand(1, 64, 64, 4, device="cuda")
    stages = [{"code": "float t=$amt; @OUT = clamp(@A*t + vec4(0.01,0.02,0.03,0.0), 0.0, 1.0);",
               "chain_input": None, "bindings": {"A": img, "amt": 1.05}}]
    for i in range(2):
        stages.append({"code": "float t=$amt; @OUT = vec4(@X.rgb*t, @X.a);",
                       "chain_input": "X", "bindings": {"amt": 1.02 + i*0.01}})

    # The fingerprint must be reproducible and value-independent.
    try:
        fp1 = FUS._fused_fp(FUS._fused_memo_key(stages, _infer_binding_type))
        s2 = [dict(s) for s in stages]; s2[0] = dict(s2[0], bindings={"A": img, "amt": 9.9})
        fp2 = FUS._fused_fp(FUS._fused_memo_key(s2, _infer_binding_type))
        assert fp1 == fp2, "fused fp changed with a param VALUE (should be value-independent)"
        assert fp1.startswith("fused_")
        r.ok("fused fingerprint reproducible + value-independent")
    except Exception as e:
        r.fail("fused fingerprint reproducible + value-independent", str(e))

    # The fused chain captures as one graph and replays bit-exact vs interpreter.
    try:
        prog, tm, ref, asg, par, used, merged = FUS.compile_fused(stages, _infer_binding_type)
        fp = FUS._fused_fp(FUS._fused_memo_key(stages, _infer_binding_type))
        out = sorted(asg.keys())
        assert G._capturable(prog)[0], "pointwise fused chain should be capturable"
        G.clear_graph_cache()
        ref_out = Interpreter().execute(prog, dict(merged), tm, device="cuda",
                                        output_names=out, used_builtins=used)
        g = G.run_graphed(prog, dict(merged), tm, "cuda", fp, output_names=out, used_builtins=used)
        assert g is not None, f"fused chain not captured (err {G._last_capture_error[0]})"
        k = out[0]
        md = (g[k].float() - ref_out[k].float()).abs().max().item()
        assert md < 1e-5, f"fused graph diverges (maxdiff {md})"
        r.ok(f"fused chain captured as one graph, bit-exact (maxdiff {md:.1e})")
    except Exception as e:
        r.fail("fused chain captured as one graph, bit-exact", str(e))
    finally:
        G.clear_graph_cache()


def test_uc2_stencil_routing(r: SubTestResult):
    print("\n--- UC-2: exact-stencil default routing ---")
    from TEX_Wrangle.tex_runtime.codegen import detect_stencil_route
    from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute, _plain_execute

    def _prog(code, bt):
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        return prog, tm

    FETCH = """
    i$radius = 2;
    vec3 acc = vec3(0.0);
    float cnt = 0.0;
    for (int dy = -$radius; dy <= $radius; dy = dy + 1) {
        for (int dx = -$radius; dx <= $radius; dx = dx + 1) {
            acc = acc + fetch(@A, ix + dx, iy + dy).rgb;
            cnt = cnt + 1.0;
        }
    }
    @OUT = vec4(acc / cnt, 1.0);
    """
    SAMPLE = FETCH.replace("fetch(@A, ix + dx, iy + dy)", "sample(@A, u + float(dx)*px, v + float(dy)*py)")
    bt = {"A": TEXType.VEC3, "radius": TEXType.INT, "OUT": TEXType.VEC4}

    # A fetch-based (exact) stencil routes.
    try:
        prog, tm = _prog(FETCH, bt)
        assert detect_stencil_route(prog) is True, "fetch stencil should route"
        r.ok("fetch-based stencil routes")
    except Exception as e:
        r.fail("fetch-based stencil routes", str(e))

    # A sample-based stencil does NOT route (avg_pool lowering would diverge).
    try:
        prog, tm = _prog(SAMPLE, bt)
        assert detect_stencil_route(prog) is False, "sample stencil must NOT route"
        r.ok("sample-based stencil excluded (policy a)")
    except Exception as e:
        r.fail("sample-based stencil excluded (policy a)", str(e))

    # A non-stencil program does NOT route.
    try:
        prog, tm = _prog("@OUT = vec4(@A * 0.5, 1.0);", bt)
        assert detect_stencil_route(prog) is False
        r.ok("non-stencil program not routed")
    except Exception as e:
        r.fail("non-stencil program not routed", str(e))

    # The routed program's codegen output is bit-exact to the interpreter
    # (this is a DEFAULT-path change — divergence would silently alter output).
    try:
        img = make_img(1, 32, 32, 3)
        prog, tm = _prog(FETCH, bt)
        b = {"A": img, "radius": 2}
        cg = _codegen_only_execute(prog, dict(b), tm, "cpu", output_names=["OUT"],
                                   used_builtins=None, fingerprint="uc2test", time_context=None)
        it = _plain_execute(prog, dict(b), tm, "cpu", output_names=["OUT"], time_context=None)
        cg = cg["OUT"] if isinstance(cg, dict) else cg
        it = it["OUT"] if isinstance(it, dict) else it
        md = (cg.float() - it.float()).abs().max().item()
        assert md < 1e-5, f"routed codegen diverges from interpreter (maxdiff {md})"
        r.ok(f"routed fetch stencil bit-exact (maxdiff {md:.1e})")
    except Exception as e:
        r.fail("routed fetch stencil bit-exact", str(e))

    # A WEIGHTED or otherwise composed tap must DECLINE the route. The pool
    # lowerings (avg_pool2d / max_pool2d / unfold) can only express an UNWEIGHTED
    # neighbourhood, so the tap has to BE the whole accumulated term. Before
    # v0.22.0 the matcher SEARCHED the term's subtree for a fetch and found one
    # here, matching as if the term were the bare tap — the `* 0.5` never reached
    # the emitted kernel, so this routed DEFAULT path returned exactly 2x, with
    # shapes intact (silently). Declining is correct, just not accelerated.
    wbt = dict(bt, w=TEXType.FLOAT)
    HDR = "i$radius = 1;\n    f$w = 0.5;\n"
    LOOP = ("    for (int dy = -$radius; dy <= $radius; dy = dy + 1) {\n"
            "        for (int dx = -$radius; dx <= $radius; dx = dx + 1) {\n"
            "            %s\n        }\n    }\n")

    def _sum(term):
        return (HDR + "    vec3 acc = vec3(0.0);\n" + (LOOP % term)
                + "    @OUT = vec4(acc / 9.0, 1.0);")

    def _mm(term):
        return (HDR + "    vec3 m = vec3(0.0);\n" + (LOOP % term)
                + "    @OUT = vec4(m, 1.0);")

    TAP = "fetch(@A, ix + dx, iy + dy)"
    composed = [
        ("sum: scaled tap (* 0.5)", _sum(f"acc = acc + {TAP}.rgb * 0.5;")),
        ("sum: divided tap (/ 2.0)", _sum(f"acc = acc + {TAP}.rgb / 2.0;")),
        ("sum: param-weighted tap (* $w)", _sum(f"acc = acc + {TAP}.rgb * $w;")),
        ("sum: offset tap (+ 0.1)", _sum(f"acc = acc + ({TAP}.rgb + vec3(0.1));")),
        # Swizzle UNDER the BinOp: the old code dove past the ChannelAccess to the
        # fetch, so it lost the weight AND left channels=None — corrupting channel
        # ORDER as well as magnitude.
        ("sum: swizzle under a BinOp (.bgr * 0.5)", _sum(f"acc = acc + {TAP}.bgr * 0.5;")),
        ("minmax: scaled tap (max)", _mm(f"m = max(m, {TAP}.rgb * 0.5);")),
        ("minmax: swizzle under a BinOp (max)", _mm(f"m = max(m, {TAP}.bgr * 0.5);")),
    ]
    for name, code in composed:
        try:
            prog, _tm = _prog(code, wbt)
            assert detect_stencil_route(prog) is False, \
                "composed tap must not route (its weight cannot survive the pool lowering)"
            r.ok(f"composed tap declines: {name}")
        except Exception as e:
            r.fail(f"composed tap declines: {name}", str(e))

    # Guard the other direction: the fix must not over-decline. A BARE tap in the
    # same shape still has to route, or this became a silent perf regression.
    for name, code in [("sum", _sum(f"acc = acc + {TAP}.rgb;")),
                       ("minmax", _mm(f"m = max(m, {TAP}.rgb);"))]:
        try:
            prog, _tm = _prog(code, wbt)
            assert detect_stencil_route(prog) is True, "bare tap must still route"
            r.ok(f"bare tap still routes: {name}")
        except Exception as e:
            r.fail(f"bare tap still routes: {name}", str(e))

    # The declining programs must also be CORRECT, not merely unrouted: force
    # codegen and require bit-exactness with the interpreter oracle.
    try:
        img = make_img(1, 16, 16, 3)
        worst = 0.0
        for name, code in composed:
            prog, tm = _prog(code, wbt)
            b = {"A": img, "radius": 1, "w": 0.5}
            cg = _codegen_only_execute(prog, dict(b), tm, "cpu", output_names=["OUT"],
                                       used_builtins=None, fingerprint=f"uc2w{abs(hash(name))}", time_context=None)
            it = _plain_execute(prog, dict(b), tm, "cpu", output_names=["OUT"], time_context=None)
            cg = cg["OUT"] if isinstance(cg, dict) else cg
            it = it["OUT"] if isinstance(it, dict) else it
            assert cg.shape == it.shape, f"{name}: shape {tuple(cg.shape)} != {tuple(it.shape)}"
            worst = max(worst, (cg.float() - it.float()).abs().max().item())
        assert worst < 1e-5, f"composed tap diverges from interpreter (maxdiff {worst})"
        r.ok(f"composed taps bit-exact under forced codegen (maxdiff {worst:.1e})")
    except Exception as e:
        r.fail("composed taps bit-exact under forced codegen", str(e))


def test_uc3_uniform_loop(r: SubTestResult):
    print("\n--- UC-3: uniform (param-bounded) loop resolution ---")
    from TEX_Wrangle.tex_compiler.ast_nodes import ForLoop
    img = make_img(1, 4, 4, 3)

    # A param-radius loop (box_blur pattern) resolves to a range and is correct
    # across radius values; and takes the uniform-range fast path.
    code = """
    i$radius = 2;
    vec3 acc = vec3(0.0);
    float cnt = 0.0;
    for (int dy = -$radius; dy <= $radius; dy = dy + 1) {
        acc = acc + @A.rgb;
        cnt = cnt + 1.0;
    }
    @OUT = vec4(acc / cnt, 1.0);
    """
    def _run(radius):
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types={"A": TEXType.VEC3, "radius": TEXType.INT,
                                        "OUT": TEXType.VEC4}, source=code).check(prog)
        it = Interpreter()
        out = it.execute(prog, {"A": img, "radius": radius}, tm, device="cpu",
                         output_names=["OUT"])["OUT"]
        floops = [s for s in prog.statements if isinstance(s, ForLoop)]
        elig = it._uniform_range_cache.get(id(floops[0]))
        return out, elig

    try:
        out2, elig = _run(2)
        assert elig not in (None, False), "uniform-range path not taken"
        # 5 identical adds / 5 -> mean == @A.rgb
        assert torch.allclose(out2[..., :3], img, atol=1e-6)
        out3, _ = _run(3)
        assert torch.allclose(out3[..., :3], img, atol=1e-6)
        r.ok("param-radius loop resolves + correct across radii")
    except Exception as e:
        r.fail("param-radius loop resolves + correct across radii", str(e))

    # A loop whose bound depends on a body-assigned var must NOT be treated as
    # uniform (falls back to the general path) — and stays correct.
    try:
        code2 = """
        float n = 3.0;
        float s = 0.0;
        for (int i = 0; i < n; i = i + 1) {
            s = s + 1.0;
            n = n + 0.0;
        }
        @OUT = vec4(@A * (s * 0.1), 1.0);
        """
        prog = Parser(Lexer(code2).tokenize(), source=code2).parse()
        tm = TypeChecker(binding_types={"A": TEXType.VEC3, "OUT": TEXType.VEC4},
                         source=code2).check(prog)
        it = Interpreter()
        out = it.execute(prog, {"A": img}, tm, device="cpu", output_names=["OUT"])["OUT"]
        floops = [s for s in prog.statements if isinstance(s, ForLoop)]
        elig = it._uniform_range_cache.get(id(floops[0]))
        assert elig is False, "loop with body-mutated bound wrongly marked uniform"
        assert abs(out[0, 0, 0, 0].item() - img[0, 0, 0, 0].item() * 0.3) < 1e-5
        r.ok("body-mutated bound falls back to general path")
    except Exception as e:
        r.fail("body-mutated bound falls back to general path", str(e))
