"""
v0.17 Phase 3 — decomposition.

STR-5 optimizer PASSES · STR-6 codegen emit-dispatch registry.
"""
from helpers import *


def test_str5_passes_order(r: SubTestResult):
    print("\n--- STR-5: optimizer PASSES pipeline (order is load-bearing) ---")
    from TEX_Wrangle.tex_compiler import optimizer as O
    try:
        names = [n for n, _ in O.PASSES]
        # The load-bearing order: dce-repeat AFTER cse; unroll AFTER licm.
        assert names.index("cse") < names.index("dce-repeat"), "CSE must precede dce-repeat"
        assert names.index("licm") < names.index("unroll"), "LICM must precede unroll"
        assert names[0] == "const-propagate-locals", "literal-locals prop must run first"
        assert all(callable(run) for _, run in O.PASSES), "every pass must be callable"
        r.ok(f"PASSES = {names} (order invariants hold)")
    except Exception as e:
        r.fail("STR-5 PASSES", f"{type(e).__name__}: {e}")


def test_str6_emit_dispatch_registry(r: SubTestResult):
    print("\n--- STR-6: codegen emit-dispatch built from @_emits registry ---")
    from TEX_Wrangle.tex_runtime import codegen as C
    try:
        # The registry is populated at class-definition time by the @_emits
        # decorators; the per-instance _fn_dispatch is a view of it.
        assert len(C._EMIT_DISPATCH) > 0, "_EMIT_DISPATCH empty — decorators didn't run"
        cg = C._CodeGen({})
        assert set(cg._fn_dispatch) == set(C._EMIT_DISPATCH), (
            "instance _fn_dispatch != registry: "
            f"{set(cg._fn_dispatch) ^ set(C._EMIT_DISPATCH)}")
        assert all(callable(h) for h in cg._fn_dispatch.values()), "handler not callable"
        # every emit-dispatch name is bound to an actual _emit_fn_* method
        assert all(getattr(h, "__name__", "").startswith("_emit_fn_")
                   for h in cg._fn_dispatch.values()), "handler is not an _emit_fn_*"
        r.ok(f"{len(cg._fn_dispatch)} dispatch entries, all bound to @_emits handlers")
    except Exception as e:
        r.fail("STR-6 dispatch", f"{type(e).__name__}: {e}")


def test_str7_codegen_split(r: SubTestResult):
    print("\n--- STR-7: codegen split modules (acyclic, re-export identity) ---")
    import ast
    import os
    from TEX_Wrangle.tex_runtime import codegen as C
    from TEX_Wrangle.tex_runtime import codegen_stdfns, codegen_stencil, codegen_persist
    rt = os.path.dirname(os.path.abspath(C.__file__))
    try:
        # (1) the extracted modules must NOT import back into codegen (strict DAG).
        for mod in ("codegen_stdfns", "codegen_stencil", "codegen_persist"):
            tree = ast.parse(open(os.path.join(rt, mod + ".py"), encoding="utf-8").read())
            back = [n.module for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom) and n.module and n.module.endswith("codegen")]
            assert not back, f"{mod} imports back into codegen: {back}"
        r.ok("codegen_stdfns/stencil/persist form a strict DAG (no edge back to codegen)")
    except Exception as e:
        r.fail("STR-7 DAG", f"{type(e).__name__}: {e}")
    try:
        # (2) re-export identities (external callers resolve via codegen.*).
        assert C.detect_stencil_route is codegen_stencil.detect_stencil_route
        assert C.materialize_codegen is codegen_persist.materialize_codegen
        # (3) _SPATIAL_STDLIB single-sourced from the registry.
        from TEX_Wrangle.tex_runtime import stdlib_registry
        assert set(C._SPATIAL_STDLIB) == set(stdlib_registry.spatial_names())
        r.ok("re-export identities hold; _SPATIAL_STDLIB == registry.spatial_names()")
    except Exception as e:
        r.fail("STR-7 re-export", f"{type(e).__name__}: {e}")


def test_str9_stmt_dispatch(r: SubTestResult):
    print("\n--- STR-9: codegen statement dispatch table (exhaustive) ---")
    from TEX_Wrangle.tex_runtime import codegen as C
    from TEX_Wrangle.tex_compiler.ast_nodes import (
        VarDecl, ArrayDecl, Assignment, ReturnStmt, ExprStatement, ParamDecl,
        IfElse, ForLoop, WhileLoop, FunctionDef, BreakStmt, ContinueStmt)
    try:
        cg = C._CodeGen({})
        expected = {VarDecl, ArrayDecl, Assignment, ReturnStmt, ExprStatement, ParamDecl,
                    IfElse, ForLoop, WhileLoop, FunctionDef, BreakStmt, ContinueStmt}
        assert set(cg._stmt_dispatch) == expected, (
            f"dispatch != 12 statement types: {set(cg._stmt_dispatch) ^ expected}")
        assert all(callable(h) for h in cg._stmt_dispatch.values())
        r.ok("statement dispatch covers all 12 statement types (no silent fallthrough)")
    except Exception as e:
        r.fail("STR-9 dispatch", f"{type(e).__name__}: {e}")


def test_str4_write_collectors(r: SubTestResult):
    print("\n--- STR-4: write/reassign collectors (selectivity pinned) ---")
    from TEX_Wrangle.tex_compiler.lexer import Lexer
    from TEX_Wrangle.tex_compiler.parser import Parser
    from TEX_Wrangle.tex_compiler import optimizer as O

    def collect(code):
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        return (O._collect_written_vars(prog.statements),
                O._collect_reassigned_vars(prog.statements))

    try:
        # (a) an index/RHS read must NOT be collected as a *reassignment*; only the
        #     assignment's container name is (reassigned ignores decls, so a stray `j`
        #     could only come from the index — the bug this pins).
        w, ra = collect("float arr[4]; int j = 2; arr[j] = u; @OUT = vec4(0.0, 0.0, 0.0, 1.0);")
        assert "arr" in ra and "j" not in ra and "u" not in ra, f"index/RHS leak: ra={ra}"
        assert "arr" in w and "j" in w, f"decls written: w={w}"
        # (b) loop asymmetry: `i` is WRITTEN (init/update) but NOT reassigned at block
        #     level (reassigned visits the body only); `acc` (body assignment) is both.
        w2, ra2 = collect("float acc = 0.0; for (int i = 0; i < 4; i = i + 1) { acc = acc + u; } @OUT = vec4(acc, acc, acc, 1.0);")
        assert "i" in w2 and "acc" in w2, f"loop writes: {w2}"
        assert "acc" in ra2 and "i" not in ra2, f"reassign asymmetry: {ra2}"
        # (c) a VarDecl name is written but not reassigned.
        w3, ra3 = collect("float x = 1.0; @OUT = vec4(x, x, x, 1.0);")
        assert "x" in w3 and "x" not in ra3, f"decl vs reassign: w={w3} ra={ra3}"
        r.ok("write/reassign collectors: write-only, index/read/condition excluded, asymmetry held")
    except Exception as e:
        r.fail("STR-4 collectors", f"{type(e).__name__}: {e}")


def test_str2_select_tier_matrix(r: SubTestResult):
    print("\n--- STR-2: pure tier-SELECTION matrix (CPU-testable, no execution) ---")
    from TEX_Wrangle.tex_node import TEXWrangleNode
    S = TEXWrangleNode.select_tier

    def expected(mode, device, fused, fp_present):
        # the cascade guards, restated independently as the oracle
        if mode == "torch_compile" and not fused:
            return "torch_compile"
        if mode == "auto" and not fused:
            return "auto"
        if str(device).startswith("cuda") and mode == "cuda_graph" and (not fused or fp_present):
            return "cuda_graph"
        return "default"

    try:
        fails = []
        for mode in ("none", "torch_compile", "auto", "cuda_graph"):
            for device in ("cpu", "cuda:0"):          # device is just a string to startswith
                for fused in (False, True):
                    for fp in (False, True):
                        got = S(mode, device, fused, fp)
                        exp = expected(mode, device, fused, fp)
                        if got != exp:
                            fails.append(f"{(mode,device,fused,fp)}: got {got} exp {exp}")
        assert not fails, "; ".join(fails[:8])
        r.ok("select_tier matches the cascade guards across all 4×2×2×2 combos (CPU)")
    except Exception as e:
        r.fail("STR-2 select_tier", f"{type(e).__name__}: {e}")

    # live GPU confidence pass — cuda_graph must actually route + stay bit-exact
    import torch
    if torch.cuda.is_available():
        try:
            from failure_harness import run_tier, max_diff
            g = make_img(1, 128, 128, 3).cuda()
            code = "vec3 d = @A.rgb * 2.0 - vec3(0.25); @OUT = vec4(clamp(d, 0.0, 1.0), 1.0);"
            base = run_tier(code, {"A": g}, "interp", device="cuda")
            got = run_tier(code, {"A": g}, "graph", device="cuda")
            assert max_diff(base, got) < 1e-5, f"cuda_graph diverged: {max_diff(base, got):.2e}"
            r.ok("cuda_graph strategy routes on live GPU + bit-exact vs interpreter")
        except Exception as e:
            r.fail("STR-2 cuda_graph e2e", f"{type(e).__name__}: {e}")
    else:
        r.ok("STR-2 cuda_graph e2e skipped (no CUDA)")


def test_c2_clamp_mixed_bounds(r: SubTestResult):
    print("\n--- C2: clamp with mixed scalar/tensor bounds codegens (no fallback) ---")
    from TEX_Wrangle.tex_runtime import tier_trace
    from failure_harness import run_tier, max_diff
    binds = {"A": make_img(1, 16, 16, 3, seed=5), "B": make_img(1, 16, 16, 3, seed=6)}
    # every bound combo: torch.clamp rejects (Tensor, scalar, Tensor); the fix uses
    # clamp_min().clamp_max() so codegen ENGAGES (no interpreter fallback) + stays
    # bit-exact vs fn_clamp's minimum(maximum(...)) spatial path.
    cases = {
        "scalar,scalar": "@OUT = vec4(clamp(u, 0.2, 0.7), u, v, 1.0);",
        "scalar,tensor": "@OUT = vec4(clamp(u, 0.3, @A.r), u, v, 1.0);",
        "tensor,scalar": "@OUT = vec4(clamp(@A.g, @B.r, 0.8), u, v, 1.0);",
        "tensor,tensor": "@OUT = vec4(clamp(@A.g, @B.r, @A.b), u, v, 1.0);",
    }
    fails = []
    for label, code in cases.items():
        try:
            tier_trace.reset()
            c = run_tier(code, binds, "codegen")
            rec = tier_trace.last()
            a = run_tier(code, binds, "interp")
            if rec is None or rec.tier != "codegen":
                fails.append(f"{label}: codegen fell back ({rec}) — mixed-bound clamp not lowered")
            elif max_diff(a, c) >= 1e-5:
                fails.append(f"{label}: maxdiff {max_diff(a, c):.2e}")
        except Exception as e:
            fails.append(f"{label}: {type(e).__name__}: {e}")
    if fails:
        r.fail("C2 clamp mixed bounds", "; ".join(fails))
    else:
        r.ok("mixed-bound clamp codegens (tier=codegen, no fallback) + bit-exact, all 4 combos")
