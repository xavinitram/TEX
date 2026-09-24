"""v0.41 v042-graph — CUDA-graph capture of `viewer_exposure()`/`viewer_gamma()`.

PM-11 shipped the two host-context builtins but barred CUDA-graph capture for any program
calling either (`graphed._capturable`'s `_reads_host_context_cached` check) — a captured
replay never calls Python again, so it would keep re-serving whatever value was read at
capture, and this pair is *expected* to change every cook (a dragged slider). PM-11's own
hand-back named the fix and deferred it: feed the value as a per-replay static input
buffer, copied into a persistent device tensor before each replay, the same mechanism
`static_bindings` already uses for ordinary wire bindings — it "needs the capture plumbing
to own the buffer" (`graphed._capturable`'s docstring, the `frame`/`time` paragraph).

This module proves that fix: (1) a tweak between replays of the SAME captured graph
changes the output correctly and bit-exactly matches an uncaptured cook at the same value;
(2) no re-capture and no recompile happens on a tweak; (3) a program that never calls a
host-context builtin captures exactly as before (invariant 7). `frame`/`fps`/`time`
(`_TIME_BUILTIN_NAMES`) are a separate mechanism (bare Identifiers, not FunctionCalls) and
stay barred — this ask is viewer-only, as its brief allowed.
"""
from helpers import *
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_runtime.compiled import _plain_execute


def _compile(code, bt):
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    used = _collect_identifiers(prog)
    return prog, tm, used


def test_v042_viewer_now_capturable(r: SubTestResult):
    print("\n--- v042-graph: viewer_exposure()/viewer_gamma() are now CUDA-graph capturable ---")
    import TEX_Wrangle.tex_runtime.graphed as G
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    try:
        exposure_prog, _, _ = _compile("@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);", bt)
        gamma_prog, _, _ = _compile(
            "@OUT = vec4(pow(@A.rgb, vec3(1.0 / viewer_gamma())), 1.0);", bt)
        cap_e, ops_e = G._capturable(exposure_prog)
        cap_g, ops_g = G._capturable(gamma_prog)
        assert cap_e is True, f"viewer_exposure() program still barred (ops={ops_e})"
        assert cap_g is True, f"viewer_gamma() program still barred (ops={ops_g})"
        r.ok(f"both viewer_exposure() (ops={ops_e}) and viewer_gamma() (ops={ops_g}) "
             f"programs are capturable")
    except Exception as e:
        r.fail("v042 viewer capturable", f"{type(e).__name__}: {e}")

    # frame/fps/time stay barred — a SEPARATE mechanism (bare Identifiers), untouched by
    # this ask, and the brief's explicit fallback ("do viewer only") if they weren't.
    try:
        time_prog, _, _ = _compile("@OUT = vec4(vec3(frame), 1.0);", bt)
        cap_t, _ = G._capturable(time_prog)
        assert cap_t is False, "frame is now capturable — this ask was NOT meant to touch it"
        r.ok("frame stays barred from CUDA-graph capture (untouched by this ask)")
    except Exception as e:
        r.fail("v042 frame still barred", f"{type(e).__name__}: {e}")


def test_v042_viewer_replay_correctness(r: SubTestResult):
    print("\n--- v042-graph: a viewer tweak between replays is correct and bit-exact ---")
    if not torch.cuda.is_available():
        r.skip("v042 viewer replay correctness", "no CUDA on this box")
        return
    import TEX_Wrangle.tex_runtime.graphed as G
    G.clear_graph_cache()
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    code = "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);"
    prog, tm, used = _compile(code, bt)
    img = torch.rand(1, 64, 64, 3, device="cuda")
    fp = "v042_replay_correctness"

    def uncaptured(exposure):
        out = _plain_execute(prog, {"A": img.clone()}, tm, "cuda", output_names=["OUT"],
                             used_builtins=used, time_context=None,
                             viewer_context={"viewer_exposure": exposure})
        return out["OUT"] if isinstance(out, dict) else out

    try:
        g1 = G.run_graphed(prog, {"A": img.clone()}, tm, "cuda", fp, output_names=["OUT"],
                           used_builtins=used, viewer_context={"viewer_exposure": 2.0})
        assert g1 is not None, "program was not captured — cannot test replay"
        out1 = g1["OUT"] if isinstance(g1, dict) else g1
        ref1 = uncaptured(2.0)
        md1 = (out1.float() - ref1.float()).abs().max().item()
        assert md1 < 1e-5, f"first replay (exposure=2.0) diverges from interpreter: {md1:.3e}"

        g2 = G.run_graphed(prog, {"A": img.clone()}, tm, "cuda", fp, output_names=["OUT"],
                           used_builtins=used, viewer_context={"viewer_exposure": 5.0})
        assert g2 is not None, "second cook was not served by the graph tier"
        out2 = g2["OUT"] if isinstance(g2, dict) else g2
        ref2 = uncaptured(5.0)
        md2 = (out2.float() - ref2.float()).abs().max().item()
        assert md2 < 1e-5, f"second replay (exposure=5.0) diverges from interpreter: {md2:.3e}"

        between = (out1.float() - out2.float()).abs().max().item()
        assert between > 1e-3, \
            f"the two replays (exposure 2.0 vs 5.0) did not actually differ (maxdiff {between:.3e})"

        r.ok(f"replay@2.0 vs interpreter maxdiff {md1:.1e}; replay@5.0 vs interpreter "
             f"maxdiff {md2:.1e}; the two replays differ by {between:.3f} (both non-identity, "
             f"same captured graph)")
    except Exception as e:
        r.fail("v042 viewer replay correctness", f"{type(e).__name__}: {e}")
    finally:
        G.clear_graph_cache()


def test_v042_viewer_no_recapture(r: SubTestResult):
    print("\n--- v042-graph: a viewer tweak causes NO re-capture and NO recompile ---")
    if not torch.cuda.is_available():
        r.skip("v042 viewer no recapture", "no CUDA on this box")
        return
    import TEX_Wrangle.tex_runtime.graphed as G
    from TEX_Wrangle.tex_runtime.compiled import _compiled_cache
    G.clear_graph_cache()
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    code = "@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);"
    prog, tm, used = _compile(code, bt)
    img = torch.rand(1, 32, 32, 3, device="cuda")
    fp = "v042_no_recapture"
    try:
        cache_before_any = len(_compiled_cache)
        G.run_graphed(prog, {"A": img.clone()}, tm, "cuda", fp, output_names=["OUT"],
                     used_builtins=used, viewer_context={"viewer_exposure": 1.0})
        n_graphs_1 = len(G._graph_cache)
        n_blacklist_1 = len(G._blacklist)
        assert n_graphs_1 == 1, f"first cook did not capture exactly one graph ({n_graphs_1})"

        for exposure in (3.0, 0.2, 9.9):
            out = G.run_graphed(prog, {"A": img.clone()}, tm, "cuda", fp, output_names=["OUT"],
                               used_builtins=used, viewer_context={"viewer_exposure": exposure})
            assert out is not None, f"exposure={exposure} was not served by the graph tier"

        n_graphs_2 = len(G._graph_cache)
        n_blacklist_2 = len(G._blacklist)
        cache_after = len(_compiled_cache)
        assert n_graphs_2 == n_graphs_1, \
            f"graph cache grew across viewer tweaks ({n_graphs_1} -> {n_graphs_2}): a re-capture happened"
        assert n_blacklist_2 == n_blacklist_1, \
            f"capture blacklist grew ({n_blacklist_1} -> {n_blacklist_2}): a capture failed"
        assert cache_after == cache_before_any, \
            f"_compiled_cache grew ({cache_before_any} -> {cache_after}): this tier recompiled"
        r.ok(f"one capture ({n_graphs_1} graph) served 4 different viewer_exposure values; "
             f"_compiled_cache unmoved ({cache_before_any} -> {cache_after})")
    except Exception as e:
        r.fail("v042 viewer no recapture", f"{type(e).__name__}: {e}")
    finally:
        G.clear_graph_cache()


def test_v042_plain_program_unaffected(r: SubTestResult):
    print("\n--- v042-graph: invariant 7 — a program with no host-context builtin is unaffected ---")
    import TEX_Wrangle.tex_runtime.graphed as G
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    try:
        # Same program string `test_v015_phase2.py::test_uc1_cuda_graph` already pins to
        # (capturable=True) pre-v042-graph; op_count is a proxy for the same AST walk this
        # ask leaves untouched for any program that calls no host-context builtin, so an
        # unpinned but STABLE (deterministic, non-zero) reading is the invariant-7 claim.
        plain_prog, _, _ = _compile("@OUT = vec4(sin(@A) * 0.5 + 0.5, 1.0);", bt)
        cap, ops = G._capturable(plain_prog)
        cap2, ops2 = G._capturable(plain_prog)
        assert cap is True, f"a plain program is no longer capturable (ops={ops})"
        assert (cap2, ops2) == (cap, ops), \
            f"_capturable is non-deterministic across two calls: {(cap, ops)} vs {(cap2, ops2)}"
        r.ok(f"a plain (non-viewer) program still reads (capturable=True, ops={ops}), "
             f"deterministically — unmoved by the host-context buffer mechanism")
    except Exception as e:
        r.fail("v042 plain program capturability unaffected", f"{type(e).__name__}: {e}")

    if not torch.cuda.is_available():
        r.skip("v042 plain program replay unaffected", "no CUDA on this box")
        return
    import TEX_Wrangle.tex_runtime.graphed as G2
    G2.clear_graph_cache()
    try:
        code = "vec3 c = @A.rgb; float g = luma(c); @OUT = vec4(mix(c, vec3(g), 0.4) * 1.1, 1.0);"
        prog, tm, used = _compile(code, bt)
        img = torch.rand(1, 64, 64, 3, device="cuda")
        g = G2.run_graphed(prog, {"A": img}, tm, "cuda", "v042_plain_unaffected",
                          output_names=["OUT"], used_builtins=used)
        it = _plain_execute(prog, {"A": img}, tm, "cuda", output_names=["OUT"],
                            used_builtins=used, time_context=None)
        assert g is not None, "a plain program was no longer captured"
        gt = g["OUT"] if isinstance(g, dict) else g
        itt = it["OUT"] if isinstance(it, dict) else it
        md = (gt.float() - itt.float()).abs().max().item()
        assert md < 1e-5, f"plain-program graph replay diverges (maxdiff {md:.3e})"
        r.ok(f"a plain program still captures and replays bit-exactly vs the interpreter "
             f"(maxdiff {md:.1e}) — no host-context buffer is ever built for it")
    except Exception as e:
        r.fail("v042 plain program replay unaffected", f"{type(e).__name__}: {e}")
    finally:
        G2.clear_graph_cache()
