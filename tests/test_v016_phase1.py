"""
v0.16 Phase 1 — graph-tier honesty regression tests.

PF-1 (resolution/kernel-count crossover gate) + PF-2 (0-kernel trivial skip):
the CUDA-graph tier must capture ONLY where it measurably beats eager, and fall
to the interpreter everywhere else — so cuda_graph is never worse than eager.
"""
from helpers import *
from TEX_Wrangle.tex_runtime import graphed as G


def test_pf1_pf2_graph_gate(r: SubTestResult):
    print("\n--- PF-1/PF-2: CUDA-graph crossover gate ---")
    # Pure predicate — the measured win region, CPU-testable (run_graphed itself
    # early-returns off CUDA, so the decision logic is factored out to test here).
    try:
        W = G._graph_capture_worthwhile
        # The ceilings are per-arch (S-5: arch_support gate profiles) — assert
        # against the LIVE module constants, not Turing literals, so the contract
        # ("declines past the measured crossover") holds on every verified arch.
        base_ceil, high_ceil = G._GRAPH_BASE_PX_CEIL, G._GRAPH_HIGH_PX_CEIL
        # PF-2: 0/1-op programs capture a near-empty graph → pure loss at all res.
        assert W(0, 256 * 256) is False
        assert W(1, 256 * 256) is False
        # Low-kernel: win up to the base ceiling, decline strictly above it.
        assert W(15, base_ceil) is True
        assert W(15, base_ceil + 1) is False
        assert W(15, 4 * base_ceil) is False
        # Kernel-heavy: win up to the high ceiling (the hard cap), decline above.
        assert W(80, high_ceil) is True
        assert W(80, high_ceil + 1) is False
        assert W(80, 4 * high_ceil) is False
        # No spatial input → can't size → allow (tiny scalar programs).
        assert W(10, 0) is True
        r.ok("crossover predicate matches the measured win region "
             f"(base_ceil={base_ceil}, high_ceil={high_ceil})")
    except Exception as e:
        r.fail("PF-1/PF-2 predicate", str(e))

    if not torch.cuda.is_available():
        r.ok("PF-1/PF-2 end-to-end skipped (no CUDA)")
        return

    # End-to-end: the gate actually declines losing configs and still captures
    # winning ones bit-exactly. Uses run_tier, which raises TierUnavailable on a
    # graph decline (never silently counts a decline as a match).
    try:
        from failure_harness import run_tier, TierUnavailable, max_diff
        G.clear_graph_cache()   # reset any session-wide disable from earlier tests
        code = ("vec4 a = vec4(u, v, u * v, 1.0); vec3 d = a.rgb * 2.0 - vec3(0.5);"
                "@OUT = vec4(clamp(d.r,0.0,1.0), clamp(d.g,0.0,1.0), clamp(d.b,0.0,1.0), 1.0);")
        # Sizes derive from the LIVE per-arch base ceiling (S-5): 256² is inside
        # every profile's win region; the decline probe sits strictly above it
        # (2x the ceiling side → 4x the px).
        over_side = 2 * int(G._GRAPH_BASE_PX_CEIL ** 0.5)
        img256 = make_img(1, 256, 256, 3).cuda()
        img_over = make_img(1, over_side, over_side, 3).cuda()

        # 256² low-kernel → win region → captures and matches the interpreter.
        base = run_tier(code, {"A": img256}, "interp", device="cuda")
        got = run_tier(code, {"A": img256}, "graph", device="cuda")
        assert max_diff(base, got) < 1e-4, "graph@256² diverged from interpreter"
        r.ok("graph captures + matches interp at 256² (win region)")

        # Above the base ceiling → past the crossover → declines (PF-1).
        declined = False
        try:
            run_tier(code, {"A": img_over}, "graph", device="cuda")
        except TierUnavailable:
            declined = True
        assert declined, f"low-kernel program NOT declined at {over_side}² (PF-1 gate leak)"
        r.ok(f"graph declines low-kernel program at {over_side}² (PF-1)")

        # passthrough (0 ops) → declines at any resolution (PF-2).
        declined2 = False
        try:
            run_tier("@OUT = @A;", {"A": img256}, "graph", device="cuda")
        except TierUnavailable:
            declined2 = True
        assert declined2, "0-op passthrough NOT declined (PF-2 gate leak)"
        r.ok("graph declines 0-kernel passthrough at 256² (PF-2)")
    except Exception as e:
        r.fail("PF-1/PF-2 end-to-end", f"{type(e).__name__}: {e}")


def test_static_gate_noise(r: SubTestResult):
    print("\n--- P1-UC1-STATIC-GATE: octave-noise capture exclusion ---")
    from failure_harness import compile_program
    # Octave/count noise resolves its loop count via .item() → capture-illegal
    # sync → statically non-capturable (skips the doomed capture + RNG recovery).
    try:
        for expr in ("fbm(u*6.0,v*6.0,6)", "ridged(u*6.0,v*6.0,6)",
                     "turbulence(u*6.0,v*6.0,6)", "billow(u*6.0,v*6.0,6)"):
            prog, _, _ = compile_program(f"float n = {expr}; @OUT = vec4(n*0.5+0.5);", {})
            assert G._capturable(prog)[0] is False, f"{expr} should be gated out (count sync)"
        # Single-eval noise has no such sync → stays capturable (measured 6.4× @256²).
        for expr in ("perlin(u*8.0,v*8.0)", "simplex(u*8.0,v*8.0)",
                     "worley_f1(u*8.0,v*8.0)", "curl(u*8.0,v*8.0).x"):
            prog, _, _ = compile_program(f"float n = {expr}; @OUT = vec4(n*0.5+0.5);", {})
            assert G._capturable(prog)[0] is True, f"{expr} should stay capturable"
        r.ok("octave-noise gated out; single-eval noise stays capturable")
    except Exception as e:
        r.fail("static-gate noise classification", str(e))

    if not torch.cuda.is_available():
        r.ok("static-gate end-to-end skipped (no CUDA)")
        return

    # End-to-end: an fbm program declines via the static gate WITHOUT recording a
    # capture error — proving no doomed capture / RNG-poison recovery was run.
    try:
        from failure_harness import run_tier, TierUnavailable
        G.clear_graph_cache()
        G._last_capture_error[0] = None
        img = make_img(1, 256, 256, 3).cuda()
        declined = False
        try:
            run_tier("float n = fbm(u*6.0, v*6.0, 6); @OUT = vec4(n*0.5+0.5, 0.0, 0.0, 1.0);",
                     {"A": img}, "graph", device="cuda")
        except TierUnavailable:
            declined = True
        assert declined, "fbm not declined by the static gate"
        assert G._last_capture_error[0] is None, \
            "static gate still attempted a doomed fbm capture (recorded an error)"
        r.ok("fbm declines via static gate — no doomed capture attempted")
    except Exception as e:
        r.fail("static-gate end-to-end", f"{type(e).__name__}: {e}")
