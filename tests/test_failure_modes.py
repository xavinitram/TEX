"""
PROC-1 — the five failure-mode classes, exercised through failure_harness.

Each test drives ONE class of the harness against a mode that shipped
green-but-broken in v0.15 (now fixed), proving the pattern catches it. Future
v0.16 fixes add their regression here (or via these helpers) rather than adding
another happy-path assertion.

  A async-lifecycle · B restart/persist · C real-entry-point ·
  D cross-tier-equiv · E full-surface-sweep
"""
from helpers import *
from failure_harness import (
    run_tier, assert_tier_equiv, sweep_precision, preflight_via_endpoint,
    drive_auto, simulate_restart, max_diff, TierUnavailable,
)


# ── Class A — async background-compile lifecycle ──────────────────────
def test_fm_class_a_auto_lifecycle(r: SubTestResult):
    print("\n--- PROC-1 [A]: auto-tier lifecycle (correctness every cook) ---")
    # The v0.15 auto test never advanced past the first sync cook, so it never
    # reached the crashing TRIAL. drive_auto cooks repeatedly; the invariant that
    # MUST hold regardless of which internal tier serves each cook is: output is
    # bit-equal to the interpreter. (The stricter "reaches COMMITTED" assertion
    # lands with the CC-2 hardening in Phase 2.)
    img = make_img(seed=11)
    code = "@OUT = vec4(sin(@A * 3.14159) * 0.5 + 0.5, 1.0);"
    try:
        ref = run_tier(code, {"A": img}, "interp")
        cooks = drive_auto(code, {"A": img}, cooks=3)
        worst = max(max_diff(ref, c) for c in cooks)
        assert worst < 1e-5, f"auto diverged from interpreter across cooks (max {worst:.3e})"
        r.ok(f"auto tier bit-matches interpreter across 3 cooks (max {worst:.2e})")
    except Exception as e:
        r.fail("[A] auto lifecycle correctness", f"{type(e).__name__}: {e}")


# ── Class B — restart / persistence ───────────────────────────────────
def test_fm_class_b_restart(r: SubTestResult):
    print("\n--- PROC-1 [B]: restart reconstruction stays correct ---")
    # A loop-heavy program routes through codegen and persists a sidecar. After a
    # simulated restart (in-memory caches dropped) the reconstructed path must
    # produce output bit-identical to the interpreter — persistence must never
    # silently corrupt (the class that hid the M-2 eviction corruption).
    img = make_img(seed=13)
    code = ("vec3 acc = vec3(0.0); for (int i = 0; i < 4; i = i + 1) "
            "{ acc = acc + @A.rgb * 0.1; } @OUT = vec4(acc, 1.0);")
    try:
        ref = run_tier(code, {"A": img}, "interp")
        warm = run_tier(code, {"A": img}, "codegen")           # populate memory + disk
        assert max_diff(ref, warm) < 1e-6, "warm codegen already diverges"
        simulate_restart()                                     # drop memory tier
        cold = run_tier(code, {"A": img}, "codegen")           # reconstruct from disk
        md = max_diff(ref, cold)
        assert md < 1e-6, f"post-restart codegen diverged (max {md:.3e})"
        r.ok(f"codegen output identical before/after restart (max {md:.2e})")
    except Exception as e:
        r.fail("[B] restart reconstruction", f"{type(e).__name__}: {e}")


# ── Class C — real entry-point (not the inner helper) ─────────────────
def test_fm_class_c_entrypoint(r: SubTestResult):
    print("\n--- PROC-1 [C]: preflight via the endpoint, not the helper ---")
    # The v0.15 test called chain_preflight directly and missed that
    # preflight_from_spec returned not-ok for EVERY chain. Drive the endpoint.
    try:
        good = {'stages': [{'code': '@OUT = @A * 2.0;', 'image_input': 'A', 'params': {}}],
                'terminal_image_input': 'A'}
        rg = preflight_via_endpoint(good, '@OUT = @A + 0.1;')
        assert rg['ok'] is True, f"valid chain preflights not-ok: {rg.get('error')}"
        bad = {'stages': [{'code': '@OUT = @A * "x";', 'image_input': 'A', 'params': {}}],
               'terminal_image_input': 'A'}
        rb = preflight_via_endpoint(bad, '@OUT = @A + 0.1;')
        assert rb['ok'] is False, "broken chain wrongly preflights ok (always-green)"
        r.ok("endpoint discriminates valid vs broken chains")
    except Exception as e:
        r.fail("[C] preflight endpoint", f"{type(e).__name__}: {e}")

    # Second entry-point: the node's own execute() must accept the harness kwargs
    # and cook a real result (the surface the P0-4 _tex_preview pop lives on).
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        img = make_img()
        out = TEXWrangleNode().execute(code="@OUT = @A * 0.5;", device="cpu",
                                       compile_mode="none", A=img)
        res = out[0] if isinstance(out, tuple) else out
        assert isinstance(res, torch.Tensor) and res.shape[0] == 1
        r.ok("node execute() entry-point cooks a valid tensor")
    except Exception as e:
        r.fail("[C] node execute entry-point", f"{type(e).__name__}: {e}")


# ── Class D — cross-tier equivalence ──────────────────────────────────
def test_fm_class_d_cross_tier(r: SubTestResult):
    print("\n--- PROC-1 [D]: opt-in tiers bit-match the interpreter ---")
    img = make_img(seed=17)
    # Includes the two shapes whose tier bugs slipped v0.15: a vec/color param
    # (UC-1 graph staging) and a fractional-bound loop (UC-3 floor).
    corpus = [
        ("vec_param", "vec3 tint = vec3(0.2,0.5,0.9); @OUT = vec4(@A * tint, 1.0);", {"A": img}),
        ("frac_loop", "float s=0.0; for(float i=0.5;i<3.0;i=i+1.0){s=s+i;} @OUT=vec4(s*0.1,0,0,1);", {}),
        ("math_chain", "@OUT = vec4(sin(@A*3.14159)*0.5+0.5, 1.0);", {"A": img}),
    ]
    # CPU: codegen tier must never produce a wrong result (self-falls-back if
    # unsupported, so this is a safety net, not a codegen-engaged assertion).
    for nm, code, binds in corpus:
        assert_tier_equiv(r, nm, code, binds, tiers=("codegen",), tol=1e-5)

    # GPU: the graph tier is the real UC-1 regression surface — compare bitwise
    # where CUDA exists; a decline is acceptable (not every program is graphable).
    if torch.cuda.is_available():
        for nm, code, binds in corpus:
            g = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in binds.items()}
            assert_tier_equiv(r, f"{nm}@cuda", code, g, tiers=("graph",),
                              device="cuda", tol=1e-4)
    else:
        r.ok("[D] graph tier skipped (no CUDA)")


# ── Class E — full-surface sweep (fp16 dtype-reconcile family) ─────────
def test_fm_class_e_fp16_sweep(r: SubTestResult):
    print("\n--- PROC-1 [E]: fp16 across the grade/blend family ---")
    # The v0.15 fp16 test hit one function; the dtype-reconcile bug lived in
    # several. Sweep the whole family so a reconciler missing on ANY one fails.
    img = make_img(seed=19)
    programs = {
        "lerp":       "@OUT = vec4(lerp(@A, @A * 0.5, 0.3), 1.0);",
        "mix":        "@OUT = vec4(mix(@A, @A + 0.1, v), 1.0);",
        "fit":        "@OUT = vec4(fit(@A, 0.0, 1.0, 0.2, 0.8), 1.0);",
        "smin":       "@OUT = vec4(smin(@A, @A * 0.5, 0.1), 1.0);",
        "smax":       "@OUT = vec4(smax(@A, @A * 0.5, 0.1), 1.0);",
        "gauss_blur": "@OUT = vec4(gauss_blur(@A, 2.0), 1.0);",
    }
    sweep_precision(r, "fp16", programs, {"A": img}, {"A": img.half()}, tol=3e-2)
