"""v0.19.0 Phase 1 — harden the net.

A1-2: the combined fusion x lazy x auto/cuda_graph interaction had near-zero coverage
(doc 34 weakness #6) — each axis was tested alone, but they compose in production and
fusion rewrites the AST the lazy analysis walks. This drives a real fused chain through
TEXWrangleNode.execute (not compile_fused directly) under multiple tiers/precisions and
asserts (a) bit-exactness vs the unfused fp32 reference and (b) that lazy is disabled
under fusion (the documented v1 "fusion requests everything" rule) — pinning it explicitly.
"""
import json
from pathlib import Path
from helpers import *
from TEX_Wrangle.tex_node import TEXWrangleNode as _N

_PKG = Path(__file__).resolve().parent.parent


def _first(out):
    return out[0] if isinstance(out, tuple) else out


def test_a1_6_cli_argv(r: SubTestResult):
    print("\n--- A1-6: tex CLI argv/main() path (lowest-covered module) ---")
    # The CLI's argv/main() path (flag parsing, --help, bad-arg exit codes, the real
    # run() round-trip) had zero coverage — every prior CLI test called run_program/
    # save_image directly. This exercises main(argv) end-to-end.
    import tempfile, os
    from TEX_Wrangle import tex_cli
    try:
        import torchvision.io as tvio
    except Exception:
        r.ok("A1-6 CLI argv (torchvision absent, SKIPPED)")
        return
    d = tempfile.mkdtemp()
    try:
        # a known input PNG (uint8), a trivial program
        img = (torch.rand(1, 12, 16, 3) * 255).to(torch.uint8)
        in_png = os.path.join(d, "in.png")
        tvio.write_png(img[0].permute(2, 0, 1), in_png)
        prog = os.path.join(d, "p.tex")
        open(prog, "w", encoding="utf-8").write("@OUT = vec4(@image.rgb * 0.5, 1.0);")
        out_png = os.path.join(d, "out.png")

        # (1) happy path through main(argv) — no exception, output written + correct.
        tex_cli.main(["run", prog, "--in", in_png, "--out", out_png, "--device", "cpu"])
        assert os.path.exists(out_png), "main() did not write --out"
        dec = tvio.decode_image(tvio.read_file(out_png)).float() / 255.0
        exp = (img[0].permute(2, 0, 1).float() / 255.0) * 0.5
        assert (dec[:3] - exp).abs().max().item() < 1.5 / 255, "CLI output wrong (>1 quantum)"
        r.ok("main(['run', ...]) round-trips a PNG within the 8-bit quantum")

        # (2) missing required flag -> argparse SystemExit(2)
        try:
            tex_cli.main(["run", prog])
            r.fail("A1-6 bad-args", "missing --in/--out did not exit")
        except SystemExit as se:
            r.ok(f"missing required flag exits (code {se.code})")

        # (3) --help -> SystemExit(0)
        try:
            tex_cli.main(["--help"])
            r.fail("A1-6 --help", "--help did not exit")
        except SystemExit as se:
            r.ok(f"--help exits cleanly (code {se.code})")
    except Exception as e:
        r.fail("A1-6 CLI argv", f"{type(e).__name__}: {e}")
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_c4st_js_loc_ratchet(r: SubTestResult):
    print("\n--- C4-st: js/tex_extension.js LOC ratchet ---")
    # doc 34 weakness: the single-file frontend has no LOC governance (REG-2 watches
    # only .py). Grandfather the current size with modest headroom; further growth reds
    # the suite (the on-ramp to the deferred v0.20 JS decomposition).
    # v0.21 (FUS-1): +~120 lines for DAG-region serialize/collapse. This growth is
    # explicitly temporary — SCHED-1 (roadmap, mid-term) moves fusion detection OUT of
    # the JS into Python (the JS becomes a thin route caller), which will shrink this
    # back below the pre-FUS-1 line; the budget is raised with that plan of record.
    # v0.23 (Authoring): +~100 lines across LANG-1 (param-metadata widgets), LANG-2
    # (debounced live-lint fetch → CM6 setDiagnostics) and LANG-5 (server-backed snippet
    # store sync). All frontend-facing; the decomposition on-ramp still stands.
    # v0.23 (LANG-5 data-loss hardening): +~86 lines for the pending-set + non-destructive,
    # preservation-favoring server-truth merge (a rejected/offline save is retried, not
    # silently discarded; a transient server read-error no longer wipes the cache) plus the
    # single-flight POST chain (concurrent whole-map writes land in order). Still
    # frontend-only; the SCHED-1 decomposition that shrinks this file is unaffected.
    # v0.26 (TOOL-2): +~50 lines for the "Publish as TEX tool…" node command (parse the
    # program's params → a single-stage .textool manifest → POST /tex_wrangle/publish_tool).
    # Frontend-only; the backend validates schema-first (tex_tool). The SCHED-1 decomposition
    # on-ramp still stands.
    # v0.26 audit#6 (#5): +~12 lines so publish records the true socket type (m@→MASK, l@→LATENT)
    # from each binding's prefix instead of flattening every input/output to IMAGE. Still
    # frontend-only; the SCHED-1 file split remains the real remedy — this bump buys that plan time.
    # v0.29 (FUS-1c): +~13 lines for the linear/region coordination — region detection split into
    # detect + apply, and REGIONS now run first so the linear pass coordinates by observation
    # (_texCollapseOne's own "every chain node still present" check) instead of a predicted
    # skip-set. Frontend-only pass reordering; the SCHED-1 decomposition (fusion detection already
    # lives in Python) still stands as the file-shrinking remedy — this bump buys that plan time.
    JS_HARD = 4290
    js = _PKG / "js" / "tex_extension.js"
    if not js.exists():
        r.fail("C4-st JS ratchet", "js/tex_extension.js missing")
        return
    n = sum(1 for _ in js.open(encoding="utf-8"))
    if n > JS_HARD:
        r.fail("C4-st JS ratchet", f"tex_extension.js is {n} lines > {JS_HARD} — split by "
               "concern (editor/fusion/lazy/HUD/preflight) or raise the budget with a plan")
    else:
        r.ok(f"tex_extension.js {n} lines <= budget {JS_HARD} (ratchet holds)")


def test_c1st_execute_line_budget(r: SubTestResult):
    print("\n--- C1-st: execute() per-function line budget (anti-regrowth tripwire) ---")
    # doc 34 weakness #10: execute() regrew 276->395 across v0.18 (the v0.17 decomposition
    # silently reversed) because REG-2 watches MODULES, not functions. This is the missing
    # per-function tripwire doc 27 asked for.
    # RATCHET, v0.22 (ENG-1): execute() is now marshal-in -> engine.prepare/run ->
    # marshal-out and measures 205 lines (was 376 at v0.21). The budget drops 385 -> 240 to
    # LOCK THAT IN — the cook belongs in tex_engine now, so any regrowth here is the
    # ComfyUI adapter re-absorbing engine logic, which is exactly the S-1 drift to catch.
    import ast
    src = (_PKG / "tex_node.py").read_text(encoding="utf-8")
    BUDGET = 240
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "execute":
            span = node.end_lineno - node.lineno
            if span > BUDGET:
                r.fail("C1-st execute budget",
                       f"execute() is {span} lines > budget {BUDGET} — the cook moved to "
                       "tex_engine (ENG-1); marshalling logic that grows past this budget "
                       "usually means engine logic leaked back into the ComfyUI adapter")
            else:
                r.ok(f"execute() {span} lines <= budget {BUDGET} (376 at v0.21, pre-ENG-1)")
            return
    r.fail("C1-st execute budget", "execute() not found in tex_node.py")


def test_c2st_fp16_taxonomy_federated(r: SubTestResult):
    print("\n--- C2-st: fp16 taxonomy federated into the registry (single source) ---")
    from TEX_Wrangle.tex_runtime import precision_policy as pp
    from TEX_Wrangle.tex_runtime import stdlib_registry as R
    import TEX_Wrangle.tex_runtime.stdlib  # noqa: populate REGISTRY
    try:
        # (a) precision_policy's sets ARE the registry's (single source, not a copy).
        assert pp._FP16_FRAGILE_FNS is R.FP16_FRAGILE, "fragile set not registry-derived"
        assert pp._BOUNDED_FNS is R.FP16_BOUNDED, "bounded set not registry-derived"
        # (b) no drift: every classified name is a real registered stdlib fn.
        names = set(R.functions().keys())
        stale = [n for n in (R.FP16_FRAGILE | R.FP16_BOUNDED) if n not in names]
        assert not stale, f"classified fp16 names that aren't registered fns: {stale}"
        # (c) loud guard: no registered fn LOOKS fp16-fragile yet is unclassified.
        cand = R.unclassified_fragile_candidates()
        assert not cand, (f"unclassified fp16-fragile-looking fns: {cand} — tag each "
                          "FP16_FRAGILE/BOUNDED in stdlib_registry or confirm it's safe")
        # (d) G4: the impl guard resolves one level of delegation (`under` -> `over`), and
        # scans only the body (not `ex=` decorator text).
        under = next(e.fn for e in R.REGISTRY if "under" in e.names)
        assert R._impl_looks_fragile(under), "impl guard must catch the `under`->`over` delegator"
        r.ok(f"fp16 taxonomy single-sourced ({len(R.FP16_FRAGILE)} fragile, "
             f"{len(R.FP16_BOUNDED)} bounded); no drift; loud guard clean (+delegation)")
    except Exception as e:
        r.fail("C2-st federation", f"{type(e).__name__}: {e}")


def test_c3st_gm_rules(r: SubTestResult):
    print("\n--- C3-st: per-rule units for the precision gate (_gm / resolve) ---")
    from TEX_Wrangle.tex_runtime import precision_policy as pp
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

    def gate(code):
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        # px sits AT the per-arch fp16 floor (S-5) — this test exercises the
        # amplify/fragile/bounded RULES, not the resolution gate, so it must
        # stay above the floor on every verified arch (2048² on sm_120).
        return pp.resolve_auto_precision(prog, pp._MIN_FP16_PX, "cuda")[0]

    cases = [
        # (program, expected precision, why)
        ("@OUT = vec4(vec3(@A.r * 0.5 + 0.1), 1.0);", "fp16", "small pointwise scale — accepted"),
        ("@OUT = vec4(vec3(@A.r * 50.0), 1.0);", "fp32", "gain 50 >= _AMP — declined"),
        ("@OUT = vec4(vec3(degrees(@A.r)), 1.0);", "fp32", "degrees ~57x amplify — declined (A1-1 find)"),
        ("@OUT = vec4(vec3(radians(@A.r) * 100.0), 1.0);", "fp16", "radians de-amplifies (0.0175x) — stays low-gain"),
        ("@OUT = vec4(vec3(log10(@A.r)), 1.0);", "fp32", "log10 fragile — declined (C2-st find)"),
        ("@OUT = vec4(vec3(sin(@A.r * 3.0)), 1.0);", "fp16", "bounded sin at small arg — accepted"),
        ("float f(float x){ return x * 50.0; } @OUT = vec4(vec3(f(@A.r)), 1.0);", "fp32", "user-fn amplifier — declined (F1)"),
    ]
    fails = []
    for code, want, why in cases:
        try:
            got = gate(code)
            if got != want:
                fails.append(f"want {want} got {got}: {why}")
        except Exception as e:
            fails.append(f"{type(e).__name__} on '{why}': {e}")
    if fails:
        r.fail("C3-st gate rules", "; ".join(fails))
    else:
        r.ok(f"all {len(cases)} per-rule gate cases correct (amplify/fragile/bounded/user-fn)")


def test_a1_2_fusion_lazy_precision_tiers(r: SubTestResult):
    print("\n--- A1-2: fusion x lazy x precision x tier combined e2e ---")
    torch.manual_seed(3)
    # 2-stage chain: stage0 grades @A; the terminal reads the chain source @X and adds.
    # (Pointwise so the fp32 reference is exact and auto/cuda_graph are eligible.)
    src = torch.rand(1, 64, 64, 3)
    stage0_code = "@OUT = @A * 1.2 - 0.05;"          # vec3 -> vec3
    terminal = "@OUT = vec4(@X * 0.9 + 0.05, 1.0);"   # @X is the vec3 chain result
    spec = {
        "stages": [{"code": stage0_code, "image_input": "A", "params": {}}],
        "terminal_image_input": "X",
    }
    chain = json.dumps(spec)

    # Reference: the SAME math unfused, fp32 interpreter (stage0 then terminal).
    def _run(code, bt, binds, prec="fp32"):
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        return Interpreter().execute(prog, binds, tm, device="cpu",
                                     output_names=["OUT"], precision=prec)["OUT"]
    stage0 = _run(stage0_code, {"A": TEXType.VEC3, "OUT": TEXType.VEC3}, {"A": src})
    ref = _run(terminal, {"X": TEXType.VEC3, "OUT": TEXType.VEC4}, {"X": stage0})

    # (a) fused chain through the REAL node path, CPU, compile_mode none + auto-precision.
    try:
        cases = [("cpu", "none", "fp32"), ("cpu", "none", "auto")]
        if torch.cuda.is_available():
            cases += [("cuda", "cuda_graph", "fp32"), ("cuda", "auto", "auto")]
        worst = 0.0
        # the node returns a 3-channel IMAGE clamped to [0,1] (the IMAGE contract);
        # match it so the comparison tests the fusion math, not the output convention.
        ref3 = ref.float()[..., :3].clamp(0.0, 1.0)
        for dev, cm, prec in cases:
            out = _first(_N.execute(code=terminal, X=(src.cuda() if dev == "cuda" else src),
                                    device=dev, compile_mode=cm, precision=prec, _tex_chain=chain))
            worst = max(worst, (out.float().cpu()[..., :3] - ref3).abs().max().item())
        if worst < 1e-5:
            r.ok(f"fused chain bit-exact across {len(cases)} tier/precision combos (maxdiff {worst:.1e})")
        else:
            r.fail("A1-2 fusion tiers", f"fused output diverged from unfused fp32 ref: {worst:.2e}")
    except Exception as e:
        r.fail("A1-2 fusion tiers", f"{type(e).__name__}: {e}")

    # (b) lazy is disabled under fusion (v1 rule): check_lazy_status with a _tex_chain
    # present must request EVERYTHING (no pruning of a source the splice folded away).
    try:
        # a slot map wiring two inputs; in_1 is uncooked (None). Without a chain,
        # a terminal that references only @in_0 would prune @in_1; WITH a chain the
        # v1 rule forces _pending() (every uncooked wired slot), so in_1 must appear.
        slot_map = json.dumps([
            {"name": "X", "slot": "in_0", "type": "IMAGE"},
            {"name": "Y", "slot": "in_1", "type": "IMAGE"},
        ])
        # control: NO chain, terminal references only @X -> in_1 (Y) is pruned.
        no_chain = _N.check_lazy_status(code="@OUT = vec4(@X.rgb, 1.0);",
                                        _tex_slot_map=slot_map, in_0=src, in_1=None)
        # fused: chain present -> everything requested (in_1 present despite being unused).
        fused = _N.check_lazy_status(_tex_chain=chain, _tex_slot_map=slot_map,
                                     in_0=src, in_1=None)
        if "in_1" not in (no_chain or []) and "in_1" in (fused or []):
            r.ok("lazy prunes unused input normally; fusion requests all (v1 rule pinned)")
        else:
            r.fail("A1-2 lazy-under-fusion",
                   f"expected in_1 pruned without chain ({no_chain}) but kept with chain ({fused})")
    except Exception as e:
        r.fail("A1-2 lazy-under-fusion", f"{type(e).__name__}: {e}")
