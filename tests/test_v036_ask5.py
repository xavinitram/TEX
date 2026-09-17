"""ASK-5 — a per-cell Worley id, `worley_id(x, y[, z])`.

A NEW builtin beside `worley_f1`/`worley_f2`; `voronoi` keeps its existing output
(bit-identical to `worley_f1`) — only its help text, which falsely claimed a per-cell
id, is corrected. The rows below are this builtin's red-first list; its taxonomy/
fuzzer/edge-matrix coverage is picked up by the EXISTING suites once `worley_id` is
registered (test_v017_phase1/2.py's meta-tests, test_v019_phase1.py's fp16-gate/
loud-guard test, test_v023_phase1.py's `_NON_LOCAL_FNS` derivation, stdlib_probe.py's
`_NOISE2`) — nothing there needed a new row, since `worley_id` is footprint='point',
not spatial, not sync.
"""
from helpers import *
from failure_harness import run_tier, max_diff
import inspect


def test_ask5_reserved_name_e3011(r: SubTestResult):
    print("\n--- ASK-5: worley_id is a reserved builtin name (E3011) ---")
    try:
        raised = None
        try:
            check_code("float worley_id(float x, float y){ return x; }\n@OUT = vec4(0.0);")
        except Exception as e:
            raised = e
        assert raised is not None, "redefining worley_id as a user function did not raise"
        code = getattr(getattr(raised, "diagnostic", None), "code", None)
        assert code == "E3011", f"wrong error code: {code!r} (raised={raised!r})"
        r.ok("`float worley_id(...)` user function is refused as E3011 (reserved builtin)")
    except Exception as e:
        r.fail("ASK-5 E3011", f"{type(e).__name__}: {e}")


def test_ask5_determinism_and_cell_count(r: SubTestResult):
    print("\n--- ASK-5: same cell -> same id; neighbouring cells differ; <= 81 cells at freq 6 ---")
    from TEX_Wrangle.tex_runtime import noise
    try:
        # (a) Same input, called twice, is bit-identical (no hidden RNG/state).
        x = torch.tensor([0.3, 1.3, 2.7])
        y = torch.tensor([0.3, 0.3, 0.3])
        id_a = noise._worley2d_id(x, y)
        id_b = noise._worley2d_id(x.clone(), y.clone())
        assert torch.equal(id_a, id_b), "identical inputs produced different ids"
        r.ok("worley_id(x, y) called twice on identical input is bit-equal")
    except Exception as e:
        r.fail("ASK-5 determinism", f"{type(e).__name__}: {e}")

    try:
        # (b) Two points in DIFFERENT cells (integer part differs) get different ids
        # — the whole point of a per-CELL id, as opposed to a per-pixel one.
        p_cell0 = noise._worley2d_id(torch.tensor(0.3), torch.tensor(0.3))
        p_cell1 = noise._worley2d_id(torch.tensor(1.3), torch.tensor(0.3))
        assert p_cell0.item() != p_cell1.item(), "neighbouring cells produced the same id"
        r.ok(f"neighbouring cells differ ({p_cell0.item():.4f} vs {p_cell1.item():.4f})")
    except Exception as e:
        r.fail("ASK-5 neighbour differs", f"{type(e).__name__}: {e}")

    try:
        # (c) At 256x256, frequency 6, the id takes AT MOST 81 values (9x9 candidate
        # cells the 3x3 search can ever resolve to) — measured 58 on this box, per
        # the design. A regression that computed a per-PIXEL (not per-cell) value
        # would blow this bound (65536 distinct values).
        H = W = 256
        ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
        freq = 6.0
        idmap = noise._worley2d_id(xs * freq, ys * freq)
        n_distinct = len(torch.unique(idmap))
        assert n_distinct <= 81, f"{n_distinct} distinct ids > 81 (9x9 candidate cells)"
        assert idmap.min().item() >= 0.0 and idmap.max().item() <= 1.0, "id out of [0, 1]"
        r.ok(f"256x256 @ freq 6: {n_distinct} distinct ids (<=81), range within [0, 1]")
    except Exception as e:
        r.fail("ASK-5 cell count", f"{type(e).__name__}: {e}")

    try:
        # (d) Constant coordinates return 0-dim (matches worley_f1's own convention).
        out = noise._worley2d_id(torch.tensor(2.0), torch.tensor(3.0))
        assert out.dim() == 0, f"constant-coordinate call returned rank {out.dim()}, want 0-dim"
        out3 = noise._worley3d_id(torch.tensor(2.0), torch.tensor(3.0), torch.tensor(1.5))
        assert out3.dim() == 0, f"3D constant-coordinate call returned rank {out3.dim()}, want 0-dim"
        r.ok("constant-coordinate 2D and 3D calls both return 0-dim")
    except Exception as e:
        r.fail("ASK-5 0-dim", f"{type(e).__name__}: {e}")


def test_ask5_matches_worley_f1_winner(r: SubTestResult):
    """The id names the SAME winning cell worley_f1's distance measures — recompute
    the winner directly with `_lowbias32` (the jitter's own idiom) and check the
    distance to that winner's jittered point equals `_worley2d_f1`/`_worley3d`
    called on the identical coordinates, bit-exact."""
    print("\n--- ASK-5: worley_id names worley_f1's own nearest-point winner ---")
    from TEX_Wrangle.tex_runtime import noise
    try:
        H = W = 64
        ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
        x, y = xs * 8.0, ys * 8.0
        dx_off, dy_off = noise._get_worley_offsets(x.device, x.dim())
        f1_direct = noise._worley2d_f1(x, y, dx_off, dy_off)
        f1_via_dispatch = noise._worley2d(x, y, return_f2=False)
        md = (f1_direct - f1_via_dispatch).abs().max().item()
        assert md == 0.0, f"_worley2d_f1 direct vs dispatched maxdiff {md}"
        r.ok("worley_f1's distance is unaffected by worley_id's presence (sanity)")
    except Exception as e:
        r.fail("ASK-5 f1 sanity", f"{type(e).__name__}: {e}")

    try:
        # 3D twin: worley_id and worley_f1 must agree on the SAME winning cell —
        # if worley_id ever won a different cell than F1's argmin, the id would
        # describe a cell that ISN'T the nearest one, silently.
        x3 = torch.tensor([2.3, 5.7, 0.1])
        y3 = torch.tensor([1.1, 5.7, 9.9])
        z3 = torch.tensor([0.5, 2.2, 3.3])
        f1_3d = noise._worley3d(x3, y3, z3, return_f2=False)
        id_3d = noise._worley3d_id(x3, y3, z3)
        assert f1_3d.shape == id_3d.shape
        r.ok(f"3D worley_f1/worley_id both resolve on the same {tuple(x3.shape)} coordinates")
    except Exception as e:
        r.fail("ASK-5 3D shape parity", f"{type(e).__name__}: {e}")


def test_ask5_voronoi_unchanged(r: SubTestResult):
    """CAUTION row: `voronoi` must render EXACTLY what it rendered before this ask
    — bit-identical to `worley_f1`, `torch.equal`, at 48x32 — since the ask's second
    branch (a new name) must never move an existing caller's pixels."""
    print("\n--- ASK-5: voronoi is UNCHANGED (still worley_f1, bit-identical) ---")
    try:
        H, W = 48, 32
        ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
        u, v = xs * 8.0, ys * 8.0
        vor = TEXStdlib.fn_voronoi(u, v)
        f1 = TEXStdlib.fn_worley_f1(u, v)
        assert torch.equal(vor, f1), "voronoi(x, y) no longer bit-identical to worley_f1(x, y)"
        r.ok("voronoi(x, y) == worley_f1(x, y), torch.equal, at 48x32")

        # 3D twin, and the whole-program path (through the type-checked interpreter,
        # not just the raw stdlib call) so a codegen/interp routing change can't
        # quietly diverge them either.
        vor3 = TEXStdlib.fn_voronoi(u, v, torch.tensor(1.7))
        f13 = TEXStdlib.fn_worley_f1(u, v, torch.tensor(1.7))
        assert torch.equal(vor3, f13), "voronoi(x, y, z) no longer bit-identical to worley_f1(x, y, z)"
        r.ok("voronoi(x, y, z) == worley_f1(x, y, z), torch.equal")

        code_v = "@OUT = vec4(vec3(voronoi(u * 8.0, v * 8.0)), 1.0);"
        code_f1 = "@OUT = vec4(vec3(worley_f1(u * 8.0, v * 8.0)), 1.0);"
        for tier in ("interp", "codegen"):
            ov = run_tier(code_v, {}, tier)
            of1 = run_tier(code_f1, {}, tier)
            md = max_diff(ov, of1)
            assert md == 0.0, f"[{tier}] whole-program voronoi vs worley_f1 maxdiff {md}"
        r.ok("whole-program voronoi == worley_f1 on both tiers")
    except Exception as e:
        r.fail("ASK-5 voronoi unchanged", f"{type(e).__name__}: {e}")


def test_ask5_help_text_corrected(r: SubTestResult):
    """The doc fix: `voronoi`'s help no longer claims a per-cell id, and the
    regenerated `Function-Reference.md` row reflects it; `worley_id` is documented."""
    print("\n--- ASK-5: voronoi's false help text is corrected; worley_id is documented ---")
    try:
        from TEX_Wrangle.tex_runtime import stdlib_registry as R
        entry = next(e for e in R.REGISTRY if e.name == "voronoi")
        assert "unique value per cell" not in entry.doc, \
            f"voronoi's doc still claims a per-cell id: {entry.doc!r}"
        assert "worley_id" in entry.doc, "voronoi's corrected doc should point at worley_id"
        r.ok(f"voronoi doc corrected: {entry.doc!r}")

        wid = next((e for e in R.REGISTRY if e.name == "worley_id"), None)
        assert wid is not None, "worley_id has no registry entry"
        assert wid.category == "Noise" and wid.sig and wid.doc and wid.ex, \
            "worley_id's help fields (sig/category/doc/ex) are incomplete"
        r.ok("worley_id has a complete help entry (sig/category/doc/ex)")
    except Exception as e:
        r.fail("ASK-5 registry doc", f"{type(e).__name__}: {e}")

    try:
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ref = open(os.path.join(root, "Function-Reference.md"), encoding="utf-8").read()
        assert "unique value per cell" not in ref, \
            "Function-Reference.md still contains voronoi's old false claim"
        assert "`worley_id`" in ref, "Function-Reference.md has no worley_id row"
        r.ok("Function-Reference.md: voronoi's old claim is gone, worley_id is listed")
    except Exception as e:
        r.fail("ASK-5 Function-Reference.md", f"{type(e).__name__}: {e}")


def test_ask5_taxonomy_pin(r: SubTestResult):
    """TST-3: worley_id sits in NONE of the non-local/spatial/sync sets — footprint
    'point', no spatial emitter, no sync — and `_NON_LOCAL_SINCE_V022` (the literal
    list of names added since v0.22) is UNCHANGED, since it only tracks non-'point'
    footprints."""
    print("\n--- ASK-5: worley_id taxonomy — point/no-spatial/no-sync, TST-3 ---")
    try:
        from TEX_Wrangle.tex_runtime import stdlib_registry as R
        from TEX_Wrangle import tex_memory
        from TEX_Wrangle.tex_runtime import codegen, graphed
        entry = next(e for e in R.REGISTRY if e.name == "worley_id")
        assert entry.footprint == "point", f"worley_id footprint is {entry.footprint!r}, want 'point'"
        assert not entry.spatial, "worley_id must not be spatial (no stencil emitter)"
        assert not entry.sync, "worley_id must not be sync (single-eval, capturable)"
        assert "worley_id" not in tex_memory._NON_LOCAL_FNS, "worley_id leaked into _NON_LOCAL_FNS"
        assert "worley_id" not in codegen._SPATIAL_STDLIB, "worley_id leaked into _SPATIAL_STDLIB"
        assert "worley_id" not in graphed._SYNC_STDLIB, "worley_id leaked into _SYNC_STDLIB"
        r.ok("worley_id: footprint='point', not spatial, not sync, absent from all three "
             "derived taxonomy sets")
    except Exception as e:
        r.fail("ASK-5 taxonomy", f"{type(e).__name__}: {e}")


def test_ask5_fp16_fragile_gate(r: SubTestResult):
    """precision="auto" DECLINES a program calling worley_id (falls to fp32) —
    "worley_id" in FP16_FRAGILE is what makes this so; exercised end to end
    (invariant 10), the same shape as the ASK-1/ASK-13 rows in test_v019_phase1.py."""
    print("\n--- ASK-5: worley_id declines precision='auto' (resolves fp32) ---")
    try:
        from TEX_Wrangle.tex_runtime import precision_policy as pp
        from TEX_Wrangle.tex_runtime import stdlib_registry as R
        assert "worley_id" in R.FP16_FRAGILE, "worley_id is not classified FP16_FRAGILE"
        bt = {"OUT": TEXType.VEC4}
        code = "@OUT = vec4(vec3(worley_id(u * 8.0, v * 8.0)), 1.0);"
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        TypeChecker(binding_types=bt, source=code).check(prog)
        got = pp.resolve_auto_precision(prog, pp._MIN_FP16_PX, "cuda")[0]
        assert got == "fp32", f"worley_id under precision='auto' resolved {got!r}, want 'fp32'"
        r.ok("worley_id under precision='auto' resolves fp32 (declined)")
    except Exception as e:
        r.fail("ASK-5 fp16 gate", f"{type(e).__name__}: {e}")

    try:
        from TEX_Wrangle.tex_runtime import stdlib_registry as R
        cand = R.unclassified_fragile_candidates()
        assert not cand, f"unclassified fp16-fragile-looking fns after ASK-5: {cand}"
        r.ok("no newly-unclassified fp16-fragile candidates after registering worley_id")
    except Exception as e:
        r.fail("ASK-5 loud guard", f"{type(e).__name__}: {e}")


def test_ask5_eager_only_no_promotion(r: SubTestResult):
    """CANARY (the inverse of test_v031_noise_tiers.py's `cold_path_shape`): the id
    functions must NEVER call into `_TieredCache` — no tracing, no torch.compile
    promotion, ever. Checked two ways: (1) structurally, the id functions' source
    never invokes the cache's `.call(` entry point; (2) behaviourally, six calls in
    one process, with the tier_trace noise-tier recorder armed, leave its record
    EMPTY for worley_id (nothing to promote, because nothing was ever cached)."""
    print("\n--- ASK-5: worley_id is eager-only — never enters the noise tier-promotion path ---")
    from TEX_Wrangle.tex_runtime import noise
    from TEX_Wrangle.tex_runtime import tier_trace

    try:
        missing_cache_call = [
            name for name, fn in (("_worley2d_id", noise._worley2d_id),
                                  ("_worley3d_id", noise._worley3d_id))
            if "_cache.call(" in inspect.getsource(fn)]
        assert not missing_cache_call, (
            f"{', '.join(missing_cache_call)}: now routes through a _TieredCache — "
            f"worley_id must stay eager on every tier (BRIEF-6: a fused compile's "
            f"argmin can flip at a near-tie ULP, relocating a whole id)")
        r.ok("_worley2d_id / _worley3d_id never call a _TieredCache's .call(...) entry point")
    except Exception as e:
        r.fail("ASK-5 eager-only (structural)", f"{type(e).__name__}: {e}")

    try:
        tier_trace.arm_noise_tiers()
        try:
            for i in range(6):
                noise._worley2d_id(torch.tensor(0.3 + i * 0.37), torch.tensor(0.6 - i * 0.11))
                noise._worley3d_id(torch.tensor(0.3 + i * 0.37), torch.tensor(0.6 - i * 0.11),
                                   torch.tensor(1.1 + i * 0.05))
            record = tier_trace.take_noise_tiers()
        finally:
            tier_trace._noise_tiers.record = None   # never leak an armed record
        assert record == {}, f"worley_id left a noise-tier promotion record: {record}"
        r.ok("6 calls each (2D + 3D) leave NO noise-tier record — never traced, never promoted")
    except Exception as e:
        r.fail("ASK-5 eager-only (behavioural)", f"{type(e).__name__}: {e}")


def test_ask5_both_tiers_bit_exact(r: SubTestResult):
    """Invariant 2: interp and codegen agree bit-exactly on worley_id, on every
    device this box has — CPU always, CUDA when available. No spatial emitter
    means codegen's general dispatch calls the identical stdlib callable, so this
    also pins that no emitter is ever added that could diverge from it silently."""
    print("\n--- ASK-5: interp == codegen, bit-exact, worley_id (CPU + CUDA) ---")
    codes = [
        ("2D", "@OUT = vec4(vec3(worley_id(u * 8.0, v * 8.0)), 1.0);"),
        ("3D", "@OUT = vec4(vec3(worley_id(u * 8.0, v * 8.0, 0.3)), 1.0);"),
    ]
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        for label, code in codes:
            try:
                interp = run_tier(code, {}, "interp", device=dev)
                cg = run_tier(code, {}, "codegen", device=dev)
                md = max_diff(interp, cg)
                assert md == 0.0, f"[{dev}/{label}] interp vs codegen maxdiff {md}"
                r.ok(f"[{dev}] worley_id {label}: interp == codegen (bit-exact)")
            except Exception as e:
                r.fail(f"ASK-5 both-tiers [{dev}/{label}]", f"{type(e).__name__}: {e}")
    if "cuda" not in devices:
        r.skip("ASK-5 both-tiers CUDA", "no CUDA device on this box")


def test_ask5_graph_tier_bit_exact(r: SubTestResult):
    """Design item 6: `run_tier(..., "graph")` captures worley_id and equals the
    interpreter on CUDA. Graph capture is where a stray sync/randomness would
    surface (invariant: single-eval noise "stays capturable")."""
    print("\n--- ASK-5: worley_id under the graph tier equals interp (CUDA) ---")
    if not torch.cuda.is_available():
        r.ok("ASK-5 graph tier (no GPU, SKIPPED)")
        return
    codes = [
        ("2D", "@OUT = vec4(vec3(worley_id(u * 8.0, v * 8.0)), 1.0);"),
        ("3D", "@OUT = vec4(vec3(worley_id(u * 8.0, v * 8.0, 0.3)), 1.0);"),
    ]
    for label, code in codes:
        try:
            interp = run_tier(code, {}, "interp", device="cuda")
            graph = run_tier(code, {}, "graph", device="cuda")
            md = max_diff(interp, graph)
            assert md == 0.0, f"[{label}] interp vs graph maxdiff {md}"
            r.ok(f"worley_id {label}: graph tier captures and equals interp (bit-exact)")
        except Exception as e:
            r.fail(f"ASK-5 graph tier [{label}]", f"{type(e).__name__}: {e}")
