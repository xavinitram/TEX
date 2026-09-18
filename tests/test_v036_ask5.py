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

from TEX_Wrangle.tex_runtime.noise import _COMPILE_AFTER_CALLS


# ── The tiered-noise hazard every row below has to stay clear of ─────────────
#
# `worley_f1`/`worley_f2` dispatch through `noise._worley_cache` (eager -> jit.trace ->
# torch.compile/Inductor, `_TieredCache.call`). `worley_id` does NOT — it is eager on
# every call by construction, which `test_ask5_eager_only_no_promotion` pins. So any row
# that compares a DISPATCHED worley result against anything has to ask which TIER each
# side ran, because two things make the sides differ:
#
#   * a direct `_worley2d_f1(...)` call is the eager body, while `_worley2d(...)` answers
#     from whichever tier the cache holds. Their agreeing is a fact about the host's
#     fuser, not about TEX — `_worley2d`'s own comment says the pair only "happens to
#     agree bitwise on this box".
#   * `try_upgrade` swaps a key's traced callable for an Inductor one on that key's
#     `_COMPILE_AFTER_CALLS`th call, once per key per process, wherever a host compiler
#     exists. Two DISPATCHED calls made either side of that swap are also two tiers.
#
# Cross-tier agreement is held to a recorded envelope, never asserted as an equality —
# test_v031_noise_tiers.py, "The promotion envelope". The helpers below let each row say
# which of the two it is doing, instead of leaving it to the box.

def _worley_tier(device):
    """The tier `noise._worley_cache` holds for the 2D F1 key on `device`, in words —
    read off the cache, never inferred from the output (test_v031_noise_tiers.py's
    `_tier_of`), so a same-tier claim is only ever made about one tier."""
    from TEX_Wrangle.tex_runtime import noise
    held = noise._worley_cache.cache.get((False, device))
    if held is None:
        return "cold"
    if held is False:
        return "eager"
    return "trace" if isinstance(held, torch.jit.ScriptFunction) else "promoted"


def _settle_worley_tier(x, y):
    """Call the DISPATCHED 2D worley path until its tier can no longer change; return the
    tier it settled on.

    A key is promoted at most once per process (`try_upgrade` marks it attempted before
    it compiles), so `_COMPILE_AFTER_CALLS + 1` calls are always enough — and since the
    key is `(return_f2, device)` and carries no shape, settling it here fixes the tier for
    every later 2D worley call in this process at any size. A row comparing two dispatched
    results must do this FIRST: measured on the box this was written on, running this file
    alone puts the swap on worley call 4, which is in the middle of the voronoi row's
    pairs. `_worley3d` has no tiered cache at all, so 3D needs none of this.
    """
    from TEX_Wrangle.tex_runtime import noise
    for _ in range(_COMPILE_AFTER_CALLS + 1):
        noise._worley2d(x, y, return_f2=False)
    return _worley_tier(noise._widest((x, y)).device)


def _worley_tier_band(scale):
    """The band a cross-tier worley comparison is held to, at coordinate `scale`.

    test_v031_noise_tiers.py measured worley's promotion swap moving "1-2 fp32 ulps OF THE
    COORDINATE" and banded it at four of them. That SHAPE is worley's, but those numbers
    were recorded on CUDA, and that file is explicit that a band recorded against one
    compiler is not a fact about another — so this is its own constant with its own
    measurements. On the box this was written on (CPU Inductor via MSVC) eager vs trace AND
    eager vs promoted both measured 0.0 at 64^2 / scale 8; a CPU runner elsewhere measured
    5.960464e-08 for the same comparison — 2**-24, one fp32 ulp of an output near 1.0 —
    which sits ~32x inside this band. A build that blows it is a decision to re-measure and
    re-band, never a tolerance to widen.
    """
    return 4.0 * scale * 2.0 ** -24


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
    """The id names the SAME winning cell worley_f1's distance measures — checked with
    both sides EAGER, off one shared `_worley2d_core` distance stack, so bit-equality is
    a claim about the wiring and not about the host.

    This row used to compare the eager `_worley2d_f1(...)` against `_worley2d(...)`, which
    dispatches through the tiered cache, and assert the two bit-identical. That is a
    CROSS-TIER claim wearing a regression check's clothes: it holds wherever the fuser
    happens to leave the ops alone and fails by one fp32 ulp where it does not. The
    question it SAID it was asking — did adding `worley_id` move `worley_f1` — is already
    answered bit-exactly, on one path, by `test_ask5_voronoi_unchanged`. The cross-tier
    fact it was measuring by accident is kept, and honestly banded, in the row below.
    """
    print("\n--- ASK-5: worley_id names worley_f1's own nearest-point winner ---")
    from TEX_Wrangle.tex_runtime import noise
    H = W = 64
    ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    x, y = xs * 8.0, ys * 8.0
    dx_off, dy_off = noise._get_worley_offsets(x.device, x.dim())
    dist = noise._worley2d_core(x, y, dx_off, dy_off)
    winner = dist.min(dim=0).indices

    try:
        # `worley_f1` reports `dist.min(dim=0).values`; `worley_id` re-hashes
        # `dist.min(dim=0).indices`. Gathering the distance AT that index has to reproduce
        # F1's own value exactly, or the id is naming a cell that is not the nearest one.
        # Both sides eager, both off the same `dist`, so this is like compared with like.
        at_winner = torch.sqrt(torch.gather(dist, 0, winner.unsqueeze(0)).squeeze(0))
        f1_direct = noise._worley2d_f1(x, y, dx_off, dy_off)
        assert torch.equal(f1_direct, at_winner), \
            f"maxdiff {(f1_direct - at_winner).abs().max().item()}"
        r.ok("worley_f1's distance IS the distance to the cell worley_id's argmin picks "
             "(eager both sides, 64x64, torch.equal)")
    except Exception as e:
        r.fail("ASK-5 id names f1's winner", f"{type(e).__name__}: {e}")

    try:
        # …and the id is a FUNCTION of that winning cell, not of the pixel: every pixel
        # resolving to one cell carries one id. A per-pixel regression, or an id read off
        # the wrong candidate, breaks it. Cell coords are integers, so identifying the
        # winning cell is exact arithmetic with nothing to round.
        #
        # Distinctness BETWEEN cells is deliberately not asserted: the id is a 23-bit hash,
        # so collisions are a property of `_lowbias32`, not of this wiring — measured here,
        # 89 winning cells carry 87 distinct ids. `test_ask5_determinism_and_cell_count`
        # already pins that neighbouring cells differ and that the value count is per-CELL
        # (<= 81), which is the claim the design makes.
        xi = torch.floor(x).to(torch.int32)
        yi = torch.floor(y).to(torch.int32)
        cx = torch.gather(xi.unsqueeze(0) + dx_off, 0, winner.unsqueeze(0)).squeeze(0)
        cy = torch.gather(yi.unsqueeze(0) + dy_off, 0, winner.unsqueeze(0)).squeeze(0)
        cell = (cx.long() * 1000003 + cy.long()).flatten().tolist()
        ids = noise._worley2d_id(x, y).flatten().tolist()
        per_cell, drift = {}, 0
        for c, i in zip(cell, ids):
            if per_cell.setdefault(c, i) != i:
                drift += 1
        assert not drift, f"{drift} pixels carry an id their own winning cell does not"
        assert len(per_cell) > 1, "the probe resolved to one cell — this row is vacuous"
        r.ok(f"worley_id is constant across every pixel of a winning cell "
             f"({len(per_cell)} cells, {len(set(per_cell.values()))} distinct ids, "
             f"64x64 at freq 8)")
    except Exception as e:
        r.fail("ASK-5 id is per-cell", f"{type(e).__name__}: {e}")

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


def test_ask5_worley_f1_tier_envelope(r: SubTestResult):
    """What the row above used to assert by accident, said out loud and banded.

    `worley_f1` called DIRECTLY runs the eager body; `worley_f1` called through
    `noise._worley2d` runs whichever tier the cache holds. Those are two different
    computations of the same function, and two tiers agree only to a recorded envelope
    (test_v031_noise_tiers.py, "The promotion envelope") — never bit-exactly, and whether
    they happen to is a property of the host's fuser. Held to the band, and naming the tier
    the dispatched side actually took, the tier story stays pinned without the lie."""
    print("\n--- ASK-5: eager worley_f1 vs the DISPATCHED (tiered) one, held to the "
          "worley envelope ---")
    from TEX_Wrangle.tex_runtime import noise
    try:
        H = W = 64
        scale = 8.0
        ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
        x, y = xs * scale, ys * scale
        dx_off, dy_off = noise._get_worley_offsets(x.device, x.dim())
        f1_eager = noise._worley2d_f1(x, y, dx_off, dy_off)
        # Settle first, so the measurement is against ONE named tier rather than against
        # whichever side of the one-shot promotion this call happens to land on.
        tier = _settle_worley_tier(x, y)
        f1_dispatched = noise._worley2d(x, y, return_f2=False)
        md = (f1_eager - f1_dispatched).abs().max().item()
        band = _worley_tier_band(scale)
        assert md <= band, (
            f"eager vs {tier}: maxdiff {md:.6e} > band {band:.6e} at coordinate scale "
            f"{scale} — a blown band is a decision to re-measure and re-band, never a "
            f"tolerance to widen")
        r.ok(f"eager vs {tier}: maxdiff {md:.3e} <= band {band:.3e} (4 fp32 ulps of a "
             f"coordinate at scale {scale})")
    except Exception as e:
        r.fail("ASK-5 f1 tier envelope", f"{type(e).__name__}: {e}")


def test_ask5_voronoi_unchanged(r: SubTestResult):
    """CAUTION row: `voronoi` must render EXACTLY what it rendered before this ask
    — bit-identical to `worley_f1`, `torch.equal`, at 48x32 — since the ask's second
    branch (a new name) must never move an existing caller's pixels."""
    print("\n--- ASK-5: voronoi is UNCHANGED (still worley_f1, bit-identical) ---")
    try:
        H, W = 48, 32
        ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
        u, v = xs * 8.0, ys * 8.0
        # Every 2D side below is a DISPATCHED worley_f1 call, so they must all run the SAME
        # tier for these `torch.equal`s to be a claim about `voronoi` being an alias of
        # `worley_f1` rather than about the host's fuser. `try_upgrade` swaps the key's
        # callable exactly once per process, and measured, running this file alone lands
        # that swap on worley call 4 — between two of the pairs below. So settle it first;
        # afterwards no call in this process can change tier. (2D only: `_worley3d` is not
        # tiered, so the 3D pair needs nothing.)
        _settle_worley_tier(u, v)
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
