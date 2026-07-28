"""v0.32 CACHE-7 — effort-based checkpoint placement.

Design note: docs/effort-based-checkpoints.md. What has to be pinned, and why each is a bug
the note names:

  * THE DIFFERENTIAL ORACLE — a checkpointed cook equals the straight-through fused cook
    BIT-EXACTLY, at every cut, on every device. This is the ship gate (roadmap §10.4), and it
    is what CACHE-6's own oracle asserts for one cut; CACHE-7 asserts it for N.
  * fp16 IS REFUSED (P0-2, v0.33) — and this bullet used to say the opposite. v0.32 lifted the
    fp32 gate on a 22-row measurement showing a split at fp16 bit-exact. The measurement was
    real and its conclusion was wrong: every row in its matrix produced an fp16-REPRESENTABLE
    boundary. The counterexample — a coordinate builtin amplified past fp16's precision AT the
    cut — diverges 9.00e-01 on both devices while the fp32 control is bit-exact. The row is
    inverted, not deleted, so the gate stays pinned in both directions.
  * THE RESOLUTION HOLE — `boundary_lineage_key` used to default `canvas` to nothing, so a
    tap's identity carried NO shape and `ResultCache.get` validates none. Reproduced: a 64²
    and a 128² cook minted the same key and the 128² request was SERVED THE 64² FRAME. This
    row is the regression.
  * PLACEMENT REFUSES RATHER THAN GUESSES — no profiler, too few samples, a DAG cut set, or a
    threshold under the materialization floor all yield NO checkpoints, i.e. today's cook.
    Placing on bad numbers spends memory on the wrong boundaries and reports success.
  * MULTI-TAP HARVEST — one cook exports every boundary (`@_tap_s{i}`), each equal to the
    standalone prefix cook, with `@OUT` unchanged by the tapping.
  * DEEPEST-FIRST — the serve path uses the deepest CACHED checkpoint, which is what makes an
    edit's cost depend on distance-to-checkpoint rather than on chain length.
  * INVARIANT #7 — the ComfyUI cook path neither imports nor reaches any of it.

Shapes: DIFFERENTIAL ORACLE (the equality rows), NEVER-SEVER ROWS (the placement refusals),
CANARY (the invariant-#7 contract), and one REGRESSION row for the resolution hole.
"""
import threading

from helpers import *

from TEX_Wrangle import tex_checkpoint as CK
from TEX_Wrangle import tex_engine, tex_fusion, tex_results
from TEX_Wrangle.tex_runtime import profile as _profile

_POOL = [
    "@OUT = vec4(@IN.rgb * 1.05, 1.0);",
    "@OUT = vec4(max(@IN.rgb - vec3(0.02), vec3(0.0)), 1.0);",
    "@OUT = gauss_blur(@IN, 4.0);",
    "@OUT = vec4(spow(@IN.rgb, vec3(0.95)), 1.0);",
    "float y = luma(@IN);\n@OUT = vec4(mix(vec3(y), @IN.rgb, 1.10), 1.0);",
    "@OUT = vec4((@IN.rgb - vec3(0.5)) * 1.08 + vec3(0.5), 1.0);",
]
# A pointwise-only pool, kept as the fallback for any environment whose fp16 halo kernels are
# missing (torch 2.5 has no CPU `replication_pad2d_channels_last` for Half; 2.10 does). The
# fp16 rows try the full pool first and fall back rather than skipping, so the coverage
# follows the toolchain instead of being permanently narrowed by the weakest one.
_POOL_POINTWISE = [c for c in _POOL if "gauss_blur" not in c]


def _fp16_pool(device):
    """The richest fp16 pool this box can actually cook — with the halo op if its kernels
    exist, pointwise otherwise."""
    try:
        tex_engine.cook_stage_list(
            [{"code": "@OUT = gauss_blur(@IN, 4.0);", "chain_input": None,
              "bindings": {"IN": torch.rand(1, 16, 16, 3, device=device)}}],
            device=device, precision="fp16")
        return _POOL, "with-halo"
    except Exception:
        return _POOL_POINTWISE, "pointwise-only"


def _stages(src, n, pool=None, tap_at=()):
    pool = pool or _POOL
    out = []
    for i in range(n):
        st = {"code": pool[i % len(pool)],
              "chain_input": (None if i == 0 else "IN"),
              "bindings": ({"IN": src} if i == 0 else {})}
        if i in tap_at:
            st["tap"] = True
        out.append(st)
    return out


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ── the ship gate: a checkpointed cook == the straight-through cook ──────────────────────

def test_v032_cache7_differential_oracle(r: SubTestResult):
    """N checkpoints, every cut, both devices: bit-exact against the full fused cook.

    Bit-exact (`torch.equal`) rather than a tolerance, because that is what CACHE-6's oracle
    already asserts for one cut — a suffix splice recomputes nothing differently, it just
    starts later. A tolerance here would hide a real divergence as rounding."""
    print("\n--- v0.32 CACHE-7: differential oracle (checkpointed == full) ---")
    N = 6
    for device in _devices():
        torch.manual_seed(0)
        src = torch.rand(1, 96, 96, 3, device=device)
        up = ("oracle-src",)
        full = tex_engine.cook_stage_list(_stages(src, N), device=device,
                                          precision="fp32")["OUT"]
        for cuts in ([2], [1, 3], [1, 2, 3, 4, 5]):
            cache = tex_results.ResultCache()
            done = CK.materialize(_stages(src, N), cache, device=device,
                                  precision="fp32", upstream=up, cuts=cuts)
            if sorted(done) != sorted(cuts):
                r.fail(f"CACHE-7 harvest {device} {cuts}",
                       f"materialized {done}, expected {cuts}")
                continue
            got = CK.cook_checkpointed(_stages(src, N), cache, device=device,
                                       precision="fp32", upstream=up, cuts=cuts)["OUT"]
            if torch.equal(got, full):
                r.ok(f"CACHE-7 oracle {device} cuts={cuts}: bit-exact vs full fused")
            else:
                d = (got.float() - full.float()).abs().max().item()
                r.fail(f"CACHE-7 oracle {device} cuts={cuts}", f"maxdiff {d:.3e}")


def test_v032_cache7_fp16_gate_is_lifted_and_exact(r: SubTestResult):
    """P0-2 (v0.33). This row used to assert the OPPOSITE — that fp16 taps are admitted and
    bit-exact — on the strength of a 22-row measurement. The measurement was real; its
    conclusion was not, because every row in it produced an fp16-REPRESENTABLE boundary.

    M-3 keeps coordinate builtins fp32 (invariant #4), so `u * 1000.0 + 0.123` carries an
    interior local at fp32 precision through a straight-through fused cook, while the
    checkpointed path materializes that boundary and the suffix downcasts it at ingest.

    The row is INVERTED rather than deleted, so the gate stays pinned in both directions:
      (a) fp16 taps are REFUSED — `materialize` returns nothing;
      (b) the counterexample is carried here as a differential row, so if anyone lifts the gate
          again on a representable-values corpus, THIS is what goes red;
      (c) the fp32 control over the same stages is still bit-exact — proving the counterexample
          indicts the precision label, not the checkpoint mechanism.
    """
    print("\n--- v0.32/P0-2 CACHE-7: fp16 taps REFUSED (the counterexample) ---")
    # THE SHAPE OF THE HAZARD, and it took two attempts to get right — the first draft applied
    # `fract` INSIDE the tapped stage, which collapses the magnitude before the boundary, so the
    # boundary was fp16-representable after all and the divergence was only 1.07e-03.
    #
    # It has to be:
    #   stage 0  produce a LARGE value from a COORDINATE builtin. M-3 keeps coords fp32
    #            (invariant #4), so `u * 1000.0` is an fp32 local even under precision="fp16",
    #            and in a straight-through fused cook it stays one all the way into stage 1.
    #   stage 1  be SENSITIVE to that value's low bits. `fract` is the sharpest such function:
    #            fp16's ULP at [512,1024) is 0.5, so a downcast at the boundary moves the
    #            fractional part by O(1), not by an ulp.
    #
    # The checkpointed path materializes stage 0's output and the suffix DOWNCASTS it at ingest;
    # the straight-through path never does. That difference is the whole defect.
    #
    # Kept out of `_POOL` deliberately: the pool is the one the original (over-confident)
    # measurement used, and it must stay able to reproduce that measurement's success.
    HAZARD = "@OUT = vec4(vec3(u * 1000.0 + 0.123) + @IN.rgb, 1.0);"
    SENSITIVE = "@OUT = vec4(fract(@IN.rgb), 1.0);"

    for device in _devices():
        torch.manual_seed(1)
        src = torch.rand(1, 128, 128, 3, device=device)
        stages = [{"code": HAZARD, "chain_input": None, "bindings": {"IN": src}},
                  {"code": SENSITIVE, "chain_input": "IN"},
                  {"code": "@OUT = vec4(@IN.rgb + vec3(0.05), 1.0);", "chain_input": "IN"},
                  {"code": "@OUT = vec4(@IN.rgb * 0.9, 1.0);", "chain_input": "IN"}]
        up = ("fp16-src",)

        # (a) the gate refuses fp16 outright. CUT AT 1 — immediately after the hazard stage,
        # while the boundary still holds the large value. A first draft cut at 2, i.e. AFTER
        # `fract` had already folded it back into [0,1) where fp16 represents it fine, and
        # measured a harmless 6.8e-04. The cut position IS the counterexample.
        cache = tex_results.ResultCache()
        done = CK.materialize(stages, cache, device=device, precision="fp16",
                              upstream=up, cuts=[1])
        if done:
            r.fail(f"CACHE-7 fp16 gate {device}",
                   f"materialized {done} — the unsound fp16 lift is back")
            continue

        # (b) THE COUNTEREXAMPLE, measured rather than asserted: had the gate admitted the tap,
        # this is the divergence it would have shipped. Driven through the same prefix/suffix
        # splice the checkpoint path uses, with the cache bypassed so the gate cannot hide it.
        full16 = tex_engine.cook_stage_list(stages, device=device, precision="fp16")["OUT"]
        boundary = tex_engine.cook_stage_list(stages[:1], device=device,
                                              precision="fp16")["OUT"]
        suffix = tex_fusion.suffix_stage_list(stages, 1, boundary)
        split16 = tex_engine.cook_stage_list(suffix, device=device, precision="fp16")["OUT"]
        d16 = float((split16.float() - full16.float()).abs().max())

        # (c) the fp32 control over the SAME stages: bit-exact, so the hazard is the label
        full32 = tex_engine.cook_stage_list(stages, device=device, precision="fp32")["OUT"]
        b32 = tex_engine.cook_stage_list(stages[:1], device=device, precision="fp32")["OUT"]
        split32 = tex_engine.cook_stage_list(
            tex_fusion.suffix_stage_list(stages, 1, b32), device=device,
            precision="fp32")["OUT"]

        if d16 > 1e-2 and torch.equal(split32, full32):
            r.ok(f"CACHE-7 {device}: fp16 taps refused; the split they would have served "
                 f"diverges {d16:.2e} while the fp32 split is bit-exact")
        else:
            r.fail(f"CACHE-7 fp16 counterexample {device}",
                   f"fp16 split maxdiff {d16:.2e} (want > 1e-2), fp32 split exact="
                   f"{torch.equal(split32, full32)} — if the fp16 divergence has vanished the "
                   f"counterexample no longer proves the gate is needed; re-derive it before "
                   f"lifting anything")

    # And the clause that was never lifted: a LATENT still refuses (M-3 forces fp32 and narrows
    # the wrong axis). Belt-and-braces on the gate, not an accident of the cook.
    src = torch.rand(1, 32, 32, 3)
    cache = tex_results.ResultCache()
    done = CK.materialize(_stages(src, 4), cache, device="cpu", precision="fp32",
                          upstream=("s",), cuts=[2], latent_channel_count=4)
    r.ok("CACHE-7 gate: a LATENT still refuses taps") if done == [] else         r.fail("CACHE-7 latent gate", f"materialized {done} under a LATENT")


# ── the regression: the resolution hole in the tap key ───────────────────────────────────

def test_v032_cache7_boundary_key_carries_resolution(r: SubTestResult):
    """REGRESSION. `cook_fused_cached` minted its boundary key without `canvas=`, and
    `ResultCache.get` validates neither shape nor device — so with one host source key a 64²
    and a 128² cook collided and the 128² request was served the 64² frame. Silent, wrong
    size, no error.

    Both paths are asserted, because the fix went into the KEY MINTER's default rather than
    into the new call site: fixing it only in CACHE-7 would have left shipped CACHE-6 wrong."""
    print("\n--- v0.32 CACHE-7: the tap key carries resolution (regression) ---")
    code0 = "@OUT = vec4(@IN.rgb * 1.1, 1.0);"
    code1 = "@OUT = vec4(@IN.rgb * $knob, 1.0);"

    def S(src):
        return [{"code": code0, "chain_input": None, "bindings": {"IN": src}},
                {"code": code1, "chain_input": "IN", "bindings": {"knob": 0.5}}]

    small, big = torch.rand(1, 64, 64, 3), torch.rand(1, 128, 128, 3)
    up = ("host-source-A",)          # NOT resolution-sensitive — the shipping hazard

    k_small = tex_engine.boundary_lineage_key(S(small), 1, "cpu", "fp32", upstream=up)
    k_big = tex_engine.boundary_lineage_key(S(big), 1, "cpu", "fp32", upstream=up)
    if k_small != k_big:
        r.ok("CACHE-7 key: 64² and 128² boundaries key apart")
    else:
        r.fail("CACHE-7 key resolution", "64² and 128² mint the SAME boundary key")

    # The end-to-end consequence, through the SHIPPED single-tap path.
    cache = tex_results.ResultCache()
    a = tex_engine.cook_fused_cached(S(small), 1, cache, device="cpu",
                                     precision="fp32", upstream=up)["OUT"]
    b = tex_engine.cook_fused_cached(S(big), 1, cache, device="cpu",
                                     precision="fp32", upstream=up)["OUT"]
    if tuple(a.shape)[1:3] == (64, 64) and tuple(b.shape)[1:3] == (128, 128):
        r.ok("CACHE-6 tap: a 128² cook is no longer served the 64² boundary")
    else:
        r.fail("CACHE-6 tap resolution",
               f"64² cook -> {tuple(a.shape)}, 128² cook -> {tuple(b.shape)}")


# ── placement refuses rather than guesses ───────────────────────────────────────────────

def test_v032_cache7_placement_refuses_rather_than_guesses(r: SubTestResult):
    """NEVER-SEVER ROWS. Every input that would make placement a guess must yield NO
    checkpoints — which is exactly today's cook, and always correct."""
    print("\n--- v0.32 CACHE-7: placement refuses rather than guesses ---")
    src = torch.rand(1, 64, 64, 3)
    S = _stages(src, 6)
    costs = {0: 5.0, 1: 5.0, 2: 200.0, 3: 5.0, 4: 5.0, 5: 5.0}
    px = 512 * 512

    rows = [
        ("no costs at all (the profiler is disarmed — the DEFAULT)",
         dict(costs=None, settled=True)),
        ("an UNSETTLED estimate (the EWMA still carries the cold cook)",
         dict(costs=costs, settled=False)),
        ("only the unfused `None` stage key (not a fused chain)",
         dict(costs={"None": 12.0}, settled=True)),
    ]
    for label, kw in rows:
        got = CK.plan_checkpoints(S, threshold_ms=10.0, px=px, **kw)
        r.ok(f"CACHE-7 refuses: {label}") if got == [] else \
            r.fail(f"CACHE-7 refuse ({label})", f"placed {got}")

    # A threshold BELOW the materialization floor must not place a tap per stage: at 1024² a
    # `put` is ~4.5 ms, so a 0.1 ms threshold would otherwise make the chain slower while
    # reporting success. The floor is a MULTIPLE of the put, so a stage that merely breaks
    # even against it is refused too.
    floor_px = 1024 * 1024
    cheap = {i: 1.0 for i in range(6)}          # every stage far below the floor
    if CK.plan_checkpoints(S, costs=cheap, threshold_ms=0.1, px=floor_px, settled=True) == []:
        r.ok("CACHE-7 floor: stages cheaper than a `put` get no checkpoints at all")
    else:
        r.fail("CACHE-7 materialization floor", "placed taps on stages cheaper than a `put`")

    marginal = {i: 5.0 for i in range(6)}       # ~break-even against a 4.5 ms put
    dense = CK.plan_checkpoints(S, costs=marginal, threshold_ms=0.1, px=floor_px, settled=True)
    if len(dense) <= 3:
        r.ok(f"CACHE-7 floor: break-even stages placed {len(dense)} tap(s), not 5")
    else:
        r.fail("CACHE-7 materialization floor",
               f"placed {len(dense)} taps on break-even stages at 1024²")

    # And the positive control, so none of the above passes vacuously.
    placed = CK.plan_checkpoints(S, costs=costs, threshold_ms=100.0, px=px, settled=True)
    if placed == [3]:
        r.ok("CACHE-7 places the cut AFTER the expensive stage (cumulative cost, not node count)")
    else:
        r.fail("CACHE-7 placement", f"expected [3] (the cut after the 200 ms stage), got {placed}")


def test_v032_cache7_dag_cut_set_is_analysis_only(r: SubTestResult):
    """A DAG cut crosses a SET of edges, not one. v0.32 ships the analysis and refuses to
    place there — correct, just not incremental, exactly CACHE-6 v1's posture."""
    print("\n--- v0.32 CACHE-7: DAG cut sets are analysed, not executed ---")
    src = torch.rand(1, 64, 64, 3)
    lin = _stages(src, 4)
    if CK.cut_set(lin, 2) == [(1, "OUT")]:
        r.ok("CACHE-7 cut_set: a linear cut crosses exactly one edge")
    else:
        r.fail("CACHE-7 cut_set linear", f"got {CK.cut_set(lin, 2)}")

    # A diamond: stage 3 reads BOTH stage 0 and stage 2, so a cut at 2 crosses two edges.
    dag = [dict(st) for st in lin]
    dag[3] = {"code": "@OUT = vec4((@A.rgb + @B.rgb) * 0.5, 1.0);", "chain_input": None,
              "bindings": {}, "chain_inputs": {"A": [0, "OUT"], "B": [2, "OUT"]}}
    cs = CK.cut_set(dag, 2)
    if sorted(cs) == [(0, "OUT"), (1, "OUT")]:
        r.ok(f"CACHE-7 cut_set: a DAG cut crosses {len(cs)} edges — a SET, not a tensor")
    else:
        r.fail("CACHE-7 cut_set DAG", f"expected [(0,'OUT'),(1,'OUT')], got {sorted(cs)}")

    costs = {i: 500.0 for i in range(4)}
    placed = CK.plan_checkpoints(dag, costs=costs, threshold_ms=1.0,
                                 px=64 * 64, settled=True)
    if 2 not in placed:
        r.ok("CACHE-7 refuses to place a checkpoint on a multi-edge DAG cut")
    else:
        r.fail("CACHE-7 DAG placement", f"placed {placed}, which includes the DAG cut 2")


# ── the harvest, and deepest-first serving ───────────────────────────────────────────────

def test_v032_cache7_one_cook_harvests_every_boundary(r: SubTestResult):
    """Phase 2 is ONE cook, not N segment cooks: `compile_fused` exports each tapped stage's
    handoff as `@_tap_s{i}`. Assert the taps come back, each equals the standalone prefix
    cook, and `@OUT` is unchanged by asking for them."""
    print("\n--- v0.32 CACHE-7: one cook harvests every boundary ---")
    N, cuts = 6, [2, 4]
    for device in _devices():
        torch.manual_seed(2)
        src = torch.rand(1, 96, 96, 3, device=device)
        plain = tex_engine.cook_stage_list(_stages(src, N), device=device, precision="fp32")
        tapped = tex_engine.cook_stage_list(
            _stages(src, N, tap_at={k - 1 for k in cuts}), device=device, precision="fp32")
        if not torch.equal(plain["OUT"], tapped["OUT"]):
            r.fail(f"CACHE-7 harvest {device}", "@OUT changed when taps were armed")
            continue
        r.ok(f"CACHE-7 harvest {device}: @OUT unchanged by arming taps")
        for k in cuts:
            got = tapped.get(f"_tap_s{k - 1}")
            ref = tex_engine.cook_stage_list(_stages(src, N)[:k], device=device,
                                             precision="fp32")["OUT"]
            if got is not None and torch.equal(got, ref):
                r.ok(f"CACHE-7 harvest {device}: tap at cut {k} == the standalone prefix cook")
            else:
                r.fail(f"CACHE-7 harvest {device} cut {k}",
                       "tap absent" if got is None else "tap != prefix cook")


def test_v032_cache7_serves_the_deepest_cached_checkpoint(r: SubTestResult):
    """Deepest-first is what makes an edit's cost depend on distance-to-checkpoint rather than
    on chain length. Assert it by COUNTING work: with the deep checkpoint cached, the cook
    must touch fewer stages than with only the shallow one."""
    print("\n--- v0.32 CACHE-7: the deepest cached checkpoint wins ---")
    src = torch.rand(1, 64, 64, 3)
    N, cuts = 6, [1, 4]
    up = ("deep-src",)
    full = tex_engine.cook_stage_list(_stages(src, N), device="cpu", precision="fp32")["OUT"]

    seen = []
    real = tex_fusion.suffix_stage_list

    def spy(stages, k, boundary):
        seen.append(k)
        return real(stages, k, boundary)

    cache = tex_results.ResultCache()
    CK.materialize(_stages(src, N), cache, device="cpu", precision="fp32",
                   upstream=up, cuts=cuts)
    tex_fusion.suffix_stage_list = spy
    try:
        got = CK.cook_checkpointed(_stages(src, N), cache, device="cpu",
                                   precision="fp32", upstream=up, cuts=cuts)["OUT"]
    finally:
        tex_fusion.suffix_stage_list = real

    if seen == [4]:
        r.ok("CACHE-7 serve: spliced from cut 4 (the deepest), not cut 1")
    else:
        r.fail("CACHE-7 deepest-first", f"suffix_stage_list called with {seen}, expected [4]")
    r.ok("CACHE-7 serve: deepest-splice result is bit-exact") if torch.equal(got, full) else \
        r.fail("CACHE-7 deepest-first value", "spliced result != full cook")

    # Nothing cached -> the whole chain cooks, exactly as today. That is phase 1, and it is
    # why arming CACHE-7 cannot make a FIRST cook slower.
    empty = tex_results.ResultCache()
    seen.clear()
    tex_fusion.suffix_stage_list = spy
    try:
        got2 = CK.cook_checkpointed(_stages(src, N), empty, device="cpu",
                                    precision="fp32", upstream=up, cuts=cuts)["OUT"]
    finally:
        tex_fusion.suffix_stage_list = real
    if seen == [] and torch.equal(got2, full):
        r.ok("CACHE-7 serve: an empty cache cooks the whole chain (phase 1 unchanged)")
    else:
        r.fail("CACHE-7 phase 1", f"empty cache spliced at {seen}")


def test_v032_cache7_upstream_source_key_is_mandatory(r: SubTestResult):
    """Inherited from CACHE-6 and just as load-bearing with N taps: without a
    content-sensitive source key covering EVERY tensor the prefix reads, a cached boundary
    could be served for a different image. No key -> no caching, and a correct full cook."""
    print("\n--- v0.32 CACHE-7: the upstream source key is mandatory ---")
    src = torch.rand(1, 64, 64, 3)
    cache = tex_results.ResultCache()
    done = CK.materialize(_stages(src, 4), cache, device="cpu", precision="fp32",
                          upstream=(), cuts=[2])
    if done == [] and cache.stats()["ram_entries"] == 0:
        r.ok("CACHE-7: no `upstream` -> nothing cached")
    else:
        r.fail("CACHE-7 upstream gate", f"materialized {done} with no source key")

    full = tex_engine.cook_stage_list(_stages(src, 4), device="cpu", precision="fp32")["OUT"]
    got = CK.cook_checkpointed(_stages(src, 4), cache, device="cpu", precision="fp32",
                               upstream=(), cuts=[2])["OUT"]
    r.ok("CACHE-7: no `upstream` still cooks correctly (full chain)") \
        if torch.equal(got, full) else \
        r.fail("CACHE-7 upstream fallback", "the no-key cook diverged from the full cook")


# ── the store is now safe for a background writer ───────────────────────────────────────

def test_v032_cache7_result_cache_is_thread_safe(r: SubTestResult):
    """CACHE-7's phase 2 makes the ENGINE a writer on the SCHED-4 worker thread while the host
    `get`s on the main thread. Before the lock that was concurrent `move_to_end`/`popitem` on
    one OrderedDict — a corrupted LRU or a RuntimeError, not a stale read."""
    print("\n--- v0.32 CACHE-7: ResultCache is thread-safe ---")
    cache = tex_results.ResultCache(budget_mb=2)      # tight, so eviction races too
    frame = torch.rand(1, 128, 128, 4)
    errors = []
    stop = threading.Event()

    def writer(tag):
        try:
            i = 0
            while not stop.is_set():
                cache.put(f"{tag}-{i % 24}", frame)
                i += 1
        except Exception as exc:                     # noqa: BLE001 — that IS the finding
            errors.append(f"{tag}: {type(exc).__name__}: {exc}")

    def reader():
        try:
            i = 0
            while not stop.is_set():
                cache.get(f"w0-{i % 24}")
                cache.stats()
                i += 1
        except Exception as exc:                     # noqa: BLE001
            errors.append(f"reader: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=writer, args=(f"w{i}",), daemon=True)
               for i in range(2)] + [threading.Thread(target=reader, daemon=True)]
    for t in threads:
        t.start()
    time.sleep(0.6)
    stop.set()
    for t in threads:
        t.join(timeout=5.0)

    if errors:
        r.fail("CACHE-7 ResultCache thread safety", "; ".join(errors[:3]))
    else:
        r.ok("CACHE-7: 2 writers + 1 reader for 0.6 s raised nothing")
    st = cache.stats()
    if st["ram_bytes"] >= 0 and st["ram_entries"] >= 0:
        r.ok(f"CACHE-7: byte accounting survived the race "
             f"({st['ram_entries']} entries, {st['ram_bytes']} B)")
    else:
        r.fail("CACHE-7 accounting", f"negative accounting after the race: {st}")


def test_v032_cache7_harvest_respects_the_tap_budget(r: SubTestResult):
    """A fused program carries at most MAX_OUTPUTS-1 taps beside its own @OUT, and
    `compile_fused` drops the overflow in ASCENDING stage order — keeping the SHALLOWEST and
    discarding the deepest, exactly inverting deepest-first serving.

    Measured before the fix: asked 8 taps -> got [1..7]; asked 12 -> got [1..7]. So a chain
    long enough to need many checkpoints silently kept only the ones nearest the source.
    `materialize` now harvests in passes, deepest first."""
    print("\n--- v0.32 CACHE-7: the harvest respects the tap budget ---")
    budget = tex_engine.MAX_OUTPUTS - 1
    n = budget + 5
    src = torch.rand(1, 48, 48, 3)
    up = ("budget-src",)
    cuts = list(range(1, n))                      # far more cuts than one pass can carry

    if CK._tap_budget() == budget:
        r.ok(f"CACHE-7 tap budget: {budget} taps beside @OUT (MAX_OUTPUTS={tex_engine.MAX_OUTPUTS})")
    else:
        r.fail("CACHE-7 tap budget", f"got {CK._tap_budget()}, expected {budget}")

    # A single cook genuinely cannot carry them all — so if `materialize` returns every cut
    # below, it did so by retrying, which is the property under test.
    one_pass = tex_engine.cook_stage_list(
        _stages(src, n, tap_at={k - 1 for k in cuts}), device="cpu", precision="fp32")
    carried = sorted(int(k.split("_s")[1]) for k in one_pass if k.startswith("_tap_s"))
    if len(carried) < len(cuts):
        r.ok(f"CACHE-7: one cook carries only {len(carried)} of {len(cuts)} taps "
             f"(deepest dropped: kept {carried[-1]} of {max(cuts) - 1})")
    else:
        r.fail("CACHE-7 tap budget", "one cook carried every tap — the batching is untested")

    cache = tex_results.ResultCache()
    done = CK.materialize(_stages(src, n), cache, device="cpu", precision="fp32",
                          upstream=up, cuts=cuts)
    if sorted(done) == sorted(cuts):
        r.ok(f"CACHE-7 harvest: all {len(cuts)} checkpoints materialized (> the {budget}-tap cap)")
    else:
        missing = sorted(set(cuts) - set(done))
        r.fail("CACHE-7 harvest budget", f"missing cuts {missing}")

    full = tex_engine.cook_stage_list(_stages(src, n), device="cpu", precision="fp32")["OUT"]
    got = CK.cook_checkpointed(_stages(src, n), cache, device="cpu", precision="fp32",
                               upstream=up, cuts=cuts)["OUT"]
    r.ok("CACHE-7 harvest: the multi-pass result is bit-exact") if torch.equal(got, full) else \
        r.fail("CACHE-7 multi-pass value", "diverged from the full cook")


def test_v032_cache7_prefix_fingerprint_range_guard(r: SubTestResult):
    """REGRESSION. Python slicing clamps, so `stages[:k]` for k >= len is the WHOLE list —
    and `prefix_fingerprint(S, 99)` silently returned the whole chain's fingerprint, which is
    the string PROF-1 keys its per-stage table on. One hand-audited caller never hit it; a
    multi-tap planner generates k programmatically."""
    print("\n--- v0.32 CACHE-7: prefix_fingerprint range guard (regression) ---")
    src = torch.rand(1, 32, 32, 3)
    S = _stages(src, 3)
    whole = tex_fusion._fused_fp(
        tex_fusion._fused_memo_key(S, tex_engine._infer_binding_type))
    for k in (0, 3, 99, -1):
        try:
            fp = tex_fusion.prefix_fingerprint(S, k, tex_engine._infer_binding_type)
            r.fail("CACHE-7 prefix range", f"k={k} returned {fp[:16]}"
                   + (" == THE WHOLE-CHAIN fp" if fp == whole else ""))
        except tex_fusion.FusionError:
            r.ok(f"CACHE-7 prefix_fingerprint: k={k} raises FusionError")
    for k in (1, 2):
        fp = tex_fusion.prefix_fingerprint(S, k, tex_engine._infer_binding_type)
        r.ok(f"CACHE-7 prefix_fingerprint: k={k} still returns a distinct prefix fp") \
            if fp != whole else \
            r.fail("CACHE-7 prefix range", f"k={k} collides with the whole-chain fp")


def test_v032_cache7_suffix_preserves_stage_keys(r: SubTestResult):
    """`suffix_stage_list` rebuilt the head stage from `code` alone, dropping every other key
    it carried — `tap` and `exports` above all. So a host that asked for a preview tap on the
    cut stage got it from a full cook and lost it from an incremental one."""
    print("\n--- v0.32 CACHE-7: the suffix preserves the head stage's keys ---")
    src = torch.rand(1, 32, 32, 3)
    S = _stages(src, 4)
    S[2] = {**S[2], "tap": True, "exports": []}
    suffix = tex_fusion.suffix_stage_list(S, 2, torch.rand(1, 32, 32, 4))
    if suffix[0].get("tap") is True:
        r.ok("CACHE-7 suffix: the head stage keeps its `tap` flag")
    else:
        r.fail("CACHE-7 suffix keys", "the head stage lost `tap` across the rebind")
    if suffix[0].get("chain_input") is None:
        r.ok("CACHE-7 suffix: the head stage is still rebound to the boundary")
    else:
        r.fail("CACHE-7 suffix keys", "the head stage kept its chain_input")


def test_v032_cache7_precision_must_be_resolved(r: SubTestResult):
    """`prepare()` is where "auto" resolves, and this path never calls it — `cook_stage_list`
    hands the string to the interpreter, which cooks "auto" as fp32. Caching under the label
    "auto" would mean a boundary the same host could never find once it resolved."""
    print("\n--- v0.32 CACHE-7: precision must be resolved ---")
    src = torch.rand(1, 32, 32, 3)
    S = _stages(src, 4)
    full = tex_engine.cook_stage_list(S, device="cpu", precision="fp32")["OUT"]
    for prec in ("auto", "bogus"):
        cache = tex_results.ResultCache()
        done = CK.materialize(_stages(src, 4), cache, device="cpu", precision=prec,
                              upstream=("s",), cuts=[2])
        if done == [] and cache.stats()["ram_entries"] == 0:
            r.ok(f"CACHE-7: precision={prec!r} caches nothing")
        else:
            r.fail("CACHE-7 precision gate", f"precision={prec!r} materialized {done}")
        got = CK.cook_checkpointed(_stages(src, 4), cache, device="cpu", precision=prec,
                                   upstream=("s",), cuts=[2])["OUT"]
        r.ok(f"CACHE-7: precision={prec!r} still cooks correctly") \
            if torch.equal(got, full) else \
            r.fail("CACHE-7 precision fallback", f"precision={prec!r} diverged")


def test_v032_cache7_stage_costs_cross_bucket_fallback(r: SubTestResult):
    """`predict` falls back to the nearest measured bucket scaled by pixels; `stage_costs`
    did not, so placement refused at every resolution the session had not already cooked —
    while the whole-cook estimator answered happily for the same key. profile.py's own
    docstring deferred this to CACHE-7."""
    print("\n--- v0.32 CACHE-7: stage_costs falls back across buckets ---")
    _profile.reset()
    key = _profile.make_key("fp-crossbucket", "cpu", "fp32")
    small = (1, 128, 128)
    _profile.record_stages(key, {0: 1.0, 1: 4.0}, small)
    _profile.record(key, 5.0, small)

    same = _profile.stage_costs(key, small)
    if same and abs(same.get(1, 0) - 4.0) < 1e-6:
        r.ok("CACHE-7 stage_costs: an exact bucket still returns its own EWMA")
    else:
        r.fail("CACHE-7 stage_costs exact", f"got {same}")

    big = (1, 512, 512)
    scaled = _profile.stage_costs(key, big)
    if not scaled:
        r.fail("CACHE-7 stage_costs fallback", "an unmeasured bucket still returns {}")
    else:
        ratio = scaled.get(1, 0) / 4.0
        if 8.0 < ratio < 24.0:            # 512²/128² = 16x pixels
            r.ok(f"CACHE-7 stage_costs: an unmeasured bucket scales by pixels ({ratio:.1f}x)")
        else:
            r.fail("CACHE-7 stage_costs fallback", f"scaled by {ratio:.2f}x, expected ~16x")
        if scaled.get(1, 0) > scaled.get(0, 0):
            r.ok("CACHE-7 stage_costs: the fallback preserves the stage RANKING")
        else:
            r.fail("CACHE-7 stage_costs fallback", "the scaled costs inverted the ranking")
    _profile.reset()


# ── invariant #7 ────────────────────────────────────────────────────────────────────────

def test_v032_cache7_off_the_default_path(r: SubTestResult):
    """CANARY. The ComfyUI cook path must neither import nor reach any of this."""
    print("\n--- v0.32 CACHE-7: off the default path ---")
    node_src = (Path(__file__).resolve().parent.parent / "tex_node.py").read_text(
        encoding="utf-8")
    for name in ("tex_checkpoint", "cook_checkpointed", "ResultCache", "tex_results"):
        if name in node_src:
            r.fail("CACHE-7 invariant #7", f"tex_node.py references {name}")
        else:
            r.ok(f"CACHE-7 invariant #7: tex_node.py never mentions {name}")

    # Placement with the profiler DISARMED (the default) must be empty, which is what makes
    # "off by default" a property of the wiring rather than a promise.
    was = _profile.enabled()
    if was:
        _profile.disable()
    try:
        src = torch.rand(1, 32, 32, 3)
        got = CK._plan_from_profile(_stages(src, 4), threshold_ms=100.0,
                                    profile_key=("nope", "cpu", "fp32"), spatial=(1, 32, 32))
        r.ok("CACHE-7: a disarmed profiler places no checkpoints") if got == [] else \
            r.fail("CACHE-7 disarmed placement", f"placed {got} with the profiler off")
    finally:
        if was:
            _profile.enable()


# ── audit fixes (the ones a test would have caught) ─────────────────────────────────────

def test_v032_cache7_refuses_dag_stage_lists(r: SubTestResult):
    """BLOCKER. A single-edge cut is NECESSARY but NOT SUFFICIENT.

    `suffix_stage_list` rebuilds the suffix as `[rebased head] + stages[k+1:]`, which RENUMBERS
    every stage — a stage that was index 7 becomes 7-k. Its `chain_inputs` are ABSOLUTE stage
    indices, copied verbatim, so on a DAG the suffix is silently mis-wired even when the cut
    itself crosses one edge. Measured through the weaker per-cut gate: 425 DAG lists admitted,
    30 returning WRONG PIXELS (maxdiff 0.146), against 0 and 0 through CACHE-6."""
    print("\n--- v0.32 CACHE-7: DAG stage lists are refused, not mis-wired ---")
    src = torch.rand(1, 64, 64, 3)
    up = ("dag-src",)

    def _lin(n):
        return _stages(src, n)

    # A diamond: stage 3 reads stage 0 AND stage 2. Cuts at 1 and 2 each cross >1 edge, but a
    # cut at 3 crosses only (2,'OUT') — single-edge, and still un-rebasable.
    dag = _lin(5)
    dag[3] = {"code": "@OUT = vec4((@A.rgb + @B.rgb) * 0.5, 1.0);", "chain_input": None,
              "bindings": {}, "chain_inputs": {"A": [0, "OUT"], "B": [2, "OUT"]}}

    cache = tex_results.ResultCache()
    done = CK.materialize(dag, cache, device="cpu", precision="fp32", upstream=up,
                          cuts=[1, 2, 3, 4])
    if done == [] and cache.stats()["ram_entries"] == 0:
        r.ok("CACHE-7: a DAG stage list materializes nothing")
    else:
        r.fail("CACHE-7 DAG gate", f"materialized {done} on a DAG list")

    # And the cook still has to be RIGHT — the refusal must fall back, not diverge.
    full = tex_engine.cook_stage_list(dag, device="cpu", precision="fp32")["OUT"]
    got = CK.cook_checkpointed(dag, cache, device="cpu", precision="fp32", upstream=up,
                               cuts=[1, 2, 3, 4])["OUT"]
    if torch.equal(got, full):
        r.ok("CACHE-7: a refused DAG cooks the whole chain, bit-exact")
    else:
        d = (got.float() - full.float()).abs()
        r.fail("CACHE-7 DAG correctness",
               f"maxdiff {d.max().item():.3e} over {int((d > 0).sum())} elements")

    # A small sweep, because the single case above is the one I happened to think of.
    bad = admitted = 0
    for n in range(3, 7):
        for a in range(0, n - 2):
            for b in range(a + 1, n - 1):
                S = _lin(n)
                S[n - 1] = {"code": "@OUT = vec4((@A.rgb + @B.rgb) * 0.5, 1.0);",
                            "chain_input": None, "bindings": {},
                            "chain_inputs": {"A": [a, "OUT"], "B": [b, "OUT"]}}
                ref = tex_engine.cook_stage_list(S, device="cpu", precision="fp32")["OUT"]
                c = tex_results.ResultCache()
                if CK.materialize(S, c, device="cpu", precision="fp32", upstream=up,
                                  cuts=list(range(1, n))):
                    admitted += 1
                try:
                    out = CK.cook_checkpointed(S, c, device="cpu", precision="fp32",
                                               upstream=up, cuts=list(range(1, n)))["OUT"]
                    if not torch.equal(out, ref):
                        bad += 1
                except Exception:
                    bad += 1
    if admitted == 0 and bad == 0:
        r.ok("CACHE-7 sweep: 0 DAG lists admitted, 0 wrong pixels (CACHE-6's posture)")
    else:
        r.fail("CACHE-7 DAG sweep", f"{admitted} admitted, {bad} wrong/raised")


def test_v032_cache7_profile_costs_and_confidence_agree(r: SubTestResult):
    """`stage_costs` filters to buckets carrying a per-stage breakdown; `samples` did not. Two
    calls could therefore resolve DIFFERENT buckets and hand a planner costs from one with
    confidence from another. `stage_snapshot` resolves once."""
    print("\n--- v0.32 CACHE-7: costs and confidence come from one bucket ---")
    _profile.reset()
    key = _profile.make_key("fp-agree", "cpu", "fp32")
    small, big = (1, 128, 128), (1, 512, 512)
    # A bucket with MANY whole-cook samples but NO per-stage breakdown, plus a distant bucket
    # that has one. Two independent queries land on different buckets here.
    for _ in range(30):
        _profile.record(key, 5.0, big)
    _profile.record_stages(key, {0: 1.0, 1: 4.0}, small)
    _profile.record(key, 5.0, small)

    costs, is_settled = _profile.stage_snapshot(key, big, need=12)
    n = _profile.samples(key, big, need_stages=True)
    if costs and (is_settled == (n >= 12)):
        r.ok(f"CACHE-7: stage_snapshot's verdict matches its own bucket "
             f"(settled={is_settled}, samples={n})")
    else:
        r.fail("CACHE-7 snapshot agreement",
               f"costs={bool(costs)} settled={is_settled} samples_of_that_bucket={n}")
    loose = _profile.samples(key, big)
    if loose != n:
        r.ok(f"CACHE-7: the two bucket filters genuinely differ here ({loose} vs {n}) — "
             "so the row is not vacuous")
    else:
        r.ok("CACHE-7: bucket filters agree on this table (row still valid)")
    _profile.reset()
