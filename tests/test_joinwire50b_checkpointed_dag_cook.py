"""JOINWIRE-50b — wiring checkpoint boundaries into `tex_chain.cook_stage_dag`.

JOINWIRE-50 (v0.50 wave 1) shipped `cook_stage_dag` — a DAG-shaped, node-by-node, windowed
join cook — but its own design record named what it did NOT ship: no way for a host that uses
checkpoint boundaries (a `ResultCache`, the same object `tex_checkpoint.cook_checkpointed`
already takes) to get the same windowed-join win. This file proves the answer this ask
picked: teach `cook_stage_dag` ITSELF the checkpoint semantics (`result_cache=`, `upstream=`),
rather than teaching `cook_checkpointed` the DAG shape.

Why that side, not the other: `cook_checkpointed`'s own mechanism (`suffix_stage_list` /
`compile_fused` splicing ONE program) is fused-linear-chain shaped down to its `cut_set`
analysis, which already documents the DAG execution half as DEFERRED (a multi-edge cut's
suffix rebind is unsolved, not merely untested) — `gate_refusal` refuses a non-linear stage
list outright (`REFUSE_NOT_LINEAR`). `cook_stage_dag`, by contrast, already cooks node-by-node
with NO fusion: a "checkpoint boundary" there is just a stage's own already-valid full-frame
output persisted across ticks — exactly what `known_outputs` already represents, except held
in the HOST's memory instead of a `ResultCache`. Wiring `result_cache` as an optional
cache-backed fallback for `known_outputs` costs zero new keying: `tex_chain.boundary_lineage_key`
already fingerprints a stage list generically over `chain_inputs` (`_fused_memo_key`'s Q-3
topology tuple), so it needs no DAG-specific variant — see `cook_stage_dag`'s own docstring
for the full argument.

THE LOAD-BEARING SAFETY RULE this file exists to prove, not merely assert: a boundary is
`put` into `result_cache` ONLY when `tier_trace.last_roi()` says the stage's OWN cook this
tick actually served whole-frame (`served_roi is None`) — never when it was windowed and
re-embedded (which carries an unread-garbage region outside its window). A windowed result
must never be served back as a whole frame later.

Every allowed round trip is proven by PIXEL IDENTITY (`torch.equal`) against a fresh
whole-frame cook of the same graph. CPU only (no device-specific mechanism here — the get/put
plumbing is torch-free string keys and Python dict lookups; `cook_stage_dag`'s own CUDA row
already covers the device axis for the underlying per-stage cook).
"""
from __future__ import annotations

from helpers import *
from helpers import _crop  # FIX-DAG G3 (R1#4): shared with test_joinwire50_dag_cook's own copy

from TEX_Wrangle import tex_chain
from TEX_Wrangle import tex_results


# A blur heavy enough that ANY partial roi, grown by its halo (3x sigma, invariant #5's
# `gauss_blur` reach multiplier), clamps to the full frame on a 16x16 canvas: 3*20=60 >> 16.
_HEAVY_SIGMA = 20.0


def _cp_merge_stages(A, B):
    """stage0: the checkpoint CANDIDATE (a plain source). stage1: a heavy blur of stage0 —
    forces stage0's OWN planned window to clamp to the full frame (see module docstring),
    so stage0 SERVES whole-frame even though it was "planned" a window; that is exactly the
    shape a checkpoint boundary needs, and exactly why the cache is populated off
    `served_roi`, never off "was a window planned". stage2: an independent live edit of B.
    stage3: the sink Merge, joining stage1 and stage2, pointwise (no growth of its own)."""
    return [
        {"code": "@OUT = @A;", "bindings": {"A": A}},
        {"code": f"@OUT = gauss_blur(@P, {_HEAVY_SIGMA});", "bindings": {},
         "chain_inputs": {"P": [0, "OUT"]}},
        {"code": "@OUT = @B + 0.1;", "bindings": {"B": B}},
        {"code": "@OUT = (@bg + @fg) * 0.5;", "bindings": {},
         "chain_inputs": {"bg": [1, "OUT"], "fg": [2, "OUT"]}},
    ]


def _cp_diamond_stages(A):
    """stage0: the checkpoint candidate, feeding TWO consumers — stage1 (heavy blur, forces
    the union demand on stage0 to clamp full) and stage2 (a plain pointwise pass). stage3:
    the sink, joining stage1+stage2 pointwise."""
    return [
        {"code": "@OUT = @A;", "bindings": {"A": A}},
        {"code": f"@OUT = gauss_blur(@P, {_HEAVY_SIGMA});", "bindings": {},
         "chain_inputs": {"P": [0, "OUT"]}},
        {"code": "@OUT = @P * 1.0;", "bindings": {},
         "chain_inputs": {"P": [0, "OUT"]}},
        {"code": "@OUT = (@bg + @fg) * 0.5;", "bindings": {},
         "chain_inputs": {"bg": [1, "OUT"], "fg": [2, "OUT"]}},
    ]


# ── Merge-below-edit, checkpoint-backed ──────────────────────────────────────────────────

def test_joinwire50b_checkpointed_merge_below_edit_pixel_identity(r: SubTestResult):
    print("\n--- JOINWIRE-50b: Merge-below-edit, checkpoint boundary round trip ---")
    torch.manual_seed(5001)
    W = H = 16
    A = torch.rand(1, H, W, 3)
    B_old = torch.rand(1, H, W, 3)
    B_new = torch.rand(1, H, W, 3)
    rc = tex_results.ResultCache()
    up = ("jw50b-merge-src",)

    # Tick 1: everything dirty (first cook). Stage 0 is PLANNED a window (it has a real
    # downstream demand) but its own cook SERVES whole-frame (the heavy-blur clamp) —
    # exactly the shape that should populate the cache.
    roi1 = (1, 1, 4, 4, W, H)
    stages_old = _cp_merge_stages(A, B_old)
    win1 = tex_chain.cook_stage_dag(stages_old, roi=roi1, roi_exec=True, dirty_from=0,
                                    result_cache=rc, upstream=up)
    if win1["windows"] is None or win1["windows"][0] is None:
        r.fail("JOINWIRE-50b merge checkpoint seed",
               "expected stage 0 to be planned a window (a real downstream demand exists)")
        return
    if win1["windows"][0][2:4] != (W, H):
        r.fail("JOINWIRE-50b merge checkpoint seed",
               f"expected stage 0's grown window to clamp to the full frame, got "
               f"{win1['windows'][0]!r}")
        return

    # Tick 2: an EDIT lands on stage 2 (B changes); stage 0 is now CLEAN (dirty_from=1)
    # and NOT supplied via known_outputs at all -- only `result_cache` can satisfy it.
    roi2 = (9, 9, 3, 3, W, H)
    stages_new = _cp_merge_stages(A, B_new)
    win2 = tex_chain.cook_stage_dag(stages_new, roi=roi2, roi_exec=True, dirty_from=1,
                                    valid=[None, (0, 0, W, H), None, None],
                                    result_cache=rc, upstream=up)
    fresh = tex_chain.cook_stage_dag(stages_new)
    ref = _crop(fresh["result"]["OUT"], roi2)
    got = win2["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("JOINWIRE-50b merge checkpoint round trip",
               f"maxdiff={(got - ref).abs().max().item()}")
        return
    r.ok("JOINWIRE-50b: a checkpoint boundary written by tick 1 (whole-frame, off "
        "`served_roi`) correctly serves tick 2's clean stage with no `known_outputs` at "
        "all, and the edited graph still matches a fresh whole-frame cook exactly")


# ── Diamond, checkpoint-backed ────────────────────────────────────────────────────────────

def test_joinwire50b_checkpointed_diamond_pixel_identity(r: SubTestResult):
    print("\n--- JOINWIRE-50b: diamond, checkpoint boundary round trip ---")
    torch.manual_seed(5002)
    W = H = 16
    A = torch.rand(1, H, W, 3)
    rc = tex_results.ResultCache()
    up = ("jw50b-diamond-src",)
    stages = _cp_diamond_stages(A)

    roi1 = (2, 2, 3, 3, W, H)
    win1 = tex_chain.cook_stage_dag(stages, roi=roi1, roi_exec=True, dirty_from=0,
                                    result_cache=rc, upstream=up)
    if win1["windows"] is None or win1["windows"][0][2:4] != (W, H):
        r.fail("JOINWIRE-50b diamond checkpoint seed",
               f"expected stage 0's UNION demand to clamp to the full frame, got "
               f"{win1['windows'] and win1['windows'][0]!r}")
        return

    roi2 = (10, 10, 4, 4, W, H)
    win2 = tex_chain.cook_stage_dag(stages, roi=roi2, roi_exec=True, dirty_from=1,
                                    result_cache=rc, upstream=up)
    fresh = tex_chain.cook_stage_dag(stages)
    ref = _crop(fresh["result"]["OUT"], roi2)
    got = win2["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("JOINWIRE-50b diamond checkpoint round trip",
               f"maxdiff={(got - ref).abs().max().item()}")
        return
    r.ok("JOINWIRE-50b: a diamond's checkpointed source (union demand forced it whole) "
        "serves a second, differently-windowed tick correctly with no known_outputs")


# ── dirty_from mid-DAG, checkpoint vs known_outputs equivalence ─────────────────────────

def test_joinwire50b_checkpointed_dirty_from_mid_dag_matches_known_outputs(r: SubTestResult):
    """The cache-backed path and the pre-existing (JOINWIRE-50) `known_outputs` path must
    reach the IDENTICAL pixels for the SAME dirty_from mid-DAG edit — `result_cache` is a
    second SOURCE for the same clean-stage value, not a second contract."""
    print("\n--- JOINWIRE-50b: dirty_from mid-DAG, cache path == known_outputs path ---")
    torch.manual_seed(5003)
    W = H = 16
    A = torch.rand(1, H, W, 3)
    B_new = torch.rand(1, H, W, 3)
    rc = tex_results.ResultCache()
    up = ("jw50b-dirtyfrom-src",)
    stages = _cp_merge_stages(A, torch.rand(1, H, W, 3))  # B (tick 1) unused downstream here

    roi_seed = (1, 1, 4, 4, W, H)
    win1 = tex_chain.cook_stage_dag(stages, roi=roi_seed, roi_exec=True, dirty_from=0,
                                    result_cache=rc, upstream=up)
    stage0_output = win1["stage_outputs"][0]

    stages_new = _cp_merge_stages(A, B_new)
    roi2 = (7, 7, 5, 5, W, H)
    via_cache = tex_chain.cook_stage_dag(
        stages_new, roi=roi2, roi_exec=True, dirty_from=1,
        valid=[None, (0, 0, W, H), None, None], result_cache=rc, upstream=up)
    via_known = tex_chain.cook_stage_dag(
        stages_new, roi=roi2, roi_exec=True, dirty_from=1,
        valid=[None, (0, 0, W, H), None, None], known_outputs={0: stage0_output})

    got_cache = via_cache["result"]["OUT"]
    got_known = via_known["result"]["OUT"]
    if not torch.equal(got_cache, got_known):
        r.fail("JOINWIRE-50b cache vs known_outputs",
               f"diverged, maxdiff={(got_cache - got_known).abs().max().item()}")
        return
    r.ok("JOINWIRE-50b: result_cache and known_outputs serve the identical clean-stage "
        "value for the same dirty_from mid-DAG edit")


# ── Refusal (divergent-validity join): result_cache must stay untouched ─────────────────

_BG_CODE = "@OUT = @A;"
_FG_BLUR_CODE = f"@OUT = gauss_blur(@A, {_HEAVY_SIGMA / 6.0});"
_JOIN_CODE = "@OUT = (@bg + @fg) * 0.5;"


def test_joinwire50b_refusal_leaves_result_cache_untouched(r: SubTestResult):
    """A divergent-validity join refuses the WHOLE plan (`chain_windows_dag` -> None); when
    that happens `cook_stage_dag` cooks everything from source, and `result_cache` must be
    consulted for NOTHING (there is no "clean, windowed" case to checkpoint at all — the
    whole point of the refusal is that nothing here is safely narrow-servable this tick).
    A cache that a PRIOR tick already populated must not leak a stale value into the
    refused cook either."""
    print("\n--- JOINWIRE-50b: a whole-plan refusal never touches result_cache ---")
    torch.manual_seed(5004)
    W = H = 20
    src = torch.rand(1, H, W, 3)
    stages = [
        {"code": _BG_CODE, "bindings": {"A": src}},
        {"code": _FG_BLUR_CODE, "bindings": {"A": src}},
        {"code": _JOIN_CODE, "bindings": {},
         "chain_inputs": {"bg": [0, "OUT"], "fg": [1, "OUT"]}},
    ]
    rc = tex_results.ResultCache()
    up = ("jw50b-refusal-src",)
    roi_far = (14, 14, 4, 4, W, H)
    valid = [(0, 0, W, H), (0, 0, 6, 6), None]   # fg's tracked-valid region is a small corner

    win = tex_chain.cook_stage_dag(stages, roi=roi_far, roi_exec=True, dirty_from=2,
                                   valid=valid, result_cache=rc, upstream=up)
    if win["windows"] is not None:
        r.fail("JOINWIRE-50b refusal", f"expected a whole-plan refusal, got {win['windows']!r}")
        return
    if rc.stats()["ram_entries"]:
        r.fail("JOINWIRE-50b refusal", "result_cache was written to during a refused plan")
        return
    full = tex_chain.cook_stage_dag(stages)
    ref = full["result"]["OUT"]
    got = win["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("JOINWIRE-50b refusal fallback pixel identity",
               f"maxdiff={(got - ref).abs().max().item()}")
        return
    r.ok("JOINWIRE-50b: a whole-plan refusal leaves result_cache empty and still matches "
        "the whole-frame fallback exactly")


# ── Negative control: a windowed stage must NEVER reach result_cache ────────────────────

def test_joinwire50b_negative_control_windowed_output_would_corrupt_if_cached(r: SubTestResult):
    """Proves the load-bearing rule by contradiction: manually PUT a stage's WINDOWED
    (cropped, re-embedded) output under the exact key a whole-frame boundary would use, then
    show a later clean lookup reading it back produces WRONG pixels against a true
    whole-frame cook — and separately confirm the REAL code path never does this itself
    (the cache stays empty for a stage that served windowed this tick)."""
    print("\n--- JOINWIRE-50b negative control: a windowed output cached as whole-frame is wrong ---")
    torch.manual_seed(5005)
    W = H = 16
    A = torch.rand(1, H, W, 3)
    B = torch.rand(1, H, W, 3)
    stages = _cp_merge_stages(A, B)

    # A window request small enough that stage 1 (the heavy blur consumer feeding the
    # sink) itself still crops (not the clamp case) -- pick roi identical to what stage 1
    # actually serves so we can steal its own genuinely-windowed output.
    roi = (3, 3, 4, 4, W, H)
    rc_real = tex_results.ResultCache()
    up = ("jw50b-negctrl-a", "jw50b-negctrl-b")
    win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True, dirty_from=0,
                                   result_cache=rc_real, upstream=up)
    # Stage 2 (the plain "@B + 0.1" edit, no upstream halo) is windowed exactly to `roi`
    # (never clamps full on this small request) -- confirm that, then confirm the REAL
    # cache never stored a boundary for it.
    if win["windows"][2] is None or win["windows"][2][2:4] == (W, H):
        r.fail("JOINWIRE-50b negative control setup",
               f"expected stage 2 to be genuinely windowed (not clamped full), got "
               f"{win['windows'][2]!r}")
        return

    key = tex_chain.boundary_lineage_key(stages, 3, "cpu", "fp32", upstream=up,
                                         latent_channel_count=0, scale=None)
    if rc_real.get(key) is not None:
        r.fail("JOINWIRE-50b negative control",
               "the real code path cached a windowed stage's output as a whole-frame "
               "boundary -- exactly the bug this rule exists to prevent")
        return

    # FIX-DAG G1: `stage_outputs[2]["OUT"]` is now the bare CROP `cook_stage_list` itself
    # returned — the public surface no longer carries a padded, mostly-garbage full canvas
    # at all (that used to be exactly this attack's own setup). Confirm the crop really is
    # small (not accidentally full-size, which would silently defang this test), then
    # reconstruct the OLD full-size, only-`roi`-real shape by hand via `_embed_window` — the
    # one place that re-embed still exists — to show WHY the rule matters: manually put that
    # reconstructed value under the boundary key, and use it as a "clean" boundary for a
    # DIFFERENT window.
    stage2_crop = win["stage_outputs"][2]["OUT"]
    if list(stage2_crop.shape[1:3]) == [H, W]:
        r.fail("JOINWIRE-50b negative control setup",
               "stage 2's stage_outputs entry is already full-size -- the crop-vs-embed "
               "distinction this attack needs is gone")
        return
    windowed_embedded = tex_chain._embed_window(stage2_crop, win["stage_windows"][2])
    rc_poisoned = tex_results.ResultCache()
    rc_poisoned.put(key, windowed_embedded, canvas={"shape": list(windowed_embedded.shape)})

    # dirty_from=3 marks stages 0-2 all "clean"; stage 2 must come from the (poisoned)
    # result_cache — the thing under test — but stages 0/1 need a genuine TRUE whole-frame
    # value so the corruption under test is isolated to stage 2 alone, not a side effect of
    # an unrelated missing/garbage upstream.
    b0 = tex_chain.cook_stage_list([stages[0]])
    b1 = tex_chain.cook_stage_list([dict(stages[1], bindings={"P": b0["OUT"]})])
    roi2 = (11, 11, 3, 3, W, H)
    served = tex_chain.cook_stage_dag(stages, roi=roi2, roi_exec=True, dirty_from=3,
                                      known_outputs={0: b0, 1: b1},
                                      result_cache=rc_poisoned, upstream=up)
    fresh = tex_chain.cook_stage_dag(stages)
    ref = _crop(fresh["result"]["OUT"], roi2)
    got = served["result"]["OUT"]
    if torch.equal(got, ref):
        r.fail("JOINWIRE-50b negative control",
               "expected serving a windowed value as a whole-frame boundary to be WRONG, "
               "but it matched the true whole-frame cook")
        return
    r.ok(f"JOINWIRE-50b: serving a windowed output as a whole-frame boundary IS pixel-wrong "
        f"(maxdiff={(got - ref).abs().max().item()}), confirming why `cook_stage_dag` only "
        f"ever puts off `served_roi is None`, and confirming the real code path never "
        f"does this itself")


# ── The sink has no valid boundary cut point (regression for the n-1 guard) ─────────────

def test_joinwire50b_sink_whole_frame_with_result_cache_does_not_raise(r: SubTestResult):
    """A `convolve`-style call makes a WHOLE stage list fall back to whole-frame, including
    the SINK (the last stage) -- under an active window plan (`windows is not None`), not
    the plain `roi=None` no-op. `boundary_lineage_key`/`prefix_fingerprint` only accepts a
    cut `1 <= k < len(stages)` (a boundary is always "after stage k-1", which needs a stage
    AT k to exist) -- the sink has no such cut, so `cook_stage_dag` must skip caching it
    rather than mint `k == len(stages)` and raise `FusionError` out of a serve path."""
    print("\n--- JOINWIRE-50b: a whole-frame sink under result_cache never raises ---")
    torch.manual_seed(5006)
    W = H = 24
    kernel = torch.rand(1, 3, 3, 1)
    img = torch.rand(1, H, W, 3)
    stages = [
        {"code": "@OUT = @K;", "bindings": {"K": kernel}},
        {"code": "@OUT = @I;", "bindings": {"I": img}},
        {"code": "@OUT = convolve(@img, @kernel);", "bindings": {},
         "chain_inputs": {"img": [1, "OUT"], "kernel": [0, "OUT"]}},
    ]
    rc = tex_results.ResultCache()
    up = ("jw50b-sink-src",)
    roi = (4, 4, 6, 6, W, H)
    try:
        win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True, dirty_from=0,
                                       result_cache=rc, upstream=up)
    except Exception as e:
        r.fail("JOINWIRE-50b sink whole-frame regression", f"raised {type(e).__name__}: {e}")
        return
    full = tex_chain.cook_stage_dag(stages)
    if not torch.equal(win["result"]["OUT"], full["result"]["OUT"]):
        r.fail("JOINWIRE-50b sink whole-frame regression", "pixel mismatch vs whole-frame cook")
        return
    r.ok("JOINWIRE-50b: a convolve-style whole-graph fallback with result_cache armed "
        "cooks correctly and never tries to mint a boundary key for the sink")
