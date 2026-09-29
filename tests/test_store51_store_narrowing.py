"""STORE-51 (v0.51 wave 2, host item 3's ONE small ask) — `tex_chain.cook_stage_dag` gains an
additive `store=` parameter.

CKPT-51 (this same wave) confirmed the composition question: `cook_stage_dag` is where a host's
join-shaped checkpointing lives, and the host owns cut placement (its own choice of cost model,
manual pins or bans) while TEX owns the one safety rule (a windowed output is never storable as
a whole-frame boundary). The host's actual ask, once that composition was settled: let it choose
WHICH clean whole-frame stages actually get `put` into `result_cache`, without touching how a
boundary is served (by `boundary_lineage_key`) or the never-store-a-window rule.

`store` (an optional `set[int]` of stage indices) intersects with the EXISTING `put` eligibility
test (clean, `result_cache is not None`, `windows is not None`, not the sink, `served_roi is
None`) — it can only NARROW what gets stored, never widen it. `store=None` (the default) keeps
every caller before this ask working exactly as before: every eligible stage is stored, as today.

**The shape used throughout** (`_two_eligible_branches`, below) is deliberately reused rather than
invented per row: two independent sources (stage 0, stage 2), each read by its own heavy-blur
CONSUMER (stage 1, stage 3), which is what makes stage 0/2 -- not the blurs -- the ELIGIBLE
whole-frame boundaries here. A heavy blur's halo inflates the INPUT window it needs from its
source, and when that inflated demand saturates the whole canvas the backward-projected `windows[
i]` for the SOURCE becomes `(0, 0, W, H, W, H)` -- indistinguishable from "no crop needed" by
`cook_stage_dag`'s own `w[2:4] != w[4:6]` test, so the source serves whole-frame (`stage_roi=None`,
`served_roi=None`) and is `result_cache`-eligible. The blurs themselves (stage 1, 3) are asked for
only the SAME small OUTPUT window the sink wants and (being ROI-executable) genuinely narrow to
it -- confirmed via `stage_windows` below, never assumed, so this file has a genuinely windowed
sibling (1, 3) sitting right next to a genuinely eligible one (0, 2) in every row.

Four rows, matching the ask's own red-first list:

1. `store={i}` stores exactly stage `i` — of the two eligible siblings (0 and 2), naming only
   one in `store` leaves the other's boundary key absent from `result_cache`.
2. `store=set()` stores nothing this tick, but a PRE-EXISTING boundary (put by an earlier,
   unrestricted call) is still served correctly — narrowing what is WRITTEN is not the same as
   forgetting what was already READ.
3. A genuinely windowed stage (1) named in `store=` alongside an eligible one (0) is still not
   stored (negative control) — `store` can only narrow the existing eligibility test, never
   override the `served_roi is None` gate that decides whether a stage was windowed at all.
4. The default (`store=None`, and every caller from before this ask) stores every eligible stage,
   unchanged in count from CKPT-51/JOINWIRE-50b's own behaviour.

Plus one validation row: an out-of-range `store` index is a host wiring mistake, not a shape this
cook can act on, and is rejected up front with a clear diagnostic naming the bad index.

CPU only (no device-specific mechanism exercised beyond what JOINWIRE-50/50b/CKPT-51's own CUDA
rows already cover).
"""
from __future__ import annotations

import torch

from helpers import *
from helpers import _crop

from TEX_Wrangle import tex_chain
from TEX_Wrangle import tex_results
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib as _Sample  # populates REGISTRY


_HEAVY_SIGMA = 20.0  # halo = 3*20 = 60, far past a 16x16 canvas
_W = _H = 16


def _two_eligible_branches(A, B):
    """stage0/2: two independent sources — THESE are the eligible `result_cache` boundaries,
    not the blurs: each is read by a heavy-blur CONSUMER (stage1/3) whose halo (60px) needs an
    input patch that saturates the whole 16x16 frame, so the demand `chain_windows_dag`
    backward-projects onto stage0/2 is the FULL frame (`windows[i]` a `(0, 0, W, H, W, H)`
    tuple whose crop equals the whole canvas) — indistinguishable from "not windowed" by
    `cook_stage_dag`'s own `w[2:4] != w[4:6]` test, hence `stage_roi=None`, hence
    `served_roi=None`: a whole-frame serve, `stages_whole`-counted and `result_cache`-eligible.
    stage1/3 (the blurs themselves) are asked for only the SAME small output window the sink
    wants and (being ROI-executable) genuinely narrow to it — verified in `stage_windows`
    below, not assumed. stage4: the join (sink)."""
    return [
        {"code": "@OUT = @A;", "bindings": {"A": A}},
        {"code": f"@OUT = gauss_blur(@P, {_HEAVY_SIGMA});", "bindings": {},
         "chain_inputs": {"P": [0, "OUT"]}},
        {"code": "@OUT = @B;", "bindings": {"B": B}},
        {"code": f"@OUT = gauss_blur(@Q, {_HEAVY_SIGMA});", "bindings": {},
         "chain_inputs": {"Q": [2, "OUT"]}},
        {"code": "@OUT = (@bg + @fg) * 0.5;", "bindings": {},
         "chain_inputs": {"bg": [1, "OUT"], "fg": [3, "OUT"]}},
    ]


def _keys_for(stages, up):
    """The SAME `boundary_lineage_key` calls `cook_stage_dag`'s own `_checkpoint_key` closure
    makes internally, for the two ELIGIBLE stage indices 0 and 2 (k = idx + 1) — spelled once
    here so every row below reads a cache entry the same way the mechanism itself writes it."""
    k0 = tex_chain.boundary_lineage_key(stages, 1, "cpu", "fp32", upstream=up)
    k2 = tex_chain.boundary_lineage_key(stages, 3, "cpu", "fp32", upstream=up)
    return k0, k2


def _assert_shape(r, win, tag):
    """The shared sanity check every row relies on: stage 0/2 serve whole-frame (absent from
    `stage_windows`), stage 1/3 genuinely narrow (present). Returns True when the shape holds."""
    sw = win["stage_windows"]
    if 0 in sw or 2 in sw:
        r.fail(tag, f"expected stage 0 and 2 (the sources) to serve whole-frame, got "
               f"stage_windows={sw!r}")
        return False
    if 1 not in sw or 3 not in sw:
        r.fail(tag, f"expected stage 1 and 3 (the heavy blurs) to genuinely window, got "
               f"stage_windows={sw!r}")
        return False
    return True


# ── 1. store={i} stores exactly stage i ─────────────────────────────────────────────────────

def test_store51_store_names_stores_exactly_that_stage(r: SubTestResult):
    print("\n--- STORE-51: store={i} stores exactly stage i, not its eligible sibling ---")
    torch.manual_seed(5111)
    A = torch.rand(1, _H, _W, 3)
    B = torch.rand(1, _H, _W, 3)
    rc = tex_results.ResultCache()
    up = ("store51-two-branch-src", "second-source",)
    stages = _two_eligible_branches(A, B)
    key0, key2 = _keys_for(stages, up)

    roi = (1, 1, 4, 4, _W, _H)
    win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True, dirty_from=0,
                                   result_cache=rc, upstream=up, store={0})
    if not _assert_shape(r, win, "STORE-51 store={0} seed"):
        return

    if rc.get(key0) is None:
        r.fail("STORE-51 store={0}", "stage 0 was named in store= and was eligible, but "
               "result_cache has no entry for it")
        return
    if rc.get(key2) is not None:
        r.fail("STORE-51 store={0}", "stage 2 was NOT named in store= but was stored anyway "
               "-- store= must narrow, never widen implicitly")
        return

    fresh = tex_chain.cook_stage_dag(stages)
    ref = fresh["stage_outputs"][0]["OUT"]
    if not torch.equal(rc.get(key0), ref):
        r.fail("STORE-51 store={0} content",
               f"maxdiff={(rc.get(key0) - ref).abs().max().item()}")
        return
    r.ok("STORE-51: store={0} stored exactly stage 0's boundary (content-correct) and left "
        "stage 2's -- equally eligible but not named -- absent from result_cache")


# ── 2. store=set() stores nothing, but still serves an existing boundary ───────────────────

def test_store51_empty_store_stores_nothing_but_still_serves_existing(r: SubTestResult):
    print("\n--- STORE-51: store=set() stores nothing this tick, but serves what is already "
          "cached ---")
    torch.manual_seed(5112)
    A = torch.rand(1, _H, _W, 3)
    B = torch.rand(1, _H, _W, 3)
    rc = tex_results.ResultCache()
    up = ("store51-empty-store-src", "second-source",)
    stages = _two_eligible_branches(A, B)
    key0, key2 = _keys_for(stages, up)
    roi = (1, 1, 4, 4, _W, _H)

    # (a) A fresh cache, everything dirty, store=set(): NOTHING is written, even though both
    # sources are eligible -- the negative half of this row.
    win_a = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True, dirty_from=0,
                                     result_cache=rc, upstream=up, store=set())
    if not _assert_shape(r, win_a, "STORE-51 store=set() seed"):
        return
    if rc.get(key0) is not None or rc.get(key2) is not None:
        r.fail("STORE-51 store=set() writes",
               f"expected NO entries after a store=set() cook, got key0={rc.get(key0)!r} "
               f"key2={rc.get(key2)!r}")
        return

    # (b) Seed the cache properly (store=None, today's behaviour), THEN re-cook with
    # dirty_from=1 (only stage 0 is clean) and store=set() again -- this can only produce a
    # correct result by SERVING stage 0's pre-existing boundary out of result_cache, which
    # store=set() must not block: reading is unconditional on store=, only WRITING is gated.
    tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True, dirty_from=0,
                             result_cache=rc, upstream=up)
    if rc.get(key0) is None:
        r.fail("STORE-51 seed for the serve half", "expected stage 0's boundary stored by "
               "the unrestricted seed cook")
        return

    roi2 = (9, 9, 3, 3, _W, _H)
    win2 = tex_chain.cook_stage_dag(
        stages, roi=roi2, roi_exec=True, dirty_from=1,
        valid=[(0, 0, _W, _H), None, None, None, None],
        result_cache=rc, upstream=up, store=set())
    fresh = tex_chain.cook_stage_dag(stages)
    ref = _crop(fresh["result"]["OUT"], roi2)
    got = win2["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("STORE-51 store=set() round trip", f"maxdiff={(got - ref).abs().max().item()}")
        return
    r.ok("STORE-51: store=set() wrote nothing to a fresh cache, and separately did not "
        "prevent a later tick from being served correctly out of a boundary a prior, "
        "unrestricted cook had already stored")


# ── 3. A genuinely windowed stage named in store= is still not stored (negative control) ───

def test_store51_windowed_stage_named_in_store_is_still_not_stored(r: SubTestResult):
    print("\n--- STORE-51: naming a genuinely WINDOWED stage in store= does not store it "
          "(negative control) ---")
    torch.manual_seed(5113)
    A = torch.rand(1, _H, _W, 3)
    B = torch.rand(1, _H, _W, 3)
    rc = tex_results.ResultCache()
    up = ("store51-negative-control-src", "second-source",)
    stages = _two_eligible_branches(A, B)
    key0, _key2 = _keys_for(stages, up)
    key1 = tex_chain.boundary_lineage_key(stages, 2, "cpu", "fp32", upstream=up)

    roi = (1, 1, 4, 4, _W, _H)
    win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True, dirty_from=0,
                                   result_cache=rc, upstream=up, store={0, 1})
    if not _assert_shape(r, win, "STORE-51 negative-control seed"):
        return

    if rc.get(key1) is not None:
        r.fail("STORE-51 negative control", "stage 1 was genuinely windowed this tick, and "
               "was still stored despite being named in store= -- a windowed output must "
               "NEVER be stored as a whole-frame boundary, regardless of store=")
        return
    if rc.get(key0) is None:
        r.fail("STORE-51 negative control", "stage 0 was eligible AND named in store=, but "
               "was not stored")
        return
    r.ok("STORE-51: store={0, 1} left the genuinely windowed stage 1 out of result_cache "
        "entirely -- naming a windowed stage in store= cannot force it to be stored -- while "
        "still storing the eligible, named stage 0")


# ── 4. Default (store= omitted) is unchanged: every eligible stage is stored ────────────────

def test_store51_default_store_none_stores_every_eligible_stage(r: SubTestResult):
    print("\n--- STORE-51: default store=None keeps today's behaviour (both eligible stages "
          "stored) ---")
    torch.manual_seed(5114)
    A = torch.rand(1, _H, _W, 3)
    B = torch.rand(1, _H, _W, 3)
    rc = tex_results.ResultCache()
    up = ("store51-default-src", "second-source",)
    stages = _two_eligible_branches(A, B)
    key0, key2 = _keys_for(stages, up)

    roi = (1, 1, 4, 4, _W, _H)
    # store= omitted entirely -- exactly the call shape every caller before this ask used.
    win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True, dirty_from=0,
                                   result_cache=rc, upstream=up)
    if not _assert_shape(r, win, "STORE-51 default seed"):
        return
    stored = sum(1 for k in (key0, key2) if rc.get(k) is not None)
    if stored != 2:
        r.fail("STORE-51 default count", f"expected both eligible stages stored by default "
               f"(store= omitted), got {stored}/2")
        return
    r.ok("STORE-51: with store= omitted (the default, and every pre-STORE-51 call shape), "
        "both eligible whole-frame stages are stored, unchanged in count from CKPT-51/"
        "JOINWIRE-50b's own behaviour")


# ── Validation: an out-of-range store index is a clear diagnostic, not a stack trace ────────

def test_store51_out_of_range_index_raises_clear_diagnostic(r: SubTestResult):
    print("\n--- STORE-51: an out-of-range store index is rejected with a clear diagnostic "
          "---")
    torch.manual_seed(5115)
    A = torch.rand(1, _H, _W, 3)
    B = torch.rand(1, _H, _W, 3)
    stages = _two_eligible_branches(A, B)  # 5 stages: valid indices are 0..4
    rc = tex_results.ResultCache()

    try:
        tex_chain.cook_stage_dag(stages, roi=(1, 1, 4, 4, _W, _H), roi_exec=True,
                                 dirty_from=0, result_cache=rc,
                                 upstream=("store51-oob-src",), store={7})
    except ValueError as e:
        msg = str(e)
        if "7" not in msg:
            r.fail("STORE-51 out-of-range diagnostic",
                   f"expected the bad index (7) named in the message, got: {msg!r}")
            return
        if "cook_stage_dag" not in msg:
            r.fail("STORE-51 out-of-range diagnostic",
                   f"expected the diagnostic to identify its own call, got: {msg!r}")
            return
        r.ok(f"STORE-51: store={{7}} against a 5-stage list raised a clear ValueError naming "
            f"the bad index: {msg!r}")
        return
    r.fail("STORE-51 out-of-range diagnostic", "expected a ValueError for store={7} against "
           "a 5-stage list, none was raised")
