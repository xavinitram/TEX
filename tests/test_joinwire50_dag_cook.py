"""JOINWIRE-50 — a DAG-shaped stage list, cooked node-by-node, windowed end-to-end.

`tex_roi.chain_windows_dag` (JOIN-49) has planned per-stage windows for a multi-input join
since v0.49, but nothing in the tree ever COOKED one — every real cook path (`cook_stage_list`
single-stage, `cook_checkpointed`'s suffix loop) is still linear-only. This file proves the
new caller (`tex_chain.cook_stage_dag`, plus `tex_roi.stage_dag_arg_halos` — the per-stage
reach resolver `StageSpec.arg_halo`'s own docstring named as owed) end-to-end: every window it
allows is proven by PIXEL IDENTITY (`torch.equal` windowed vs whole-frame) on the DAG shapes
the brief names — a Merge below an edit, a diamond, one node feeding two joins, `dirty_from`
mid-DAG, a convolve-style per-argument call, and a join whose inputs disagree on validity
(which must refuse) — with negative controls showing the refused case really would be wrong.
CPU always; a CUDA row mirrors JOIN-49's own gate (`torch.cuda.is_available()` alone).
"""
from __future__ import annotations

from helpers import *

from TEX_Wrangle import tex_chain


# ── Shared small DAG builders ──────────────────────────────────────────────────────────────

def _merge_below_edit_stages(A, B):
    """stage0: an EDIT of @A (the "below an edit" shape). stage1: a blur of @B (asymmetric
    per-argument reach — this is the exact JOIN-49 pixel-identity shape, now cooked for
    real rather than hand-windowed). stage2: the Merge, reading both."""
    return [
        {"code": "@OUT = @A + 0.1;", "bindings": {"A": A}},
        {"code": "@OUT = gauss_blur(@B, 3.0);", "bindings": {"B": B}},
        {"code": "@OUT = (@bg + @fg) * 0.5;", "bindings": {},
         "chain_inputs": {"bg": [0, "OUT"], "fg": [1, "OUT"]}},
    ]


def _diamond_stages(A):
    """stage0 feeds TWO consumers (stage1's blur, stage2's pass-through) — one input, two
    consumers: composition rule 1 (same-input union), not a join."""
    return [
        {"code": "@OUT = @A;", "bindings": {"A": A}},
        {"code": "@OUT = gauss_blur(@P, 2.0);", "bindings": {},
         "chain_inputs": {"P": [0, "OUT"]}},
        {"code": "@OUT = @P * 1.0;", "bindings": {},
         "chain_inputs": {"P": [0, "OUT"]}},
        {"code": "@OUT = (@bg + @fg) * 0.5;", "bindings": {},
         "chain_inputs": {"bg": [1, "OUT"], "fg": [2, "OUT"]}},
    ]


def _feeds_two_joins_stages(A, B, C):
    """stage1 (@B's source) feeds TWO different joins (stage2 and stage4), each of which
    also has its own second input — "one node feeding two joins", named separately from the
    diamond because both consumers are themselves joins, not plain pass-throughs."""
    return [
        {"code": "@OUT = @A;", "bindings": {"A": A}},                                # 0
        {"code": "@OUT = @B;", "bindings": {"B": B}},                                # 1
        {"code": "@OUT = (@x + gauss_blur(@y, 2.0)) * 0.5;", "bindings": {},
         "chain_inputs": {"x": [0, "OUT"], "y": [1, "OUT"]}},                        # 2
        {"code": "@OUT = @C;", "bindings": {"C": C}},                                # 3
        {"code": "@OUT = (gauss_blur(@x, 1.0) + @y) * 0.5;", "bindings": {},
         "chain_inputs": {"x": [1, "OUT"], "y": [3, "OUT"]}},                        # 4
        {"code": "@OUT = (@p + @q) * 0.5;", "bindings": {},
         "chain_inputs": {"p": [2, "OUT"], "q": [4, "OUT"]}},                        # 5 (sink)
    ]


def _crop(full, roi):
    x0, y0, w, h, _W, _H = roi
    return full[:, y0:y0 + h, x0:x0 + w]


# ── Merge below an edit ───────────────────────────────────────────────────────────────────

def test_joinwire50_merge_below_edit_pixel_identity(r: SubTestResult):
    print("\n--- JOINWIRE-50: Merge-below-an-edit, real cook, windowed vs whole-frame ---")
    torch.manual_seed(490)
    W = H = 32
    A = torch.rand(1, H, W, 3)
    B = torch.rand(1, H, W, 3)
    stages = _merge_below_edit_stages(A, B)
    roi = (10, 10, 8, 8, W, H)

    full = tex_chain.cook_stage_dag(stages)
    win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True)

    ref = _crop(full["result"]["OUT"], roi)
    got = win["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("JOINWIRE-50 merge-below-edit pixel identity",
               f"maxdiff={ (got - ref).abs().max().item() }")
        return
    if win["stages_windowed"] != 3 or win["stages_whole"] != 0:
        r.fail("JOINWIRE-50 merge-below-edit windowed count",
               f"expected 3 windowed/0 whole, got {win['stages_windowed']}/"
               f"{win['stages_whole']} (windows={win['windows']!r})")
        return
    if full["stages_windowed"] != 0 or full["stages_whole"] != 3:
        r.fail("JOINWIRE-50 merge-below-edit whole-frame count",
               f"the no-roi cook must report 0 windowed/3 whole, got "
               f"{full['stages_windowed']}/{full['stages_whole']}")
        return
    r.ok("JOINWIRE-50: Merge-below-an-edit windows all 3 stages and matches the whole-frame "
        "cook pixel-for-pixel")


def test_joinwire50_merge_below_edit_poisoned_fill_is_never_read(r: SubTestResult):
    """The load-bearing claim behind `_embed_window`: the region OUTSIDE a re-embedded
    intermediate's window is never read by anything this cook does. Proven, not merely
    argued — poison that region with a non-zero, non-finite sentinel instead of zeros and
    check the SINK still matches the whole-frame cook exactly. If any downstream stage's
    own `run_roi` ever read outside the window this function computed, the sentinel would
    leak into the result (or produce NaN outputs), and this test would catch it."""
    print("\n--- JOINWIRE-50: poisoned-fill proof (unread garbage is truly unread) ---")
    torch.manual_seed(491)
    W = H = 32
    A = torch.rand(1, H, W, 3)
    B = torch.rand(1, H, W, 3)
    stages = _merge_below_edit_stages(A, B)
    roi = (10, 10, 8, 8, W, H)
    full = tex_chain.cook_stage_dag(stages)
    ref = _crop(full["result"]["OUT"], roi)

    def _poisoned_embed(val, window):
        if not isinstance(val, torch.Tensor) or val.dim() < 3:
            return val
        x0, y0, w, h, W2, H2 = window
        full_t = val.new_full((val.shape[0], H2, W2, *val.shape[3:]), float("nan"))
        full_t[:, y0:y0 + h, x0:x0 + w] = val
        return full_t

    orig = tex_chain._embed_window
    tex_chain._embed_window = _poisoned_embed
    try:
        win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True)
    finally:
        tex_chain._embed_window = orig
    got = win["result"]["OUT"]
    if torch.isnan(got).any():
        r.fail("JOINWIRE-50 poisoned-fill", "NaN leaked into the sink's own output")
        return
    if not torch.equal(got, ref):
        r.fail("JOINWIRE-50 poisoned-fill",
               f"result changed under a poisoned fill, maxdiff="
               f"{(got - ref).abs().max().item()}")
        return
    r.ok("JOINWIRE-50: an NaN-poisoned unread region never reaches the sink — the fill is "
        "provably inert, not just conveniently zero")


# ── Diamond (same-input, multiple-consumer union) ────────────────────────────────────────

def test_joinwire50_diamond_pixel_identity(r: SubTestResult):
    print("\n--- JOINWIRE-50: diamond (one input, two consumers) ---")
    torch.manual_seed(492)
    W = H = 32
    A = torch.rand(1, H, W, 3)
    stages = _diamond_stages(A)
    roi = (12, 12, 6, 6, W, H)
    full = tex_chain.cook_stage_dag(stages)
    win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True)
    ref = _crop(full["result"]["OUT"], roi)
    got = win["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("JOINWIRE-50 diamond pixel identity",
               f"maxdiff={(got - ref).abs().max().item()}")
        return
    # stage0 is read by BOTH stage1 (blur, halo>0) and stage2 (halo 0) — its planned window
    # must be the UNION (i.e. at least as large as stage1's own demand), never merely
    # stage2's pointwise one; a wrong (too-small) union would have already failed the
    # pixel-identity check above, but the shape assertion pins the composition rule by name.
    w0 = win["windows"][0]
    if w0 is None or w0[2] < roi[2] or w0[3] < roi[3]:
        r.fail("JOINWIRE-50 diamond union window", f"stage 0's window {w0!r} looks too small")
        return
    r.ok("JOINWIRE-50: diamond union windows correctly and matches the whole-frame cook")


# ── One node feeding two joins ────────────────────────────────────────────────────────────

def test_joinwire50_node_feeds_two_joins_pixel_identity(r: SubTestResult):
    print("\n--- JOINWIRE-50: one node feeding two joins ---")
    torch.manual_seed(493)
    W = H = 32
    A = torch.rand(1, H, W, 3)
    B = torch.rand(1, H, W, 3)
    C = torch.rand(1, H, W, 3)
    stages = _feeds_two_joins_stages(A, B, C)
    roi = (14, 14, 6, 6, W, H)
    full = tex_chain.cook_stage_dag(stages)
    win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True)
    ref = _crop(full["result"]["OUT"], roi)
    got = win["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("JOINWIRE-50 node-feeds-two-joins pixel identity",
               f"maxdiff={(got - ref).abs().max().item()}")
        return
    if win["stages_windowed"] != 6:
        r.fail("JOINWIRE-50 node-feeds-two-joins windowed count",
               f"expected all 6 stages windowed, got {win['stages_windowed']} "
               f"(windows={win['windows']!r})")
        return
    r.ok("JOINWIRE-50: a node feeding two separate joins windows correctly on both edges "
        "and matches the whole-frame cook")


# ── dirty_from mid-DAG ────────────────────────────────────────────────────────────────────

def test_joinwire50_dirty_from_mid_dag_pixel_identity(r: SubTestResult):
    """A prior cook's stage 0 is CLEAN (unaffected by the edit) and supplied via
    `known_outputs`; only stages 1-2 recook, `dirty_from=1`. Proves the incremental path
    (the one an embedding host's zoomed-gesture windowing depends on) reaches the same pixels a fresh
    from-scratch cook of the edited graph would."""
    print("\n--- JOINWIRE-50: dirty_from mid-DAG (incremental edit) ---")
    torch.manual_seed(494)
    W = H = 28
    A = torch.rand(1, H, W, 3)
    B_old = torch.rand(1, H, W, 3)
    B_new = torch.rand(1, H, W, 3)

    old = tex_chain.cook_stage_dag(_merge_below_edit_stages(A, B_old))
    stage0_output = old["stage_outputs"][0]

    roi = (6, 6, 6, 6, W, H)
    stages_new = _merge_below_edit_stages(A, B_new)
    win = tex_chain.cook_stage_dag(
        stages_new, roi=roi, roi_exec=True, dirty_from=1,
        valid=[(0, 0, W, H), None, None], known_outputs={0: stage0_output})
    fresh = tex_chain.cook_stage_dag(stages_new)
    ref = _crop(fresh["result"]["OUT"], roi)
    got = win["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("JOINWIRE-50 dirty_from mid-DAG",
               f"maxdiff={(got - ref).abs().max().item()}")
        return
    if win["windows"][0] is not None:
        r.fail("JOINWIRE-50 dirty_from mid-DAG",
               f"stage 0 (below dirty_from) must not get a window, got {win['windows'][0]!r}")
        return
    if 0 in win["stage_outputs"] and win["stage_outputs"][0] is not stage0_output:
        r.fail("JOINWIRE-50 dirty_from mid-DAG",
               "stage 0 was recooked instead of reusing known_outputs")
        return
    r.ok("JOINWIRE-50: dirty_from mid-DAG reuses the clean prefix and matches a fresh cook "
        "of the edited graph")


def test_joinwire50_dirty_from_missing_known_output_raises(r: SubTestResult):
    """A dirty stage that needs a CLEAN upstream's value, with no `known_outputs` supplied
    for it, must raise rather than fabricate a value — the same fail-loud posture
    `chain_windows_dag`'s own topological guard already takes for a malformed graph."""
    print("\n--- JOINWIRE-50: missing known_outputs for a clean upstream raises ---")
    torch.manual_seed(495)
    W = H = 20
    A = torch.rand(1, H, W, 3)
    B = torch.rand(1, H, W, 3)
    stages = _merge_below_edit_stages(A, B)
    roi = (4, 4, 6, 6, W, H)
    try:
        tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True, dirty_from=1,
                                 valid=[(0, 0, W, H), None, None])   # known_outputs omitted
        r.fail("JOINWIRE-50 missing known_outputs", "expected ValueError, got a result")
        return
    except ValueError:
        r.ok("JOINWIRE-50: a dirty stage missing its clean upstream's known_outputs raises "
            "ValueError instead of fabricating a value")


# ── Convolve-style per-argument reach (both args unbounded -> whole-graph fallback) ───────

def test_joinwire50_convolve_style_falls_back_whole_frame(r: SubTestResult):
    """`convolve(@img, @kernel)` declares BOTH its own arg-0 footprint ('image') and a
    per-argument declaration for @kernel (REACH-48's `arg_footprint`) — so a stage that
    calls it is not ROI-3-executable at ALL (`roi_plan(...).executable` is False), and
    `stage_dag_arg_halos`'s whitelist gate must report WHOLE_FRAME for every one of its
    named inputs rather than trusting @kernel's own per-argument declaration in isolation.
    Proves the mechanism correctly DECLINES to narrow this shape (falls back to whole-frame
    for every stage, still bit-exact — never a crash, never a silently-wrong narrow)."""
    print("\n--- JOINWIRE-50: convolve-style call declines to narrow (whole-frame, correct) ---")
    torch.manual_seed(496)
    W = H = 24
    kernel = torch.rand(1, 3, 3, 1)
    img = torch.rand(1, H, W, 3)
    stages = [
        {"code": "@OUT = @K;", "bindings": {"K": kernel}},
        {"code": "@OUT = @I;", "bindings": {"I": img}},
        {"code": "@OUT = convolve(@img, @kernel);", "bindings": {},
         "chain_inputs": {"img": [1, "OUT"], "kernel": [0, "OUT"]}},
    ]
    roi = (4, 4, 6, 6, W, H)
    full = tex_chain.cook_stage_dag(stages)
    win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True)
    got, ref = win["result"]["OUT"], full["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("JOINWIRE-50 convolve-style fallback",
               f"maxdiff={(got - ref).abs().max().item()}")
        return
    if win["stages_windowed"] != 0 or win["stages_whole"] != 3:
        r.fail("JOINWIRE-50 convolve-style fallback count",
               f"expected 0 windowed / 3 whole, got {win['stages_windowed']}/"
               f"{win['stages_whole']}")
        return
    r.ok("JOINWIRE-50: a convolve-style (per-argument-declared, but globally unbounded) "
        "call correctly declines every stage to whole-frame and still matches exactly")


# ── Divergent-validity join: must refuse ─────────────────────────────────────────────────

_BG_CODE = "@OUT = @A;"
_FG_BLUR_CODE = "@OUT = gauss_blur(@A, 3.0);"
_JOIN_CODE = "@OUT = (@bg + @fg) * 0.5;"
_JOIN_HALO_FG = 9    # ceil(3 * 3.0), gauss_blur's own halo_arg mult


def test_joinwire50_divergent_validity_join_refuses_and_stays_safe(r: SubTestResult):
    """A join whose two inputs disagree on validity (fg's canvas is only fresh over a small
    corner far from the window the join needs) must refuse to plan a narrow window at all
    (`chain_windows_dag` returns `None`) — and `cook_stage_dag` must still produce a
    CORRECT picture by falling back to recooking every stage from source (never silently
    narrow over the stale region), even though this test never even asks it to use
    `known_outputs`."""
    print("\n--- JOINWIRE-50: divergent-validity join refuses, cook_stage_dag stays correct ---")
    torch.manual_seed(497)
    W = H = 40
    src = torch.rand(1, H, W, 3)
    stages = [
        {"code": _BG_CODE, "bindings": {"A": src}},
        {"code": _FG_BLUR_CODE, "bindings": {"A": src}},
        {"code": _JOIN_CODE, "bindings": {},
         "chain_inputs": {"bg": [0, "OUT"], "fg": [1, "OUT"]}},
    ]
    # fg's tracked valid region: only a small corner near (0,0) -- everywhere else is
    # (by construction) stale relative to `dirty_from`. bg is valid everywhere.
    roi_far = (30, 30, 6, 6, W, H)         # far from fg's "valid" corner
    valid = [(0, 0, W, H), (0, 0, 8, 8), None]
    win = tex_chain.cook_stage_dag(stages, roi=roi_far, roi_exec=True, dirty_from=2,
                                   valid=valid)
    if win["windows"] is not None:
        r.fail("JOINWIRE-50 divergent-validity refusal",
               f"expected chain_windows_dag to refuse (None), got {win['windows']!r}")
        return
    full = tex_chain.cook_stage_dag(stages)
    # A whole-plan refusal means EVERY stage (including the sink) cooks whole-frame — the
    # documented "cook the whole chain from the source" contract, not a crop of it.
    ref = full["result"]["OUT"]
    got = win["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("JOINWIRE-50 divergent-validity refusal",
               f"the safe whole-graph fallback itself diverged, maxdiff="
               f"{(got - ref).abs().max().item()}")
        return
    if win["stages_windowed"] != 0 or win["stages_whole"] != 3:
        r.fail("JOINWIRE-50 divergent-validity refusal",
               f"expected every stage whole-frame on refusal, got "
               f"{win['stages_windowed']} windowed / {win['stages_whole']} whole")
        return
    r.ok("JOINWIRE-50: the divergent-validity join is refused a narrow window, and the "
        "whole-graph fallback it takes instead is still pixel-correct")


def test_joinwire50_negative_control_divergent_validity_would_be_wrong_if_served(r: SubTestResult):
    """The refusal above is not bureaucratic caution: prove it by constructing the EXACT
    scenario chain_windows_dag refuses, then SERVING the narrow window anyway (bypassing
    the refusal, mirroring test_join49's own negative control) — reading fg from a canvas
    that is only actually fresh over a small corner. The result must come back WRONG
    (nonzero maxdiff against the true whole-frame cook), proving the refusal prevents a
    real, not merely theoretical, silent bad picture."""
    print("\n--- JOINWIRE-50 negative control: serving the refused window IS wrong ---")
    torch.manual_seed(498)
    W = H = 40
    src = torch.rand(1, H, W, 3)
    from TEX_Wrangle import tex_engine
    bg_whole = tex_engine.cook(_BG_CODE, {"A": src}, device_mode="cpu",
                              precision="fp32").outputs["OUT"]
    fg_whole = tex_engine.cook(_FG_BLUR_CODE, {"A": src}, device_mode="cpu",
                              precision="fp32").outputs["OUT"]
    whole_out = tex_engine.cook(_JOIN_CODE, {"bg": bg_whole, "fg": fg_whole},
                                device_mode="cpu", precision="fp32").outputs["OUT"]

    # fg's canvas as it ACTUALLY stands: fresh only over an 8x8 corner, stale (zero) elsewhere.
    stale_fg = torch.zeros_like(fg_whole)
    stale_fg[:, 0:8, 0:8] = fg_whole[:, 0:8, 0:8]

    roi_far = (30, 30, 6, 6, W, H)
    x0, y0, w, h = roi_far[:4]
    # Serve the join narrow ANYWAY, reading the stale fg canvas over the window the refused
    # plan would have used (bg 0-halo, fg _JOIN_HALO_FG-halo -- the exact composition the
    # refused chain_windows_dag call would have grown, had it not refused).
    pad = _JOIN_HALO_FG
    fx0, fy0 = max(0, x0 - pad), max(0, y0 - pad)
    fx1, fy1 = min(W, x0 + w + pad), min(H, y0 + h + pad)
    bg_window = bg_whole[:, y0:y0 + h, x0:x0 + w]
    fg_window = stale_fg[:, fy0:fy1, fx0:fx1]
    # fg_window is the (grown) cook-region crop; the join itself reads it at the SAME
    # (ungrown) window as bg since the join's own code has no spatial op -- crop back down.
    lx, ly = x0 - fx0, y0 - fy0
    fg_for_join = fg_window[:, ly:ly + h, lx:lx + w]
    served = ((bg_window + fg_for_join) * 0.5)
    ref = whole_out[:, y0:y0 + h, x0:x0 + w]
    if torch.equal(served, ref):
        r.fail("JOINWIRE-50 negative control", "expected the stale-fg serve to be WRONG, "
               "but it matched the true whole-frame cook")
        return
    maxdiff = (served - ref).abs().max().item()
    r.ok(f"JOINWIRE-50: serving the divergent-validity case anyway IS pixel-wrong "
        f"(maxdiff={maxdiff}), confirming the refusal is load-bearing, not conservative "
        f"decoration")


# ── Invariant 7: roi=None is unaffected ───────────────────────────────────────────────────

def test_joinwire50_roi_none_is_a_noop(r: SubTestResult):
    """Invariant #7: a `cook_stage_dag` call with `roi=None` (every pre-existing caller,
    since this function is brand new) plans no window at all and cooks every stage
    whole-frame — the identical result `cook_stage_list` would give per stage, one at a
    time, with no DAG machinery invoked."""
    print("\n--- JOINWIRE-50: roi=None plans nothing (invariant 7) ---")
    torch.manual_seed(499)
    W = H = 16
    A = torch.rand(1, H, W, 3)
    B = torch.rand(1, H, W, 3)
    stages = _merge_below_edit_stages(A, B)
    out = tex_chain.cook_stage_dag(stages)
    if out["windows"] is not None:
        r.fail("JOINWIRE-50 roi=None invariant", f"expected windows=None, got {out['windows']!r}")
        return
    if out["stages_windowed"] != 0 or out["stages_whole"] != 3:
        r.fail("JOINWIRE-50 roi=None invariant",
               f"expected 0 windowed / 3 whole, got {out['stages_windowed']}/"
               f"{out['stages_whole']}")
        return
    # Must equal cooking each stage individually via cook_stage_list (bit-exact).
    b0 = tex_chain.cook_stage_list([stages[0]])
    b1 = tex_chain.cook_stage_list([stages[1]])
    bindings2 = dict(stages[2].get("bindings") or {})
    bindings2["bg"], bindings2["fg"] = b0["OUT"], b1["OUT"]
    b2 = tex_chain.cook_stage_list([dict(stages[2], bindings=bindings2)])
    if not torch.equal(out["result"]["OUT"], b2["OUT"]):
        r.fail("JOINWIRE-50 roi=None invariant", "diverged from the manual per-stage cook")
        return
    r.ok("JOINWIRE-50: roi=None plans no window and matches a manual per-stage cook exactly")


# ── Topological guard ─────────────────────────────────────────────────────────────────────

def test_joinwire50_forward_reference_raises(r: SubTestResult):
    """A stage naming a chain_inputs index that is not strictly earlier must raise, mirroring
    `chain_windows_dag`'s own topological guard -- a defect in the CALLER's graph
    construction, not a case this function can serve any answer for."""
    print("\n--- JOINWIRE-50: a forward chain_inputs reference raises ---")
    stages = [
        {"code": "@OUT = (@bg + @fg) * 0.5;", "bindings": {},
         "chain_inputs": {"bg": [1, "OUT"], "fg": [1, "OUT"]}},   # stage 0 names stage 1!
        {"code": "@OUT = @A;", "bindings": {"A": torch.rand(1, 8, 8, 3)}},
    ]
    try:
        tex_chain.cook_stage_dag(stages)
        r.fail("JOINWIRE-50 forward reference", "expected ValueError, got a result")
    except ValueError:
        r.ok("JOINWIRE-50: a forward chain_inputs reference raises ValueError")


# ── CUDA (gated on device presence alone, JOIN-49's own convention) ──────────────────────

def test_joinwire50_merge_below_edit_pixel_identity_cuda(r: SubTestResult):
    print("\n--- JOINWIRE-50: Merge-below-an-edit, CUDA ---")
    if not torch.cuda.is_available():
        r.skip("JOINWIRE-50 CUDA pixel identity", "no CUDA on this box")
        return
    torch.manual_seed(500)
    W = H = 32
    A = torch.rand(1, H, W, 3, device="cuda")
    B = torch.rand(1, H, W, 3, device="cuda")
    stages = _merge_below_edit_stages(A, B)
    roi = (10, 10, 8, 8, W, H)
    full = tex_chain.cook_stage_dag(stages, device="cuda")
    win = tex_chain.cook_stage_dag(stages, device="cuda", roi=roi, roi_exec=True)
    ref = _crop(full["result"]["OUT"], roi)
    got = win["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("JOINWIRE-50 CUDA merge-below-edit", f"maxdiff={(got - ref).abs().max().item()}")
        return
    r.ok("JOINWIRE-50: Merge-below-an-edit matches on CUDA too")
