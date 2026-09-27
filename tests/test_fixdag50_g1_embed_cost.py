"""FIX-DAG G1 (the efficiency review) — a non-sink windowed stage's re-embed no longer
allocates and zero-fills a full canvas, and the padded buffer it used to produce is no
longer part of `cook_stage_dag`'s public `stage_outputs` surface.

Red-first: at base `e3ba26b`, cooking the Merge-below-edit DAG (JOINWIRE-50's own shape)
windowed, at 4K, calls `torch.Tensor.new_zeros` once per non-sink windowed stage, and
`stage_outputs[1]["OUT"]` is a full-canvas-shaped tensor even though only a small window of
it is real (the rest is zero-filled padding a caller could misread as real pixels). Both are
false against this fix: `torch.Tensor.new_zeros` is never called by `cook_stage_dag`'s own
re-embed path at all (it now uses an ephemeral, uninitialised `new_empty` buffer that is
never assigned into `stage_outputs`), and `stage_outputs[1]["OUT"]` stays the bare crop.

CPU here (matches this repository's own CPU-only test convention for non-timing rows); a
per-tick wall-clock measurement (before/after, named by box+GPU) lives in this ask's
own record, not in the suite, per AGENTS.md's "Timing tests are the orchestrator's landing
ceremony" convention — this file asserts the ALLOCATION SHAPE, not a wall-clock number.
"""
from __future__ import annotations

from helpers import *

from TEX_Wrangle import tex_chain


def _merge_below_edit_stages(A, B):
    """Byte-for-byte the same DAG shape `test_joinwire50_dag_cook.py`'s own
    `_merge_below_edit_stages` builds — an edit of @A, an asymmetric-reach blur of @B, and a
    Merge reading both. Not imported from that file: this file is deliberately independent of
    JOINWIRE-50's own test module, matching this repository's existing convention of a
    private per-file DAG builder (`test_joinwire50b_checkpointed_dag_cook.py` has its own
    `_cp_merge_stages` rather than importing JOINWIRE-50's)."""
    return [
        {"code": "@OUT = @A + 0.1;", "bindings": {"A": A}},
        {"code": "@OUT = gauss_blur(@B, 3.0);", "bindings": {"B": B}},
        {"code": "@OUT = (@bg + @fg) * 0.5;", "bindings": {},
         "chain_inputs": {"bg": [0, "OUT"], "fg": [1, "OUT"]}},
    ]


def test_fixdag50_g1_no_zero_fill_per_windowed_stage(r: SubTestResult):
    print("\n--- FIX-DAG G1: no full-canvas zero-fill per non-sink windowed stage (4K) ---")
    torch.manual_seed(50100)
    # 3840x2160x4 fp32 — the exact shape the efficiency review and JOINWIRE-50b's own record
    # measured (132,710,400 bytes / 126.56 MiB per full canvas).
    W, H = 3840, 2160
    A = torch.rand(1, H, W, 4)
    B = torch.rand(1, H, W, 4)
    stages = _merge_below_edit_stages(A, B)
    roi = (100, 100, 200, 200, W, H)

    zero_fill_shapes = []
    orig_new_zeros = torch.Tensor.new_zeros

    def _counting_new_zeros(self, *a, **kw):
        zero_fill_shapes.append(tuple(a[0]) if a else tuple(kw.get("size", ())))
        return orig_new_zeros(self, *a, **kw)

    torch.Tensor.new_zeros = _counting_new_zeros
    try:
        win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True)
    finally:
        torch.Tensor.new_zeros = orig_new_zeros

    if zero_fill_shapes:
        r.fail("FIX-DAG G1 zero-fill count",
               f"expected 0 torch.Tensor.new_zeros calls from cook_stage_dag's own cook, "
               f"got {len(zero_fill_shapes)}: {zero_fill_shapes!r}")
        return

    if win["stages_windowed"] != 3 or win["stages_whole"] != 0:
        r.fail("FIX-DAG G1 setup",
               f"expected all 3 stages windowed on this DAG/roi, got "
               f"windowed={win['stages_windowed']} whole={win['stages_whole']}")
        return

    stage1_out = win["stage_outputs"][1]["OUT"]
    if list(stage1_out.shape[1:3]) == [H, W]:
        r.fail("FIX-DAG G1 stage_outputs contract",
               f"stage_outputs[1]['OUT'] is full-canvas-shaped ({list(stage1_out.shape)}) "
               f"-- expected the bare crop, never a re-embedded/padded full canvas")
        return
    if 1 not in win["stage_windows"]:
        r.fail("FIX-DAG G1 stage_windows contract",
               "stage 1 was genuinely windowed but stage_windows has no entry for it")
        return
    if tuple(win["stage_windows"][1])[:4] != (100, 100, 200, 200):
        r.fail("FIX-DAG G1 stage_windows contract",
               f"stage_windows[1] = {win['stage_windows'][1]!r}, expected the (100,100,200,200,...) "
               f"window stage 1 actually served")
        return

    r.ok(f"FIX-DAG G1: a 3-stage 4K Merge-below-edit DAG windows every stage with ZERO "
        f"torch.Tensor.new_zeros calls, and stage_outputs[1] stays a "
        f"{list(stage1_out.shape)} crop (stage_windows[1] names its offset) rather than a "
        f"[1, {H}, {W}, 4] padded full canvas")


def test_fixdag50_g1_pixel_identity_still_holds_at_4k(r: SubTestResult):
    """The whole point of the fix is a cost/exposure change, not a pixel change: the sink's
    result at 4K must still match a whole-frame cook of the same graph exactly, the same
    proof `test_joinwire50_merge_below_edit_pixel_identity` already runs at 32x32."""
    print("\n--- FIX-DAG G1: pixel identity holds at 4K after the re-embed change ---")
    torch.manual_seed(50101)
    W, H = 3840, 2160
    A = torch.rand(1, H, W, 4)
    B = torch.rand(1, H, W, 4)
    stages = _merge_below_edit_stages(A, B)
    roi = (100, 100, 200, 200, W, H)

    win = tex_chain.cook_stage_dag(stages, roi=roi, roi_exec=True)
    full = tex_chain.cook_stage_dag(stages)
    x0, y0, w, h, _W, _H = roi
    ref = full["result"]["OUT"][:, y0:y0 + h, x0:x0 + w]
    got = win["result"]["OUT"]
    if not torch.equal(got, ref):
        r.fail("FIX-DAG G1 4K pixel identity",
               f"maxdiff={(got - ref).abs().max().item()}")
        return
    r.ok("FIX-DAG G1: the sink's own window still matches the whole-frame cook exactly at "
        "4K, with the cheaper (uninitialised, not zero-filled) intermediate re-embed")
