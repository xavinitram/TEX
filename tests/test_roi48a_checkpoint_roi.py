"""ROI-48A — `roi=`/`roi_exec=` on `tex_checkpoint.cook_checkpointed`, with the SAME
contract `tex_engine.cook(roi=...)` already has: an armed, well-formed, ROI-executable
window narrows the cook; anything else is a documented no-op that falls back to the
whole-frame serve, exactly as before this ask (an embedding host's ranked ask 2).

The scenario this exists for: a mid-graph edit lands on the LAST stage of a chain whose
earlier stages are already checkpointed. `cook_checkpointed` splices the cached prefix's
boundary and re-cooks only the SUFFIX — often exactly one stage, which is precisely the
shape `roi=` can now narrow too, so an edit near a checkpoint recooks only the changed
window instead of the whole frame.

Window semantics, lineage/boundary keys, refusal behaviour:
  * the SUFFIX cook (the whole chain when nothing is cached yet, or the cut's suffix when
    it is) is the only thing that can narrow — `cook_stage_list`'s own gate declines
    a fused (>1 stage) suffix, a LATENT, an unarmed/malformed/whole-frame window, or a
    program `roi_plan` cannot prove executable, and reports the decline on
    `tier_trace.last_roi()` exactly like `tex_engine.cook` does;
  * the CACHED boundary (the prefix's output) is never affected by `roi=` at all — it is
    always materialized whole-frame by `materialize()` (which does not accept `roi`, on
    purpose: a checkpoint must serve ANY future window, not one baked to a single
    request), so `boundary_lineage_key` needs no change and a windowed result can never be
    mistaken for — or stored as — a whole frame;
  * `roi=None` (every caller before this ask) never touches any of the new code at all —
    pinned below by a call-count spy, not a timing.
"""
from helpers import *

from TEX_Wrangle import tex_checkpoint as CK
from TEX_Wrangle import tex_engine, tex_results
from TEX_Wrangle import tex_roi as _R


# Three plain pointwise prefix stages, then a $mode-switched tail (mirrors
# test_roi48a_uniform_branch.py's BLUR_SRC): mode=0 is a bounded, narrowable `gauss_blur`;
# mode=1 is a whole-image gather. Chained through `@IN`, the stage-list convention.
_PREFIX = [
    "@OUT = vec4(@IN.rgb * 1.05, 1.0);",
    "@OUT = vec4(max(@IN.rgb - vec3(0.02), vec3(0.0)), 1.0);",
    "@OUT = vec4(spow(@IN.rgb, vec3(0.95)), 1.0);",
]
_TAIL = """
i$mode = 0;
f$sigma = 2.0;
if ($mode == 0) {
    @OUT = vec4(gauss_blur(@IN, $sigma).rgb, 1.0);
} else {
    @OUT = vec4(sample(@IN, u, v).rgb, 1.0);
}
"""


def _stages(src, mode=0):
    out = []
    for i, code in enumerate(_PREFIX):
        out.append({"code": code, "chain_input": (None if i == 0 else "IN"),
                    "bindings": ({"IN": src} if i == 0 else {})})
    out.append({"code": _TAIL, "chain_input": "IN", "bindings": {"mode": mode, "sigma": 2.0}})
    return out


def test_roi48a_checkpoint_windowed_pixel_identity(r: SubTestResult):
    print("\n--- ROI-48A: cook_checkpointed(roi=...) windows the suffix, CPU and CUDA ---")
    N = len(_PREFIX) + 1     # 3 prefix stages + the $mode tail
    cut = N - 1              # cache everything BEFORE the tail -> suffix is exactly 1 stage
    W, H = 48, 40
    roi = (10, 8, 16, 14, W, H)
    x0, y0, w, h, _, _ = roi
    up = ("roi48a-src",)
    try:
        for device in devices():
            torch.manual_seed(2048)
            src = torch.rand(1, H, W, 3, device=device)
            stages = _stages(src, mode=0)

            _R.clear_roi_memo()
            full = tex_engine.cook_stage_list(stages, device=device, precision="fp32")["OUT"]

            cache = tex_results.ResultCache()
            done = CK.materialize(_stages(src, mode=0), cache, device=device,
                                  precision="fp32", upstream=up, cuts=[cut])
            if sorted(done) != [cut]:
                r.fail(f"ROI-48A checkpoint harvest ({device})",
                       f"materialized {done}, expected [{cut}]")
                continue

            _R.clear_roi_memo()
            from TEX_Wrangle.tex_runtime import tier_trace as _tt
            _tt.reset()
            got = CK.cook_checkpointed(stages, cache, device=device, precision="fp32",
                                       upstream=up, cuts=[cut], roi=roi, roi_exec=True)["OUT"]
            served = _tt.last_roi()[0]
            if served != roi:
                r.fail(f"ROI-48A checkpoint window ({device})",
                       f"cook_checkpointed did not report a served window: {_tt.last_roi()}")
                continue
            crop = full[:, y0:y0 + h, x0:x0 + w]
            if tuple(got.shape) == tuple(crop.shape) and torch.equal(got, crop):
                r.ok(f"{device}: cook_checkpointed(roi=...) suffix is torch.equal the "
                     f"whole-frame crop")
            else:
                md = (got.float() - crop.float()).abs().max().item() \
                    if tuple(got.shape) == tuple(crop.shape) else float("nan")
                r.fail(f"ROI-48A checkpoint pixel identity ({device})",
                       f"shape {tuple(got.shape)} vs {tuple(crop.shape)}, maxdiff {md:.3e}")
    except Exception as e:
        r.fail("ROI-48A checkpoint pixel identity", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


def test_roi48a_checkpoint_roi_refusal_never_served_as_whole_frame(r: SubTestResult):
    print("\n--- ROI-48A: a multi-stage suffix declines the window (never a mislabeled whole frame) ---")
    # cut=1: prefix=[stage0], suffix=[stage1,stage2,tail] -- THREE stages, so
    # cook_stage_list's own single-stage gate must decline (the same posture as a fused
    # chain under tex_engine.cook). The important thing is what it does NOT do: it must
    # never report `roi` served while actually returning the whole frame.
    up = ("roi48a-src2",)
    roi = (4, 4, 8, 8, 48, 40)
    try:
        torch.manual_seed(4)
        src = torch.rand(1, 40, 48, 3)
        stages = _stages(src, mode=0)
        cache = tex_results.ResultCache()
        done = CK.materialize(_stages(src, mode=0), cache, device="cpu", precision="fp32",
                              upstream=up, cuts=[1])
        if 1 not in done:
            r.fail("ROI-48A multi-stage suffix setup", f"materialized {done}, expected 1 in it")
            return
        _R.clear_roi_memo()
        from TEX_Wrangle.tex_runtime import tier_trace as _tt
        _tt.reset()
        out = CK.cook_checkpointed(stages, cache, device="cpu", precision="fp32",
                                   upstream=up, cuts=[1], roi=roi, roi_exec=True)["OUT"]
        served = _tt.last_roi()[0]
        if served is not None:
            r.fail("ROI-48A multi-stage suffix", f"a 3-stage suffix reported a served window: {served}")
        elif tuple(out.shape[1:3]) == (40, 48):
            r.ok("a multi-stage suffix declines the window and returns the true whole frame")
        else:
            r.fail("ROI-48A multi-stage suffix",
                  f"declined but shape {tuple(out.shape)} is not the whole {40}x{48} frame")
    except Exception as e:
        r.fail("ROI-48A multi-stage suffix refusal", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


def test_roi48a_checkpoint_roi_none_is_a_no_op(r: SubTestResult):
    print("\n--- ROI-48A invariant #7: roi=None never touches the new code (counts, not timings) ---")
    # A call-count spy, not a timing: `tier_trace.record_roi` is the ONE function every new
    # branch in `cook_stage_list` funnels through before it can do anything else (arm,
    # decline, or serve). `roi=None` must never call it at all -- proving the new gate
    # block is skipped in its entirety, not merely that it happens to decide "no window"
    # every time. `roi=<a real window>` on the SAME chain calls it, so the spy is proven to
    # actually observe calls rather than being wired to something dead.
    from TEX_Wrangle.tex_runtime import tier_trace as _tt
    calls = []
    orig = _tt.record_roi

    def _spy(*a, **kw):
        calls.append((a, kw))
        return orig(*a, **kw)

    N = len(_PREFIX) + 1
    cut = N - 1
    up = ("roi48a-src3",)
    torch.manual_seed(9)
    src = torch.rand(1, 40, 48, 3)
    stages = _stages(src, mode=0)
    cache = tex_results.ResultCache()
    CK.materialize(_stages(src, mode=0), cache, device="cpu", precision="fp32",
                   upstream=up, cuts=[cut])
    try:
        _tt.record_roi = _spy
        _R.clear_roi_memo()
        calls.clear()
        CK.cook_checkpointed(stages, cache, device="cpu", precision="fp32",
                             upstream=up, cuts=[cut])           # roi=None (the default)
        if calls:
            r.fail("ROI-48A invariant #7", f"roi=None still called record_roi: {calls}")
        else:
            r.ok("roi=None: record_roi (and everything behind it) is never called")

        calls.clear()
        _R.clear_roi_memo()
        CK.cook_checkpointed(stages, cache, device="cpu", precision="fp32", upstream=up,
                             cuts=[cut], roi=(2, 2, 6, 6, 48, 40), roi_exec=True)
        if calls:
            r.ok(f"roi=<window>: record_roi IS called ({len(calls)}x) -- the spy observes")
        else:
            r.fail("ROI-48A spy sanity", "roi=<window> never called record_roi either")
    except Exception as e:
        r.fail("ROI-48A invariant #7 (roi=None no-op)", f"{type(e).__name__}: {e}")
    finally:
        _tt.record_roi = orig
        _R.clear_roi_memo()
