"""JOIN-49 — windows across multi-input joins (host ask 3, TIERS-48-design.md SS B).

`chain_windows` composes ROI margins across a LINEAR chain (stage i has exactly one producer
and one consumer). `chain_windows_dag`/`StageSpec` generalise the same composition to a DAG:
per-stage input maps, same-input union (SS B.2 rule 1), per-argument backward halo projection
(SS B.2 rule 2), and the divergent-validity refusal case (SS B.3).

`chain_windows` itself is UNTOUCHED by this lane — every existing call site and test keeps
calling exactly the function it always did. What this file proves instead: `chain_windows_dag`
fed the synthesized linear stage list is BYTE-IDENTICAL to `chain_windows` (the "the existing
linear API is the inputs=(i-1,) special case" contract), the two NEW composition rules do what
SS B.2 says, the divergent-validity refusal fires exactly when SS B.3 says it must, and every
window this module allows for a real multi-input join is proven by PIXEL IDENTITY: a windowed
cook (crop, run, patch) equals the whole-frame cook, on CPU always and on CUDA whenever a
device is present (FIX-ROI49 Q2: gated on `torch.cuda.is_available()` alone, same as every
other CUDA-only row in this tree — no lease/env-var vocabulary). Negative controls (cases that
MUST refuse) are pixel-proven too — the case is "this window is silently wrong", not merely
"this window is smaller than the frame".
"""
from __future__ import annotations

import itertools
import random

from helpers import *

from TEX_Wrangle import tex_engine, tex_roi as R
from TEX_Wrangle.tex_roi import StageSpec, chain_windows_dag


# ── Part 1: the linear special case is byte-identical to chain_windows ───────────────────────

def _linear_stages(halos):
    return [StageSpec(h, () if i == 0 else (i - 1,)) for i, h in enumerate(halos)]


def _assert_same(r: SubTestResult, label: str, halos, roi, dirty_from=0, valid=None,
                 declined=()):
    want = R.chain_windows(halos, roi, dirty_from, valid=valid, declined=declined)
    got = chain_windows_dag(_linear_stages(halos), roi, dirty_from, valid=valid,
                            declined=declined)
    if got != want:
        r.fail(f"JOIN-49 linear-equivalence oracle ({label})",
               f"chain_windows={want!r} but chain_windows_dag(linear)={got!r} for "
               f"halos={halos!r} roi={roi!r} dirty_from={dirty_from!r} valid={valid!r} "
               f"declined={declined!r}")
        return False
    return True


def test_join49_linear_equivalence_named_corpus(r: SubTestResult):
    """The exact call shapes the existing `chain_windows` test files already exercise
    (test_v032_region.py, test_v033_phase0.py, test_scale47b_halo_math.py,
    test_cache10_region_advisory.py) — the "existing linear corpus" the brief names."""
    print("\n--- JOIN-49: linear-equivalence oracle over the named existing corpus ---")
    W = H = 64
    cases = [
        ([0, 0, 4, 0], (40, 40, 16, 16, W, H), 0, None, ()),
        ([1, 2, 3, 1], (40, 40, 16, 16, W, H), 0, None, ()),
        ([1, 1, 1, 1], (10, 10, 8, 8, W, H), 2, None, ()),
        ([0, 0, 4, 0], (8, 8, 16, 16, 128, 128), 0, None, ()),
        ([0, 0, 4, 0], (96, 96, 16, 16, 128, 128), 2,
         [(8 - p, 8 - p, 16 + 2 * p, 16 + 2 * p, 128, 128) for p in (0, 0, 4, 4)], ()),
        ([1, 1, 4, 1], (0, 0, 4, 4, 16, 16), 1, [(0, 0, 16, 16, 16, 16)] * 4, [1]),
        ([1, 1, 4, 1], (0, 0, 4, 4, 16, 16), 1, [None] * 4, [1]),
        ([1, 1, 1, 1], (0, 0, 4, 4, 16, 16), 5, [(0, 0, 16, 16, 16, 16)] * 4, ()),
        ([1, 1, 1, 1], (0, 0, 4, 4, 16, 16), 5, None, ()),
        ([R.WHOLE_FRAME, 1, 1], (0, 0, 4, 4, W, H), 0, None, ()),
        ([], (0, 0, W, H, W, H), 0, None, ()),
        ([5], (0, 0, W, H, W, H), 0, None, ()),
        ([5], (0, 0, W, H, W, H), 0, [None], ()),
    ]
    all_ok = True
    for i, (halos, roi, dirty_from, valid, declined) in enumerate(cases):
        all_ok &= _assert_same(r, f"row {i}", halos, roi, dirty_from, valid, declined)
    if all_ok:
        r.ok(f"JOIN-49: chain_windows_dag(linear) == chain_windows over "
             f"{len(cases)} named corpus rows")


def test_join49_linear_equivalence_random_sweep(r: SubTestResult):
    """A randomized oracle sweep over the same degenerate (linear) input shape — dirty_from,
    valid presence/values, declined sets, window position, all varied — because the named
    corpus above is necessarily finite and this is a correctness-sensitive walk (the linear
    version needed two silent-wrong fixes before it was trusted; ROI-48A/FIX-ROI's own
    history). Deterministic seed, so a failure is reproducible."""
    print("\n--- JOIN-49: linear-equivalence oracle, randomized sweep ---")
    rng = random.Random(20490)
    W = H = 96
    all_ok = True
    n_cases = 300
    for _ in range(n_cases):
        n = rng.randint(0, 6)
        halos = [rng.choice([0, 1, 2, 3, R.WHOLE_FRAME]) for _ in range(n)]
        x0 = rng.randint(0, W - 1)
        y0 = rng.randint(0, H - 1)
        w = rng.randint(1, W - x0)
        h = rng.randint(1, H - y0)
        roi = (x0, y0, w, h, W, H)
        dirty_from = rng.randint(-1, n + 1)
        if rng.random() < 0.5 or n == 0:
            valid = None
        else:
            valid = []
            for _ in range(n):
                if rng.random() < 0.4:
                    valid.append(None)
                else:
                    vx0 = rng.randint(0, W - 1)
                    vy0 = rng.randint(0, H - 1)
                    valid.append((vx0, vy0, rng.randint(1, W - vx0), rng.randint(1, H - vy0)))
        declined = tuple(sorted(set(rng.sample(range(max(n, 1)), k=rng.randint(0, min(2, n))))))\
            if n > 0 else ()
        all_ok &= _assert_same(r, "random", halos, roi, dirty_from, valid, declined)
    if all_ok:
        r.ok(f"JOIN-49: chain_windows_dag(linear) == chain_windows over "
             f"{n_cases} randomized configurations (seed 20490)")


def test_join49_dag_walker_rejects_non_topological_input(r: SubTestResult):
    """A stage naming an input that is not strictly earlier (a cycle, or a forward
    reference) has no meaning in this model — `chain_windows` never had to reject this
    because a flat `halos` list cannot even express it. Refuse loudly rather than compute a
    plausible-looking wrong answer."""
    print("\n--- JOIN-49: non-topological stage graphs are refused, not silently walked ---")
    bad_shapes = [
        [StageSpec(0, (1,)), StageSpec(0, ())],       # stage 0 reads stage 1 (forward ref)
        [StageSpec(0, (0,))],                          # self-loop
        [StageSpec(0, ()), StageSpec(0, (5,))],        # out-of-range input
    ]
    all_ok = True
    for stages in bad_shapes:
        try:
            chain_windows_dag(stages, (0, 0, 1, 1, 4, 4))
            all_ok = False
            r.fail("JOIN-49 topological guard", f"{stages!r} was accepted, not refused")
        except ValueError:
            pass
    if all_ok:
        r.ok("JOIN-49: every non-topological stage graph raised ValueError")


# ── Part 2: the two NEW composition rules, in pure arithmetic ────────────────────────────────

def test_join49_same_input_multiple_consumer_union(r: SubTestResult):
    """SS B.2 rule 1: stage 0 feeds BOTH stage 1 (halo 6) and stage 2 (halo 0, a
    pass-through), a diamond. Stage 0's required window must be the UNION of what each
    demands — not stage 1's alone (which would starve stage 2's demand if it were the
    wider one) and not stage 2's alone (which would starve stage 1's halo)."""
    print("\n--- JOIN-49: same-input union across two consumers (diamond) ---")
    W = H = 64
    # stage 0: source. stage 1: reads 0, halo 6. stage 2: reads 0 AND 1 (the sink), halo 0.
    stages = [StageSpec(0, ()), StageSpec(6, (0,)), StageSpec(0, (0, 1))]
    roi = (20, 20, 4, 4, W, H)
    out = chain_windows_dag(stages, roi, dirty_from=0)
    if out is None:
        r.fail("JOIN-49 diamond union", "plan unexpectedly refused")
        return
    w0 = out[0]
    # Demand from stage 1: grow(out[1], stage1.halo=6). out[1] = grow(out[2]=roi, stage2's
    # halo_for(1)=0) = roi itself. So stage-1-via demand = grow(roi, 6).
    via_1 = (max(0, 20 - 6), max(0, 20 - 6), 4 + 12, 4 + 12, W, H)
    # Demand from stage 2 directly (stage 2 reads stage 0 too, halo 0): grow(roi, 0) = roi.
    via_2 = roi
    want = (min(via_1[0], via_2[0]), min(via_1[1], via_2[1]),
            max(via_1[0] + via_1[2], via_2[0] + via_2[2]) - min(via_1[0], via_2[0]),
            max(via_1[1] + via_1[3], via_2[1] + via_2[3]) - min(via_1[1], via_2[1]), W, H)
    if w0 == want:
        r.ok(f"JOIN-49: diamond union window[0] = {w0} (union of both consumers' demands)")
    else:
        r.fail("JOIN-49 diamond union", f"want {want}, got {w0}")

    # NEGATIVE CONTROL: `via_1` happens to be a superset of `via_2` here (padding from the
    # same underlying box always nests), so the correct union legitimately EQUALS `via_1` —
    # that is not a bug. The bug this guards against is a walker that used only stage 2's
    # (smaller, dominated) demand instead of the max/union: prove `via_1 != via_2` (so the
    # two demands are actually distinguishable) and that the real answer is NOT the smaller
    # one alone.
    if via_1 == via_2:
        r.fail("JOIN-49 diamond union (negative control)",
               "test setup is vacuous: the two consumers' demands coincide")
    elif w0 == via_2:
        r.fail("JOIN-49 diamond union (negative control)",
               "window equals ONLY the smaller consumer's demand — the larger consumer's "
               "halo was dropped, not unioned")
    else:
        r.ok("JOIN-49: window is the union (here, correctly dominated by the larger demand), "
             "not the smaller consumer's demand alone")


def test_join49_per_argument_backward_projection(r: SubTestResult):
    """SS B.2 rule 2: a join stage 2 reads stage 0 (as its `bg` argument, halo 0 — a plain
    pass-through) and stage 1 (as its `fg` argument, halo 9 — as if it fed the join through
    a feathering blur). The SAME upstream union machinery must NOT blur the two demands
    together: stage 1's window grows by 9, stage 0's grows by 0, from the SAME consumer."""
    print("\n--- JOIN-49: per-argument backward halo projection (asymmetric join inputs) ---")
    W = H = 64
    stages = [
        StageSpec(0, ()),                                  # 0: bg source
        StageSpec(0, ()),                                  # 1: fg source
        StageSpec(0, (0, 1), arg_halo={0: 0, 1: 9}),        # 2: join, asymmetric reach
    ]
    roi = (30, 30, 4, 4, W, H)
    out = chain_windows_dag(stages, roi, dirty_from=0)
    if out is None:
        r.fail("JOIN-49 asymmetric join", "plan unexpectedly refused")
        return
    want_bg = roi                                            # halo 0 -> unchanged
    want_fg = (30 - 9, 30 - 9, 4 + 18, 4 + 18, W, H)          # halo 9 -> grown by 9
    ok = out[0] == want_bg and out[1] == want_fg
    if ok:
        r.ok(f"JOIN-49: bg window {out[0]} unchanged, fg window {out[1]} grown by 9 — the "
             "SAME consumer projects two different demands backward")
    else:
        r.fail("JOIN-49 asymmetric join", f"want bg={want_bg} fg={want_fg}, got "
               f"bg={out[0]} fg={out[1]}")

    # NEGATIVE CONTROL: a walker that used the uniform `halo` (0, since no override applied
    # at the STAGE level) for both inputs would report bg AND fg both == roi, silently
    # under-sizing fg's true 9px reach.
    if out[1] == roi:
        r.fail("JOIN-49 asymmetric join (negative control)",
               "fg window was NOT grown — arg_halo override is not being consulted")
    else:
        r.ok("JOIN-49: fg's per-argument override was honoured, not the stage's uniform halo")


def test_join49_declined_poisoning_generalised_to_every_input(r: SubTestResult):
    """P0-4a, generalised: a join stage that DECLINED (cooked whole-frame from its inputs)
    poisons validity if ANY of its several inputs was itself only a patched region — not
    only its first input, which is all the linear model could name."""
    print("\n--- JOIN-49: declined-stage poisoning checks EVERY input, not just the first ---")
    W = H = 64
    stages = [StageSpec(0, ()), StageSpec(0, ()), StageSpec(0, (0, 1))]
    roi = (10, 10, 4, 4, W, H)
    # Convention (matches `chain_windows`/`covers`): `None` means "whole-frame valid"; a
    # tuple means "valid only over THIS region" (a patched, partial canvas).
    patched = (0, 0, 8, 8)
    # Stage 2 declined; its SECOND input (stage 1) is only patched over a small region while
    # its first (stage 0) is whole-frame valid. Poisoning must still fire.
    valid_second_patched = [None, patched, None]
    out = chain_windows_dag(stages, roi, dirty_from=0, valid=valid_second_patched,
                            declined=[2])
    if out is not None:
        r.fail("JOIN-49 declined poisoning (second input)",
               f"expected None (poisoned via input 1), got {out!r}")
    else:
        r.ok("JOIN-49: a declined join is poisoned by its SECOND input's partial validity")

    # Sanity: with BOTH inputs whole-frame valid (`None`), no poisoning.
    valid_both_whole = [None, None, None]
    out2 = chain_windows_dag(stages, roi, dirty_from=0, valid=valid_both_whole, declined=[2])
    if out2 is None:
        r.fail("JOIN-49 declined poisoning (sanity)",
               "both inputs whole-frame valid, but the plan still refused")
    else:
        r.ok("JOIN-49: no poisoning when every input of a declined stage is whole-frame valid")


def test_join49_divergent_validity_refusal(r: SubTestResult):
    """SS B.3's new refusal shape: a join stage reads TWO clean (not-being-recomputed)
    upstream canvases whose valid regions do not both cover what the join needs from each
    — even though NEITHER input alone triggers the linear model's rules (neither is
    declined, neither is individually starved by a single-input check). This is a genuinely
    NEW refusal case a linear chain (one upstream per stage) never had to ask."""
    print("\n--- JOIN-49: divergent-validity join refusal (new SS B.3 case) ---")
    W = H = 64
    # stage 0, 1: clean upstreams (dirty_from=2). stage 2: the join, dirty, reads both.
    stages = [StageSpec(0, ()), StageSpec(0, ()), StageSpec(0, (0, 1))]
    roi = (40, 40, 8, 8, W, H)
    # stage 0 is valid over a window covering the roi; stage 1 is valid only over a DISJOINT
    # corner far from the roi — the join's demand on stage 1 is NOT covered.
    valid = [(30, 30, 30, 30), (0, 0, 8, 8), None]
    out = chain_windows_dag(stages, roi, dirty_from=2, valid=valid)
    if out is not None:
        r.fail("JOIN-49 divergent validity", f"expected None (input 1 uncovered), got {out!r}")
        return
    r.ok("JOIN-49: refused when one of two clean join inputs does not cover the join's demand")

    # Negative control (must NOT refuse): both clean inputs cover the demand.
    valid_ok = [(30, 30, 30, 30), (30, 30, 30, 30), None]
    out2 = chain_windows_dag(stages, roi, dirty_from=2, valid=valid_ok)
    if out2 is None:
        r.fail("JOIN-49 divergent validity (sanity)",
               "both clean inputs cover the demand, but the plan still refused")
    else:
        r.ok("JOIN-49: NOT refused when both clean join inputs cover the join's demand")


# ── Part 3: pixel identity — every window this module allows is proven, not asserted ─────────

def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# BG is a plain pass-through (halo 0 into the join). FG is fed through a real gauss_blur
# stage before the join reads it — the join's OWN reach into FG (via that upstream blur) is
# a real, non-zero pixel radius, not a hand-picked number.
_BG_CODE = "@OUT = vec4(@IN.rgb * 0.6, 1.0);"
_FG_BLUR_CODE = "@OUT = gauss_blur(@IN, 3.0);"          # halo = ceil(3*3) = 9
_JOIN_CODE = "@OUT = vec4(@BG.rgb * (1.0 - $w) + @FG.rgb * $w, 1.0);"
_JOIN_HALO_FG = 9
_JOIN_PARAMS = {"w": 0.5}


def _cook_whole(src, device):
    bg = tex_engine.cook(_BG_CODE, {"IN": src}, device_mode=device,
                         precision="fp32").outputs["OUT"]
    fg = tex_engine.cook(_FG_BLUR_CODE, {"IN": src}, device_mode=device,
                         precision="fp32").outputs["OUT"]
    out = tex_engine.cook(_JOIN_CODE, {"BG": bg, "FG": fg, **_JOIN_PARAMS}, device_mode=device,
                          precision="fp32").outputs["OUT"]
    return bg, fg, out


def _cook_windowed(src, device, roi):
    """Cook the 3-stage join DAG using exactly the windows `chain_windows_dag` plans, and
    reassemble a full-size frame by pasting each stage's windowed output where it belongs
    (mirroring `test_v032_region.py`'s own `_host_edit` pattern)."""
    stages = [StageSpec(0, ()), StageSpec(_JOIN_HALO_FG, ()),
             StageSpec(0, (0, 1), arg_halo={0: 0, 1: _JOIN_HALO_FG})]
    plan = chain_windows_dag(stages, roi, dirty_from=0)
    assert plan is not None, "test setup: this shape must be serviceable"
    w_bg, w_fg, w_out = plan
    W, H = roi[4], roi[5]

    def _cook_win(code, params, win):
        res = tex_engine.cook(code, params, device_mode=device, precision="fp32",
                              roi=win, roi_exec=True)
        # `cooked_roi=None` means the engine DECLINED the window and cooked whole-frame
        # instead — the served extent is then the whole frame, not `win`.
        served = res.cooked_roi if res.cooked_roi is not None else (0, 0, W, H, W, H)
        return res.outputs["OUT"], served

    def _local_crop(tensor, served_win, target_win):
        """`target_win` (absolute frame coords) out of `tensor`, which itself only covers
        `served_win` (absolute frame coords) — i.e. LOCAL offsets relative to `served_win`'s
        own origin, not `tensor`'s own (zero-based, window-relative) indices."""
        sx0, sy0 = served_win[0], served_win[1]
        tx0, ty0, tw, th = target_win[:4]
        lx0, ly0 = tx0 - sx0, ty0 - sy0
        return tensor[:, ly0:ly0 + th, lx0:lx0 + tw]

    bg_out, bg_served = _cook_win(_BG_CODE, {"IN": src}, w_bg)
    fg_out, fg_served = _cook_win(_FG_BLUR_CODE, {"IN": src}, w_fg)
    # The join reads bg/fg over stage 2's own window — crop each of bg_out/fg_out (which
    # cover bg_served/fg_served, not necessarily the frame) down to that window.
    bg_for_join = _local_crop(bg_out, bg_served, w_out)
    fg_for_join = _local_crop(fg_out, fg_served, w_out)
    # `bg_for_join`/`fg_for_join` are ALREADY exactly `w_out`-sized (the join's own window) —
    # there is nothing left to narrow, so the join stage is cooked plainly (no `roi=`), the
    # same way a host that already has only the window's worth of upstream pixels would.
    out_win = tex_engine.cook(_JOIN_CODE, {"BG": bg_for_join, "FG": fg_for_join,
                                           **_JOIN_PARAMS}, device_mode=device,
                              precision="fp32").outputs["OUT"]
    return out_win, w_out


def _run_pixel_identity(r: SubTestResult, device: str):
    torch.manual_seed(490)
    W = H = 96
    src = torch.rand(1, H, W, 3)
    _, _, whole_out = _cook_whole(src, device)
    roi = (30, 30, 12, 12, W, H)
    win_out, served = _cook_windowed(src, device, roi)
    x0, y0, w, h = served[:4] if served is not None else roi[:4]
    ref_crop = whole_out[:, y0:y0 + h, x0:x0 + w]
    if torch.equal(win_out, ref_crop):
        r.ok(f"JOIN-49 pixel identity [{device}]: windowed 3-stage join DAG cook == "
             f"whole-frame crop, torch.equal True")
    else:
        maxdiff = (win_out - ref_crop).abs().max().item()
        r.fail(f"JOIN-49 pixel identity [{device}]",
               f"windowed join diverges from whole-frame, maxdiff={maxdiff}")


def test_join49_pixel_identity_join_dag(r: SubTestResult):
    """The proof the brief requires for every window this module allows: a real 3-stage
    join DAG (bg pass-through + fg gauss_blur + a per-pixel join reading both), windowed
    per `chain_windows_dag`'s own plan, produces PIXELS equal to the whole-frame cook — not
    merely a plausible-looking window. Runs on CPU always; CUDA is exercised by the
    lease-gated test below."""
    print("\n--- JOIN-49: pixel-identity proof, 3-stage join DAG, CPU ---")
    _run_pixel_identity(r, "cpu")


def test_join49_pixel_identity_join_dag_cuda(r: SubTestResult):
    """Same proof, CUDA. Gates on a CUDA device ONLY — this file's own CPU witness
    (`test_join49_pixel_identity_join_dag`, same construction, above) already proves the
    same pixels, so this row needs nothing beyond the device the other CUDA-only rows in
    this tree need (FIX-ROI49 Q2, B4#1/R2#1): a non-run reports through `r.skip`, so
    SIMP-3's skip census counts it, instead of `r.ok` in skip-shaped words that dodge the
    census vocabulary and read as a pass that measured nothing."""
    print("\n--- JOIN-49: pixel-identity proof, 3-stage join DAG, CUDA ---")
    if not torch.cuda.is_available():
        r.skip("JOIN-49 CUDA pixel identity", "no CUDA on this box")
        return
    _run_pixel_identity(r, "cuda")


# ── Part 4: negative controls that MUST refuse — proven by pixel divergence, not just None ───

def test_join49_negative_control_divergent_validity_is_pixel_wrong_if_served(r: SubTestResult):
    """Proves the SS B.3 refusal is not merely conservative bureaucracy: if the divergent-
    validity case were served ANYWAY (the bug this refusal exists to prevent), the result
    would be a genuinely WRONG picture, not just a smaller-than-ideal window. Builds the
    scenario by hand: stage 1 (fg) is only valid over a small corner; the join demands a
    window near the frame's OTHER corner, so serving it narrow would read fg pixels
    OUTSIDE where fg was ever recomputed — i.e. STALE pixels, not merely under-windowed
    ones."""
    print("\n--- JOIN-49 negative control: serving the divergent-validity case IS wrong ---")
    device = "cpu"
    torch.manual_seed(491)
    W = H = 96
    src = torch.rand(1, H, W, 3)
    _, fg_whole, whole_out = _cook_whole(src, device)

    # Simulate fg's canvas being valid ONLY over a small corner near (0,0): everywhere else
    # holds STALE (pre-edit) pixels rather than the current fg — construct that explicitly.
    stale_fg = torch.zeros_like(fg_whole)
    stale_fg[:, 0:20, 0:20] = fg_whole[:, 0:20, 0:20]           # the only "still fresh" part
    fg_valid_region = (0, 0, 20, 20)

    bg_whole = tex_engine.cook(_BG_CODE, {"IN": src}, device_mode=device,
                              precision="fp32").outputs["OUT"]
    roi_far = (60, 60, 12, 12, W, H)                             # far from fg's valid corner

    # 1) chain_windows_dag must refuse this shape.
    stages = [StageSpec(0, ()), StageSpec(_JOIN_HALO_FG, ()),
             StageSpec(0, (0, 1), arg_halo={0: 0, 1: _JOIN_HALO_FG})]
    valid = [(0, 0, W, H), fg_valid_region, None]
    plan = chain_windows_dag(stages, roi_far, dirty_from=2, valid=valid)
    if plan is not None:
        r.fail("JOIN-49 negative control setup", f"expected refusal, got plan {plan!r}")
        return
    r.ok("JOIN-49: divergent-validity join correctly refused (plan is None)")

    # 2) Prove serving it anyway (bypassing the refusal, as the pre-fix code would have)
    # produces WRONG pixels against the true whole-frame join, not merely a narrower window.
    x0, y0, w, h = roi_far[:4]
    fg_crop_from_stale = stale_fg[:, y0:y0 + h, x0:x0 + w]
    bg_crop = bg_whole[:, y0:y0 + h, x0:x0 + w]
    served_anyway = tex_engine.cook(_JOIN_CODE, {"BG": bg_crop, "FG": fg_crop_from_stale,
                                                 **_JOIN_PARAMS}, device_mode=device,
                                    precision="fp32").outputs["OUT"]
    true_crop = whole_out[:, y0:y0 + h, x0:x0 + w]
    if torch.equal(served_anyway, true_crop):
        r.fail("JOIN-49 negative control",
               "serving the divergent-validity case produced CORRECT pixels — the negative "
               "control is vacuous, it proves nothing about why the refusal matters")
    else:
        maxdiff = (served_anyway - true_crop).abs().max().item()
        r.ok(f"JOIN-49: serving the refused shape anyway IS pixel-wrong (maxdiff={maxdiff}) — "
             "the refusal is protecting a real silent-wrong case, not a hypothetical one")


def test_join49_negative_control_asymmetric_halo_if_uniform_applied(r: SubTestResult):
    """Proves rule 2 (per-argument projection) is load-bearing: if the walker had instead
    applied the join's UNIFORM halo (0, since it has no single scalar reach) or FG's halo to
    BOTH inputs, the served window would either starve fg's true blur-radius reach (wrong
    pixels) or over-pad bg needlessly. Constructs the "starve fg" case explicitly and shows
    the resulting join output diverges from the whole-frame answer."""
    print("\n--- JOIN-49 negative control: uniform halo would starve fg's true reach ---")
    device = "cpu"
    torch.manual_seed(492)
    W = H = 96
    src = torch.rand(1, H, W, 3)
    bg_whole, fg_whole, whole_out = _cook_whole(src, device)
    roi = (40, 40, 6, 6, W, H)
    x0, y0, w, h = roi[:4]

    # Correct: fg cropped with its true 9px halo (what chain_windows_dag actually plans),
    # THEN the join-window slice taken out of that — matches production: the blur only
    # ever sees the padded window's worth of source, never the whole frame.
    fx0, fy0 = max(0, x0 - _JOIN_HALO_FG), max(0, y0 - _JOIN_HALO_FG)
    fx1, fy1 = min(W, x0 + w + _JOIN_HALO_FG), min(H, y0 + h + _JOIN_HALO_FG)
    src_padded = src[:, fy0:fy1, fx0:fx1]
    fg_blurred_from_correct = tex_engine.cook(
        _FG_BLUR_CODE, {"IN": src_padded}, device_mode=device,
        precision="fp32").outputs["OUT"]
    fg_correct_crop = fg_blurred_from_correct[:, y0 - fy0:y0 - fy0 + h, x0 - fx0:x0 - fx0 + w]

    # Wrong (the bug a uniform-halo walker would produce): the blur is fed ONLY the
    # unpadded join window's worth of source — as if fg's own reach into its input were 0,
    # the join's uniform halo, instead of its true 9px blur radius. `gauss_blur` then has no
    # true neighbour pixels near the crop's edges to read (this is what "insufficient
    # window" actually means for a host whose canvases are only ever the size they were
    # cooked at — the single-call `roi=` growth `_cook_win` uses elsewhere in this file is
    # not in play here on purpose, since that mechanism always self-heals a too-small
    # request as long as the FULL tensor is still reachable, which is not the failure mode
    # this row is proving).
    src_unpadded = src[:, y0:y0 + h, x0:x0 + w]
    fg_blurred_from_zero_halo = tex_engine.cook(
        _FG_BLUR_CODE, {"IN": src_unpadded}, device_mode=device,
        precision="fp32").outputs["OUT"]

    bg_crop = bg_whole[:, y0:y0 + h, x0:x0 + w]
    correct_out = tex_engine.cook(
        _JOIN_CODE, {"BG": bg_crop, "FG": fg_correct_crop, **_JOIN_PARAMS},
        device_mode=device, precision="fp32").outputs["OUT"]
    wrong_out = tex_engine.cook(
        _JOIN_CODE, {"BG": bg_crop, "FG": fg_blurred_from_zero_halo, **_JOIN_PARAMS},
        device_mode=device, precision="fp32").outputs["OUT"]
    true_crop = whole_out[:, y0:y0 + h, x0:x0 + w]

    if not torch.equal(correct_out, true_crop):
        r.fail("JOIN-49 negative control setup",
               "the CORRECT (per-argument-halo) window did not match the whole-frame answer "
               "— test setup is broken, not the feature")
        return
    r.ok("JOIN-49: the per-argument halo this module plans reproduces the whole-frame answer")

    if torch.equal(wrong_out, true_crop):
        r.fail("JOIN-49 negative control",
               "a uniform (zero) halo on fg ALSO matched the whole-frame answer — the "
               "negative control is vacuous on this box/seed")
    else:
        maxdiff = (wrong_out - true_crop).abs().max().item()
        r.ok(f"JOIN-49: a uniform-halo (starved fg) window IS pixel-wrong (maxdiff={maxdiff}) "
             "— per-argument projection is load-bearing, not cosmetic")
