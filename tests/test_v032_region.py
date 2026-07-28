"""v0.32 CACHE-9 — region-granular recook.

Design note: docs/region-granular-recook.md. What has to be pinned, and why each is a bug the
note names as already-measured, not hypothetical:

  * THE ROI-4 DIFFERENTIAL — a patch-assembled frame equals the whole-frame cook. This is the
    ship gate; every ROI feature ships behind this lane, and CACHE-9 is the one the risk
    register calls the correctness-sensitive item of the set (it composes footprint
    over-approximation, ENG-12 immutability, and lineage keys at once).
  * THE HALO INVERSION — `RoiPlan.halo` is 0 for a NON-EXECUTABLE plan, which means "unbounded,
    cook whole", not "reads no neighbours". A consumer trusting that 0 under-grows every
    upstream window and leaves a stale ring. Measured in the demo host before it composed
    backwards: stage-5 sharpen wrong over 2157 px, stage-9 vignette over 3987 px.
  * SATURATION — one unbounded stage clamps its own window AND every window above it.
  * COPY-ON-PATCH — the cached base is byte-identical after a patch. ENG-12's freeze is a
    tripwire, not a fence: an in-place op on an inference tensor LANDS the write and then
    raises, so a "protected" master would be silently corrupted and re-served.
  * DECLINED WINDOWS — a stage can hand back a whole frame (`cooked_roi is None`); pasting that
    into a w×h slice is a crash, which is what the demo did before it checked.
  * PROVENANCE — a patched frame keys apart from its base and carries it in `upstream`, because
    the version STAMP cannot express it (`frame_version` is a constant 0 for frozen entries,
    and `put` always freezes).

Shapes: DIFFERENTIAL ORACLE, NEVER-SEVER ROWS (the inversion + saturation + declines), CANARY
(ownership), REGRESSION (the stale ring).
"""
import threading

from helpers import *

from TEX_Wrangle import tex_engine, tex_results, tex_roi

# A chain with a real halo op in the middle, so a patch that ignores composition leaves a
# visibly stale ring rather than a rounding difference.
_CHAIN = [
    "@OUT = vec4(@IN.rgb * $knob, 1.0);",
    "@OUT = vec4(max(@IN.rgb - vec3(0.02), vec3(0.0)), 1.0);",
    "@OUT = gauss_blur(@IN, 3.0);",
    "@OUT = vec4((@IN.rgb - vec3(0.5)) * 1.08 + vec3(0.5), 1.0);",
]
_PARAMS = [{"knob": 0.8}, {}, {}, {}]


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _cook_stage(code, params, src, roi=None, device="cpu"):
    """One unfused engine cook, optionally over a window. Returns (tensor, cooked_roi)."""
    res = tex_engine.cook(code, {"IN": src, **params}, device_mode=device, precision="fp32",
                          roi=roi, roi_exec=(roi is not None))
    return res.outputs["OUT"], res.cooked_roi


def _cook_chain_whole(src, device, knob):
    """The reference: every stage cooked whole-frame, in order."""
    cur = src
    for code, params in zip(_CHAIN, _PARAMS):
        p = dict(params)
        if "knob" in p:
            p["knob"] = knob
        cur, _ = _cook_stage(code, p, cur, None, device)
    return cur


# ── the ship gate ───────────────────────────────────────────────────────────────────────

def test_v032_cache9_region_recook_oracle(r: SubTestResult):
    """ROI-4 DIFFERENTIAL. Patch-assembled == whole-frame cook, over a chain with a halo op.

    `< 1e-5` is the FUS-3 convention: pointwise/morphology land bit-exact, conv within ~1 ulp
    of size-dependent kernel dispatch. A tolerance is right HERE (unlike CACHE-7's taps, which
    recompute nothing) because an ROI cook genuinely re-dispatches convolutions at a different
    extent."""
    print("\n--- v0.32 CACHE-9: region recook oracle (patched == whole) ---")
    for device in _devices():
        torch.manual_seed(3)
        src = torch.rand(1, 96, 96, 3, device=device)
        halos = [tex_roi.stage_halo(c, p) for c, p in zip(_CHAIN, _PARAMS)]

        # Cook the whole chain once at the OLD param, keeping every stage's frame.
        canvases, cur = [], src
        for code, params in zip(_CHAIN, _PARAMS):
            cur, _ = _cook_stage(code, params, cur, None, device)
            canvases.append(cur)

        # The edit: stage 0's knob moves. Recook only the window through the whole chain.
        roi = (24, 24, 32, 32, 96, 96)
        windows = tex_roi.chain_windows(halos, roi, dirty_from=0)
        cache = tex_results.ResultCache()
        cur = src
        patched = list(canvases)
        ok = True
        for i, (code, params) in enumerate(zip(_CHAIN, _PARAMS)):
            p = dict(params)
            if "knob" in p:
                p["knob"] = 0.45
            out, served = _cook_stage(code, p, cur, windows[i], device)
            if served is None:
                patched[i] = out                       # declined: a whole frame REPLACES
            else:
                key = f"stage{i}@{served}"
                got = cache.patch_region(key, out, served, base=patched[i])
                if got is None:
                    r.fail(f"CACHE-9 patch {device} stage {i}", "patch_region refused")
                    ok = False
                    break
                patched[i] = got
            cur = patched[i]

        if not ok:
            continue
        ref = _cook_chain_whole(src, device, 0.45)
        # Compared INSIDE the requested window only. That is the contract: a region recook
        # makes the REGION correct. Outside it the patched frame still legitimately holds
        # pre-edit pixels — the edit here is a pointwise multiply, so a whole-frame comparison
        # would be asserting that a region cook did a whole-frame cook's job.
        x0, y0, w, h = roi[:4]
        d = (patched[-1][:, y0:y0 + h, x0:x0 + w].float()
             - ref[:, y0:y0 + h, x0:x0 + w].float()).abs().max().item()
        if d < 1e-5:
            r.ok(f"CACHE-9 oracle {device}: the patched WINDOW == the whole cook "
                 f"(maxdiff {d:.2e})")
        else:
            r.fail(f"CACHE-9 oracle {device}", f"maxdiff {d:.3e} inside the window (>= 1e-5)")


def test_v032_cache9_stale_ring_regression(r: SubTestResult):
    """REGRESSION. Patching only the REQUESTED rect — ignoring the composition — must be
    provably worse. If this row ever stops finding a difference, `chain_windows` has stopped
    doing anything and the oracle above would pass vacuously."""
    print("\n--- v0.32 CACHE-9: the stale ring the composition prevents ---")
    torch.manual_seed(4)
    device = "cpu"
    src = torch.rand(1, 96, 96, 3, device=device)
    halos = [tex_roi.stage_halo(c, p) for c, p in zip(_CHAIN, _PARAMS)]
    roi = (24, 24, 32, 32, 96, 96)

    composed = tex_roi.chain_windows(halos, roi, dirty_from=0)
    naive = [roi] * len(_CHAIN)                        # the wrong thing: same rect everywhere

    def _assemble(windows):
        canvases, cur = [], src
        for code, params in zip(_CHAIN, _PARAMS):
            cur, _ = _cook_stage(code, params, cur, None, device)
            canvases.append(cur)
        cur, out_frames = src, list(canvases)
        for i, (code, params) in enumerate(zip(_CHAIN, _PARAMS)):
            p = dict(params)
            if "knob" in p:
                p["knob"] = 0.45
            out, served = _cook_stage(code, p, cur, windows[i], device)
            if served is None:
                out_frames[i] = out
            else:
                x0, y0, w, h = served[:4]
                buf = out_frames[i].clone()
                buf[:, y0:y0 + h, x0:x0 + w] = out
                out_frames[i] = buf
            cur = out_frames[i]
        return out_frames[-1]

    ref = _cook_chain_whole(src, device, 0.45)
    x0, y0, w, h = roi[:4]

    def _err(frame):
        # Inside the requested window only — see the oracle row.
        return (frame[:, y0:y0 + h, x0:x0 + w].float()
                - ref[:, y0:y0 + h, x0:x0 + w].float()).abs().max().item()

    d_naive = _err(_assemble(naive))
    d_comp = _err(_assemble(composed))
    if composed[0][2] > roi[2]:
        r.ok(f"CACHE-9: the composed upstream window is grown "
             f"({roi[2]}px -> {composed[0][2]}px by the blur's halo)")
    else:
        r.fail("CACHE-9 composition", "the upstream window was not grown at all")
    if d_naive > d_comp:
        r.ok(f"CACHE-9: composing beats the naive same-rect patch "
             f"({d_naive:.2e} -> {d_comp:.2e})")
    else:
        r.fail("CACHE-9 stale ring", f"naive={d_naive:.2e} composed={d_comp:.2e} — "
               "the composition is not doing anything")


# ── never-sever rows ────────────────────────────────────────────────────────────────────

def test_v032_cache9_unbounded_reach_inverts_to_whole_frame(r: SubTestResult):
    """THE INVERSION. A non-executable plan reports halo=0 meaning "unbounded"; `stage_halo`
    must answer WHOLE_FRAME, and one such stage must saturate every window above it."""
    print("\n--- v0.32 CACHE-9: unbounded reach inverts to a whole frame ---")
    gathers = [
        ("a sample() gather", "@OUT = sample(@IN, u * 0.5, v * 0.5);"),
        ("a fetch() gather", "@OUT = fetch(@IN, ix / 2, iy / 2);"),
        ("a reduction", "@OUT = vec4(vec3(img_mean(@IN).r), 1.0);"),
    ]
    for label, code in gathers:
        plan = tex_roi.roi_plan(code, {})
        h = tex_roi.stage_halo(code, {})
        if not plan.executable and h == tex_roi.WHOLE_FRAME:
            r.ok(f"CACHE-9 stage_halo: {label} -> WHOLE_FRAME (plan.halo was {plan.halo})")
        elif plan.executable:
            r.ok(f"CACHE-9 stage_halo: {label} is ROI-executable here (halo={h}) — no inversion "
                 "needed")
        else:
            r.fail("CACHE-9 halo inversion",
                   f"{label}: non-executable but stage_halo returned {h}")

    # Saturation: an unbounded stage 2 must make windows 0..1 the whole frame.
    halos = [1, 2, tex_roi.WHOLE_FRAME, 1]
    W = H = 128
    wins = tex_roi.chain_windows(halos, (40, 40, 16, 16, W, H), dirty_from=0)
    if wins[1] == (0, 0, W, H, W, H):
        r.ok("CACHE-9 chain_windows: an unbounded consumer saturates its producer to the frame")
    else:
        r.fail("CACHE-9 saturation", f"window below the unbounded stage was {wins[1]}")
    if wins[0] == (0, 0, W, H, W, H):
        r.ok("CACHE-9 chain_windows: saturation propagates all the way down")
    else:
        r.fail("CACHE-9 saturation", f"the bottom window was {wins[0]}, not the whole frame")

    # And the ordinary case still narrows, so the rows above are not passing vacuously.
    wins2 = tex_roi.chain_windows([1, 2, 3, 1], (40, 40, 16, 16, W, H), dirty_from=0)
    if all(w[2] < W for w in wins2):
        r.ok(f"CACHE-9 chain_windows: bounded reaches stay narrow (widths "
             f"{[w[2] for w in wins2]})")
    else:
        r.fail("CACHE-9 composition", f"a bounded chain saturated anyway: {wins2}")

    # Windows grow monotonically DOWNstream-to-upstream, never shrink.
    widths = [w[2] for w in wins2]
    if widths == sorted(widths, reverse=True):
        r.ok("CACHE-9 chain_windows: windows grow monotonically towards the source")
    else:
        r.fail("CACHE-9 composition", f"window widths are not monotonic: {widths}")


def test_v032_cache9_dirty_from_leaves_clean_stages_alone(r: SubTestResult):
    """Stages below `dirty_from` are not cooking, so they get no window at all — the
    interactive case (the user drags the LAST node's slider) is the whole point."""
    print("\n--- v0.32 CACHE-9: a clean prefix gets no window ---")
    wins = tex_roi.chain_windows([1, 1, 1, 1], (10, 10, 8, 8, 64, 64), dirty_from=2)
    if wins[0] is None and wins[1] is None and wins[2] is not None and wins[3] is not None:
        r.ok("CACHE-9 chain_windows: dirty_from=2 -> stages 0,1 have no window")
    else:
        r.fail("CACHE-9 dirty_from", f"got {wins}")


# ── ownership + provenance ──────────────────────────────────────────────────────────────

def test_v032_cache9_patch_never_touches_the_cached_master(r: SubTestResult):
    """CANARY. ENG-12's freeze is a tripwire, not a fence — an in-place op on an inference
    tensor lands the write and THEN raises. So the base frame must be byte-identical after a
    patch, and the only thing that guarantees that is copy-on-patch."""
    print("\n--- v0.32 CACHE-9: a patch never reaches the cached master ---")
    cache = tex_results.ResultCache()
    base = torch.rand(1, 32, 32, 4)
    snapshot = base.clone()
    cache.put("base", base)

    patch = torch.zeros(1, 8, 8, 4)
    out = cache.patch_region("patched", patch, (4, 4, 8, 8, 32, 32), base_key="base")
    if out is None:
        r.fail("CACHE-9 patch_region", "refused a well-formed patch")
        return
    served_base = cache.get("base")
    if torch.equal(served_base, snapshot):
        r.ok("CACHE-9 patch_region: the base frame is byte-identical after the patch")
    else:
        r.fail("CACHE-9 ownership", "the patch reached the cached base frame")
    if torch.equal(out[:, 4:12, 4:12], patch):
        r.ok("CACHE-9 patch_region: the window carries the patch")
    else:
        r.fail("CACHE-9 patch_region", "the patched window does not hold the patch")
    if torch.equal(out[:, 0:4, 0:4], snapshot[:, 0:4, 0:4]):
        r.ok("CACHE-9 patch_region: pixels outside the window are the base's")
    else:
        r.fail("CACHE-9 patch_region", "pixels outside the window changed")

    # A frozen explicit base must be cloned too, not written through.
    frozen = tex_engine.freeze(torch.rand(1, 32, 32, 4))
    fsnap = frozen.clone()
    got = cache.patch_region("p2", torch.ones(1, 8, 8, 4), (0, 0, 8, 8, 32, 32), base=frozen)
    if got is not None and torch.equal(frozen, fsnap):
        r.ok("CACHE-9 patch_region: an explicit FROZEN base is cloned, not written through")
    else:
        r.fail("CACHE-9 ownership", "a frozen explicit base was mutated")


def test_v032_cache9_patch_refuses_a_mismatched_window(r: SubTestResult):
    """A stage can DECLINE a window and hand back a whole frame. Pasting that into a w×h slice
    is a crash — which is what the demo host did before it checked `cooked_roi`. Refusing beats
    writing wrong pixels into a frame that then looks authoritative."""
    print("\n--- v0.32 CACHE-9: a mismatched window is refused ---")
    cache = tex_results.ResultCache()
    cache.put("base", torch.rand(1, 32, 32, 4))
    rows = [
        ("a whole frame offered as a window patch", torch.rand(1, 32, 32, 4), (4, 4, 8, 8, 32, 32)),
        ("a patch of the wrong extent", torch.rand(1, 4, 4, 4), (4, 4, 8, 8, 32, 32)),
        ("a window describing a different frame size", torch.rand(1, 8, 8, 4), (0, 0, 8, 8, 64, 64)),
    ]
    for label, patch, window in rows:
        got = cache.patch_region("bad", patch, window, base_key="base")
        r.ok(f"CACHE-9 patch_region refuses: {label}") if got is None else \
            r.fail("CACHE-9 window validation", f"accepted {label}")
    if cache.get("bad") is None:
        r.ok("CACHE-9 patch_region: a refused patch stored nothing")
    else:
        r.fail("CACHE-9 window validation", "a refused patch still wrote a cache entry")

    # No base at all -> None, so the caller cooks whole (always correct).
    empty = tex_results.ResultCache()
    if empty.patch_region("x", torch.rand(1, 8, 8, 4), (0, 0, 8, 8, 32, 32),
                          base_key="missing") is None:
        r.ok("CACHE-9 patch_region: no base -> None (the caller cooks whole)")
    else:
        r.fail("CACHE-9 patch_region", "patched with no base frame")


def test_v032_cache9_provenance_is_in_the_key(r: SubTestResult):
    """The version STAMP cannot express "patched descendant of": `frame_version` is a constant
    0 for frozen tensors and `put` always freezes. So provenance rides CACHE-1's `upstream`."""
    print("\n--- v0.32 CACHE-9: provenance rides the key, not the stamp ---")
    frozen = tex_engine.freeze(torch.rand(1, 16, 16, 4))
    if tex_engine.frame_version(frozen) == 0:
        r.ok("CACHE-9: frame_version is 0 for a frozen frame (the stamp cannot carry lineage)")
    else:
        r.fail("CACHE-9 provenance", "frame_version is no longer constant for frozen frames — "
               "the design note's reasoning needs revisiting")

    base_key = tex_results.lineage_key(program_fp="fp-base", device="cpu", precision="fp32")
    patched_key = tex_results.lineage_key(
        program_fp="fp-base", device="cpu", precision="fp32", upstream=(base_key,),
        canvas={"shape": [1, 32, 32, 4], "roi": [4, 4, 8, 8]})
    if base_key != patched_key:
        r.ok("CACHE-9: a patched frame keys apart from its base")
    else:
        r.fail("CACHE-9 provenance", "the patched key collides with the base key")
    other = tex_results.lineage_key(
        program_fp="fp-base", device="cpu", precision="fp32", upstream=(base_key,),
        canvas={"shape": [1, 32, 32, 4], "roi": [0, 0, 8, 8]})
    if other != patched_key:
        r.ok("CACHE-9: two patches of the same base at different windows key apart")
    else:
        r.fail("CACHE-9 provenance", "the window does not discriminate the key")


# ── audit fixes ─────────────────────────────────────────────────────────────────────────

def test_v032_cache9_second_deeper_edit_needs_valid_regions(r: SubTestResult):
    """BLOCKER-adjacent. A region cook leaves the canvases it patched valid ONLY over their
    composed windows. A LATER, deeper edit derives windows from a different `roi` and reads an
    upstream canvas wherever that lands — including outside the region the earlier patch made
    correct, where it finds PRE-EDIT pixels. Measured through documented `dirty_from` usage:
    2.17e-01 on the second edit (see the pixel-level row below, which is the one that matters —
    this row only checks the arithmetic, and an earlier version of it passed while the pixels
    were still wrong).

    `chain_windows(..., valid=)` closes it by returning **None**: "not serviceable, cook from
    the source". Widening the returned WINDOWS does not work — the upstream canvas is wrong
    outside the earlier patch, not merely read too narrowly, and no downstream window repairs a
    stale input."""
    print("\n--- v0.32 CACHE-9: a second deeper edit respects canvas validity ---")
    W = H = 128
    halos = [0, 0, 4, 0]

    # Edit 1 at stage 0 over a small window in one corner: canvases end up valid only there.
    w1 = tex_roi.chain_windows(halos, (8, 8, 16, 16, W, H), dirty_from=0)
    valid = list(w1)

    # Edit 2 at stage 2, over a window in the OPPOSITE corner. Stage 2 reads canvas 1, which
    # edit 1 only made correct near (8,8).
    far = (96, 96, 16, 16, W, H)
    unguarded = tex_roi.chain_windows(halos, far, dirty_from=2)
    guarded = tex_roi.chain_windows(halos, far, dirty_from=2, valid=valid)

    if unguarded[2] is not None and unguarded[2][2] < W:
        r.ok("CACHE-9: without `valid`, the second edit narrows (the stale-ring shape)")
    else:
        r.fail("CACHE-9 validity setup", "the unguarded plan did not narrow — row is vacuous")

    if guarded is None:
        r.ok("CACHE-9: with `valid`, an uncovered read reports NOT SERVICEABLE (None)")
    else:
        r.fail("CACHE-9 validity",
               f"expected None (cook from the source), got {guarded} — widening the WINDOWS "
               "cannot fix a stale upstream canvas")

    # And the covered case must STILL narrow, or the guard is just "always cook whole".
    near = (10, 10, 8, 8, W, H)
    ok_plan = tex_roi.chain_windows(halos, near, dirty_from=2, valid=valid)
    if ok_plan[2] is not None and ok_plan[2][2] < W:
        r.ok(f"CACHE-9: a COVERED second edit still narrows (w={ok_plan[2][2]})")
    else:
        r.fail("CACHE-9 validity", "a covered edit was needlessly widened to the frame")

    # `covers` itself, since the guard is only as good as it.
    rows = [((0, 0, 10, 10, W, H), (2, 2, 4, 4, W, H), True),
            ((0, 0, 10, 10, W, H), (8, 8, 8, 8, W, H), False),
            (None, (0, 0, W, H, W, H), True)]
    for valid_r, need_r, want in rows:
        got = tex_roi.covers(valid_r, need_r)
        r.ok(f"CACHE-9 covers({'whole' if valid_r is None else 'rect'}, rect) == {want}") \
            if got == want else \
            r.fail("CACHE-9 covers", f"{valid_r} vs {need_r}: got {got}, want {want}")


def test_v032_cache9_patch_region_is_atomic(r: SubTestResult):
    """`patch_region` composes `get` + `put`. Per-call safe is not ATOMIC: two threads patching
    one base both read it, both write their own window, and the second `put` discards the
    first's. An earlier draft carried a comment claiming a lock it never took — 200/200 lost
    updates."""
    print("\n--- v0.32 CACHE-9: patch_region is atomic ---")
    cache = tex_results.ResultCache()
    N = 16
    base = torch.zeros(1, 8, N * 4, 4)
    cache.put("shared", base)
    errors = []

    # WIDEN THE CRITICAL SECTION. Without this the threads simply never interleave at this
    # scale and the test passes with the lock REMOVED — verified by mutation: the row below is
    # the discriminator, not the thread count. A delay inside the read is what turns
    # "read-modify-write" into a window big enough to lose an update through. Under the lock
    # the delay serializes; without it, every thread reads the same base and the last `put`
    # wins.
    real_get = cache.get

    def slow_get(key, *, copy=True):
        time.sleep(0.01)
        return real_get(key, copy=copy)

    cache.get = slow_get

    def worker(i):
        try:
            # Each thread owns a disjoint 4-px column band of ONE shared key.
            patch = torch.full((1, 8, 4, 4), float(i + 1))
            cache.patch_region("shared", patch, (i * 4, 0, 4, 8, N * 4, 8),
                               base_key="shared")
        except Exception as exc:                      # noqa: BLE001 — that IS the finding
            errors.append(f"{i}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60.0)
    cache.get = real_get

    if errors:
        r.fail("CACHE-9 patch atomicity", "; ".join(errors[:3]))
        return
    r.ok(f"CACHE-9: {N} concurrent patchers on one key raised nothing")
    final = cache.get("shared")
    if final is None:
        r.fail("CACHE-9 patch atomicity", "no frame survived the race")
        return
    written = sum(1 for i in range(N) if float(final[0, 0, i * 4, 0]) == float(i + 1))
    # EVERY band must survive. Without the lock this collapses to ~1 (each thread reads the
    # same base through the widened window and the last `put` discards the rest). `>= N//2`
    # would NOT discriminate — mutation-tested: the lock can be removed entirely and a
    # majority-threshold assertion still passes.
    if written == N:
        r.ok(f"CACHE-9: all {N} bands survived a deliberately interleaved read-modify-write")
    else:
        r.fail("CACHE-9 patch atomicity",
               f"only {written}/{N} bands survived — read-modify-write is not serialized")


def test_v032_cache9_second_deeper_edit_pixels(r: SubTestResult):
    """The PIXEL-level form of the row above. Window arithmetic changing is not the same claim
    as the stale ring being gone, and the audit measured the bug in pixels (7.79e-02), so this
    reproduces it end to end: edit once at stage 0 over one corner, then again at stage 2 over
    the opposite corner, and compare against the whole-frame reference inside the second
    window."""
    print("\n--- v0.32 CACHE-9: a second deeper edit, measured in pixels ---")
    device = "cpu"
    torch.manual_seed(7)
    N = 96
    src = torch.rand(1, N, N, 3, device=device)
    halos = [tex_roi.stage_halo(c, p) for c, p in zip(_CHAIN, _PARAMS)]

    def _prime(knob):
        cur, canv = src, []
        for code, params in zip(_CHAIN, _PARAMS):
            p = dict(params)
            if "knob" in p:
                p["knob"] = knob
            cur, _ = _cook_stage(code, p, cur, None, device)
            canv.append(cur)
        return canv

    def _edit(canv, valid, roi, dirty_from, knob, use_valid):
        plan = tex_roi.chain_windows(halos, roi, dirty_from,
                                     valid=(valid if use_valid else None))
        if plan is None:
            # Not serviceable: the upstream canvas is stale where this edit needs to read it.
            # The only correct remedy is to re-cook from the source, whole-frame.
            dirty_from, plan = 0, [None] * len(_CHAIN)
        canv, valid = list(canv), list(valid)
        cur = src if dirty_from == 0 else canv[dirty_from - 1]
        for i in range(dirty_from, len(_CHAIN)):
            p = dict(_PARAMS[i])
            if "knob" in p:
                p["knob"] = knob
            out, served = _cook_stage(_CHAIN[i], p, cur, plan[i], device)
            if served is None:
                canv[i], valid[i] = out, None
            else:
                x0, y0, w, h = served[:4]
                buf = canv[i].clone()
                buf[:, y0:y0 + h, x0:x0 + w] = out
                canv[i], valid[i] = buf, served
            cur = canv[i]
        return canv, valid

    corner_a = (4, 4, 24, 24, N, N)
    corner_b = (N - 28, N - 28, 24, 24, N, N)

    for use_valid in (False, True):
        canv = _prime(0.80)
        valid = [None] * len(_CHAIN)
        canv, valid = _edit(canv, valid, corner_a, 0, 0.45, use_valid)     # edit 1
        canv, valid = _edit(canv, valid, corner_b, 2, 0.45, use_valid)     # edit 2, deeper
        ref = _cook_chain_whole(src, device, 0.45)
        x0, y0, w, h = corner_b[:4]
        err = (canv[-1][:, y0:y0 + h, x0:x0 + w].float()
               - ref[:, y0:y0 + h, x0:x0 + w].float()).abs().max().item()
        label = "with `valid`" if use_valid else "without `valid`"
        if use_valid:
            if err < 1e-5:
                r.ok(f"CACHE-9 {label}: the second window is correct (maxdiff {err:.2e})")
            else:
                r.fail("CACHE-9 second edit", f"{label}: maxdiff {err:.3e} — ring still stale")
        else:
            if err >= 1e-5:
                r.ok(f"CACHE-9 {label}: the stale ring IS reproduced ({err:.2e}) — "
                     "so the guarded row is not vacuous")
            else:
                r.fail("CACHE-9 second edit",
                       f"{label}: expected a stale ring, measured {err:.2e} — "
                       "the negative control found nothing, so this test proves nothing")
