"""CACHE-10 — is a region recook worth it?

Design note: docs/region-granular-recook.md §4, and the reopen gate recorded in
DEVELOPMENT.md's rejected-decisions register (Doc 41 §2.4(a)): CACHE-9's own note fenced the
all-dirty cliff — region recook measured **0.21×/0.04×** against whole-frame at 2048², all 50
stages dirty — with a paragraph telling a host to route that case around the mechanism itself
instead of a `not-worth-it` return from the serviceability API.

`tex_roi.region_advisory` is the wiring the deferral asked for: given exactly the inputs a
region-recook planner already has (`chain_windows`'s own parameters) plus PROF-1-shaped
per-stage costs, it prices both sides and hands back a `RegionAdvisory` — mirroring
`tex_checkpoint.GateRefusal`'s shape and stability promise — only when the region path is
expected to lose. It never touches what `chain_windows` returns, and a host that never calls
it sees no behaviour change.

Shapes: NEVER-SEVER ROWS (every input that would make the price a guess must yield NO
advisory, mirroring CACHE-7's "placement refuses rather than guesses"), the POSITIVE CONTROL
(the all-dirty shape the deferral names, priced and pinned exactly — not just in direction),
the NEGATIVE CONTROL (a narrow mid-graph edit the mechanism is supposed to win, pinned as a
non-advisory), REFUSAL PASSTHROUGH (`chain_windows`'s own correctness refusal is never
second-guessed), a structure/stability check, and a CANARY for invariant #7.
"""
from helpers import *

from TEX_Wrangle import tex_checkpoint as CK
from TEX_Wrangle import tex_roi as ROI


def test_cache10_all_dirty_edit_is_priced_slower(r: SubTestResult):
    """POSITIVE CONTROL. An edit whose requested window IS the whole frame is the textbook
    all-dirty shape the deferral names: every stage's `chain_windows` window is necessarily the
    full frame too (there is nowhere left to grow from), so the region path pays a full-frame
    clone per stage on top of the SAME per-stage compute the whole-frame path pays — never
    less. Pinned exactly, not just in direction, against `tex_checkpoint.put_cost_ms` itself so
    the two modules cannot drift apart silently."""
    print("\n--- CACHE-10: an all-dirty edit is priced slower, exactly ---")
    halos = [0, 2, 3]                       # any reach; irrelevant once the window is full-frame
    roi = (0, 0, 64, 64, 64, 64)             # the WHOLE frame requested -> dirty_from=0 is all-dirty
    costs = {0: 12.0, 1: 30.0, 2: 8.0}
    px = 64 * 64
    for device in ("cpu", "cuda"):
        adv = ROI.region_advisory(halos, roi, 0, costs=costs, px=px, device=device,
                                  settled=True)
        want_whole = sum(costs.values())
        want_region = want_whole + 3 * CK.put_cost_ms(px, device)
        if adv is None:
            r.fail(f"CACHE-10 all-dirty ({device})", "no advisory; expected the region path "
                   "flagged as slower")
            continue
        ok = (adv.code == ROI.ADVISE_REGION_SLOWER
              and abs(adv.whole_ms - want_whole) < 1e-9
              and abs(adv.region_ms - want_region) < 1e-9)
        if ok:
            r.ok(f"CACHE-10 all-dirty ({device}): region {adv.region_ms:.3f} ms > "
                 f"whole {adv.whole_ms:.3f} ms, by exactly 3 clones")
        else:
            r.fail(f"CACHE-10 all-dirty ({device})",
                   f"got {adv!r}, wanted whole={want_whole!r} region={want_region!r}")


def test_cache10_mid_graph_edit_is_not_flagged(r: SubTestResult):
    """NEGATIVE CONTROL. A small, deep edit is the shape CACHE-9 exists for: the window stays
    tiny relative to the frame, so even after the per-stage clone tax the region path is
    cheaper — no advisory, and a caller that always asks learns nothing it has to act on."""
    print("\n--- CACHE-10: a narrow mid-graph edit is not flagged ---")
    halos = [0, 0, 0, 0]
    roi = (490, 490, 20, 20, 1000, 1000)     # a 20x20 window out of 1000x1000, no halo growth
    costs = {2: 500.0, 3: 500.0}             # only the dirty suffix carries a measured cost
    px = 1000 * 1000
    adv = ROI.region_advisory(halos, roi, 2, costs=costs, px=px, device="cpu", settled=True)
    if adv is None:
        r.ok("CACHE-10: the narrow edit is not flagged (the region path is expected to win)")
    else:
        r.fail("CACHE-10 mid-graph", f"flagged a narrow edit as slower: {adv!r}")


def test_cache10_refuses_rather_than_guesses(r: SubTestResult):
    """NEVER-SEVER ROWS, mirroring CACHE-7's placement discipline (`plan_checkpoints`): every
    input that would make the price a guess yields NO advisory rather than a fabricated one."""
    print("\n--- CACHE-10: never fabricates a verdict ---")
    halos = [0, 0, 0]
    roi = (0, 0, 32, 32, 32, 32)
    good_costs = {0: 10.0, 1: 10.0, 2: 10.0}
    px = 32 * 32

    rows = [
        ("no costs at all (the profiler is disarmed — the DEFAULT)",
         dict(halos=halos, roi=roi, dirty_from=0, costs=None, px=px, settled=True)),
        ("an empty costs dict",
         dict(halos=halos, roi=roi, dirty_from=0, costs={}, px=px, settled=True)),
        ("UNSETTLED (the EWMA still carries the cold cook)",
         dict(halos=halos, roi=roi, dirty_from=0, costs=good_costs, px=px, settled=False)),
        ("only the unfused `None` stage key (not a fused chain)",
         dict(halos=halos, roi=roi, dirty_from=0, costs={"None": 12.0}, px=px, settled=True)),
        ("dirty_from past the end (nothing left to recook)",
         dict(halos=halos, roi=roi, dirty_from=5, costs=good_costs, px=px, settled=True)),
    ]
    for label, kw in rows:
        got = ROI.region_advisory(**kw)
        if got is None:
            r.ok(f"CACHE-10 refuses: {label}")
        else:
            r.fail(f"CACHE-10 refuse ({label})", f"got {got!r}")

    # The positive control, so none of the rows above passed vacuously.
    placed = ROI.region_advisory(halos, roi, 0, costs=good_costs, px=px, settled=True)
    if placed is not None:
        r.ok("CACHE-10: the positive control (same shape, settled+costed) DOES advise")
    else:
        r.fail("CACHE-10 positive control", "no advisory with a fully-priced all-dirty edit")


def test_cache10_never_overrides_chain_windows(r: SubTestResult):
    """REFUSAL PASSTHROUGH. `chain_windows` may already refuse an edit for correctness
    reasons — here, P0-4a: stage 1 DECLINED its last cook (cooked whole-frame from a possibly-
    stale input) while its own upstream input (stage 0) was only ever valid over a small
    patched window, which poisons every later edit above it (see `tests/test_v033_phase0.py`
    for the same shape). `region_advisory` must never re-derive or second-guess that verdict,
    even handed perfect, settled costs."""
    print("\n--- CACHE-10: never overrides chain_windows's own refusal ---")
    halos = [0, 2, 0]
    roi = (0, 0, 8, 8, 32, 32)
    valid = [(0, 0, 4, 4, 32, 32), None, None]
    declined = [1]
    if ROI.chain_windows(halos, roi, 1, valid=valid, declined=declined) is not None:
        r.fail("CACHE-10 test setup", "chain_windows was expected to refuse this shape")
        return

    costs = {1: 999.0, 2: 999.0}
    adv = ROI.region_advisory(halos, roi, 1, costs=costs, px=32 * 32, settled=True,
                              valid=valid, declined=declined)
    if adv is None:
        r.ok("CACHE-10: passes through chain_windows's correctness refusal as None")
    else:
        r.fail("CACHE-10 refusal passthrough", f"priced a shape chain_windows refused: {adv!r}")


def test_cache10_structured_and_stable(r: SubTestResult):
    """The advisory mirrors `tex_checkpoint.GateRefusal`'s shape and stability promise: a
    stable `code` a host can key on, a frozen record, and a `message` that carries no contract
    but does carry both priced sides."""
    print("\n--- CACHE-10: the advisory is structured data, not a decision ---")
    halos = [0]
    roi = (0, 0, 16, 16, 16, 16)
    adv = ROI.region_advisory(halos, roi, 0, costs={0: 5.0}, px=16 * 16, settled=True)
    wrong = []
    if adv is None:
        wrong.append("expected an advisory for a trivial all-dirty, single-stage edit")
    else:
        if adv.code != ROI.ADVISE_REGION_SLOWER:
            wrong.append(f"code {adv.code!r}, wanted {ROI.ADVISE_REGION_SLOWER!r}")
        if (f"{adv.region_ms:.3f}" not in adv.message
                or f"{adv.whole_ms:.3f}" not in adv.message):
            wrong.append(f"message does not carry both priced sides: {adv.message!r}")
        try:
            adv.region_ms = 0.0
            wrong.append("RegionAdvisory is not frozen: a field was reassigned")
        except Exception:
            pass
    if wrong:
        r.fail("CACHE-10 structure", "; ".join(wrong))
    else:
        r.ok("CACHE-10: code is stable, message carries both figures, the record is frozen")


def test_cache10_off_the_default_path(r: SubTestResult):
    """CANARY. The ComfyUI cook path must neither import nor reach any of this — it never
    drives a region recook at all, so its scenario reads zero on every ROI and results-cache
    row by construction (invariant #7)."""
    print("\n--- CACHE-10: off the default path ---")
    node_src = (Path(__file__).resolve().parent.parent / "tex_node.py").read_text(
        encoding="utf-8")
    for name in ("tex_roi", "region_advisory", "RegionAdvisory", "chain_windows"):
        if name in node_src:
            r.fail("CACHE-10 invariant #7", f"tex_node.py references {name}")
        else:
            r.ok(f"CACHE-10 invariant #7: tex_node.py never mentions {name}")
