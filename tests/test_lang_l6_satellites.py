"""LANG-L6 — the satellite tiers under language `0.25`.

L6 of `docs/masked-control-flow.md` §8: the three tiers that sit BESIDE the interpreter and
codegen — CUDA-graph capture, `precision="auto"`, and the ROI / strip / batch-strip
planners — each asked what the `0.25` rules mean for it, plus the lazy analysis, which is
asked to prove it means nothing.

  * `graphed._capturable` declines a flagged program with sync points (a loop whose live
    mask is re-tested per pass) and keeps a sync-free `0.25` program at its exact
    `(True, op_count)`.
  * `precision_policy.resolve_auto_precision` declines a `0.25` per-pixel `for` bound (a
    per-pixel data branch) and keeps everything it accepts today.
  * The region-dependence sunset (`tex_roi.region_dependent`, clauses (a)/(b)) is
    CHARACTERIZED: with the engine at `0.25` and the program asking for it, the ROI window,
    the halo strips and the batch strips all equal the whole frame bitwise — and the same
    programs under `0.23`/`0.24` rules are still declined by every planner and still
    disagree with the whole frame when an executor is driven directly.
  * `tex_lazy.lazy_required_bindings` answers the same set with and without the pragma, and
    with the engine at either level, because it never reads the language at all.

**The two seams.** `LANGUAGE_VERSION` is `"0.24"` at this head (L7 moves it), and the gate is
`min(pragma, LANGUAGE_VERSION)`, so no program can reach the `0.25` rules through any engine
call site. Where the tier has an entry point of its own, the rules are asked for through the
same leading-underscore `_masked_flow` keyword LANG-L4/L5 gave `Interpreter.execute` and
`codegen.try_compile` (`None` = the engine's gate). The executors have no such seam — they
call `Interpreter.execute` with the default — so the characterization rows open the engine's
own gate the one way that already exists in this tree: `test_v036_region_dependence`'s T8
sets `tex_api.LANGUAGE_VERSION` to `"0.25"` under `try/finally` and clears the ROI memo.
That is deliberately the REAL gate, not a seam: those rows prove the wiring, not a stub.

Every row runs on the CPU interpreter. No ComfyUI, no CUDA (the device is a STRING the
precision gate compares), no compiler, no numpy, no Windows path, no timing.
"""
import os

import torch

from helpers import *

import compat_corpus as cc
import test_lang_l4_masked_flow as L4
import test_v036_region_dependence as V36
from TEX_Wrangle import tex_api, tex_lazy, tex_memory, tex_roi
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_lazy import clear_lazy_memo, lazy_required_bindings
from TEX_Wrangle.tex_runtime import graphed, masked_flow, precision_policy
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_runtime.precision_policy import resolve_auto_precision

PRAGMA = L4.PRAGMA                       # "//!tex 0.25\n"
_PX = 2048 * 2048                        # above every arch's fp16 floor

# ── the programs ──────────────────────────────────────────────────────────────

# R-BREAK's shape: a STATIC-range `for` that directly encloses a `break` under a per-pixel
# `if`. Capturable today (the transfer unwinds; nothing syncs); under `0.25` the loop's live
# mask is re-tested every pass — `flow_plan` names it a sync point.
BREAK_IN_STATIC_FOR = ("float a = @A.r; float hit = -1.0;\n"
                       "for (int i = 0; i < 3; i = i + 1) {\n"
                       "  if (a > 0.5) { hit = float(i) + 10.0; break; }\n"
                       "  hit = hit - 1.0;\n"
                       "}\n"
                       "@OUT = vec4(hit, hit, hit, 1.0);\n")

# A scatter under a per-pixel `if`: M5's source-gated compaction reads the mask back.
SCATTER_UNDER_IF = "if (@A.r > 0.5) { @T[ix, iy] += 1.0; }\n@OUT = @A;\n"

# Transfer-free `0.25` program: an empty plan. Must keep its exact (True, op_count).
STATIC_FOR = ("float s = 0.0; for (int i = 0; i < 4; i = i + 1) { s = s + @A.r * float(i); } "
              "@OUT = vec4(s, s, s, 1.0);\n")

# A transfer that never syncs: a `return` under a per-pixel `if`, called at uniform live.
RETURN_ONLY = ("float pick(float a) { if (a > 0.5) { return a * 10.0; } return a * 100.0; }\n"
               "float r = pick(@A.r);\n"
               "@OUT = vec4(r, r, r, 1.0);\n")

# A `for` bounded by a COORDINATE: fp16-eligible today (no image lineage in the comparison,
# nothing image-tainted accumulates), a per-pixel data branch under `0.25`.
U_BOUNDED_FOR = ("float c = 0.0; for (int i = 0; float(i) < u * 6.0; i = i + 1) { c = c + 1.0; } "
                 "@OUT = vec4(c * 0.1, c * 0.1, c * 0.1, 1.0);\n")
# The batch-axis twin (clause (b)): bounded by the frame index.
FI_BOUNDED_FOR = ("float c = 0.0; for (int i = 0; float(i) < fi + 1.0; i = i + 1) { c = c + 1.0; } "
                  "@OUT = vec4(c * 0.1, c * 0.1, c * 0.1, 1.0);\n")
# A static `for` over a coordinate: fp16 today and fp16 under `0.25` (an empty plan).
STATIC_FOR_OVER_U = ("float c = 0.0; for (int i = 0; i < 6; i = i + 1) { c = c + u * 0.1; } "
                     "@OUT = vec4(c, c, c, 1.0);\n")
# An image-lineage bound: declined TODAY by the comparison clause, with today's reason.
IMAGE_BOUNDED_FOR = ("float n = @A.r * 6.0; float c = 0.0; "
                     "for (int i = 0; float(i) < n; i = i + 1) { c = c + 1.0; } "
                     "@OUT = vec4(c * 0.1, c * 0.1, c * 0.1, 1.0);\n")

# TRK-25's own repro (H=8, W=2, column 0 read top to bottom): whole-frame under `0.23` is
# 4 passes on every row; per pixel, `v` runs 0 … 1 INCLUSIVE down the rows (y/(H-1)), so a
# row needs ceil((1-v)/0.25) passes and the bottom row needs none. Note this is NOT T1's
# 4-strip column ([…, 1, 1]): a strip still runs its own maximum, a pixel runs its own.
REPRO = V36.REPRO
REPRO_PER_PIXEL_COLUMN = [4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 0.0]
REPRO_REGION_MAX_COLUMN = [4.0] * 8
# A per-pixel `for` bound on the same axis.
V_BOUNDED_FOR = ("float n = 0.0;\n"
                 "for (int i = 0; float(i) < v * 4.0; i = i + 1) { n = n + 1.0; }\n"
                 "@OUT = vec4(n,n,n,1.0);\n")

_ATOMS = L4._ATOM_PROGRAMS
_WORKED = {name: src for name, (src, _b, _a) in L4._WORKED.items()}
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

# LANG-L7: the corpus rows (and the shipped example) that ASK for `0.25` on purpose — the real
# engine now grants it, so every OTHER invariant-7 "nothing moved" sweep in this file excludes
# exactly these. Kept as its own local constant (not imported from
# test_lang_l5_codegen_masking) so this file's corpus sweeps do not depend on that file's
# import order.
_LANG_L7_MASKED_CORPUS_NAMES = frozenset({
    "adv025_break", "adv025_continue", "adv025_return",
    "adv025_for_bound", "adv025_while_bound", "per_pixel_control_flow",
})


def _parse(src):
    return Parser(Lexer(src).tokenize(), source=src).parse()


class _engine_at:
    """T8's mechanism: the engine's OWN language level, under try/finally, with every memo
    that caches a level-dependent verdict cleared on the way in and on the way out."""

    def __init__(self, version):
        self.version = version

    def __enter__(self):
        self.saved = tex_api.LANGUAGE_VERSION
        tex_api.LANGUAGE_VERSION = self.version
        tex_roi.clear_roi_memo()
        clear_lazy_memo()
        return self

    def __exit__(self, *exc):
        tex_api.LANGUAGE_VERSION = self.saved
        tex_roi.clear_roi_memo()
        clear_lazy_memo()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 1. Graph capture
# ══════════════════════════════════════════════════════════════════════════════

def test_l6_capture_gate_declines_a_flagged_program_with_sync_points(r: SubTestResult):
    print("\n--- L6: _capturable is False for a flagged program with sync points ---")
    for label, src in (("break in a static for", BREAK_IN_STATIC_FOR),
                       ("scatter under a per-pixel if", SCATTER_UNDER_IF)):
        try:
            prog = _parse(PRAGMA + src)
            plan = tex_api.flow_plan(prog)
            assert plan.sync_points or plan.scatter_sites, "the plan names no sync"
            unflagged = graphed._capturable(prog, _masked_flow=False)
            assert unflagged[0] is True, f"the 0.23 walk already declines it: {unflagged}"
            flagged = graphed._capturable(prog, _masked_flow=True)
            assert flagged == (False, 0), f"flagged verdict {flagged}"
            r.ok(f"{label}: capturable {unflagged} unflagged, {flagged} under 0.25")
        except Exception as e:
            r.fail(f"L6 capture declines {label}", f"{type(e).__name__}: {e}")

    # Every L4 atom whose plan has a sync point (TRK-154: including a masked call site) and
    # whose 0.23 verdict is True must flip.
    try:
        flipped = []
        for name in sorted(_ATOMS):
            prog = _parse(PRAGMA + _ATOMS[name])
            plan = tex_api.flow_plan(prog)
            if not (plan.sync_points or plan.scatter_sites or plan.call_sites):
                continue
            if graphed._capturable(prog, _masked_flow=False)[0] is not True:
                continue                                # declined at every level already
            assert graphed._capturable(prog, _masked_flow=True) == (False, 0), name
            flipped.append(name)
        assert len(flipped) >= 3, f"too few atoms exercise the decline: {flipped}"
        assert "binding_write_in_call" in flipped, (
            "TRK-154: a call reached under a per-pixel `if` syncs (M4's empty-call skip) "
            "regardless of what its body does — this atom's plan must carry a call_sites "
            "entry and its capturable verdict must flip under 0.25")
        r.ok(f"{len(flipped)} L4 atoms capturable under 0.23 decline under 0.25: "
             f"{', '.join(flipped)}")
    except Exception as e:
        r.fail("L6 capture declines the sync-bearing atoms", f"{type(e).__name__}: {e}")

    # The ENGINE's gate, not the seam: with `_masked_flow=None` the verdict follows the
    # language level — shut below 0.25, open when the engine implements it. LANG-L7 moved
    # the real engine TO 0.25, so the "shut" half is exercised via `_engine_at("0.24")`.
    try:
        with _engine_at("0.24"):
            prog = _parse(PRAGMA + BREAK_IN_STATIC_FOR)
            assert graphed._capturable(prog) == graphed._capturable(prog, _masked_flow=False), \
                "a 0.25 pragma alone moved the verdict while the engine is below 0.25"
        with _engine_at("0.25"):
            assert graphed._capturable(_parse(PRAGMA + BREAK_IN_STATIC_FOR)) == (False, 0), \
                "engine and program both at 0.25: the gate did not open"
            assert graphed._capturable(_parse(BREAK_IN_STATIC_FOR))[0] is True, \
                "engine at 0.25, no pragma: must keep 0.23's verdict"
            assert graphed._capturable(_parse("//!tex 0.24\n" + BREAK_IN_STATIC_FOR))[0] is True
        r.ok("the default seam follows the engine's own gate (shut at 0.24, open at 0.25)")
    except Exception as e:
        r.fail("L6 capture gate wiring", f"{type(e).__name__}: {e}")


def test_l6_capture_gate_keeps_a_sync_free_025_program(r: SubTestResult):
    print("\n--- L6: _capturable stays (True, op_count) for a sync-free 0.25 program ---")
    for label, src in (("transfer-free (empty plan)", STATIC_FOR),
                       ("return under a per-pixel if (no per-pass live test)", RETURN_ONLY)):
        try:
            prog = _parse(PRAGMA + src)
            unflagged = graphed._capturable(prog, _masked_flow=False)
            flagged = graphed._capturable(prog, _masked_flow=True)
            assert unflagged[0] is True, unflagged
            assert flagged == unflagged, f"{flagged} != {unflagged}: the op count moved too"
            r.ok(f"{label}: {flagged} under both rule sets")
        except Exception as e:
            r.fail(f"L6 capture keeps {label}", f"{type(e).__name__}: {e}")

    # …and across every L4 atom: a plan with no sync/scatter/call site (TRK-154: the third
    # is now part of "sync-free") leaves the verdict AND the op count exactly where 0.23
    # put them.
    try:
        kept = 0
        for name in sorted(_ATOMS):
            prog = _parse(PRAGMA + _ATOMS[name])
            plan = tex_api.flow_plan(prog)
            if plan.sync_points or plan.scatter_sites or plan.call_sites:
                continue
            assert graphed._capturable(prog, _masked_flow=True) == \
                graphed._capturable(prog, _masked_flow=False), name
            kept += 1
        assert kept >= 1, "no atom is sync-free — the row is vacuous"
        r.ok(f"{kept} sync-free atoms keep their exact (capturable, op_count)")
    except Exception as e:
        r.fail("L6 capture keeps the sync-free atoms", f"{type(e).__name__}: {e}")

    # An INCOMPLETE plan declines — the walk could not say where the syncs are.
    try:
        prog = _parse(PRAGMA + STATIC_FOR)
        saved = tex_api.flow_plan
        tex_api.flow_plan = lambda p, bt=None: tex_api._INCOMPLETE_FLOW_PLAN
        try:
            assert graphed._capturable(prog, _masked_flow=True) == (False, 0)
        finally:
            tex_api.flow_plan = saved
        r.ok("an incomplete plan declines (fail-closed)")
    except Exception as e:
        r.fail("L6 capture incomplete plan", f"{type(e).__name__}: {e}")


def test_l6_capture_gate_mutations_both_directions(r: SubTestResult):
    print("\n--- L6: the capture check is load-bearing in both directions ---")
    saved = graphed._masked_flow_syncs
    try:
        # Direction 1: the check is silenced — the flagged program would read capturable.
        graphed._masked_flow_syncs = lambda p, m=None: False
        prog = _parse(PRAGMA + BREAK_IN_STATIC_FOR)
        assert graphed._capturable(prog, _masked_flow=True)[0] is True, \
            "silencing the check moved nothing — `_capturable` is not consulting it"
        r.ok("removing the check re-arms the doomed capture (the decline row would red)")
    except Exception as e:
        r.fail("L6 capture mutation (removed)", f"{type(e).__name__}: {e}")
    finally:
        graphed._masked_flow_syncs = saved
    try:
        # Direction 2: the check declines EVERY flagged program — the sync-free row reds.
        graphed._masked_flow_syncs = lambda p, m=None: bool(m)
        prog = _parse(PRAGMA + STATIC_FOR)
        assert graphed._capturable(prog, _masked_flow=True) == (False, 0), \
            "a blanket decline moved nothing — the seam is not reaching the check"
        r.ok("a blanket decline is visible (the keep row would red)")
    except Exception as e:
        r.fail("L6 capture mutation (widened)", f"{type(e).__name__}: {e}")
    finally:
        graphed._masked_flow_syncs = saved
    try:
        assert graphed._capturable(_parse(PRAGMA + STATIC_FOR), _masked_flow=True)[0] is True
        assert graphed._capturable(_parse(PRAGMA + BREAK_IN_STATIC_FOR),
                                   _masked_flow=True)[0] is False
        r.ok("restored")
    except Exception as e:
        r.fail("L6 capture mutation restore", f"{type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. precision="auto"
# ══════════════════════════════════════════════════════════════════════════════

def test_l6_auto_declines_a_025_per_pixel_for(r: SubTestResult):
    print("\n--- L6: precision=auto declines a 0.25 per-pixel `for` ---")
    for label, src in (("a coordinate-bounded for", U_BOUNDED_FOR),
                       ("a frame-index-bounded for (clause b)", FI_BOUNDED_FOR)):
        try:
            prog = _parse(PRAGMA + src)
            plan = tex_api.flow_plan(prog)
            assert plan.per_pixel_loops, "the plan names no per-pixel loop"
            today = resolve_auto_precision(prog, _PX, "cuda", _masked_flow=False)
            assert today[0] == "fp16", f"not fp16-eligible today: {today}"
            flagged = resolve_auto_precision(prog, _PX, "cuda", _masked_flow=True)
            assert flagged[0] == "fp32", flagged
            assert "0.25" in flagged[1] and "for" in flagged[1], flagged[1]
            r.ok(f"{label}: {today[0]} under 0.23 -> {flagged[0]} under 0.25 ({flagged[1]})")
        except Exception as e:
            r.fail(f"L6 auto declines {label}", f"{type(e).__name__}: {e}")

    try:
        with _engine_at("0.24"):
            prog = _parse(PRAGMA + U_BOUNDED_FOR)
            assert resolve_auto_precision(prog, _PX, "cuda") == \
                resolve_auto_precision(prog, _PX, "cuda", _masked_flow=False), \
                "a 0.25 pragma alone moved the verdict while the engine is below 0.25"
        with _engine_at("0.25"):
            assert resolve_auto_precision(_parse(PRAGMA + U_BOUNDED_FOR), _PX, "cuda")[0] == "fp32"
            assert resolve_auto_precision(_parse(U_BOUNDED_FOR), _PX, "cuda")[0] == "fp16", \
                "engine at 0.25, no pragma: must keep 0.23's verdict"
            assert resolve_auto_precision(_parse("//!tex 0.24\n" + U_BOUNDED_FOR),
                                          _PX, "cuda")[0] == "fp16"
        r.ok("the default seam follows the engine's own gate (shut at 0.24, open at 0.25)")
    except Exception as e:
        r.fail("L6 auto gate wiring", f"{type(e).__name__}: {e}")

    # The earlier clauses keep priority: the CPU answer and the floor come first.
    try:
        prog = _parse(PRAGMA + U_BOUNDED_FOR)
        assert resolve_auto_precision(prog, _PX, "cpu", _masked_flow=True)[1].startswith(
            "auto->fp32: CPU")
        assert "^2" in resolve_auto_precision(prog, 16, "cuda", _masked_flow=True)[1]
        r.ok("the CPU and resolution clauses still answer first")
    except Exception as e:
        r.fail("L6 auto clause order", f"{type(e).__name__}: {e}")


def test_l6_auto_keeps_what_it_accepts_today(r: SubTestResult):
    print("\n--- L6: precision=auto is not a blanket decline of 0.25 programs ---")
    try:
        prog = _parse(PRAGMA + STATIC_FOR_OVER_U)
        assert tex_api.flow_plan(prog).is_empty()
        today = resolve_auto_precision(prog, _PX, "cuda", _masked_flow=False)
        flagged = resolve_auto_precision(prog, _PX, "cuda", _masked_flow=True)
        assert today[0] == "fp16" and flagged == today, (today, flagged)
        r.ok("a 0.25 program with a static for keeps its fp16, reason string included")
    except Exception as e:
        r.fail("L6 auto keeps a static for", f"{type(e).__name__}: {e}")
    try:
        prog = _parse(PRAGMA + IMAGE_BOUNDED_FOR)
        today = resolve_auto_precision(prog, _PX, "cuda", _masked_flow=False)
        flagged = resolve_auto_precision(prog, _PX, "cuda", _masked_flow=True)
        assert today[0] == "fp32" and "0.25" not in today[1], today
        assert flagged == today, f"the reason string moved: {flagged} != {today}"
        r.ok("an image-bounded for keeps TODAY's declining reason (asked last)")
    except Exception as e:
        r.fail("L6 auto keeps today's reason", f"{type(e).__name__}: {e}")
    # Every L4 atom: the verdict under 0.25 may only move fp16 -> fp32, never the reverse,
    # and only for a program whose plan holds a per-pixel `for`.
    try:
        moved, same = [], 0
        for name in sorted(_ATOMS):
            prog = _parse(PRAGMA + _ATOMS[name])
            today = resolve_auto_precision(prog, _PX, "cuda", _masked_flow=False)
            flagged = resolve_auto_precision(prog, _PX, "cuda", _masked_flow=True)
            if flagged == today:
                same += 1
                continue
            assert today[0] == "fp16" and flagged[0] == "fp32", (name, today, flagged)
            assert tex_api.flow_plan(prog).per_pixel_loops, name
            moved.append(name)
        r.ok(f"atoms: {same} unchanged, {len(moved)} fp16->fp32 ({', '.join(moved) or 'none'})")
    except Exception as e:
        r.fail("L6 auto over the atoms", f"{type(e).__name__}: {e}")


def test_l6_auto_decline_mutations_both_directions(r: SubTestResult):
    print("\n--- L6: the auto decline is load-bearing in both directions ---")
    saved = precision_policy._masked_per_pixel_for
    try:
        precision_policy._masked_per_pixel_for = lambda p, m=None: False
        assert resolve_auto_precision(_parse(PRAGMA + U_BOUNDED_FOR), _PX, "cuda",
                                      _masked_flow=True)[0] == "fp16", \
            "silencing the check moved nothing — `resolve_auto_precision` is not consulting it"
        r.ok("removing the check hands the per-pixel for back to fp16 (the decline row would red)")
    except Exception as e:
        r.fail("L6 auto mutation (removed)", f"{type(e).__name__}: {e}")
    finally:
        precision_policy._masked_per_pixel_for = saved
    try:
        precision_policy._masked_per_pixel_for = lambda p, m=None: bool(m)
        assert resolve_auto_precision(_parse(PRAGMA + STATIC_FOR_OVER_U), _PX, "cuda",
                                      _masked_flow=True)[0] == "fp32", \
            "a blanket decline moved nothing — the seam is not reaching the check"
        r.ok("a blanket decline is visible (the keep row would red)")
    except Exception as e:
        r.fail("L6 auto mutation (widened)", f"{type(e).__name__}: {e}")
    finally:
        precision_policy._masked_per_pixel_for = saved
    try:
        assert resolve_auto_precision(_parse(PRAGMA + STATIC_FOR_OVER_U), _PX, "cuda",
                                      _masked_flow=True)[0] == "fp16"
        assert resolve_auto_precision(_parse(PRAGMA + U_BOUNDED_FOR), _PX, "cuda",
                                      _masked_flow=True)[0] == "fp32"
        r.ok("restored")
    except Exception as e:
        r.fail("L6 auto mutation restore", f"{type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. The region-dependence sunset, characterized
# ══════════════════════════════════════════════════════════════════════════════

def _triple(src, shape=(2, 8, 2)):
    """Whole frame, then every executor driven DIRECTLY (they are dumb by design)."""
    b, h, w = shape
    prog, whole = V36._cook(src, shape=shape)
    _, t2 = V36._cook(src, shape=shape, tiles=2)
    _, t4 = V36._cook(src, shape=shape, tiles=4)
    _, roi = V36._cook(src, shape=shape, roi=(0, h // 2, w, h // 2, w, h),
                       narrow=frozenset({"A"}), halo=0)
    _, batch = V36._cook(src, shape=shape, batch=2)
    return prog, whole["OUT"], {"2 strips": t2["OUT"], "4 strips": t4["OUT"],
                                "ROI window": roi["OUT"], "batch strips": batch["OUT"]}


def _same_as_whole(whole, name, out):
    if name == "ROI window":
        return torch.equal(whole[:, whole.shape[1] // 2:], out)
    return torch.equal(whole, out)


def _halo_pair(src):
    prog = tex_api.compile(src, {"A": TEXType.VEC4})
    names = sorted(prog.assigned.keys())
    bindings = {"A": V36._img(1, 64, 64)}
    interp = Interpreter()
    whole = interp.execute(prog.ast, bindings, prog.type_map, device="cpu",
                           latent_channel_count=0, output_names=names,
                           used_builtins=prog.used_builtins, precision="fp32")
    striped = tex_memory.run_tiled_halo(interp, prog.ast, bindings, prog.type_map, "cpu", 0,
                                        names, prog.used_builtins, "fp32", 2,
                                        frozenset({"A"}), 6)
    return prog, whole["OUT"], striped["OUT"]


def test_l6_split_triple_equals_the_whole_frame_under_025(r: SubTestResult):
    print("\n--- L6: engine at 0.25 + `//!tex 0.25`: every split equals the whole frame ---")
    with _engine_at("0.25"):
        for label, src in (("TRK-25's repro (a per-pixel while)", REPRO),
                           ("a per-pixel for bound", V_BOUNDED_FOR),
                           ("a break under a per-pixel guard", BREAK_IN_STATIC_FOR)):
            code = PRAGMA + src
            try:
                assert tex_roi.region_dependent(_parse(code), code=code) is False
                assert tex_roi.roi_plan(code, {}).executable is True
                assert tex_roi.batch_sliceable(code, {}) is True
                assert V36._tile_plan_for(code, free_hint=1024.0) is not None, \
                    "_tile_plan under pressure still declines"
                r.ok(f"{label}: region_dependent False; ROI, strip and batch planners all accept")
            except Exception as e:
                r.fail(f"L6 planners accept {label}", f"{type(e).__name__}: {e}")
            try:
                prog, whole, outs = _triple(code)
                assert prog.ast.language == "0.25", prog.ast.language
                for name, out in outs.items():
                    assert _same_as_whole(whole, name, out), \
                        f"{name} != whole frame (maxdiff {V36._maxdiff(whole, out)})"
                r.ok(f"{label}: {', '.join(outs)} are all torch.equal to the whole frame")
            except Exception as e:
                r.fail(f"L6 triple equals whole {label}", f"{type(e).__name__}: {e}")

        # The masked path genuinely ran: the whole frame is the PER-PIXEL answer, which
        # under 0.23 only a 4-strip split ever produced (T1's own characterization).
        try:
            _prog, whole = V36._cook(PRAGMA + REPRO)
            assert V36._column(whole["OUT"]) == REPRO_PER_PIXEL_COLUMN, V36._column(whole["OUT"])
            r.ok(f"the whole frame is the per-pixel column {REPRO_PER_PIXEL_COLUMN}")
        except Exception as e:
            r.fail("L6 masked path ran", f"{type(e).__name__}: {e}")

        # The HALO strip route (blur + per-pixel loop, the class `is_tile_safe` refuses).
        try:
            code = PRAGMA + V36.HALO_REPRO
            plan = tex_roi.roi_plan(code, {})
            assert plan.executable is True and plan.halo == 6, (plan.executable, plan.halo)
            _prog, whole, striped = _halo_pair(code)
            assert torch.equal(whole, striped), f"maxdiff {V36._maxdiff(whole, striped)}"
            r.ok("halo strips (halo 6) are torch.equal to the whole frame")
        except Exception as e:
            r.fail("L6 halo strips equal whole", f"{type(e).__name__}: {e}")

        # The batch axis (clause (b)): an `fi`-bounded loop, frame strips of 2.
        try:
            code = PRAGMA + V36.FI_REPRO
            assert tex_roi.batch_sliceable(code, {}) is True
            _p, whole = V36._cook(code, shape=(4, 2, 4))
            _p2, striped = V36._cook(code, shape=(4, 2, 4), batch=2)
            assert torch.equal(whole["OUT"], striped["OUT"])
            per_frame = [round(float(whole["OUT"][b, 0, 0, 0]), 4) for b in range(4)]
            assert per_frame == [1.0, 2.0, 3.0, 4.0], per_frame
            r.ok(f"batch strips of an fi-bounded loop equal the whole batch ({per_frame})")
        except Exception as e:
            r.fail("L6 batch strips equal whole", f"{type(e).__name__}: {e}")


def test_l6_split_triple_still_differs_below_025(r: SubTestResult):
    print("\n--- L6: the same programs under 0.23/0.24 rules: declined, and they differ ---")
    # Three ways to be below 0.25: no pragma, an older pragma, and — at THIS head — the
    # 0.25 pragma itself, since the engine is at 0.24 and the gate is the minimum.
    headers = [("no pragma", ""), ("//!tex 0.23", "//!tex 0.23\n"),
               ("//!tex 0.24", "//!tex 0.24\n")]
    if tex_api._ver_tuple(tex_api.LANGUAGE_VERSION) < tex_roi.MASKED_FLOW_SINCE:
        headers.append((f"//!tex 0.25 on a {tex_api.LANGUAGE_VERSION} engine", PRAGMA))
    for hlabel, header in headers:
        for label, src in (("TRK-25's repro", REPRO), ("a per-pixel for bound", V_BOUNDED_FOR)):
            code = header + src
            try:
                assert tex_roi.region_dependent(_parse(code), code=code) is True
                assert tex_roi.roi_plan(code, {}).executable is False
                assert tex_roi.batch_sliceable(code, {}) is False
                assert V36._tile_plan_for(code, free_hint=1024.0) is None
                r.ok(f"{hlabel}, {label}: every planner declines")
            except Exception as e:
                r.fail(f"L6 planners decline {hlabel} {label}", f"{type(e).__name__}: {e}")
        try:
            code = header + REPRO
            _prog, whole, outs = _triple(code)
            assert V36._column(whole) == REPRO_REGION_MAX_COLUMN, V36._column(whole)
            differing = [n for n, o in outs.items() if not _same_as_whole(whole, n, o)]
            assert "2 strips" in differing and "4 strips" in differing and "ROI window" in differing, \
                f"only {differing} differ"
            r.ok(f"{hlabel}: whole frame is {REPRO_REGION_MAX_COLUMN[:2]}…; "
                 f"{', '.join(differing)} disagree with it")
        except Exception as e:
            r.fail(f"L6 triple differs {hlabel}", f"{type(e).__name__}: {e}")
    # The same with the engine at 0.25 and the program NOT asking: still declined.
    with _engine_at("0.25"):
        for hlabel, header in headers[:3]:
            code = header + REPRO
            try:
                assert tex_roi.region_dependent(_parse(code), code=code) is True
                assert tex_roi.roi_plan(code, {}).executable is False
                assert tex_roi.batch_sliceable(code, {}) is False
                _prog, whole, outs = _triple(code)
                assert V36._column(whole) == REPRO_REGION_MAX_COLUMN
                assert not _same_as_whole(whole, "4 strips", outs["4 strips"])
                r.ok(f"engine at 0.25, {hlabel}: declined, and 4 strips still differ")
            except Exception as e:
                r.fail(f"L6 engine-0.25 {hlabel}", f"{type(e).__name__}: {e}")
        try:
            code = V36.HALO_REPRO
            _prog, whole, striped = _halo_pair(code)
            assert abs(V36._maxdiff(whole, striped) - 1.0) < 1e-5
            code = V36.FI_REPRO
            _p, whole = V36._cook(code, shape=(4, 2, 4))
            _p2, striped = V36._cook(code, shape=(4, 2, 4), batch=2)
            assert not torch.equal(whole["OUT"], striped["OUT"])
            r.ok("engine at 0.25, no pragma: halo strips and batch strips still differ")
        except Exception as e:
            r.fail("L6 engine-0.25 halo/batch differ", f"{type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. The lazy analysis: syntactic, and blind to the language on purpose
# ══════════════════════════════════════════════════════════════════════════════

_NEVER_SEVER = {
    "*0 (NaN*0 = NaN)": "@OUT = @A * 0.0 + @B * 0.0;",
    "&& operand": "if ($t > 0.5 && luma(@B) > 0.2) { @OUT = @B; } else { @OUT = @A * 0.1; }",
    "spatial-condition branch": "if (@A.r > 0.5) { @OUT = @B; } else { @OUT = @A; }",
}


def test_l6_lazy_analysis_never_reads_the_language(r: SubTestResult):
    print("\n--- L6: the never-sever lazy row: the pragma moves no required set ---")
    programs = {}
    programs.update({f"worked:{k}": v for k, v in _WORKED.items()})
    programs.update({f"atom:{k}": v for k, v in _ATOMS.items()})
    programs.update({"repro": REPRO, "halo repro": V36.HALO_REPRO, "fi repro": V36.FI_REPRO})
    programs.update({f"never-sever:{k}": v for k, v in _NEVER_SEVER.items()})
    try:
        clear_lazy_memo()
        plain = {k: lazy_required_bindings(v, {"t": 0.0}) for k, v in programs.items()}
        pragma = {k: lazy_required_bindings(PRAGMA + v, {"t": 0.0}) for k, v in programs.items()}
        with _engine_at("0.25"):
            at25 = {k: lazy_required_bindings(PRAGMA + v, {"t": 0.0}) for k, v in programs.items()}
        none = [k for k, v in plain.items() if v is None]
        assert not none, f"the analysis failed on {none}"
        moved = [k for k in programs if not (plain[k] == pragma[k] == at25[k])]
        assert not moved, f"the pragma or the engine level moved the required set of {moved}"
        assert all("A" in plain[k] for k in programs if "@A" in programs[k])
        assert all("B" in plain[k] for k in programs if "@B" in programs[k])
        r.ok(f"{len(programs)} programs: identical required sets with/without the pragma, "
             f"engine at 0.24 and at 0.25 (never-sever rows keep A and B)")
    except Exception as e:
        r.fail("L6 lazy verdicts", f"{type(e).__name__}: {e}")
    finally:
        clear_lazy_memo()
    # The structural half: `tex_lazy` has no vocabulary for the language at all.
    try:
        with open(tex_lazy.__file__, encoding="utf-8") as f:
            src = f.read()
        for token in ("LANGUAGE_VERSION", "masked_flow", "flow_plan", "MASKED_FLOW_SINCE",
                      "_language_tuple", "language_pragma", ".language"):
            assert token not in src, f"tex_lazy.py now mentions {token!r}"
        r.ok("tex_lazy.py names no language-level symbol — the analysis is syntactic")
    except Exception as e:
        r.fail("L6 lazy is syntactic", f"{type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Invariant 7, in-tree: no corpus verdict moves, and the version has not
# ══════════════════════════════════════════════════════════════════════════════

def test_l6_corpus_satellite_verdicts_are_unmoved(r: SubTestResult):
    print("\n--- L6: every corpus program's capture and auto verdicts equal the forced-off ones ---")
    try:
        programs = dict(cc._corpus_programs())
        assert len(programs) >= 100, len(programs)
        moved, checked = [], 0
        for name, src in sorted(programs.items()):
            if name in _LANG_L7_MASKED_CORPUS_NAMES:
                continue    # LANG-L7's own rows ASK for 0.25 and the real engine now grants
                            # it — excluded here, proved masked-and-unmoved-in-VALUE by
                            # test_masked_gate_is_open_only_for_this_lanes_pragma_rows
                            # (test_lang_l4_masked_flow.py) and this file's own §3 split-triple
                            # characterization, not by this invariant-7 sweep.
            try:
                prog = _parse(src)
            except Exception:
                continue
            assert masked_flow.enabled_for(prog, src) is False, name
            if graphed._capturable(prog) != graphed._capturable(prog, _masked_flow=False):
                moved.append(f"{name}:capture")
            if resolve_auto_precision(prog, _PX, "cuda") != \
                    resolve_auto_precision(prog, _PX, "cuda", _masked_flow=False):
                moved.append(f"{name}:auto")
            checked += 1
        assert not moved, moved
        r.ok(f"{checked} corpus programs: default-seam verdicts == forced-off verdicts")
    except Exception as e:
        r.fail("L6 corpus verdicts", f"{type(e).__name__}: {e}")


def test_l6_language_version_reached_masked_flow(r: SubTestResult):
    print("\n--- L6->L7: LANGUAGE_VERSION reached masked flow ---")
    try:
        assert tex_api.LANGUAGE_VERSION == "0.25"
        assert tex_roi.MASKED_FLOW_SINCE == (0, 25)
        r.ok("LANGUAGE_VERSION 0.25, MASKED_FLOW_SINCE (0, 25)")
    except Exception as e:
        r.fail("L6->L7 version pin", f"{type(e).__name__}: {e}")
