"""TRK-219 (PRUNE-49) — the verified static-flow prune lives in `tex_roi._fold_program` itself,
so every one of its four consumers (`_walk`, `frame_window`, `batch_sliceable`,
`_scale_verdict_uncached`) — and, through `_walk`/`batch_sliceable`, `region_dependent` — sees
the PRUNED program, not just the two walkers FIX-ROI/O2 scoped it to.

WHY THIS IS SAFE (invariant #11). `_prune_static_flow` only ever removes a branch whose
condition `_fold_program`'s own fp32-verified fold (ROI-48A/O1: `_capture_pre_fold_conditions`
+ `_revert_unverified_folds`) has PROVEN a `NumberLiteral` — never a guess (see
`tex_roi._fold_program`'s docstring). Removing a construct that can never execute at these
exact `$param` values can only make a static analysis's answer MORE PERMISSIVE (a narrower
frame window, `batch_sliceable`/`scale_verdict` moving refused -> allowed, `region_dependent`
moving True -> False) — never wrong, and never in the other direction. Every row below is
RED-FIRST (asserts the value the pre-TRK-219 `tex_roi` produces first, reproducing FIX-ROI's own
finding that these four consumers do not yet see the pruned tree), then proven GREEN, then
backed by a PIXEL-IDENTITY cook (a windowed / strip / batch-strip cook is `torch.equal` to the
whole-frame crop) so the verdict move is proven correct, not merely different.

`tests/test_perf1_roi_walk_memo.py` carries the derivation-oracle characterization of the two
classes this change moves in the existing corpus (`_trk219_characterize_move`); this file is the
consumer-level red-first/pixel-identity half of the same change.

CPU only (`CUDA_VISIBLE_DEVICES=-1`, set by the harness / CI). A CUDA confirmation of the same
mechanism reaching `tex_tiling`'s OOM/TDR strip planners via `roi_plan` under real memory
pressure is a follow-up owed under a held bench lease — no CUDA-specific row lives in this file.
"""
import torch

from helpers import *

from TEX_Wrangle import tex_api, tex_memory, tex_roi
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_runtime.interpreter import Interpreter


def _drop_memos():
    tex_roi.clear_roi_memo()


def _img(b, h, w, seed=0):
    torch.manual_seed(seed)
    return torch.rand(b, h, w, 4)


def _cook(src, *, shape=(1, 8, 8), roi=None, narrow=None, halo=0, batch=None,
          params=None, seed=0):
    """Compile + cook `src` whole-frame, or through `run_roi`/`run_batch_strips` directly —
    the same harness shape `test_v036_region_dependence.py` uses. Returns (prog, outputs)."""
    b, h, w = shape
    bindings = {"A": _img(b, h, w, seed)}
    btypes = {"A": TEXType.VEC4}
    for name, value in (params or {}).items():
        bindings[name] = value
        btypes[name] = TEXType.FLOAT
    prog = tex_api.compile(src, btypes)
    names = sorted(prog.assigned.keys())
    interp = Interpreter()
    head = (interp, prog.ast, bindings, prog.type_map, "cpu", 0, names,
            prog.used_builtins, "fp32")
    if batch:
        out = tex_memory.run_batch_strips(*head, batch)
    elif roi is not None:
        out = tex_memory.run_roi(*head, roi,
                                 frozenset() if narrow is None else narrow, halo)
    else:
        out = interp.execute(prog.ast, bindings, prog.type_map, device="cpu",
                             latent_channel_count=0, output_names=names,
                             used_builtins=prog.used_builtins, precision="fp32")
    return prog, out


# ── a dead branch under a $param that folds to a verified literal-false ──────

FRAME_DEAD = ("f$k = 0.0;\n"
              "if ($k > 0.5) {\n"
              "    @OUT = fetch_frame(@A, fi + 5, ix, iy);\n"
              "} else {\n"
              "    @OUT = @A;\n"
              "}\n")

BATCH_FRAME_DEAD = ("f$k = 0.0;\n"
                    "if ($k > 0.5) {\n"
                    "    @OUT = fetch_frame(@A, fi + 1, ix, iy);\n"
                    "} else {\n"
                    "    @OUT = @A;\n"
                    "}\n")

BATCH_REGDEP_DEAD = ("f$k = 0.0;\n"
                     "if ($k > 0.5) {\n"
                     "    float x = 0.0;\n"
                     "    while (x < img_width(@A)) { x = x + 1.0; }\n"
                     "}\n"
                     "@OUT = @A;\n")

SCALE_DEAD = ("f$k = 0.0;\n"
             "if ($k > 0.5) {\n"
             "    @OUT = vec4(ix * 0.001, iy * 0.001, 0.0, 1.0);\n"
             "} else {\n"
             "    @OUT = @A;\n"
             "}\n")

# A HALO op (gauss_blur) gated behind the same dead branch, so the region-dependence prune's
# widening is exercised where `roi_plan` actually narrows a real cook (halo > 0).
ROI_REGDEP_DEAD = ("f$k = 0.0;\n"
                   "if ($k > 0.5) {\n"
                   "    float x = 0.0;\n"
                   "    while (x < img_width(@A)) { x = x + 1.0; }\n"
                   "}\n"
                   "@OUT = gauss_blur(@A, 2.0);\n")


def _without_fold_level_prune(code, param_values):
    """The PRE-TRK-219 shape: `_fold_program`'s fold with no `_prune_static_flow` step —
    reproduces exactly what `frame_window`/`batch_sliceable`/`_scale_verdict_uncached` saw
    before this lane, by calling the private fold-only helper the module still exposes for
    the fp32 verification (`_capture_pre_fold_conditions`/`_revert_unverified_folds`) and
    stopping one step short of the prune this lane added. Kept local to this file (not a
    `tex_roi` export) — it exists only to make the red-first direction checkable without a
    second frozen implementation to maintain, mirroring PERF-1's own `_base_fold_program`."""
    from TEX_Wrangle.tex_compiler.ast_nodes import NumberLiteral, clone_tree
    from TEX_Wrangle.tex_lazy import (_fp32, _substitute_params, _capture_pre_fold_conditions,
                                      _revert_unverified_folds)
    from TEX_Wrangle.tex_compiler.optimizer import _propagate_literal_locals, _fold_all
    program = clone_tree(tex_roi._pristine_program(code))
    subs = {name: NumberLiteral(value=_fp32(v), is_int=isinstance(v, (bool, int)))
            for name, v in param_values.items() if isinstance(v, (bool, int, float))}
    stmts = program.statements
    if subs:
        for stmt in stmts:
            _substitute_params(stmt, subs)
        pre1 = _capture_pre_fold_conditions(stmts)
        stmts = _fold_all(stmts)
        stmts = _propagate_literal_locals(stmts)
        pre2 = _capture_pre_fold_conditions(stmts)
        stmts = _fold_all(stmts)
        _revert_unverified_folds(stmts, pre1, pre2)
        program.statements = stmts
    return program


def test_frame_window_tightens_once_the_dead_branch_is_pruned(r: SubTestResult):
    print("\n--- TRK-219: frame_window over a dead-branch fetch_frame ---")
    _drop_memos()
    pre = _without_fold_level_prune(FRAME_DEAD, {"k": 0.0})
    old_lo = old_hi = 0
    for frame_arg in tex_roi._frame_ops(pre):
        off = tex_roi._st._extract_pixel_offset(frame_arg, "fi")
        if off is None:
            old_lo, old_hi = None, None
            break
        old_lo, old_hi = min(old_lo, off), max(old_hi, off)
    r.ok(f"RED-FIRST: pre-TRK-219 shape reports a WIDER window {(old_lo, old_hi)!r} "
         f"than the live code needs ((0, 0))") if (old_lo, old_hi) != (0, 0) else \
        r.fail("TRK-219 frame_window", "the repro of the pre-change shape is already (0, 0) — "
                                       "not exercising the bug this row characterizes")
    got = tex_roi.frame_window(FRAME_DEAD, {"k": 0.0})
    r.ok(f"GREEN: frame_window is now the tight (0, 0) — the dead fetch_frame is pruned away") \
        if got == (0, 0) else \
        r.fail("TRK-219 frame_window", f"expected (0, 0), got {got!r}")


def test_batch_sliceable_permits_a_dead_frame_op(r: SubTestResult):
    print("\n--- TRK-219: batch_sliceable over a dead-branch fetch_frame ---")
    _drop_memos()
    pre = _without_fold_level_prune(BATCH_FRAME_DEAD, {"k": 0.0})
    pre_has_frame_op = any(True for _ in tex_roi._frame_ops(pre))
    r.ok("RED-FIRST: the pre-TRK-219 shape still sees the dead fetch_frame") \
        if pre_has_frame_op else \
        r.fail("TRK-219 batch_sliceable", "repro found no frame op pre-prune — not exercising it")
    got = tex_roi.batch_sliceable(BATCH_FRAME_DEAD, {"k": 0.0})
    r.ok("GREEN: batch_sliceable(k=0.0) is now True (the dead frame op is pruned away)") \
        if got is True else \
        r.fail("TRK-219 batch_sliceable", f"expected True, got {got!r}")
    # ... and the branch TAKEN (k=1.0) must still correctly refuse — never over-widened.
    live = tex_roi.batch_sliceable(BATCH_FRAME_DEAD, {"k": 1.0})
    r.ok("the LIVE branch (k=1.0) still correctly refuses (a real frame op)") \
        if live is False else \
        r.fail("TRK-219 batch_sliceable", f"k=1.0 (frame op LIVE) must stay False, got {live!r}")


def test_batch_sliceable_permits_a_dead_region_dependent_loop(r: SubTestResult):
    print("\n--- TRK-219: batch_sliceable over a dead-branch region-dependent while ---")
    _drop_memos()
    got = tex_roi.batch_sliceable(BATCH_REGDEP_DEAD, {"k": 0.0})
    r.ok("GREEN: batch_sliceable(k=0.0) is now True (the dead while loop can't run)") \
        if got is True else \
        r.fail("TRK-219 batch_sliceable/region-dep", f"expected True, got {got!r}")
    live = tex_roi.batch_sliceable(BATCH_REGDEP_DEAD, {"k": 1.0})
    r.ok("the LIVE branch (k=1.0) still correctly refuses (a real region-dependent loop)") \
        if live is False else \
        r.fail("TRK-219 batch_sliceable/region-dep", f"k=1.0 must stay False, got {live!r}")


def test_scale_verdict_permits_a_dead_scale_unsafe_read(r: SubTestResult):
    print("\n--- TRK-219: scale_verdict over a dead-branch ix/iy read ---")
    _drop_memos()
    pre = _without_fold_level_prune(SCALE_DEAD, {"k": 0.0})
    pre_unsafe = any(tex_roi._scale_unsafe_walk(stmt) for stmt in pre.statements)
    r.ok("RED-FIRST: the pre-TRK-219 shape still sees the dead ix/iy read as scale-unsafe") \
        if pre_unsafe else \
        r.fail("TRK-219 scale_verdict", "repro found no unsafe read pre-prune — not exercising it")
    got = tex_roi.scale_verdict(SCALE_DEAD, {"k": 0.0})
    r.ok(f"GREEN: scale_verdict(k=0.0).safe is now True (the dead ix/iy read is pruned away)") \
        if got.safe else \
        r.fail("TRK-219 scale_verdict", f"expected safe=True, got {got!r}")
    live = tex_roi.scale_verdict(SCALE_DEAD, {"k": 1.0})
    r.ok("the LIVE branch (k=1.0) still correctly refuses (a real ix/iy read)") \
        if not live.safe else \
        r.fail("TRK-219 scale_verdict", f"k=1.0 (ix/iy LIVE) must stay unsafe, got {live!r}")


def test_walk_region_dependent_permits_a_dead_loop(r: SubTestResult):
    print("\n--- TRK-219: region_dependent (via _walk) over a dead-branch while loop ---")
    _drop_memos()
    walked = tex_roi._walk(BATCH_REGDEP_DEAD, {"k": 0.0})
    r.ok("GREEN: _walk's region_dep is now False for the dead-branch loop") \
        if walked is not None and walked[4] is False else \
        r.fail("TRK-219 region_dependent", f"expected region_dep=False, got {walked!r}")
    live = tex_roi._walk(BATCH_REGDEP_DEAD, {"k": 1.0})
    r.ok("the LIVE branch (k=1.0) still correctly reports region_dep=True") \
        if live is not None and live[4] is True else \
        r.fail("TRK-219 region_dependent", f"k=1.0 must stay region_dep=True, got {live!r}")


# ── pixel identity: the widened verdict is not just DIFFERENT, it is CORRECT ─

def test_pixel_identity_batch_strips_over_a_dead_frame_op(r: SubTestResult):
    """`batch_sliceable` now says True for `BATCH_FRAME_DEAD` at k=0.0 — cook it in 3 batch
    strips (`run_batch_strips`) and require the stitched result to be `torch.equal` to a
    whole-batch cook. A wrongly-widened verdict would show up here as a strip boundary
    artifact (a strip-local `fi` reading garbage from the pruned-but-still-compiled dead
    branch) — this proves it does not, not just that the verdict moved."""
    print("\n--- TRK-219: pixel identity — batch strips over a dead frame op ---")
    _drop_memos()
    assert tex_roi.batch_sliceable(BATCH_FRAME_DEAD, {"k": 0.0}) is True
    _, whole = _cook(BATCH_FRAME_DEAD, shape=(6, 8, 8), params={"k": 0.0})
    _, strips = _cook(BATCH_FRAME_DEAD, shape=(6, 8, 8), params={"k": 0.0}, batch=3)
    ok = torch.equal(whole["OUT"], strips["OUT"])
    r.ok("3 batch-strips == whole-batch cook, bit-exact, for the widened program") if ok else \
        r.fail("TRK-219 pixel identity", f"batch strips diverged from whole-frame: "
                                         f"max diff {(whole['OUT'] - strips['OUT']).abs().max()}")


def test_pixel_identity_roi_window_over_a_dead_region_dependent_loop(r: SubTestResult):
    """`roi_plan` (via `_walk`) now reports this program executable (region_dep=False,
    halo=6 from the gauss_blur) at k=0.0 — cook a windowed sub-region (`run_roi`) directly and
    require it to be `torch.equal` to the same crop of a whole-frame cook."""
    print("\n--- TRK-219: pixel identity — ROI window over a dead region-dependent loop ---")
    _drop_memos()
    plan = tex_roi.roi_plan(ROI_REGDEP_DEAD, {"k": 0.0})
    r.ok(f"roi_plan is executable, halo={plan.halo}, narrow={sorted(plan.narrow)}") \
        if plan.executable and plan.halo > 0 else \
        r.fail("TRK-219 pixel identity", f"expected an executable, haloed plan, got {plan!r}")
    if not plan.executable:
        return
    shape = (1, 24, 24)
    roi = (6, 5, 8, 7, shape[2], shape[1])
    x0, y0, w, h, W, H = roi
    _, whole = _cook(ROI_REGDEP_DEAD, shape=shape, params={"k": 0.0})
    _, windowed = _cook(ROI_REGDEP_DEAD, shape=shape, params={"k": 0.0},
                        roi=roi, narrow=plan.narrow, halo=plan.halo)
    crop = whole["OUT"][:, y0:y0 + h, x0:x0 + w]
    ok = torch.equal(windowed["OUT"], crop)
    r.ok("the ROI window == the whole-frame crop, bit-exact") if ok else \
        r.fail("TRK-219 pixel identity", f"windowed cook diverged: max diff "
                                         f"{(windowed['OUT'] - crop).abs().max()}")


def test_pixel_identity_class1_out_write_target_is_inert(r: SubTestResult):
    """The oracle's CLASS 1 (`test_perf1_roi_walk_memo.py`): `@OUT` moving in or out of
    `fold_erased`/`narrow` cannot change any cook, because `run_roi` only narrows names that
    are actual BINDINGS — `@OUT` is a write target, never one. Proves it directly: cook the
    same windowed program once with `narrow={"A"}` (what TRK-219 now reports) and once with
    the pre-TRK-219 `narrow={"A", "OUT"}`, and require identical output."""
    print("\n--- TRK-219: pixel identity — class 1 (@OUT in/out of narrow) is inert ---")
    code = ("f$k = 0.0;\nif ($k > 0.5) {\n    @OUT = gauss_blur(@A, 2.0);\n} else {\n"
            "    @OUT = gauss_blur(@A, 2.0);\n}\n")
    shape = (1, 24, 24)
    roi = (6, 5, 8, 7, shape[2], shape[1])
    _, without_out = _cook(code, shape=shape, params={"k": 0.0}, roi=roi,
                           narrow=frozenset({"A"}), halo=6)
    _, with_out = _cook(code, shape=shape, params={"k": 0.0}, roi=roi,
                        narrow=frozenset({"A", "OUT"}), halo=6)
    ok = torch.equal(without_out["OUT"], with_out["OUT"])
    r.ok("narrow={'A'} and narrow={'A','OUT'} cook byte-identical — @OUT's presence is inert") \
        if ok else \
        r.fail("TRK-219 pixel identity", "narrowing (or not) the non-binding name 'OUT' "
                                         "changed the cook — class 1 is NOT inert after all")
