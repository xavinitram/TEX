"""FIX-ROI49 Q1 — TRK-221 covered the plain default-tier tail, but a SUCCESSFUL `run_tiled`/
`run_tiled_halo` cook never called `tier_trace.record(...)` at all: `tex_engine_tiers.py`'s
`_run_default` returns their result directly (`return run_tiled(...)` / `return
run_tiled_halo(...)`), so its own TRK-221 tail — the only place that function recorded a tier —
is reached only on a DECLINE (an exception) or when no tile/halo plan was ever taken, never on
this, the actual tiled-success path. `run_tiled_halo`'s per-strip `run_roi` calls pass
`record_trace=False` (deliberately, to keep a strip's own ROI rect off the trace), which
suppresses only `record_roi`'s window field, never a tier tag — and nothing else in either
function wrote one either. `tier_trace.last()` after a successful tiled/halo-tiled cook
therefore kept reading whatever tier a PRIOR, unrelated cook happened to leave behind — the
exact "silently wrong, never 'no tier ran'" failure class TRK-221's own module docstring says
`tier_trace` exists to make impossible, reopened for these two routes.

RED AT BASE (`32f6917`, confirmed by running): a direct call to `run_tiled`/`run_tiled_halo`
right after an unrelated `codegen`-tagged run, with no intervening `tier_trace.reset()` (the
same run()-after-run() shape `test_trk221_run_after_run_is_never_stale` uses for the plain
interpreter tail — `tex_engine.run(plan)` called directly on a prepared plan never resets the
trace; only `prepare()` does), leaves `tier_trace.last()` reading `codegen` — the unrelated
prior cook's tier, not `tiled`/`halo_tiled`.

FIX: `run_tiled` and `run_tiled_halo` each call `tier_trace.record("tiled")` /
`tier_trace.record("halo_tiled")` once, immediately before returning their assembled output —
mirroring TRK-221's own "record right before the return that would otherwise leave silently".

ComfyUI-invisible because: this only writes to a diagnostics-only, thread-local trace read by
`tex doctor` / the DBG-1 HUD / tests — it changes no pixel, no cache key and no control flow of
any cook. `run_tiled`/`run_tiled_halo` are cost-unaffected: one attribute write per cook, same
class TST-5's own docstring already prices for `tier_trace.record`'s other call sites.
"""
from helpers import *

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_memory import run_tiled, run_tiled_halo
from TEX_Wrangle.tex_runtime import tier_trace

_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")

_PLAIN = "@OUT = @A * 2.0;"                      # pointwise; tile-safe, no halo needed
_BLUR_BUILTIN = "@OUT = gauss_blur(@image, 2.0);"  # non-point footprint; not tile-safe


def _blur_source():
    with open(os.path.join(_EXAMPLES, "blur.tex"), encoding="utf-8") as f:
        return f.read()


def test_q1_run_tiled_records_its_own_tier(r: SubTestResult):
    """A real, successful `run_tiled` cook (forced `n_strips=2` — the exact bypass B3's own
    probe used, since the CUDA-only memory-pressure planner can't be reached on this box)
    must leave `tier_trace.last().tier == 'tiled'`, not `None` and not whatever ran before it."""
    print("\n--- FIX-ROI49 Q1: a successful run_tiled cook records tier='tiled' ---")
    try:
        img = make_img(1, 8, 8, 3, seed=491)
        plan = tex_engine.prepare(_PLAIN, {"A": img}, device_mode="cpu", compile_mode="none")
        ctx = plan.ctx
        interp = tex_engine._get_interpreter()
        tier_trace.reset()
        out = run_tiled(interp, ctx.program, ctx.bindings, ctx.type_map, ctx.device,
                        ctx.latent_channel_count, ctx.output_names, ctx.used_builtins,
                        ctx.eff_precision, n_strips=2, time_context=ctx.time_context)
        if "OUT" not in out:
            r.fail("Q1 run_tiled", "run_tiled returned no OUT")
            return
        rec = tier_trace.last()
        if rec is None or rec.tier != "tiled":
            r.fail("Q1 run_tiled record", f"got {rec!r}, expected tier='tiled'")
            return
        r.ok(f"recorded: {rec!r}")
    except Exception as e:
        r.fail("Q1 run_tiled", f"{type(e).__name__}: {e}")


def test_q1_run_tiled_halo_records_its_own_tier(r: SubTestResult):
    """Same claim for `run_tiled_halo` (forced `n_strips=2`, a real halo margin) — the
    `record_trace=False` each strip's `run_roi` call passes must not be mistaken for
    suppressing THIS function's own tier tag; the two are different thread-local fields
    (`_local.roi` vs `_local.last`, `tier_trace.py`'s own module docstring)."""
    print("\n--- FIX-ROI49 Q1: a successful run_tiled_halo cook records tier='halo_tiled' ---")
    try:
        img = make_img(1, 12, 12, 3, seed=492)
        plan = tex_engine.prepare(_BLUR_BUILTIN, {"image": img}, device_mode="cpu",
                                  compile_mode="none")
        ctx = plan.ctx
        interp = tex_engine._get_interpreter()
        tier_trace.reset()
        out = run_tiled_halo(interp, ctx.program, ctx.bindings, ctx.type_map, ctx.device,
                             ctx.latent_channel_count, ctx.output_names, ctx.used_builtins,
                             ctx.eff_precision, n_strips=2, narrow_names=frozenset({"image"}),
                             halo=6, time_context=ctx.time_context)
        if "OUT" not in out:
            r.fail("Q1 run_tiled_halo", "run_tiled_halo returned no OUT")
            return
        rec = tier_trace.last()
        if rec is None or rec.tier != "halo_tiled":
            r.fail("Q1 run_tiled_halo record", f"got {rec!r}, expected tier='halo_tiled'")
            return
        r.ok(f"recorded: {rec!r}")
    except Exception as e:
        r.fail("Q1 run_tiled_halo", f"{type(e).__name__}: {e}")


def test_q1_tiled_after_codegen_run_is_never_stale(r: SubTestResult):
    """B3's own run()-after-run() repro (B3-pacing.md finding 1), through the TILED path:
    `tex_engine.run(plan1)` on a stencil-routed program records `codegen`; a direct
    `run_tiled` call right afterwards, with NO `tier_trace.reset()` in between (the exact
    plan-reusing-caller shape `test_trk221_run_after_run_is_never_stale` already pins for the
    plain interpreter tail), must read its OWN tier, not plan1's leftover `codegen`."""
    print("\n--- FIX-ROI49 Q1: run_tiled right after a codegen run is never stale ---")
    try:
        img = make_img(1, 8, 8, 3, seed=493)
        plan1 = tex_engine.prepare(_blur_source(), {"image": img.clone(), "radius": 2},
                                   device_mode="cpu", compile_mode="none")
        plan2 = tex_engine.prepare(_PLAIN, {"A": img.clone()}, device_mode="cpu",
                                   compile_mode="none")

        tier_trace.reset()
        tex_engine.run(plan1)
        rec1 = tier_trace.last()
        if rec1 is None or rec1.tier != "codegen":
            r.fail("Q1 setup", f"expected plan1 (blur.tex, a UC-2 stencil) to record codegen, "
                   f"got {rec1!r} — the repro's own premise did not hold")
            return

        ctx2 = plan2.ctx
        interp = tex_engine._get_interpreter()
        out2 = run_tiled(interp, ctx2.program, ctx2.bindings, ctx2.type_map, ctx2.device,
                         ctx2.latent_channel_count, ctx2.output_names, ctx2.used_builtins,
                         ctx2.eff_precision, n_strips=2,
                         time_context=ctx2.time_context)   # deliberately no reset() in between
        rec2 = tier_trace.last()
        if "OUT" not in out2:
            r.fail("Q1 run_tiled(plan2)", "run_tiled returned no OUT")
            return
        if rec2 is None:
            r.fail("Q1 stale record",
                   "tier_trace.last() is None after run_tiled(plan2) — still not recorded")
            return
        if rec2.tier == "codegen":
            r.fail("Q1 stale record", f"tier_trace.last() still reads plan1's tier ({rec2!r}) "
                   f"after plan2's own tiled cook — the stale-record bug")
            return
        if rec2.tier != "tiled":
            r.fail("Q1 stale record", f"got {rec2!r}, expected tier='tiled' for plan2's own "
                   f"run_tiled call")
            return
        r.ok(f"plan1 -> {rec1!r}, run_tiled(plan2, no reset in between) -> {rec2!r}: not stale")
    except Exception as e:
        r.fail("Q1 run-after-run (tiled)", f"{type(e).__name__}: {e}")
