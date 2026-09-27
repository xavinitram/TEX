"""TRK-221 — `tex_engine_tiers.py::_run_default`'s plain success path never called
`tier_trace.record(...)`, only its own fallback branches did (the `torch_compile`/`auto`/
`cuda_graph` strategies each record on decline; the ROI branch records via `record_roi`; a
successful UC-2 stencil route records "codegen" inside `_codegen_only_execute`; a tiled/
halo-tiled cook records inside `run_tiled`/`run_tiled_halo`). The one route with NO record of
its own is the tail of `_run_default` itself: no ROI, no stencil route, no tile/halo pressure,
not a fused chain, no scale — i.e. the plain, ordinary default-tier cook. `tier_trace.last()`
after that cook is either `None` (if `tier_trace.reset()` ran freshly, as `tex_engine.prepare()`
does on every `cook()` call) or, for a caller who calls `tex_engine.run(plan)` directly on an
already-prepared plan (as a host reusing a plan, or `tex_engine.run()`'s own re-cook paths,
can) WITHOUT an intervening `prepare()`, the PRIOR run's leftover record — silently wrong,
never "no tier ran", exactly the failure mode `tier_trace` exists to make impossible
(`tier_trace.py`'s own module docstring: "a tier that *stops* engaging becomes a red test").

RED AT BASE (`7477a93`, confirmed by running): `tex_engine.run(plan1)` on a stencil-routed
program (`examples/blur.tex`) records `codegen`; `tex_engine.run(plan2)` on an ordinary
`@OUT = @A * 2.0;` program (a plain default-tier cook, no stencil/tile/halo route taken)
called directly afterwards (no `prepare()`/`tier_trace.reset()` between the two `run()`
calls — the exact shape a plan-reusing caller produces) still read back `codegen` — plan1's
tier, not plan2's own (plan2 never touched codegen at all).

FIX: `_run_default` now tracks whether a tiled/halo-tiled attempt was tried and declined
(`_default_fallback`/`_default_fallback_reason`), and calls
`tier_trace.record("interpreter", fallback_from=..., reason=...)` once, immediately before
its final `interp.execute(...)` — the one fall-through every earlier return in the function
already bypasses with its own record. `fallback_from` is `None` on the ordinary route (never
attempted a tile/halo strategy) and names which one was tried and declined otherwise.

ComfyUI-invisible because: this only writes to a diagnostics-only, thread-local trace read by
`tex doctor` / the DBG-1 HUD / tests — it changes no pixel, no cache key and no control flow
of any cook.
"""
from helpers import *

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import tier_trace

_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")

_PLAIN = "@OUT = @A * 2.0;"   # pointwise; no stencil route, no tile/halo pressure at this size


def _blur_source():
    with open(os.path.join(_EXAMPLES, "blur.tex"), encoding="utf-8") as f:
        return f.read()


def test_trk221_plain_default_cook_is_recorded_via_cook(r: SubTestResult):
    """The ordinary path most hosts use (`tex_engine.cook`, which calls `prepare()` and its
    own `tier_trace.reset()` every time): a plain default-tier cook must leave a record naming
    ITSELF (`tier="interpreter"`, `fallback_from=None`), not just leave the trace at `None`
    forever indistinguishable from "nothing ran"."""
    print("\n--- TRK-221: a plain default-tier cook records its own tier (via cook()) ---")
    try:
        img = make_img(1, 8, 8, 3, seed=221)
        tier_trace.reset()
        res = tex_engine.cook(_PLAIN, {"A": img}, device_mode="cpu", compile_mode="none")
        if "OUT" not in res.outputs:
            r.fail("TRK-221 plain cook", "cook() returned no OUT")
            return
        rec = tier_trace.last()
        if rec is None:
            r.fail("TRK-221 plain cook record", "tier_trace.last() is None after a "
                   "successful default-tier cook — the plain route still records nothing")
            return
        if rec.tier != "interpreter" or rec.fallback_from is not None:
            r.fail("TRK-221 plain cook record", f"got {rec!r}, expected "
                   "tier='interpreter', fallback_from=None")
            return
        r.ok(f"recorded: {rec!r}")
    except Exception as e:
        r.fail("TRK-221 plain cook", f"{type(e).__name__}: {e}")


def test_trk221_run_after_run_is_never_stale(r: SubTestResult):
    """The RED-AT-BASE repro: two prepared plans, `run()` called directly on each with no
    `prepare()`/`reset()` between them (the shape a plan-reusing caller — or a re-cook path
    that calls `run()` more than once — produces). The second, plain-default-tier plan's own
    record must not still name the first plan's tier."""
    print("\n--- TRK-221: run(plan2) right after run(plan1) is never stale ---")
    try:
        img = make_img(1, 8, 8, 3, seed=222)
        plan1 = tex_engine.prepare(_blur_source(), {"image": img.clone(), "radius": 2},
                                   device_mode="cpu", compile_mode="none")
        plan2 = tex_engine.prepare(_PLAIN, {"A": img.clone()},
                                   device_mode="cpu", compile_mode="none")

        out1 = tex_engine.run(plan1)
        rec1 = tier_trace.last()
        if rec1 is None or rec1.tier != "codegen":
            r.fail("TRK-221 setup", f"expected plan1 (blur.tex, a UC-2 stencil) to record "
                   f"codegen, got {rec1!r} — the repro's own premise did not hold")
            return

        out2 = tex_engine.run(plan2)   # deliberately no reset() in between
        rec2 = tier_trace.last()
        if "OUT" not in out2.outputs:
            r.fail("TRK-221 run(plan2)", "run() returned no OUT")
            return
        if rec2 is None:
            r.fail("TRK-221 stale record",
                   "tier_trace.last() is None after run(plan2) — still not recorded")
            return
        if rec2.tier == "codegen":
            r.fail("TRK-221 stale record",
                   f"tier_trace.last() still reads plan1's tier ({rec2!r}) after plan2's "
                   f"own (plain default, non-stencil) run — the stale-record bug")
            return
        if rec2.tier != "interpreter" or rec2.fallback_from is not None:
            r.fail("TRK-221 stale record", f"got {rec2!r}, expected "
                   "tier='interpreter', fallback_from=None for plan2's own cook")
            return
        r.ok(f"plan1 -> {rec1!r}, plan2 (no reset in between) -> {rec2!r}: not stale")
    except Exception as e:
        r.fail("TRK-221 run-after-run", f"{type(e).__name__}: {e}")


def test_trk221_tier141_ordinary_decline_still_unaffected(r: SubTestResult):
    """Control: TRK-141's `_interp_fallback` route (a DIFFERENT function, reached only from the
    torch_compile/auto/cuda_graph tier strategies, never from `_run_default`) must keep its own
    already-pinned "ordinary decline stays unrecorded" contract — this fix touches only
    `_run_default`'s own fall-through, not `_interp_fallback`."""
    print("\n--- TRK-221 control: _interp_fallback's ordinary-decline contract is untouched ---")
    bindings = {"A": make_img(1, 4, 4, 3, seed=223)}
    orig = tex_engine.execute_compiled

    def _raise_ordinary(*a, **kw):
        raise RuntimeError("unrelated compile failure")

    tex_engine.execute_compiled = _raise_ordinary
    try:
        tier_trace.reset()
        result = tex_engine.cook("@OUT = @A * 1.5;", dict(bindings), device_mode="cpu",
                                 compile_mode="torch_compile")
        rec = tier_trace.last()
        if result is None or "OUT" not in result.outputs:
            r.fail("TRK-221 control", "torch_compile still serves a result")
            return
        if rec is not None:
            r.fail("TRK-221 control", f"expected an ordinary decline to stay unrecorded "
                   f"(TRK-141's pinned contract), got {rec!r}")
            return
        r.ok("unaffected: still None, as TRK-141 pins")
    except Exception as e:
        r.fail("TRK-221 control", f"{type(e).__name__}: {e}")
    finally:
        tex_engine.execute_compiled = orig
