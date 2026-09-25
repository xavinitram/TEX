"""
SPLIT-E — `tex_engine_tiers`: tier selection + the tier execution strategies.

Mechanical move out of `tex_engine.py`, modelled on ENG-14/NEG-2: bodies verbatim, no
surviving line of `tex_engine.py` changed, and every name is re-exported there so
`tex_engine.NAME` still answers for all of them (the seam-freeze test pins it). This is
the STR-2 domain — pure tier SELECTION (`select_tier`) plus the four strategies that
actually run one (`_run_torch_compile`/`_run_auto`/`_run_cuda_graph`/`_run_default`), the
shared interpreter-fallback recovery path, and the `_run_tier` dispatcher `run()` and the
C2 finiteness net call. `tex_engine.py` still PREPARES and RUNS a cook (`prepare`/`run`/
`cook` stay there, unmoved and easy to find); this module only decides and executes ONE
tier once a plan already exists.

**ROUTE-45 — why every cross-module call here routes through `tex_engine.` instead of
importing directly.** Before this split every one of these functions lived inside
`tex_engine.py` itself, so ANY name it used — whether tex_engine physically defined it
(`_get_interpreter`/`_tile_plan`/`_halo_tile_plan`/`_scalar_params`, relocated to
`tex_chain`/`tex_tiling` by NEG-2/ENG-14 and re-exported back) or merely imported it at
module scope from elsewhere (`execute_compiled`/`_codegen_only_execute`/
`should_stencil_route` from `tex_runtime.compiled`, `CookCancelled`/`_cancel_check` from
`tex_runtime.host`) — resolved against `tex_engine`'s OWN module dict. A test or host that
monkeypatches `tex_engine.NAME` (several already do — `test_trk141_codegen_defect_
fallback.py` patches `tex_engine.execute_compiled` to prove `_run_torch_compile`'s codegen-
defect catch, and `tex_v022_phase1`/`test_v042_hostaudit4a` patch `tex_engine._run_tier`)
therefore always saw the patch, because the wrap and the call resolved against the exact
same namespace.

Importing any of those names straight from their true source module into THIS module
would silently drop that visibility for a caller now living here — exactly the ROUTE-45
audit's finding for `tex_chain.cook_fused_cached` calling `tex_engine.cook_stage_list` by
local name, generalised: it is not only a name
`tex_engine` used to physically define, it is any name a caller reached by resolving
through `tex_engine`'s namespace before the caller moved. `test_trk141` is exactly this
class of test and is what proved the general form matters here, not just the four names
ENG-14/NEG-2 relocated.

So this module holds a deferred reference to the `tex_engine` module object
(`_tex_engine`, bound once at import — safe under the circular import because nothing
calls through it until long after both modules have finished loading) and reads every one
of those names off it — `_tex_engine._get_interpreter()`, `_tex_engine.execute_compiled(
...)`, `_tex_engine.CookCancelled`, etc. — rather than importing them directly. Only names
that (a) are called through a FUNCTION-LOCAL import (`run_auto`, `run_graphed`, `tier_trace`,
`run_roi`/`run_tiled`/`run_tiled_halo`/`is_tile_safe_cached`, `_CgBreak`/`_CgContinue`) are
imported directly here, because a function-local `from X import Y` re-resolves against `X`'s
OWN namespace on every call regardless of which module the caller lives in — it never goes
through `tex_engine` even today, so moving the caller changes nothing a wrap could see.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("TEX")

# ROUTE-45: a deferred reference to `tex_engine`, NOT a direct `from .tex_chain import
# _get_interpreter` / `from .tex_runtime.compiled import execute_compiled` / etc. — see the
# module docstring. `tex_engine` is already in `sys.modules` (mid-import) the moment this
# module is first imported (from inside `tex_engine.py`'s own top-level import statement),
# so this binds the module object without re-running it; nothing here reads an attribute
# off it until a cook actually runs, long after both modules have finished loading.
from . import tex_engine as _tex_engine


# ── Tier selection + the tier strategies ─────────────────────────────────────

def _interp_fallback(ctx, *, reset_dynamo: bool, pass_precision: bool):
    """STR-2: the single copy of the tier→interpreter recovery path, previously
    duplicated in the torch_compile / auto / cuda_graph branches. The two flags
    reproduce each branch's *exact* original call:
      - torch_compile: reset_dynamo=True,  pass_precision=False
      - auto:          reset_dynamo=True,  pass_precision=True
      - cuda_graph:    reset_dynamo=False, pass_precision=False
    (dynamo state is process-global — see compiled.py — so a failed compile/auto
    cook resets it on THIS thread before retrying; the graph path never touched
    dynamo, so it does not reset.)"""
    _tex_engine._cancel_check(ctx.cancel)   # SCHED-3 yield C: a tier failed — don't fall
                                            # back into a stale cook
    if reset_dynamo:
        try:
            torch._dynamo.reset()
        except Exception:
            pass
    interp = _tex_engine._get_interpreter()
    # REG-1d: `ctx.code` is the TERMINAL stage's own source only -- on a fused chain
    # (`ctx.fused_chain`) the executed `ctx.program` is the multi-stage SPLICE, and an
    # upstream (non-terminal) stage's call is invisible to `ctx.code`. `_consensus_extent`
    # trusts `source` to name every call the executed Program can make (its non-spatial-
    # exclusion fast path), so a partial source here is not a cosmetic risk the way it is
    # for error-line rendering -- it can wrongly skip the walk. Blank it exactly like the
    # OTHER interpreter call site below (`source=("" if ctx.fused_chain else ctx.code)`),
    # so every route to the interpreter agrees on what "unknown" means.
    kw = dict(source=("" if ctx.fused_chain else ctx.code),
              latent_channel_count=ctx.latent_channel_count,
              output_names=ctx.output_names, used_builtins=ctx.used_builtins,
              time_context=ctx.time_context,
              cancel=ctx.cancel, on_progress=ctx.on_progress)  # SCHED-3: token survives the fallback
    if pass_precision:
        kw["precision"] = ctx.eff_precision
    return interp.execute(ctx.program, ctx.bindings, ctx.type_map,
                          device=ctx.device, **kw)


def _record_codegen_defect_fallback(tier: str, exc: Exception) -> None:
    """TRK-141: a bare `_CgBreak`/`_CgContinue` (codegen's internal control-flow
    signal, meant to be consumed by the loop emitter that raises it) escaping all
    the way out to a tier strategy's own `except Exception` is a CODEGEN DEFECT,
    not an ordinary compile decline — every other reason this catch fires (a
    missing backend, an unsupported construct, a genuine runtime error in the
    generated code) is a legitimate reason to fall back quietly. Before this fix
    the fallback was recorded nowhere: `tier_trace.record` is never called on this
    path, so `tier_trace.last()` still read the PRIOR cook's record (or None,
    right after `prepare()`'s `tier_trace.reset()`) — indistinguishable from "no
    fallback happened", and the only log line was a `logger.warning` with the
    bare `str(exc)`, which is empty for these two classes. This does not touch the
    warning already logged by the caller for the ordinary case; it only adds an
    ERROR-level line naming the class and a tier_trace record for THIS class."""
    from .tex_runtime.codegen import _CgBreak, _CgContinue
    if not isinstance(exc, (_CgBreak, _CgContinue)):
        return
    from .tex_runtime import tier_trace
    reason = f"codegen defect: {type(exc).__name__} escaped generated code"
    logger.error("[TEX] %s tier fell back to interpreter on a codegen defect (%s).",
                 tier, reason)
    tier_trace.record("interpreter", fallback_from=tier, reason=reason)


def select_tier(compile_mode, device, fused_chain: bool, fused_fp_present: bool) -> str:
    """STR-2: PURE tier SELECTION — which acceleration strategy `(mode, device,
    fused)` picks, WITHOUT executing it. The branch ORDER and every guard mirror
    the old cascade verbatim; this is the CPU-testable core where the routing
    complexity lives (a fake `device="cuda:0"` string exercises the cuda_graph
    classification without a GPU)."""
    # v0.20: fused chains may take the compile tiers too — keyed by fused_fp
    # (same pattern cuda_graph used since v0.17). Measured on a fused-chain-
    # shaped program (sm_120 + Triton): inductor 2.63x vs interpreter at
    # 1024²; on toolchain-less boxes the tiers self-fall-back (and `auto`
    # measures-then-rejects), so enabling them is never a regression.
    if compile_mode == "torch_compile" and (not fused_chain or fused_fp_present):
        return "torch_compile"
    if compile_mode == "auto" and (not fused_chain or fused_fp_present):
        return "auto"
    if (compile_mode == "cuda_graph" and str(device).startswith("cuda")
            and (not fused_chain or fused_fp_present)):
        return "cuda_graph"
    return "default"


# ── STR-2 tier strategies: each runs one tier and returns raw_output; the
#    dict-normalization is lifted to _run_tier post-dispatch. ──

def _run_torch_compile(ctx):
    # Fused chains are keyed by their chain fingerprint (ctx.fp is None there).
    _fp = ctx.fused_fp if ctx.fused_chain else ctx.fp
    try:
        return _tex_engine.execute_compiled(
            ctx.program, ctx.bindings, ctx.type_map, ctx.device,
            _fp, latent_channel_count=ctx.latent_channel_count,
            output_names=ctx.output_names, used_builtins=ctx.used_builtins,
            time_context=ctx.time_context)
    except Exception as compile_exc:
        _record_codegen_defect_fallback("torch_compile", compile_exc)
        # Defense in depth: torch_compile must NEVER hard-fail the node.
        logger.warning("[TEX] torch_compile path failed (%s); using interpreter.",
                       compile_exc)
        return _interp_fallback(ctx, reset_dynamo=True, pass_precision=False)


def _run_auto(ctx):
    # Fused chains are keyed by their chain fingerprint (ctx.fp is None there).
    _fp = ctx.fused_fp if ctx.fused_chain else ctx.fp
    try:
        from .tex_runtime.compiled import run_auto
        return run_auto(ctx.program, ctx.bindings, ctx.type_map, ctx.device, _fp,
                        latent_channel_count=ctx.latent_channel_count,
                        output_names=ctx.output_names, used_builtins=ctx.used_builtins,
                        precision=ctx.eff_precision, time_context=ctx.time_context)
    except Exception as auto_exc:
        _record_codegen_defect_fallback("auto", auto_exc)
        logger.warning("[TEX] auto tier failed (%s); using interpreter.", auto_exc)
        return _interp_fallback(ctx, reset_dynamo=True, pass_precision=True)


def _run_cuda_graph(ctx):
    # UC-1: CUDA-graph replay. A fused chain is captured as ONE graph keyed by its
    # fused fingerprint. run_graphed returns None (→ interpreter) when the program
    # isn't graphable or capture failed — never hard-fails on this path.
    from .tex_runtime.graphed import run_graphed
    _fp = ctx.fused_fp if ctx.fused_chain else ctx.fp
    out = None
    try:
        out = run_graphed(ctx.program, ctx.bindings, ctx.type_map, ctx.device, _fp,
                          latent_channel_count=ctx.latent_channel_count,
                          output_names=ctx.output_names, used_builtins=ctx.used_builtins)
    except Exception as _g_exc:
        _record_codegen_defect_fallback("cuda_graph", _g_exc)
        logger.warning("[TEX] cuda_graph path failed (%s); using interpreter.", _g_exc)
        out = None
    if out is None:
        return _interp_fallback(ctx, reset_dynamo=False, pass_precision=False)
    return out


def _roi_codegen_exec(fp):
    """An `exec_fn` for `tex_memory.run_roi` that serves the narrowed cook from the CODEGEN
    tier instead of the tree-walking interpreter (v0.30 — the v0.27 deferral's reopen
    condition, "thread the strip offset through the coordinate-env builder").

    Safe because the pieces that make ROI correct live OUTSIDE the executor: `run_roi` narrows
    the bindings and crops the result, and `_build_codegen_env(roi=...)` now offsets the
    coordinate builtins exactly as `Interpreter._create_builtins` does — so codegen sees a
    correct grid for the window. `_codegen_only_execute` self-falls-back to the interpreter
    (forwarding `roi`) if the program is unsupported or emission fails, so this can only be a
    speed choice, never a correctness one. Signature matches `Interpreter.execute`'s keywords;
    `cancel`/`on_progress` are accepted and dropped (the compiled tiers have no yield points —
    a cancel is honoured between region cooks, not inside one)."""
    def _exec(program, bindings, type_map, *, device, latent_channel_count, output_names,
              used_builtins, precision, roi, time_context, cancel=None, on_progress=None):
        return _tex_engine._codegen_only_execute(
            program, bindings, type_map, device, latent_channel_count, output_names,
            used_builtins=used_builtins, precision=precision, fingerprint=fp,
            time_context=time_context, roi=roi)
    return _exec


def _roi_codegen_enabled() -> bool:
    """Whether an ROI cook routes through codegen. v0.30 ships this OFF by default: measured on
    this box the codegen tier is not faster than the interpreter for the small windows an ROI
    cook produces (both are launch-bound there), so the default keeps the tier the ROI-4 oracle
    validates. `TEX_ROI_CODEGEN=1` turns it on — the A/B switch the reopen condition asks for,
    and the lane a faster box (or a larger window) can re-measure without a code change."""
    return os.environ.get("TEX_ROI_CODEGEN", "0") == "1"


def _run_default(ctx):
    # ROI-3: cook only the requested sub-window (interpreter tier, flagged off — set only
    # when a host passed `roi=`, the program is ROI-executable, and TEX_ROI_EXEC is on). The
    # narrow-cook-crop is bit-exact for pointwise/morphology, ~1 ulp for conv — and on CPU also
    # ~1 ulp for NOISE, whose kernels are shape-dependent at the last ulp, which a noise
    # derivative (`curl`) then amplifies by its 1/(2*eps) factor to ~3e-5 (measured; CUDA is
    # exact for every class WITHIN a settled noise-cache tier — the one-time jit-trace->
    # Inductor promotion, `tex_runtime.noise._TieredCache.try_upgrade`, is bounded by its own
    # pinned envelope instead, see `tests/test_v031_noise_tiers.py`). See the table in
    # CHANGELOG 0.30.0. Whole-frame on any run_roi error (never hard-fail the cook).
    if ctx.roi is not None and ctx.roi_plan is not None:
        # Bind tier_trace OUTSIDE the try: it is imported function-locally per the SCC
        # convention (see the F1 note below), and an import inside the try would leave the
        # `except` branch's own record_roi raising NameError instead of reporting the failure.
        from .tex_runtime import tier_trace
        try:
            from .tex_memory import run_roi
            return run_roi(_tex_engine._get_interpreter(), ctx.program, ctx.bindings, ctx.type_map,
                           ctx.device, ctx.latent_channel_count, ctx.output_names,
                           ctx.used_builtins, ctx.eff_precision, ctx.roi,
                           ctx.roi_plan.narrow, ctx.roi_plan.halo, ctx.time_context,
                           cancel=ctx.cancel, on_progress=ctx.on_progress,
                           exec_fn=(_roi_codegen_exec(ctx.fp) if _roi_codegen_enabled() else None))
        except _tex_engine.CookCancelled:
            raise                       # SCHED-3: a cancel aborts — never fall back to whole-frame
        except Exception as _roi_exc:
            # A DETERMINISTIC failure here re-cooks whole-frame on every frame of a pan (a
            # wasted partial + a full cook + a log line), so make the reason visible. Format
            # once — this path is per-frame by construction.
            _roi_why = f"roi cook failed: {_roi_exc}"
            tier_trace.record_roi(None, _roi_why)
            logger.warning("[TEX] %s; running whole-frame.", _roi_why)
    # UC-2: default-route an exact (fetch/conv) stencil through the codegen tier
    # (avg_pool2d/conv2d/unfold). _codegen_only_execute self-falls-back; the outer
    # guard covers env-build edge cases so this can never hard-fail the node.
    if not ctx.fused_chain:
        try:
            if _tex_engine._should_stencil_route(ctx.fp, ctx.program):
                return _tex_engine._codegen_only_execute(
                    ctx.program, ctx.bindings, ctx.type_map, ctx.device,
                    latent_channel_count=ctx.latent_channel_count,
                    output_names=ctx.output_names,
                    used_builtins=ctx.used_builtins, fingerprint=ctx.fp,
                    time_context=ctx.time_context,
                    cancel=ctx.cancel)  # CANCEL-44 (Gap 2): this route had no yield point
        except _tex_engine.CookCancelled:
            raise                       # SCHED-3: a cancel aborts — never fall back to interp
        except Exception as _stencil_exc:
            logger.warning("[TEX] stencil codegen route failed (%s); using "
                           "interpreter.", _stencil_exc)
    interp = _tex_engine._get_interpreter()
    # M-4: under GPU memory pressure, run a tile-safe program in horizontal strips
    # (peak transient ~1/n). Falls back to the whole-image cook on any strip error.
    n_strips = (_tex_engine._tile_plan(ctx.program, ctx.bindings, ctx.device, ctx.latent_channel_count,
                           2 if ctx.eff_precision == "fp16" else 4, ctx.fp,
                           free_hint=ctx.free_hint, code=ctx.code, binding_types=ctx.binding_types)
                if not ctx.fused_chain else None)
    if n_strips:
        try:
            from .tex_memory import run_tiled
            return run_tiled(interp, ctx.program, ctx.bindings, ctx.type_map, ctx.device,
                             ctx.latent_channel_count, ctx.output_names, ctx.used_builtins,
                             ctx.eff_precision, n_strips, ctx.time_context,
                             cancel=ctx.cancel, on_progress=ctx.on_progress)
        except _tex_engine.CookCancelled:
            raise                       # SCHED-3: a cancel aborts — never fall back to untiled
        except Exception as _tile_exc:
            logger.warning("[TEX] tiled cook failed (%s); running untiled.", _tile_exc)
    elif not ctx.fused_chain:
        # ROI-5: `_tile_plan` refused (a non-pixel-local program — a blur/morphology), but a
        # BOUNDED-halo op can still tile with a grown strip. Under memory pressure OR the TDR
        # time cap, cook it in halo strips (an 8K gauss_blur that could not tile at all before).
        # `_halo_tile_plan` cheap-gates so a small default cook returns before any real work.
        #
        # TRK-83: `n_strips` above is also falsy on a program `is_tile_safe_cached` — the same
        # memo `_halo_tile_plan` itself would consult FIRST — already answered True on: no
        # pressure, not a halo case. `_tile_plan` just warmed that exact fingerprint's entry
        # (it is the first thing it checks), so this is a memo hit, not a second AST walk, and
        # it skips a call guaranteed to no-op on every unpressured tile-safe stage — the "10
        # calls/cook" residue named in `docs/host-path-counts.md` §6 item 6 / TRK-83.
        from .tex_memory import is_tile_safe_cached
        halo_plan = (None if is_tile_safe_cached(ctx.program, ctx.fp) else
                    _tex_engine._halo_tile_plan(ctx.program, ctx.code, ctx.bindings, ctx.device,
                                    ctx.latent_channel_count,
                                    2 if ctx.eff_precision == "fp16" else 4, ctx.fp,
                                    ctx.free_hint, ctx.eff_precision, ctx.binding_types))
        if halo_plan:
            n_h, narrow_names, halo = halo_plan
            try:
                from .tex_memory import run_tiled_halo
                return run_tiled_halo(interp, ctx.program, ctx.bindings, ctx.type_map, ctx.device,
                                      ctx.latent_channel_count, ctx.output_names, ctx.used_builtins,
                                      ctx.eff_precision, n_h, narrow_names, halo, ctx.time_context,
                                      cancel=ctx.cancel, on_progress=ctx.on_progress)
            except _tex_engine.CookCancelled:
                raise                   # SCHED-3: a cancel aborts — never fall back to untiled
            except Exception as _halo_exc:
                logger.warning("[TEX] halo-tiled cook failed (%s); running untiled.", _halo_exc)
    # Pass source so runtime (E6xxx) errors render a source-line caret. Fused chains
    # splice many sources, so leave source empty there (errors stay message-only).
    return interp.execute(ctx.program, ctx.bindings, ctx.type_map, device=ctx.device,
                          source=("" if ctx.fused_chain else ctx.code),
                          latent_channel_count=ctx.latent_channel_count,
                          output_names=ctx.output_names, used_builtins=ctx.used_builtins,
                          precision=ctx.eff_precision, time_context=ctx.time_context,
                          cancel=ctx.cancel, on_progress=ctx.on_progress)


# tier_id → strategy function (module-level; the dict holds the fn objects directly —
# the classmethod era used names + getattr to dodge the callable-in-dict binding trap,
# which plain functions do not have).
_TIER_METHOD = {
    "torch_compile": _run_torch_compile, "auto": _run_auto,
    "cuda_graph": _run_cuda_graph, "default": _run_default,
}


def _run_tier(ctx, tier_id):
    """Dispatch a cook to the selected tier strategy and normalize its result to an
    output dict. Single home for the `tier method -> {name: tensor}` idiom used by
    both run() and the C2 re-cook path (reuse review)."""
    out = _TIER_METHOD[tier_id](ctx)
    return out if isinstance(out, dict) else {ctx.output_names[0]: out}
