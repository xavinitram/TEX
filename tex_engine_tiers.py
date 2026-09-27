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
from dataclasses import dataclass

import torch

logger = logging.getLogger("TEX")

# ROUTE-45: a deferred reference to `tex_engine`, NOT a direct `from .tex_chain import
# _get_interpreter` / `from .tex_runtime.compiled import execute_compiled` / etc. — see the
# module docstring. `tex_engine` is already in `sys.modules` (mid-import) the moment this
# module is first imported (from inside `tex_engine.py`'s own top-level import statement),
# so binding the module object costs nothing extra WHEN `tex_engine` is imported first — but
# nothing here reads an attribute off it until a cook actually runs, long after both modules
# have finished loading, in EITHER import order.
#
# R1 (v0.46, FIX-OBSROUTE): that "either order" half used to be false. Importing
# `tex_engine_tiers` FIRST, in a fresh process, crashed with a circular ImportError: the
# eager `from . import tex_engine as _tex_engine` below started `tex_engine.py`'s own body
# running (it was not yet in `sys.modules`), which reaches `from .tex_engine_tiers import
# (select_tier, ...)` (the SPLIT-E re-export) while THIS module's body is still stuck on its
# own import line above — `tex_engine_tiers` is in `sys.modules` by then (added the moment
# Python started running it) but none of its functions have been defined yet, so that
# re-export raises ImportError. Binding eagerly needed `tex_engine`'s own import to finish;
# `tex_engine.py` needed `tex_engine_tiers`'s import to have already finished — a genuine
# cycle, not merely a name resolved too soon.
#
# The fix resolves `_tex_engine` LAZILY instead: nothing below imports `tex_engine` at
# module scope, so importing `tex_engine_tiers` first no longer pulls it in at all. The
# proxy's `__getattr__` performs the import on the first attribute read any `_run_*`
# strategy makes — long after both modules have finished loading, whichever one was
# imported first — and caches the result so every later attribute read is one dict lookup
# plus one `getattr`, not a fresh import.
class _LazyTexEngine:
    """A stand-in for the `tex_engine` module object, resolved on first ATTRIBUTE access
    rather than at import time. `_tex_engine.NAME` below reads exactly as it did when this
    was an eager `from . import tex_engine as _tex_engine` — this class exists only to move
    the *timing* of the import, not to change what any caller sees."""
    __slots__ = ("_mod",)

    def __init__(self):
        self._mod = None

    def __getattr__(self, name):
        mod = self._mod
        if mod is None:
            from . import tex_engine as mod
            self._mod = mod
        return getattr(mod, name)


_tex_engine = _LazyTexEngine()


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
    both run() and the C2 re-cook path (reuse review).

    A scale-active cook (`ctx.scale is not None`) never reaches `torch_compile`/`auto`/
    `cuda_graph` — those tiers do not thread a runtime scale multiplier through their
    COMPILED/CAPTURED code at all (SCALE-COMPILED-48 is a v0.49+ item;
    `docs/resolution-scale.md`), so a scale-active cook whose `tier_id` names one of them
    is forced onto the plain interpreter instead.

    A scale-active cook whose `tier_id == "default"` is NOT forced to the interpreter
    unconditionally: it routes to codegen instead whenever the "default" tier's own
    internal codegen shortcut would already fire (`_should_stencil_route`, the UC-2
    exact-fetch stencil gate) — codegen's `pixel_args=`-tagged call sites emit
    `arg * _env['__tex_scale']` as a runtime value, never a folded literal
    (`tex_runtime/codegen.py`), so that route is scale-safe. `_should_stencil_route` is
    the ONLY concrete "would codegen run here" test the default tier has today
    (ROI-codegen is a separate, still-flagged-off lane, deliberately untouched here).
    `_codegen_only_execute` self-falls-back to the interpreter (forwarding `scale`) on
    any decline/failure, so this can never hard-fail or silently drop scale. Tiling and
    ROI narrowing stay out of scope for a scale-active cook: it always cooks whole-frame
    (this branch never threads `ctx.roi`).

    Recorded via `tier_trace` on every path (`_codegen_only_execute` records "codegen" on
    success, "interpreter" fallback_from="codegen" on decline; the plain interpreter
    branch below records "interpreter" fallback_from=tier_id) — never a silent fallback.
    `ctx.scale is None` (every ComfyUI cook) never reaches this function's body at all —
    one `is not None` check, no behaviour change on the default (`scale=None`) path."""
    if ctx.scale is not None:
        from .tex_runtime import tier_trace
        if (tier_id == "default" and not ctx.fused_chain
                and _tex_engine._should_stencil_route(ctx.fp, ctx.program)):
            out = _tex_engine._codegen_only_execute(
                ctx.program, ctx.bindings, ctx.type_map, ctx.device,
                latent_channel_count=ctx.latent_channel_count,
                output_names=ctx.output_names, used_builtins=ctx.used_builtins,
                precision=ctx.eff_precision, fingerprint=ctx.fp,
                time_context=ctx.time_context, cancel=ctx.cancel, scale=ctx.scale)
            return out if isinstance(out, dict) else {ctx.output_names[0]: out}
        tier_trace.record("interpreter", fallback_from=tier_id,
                          reason="scale is active (SCALE-47b runs on the interpreter tier only)")
        interp = _tex_engine._get_interpreter()
        out = interp.execute(ctx.program, ctx.bindings, ctx.type_map, device=ctx.device,
                             source=("" if ctx.fused_chain else ctx.code),
                             latent_channel_count=ctx.latent_channel_count,
                             output_names=ctx.output_names, used_builtins=ctx.used_builtins,
                             precision=ctx.eff_precision, time_context=ctx.time_context,
                             cancel=ctx.cancel, on_progress=ctx.on_progress, scale=ctx.scale)
        return out if isinstance(out, dict) else {ctx.output_names[0]: out}
    out = _TIER_METHOD[tier_id](ctx)
    return out if isinstance(out, dict) else {ctx.output_names[0]: out}


# ── TIERQ-48: the declared-fallback query ────────────────────────────────────
#
# A public, side-effect-free query answering "which tier will a cook of THIS shape
# run on, and why" — before a host cooks anything. It exists because scale/ROI are
# SILENTLY interpreter-only past two separate choke points (`_run_tier`'s unconditional
# `ctx.scale is not None` bypass above; `tex_engine.prepare`'s `roi is not None and
# tier_id == "default"` gate), so a host previously had no way to learn that fact except
# by timing a cook and noticing it was slow.
#
# Read-only over tier selection: this calls `select_tier` (this module, unchanged) and
# the existing `tex_roi` predicates (`scale_safe`/`roi_exec_enabled`/`validate_roi`/
# `canonical_roi`/`roi_plan`) in the SAME order `tex_engine.prepare()`'s own gate already
# does, rather than re-deriving a parallel judgment — so the two answers can never
# disagree by construction. `tests/test_tierq48_agreement.py` pins this against the real
# cook path's own `CookPlan` for a matrix of inputs.
#
# STABLE REASON CODES — part of the public contract: a host may branch on these exact
# strings, and they do not change shape across a release without a CHANGELOG entry.
TIER_REASON_SCALE_UNSAFE = "scale-unsafe-refused"   # the cook itself would raise, not run
TIER_REASON_SCALE_ACTIVE = "scale-active"           # non-None scale forces "interpreter"
TIER_REASON_SCALE_ACTIVE_CODEGEN = "scale-active-codegen-stencil"
# ^ the "default" tier's own UC-2 exact-fetch-stencil shortcut fired for THIS program.
# Named for the shortcut, not for "codegen now supports scale" in general: a plain
# gauss_blur/erode/dilate/bilateral_filter call with no independent hand-written stencil
# loop still reports TIER_REASON_SCALE_ACTIVE (interpreter) — see docs/resolution-scale.md.
TIER_REASON_SELECTED = "tier-selected"              # plain select_tier verdict, scale inactive

ROI_REASON_TIER_NOT_DEFAULT = "roi-declined-tier-not-default"
ROI_REASON_FUSED_CHAIN = "roi-declined-fused-chain"
ROI_REASON_LATENT = "roi-declined-latent-input"
ROI_REASON_SCALE_ACTIVE = "roi-declined-scale-active"
ROI_REASON_NOT_ARMED = "roi-declined-not-armed"
ROI_REASON_MALFORMED = "roi-declined-malformed"
ROI_REASON_WHOLE_FRAME = "roi-declined-whole-frame"
ROI_REASON_NOT_EXECUTABLE = "roi-declined-not-executable"
ROI_REASON_PRECISION = "roi-declined-precision-not-fp32"
ROI_REASON_ARMED = "roi-armed"


@dataclass(frozen=True)
class TierVerdict:
    """TIERQ-48's answer. `tier` is one of the six strings the real dispatch can
    actually produce — `"torch_compile"` / `"auto"` / `"cuda_graph"` / `"default"` /
    `"interpreter"` / `"codegen"` — or `None` when the cook itself would REFUSE
    (`reason == TIER_REASON_SCALE_UNSAFE`): never a guess at what an exception-raising
    cook "would have" run on. `"codegen"` (`reason == TIER_REASON_SCALE_ACTIVE_CODEGEN`)
    means a scale-active cook on the `"default"` tier whose PROGRAM independently
    contains the UC-2 exact-fetch stencil shape (`_should_stencil_route`) — this is
    narrower than "the program calls a `pixel_args=` builtin": a plain
    `gauss_blur`/`erode`/`dilate`/`bilateral_filter` call with no such stencil loop still
    reports `"interpreter"` (`TIER_REASON_SCALE_ACTIVE`); see `docs/resolution-scale.md`.
    `roi_armed` is a SEPARATE question from `tier`: an otherwise-eligible
    compiled/graphed tier still runs an ROI-requesting cook whole-frame (see
    `ROI_REASON_TIER_NOT_DEFAULT`), and the `"codegen"` scale route never threads ROI
    either (`ROI_REASON_SCALE_ACTIVE`) — `tier` names what executes the cook, `roi_armed`
    names whether IT narrows to the window."""
    tier: str | None
    reason: str
    roi_armed: bool
    roi_reason: str | None


def _stencil_route_would_apply(code: str, binding_types: dict | None) -> bool | None:
    """SCALE-CG-48 support: best-effort, read-only check of whether a scale-active,
    `tier_id == "default"` cook would route to codegen (the UC-2 stencil gate,
    `_should_stencil_route`) -- the one fact `tier_verdict` needs that a raw source
    string alone cannot answer, because `_should_stencil_route` consults the COMPILED
    program, not the text. Returns `True`/`False` when it could compile `code` (against
    `binding_types`) and check, or `None` ("unknown, answer conservatively") on any
    failure -- most commonly `binding_types` being `None` or incomplete, exactly the
    same "supply it for a precise answer" contract `roi_plan`'s `binding_types`
    parameter already documents above.

    FIX-TIER T2: this reads the SAME program cache a real cook would populate, but never
    WRITES to it -- a cache HIT is served normally (`TEXCache.get`, no store), and a MISS
    compiles through the same front end (`parse_and_split` + `TEXCache.compile_ast`)
    `compile_tex` itself uses, WITHOUT calling `.put()`. `tier_verdict`'s own docstring
    promises "no compile, no cache write, no cache pollution" (matching `scale_verdict`'s
    genuine side-effect-freedom); going through `_compile_or_raise`/`compile_tex` broke
    that promise for the one branch that reaches this function, because a cache miss
    there unconditionally persists the result (memory AND disk, `TEXCache.put` ->
    `_save_to_disk`) -- exactly the "compiles and populates the ordinary program cache"
    behavior B2#1 caught by running it against a fresh `TEX_CACHE_DIR`. A speculative,
    pre-cook query must not seed a `.pkl` (or a fresh in-memory slot) for a program nobody
    has actually cooked."""
    try:
        from .tex_cache import get_cache, parse_and_split
        bt = binding_types or {}
        cache = get_cache()
        fp = cache.fingerprint(code, bt)
        cached = cache.get(code, bt, fp=fp)
        if cached is not None:
            ast = cached[0]
        else:
            program = parse_and_split(code, bt)
            ast = cache.compile_ast(program, bt, source=code)[0]
        return bool(_tex_engine._should_stencil_route(fp, ast))
    except Exception:
        return None


def tier_verdict(code: str, *, compile_mode: str = "none", device: str = "cpu",
                 precision: str | None = None, roi: tuple | None = None,
                 scale: float | None = None, param_values: dict | None = None,
                 binding_types: dict | None = None, fused_chain: bool = False,
                 fused_fp_present: bool = False, has_latent_input: bool = False,
                 roi_exec: bool | None = None) -> TierVerdict:
    """TIERQ-48: which tier a cook of `code` at `(compile_mode, device, roi, scale)`
    WILL run on, and why — computed by
    calling the exact same read-only predicates `tex_engine.prepare()` calls, in the
    same order, rather than re-deriving a parallel judgment. Side-effect-free: no
    compile, no cache write, no cook, no cache pollution.

    `precision` is the cook's EFFECTIVE precision (`"fp32"` / `"fp16"`; `None` means
    `"fp32"`, matching `prepare()`'s own default) — exactly what `prepare()` has already
    resolved by the time its own ROI gate runs. This function does NOT resolve
    `precision="auto"` itself: that resolution needs a real cook's bindings/resolution to
    size the pixel-count gate (`tex_runtime.precision_policy.resolve_auto_precision`),
    which a pre-cook, bindings-free query does not have in general. A caller predicting an
    `"auto"` cook resolves it first, exactly as `prepare()` does before this same gate,
    and passes the resolved string here.

    `roi`, when given, is the 6-tuple `(x0, y0, w, h, full_w, full_h)` `tex_engine.cook`
    takes. `param_values`/`binding_types` feed `tex_roi.roi_plan`'s reach analysis exactly
    as a real cook's `_scalar_params(bindings)`/`binding_types` would; both default to
    "not supplied", the conservative reading `roi_plan` itself documents.

    `binding_types` also decides how precisely this function can answer for a
    scale-active cook on the `"default"` tier (SCALE-CG-48): that cook routes to
    `"codegen"` instead of `"interpreter"` when the UC-2 stencil gate would already
    choose codegen, and answering that precisely means compiling `code` against
    `binding_types` (the same compile a real cook performs; no new cache, no
    execution). Without `binding_types` (or with one that cannot compile `code`), this
    conservatively reports `"interpreter"` — never wrong about there being NO wrong
    pixel risk (that gate only picks a FASTER tier for the same bytes), only
    potentially pessimistic about which tier is named.

    Never raises: a malformed `roi` is reported as a declined reason
    (`ROI_REASON_MALFORMED`), not a `TypeError`/`ValueError` — the same "over-approximate,
    never blow up" posture every other `tex_roi` query in this module already takes.
    """
    from . import tex_roi as _tex_roi

    if scale is not None and scale != 1.0 and not _tex_roi.scale_safe(code, param_values):
        return TierVerdict(None, TIER_REASON_SCALE_UNSAFE, False, None)

    eff_precision = "fp32" if precision is None else precision
    tier_id = select_tier(compile_mode, device, fused_chain, fused_fp_present)

    roi_armed = False
    roi_reason = None
    if roi is not None:
        # Mirrors `tex_engine.prepare()`'s ROI gate verbatim, in the same order: the
        # OUTER eligibility (tier/fused/latent) first, then scale, then the per-window
        # checks `run_roi`'s own cheap validation performs ahead of the expensive ones.
        if tier_id != "default":
            roi_reason = ROI_REASON_TIER_NOT_DEFAULT
        elif fused_chain:
            roi_reason = ROI_REASON_FUSED_CHAIN
        elif has_latent_input:
            roi_reason = ROI_REASON_LATENT
        elif scale is not None:
            roi_reason = ROI_REASON_SCALE_ACTIVE
        elif not _tex_roi.roi_exec_enabled(roi_exec):
            roi_reason = ROI_REASON_NOT_ARMED
        elif _tex_roi.validate_roi(roi) is not None:
            roi_reason = ROI_REASON_MALFORMED
        else:
            _canon = _tex_roi.canonical_roi(roi)
            if _canon[2:4] == _canon[4:6]:
                roi_reason = ROI_REASON_WHOLE_FRAME
            elif not _tex_roi.roi_plan(code, param_values, binding_types).executable:
                roi_reason = ROI_REASON_NOT_EXECUTABLE
            elif eff_precision != "fp32":
                roi_reason = ROI_REASON_PRECISION
            else:
                roi_armed = True
                roi_reason = ROI_REASON_ARMED

    if scale is not None:
        # On the "default" tier only (never torch_compile/auto/cuda_graph — those stay
        # forced to the interpreter), a scale-active cook routes to codegen instead when
        # the UC-2 stencil gate would already choose it. Mirrors `_run_tier`'s own branch
        # exactly (`tier_id == "default" and not fused_chain and
        # _should_stencil_route(...)`), so the two can never disagree.
        if tier_id == "default" and not fused_chain:
            stencil = _stencil_route_would_apply(code, binding_types)
            if stencil:
                return TierVerdict("codegen", TIER_REASON_SCALE_ACTIVE_CODEGEN,
                                   roi_armed, roi_reason)
        return TierVerdict("interpreter", TIER_REASON_SCALE_ACTIVE, roi_armed, roi_reason)
    return TierVerdict(tier_id, TIER_REASON_SELECTED, roi_armed, roi_reason)
