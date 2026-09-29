"""
tex_chain: cooking a STAGE LIST, and the lineage keys that name what a cook produced.

The CACHE-6 chain family (`cook_stage_list`, `cook_fused_cached`, `boundary_lineage_key`
and the two binding predicates the keys rest on), CACHE-1's per-output `_compute_lineage`,
and `cook_stage_dag`, the node-by-node windowed cook of a DAG-shaped stage list. One
domain: a stage list goes in, raw `{output: tensor}` comes out, and every boundary between
two stages, or between one cook and the next, is named by a content-derived key rather
than by an address. `tex_engine` plans and dispatches a SINGLE program; this module cooks
a chain of them and says what a cooked frame is called.

**Two engine primitives travel with the chain, and are re-exported back.** The ENG-4
single raiser (`_compile_or_raise`) and the ENG-9 per-thread interpreter pool
(`_interp_pool`, `_get_interpreter`, `_clear_all_interpreter_caches`) are what a cook
needs in order to happen at all, and `cook_stage_list` reaches both. They live here so
this module stays a LEAF: it must import nothing that can reach `tex_engine`, because that
is what lets `tex_engine` import it at load and re-bind every moved name into the same
global slot its callers already read. A function-local import at each call site was
measured at 0.286 us per site per cook, which is why the split refused it.

`_interp_pool` is a module global whose lifecycle stays in one file by design: the
sweep (`_clear_all_interpreter_caches`, reached from `tex_memory` under memory pressure)
sits beside the accessor that creates it rather than a file away.

The annotations on `_compute_lineage` name `tex_engine`'s `CookPlan` / `ExecContext`.
They are strings under PEP 563 and are never evaluated, so the import below is guarded
and no runtime edge back to `tex_engine` exists.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .tex_cache import get_cache
from .tex_compiler.diagnostics import raw_compile_errors, compile_error_from
from .tex_runtime.interpreter import Interpreter
from .tex_runtime.interp_pool import ThreadLocalInterpreterPool as _ThreadLocalInterpreterPool
# OBSERVER-46: the supported cook-observer seam. A leaf itself (stdlib only), so importing
# it here does not cost this module its own leaf status (see the module docstring above).
from .tex_runtime import cook_observer as _cook_observer
from .tex_marshalling import (
    convert_param_value as _convert_param_value,
    infer_binding_type as _infer_binding_type,
    resolve_promise_bindings as _resolve_promise_bindings,
    Promise as _Promise,
    expand_plane_bindings as _expand_plane_bindings,
    PlanesValue as _PlanesValue,
)

if TYPE_CHECKING:                       # pragma: no cover - typing only, never executed
    from .tex_engine import CookPlan, ExecContext


# ── ENG-4: the single compile raiser ─────────────────────────────────────────

def _compile_or_raise(code: str, binding_types: dict, *, fp: str | None = None):
    """Compile `code` to the cache's 6-tuple, or raise the PUBLIC `TEXCompileError`
    (carrying `[.diagnostics]`) on failure.

    This is the raiser for the CACHE compile path — the one every host-facing caller reaches:
    the ComfyUI node (via `prepare()`), `tex_cli`, and `tex_api.compile`. They catch
    `TEXCompileError`, so no host module imports the compiler's internals (ENG-4's goal).
    `tex_fusion.compile_fused` is the OTHER compile implementation (it parses + type-checks each
    stage directly, never through the cache) and raises the same public type via the same shared
    translator — so the exception TAXONOMY lives in one place even though there are two compile
    sites. `check()` (LANG-2) keeps its own collect-don't-raise path.

    PERF-5: `fp` forwards an already-computed `TEXCache.fingerprint(code, binding_types)` so a
    caller that needs the value anyway pays for it once.
    """
    try:
        return get_cache().compile_tex(code, binding_types, fp=fp)
    except raw_compile_errors() as e:
        raise compile_error_from(e, code) from e


# ── Cached interpreter (reused across executions to avoid rebuild overhead) ──
# ENG-9: the interpreter is PER-THREAD, not a process singleton — the interpreter carries
# per-instance execution state (scope stack, `_literal_cache`, `_builtins_lru`) that a
# branch-parallel executor would corrupt if shared. Single-cook ComfyUI is unchanged (one
# thread → one instance). The pool machinery is single-sourced in `interp_pool`.
_interp_pool = _ThreadLocalInterpreterPool(Interpreter)


def _get_interpreter() -> Interpreter:
    """The current thread's cook Interpreter (ENG-9), created on first use."""
    return _interp_pool.get()


def _clear_all_interpreter_caches():
    """Sweep EVERY per-thread interpreter's tensor LRUs (free_tensor_caches / memory pressure)."""
    _interp_pool.clear_all()


def _compute_lineage(plan: CookPlan, ctx: ExecContext, eff_precision: str,
                     raw_output: dict) -> dict | None:
    """CACHE-1: per-output lineage keys for this cook — the identity a frame cache keys on.
    Only reached when a caller asked (plan.want_lineage), so the deferred import and the
    walk stay entirely off the default ComfyUI cook path (invariant #7).

    program_fp is the fused fingerprint on a chain, else the single-program fp; params are
    the non-tensor bindings (widget $params) by value; a tensor input contributes its UPSTREAM
    lineage key (plan.upstream_keys), never its pixels; `eff_precision` is the precision the
    frame was ACTUALLY cooked at (fp32 if the finiteness net re-cooked). Each output is keyed by
    its OWN produced-frame shape (batch + every dim, so a batch-N or a BCHW-latent cook can't
    collide with a batch-1/BHWC one — the old (W,H)-from-an-input canvas dropped batch and read
    H as W for latents) plus the ROI rect. Best-effort: a keying failure returns None rather
    than failing the cook."""
    try:
        from . import tex_results
        program_fp = ctx.fused_fp or ctx.fp
        if program_fp is None:
            return None
        params = {n: v for n, v in ctx.bindings.items() if not isinstance(v, torch.Tensor)}
        # The WHOLE playhead keys, not just `frame`: `time`/`fps` move the output pixels too
        # (ENG-7's _TIME_BUILTIN_NAMES), so a time- or fps-animation is a distinct result at the
        # same frame — keying only `frame` would serve a stale frame. Duck-type the Mapping the
        # SAME way the interpreter does (prepare() already normalizes to a dict, but a directly
        # built ExecContext must not slip a Mapping-but-not-dict playhead past the key).
        tc = dict(ctx.time_context) if hasattr(ctx.time_context, "items") else None
        roi_rect = list(ctx.roi) if ctx.roi is not None else None
        # Cook-invariant flags that MOVE PIXELS but are neither bindings nor shape — so they
        # would otherwise fall out of the key and silent-serve a stale frame across a toggle:
        #   * debug_nan_highlight paints magenta/cyan overlays onto raw_output BEFORE this keys
        #     it (run() reassigns raw_output at the DBG-3/C4-ux block) — the node cache already
        #     folds this toggle in for the same reason (tex_node.fingerprint_inputs).
        #   * latent_channel_count materializes the program-readable `ic` builtin (interpreter/
        #     compiled env), so a program reading `ic` produces channel-count-dependent pixels
        #     even when the OUTPUT shape is channel-count-independent — graphed._capture_key
        #     already folds it into the CUDA-graph capture key for exactly this reason.
        base_flags = []
        if plan.debug_nan_highlight:
            base_flags.append("dbg:nan")
        if ctx.latent_channel_count:
            base_flags.append(f"ic:{int(ctx.latent_channel_count)}")
        out = {}
        for name in ctx.output_names:
            t = raw_output.get(name)
            shape = list(t.shape) if isinstance(t, torch.Tensor) else None
            # Key the device by the produced FRAME's concrete device, not ctx.device: device_mode
            # "cuda" resolves to the bare string "cuda" (no index), which would collide cuda:0 and
            # cuda:1 on a multi-GPU host — a cross-device wrong-pixel serve, violating the key's
            # own MANDATORY-device guarantee. The output tensor carries the real index (cuda:0).
            dev = str(t.device) if isinstance(t, torch.Tensor) else str(ctx.device)
            # canvas = the PRODUCED frame's full shape + the ROI rect (so two same-size
            # sub-windows at different offsets, or two different batch sizes, key apart).
            canvas = {"shape": shape, "roi": roi_rect}
            out[name] = tex_results.lineage_key(
                program_fp=program_fp, device=dev, precision=eff_precision,
                params=params, upstream=plan.upstream_keys, time_context=tc,
                canvas=canvas, flags=(*base_flags, f"out:{name}"), scale=ctx.scale)
        return out
    except Exception:
        return None


# ── CACHE-6: fusion ↔ caching reconciliation (the cook side) ──────────────────

def cook_stage_list(stages, *, device="cpu", precision="fp32", latent_channel_count=0,
                    time_context=None, cancel=None, on_progress=None, scale=None,
                    roi: tuple | None = None, roi_exec: bool | None = None) -> dict:
    """Cook a raw fusion stage list (≥1) and return the interpreter's RAW {output: tensor}. One
    stage cooks as a plain program; ≥2 splice through `compile_fused`. It replicates prepare()'s
    param default-inject + widget-value conversion so a SUB-chain (a CACHE-6 prefix or suffix)
    cooks BIT-IDENTICALLY to those same stages inside the full fused program — the equivalence
    the CACHE-6 oracle rests on. fp32 is forced under a LATENT (M-3), exactly as prepare does.

    `scale` (SCALE-47b) rides straight through to `Interpreter.execute` — this stage-list family
    is interpreter-only (it has no tier selection of its own), so there is no accelerated route
    to bypass here the way `tex_engine.run` needs to. `None` (every caller before this ask) is
    unaffected.

    `roi`/`roi_exec` (ROI-48A): the SAME per-cook window contract `tex_engine.cook(roi=...)`
    exposes, applied here so `tex_checkpoint.cook_checkpointed`'s suffix cook — often exactly
    one stage, the common shape right after a mid-graph edit near a checkpoint — can narrow
    too. Interpreter-tier only (this family has no other tier) and SINGLE-STAGE only: a
    fused chain (`len(stages) > 1`) declines a window for the identical reason
    `tex_engine.prepare()`'s own gate does (`not fused_chain`) — `roi_plan` is scoped to one
    program's source, and `run_roi`'s narrow-cook-crop crops one program's cook-region grid,
    neither of which describes a spliced multi-stage chain. Every other clause mirrors
    `prepare()`'s gate in the same order: LATENT narrows the wrong axis, an unarmed/malformed/
    whole-frame window is a documented no-op, and only a program `roi_plan` proves executable
    at fp32 gets a window — an `auto`/`fp16` cook (this family never resolves "auto" itself;
    the interpreter cooks it as fp32, so `eff_precision` is what actually runs) is declined
    for the same reason ROI is oracle-validated at fp32 only. Any doubt is a no-op: `roi=None`
    (every caller before this ask) never reaches this block at all — invariant #7. Recorded
    on `tier_trace` exactly like `tex_engine.cook` does, so a caller reads the same
    `tier_trace.last_roi()` signal regardless of which entry point served the cook; never
    raises — an ROI failure here falls back to the whole-frame cook below, exactly as
    `tex_engine_tiers._run_default` does for its own `run_roi` call."""
    # OBSERVER-46/O3: notify once for THIS entry point; a call nested under
    # `cook_fused_cached` or `cook_checkpointed` (both call this internally, up to three
    # times per cook) shares their outer notification instead of adding one — see
    # tex_runtime/cook_observer.py.
    with _cook_observer.scope("cook_stage_list"):
        # FIX-SCALE S1: this stage-list family is a public engine entry point in its own
        # right (not only reached via `tex_engine.prepare()`), so it must apply the SAME
        # scale-safety refusal `prepare()` applies — before this fix, none of the three
        # CACHE-6/7 entries (this one, `cook_checkpointed`, `materialize`; the latter two
        # both cook by calling this function) ever consulted it, so an unsafe program cooked
        # coarse, silently, through any of them. Scoped to the TERMINAL stage's own source,
        # the same scoping `prepare()`'s check already uses for a fused chain (an upstream
        # stage is not walked). A no-op when `scale` is `None`/`1.0` (invariant #7).
        from . import tex_roi as _tex_roi
        _tex_roi.require_scale_safe(stages[-1]["code"], scale)
        # P0-H: the stage-list family is a public engine entry point that never learned about
        # promises — a Promise in a stage's bindings produced a raw TypeError out of the
        # marshalling seam whether or not it had landed. Resolving here (and refusing an unlanded
        # one as E7007) makes every stage-list caller behave like `prepare()`, which is the whole
        # point of the family: a sub-chain must cook identically to those stages inside the full
        # program. Guarded: the rebuild allocates a list plus a dict per stage, and
        # `cook_fused_cached` calls this up to twice per cook on the CACHE-6 hot path, so
        # a no-promise chain (every chain today) must not pay for the feature — the scan is one
        # class check per binding against ~15-25 us of copying on a 50-stage chain.
        if any(v.__class__ is _Promise
               for st in stages for v in (st.get("bindings") or {}).values()):
            stages = [dict(st, bindings=_resolve_promise_bindings(st.get("bindings") or {}))
                      for st in stages]
        # DATA-6: the same expansion the single-program path does at prepare(), for the same
        # reason cook_stage_list resolves promises — a sub-chain must cook identically to those
        # stages inside the full fused program. Guarded on the same shape, so a plane-free chain
        # (every chain today) pays one class check per binding and copies nothing.
        if any(v.__class__ is _PlanesValue
               for st in stages for v in (st.get("bindings") or {}).values()):
            stages = [dict(st, bindings=_expand_plane_bindings(st.get("bindings") or {},
                                                               st.get("code") or ""))
                      for st in stages]
        if len(stages) == 1:
            st = stages[0]
            bindings = dict(st.get("bindings") or {})
            binding_types = {n: _infer_binding_type(v) for n, v in bindings.items()}
            program, type_map, referenced, assigned, param_info, used_builtins = \
                _compile_or_raise(st["code"], binding_types)
        else:
            from .tex_fusion import compile_fused
            program, type_map, referenced, assigned, param_info, used_builtins, bindings = \
                compile_fused(stages, _infer_binding_type)
        # prepare()'s shared param handling: inject code-defined defaults for referenced-but-unbound
        # params, then convert widget values (hex colour → RGB, comma vec → floats). Identical order
        # to the full cook so the merged bindings — and thus the pixels — match.
        for ref_name in referenced:
            if ref_name not in assigned and ref_name not in bindings and ref_name in param_info:
                dv = param_info[ref_name].get("default_value")
                if dv is not None:
                    bindings[ref_name] = dv
        for pname, pinfo in param_info.items():
            if pname in bindings:
                bindings[pname] = _convert_param_value(bindings[pname], pinfo, pname)
        output_names = sorted(assigned.keys())
        eff_precision = "fp32" if latent_channel_count else precision

        # FIX-TIER T1 (R1/R2#1-2): the shared ladder (`tex_roi.roi_eligibility`) — the SAME
        # function `tex_engine.prepare()` and `tex_engine_tiers.tier_verdict` call, so this
        # gate can never silently drift from theirs. `cook_stage_list` has no tier_id of
        # its own to select (it always runs the interpreter directly, single-stage-only for
        # a window) — `tier_id="default"` names the one tier this family ever windows, and
        # `fused_chain=len(stages) != 1` is this family's own "is this actually windowable"
        # proxy (a multi-stage chain is never a single program the ladder's `roi_plan` call
        # could analyze).
        roi_out = None
        roi_plan_obj = None
        if roi is not None:
            from .tex_tiling import _scalar_params
            _elig = _tex_roi.roi_eligibility(
                stages[0]["code"] if len(stages) == 1 else "",
                tier_id="default", fused_chain=len(stages) != 1,
                has_latent_input=bool(latent_channel_count), scale=scale, roi=roi,
                roi_exec=roi_exec, param_values=_scalar_params(bindings),
                binding_types=binding_types if len(stages) == 1 else None,
                eff_precision=eff_precision)
            _roi_why = _elig.message
            if _elig.armed:
                roi_out, roi_plan_obj = _elig.canonical, _elig.plan
            from .tex_runtime import tier_trace as _tier_trace
            _tier_trace.record_roi(None, _roi_why or ("roi armed" if roi_out is not None
                                                       else None))

        interp = _get_interpreter()
        if roi_out is not None:
            from .tex_memory import run_roi
            from .tex_runtime.host import CookCancelled as _CookCancelled
            try:
                return run_roi(interp, program, bindings, type_map, device,
                               latent_channel_count, output_names, used_builtins,
                               eff_precision, roi_out, roi_plan_obj.narrow, roi_plan_obj.halo,
                               time_context, cancel=cancel, on_progress=on_progress)
            except _CookCancelled:
                raise           # SCHED-3: a cancel aborts — never fall back to whole-frame
            except Exception as _roi_exc:
                # Never hard-fail a cook over a window — the same posture
                # `tex_engine_tiers._run_default` keeps for its own `run_roi` call.
                from .tex_runtime import tier_trace as _tier_trace
                _tier_trace.record_roi(None, f"roi cook failed: {_roi_exc}")
        return interp.execute(program, bindings, type_map, device=device,
                              latent_channel_count=latent_channel_count,
                              output_names=output_names, used_builtins=used_builtins,
                              precision=eff_precision,
                              time_context=time_context,
                              cancel=cancel, on_progress=on_progress, scale=scale)


# ── JOINWIRE-50: a DAG-shaped stage list, cooked node-by-node, windowed end-to-end ────
#
# `cook_stage_list` above is windowed only for a SINGLE stage — a fused (`len(stages)>1`)
# chain declines `roi=` outright (`fused_chain` in the ladder), because a spliced program's
# `roi_plan` describes the WHOLE fused source, not any one original stage. `cook_stage_dag`
# below does not fuse at all: every stage stays its own program, cooked through
# `cook_stage_list`'s existing single-stage `roi=` path, exactly the way a host's own
# node-by-node tick already cooks a LINEAR chain today (one `tex_engine.cook`/
# `cook_stage_list` call per node). The only genuinely new mechanism is planning ONE window
# per stage — via `tex_roi.chain_windows_dag`, fed per-stage reach `tex_roi.
# stage_dag_arg_halos` resolves from each stage's own source — and threading a windowed
# stage's CROPPED output to its downstream consumer(s) as a full-size tensor, which
# `run_roi`'s own per-cook narrowing step (`tex_memory.run_roi`) requires of every spatial
# binding.
#
# FIX-DAG G1 (R3#1): a non-sink windowed stage's crop used to be re-embedded into a fresh
# ZERO-FILLED full canvas the moment it was cooked, and THAT embedded (mostly-zero) tensor
# was what `stage_outputs` reported back to a caller — measured ~3.65 ms / 126.56 MiB at 4K,
# almost entirely the zero-fill (an `empty` allocation of the identical shape measured
# ~0.004 ms, a ~750x gap), above the host's own 2.0 ms windowed-tick budget from a SINGLE
# such stage. The efficiency review's own read of the brief's pooled-canvas idea REFUTED it on two
# grounds: `stage_outputs` is a documented "peek an intermediate" surface a caller (or a real
# test, `test_joinwire50b_checkpointed_dag_cook.py`'s own negative control) may read directly,
# so a pool recycled across calls could hand back another cook's real pixels or poison; and
# `stage_outputs[idx]` is memoized and can be read by more than one consumer within ONE call
# (a diamond), so a buffer keyed only on (shape, dtype, device) risks aliasing a still-live
# value. Both hazards are about the FULL-SIZE embedded buffer existing at all outside the
# ephemeral moment it feeds a downstream `run_roi` call — so the fix removes that buffer from
# the public surface instead of trying to pool it: `stage_outputs[idx]` now holds the RAW
# CROP (the same tensor `cook_stage_list` returned, never padded), and a sibling return key,
# `stage_windows`, names which entries are crops and at what absolute window — a caller that
# wants to peek gets the honest, small tensor plus its offset, never a mostly-garbage
# full-size one. The full-size re-embed still happens, but only ephemerally, in-memory, at
# the one point that still needs run_roi's full-extent contract (feeding a downstream
# consumer's own cook) — and it is cheap now (`new_empty`, not `new_zeros`) because the SAME
# poisoned-fill proof this module already had (`test_joinwire50_merge_below_edit_poisoned_
# fill_is_never_read`, unedited by this fix) already showed the padded region is provably
# never read by anything a downstream cook does, whatever it contains — closing exactly the
# gap that made a POOLED (recycled) uninitialised buffer unsafe, without recycling anything.

def _embed_window(val, window):
    """Patch a windowed stage's CROPPED output into a FRESH, ephemeral full-size `(W, H)`
    canvas at its window's offset, so a downstream stage's OWN `run_roi` narrowing step
    (which requires every spatial binding at the FULL size the window declares — `tex_memory.
    run_roi`'s own extent check) can read it. THIS FUNCTION'S RETURN VALUE MUST NEVER BE
    STORED INTO `stage_outputs` (FIX-DAG G1) — it exists only to feed one downstream cook
    call; the public record of what a windowed stage produced is its raw, un-padded crop
    (`stage_outputs[idx]`) plus its window (`stage_windows[idx]`), never this buffer.

    The region OUTSIDE the window is left at whatever `new_empty` (UNINITIALISED — deliberately
    not zero-filled, FIX-DAG G1) gives it and is NEVER READ by anything this cook does:
    JOIN-49's `chain_windows_dag` only ever grows a stage's window BY a downstream consumer's
    own demand on it (`StageSpec.halo_for`, unioned across every dirty consumer) — so `window`
    already covers every region any consumer cooked THIS tick will actually narrow into.
    Not merely argued: `test_joinwire50_merge_below_edit_poisoned_fill_is_never_read` poisons
    this fill with a distinctive non-finite sentinel instead of zeros and still gets
    pixel-identical results — the only way to show unread garbage is truly unread rather than
    coincidentally zero, and the same proof that makes skipping the zero-fill safe here: an
    uninitialised region is just a different kind of unread garbage, and this call site is the
    ONE place (never `stage_outputs`) where "unread" is actually guaranteed.

    Scalar/string outputs (rank < 3 — no spatial dims at all) pass through unchanged:
    `run_roi`'s own crop-back only ever narrows the two spatial dims of a rank>=3 tensor, so
    a non-spatial output was never cropped and has no window to re-embed."""
    if not isinstance(val, torch.Tensor) or val.dim() < 3:
        return val
    x0, y0, w, h, W, H = window
    full = val.new_empty((val.shape[0], H, W, *val.shape[3:]))
    full[:, y0:y0 + h, x0:x0 + w] = val
    return full


def cook_stage_dag(stages, *, device="cpu", precision="fp32", latent_channel_count=0,
                   time_context=None, cancel=None, on_progress=None, scale=None,
                   roi: tuple | None = None, roi_exec: bool | None = None,
                   dirty_from: int = 0, valid=None, declined=(),
                   known_outputs: dict | None = None,
                   result_cache=None, upstream=(), store: set | None = None) -> dict:
    """Cook a DAG-shaped stage list, such as a Merge reading two upstream stages below an
    edit, node by node, windowed end to end via `tex_roi.chain_windows_dag` when the sink
    (the last stage) is asked for a sub-window.

    `stages[i]["chain_inputs"]` is the same DAG payload `tex_fusion.compile_fused` reads
    (`{binding_name: [src_stage_idx, "OUT"]}`); every other binding is a plain value, as in
    `cook_stage_list`. Stage indices must be topologically ordered (every `chain_inputs`
    index `< i`, else `ValueError`, a defect in the caller's graph). `roi`, `dirty_from`,
    `valid` and `declined` are `chain_windows_dag`'s own parameters. A scale-active cook
    never windows here (the same `ROI_REASON_SCALE_ACTIVE` gate as `roi_eligibility`).
    Every per-stage cook goes through `cook_stage_list`.

    Clean stages. `known_outputs` (`{stage_index: {name: tensor}}`) supplies the already-valid
    full-frame output of a CLEAN stage (`i < dirty_from`) that a dirty consumer reads. A
    missing clean value that a dirty stage needs raises `ValueError` rather than being
    fabricated. Instead of `known_outputs`, a host may hand a `result_cache` (the object
    `cook_checkpointed` takes) plus `upstream` (the CACHE-1 source keys, as in
    `boundary_lineage_key`), and this function keeps the boundaries itself:

      * a clean stage not in `known_outputs` is looked up under `boundary_lineage_key(stages,
        i + 1, ...)`, the same key `cook_checkpointed` uses for a linear boundary;
      * a stage's own cook this tick is `put` under that key ONLY when `tier_trace.last_roi()`
        says it served whole-frame. A windowed output carries an unread region outside its
        window and must never be stored as a boundary that a later, differently-windowed tick
        could read back as whole-frame. The cache is populated from the served window, never
        from the requested one;
      * both are inert unless windows are planned (`roi=` given and eligible), so `roi=None`
        cooks every stage exactly as a bare `cook_stage_list` would;
      * a boundary is read or written only when `upstream` has at least one key per tensor
        binding in the prefix it covers (the gate `cook_fused_cached` applies); without it
        two same-shape sources would share a key, so the cache is left out.

    `store` (optional set of stage indices) narrows which clean whole-frame stages are `put`
    into `result_cache`, never widens it: a stage not in `store` is not stored, a windowed
    stage is not stored whatever `store` says, and reading a boundary back is unaffected.
    `store=set()` stores nothing this tick but still serves earlier boundaries. An index
    outside `range(len(stages))` is rejected up front with a `ValueError`.

    Returns `{"result": {name: tensor}, "stage_outputs": {idx: {name: tensor}},
    "stage_windows": {idx: window}, "windows": windows_or_None, "stages_windowed": int,
    "stages_whole": int}`. `stage_outputs[idx]` is always the RAW value the stage's cook
    produced: a windowed, non-sink stage's entry is its bare crop, and `stage_windows` names
    which entries are crops and at what absolute `(x0, y0, w, h, W, H)`. An entry absent
    from `stage_windows` is full-size (a whole-frame cook, or a `known_outputs` or
    `result_cache` value). The full-size re-embed a downstream consumer needs happens
    ephemerally in `_materialize_input` and is never reported. `windows` is `None` when
    nothing was planned (no `roi=`, or `chain_windows_dag` refused the plan)."""
    with _cook_observer.scope("cook_stage_dag"):
        from . import tex_roi as _tex_roi
        from .tex_runtime import tier_trace as _tier_trace_mod
        from .tex_tiling import _scalar_params
        n = len(stages)
        known_outputs = known_outputs or {}
        eff_precision = "fp32" if latent_channel_count else precision
        if store is not None:
            for _idx in store:
                if not (0 <= _idx < n):
                    raise ValueError(
                        f"cook_stage_dag: store names stage index {_idx}, but this stage "
                        f"list only has {n} stage(s) — I need an index between 0 and "
                        f"{n - 1} inclusive. Check the index against the `stages` list you "
                        f"passed.")

        # 1. One StageSpec per stage, resolved from EACH stage's own source alone (no
        #    cross-stage knowledge needed for this step — `stage_dag_arg_halos` reads one
        #    program at a time, exactly as `stage_halo` already does for the linear family).
        chain_map = []             # [{binding_name: (src_idx, out_name)}, ...] per stage
        specs = []
        for i, st in enumerate(stages):
            ci = {}
            for b, edge in (st.get("chain_inputs") or {}).items():
                idx = int(edge[0])
                if not (0 <= idx < i):
                    raise ValueError(
                        f"cook_stage_dag: stage {i} chain_inputs[{b!r}] names stage {idx}, "
                        f"not an earlier stage index (< {i}) — inputs must be topologically "
                        f"ordered")
                ci[b] = (idx, edge[1])
            chain_map.append(ci)
            bindings = dict(st.get("bindings") or {})
            name_to_upstream = {b: idx for b, (idx, _out) in ci.items()}
            binding_types = {name: _infer_binding_type(v) for name, v in bindings.items()
                             if name not in ci} or None
            halo, arg_halo = _tex_roi.stage_dag_arg_halos(
                st["code"], name_to_upstream, param_values=_scalar_params(bindings),
                binding_types=binding_types)
            specs.append(_tex_roi.StageSpec(
                halo, tuple(sorted(set(name_to_upstream.values()))), arg_halo or None))

        # 2. Plan windows, only when eligible. The gate is the shared ladder
        #    `tex_roi.roi_eligibility`, the same one `cook_stage_list` calls, so a change to
        #    it reaches both. Its `tier_id`, `fused_chain` and `executable` legs are answered
        #    per stage by `StageSpec` in step 1, so this call passes a neutral empty program
        #    (default tier, never fused, halo 0) and only the scale, latent, precision, armed
        #    and malformed conditions are live here.
        windows = None
        if roi is not None:
            _elig = _tex_roi.roi_eligibility(
                "", tier_id="default", fused_chain=False,
                has_latent_input=bool(latent_channel_count), scale=scale, roi=roi,
                roi_exec=roi_exec, param_values={}, binding_types=None,
                eff_precision=eff_precision)
            if _elig.armed:
                windows = _tex_roi.chain_windows_dag(
                    specs, _elig.canonical, dirty_from=dirty_from, valid=valid,
                    declined=declined)

        # JOINWIRE-50b: the ONE key function for every clean-stage boundary this call either
        # reads or writes — `boundary_lineage_key` itself, unmodified (see the docstring
        # above for why its default canvas already answers a DAG stage list correctly). A
        # closure so both the skip branch and the consumer-resolution branch below share one
        # spelling rather than two that could drift apart.
        # A boundary is keyed by `upstream` alone for tensor CONTENT, so a cache read or write
        # needs one upstream key per tensor binding in the prefix it covers — the same gate
        # `cook_fused_cached` applies. Without it two same-shape sources collide on one key.
        _prefix_tensors = []
        _seen = 0
        for st in stages:
            _seen += sum(1 for v in (st.get("bindings") or {}).values() if _is_tensor_binding(v))
            _prefix_tensors.append(_seen)

        def _cacheable(idx: int) -> bool:
            return result_cache is not None and len(upstream) >= _prefix_tensors[idx]

        def _checkpoint_key(idx: int) -> str:
            return boundary_lineage_key(
                stages, idx + 1, device, precision, upstream=upstream,
                time_context=time_context, latent_channel_count=latent_channel_count,
                scale=scale)

        def _clean_lookup(idx: int):
            """A clean stage's already-valid value: `known_outputs` first, then
            `result_cache` (JOINWIRE-50b) — `None` when neither has it. Always full-frame by
            contract (a `known_outputs` value is documented as such; a `result_cache` hit is
            `put` only off `served_roi is None`, below) — never a `stage_windows` entry."""
            v = known_outputs.get(idx)
            if v is not None:
                return v
            if _cacheable(idx):
                cached = result_cache.get(_checkpoint_key(idx))
                if cached is not None:
                    return {"OUT": cached}
            return None

        # FIX-DAG G1 (R3#1): `stage_windows[idx]` names which `stage_outputs[idx]` entries
        # are bare CROPS (a genuinely windowed stage's own cook) rather than already
        # full-size — the public record `stage_outputs` returns never carries a padded
        # buffer (see `cook_stage_dag`'s own docstring and `_embed_window`'s). `_embedded`
        # memoizes the one ephemeral full-size re-embed a crop needs PER CALL, so a stage
        # read by more than one consumer this tick (a diamond) re-embeds once, not once per
        # consumer, without storing the result anywhere `stage_outputs` exposes.
        stage_windows: dict = {}
        _embedded: dict = {}

        def _materialize_input(idx: int, src: dict) -> dict:
            """`src` (a `stage_outputs`/`_clean_lookup` value for upstream stage `idx`) as a
            downstream stage's own `run_roi` step needs it: unchanged when `idx` is already
            full-size, else an ephemeral, memoized full-size re-embed of its crop — the ONLY
            place that re-embed exists; it is never assigned back into `stage_outputs`."""
            window = stage_windows.get(idx)
            if window is None:
                return src
            cached = _embedded.get(idx)
            if cached is None:
                cached = _embedded[idx] = {name: _embed_window(val, window)
                                           for name, val in src.items()}
            return cached

        # 3. Cook, stage by stage. `chain_inputs` only ever names an EARLIER index (checked
        #    above), so index order IS topological order — no separate sort needed.
        stage_outputs: dict = {}
        stages_windowed = 0
        stages_whole = 0
        for i, st in enumerate(stages):
            if windows is not None and windows[i] is None:
                if i < dirty_from:
                    v = _clean_lookup(i)
                    if v is not None:
                        stage_outputs[i] = v
                continue           # clean-and-undemanded, or dirty-but-undemanded (JOIN-49)

            bindings = dict(st.get("bindings") or {})
            for b, (idx, out) in chain_map[i].items():
                src = stage_outputs.get(idx)
                if src is None:
                    src = _clean_lookup(idx)
                if src is None or out not in src:
                    raise ValueError(
                        f"cook_stage_dag: stage {i} needs stage {idx}'s output {out!r}, "
                        f"which was never cooked and is not in known_outputs"
                        + (" or result_cache" if result_cache is not None else ""))
                stage_outputs.setdefault(idx, src)  # memoize a known_outputs/cache hit for a
                #                                      second consumer (a diamond) or for the
                #                                      caller's own `stage_outputs` inspection
                bindings[b] = _materialize_input(idx, src)[out]

            stage_roi = None
            if windows is not None and windows[i] is not None:
                w = windows[i]
                if w[2:4] != w[4:6]:
                    stage_roi = w

            out = cook_stage_list(
                [dict(st, bindings=bindings)], device=device, precision=precision,
                latent_channel_count=latent_channel_count, time_context=time_context,
                cancel=cancel, on_progress=on_progress, scale=scale,
                roi=stage_roi, roi_exec=True if stage_roi is not None else None)

            # Count (and decide whether to re-embed) off what THIS stage's own cook actually
            # served, per `tier_trace.last_roi()` — not off `stage_roi` alone. A stage this
            # function planned to window can still decline internally (its own `roi_plan`
            # disagreeing, e.g. `convolve`'s footprint making the WHOLE program
            # non-executable) and fall back to whole-frame; trusting the REQUEST rather than
            # the SERVED window would both mis-report the count and try to re-embed a
            # full-size output into a too-small window (a shape-mismatched, silently wrong
            # patch) — the exact class of bug `_embed_window`'s postcondition must not permit.
            # `stage_roi is None` (this stage was never asked to window) short-circuits
            # WITHOUT consulting `tier_trace`: `cook_stage_list` skips its whole `roi`
            # block whenever `roi=None`, so it never calls `record_roi` for THIS cook, and
            # `last_roi()` would otherwise still read whatever an EARLIER, unrelated cook on
            # this thread last recorded — a stale-read bug, not a stale-window one.
            served_roi = None if stage_roi is None else _tier_trace_mod.last_roi()[0]
            if served_roi is not None:
                # FIX-DAG G1: `out` stays the bare crop `cook_stage_list` returned — no
                # eager re-embed here, sink or not. `stage_windows[i]` records the window so
                # `_materialize_input` can produce the ephemeral full-size feed ONLY if and
                # when a downstream consumer actually reads this stage (`i != n - 1`; the
                # sink has no consumer within this call by construction).
                stage_windows[i] = served_roi
                stages_windowed += 1
            else:
                stages_whole += 1
                # `result_cache` is populated ONLY here: `served_roi is None` is tier_trace's own
                # record that THIS cook served whole-frame, so a windowed crop (the branch
                # above) never reaches this line. `windows is not None` keeps a plain
                # `roi=None` cook from paying puts (invariant 7). `i != n - 1`: the sink has
                # no suffix, so `boundary_lineage_key` has no valid key for it and nothing
                # downstream could read it. `store` only narrows this eligibility.
                if (_cacheable(i) and windows is not None and i != n - 1
                        and "OUT" in out and (store is None or i in store)):
                    result_cache.put(_checkpoint_key(i), out["OUT"],
                                     canvas={"shape": list(out["OUT"].shape)})
            stage_outputs[i] = out

        return {"result": stage_outputs.get(n - 1, {}), "stage_outputs": stage_outputs,
                "stage_windows": stage_windows, "windows": windows,
                "stages_windowed": stages_windowed, "stages_whole": stages_whole}


def _is_tensor_binding(v) -> bool:
    """True for a binding that carries PIXELS — a tensor, or a Promise of one (P0-H).

    Spelled once because three places ask it and all three were wrong in different ways when
    they each asked it themselves: the params/tensor split in `boundary_lineage_key`, its
    canvas enumeration, and `cook_fused_cached`'s upstream-coverage gate."""
    # `__class__ is` for the Promise arm, matching `resolve_promise_bindings` and
    # `infer_binding_type`. With `isinstance`, a Promise SUBCLASS would count as a tensor
    # binding here, then miss resolution in both of those (they test exact class) and
    # surface as an E7005 out of `prefix_fingerprint` — a disagreement between three
    # predicates that are supposed to describe one thing.
    # DATA-6: a PlanesValue is a TENSOR binding for keying purposes — it carries pixels, and
    # the P0-H lesson is that a pixel-carrier left on the `params` side is folded by `repr`,
    # which for a `__slots__` object is its ADDRESS. `_binding_shape` below supplies the
    # composite shape that keeps the boundary's identity content-derived.
    return (isinstance(v, torch.Tensor) or v.__class__ is _Promise
            or v.__class__ is _PlanesValue)


def _binding_shape(v):
    """The shape a tensor binding contributes to a canvas descriptor, or None if unknowable.

    A landed Promise reports its value's real shape; an unlanded one reports its declaration.
    None means "an unlanded promise that declared no shape" — un-keyable, because the
    boundary's resolution is part of its identity."""
    if v.__class__ is _PlanesValue:
        # DATA-6: a PLANES wire's identity is EVERY declared plane's name and shape, never the
        # first plane's — two wires whose `diffuse` matches and whose `Z` does not are
        # different boundaries, and keying on one plane would serve one for the other. The
        # names ride the key because a plane SET is part of what the boundary resolved to.
        return tuple(x for n in sorted(v.planes)
                     for x in (n, *(int(d) for d in v.planes[n].shape)))
    if isinstance(v, torch.Tensor):
        return tuple(v.shape)
    val = getattr(v, "value", None)
    if val is not None:
        return tuple(val.shape)
    declared = getattr(v, "shape", None)
    return tuple(declared) if declared else None


def boundary_lineage_key(stages, k, device, precision, *, upstream, time_context=None,
                         canvas=None, latent_channel_count=0, scale=None) -> str:
    """CACHE-6: the lineage key a stage-(k-1) boundary tap is cached under — the upstream
    SUB-CHAIN fingerprint (`tex_fusion.prefix_fingerprint`) × the prefix stages' param VALUES ×
    the SOURCE identity `upstream` × device × precision × playhead × canvas, namespaced by the cut
    and the `ic` channel-count flag.

    `upstream` is MANDATORY and carries the CACHE-1 lineage key(s) of the external tensor(s)
    feeding the prefix — the host's content-sensitive source identity. It is what stops two
    different source IMAGES from colliding onto one boundary (a silent stale serve): the prefix
    program fingerprint is value-independent and the params fold only NON-tensor bindings, so
    without a source key the tensor axis is unkeyed. It must be CONTENT-sensitive — a raw
    `data_ptr` is NOT (a reused/overwritten frame buffer keeps its address; the caching allocator
    reuses freed addresses), so a video pipeline doing `src.copy_(next_frame)` would serve a stale
    boundary. A GRAPH-1 host already stamps such a key per produced value; `cook_fused_cached`
    refuses to cache without one. fp32 is the exact-handoff contract, so precision keys it too.

    CACHE-7: `canvas` DEFAULTS to the prefix's input SHAPES rather than to nothing. It used to
    default to nothing and neither caller passed it, so a tap's identity carried no resolution —
    and `ResultCache.get` validates neither shape nor device. With one host source key, a 64²
    cook and a 128² cook minted the SAME key and the 128² request was served the 64² frame:
    silently, wrong size, no error (reproduced; pinned by a regression row). Resolution rode
    entirely on the host's `upstream` string, which nothing documented as required to encode one.
    Defaulting HERE rather than at each call site is what closes it for `cook_fused_cached` and
    the CACHE-7 multi-tap path at once — the hole was in this function's contract, not in a
    caller's diligence. An explicit `canvas=` still wins, for a caller that knows better.

    `scale` (SCALE-47b): threaded straight to `lineage_key` (its own docstring says why this must
    be explicit rather than folded into `fp`) — a coarse-scale checkpoint and a full-scale one
    mint different keys, so `cook_checkpointed` can never serve one to the other. `None` (every
    caller before this ask) is unaffected."""
    # OBSERVER-46/O3: notify once for THIS entry point; a call nested under
    # `cook_fused_cached` or `cook_checkpointed` (both call this internally, once per cut)
    # shares their outer notification instead of adding one — see
    # tex_runtime/cook_observer.py. `scope()`'s `__exit__` runs unconditionally, so the
    # un-keyable-Promise ValueError below still balances it.
    with _cook_observer.scope("boundary_lineage_key"):
        from . import tex_results
        from .tex_fusion import prefix_fingerprint
        fp = prefix_fingerprint(stages, k, _infer_binding_type)
        # P0-H: a Promise is a TENSOR binding that has not arrived yet, so it belongs on the
        # tensor side of this split — not in `params`. It landed there because the test asks
        # "is it a Tensor?", and `_canon_params` folds unknown objects via `repr`, which for a
        # `__slots__` object is its ADDRESS. That made checkpoint identity address-keyed: two
        # equivalent promises minted different keys (spurious misses, verified), and CPython
        # address reuse could alias two genuinely different ones onto the same key.
        #
        # THE HALF v0.34.1's FIRST DRAFT FORGOT, and it was worse than the defect it replaced:
        # removing Promise from `params` without adding it to the tensor side made a promise-fed
        # prefix invisible to BOTH halves. `_shapes()` skipped it, so the canvas enumeration came
        # back empty, fell through to the `chain_in` branch, came back empty again — and every
        # resolution minted the SAME key. Address-keying was merely wasteful (different keys,
        # spurious misses, safe); one key for two resolutions is a wrong-size boundary served on a
        # cache HIT, which is the exact failure the P0-3 comment below says it closed.
        params = {f"s{i}:{n}": v
                  for i, st in enumerate(stages[:k])
                  for n, v in (st.get("bindings") or {}).items()
                  if not _is_tensor_binding(v)}
        if canvas is None:
            # Every tensor the prefix reads, by stage-qualified name and shape. Derivable BEFORE
            # the cook: a TEX program's output canvas equals its input canvas until LANG-6's
            # `canvas()` lands, at which point this becomes a derived shape rather than a copied one.
            #
            # A Promise contributes its DECLARED shape — declared up front for exactly this reason
            # (identity computable before the pixels land). Per-binding device is deliberately not
            # emitted here, for promises or tensors: `device` is already a mandatory top-level
            # lineage_key component, so repeating it per binding would change the key shape for
            # every existing caller to say something the key already says.
            def _shapes(sts, off=0):
                out = []
                for i, st in enumerate(sts):
                    for n, v in sorted((st.get("bindings") or {}).items()):
                        if not _is_tensor_binding(v):
                            continue
                        shape = _binding_shape(v)
                        if shape is None:
                            raise ValueError(
                                f"boundary_lineage_key cannot key stage {i + off} binding '{n}': "
                                f"it is an unlanded Promise that declared no shape, and the "
                                f"boundary's RESOLUTION is part of its identity. Declare "
                                f"`shape=` on the promise, or pass `canvas=` explicitly.")
                        out.append([f"s{i + off}:{n}", *shape])
                return out

            canvas = {"in": _shapes(stages[:k])}
            if not canvas["in"]:
                # P0-3: a GENERATOR-HEAD prefix reads no tensors, so the enumeration above is empty
                # and every resolution mints the SAME key — a 64² and a 128² cook of the same chain
                # collide and the wrong-size boundary is served (reproduced end-to-end as an
                # `InterpreterError` size mismatch; with a `sample()` suffix it would be silent
                # wrong pixels instead of a raise).
                #
                # The boundary's resolution is the FUSED PROGRAM's grid, and that is set by whatever
                # spatial binding exists anywhere in the chain — not only in the prefix. So when the
                # prefix carries none, key on the whole chain's input shapes. A chain with no tensor
                # bindings at all cooks scalar-mode, where there is genuinely one resolution and
                # nothing to collide.
                canvas = {"chain_in": _shapes(stages)}
        flags = [f"tap:s{k - 1}"]
        if latent_channel_count:
            flags.append(f"ic:{int(latent_channel_count)}")
        return tex_results.lineage_key(program_fp=fp, device=str(device), precision=precision,
                                       params=params, upstream=tuple(upstream), time_context=time_context,
                                       canvas=canvas, flags=flags, scale=scale)


def cook_fused_cached(stages, k, result_cache, *, device="cpu", precision="fp32",
                      time_context=None, latent_channel_count=0, upstream=(), cancel=None,
                      on_progress=None, scale=None) -> dict:
    """CACHE-6: cook a fused chain with a stage-(k-1) boundary TAP + SUFFIX SPLICE. On a cache
    HIT (the hot downstream param didn't touch the prefix) only stages k..N recook, reading the
    cached fp32 boundary; on a MISS the prefix is materialized, cached, and the suffix cooked.
    Equals the full fused cook within the fused-vs-sequential envelope (invariant #2, <1e-5).

    OPT-IN and dormant: a host passes the `ResultCache` + cut-point `k` + the source's CACHE-1
    identity via `upstream`; nothing on the default ComfyUI path calls this (invariant #7). Falls
    back to a whole-chain cook when the gate isn't met — non-fp32 (the boundary would be fp16, NOT
    the exact handoff), a LATENT, a DAG chain, a cut-point out of range, no cache, or NO `upstream`
    source key (without a content-sensitive source identity a cached boundary could be served for a
    different image — the safe default is a correct-but-not-incremental full cook).

    `scale` (FIX-SCALE S7): rides through to every internal `cook_stage_list` call (each of
    which applies the S1 scale-safety refusal on its own terminal stage) and to
    `boundary_lineage_key`, so a coarse-scale boundary never collides with a full-scale one —
    the SAME pattern `cook_checkpointed`/`materialize` already established for the CACHE-7
    sibling. `None` (every caller before this ask) is unaffected."""
    # OBSERVER-46/O3: notify once for THIS entry point; every internal call below —
    # `_full()`'s and the hit/miss paths' `cook_stage_list`, and `boundary_lineage_key` —
    # shares this one notification rather than adding its own (see
    # tex_runtime/cook_observer.py).
    #
    # `_full()` and the hit/miss branches below call `cook_stage_list`/`boundary_lineage_key`
    # through `_tex_engine`'s attribute (ROUTE-45's routing convention, see
    # tex_engine_tiers.py's module docstring), not this module's own local name, so a host
    # wrap on `tex_engine.cook_stage_list` sees every call this function makes.
    with _cook_observer.scope("cook_fused_cached"):
        from . import tex_engine as _tex_engine
        from .tex_fusion import is_linear_stage_list, suffix_stage_list, FusionError

        def _full():
            return _tex_engine.cook_stage_list(stages, device=device, precision=precision,
                                   latent_channel_count=latent_channel_count,
                                   time_context=time_context, cancel=cancel, on_progress=on_progress,
                                   scale=scale)

        # `upstream` must key EVERY tensor input of the prefix — the source, and any EXTRA image a
        # prefix stage reads — not just be non-empty (a partial cover could stale-serve when only an
        # unkeyed prefix tensor changes; the count also subsumes the no-upstream `()` default, since a
        # chain's prefix always has ≥1 source tensor → 0<1). The range check `not (1 <= k < len)` is
        # ordered BEFORE the tensor-count term so an out-of-range (or non-int) `k` short-circuits to a
        # full cook without ever slicing `stages[:k]`.
        if (precision != "fp32" or latent_channel_count or result_cache is None
                or not is_linear_stage_list(stages) or not (1 <= k < len(stages))
                # P0-H: count PROMISED tensors too. A promise is a tensor input whose pixels have
                # not arrived, so an uncovered one is exactly the partial cover this term exists to
                # refuse — and skipping it let a promise-fed prefix through the gate with zero
                # upstream keys naming it.
                or len(upstream) < sum(1 for st in stages[:k]
                                       for v in (st.get("bindings") or {}).values()
                                       if _is_tensor_binding(v))
                # ...and an unlanded promise with no declared shape cannot be keyed at all
                # (`_binding_shape` -> None). Refuse to a full cook rather than let
                # `boundary_lineage_key` raise out of a serve path whose contract is to fall back.
                # `isinstance` first: `_binding_shape` returns `tuple(v.shape)` for any real
                # tensor and can only be None for a promise, so without the guard this walks
                # every prefix binding allocating a torch.Size->tuple purely to compare it to
                # None — on the cache-HIT path whose whole point is to skip a prefix cook.
                or any(isinstance(v, _Promise) and _binding_shape(v) is None
                       for st in stages[:k]
                       for v in (st.get("bindings") or {}).values())):
            return _full()
        # P0-5: a tap on a stage strictly below `k-1` is inside the served prefix and the suffix
        # cook never produces it. Serving anyway drops a requested output and shifts the host's
        # output slots — cook whole instead.
        from .tex_fusion import remap_suffix_taps, unservable_prefix_taps
        if unservable_prefix_taps(stages, k):
            return _full()
        key = _tex_engine.boundary_lineage_key(stages, k, device, "fp32", time_context=time_context,
                                   latent_channel_count=latent_channel_count, upstream=upstream,
                                   scale=scale)
        boundary = result_cache.get(key)
        if boundary is None:
            b = _tex_engine.cook_stage_list(stages[:k], device=device, precision="fp32",
                                time_context=time_context, cancel=cancel, scale=scale).get("OUT")
            if b is None:            # a chain always assigns @OUT; if not, cook whole (correct)
                return _full()
            result_cache.put(key, b, canvas={"shape": list(b.shape)})
            boundary = b     # the freshly-cooked, locally-owned prefix output — put stored its own
            #                  frozen copy, so `b` is unaliased; feed it straight in (no re-get clone)
        try:
            suffix = suffix_stage_list(stages, k, boundary)
        except FusionError:          # a malformed cut (head stage lacks a chain_input to rebind) — the
            return _full()           #   documented whole-chain fallback, not a crash after the put
        # P0-5: the suffix renumbers stages, so `compile_fused` names its taps `_tap_s{j}` where the
        # original was `k+j`. Remap at the serve seam — on BOTH the miss and the hit path, which is
        # this single return.
        out = remap_suffix_taps(
            _tex_engine.cook_stage_list(suffix, device=device, precision="fp32", time_context=time_context,
                            cancel=cancel, on_progress=on_progress, scale=scale), k)
        if stages[k - 1].get("tap"):
            out.setdefault(f"_tap_s{k - 1}", boundary)   # the boundary IS that stage's output
        return out
