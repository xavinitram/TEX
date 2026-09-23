"""
tex_chain — cooking a STAGE LIST, and the lineage keys that name what a cook produced.

The CACHE-6 chain family (`cook_stage_list`, `boundary_lineage_key`, `cook_fused_cached`
and the two binding predicates the keys rest on) plus CACHE-1's per-output
`_compute_lineage`, moved here verbatim from `tex_engine.py` (NEG-2). One domain: a
stage list goes in, raw `{output: tensor}` comes out, and every boundary between two
stages — or between one cook and the next — is named by a content-derived key rather
than by an address. `tex_engine` plans and dispatches a SINGLE program; this module
cooks a chain of them and says what a cooked frame is called.

**Two engine primitives travel with the chain, and are re-exported back.** The ENG-4
single raiser (`_compile_or_raise`) and the ENG-9 per-thread interpreter pool
(`_interp_pool`, `_get_interpreter`, `_clear_all_interpreter_caches`) are what a cook
needs in order to happen at all, and `cook_stage_list` reaches both. They live here so
this module stays a LEAF — it must import nothing that can reach `tex_engine`, because
that is what lets `tex_engine` import it at load and re-bind every moved name into the
same global slot its callers already read. The alternative, a function-local import at
each surviving call site, was measured at 0.286 us per site per cook and is exactly what
the ENG-14 split refused. Nothing changed name and no body changed: the moved functions
compile to byte-identical bytecode.

`_interp_pool` is a module global whose lifecycle stays in one file by design — the
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
from .tex_runtime.interpreter import Interpreter, _reads_viewer_builtin, _VIEWER_BUILTIN_NAMES
from .tex_runtime.interp_pool import ThreadLocalInterpreterPool as _ThreadLocalInterpreterPool
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
        # PM-11: viewer_context keys ONLY when the program actually calls a viewer builtin —
        # unlike `tc` above, every program can read frame/fps/time as bare identifiers with
        # no call, so there is no "before" key shape for that one to preserve. A program that
        # never calls viewer_exposure()/viewer_gamma() must key IDENTICALLY to a pre-PM-11
        # build (invariant #7); `lineage_key` itself omits the byte entirely for `None`.
        vc = ctx.viewer_context if _reads_viewer_builtin(ctx.program) else None
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
                canvas=canvas, flags=(*base_flags, f"out:{name}"), viewer_context=vc)
        return out
    except Exception:
        return None


# ── CACHE-6: fusion ↔ caching reconciliation (the cook side) ──────────────────

def cook_stage_list(stages, *, device="cpu", precision="fp32", latent_channel_count=0,
                    time_context=None, cancel=None, on_progress=None,
                    viewer_context: dict | None = None) -> dict:
    """Cook a raw fusion stage list (≥1) and return the interpreter's RAW {output: tensor}. One
    stage cooks as a plain program; ≥2 splice through `compile_fused`. It replicates prepare()'s
    param default-inject + widget-value conversion so a SUB-chain (a CACHE-6 prefix or suffix)
    cooks BIT-IDENTICALLY to those same stages inside the full fused program — the equivalence
    the CACHE-6 oracle rests on. fp32 is forced under a LATENT (M-3), exactly as prepare does.

    `viewer_context` (PM-11) rides beside `time_context` — a VALUE, never part of any lineage
    key (contrast `boundary_lineage_key`'s own `time_context=`, which DOES key: a different
    frame is a different correct result; a different viewer setting is not a different key,
    by this ask's own ruling), so a fused chain reads a viewer tweak exactly like an unfused
    one does."""
    # P0-H: the stage-list family is a public engine entry point that never learned about
    # promises — a Promise in a stage's bindings produced a raw TypeError out of the
    # marshalling seam whether or not it had landed. Resolving here (and refusing an unlanded
    # one as E7007) makes every stage-list caller behave like `prepare()`, which is the whole
    # point of the family: a sub-chain must cook identically to those stages inside the full
    # program. Guarded: the rebuild allocates a list plus a dict per stage, and
    # `cook_fused_cached` calls this up to three times per cook on the CACHE-6 hot path, so
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
    interp = _get_interpreter()
    return interp.execute(program, bindings, type_map, device=device,
                          latent_channel_count=latent_channel_count,
                          output_names=sorted(assigned.keys()), used_builtins=used_builtins,
                          precision=("fp32" if latent_channel_count else precision),
                          time_context=time_context, viewer_context=viewer_context,
                          cancel=cancel, on_progress=on_progress)


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


def _stages_read_viewer_builtin(stages) -> bool:
    """PM-11: does any of these RAW stage dicts' source call a viewer builtin?

    `boundary_lineage_key` keys a PREFIX (`stages[:k]`) before it is ever compiled — there
    is no `Program` AST here the way `_reads_viewer_builtin` wants, and re-parsing just to
    ask would duplicate `prefix_fingerprint`'s own compile a few lines below for no reason
    a substring scan can't answer just as safely. `viewer_exposure`/`viewer_gamma` are
    RESERVED names (E3011), so a plain substring match cannot miss a real call; the only
    way it can be wrong is a false POSITIVE (the name sitting inside a string literal or a
    comment), which over-keys rather than under-keys — the safe direction, same as the
    name-prefix heuristics elsewhere in this codebase."""
    return any(name in (st.get("code") or "") for st in stages for name in _VIEWER_BUILTIN_NAMES)


def boundary_lineage_key(stages, k, device, precision, *, upstream, time_context=None,
                         canvas=None, latent_channel_count=0,
                         viewer_context: dict | None = None) -> str:
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
    caller's diligence. An explicit `canvas=` still wins, for a caller that knows better."""
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
    # PM-11: key the boundary TAP on viewer_context too, when the PREFIX (stages[:k], the
    # portion this tap actually covers) calls a viewer builtin — the cached boundary pixels
    # are viewer-dependent then, and the key must say so or a later cook at a different
    # viewer value would hit this tap and silently serve the wrong exposure/gamma. Scoped to
    # the prefix, not the whole chain: a viewer stage only in the SUFFIX never touches this
    # tap's own cached pixels, so keying it in would only cost cache hits for nothing.
    vc = viewer_context if _stages_read_viewer_builtin(stages[:k]) else None
    return tex_results.lineage_key(program_fp=fp, device=str(device), precision=precision,
                                   params=params, upstream=tuple(upstream), time_context=time_context,
                                   canvas=canvas, flags=flags, viewer_context=vc)


def cook_fused_cached(stages, k, result_cache, *, device="cpu", precision="fp32",
                      time_context=None, latent_channel_count=0, upstream=(), cancel=None,
                      on_progress=None, viewer_context: dict | None = None) -> dict:
    """CACHE-6: cook a fused chain with a stage-(k-1) boundary TAP + SUFFIX SPLICE. On a cache
    HIT (the hot downstream param didn't touch the prefix) only stages k..N recook, reading the
    cached fp32 boundary; on a MISS the prefix is materialized, cached, and the suffix cooked.
    Equals the full fused cook within the fused-vs-sequential envelope (invariant #2, <1e-5).

    OPT-IN and dormant: a host passes the `ResultCache` + cut-point `k` + the source's CACHE-1
    identity via `upstream`; nothing on the default ComfyUI path calls this (invariant #7). Falls
    back to a whole-chain cook when the gate isn't met — non-fp32 (the boundary would be fp16, NOT
    the exact handoff), a LATENT, a DAG chain, a cut-point out of range, no cache, or NO `upstream`
    source key (without a content-sensitive source identity a cached boundary could be served for a
    different image — the safe default is a correct-but-not-incremental full cook)."""
    from .tex_fusion import is_linear_stage_list, suffix_stage_list, FusionError

    def _full():
        return cook_stage_list(stages, device=device, precision=precision,
                               latent_channel_count=latent_channel_count,
                               time_context=time_context, cancel=cancel, on_progress=on_progress,
                               viewer_context=viewer_context)

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
    key = boundary_lineage_key(stages, k, device, "fp32", time_context=time_context,
                               latent_channel_count=latent_channel_count, upstream=upstream,
                               viewer_context=viewer_context)
    boundary = result_cache.get(key)
    if boundary is None:
        b = cook_stage_list(stages[:k], device=device, precision="fp32",
                            time_context=time_context, cancel=cancel,
                            viewer_context=viewer_context).get("OUT")
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
        cook_stage_list(suffix, device=device, precision="fp32", time_context=time_context,
                        cancel=cancel, on_progress=on_progress,
                        viewer_context=viewer_context), k)
    if stages[k - 1].get("tap"):
        out.setdefault(f"_tap_s{k - 1}", boundary)   # the boundary IS that stage's output
    return out
