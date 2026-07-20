"""
ENG-1 / PORT-2b — `tex_engine`: the host-agnostic cook engine.

Until v0.22 the full engine — tier selection, the interpreter fallbacks, the OOM
ladder, strip tiling, the `precision="auto"` gate — was reachable ONLY through
`TEXWrangleNode.execute`, a ComfyUI v3 node classmethod. Even `tex run` (PORT-3, the
CLI whose whole job is proving TEX is host-agnostic) had to import the node facade to
cook a frame. This module is that engine, lifted out whole.

`tex_node.execute` is now marshal-in → `prepare`/`run` → marshal-out, and keeps only what
is genuinely ComfyUI adaptation (S-1): the kwargs/slot-pool protocol, LATENT dict
wrap/unwrap, the `ui=` HUD payload, and the `TEX_DIAG:`-suffixed RuntimeError the
frontend parses.

The cook is a two-step, and the split is load-bearing rather than stylistic:

    plan   = prepare(code, bindings, ...)   # compile/fuse, resolve device+precision, pick tier
    result = run(plan)                      # dispatch, safety nets, budgets
    cook(...)                               # = run(prepare(...)) — the one-shot public call

A `prepare` failure means *nothing cooked*; a `run` failure means the program started.
Hosts need that line: it is exactly what lets tex_node decide whether a stage-tagged
error may be attributed to a linked node (a chain that failed to assemble has no
running stage to blame).

**Scope — this is v0.22's MECHANICAL move.** The cascade below is v0.21's, comments
intact, re-homed. Behaviour is byte-identical and the benchmark is the gate (invariant
#7: a refactor release must be invisible) — measured at +1.3 us/cook, O(1), which is
+0.19% of a 1024² CUDA cook and under the jitter.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, replace
from typing import Any

import torch

logger = logging.getLogger("TEX")

from .tex_compiler.ast_nodes import SourceLoc
from .tex_runtime.interpreter import Interpreter, InterpreterError
from .tex_runtime.interp_pool import ThreadLocalInterpreterPool as _ThreadLocalInterpreterPool
from .tex_cache import get_cache
from .tex_runtime.compiled import (
    execute_compiled,
    _codegen_only_execute,
    should_stencil_route as _should_stencil_route,
)
from .tex_fusion import (
    prepare_fused as _prepare_fused,
    fused_fingerprint as _fused_fingerprint,
    strip_user_prefix as _strip_user_prefix,
)
from .tex_marshalling import (
    convert_param_value as _convert_param_value,
    infer_binding_type as _infer_binding_type,
    egress_meta as _egress_meta,
)
from .tex_runtime.host import (get_host_services, CookCancelled,
                               _cancel_check, _report_progress)
from . import tex_lazy as _tex_lazy


# ── HW-4: opt-in CPU thread pinning ──────────────────────────────────────────

def _apply_cpu_threads_env() -> None:
    """HW-4: if TEX_CPU_THREADS is set, pin torch's CPU thread count (measured 64t = 1.27x
    pointwise vs the torch default 32; fbm neutral +/-5%). Strictly OPT-IN and read once —
    NEVER auto-set: torch.set_num_threads is process-global in ComfyUI, and oversubscribing
    it harms concurrent nodes. Unset -> the thread count is left untouched."""
    val = os.environ.get("TEX_CPU_THREADS")
    if not val:
        return
    try:
        n = int(val)
        if n > 0:
            torch.set_num_threads(n)
    except Exception:
        pass


_apply_cpu_threads_env()  # HW-4: opt-in, once at import


# ── OOM plumbing (shared by the node's error mapping and ENG-2's ladder) ─────

def _is_oom_error(e: BaseException) -> bool:
    """True when *e* is an out-of-memory error.

    Prefers ComfyUI's ``model_management.is_oom`` — on recent torch, OOM can
    surface as ``torch.AcceleratorError`` (error_code==2), not just
    ``torch.cuda.OutOfMemoryError``, and ``is_oom`` also clears poisoned async
    CUDA error state. Falls back to ``OOM_EXCEPTION`` then the torch type so
    standalone runs still detect it (the Null host falls back to the torch type)."""
    return get_host_services().is_oom(e)


def _oom_in_chain(e: BaseException) -> BaseException | None:
    """The OOM error in *e*'s ``__cause__`` chain (or *e* itself), else None.

    M-1: an OOM raised inside a stdlib call is re-wrapped as ``InterpreterError``
    (``from e``), so the node's InterpreterError branch must look PAST the wrapper
    — otherwise ComfyUI's OOM handling (memory summary + model unload) never fires
    for the very allocations most likely to OOM (sample_mip/gauss_blur pyramids)."""
    seen: set[int] = set()
    cur: BaseException | None = e
    depth = 0
    while cur is not None and id(cur) not in seen and depth < 8:
        if _is_oom_error(cur):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__
        depth += 1
    return None


def _drop_tex_caches_on_oom() -> None:
    """P1-M1-FREERETRY: drop TEX's own module-level tensor caches (mip/grid/sampler
    pyramids) on OOM before re-raising, so ComfyUI's OOM handling (unload_all_models
    + execution retry) has that memory available — its unload frees resident models
    but never TEX's caches. Best-effort; the caches rebuild lazily next cook."""
    try:
        from .tex_memory import free_tensor_caches
        free_tensor_caches()
    except Exception:
        pass


# ── The auto-precision decision memo (cache #15) ─────────────────────────────

# PR-LP2: memoize the auto-precision DECISION per program (B1) so steady-state cooks skip
# the resolve_auto AST walk. Keyed by (fingerprint, resolution-bucket, device). NOTE: the
# per-cook fp16 finiteness net is NOT memoized (doc 32 C2) — trusting a first-cook verdict
# shipped NaN silently when the same program met a new input; the net runs every fp16 cook.
_AUTO_DECISION: dict = {}          # (fp, px>=min, dev_type) -> (precision, reason)
# _MIN_FP16_PX (the fp16 resolution floor) is single-sourced in precision_policy (F5, doc
# 32) — imported function-locally in prepare() so the decision-cache resolution bucket can
# never drift from the gate's actual threshold.
_AUTO_CACHE_MAX = 512              # bound (audit): a long session editing code = new fp each


def _cap_auto_caches() -> None:
    """Keep the auto-decision memo bounded — a long editing session mints a new fingerprint
    per code edit. Clear-on-overflow (not LRU): the decision recomputes cheaply next cook."""
    if len(_AUTO_DECISION) > _AUTO_CACHE_MAX:
        _AUTO_DECISION.clear()


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


# ── The cook value bundles ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecContext:
    """STR-2/STR-3: the value bundle threaded from `run()` into the tier strategies
    (`_run_torch_compile`/`_run_auto`/`_run_cuda_graph`/`_run_default`) and the shared
    `_interp_fallback` recovery path. Built once per cook so a strategy never re-touches
    prepare()'s locals; `select_tier` picks the strategy, `run()` dispatches via
    `_TIER_METHOD` and normalizes the output shape in one place."""
    program: Any
    bindings: dict
    type_map: dict
    device: Any
    code: str
    latent_channel_count: int
    output_names: list
    used_builtins: Any
    eff_precision: str  # M-3: fp32 when a LATENT input is present, else `precision`
    # STR-2: fields the accelerated/default tier strategies read (hoisted once so a
    # strategy never re-touches prepare()'s locals). `fp` is the value-independent
    # fingerprint, or None on a fused chain (which is keyed by `fused_fp` instead).
    fp: Any = None
    fused_chain: bool = False
    fused_fp: Any = None
    # ENG-7: the host's playhead for this cook — {"frame","fps","time"} or None. A VALUE,
    # never part of any key: it is deliberately absent from `fp` (the program fingerprint)
    # and from the tex_lazy memo, so an animating playhead never invalidates a compile.
    time_context: Any = None
    # P1: free VRAM as the M-1 preflight measured it, for _tile_plan to reuse. None when
    # unknown or when the preflight freed models (which makes the reading stale-low).
    free_hint: Any = None
    # ROI-3: the output sub-window (x0,y0,w,h,W,H) to cook, and the tex_roi plan (narrow
    # names + halo). Both None unless a host passed `roi=`, the program is ROI-executable,
    # and TEX_ROI_EXEC is on — then `_run_default` routes through tex_memory.run_roi. Flagged
    # off by default (no ComfyUI caller passes roi=). See docs/roi-spatial-laziness.md.
    roi: Any = None
    roi_plan: Any = None
    # SCHED-3: the host's cancel token and progress callback for this cook — pure VALUES on
    # the `time_context` model (never in any key). `cancel.check()` raises CookCancelled at a
    # yield point; `on_progress(phase, frac)` reports. Both None unless a host passed them;
    # `replace(ctx, ...)` at the fp16 re-cook preserves them automatically.
    cancel: Any = None
    on_progress: Any = None
    # DATA-1: per-binding colour/alpha/frame tags {name: BufferMeta} — a VALUE on the same
    # never-keyed channel (a tag does not move a pixel, unlike time_context). None unless a
    # host supplied it; the merged output tags surface on CookResult.out_meta.
    binding_meta: Any = None


@dataclass(frozen=True)
class CookPlan:
    """What `prepare()` resolved: the bound context, the tier that will run it, and the
    facts `run()` and the host need afterwards. Everything here is decided; `run(plan)`
    only executes."""
    ctx: ExecContext
    tier_id: str
    assigned: dict                 # output name -> TEXType (the host's egress typing)
    auto_fp16: bool                # this cook resolved auto -> fp16 (arms the C2 net)
    debug_nan_highlight: bool
    cook_px: int = 0               # H*W of the first spatial binding; MEM-2's trim reads
    #                                it in run() so the O(#bindings) scan stays once/cook
    # The _AUTO_DECISION key prepare() looked up, so the C2 net PINS under the same key
    # instead of rebuilding one that drifts (see _fp16_finiteness_net). (The P1 free-VRAM
    # hint rides ExecContext.free_hint, which is where _tile_plan reads it.)
    auto_ckey: Any = None
    # ENG-1/ENG-3: does `run()` guarantee the outputs don't alias the input bindings?
    # DEFAULTS TRUE, because that is what tex_api documents `cook()` to do and a public
    # guarantee cannot be conditional on a process-global the caller never set. A host that
    # PROVABLY materializes downstream passes False to skip the clone — today that is
    # exactly one caller, `tex_node`, whose egress clamp allocates anyway. See `prepare()`.
    disown: bool = True
    # CACHE-1: when True, run() attaches a per-output lineage key to the CookResult (the
    # identity a frame cache keys on). Off by default so the ComfyUI cook path is untouched
    # (invariant #7). `upstream_keys` are the lineage keys of this cook's tensor inputs,
    # supplied by a graph host that tracks them (empty under ComfyUI — no TEX-internal
    # upstream edge yet).
    want_lineage: bool = False
    upstream_keys: tuple = ()

    @property
    def fused_chain(self) -> bool:
        """Whether the chain ASSEMBLED. Derived from the context rather than stored, so
        the two can't drift; `tex_node` reads it between prepare() and run() to decide
        whether a stage-tagged error has a running stage to blame (Q-4)."""
        return self.ctx.fused_chain


@dataclass(frozen=True)
class CookResult:
    """What `run()` produced. `outputs` are the engine's RAW per-output tensors — no
    ComfyUI post-formatting (that is the host's egress profile, ENG-3)."""
    outputs: dict                  # {output_name: tensor}
    output_names: list
    assigned: dict                 # output name -> TEXType
    device: Any
    precision: str                 # the EFFECTIVE precision this cook ran at
    binding_names: list            # effective bindings (merged, on a fused chain)
    near_singularities: int | None = None   # C4-ux, only when the debug toggle is on
    # CACHE-1: {output_name: lineage_key} — the identity of each cooked frame, or None
    # unless the caller asked for it (prepare(want_lineage=True)). Off the default path so a
    # ComfyUI cook, which has no result cache to key, never pays for it (invariant #7).
    lineage: dict | None = None
    # DATA-1: {output_name: BufferMeta} — the merged colour/alpha/frame tags of this cook's
    # inputs, or None unless a host supplied binding_meta. A host re-attaches these to the
    # egress buffers; the default ComfyUI path supplies no tags, so this stays None.
    out_meta: dict | None = None


# ── ENG-6: zero-copy AI handoff (DLPack) ─────────────────────────────────────
# A cook output ALREADY is what a vision model wants: a device-resident, fp32,
# channels-last [B,H,W,C] image (or [B,H,W] mask). These helpers hand one to another
# framework over the DLPack protocol with no host round-trip.
#
# CONTRACT (canary-pinned, test_v023_phase1): an engine output is (1) a torch.Tensor,
# (2) fp32, (3) on the cook's device, (4) BHWC. `to_dlpack` can transpose to BCHW (the
# NCHW most models expect) as a zero-copy view.
#
# OWNERSHIP — copy=True is the DEFAULT and the safe posture. Codegen may reuse an output
# buffer (M-5 `out=`), and in the engine era an output may be a CACHED frame (CACHE-2);
# a consumer writing in place through a zero-copy view would corrupt engine state. So by
# default we hand out an OWNED contiguous copy. Pass copy=False only for a buffer you own
# and will not let the model mutate.
#
# AUTOGRAD — every cook runs under torch.inference_mode(), so a raw output carries the
# inference flag. The DLPack round-trip drops it (from_dlpack hands back an ORDINARY tensor
# over the shared memory either way), so a consumer CAN attach the result to an autograd
# graph — but for copy=False that graph would be backed by an engine buffer the next cook
# overwrites. Use copy=True (default) whenever the consumer will train through or mutate the
# tensor; copy=False is for read-only, same-tick consumption. Differentiable cooking (grad
# flowing back INTO the cook) is out of scope until the engine era.

def _owned_copy(t):
    """An owned, contiguous copy of `t` that is NOT inference-flagged (so an ML consumer
    can attach it to an autograd graph). `empty_like`+`copy_` runs outside the cook's
    inference_mode, unlike `.clone()`, which would inherit the flag."""
    src = t.contiguous()
    out = torch.empty_like(src)
    out.copy_(src)
    return out


def to_dlpack(tensor, *, layout="bhwc", copy=True):
    """Export a cooked output tensor as a DLPack capsule (ENG-6). `layout='bchw'` returns
    an NCHW-shaped view (a zero-copy permute); `copy=True` (default) first re-materializes
    an owned, contiguous, grad-ready tensor — see the ownership/autograd notes above."""
    import torch.utils.dlpack as _dl
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"to_dlpack expects a torch.Tensor, got {type(tensor).__name__}")
    t = tensor
    if layout == "bchw":
        if t.dim() != 4:
            raise ValueError(f"layout='bchw' needs a 4-D [B,H,W,C] tensor, got {tuple(t.shape)}")
        t = t.permute(0, 3, 1, 2)
    elif layout != "bhwc":
        raise ValueError(f"layout must be 'bhwc' or 'bchw', got {layout!r}")
    if copy:
        t = _owned_copy(t)
    return _dl.to_dlpack(t)


def from_dlpack(capsule):
    """Import a DLPack capsule (from torch or another framework) as a torch tensor that
    shares its memory (ENG-6). The inverse of `to_dlpack`."""
    import torch.utils.dlpack as _dl
    return _dl.from_dlpack(capsule)


# ── ENG-12: buffer ownership & immutability contract ─────────────────────────
# WHO MAY WRITE A COOKED TENSOR AFTER IT IS PRODUCED. Undefined write discipline is the
# bug class that killed Natron's engine, so it lands BEFORE any frame is cached (CACHE-2)
# and every later cross-owner edge (GRAPH-2 threads, XPU-2 frame handles) inherits it.
#
# FROZEN IS A SIGNAL, NOT A FENCE. A cook output is usually born an *inference tensor* (the
# interpreter and compiled tiers run under torch.inference_mode()), and torch RAISES on an in-place
# op on one — BUT that raise is not a rollback: on torch 2.12 (verified, CPU + CUDA) the in-place op
# LANDS THE WRITE and *then* raises. So a frozen frame handed straight back to a consumer that does
# `frame.clamp_(...)` is silently corrupted (the write took) even though the op "failed", and a
# version stamp can't catch it (frame_version of an inference tensor is a constant 0). Freezing is
# therefore a loud tripwire for a cache-INTERNAL mistake, never a guarantee against a CONSUMER write.
# (This is also why the born-frozen floor is not universal anyway: the cuda_graph replay hands back
# a normal .clone(), the debug paint a normal tensor.)
#
# THE CACHE CONTRACT (CACHE-2) — the real guarantee is COPY-ON-READ, mirroring to_dlpack(copy=True):
#   * put() stores the frame frozen (the canonical master — freezing keeps the cache's own code from
#     scribbling it, and torch's raise surfaces such a bug loudly).
#   * get()/_restore return an OWNED COPY by default (frame.clone() — a normal, mutable, independent
#     buffer), so a consumer's in-place write can never reach the stored master. copy=False is the
#     opt-in fast path for a read-only consumer that promises not to mutate (same posture as
#     to_dlpack(copy=False) / a `.data` alias — the caller opting out, not a hole).
#   * verify_unmutated (stratum 2, torch's t._version) is a live defense only for a NORMAL
#     (host-supplied, non-frozen) entry — a bumped counter drops it; it is inert for a frozen master
#     (which is why copy-on-read, not the version stamp, is what protects that master).
# _disown_inputs (input-alias net) and to_dlpack's copy=True default uphold the same ownership at
# egress.
#
# M-5 out= reuse (codegen, a DO-NOT-TOUCH) can NEVER target a cached frame: its reuse set is
# codegen _tN arithmetic temps only — never a binding — and a cached frame can only re-enter
# a cook AS an input binding. The one residual write into a binding, a scatter, is already
# COW-guarded (interpreter _scatter_owned / codegen _scat_owned: clone-before-first-write).
# See docs/results-caching.md and the AGENTS.md DO-NOT-TOUCH register.

def is_frozen(t) -> bool:
    """True if `t` is an inference tensor — one torch RAISES on for an in-place op. Note that on
    torch 2.12 the raise does not roll back the write (the op lands, then raises), so "frozen" is a
    loud tripwire, not a write fence; CACHE-2's actual consumer-facing guarantee is copy-on-read."""
    return isinstance(t, torch.Tensor) and t.is_inference()


def frame_version(t) -> int:
    """ENG-12 version stamp: torch's in-place mutation counter `t._version` for a normal tensor,
    else a constant 0 for a frozen (inference) tensor — reading _version on one would itself raise.
    Mirrors stdlib._safe_version. Because it is constant for a frozen tensor, verify_unmutated is a
    live mutation-detector only for NORMAL entries (a frozen master is protected by copy-on-read)."""
    return 0 if (not isinstance(t, torch.Tensor) or t.is_inference()) else t._version


def verify_unmutated(t, stamp) -> bool:
    """ENG-12 re-entry check: True iff `t`'s in-place mutation counter is unchanged since `stamp`.
    A live detector for a NORMAL entry (a bumped counter -> drop it); always True for a frozen
    master (stamp is a constant 0), which the cache protects by copy-on-read instead."""
    return frame_version(t) == stamp


def frozen_copy(t):
    """An immutable (inference-flagged) copy of `t`: any in-place write to the RESULT raises.
    Made inside inference_mode so the clone carries the inference flag even when `t` is a
    normal tensor — the exact inverse of _owned_copy (which strips the flag for autograd).
    Use to store a tamper-PROOF frame when the source is a normal (mutable) buffer."""
    with torch.inference_mode():
        return t.clone()


def freeze(t):
    """Idempotent hard-freeze: return `t` unchanged if it is already frozen (immutable),
    else a frozen_copy. The one call a frame cache uses to guarantee an entry cannot be
    written through, whatever the provenance of the tensor handed to it."""
    return t if is_frozen(t) else frozen_copy(t)


# ── Debug overlays (DBG-3 magenta NaN, C4-ux cyan near-singularity) ──────────

def _is_channels_last_image(t):
    """A channels-last IMAGE ([B,H,W,C], C<=4). The C<=4 guard avoids mis-reading a
    non-channels-last tensor ([B,C,H,W]) as a giant channel dim."""
    return isinstance(t, torch.Tensor) and t.is_floating_point() and t.dim() >= 4 \
        and t.shape[-1] <= 4


def _paint_pixels(t, pixel_mask_keepdim, rgba):
    """Paint the image pixels selected by `pixel_mask_keepdim` ([...,H,W,1]) a constant
    RGBA (clipped to the channel count). The single home of the debug pixel-paint contract,
    shared by DBG-3's magenta NaN flag and C4-ux's cyan near-singularity flag."""
    color = torch.tensor(list(rgba)[:t.shape[-1]], dtype=t.dtype, device=t.device)
    return torch.where(pixel_mask_keepdim, color, t)


def _nan_highlight(t):
    """DBG-3: paint every NaN/Inf PIXEL magenta so non-finite output is obvious. For a
    [B,H,W,C] image, any non-finite channel magenta-flags the whole pixel; for a
    [B,H,W] mask (or other), non-finite elements are set to 1.0. A no-op (returns the
    input) for a finite tensor or a non-tensor — the caller only calls it when the
    toggle is on, so a finite cook pays one isfinite reduction and nothing else."""
    if not (isinstance(t, torch.Tensor) and t.is_floating_point()):
        return t
    finite = torch.isfinite(t)
    if bool(finite.all()):
        return t
    if _is_channels_last_image(t):
        return _paint_pixels(t, (~finite).any(dim=-1, keepdim=True), (1.0, 0.0, 1.0, 1.0))
    return torch.where(finite, t, torch.ones_like(t))


def _singularity_highlight(t, pix_mask):
    """C4-ux: paint pixels where a guarded division hit the epsilon branch CYAN (0,1,1) —
    distinct from DBG-3's magenta NaN. `pix_mask` is guard_trace's accumulated per-pixel
    boolean. A no-op for a non-image tensor, an empty mask, or a shape that won't align
    (a diagnostic must never break a cook). Applied only when debug_nan_highlight is on."""
    if pix_mask is None or not _is_channels_last_image(t):
        return t
    try:
        m = pix_mask
        while m.dim() < t.dim() - 1:   # align rank to [..., H, W]
            m = m.unsqueeze(0)
        m = m.unsqueeze(-1)            # -> [..., H, W, 1] to broadcast over channels
        if m.shape[-2] != t.shape[-2] or m.shape[-3] != t.shape[-3]:
            return t                   # spatial dims disagree — can't localise safely
        return _paint_pixels(t, m, (0.0, 1.0, 1.0, 1.0))
    except Exception:
        return t


# ── Tier selection + the tier strategies ─────────────────────────────────────

def _interp_fallback(ctx: ExecContext, *, reset_dynamo: bool, pass_precision: bool):
    """STR-2: the single copy of the tier→interpreter recovery path, previously
    duplicated in the torch_compile / auto / cuda_graph branches. The two flags
    reproduce each branch's *exact* original call:
      - torch_compile: reset_dynamo=True,  pass_precision=False
      - auto:          reset_dynamo=True,  pass_precision=True
      - cuda_graph:    reset_dynamo=False, pass_precision=False
    (dynamo state is process-global — see compiled.py — so a failed compile/auto
    cook resets it on THIS thread before retrying; the graph path never touched
    dynamo, so it does not reset.)"""
    _cancel_check(ctx.cancel)   # SCHED-3 yield C: a tier failed — don't fall back into a stale cook
    if reset_dynamo:
        try:
            torch._dynamo.reset()
        except Exception:
            pass
    interp = _get_interpreter()
    kw = dict(source=ctx.code, latent_channel_count=ctx.latent_channel_count,
              output_names=ctx.output_names, used_builtins=ctx.used_builtins,
              time_context=ctx.time_context,
              cancel=ctx.cancel, on_progress=ctx.on_progress)  # SCHED-3: token survives the fallback
    if pass_precision:
        kw["precision"] = ctx.eff_precision
    return interp.execute(ctx.program, ctx.bindings, ctx.type_map,
                          device=ctx.device, **kw)


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

def _run_torch_compile(ctx: ExecContext):
    # Fused chains are keyed by their chain fingerprint (ctx.fp is None there).
    _fp = ctx.fused_fp if ctx.fused_chain else ctx.fp
    try:
        return execute_compiled(ctx.program, ctx.bindings, ctx.type_map, ctx.device,
                                _fp, latent_channel_count=ctx.latent_channel_count,
                                output_names=ctx.output_names, used_builtins=ctx.used_builtins,
                                time_context=ctx.time_context)
    except Exception as compile_exc:
        # Defense in depth: torch_compile must NEVER hard-fail the node.
        logger.warning("[TEX] torch_compile path failed (%s); using interpreter.",
                       compile_exc)
        return _interp_fallback(ctx, reset_dynamo=True, pass_precision=False)


def _run_auto(ctx: ExecContext):
    # Fused chains are keyed by their chain fingerprint (ctx.fp is None there).
    _fp = ctx.fused_fp if ctx.fused_chain else ctx.fp
    try:
        from .tex_runtime.compiled import run_auto
        return run_auto(ctx.program, ctx.bindings, ctx.type_map, ctx.device, _fp,
                        latent_channel_count=ctx.latent_channel_count,
                        output_names=ctx.output_names, used_builtins=ctx.used_builtins,
                        precision=ctx.eff_precision, time_context=ctx.time_context)
    except Exception as auto_exc:
        logger.warning("[TEX] auto tier failed (%s); using interpreter.", auto_exc)
        return _interp_fallback(ctx, reset_dynamo=True, pass_precision=True)


def _run_cuda_graph(ctx: ExecContext):
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
        logger.warning("[TEX] cuda_graph path failed (%s); using interpreter.", _g_exc)
        out = None
    if out is None:
        return _interp_fallback(ctx, reset_dynamo=False, pass_precision=False)
    return out


def _run_default(ctx: ExecContext):
    # ROI-3: cook only the requested sub-window (interpreter tier, flagged off — set only
    # when a host passed `roi=`, the program is ROI-executable, and TEX_ROI_EXEC is on). The
    # narrow-cook-crop is bit-exact for pointwise/morphology, ~1 ulp for conv. Whole-frame on
    # any run_roi error (never hard-fail the cook).
    if ctx.roi is not None and ctx.roi_plan is not None:
        try:
            from .tex_memory import run_roi
            return run_roi(_get_interpreter(), ctx.program, ctx.bindings, ctx.type_map,
                           ctx.device, ctx.latent_channel_count, ctx.output_names,
                           ctx.used_builtins, ctx.eff_precision, ctx.roi,
                           ctx.roi_plan.narrow, ctx.roi_plan.halo, ctx.time_context,
                           cancel=ctx.cancel, on_progress=ctx.on_progress)
        except CookCancelled:
            raise                       # SCHED-3: a cancel aborts — never fall back to whole-frame
        except Exception as _roi_exc:
            logger.warning("[TEX] ROI cook failed (%s); running whole-frame.", _roi_exc)
    # UC-2: default-route an exact (fetch/conv) stencil through the codegen tier
    # (avg_pool2d/conv2d/unfold). _codegen_only_execute self-falls-back; the outer
    # guard covers env-build edge cases so this can never hard-fail the node.
    if not ctx.fused_chain:
        try:
            if _should_stencil_route(ctx.fp, ctx.program):
                return _codegen_only_execute(
                    ctx.program, ctx.bindings, ctx.type_map, ctx.device,
                    latent_channel_count=ctx.latent_channel_count,
                    output_names=ctx.output_names,
                    used_builtins=ctx.used_builtins, fingerprint=ctx.fp,
                    time_context=ctx.time_context)
        except Exception as _stencil_exc:
            logger.warning("[TEX] stencil codegen route failed (%s); using "
                           "interpreter.", _stencil_exc)
    interp = _get_interpreter()
    # M-4: under GPU memory pressure, run a tile-safe program in horizontal strips
    # (peak transient ~1/n). Falls back to the whole-image cook on any strip error.
    n_strips = (_tile_plan(ctx.program, ctx.bindings, ctx.device, ctx.latent_channel_count,
                           2 if ctx.eff_precision == "fp16" else 4, ctx.fp,
                           free_hint=ctx.free_hint)
                if not ctx.fused_chain else None)
    if n_strips:
        try:
            from .tex_memory import run_tiled
            return run_tiled(interp, ctx.program, ctx.bindings, ctx.type_map, ctx.device,
                             ctx.latent_channel_count, ctx.output_names, ctx.used_builtins,
                             ctx.eff_precision, n_strips, ctx.time_context,
                             cancel=ctx.cancel, on_progress=ctx.on_progress)
        except CookCancelled:
            raise                       # SCHED-3: a cancel aborts — never fall back to untiled
        except Exception as _tile_exc:
            logger.warning("[TEX] tiled cook failed (%s); running untiled.", _tile_exc)
    elif not ctx.fused_chain:
        # ROI-5: `_tile_plan` refused (a non-pixel-local program — a blur/morphology), but a
        # BOUNDED-halo op can still tile with a grown strip. Under memory pressure OR the TDR
        # time cap, cook it in halo strips (an 8K gauss_blur that could not tile at all before).
        # `_halo_tile_plan` cheap-gates so a small default cook returns before any real work.
        halo_plan = _halo_tile_plan(ctx.program, ctx.code, ctx.bindings, ctx.device,
                                    ctx.latent_channel_count,
                                    2 if ctx.eff_precision == "fp16" else 4, ctx.fp,
                                    ctx.free_hint, ctx.eff_precision)
        if halo_plan:
            n_h, narrow_names, halo = halo_plan
            try:
                from .tex_memory import run_tiled_halo
                return run_tiled_halo(interp, ctx.program, ctx.bindings, ctx.type_map, ctx.device,
                                      ctx.latent_channel_count, ctx.output_names, ctx.used_builtins,
                                      ctx.eff_precision, n_h, narrow_names, halo, ctx.time_context,
                                      cancel=ctx.cancel, on_progress=ctx.on_progress)
            except CookCancelled:
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


def _fp16_finiteness_net(raw_output, auto_fp16, ctx, tier_id, auto_ckey=None):
    """C2 (extracted from execute, C1-st): the residual data-dependent safety net
    for `precision="auto"`. Every auto->fp16 cook's output is checked for NaN/Inf
    (the case the static gate can't rule out — a blind spot, or a pow going
    negative). On non-finite: discard the fp16 probes, re-cook fp32, and PIN the
    auto decision to fp32 so the program skips fp16 for all future inputs. Returns
    `(output_dict, effective_precision)`; a no-op (returning the original precision) when the
    cook wasn't auto->fp16 or stayed finite.

    `auto_ckey` is the memo key prepare() actually LOOKED UP, handed over rather than
    rebuilt here. Rebuilding it is what broke the pin: v0.22's B3 fix added compile_mode to
    the reader's key and left this writer at the old 3-tuple, and a 3-tuple can never match
    a 4-tuple lookup — so "pin the program to fp32 thereafter" never read back. Silently,
    because nothing about correctness depends on it (the net re-runs every cook), only the
    cost: every such program double-cooks forever. Passing the key deletes the second
    construction site rather than re-synchronising it."""
    # Returns (raw_output, effective_precision): the SECOND element is the precision the
    # returned frame was ACTUALLY cooked at — "fp32" when the net re-cooked, else the original
    # eff_precision. The caller stamps CookResult.precision AND the lineage key from it, or a
    # re-cooked-fp32 frame would be mislabelled fp16 and keyed under fp16 (a false cache miss
    # against the auto-pinned-fp32 cook of the same inputs, and a lie to a host reading .precision).
    if not (auto_fp16 and ctx.eff_precision == "fp16"):
        return raw_output, ctx.eff_precision
    if not any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
               for v in raw_output.values()):
        return raw_output, ctx.eff_precision
    # F1 fix: tier_trace is imported function-locally per method (the SCC convention);
    # the C1-st extraction moved this block out of execute() without carrying the import,
    # so the recovery path NameError-crashed exactly when auto-fp16 overflowed.
    from .tex_runtime import tier_trace
    tier_trace.clear_probes()  # discard the fp16 cook's probes (no dup)
    tier_trace.record_precision("fp32", "auto: fp16 non-finite -> fp32 (pinned)")
    if auto_ckey is not None:
        _AUTO_DECISION[auto_ckey] = ("fp32", "auto: fp16 non-finite -> fp32 (pinned)")
    _cancel_check(ctx.cancel)   # SCHED-3 yield D: don't climb into an fp32 re-cook if abandoned
    return _run_tier(replace(ctx, eff_precision="fp32"), tier_id), "fp32"


# ── Memory planning ──────────────────────────────────────────────────────────

def _tile_plan(program, bindings: dict[str, Any], device,
               latent_channel_count: int = 0, dtype_bytes: int = 4,
               fingerprint: str | None = None,
               free_hint: float | None = None) -> int | None:
    """M-4: strip count if the cook should be tiled (tile-safe + under memory
    pressure), else None. cuda only; needs the host's free-memory query.
    MEM-3: dtype_bytes=2 in fp16 mode halves the peak estimate (a fp16 cook that
    fits shouldn't be tiled as if it were fp32).

    P1: `free_hint` is the free-VRAM reading prepare()'s M-1 preflight already bought. On
    a default CUDA cook these two sites were the ONLY callers and ran microseconds apart
    with nothing allocating between them, yet each paid for its own query — measured at
    2 x ~68 us, ~42% of a 345 us 256² cook, ~94% of it in the query rather than the
    estimator. The caller passes None when the reading is stale (the preflight asked the
    host to unload) or absent, and then we buy our own.

    The ESTIMATE is deliberately NOT hinted, though it looks like the same redundancy: the
    two sites call it with different `dtype_bytes`. The preflight runs before `auto`
    resolves, so it passes 4; here, post-resolution, an auto->fp16 cook passes 2 — reusing
    the preflight's number would hand this function a 2x-inflated peak and over-tile
    exactly the cooks `auto` accepted (measured 67108864 vs 33554432 at 2048²). It is also
    only ~6% of the cost."""
    if not str(device).startswith("cuda"):
        return None  # host.get_free_memory returns None off a host → no tiling
    # M-4 safety: never tile a LATENT ([B,C,H,W] — dim 1 is channels, not
    # height) or a cook whose spatial bindings disagree on height (they can't
    # be co-tiled). run_tiled re-checks, but planning here avoids a bogus
    # peak estimate off the wrong axis.
    if latent_channel_count:
        return None
    try:
        from .tex_memory import is_tile_safe_cached, estimate_peak_bytes, shared_tile_height
        if not is_tile_safe_cached(program, fingerprint):  # P4: memoized per fingerprint
            return None
        H = shared_tile_height(bindings)
        if H is None:
            return None
        spatial = None
        for v in bindings.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == H:
                spatial = (v.shape[0], v.shape[1], v.shape[2])
                break
        if spatial is None:
            return None
        est = estimate_peak_bytes(program, spatial, dtype_bytes, fingerprint)  # P4/LAT-2: memoized walk
        free = (free_hint if free_hint is not None
                else get_host_services().get_free_memory(torch.device(device)))
        if not free or est <= 0:
            return None
        budget = 0.25 * free
        if est <= budget:
            return None  # no pressure — don't pay the launch tax
        n = math.ceil(est / budget)
        max_strips = max(1, spatial[1] // 64)  # ≥64-row strip floor
        n = min(n, max_strips)
        return n if n >= 2 else None
    except Exception:
        return None


# ROI-5/WDDM: a per-strip cook-time ceiling. A display GPU's driver resets (TDR) any kernel
# that runs past ~2 s, killing the cook — so the strip planner caps estimated per-STRIP time,
# not just bytes. ~1.8 s leaves headroom under the 2 s watchdog.
_TDR_BUDGET_MS = 1800.0


def _tdr_strip_floor(fingerprint, spatial, precision, device) -> int:
    """ROI-5: the minimum strip count that keeps each strip's estimated cook time under the ~2 s
    WDDM TDR watchdog, derived from autotier's persisted WHOLE-FRAME median (BlinkScript's driver
    timeouts are this failure, un-planned-for). Best-effort: 0 when the program was never measured
    on this device (the very first big cook can't be pre-timed — the reach of persisted medians)."""
    if not str(device).startswith("cuda"):
        return 0
    try:
        from .tex_runtime import autotier
        whole_ms = autotier.cook_ms(autotier.make_key(fingerprint, "cuda", precision, spatial))
        if whole_ms and whole_ms > _TDR_BUDGET_MS:
            return int(math.ceil(whole_ms / _TDR_BUDGET_MS))
    except Exception:
        pass
    return 0


def _scalar_params(bindings) -> dict:
    """The foldable scalar/bool/int params of a binding set — what `tex_roi`/`tex_lazy` fold to
    resolve halo radii and dead branches. Shared by the halo planner and the OOM ladder (the
    default `prepare()` param loop keeps its own inline copy to stay off a per-cook call frame)."""
    return {n: v for n, v in bindings.items() if isinstance(v, (bool, int, float))}


def _halo_tile_plan(program, code, bindings, device, latent_channel_count, dtype_bytes,
                    fingerprint, free_hint, precision):
    """ROI-5: `(n_strips, narrow_names, halo)` when a NON-tile-safe program is HALO-tileable — a
    bounded direct-tensor neighbourhood op (blur / erode / dilate), which `tex_roi.roi_plan`
    reports executable with a positive cook halo — and either memory pressure OR the TDR time cap
    calls for strips; else None. This is the class `is_tile_safe` refuses, so it never overlaps
    the pointwise `_tile_plan`. cuda + non-latent only (fused chains are excluded by the caller).

    CHEAP-GATED for invariant #7: a small cook returns BEFORE `roi_plan`, `_scalar_params`, and the
    free-VRAM driver query. `est` is a memo hit (the M-1 preflight already walked it this cook),
    `device_total_mem` is cached, and `_tdr_strip_floor` is a dict lookup — so a program a small
    fraction of VRAM with no TDR-risk median can't need tiling and never pays the ~61 µs
    `get_free_memory`, exactly mirroring `_preflight_memory`'s `est < total // 8` skip."""
    if latent_channel_count or not str(device).startswith("cuda"):
        return None
    try:
        # Hot-path bail FIRST: a POINTWISE (tile-safe) program reaches here whenever `_tile_plan`
        # found no pressure, but halo tiling is only for the NON-tile-safe blur/morphology class (a
        # tile-safe program's roi_plan has halo 0 → None below anyway). The memoized tile-safe check
        # lets a pointwise cook skip roi_plan/estimate/the dim scans AND the imports below entirely —
        # this runs on every default pointwise CUDA cook (invariant #7).
        from .tex_memory import is_tile_safe_cached
        if is_tile_safe_cached(program, fingerprint):
            return None
        from .tex_memory import (shared_tile_height, shared_tile_width, estimate_peak_bytes,
                                 device_total_mem)
        H = shared_tile_height(bindings)
        W = shared_tile_width(bindings)
        if H is None or W is None:
            return None
        # Batch of the height-H anchor image — guaranteed present, since `shared_tile_height`
        # derived `H` from exactly such a tensor (a StopIteration if that ever breaks is caught by
        # the outer guard → None). The cook is sized off the SHARED height/width (both skip
        # broadcast singletons), never the anchor's own width — a [B,H,1] companion bound first
        # would otherwise under-size `est` by a factor of W and silently zero the TDR px-bucket.
        batch = next(v.shape[0] for v in bindings.values()
                     if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == H)
        spatial = (batch, H, W)
        est = estimate_peak_bytes(program, spatial, dtype_bytes, fingerprint)   # memoized walk
        tdr_floor = _tdr_strip_floor(fingerprint, spatial, precision, device)   # dict lookup
        total = device_total_mem(device)                                       # cached, no query
        if tdr_floor < 2 and total and 0 < est < total // 8:
            return None    # not big enough for memory pressure, not TDR-risky → skip (no free query)
        from . import tex_roi
        plan = tex_roi.roi_plan(code, _scalar_params(bindings))
        if not plan.executable or plan.halo <= 0 or not plan.narrow:
            return None
        halo = plan.halo
        # Strip floor: ≥ 4·halo rows/strip so the grown-halo overhead stays under ~50%.
        max_strips = max(1, H // max(64, 4 * halo))
        n_mem = 0
        free = (free_hint if free_hint is not None
                else get_host_services().get_free_memory(torch.device(device)))
        if free and est > 0 and est > 0.25 * free:
            n_mem = math.ceil(est / (0.25 * free))
        n = min(max(n_mem, tdr_floor), max_strips)
        return (n, plan.narrow, halo) if n >= 2 else None
    except Exception:
        return None


def _preflight_memory(program, bindings: dict[str, Any], device,
                      dtype_bytes: int = 4, fingerprint=None) -> float | None:
    """M-1: if the estimated cook peak exceeds free VRAM, free resident
    models first (best-effort; never raises). MEM-3: dtype_bytes=2 for an explicit
    fp16 cook (auto is still unresolved here, so it stays the conservative 4).

    Returns the free-VRAM reading it bought, as a `free_hint` for `_tile_plan` to reuse
    instead of buying a second query microseconds later (P1). Returns None whenever
    `_tile_plan` must query for itself — either the reading is unknown/uncomputed, OR the
    preflight just asked the host to unload models, which raises true free and makes the
    number STALE-LOW. Collapsing "unknown" and "stale" into a single None is exactly what
    the caller wants: both mean "don't trust this, re-read."

    LAT-2: `fingerprint` memoizes the peak-estimate AST walk (see estimate_peak_bytes), and
    the estimate then gates whether the free-VRAM query below runs at all."""
    host = get_host_services()  # PORT-1: a host with no free-memory answer → this no-ops
    try:
        spatial = None
        for v in bindings.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 3:
                spatial = (v.shape[0], v.shape[1], v.shape[2])
                break
        if spatial is None:
            return None
        from .tex_memory import estimate_peak_bytes, device_total_mem
        est = estimate_peak_bytes(program, spatial, dtype_bytes, fingerprint)
        if est <= 0:
            return None
        # LAT-2: `host.get_free_memory` below is a live driver query measured at ~61 us — 92%
        # of a 256² prepare(), on EVERY CUDA cook. Its only effect is to unload models when
        # free < est + headroom, which cannot happen when the estimate is a small fraction of
        # total VRAM and free is gigabytes. So when est is under total//8 (mirrors
        # cache_budget_bytes's VRAM fraction), skip the query AND the pre-unload entirely:
        # the rare genuine-pressure case is still caught by ComfyUI's host OOM ladder
        # (unload_all_models + retry) and the engine's own ENG-2 strip-retry ladder.
        total = device_total_mem(device)
        if total and est < total // 8:
            return None
        dev_t = torch.device(device)
        free = host.get_free_memory(dev_t)
        headroom = 128 * 1024 * 1024
        if free is not None and free < est + headroom:
            host.free_memory(est + 256 * 1024 * 1024, dev_t)
            return None            # stale-low after the unload — make _tile_plan re-query
        return free
    except Exception:
        return None


def resolve_device(device_mode: str, bindings: dict[str, Any]) -> str:
    """
    Resolve the target execution device.

    "auto" — infer from inputs (prefer GPU if any input is on GPU, else CPU)
    "cpu"  — force CPU execution
    "cuda" — force GPU execution (raises if CUDA unavailable)
    """
    if device_mode == "cpu":
        return "cpu"

    if device_mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TEX Error: device='cuda' selected but CUDA is not available. "
                "Select 'auto' or 'cpu' instead."
            )
        return "cuda"

    # "auto" mode: prefer GPU if any input tensor is on GPU
    # (latent dicts are already unwrapped to tensors before this is called)
    for val in bindings.values():
        if isinstance(val, torch.Tensor) and val.is_cuda:
            return str(val.device)
    return "cpu"


# ── The cook ─────────────────────────────────────────────────────────────────

MAX_OUTPUTS = 8   # the engine's own ceiling; the ComfyUI node's socket count matches it


def prepare(code: str, bindings: dict, *, chain_payload: Any = None,
            device_mode: str = "auto", compile_mode: str = "none",
            precision: str = "fp32", has_latent_input: bool = False,
            latent_channel_count: int = 0, forgive_dead_refs: bool = False,
            debug_nan_highlight: bool = False, time_context: dict | None = None,
            max_outputs: int = MAX_OUTPUTS, disown: bool = True,
            roi: tuple | None = None, want_lineage: bool = False,
            upstream_keys: tuple = (), cancel=None, on_progress=None,
            binding_meta: dict | None = None) -> CookPlan:
    """Resolve everything a cook needs *without running it*: compile (or splice a fused
    chain), resolve the device, gate the outputs and references, fold params, resolve
    `precision="auto"`, and select the tier.

    `bindings` is consumed and MUTATED (param defaults injected, values converted) —
    the caller hands over ownership, exactly as execute() did with its own dict.

    `forgive_dead_refs`: the host may have skipped cooking statically-dead inputs (the
    ComfyUI lazy pool), so a reference the optimizer will drop must not raise E6003.
    Off by default: a host that cooks everything wants the loud "not connected" error.

    `disown` (ENG-1/ENG-3): guarantee the outputs don't alias the input bindings — see
    `_disown_inputs`. ON by default, because `tex_api` publishes that guarantee for
    `cook()` and a guarantee with a precondition the docs don't state is not one. It was
    gated on the process-global egress profile until it was noticed that the global cannot
    answer the question: it describes what a host set, not what THIS call will do with the
    result, and `cook()`'s own contract is to return RAW tensors with no profile applied at
    all. So it is a parameter, and the caller — which is the only party that knows —
    decides.

    Pass False ONLY if you can prove your own egress already materializes. `tex_node` can:
    its clamp allocates a fresh tensor on the way out, which is the accident that kept the
    ComfyUI path correct before any of this existed. Cloning first and clamping second
    allocates the clone, copies into it, and drops it — measured +0.236 ms at 2048²
    passthrough, +118% of egress, on the DEFAULT path, in a release whose whole claim is
    +1.3 us/cook. Invariant #7 is not a formality; neither is the guarantee, so the cost
    lands on whoever can prove they don't need it, not on everyone by default.
    """
    # fp16 is an interpreter-only mode for now — the compile/graph paths bake precision
    # into their keys but aren't validated for fp16 yet. Which precisions a tier supports
    # is engine policy, so the clamp lives here rather than in each host (it sat in
    # tex_node until v0.22, which left `cook()` free to feed fp16 into a compiled tier).
    # Must precede _preflight_memory below, whose dtype_bytes reads the CLAMPED value.
    # The `auto`-resolved twin of this rule is in the precision block further down.
    if precision == "fp16" and compile_mode != "none":
        precision = "fp32"
    # CACHE-1: the interpreter reads the playhead by duck-typing (`time_context.get(name)`, any
    # Mapping), but the lineage keyer type-checks it — so a Mapping-but-not-dict playhead (a
    # MappingProxyType read-only view a host might hand out) would drive the pixels yet fall out
    # of the key, a silent stale serve. Normalize to a plain dict ONCE here so both consumers see
    # the same object. (The ComfyUI adapter already passes a plain dict or None.)
    if time_context is not None and not isinstance(time_context, dict) \
            and hasattr(time_context, "items"):
        time_context = dict(time_context)
    fused_chain = False
    fused_fp = None
    # LAT-2: the program fingerprint, hoisted above the M-1 preflight so it can memoize its
    # peak-estimate walk. Stays None for a fused chain (which is keyed by `fused_fp` instead).
    fp = None
    if chain_payload:
        # Fused path: splice the whole linked chain into one program.
        # Per-stage validation already happened in compile_fused, so a
        # failure here raises (we must NOT silently run the terminal
        # alone — the upstream nodes were collapsed away in the prompt).
        spec = json.loads(chain_payload) if isinstance(chain_payload, str) else chain_payload
        # Q-1 / v0.20: the fused fingerprint (value-independent) is the KEY
        # under which select_tier admits a fused chain to the accelerated
        # tiers — a CUDA-graph capture unit (cuda_graph) OR a torch.compile/
        # auto artifact. So it must be computed for EVERY compiling mode, not
        # cuda_graph alone: without it select_tier strands a fused chain on the
        # interpreter under torch_compile/auto (the measured inductor win is
        # then unreachable from the node). Computed from the original bindings
        # before _prepare_fused merges them; value-independent + memoized (see
        # tex_fusion.fused_fingerprint), so cheap on every mode.
        if compile_mode in ("cuda_graph", "torch_compile", "auto"):
            fused_fp = _fused_fingerprint(spec, code, bindings, _infer_binding_type)
        (program, type_map, referenced, assigned_bindings, param_info,
         used_builtins, bindings) = _prepare_fused(spec, code, bindings, _infer_binding_type)
        fused_chain = True
    else:
        # Infer binding types for inputs
        binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}

        # Compile (uses two-tier Mega-Cache: memory LRU + disk persistence)
        cache = get_cache()
        program, type_map, referenced, assigned_bindings, param_info, used_builtins = \
            cache.compile_tex(code, binding_types)
        # LAT-2: compute the fingerprint here (memoized in TEXCache, so this is the SAME
        # single call that used to sit below the preflight — value-identical, since
        # binding_types is snapshotted before param injection) so _preflight_memory can
        # memoize its peak-estimate AST walk on it.
        fp = cache.fingerprint(code, binding_types)

    # Resolve target device (from the effective bindings — merged for a chain)
    device = resolve_device(device_mode, bindings)

    # M-1: preflight — estimate this cook's peak and, if it exceeds free
    # VRAM, ask the host to free resident models first (what model loaders
    # do). Prevents a big cook OOMing while GBs sit locked in models. LAT-2:
    # the ~61 us free-VRAM query is skipped when the estimate is a small
    # fraction of total VRAM (the common case), so the common path is
    # near-free. cuda only.
    free_hint = None
    if str(device).startswith("cuda"):
        free_hint = _preflight_memory(
            program, bindings, device,
            2 if precision == "fp16" and not has_latent_input else 4,
            fingerprint=fp)

    # Determine output names (sorted alphabetically for stable ordering)
    output_names = sorted(assigned_bindings.keys())
    if not output_names:
        raise InterpreterError(
            "TEX program has no outputs. Assign to @OUT or another @name.",
            loc=SourceLoc(1, 1), source=code, code="E6001",
        )
    if len(output_names) > max_outputs:
        raise InterpreterError(
            f"TEX program has {len(output_names)} outputs, "
            f"exceeding the maximum ({max_outputs}).",
            loc=SourceLoc(1, 1), source=code, code="E6002",
        )

    # Check that referenced input bindings are available.
    # For $params with code-defined defaults, inject the default as a
    # fallback when no widget value or wire connection is provided.
    # Lazy gate: `referenced` is the pre-optimization set, so it still
    # contains names whose only uses are statically dead given the
    # current params — exactly the inputs check_lazy_status skipped.
    # Recompute the live set (memo hit: same code + same scalar values
    # the lazy round saw) and forgive dead references instead of
    # raising E6003. Single programs only; fused chains keep v0.17
    # behaviour (their lazy round requests everything).
    lazy_needed = None
    if not fused_chain and forgive_dead_refs:
        scalar_params = {n: v for n, v in bindings.items()
                         if isinstance(v, (bool, int, float))}
        lazy_needed = _tex_lazy.lazy_required_bindings(code, scalar_params)
    for ref_name in referenced:
        if ref_name not in assigned_bindings and ref_name not in bindings:
            if lazy_needed is not None and ref_name not in lazy_needed:
                continue  # statically dead under the current params
            # Try param default before erroring
            if ref_name in param_info:
                default_val = param_info[ref_name].get("default_value")
                if default_val is not None:
                    bindings[ref_name] = default_val
                    continue
            sigil = "$" if ref_name in param_info else "@"
            # On the fused path user bindings are stage-prefixed
            # (_s0_u_amt — see tex_fusion._user_prefix); show the
            # user's original name, not the synthetic one.
            disp = _strip_user_prefix(ref_name) if fused_chain else ref_name
            raise InterpreterError(
                f"TEX code references {sigil}{disp} but no input is connected to slot '{disp}'.",
                loc=SourceLoc(1, 1), source=code, code="E6003",
            )

    # Convert param widget values to appropriate types for the interpreter
    # (e.g. hex color strings → RGB lists, comma vec strings → float lists)
    for pname, pinfo in param_info.items():
        if pname in bindings:
            bindings[pname] = _convert_param_value(bindings[pname], pinfo, pname)

    # STR-2/STR-3: bundle every arg the tiers need once (fp hoisted here so no
    # strategy re-touches prepare()'s locals), select the tier (pure), dispatch
    # to its strategy, and normalize the output shape in ONE place.
    # PR-LP2: resolve precision="auto" now that the program + resolution are
    # known — fp16 only in the measured win region (CUDA, >=1024^2, pointwise,
    # no image-derived threshold); fp32 otherwise. Record the decision + reason
    # (tier_trace) so it's never silent. fp16 stays out of the compiled/graph
    # tiers this cycle (mirrors the compile-mode fp32 force in the host).
    # DBG-1: clear the tier/precision trace so the HUD payload reflects
    # THIS cook (the trace is thread-local and persists across cooks).
    from .tex_runtime import tier_trace, guard_trace
    tier_trace.reset()
    # C4-ux: arm the near-singularity trace ONLY with the debug toggle (guard hooks
    # are a no-op otherwise — the zero-cost-when-off contract). Set the state
    # explicitly every cook so a prior debug cook that threw can't leave it armed.
    guard_trace.arm() if debug_nan_highlight else guard_trace.disarm()
    # `fp` was computed above (non-fused branch), hoisted so the M-1 preflight could reuse it
    # (LAT-2); it is None for a fused chain, which is keyed by `fused_fp`.
    # cook resolution (H*W of the first spatial binding) — used by the auto gate
    # AND the MEM-2 downshift trim; computed ONCE (audit: was scanned twice/cook).
    cook_px = next((v.shape[1] * v.shape[2] for v in bindings.values()
                    if isinstance(v, torch.Tensor) and v.dim() >= 3), 0)
    auto_fp16 = False
    auto_ckey = None
    if precision == "auto":
        if has_latent_input:
            # M-3 forces LATENT to fp32; short-circuit so the trace is honest and
            # we never size the gate off a LATENT's [B,H,W,C] axis.
            precision, auto_reason = "fp32", "auto->fp32: LATENT input (stays fp32)"
        else:
            dev_type = torch.device(device).type
            from .tex_runtime.precision_policy import (
                resolve_auto_precision, _MIN_FP16_PX)
            # B1: memoize the DECISION per (program, resolution-bucket, device) so
            # the resolve_auto AST walk runs once, not per cook (a static property of
            # code+resolution+device — value-independent).
            # B3: compile_mode is part of the KEY because it is part of the VALUE — the
            # "[compiled tier: fp32]" adjustment below happens on the miss path, so
            # without it whichever compile_mode cooked this program FIRST would decide the
            # precision for every later mode. A torch_compile cook would pin fp32 and the
            # next compile_mode="none" cook would silently lose fp16 (and vice versa).
            auto_ckey = ((fp, cook_px >= _MIN_FP16_PX, dev_type,
                          compile_mode != "none") if fp is not None else None)
            cached = _AUTO_DECISION.get(auto_ckey) if auto_ckey is not None else None
            if cached is not None:
                precision, auto_reason = cached
            else:
                precision, auto_reason = resolve_auto_precision(program, cook_px, dev_type)
                if precision == "fp16" and compile_mode != "none":
                    precision, auto_reason = "fp32", auto_reason + " [compiled tier: fp32]"
                if auto_ckey is not None:
                    _AUTO_DECISION[auto_ckey] = (precision, auto_reason)
                    _cap_auto_caches()
            # C2 (doc 32): the finiteness safety-net runs on EVERY fp16 cook. The B1
            # "check once then trust the fingerprint" shipped NaN silently when the
            # SAME program met a new input value (3.1M NaN pixels) — and in real
            # ComfyUI use every cook already HAS a new input (identical inputs are
            # cache hits), so a value-keyed skip is pure overhead on top of the check
            # it can't avoid. Checking every cook is correct AND, measured, cheaper on
            # the real path than hashing the input to decide whether to check. A
            # non-finite result re-cooks + pins the program to fp32 thereafter.
            auto_fp16 = precision == "fp16"
        tier_trace.record_precision(precision, auto_reason)
    # M-3: LATENT data exceeds [0,1] and feeds further math, so it must stay
    # fp32 even in fp16 mode.
    eff_precision = "fp32" if has_latent_input else precision
    tier_id = select_tier(compile_mode, device, fused_chain, fused_fp is not None)
    # ROI-3: gate the sub-window cook. Flagged OFF (TEX_ROI_EXEC) and interpreter-tier only —
    # compiled tiers don't thread roi, so a would-be ROI cook that resolved to a compiled tier
    # simply runs whole-frame (same posture as tiling). Non-fused, non-latent, executable.
    roi_out = roi_plan_obj = None
    if roi is not None and tier_id == "default" and not fused_chain and not has_latent_input:
        from . import tex_roi as _tex_roi
        if _tex_roi.roi_exec_enabled():
            scalar_params = {n: v for n, v in bindings.items()
                             if isinstance(v, (bool, int, float))}
            _plan = _tex_roi.roi_plan(code, scalar_params)
            if _plan.executable:
                roi_out, roi_plan_obj = roi, _plan
                # ROI is validated (oracle) at fp32 only, and the ~1-ulp conv slack the
                # narrow-cook-crop leaves would scale up at fp16 — clamp the ROI cook to fp32
                # (the same conservative posture as the compiled-tier / LATENT fp32 forces).
                eff_precision = "fp32"
    ctx = ExecContext(program, bindings, type_map, device, code,
                      latent_channel_count, output_names, used_builtins,
                      eff_precision, fp, fused_chain, fused_fp, time_context,
                      free_hint, roi_out, roi_plan_obj,  # ROI-3 window + plan (None unless armed)
                      cancel, on_progress,               # SCHED-3 (None unless a host passed them)
                      binding_meta)                      # DATA-1 tags (None unless a host passed them)
    return CookPlan(ctx=ctx, tier_id=tier_id, assigned=assigned_bindings,
                    auto_fp16=auto_fp16, debug_nan_highlight=debug_nan_highlight,
                    cook_px=cook_px, auto_ckey=auto_ckey, disown=disown,
                    want_lineage=want_lineage, upstream_keys=tuple(upstream_keys))


def _oom_retry(ctx: ExecContext, caught: BaseException, oom: BaseException):
    """ENG-2: the engine's OOM recovery ladder. Returns the output dict on success, or
    None to let the caller re-raise the ORIGINAL OOM.

    Strictly ADDITIVE, and that constraint shapes the whole design. Under ComfyUI an
    escaped OOM already triggers a good host-level ladder (unload_all_models + re-run the
    prompt), and that must keep working — so this never swallows an OOM it could not
    actually recover. It runs where the old code was already on its way to raising:
    either it turns a failure into a picture, or nothing observable changes.

      1. drop TEX's own tensor caches (mip/grid/sampler pyramids). The host's unload
         frees models but never these; on a Null host nothing else would free them at all.
      2. re-cook in horizontal strips, forcing the tile plan rather than asking it —
         the estimator just demonstrably under-called this cook, so its verdict is not
         evidence. Only for tile-safe programs; run_tiled is seam-exact (M-4).

    There is deliberately NO automatic CPU rung, though the roadmap floats one: device is
    not a free variable. Invariant #9 makes CPU↔GPU a characterization envelope (up to
    6.1e-2), so silently finishing on the CPU would hand back different pixels than the
    user asked for, with only a log line to say so — and CACHE-1 will make device part of
    every result key precisely because it is visible. A host that wants CPU can ask for
    CPU; the engine will not decide that quietly.
    """
    logger.warning("[TEX] cook hit OOM (%s); dropping TEX caches and retrying.", oom)
    # The failed cook's traceback holds a frame for every level of it, and the interpreter
    # keeps whole [B,H,W,C] tensors as frame locals (`_eval_binop`'s left/right,
    # `_eval_function_call`'s args) — ~127 MB apiece at 4K fp32. Without clearing them the
    # ladder frees the mip caches and then retries into the very peak it just failed on.
    #
    # BOTH exceptions, and that is the whole point: `oom` is what `_oom_in_chain` dug out
    # of the __cause__ chain, NOT what run() caught. M-1 re-wraps a stdlib OOM as
    # InterpreterError(...) from e — the likeliest OOM there is, per M-1's own docstring
    # (sample_mip / gauss_blur) — and in that shape the big tree-walk frames hang off the
    # WRAPPER, while the inner OOM carries only the innermost two. Measured: clearing
    # `oom` alone freed 0 of 24 MB; clearing the wrapper freed all of it.
    # clear_frames() silently skips still-executing frames and leaves the traceback
    # renderable, so the "re-raise THE ORIGINAL" contract below is untouched.
    try:
        import traceback as _tb
        for _exc in (caught, oom):
            if _exc is not None:
                _tb.clear_frames(_exc.__traceback__)
    except Exception:
        pass
    # B4: discard the doomed attempt's debug_print probes before re-cooking, or the HUD
    # shows every probe twice — and the re-cook is TILED, so its per-strip probes read
    # narrowed bindings and would report the wrong pixel. Same reason the C2 finiteness
    # net clears before its own re-cook, two functions away.
    try:
        from .tex_runtime import tier_trace
        tier_trace.clear_probes()
    except Exception:
        pass
    _drop_tex_caches_on_oom()
    get_host_services().soft_empty_cache()

    if ctx.fused_chain or ctx.latent_channel_count or not str(ctx.device).startswith("cuda"):
        return None            # rung 2 is strip tiling; these are the shapes it can't tile
    try:
        from .tex_memory import (is_tile_safe_cached, shared_tile_height, run_tiled,
                                 shared_tile_width, run_tiled_halo)
        H = shared_tile_height(ctx.bindings)
        if H is None:
            return None
        # The estimator said this fit and it did not, so don't re-consult it for the
        # strip count — take the deepest split the >=64-row floor allows and let the
        # launch tax buy a picture. Capped, so a pathological program still terminates.
        n_strips = max(2, min(H // 64, 16))
        if n_strips < 2:
            return None
        _cancel_check(ctx.cancel)   # SCHED-3 yield E: don't start a tiled OOM re-cook if abandoned
        if is_tile_safe_cached(ctx.program, ctx.fp):
            logger.warning("[TEX] retrying the cook in %d strips.", n_strips)
            return run_tiled(_get_interpreter(), ctx.program, ctx.bindings, ctx.type_map,
                             ctx.device, ctx.latent_channel_count, ctx.output_names,
                             ctx.used_builtins, ctx.eff_precision, n_strips, ctx.time_context,
                             cancel=ctx.cancel, on_progress=ctx.on_progress)
        # ROI-5: not pixel-local, but a bounded blur/morphology can still HALO-tile out of an OOM.
        from . import tex_roi
        rplan = tex_roi.roi_plan(ctx.code, _scalar_params(ctx.bindings))
        if (rplan.executable and rplan.halo > 0 and rplan.narrow
                and shared_tile_width(ctx.bindings) is not None):
            n_h = max(2, min(H // max(64, 4 * rplan.halo), 16))
            if n_h >= 2:
                logger.warning("[TEX] retrying the cook in %d halo strips.", n_h)
                return run_tiled_halo(_get_interpreter(), ctx.program, ctx.bindings, ctx.type_map,
                                      ctx.device, ctx.latent_channel_count, ctx.output_names,
                                      ctx.used_builtins, ctx.eff_precision, n_h, rplan.narrow,
                                      rplan.halo, ctx.time_context,
                                      cancel=ctx.cancel, on_progress=ctx.on_progress)
        return None
    except CookCancelled:
        raise                       # SCHED-3: a cancel at yield E (or inside the tiled/halo OOM
        #                             re-cook) must abort — never be reclassified as the original
        #                             OOM and retried on a cook the user already abandoned.
    except Exception as retry_exc:
        # The retry failed too (possibly with a second OOM). Report nothing recovered and
        # let the ORIGINAL OOM propagate — the host's own ladder is the next rung, and it
        # must see the OOM, not this.
        logger.warning("[TEX] tiled OOM retry failed (%s); re-raising the original OOM.",
                       retry_exc)
        return None


def _disown_inputs(raw_output: dict, bindings: dict) -> dict:
    """Make sure no cooked output shares storage with an INPUT BINDING.

    `@OUT = @A;` binds the output name straight to the input tensor, so the "result" IS
    the caller's buffer — and const-folding widens that past literal identity (`@OUT =
    @A * 1.0;` folds to the same thing). The ComfyUI node never noticed because its egress
    clamp materializes a fresh tensor on the way out; ENG-3's `engine` profile removed the
    clamp and, with it, that accidental copy. A host recycling frame buffers would then
    have its input silently rewritten by its own output.

    This is the right layer for it: only the engine knows what the bindings were. The
    egress profile downstream is a format conversion and cannot tell an aliased input from
    a freshly computed tensor, so a clone there would have to be unconditional — measured
    at 39.7 ms on a 398 MB buffer, paid on every cook including the overwhelming majority
    that never alias.

    Keyed on the STORAGE POINTER, and that word is load-bearing twice over:

      * NOT object identity. A reshape (`unsqueeze`) hands back a new object over the same
        storage — different object, same pixels, same corruption.
      * NOT `.data_ptr()`, which is the address of the tensor's FIRST ELEMENT, not of its
        buffer. A view at a non-zero offset has a different one. `@X = @A.rgb;` starts at
        offset 0 and looks caught, which is exactly what makes that spelling dangerous —
        `@X = @A.a;` starts at offset 3, compares unequal, and sails through aliased.
        `untyped_storage().data_ptr()` is the buffer, and is the same for every view of it.

    Whether it runs at all is `plan.disown`, decided by the CALLER — see `prepare()`. It
    is not read off the process-global egress profile: a caller that pins its own profile
    per-call (`tex_cli`) or applies none at all (the documented `cook()` contract — raw
    tensors) is invisible to that global, so consulting it answered a question nobody
    asked. Ownership is a property of the call.
    """
    try:
        src = {v.untyped_storage().data_ptr()
               for v in bindings.values() if isinstance(v, torch.Tensor)}
        if not src:
            return raw_output
        return {name: (out.clone() if isinstance(out, torch.Tensor)
                       and out.untyped_storage().data_ptr() in src else out)
                for name, out in raw_output.items()}
    except Exception:
        return raw_output   # ownership is a safety net; it must never fail a cook


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
                canvas=canvas, flags=(*base_flags, f"out:{name}"))
        return out
    except Exception:
        return None


def run(plan: CookPlan) -> CookResult:
    """Execute a prepared plan: dispatch to the tier, apply the fp16 finiteness net and
    the debug overlays, enforce the cache budgets. Returns RAW outputs (no host egress
    formatting — that is ENG-3's profile, applied by the caller)."""
    ctx = plan.ctx
    _cancel_check(ctx.cancel)                         # SCHED-3 yield A: abort a stale cook up front
    _report_progress(ctx.on_progress, "tier", 0.0)
    try:
        raw_output = _run_tier(ctx, plan.tier_id)
    except BaseException as e:                       # ENG-2: the OOM ladder
        # SCHED-3: a cancellation raised inside the tier must propagate straight out, never be
        # mistaken for a recoverable OOM and silently retried (GOTCHA: _oom_in_chain returns
        # None for it today, but an explicit guard makes the intent load-bearing, not incidental).
        if isinstance(e, CookCancelled):
            raise
        oom = _oom_in_chain(e)
        if oom is None:
            raise
        retried = _oom_retry(ctx, e, oom)
        if retried is None:
            raise
        raw_output = retried

    # PR-LP2 safety net (C2): re-cook fp32 (and pin the auto decision) if an
    # auto->fp16 cook went non-finite. Extracted to a helper (C1-st) so the cook
    # stays within its per-function line budget and can't silently re-inline.
    raw_output, eff_precision = _fp16_finiteness_net(raw_output, plan.auto_fp16, ctx,
                                                     plan.tier_id, plan.auto_ckey)

    # M-2: cap TEX's tensor-cache residency at a byte budget (evicts
    # oldest mip/grid entries; allocate-and-hold semantics untouched).
    # P1-M2-CPU: enforce on CPU cooks too — the mip/grid/sampler caches
    # grow unbounded there otherwise. cache_budget_bytes returns a 512 MB
    # CPU budget; the eviction is cheap (early-returns when under budget)
    # and the graph-cache teardown on eviction is a no-op off CUDA.
    try:
        from .tex_memory import enforce_cache_budget, trim_reserved_pool
        enforce_cache_budget(ctx.device)
        # MEM-2 (B2): the trim only queries the allocator after a resolution
        # downshift (zero cost on same-size steady state). cook_px rides the plan.
        trim_reserved_pool(ctx.device, plan.cook_px)
    except Exception:
        pass

    # ENG-1/ENG-3: make good on the no-alias guarantee, unless this caller proved it does
    # not need it (see prepare()'s `disown`).
    if plan.disown:
        raw_output = _disown_inputs(raw_output, ctx.bindings)

    # DBG-3 magenta NaN + C4-ux cyan near-singularity. Applied to the RAW outputs, above
    # every tier, before any host egress formatting — the same order the node applied
    # them in when it owned the loop.
    near_sing = None
    if plan.debug_nan_highlight:
        from .tex_runtime import guard_trace   # off the default path: only the toggle pays
        mask = guard_trace.mask()              # hoisted out of the loop (was once/output)
        raw_output = {name: _nan_highlight(_singularity_highlight(raw, mask))
                      for name, raw in raw_output.items()}
        near_sing = guard_trace.count()   # read BEFORE the disarm below
        guard_trace.disarm()

    # CACHE-1: attach per-output lineage keys when the caller asked (a frame cache / graph
    # host). Off the default path — computed only under plan.want_lineage. `eff_precision` is
    # the precision the frame was ACTUALLY cooked at (fp32 if the finiteness net re-cooked), so
    # the key and CookResult.precision match the pixels.
    lineage = _compute_lineage(plan, ctx, eff_precision, raw_output) if plan.want_lineage else None

    _report_progress(ctx.on_progress, "tier", 1.0)   # SCHED-3: the cook produced a frame
    return CookResult(
        outputs=raw_output, output_names=ctx.output_names, assigned=plan.assigned,
        device=ctx.device, precision=eff_precision,
        binding_names=list(ctx.bindings.keys()), near_singularities=near_sing,
        lineage=lineage,
        # DATA-1: merged TENSOR-input tags for every output, or None when the host passed no
        # binding_meta (the default ComfyUI path) — value channel, never keyed. The engine
        # resolves the ORIGINAL tensor-input names here (de-prefixing fused stage renames via
        # tex_fusion.strip_user_prefix) so the host's tags match and marshalling stays free of
        # fusion's naming; only tensor inputs contribute (a scalar param can't downgrade an output).
        out_meta=_egress_meta(
            ctx.binding_meta, ctx.output_names,
            {(_strip_user_prefix(n) if ctx.fused_chain else n)   # de-prefix ONLY on a fused chain
             for n, v in ctx.bindings.items() if isinstance(v, torch.Tensor)}  # (mirror the E6003 site)
            if ctx.binding_meta else None),
    )


def cook(code: str, bindings: dict, **kwargs) -> CookResult:
    """The one-shot public cook: `run(prepare(...))`.

    The host-agnostic engine entry point, for a host with nothing to marshal: no ComfyUI,
    no node, no `ui=` payload. `tex_node` uses the two-step form instead so it can tell a
    chain that never assembled from one that failed mid-cook.

        from TEX_Wrangle import tex_engine
        res = tex_engine.cook("@OUT = vec4(@A.rgb * 1.2, 1.0);", {"A": img}, device_mode="cuda")
        res.outputs["OUT"]   # raw [B,H,W,4], unclamped

    Accepts every keyword `prepare()` does. Returns a `CookResult` whose `outputs` are
    RAW (no clamp / alpha-drop / gray-expand — apply an egress profile for that; ENG-3).
    """
    return run(prepare(code, bindings, **kwargs))


# ── CACHE-6: fusion ↔ caching reconciliation (the cook side) ──────────────────

def cook_stage_list(stages, *, device="cpu", precision="fp32", latent_channel_count=0,
                    time_context=None, cancel=None, on_progress=None) -> dict:
    """Cook a raw fusion stage list (≥1) and return the interpreter's RAW {output: tensor}. One
    stage cooks as a plain program; ≥2 splice through `compile_fused`. It replicates prepare()'s
    param default-inject + widget-value conversion so a SUB-chain (a CACHE-6 prefix or suffix)
    cooks BIT-IDENTICALLY to those same stages inside the full fused program — the equivalence
    the CACHE-6 oracle rests on. fp32 is forced under a LATENT (M-3), exactly as prepare does."""
    from .tex_cache import get_cache
    if len(stages) == 1:
        st = stages[0]
        bindings = dict(st.get("bindings") or {})
        binding_types = {n: _infer_binding_type(v) for n, v in bindings.items()}
        program, type_map, referenced, assigned, param_info, used_builtins = \
            get_cache().compile_tex(st["code"], binding_types)
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
                          time_context=time_context, cancel=cancel, on_progress=on_progress)


def boundary_lineage_key(stages, k, device, precision, *, upstream, time_context=None,
                         canvas=None, latent_channel_count=0) -> str:
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
    refuses to cache without one. fp32 is the exact-handoff contract, so precision keys it too."""
    from . import tex_results
    from .tex_fusion import prefix_fingerprint
    fp = prefix_fingerprint(stages, k, _infer_binding_type)
    params = {f"s{i}:{n}": v
              for i, st in enumerate(stages[:k])
              for n, v in (st.get("bindings") or {}).items()
              if not isinstance(v, torch.Tensor)}
    flags = [f"tap:s{k - 1}"]
    if latent_channel_count:
        flags.append(f"ic:{int(latent_channel_count)}")
    return tex_results.lineage_key(program_fp=fp, device=str(device), precision=precision,
                                   params=params, upstream=tuple(upstream), time_context=time_context,
                                   canvas=canvas, flags=flags)


def cook_fused_cached(stages, k, result_cache, *, device="cpu", precision="fp32",
                      time_context=None, latent_channel_count=0, upstream=(), cancel=None,
                      on_progress=None) -> dict:
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
                               time_context=time_context, cancel=cancel, on_progress=on_progress)

    # `upstream` must key EVERY tensor input of the prefix — the source, and any EXTRA image a
    # prefix stage reads — not just be non-empty (a partial cover could stale-serve when only an
    # unkeyed prefix tensor changes; the count also subsumes the no-upstream `()` default, since a
    # chain's prefix always has ≥1 source tensor → 0<1). The range check `not (1 <= k < len)` is
    # ordered BEFORE the tensor-count term so an out-of-range (or non-int) `k` short-circuits to a
    # full cook without ever slicing `stages[:k]`.
    if (precision != "fp32" or latent_channel_count or result_cache is None
            or not is_linear_stage_list(stages) or not (1 <= k < len(stages))
            or len(upstream) < sum(1 for st in stages[:k]
                                   for v in (st.get("bindings") or {}).values()
                                   if isinstance(v, torch.Tensor))):
        return _full()
    key = boundary_lineage_key(stages, k, device, "fp32", time_context=time_context,
                               latent_channel_count=latent_channel_count, upstream=upstream)
    boundary = result_cache.get(key)
    if boundary is None:
        b = cook_stage_list(stages[:k], device=device, precision="fp32",
                            time_context=time_context, cancel=cancel).get("OUT")
        if b is None:            # a chain always assigns @OUT; if not, cook whole (correct)
            return _full()
        result_cache.put(key, b, canvas={"shape": list(b.shape)})
        boundary = b     # the freshly-cooked, locally-owned prefix output — put stored its own
        #                  frozen copy, so `b` is unaliased; feed it straight in (no re-get clone)
    try:
        suffix = suffix_stage_list(stages, k, boundary)
    except FusionError:          # a malformed cut (head stage lacks a chain_input to rebind) — the
        return _full()           #   documented whole-chain fallback, not a crash after the put
    return cook_stage_list(suffix, device=device, precision="fp32",
                           time_context=time_context, cancel=cancel, on_progress=on_progress)
