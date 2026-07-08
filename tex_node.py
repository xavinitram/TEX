"""
TEX Wrangle Node — ComfyUI custom node for the TEX (Tensor Expression Language).

Provides a single node where users write compact TEX scripts to process
images, masks, and scalar values using per-pixel tensor operations.

Requires ComfyUI with Nodes v3 API support (comfy_api.latest).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import torch
import traceback
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("TEX")

from .tex_compiler.lexer import LexerError
from .tex_compiler.parser import ParseError
from .tex_compiler.type_checker import TypeCheckError
from .tex_compiler.diagnostics import TEXMultiError, TEX_BUG_REPORT_URL
from .tex_compiler.ast_nodes import SourceLoc
from .tex_runtime.interpreter import Interpreter, InterpreterError
from .tex_cache import get_cache
from .tex_runtime.compiled import (
    execute_compiled,
    _codegen_only_execute,
    should_stencil_route as _should_stencil_route,
)
from .tex_fusion import (
    prepare_fused as _prepare_fused,
    fused_fingerprint as _fused_fingerprint,
    FusionError,
)

# Marshalling and type inference utilities (extracted to tex_marshalling.py)
from .tex_marshalling import (
    tensor_fingerprint as _tensor_fingerprint,
    unwrap_latent as _unwrap_latent,
    convert_param_value as _convert_param_value,
    infer_binding_type as _infer_binding_type,
    prepare_output as _prepare_output,
    map_inferred_type as _map_inferred_type,
)

# ── v3 API import ──
# Falls back to a stub base class when running outside ComfyUI (e.g. standalone tests).
try:
    from comfy_api.latest import IO
    _V3_AVAILABLE = True
except ImportError:
    IO = None
    _V3_AVAILABLE = False

# ── ComfyUI memory-management (optional; absent in standalone tests) ──
try:
    import comfy.model_management as _mm
except Exception:
    _mm = None


@dataclass(frozen=True)
class ExecContext:
    """STR-2/STR-3: the value bundle threaded from `execute()` into the tier strategies
    (`_run_torch_compile`/`_run_auto`/`_run_cuda_graph`/`_run_default`) and the shared
    `_interp_fallback` recovery path. Built once per cook so a strategy never re-touches
    execute()'s locals; `select_tier` picks the strategy, `execute()` dispatches via
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
    # strategy never re-touches execute()'s locals). `fp` is the value-independent
    # fingerprint, or None on a fused chain (which is keyed by `fused_fp` instead).
    fp: Any = None
    fused_chain: bool = False
    fused_fp: Any = None


def _downscale_for_preview(img: torch.Tensor, max_dim: int = 256) -> torch.Tensor:
    """Q-6: downscale a [B,H,W,C] image so its largest spatial dim ≤ max_dim
    (aspect preserved), for the debounced low-res live preview. Returns the
    input unchanged when it already fits. Bilinear, antialias-free (a preview)."""
    if not (isinstance(img, torch.Tensor) and img.dim() == 4):
        return img
    _b, h, w, _c = img.shape
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / m
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    bchw = img.permute(0, 3, 1, 2)
    out = torch.nn.functional.interpolate(bchw, size=(nh, nw), mode="bilinear",
                                          align_corners=False)
    return out.permute(0, 2, 3, 1).contiguous()


def _is_oom_error(e: BaseException) -> bool:
    """True when *e* is an out-of-memory error.

    Prefers ComfyUI's ``model_management.is_oom`` — on recent torch, OOM can
    surface as ``torch.AcceleratorError`` (error_code==2), not just
    ``torch.cuda.OutOfMemoryError``, and ``is_oom`` also clears poisoned async
    CUDA error state. Falls back to ``OOM_EXCEPTION`` then the torch type so
    standalone runs still detect it."""
    if _mm is not None:
        try:
            if hasattr(_mm, "is_oom"):
                return bool(_mm.is_oom(e))
            if hasattr(_mm, "OOM_EXCEPTION"):
                return isinstance(e, _mm.OOM_EXCEPTION)
        except Exception:
            pass
    oom_t = getattr(torch.cuda, "OutOfMemoryError", None)
    return isinstance(oom_t, type) and isinstance(e, oom_t)


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


# ── Cached interpreter (reused across executions to avoid rebuild overhead) ──
_interpreter: Interpreter | None = None


def _get_interpreter() -> Interpreter:
    """Get or create the cached Interpreter singleton."""
    global _interpreter
    if _interpreter is None:
        _interpreter = Interpreter()
    return _interpreter


# ── Node class ──
# When running inside ComfyUI, inherit from IO.ComfyNode (v3 API).
# When running standalone (tests), fall back to plain object.
_BaseClass = IO.ComfyNode if _V3_AVAILABLE else object


class TEXWrangleNode(_BaseClass):
    """
    TEX Wrangle — per-pixel tensor expression processor.

    Write compact TEX scripts to process images, masks, and scalars.
    Reference inputs with @name (e.g. @A, @base_image, @strength).
    Write output to @OUT.

    Example:
        float gray = luma(@A);
        @OUT = vec4(gray, gray, gray, 1.0);
    """

    # Maximum number of output slots (pre-allocated for dynamic outputs)
    MAX_OUTPUTS = 8

    # System kwargs that are NOT TEX bindings
    _SYSTEM_KWARGS = {"code", "device", "compile_mode", "precision", "_tex_any",
                      "_tex_chain", "_tex_preview"}

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TEX_Wrangle",
            display_name="TEX Wrangle",
            category="TEX",
            description=(
                "TEX Wrangle: per-pixel tensor expressions for images, masks, and scalars.\n"
                "Reference inputs with @name (e.g. @A, @base_image).\n"
                "Write outputs with @name = expr (e.g. @OUT, @mask, @result).\n"
                "Add parameter widgets with $name (e.g. f$strength = 0.5).\n"
                "Supports float, int, vec2/vec3/vec4, mat3/mat4, string, and array types "
                "with if/else, for/while loops, and 100+ stdlib functions.\n"
                "Click the ? icon for a quick reference."
            ),
            inputs=[
                IO.String.Input(
                    "code",
                    multiline=True,
                    dynamic_prompts=False,
                    default=(
                        "// TEX Wrangle\n"
                        "// Read inputs with @A, @B, etc.\n"
                        "// Write output to @OUT\n\n"
                        "float gray = luma(@IN);\n"
                        "@OUT = vec3(gray);\n"
                    ),
                    placeholder="// TEX code here...",
                    tooltip="TEX source code. Use @name for inputs, @name = expr for outputs. Use $name for parameter widgets.",
                ),
                IO.Combo.Input(
                    "device",
                    options=["auto", "cpu", "cuda"],
                    default="auto",
                    tooltip="Execution device. auto: follows input tensors. cpu/cuda: force a specific device.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "compile_mode",
                    options=["none", "auto", "torch_compile", "cuda_graph"],
                    default="none",
                    tooltip="none: standard interpreter (recommended default). auto: EXPERIMENTAL measured auto-tier — runs the fast codegen path and trials torch.compile in the background, committing only on a measured win (the background-compile timing is still being hardened; safe — it only ever falls back to a correct path). torch_compile: force JIT-compile via torch.compile. cuda_graph: CUDA-graph replay of the interpreter (GPU only; big win for small launch-bound programs). All fall back to the interpreter on failure.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "precision",
                    options=["fp32", "auto", "fp16"],
                    default="fp32",
                    tooltip="fp32: full precision (default). auto: fp16 ONLY where it measurably wins and stays accurate — CUDA, >=1024x1024, pointwise programs with no sampling/scatter/reduction and no image-derived threshold (measured 1.4-1.5x on grade-class); everything else runs fp32. fp16: EXPERIMENTAL — force half-precision IMAGE-class temps (coordinates & sampling stay fp32; ~1e-3 accuracy, diverges on threshold/branch programs). LATENT stays fp32.",
                    optional=True,
                ),
                # Wildcard input so the node appears in the search panel
                # when dragging a wire of any type. The frontend extension
                # removes all initial input slots in onNodeCreated and
                # manages them dynamically, so this slot never renders.
                IO.AnyType.Input(
                    "_tex_any",
                    optional=True,
                    tooltip="TEX accepts any input type. Use @name in code to reference it.",
                ),
            ],
            outputs=[
                IO.AnyType.Output(
                    id=f"out_{i}",
                    display_name=f"out_{i}",
                    tooltip="TEX output. Type auto-inferred from code.",
                )
                for i in range(cls.MAX_OUTPUTS)
            ],
            accept_all_inputs=True,
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        """Cache-busting: re-execute if code, inputs, device, or compile_mode change."""
        code = kwargs.get("code", "")
        parts = [code]
        # Include device and compile_mode in the hash
        parts.append(kwargs.get("device", "auto"))
        parts.append(kwargs.get("compile_mode", "none"))
        parts.append(kwargs.get("precision", "fp32"))
        # LOAD-BEARING: the fused-chain payload (upstream code + params) must
        # be hashed explicitly. It sits in _SYSTEM_KWARGS so the loop below
        # never treats it as a TEX binding — without this line, editing an
        # upstream node's code would no longer recook the fused terminal.
        chain_payload = kwargs.get("_tex_chain")
        if chain_payload is not None:
            parts.append(f"_tex_chain:{chain_payload}")
        # Hash all binding inputs (everything except system kwargs)
        for name in sorted(kwargs.keys()):
            if name in cls._SYSTEM_KWARGS:
                continue
            val = kwargs[name]
            if val is not None:
                if isinstance(val, list):
                    list_parts = []
                    for item in val:
                        if isinstance(item, torch.Tensor):
                            list_parts.append(_tensor_fingerprint(item))
                        else:
                            list_parts.append(str(item))
                    parts.append(f"{name}:LIST:{','.join(list_parts)}")
                elif isinstance(val, dict) and "samples" in val:
                    lat = f"{name}:LATENT:{_tensor_fingerprint(val['samples'])}"
                    # Include noise_mask + scalar metadata so a SetLatentNoiseMask
                    # change (same samples, new mask) still busts the cache.
                    mask = val.get("noise_mask")
                    if isinstance(mask, torch.Tensor):
                        lat += f":mask:{_tensor_fingerprint(mask)}"
                    for mk in sorted(k for k in val if k not in ("samples", "noise_mask")):
                        mv = val[mk]
                        if isinstance(mv, (int, float, str, bool)):
                            lat += f":{mk}={mv}"
                        elif isinstance(mv, torch.Tensor):
                            lat += f":{mk}:{_tensor_fingerprint(mv)}"
                        else:
                            lat += f":{mk}={mv!r}"
                    parts.append(lat)
                elif isinstance(val, torch.Tensor):
                    parts.append(f"{name}:{_tensor_fingerprint(val)}")
                else:
                    parts.append(f"{name}:{val}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    @classmethod
    def _interp_fallback(cls, ctx: ExecContext, *, reset_dynamo: bool,
                         pass_precision: bool):
        """STR-2: the single copy of the tier→interpreter recovery path, previously
        duplicated in the torch_compile / auto / cuda_graph branches. The two flags
        reproduce each branch's *exact* original call:
          - torch_compile: reset_dynamo=True,  pass_precision=False
          - auto:          reset_dynamo=True,  pass_precision=True
          - cuda_graph:    reset_dynamo=False, pass_precision=False
        (dynamo state is process-global — see compiled.py — so a failed compile/auto
        cook resets it on THIS thread before retrying; the graph path never touched
        dynamo, so it does not reset.)"""
        if reset_dynamo:
            try:
                torch._dynamo.reset()
            except Exception:
                pass
        interp = _get_interpreter()
        kw = dict(source=ctx.code, latent_channel_count=ctx.latent_channel_count,
                  output_names=ctx.output_names, used_builtins=ctx.used_builtins)
        if pass_precision:
            kw["precision"] = ctx.eff_precision
        return interp.execute(ctx.program, ctx.bindings, ctx.type_map,
                              device=ctx.device, **kw)

    @staticmethod
    def select_tier(compile_mode, device, fused_chain: bool, fused_fp_present: bool) -> str:
        """STR-2: PURE tier SELECTION — which acceleration strategy `(mode, device,
        fused)` picks, WITHOUT executing it. The branch ORDER and every guard mirror
        the old cascade verbatim; this is the CPU-testable core where the routing
        complexity lives (a fake `device="cuda:0"` string exercises the cuda_graph
        classification without a GPU)."""
        if compile_mode == "torch_compile" and not fused_chain:
            return "torch_compile"
        if compile_mode == "auto" and not fused_chain:
            return "auto"
        if (compile_mode == "cuda_graph" and str(device).startswith("cuda")
                and (not fused_chain or fused_fp_present)):
            return "cuda_graph"
        return "default"

    # ── STR-2 tier strategies: each runs one tier and returns raw_output; the
    #    dict-normalization is lifted to execute() post-dispatch. ──
    @classmethod
    def _run_torch_compile(cls, ctx: ExecContext):
        try:
            return execute_compiled(ctx.program, ctx.bindings, ctx.type_map, ctx.device,
                                    ctx.fp, latent_channel_count=ctx.latent_channel_count,
                                    output_names=ctx.output_names, used_builtins=ctx.used_builtins)
        except Exception as compile_exc:
            # Defense in depth: torch_compile must NEVER hard-fail the node.
            logger.warning("[TEX] torch_compile path failed (%s); using interpreter.",
                           compile_exc)
            return cls._interp_fallback(ctx, reset_dynamo=True, pass_precision=False)

    @classmethod
    def _run_auto(cls, ctx: ExecContext):
        try:
            from .tex_runtime.compiled import run_auto
            return run_auto(ctx.program, ctx.bindings, ctx.type_map, ctx.device, ctx.fp,
                            latent_channel_count=ctx.latent_channel_count,
                            output_names=ctx.output_names, used_builtins=ctx.used_builtins,
                            precision=ctx.eff_precision)
        except Exception as auto_exc:
            logger.warning("[TEX] auto tier failed (%s); using interpreter.", auto_exc)
            return cls._interp_fallback(ctx, reset_dynamo=True, pass_precision=True)

    @classmethod
    def _run_cuda_graph(cls, ctx: ExecContext):
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
            return cls._interp_fallback(ctx, reset_dynamo=False, pass_precision=False)
        return out

    @classmethod
    def _run_default(cls, ctx: ExecContext):
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
                        used_builtins=ctx.used_builtins, fingerprint=ctx.fp)
            except Exception as _stencil_exc:
                logger.warning("[TEX] stencil codegen route failed (%s); using "
                               "interpreter.", _stencil_exc)
        interp = _get_interpreter()
        # M-4: under GPU memory pressure, run a tile-safe program in horizontal strips
        # (peak transient ~1/n). Falls back to the whole-image cook on any strip error.
        n_strips = (cls._tile_plan(ctx.program, ctx.bindings, ctx.device, ctx.latent_channel_count)
                    if not ctx.fused_chain else None)
        if n_strips:
            try:
                from .tex_memory import run_tiled
                return run_tiled(interp, ctx.program, ctx.bindings, ctx.type_map, ctx.device,
                                 ctx.latent_channel_count, ctx.output_names, ctx.used_builtins,
                                 ctx.eff_precision, n_strips)
            except Exception as _tile_exc:
                logger.warning("[TEX] tiled cook failed (%s); running untiled.", _tile_exc)
        # Pass source so runtime (E6xxx) errors render a source-line caret. Fused chains
        # splice many sources, so leave source empty there (errors stay message-only).
        return interp.execute(ctx.program, ctx.bindings, ctx.type_map, device=ctx.device,
                              source=("" if ctx.fused_chain else ctx.code),
                              latent_channel_count=ctx.latent_channel_count,
                              output_names=ctx.output_names, used_builtins=ctx.used_builtins,
                              precision=ctx.eff_precision)

    # tier_id → strategy method name (getattr avoids the callable-in-dict binding trap)
    _TIER_METHOD = {
        "torch_compile": "_run_torch_compile", "auto": "_run_auto",
        "cuda_graph": "_run_cuda_graph", "default": "_run_default",
    }

    @classmethod
    def execute(cls, **kwargs):
        """Execute TEX code with the provided inputs."""
        start_time = time.perf_counter()

        # Pop system parameters so they don't get treated as bindings
        code = kwargs.pop("code", "")
        device_mode = kwargs.pop("device", "auto")
        compile_mode = kwargs.pop("compile_mode", "none")
        precision = kwargs.pop("precision", "fp32")
        # fp16 is an interpreter-only mode for now — the compile/graph paths bake
        # precision into their keys but aren't validated for fp16 yet.
        if precision == "fp16" and compile_mode != "none":
            precision = "fp32"
        kwargs.pop("_tex_any", None)  # search-panel wildcard slot, unused
        kwargs.pop("_tex_preview", None)  # Q-6 preview hint; pop so it never
        # becomes a phantom @_tex_preview binding (it is a _SYSTEM_KWARG).
        # Cross-node fusion: when the frontend collapses a linked TEX chain into
        # this (terminal) node, it passes the upstream stages here. Absent → the
        # node behaves exactly as a single program (no behaviour change).
        chain_payload = kwargs.pop("_tex_chain", None)

        # Everything remaining in kwargs is a TEX binding (inputs + params)
        bindings: dict[str, Any] = {
            name: val for name, val in kwargs.items() if val is not None
        }

        # Unwrap lists — use first element (TEX doesn't support per-image indexing)
        for name, val in list(bindings.items()):
            if isinstance(val, list):
                if len(val) > 0:
                    bindings[name] = val[0]
                else:
                    bindings[name] = torch.tensor(0.0)

        # Unwrap LATENT dicts into channel-last tensors, preserving metadata
        latent_meta: dict[str, Any] = {}
        has_latent_input = False
        latent_channel_count = 0
        for name, val in list(bindings.items()):
            if isinstance(val, str):
                continue  # strings pass through as-is
            if isinstance(val, dict) and "samples" in val:
                tensor_cl, meta = _unwrap_latent(val)
                bindings[name] = tensor_cl
                has_latent_input = True
                if not latent_meta and meta:
                    latent_meta = meta
                if not latent_channel_count:
                    latent_channel_count = tensor_cl.shape[-1]

        try:
            fused_chain = False
            fused_fp = None
            if chain_payload:
                # Fused path: splice the whole linked chain into one program.
                # Per-stage validation already happened in compile_fused, so a
                # failure here raises (we must NOT silently run the terminal
                # alone — the upstream nodes were collapsed away in the prompt).
                spec = json.loads(chain_payload) if isinstance(chain_payload, str) else chain_payload
                # Q-1: the fused fingerprint (value-independent) lets the whole
                # spliced chain be a CUDA-graph capture unit — computed from the
                # original bindings before _prepare_fused merges them. Only needed
                # for cuda_graph mode, so skip the extra stage-assembly otherwise.
                if compile_mode == "cuda_graph":
                    fused_fp = _fused_fingerprint(spec, code, bindings, _infer_binding_type)
                (program, type_map, referenced, assigned_bindings, param_info,
                 used_builtins, bindings) = _prepare_fused(spec, code, bindings, _infer_binding_type)
                fused_chain = True
            else:
                # Infer binding types for inputs
                binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}

                # Compile (uses two-tier Mega-Cache: memory LRU + disk persistence)
                cache = get_cache()
                program, type_map, referenced, assigned_bindings, param_info, used_builtins = cache.compile_tex(code, binding_types)

            # Resolve target device (from the effective bindings — merged for a chain)
            device = cls._resolve_device(device_mode, bindings)

            # M-1: preflight — estimate this cook's peak and, if it exceeds free
            # VRAM, ask ComfyUI to free resident models first (what model loaders
            # do). Prevents a big cook OOMing while GBs sit locked in models. Zero
            # cost on the common path (one memory-stats read). cuda only.
            if _mm is not None and str(device).startswith("cuda"):
                cls._preflight_memory(program, bindings, device)

            # Determine output names (sorted alphabetically for stable ordering)
            output_names = sorted(assigned_bindings.keys())
            if not output_names:
                raise InterpreterError(
                    "TEX program has no outputs. Assign to @OUT or another @name.",
                    loc=SourceLoc(1, 1), source=code, code="E6001",
                )
            if len(output_names) > cls.MAX_OUTPUTS:
                raise InterpreterError(
                    f"TEX program has {len(output_names)} outputs, "
                    f"exceeding the maximum ({cls.MAX_OUTPUTS}).",
                    loc=SourceLoc(1, 1), source=code, code="E6002",
                )

            # Check that referenced input bindings are available.
            # For $params with code-defined defaults, inject the default as a
            # fallback when no widget value or wire connection is provided.
            for ref_name in referenced:
                if ref_name not in assigned_bindings and ref_name not in bindings:
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
                    disp = re.sub(r"^_s\d+_u_", "", ref_name) if fused_chain else ref_name
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
            # strategy re-touches execute()'s locals), select the tier (pure), dispatch
            # to its strategy, and normalize the output shape in ONE place.
            # PR-LP2: resolve precision="auto" now that the program + resolution are
            # known — fp16 only in the measured win region (CUDA, >=1024^2, pointwise,
            # no image-derived threshold); fp32 otherwise. Record the decision + reason
            # (tier_trace) so it's never silent. fp16 stays out of the compiled/graph
            # tiers this cycle (mirrors the compile-mode fp32 force above).
            auto_fp16 = False
            if precision == "auto":
                from .tex_runtime.precision_policy import resolve_auto_precision
                from .tex_runtime import tier_trace
                dev_type = torch.device(device).type
                spatial_px = next((v.shape[1] * v.shape[2] for v in bindings.values()
                                   if isinstance(v, torch.Tensor) and v.dim() >= 3), 0)
                precision, auto_reason = resolve_auto_precision(program, spatial_px, dev_type)
                if precision == "fp16" and compile_mode != "none":
                    precision, auto_reason = "fp32", auto_reason + " [compiled tier: fp32]"
                auto_fp16 = precision == "fp16"
                tier_trace.record_precision(precision, auto_reason)
            # M-3: LATENT data exceeds [0,1] and feeds further math, so it must stay
            # fp32 even in fp16 mode.
            eff_precision = "fp32" if has_latent_input else precision
            fp = cache.fingerprint(code, binding_types) if not fused_chain else None
            ctx = ExecContext(program, bindings, type_map, device, code,
                              latent_channel_count, output_names, used_builtins,
                              eff_precision, fp, fused_chain, fused_fp)
            tier_id = cls.select_tier(compile_mode, device, fused_chain,
                                      fused_fp is not None)
            raw_output = getattr(cls, cls._TIER_METHOD[tier_id])(ctx)
            if not isinstance(raw_output, dict):
                raw_output = {output_names[0]: raw_output}

            # PR-LP2 safety net: an auto->fp16 cook can still go non-finite on a
            # fp16-fragile pointwise program the static gate can't tell from grade's
            # safe pow (e.g. brightness_contrast's pivot pow -> NaN). Cheap all-reduce
            # on the auto-fp16 path only; on a NaN/Inf, transparently re-cook fp32.
            if auto_fp16 and eff_precision == "fp16" and any(
                    isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
                    for v in raw_output.values()):
                from .tex_runtime import tier_trace
                tier_trace.record_precision("fp32", "auto: fp16 non-finite -> fp32 fallback")
                ctx = ExecContext(program, bindings, type_map, device, code,
                                  latent_channel_count, output_names, used_builtins,
                                  "fp32", fp, fused_chain, fused_fp)
                raw_output = getattr(cls, cls._TIER_METHOD[tier_id])(ctx)
                if not isinstance(raw_output, dict):
                    raw_output = {output_names[0]: raw_output}

            # M-2: cap TEX's tensor-cache residency at a byte budget (evicts
            # oldest mip/grid entries; allocate-and-hold semantics untouched).
            # P1-M2-CPU: enforce on CPU cooks too — the mip/grid/sampler caches
            # grow unbounded there otherwise. cache_budget_bytes returns a 512 MB
            # CPU budget; the eviction is cheap (early-returns when under budget)
            # and the graph-cache teardown on eviction is a no-op off CUDA.
            try:
                from .tex_memory import enforce_cache_budget
                enforce_cache_budget(device)
            except Exception:
                pass

            # Format each output based on inferred type
            results = []
            output_types_log = []
            for name in output_names:
                raw = raw_output[name]
                inferred_type = assigned_bindings[name]
                effective_type = _map_inferred_type(inferred_type, has_latent_input)
                result = _prepare_output(raw, effective_type)
                if effective_type == "LATENT":
                    result = {"samples": result, **latent_meta}
                results.append(result)
                output_types_log.append(f"{name}:{effective_type}")

            # Pad with None for unused output slots
            while len(results) < cls.MAX_OUTPUTS:
                results.append(None)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            compile_tag = " [compiled]" if compile_mode == "torch_compile" else ""
            logger.info(
                "Executed in %.1fms%s | device: %s | outputs: [%s] | bindings: %s",
                elapsed_ms, compile_tag, device,
                ", ".join(output_types_log), list(bindings.keys()),
            )

            # Return NodeOutput for v3, or tuple for standalone/test usage
            if _V3_AVAILABLE:
                return IO.NodeOutput(*results)
            return tuple(results)

        except TEXMultiError as e:
            # Multiple errors — human-readable first, JSON at end for frontend
            readable = "\n\n".join(d.render() for d in e.diagnostics)
            payload = json.dumps([d.to_dict() for d in e.diagnostics])
            raise RuntimeError(f"{readable}\nTEX_DIAG:{payload}") from e
        except (LexerError, ParseError, TypeCheckError, InterpreterError) as e:
            # M-1: an OOM raised inside a stdlib call arrives here wrapped as an
            # InterpreterError — re-raise the underlying OOM UNWRAPPED so ComfyUI's
            # OOM handling (memory summary + unload_all_models) still fires. Must
            # precede the diagnostic re-wrap, which would otherwise mask it.
            _oom = _oom_in_chain(e)
            if _oom is not None:
                _drop_tex_caches_on_oom()   # P1-M1-FREERETRY
                raise _oom
            # Q-4: attribute a fused-chain runtime error to its originating linked
            # node (the stage tag survives the splice; the frontend collapsed the
            # nodes so line numbers alone wouldn't map back).
            stage_note = ""
            _loc = getattr(e, "loc", None)
            _stage = getattr(_loc, "stage", None) if _loc is not None else None
            if fused_chain and _stage is not None:
                stage_note = (f"\n\nThis error came from linked TEX node #{_stage + 1} "
                              f"in the fused chain (counting from the source image). "
                              f"Break the chain there to debug it in isolation.")
            # Single error — human-readable first, JSON at end for frontend
            if hasattr(e, '_build_diagnostic'):
                e._build_diagnostic()
            if hasattr(e, 'diagnostic') and e.diagnostic:
                readable = e.diagnostic.render()
                payload = json.dumps([e.diagnostic.to_dict()])
                raise RuntimeError(f"{readable}{stage_note}\nTEX_DIAG:{payload}") from e
            # Fallback for errors without diagnostic (shouldn't happen)
            raise RuntimeError(f"TEX Error: {e}{stage_note}") from e
        except FusionError as e:
            # A chain couldn't be fused (e.g. an upstream node scatter-writes @OUT,
            # or has multiple outputs). A structural/config condition — show it
            # cleanly, no traceback or bug-report link. We must still raise: the
            # upstream nodes were collapsed out of the prompt, so running this
            # terminal alone would be wrong.
            raise RuntimeError(
                f"TEX couldn't fuse this chain of nodes:\n  {e}\n\n"
                f"Turn off 'TEX Fusion: Compile linked TEX nodes together' in "
                f"settings, or break the chain at the node mentioned above."
            ) from e
        except (ValueError, KeyError, TypeError, IndexError) as e:
            # Usually a value or type a built-in didn't expect — a user-side
            # problem, not a TEX bug. Show it cleanly, without a traceback.
            raise RuntimeError(
                f"TEX couldn't finish running your program:\n  {e}\n\n"
                f"This usually points to an input or parameter value that a built-in "
                f"didn't expect — check the inputs feeding this node."
            ) from e
        except RuntimeError as e:
            # OutOfMemoryError subclasses RuntimeError — re-raise it UNWRAPPED so
            # ComfyUI's own OOM handling (execution.py: is_oom → memory summary +
            # unload_all_models + user tips) still fires; the generic re-wrap
            # below would hide it. (Preflight/retry recovery is M-1, Phase 3.)
            _oom = _oom_in_chain(e)
            if _oom is not None:
                _drop_tex_caches_on_oom()   # P1-M1-FREERETRY
                raise _oom
            msg = str(e)
            # Ordinary user/config RuntimeErrors — a device selection issue, a
            # string output without the s@ prefix, or inputs of mismatched
            # resolution — are NOT TEX bugs. Show them cleanly (no traceback / URL).
            if "TEX Error:" in msg or "size of tensor" in msg or "must match" in msg:
                raise RuntimeError(
                    f"TEX couldn't run your program:\n  {msg}\n\n"
                    f"This is usually a device, output-type, or input-size mismatch "
                    f"— check the inputs and settings feeding this node."
                ) from e
            tb = traceback.format_exc()
            raise RuntimeError(
                f"TEX hit an unexpected problem while running your code:\n  {e}\n\n"
                f"If your code looks correct, this may be a TEX bug worth reporting:\n  "
                f"{TEX_BUG_REPORT_URL}\n{tb}"
            ) from e
        except Exception as e:
            # OOM can also surface here as torch.AcceleratorError — re-raise it
            # unwrapped for ComfyUI's OOM handling (see the RuntimeError branch).
            _oom = _oom_in_chain(e)
            if _oom is not None:
                _drop_tex_caches_on_oom()   # P1-M1-FREERETRY
                raise _oom
            # Genuinely unexpected — this may be a TEX bug. Include the traceback
            # and a report link so it can be diagnosed.
            tb = traceback.format_exc()
            raise RuntimeError(
                f"TEX hit an unexpected problem while running your code:\n  {e}\n\n"
                f"If your code looks correct, this may be a TEX bug worth reporting:\n  "
                f"{TEX_BUG_REPORT_URL}\n{tb}"
            ) from e

    @staticmethod
    def _tile_plan(program, bindings: dict[str, Any], device,
                   latent_channel_count: int = 0) -> int | None:
        """M-4: strip count if the cook should be tiled (tile-safe + under memory
        pressure), else None. cuda only; needs ComfyUI's free-memory query."""
        if _mm is None or not (hasattr(_mm, "get_free_memory") and str(device).startswith("cuda")):
            return None
        # M-4 safety: never tile a LATENT ([B,C,H,W] — dim 1 is channels, not
        # height) or a cook whose spatial bindings disagree on height (they can't
        # be co-tiled). run_tiled re-checks, but planning here avoids a bogus
        # peak estimate off the wrong axis.
        if latent_channel_count:
            return None
        try:
            from .tex_memory import is_tile_safe, estimate_peak_bytes, shared_tile_height
            if not is_tile_safe(program):
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
            est = estimate_peak_bytes(program, spatial)
            free = _mm.get_free_memory(torch.device(device))
            if not free or est <= 0:
                return None
            budget = 0.25 * free
            if est <= budget:
                return None  # no pressure — don't pay the launch tax
            import math as _math
            n = _math.ceil(est / budget)
            max_strips = max(1, spatial[1] // 64)  # ≥64-row strip floor
            n = min(n, max_strips)
            return n if n >= 2 else None
        except Exception:
            return None

    @staticmethod
    def _preflight_memory(program, bindings: dict[str, Any], device) -> None:
        """M-1: if the estimated cook peak exceeds free VRAM, free resident
        models first (best-effort; never raises)."""
        if _mm is None or not (hasattr(_mm, "get_free_memory") and hasattr(_mm, "free_memory")):
            return
        try:
            spatial = None
            for v in bindings.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 3:
                    spatial = (v.shape[0], v.shape[1], v.shape[2])
                    break
            if spatial is None:
                return
            from .tex_memory import estimate_peak_bytes
            est = estimate_peak_bytes(program, spatial)
            if est <= 0:
                return
            dev_t = torch.device(device)
            free = _mm.get_free_memory(dev_t)
            headroom = 128 * 1024 * 1024
            if free is not None and free < est + headroom:
                _mm.free_memory(est + 256 * 1024 * 1024, dev_t)
        except Exception:
            pass

    @staticmethod
    def _resolve_device(device_mode: str, bindings: dict[str, Any]) -> str:
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
