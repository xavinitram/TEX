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
                    options=["fp32", "fp16"],
                    default="fp32",
                    tooltip="fp32: full precision (default). fp16: half-precision IMAGE-class temps (coordinates & sampling stay fp32; ~1e-3 accuracy). Cuts peak/churn ~30% when the INPUT is already fp16; for an fp32 input the cast can raise peak, so it's most useful in all-fp16 pipelines. LATENT stays fp32.",
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

            # Execute — optionally with torch.compile acceleration.
            # Fused chains always use the interpreter (the spliced program isn't
            # keyed in the torch_compile fingerprint cache).
            if compile_mode == "torch_compile" and not fused_chain:
                fp = cache.fingerprint(code, binding_types)
                try:
                    raw_output = execute_compiled(program, bindings, type_map, device, fp,
                                                  latent_channel_count=latent_channel_count,
                                                  output_names=output_names,
                                                  used_builtins=used_builtins)
                except Exception as compile_exc:
                    # Defense in depth: selecting torch_compile must NEVER hard-fail
                    # the node. If execute_compiled's own fallback still raised,
                    # reset dynamo on THIS thread and run the plain interpreter
                    # (dynamo state is process-global — see compiled.py).
                    logger.warning(
                        "[TEX] torch_compile path failed (%s); using interpreter.",
                        compile_exc,
                    )
                    try:
                        torch._dynamo.reset()
                    except Exception:
                        pass
                    interp = _get_interpreter()
                    raw_output = interp.execute(program, bindings, type_map, device=device,
                                                source=code,
                                                latent_channel_count=latent_channel_count,
                                                output_names=output_names,
                                                used_builtins=used_builtins)
                # torch_compile returns single value — wrap for compat
                if not isinstance(raw_output, dict):
                    raw_output = {output_names[0]: raw_output}
            elif compile_mode == "auto" and not fused_chain:
                # CC-2: measured auto-tier — run the always-safe codegen path,
                # background-compile without stalling, and switch to torch.compile
                # only on a measured per-machine win. Never hard-fails the node.
                fp = cache.fingerprint(code, binding_types)
                try:
                    from .tex_runtime.compiled import run_auto
                    raw_output = run_auto(program, bindings, type_map, device, fp,
                                          latent_channel_count=latent_channel_count,
                                          output_names=output_names,
                                          used_builtins=used_builtins,
                                          precision=("fp32" if has_latent_input else precision))
                except Exception as auto_exc:
                    logger.warning("[TEX] auto tier failed (%s); using interpreter.",
                                   auto_exc)
                    try:
                        torch._dynamo.reset()
                    except Exception:
                        pass
                    interp = _get_interpreter()
                    raw_output = interp.execute(program, bindings, type_map, device=device,
                                                source=code,
                                                latent_channel_count=latent_channel_count,
                                                output_names=output_names,
                                                used_builtins=used_builtins,
                                                precision=("fp32" if has_latent_input else precision))
                if not isinstance(raw_output, dict):
                    raw_output = {output_names[0]: raw_output}
            elif (compile_mode == "cuda_graph" and str(device).startswith("cuda")
                  and (not fused_chain or fused_fp is not None)):
                # UC-1: CUDA-graph replay of the interpreter. Q-1: a fused chain is
                # captured as ONE graph, keyed by its fused fingerprint. Returns
                # None (falls through to the interpreter) when the program isn't
                # graphable or capture failed — never hard-fails on this path.
                from .tex_runtime.graphed import run_graphed
                _fp = fused_fp if fused_chain else cache.fingerprint(code, binding_types)
                raw_output = None
                try:
                    raw_output = run_graphed(
                        program, bindings, type_map, device, _fp,
                        latent_channel_count=latent_channel_count,
                        output_names=output_names, used_builtins=used_builtins)
                except Exception as _g_exc:
                    logger.warning("[TEX] cuda_graph path failed (%s); using interpreter.", _g_exc)
                    raw_output = None
                if raw_output is None:
                    interp = _get_interpreter()
                    raw_output = interp.execute(program, bindings, type_map, device=device,
                                                source=code,
                                                latent_channel_count=latent_channel_count,
                                                output_names=output_names,
                                                used_builtins=used_builtins)
                elif not isinstance(raw_output, dict):
                    raw_output = {output_names[0]: raw_output}
            else:
                # UC-2: default-route programs with an exact (fetch/conv) stencil
                # through the codegen tier, which lowers the stencil to
                # avg_pool2d/conv2d/unfold (10-40x CPU on box_blur-class programs).
                # Sample-based stencils are excluded by the gate (their lowering
                # would diverge). _codegen_only_execute self-falls-back to the
                # interpreter; the outer guard covers env-build edge cases so this
                # default-path change can never hard-fail the node.
                routed = False
                if not fused_chain:
                    try:
                        _fp = cache.fingerprint(code, binding_types)
                        if _should_stencil_route(_fp, program):
                            raw_output = _codegen_only_execute(
                                program, bindings, type_map, device,
                                latent_channel_count=latent_channel_count,
                                output_names=output_names,
                                used_builtins=used_builtins, fingerprint=_fp)
                            if not isinstance(raw_output, dict):
                                raw_output = {output_names[0]: raw_output}
                            routed = True
                    except Exception as _stencil_exc:
                        logger.warning("[TEX] stencil codegen route failed (%s); "
                                       "using interpreter.", _stencil_exc)
                        routed = False
                if not routed:
                    interp = _get_interpreter()
                    # M-3: LATENT data exceeds [0,1] and feeds further math, so it
                    # must stay fp32 even in fp16 mode.
                    eff_precision = "fp32" if has_latent_input else precision
                    # M-4: under GPU memory pressure, run a tile-safe program in
                    # horizontal strips (peak transient ~1/n). Falls back to the
                    # whole-image cook on any strip error.
                    n_strips = (cls._tile_plan(program, bindings, device, latent_channel_count)
                                if not fused_chain else None)
                    if n_strips:
                        try:
                            from .tex_memory import run_tiled
                            raw_output = run_tiled(interp, program, bindings, type_map,
                                                   device, latent_channel_count,
                                                   output_names, used_builtins, eff_precision,
                                                   n_strips)
                        except Exception as _tile_exc:
                            logger.warning("[TEX] tiled cook failed (%s); running untiled.",
                                           _tile_exc)
                            n_strips = None
                    if not n_strips:
                        # Pass source so runtime (E6xxx) errors render a source-line caret.
                        # Fused chains splice many sources, so their line numbers wouldn't
                        # map to `code` — leave source empty there (errors stay message-only).
                        raw_output = interp.execute(program, bindings, type_map, device=device,
                                                    source=("" if fused_chain else code),
                                                    latent_channel_count=latent_channel_count,
                                                    output_names=output_names,
                                                    used_builtins=used_builtins,
                                                    precision=eff_precision)

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
