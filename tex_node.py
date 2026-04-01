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
import time
import torch
import traceback
from typing import Any

logger = logging.getLogger("TEX")

from .tex_compiler.lexer import LexerError
from .tex_compiler.parser import ParseError
from .tex_compiler.type_checker import TypeCheckError
from .tex_compiler.diagnostics import TEXMultiError
from .tex_compiler.ast_nodes import SourceLoc
from .tex_runtime.interpreter import Interpreter, InterpreterError
from .tex_cache import get_cache
from .tex_runtime.compiled import execute_compiled

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
    _SYSTEM_KWARGS = {"code", "device", "compile_mode", "_tex_any"}

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
                "Supports float, int, vec3, vec4, string types with if/else, for loops, and 100+ stdlib functions.\n"
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
                    options=["none", "torch_compile"],
                    default="none",
                    tooltip="none: standard interpreter. torch_compile: JIT-compile via torch.compile (falls back on failure).",
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
                for i in range(8)
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
                    parts.append(f"{name}:LATENT:{_tensor_fingerprint(val['samples'])}")
                elif isinstance(val, torch.Tensor):
                    parts.append(f"{name}:{_tensor_fingerprint(val)}")
                else:
                    parts.append(f"{name}:{val}")
        return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()

    @classmethod
    def execute(cls, **kwargs):
        """Execute TEX code with the provided inputs."""
        start_time = time.perf_counter()

        # Pop system parameters so they don't get treated as bindings
        code = kwargs.pop("code", "")
        device_mode = kwargs.pop("device", "auto")
        compile_mode = kwargs.pop("compile_mode", "none")
        kwargs.pop("_tex_any", None)  # search-panel wildcard slot, unused

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

        # Resolve target device
        device = cls._resolve_device(device_mode, bindings)

        try:
            # Infer binding types for inputs
            binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}

            # Compile (uses two-tier Mega-Cache: memory LRU + disk persistence)
            cache = get_cache()
            program, type_map, referenced, assigned_bindings, param_info, used_builtins = cache.compile_tex(code, binding_types)

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
                    raise InterpreterError(
                        f"TEX code references {sigil}{ref_name} but no input is connected to slot '{ref_name}'.",
                        loc=SourceLoc(1, 1), source=code, code="E6003",
                    )

            # Convert param widget values to appropriate types for the interpreter
            # (e.g. hex color strings → RGB lists, comma vec strings → float lists)
            for pname, pinfo in param_info.items():
                if pname in bindings:
                    bindings[pname] = _convert_param_value(bindings[pname], pinfo)

            # Execute — optionally with torch.compile acceleration
            if compile_mode == "torch_compile":
                fp = cache.fingerprint(code, binding_types)
                raw_output = execute_compiled(program, bindings, type_map, device, fp,
                                              latent_channel_count=latent_channel_count,
                                              output_names=output_names)
                # torch_compile returns single value — wrap for compat
                if not isinstance(raw_output, dict):
                    raw_output = {output_names[0]: raw_output}
            else:
                interp = _get_interpreter()
                raw_output = interp.execute(program, bindings, type_map, device=device,
                                            latent_channel_count=latent_channel_count,
                                            output_names=output_names,
                                            used_builtins=used_builtins)

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
                f"Executed in {elapsed_ms:.1f}ms{compile_tag} | "
                f"device: {device} | outputs: [{', '.join(output_types_log)}] | "
                f"bindings: {list(bindings.keys())}"
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
            # Single error — human-readable first, JSON at end for frontend
            if hasattr(e, '_build_diagnostic'):
                e._build_diagnostic()
            if hasattr(e, 'diagnostic') and e.diagnostic:
                readable = e.diagnostic.render()
                payload = json.dumps([e.diagnostic.to_dict()])
                raise RuntimeError(f"{readable}\nTEX_DIAG:{payload}") from e
            # Fallback for errors without diagnostic (shouldn't happen)
            raise RuntimeError(f"TEX Error: {e}") from e
        except Exception as e:
            # Unexpected error — include traceback for debugging
            tb = traceback.format_exc()
            raise RuntimeError(f"TEX Internal Error: {e}\n{tb}") from e

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
