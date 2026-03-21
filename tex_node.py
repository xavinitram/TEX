"""
TEX Wrangle Node — ComfyUI custom node for the TEX (Tensor Expression Language).

Provides a single node where users write compact TEX scripts to process
images, masks, and scalar values using per-pixel tensor operations.
"""
from __future__ import annotations

import hashlib
import logging
import time
import torch
import traceback
from typing import Any

logger = logging.getLogger("TEX")

from .tex_compiler.lexer import LexerError
from .tex_compiler.parser import ParseError
from .tex_compiler.type_checker import TypeCheckError, TEXType
from .tex_runtime.interpreter import Interpreter, InterpreterError
from .tex_cache import get_cache
from .tex_runtime.compiled import execute_compiled


# ── Cached interpreter (reused across executions to avoid rebuild overhead) ──
_interpreter: Interpreter | None = None


def _get_interpreter() -> Interpreter:
    """Get or create the cached Interpreter singleton."""
    global _interpreter
    if _interpreter is None:
        _interpreter = Interpreter()
    return _interpreter


# Wildcard type that matches any ComfyUI type
ANY_TYPE = "*"


class ContainsAnyDict(dict):
    """Dict subclass that claims to contain any key.

    Used to let ComfyUI accept dynamically-named inputs created by the
    frontend JS extension.  Official pattern from ComfyUI docs:
    https://docs.comfy.org/custom-nodes/backend/more_on_inputs
    """

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            # Dynamic inputs (e.g. @A, @B) — return wildcard type
            return (ANY_TYPE,)


def _tensor_fingerprint(t: torch.Tensor) -> str:
    """Sample 256 evenly-spaced values from a tensor for fast hashing."""
    flat = t.flatten()
    stride = max(1, len(flat) // 256)
    samples = flat[::stride][:256]
    return f"{t.shape}:{t.dtype}:{samples.cpu().tolist()}"


def _unwrap_latent(value: dict) -> tuple[torch.Tensor, dict]:
    """
    Unwrap a ComfyUI LATENT dict into a channel-last tensor + metadata.

    Input:  {"samples": tensor [B,C,H,W], "noise_mask": ..., ...}
    Output: (tensor [B,H,W,C], {"noise_mask": ..., ...})
    """
    samples = value["samples"]  # [B, C, H, W]
    tensor_cl = samples.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
    metadata = {k: v for k, v in value.items() if k != "samples"}
    return tensor_cl, metadata


def _infer_binding_type(value: Any) -> TEXType:
    """Infer the TEX type of a ComfyUI input value."""
    # Image/latent lists — use first element for type inference
    if isinstance(value, list):
        if len(value) > 0:
            return _infer_binding_type(value[0])
        return TEXType.FLOAT
    if isinstance(value, dict) and "samples" in value:
        # LATENT dict — infer from channel count (dim 1 = C before permute)
        c = value["samples"].shape[1]
        if c == 3:
            return TEXType.VEC3
        return TEXType.VEC4
    if isinstance(value, str):
        return TEXType.STRING
    if isinstance(value, torch.Tensor):
        if value.dim() == 4:
            # [B, H, W, C]
            c = value.shape[-1]
            if c == 4:
                return TEXType.VEC4
            elif c == 3:
                return TEXType.VEC3
            else:
                return TEXType.FLOAT
        elif value.dim() == 3:
            # [B, H, W] — mask
            return TEXType.FLOAT
        else:
            return TEXType.FLOAT
    elif isinstance(value, (int, bool)):
        return TEXType.INT
    elif isinstance(value, float):
        return TEXType.FLOAT
    return TEXType.FLOAT


def _prepare_output(raw: torch.Tensor | str, output_type: str) -> Any:
    """Convert the interpreter's raw output to the expected ComfyUI format."""
    if output_type == "STRING":
        if isinstance(raw, str):
            return raw
        # Tensor -> string: convert scalar to string
        if isinstance(raw, torch.Tensor):
            if raw.dim() == 0:
                v = raw.item()
                return str(int(v)) if v == int(v) else str(v)
            return str(raw.float().mean().item())
        return str(raw)

    # For non-STRING outputs, if raw is a string, error
    if isinstance(raw, str):
        raise RuntimeError(
            f"TEX Error: Output is a string but inferred type is '{output_type}'. "
            f"Use s@name prefix for string outputs."
        )

    if output_type == "IMAGE":
        # IMAGE expects [B, H, W, C] float32 in [0, 1]
        if raw.dim() == 3:
            # [B, H, W] -> [B, H, W, 3] (grayscale to RGB)
            raw = raw.unsqueeze(-1).expand(-1, -1, -1, 3)
        elif raw.dim() == 4 and raw.shape[-1] == 3:
            pass  # already [B, H, W, 3]
        elif raw.dim() == 4 and raw.shape[-1] == 4:
            # Drop alpha for IMAGE output (ComfyUI standard is 3-channel)
            raw = raw[..., :3]
        elif raw.dim() == 0:
            # Scalar -> 1x1 image
            raw = raw.view(1, 1, 1, 1).expand(1, 1, 1, 3)
        return raw.clamp(0, 1).cpu()

    elif output_type == "MASK":
        # MASK expects [B, H, W] float32 in [0, 1]
        if raw.dim() == 4:
            # Vec3/Vec4 image → luminance scalar
            raw = 0.2126 * raw[..., 0] + 0.7152 * raw[..., 1] + 0.0722 * raw[..., 2]
        elif raw.dim() == 2:
            # [H, W] without batch dim — add it
            raw = raw.unsqueeze(0)
        elif raw.dim() == 0:
            raw = raw.view(1, 1, 1)
        # dim == 3 ([B, H, W]) is the correct format — pass through
        return raw.clamp(0, 1).cpu()

    elif output_type == "FLOAT":
        if raw.dim() == 0:
            return raw.item()
        return raw.mean().item()

    elif output_type == "LATENT":
        # LATENT expects [B, C, H, W] — permute back from channel-last
        if raw.dim() == 4:
            raw = raw.permute(0, 3, 1, 2).contiguous()
        elif raw.dim() == 3:
            # [B, H, W] -> [B, 1, H, W]
            raw = raw.unsqueeze(1)
        elif raw.dim() == 0:
            raw = raw.view(1, 1, 1, 1)
        # NO clamping — latent values are not bounded to [0,1]
        return raw

    elif output_type == "INT":
        if raw.dim() == 0:
            return int(raw.item())
        return int(raw.mean().item())

    return raw.cpu()


def _map_inferred_type(inferred: TEXType | None, has_latent_input: bool) -> str:
    """Map an inferred TEXType to a ComfyUI output_type string."""
    if inferred is None:
        return "IMAGE"
    return {
        TEXType.VEC4: "LATENT" if has_latent_input else "IMAGE",
        TEXType.VEC3: "IMAGE",
        TEXType.FLOAT: "MASK",
        TEXType.INT: "INT",
        TEXType.STRING: "STRING",
    }.get(inferred, "IMAGE")


class TEXWrangleNode:
    """
    TEX Wrangle — per-pixel tensor expression processor.

    Write compact TEX scripts to process images, masks, and scalars.
    Reference inputs with @name (e.g. @A, @base_image, @strength).
    Write output to @OUT.

    Example:
        float gray = luma(@A);
        @OUT = vec4(gray, gray, gray, 1.0);
    """

    CATEGORY = "TEX"
    FUNCTION = "execute"
    DESCRIPTION = (
        "TEX Wrangle: per-pixel tensor expressions for images, masks, and scalars.\n"
        "Reference inputs with @name (e.g. @A, @base_image).\n"
        "Write outputs with @name = expr (e.g. @OUT, @mask, @result).\n"
        "Add parameter widgets with $name (e.g. f$strength = 0.5).\n"
        "Supports float, int, vec3, vec4, string types with if/else, for loops, and 60+ stdlib functions.\n"
        "Click the ? icon for a quick reference."
    )

    # Maximum number of output slots (pre-allocated for dynamic outputs)
    MAX_OUTPUTS = 8

    # System kwargs that are NOT TEX bindings
    _SYSTEM_KWARGS = {"device", "compile_mode", "output_type", "_tex_any"}

    @classmethod
    def INPUT_TYPES(cls):
        # ContainsAnyDict lets the backend accept dynamically-named inputs
        # created by the frontend JS extension.  The frontend parses TEX code
        # for @name references and creates matching sockets on the fly.
        optional = ContainsAnyDict()

        # System parameters (always present as combo widgets)
        optional["device"] = (
            ["auto", "cpu", "cuda"],
            {
                "default": "auto",
                "tooltip": "Execution device. auto: follows input tensors. cpu/cuda: force a specific device.",
            },
        )
        optional["compile_mode"] = (
            ["none", "torch_compile"],
            {
                "default": "none",
                "tooltip": "none: standard interpreter. torch_compile: JIT-compile via torch.compile (falls back on failure).",
            },
        )
        # NOTE: output_type was removed in v0.3. Old workflows that pass it
        # will still work — ContainsAnyDict accepts any key, and execute()
        # pops it from kwargs before processing bindings.

        # Register a wildcard-type input so the node appears in the
        # search panel when dragging a wire of any type.  The frontend
        # extension removes all initial input slots in onNodeCreated and
        # manages them dynamically, so this slot never actually renders.
        optional["_tex_any"] = (ANY_TYPE, {
            "tooltip": "TEX accepts any input type. Use @name in code to reference it.",
        })

        return {
            "required": {
                "code": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "// TEX Wrangle\n"
                               "// Read inputs with @A, @B, etc.\n"
                               "// Write output to @OUT\n\n"
                               "float gray = luma(@IN);\n"
                               "@OUT = vec3(gray);\n",
                    "placeholder": "// TEX code here...",
                    "tooltip": "TEX source code. Use @name for inputs, @name = expr for outputs. Use $name for parameter widgets.",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = tuple([ANY_TYPE] * 8)
    RETURN_NAMES = tuple([f"out_{i}" for i in range(8)])
    OUTPUT_TOOLTIPS = tuple(["TEX output. Type auto-inferred from code."] * 8)

    @classmethod
    def IS_CHANGED(cls, code, **kwargs):
        """Cache-busting: re-execute if code, inputs, device, or compile_mode change."""
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

    def execute(self, code: str, **kwargs) -> tuple:
        """Execute TEX code with the provided inputs."""
        start_time = time.perf_counter()

        # Pop system parameters so they don't get treated as bindings
        device_mode = kwargs.pop("device", "auto")
        compile_mode = kwargs.pop("compile_mode", "none")
        output_type_compat = kwargs.pop("output_type", "auto")  # deprecated, ignored
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
        device = self._resolve_device(device_mode, bindings)

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
                    "TEX program has no outputs. Assign to @OUT or another @name."
                )
            if len(output_names) > self.MAX_OUTPUTS:
                raise InterpreterError(
                    f"TEX program has {len(output_names)} outputs, "
                    f"exceeding the maximum ({self.MAX_OUTPUTS})."
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
                        f"TEX code references {sigil}{ref_name} but no input is connected to slot '{ref_name}'."
                    )

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
            while len(results) < self.MAX_OUTPUTS:
                results.append(None)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            compile_tag = " [compiled]" if compile_mode == "torch_compile" else ""
            logger.info(
                f"Executed in {elapsed_ms:.1f}ms{compile_tag} | "
                f"device: {device} | outputs: [{', '.join(output_types_log)}] | "
                f"bindings: {list(bindings.keys())}"
            )

            return tuple(results)

        except (LexerError, ParseError, TypeCheckError, InterpreterError) as e:
            # Clean user-facing error
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
        for val in bindings.values():
            if isinstance(val, str):
                continue
            if isinstance(val, dict) and "samples" in val:
                if val["samples"].is_cuda:
                    return str(val["samples"].device)
            elif isinstance(val, torch.Tensor) and val.is_cuda:
                return str(val.device)
        return "cpu"
