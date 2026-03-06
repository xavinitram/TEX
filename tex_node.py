"""
TEX Wrangle Node — ComfyUI custom node for the TEX (Tensor Expression Language).

Provides a single node where users write compact TEX scripts to process
images, masks, and scalar values using per-pixel tensor operations.
"""
from __future__ import annotations

import hashlib
import time
import torch
import traceback
from typing import Any

from .tex_compiler.lexer import LexerError
from .tex_compiler.parser import ParseError
from .tex_compiler.type_checker import TypeCheckError, TEXType
from .tex_runtime.interpreter import Interpreter, InterpreterError
from .tex_cache import get_cache
from .tex_runtime.compiled import execute_compiled


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
            f"TEX Error: @OUT is a string but output_type is '{output_type}'. "
            f"Set output_type to 'STRING' when using string output."
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
            # Vec -> luminance
            raw = 0.2126 * raw[..., 0] + 0.7152 * raw[..., 1] + 0.0722 * raw[..., 2]
        elif raw.dim() == 0:
            raw = raw.view(1, 1, 1)
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
        "Reference inputs with @name (e.g. @A, @base_image), write to @OUT.\n"
        "Supports float, int, vec3, vec4, string types with if/else, for loops, and 60+ stdlib functions.\n"
        "Click the ? icon for a quick reference."
    )

    # System kwargs that are NOT TEX bindings
    _SYSTEM_KWARGS = {"device", "compile_mode"}

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

        return {
            "required": {
                "code": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "// TEX Wrangle — write your expression here\n"
                               "// Reference inputs with @A, @B, etc.\n"
                               "// Write output to @OUT\n\n"
                               "float gray = luma(@A);\n"
                               "@OUT = vec3(gray, gray, gray);\n",
                    "placeholder": "// TEX code here...",
                    "tooltip": "TEX source code. Use @name to reference inputs, @OUT for output. Types: float, int, vec3, vec4, string.",
                }),
                "output_type": (["auto", "IMAGE", "MASK", "LATENT", "FLOAT", "INT", "STRING"], {
                    "default": "auto",
                    "tooltip": "Output format. auto: inferred from code. IMAGE: [B,H,W,3]. MASK: [B,H,W]. LATENT: [B,C,H,W]. FLOAT/INT: scalar. STRING: text.",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("result",)
    OUTPUT_TOOLTIPS = ("TEX result. Format depends on output_type: IMAGE [B,H,W,3], MASK [B,H,W], LATENT [B,C,H,W], FLOAT, or INT.",)

    @classmethod
    def IS_CHANGED(cls, code, output_type, **kwargs):
        """Cache-busting: re-execute if code, inputs, device, or compile_mode change."""
        parts = [code, output_type]
        # Include device and compile_mode in the hash
        parts.append(kwargs.get("device", "auto"))
        parts.append(kwargs.get("compile_mode", "none"))
        # Hash all binding inputs (everything except system kwargs)
        for name in sorted(kwargs.keys()):
            if name in cls._SYSTEM_KWARGS:
                continue
            val = kwargs[name]
            if val is not None:
                if isinstance(val, dict) and "samples" in val:
                    s = val["samples"]
                    parts.append(f"{name}:LATENT:{s.shape}:{s.dtype}:{s.device}")
                elif isinstance(val, torch.Tensor):
                    parts.append(f"{name}:{val.shape}:{val.dtype}:{val.device}")
                else:
                    parts.append(f"{name}:{val}")
        return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()

    def execute(self, code: str, output_type: str, **kwargs) -> tuple:
        """Execute TEX code with the provided inputs."""
        start_time = time.perf_counter()

        # Pop system parameters so they don't get treated as bindings
        device_mode = kwargs.pop("device", "auto")
        compile_mode = kwargs.pop("compile_mode", "none")

        # Everything remaining in kwargs is a TEX binding
        bindings: dict[str, Any] = {
            name: val for name, val in kwargs.items() if val is not None
        }

        # Unwrap LATENT dicts into channel-last tensors, preserving metadata
        latent_meta: dict[str, Any] = {}
        latent_channel_count = 0
        for name, val in list(bindings.items()):
            if isinstance(val, str):
                continue  # strings pass through as-is
            if isinstance(val, dict) and "samples" in val:
                tensor_cl, meta = _unwrap_latent(val)
                bindings[name] = tensor_cl
                if not latent_meta:
                    latent_meta = meta
                    latent_channel_count = tensor_cl.shape[-1]

        # Resolve target device
        device = self._resolve_device(device_mode, bindings)

        try:
            # Infer binding types
            binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}

            # Set output type: auto mode omits OUT from binding_types (triggers inference)
            if output_type != "auto":
                out_type_map = {
                    "IMAGE": TEXType.VEC4,
                    "MASK": TEXType.FLOAT,
                    "LATENT": TEXType.VEC4,
                    "FLOAT": TEXType.FLOAT,
                    "INT": TEXType.INT,
                    "STRING": TEXType.STRING,
                }
                binding_types["OUT"] = out_type_map.get(output_type, TEXType.VEC4)

            # Compile (uses two-tier Mega-Cache: memory LRU + disk persistence)
            cache = get_cache()
            program, type_map, referenced, inferred_out = cache.compile_tex(code, binding_types)

            # Resolve effective output type
            if output_type == "auto":
                effective_output_type = _map_inferred_type(inferred_out, bool(latent_meta))
            else:
                effective_output_type = output_type

            # Check that referenced bindings are available
            for ref_name in referenced:
                if ref_name != "OUT" and ref_name not in bindings:
                    raise InterpreterError(
                        f"TEX code references @{ref_name} but no input is connected to slot '{ref_name}'."
                    )

            # Execute — optionally with torch.compile acceleration
            if compile_mode == "torch_compile":
                fp = cache.fingerprint(code, binding_types)
                raw_output = execute_compiled(program, bindings, type_map, device, fp,
                                              latent_channel_count=latent_channel_count)
            else:
                interp = Interpreter()
                raw_output = interp.execute(program, bindings, type_map, device=device,
                                            latent_channel_count=latent_channel_count)

            # Format output
            result = _prepare_output(raw_output, effective_output_type)

            # Wrap LATENT output in dict with preserved metadata
            if effective_output_type == "LATENT":
                result = {"samples": result, **latent_meta}

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            compile_tag = " [compiled]" if compile_mode == "torch_compile" else ""
            auto_tag = f" (auto->{effective_output_type})" if output_type == "auto" else ""
            print(f"[TEX] Executed in {elapsed_ms:.1f}ms{compile_tag} | "
                  f"device: {device} | output: {effective_output_type}{auto_tag} | "
                  f"bindings: {list(bindings.keys())}")

            return (result,)

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
