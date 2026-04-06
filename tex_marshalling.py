"""
TEX Wrangle — marshalling and type inference utilities.

Converts between ComfyUI wire formats (IMAGE, MASK, LATENT, FLOAT, INT,
STRING) and the channel-last [B, H, W, C] tensors the TEX runtime expects.

Also provides type inference for input bindings and parameter widget
conversion (hex colors, comma-separated vectors, booleans).
"""
from __future__ import annotations

import torch
from typing import Any

from .tex_compiler.type_checker import TEXType
from .tex_runtime.stdlib import LUMA_R, LUMA_G, LUMA_B


# ── Tensor fingerprinting ──

def tensor_fingerprint(t: torch.Tensor) -> str:
    """Sample 256 evenly-spaced values from a tensor for fast hashing."""
    flat = t.flatten()
    stride = max(1, len(flat) // 256)
    samples = flat[::stride][:256]
    return f"{t.shape}:{t.dtype}:{samples.cpu().tolist()}"


# ── Latent dict handling ──

def unwrap_latent(value: dict) -> tuple[torch.Tensor, dict]:
    """
    Unwrap a ComfyUI LATENT dict into a channel-last tensor + metadata.

    Input:  {"samples": tensor [B,C,H,W], "noise_mask": ..., ...}
    Output: (tensor [B,H,W,C], {"noise_mask": ..., ...})
    """
    samples = value["samples"]  # [B, C, H, W]
    tensor_cl = samples.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
    metadata = {k: v for k, v in value.items() if k != "samples"}
    return tensor_cl, metadata


# ── Color / vector conversion ──

def hex_to_rgb(hex_str: str) -> list[float]:
    """Convert a hex color string like '#FF8800' to [R, G, B] floats in [0, 1]."""
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    if len(h) < 6:
        h = h.ljust(6, "0")
    # Only read first 6 hex chars (ignore alpha if present)
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]


# ── Parameter widget conversion ──

# Valid type_hint values for parameter widgets ($name declarations):
#   "f"  — float (default, no conversion needed)
#   "i"  — integer (no conversion needed)
#   "s"  — string (no conversion needed)
#   "b"  — boolean toggle (truthy value → 0.0/1.0 float)
#   "c"  — hex color string (e.g. "#FF8800" → [R, G, B] float list)
#   "v2" — vec2 comma string (e.g. "1.0, 2.0" → [1.0, 2.0])
#   "v3" — vec3 comma string (e.g. "1, 2, 3" → [1.0, 2.0, 3.0])
#   "v4" — vec4 comma string (e.g. "1, 2, 3, 4" → [1.0, 2.0, 3.0, 4.0])

def convert_param_value(value: Any, param_info: dict) -> Any:
    """Convert a param widget value to the appropriate Python type for the interpreter.

    Boolean toggles → float 0/1, color hex → [R,G,B] list,
    vec2/vec3 comma strings → list of floats. Other values pass through.
    """
    hint = param_info.get("type_hint", "f")
    if hint in ("f", "i", "s"):
        return value  # common cases need no conversion
    if hint == "b":
        if isinstance(value, (bool, int, float)):
            # bool() clamps any truthy number to 1.0 (e.g. 2 → True → 1.0)
            return float(bool(value))
    elif hint == "c" and isinstance(value, str):
        if value.startswith("#"):
            try:
                return hex_to_rgb(value)
            except ValueError:
                pass
        else:
            # Comma-separated RGB floats (0-1): "0.5, 0.3, 0.1"
            try:
                parts = [float(x.strip()) for x in value.split(",")]
                if len(parts) >= 3:
                    return parts[:3]
            except ValueError:
                pass
    elif hint in ("v2", "v3", "v4") and isinstance(value, str):
        expected = {"v2": 2, "v3": 3, "v4": 4}[hint]
        try:
            parts = [float(x.strip()) for x in value.split(",")]
            # Pad or truncate to expected component count
            if len(parts) < expected:
                parts.extend([0.0] * (expected - len(parts)))
            elif len(parts) > expected:
                parts = parts[:expected]
            return parts
        except ValueError:
            pass
    return value


# ── Type inference ──

def infer_binding_type(value: Any) -> TEXType:
    """Infer the TEX type of a ComfyUI input value."""
    # Image/latent lists — use first element for type inference
    if isinstance(value, list):
        if len(value) > 0:
            return infer_binding_type(value[0])
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
            elif c == 2:
                return TEXType.VEC2
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


def map_inferred_type(inferred: TEXType | None, has_latent_input: bool) -> str:
    """Map an inferred TEXType to a ComfyUI output_type string."""
    if inferred is None:
        return "IMAGE"
    return {
        TEXType.VEC4: "LATENT" if has_latent_input else "IMAGE",
        TEXType.VEC3: "IMAGE",
        TEXType.VEC2: "IMAGE",
        TEXType.FLOAT: "MASK",
        TEXType.INT: "INT",
        TEXType.STRING: "STRING",
    }.get(inferred, "IMAGE")


# ── Output formatting ──

def prepare_output(raw: torch.Tensor | str, output_type: str) -> Any:
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
        elif raw.dim() == 4 and raw.shape[-1] == 2:
            # [B, H, W, 2] -> [B, H, W, 3] (pad with zeros)
            raw = torch.nn.functional.pad(raw, (0, 1))
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
            raw = LUMA_R * raw[..., 0] + LUMA_G * raw[..., 1] + LUMA_B * raw[..., 2]
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
