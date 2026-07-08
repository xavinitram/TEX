"""
TEX Wrangle — marshalling and type inference utilities.

Converts between ComfyUI wire formats (IMAGE, MASK, LATENT, FLOAT, INT,
STRING) and the channel-last [B, H, W, C] tensors the TEX runtime expects.

Also provides type inference for input bindings and parameter widget
conversion (hex colors, comma-separated vectors, booleans).
"""
from __future__ import annotations

import hashlib
import struct
import torch
from typing import Any

from .tex_compiler.types import TEXType
from .tex_runtime.stdlib import LUMA_R, LUMA_G, LUMA_B


def to_fp32_if_int_image(t):
    """M5-INT: an integer image-like tensor binding (dim>=3, not bool) has TEX type
    VECn (float), so BOTH the interpreter and codegen cast it to fp32 at ingestion —
    otherwise the tiers diverge for FLOAT/LATENT outputs (or int64 values > 2^24).
    Scalar int params / int index arrays (dim<3) and bool masks pass through.

    Single source for both ingestion paths (the interpreter binding loop and
    codegen's `_contiguous_bindings`); guarded by the M5-INT bit-exactness test."""
    if (isinstance(t, torch.Tensor) and t.dim() >= 3
            and not t.is_floating_point() and t.dtype != torch.bool):
        return t.to(torch.float32)
    return t


# ── Tensor fingerprinting ──

def tensor_fingerprint(t: torch.Tensor) -> str:
    """Hash a tensor cheaply: 256 strided samples PLUS whole-tensor reductions.

    The strided sample alone collides on localized edits (a single off-stride
    pixel, a painted mask band, a small moved object) — distinct inputs would
    then reuse a stale cached result. numel + sum + mean over all elements catch
    those changes. (sum/mean add one batched GPU sync, ~0.07ms at 512x512.)
    """
    flat = t.flatten()
    n = flat.numel()
    stride = max(1, n // 256)
    samples = flat[::stride][:256].float()
    # sum(dtype=float32) reduces without copying the whole tensor; fold it into
    # the SAME host transfer as the samples (one GPU->CPU sync), then derive the
    # mean host-side.
    s = t.sum(dtype=torch.float32)
    host = torch.cat([samples, s.reshape(1)]).cpu()
    sample_vals = host[:-1].tolist()
    total = host[-1].item()
    mean = total / max(n, 1)
    # Hash the packed sample bytes instead of formatting 256 floats into the key
    # (the old f-string over the full list was ~25x the cost). struct.pack is
    # C-level and numpy-free — TEX is torch-only and CI runs PyTorch without numpy.
    digest = hashlib.sha256(struct.pack(f"{len(sample_vals)}f", *sample_vals)).hexdigest()[:16]
    return f"{t.shape}:{t.dtype}:{n}:{total:.6g}:{mean:.6g}:{digest}"


# ── Latent dict handling ──

def unwrap_latent(value: dict) -> tuple[torch.Tensor, dict]:
    """
    Unwrap a ComfyUI LATENT dict into a channel-last tensor + metadata.

    Input:  {"samples": tensor [B,C,H,W], "noise_mask": ..., ...}
    Output: (tensor [B,H,W,C], {"noise_mask": ..., ...})
    """
    samples = value["samples"]  # [B, C, H, W]
    if not isinstance(samples, torch.Tensor) or samples.dim() != 4:
        got = (f"a {samples.dim()}D tensor" if isinstance(samples, torch.Tensor)
               else type(samples).__name__)
        raise ValueError(
            f"This LATENT input's 'samples' isn't the expected [batch, channels, height, "
            f"width] shape (got {got}). This usually means an upstream node produced an "
            f"empty or malformed latent — try re-running it.")
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

def convert_param_value(value: Any, param_info: dict, param_name: str = "") -> Any:
    """Convert a param widget value to the appropriate Python type for the interpreter.

    Boolean toggles → float 0/1, color hex → [R,G,B] list,
    vec2/vec3 comma strings → list of floats. Non-string values (already in the
    target form) pass through. A malformed STRING for a c/v* hint raises a clear,
    param-named error instead of silently passing through (which previously
    surfaced as a confusing downstream error with no mention of the parameter).
    """
    hint = param_info.get("type_hint", "f")
    if hint in ("f", "i", "s"):
        return value  # common cases need no conversion
    if hint == "b":
        if isinstance(value, (bool, int, float)):
            # bool() clamps any truthy number to 1.0 (e.g. 2 → True → 1.0)
            return float(bool(value))
        return value  # already-typed boolean value — leave as-is
    label = f"${param_name}" if param_name else "parameter"
    if hint == "c" and isinstance(value, str):
        if value.startswith("#"):
            try:
                return hex_to_rgb(value)
            except ValueError:
                raise ValueError(
                    f"Parameter {label}: '{value}' is not a valid hex color "
                    f"(expected like #FF8800).")
        # Comma-separated RGB floats (0-1): "0.5, 0.3, 0.1"
        try:
            parts = [float(x.strip()) for x in value.split(",")]
        except ValueError:
            raise ValueError(
                f"Parameter {label}: color '{value}' must be a hex string "
                f"(#RRGGBB) or comma-separated RGB floats.")
        if len(parts) >= 3:
            return parts[:3]
        raise ValueError(
            f"Parameter {label}: color '{value}' needs at least 3 values (R, G, B).")
    elif hint in ("v2", "v3", "v4") and isinstance(value, str):
        expected = {"v2": 2, "v3": 3, "v4": 4}[hint]
        try:
            parts = [float(x.strip()) for x in value.split(",")]
        except ValueError:
            raise ValueError(
                f"Parameter {label}: {hint} value '{value}' must be "
                f"comma-separated numbers (e.g. \"1.0, 2.0\").")
        # Pad or truncate to expected component count
        if len(parts) < expected:
            parts.extend([0.0] * (expected - len(parts)))
        elif len(parts) > expected:
            parts = parts[:expected]
        return parts
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
        # LATENT dict — infer from channel count (dim 1 = C before permute),
        # using the same mapping as the post-unwrap [B,H,W,C] tensor branch
        # below (the node unwraps latent dicts before inference, so this
        # branch only serves direct API callers).
        c = value["samples"].shape[1]
        if c == 4:
            return TEXType.VEC4
        elif c == 3:
            return TEXType.VEC3
        elif c == 2:
            return TEXType.VEC2
        else:
            return TEXType.FLOAT
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
        elif raw.dim() == 4 and raw.shape[-1] == 1:
            # [B, H, W, 1] -> [B, H, W, 3] (grayscale to RGB)
            raw = raw.expand(-1, -1, -1, 3)
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
        # Keep on the compute device (GPU stays GPU): forcing .cpu() here makes
        # chained TEX nodes round-trip over PCIe per link and collapses the next
        # node's auto-device to CPU. Terminal nodes (Save/Preview) .cpu() themselves.
        return raw.clamp(0, 1)

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
        # dim == 3 ([B, H, W]) is the correct format — pass through.
        # Keep on the compute device (see IMAGE above).
        return raw.clamp(0, 1)

    elif output_type == "FLOAT":
        if raw.dim() == 0:
            return raw.item()
        return raw.float().mean().item()  # .float(): mean() rejects integer dtypes

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
        return int(raw.float().mean().item())  # .float(): mean() rejects integer dtypes

    # Unmapped output types pass through unchanged — keep on the compute
    # device (see IMAGE above). Unreachable from the node (map_inferred_type
    # only emits the strings handled above); serves direct prepare_output
    # callers (see DEVELOPMENT.md extension points).
    return raw
