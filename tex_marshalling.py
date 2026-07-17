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


# ── XPU: pinned egress + non-blocking ingestion (v0.20) ──────────────────────
#
# A mixed-device chain (CPU-cooked TEX node → CUDA-cooked TEX node) pays one H2D
# copy at ingestion. In v0.19.1 that copy was strictly serial: the source was
# ordinary pageable memory, and CUDA can only DMA asynchronously from page-locked
# (pinned) memory — PyTorch silently downgrades non_blocking=True to a sync copy
# from pageable sources. So the timeline was copy-finishes → compute-starts.
#
# v0.20 staggers them: a CPU cook (with CUDA present) writes its output into
# pinned memory (same write, page-locked allocation — torch's caching host
# allocator amortizes the lock cost across cooks), and ingestion issues
# .to("cuda", non_blocking=True) for pinned sources. The DMA rides in the
# background while Python keeps working (remaining bindings, coordinate
# builtins, interpreter dispatch); CUDA stream ordering makes the first kernel
# that touches the data wait exactly until the copy lands. No events, no manual
# sync, bit-identical — the standard DataLoader pattern. ~1.2-1.5ms hidden per
# 16MB 1024² vec4 frame on PCIe; proportionally more at 4K (67MB).
#
# All-CPU and all-GPU chains are untouched (zero copies either way); pinned
# tensors read/write like any CPU memory for a downstream CPU consumer.

_PIN_MIN_BYTES = 1 << 20    # <1MB DMAs in <0.1ms — nothing worth hiding
# Upper cap: pinned pages are unswappable, and torch's caching host allocator
# retains freed blocks for the process lifetime — an uncapped video-scale egress
# (e.g. a [81,1088,1920,3] fp32 batch ≈ 1.9GB) would permanently page-lock RAM
# the OS can never reclaim. Above the cap the copy is seconds-scale anyway; no
# amount of Python-side overlap hides it, so pageable loses nothing.
_PIN_MAX_BYTES = 256 << 20  # 256MB


def _pin_worthwhile(t) -> bool:
    """True when an egress alloc for `t` should be page-locked: it's a CPU tensor
    in the size band where a background H2D copy has real latency to hide (1MB —
    256MB), and CUDA exists to DMA to. Never raises (a wedged driver reads as
    no-CUDA)."""
    try:
        nbytes = t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0
        return (isinstance(t, torch.Tensor)
                and t.device.type == "cpu"
                and _PIN_MIN_BYTES <= nbytes <= _PIN_MAX_BYTES
                and torch.cuda.is_available())
    except Exception:
        return False


def _pinned_clamp01(raw: torch.Tensor) -> torch.Tensor:
    """`raw.clamp(0, 1)` whose fresh output tensor is page-locked when worthwhile
    (same write either way). Falls back to a pageable clamp on any host-memory
    pressure — pinning is an optimization, never a requirement."""
    if _pin_worthwhile(raw):
        try:
            out = torch.empty(raw.shape, dtype=raw.dtype, pin_memory=True)
            return torch.clamp(raw, 0.0, 1.0, out=out)
        except Exception:
            pass
    return raw.clamp(0, 1)


def _pinned_contiguous(t: torch.Tensor) -> torch.Tensor:
    """`t.contiguous()` whose materialization is page-locked when worthwhile.
    Used by the LATENT permute copies (egress AND unwrap) so latent chains keep
    pinned-ness through to ingestion. Already-contiguous tensors pass through."""
    if t.is_contiguous():
        return t
    if _pin_worthwhile(t):
        try:
            out = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            out.copy_(t)
            return out
        except Exception:
            pass
    return t.contiguous()


def to_fp32_if_int_image(t, device=None):
    """M5-INT: an integer image-like tensor binding (dim>=3, not bool) has TEX type
    VECn (float), so BOTH the interpreter and codegen cast it to fp32 at ingestion —
    otherwise the tiers diverge for FLOAT/LATENT outputs (or int64 values > 2^24).
    Scalar int params / int index arrays (dim<3) and bool masks pass through.

    `device` (optional) additionally co-locates the tensor on the compute device,
    FUSED with the cast when both apply — one copy instead of two. This makes a
    cross-device handoff (e.g. a CPU-cooked TEX node feeding a CUDA-forced one)
    a single direct hop on every tier, instead of an exception + interpreter
    retry on the codegen tiers.

    Single source for both ingestion paths (the interpreter binding loop and
    codegen's `_contiguous_bindings`); guarded by the M5-INT bit-exactness test."""
    if not isinstance(t, torch.Tensor):
        return t
    needs_cast = (t.dim() >= 3 and not t.is_floating_point()
                  and t.dtype != torch.bool)
    needs_move = device is not None and t.device != device
    if needs_cast and needs_move:
        return t.to(device=device, dtype=torch.float32)
    if needs_cast:
        return t.to(torch.float32)
    if needs_move:
        # XPU (v0.20): a pinned CPU source headed to CUDA rides the DMA engine in
        # the background — the CPU returns immediately and the rest of ingestion
        # (remaining bindings, coordinate builtins, dispatch) overlaps the copy.
        # Stream ordering makes the first kernel touching the data wait exactly
        # until the copy lands: no events, no manual sync, bit-identical. Pure
        # device move only (a dtype-converting copy stays synchronous above);
        # pageable sources take the plain path (PyTorch would silently degrade
        # non_blocking to sync anyway). D2H is never non_blocking (a CPU-side
        # read could observe an in-flight buffer).
        if getattr(device, "type", None) == "cuda" and t.is_pinned():
            return t.to(device, non_blocking=True)
        return t.to(device)
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
    # The BCHW→BHWC materialization goes to pinned memory when this latent is a
    # CPU tensor on a CUDA-capable box — without this, the unwrap copy would
    # strip the pinned-ness a pinned LATENT egress just paid for, and ingestion's
    # non-blocking H2D would silently degrade to a sync copy (XPU, v0.20).
    tensor_cl = _pinned_contiguous(samples.permute(0, 2, 3, 1))  # [B, H, W, C]
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


# ── ENG-3: egress profiles ───────────────────────────────────────────────────
#
# TEX cooks in fp32 and can produce values outside [0,1] (scene-linear highlights, a
# normal map, a signed flow field, an unpremultiplied alpha). ComfyUI's IMAGE wire
# cannot: it is 3-channel, [0,1]. So egress has always clamped, dropped alpha, and
# expanded gray -> RGB. That is CORRECT for ComfyUI and LOSSY for a compositor — the
# same node hop that ships a clean IMAGE also destroys every HDR value TEX computed.
#
# So the format is the HOST's, not the node's:
#
#   'comfy'  (default) — exactly today's behaviour, byte-identical. Canary-pinned by
#              test_eng3_comfy_profile_canary: this profile may never drift.
#   'engine' — value-preserving: fp32 BHWC, NO clamp, alpha kept, channels kept.
#              Scene-linear values survive a node hop. For a standalone host / PORT-5.
#
# Set once by the host, never per-node — a per-node toggle would let two TEX nodes in
# one graph disagree about what an IMAGE *is*, and the wire type would stop meaning
# anything. Lossless reshapes (adding a channel axis, the LATENT permute) are kept in
# both profiles; only the LOSSY steps (clamp, alpha-drop, gray-expand, 2ch-pad) are
# what 'engine' declines.

EGRESS_PROFILES = ("comfy", "engine")
_egress_profile = "comfy"


def set_egress_profile(name: str) -> None:
    """Set the process-wide egress profile (host-level; see EGRESS_PROFILES)."""
    global _egress_profile
    if name not in EGRESS_PROFILES:
        raise ValueError(f"unknown egress profile {name!r} (expected one of "
                         f"{', '.join(EGRESS_PROFILES)})")
    _egress_profile = name


def get_egress_profile() -> str:
    """The process-wide egress profile."""
    return _egress_profile


def egress_materializes(profile: str | None = None) -> bool:
    """True when `profile` (default: the process-wide one) is GUARANTEED to allocate a
    fresh tensor on the way out, so a caller that formats through it needs no ownership
    clone of its own — `tex_engine.run(plan)`'s `disown` would be dead work.

    `comfy` clamps, and a clamp always materializes: that accident is why the ComfyUI node
    never had an aliasing bug in the first place. `engine` exists precisely to REMOVE the
    lossy steps, so it may hand back a view (`_prepare_output_engine` says so itself) and
    guarantees nothing.

    This lives here because the answer is a property of the PROFILE, and profiles are this
    module's business — the engine must not consult it (ownership is a property of the
    CALL, not of a process global; that confusion is exactly what made `cook()`'s published
    no-alias guarantee false by default). It is for a caller deciding about ITS OWN egress:
    `tex_node` formats through the process-wide profile without pinning one, so the
    process-wide answer IS the node's answer, and it passes `disown=not egress_materializes()`.
    A caller that PINS a profile (`tex_cli`) must pass that profile here, not read the global
    — the global does not describe what that call will do.
    """
    return (profile or _egress_profile) == "comfy"


# ── Output formatting ──

def _reject_string_output(raw, output_type: str) -> None:
    """A string output must be declared `s@name`. Shared by every egress profile: the
    rule is TEX's, not a profile's, and the message is user-facing (ENG-3)."""
    if isinstance(raw, str):
        raise RuntimeError(
            f"TEX Error: Output is a string but inferred type is '{output_type}'. "
            f"Use s@name prefix for string outputs."
        )


def _to_mask_shape(raw: torch.Tensor) -> torch.Tensor:
    """Reduce any cooked output to MASK's [B, H, W]. Shared by every egress profile: a
    vec image becomes luminance (that IS what MASK means) and a bare [H,W]/scalar gains
    its batch axis. Range is NOT touched here — clamping is per-profile (ENG-3)."""
    if raw.dim() == 4:
        # Vec3/Vec4 image → luminance scalar
        return LUMA_R * raw[..., 0] + LUMA_G * raw[..., 1] + LUMA_B * raw[..., 2]
    if raw.dim() == 2:
        return raw.unsqueeze(0)        # [H, W] without batch dim — add it
    if raw.dim() == 0:
        return raw.view(1, 1, 1)
    return raw                         # dim == 3 ([B, H, W]) is already correct


def prepare_output(raw: torch.Tensor | str, output_type: str, *,
                   profile: str | None = None) -> Any:
    """Convert a cooked output to the host's wire format — the public egress entry point.

    Dispatches to the process-wide egress profile (ENG-3). `profile=` overrides it for
    ONE call, and exists for a host pinning its own conversion (`tex run` writes an 8-bit
    PNG, so it asks for 'comfy' regardless of what an embedding host chose) — NOT for a
    node to disagree with its neighbours about what an IMAGE is. That distinction is the
    caller's to keep; the profile itself stays host-level."""
    return _EGRESS[profile or _egress_profile](raw, output_type)


def _prepare_output_comfy(raw: torch.Tensor | str, output_type: str) -> Any:
    """The ComfyUI wire format: clamp to [0,1], drop alpha, expand gray -> RGB. Lossy by
    necessity — that IS the IMAGE contract every downstream node expects. Byte-identical
    to every version before v0.22 and canary-pinned (test_eng3_comfy_profile_canary)."""
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

    _reject_string_output(raw, output_type)

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
        # CPU cooks (with CUDA present) egress into pinned memory so a downstream
        # CUDA cook's H2D copy can overlap its Python-side setup (XPU, v0.20).
        return _pinned_clamp01(raw)

    elif output_type == "MASK":
        # MASK expects [B, H, W] float32 in [0, 1]. The shape half is shared with the
        # engine profile (ENG-3); only the clamp is this profile's own decision.
        # Keep on the compute device (see IMAGE above); pinned egress ditto.
        return _pinned_clamp01(_to_mask_shape(raw))

    elif output_type == "FLOAT":
        if raw.dim() == 0:
            return raw.item()
        return raw.float().mean().item()  # .float(): mean() rejects integer dtypes

    elif output_type == "LATENT":
        # LATENT expects [B, C, H, W] — permute back from channel-last.
        # The materializing copy goes to pinned memory on CPU cooks (see IMAGE).
        if raw.dim() == 4:
            raw = _pinned_contiguous(raw.permute(0, 3, 1, 2))
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


def _prepare_output_engine(raw: torch.Tensor | str, output_type: str) -> Any:
    """ENG-3's 'engine' profile: the value-preserving egress.

    Differs from 'comfy' in exactly the LOSSY steps, and nowhere else:
      IMAGE   no clamp, no alpha-drop, no gray->RGB expand, no 2ch pad — the cooked
              channels and magnitudes survive verbatim, normalized to fp32 BHWC.
      MASK    no clamp. A vec image still reduces to luma (that IS what MASK means),
              but a >1.0 or negative mask value is now preserved rather than crushed.
      LATENT  identical to 'comfy' — the BCHW permute is lossless and latents were
              never clamped, so there is nothing to preserve.
      scalars identical (STRING/FLOAT/INT carry no range to lose).
    Never pinned — including LATENT, which is why this profile spells out the permute
    instead of delegating: pinning is an H2D-overlap optimization for the ComfyUI node-hop
    path (XPU, v0.20), and an engine host owns its own transfers (XPU-2). Page-locking
    RAM the host never asked to be page-locked is a cost, not a favour.

    OWNERSHIP: this is a format CONVERSION, not a transfer of ownership — it may return a
    view of `raw` (e.g. an already-fp32 IMAGE passes straight through). The guarantee that
    a cooked output does not alias an INPUT BINDING belongs one level up, in
    `tex_engine.run`, which is the only layer that knows what the bindings were."""
    if output_type == "STRING":
        return _prepare_output_comfy(raw, "STRING")   # no range to preserve
    _reject_string_output(raw, output_type)

    if output_type == "IMAGE":
        if raw.dim() == 3:            # [B,H,W] -> [B,H,W,1]: add the axis, keep the value
            raw = raw.unsqueeze(-1)
        elif raw.dim() == 0:
            raw = raw.view(1, 1, 1, 1)
        return raw.float()
    if output_type == "MASK":
        return _to_mask_shape(raw).float()        # same shape rule, no clamp
    if output_type == "LATENT":
        # Same shape rule as 'comfy', minus the page-locking (see the docstring). Latents
        # were never clamped, so there is no range to preserve here — only the pinning to
        # decline.
        if raw.dim() == 4:
            return raw.permute(0, 3, 1, 2).contiguous()
        if raw.dim() == 3:
            return raw.unsqueeze(1)
        if raw.dim() == 0:
            return raw.view(1, 1, 1, 1)
        return raw
    # FLOAT / INT / unmapped: 'comfy' is already value-preserving and allocation-free for
    # these (they return Python scalars). Call the profile body directly, not back through
    # the public dispatcher — re-entering would re-run every pre-dispatch step it grows.
    return _prepare_output_comfy(raw, output_type)


# ENG-3: the profile registry. A third profile is a dict entry, not another branch in
# the dispatcher — and no profile can reach another through the public entry point.
_EGRESS = {"comfy": _prepare_output_comfy, "engine": _prepare_output_engine}
