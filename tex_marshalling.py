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
from dataclasses import dataclass
from typing import Any, Iterable

from .tex_compiler.types import TEXType, set_array_wires, array_wires_enabled, _VEC_SIZE_TYPE
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

def _spatial_channels_to_type(c: int) -> TEXType:
    """Map a spatial buffer's channel count → a TEX type. The SINGLE policy shared by the
    LATENT-dict branch and the [B,H,W,C] tensor branch of `infer_binding_type` — which used to
    be two separate copies of this ladder (both `else → FLOAT` for C≥5), free to drift apart.
    TEX's type vocabulary tops out at vec4, so C∉{1,2,3,4} is a
    DELIBERATE refusal, not a silent `else → FLOAT` collapse (which lost every channel past
    luma at the MASK egress). C=1 is a scalar field (mask), egressed losslessly as FLOAT.

    The refusal is REACHABLE — not just via the exotic EXR/AOV path but via the mainstream
    LATENT wire: the node unwraps a >4-channel latent (SD3/Flux/Wan 16ch, LTX-2 128ch) into a
    [B,H,W,C] tensor before inference. It surfaces as a clean cook error (the node's handler
    wraps the ValueError), replacing the old silent luma-MASK that couldn't even re-wire to a
    latent consumer."""
    if c == 1:
        return TEXType.FLOAT
    if c in _VEC_SIZE_TYPE:           # {2: VEC2, 3: VEC3, 4: VEC4} — the one canonical size→vec map
        return _VEC_SIZE_TYPE[c]
    raise ValueError(
        f"a {c}-channel buffer isn't representable in TEX's types (vec4 is the widest). "
        f"A >4-channel LATENT (SD3/Flux/Wan/LTX) is not a decoded image — VAE-decode it to "
        f"an image first; a depth/AOV/multi-plane EXR needs a host with named planes (DATA-6).")


def infer_binding_type(value: Any) -> TEXType:
    """Infer the TEX type of a ComfyUI input value."""
    # Image/latent lists — use first element for type inference
    if isinstance(value, list):
        if len(value) > 0:
            return infer_binding_type(value[0])
        return TEXType.FLOAT
    if isinstance(value, dict) and "samples" in value:
        # LATENT dict — infer from the channel count (dim 1 = C before the BCHW→BHWC permute),
        # via the SAME policy as the post-unwrap [B,H,W,C] tensor branch below. The node unwraps
        # latent dicts before inference, so this branch only serves direct-API callers — but the
        # two MUST agree, else a wide latent refuses via the node yet types as FLOAT via a raw dict.
        return _spatial_channels_to_type(value["samples"].shape[1])
    if isinstance(value, str):
        return TEXType.STRING
    if isinstance(value, torch.Tensor):
        if value.dim() == 4:
            return _spatial_channels_to_type(value.shape[-1])   # [B, H, W, C]
        elif value.dim() == 3:
            # [B, H, W] — mask
            return TEXType.FLOAT
        elif value.dim() in (1, 2) and array_wires_enabled():
            # DATA-3: a low-rank tensor is an ARRAY wire under the engine profile — [N] (scalar
            # array) or [N, C] (vec array). Under ComfyUI (default) this stays FLOAT below, so a
            # host that never enables array wires is byte-identical.
            return TEXType.ARRAY
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
        TEXType.ARRAY: "ARRAY",   # DATA-3: an array wire (engine profile only)
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
    """Set the process-wide egress profile (host-level; see EGRESS_PROFILES). DATA-3: the
    `engine` profile ALSO enables ARRAY host wires (a host that carries `engine` values can carry
    array values too), and `comfy` disables them — arrays are an engine-profile capability, so
    the one host-level switch governs both."""
    global _egress_profile
    if name not in EGRESS_PROFILES:
        raise ValueError(f"unknown egress profile {name!r} (expected one of "
                         f"{', '.join(EGRESS_PROFILES)})")
    _egress_profile = name
    set_array_wires(name == "engine")


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
        # Vec image → luminance scalar. GUARD the channel count: a low-channel 4-D image
        # (a 1-/2-channel FLOAT/AOV buffer) reaches a MASK egress too, and the old
        # unguarded raw[..., 2] IndexError'd on it. Missing channels read as 0, matching
        # the IMAGE path's 2ch zero-pad, so a mask egress of any-width image is DEFINED.
        c = raw.shape[-1]
        if c >= 3:
            return LUMA_R * raw[..., 0] + LUMA_G * raw[..., 1] + LUMA_B * raw[..., 2]
        if c == 2:
            return LUMA_R * raw[..., 0] + LUMA_G * raw[..., 1]   # no blue → reads as 0
        return raw[..., 0]                                        # c == 1: the channel IS the mask
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

    if output_type == "ARRAY":
        # DATA-3: the always-on guard. ARRAY outputs are an engine-profile capability; ComfyUI's
        # IMAGE/MASK/LATENT wires cannot carry one, so refuse here regardless of whether the
        # compile-time E3203 gate ran (a cross-profile disk-cache hit could have skipped it).
        raise RuntimeError(
            "TEX Error: an ARRAY output needs a host that carries array wires (the 'engine' "
            "egress profile). ComfyUI's wires cannot represent an array — output individual "
            "elements or a string (join()) instead.")

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

    if output_type == "ARRAY":
        # DATA-3: an array wire — [N] (scalar) or [N,C] (vec) — passes through as fp32, the whole
        # point of the engine profile. A string array (Python list) passes through untouched.
        return raw.float() if isinstance(raw, torch.Tensor) else raw

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


# ── DATA-1: buffer metadata sidecar ──────────────────────────────────────────
#
# A cooked buffer is not just pixels — it has a colour interpretation (is 0.5 half the
# LIGHT, or half the perceptual lightness?) and an alpha convention (are the RGB values
# already multiplied by alpha?). ComfyUI's IMAGE wire carries none of that; a compositor's
# does. DATA-1 is the seam that carries it: a per-binding `{colorspace, premult, frame,
# host-opaque extra}` tag that rides the cook and re-attaches at egress.
#
# THREE hard boundaries (roadmap §7):
#   - Tags only, NEVER transforms. A buffer tagged `srgb` is not converted to `linear` by
#     anything here — conversions stay the user's explicit `srgb_to_linear()` call. This is
#     exactly what lets the ACES/OCIO rejection stand (DATA-1 is not a colour engine).
#   - It rides the VALUE channel (the ExecContext.time_context / cancel model), so it NEVER
#     enters a fingerprint, cache, or lineage key — a tag does not move a pixel, and keying on
#     it would split the cache for nothing. (Contrast time_context, which DOES move pixels and
#     so IS keyed.) The default ComfyUI cook supplies no tags → the whole path is dormant
#     (invariant #7).
#   - Merge on conflict is `unknown`, NEVER a silent pick. Two inputs tagged `srgb` and
#     `linear` produce an output tagged `unknown` — the honest answer, not whichever came
#     first. Losing the tag is safe; asserting the wrong one corrupts a downstream transform.

COLORSPACES = ("srgb", "linear", "oklab", "unknown")
PREMULT = ("premultiplied", "unassociated", "opaque", "unknown")


@dataclass(frozen=True)
class BufferMeta:
    """One binding's non-pixel metadata (DATA-1). All fields default to the honest `unknown`
    (or None for `frame`), so an untagged buffer is a valid, fully-`unknown` BufferMeta and
    the default path never has to special-case its absence. `extra` is a host-opaque mapping
    (a LATENT's `noise_mask`, a frame's source path) merged key-wise; keep it small and
    hashable-valued if a host means to compare it. Frozen: a buffer's tag is as immutable as
    the buffer (ENG-12) — re-tagging mints a new BufferMeta, it never mutates one in place."""
    colorspace: str = "unknown"
    premult: str = "unknown"
    frame: int | None = None
    extra: dict | None = None

    def __post_init__(self):
        if self.colorspace not in COLORSPACES:
            raise ValueError(f"unknown colorspace {self.colorspace!r} (expected one of "
                             f"{', '.join(COLORSPACES)})")
        if self.premult not in PREMULT:
            raise ValueError(f"unknown premult {self.premult!r} (expected one of "
                             f"{', '.join(PREMULT)})")


def merge_buffer_meta(metas: Iterable[BufferMeta]) -> BufferMeta:
    """The DATA-1 merge policy for an output derived from several tagged inputs: a field
    survives ONLY if every input agrees on it, else it collapses to `unknown`/None (never a
    silent pick). `extra` keeps only keys all inputs carry with an equal value. Empty input →
    all-`unknown` (nothing is known); a single input → itself. This is deliberately
    conservative in the safe direction: an output that IS srgb but reads `unknown` merely
    forfeits an advisory; an output that is linear but reads `srgb` would mislead a transform."""
    metas = [m for m in metas if m is not None]
    if not metas:
        return BufferMeta()
    if len(metas) == 1:
        return metas[0]

    def _agree(vals, absent):
        return vals[0] if len(set(vals)) == 1 else absent

    colorspace = _agree([m.colorspace for m in metas], "unknown")
    premult = _agree([m.premult for m in metas], "unknown")
    frame = _agree([m.frame for m in metas], None)
    # extra: keys present in EVERY input with an equal value (an intersection where the value
    # agrees). `k in common` already means every input carries it, so `m.extra[k]` is safe.
    common = set(metas[0].extra or {})
    for m in metas[1:]:
        common &= set(m.extra or {})
    kept = {k: metas[0].extra[k] for k in common
            if all(m.extra[k] == metas[0].extra[k] for m in metas)}
    return BufferMeta(colorspace, premult, frame, kept or None)


def egress_meta(binding_meta: dict | None, output_names: Iterable[str],
                tensor_names: set | None = None) -> dict | None:
    """The output-side tags for a cook, given the host's per-input `binding_meta` (or None).
    Every output carries the SAME merge of every TENSOR-input tag — v1 does not track which input a
    given output derived from (the LATENT `noise_mask` round-trip re-attaches to every LATENT output
    the same way). `tensor_names` is the set of ORIGINAL binding names that hold a tensor; only those
    contribute, so a scalar `$param`'s tag can't silently downgrade every output to `unknown`. The
    ENGINE supplies it (de-prefixing any fused `_s{i}_u_` stage rename via
    `tex_fusion.strip_user_prefix`), so this Runtime-layer seam stays free of fusion's naming; `None`
    means "merge every tag" (a legacy/positional caller). None in → None out (and no outputs → None),
    so the default ComfyUI path stays byte-identical."""
    if not binding_meta:
        return None
    if tensor_names is not None:
        metas = [m for n, m in binding_meta.items() if n in tensor_names]
    else:
        metas = list(binding_meta.values())
    if not metas or not output_names:
        return None
    merged = merge_buffer_meta(metas)
    return {name: merged for name in output_names}

# The DATA-1 gamma-space halo lint (W7005) lives in tex_api, not here: it is a footprint ×
# tag ANALYSIS (tex_roi + diagnostics), not a wire conversion, so it belongs in the host-facing
# facade beside check(), and keeps this low-level module free of an upward tex_roi dependency.
