"""
TEX Wrangle — storage I/O (DATA-2).

TEX cooks in fp32 and its wire is fp32 (invariant: the compute type never changes). But a
buffer is *stored* in something narrower — an 8-bit PNG, a 16-bit PNG, a half or float EXR.
`BufferDesc` names that storage envelope, and this package resolves it at the ONE place it
matters: ingestion (decode storage → fp32) and egress (encode fp32 → storage). Everything in
between is fp32, exactly as a uint8 PNG has always been decoded to fp32 [0,1] and re-encoded
on the way out — half and uint16 are just more storage dtypes on the same seam.

Boundary (roadmap §7): a storage dtype is NOT whole-pipeline fp16. `BufferDesc(storage=
"float16")` means "this file holds halfs"; the moment it enters TEX it is fp32, and it only
becomes half again at write. Compute and wire stay fp32.

`exr.py` is the EXR reader/writer — pure torch (struct + zlib + torch.frombuffer), because the
numpy-based OpenEXR bindings are banned (invariant #1). NONE/ZIP/ZIPS scanline only.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch


def _raw_bytes(t: torch.Tensor) -> bytes:
    """Reinterpret a tensor's storage as raw bytes in native (little-endian) order, numpy-free —
    the fast path the EXR/PNG writers hand to zlib/disk instead of `struct.pack(*tensor.tolist())`.
    The tensor is made contiguous and BOUND TO A LOCAL that outlives the `string_at` read, so
    passing even a temporary (`_raw_bytes(x.to(torch.uint8))`) is safe — the argument holds the
    reference for the call, unlike the inline `x.contiguous().data_ptr()` form which can free the
    contiguous copy before the read (a use-after-free). Length is `numel × element_size`, correct
    for any dtype. The one reviewed home for this reinterpret + its contiguity/lifetime contract.
    Self-defends against a device pointer: a CUDA `data_ptr()` read as host memory is a segfault,
    so a stray CUDA tensor is copied to CPU here (a no-op for every present caller, which already
    `.cpu()`s upstream) rather than left to crash."""
    t = t.cpu().contiguous()
    return ctypes.string_at(t.data_ptr(), t.numel() * t.element_size())


# Storage dtypes a buffer can live in on disk / the wire. Integer dtypes encode a normalized
# [0,1] range (like uint8 always has); float dtypes hold real values verbatim (HDR-capable).
STORAGE_DTYPES = ("uint8", "uint16", "float16", "float32")

_STORAGE_TORCH = {
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "float16": torch.float16,
    "float32": torch.float32,
}
_INT_MAXVAL = {"uint8": 255.0, "uint16": 65535.0}


@dataclass(frozen=True)
class BufferDesc:
    """How a buffer is STORED (DATA-2), resolved only at the ingestion/egress seam. `storage`
    is one of STORAGE_DTYPES; `transfer` is an optional colour-transfer hint that pairs with
    DATA-1's `BufferMeta.colorspace` (e.g. a uint8 PNG is conventionally `srgb`-encoded, a
    float EXR conventionally `linear`) — a HINT the loader can attach as a tag, never a
    transform applied here. Frozen; the default is the identity (fp32, no conversion)."""
    storage: str = "float32"
    transfer: str = "unknown"

    def __post_init__(self):
        if self.storage not in STORAGE_DTYPES:
            raise ValueError(f"unknown storage dtype {self.storage!r} (expected one of "
                             f"{', '.join(STORAGE_DTYPES)})")

    @property
    def is_float(self) -> bool:
        return self.storage in ("float16", "float32")


def decode_to_fp32(t: torch.Tensor, desc: BufferDesc) -> torch.Tensor:
    """Storage tensor → fp32, at ingestion. Integer storage divides by its max (uint8 → [0,1],
    exactly as today); float storage widens to fp32. The single inbound conversion — nothing
    downstream sees anything but fp32."""
    if desc.is_float:
        return t.to(torch.float32)
    return t.to(torch.float32) / _INT_MAXVAL[desc.storage]


def encode_from_fp32(t: torch.Tensor, desc: BufferDesc) -> torch.Tensor:
    """fp32 → storage tensor, at egress. Integer storage clamps to [0,1] and rounds (not
    truncates — the bit-exact uint8↔float round-trip the CLI already relies on); float storage
    narrows (float16) or passes through (float32). The single outbound conversion."""
    if desc.is_float:
        return t.to(_STORAGE_TORCH[desc.storage])
    maxval = _INT_MAXVAL[desc.storage]
    return (t.to(torch.float32).clamp(0.0, 1.0) * maxval).round().to(_STORAGE_TORCH[desc.storage])
