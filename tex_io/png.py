"""
Minimal 16-bit PNG writer (DATA-2).

torchvision's `encode_png` is uint8-only, so a 16-bit sink needs a hand-rolled encoder —
which the PNG container makes small: a signature, three chunks (IHDR / IDAT / IEND), each
length-prefixed and CRC32-tailed (zlib.crc32, already a dependency). Pixel data is filter-type
0 (None) scanlines, samples big-endian per the spec, deflated in one IDAT. struct + zlib, no
numpy (invariant #1). Reading 16-bit PNG stays torchvision's job (`decode_image` handles it) —
this is write-only, the one direction torchvision can't do.
"""
from __future__ import annotations

import struct
import zlib

import torch

from . import _raw_bytes

_SIG = b"\x89PNG\r\n\x1a\n"
# PNG colour types keyed by channel count: 1 gray, 2 gray+alpha, 3 RGB, 4 RGBA.
_COLOR_TYPE = {1: 0, 2: 4, 3: 2, 4: 6}


def _chunk(typ: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + typ + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))


def write_png16(path, u16: torch.Tensor) -> None:
    """Write a uint16 [H,W,C] tensor (C in 1..4) to a 16-bit PNG. Samples are stored
    big-endian (the PNG convention); a filter-0 byte leads each scanline. Values must already
    be the integer samples (0..65535) — the fp32→uint16 mapping is `tex_io.encode_from_fp32`."""
    if u16.dim() != 3 or u16.shape[-1] not in _COLOR_TYPE:
        raise ValueError(f"write_png16 expects [H,W,C] with C in 1..4, got {tuple(u16.shape)}")
    if u16.dtype != torch.uint16:
        raise ValueError(f"write_png16 expects a uint16 tensor, got {u16.dtype}")
    H, W, C = u16.shape
    u16 = u16.cpu().contiguous()

    ihdr = struct.pack(">IIBBBBB", W, H, 16, _COLOR_TYPE[C], 0, 0, 0)
    # P1: build the whole raw image (a filter-0 byte + big-endian samples per scanline) with
    # vectorized torch ops + ONE ctypes memcpy, instead of struct.pack('>...H', *tolist()) per
    # scanline. PNG samples are big-endian; torch stores uint16 little-endian, so swap each 2-byte
    # pair. Bit-identical payload; measured ~25x less marshalling at megapixel sizes.
    be = u16.view(torch.uint8).reshape(H, W * C, 2).flip(-1)     # [H, W*C, 2] -> big-endian bytes
    scan = torch.zeros(H, 1 + W * C * 2, dtype=torch.uint8)      # col 0 = filter-type-0 byte
    scan[:, 1:] = be.reshape(H, W * C * 2)
    idat = zlib.compress(_raw_bytes(scan), 6)

    with open(path, "wb") as f:
        f.write(_SIG)
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", idat))
        f.write(_chunk(b"IEND", b""))
