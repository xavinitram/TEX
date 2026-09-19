"""
Pure-torch scanline OpenEXR reader/writer (DATA-2).

WHY pure torch: the numpy-based OpenEXR / Imath bindings are banned (invariant #1 — TEX is
torch-only so it stays embeddable). This implements the EXR container by hand: struct for the
header, zlib for ZIP, and torch for the pixel plumbing (`.view(dtype)` reinterpret on read,
`struct.pack` on write — verified bit-identical to torch's own half/float encoding).

SCOPE (honest, roadmap): scanline images only — single-part, NONE / ZIPS / ZIP compression,
HALF or FLOAT (UINT read-only). Layers — `layer.channel` channel names inside that one part —
are in (`read_layers` / `write_layers`, DATA-6); parts, deep and tiled remain out. Tiled,
multipart, deep, and the lossy codecs (PIZ, PXR24, B44, DWA) raise a clear error rather than
mis-decode. Little-endian hosts only (every target; EXR is LE and torch reinterpret uses
native order).

The value contract: a written buffer read back is bit-exact for FLOAT and half-rounded for
HALF (the storage dtype's own precision, nothing lost beyond it).
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import torch

from . import BufferDesc, _raw_bytes

# EXR magic + the pixel-type / compression / lineOrder enums.
_MAGIC = 20000630                      # 0x76 0x2f 0x31 0x01 as LE int32
_PT_UINT, _PT_HALF, _PT_FLOAT = 0, 1, 2
_PT_BYTES = {_PT_UINT: 4, _PT_HALF: 2, _PT_FLOAT: 4}
_PT_VIEW = {_PT_HALF: torch.float16, _PT_FLOAT: torch.float32, _PT_UINT: torch.int32}
_PT_STORAGE = {_PT_HALF: "float16", _PT_FLOAT: "float32"}   # UINT reads back as fp32 (no int storage)
_C_NONE, _C_ZIPS, _C_ZIP = 0, 2, 3
_LINES_PER_BLOCK = {_C_NONE: 1, _C_ZIPS: 1, _C_ZIP: 16}
_SUPPORTED_COMPRESSION = {"none": _C_NONE, "zips": _C_ZIPS, "zip": _C_ZIP}


class EXRError(ValueError):
    """A malformed or out-of-scope EXR (tiled / multipart / unsupported compression)."""


@dataclass(frozen=True)
class ExrImage:
    """A decoded EXR: `pixels` is [H, W, C] fp32 with channels in `channels` order (canonical
    R,G,B,A when the file's channels are a subset of those, else the file's sorted order).
    `desc` carries the file's storage dtype (float16/float32) so a host that re-saves keeps the
    precision it read. `pixel_types` is the file's pixelType per channel (0 UINT, 1 HALF,
    2 FLOAT) in `channels` order — `desc` records the FIRST channel's storage only, and a
    layered read (`read_layers`) needs the per-channel truth to refuse a UINT id plane."""
    pixels: torch.Tensor
    channels: list
    desc: BufferDesc
    pixel_types: tuple = ()


# ── ZIP codec (EXR's interleave + delta predictor, vectorized) ────────────────

def _zip_decompress(data: bytes) -> torch.Tensor:
    """Inflate one ZIP/ZIPS block → the raw scanline bytes as a uint8 tensor. Undoes EXR's two
    reversible transforms (a delta predictor, then a two-half byte de-interleave) with torch
    ops so a megabyte block is not a per-byte Python loop: the predictor is a prefix-sum
    (`d[i] = Σb[..i] − 128·i mod 256`), the de-interleave two strided assigns."""
    raw = zlib.decompress(data)
    b = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.int64)
    n = b.numel()
    # Undo predictor: running (b[i-1] + b[i] - 128) mod 256 == cumsum(b) - 128*i, mod 256.
    idx = torch.arange(n, dtype=torch.int64)
    b = ((torch.cumsum(b, 0) - 128 * idx) % 256).to(torch.uint8)
    # Undo interleave: the compressor split into t1 = out[0::2], t2 = out[1::2].
    half = (n + 1) // 2
    out = torch.empty(n, dtype=torch.uint8)
    out[0::2] = b[:half]
    out[1::2] = b[half:]
    return out


def _zip_compress(raw_u8: torch.Tensor) -> bytes:
    """The inverse of `_zip_decompress`: interleave the raw scanline bytes into two halves,
    delta-encode, then deflate. Mirrors OpenEXR's `Zip::compress` exactly (interleave FIRST,
    predictor SECOND) so any conformant reader — not just this one — decodes it."""
    n = raw_u8.numel()
    half = (n + 1) // 2
    inter = torch.empty(n, dtype=torch.uint8)
    inter[:half] = raw_u8[0::2]
    inter[half:] = raw_u8[1::2]
    # Delta-encode: d[i] = (orig[i] - orig[i-1] + 384) mod 256; d[0] = orig[0].
    i = inter.to(torch.int64)
    d = i.clone()
    d[1:] = (i[1:] - i[:-1] + 384) % 256
    # P1: hand zlib the raw uint8 bytes directly (no endianness for 1-byte samples) instead of
    # bytes(...tolist()) over millions of ints. Identical payload, ~25x less marshalling.
    return zlib.compress(_raw_bytes(d.to(torch.uint8)), 6)


# ── Header attributes ─────────────────────────────────────────────────────────

def _read_attr_header(buf: memoryview, pos: int):
    """Parse the attribute list at `pos` → (attrs dict, pos after the terminating null)."""
    attrs = {}
    while True:
        name, pos = _read_cstr(buf, pos)
        if name == "":
            return attrs, pos                     # empty name terminates the header
        atype, pos = _read_cstr(buf, pos)
        (size,) = struct.unpack_from("<i", buf, pos); pos += 4
        attrs[name] = (atype, bytes(buf[pos:pos + size])); pos += size


def _read_cstr(buf: memoryview, pos: int):
    end = pos
    while buf[end] != 0:
        end += 1
    return bytes(buf[pos:end]).decode("latin-1"), end + 1


def _parse_channels(value: bytes):
    """chlist attribute → [(name, pixelType, bytesPerSample)], in file (sorted) order."""
    chans, pos, mv = [], 0, memoryview(value)
    while pos < len(value) and value[pos] != 0:
        name, pos = _read_cstr(mv, pos)
        (ptype,) = struct.unpack_from("<i", mv, pos); pos += 4
        pos += 4                                   # pLinear (1) + reserved (3)
        xs, ys = struct.unpack_from("<ii", mv, pos); pos += 8
        if ptype not in _PT_BYTES:
            raise EXRError(f"unsupported channel pixel type {ptype} for '{name}'")
        if xs != 1 or ys != 1:                      # every plane must be full-resolution: the
            raise EXRError(f"subsampled channel '{name}' (xSampling={xs}, ySampling={ys}) "
                           f"is out of scope")      # block byte-math assumes W×H samples/channel
        chans.append((name, ptype, _PT_BYTES[ptype]))
    return chans


def _build_channels(names, ptype: int) -> bytes:
    """Serialize a chlist: names sorted, each HALF/FLOAT, sampling 1×1, pLinear 0."""
    out = bytearray()
    for name in sorted(names):
        out += name.encode("latin-1") + b"\x00"
        out += struct.pack("<i", ptype)            # pixelType
        out += struct.pack("<BBBB", 0, 0, 0, 0)    # pLinear + 3 reserved
        out += struct.pack("<ii", 1, 1)            # xSampling, ySampling
    out += b"\x00"
    return bytes(out)


# ── Public reader ─────────────────────────────────────────────────────────────

def read_exr(src) -> ExrImage:
    """Decode an EXR file (path str or raw bytes) → an `ExrImage` with [H,W,C] fp32 pixels. A
    malformed / truncated / out-of-scope file raises `EXRError`, never a raw struct/zlib error."""
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    else:
        with open(src, "rb") as f:
            data = f.read()
    try:
        return _decode_exr(memoryview(data))
    except EXRError:
        raise
    except (struct.error, IndexError, zlib.error, ValueError) as e:   # ValueError subsumes
        # ValueError covers torch.frombuffer choking on a zero-length block (a NONE/raw chunk
        # whose declared dataSize is 0) and any other malformed-byte-count reinterpret — the
        # docstring promises EXRError, never a raw error, so it must be in the net. (EXRError
        # is itself a ValueError but is re-raised above, so a genuine out-of-scope message wins.)
        raise EXRError(f"malformed or truncated EXR: {e}") from e


def _decode_exr(buf: memoryview) -> ExrImage:
    (magic,) = struct.unpack_from("<i", buf, 0)
    if magic != _MAGIC:
        raise EXRError("not an EXR file (bad magic)")
    (version,) = struct.unpack_from("<i", buf, 4)
    flags = version >> 8
    if flags & 0x2:
        raise EXRError("tiled EXR is out of scope (scanline only)")
    if flags & 0x18:                               # multipart (0x10) or deep (0x8)
        raise EXRError("multipart / deep EXR is out of scope")

    attrs, pos = _read_attr_header(buf, 8)
    for req in ("channels", "compression", "dataWindow"):
        if req not in attrs:
            raise EXRError(f"EXR missing required attribute '{req}'")

    channels = _parse_channels(attrs["channels"][1])
    compression = attrs["compression"][1][0]
    if compression not in _LINES_PER_BLOCK:
        raise EXRError(f"unsupported EXR compression {compression} "
                       f"(only none / zips / zip)")
    x_min, y_min, x_max, y_max = struct.unpack_from("<iiii", attrs["dataWindow"][1], 0)
    W, H = x_max - x_min + 1, y_max - y_min + 1

    lines_per_block = _LINES_PER_BLOCK[compression]
    n_blocks = (H + lines_per_block - 1) // lines_per_block
    offsets = struct.unpack_from("<%dQ" % n_blocks, buf, pos)

    row_stride = sum(W * bps for _, _, bps in channels)
    planes = {name: torch.empty(H, W, dtype=torch.float32) for name, _, _ in channels}

    # Each block is placed by its ABSOLUTE chunk y-coordinate (`y0 - y_min`), and scanlines within
    # a block are always increasing-y — so orientation is correct regardless of the file's lineOrder
    # (which is only the order chunks were *written*, a streaming hint). NO post-hoc flip.
    for off in offsets:
        (y0,) = struct.unpack_from("<i", buf, off)
        (dsize,) = struct.unpack_from("<i", buf, off + 4)
        payload = buf[off + 8:off + 8 + dsize]
        y0 -= y_min
        L = min(lines_per_block, H - y0)
        expected = L * row_stride
        if dsize == expected or compression == _C_NONE:
            block = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        else:
            block = _zip_decompress(bytes(payload))
        if block.numel() != expected:
            raise EXRError(f"EXR block at y={y0 + y_min} decoded to {block.numel()} bytes, "
                           f"expected {expected}")
        block = block.view(L, row_stride)
        col = 0
        for name, ptype, bps in channels:
            span = W * bps
            samples = block[:, col:col + span].reshape(-1).view(_PT_VIEW[ptype])   # reshape(-1) is already contiguous
            planes[name][y0:y0 + L] = samples.to(torch.float32).view(L, W)
            col += span

    ordered = _canonical_order([n for n, _, _ in channels])
    pixels = torch.stack([planes[n] for n in ordered], dim=-1)   # [H, W, C]
    # `transfer="linear"`: EXR is a scene-linear format by convention, so the loaded buffer's
    # colour-transfer HINT is `linear` (BufferDesc.transfer — DATA-1/DATA-2 seam). A pure hint,
    # never a transform; a host that knows the file is otherwise overrides it. This is what gives
    # the field a producer (it was declared-but-dead before); full read→BufferMeta.colorspace
    # wiring is a DATA-1↔DATA-2 host-integration follow-up.
    desc = BufferDesc(storage=_PT_STORAGE.get(channels[0][1], "float32"), transfer="linear")
    ptype = {name: pt for name, pt, _ in channels}
    return ExrImage(pixels=pixels, channels=ordered, desc=desc,
                    pixel_types=tuple(ptype[n] for n in ordered))


def _canonical_order(names):
    """Return channels in R,G,B,A order when they are exactly a subset of those (the compositor
    case), else the file's own sorted order. Keeps a read→write round-trip's channels stable."""
    rgba = [c for c in ("R", "G", "B", "A") if c in names]
    if set(rgba) == set(names):
        return rgba
    return list(names)


# ── Public writer ─────────────────────────────────────────────────────────────

_DEFAULT_NAMES = {1: ["Y"], 2: ["R", "G"], 3: ["R", "G", "B"], 4: ["R", "G", "B", "A"]}


def _as_hwc(pixels: torch.Tensor) -> torch.Tensor:
    """The writer's shape rule, shared by `write_exr` and `write_layers`: [H,W,C] passes,
    [H,W] is a mask → [H,W,1], [1,H,W,C] drops the batch, and a dim-3 [1,H,W] is a BATCHED
    mask → [H,W,1] — NOT a [H=1,W,C] image (see `write_exr`). Anything else is an `EXRError`.
    Returns fp32 on the CPU."""
    t = pixels.detach().to(torch.float32).cpu()
    if t.dim() == 4 and t.shape[0] == 1:
        t = t[0]                              # [1,H,W,C] -> [H,W,C]
    elif t.dim() == 3 and t.shape[0] == 1:    # [1,H,W] batched mask/scalar -> [H,W,1]
        t = t[0].unsqueeze(-1)                #   (NOT slice the width axis into channels)
    if t.dim() == 2:
        t = t.unsqueeze(-1)                   # [H,W] mask -> [H,W,1]
    if t.dim() != 3:
        raise EXRError(f"cannot write a tensor of shape {tuple(pixels.shape)} as EXR "
                       f"(expected [H,W,C], [H,W], or [1,H,W,C])")
    return t


def write_exr(path, pixels: torch.Tensor, *, channels=None, half: bool = False,
              compression: str = "zip") -> None:
    """Write [H,W,C] (or [H,W] / [1,H,W,C] / [1,H,W]) fp32 pixels to an EXR. `half=True` stores
    HALF (compact, HDR-capable, ~1e-3 precision); default FLOAT is exact. `channels` names the C
    channels (default R,G,B,A by count); `compression` is 'zip' (default) / 'zips' / 'none'.

    Doc 32 B1: a MASK / scalar output egresses as `[B,H,W]` (batch-leading, no channel axis) —
    the same shape `save_image` handles for the PNG sink. A dim-3 `[1,H,W]` is that batched mask
    and becomes `[H,W,1]`; it is NOT read as a `[H=1,W,C]` image (which the raw `[H,W,C]`
    interpretation below would do — silently transposing a narrow mask, or demanding W channel
    names for a wide one). Mirrors `tex_cli.save_image`'s `[1,H,W]` branch."""
    if compression not in _SUPPORTED_COMPRESSION:
        raise EXRError(f"unsupported compression {compression!r} "
                       f"(only {', '.join(_SUPPORTED_COMPRESSION)})")
    t = _as_hwc(pixels)
    H, W, C = t.shape
    names = list(channels) if channels is not None else _DEFAULT_NAMES.get(C)
    if names is None or len(names) != C:
        raise EXRError(f"need {C} channel names for a {C}-channel image (got {names})")

    ptype = _PT_HALF if half else _PT_FLOAT
    order = sorted(range(C), key=lambda i: names[i])   # storage order = channels sorted by name
    sorted_t = t[:, :, order].contiguous()             # [H, W, C] in sorted-name order
    if half:
        # Round-to-half here so an HDR value above the half range becomes `inf` (matching torch,
        # and matching the fp16 cast in the block loop below) — the whole point of a HALF sink.
        sorted_t = sorted_t.half().float()

    comp = _SUPPORTED_COMPRESSION[compression]
    lines_per_block = _LINES_PER_BLOCK[comp]
    header = _exr_header(W, H, [names[i] for i in order], ptype, comp)

    chunks, y = [], 0
    while y < H:
        L = min(lines_per_block, H - y)
        # Block byte order is row-major over rows, then channels, then pixels: [L, C, W].
        # P1: reinterpret the contiguous tensor's raw bytes (EXR is little-endian == torch's
        # native order) rather than materializing millions of Python floats via .tolist() +
        # struct.pack. `sorted_t` is already half-rounded when `half`, so the fp16 cast is exact
        # and its LE bytes are bit-identical to the old struct.pack('<e', ...); FLOAT stores the
        # fp32 bytes verbatim (== struct.pack('<f', ...)). Measured ~25x faster at 1024^2x4.
        blk = sorted_t[y:y + L].permute(0, 2, 1).reshape(-1)
        if half:
            blk = blk.to(torch.float16)
        raw = _raw_bytes(blk)                          # fp16/fp32 LE bytes, verbatim (== struct.pack)
        if comp == _C_NONE:
            payload = raw
        else:
            packed = _zip_compress(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
            payload = packed if len(packed) < len(raw) else raw   # store raw if it didn't shrink
        chunks.append((y, payload))
        y += L

    # Offset table then chunks. Each chunk on disk: [int32 y][int32 dataSize][data].
    off = len(header) + 8 * len(chunks)
    offsets = bytearray()
    body = bytearray()
    for cy, payload in chunks:
        offsets += struct.pack("<Q", off)
        rec = struct.pack("<ii", cy, len(payload)) + payload
        body += rec
        off += len(rec)

    with open(path, "wb") as f:
        f.write(header)
        f.write(bytes(offsets))
        f.write(bytes(body))


def _exr_header(W: int, H: int, names, ptype: int, comp: int) -> bytes:
    """The fixed scanline-EXR header: magic, version 2, and the eight required attributes."""
    out = bytearray()
    out += struct.pack("<ii", _MAGIC, 2)           # magic + version 2, flags 0 (scanline)

    def attr(name, atype, value):
        return (name.encode() + b"\x00" + atype.encode() + b"\x00"
                + struct.pack("<i", len(value)) + value)

    out += attr("channels", "chlist", _build_channels(names, ptype))
    out += attr("compression", "compression", bytes([comp]))
    out += attr("dataWindow", "box2i", struct.pack("<iiii", 0, 0, W - 1, H - 1))
    out += attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, W - 1, H - 1))
    out += attr("lineOrder", "lineOrder", bytes([0]))                # INCREASING_Y
    out += attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    out += attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    out += attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    out += b"\x00"                                  # end of header
    return bytes(out)


# ── Layers (DATA-6): `layer.channel` names grouped into per-plane tensors ─────
#
# An EXR "layer" is a NAMING convention inside one part's channel list (`beauty.R`,
# `specular.G`, a bare `Z`), not the multipart container. The reader above already returns
# every such channel verbatim, and a write→read round-trip through it is bitwise — so this
# section is a grouping over `ExrImage` plus its inverse flattener, never a decoder change.
# Multipart / deep / tiled stay refused exactly where `_decode_exr` refuses them.

_ROOT_PLANE = "beauty"                 # the implicit root layer's colour group, as a plane
_ROOT_COLOUR = ("R", "G", "B", "A")    # its channels — and the per-position letters on write
_LONG_COLOUR = ("red", "green", "blue", "alpha")   # the long spelling (a Nuke-written layer)


def _plane_order(chans):
    """Channel order within one plane: R,G,B,A when the set is a subset of the short spelling,
    red,green,blue,alpha when it is a subset of the long one (a Nuke-written `beauty.red`
    layer — the sorted fallback would hand it back blue, green, red, a silent BGR), else the
    file's own sorted order. Case-sensitive, and ONLY those two spellings: a mixed set
    (`R` beside `green`), a lowercase `r`, or any other convention takes the sorted fallback."""
    for spelling in (_ROOT_COLOUR, _LONG_COLOUR):
        if set(chans) <= set(spelling):
            return [c for c in spelling if c in chans]
    return list(chans)


class ExrLayers(dict):
    """What `read_layers` returns: a plain ``{plane: (pixels, BufferDesc)}`` dict, plus
    `channels` — each plane's verbatim file channel names in the plane's channel order — so
    `write_layers` can put back the exact spelling it read (a `beauty.red` layer is written
    `beauty.red`, never renamed to `beauty.R`). The attribute is metadata a host may ignore;
    the dict contract is the whole value."""
    __slots__ = ("channels",)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.channels = {}


def _split_layer(name: str):
    """`layer.channel` split on the LAST dot — the same split the compiler applies to a dotted
    `@src.plane` read, so one rule governs both ends. No dot → the implicit root layer, ``""``."""
    layer, dot, chan = name.rpartition(".")
    return (layer, chan) if dot else ("", name)


def read_layers(src) -> dict:
    """Decode an EXR (path or bytes) and group its channels into named planes:
    ``{plane: (pixels [H,W,C<=4] fp32, BufferDesc)}``. The grouping rule:

    * a channel name splits on its LAST dot into `layer` and `channel`; a layered channel
      belongs to the plane named `layer` (`beauty.diffuse.R` → plane `beauty.diffuse`);
    * the root layer's `R`/`G`/`B`/`A` group into a plane named `beauty` — only when the file
      has no explicit `beauty` layer. A file carrying BOTH is refused: the container gives
      the root group no name of its own (its layer name is the empty string), and a plane no
      program can address would vanish silently;
    * any other bare name (`Z`, `N`, `id`) is a single-channel plane of its own name, [H,W,1];
    * within a plane, channels come R,G,B,A when they are a subset of that spelling, or
      red,green,blue,alpha when they are a subset of the long one (case-sensitive; a
      Nuke-written layer), else in the file's own sorted order (the file stores `B,G,R`;
      the plane comes back `R,G,B`). No other spelling is recognised (`_plane_order`);
    * a layer with more than four channels, or with a UINT channel (an integer id plane —
      cryptomatte), raises `EXRError` naming the layer rather than yielding a wrong plane.

    Per-plane `BufferDesc.storage` is `float16` when every channel of that plane is HALF,
    else `float32`; `transfer` is `linear` as for `read_exr`. A plane name that collides with
    a swizzle (`rgb`, `z`, …) is reported as-is — refusing it is the compiler's job, at the
    point a program reads the plane; the reader's job is to report the file. Tiled /
    multipart / deep files raise exactly as `read_exr` does. Returns an `ExrLayers` — a
    plain dict (so this module stays a leaf and the host wraps it in its own plane carrier)
    whose `.channels` records each plane's verbatim file channel names for `write_layers`."""
    img = read_exr(src)
    ptypes = img.pixel_types or (_PT_FLOAT,) * len(img.channels)
    ptype = dict(zip(img.channels, ptypes))
    members = {}                                   # plane -> [(channel, column)]
    origin = {}                                    # plane -> "root" | "layer" | "bare"
    for col, name in enumerate(img.channels):
        layer, chan = _split_layer(name)
        if layer:
            plane, kind = layer, "layer"
        elif chan in _ROOT_COLOUR:
            plane, kind = _ROOT_PLANE, "root"
        else:
            plane, kind = chan, "bare"
        prev = origin.setdefault(plane, kind)
        if prev != kind:
            if plane == _ROOT_PLANE and {prev, kind} == {"root", "layer"}:
                raise EXRError(f"EXR has both root R/G/B/A channels and an explicit "
                               f"'{_ROOT_PLANE}' layer: the root colour group has no name of "
                               f"its own in the container, so the two cannot both be planes "
                               f"— rename the '{_ROOT_PLANE}' layer")
            raise EXRError(f"EXR channel groups collide on plane '{plane}' (a bare "
                           f"'{plane}' channel beside a '{plane}.*' layer)")
        members.setdefault(plane, []).append((chan, col))

    out = ExrLayers()
    for plane, rows in members.items():
        chans = [c for c, _ in rows]
        if len(chans) > 4:
            raise EXRError(f"EXR layer '{plane}' has {len(chans)} channels "
                           f"({', '.join(chans)}); a plane is at most 4 (VEC4) — split the "
                           f"layer, or read the raw channels with read_exr")
        uint = [c for c, col in rows if ptype[img.channels[col]] == _PT_UINT]
        if uint:
            raise EXRError(f"EXR layer '{plane}' has UINT channel(s) {', '.join(uint)}: an "
                           f"integer id plane (cryptomatte) is out of scope, and a float plane "
                           f"in its place would be silently wrong")
        column = dict(rows)
        cols = [column[c] for c in _plane_order(chans)]
        storage = ("float16" if all(ptype[img.channels[col]] == _PT_HALF for _, col in rows)
                   else "float32")
        out[plane] = (img.pixels[:, :, cols], BufferDesc(storage=storage, transfer="linear"))
        out.channels[plane] = [img.channels[col] for col in cols]
    return out


def write_layers(path, planes: dict, *, channels=None, root=_ROOT_PLANE, half: bool = False,
                 compression: str = "zip") -> None:
    """The inverse of `read_layers`: flatten ``{plane: pixels [H,W,C<=4]}`` into `layer.channel`
    names and write ONE scanline part through `write_exr`. A plane's value may also be the
    `(pixels, BufferDesc)` pair `read_layers` returns, so ``write_layers(p, read_layers(f))``
    is the whole round-trip (the desc is informational here; `half` picks the storage).

    Channel names, per plane, in this order of precedence: (1) `channels[plane]` — a list
    of the C full file channel names, verbatim; defaults to the `.channels` an `ExrLayers`
    carries, which is how a read file is written back with the SAME spelling it had (a
    Nuke `beauty.red` layer stays `beauty.red`, never renamed). Otherwise (2) the plane
    named `root` (default `beauty`, `read_layers`' name for the root colour group) is
    written as the bare root `R,G,B,A`; (3) any other single-channel plane whose name has
    no dot is written as that bare name (`Z` → `Z`); (4) everything else is `name.R`,
    `name.G`, `name.B`, `name.A` by position. `root=None` disables (2). Each plane takes
    the shapes `write_exr` takes; all planes must share one H×W.

    Round-trip contract: `read_layers` of what `write_layers` wrote from a `read_layers`
    result is bitwise equal per plane (half-rounded under `half=True`) AND names every
    channel as the source file did. Only a plane written WITHOUT names — a host-built
    tensor — takes the (2)-(4) spelling; the grouping rule, not the file, owns channel
    order either way."""
    if not planes:
        raise EXRError("write_layers needs at least one plane")
    if channels is None:
        channels = getattr(planes, "channels", None) or {}
    names, cols, extent = [], [], None
    for plane, pixels in planes.items():
        if not isinstance(plane, str) or not plane:
            raise EXRError(f"plane names must be non-empty strings (got {plane!r})")
        if isinstance(pixels, tuple):              # a `read_layers` (pixels, desc) pair
            pixels = pixels[0]
        t = _as_hwc(pixels)
        if extent is None:
            extent = tuple(t.shape[:2])
        elif tuple(t.shape[:2]) != extent:
            raise EXRError(f"plane '{plane}' is {tuple(t.shape[:2])} but the first plane is "
                           f"{extent}: every plane in one EXR part shares one extent")
        C = t.shape[2]
        if not 1 <= C <= 4:
            raise EXRError(f"plane '{plane}' has {C} channels; a plane is 1 to 4 (VEC4)")
        if plane in channels:
            chans = [str(c) for c in channels[plane]]
            if len(chans) != C:
                raise EXRError(f"plane '{plane}' has {C} channels but {len(chans)} channel "
                               f"name(s) were given ({', '.join(chans)})")
        elif plane == root:
            chans = list(_ROOT_COLOUR[:C])
        elif C == 1 and "." not in plane:
            chans = [plane]
        else:
            chans = [f"{plane}.{c}" for c in _ROOT_COLOUR[:C]]
        names += chans
        cols.append(t)
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise EXRError(f"planes flatten to duplicate EXR channel name(s) {', '.join(dup)} — "
                       f"rename a plane")
    write_exr(path, torch.cat(cols, dim=-1), channels=names, half=half, compression=compression)
