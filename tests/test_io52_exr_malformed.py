"""
Malformed and hostile EXR input: every repro is built in memory as bytes, nothing is committed.

The reader's contract is "a bad file raises `EXRError`, never a hang, a raw torch/struct error, or
a picture with rows nobody wrote". Each test builds the smallest file that used to break that,
and pins the loud refusal. A file the reader should still accept has its own test, so a bound
cannot be tightened into a false refusal unnoticed.
"""
import os
import struct
import tempfile
import threading
import tracemalloc
import zlib

import pytest
import torch

from TEX_Wrangle.tex_io import exr as tex_exr

_MAGIC = 20000630
_SIZE = {0: 4, 1: 2, 2: 4}


def _attr(name, atype, value):
    return name.encode() + b"\0" + atype.encode() + b"\0" + struct.pack("<i", len(value)) + value


def _chlist(chs):
    out = bytearray()
    for name, pt in chs:
        out += name.encode() + b"\0" + struct.pack("<i", pt) + b"\0\0\0\0" + struct.pack("<ii", 1, 1)
    return bytes(out) + b"\0"


def _sample(pt, v):
    return {0: struct.pack("<I", int(v)), 1: struct.pack("<e", v), 2: struct.pack("<f", v)}[pt]


def _row(chs, W, y):
    """One scanline's raw bytes, channel-major, with values that depend on (channel, x, y)."""
    out = b""
    for ci, (_, pt) in enumerate(chs):
        for x in range(W):
            out += _sample(pt, float(ci * 8 + x + y * 0.5) if pt != 0 else ci * 8 + x + y)
    return out


def _exr(W=8, H=4, comp=0, chs=(("R", 2),), ver=2, dw=None, offs=None, blocks=None, payloads=None):
    """Assemble an EXR by hand. `blocks` = [(y0, payload)] overrides the default one-chunk-per-
    block layout, `offs` overrides the offset table's values, `dw` the dataWindow box."""
    lpb = {0: 1, 2: 1, 3: 16}[comp]
    head = struct.pack("<ii", _MAGIC, ver)
    head += _attr("channels", "chlist", _chlist(chs))
    head += _attr("compression", "compression", bytes([comp]))
    head += _attr("dataWindow", "box2i", struct.pack("<iiii", *(dw or (0, 0, W - 1, H - 1))))
    head += _attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, W - 1, H - 1))
    head += _attr("lineOrder", "lineOrder", bytes([0]))
    head += _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    head += _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    head += _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    head += b"\0"
    if blocks is None:
        blocks = []
        for y in range(0, H, lpb):
            rows = b"".join(_row(chs, W, yy) for yy in range(y, min(y + lpb, H)))
            blocks.append((y, rows))
    table_at = len(head)
    n = len(blocks) if offs is None else len(offs)
    at = table_at + 8 * n
    real, body = [], b""
    for y0, payload in blocks:
        real.append(at + len(body))
        body += struct.pack("<ii", y0, len(payload)) + payload
    table = struct.pack("<%dQ" % n, *(offs if offs is not None else real))
    return head + table + body


def _read_within(data, seconds=3.0):
    """Run `read_exr(data)` on a daemon thread; a spin fails the test instead of hanging it."""
    box = {}

    def go():
        try:
            box["ok"] = tex_exr.read_exr(data)
        except BaseException as e:      # noqa: BLE001 — the test inspects the type
            box["err"] = e

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), "read_exr did not return (hung)"
    if "err" in box:
        raise box["err"]
    return box["ok"]


def _refused(data, *needles):
    with pytest.raises(tex_exr.EXRError) as ei:
        _read_within(data)
    for n in needles:
        assert n in str(ei.value), str(ei.value)


# ── the reference file the builder must agree with ───────────────────────────

def test_hand_built_file_reads_back_its_pixels():
    chs = (("G", 1), ("R", 1), ("Z", 2))
    img = _read_within(_exr(W=4, H=3, comp=0, chs=chs))
    assert img.pixels.shape == (3, 4, 3) and img.channels == ["G", "R", "Z"]
    for y in range(3):
        for x in range(4):
            assert img.pixels[y, x, 2].item() == pytest.approx(2 * 8 + x + y * 0.5)


# ── header ───────────────────────────────────────────────────────────────────

def test_negative_attribute_size_is_refused_not_spun_on():
    data = struct.pack("<ii", _MAGIC, 2) + b"a\0t\0" + struct.pack("<i", -8) + b"\0" * 64
    _refused(data, "attribute")


def test_attribute_size_past_the_end_is_refused():
    data = struct.pack("<ii", _MAGIC, 2) + b"a\0t\0" + struct.pack("<i", 1 << 30) + b"\0" * 16
    _refused(data, "attribute")


@pytest.mark.parametrize("ver", [0, 1, 3, 0x7FFFFFFF])
def test_wrong_version_number_is_refused(ver):
    _refused(_exr(ver=ver))


def test_unknown_format_flag_is_refused():
    _refused(_exr(ver=2 | (1 << 13)), "version")


def test_duplicate_channel_names_are_refused():
    _refused(_exr(W=8, H=4, chs=(("R", 2), ("R", 2))), "duplicate")


# ── dataWindow ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dw", [(5, 0, 2, 3), (0, 5, 3, 2), (0, 0, -3, 3)])
def test_degenerate_data_window_is_refused(dw):
    _refused(_exr(dw=dw, blocks=[(y, b"\0" * 8) for y in range(4)]))


def _no_big_alloc(monkeypatch, limit=1 << 22):
    real = torch.empty

    def guarded(*a, **k):
        shape = a[0] if a and not isinstance(a[0], int) else a
        n = 1
        for s in (shape if isinstance(shape, (tuple, list, torch.Size)) else (shape,)):
            n *= int(s)
        assert n <= limit, f"read_exr tried to allocate {n} elements before checking the file"
        return real(*a, **k)

    monkeypatch.setattr(tex_exr.torch, "empty", guarded)


def test_huge_window_is_refused_before_any_plane_is_allocated(monkeypatch):
    _no_big_alloc(monkeypatch)
    # 2^20 x 2^10 float samples declared by a file of a few hundred bytes
    data = _exr(W=8, H=1, dw=(0, 0, (1 << 20) - 1, 1023),
                blocks=[(y, b"\0" * 64) for y in range(1024)])
    _refused(data)


def test_window_wider_than_int_range_is_refused(monkeypatch):
    _no_big_alloc(monkeypatch)
    _refused(_exr(W=8, H=1, dw=(-(1 << 31), 0, (1 << 31) - 1, 0), blocks=[(0, b"\0" * 64)]))


def test_a_highly_compressible_large_image_still_reads():
    px = torch.zeros(256, 256, 4)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "z.exr")
        tex_exr.write_exr(p, px, compression="zip")
        assert os.path.getsize(p) < 16 * 1024          # ~1 MB of pixels in a few KB of file
        assert torch.equal(_read_within(p).pixels, px)


# ── offset table ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("off", [1 << 63, (1 << 64) - 1, 3, 10 ** 9])
def test_offset_outside_the_file_is_refused(off):
    _refused(_exr(W=8, H=4, offs=[off] * 4))


def test_duplicate_blocks_are_refused():
    row = b"\x09" * 32
    _refused(_exr(W=8, H=4, blocks=[(0, row)] * 4), "block")


def test_negative_and_out_of_range_block_rows_are_refused():
    row = b"\x09" * 32
    for bad in (-5, -(1 << 31), 4, 1 << 20):
        _refused(_exr(W=8, H=4, blocks=[(bad, row), (1, row), (2, row), (3, row)]), "block")


def test_misaligned_zip_block_is_refused():
    raw = b"\x09" * (8 * 4 * 16)
    _refused(_exr(W=8, H=32, comp=3, blocks=[(1, raw), (16, raw)]), "block")


def test_missing_blocks_are_refused():
    row = b"\x09" * 32
    _refused(_exr(W=8, H=4, blocks=[(0, row), (1, row), (2, row), (0, row)]), "block")


def test_a_valid_file_with_blocks_written_out_of_order_still_reads():
    chs = (("R", 2),)
    rows = {y: _row(chs, 8, y) for y in range(4)}
    img = _read_within(_exr(W=8, H=4, blocks=[(3, rows[3]), (2, rows[2]), (1, rows[1]), (0, rows[0])]))
    assert img.pixels[3, 2, 0].item() == pytest.approx(2 + 3 * 0.5)
    assert img.pixels[0, 7, 0].item() == pytest.approx(7)


def test_a_window_with_a_nonzero_origin_still_reads():
    chs = (("R", 2),)
    img = _read_within(_exr(W=8, H=4, dw=(-4, 10, 3, 13),
                            blocks=[(10 + y, _row(chs, 8, y)) for y in range(4)]))
    assert img.pixels.shape == (4, 8, 1)
    assert img.pixels[2, 5, 0].item() == pytest.approx(5 + 1.0)


# ── decoded content ──────────────────────────────────────────────────────────

def test_mixed_half_and_float_channels_on_single_line_blocks():
    """FLOAT after an odd number of HALF channels, W odd: the float channel starts at an
    unaligned byte offset inside the block."""
    chs = (("B", 1), ("G", 1), ("R", 1), ("Z", 2))
    for comp in (0, 2):
        raw = [_row(chs, 5, y) for y in range(3)]
        if comp == 2:
            blocks = [(y, _zip_block(r)) for y, r in enumerate(raw)]
        else:
            blocks = list(enumerate(raw))
        img = _read_within(_exr(W=5, H=3, comp=comp, chs=chs, blocks=blocks))
        assert img.pixels.shape == (3, 5, 4)
        assert img.pixels[2, 4, 3].item() == pytest.approx(3 * 8 + 4 + 1.0)
        assert img.pixels[2, 4, 0].item() == pytest.approx(4 + 1.0)


def _zip_block(raw: bytes) -> bytes:
    packed = tex_exr._zip_compress(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
    return packed if len(packed) < len(raw) else raw


def test_uint_channel_above_int32_range_reads_as_its_unsigned_value():
    chs = (("id", 0),)
    raw = struct.pack("<II", 4000000000, 7)
    img = _read_within(_exr(W=2, H=1, chs=chs, blocks=[(0, raw)]))
    assert img.pixels[0, 0, 0].item() == float(4000000000)
    assert img.pixels[0, 1, 0].item() == 7.0


# ── ZIP inflate is bounded by what the block may hold ────────────────────────

def test_zip_bomb_is_refused_without_inflating_it():
    bomb = zlib.compress(bytes(16 * 1024 * 1024), 9)         # ~16 KB on disk
    data = _exr(W=8, H=1, comp=2, chs=(("R", 2),), blocks=[(0, bomb)])
    tracemalloc.start()
    try:
        with pytest.raises(tex_exr.EXRError):
            tex_exr.read_exr(data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024, f"inflated {peak} bytes of a block that may hold 32"


def test_zip_block_of_the_wrong_length_is_refused():
    short = zlib.compress(b"\0" * 16)
    _refused(_exr(W=8, H=1, comp=2, chs=(("R", 2),), blocks=[(0, short)]), "expected 32")


# ── the writer replaces the destination atomically ───────────────────────────

def test_write_exr_replaces_the_destination_atomically(monkeypatch):
    px = torch.rand(4, 4, 3)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "out.exr")
        with open(p, "wb") as f:
            f.write(b"previous good file")
        real = os.replace
        seen = []
        monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(b), real(a, b))[1])
        tex_exr.write_exr(p, px)
        assert seen == [p]
        assert torch.equal(tex_exr.read_exr(p).pixels, px)
        assert os.listdir(td) == ["out.exr"]

        with open(p, "wb") as f:
            f.write(b"previous good file")

        def boom(a, b):
            raise PermissionError("locked")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            tex_exr.write_exr(p, px)
        with open(p, "rb") as f:
            assert f.read() == b"previous good file"
        assert os.listdir(td) == ["out.exr"]
