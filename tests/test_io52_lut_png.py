"""
Malformed LUT text and the PNG sink's edge cases. Every input is built in memory.

`read_cube` / `read_spi1d` promise `LutError` (a ValueError) for any bad content, and a table
that loads is a table whose every sample is a finite number in the rows the file declared.
"""
import os
import tempfile

import pytest
import torch

from TEX_Wrangle.tex_io import BufferDesc, encode_from_fp32
from TEX_Wrangle.tex_io import lut as L
from TEX_Wrangle.tex_io import png as P

_ROW = b"0.25 0.5 0.75\n"


def _cube(size=2, header=b"", rows=None, title=None):
    rows = rows if rows is not None else [_ROW] * (size ** 3)
    t = b"" if title is None else b"TITLE " + title + b"\n"
    return t + header + b"LUT_3D_SIZE %d\n" % size + b"".join(rows)


def _spi(length=2, comps=1, rows=None, extra=b""):
    rows = rows if rows is not None else [b"0.5 " * comps + b"\n"] * length
    return (b"Version 1\nFrom 0 1\nLength %d\nComponents %d\n%s{\n" % (length, comps, extra)
            + b"".join(rows) + b"}\n")


# ── decoding ─────────────────────────────────────────────────────────────────

def test_cube_with_a_utf8_bom_reads():
    lut = L.read_cube(b"\xef\xbb\xbf" + _cube(title=b'"warm"'))
    assert lut.size == 2 and lut.title == "warm"


def test_spi1d_with_a_utf8_bom_reads():
    assert L.read_spi1d(b"\xef\xbb\xbf" + _spi()).values.shape == (2,)


@pytest.mark.parametrize("read,data", [(L.read_cube, _cube(title=b"caf\xe9")),
                                       (L.read_spi1d, _spi(extra=b"# caf\xe9\n"))])
def test_undecodable_text_is_a_lut_error(read, data):
    with pytest.raises(L.LutError):
        read(data)


def test_title_keeps_a_hash_and_exotic_separators():
    assert L.read_cube(_cube(title=b'"a # b"')).title == "a # b"
    assert L.read_cube(_cube(title='"a b"'.encode())).title == "a b"
    assert L.read_cube(_cube(title=b'"abc" # note')).title == "abc"


def test_trailing_comment_and_crlf_still_parse():
    data = _cube(rows=[b"0.25 0.5 0.75  # note\r\n"] * 8).replace(b"\n", b"\r\n")
    assert L.read_cube(data).grid.shape == (2, 2, 2, 3)


# ── cube header ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("hdr", [b"DOMAIN_MIN 0 0 0 5\n", b"DOMAIN_MAX 1 1 1 9\n",
                                 b"DOMAIN_MIN 0 0\n", b"DOMAIN_MAX\n"])
def test_malformed_domain_line_is_refused_as_malformed(hdr):
    with pytest.raises(L.LutError, match="malformed"):
        L.read_cube(_cube(header=hdr))


def test_duplicate_size_is_refused():
    with pytest.raises(L.LutError, match="LUT_3D_SIZE"):
        L.read_cube(b"LUT_3D_SIZE 3\n" + _cube())


@pytest.mark.parametrize("bad", [b"nan", b"inf", b"-inf", b"1e999", b"infinity"])
def test_cube_non_finite_sample_is_refused(bad):
    rows = [bad + b" 0 0\n"] + [_ROW] * 7
    with pytest.raises(L.LutError, match="finite"):
        L.read_cube(_cube(rows=rows))


def test_cube_float32_overflow_is_refused():
    with pytest.raises(L.LutError, match="finite"):
        L.read_cube(_cube(rows=[b"1e39 0 0\n"] + [_ROW] * 7))


def test_cube_underscore_number_is_refused():
    with pytest.raises(L.LutError):
        L.read_cube(_cube(rows=[b"1_0 0 0\n"] + [_ROW] * 7))


# ── spi1d ────────────────────────────────────────────────────────────────────

def test_spi1d_ragged_rows_with_matching_totals_are_refused():
    data = _spi(length=2, comps=3, rows=[b"1 2\n", b"3 4 5 6\n"])
    with pytest.raises(L.LutError, match="row"):
        L.read_spi1d(data)


@pytest.mark.parametrize("comps,rows", [(1, [b"1 2 3\n", b"1 2 3\n"]),
                                        (3, [b"1\n", b"2\n"])])
def test_spi1d_row_width_mismatch_is_a_lut_error_not_a_runtime_error(comps, rows):
    with pytest.raises(L.LutError):
        L.read_spi1d(_spi(length=2, comps=comps, rows=rows))


@pytest.mark.parametrize("bad", [b"nan", b"inf", b"1e999"])
def test_spi1d_non_finite_sample_is_refused(bad):
    with pytest.raises(L.LutError, match="finite"):
        L.read_spi1d(_spi(length=2, rows=[bad + b"\n", b"0.5\n"]))


def test_spi1d_valid_three_component_table_reads():
    v = L.read_spi1d(_spi(length=2, comps=3, rows=[b"0 0.1 0.2\n", b"0.3 0.4 0.5\n"])).values
    assert v.shape == (2, 3) and v[1, 2].item() == pytest.approx(0.5)


# ── the PNG sink ─────────────────────────────────────────────────────────────

def test_nan_and_infinities_egress_deterministically():
    t = torch.tensor([float("nan"), float("inf"), float("-inf"), 0.5])
    assert encode_from_fp32(t, BufferDesc("uint16")).tolist() == [0, 65535, 0, 32768]
    assert encode_from_fp32(t, BufferDesc("uint8")).tolist() == [0, 255, 0, 128]


def test_write_png16_replaces_the_destination_atomically(monkeypatch):
    px = torch.arange(12, dtype=torch.int32).to(torch.uint16).reshape(2, 2, 3)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "o.png")
        with open(p, "wb") as f:
            f.write(b"previous good file")
        real, seen = os.replace, []
        monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(b), real(a, b))[1])
        P.write_png16(p, px)
        assert seen == [p] and os.listdir(td) == ["o.png"]
        with open(p, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"

        with open(p, "wb") as f:
            f.write(b"previous good file")

        def boom(a, b):
            raise PermissionError("locked")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            P.write_png16(p, px)
        with open(p, "rb") as f:
            assert f.read() == b"previous good file"
