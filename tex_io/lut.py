"""
Pure-Python `.cube` / `.spi1d` LUT reader (COLOR-1, v0.40).

WHY pure Python, no numpy: invariant #1 (TEX is torch-only, so it stays embeddable) — the
same rule `exr.py` follows. This is text parsing only, so there is no compiled-parser
dependency either (no PyOpenColorIO, no third-party LUT library): a `.cube`/`.spi1d` file is
whitespace-delimited ASCII, read line by line into plain Python floats and packed into a
torch tensor with `torch.tensor(...)` — never routed through NumPy in either direction.

SCOPE (honest, mirrors `exr.py`'s own "scanline only" line): a `.cube` file's 3D table only
(`LUT_1D_SIZE` tables in a `.cube` file are out of scope — use `.spi1d` for a 1D shaper) with
the DEFAULT domain (`DOMAIN_MIN 0 0 0` / `DOMAIN_MAX 1 1 1`); a non-default domain raises
rather than silently rescaling. `.spi1d` reads `Version 1` files with 1 or 3 components.

Not a stdlib function: TEX's stdlib never touches a filesystem (the same boundary `tex_io`
draws for every other format) — a host loads a file with this module and binds the resulting
plain tensor in like any other input, then a TEX program consumes it via `apply_lut3d`
(COLOR-1 ruling 5: a LUT is an ordinary bound tensor, not a new TEXType).

TENSOR LAYOUT (`read_cube`): the returned `grid` is `[N, N, N, 3]`, indexed
`grid[b_idx, g_idx, r_idx]`, matching a `.cube` file's own entry order (RED fastest-varying,
then GREEN, then BLUE) without any reshuffle — `apply_lut3d` (`tex_runtime/stdlib_color.py`)
depends on this exact axis order for its `grid_sample` volume permute.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import _read_source


class LutError(ValueError):
    """A malformed, truncated, or out-of-scope LUT file (non-default domain, a 1D table in
    a `.cube` file, a row count that doesn't match the declared size)."""


@dataclass(frozen=True)
class CubeLut:
    """A decoded `.cube` 3D LUT: `grid` is `[N, N, N, 3]` fp32 (see module docstring for the
    axis order), `size` is N, `title` the file's `TITLE` (or "")."""
    grid: torch.Tensor
    size: int
    title: str = ""


@dataclass(frozen=True)
class Spi1D:
    """A decoded `.spi1d` 1D LUT: `values` is `[N]` (1 component) or `[N, C]` fp32,
    `domain` the `(from, to)` input range the file declared."""
    values: torch.Tensor
    domain: tuple = (0.0, 1.0)




def _strip_comment(line: str) -> str:
    i = line.find("#")
    return line if i < 0 else line[:i]


def _lines(text: str) -> list:
    """Split on `\\n`, `\\r\\n` and `\\r` only: `str.splitlines()` also breaks on form feed, NEL
    and U+2028, which are ordinary characters inside a `.cube` title."""
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _finite(values: torch.Tensor, kind: str) -> None:
    if not bool(torch.isfinite(values).all()):
        raise LutError(f"non-finite value in .{kind} data (nan, inf, or beyond fp32 range)")


def _decode_guarded(fn, src, kind: str):
    """Read `src` (a path, or the file's bytes) as UTF-8 text, with or without a BOM, and call
    `fn(text)`, converting a bare `ValueError`/`IndexError` (an undecodable file included) into
    a `LutError` tagged `kind` (`"cube"`/`"spi1d"`). A `LutError` `fn` raised passes through."""
    try:
        return fn(_read_source(src).decode("utf-8-sig"))
    except LutError:
        raise
    except (ValueError, IndexError) as e:
        raise LutError(f"malformed .{kind} file: {e}") from e


def read_cube(src) -> CubeLut:
    """Decode a `.cube` file (a path, or its raw bytes/bytearray, as for `read_exr`) into a
    `CubeLut`.

    Raises `LutError` for any bad content: undecodable text, a `LUT_1D_SIZE` table, a
    non-default or malformed `DOMAIN_MIN`/`DOMAIN_MAX`, a repeated or malformed size, a body
    whose row count doesn't match `size**3`, or a non-finite sample. An unreadable path
    raises `OSError`.
    """
    return _decode_guarded(_decode_cube, src, "cube")


def _decode_cube(text: str) -> CubeLut:
    size = None
    title = ""
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    rows: list = []
    for raw_line in _lines(text):
        head_parts = raw_line.split(None, 1)
        if head_parts and head_parts[0].upper() == "TITLE":   # before comment stripping: a title may hold '#'
            rest = head_parts[1].strip() if len(head_parts) > 1 else ""
            title = rest[1:].split('"', 1)[0] if rest.startswith('"') else _strip_comment(rest).strip()
            continue
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        parts = line.split()
        head = parts[0].upper()
        if head == "LUT_1D_SIZE":
            raise LutError("a .cube LUT_1D_SIZE (1D) table is out of scope for read_cube — "
                           "use read_spi1d for a 1D shaper")
        elif head == "LUT_3D_SIZE":
            if len(parts) != 2:
                raise LutError(f"malformed LUT_3D_SIZE line: {raw_line!r}")
            if size is not None:
                raise LutError("LUT_3D_SIZE declared more than once")
            size = int(parts[1])
            if size < 2:
                raise LutError(f"LUT_3D_SIZE must be >= 2, got {size}")
        elif head in ("DOMAIN_MIN", "DOMAIN_MAX"):
            if len(parts) != 4 or "_" in line:
                raise LutError(f"malformed {head} line: {raw_line!r}")
            vals = tuple(float(x) for x in parts[1:])
            if head == "DOMAIN_MIN":
                domain_min = vals
            else:
                domain_max = vals
        else:
            # Anything that is not a recognised keyword must be an 'r g b' data row; an unknown
            # keyword (LUT_IN_VIDEO_RANGE, ...) lands here and fails the 3-column check.
            if len(parts) != 3 or "_" in line:            # float() would read '1_0' as 10.0
                raise LutError(f"expected a 'r g b' data row, got: {raw_line!r}")
            rows.append(tuple(float(x) for x in parts))
    if size is None:
        raise LutError("no LUT_3D_SIZE declared")
    if domain_min != (0.0, 0.0, 0.0) or domain_max != (1.0, 1.0, 1.0):
        raise LutError(
            f"non-default domain (DOMAIN_MIN={domain_min}, DOMAIN_MAX={domain_max}) is out of "
            f"scope — apply_lut3d assumes a [0,1] domain; rescale the LUT before loading it")
    expected = size ** 3
    if len(rows) != expected:
        raise LutError(f"LUT_3D_SIZE {size} needs {expected} data rows, found {len(rows)}")
    flat = [c for row in rows for c in row]         # R fastest-varying, then G, then B (file order)
    grid = torch.tensor(flat, dtype=torch.float32).reshape(size, size, size, 3)
    _finite(grid, "cube")
    return CubeLut(grid=grid, size=size, title=title)


def read_spi1d(src) -> Spi1D:
    """Decode a `.spi1d` file (a path, or its raw bytes/bytearray) into a `Spi1D`.

    Raises `LutError` for undecodable text, a missing `Version`/`Length`/`{ }` data block, a
    `Components` other than 1 or 3, a row count that doesn't match `Length`, a row whose width
    isn't `Components`, or a non-finite sample. An unreadable path raises `OSError`.
    """
    return _decode_guarded(_decode_spi1d, src, "spi1d")


def _decode_spi1d(text: str) -> Spi1D:
    length = None
    components = 1
    domain = (0.0, 1.0)
    rows: list = []
    in_data = False
    for raw_line in _lines(text):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if line == "{":
            in_data = True
            continue
        if line == "}":
            break
        if in_data:
            if "_" in line:                               # float() would read '1_0' as 10.0
                raise LutError(f"malformed data row: {raw_line!r}")
            vals = tuple(float(x) for x in line.split())
            rows.append(vals)
            continue
        parts = line.split()
        head = parts[0]
        if head == "Version":
            if parts[1] != "1":
                raise LutError(f"unsupported .spi1d version {parts[1]!r} (only '1' is)")
        elif head == "From":
            domain = (float(parts[1]), float(parts[2]))
        elif head == "Length":
            length = int(parts[1])
        elif head == "Components":
            components = int(parts[1])
            if components not in (1, 3):
                raise LutError(f"unsupported .spi1d Components {components} (expected 1 or 3)")
    if length is None:
        raise LutError("no Length declared")
    if not rows:
        raise LutError("no '{ ... }' data block found")
    if len(rows) != length:
        raise LutError(f"Length {length} needs {length} data rows, found {len(rows)}")
    for i, row in enumerate(rows):
        if len(row) != components:
            raise LutError(f"data row {i} has {len(row)} value(s), Components is {components}")
    flat = [c for row in rows for c in row]
    if components == 1:
        values = torch.tensor(flat, dtype=torch.float32).reshape(length)
    else:
        values = torch.tensor(flat, dtype=torch.float32).reshape(length, components)
    _finite(values, "spi1d")
    return Spi1D(values=values, domain=domain)
