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


def _to_text(src) -> str:
    """`src` -> decoded text, via the shared `tex_io._read_source` (COLOR-1 simplify) that
    `exr.py`'s `read_exr` also uses: a path is read in BINARY mode, like every other
    `tex_io` format, then decoded as UTF-8 text here. Not text-mode-open + its universal-
    newline translation — this module's parser doesn't need it (`str.splitlines()` already
    treats `\\r\\n` as one line break, so a CRLF file parses identically either way)."""
    return _read_source(src).decode("utf-8")


def _decode_guarded(fn, text: str, kind: str):
    """Call `fn(text)`, converting a bare `ValueError`/`IndexError` into a `LutError`
    tagged `kind` (`"cube"`/`"spi1d"`) — `read_cube`/`read_spi1d`'s identical try/except,
    factored once. A `LutError` `fn` already raised passes through unchanged (it already
    carries the right out-of-scope message)."""
    try:
        return fn(text)
    except LutError:
        raise
    except (ValueError, IndexError) as e:
        raise LutError(f"malformed .{kind} file: {e}") from e


def read_cube(src) -> CubeLut:
    """Decode a `.cube` file (path str, or its raw bytes/text already read — mirroring
    `read_exr`'s `src` contract) into a `CubeLut`.

    Raises `LutError` — never a raw `ValueError`/`IndexError` — for anything out of scope:
    a `LUT_1D_SIZE` table, a non-default `DOMAIN_MIN`/`DOMAIN_MAX`, a malformed size, or a
    body whose row count doesn't match `size**3`.
    """
    return _decode_guarded(_decode_cube, _to_text(src), "cube")


def _decode_cube(text: str) -> CubeLut:
    size = None
    title = ""
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    rows: list = []
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        parts = line.split()
        head = parts[0].upper()
        if head == "TITLE":
            title = line.split(None, 1)[1].strip().strip('"') if len(parts) > 1 else ""
        elif head == "LUT_1D_SIZE":
            raise LutError("a .cube LUT_1D_SIZE (1D) table is out of scope for read_cube — "
                           "use read_spi1d for a 1D shaper")
        elif head == "LUT_3D_SIZE":
            if len(parts) != 2:
                raise LutError(f"malformed LUT_3D_SIZE line: {raw_line!r}")
            size = int(parts[1])
            if size < 2:
                raise LutError(f"LUT_3D_SIZE must be >= 2, got {size}")
        elif head == "DOMAIN_MIN":
            domain_min = tuple(float(x) for x in parts[1:4])
        elif head == "DOMAIN_MAX":
            domain_max = tuple(float(x) for x in parts[1:4])
        else:
            # A data row: exactly 3 floats. Any other unrecognised keyword (e.g. TITLE with no
            # quotes handled above) would already have been consumed; a row that ISN'T 3 floats
            # here is genuinely malformed input.
            if len(parts) != 3:
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
    return CubeLut(grid=grid, size=size, title=title)


def read_spi1d(src) -> Spi1D:
    """Decode a `.spi1d` file (path str, or its raw bytes/text already read) into a `Spi1D`.

    Raises `LutError` for a missing `Version`/`Length`/`{ }` data block, a `Components` other
    than 1 or 3, or a row count that doesn't match the declared `Length`.
    """
    return _decode_guarded(_decode_spi1d, _to_text(src), "spi1d")


def _decode_spi1d(text: str) -> Spi1D:
    length = None
    components = 1
    domain = (0.0, 1.0)
    rows: list = []
    in_data = False
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if line == "{":
            in_data = True
            continue
        if line == "}":
            break
        if in_data:
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
    flat = [c for row in rows for c in row]
    if components == 1:
        values = torch.tensor(flat, dtype=torch.float32).reshape(length)
    else:
        values = torch.tensor(flat, dtype=torch.float32).reshape(length, components)
    return Spi1D(values=values, domain=domain)
