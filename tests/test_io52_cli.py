"""
`tex run`: awkward inputs and outputs fail like a CLI (a clean exit message) or work. Every image
is built in memory as PNG bytes and written to a temp directory.
"""
import os
import struct
import tempfile
import zlib

import pytest
import torch

from TEX_Wrangle import tex_cli

pytest.importorskip("torchvision")


def _png(width, height, color_type, samples):
    """An 8-bit PNG; `samples` is the flat per-pixel sample list (channels interleaved)."""
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    raw = b"".join(b"\0" + bytes(samples[(y * width) * ch:(y * width + width) * ch])
                   for y in range(height))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.fixture
def td():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _write(td, name, data, mode="wb"):
    p = os.path.join(td, name)
    with open(p, mode) as f:
        f.write(data)
    return p


def test_gray_plus_alpha_png_loads_as_rgb(td):
    p = _write(td, "ga.png", _png(2, 1, 4, [10, 255, 200, 128]))
    img = tex_cli.load_image(p)
    assert tuple(img.shape) == (1, 1, 2, 3)
    assert img[0, 0, 0].tolist() == pytest.approx([10 / 255] * 3)
    assert img[0, 0, 1].tolist() == pytest.approx([200 / 255] * 3)


def test_run_program_ignores_an_unrelated_secondary_output():
    img = torch.full((1, 2, 2, 3), 0.25)
    code = '@OUT = vec4(@A.rgb, 1.0); s@note = "hello";'
    out = tex_cli.run_program(code, img)
    assert out.shape[-1] in (3, 4) and out.shape[1:3] == (2, 2)


def test_string_primary_output_is_a_value_error_not_a_traceback():
    img = torch.zeros(1, 2, 2, 3)
    with pytest.raises(ValueError, match="not an image"):
        tex_cli.run_program('s@OUT = "hello";', img)


def _run_cli(td, *extra, prog=b"@OUT = vec4(@A.rgb, 1.0);", out="o.png"):
    src = _write(td, "in.png", _png(2, 2, 2, [9, 50, 200] * 4))
    tex = _write(td, "p.tex", prog)
    return tex_cli.main(["run", tex, "--in", src, "--out", os.path.join(td, out), *extra])


def test_bom_prefixed_program_runs(td):
    _run_cli(td, prog=b"\xef\xbb\xbf@OUT = vec4(@A.rgb, 1.0);")
    assert os.path.exists(os.path.join(td, "o.png"))


def test_device_auto_is_accepted(td):
    _run_cli(td, "--device", "auto")
    assert os.path.exists(os.path.join(td, "o.png"))


def test_unknown_device_is_a_usage_error(td):
    with pytest.raises(SystemExit) as ei:
        _run_cli(td, "--device", "tpu")
    assert ei.value.code == 2


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a box without CUDA")
def test_cuda_without_cuda_is_a_clean_exit(td):
    with pytest.raises(SystemExit) as ei:
        _run_cli(td, "--device", "cuda")
    assert "CUDA" in str(ei.value.code)


def test_non_png_extension_is_noted_on_stderr(td, capsys):
    _run_cli(td, out="o.jpg")
    assert "writing PNG data" in capsys.readouterr().err
    _run_cli(td, out="o.png")
    assert "writing PNG data" not in capsys.readouterr().err
