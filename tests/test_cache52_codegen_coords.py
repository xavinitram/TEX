"""The codegen environment builds coordinate builtins in fp32 whatever the image precision,
as the interpreter does; only the constants that mix with image data follow it."""
import pytest
import torch

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_runtime.compiled import _build_codegen_env


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_codegen_coordinates_stay_fp32(precision):
    code = ("@OUT = vec4(u, v, ix / iw, iy / ih) * (px + py) + fi + fn;")
    bt = {"OUT": TEXType.VEC4}
    prog = parse_and_split(code, bt)
    TypeChecker(binding_types=bt).check(prog)
    frame = torch.zeros(1, 4, 6, 4)
    env, sp, _used = _build_codegen_env(prog, {"A": frame}, torch.device("cpu"), 0,
                                        precision=precision)
    assert sp == (1, 4, 6)
    for name in ("ix", "iy", "u", "v", "iw", "ih", "px", "py", "fi", "fn"):
        assert env[name].dtype == torch.float32, name
