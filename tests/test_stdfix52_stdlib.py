"""Stdlib correctness fixes: mask blurs, mip identity detection, NaN coordinates, hash_int,
fp16 gate walks, colour maths, non-finite text/INT conversion, sdf_polygon, fp16 bilateral."""
import math

import pytest
import torch

from helpers import *
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib as S
from TEX_Wrangle.tex_runtime.precision_policy import resolve_auto_precision

NAN = float("nan")


def _img(B=1, H=8, W=8, C=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, H, W, C, generator=g)


# -- gauss_blur / bilateral_filter on a [B,H,W] mask ------------------------------------

def _mask(H=16, W=16):
    m = torch.zeros(1, H, W)
    m[0, H // 2, W // 2] = 1.0
    return m


def test_gauss_blur_mask_is_blurred_like_a_one_channel_image():
    m = _mask()
    out = S.fn_gauss_blur(m, 2.0)
    ref = S.fn_gauss_blur(m.unsqueeze(-1), 2.0).squeeze(-1)
    assert out.shape == m.shape
    assert not torch.equal(out, m)
    assert torch.equal(out, ref)


def test_bilateral_filter_mask_is_filtered_like_a_one_channel_image():
    g = torch.Generator().manual_seed(1)
    m = torch.rand(1, 16, 16, generator=g)
    out = S.fn_bilateral_filter(m, 2.0, 0.3)
    ref = S.fn_bilateral_filter(m.unsqueeze(-1), 2.0, 0.3).squeeze(-1)
    assert out.shape == m.shape
    assert not torch.equal(out, m)
    assert torch.equal(out, ref)


def test_mask_blur_program_matches_across_tiers():
    m = _mask()
    for call in ("gauss_blur(@M, 2.0)", "bilateral_filter(@M, 2.0, 0.3)"):
        interp, cg = run_both(f"@OUT = {call};", {"M": m}, H=16, W=16)
        o = interp["OUT"]
        assert not torch.equal(o.reshape(m.shape), m), call
        if cg is not None:
            assert (cg["OUT"].float() - o.float()).abs().max() < 1e-5, call


