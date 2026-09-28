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


# -- sample_mip identity-UV detection ---------------------------------------------------

def _uv(H=8, W=8, B=1):
    u = (torch.arange(W).float() / (W - 1)).view(1, 1, W).expand(B, H, W).contiguous()
    v = (torch.arange(H).float() / (H - 1)).view(1, H, 1).expand(B, H, W).contiguous()
    return u, v


def test_sample_mip_honours_a_warp_that_fixes_the_corners():
    img = _img(H=16, W=16)
    u, v = _uv(16, 16)
    warp = u + 0.08 * torch.sin(v * math.pi)
    assert (S.fn_sample_mip(img, warp, v, 1.0) - S.fn_sample_mip(img, u, v, 1.0)).abs().max() > 1e-3
    assert (S.fn_sample_mip_gauss(img, warp, v, 1.0)
            - S.fn_sample_mip_gauss(img, u, v, 1.0)).abs().max() > 1e-3


def test_sample_mip_identity_uv_still_takes_the_plain_level():
    img = _img(H=16, W=16)
    u, v = _uv(16, 16)
    out = S.fn_sample_mip(img, u, v, 1.0)
    assert out.shape == img.shape


def test_sample_mip_mixed_rank_uv_broadcasts():
    img = _img(H=8, W=8)
    u, v = _uv(8, 8)
    a = S.fn_sample_mip(img, u, torch.tensor(0.5), 0.0)
    b = S.fn_sample_mip(img, u, torch.full_like(v, 0.5), 0.0)
    assert a.shape == b.shape
    assert (a - b).abs().max() < 1e-6
    c = S.fn_sample_mip(img, torch.tensor(0.25), v, 0.0)
    d = S.fn_sample_mip(img, torch.full_like(u, 0.25), v, 0.0)
    assert (c - d).abs().max() < 1e-6


