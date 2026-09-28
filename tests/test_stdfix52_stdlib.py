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


# -- NaN coordinates never become an out-of-range index ---------------------------------

def test_fetch_nan_coordinate_lands_on_a_valid_pixel():
    img = _img(H=4, W=4)
    px = torch.full((1, 4, 4), NAN)
    py = torch.zeros(1, 4, 4)
    out = S.fn_fetch(img, px, py)
    assert torch.isfinite(out).all()
    assert torch.equal(out[0, 0, 0], img[0, 0, 0])
    out2 = S.fn_fetch(img, torch.tensor(NAN), torch.tensor(NAN))
    assert torch.isfinite(out2).all()


def test_fetch_finite_coordinates_are_unchanged():
    img = _img(H=4, W=4)
    px = torch.tensor([-2.0, 0.4, 1.9, 9.0]).view(1, 1, 4).expand(1, 4, 4)
    py = torch.zeros(1, 4, 4)
    out = S.fn_fetch(img, px, py)
    assert torch.equal(out[0, 0, 0], img[0, 0, 0])
    assert torch.equal(out[0, 0, 1], img[0, 0, 0])
    assert torch.equal(out[0, 0, 2], img[0, 0, 1])
    assert torch.equal(out[0, 0, 3], img[0, 0, 3])


def test_sample_lanczos_nan_coordinate_is_safe():
    img = _img(H=4, W=4)
    u = torch.full((1, 4, 4), NAN)
    v = torch.full((1, 4, 4), 0.5)
    out = S.fn_sample_lanczos(img, u, v)
    assert out.shape == img.shape


def test_sample_frame_nan_coordinate_is_safe():
    img = _img(B=2, H=4, W=4)
    u = torch.full((2, 4, 4), NAN)
    v = torch.full((2, 4, 4), 0.5)
    out = S.fn_sample_frame(img, 0.0, u, v)
    assert out.shape == (2, 4, 4, 3)


@pytest.mark.parametrize("code", [
    "@OUT = fetch(@A, ix + @F.r, iy);",
    "@OUT = @A[ix + @F.r, iy];",
    "@OUT = sample(@A, u + @F.r * px, v);",
])
def test_nan_coordinate_program_runs_on_both_tiers(code):
    A = _img(H=4, W=4, C=4)
    F = torch.zeros(1, 4, 4, 4)
    F[0, 1, 1, 0] = NAN
    interp, cg = run_both(code, {"A": A, "F": F})
    if cg is not None:
        a, b = interp["OUT"], cg["OUT"]
        assert torch.equal(torch.isnan(a), torch.isnan(b))
        assert (torch.nan_to_num(a) - torch.nan_to_num(b)).abs().max() < 1e-5


# -- hash_int ---------------------------------------------------------------------------

def test_hash_int_without_max_varies_per_string():
    vals = {S.fn_hash_int(s).item() for s in ("a", "b", "c", "frame_001", "frame_002")}
    assert len(vals) >= 4
    for v in vals:
        assert 0 <= v < 2 ** 24 and v == int(v)


@pytest.mark.parametrize("mx", [0, -5, 2 ** 25])
def test_hash_int_unusable_max_still_varies(mx):
    vals = {S.fn_hash_int(s, mx).item() for s in ("a", "b", "c", "d")}
    assert len(vals) >= 3
    assert all(0 <= v < 2 ** 24 for v in vals)


def test_hash_int_with_max_is_a_modulo():
    assert 0 <= S.fn_hash_int("abc", 100).item() < 100


# -- fp16 gate --------------------------------------------------------------------------

def _resolve(code):
    return resolve_auto_precision(parse_and_split(code, {}), 2048 * 2048, "cuda")[0]


def test_gate_long_taint_chain_is_followed_to_the_end():
    lines = ["float v0 = @A.r;"] + [f"float v{i} = v{i-1} * 1.0;" for i in range(1, 13)]
    code = "\n".join(lines) + "\nfloat o = v12 > 0.5;\n@OUT = vec4(o, o, o, 1.0);"
    assert _resolve(code) == "fp32"


def test_gate_swizzle_store_of_image_taints_the_local():
    code = ("vec3 c = vec3(0.0);\nc.r = @A.r;\nfloat o = c.r > 0.5;\n"
            "@OUT = vec4(o, o, o, 1.0);")
    assert _resolve(code) == "fp32"


def test_gate_swizzle_store_carries_gain():
    code = "vec3 c = vec3(0.0);\nc.r = @A.r * 3.0;\n@OUT = vec4(sin(c * 3.0), 1.0);"
    assert _resolve(code) == "fp32"


def test_gate_swizzle_accumulation_in_a_loop_is_declined():
    code = ("vec3 acc = vec3(0.0);\nfor (int i = 0; i < 4; i++) { acc.r += @A.r; }\n"
            "@OUT = vec4(acc, 1.0);")
    assert _resolve(code) == "fp32"


def test_gate_loop_carried_constant_is_not_folded():
    code = ("float s = 1.0;\nfor (int i = 0; i < 10; i++) { s = s * 2.0; }\n"
            "@OUT = vec4(sin(@A.rgb * s), 1.0);")
    assert _resolve(code) == "fp32"


def test_gate_loop_counter_magnitude_is_unbounded():
    code = ("vec3 c = vec3(0.0);\nfor (int i = 0; i < 40; i++) { c = sin(@A.rgb * i); }\n"
            "@OUT = vec4(c, 1.0);")
    assert _resolve(code) == "fp32"


def test_gate_still_accepts_a_smooth_pointwise_program():
    assert _resolve("@OUT = vec4(@A.rgb * 1.1, 1.0);") == "fp16"


# -- colour -----------------------------------------------------------------------------

def _px(*c):
    return torch.tensor(c, dtype=torch.float32).view(1, 1, 1, len(c))


@pytest.mark.parametrize("h", [-0.1, -0.2, -0.5, -0.9, -1.3])
def test_hsv2rgb_negative_hue_wraps(h):
    a = S.fn_hsv2rgb(_px(h, 1.0, 1.0))
    b = S.fn_hsv2rgb(_px(h % 1.0, 1.0, 1.0))
    assert (a - b).abs().max() < 1e-5


def test_rgb2hsv_gray_is_finite_in_fp16():
    gray = torch.full((1, 2, 2, 3), 0.5, dtype=torch.float16)
    out = S.fn_rgb2hsv(gray)
    assert torch.isfinite(out).all()
    assert (out[..., 0] == 0).all() and (out[..., 1] < 1e-3).all()
    black = S.fn_rgb2hsv(torch.zeros(1, 1, 1, 3, dtype=torch.float16))
    assert torch.isfinite(black).all()


def test_rgb2hsv_fp32_unchanged():
    out = S.fn_rgb2hsv(_px(1.0, 0.0, 0.0))
    assert abs(out[0, 0, 0, 0].item()) < 1e-6 and abs(out[0, 0, 0, 1].item() - 1.0) < 1e-6


def test_dodge_burn_hdr_ranges():
    assert S.fn_color_dodge(_px(0.5, 0.5, 0.5), _px(2.0, 2.0, 2.0)).min() == 1.0
    assert S.fn_color_dodge(_px(0.5, 0.5, 0.5), _px(1.0, 1.0, 1.0)).min() == 1.0
    assert S.fn_color_dodge(_px(0.0, 0.0, 0.0), _px(2.0, 2.0, 2.0)).max() == 0.0
    assert S.fn_color_burn(_px(0.5, 0.5, 0.5), _px(-1.0, -1.0, -1.0)).max() == 0.0
    assert S.fn_color_burn(_px(1.0, 1.0, 1.0), _px(-1.0, -1.0, -1.0)).min() == 1.0
    for b in (3.0, -3.0):
        v = S.fn_vivid_light(_px(0.5, 0.5, 0.5), _px(b, b, b))
        assert v.max() <= 1.0 and v.min() >= 0.0


def test_dodge_burn_in_range_values_unchanged():
    assert abs(S.fn_color_dodge(_px(0.25, 0.25, 0.25), _px(0.5, 0.5, 0.5))[0, 0, 0, 0].item() - 0.5) < 1e-6
    assert abs(S.fn_color_burn(_px(0.75, 0.75, 0.75), _px(0.5, 0.5, 0.5))[0, 0, 0, 0].item() - 0.5) < 1e-6


