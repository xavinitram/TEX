"""A scalar RHS to a multi-channel swizzle is not read as multi-channel.

Both tiers are compared bitwise where a program runs on both; the interpreter is the oracle."""
import pytest
import torch

from helpers import *   # noqa: F403

import test_lang_l5_codegen_masking as L5

PRAGMA = "//!tex 0.25\n"


def _wire(B=2, H=3, W=5):
    n = B * H * W
    t = torch.tensor([((i * 7 + 3) % 17) / 17.0 for i in range(n)]).reshape(B, H, W, 1)
    return torch.cat([t, (t + 0.13) % 1.0, (t + 0.41) % 1.0, torch.ones_like(t)], dim=-1)


def _both(src, bindings, masked=False):
    iout, cout, names = L5.cook_both(src, bindings, masked=masked)
    L5.assert_bitwise("tiers", iout, cout, names)
    return iout


# ── a scalar RHS to a multi-channel swizzle ─────────────────────────────────────

@pytest.mark.parametrize("hw", [(3, 5), (4, 4), (2, 6)])
def test_scalar_rhs_to_rgb_swizzle(hw):
    H, W = hw
    A = _wire(2, H, W)
    src = """
vec4 c = @A;
float m = @A.r;
c.rgb = 0.5;
vec4 d = @A;
d.rgb = m;
vec4 e = @A;
e.rg = vec2(0.25, 0.75);
@OUT = vec4(c.r + c.g + c.b, d.r, d.b, e.r + e.g * 10.0 + e.b * 100.0);
"""
    out = _both(src, {"A": A})["OUT"]
    assert torch.allclose(out[..., 0], torch.full_like(out[..., 0], 1.5))
    assert torch.equal(out[..., 1], A[..., 0]) and torch.equal(out[..., 2], A[..., 0])
    assert torch.allclose(out[..., 3], 0.25 + 7.5 + 100.0 * A[..., 2])
