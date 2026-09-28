"""A uniform return/break leaves the live mask a Python bool.

Both tiers are compared bitwise where a program runs on both; the interpreter is the oracle."""
import torch

from helpers import *   # noqa: F403

import test_lang_l5_codegen_masking as L5
from TEX_Wrangle.tex_runtime import masked_flow as MF

PRAGMA = "//!tex 0.25\n"


def _wire(B=2, H=3, W=5):
    n = B * H * W
    t = torch.tensor([((i * 7 + 3) % 17) / 17.0 for i in range(n)]).reshape(B, H, W, 1)
    return torch.cat([t, (t + 0.13) % 1.0, (t + 0.41) % 1.0, torch.ones_like(t)], dim=-1)


def _both(src, bindings, masked=False):
    iout, cout, names = L5.cook_both(src, bindings, masked=masked)
    L5.assert_bitwise("tiers", iout, cout, names)
    return iout


# ── uniform transfers leave the mask a Python bool ──────────────────────────────

def test_uniform_break_then_write_does_not_crash():
    src = PRAGMA + """
float acc = @A.r;
for (int i = 0; i < 3; i = i + 1) {
  float k = 1.0;
  if (k > 0.5) { break; }
  acc = acc + 1.0;
}
acc = acc + 0.25;
@OUT = vec4(acc, acc, acc, 1.0);
"""
    A = _wire()
    out = _both(src, {"A": A}, masked=True)
    assert torch.allclose(out["OUT"][..., 0], A[..., 0] + 0.25)


def test_uniform_return_then_later_statements():
    src = PRAGMA + """
float f(float x) {
  float k = 1.0;
  float y = x;
  if (k > 0.5) { return 1.0; }
  y = y + 5.0;
  return y + 2.0;
}
@OUT = vec4(f(@A.r), 0.0, 0.0, 1.0);
"""
    out = _both(src, {"A": _wire()}, masked=True)
    assert torch.all(out["OUT"][..., 0] == 1.0)


def test_uniform_return_then_per_pixel_return_and_if():
    src = PRAGMA + """
float f(float x) {
  float k = 1.0;
  if (k > 0.5) { return 3.0; }
  if (x > 0.5) { return 9.0; }
  return 7.0;
}
@OUT = vec4(f(@A.r), 0.0, 0.0, 1.0);
"""
    out = _both(src, {"A": _wire()}, masked=True)
    assert torch.all(out["OUT"][..., 0] == 3.0)


def test_mask_algebra_treats_false_as_no_pixel():
    live = torch.tensor([True, False])
    assert MF.m_sub(live, False) is live
    assert MF.m_or(None, False) is None
    assert MF.m_or(live, False) is live
    assert MF.m_and(False, live) is False
    fr = MF._Frame("call")
    assert MF.apply_transfer(fr, False) is False and fr.dead is None
    assert MF.record_return(fr, torch.tensor(1.0), False) is False and fr.ret is None
    before = torch.tensor([1.0, 2.0])
    assert MF.merge_write(False, torch.tensor([9.0, 9.0]), before) is before
