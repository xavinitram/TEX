"""A non-finite host index declines instead of raising.

Both tiers are compared bitwise where a program runs on both; the interpreter is the oracle."""
import torch

from helpers import *   # noqa: F403

import test_lang_l5_codegen_masking as L5
from TEX_Wrangle.tex_runtime.interpreter_values import _host_index

PRAGMA = "//!tex 0.25\n"


def _wire(B=2, H=3, W=5):
    n = B * H * W
    t = torch.tensor([((i * 7 + 3) % 17) / 17.0 for i in range(n)]).reshape(B, H, W, 1)
    return torch.cat([t, (t + 0.13) % 1.0, (t + 0.41) % 1.0, torch.ones_like(t)], dim=-1)


def _both(src, bindings, masked=False):
    iout, cout, names = L5.cook_both(src, bindings, masked=masked)
    L5.assert_bitwise("tiers", iout, cout, names)
    return iout


# ── non-finite host index ───────────────────────────────────────────────────────

def test_host_index_declines_a_non_finite_reading():
    from TEX_Wrangle.tex_runtime.stdlib import _host_scalar  # noqa: F401
    import TEX_Wrangle.tex_runtime.interpreter_values as iv
    for bad in (float("nan"), float("inf"), float("-inf")):
        t = torch.tensor(bad)
        orig = iv._host_scalar
        iv._host_scalar = lambda x, _b=bad: _b
        try:
            assert _host_index(t, 4) is None
        finally:
            iv._host_scalar = orig
