"""A runtime string-array index floors and clamps the same on both tiers.

Both tiers are compared bitwise where a program runs on both; the interpreter is the oracle."""
import pytest
import torch

from helpers import *   # noqa: F403

import test_lang_l5_codegen_masking as L5
import TEX_Wrangle.tex_runtime.interpreter_values as _ivals

PRAGMA = "//!tex 0.25\n"


def _wire(B=2, H=3, W=5):
    n = B * H * W
    t = torch.tensor([((i * 7 + 3) % 17) / 17.0 for i in range(n)]).reshape(B, H, W, 1)
    return torch.cat([t, (t + 0.13) % 1.0, (t + 0.41) % 1.0, torch.ones_like(t)], dim=-1)


def _both(src, bindings, masked=False):
    iout, cout, names = L5.cook_both(src, bindings, masked=masked)
    L5.assert_bitwise("tiers", iout, cout, names)
    return iout


# ── string-array runtime index ──────────────────────────────────────────────────

@pytest.mark.parametrize("v,want", [(1.5, 1), (1.99, 1), (0.0, 0), (-3.0, 0), (99.0, 2),
                                    (float("nan"), 0), (float("inf"), 2), (float("-inf"), 0)])
def test_list_index_floors_and_clamps(v, want):
    assert _ivals._list_index(v, 3) == want


def test_string_array_runtime_index_same_on_both_tiers():
    src = """
string s[] = {"a", "bb", "ccc"};
float k = 1.5;
float n = float(len(s[k]));
@OUT = vec4(n, 0.0, 0.0, 1.0);
"""
    out = _both(src, {"A": _wire()})["OUT"]
    assert torch.all(out[..., 0] == 2.0)   # floor(1.5) = 1 -> "bb"
