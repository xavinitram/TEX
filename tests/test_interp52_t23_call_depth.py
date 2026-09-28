"""A raising call argument does not leak a call-depth level.

Both tiers are compared bitwise where a program runs on both; the interpreter is the oracle."""
import pytest
import torch

from helpers import *   # noqa: F403

import test_lang_l5_codegen_masking as L5
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, InterpreterError

PRAGMA = "//!tex 0.25\n"


def _wire(B=2, H=3, W=5):
    n = B * H * W
    t = torch.tensor([((i * 7 + 3) % 17) / 17.0 for i in range(n)]).reshape(B, H, W, 1)
    return torch.cat([t, (t + 0.13) % 1.0, (t + 0.41) % 1.0, torch.ones_like(t)], dim=-1)


def _both(src, bindings, masked=False):
    iout, cout, names = L5.cook_both(src, bindings, masked=masked)
    L5.assert_bitwise("tiers", iout, cout, names)
    return iout


# ── call depth ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("masked", [False, True])
def test_raising_argument_does_not_leak_call_depth(masked):
    # `@P` is a scalar input, so indexing it as an image raises while the call's argument
    # is being evaluated.
    src = (PRAGMA if masked else "") + """
float f(float x) { if (x > 2.0) { return x; } return 1.0; }
float y = f(@P[0, 0]);
@OUT = vec4(y, 0.0, 0.0, 1.0);
"""
    A = _wire()
    bindings = {"A": A, "P": 1.0}
    bt = {k: _infer_binding_type(v) for k, v in bindings.items()}
    from TEX_Wrangle.tex_cache import parse_and_split
    program = parse_and_split(src, bt)
    checker = TypeChecker(binding_types=bt, source=src)
    type_map = checker.check(program)
    interp = Interpreter()
    for _ in range(3):
        with pytest.raises(InterpreterError):
            interp.execute(program, dict(bindings), type_map, device="cpu",
                           output_names=["OUT"], source=src,
                           _masked_flow=True if masked else None)
        assert interp._call_depth == 0
