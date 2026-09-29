"""compile_fused: the merged bindings are one prefixed dict, and a stage with no bindings key
value (None) splices like an empty one."""
import torch

from helpers import *  # noqa: F401,F403
from TEX_Wrangle import tex_fusion as F
from TEX_Wrangle.tex_marshalling import infer_binding_type as _ibt


def test_stage_with_none_bindings_splices_and_merged_bindings_are_prefixed():
    src = torch.rand(1, 8, 8, 4)
    stages = [{"code": "@OUT = @in * 2.0;", "chain_input": None, "bindings": {"in": src}},
              {"code": "@OUT = @in + 0.25;", "chain_input": "in", "bindings": None}]
    prog, tm, refs, asg, params, used, merged = F.compile_fused(stages, _ibt)
    assert list(merged.values()) == [src] and len(merged) == 1
    assert list(merged)[0].endswith("in") and "_s0_" in list(merged)[0]
    got = Interpreter().execute(prog, merged, tm, device="cpu", output_names=sorted(asg))["OUT"]
    assert torch.allclose(got, src * 2.0 + 0.25, atol=1e-6)
    # the memoized second call answers the same merged bindings
    again = F.compile_fused(stages, _ibt)[-1]
    assert list(again) == list(merged)
