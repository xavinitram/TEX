"""A fused terminal that read-modify-writes a non-chain external reads the external."""
import torch

from helpers import *  # noqa: F401,F403
from TEX_Wrangle import tex_fusion as F
from TEX_Wrangle.tex_marshalling import infer_binding_type as _ibt


def test_fused_terminal_can_read_modify_write_a_non_chain_external():
    torch.manual_seed(3)
    src = torch.rand(1, 8, 8, 4)
    mask = torch.rand(1, 8, 8, 4)
    stages = [{"code": "@OUT = @in * 1.3;", "chain_input": None, "bindings": {"in": src}},
              {"code": "@mask = @mask * 0.5; @OUT = @in * @mask;",
               "chain_input": "in", "bindings": {"mask": mask}}]
    prog, tm, refs, asg, params, used, merged = F.compile_fused(stages, _ibt)
    got = Interpreter().execute(prog, merged, tm, device="cpu", output_names=sorted(asg))["OUT"]
    assert torch.allclose(got, (src * 1.3) * (mask * 0.5), atol=1e-6)
