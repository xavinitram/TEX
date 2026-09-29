"""tex_tiling: the pointwise strip planner sizes its peak off the real frame, not off a
width-1 companion that happens to be bound first."""
import torch

from TEX_Wrangle import tex_tiling
from TEX_Wrangle.tex_cache import parse_and_split


def _plan(bindings, free):
    prog = parse_and_split("@OUT = @A * 2.0;", {})
    return tex_tiling._tile_plan(prog, bindings, "cuda", 0, 4, None, free_hint=free)


def test_width_one_companion_bound_first_does_not_hide_the_pressure():
    H = W = 1024
    img = torch.zeros(1, H, W, 3)
    col = torch.zeros(1, H, 1, 3)
    free = 32 * 1024 * 1024               # the 16 MB frame is over a quarter of it
    assert _plan({"A": img}, free) == 2
    assert _plan({"col": col, "A": img}, free) == 2
