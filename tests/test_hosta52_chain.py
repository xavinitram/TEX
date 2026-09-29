"""Chain-cook cache gates: the stage DAG cook only caches a boundary its upstream keys cover."""
import pytest
import torch

from TEX_Wrangle import tex_chain, tex_results

_W = _H = 16


def _stages(A, B):
    return [
        {"code": "@OUT = @A;", "bindings": {"A": A}},
        {"code": "@OUT = gauss_blur(@P, 20.0);", "bindings": {}, "chain_inputs": {"P": [0, "OUT"]}},
        {"code": "@OUT = @B + 0.1;", "bindings": {"B": B}},
        {"code": "@OUT = (@bg + @fg) * 0.5;", "bindings": {},
         "chain_inputs": {"bg": [1, "OUT"], "fg": [2, "OUT"]}},
    ]


def test_dag_cook_without_full_upstream_does_not_cache_a_boundary():
    torch.manual_seed(7)
    A1, A2, B = (torch.rand(1, _H, _W, 3) for _ in range(3))
    rc = tex_results.ResultCache()
    roi = (1, 1, 4, 4, _W, _H)
    tex_chain.cook_stage_dag(_stages(A1, B), roi=roi, roi_exec=True, dirty_from=0,
                             result_cache=rc, upstream=())
    assert rc.stats()["ram_entries"] == 0
    # a second, different source of the same shape must not be answered from the first
    with pytest.raises(ValueError, match="never cooked"):
        tex_chain.cook_stage_dag(_stages(A2, B), roi=roi, roi_exec=True, dirty_from=1,
                                 valid=[None, (0, 0, _W, _H), None, None],
                                 result_cache=rc, upstream=())


def test_dag_cook_with_full_upstream_still_caches():
    torch.manual_seed(8)
    A, B = torch.rand(1, _H, _W, 3), torch.rand(1, _H, _W, 3)
    rc = tex_results.ResultCache()
    tex_chain.cook_stage_dag(_stages(A, B), roi=(1, 1, 4, 4, _W, _H), roi_exec=True,
                             dirty_from=0, result_cache=rc, upstream=("a", "b"))
    assert rc.stats()["ram_entries"] >= 1
