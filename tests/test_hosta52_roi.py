"""tex_roi: a short `valid` list is legal input to chain_windows, and the memo hit paths
tolerate a concurrent eviction."""
from collections import OrderedDict

from TEX_Wrangle import tex_roi


def test_declined_stage_past_the_end_of_a_short_valid_list_does_not_raise():
    plan = tex_roi.chain_windows([0, 2, 2, 2], (1, 1, 4, 4, 32, 32), dirty_from=0,
                                 valid=[None, None], declined={3})
    assert plan is None or len(plan) == 4


def test_declined_stage_with_a_partial_upstream_still_refuses():
    partial = (0, 0, 8, 8)
    assert tex_roi.chain_windows([0, 2, 2], (1, 1, 4, 4, 32, 32), dirty_from=0,
                                 valid=[None, partial, None], declined={2}) is None


class _EvictedOnTouch(OrderedDict):
    def move_to_end(self, key, last=True):
        raise KeyError(key)


def test_scale_verdict_and_region_memos_tolerate_a_concurrent_eviction(monkeypatch):
    src = "@OUT = gauss_blur(@A, 2.0);"
    monkeypatch.setattr(tex_roi, "_scale_verdict_memo", _EvictedOnTouch())
    monkeypatch.setattr(tex_roi, "_walk_memo", _EvictedOnTouch())
    first = tex_roi.scale_verdict(src, {})
    assert tex_roi.scale_verdict(src, {}) == first
    plan = tex_roi.roi_plan(src, {}, None)
    assert tex_roi.roi_plan(src, {}, None).halo == plan.halo
