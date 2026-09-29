"""v0.52 sweep: the idle harvest re-cooks nothing it has, and skips cuts nobody serves."""
import torch

from TEX_Wrangle import tex_checkpoint as CK
from TEX_Wrangle import tex_engine, tex_results


def _chain(n, tap_at=()):
    out = []
    for i in range(n):
        st = {"code": f"@OUT = vec4(@IN.rgb * {1.0 + i * 0.05:.2f}, 1.0);",
              "chain_input": (None if i == 0 else "IN"),
              "bindings": ({"IN": torch.rand(1, 16, 16, 3)} if i == 0 else {})}
        if i in tap_at:
            st["tap"] = True
        out.append(st)
    return out


class _CountCooks:
    def __init__(self):
        self.n = 0

    def __enter__(self):
        self._real = tex_engine.cook_stage_list
        counter = self

        def wrapped(*a, **k):
            counter.n += 1
            return self._real(*a, **k)

        tex_engine.cook_stage_list = wrapped
        return self

    def __exit__(self, *exc):
        tex_engine.cook_stage_list = self._real


def test_harvest_skips_boundaries_already_resident():
    cache = tex_results.ResultCache()
    stages = _chain(3)
    with _CountCooks() as c:
        first = CK.materialize(stages, cache, cuts=[1, 2], upstream=("u",))
        assert first == [1, 2] and c.n == 1
        again = CK.materialize(stages, cache, cuts=[1, 2], upstream=("u",))
        assert again == [1, 2] and c.n == 1          # no second cook, no re-put


def test_harvest_skips_a_cut_the_serve_loop_would_never_use():
    cache = tex_results.ResultCache()
    stages = _chain(3, tap_at={0})          # a user tap below the cut: cut 2 is unservable
    with _CountCooks() as c:
        assert CK.materialize(stages, cache, cuts=[2], upstream=("u",)) == []
        assert c.n == 0
    assert not cache._ram
