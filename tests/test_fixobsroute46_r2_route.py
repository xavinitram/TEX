"""FIX-OBSROUTE (v0.46 Phase C) — R2 [B4#2]: route `cook_fused_cached`'s internal calls
through `tex_engine`'s attribute.

ROUTE-45 already fixed this shape for `tex_engine_tiers` (every cross-module call there
resolves through `_tex_engine.NAME` so a host's monkeypatch on `tex_engine.NAME` is seen
regardless of which module physically defines the callee) and for
`tex_checkpoint.cook_checkpointed` (it already called `tex_engine.cook_stage_list`/
`tex_engine.boundary_lineage_key` by module lookup). `tex_chain.cook_fused_cached` was the
one holdout still calling its OWN module-local `cook_stage_list`/`boundary_lineage_key` —
so a host wrapping `tex_engine.cook_stage_list` to count cooks (the exact ROUTE-45 shape)
missed 3 of the 6 calls `cook_fused_cached` makes to them: the `_full()` fallback, the
cache-miss prefix materialize, and the suffix cook (cache-hit and cache-miss share that
last one). The cook-observer seam (OBSERVER-46) already covered this independently, but a
host's own direct wrap on the old name — the documented, older seam ROUTE-45 exists to
keep alive — did not.

Fixed by having `cook_fused_cached` resolve `tex_engine as _tex_engine` (function-local,
the same deferred-import idiom `tex_checkpoint.cook_checkpointed` already uses) and calling
`_tex_engine.cook_stage_list`/`_tex_engine.boundary_lineage_key` throughout its body."""
import torch

from TEX_Wrangle import tex_chain, tex_engine, tex_results


def _img():
    return torch.ones(1, 4, 4, 3)


def _two_stage_chain(img):
    return [
        {"code": "@OUT = vec4(@A.rgb * 1.5, 1.0);", "chain_input": None, "bindings": {"A": img}},
        {"code": "@OUT = vec4(@X.rgb + 0.1, 1.0);", "chain_input": "X", "bindings": {}},
    ]


class _Wrap:
    """A host's monkeypatch on `tex_engine.cook_stage_list` (ROUTE-45's exact shape): counts
    calls and delegates to the real function, restored on exit."""

    def __init__(self, module, name):
        self._module = module
        self._name = name
        self._real = getattr(module, name)
        self.count = 0

    def __enter__(self):
        real = self._real
        counter = self

        def _wrapped(*a, **kw):
            counter.count += 1
            return real(*a, **kw)

        setattr(self._module, self._name, _wrapped)
        return self

    def __exit__(self, *exc):
        setattr(self._module, self._name, self._real)
        return False


def test_r2_host_wrap_on_tex_engine_cook_stage_list_sees_every_internal_call(r):
    """The exact repro: wrap `tex_engine.cook_stage_list` (module attribute, ROUTE-45's own
    shape) and drive `cook_fused_cached` through its cache-MISS path (materialize the
    prefix + suffix cook: 2 internal `cook_stage_list` calls) and its cache-HIT path (a
    `boundary_lineage_key` probe + the suffix cook: 1 internal `cook_stage_list` call, 1
    `boundary_lineage_key` call). Pre-fix, the wrap counted 0 for both — the calls went to
    `tex_chain`'s own local names, invisible to a patch on `tex_engine`'s attribute."""
    print("\n--- FIX-OBSROUTE R2: a tex_engine.cook_stage_list wrap sees every internal call ---")
    stages = _two_stage_chain(_img())
    rc = tex_results.ResultCache()

    with _Wrap(tex_engine, "cook_stage_list") as w_csl, \
         _Wrap(tex_engine, "boundary_lineage_key") as w_blk:
        # Cache MISS: cook_fused_cached materializes the prefix (1 cook_stage_list call for
        # stages[:k]) then cooks the suffix (1 more) — 2 total, 0 boundary_lineage_key calls
        # (the miss path never probes; it writes).
        out1 = tex_chain.cook_fused_cached(stages, 1, rc, device="cpu", upstream=("srckey",))
        if "OUT" not in out1:
            r.fail("FIX-OBSROUTE R2 miss", f"cook_fused_cached returned no OUT: {out1.keys()}")
        elif w_csl.count != 2:
            r.fail("FIX-OBSROUTE R2 miss", f"expected 2 cook_stage_list calls visible to the "
                   f"tex_engine wrap on the cache-MISS path, saw {w_csl.count}")
        else:
            r.ok(f"cache-MISS: tex_engine.cook_stage_list wrap saw {w_csl.count} call(s)")

        w_csl.count = 0
        w_blk.count = 0

        # Cache HIT: one boundary_lineage_key probe (finds the boundary the miss path just
        # stored) + one cook_stage_list call for the suffix.
        out2 = tex_chain.cook_fused_cached(stages, 1, rc, device="cpu", upstream=("srckey",))
        if "OUT" not in out2:
            r.fail("FIX-OBSROUTE R2 hit", f"cook_fused_cached returned no OUT: {out2.keys()}")
        elif w_csl.count != 1 or w_blk.count != 1:
            r.fail("FIX-OBSROUTE R2 hit", f"expected 1 cook_stage_list + 1 boundary_lineage_key "
                   f"call visible to the tex_engine wrap on the cache-HIT path, saw "
                   f"cook_stage_list={w_csl.count} boundary_lineage_key={w_blk.count}")
        else:
            r.ok(f"cache-HIT: tex_engine wrap saw {w_csl.count} cook_stage_list + "
                 f"{w_blk.count} boundary_lineage_key call(s)")


def test_r2_full_fallback_also_routes_through_the_wrap(r):
    """`_full()` — the whole-chain fallback `cook_fused_cached` takes when the cache-gate
    refuses (e.g. no `upstream` key) — must ALSO be visible to a `tex_engine.cook_stage_list`
    wrap; it is one of the 6 calls this ask names."""
    print("\n--- FIX-OBSROUTE R2: the _full() fallback also routes through the wrap ---")
    stages = _two_stage_chain(_img())
    with _Wrap(tex_engine, "cook_stage_list") as w_csl:
        # No upstream key -> the gate refuses -> _full() -> tex_engine.cook_stage_list once.
        out = tex_chain.cook_fused_cached(stages, 1, None, device="cpu")
        if "OUT" not in out:
            r.fail("FIX-OBSROUTE R2 full", f"cook_fused_cached returned no OUT: {out.keys()}")
        elif w_csl.count != 1:
            r.fail("FIX-OBSROUTE R2 full", f"expected 1 cook_stage_list call for the _full() "
                   f"fallback visible to the wrap, saw {w_csl.count}")
        else:
            r.ok(f"_full() fallback: tex_engine.cook_stage_list wrap saw {w_csl.count} call")
