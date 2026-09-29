"""FIX-SCALE S7 (v0.47 Phase C, B2 finding 8) — `cook_fused_cached` (CACHE-6) accepts
`scale=` too, matching its `cook_checkpointed`/`materialize` (CACHE-7) siblings, instead of
raising an immediate TypeError on any caller that tries.

Before this fix: `cook_fused_cached` carried no `scale` kwarg at all, unlike
`cook_stage_list` (which it calls internally) and unlike `cook_checkpointed`/`materialize`,
which fully support it (including in `boundary_lineage_key`, whose OWN `scale=` param
`cook_fused_cached` never passed). A caller attempting `cook_fused_cached(..., scale=0.5)`
got an immediate TypeError -- loud, not silent, so no correctness risk -- but the CACHE-6
boundary-tap family could not participate in resolution-scale at all."""
import torch

from helpers import *
from TEX_Wrangle import tex_chain, tex_results

_UNSAFE_CODE = "@OUT = vec4(ix * 0.01, iy * 0.01, 0.0, 1.0);"


def _img():
    return torch.ones(1, 8, 8, 3)


def _two_stage(img, tail_code):
    return [
        {"code": "@OUT = vec4(@A.rgb * 1.5, 1.0);", "chain_input": None, "bindings": {"A": img}},
        {"code": tail_code, "chain_input": "X", "bindings": {}},
    ]


def test_s7_cook_fused_cached_accepts_scale_kwarg(r: SubTestResult):
    print("\n--- FIX-SCALE S7: cook_fused_cached(scale=0.5) no longer TypeErrors ---")
    stages = _two_stage(_img(), "@OUT = vec4(@X.rgb + 0.1, 1.0);")
    rc = tex_results.ResultCache()
    try:
        out = tex_chain.cook_fused_cached(stages, 1, rc, device="cpu", upstream=("k1",),
                                          scale=0.5)
    except TypeError as e:
        r.fail("scale kwarg accepted", f"cook_fused_cached(scale=0.5) raised TypeError: {e}")
        return
    if "OUT" not in out:
        r.fail("cook shape", f"expected OUT in the result, got {out.keys()}")
        return
    r.ok(f"cook_fused_cached(scale=0.5) returned {list(out.keys())}")


def test_s7_cache_hit_path_also_accepts_scale(r: SubTestResult):
    print("\n--- FIX-SCALE S7: the cache-HIT path (2nd call) also honours scale= ---")
    stages = _two_stage(_img(), "@OUT = vec4(@X.rgb + 0.1, 1.0);")
    rc = tex_results.ResultCache()
    out1 = tex_chain.cook_fused_cached(stages, 1, rc, device="cpu", upstream=("k1",), scale=0.5)
    hits0, misses0 = rc.hits, rc.misses
    out2 = tex_chain.cook_fused_cached(stages, 1, rc, device="cpu", upstream=("k1",), scale=0.5)
    if "OUT" not in out1 or "OUT" not in out2:
        r.fail("cook shape", f"expected OUT in both: {out1.keys()} / {out2.keys()}")
        return
    if (rc.hits, rc.misses) != (hits0 + 1, misses0) or not torch.equal(out1["OUT"], out2["OUT"]):
        r.fail("cache hit", f"the second scale=0.5 call was not served from the cache "
               f"(hits {hits0}->{rc.hits}, misses {misses0}->{rc.misses})")
        return
    # The key must carry the scale: the same stages and upstream at no scale are a MISS, not
    # the scale=0.5 entry served back.
    tex_chain.cook_fused_cached(stages, 1, rc, device="cpu", upstream=("k1",))
    if rc.misses != misses0 + 1:
        r.fail("scale in the cache key", "a scale=1.0 call was served the scale=0.5 entry "
               f"(misses {misses0}->{rc.misses})")
        return
    r.ok("the cache-MISS call cooks, the repeat scale=0.5 call HITS, and an unscaled call misses")


def test_s7_cook_fused_cached_refuses_unsafe_program_at_scale(r: SubTestResult):
    print("\n--- FIX-SCALE S7: cook_fused_cached(scale=0.5) refuses an unsafe terminal stage ---")
    stages = _two_stage(_img(), _UNSAFE_CODE)
    rc = tex_results.ResultCache()
    raised = None
    try:
        tex_chain.cook_fused_cached(stages, 1, rc, device="cpu", upstream=("k2",), scale=0.5)
    except Exception as e:
        raised = e
    if raised is None:
        r.fail("scale refusal", "cook_fused_cached(scale=0.5) did not refuse an unsafe "
               "terminal stage -- it cooked coarse silently")
        return
    refusal = getattr(raised, "tex_refusal", None)
    if refusal is None or refusal.code != "scale-unsafe":
        r.fail("scale refusal structure", f"expected tex_refusal.code=='scale-unsafe', got "
               f"{refusal!r}")
        return
    r.ok(f"cook_fused_cached(scale=0.5) refused with tex_refusal.code={refusal.code!r}")


def test_s7_no_scale_still_works_unaffected(r: SubTestResult):
    print("\n--- FIX-SCALE S7: cook_fused_cached with no scale= is unaffected (invariant #7) ---")
    stages = _two_stage(_img(), "@OUT = vec4(@X.rgb + 0.1, 1.0);")
    rc = tex_results.ResultCache()
    try:
        out = tex_chain.cook_fused_cached(stages, 1, rc, device="cpu", upstream=("k1",))
    except Exception as e:
        r.fail("no-scale call", f"unexpected raise with no scale= at all: {e}")
        return
    if "OUT" not in out:
        r.fail("cook shape", f"expected OUT, got {out.keys()}")
        return
    r.ok("cook_fused_cached() with no scale= argument at all still works")
