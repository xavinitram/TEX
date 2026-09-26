"""SCALE-47b phase 5 — scale rides the lineage key AND the checkpoint boundary key.

`SCALE-47-design.md` §2 "Lineage key" / §(e): `scale` is deliberately NOT part of `program_fp`
(a `$param`-bound or literal sigma is a runtime scalar, never folded), so a result/checkpoint
cache that does not add it EXPLICITLY could serve a proxy-scale frame to a full-scale request —
silent-wrong, exactly the class invariant #2/#5 exist to prevent. Proven here at the key level
(two cooks differing only in scale mint different keys), which is the load-bearing mechanism: a
`ResultCache`/checkpoint cache has no invalidation protocol beyond "the key differs".
"""
from helpers import *
from TEX_Wrangle import tex_results_keys as RK
from TEX_Wrangle import tex_chain as _chain


def _base():
    return dict(program_fp="fp-scale47b", device="cpu", precision="fp32")


def test_scale47b_lineage_key_scale_none_default(r: SubTestResult):
    print("\n--- SCALE-47b: lineage_key() with no scale= is unaffected (invariant #7) ---")
    a = RK.lineage_key(**_base())
    b = RK.lineage_key(**_base(), scale=None)
    if a != b:
        r.fail("lineage_key scale=None default", "lineage_key() != lineage_key(scale=None)")
        return
    r.ok("lineage_key() == lineage_key(scale=None)")


def test_scale47b_lineage_key_scale_changes_identity(r: SubTestResult):
    print("\n--- SCALE-47b: lineage_key(scale=X) differs for different X, and from None ---")
    k_none = RK.lineage_key(**_base())
    k_full = RK.lineage_key(**_base(), scale=1.0)
    k_half = RK.lineage_key(**_base(), scale=0.5)
    k_half2 = RK.lineage_key(**_base(), scale=0.5)
    if len({k_none, k_full, k_half}) != 3:
        r.fail("lineage_key scale distinctness",
               f"expected 3 distinct keys, got {len({k_none, k_full, k_half})}: "
               f"none={k_none[:8]} full={k_full[:8]} half={k_half[:8]}")
        return
    if k_half != k_half2:
        r.fail("lineage_key scale determinism", "the same scale value minted two different keys")
        return
    r.ok("scale=None, scale=1.0 and scale=0.5 each mint a distinct, deterministic key")


def test_scale47b_boundary_lineage_key_threads_scale(r: SubTestResult):
    print("\n--- SCALE-47b: boundary_lineage_key(scale=...) mints a distinct checkpoint key ---")
    stages = [{"code": "@OUT = @A * 2.0;", "bindings": {"A": 1.0}},
              {"code": "@OUT = @IN * 3.0;", "bindings": {}}]
    k_none = _chain.boundary_lineage_key(stages, 1, "cpu", "fp32", upstream=("src#1",))
    k_half = _chain.boundary_lineage_key(stages, 1, "cpu", "fp32", upstream=("src#1",), scale=0.5)
    if k_none == k_half:
        r.fail("boundary_lineage_key scale",
               "a coarse (scale=0.5) checkpoint key collided with the full (scale=None) one -- "
               "a coarse checkpoint could be served to a full-scale request")
        return
    r.ok("boundary_lineage_key(scale=0.5) != boundary_lineage_key() -- never cross-served")
