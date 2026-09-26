"""SCALE-47b phase 8 — the proxy-size assertion helper: offered, never enforced.

`SCALE-47-design.md` §(b) / the author's decision #4: the host owns proxy selection (R5), so
TEX offers a cheap sanity check comparing a cook's bound image shape against the `scale` it
claims -- an ARITY check, nothing more -- but never runs it itself. A host that wants the
belt-and-braces call opts in explicitly; a cook with a mismatched proxy still cooks (whatever
it cooks), because enforcing this engine-side would be a second, un-asked-for gate.
"""
from helpers import *
from TEX_Wrangle import tex_api
from TEX_Wrangle import tex_engine


def test_scale47b_proxy_scale_matches(r: SubTestResult):
    print("\n--- SCALE-47b: a proxy shaped exactly for its scale reports no mismatch ---")
    A = make_img(1, 32, 32, 4)          # a 0.5x proxy of a 64x64 full frame
    msg = tex_api.check_proxy_scale({"A": A}, full_hw=(64, 64), scale=0.5)
    if msg is not None:
        r.fail("matching proxy", f"expected None (no mismatch), got {msg!r}")
        return
    r.ok("a 32x32 binding against full_hw=(64,64), scale=0.5 reports no mismatch")


def test_scale47b_proxy_scale_mismatch_reported(r: SubTestResult):
    print("\n--- SCALE-47b: a mismatched proxy names itself in the returned message ---")
    A = make_img(1, 30, 30, 4)          # NOT 0.5x of 64x64 (expected 32x32)
    msg = tex_api.check_proxy_scale({"A": A}, full_hw=(64, 64), scale=0.5)
    if msg is None or "A" not in msg:
        r.fail("mismatch report", f"expected a message naming 'A', got {msg!r}")
        return
    r.ok(f"a 30x30 binding against expected 32x32 is reported: {msg!r}")


def test_scale47b_proxy_assert_never_raises_on_odd_input(r: SubTestResult):
    print("\n--- SCALE-47b: non-tensor / scalar bindings are ignored, never raise ---")
    try:
        msg = tex_api.check_proxy_scale({"A": 1.0, "B": "plate"}, full_hw=(64, 64), scale=0.5)
    except Exception as e:
        r.fail("defensive", f"raised on scalar/string bindings: {type(e).__name__}: {e}")
        return
    r.ok(f"scalar/string bindings are skipped, not mistaken for a spatial proxy: {msg!r}")


def test_scale47b_proxy_assert_is_never_enforced(r: SubTestResult):
    print("\n--- SCALE-47b: the engine never calls this itself -- a mismatched proxy still cooks ---")
    A = make_img(1, 30, 30, 4)   # deliberately NOT the "correct" 32x32 proxy for scale=0.5
    try:
        res = tex_engine.cook("@OUT = gauss_blur(@A, 4.0);", {"A": A}, device_mode="cpu", scale=0.5)
    except Exception as e:
        r.fail("not enforced", f"a mismatched proxy shape made the cook itself refuse: "
               f"{type(e).__name__}: {e}")
        return
    r.ok(f"a cook whose proxy shape does not match its claimed scale still cooks "
         f"(shape {tuple(res.outputs['OUT'].shape)}) -- the engine never enforces this check")
