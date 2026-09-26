"""SCALE-47b phase 2 — `scale=` threaded through `tex_engine.prepare()`/`ExecContext`, inert.

No pixel-changing behaviour lands here (that is phase 3): this file proves the PLUMBING —
`prepare(..., scale=...)` reaches `ExecContext.scale` unchanged, the default is `None`, and an
ROI request declines cleanly (with a discoverable reason) when `scale` is also active, the same
way every other ROI ineligibility already does (SCALE-47-design.md §2's invariant: `scale=None`
touches zero bytes, exactly `want_lineage`'s off-by-default shape).
"""
from helpers import *
from TEX_Wrangle import tex_engine
from TEX_Wrangle import tex_roi as _R


def test_scale47b_scale_defaults_none_on_ctx(r: SubTestResult):
    print("\n--- SCALE-47b: prepare() with no scale= leaves ExecContext.scale None ---")
    code = "@OUT = @A * 2.0;"
    A = make_img(1, 4, 4, 4)
    plan = tex_engine.prepare(code, {"A": A}, device_mode="cpu")
    if plan.ctx.scale is not None:
        r.fail("default scale", f"expected None, got {plan.ctx.scale!r}")
        return
    r.ok("ExecContext.scale is None when the caller never passes scale=")


def test_scale47b_scale_roundtrips_through_ctx(r: SubTestResult):
    print("\n--- SCALE-47b: prepare(scale=0.5) reaches ExecContext.scale unchanged ---")
    code = "@OUT = @A * 2.0;"
    A = make_img(1, 4, 4, 4)
    plan = tex_engine.prepare(code, {"A": A}, device_mode="cpu", scale=0.5)
    if plan.ctx.scale != 0.5:
        r.fail("scale roundtrip", f"expected 0.5, got {plan.ctx.scale!r}")
        return
    r.ok("prepare(scale=0.5).ctx.scale == 0.5")


def test_scale47b_roi_declines_when_scale_active(r: SubTestResult):
    print("\n--- SCALE-47b: an ROI request declines when scale is also active ---")
    _R.clear_roi_memo()
    code = "@OUT = gauss_blur(@A, 2.0);"
    A = make_img(1, 16, 16, 4)
    win = (0, 0, 8, 8, 16, 16)
    res = tex_engine.cook(code, {"A": A}, device_mode="cpu", roi=win, roi_exec=True, scale=0.5)
    if res.cooked_roi is not None:
        r.fail("roi+scale", f"expected cooked_roi=None (declined), got {res.cooked_roi!r}")
        return
    if tuple(res.outputs["OUT"].shape) != (1, 16, 16, 4):
        r.fail("roi+scale shape", f"expected the whole 16x16 frame, got "
               f"{tuple(res.outputs['OUT'].shape)}")
        return
    r.ok("roi=+scale= together decline the window and cook whole-frame, never a wrong shape")


def test_scale47b_roi_alone_still_arms(r: SubTestResult):
    print("\n--- SCALE-47b: an ROI request with NO scale is unaffected (invariant #7) ---")
    _R.clear_roi_memo()
    code = "@OUT = gauss_blur(@A, 2.0);"
    A = make_img(1, 16, 16, 4)
    win = (0, 0, 8, 8, 16, 16)
    res = tex_engine.cook(code, {"A": A}, device_mode="cpu", roi=win, roi_exec=True)
    if res.cooked_roi != win:
        r.fail("roi alone", f"expected cooked_roi={win!r} (still armed), got {res.cooked_roi!r}")
        return
    r.ok("roi= with no scale= is byte-for-byte the pre-SCALE-47b behaviour")
