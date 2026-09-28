"""FIX-501 F3 -- `bilateral_filter`'s exact tier (radius<=24) used to differ, by a few
ULPs, between a windowed cook and the corresponding crop of a whole-frame cook, on CUDA
only (CPU read 0.0 in every reproduction). Root cause (an embedding host's own finding,
confirmed by reading `_bilateral_exact_bchw`): its row-tiling divides work into tiles
sized from the INPUT'S OWN WIDTH (`_BILATERAL_TILE_BUDGET_ELEMS // (W * ksize * ksize)`),
so a narrower windowed crop and the wider whole frame tile at different row heights for
the SAME output pixels -- and a CUDA reduction over the kernel window (`.sum(dim=(-2,
-1))`) is not guaranteed bit-identical across differently shaped surrounding tensors,
even though every output pixel's own math is unchanged.

The fix confines the reduction inside `_bilateral_exact_bchw` to a shape-independent,
strictly sequential accumulation (one elementwise add per kernel tap, in a fixed
row-major order) whenever `radius > 3` -- i.e. only in the row-tiled regime this bug
lives in. `radius<=3` keeps calling the ORIGINAL `_bilateral_weighted_avg` unchanged, so
today's bit-identity pin (`test_bilat50_radius.py`) is untouched.
"""
from __future__ import annotations

import torch

from helpers import *  # noqa: F401,F403
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib as _Sample  # populates REGISTRY

_CUDA = torch.cuda.is_available()


def _make_frame(H, W, seed):
    torch.manual_seed(seed)
    return torch.rand(1, H, W, 3)


def _windowed_vs_whole(code, image, roi, device):
    x0, y0, w, h, W, H = roi
    full = tex_engine.cook(code, {"A": image.clone()}, device_mode=device).outputs["OUT"]
    res = tex_engine.cook(code, {"A": image.clone()}, device_mode=device,
                           roi=roi, roi_exec=True)
    win = res.outputs["OUT"]
    crop = full[:, y0:y0 + h, x0:x0 + w]
    return res.cooked_roi, win, crop


def _repro_roi(r: SubTestResult, ss: float, seed: int, device: str):
    # A wide-but-short frame (matches the host's own 1024x576 repro shape) and a
    # narrower, non-saturating crop -- different widths feed `_bilateral_exact_bchw`'s
    # width-derived tile height a different value for the crop than for the frame it
    # is drawn from, the exact condition F3's hypothesis names.
    H, W = 576, 1024
    image = _make_frame(H, W, seed=seed)
    code = f"@OUT = bilateral_filter(@A, {ss}, 0.2);"
    roi = (300, 180, 256, 220, W, H)  # interior, halo does not saturate to the frame
    cooked_roi, win, crop = _windowed_vs_whole(code, image, roi, device)
    if cooked_roi != roi:
        r.fail(f"bilateral fix501 window identity ({device}, ss={ss})",
               f"window unexpectedly declined: cooked_roi={cooked_roi}")
        return
    if torch.equal(win, crop):
        r.ok(f"bilateral_filter exact tier (ss={ss}) windowed cook is torch.equal to the "
             f"whole-frame crop on {device}")
    else:
        md = (win.float() - crop.float()).abs().max().item()
        r.fail(f"bilateral fix501 window identity ({device}, ss={ss})",
               f"windowed cook diverges from whole-frame crop, maxdiff={md:.4e}")


def test_fix501_bilateral_exact_tier_window_identity_cpu(r: SubTestResult):
    print("\n--- FIX-501 F3: bilateral_filter exact tier, windowed vs whole-frame, CPU "
          "(control -- always read 0.0 in every reproduction) ---")
    _repro_roi(r, 2.0, seed=201, device="cpu")   # radius=6
    _repro_roi(r, 3.0, seed=202, device="cpu")   # radius=9


def test_fix501_bilateral_exact_tier_window_identity_cuda(r: SubTestResult):
    if not _CUDA:
        r.skip("FIX-501 F3 CUDA window identity", "no CUDA device present on this box")
        return
    print("\n--- FIX-501 F3: bilateral_filter exact tier, windowed vs whole-frame, CUDA "
          "(an embedding host's own repro shape: radius 6 and 9) ---")
    _repro_roi(r, 2.0, seed=201, device="cuda")   # radius=6
    _repro_roi(r, 3.0, seed=202, device="cuda")   # radius=9


def test_fix501_bilateral_taploop_tier_window_identity_cpu(r: SubTestResult):
    print("\n--- BILATX-51: the tap-loop exact tier (3<radius<=40, "
          "_bilateral_exact_taploop_bchw) keeps windowed cook == whole-frame crop, CPU "
          "(control) ---")
    _repro_roi(r, 8.0, seed=204, device="cpu")           # radius=24, old tiled ceiling
    _repro_roi(r, 10.0, seed=205, device="cpu")          # radius=30
    _repro_roi(r, 40.0 / 3.0, seed=206, device="cpu")    # radius=40, the new ceiling


def test_fix501_bilateral_taploop_tier_window_identity_cuda(r: SubTestResult):
    if not _CUDA:
        r.skip("BILATX-51 taploop CUDA window identity", "no CUDA device present on this box")
        return
    print("\n--- BILATX-51: the tap-loop exact tier's per-tap accumulation is already "
          "shape-independent (each add combines two full-frame-shaped tensors, never a "
          "tensor shaped by how the caller tiled) -- windowed cook == whole-frame crop on "
          "CUDA at radius 24, 30 and 40 (the new ceiling), without needing FIX-501 F3's own "
          "`_sum_kernel_taps_fixed_order` helper ---")
    _repro_roi(r, 8.0, seed=204, device="cuda")           # radius=24
    _repro_roi(r, 10.0, seed=205, device="cuda")          # radius=30
    _repro_roi(r, 40.0 / 3.0, seed=207, device="cuda")    # radius=40, the new ceiling


def test_fix501_bilateral_radius_le_3_unchanged_on_cuda(r: SubTestResult):
    if not _CUDA:
        r.skip("FIX-501 F3 CUDA window identity", "no CUDA device present on this box")
        return
    print("\n--- FIX-501 F3 guard: radius<=3 (ss<=1.0) is UNTOUCHED by the fix -- stays "
          "on the original _bilateral_weighted_avg path, still torch.equal windowed vs "
          "whole-frame on CUDA (it always was; this just proves the fix didn't move it) ---")
    _repro_roi(r, 1.0, seed=203, device="cuda")   # radius=3, the protected boundary
