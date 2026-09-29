"""FIX-ROI51 R1 (v0.51 Phase C, B1#1) -- `_reach_of`'s `halo_arg` branch must size the
declared reach (and decide the `approx_above` decline) from the SAME value the runtime
builtin actually computes with, not the raw fp64 AST literal.

THE BUG (confirmed at base `365fdb4` by the v0.51 Phase C bug hunt, B1 finding 1):
`_reach_of` (tex_roi.py) read a `halo_arg` argument through `_static_number`, which is
`NumberLiteral.value` -- the raw Python `float()` of the source token, never rounded.
`fn_bilateral_filter`/`gauss_blur` (tex_runtime/stdlib_sample.py) instead resolve the
SAME argument through `_host_scalar`/a bare-float fallback that fp32-rounds it via
`_dtype_rounded(raw, torch.float32)` before computing `radius = ceil(3.0*ss)`. An
ordinary 7-8 significant-digit literal can sit close enough to a `k/3` radius boundary
that fp32-rounding flips which side of the integer `ceil` it lands on -- `_reach_of`
then declares (and sizes the cook halo to) a radius ONE PIXEL SMALLER than the radius
the builtin actually loops out to, so a windowed/tiled/ROI-narrowed cook silently reads
wrong pixels near its own edge (invariant 5's "wrong only when narrowed" shape).

Repro (B1's own numbers): `spatial_sigma = 4.3333333` (raw fp64). Raw `ceil(3*4.3333333)
== 13` (what `_reach_of` declared/used at base); fp32-rounded `ceil(3*4.333333492279053)
== 14` (what `fn_bilateral_filter` actually loops out to) -- a 1-pixel halo shortfall.

THE FIX: `_reach_of`'s `halo_arg` branch now fp32-rounds `v` through the runtime's own
`_dtype_rounded` (imported from `tex_runtime.stdlib`, not copied) before computing the
declared reach AND before the `approx_above` decline comparison, with the same
fall-back-to-raw-on-`_dtype_rounded is None` the runtime's own call sites use.

RED AT BASE (`365fdb4`, verified by hand before this fix landed): `test_r1_pixel_identity_
bilateral_taploop_boundary` failed with `torch.equal: False`, maxdiff ~2.9e-4 (matching
B1's own repro number); `test_r1_reach_of_matches_runtime_radius_scan` failed at several
of the scanned "dangerous-direction" literals (declared reach one pixel below the
runtime's own radius).
"""
from __future__ import annotations

import math
import struct

import torch

from helpers import *  # noqa: F401,F403
from TEX_Wrangle import tex_engine
from TEX_Wrangle import tex_roi as _R
from TEX_Wrangle.tex_compiler.ast_nodes import NumberLiteral
from TEX_Wrangle.tex_runtime.stdlib_core import _dtype_rounded

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]

# B1's own repro literal: raw ceil(3*v) == 13, fp32-rounded ceil(3*v) == 14.
_BOUNDARY_SIGMA = 4.3333333

_BLUR_CODE = """
@OUT = bilateral_filter(@image, %s, 0.2);
""" % repr(_BOUNDARY_SIGMA)


def _runtime_radius(raw: float, mult: float = 3.0) -> int:
    """The radius `fn_bilateral_filter`/`gauss_blur` actually dispatch on for a bare
    Python float -- fp32-round through the runtime's OWN `_dtype_rounded`, falling back
    to the raw value only when rounding cannot be established (the identical fallback
    both the runtime call sites and the fix use)."""
    rounded = _dtype_rounded(float(raw), torch.float32)
    v = raw if rounded is None else rounded
    return int(math.ceil(mult * abs(v)))


def _dangerous_literals():
    """A small scan for 7-8 significant-digit literals whose raw-double `ceil(3*v)`
    disagrees with their fp32-rounded `ceil(3*v)` -- the same "dangerous-direction"
    shape B1's own brute-force scan found ~50 of, spread across the tap-loop tier's
    radius range (k=1..40ish). Seeded near successive `k/3` boundaries so this keeps
    finding examples even if the exact literal in B1's prose ever changes."""
    out = []
    for k in range(2, 40):
        base = k / 3.0
        # Perturb in the last couple of decimal digits of an 8-sig-fig literal, biased
        # toward the boundary from below (so the raw double sits just under the
        # integer, while fp32-rounding can push it just over).
        for delta in (1e-7, 3e-7, 7e-7, 1.5e-6):
            v = round(base - delta, 8)
            if v <= 0:
                continue
            raw_ceil = int(math.ceil(3.0 * v))
            fp32_v = struct.unpack("<f", struct.pack("<f", v))[0]
            fp32_ceil = int(math.ceil(3.0 * fp32_v))
            if fp32_ceil != raw_ceil:
                out.append(v)
    return out


def test_r1_reach_of_matches_runtime_radius_scan(r: SubTestResult):
    print("\n--- FIX-ROI51 R1: _reach_of's halo_arg reach agrees with the runtime's own "
          "fp32-rounded radius across a scan of dangerous literals ---")
    literals = _dangerous_literals()
    if not literals:
        r.fail("R1 dangerous-literal scan", "scan found no boundary-straddling literals "
               "-- test itself is broken")
        return
    fp = ("halo_arg", 0, 3.0, None)
    mismatches = []
    for v in literals + [_BOUNDARY_SIGMA]:
        lit = NumberLiteral(value=v, is_int=False)
        got = _R._reach_of(fp, [lit])
        want = _runtime_radius(v)
        if got != want:
            mismatches.append((v, got, want))
    if mismatches:
        r.fail("R1 reach_of vs runtime radius",
               f"{len(mismatches)}/{len(literals) + 1} literals disagree: "
               f"{mismatches[:5]}{'...' if len(mismatches) > 5 else ''}")
        return
    r.ok(f"_reach_of agrees with the runtime's fp32-rounded radius on all "
         f"{len(literals) + 1} scanned literals (including B1's own {_BOUNDARY_SIGMA})")


def test_r1_pixel_identity_bilateral_taploop_boundary(r: SubTestResult):
    print("\n--- FIX-ROI51 R1: windowed bilateral_filter at the fp32 radius boundary is "
          "torch.equal to a whole-frame crop (B1's own repro; runs on every device this "
          "box exposes) ---")
    W, H = 1024, 576
    x0, y0, w, h = 300, 180, 256, 220
    roi = (x0, y0, w, h, W, H)
    torch.manual_seed(51001)
    image_cpu = torch.rand(1, H, W, 3)

    for device in _DEVICES:
        try:
            image = image_cpu.clone().to(device)
            _R.clear_roi_memo()
            full = tex_engine.cook(_BLUR_CODE, {"image": image.clone()},
                                    device_mode=device).outputs["OUT"]
            _R.clear_roi_memo()
            res = tex_engine.cook(_BLUR_CODE, {"image": image.clone()},
                                   device_mode=device, roi=roi, roi_exec=True)
            win = res.outputs["OUT"]
            if res.cooked_roi != roi:
                r.fail(f"R1 pixel identity ({device})",
                       f"window declined: cooked_roi={res.cooked_roi}")
                continue
            crop = full[:, y0:y0 + h, x0:x0 + w]
            if tuple(win.shape) == tuple(crop.shape) and torch.equal(win, crop):
                r.ok(f"windowed bilateral_filter ({device}) torch.equal whole-frame crop "
                     f"at spatial_sigma={_BOUNDARY_SIGMA}")
            else:
                md = (win.float() - crop.float()).abs().max().item() \
                    if tuple(win.shape) == tuple(crop.shape) else float("nan")
                r.fail(f"R1 pixel identity ({device})",
                       f"shape {tuple(win.shape)} vs {tuple(crop.shape)}, maxdiff {md:.4e}")
        except Exception as e:
            r.fail(f"R1 pixel identity ({device})", f"{type(e).__name__}: {e}")
        finally:
            _R.clear_roi_memo()
    # No separate CUDA skip report: the fix is a host-side float rounding correction
    # (never a device-numerics one), so the CPU row above is already a full witness of
    # the fix on every box; `_DEVICES` simply adds the CUDA row too wherever a device
    # is present, without treating its absence as a reportable gap (SIMP-3's skip
    # budget is a ratchet on rows that genuinely cannot be proven any other way).


def test_r1_approx_above_decline_uses_rounded_value(r: SubTestResult):
    print("\n--- FIX-ROI51 R1: the approx_above decline compares the SAME rounded value "
          "the reach is computed from ---")
    # A literal whose raw double is <= an approx_above threshold but whose fp32
    # rounding pushes it just over -- the decline decision and the reach computation
    # must agree (both rounded), never split (one raw, one rounded). Hand-picked so
    # `v` and `threshold` straddle the same fp32 rounding cell: `v` rounds up to
    # 13.333333969116211, which is > `threshold`, even though the raw double `v` is
    # itself <= `threshold`.
    v = 13.3333336
    threshold = 13.3333338
    fp32_v = struct.unpack("<f", struct.pack("<f", v))[0]
    if not (v <= threshold and fp32_v > threshold):
        r.fail("R1 approx_above decline", "fixture literals stopped straddling the fp32 "
               f"rounding cell (raw={v}, fp32={fp32_v}, threshold={threshold})")
        return
    fp = ("halo_arg", 0, 1.0, threshold)
    lit = NumberLiteral(value=v, is_int=False)
    got = _R._reach_of(fp, [lit])
    if got != "unbounded":
        r.fail("R1 approx_above decline", f"raw={v}, fp32={fp32_v} > {threshold} "
               f"should decline (unbounded), got {got}")
        return
    r.ok(f"declines to unbounded when the ROUNDED value ({fp32_v}) crosses "
         f"approx_above={threshold}")
