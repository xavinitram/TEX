#!/usr/bin/env python3
"""
ROI scrub benchmark (v0.30 — First viewer)
==========================================
The measurement behind the v0.30 ROI items and PM-6's "interactive rate" claim.

Four rows, because they isolate four different costs:

    whole            whole-frame cook at the full resolution        (the thing ROI avoids)
    roi_fixed        one fixed sub-window, same window every frame  (the ROI win, warm memo)
    roi_panning      the window MOVES every frame                   (LAT-4 builtins-LRU cost)
    roi_pan_param    the window moves AND a $param scrubs           (roi_plan memo-key cost)

The last row is the one a naive benchmark misses: `tex_roi.roi_plan`'s memo key folds every
scalar param value, so a viewport scrubbing a slider re-parses + re-folds + re-walks the
program EVERY frame. A loop that holds the param constant reports the fixed-window number and
hides it entirely — which is exactly how this cost survived to v0.30.

Every CUDA timing region is `torch.cuda.synchronize()`d (the repo's standing gotcha: without
it you measure kernel-launch overhead, not execution).

Usage
-----
    python benchmarks/roi_scrub_bench.py                                  # cpu + cuda if present
    python benchmarks/roi_scrub_bench.py --device cuda --resolution 1024
    python benchmarks/roi_scrub_bench.py --save results/roi_before.json
    python benchmarks/roi_scrub_bench.py --save results/roi_after.json --compare results/roi_before.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                                # .../TEX_Wrangle
sys.path.insert(0, os.path.dirname(_PKG))                    # .../custom_nodes (package parent)

import torch                                                  # noqa: E402
from TEX_Wrangle import tex_engine, tex_roi                    # noqa: E402

# A program in the ROI-executable class (pointwise + a direct-tensor halo op) that also reads a
# $param, so the memo-key row has something to scrub. gauss_blur carries the ('halo_arg', 1, 3.0)
# reach descriptor, so the ROI cook region is ROI ⊕ ceil(3·sigma) = ROI ⊕ 6.
#
# The blur MUST stay inline: routing it through a named local (`vec4 b = gauss_blur(...)`) makes
# the program NON-ROI-executable in v1 (the name boundary blocks reach composition — the
# "precise local-variable dataflow" ROI-5 reopen item), and the whole benchmark would silently
# measure whole-frame cooks. `_roi_path_taken` is the guard that catches exactly that mistake.
_CODE = "@OUT = vec4(mix(@A.rgb, gauss_blur(@A, 2.0).rgb, $amount), 1.0);\n"


def _median_ms(fn, n: int, device: str) -> float:
    fn()                                          # warm (compile + cache + first alloc)
    if device == "cuda":
        torch.cuda.synchronize()
    samples = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        if device == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(samples), 4)


def _supports_roi_exec() -> bool:
    """v0.30 added the per-cook `roi_exec` arm; before that the only arm was TEX_ROI_EXEC.
    Feature-detect so this ONE script can measure a pre-v0.30 worktree and a v0.30 tree
    back to back (the repo's A/B worktree probing rule) — otherwise the 'before' side of
    every comparison would have to be a different, non-comparable script."""
    import inspect
    try:
        return "roi_exec" in inspect.signature(tex_engine.prepare).parameters
    except Exception:
        return False


def bench_device(device: str, res: int, roi_side: int, frames: int) -> dict:
    torch.manual_seed(7)
    A = torch.rand(1, res, res, 4, device=device)
    span = max(1, res - roi_side)                 # pan the window across the frame
    per_cook_arm = _supports_roi_exec()
    if not per_cook_arm:
        os.environ["TEX_ROI_EXEC"] = "1"          # pre-v0.30: the env var is the only arm

    def _cook(roi=None, amount=0.5):
        kw = {"roi_exec": roi is not None} if per_cook_arm else {}
        return tex_engine.cook(_CODE, {"A": A, "amount": amount}, device_mode=device,
                               precision="fp32", roi=roi, **kw)

    def _win(i):
        x0 = (i * 37) % span                      # a deterministic pan (coprime-ish stride)
        y0 = (i * 23) % span
        return (x0, y0, roi_side, roi_side, res, res)

    rows: dict = {}
    rows["whole"] = _median_ms(lambda i=0: _cook(None), frames, device)

    fixed = (span // 2, span // 2, roi_side, roi_side, res, res)
    rows["roi_fixed"] = _median_ms(lambda i=0: _cook(fixed), frames, device)
    rows["roi_panning"] = _median_ms(lambda i=0: _cook(_win(i)), frames, device)
    rows["roi_pan_param"] = _median_ms(
        lambda i=0: _cook(_win(i), amount=0.25 + (i % 100) * 0.005), frames, device)

    # Prove the rows actually took the ROI path — a silent whole-frame fallback would make
    # every number meaningless (and this benchmark would "improve" by doing nothing). Pre-v0.30
    # there is no `cooked_roi`, so fall back to the output shape as the evidence.
    probe = _cook(fixed)
    out_shape = list(probe.outputs["OUT"].shape)
    rows["_roi_path_taken"] = (probe.cooked_roi == fixed) if per_cook_arm \
        else (out_shape[1] == roi_side and out_shape[2] == roi_side)
    rows["_roi_out_shape"] = out_shape
    rows["_speedup_vs_whole"] = round(rows["whole"] / max(rows["roi_fixed"], 1e-9), 2)
    rows["_param_scrub_overhead_ms"] = round(rows["roi_pan_param"] - rows["roi_panning"], 4)
    rows["_pan_overhead_ms"] = round(rows["roi_panning"] - rows["roi_fixed"], 4)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="TEX ROI scrub benchmark (v0.30)")
    ap.add_argument("--device", default=None, help="cpu|cuda (default: both available)")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--roi", type=int, default=256, help="ROI side in px (default 256)")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--save", default=None)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    devices = [args.device] if args.device else \
        (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    out = {
        "meta": {
            "torch": torch.__version__,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "capability": ("sm_" + "".join(map(str, torch.cuda.get_device_capability(0))))
                          if torch.cuda.is_available() else None,
            "resolution": args.resolution, "roi": args.roi, "frames": args.frames,
            "roi_exec_default": tex_roi.roi_exec_enabled(),
        },
        "rows": {},
    }
    for dev in devices:
        print(f"\n=== {dev} — {args.resolution}^2 frame, {args.roi}^2 ROI, {args.frames} frames ===")
        rows = bench_device(dev, args.resolution, args.roi, args.frames)
        out["rows"][dev] = rows
        for k in ("whole", "roi_fixed", "roi_panning", "roi_pan_param"):
            print(f"  {k:16} {rows[k]:9.4f} ms")
        print(f"  {'-> roi speedup':16} {rows['_speedup_vs_whole']:9.2f}x vs whole-frame")
        print(f"  {'-> pan cost':16} {rows['_pan_overhead_ms']:9.4f} ms/frame")
        print(f"  {'-> param cost':16} {rows['_param_scrub_overhead_ms']:9.4f} ms/frame")
        if not rows["_roi_path_taken"]:
            print("  !! ROI PATH NOT TAKEN — numbers are whole-frame cooks, not ROI cooks")

    if args.save:
        path = args.save if os.path.isabs(args.save) else os.path.join(_HERE, args.save)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved {path}")

    if args.compare:
        base_path = args.compare if os.path.isabs(args.compare) else os.path.join(_HERE, args.compare)
        with open(base_path, encoding="utf-8") as f:
            base = json.load(f)
        print(f"\n=== compare vs {args.compare} (>1.00 = faster now) ===")
        for dev, rows in out["rows"].items():
            brows = base.get("rows", {}).get(dev)
            if not brows:
                print(f"  {dev}: no baseline"); continue
            for k in ("whole", "roi_fixed", "roi_panning", "roi_pan_param"):
                if k in brows and brows[k]:
                    ratio = brows[k] / max(rows[k], 1e-9)
                    flag = "  <-- REGRESSION" if ratio < 0.95 else ""
                    print(f"  {dev:5} {k:16} {brows[k]:8.4f} -> {rows[k]:8.4f} ms  "
                          f"({ratio:.2f}x){flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
