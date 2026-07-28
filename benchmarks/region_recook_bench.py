#!/usr/bin/env python3
"""CACHE-9 / PM-8 — what a mid-graph edit costs on a big unfused comp.

PM-8 asks: "on a 50-node 4K comp, a mid-graph param edit recooks in <= (threshold + suffix)
rather than the whole graph, measured against the v0.31 baseline; RAM footprint stays under the
governor budget with checkpoints on."

TWO THINGS THAT SHAPE THIS HARNESS, both measured rather than assumed:

 1. A 50-node comp has NO FUSED CHAIN. `_MAX_FUSED_REGION_STAGES` is 16 and `_grow_region`
    returns None past it rather than truncating, so a linear graph of 17+ TEX nodes yields ZERO
    fusable regions (measured: N=16 -> 1 region, N=17 -> 0, N=50 -> 0). So the 50-node shape is
    entirely UNFUSED per-stage cooks — which is CACHE-9's shape, not CACHE-7's. CACHE-7 is
    measured separately, on the <=16-stage fused chains a host can actually produce
    (`checkpoint_bench.py`). Reporting one blended number would hide which mechanism produced it.

 2. Per-stage canvases at 4K do not fit. A 4096^2 x4 fp32 frame is 268 MB, and this comp keeps
    one per stage: 50 of them is 13.4 GB. So `--resolution` is honest about what it ran, and the
    default is the largest square that fits a stated budget. The memory arithmetic is reported,
    not hidden behind a resolution nobody checks.

Rows per (device, resolution):
  whole_all      every stage recooked whole-frame     — what a host without CACHE-9 pays
  region_all     every stage recooked over its window — the composition win
  region_mid     a MID-GRAPH edit: the clean prefix stands, the dirty suffix cooks its window
  peak_mb        RAM held by the frame cache at the end, vs the governor budget
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

_bench_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_bench_dir.parent.parent))
sys.path.insert(0, str(_bench_dir))

os.environ.setdefault(
    "TEX_CACHE_DIR", str(Path(tempfile.gettempdir()) / "tex_bench_cache"))

import torch                                                          # noqa: E402
from run_benchmarks import system_info                                # noqa: E402
from TEX_Wrangle import tex_engine, tex_memory, tex_results, tex_roi  # noqa: E402

# Real compositing nodes, each ROI-EXECUTABLE (pointwise, or a direct-tensor halo op) — the
# class ROI-3 covers. The halo ops are what make the window COMPOSITION load-bearing rather
# than decorative: without growing the upstream window they leave a stale ring.
_POOL = [
    ("exposure",   "@OUT = vec4(@IN.rgb * $k, 1.0);",                              {"k": 1.05}),
    ("blackpoint", "@OUT = vec4(max(@IN.rgb - vec3($k), vec3(0.0)), 1.0);",        {"k": 0.02}),
    ("gamma",      "@OUT = vec4(spow(@IN.rgb, vec3($k)), 1.0);",                   {"k": 0.95}),
    ("saturation", "float y = luma(@IN);\n@OUT = vec4(mix(vec3(y), @IN.rgb, $k), 1.0);",
                                                                                   {"k": 1.10}),
    ("blur",       "@OUT = gauss_blur(@IN, $k);",                                  {"k": 1.5}),
    ("contrast",   "@OUT = vec4((@IN.rgb - vec3(0.5)) * $k + vec3(0.5), 1.0);",    {"k": 1.08}),
    ("tint",       "@OUT = vec4(@IN.rgb * vec3(1.0 + $k, 1.0, 1.0 - $k), 1.0);",   {"k": 0.03}),
    ("glow",       "@OUT = vec4(@IN.rgb + $k * gauss_blur(@IN, 4.0).rgb, 1.0);",   {"k": 0.12}),
]


class Comp:
    """An N-stage UNFUSED comp with a persistent full-frame canvas per stage — the shape
    `host_demo.RoiComp` proved at 10 nodes, driven here through the ENGINE's own
    `tex_roi.chain_windows` + `ResultCache.patch_region` rather than a host's hand-roll."""

    def __init__(self, n: int, res: int, device: str):
        self.n, self.res, self.device = n, res, device
        self.stages = [(_POOL[i % len(_POOL)][1], dict(_POOL[i % len(_POOL)][2]))
                       for i in range(n)]
        self.src = torch.rand(1, res, res, 3, device=device)
        self.cache = tex_results.ResultCache()
        tex_memory.register_result_cache(self.cache, name="pm8")
        self.canvas: list = [None] * n
        # The region each canvas is CURRENTLY correct over; None = the whole frame. A canvas
        # patched over a window is valid only there, so a later edit whose composed window
        # escapes it would read pre-edit pixels — `chain_windows` reports NOT SERVICEABLE
        # (None) when told. Tracking it is the host's job; this is that proof.
        self.valid: list = [None] * n
        self.halos = [tex_roi.stage_halo(c, p) for c, p in self.stages]

    def _cook(self, i, src, roi):
        code, params = self.stages[i]
        res = tex_engine.cook(code, {"IN": src, **params}, device_mode=self.device,
                              precision="fp32", roi=roi, roi_exec=(roi is not None))
        return res.outputs["OUT"], res.cooked_roi

    def cook(self, roi=None, dirty_from: int = 0, patch: bool = True):
        """Cook stages `dirty_from..n-1`, over `roi` when given, patching each stage's canvas."""
        windows = (tex_roi.chain_windows(self.halos, roi, dirty_from, valid=self.valid)
                   if roi is not None else [None] * self.n)
        if windows is None:
            # Not serviceable — an upstream canvas is stale where this edit must read it. No
            # window choice repairs a stale input; the only fix is to re-cook from the source.
            dirty_from, windows = 0, [None] * self.n
        cur = self.src if dirty_from == 0 else self.canvas[dirty_from - 1]
        for i in range(dirty_from, self.n):
            out, served = self._cook(i, cur, windows[i])
            if served is None or not patch or self.canvas[i] is None:
                self.canvas[i] = out            # a declined window REPLACES the canvas
                self.valid[i] = None                # a whole frame is valid everywhere
            else:
                got = self.cache.patch_region(
                    f"s{i}", out, served, base=self.canvas[i])
                self.canvas[i] = got if got is not None else out
                # Correct over the union of what it already was and what we just wrote. The
                # conservative reading — the newly patched window alone — is what this tracks,
                # because a union of two disjoint rects is not a rect.
                self.valid[i] = served if got is not None else None
            cur = self.canvas[i]
        if self.device == "cuda":
            torch.cuda.synchronize()
        return self.canvas[-1]


def _median(fn, reps):
    fn(-1)
    return round(statistics.median(
        [(lambda t0: (fn(i), (time.perf_counter() - t0) * 1000.0)[1])(time.perf_counter())
         for i in range(reps)]), 3)


def _fits(n, res, device) -> bool:
    """Would N canvases at `res` fit? A 4096²x4 fp32 frame is 268 MB; 50 of them is 13.4 GB."""
    need = n * res * res * 4 * 4
    if device == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info()
            return need < 0.6 * free
        except Exception:
            return res <= 1024
    return need < (8 << 30)          # a self-imposed 8 GB ceiling for the CPU lane


def measure(device, n, res, roi_side, reps, edit_at):
    comp = Comp(n, res, device)
    comp.cook(None, 0)                                   # prime every canvas
    span = max(1, res - roi_side)
    roi = (span // 2, span // 2, roi_side, roi_side, res, res)
    row = {"stages": n, "res": res, "roi": roi_side, "edit_at": edit_at}

    def _whole_all(i):
        comp.stages[0][1]["k"] = 1.05 + (i % 11) * 0.001
        comp.cook(None, 0)

    def _region_all(i):
        comp.stages[0][1]["k"] = 1.06 + (i % 11) * 0.001
        comp.cook(roi, 0)

    def _region_mid(i):
        comp.stages[edit_at][1]["k"] = 1.07 + (i % 11) * 0.001
        comp.cook(roi, edit_at)

    row["whole_all"] = _median(_whole_all, reps)
    row["region_all"] = _median(_region_all, reps)
    row["region_mid"] = _median(_region_mid, reps)
    row["speedup_region_all"] = round(row["whole_all"] / row["region_all"], 2)
    row["speedup_region_mid"] = round(row["whole_all"] / row["region_mid"], 2)

    # PM-8's second half: "RAM footprint stays under the governor budget". The CACHE-5
    # governor is HOST-DRIVEN — `arbitrate()` is a call a host makes at its own safe points,
    # not a background thread the engine runs (invariant #7: nothing arbitrates on the default
    # cook path). So this measures BOTH numbers: what the cache holds if nobody drives the
    # governor, and what it holds after one arbitration. Reporting only the second would claim
    # a guarantee the engine does not make; reporting only the first would call a host's
    # omission an engine failure.
    row["governor_budget_mb"] = round(tex_memory.governor_budget(device) / (1 << 20), 1)
    row["ram_mb_undriven"] = round(comp.cache.stats()["ram_bytes"] / (1 << 20), 1)
    freed = tex_memory.get_cache_registry().arbitrate(device)
    st = comp.cache.stats()
    row["arbitrate_freed_mb"] = round(freed / (1 << 20), 1)
    row["cache_ram_mb"] = round(st["ram_bytes"] / (1 << 20), 1)
    row["under_budget"] = row["cache_ram_mb"] <= row["governor_budget_mb"]
    row["profile"] = tex_memory.active_profile()
    row["windows"] = [w[2] for w in tex_roi.chain_windows(comp.halos, roi, edit_at)
                      if w is not None][:4]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=50)
    ap.add_argument("--resolution", type=int, default=0, help="0 = largest that fits")
    ap.add_argument("--roi", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--profile", type=str, default=None, help="GOV-1 preset to run under")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    if args.profile:
        tex_memory.set_profile(args.profile)

    devices = ["cpu"] + ([] if args.cpu_only or not torch.cuda.is_available() else ["cuda"])
    out = {"meta": {**system_info("matrix"), "stages": args.stages, "reps": args.reps,
                    "profile": tex_memory.active_profile()},
           "rows": {}}

    for device in devices:
        if args.resolution:
            res = args.resolution
            if not _fits(args.stages, res, device):
                print(f"{device}: SKIP {args.stages}x{res}^2 — "
                      f"{args.stages * res * res * 16 / (1 << 30):.1f} GB of canvases "
                      "does not fit; see the module docstring.", flush=True)
                continue
        else:
            res = next((r for r in (4096, 2048, 1024, 512)
                        if _fits(args.stages, r, device)), 512)
        roi = min(args.roi, res // 2)
        r = measure(device, args.stages, res, roi, args.reps, args.stages // 2)
        out["rows"][f"{device}/n{args.stages}/{res}"] = r
        print(f"{device} {args.stages} nodes @ {res}^2, ROI {roi}^2 "
              f"({args.stages * res * res * 16 / (1 << 30):.2f} GB of canvases)", flush=True)
        print(f"   whole-frame all-dirty : {r['whole_all']:9.2f} ms", flush=True)
        print(f"   region     all-dirty : {r['region_all']:9.2f} ms  "
              f"({r['speedup_region_all']}x)", flush=True)
        print(f"   region     MID-GRAPH : {r['region_mid']:9.2f} ms  "
              f"({r['speedup_region_mid']}x)   <- PM-8", flush=True)
        print(f"   frame cache {r['ram_mb_undriven']} MB undriven -> arbitrate freed "
              f"{r['arbitrate_freed_mb']} MB -> {r['cache_ram_mb']} MB vs budget "
              f"{r['governor_budget_mb']} MB [{r['profile']}] -> "
              f"{'UNDER' if r['under_budget'] else 'OVER'}", flush=True)

    if args.save:
        p = Path(args.save)
        if not p.is_absolute():
            p = _bench_dir / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nsaved -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
