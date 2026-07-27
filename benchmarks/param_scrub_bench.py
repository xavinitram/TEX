"""ANIM-1 — the price of the animated-parameter contract.

`tests/test_v031_anim_contract.py` proves a `$param` sweep causes 0 recompiles. This measures
what it *does* cost, which is the number a host planning a timeline actually needs.

The shape of the result is the claim. Cook the same program 1000 times, changing `$x` every
cook, and plot cost against cook index: if the contract holds the line is FLAT — cook cost
only, no compile-cost slope, no periodic spike where a recompile lands. So the report is not
one median but four: the cold cook, the first-decile median, the last-decile median, and the
max after warmup. A contract violation shows up as last >> first, or as a max far above the
median (one recompile hiding in a thousand cooks).

Two controls make the flat line meaningful:

  static   the same 1000 cooks with the param held CONSTANT — the floor. `scrub` should
           match it; the gap is what animating actually costs.
  recook   1000 cooks each with a genuinely DIFFERENT program — what a host would pay if
           params were baked into the code (the naive way to build a keyframe system).
           Reported as a ratio, because that ratio is the contract's whole value.

Rerun per release; it is the keyframe workload's proxy (doc 40 §4.1).

    python benchmarks/param_scrub_bench.py
    python benchmarks/param_scrub_bench.py --res 512 --cooks 1000 --device cuda --save results/anim1.json
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from TEX_Wrangle import tex_engine

CODE = "@OUT = vec4(@A.rgb * $x + vec3(0.02), 1.0);"


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _series(fn, n, device):
    out = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        _sync(device)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def bench(res: int, cooks: int, device: str) -> dict:
    torch.manual_seed(31)
    A = torch.rand(1, res, res, 4, device=device)
    dec = max(1, cooks // 10)

    def scrub(i):
        tex_engine.cook(CODE, {"A": A, "x": 0.5 + (i % 997) * 0.001},
                        device_mode=device, precision="fp32")

    def static(i):
        tex_engine.cook(CODE, {"A": A, "x": 0.75}, device_mode=device, precision="fp32")

    def recook(i):
        tex_engine.cook(f"@OUT = vec4(@A.rgb * {0.5 + i * 0.001:.6f} + vec3(0.02), 1.0);",
                        {"A": A}, device_mode=device, precision="fp32")

    # The COLD cook is measured on its own, before anything is warm — it is the one cook the
    # contract allows to compile, and folding it into the series would hide it in the median.
    t0 = time.perf_counter()
    scrub(0)
    _sync(device)
    cold_ms = (time.perf_counter() - t0) * 1000.0

    s = _series(scrub, cooks, device)
    st = _series(static, cooks, device)
    # The recook control is genuinely expensive (a full compile per cook), so it runs a tenth
    # as many times — enough for a stable median, and the ratio is what is reported.
    rc = _series(recook, max(20, cooks // 10), device)

    out = {
        "res": res, "cooks": cooks, "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cold_ms": round(cold_ms, 4),
        "scrub_first_decile_ms": round(statistics.median(s[:dec]), 4),
        "scrub_last_decile_ms": round(statistics.median(s[-dec:]), 4),
        "scrub_median_ms": round(statistics.median(s), 4),
        "scrub_max_after_warmup_ms": round(max(s[dec:]), 4),
        "static_median_ms": round(statistics.median(st), 4),
        "recook_median_ms": round(statistics.median(rc), 4),
    }
    out["animation_cost_ms"] = round(out["scrub_median_ms"] - out["static_median_ms"], 4)
    out["slope_last_over_first"] = round(
        out["scrub_last_decile_ms"] / max(out["scrub_first_decile_ms"], 1e-9), 3)
    out["vs_recompiling_host_x"] = round(
        out["recook_median_ms"] / max(out["scrub_median_ms"], 1e-9), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--cooks", type=int, default=1000)
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--save", type=str, default=None)
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device

    r = bench(a.res, a.cooks, device)
    print(f"\nANIM-1 — param scrub  ({r['res']}^2, {r['device']}, {r['cooks']} cooks)")
    print(f"  cold cook (compiles)          {r['cold_ms']:9.4f} ms")
    print(f"  scrub, first decile           {r['scrub_first_decile_ms']:9.4f} ms")
    print(f"  scrub, last decile            {r['scrub_last_decile_ms']:9.4f} ms   "
          f"(slope {r['slope_last_over_first']}x — flat is the contract)")
    print(f"  scrub, max after warmup       {r['scrub_max_after_warmup_ms']:9.4f} ms")
    print(f"  static param (the floor)      {r['static_median_ms']:9.4f} ms   "
          f"(animating costs {r['animation_cost_ms']:+.4f} ms)")
    print(f"  a recompiling host would pay  {r['recook_median_ms']:9.4f} ms   "
          f"({r['vs_recompiling_host_x']}x)")
    if a.save:
        path = a.save if os.path.isabs(a.save) else \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), a.save)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
