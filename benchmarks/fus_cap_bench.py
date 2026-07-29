"""FUS-cap decision spike (doc 41 §3.5) — is the 16-stage fusion cap a cliff worth fixing?

`_MAX_FUSED_REGION_STAGES = 16` with `_grow_region` returning `None` past it means a
17-stage linear graph gets **zero** fusion: N=16 -> 1 region, N=17 -> 0, N=50 -> 0. That is a
cliff, not a taper, and nothing has ever measured what it costs. This does.

Three routes, same pixels, at N = 16 / 17 / 24 / 50 on both devices:

  fused      ONE region of N stages. Only legal at N<=16 today; measured past the cap anyway
             as the upper bound the other two are judged against (the planner refuses it, the
             splicer does not — `compile_fused` will happily splice 50 stages).
  segmented  ceil(N/16) fused regions of <=16 stages, chained: the alternative the spike
             exists to price. Each region cooks and hands its output to the next.
  unfused    N single-stage cooks, chained. THIS IS WHAT SHIPS at N>16.

The verdict is `unfused / segmented` — how much of the cliff segmenting buys back.

    python benchmarks/fus_cap_bench.py --res 512
    python benchmarks/fus_cap_bench.py --res 1024 --save results/fus_cap.json
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_fusion import _MAX_FUSED_REGION_STAGES as CAP

#: One stage of the chain. Cheap and pointwise on purpose: the question is what FUSION
#: costs (per-stage materialization + per-cook overhead), and a heavy kernel would bury it.
STAGE = "@OUT = vec4(@A.rgb * {k:.4f} + vec3({b:.4f}), @A.a);"


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _stages(n):
    """A linear chain. Stage 0 reads the external `@A`; every later stage reads its
    predecessor's OUT through `chain_inputs` — the wiring `_grow_region` emits."""
    out = []
    for i in range(n):
        st = {"code": STAGE.format(k=1.0 + i * 0.001, b=i * 0.0005), "bindings": {}}
        if i:
            st["chain_inputs"] = {"A": [i - 1, "OUT"]}
        out.append(st)
    return out


def _run(stages, src, device, seg):
    """Cook `stages` in consecutive groups of `seg` (seg=len -> one fused region;
    seg=1 -> fully unfused). Returns the final tensor.

    Each group is re-based to index 0 — `chain_inputs` are ABSOLUTE stage indices, so a
    slice carried verbatim would point outside itself (the same renumbering trap CACHE-7's
    `suffix_stage_list` has a blocker row for). The group's head reads the previous group's
    output as an ordinary external binding, which is exactly what chaining regions means."""
    cur = src
    for i in range(0, len(stages), seg):
        group = []
        for j, s in enumerate(stages[i:i + seg]):
            st = {"code": s["code"], "bindings": {}}
            if j == 0:
                st["bindings"] = {"A": cur}
            else:
                st["chain_inputs"] = {"A": [j - 1, "OUT"]}
            group.append(st)
        cur = tex_engine.cook_stage_list(group, device=device, precision="fp32")["OUT"]
    return cur


def _time(fn, warmup, runs, device):
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def bench(res, device, counts, warmup, runs):
    torch.manual_seed(34)
    src = torch.rand(1, res, res, 4, device=device)
    rows = []
    for n in counts:
        stages = _stages(n)
        # Correctness first: all three routes must agree, or the timings compare
        # three different computations. fp32, so equality is exact.
        ref = _run(stages, src, device, n)
        seg_out = _run(stages, src, device, CAP)
        unf_out = _run(stages, src, device, 1)
        exact = bool(torch.equal(ref, seg_out) and torch.equal(ref, unf_out))
        maxdiff = max(float((ref - seg_out).abs().max()), float((ref - unf_out).abs().max()))

        row = {
            "n": n, "device": device, "res": res, "exact": exact, "maxdiff": maxdiff,
            "regions_today": 1 if n <= CAP else 0,
            "regions_segmented": math.ceil(n / CAP),
            "fused_ms": _time(lambda: _run(stages, src, device, n), warmup, runs, device),
            "segmented_ms": _time(lambda: _run(stages, src, device, CAP), warmup, runs, device),
            "unfused_ms": _time(lambda: _run(stages, src, device, 1), warmup, runs, device),
        }
        row["segmented_vs_unfused"] = row["unfused_ms"] / row["segmented_ms"]
        row["fused_vs_unfused"] = row["unfused_ms"] / row["fused_ms"]
        rows.append(row)
        print(f"  N={n:<3} {device:<5} fused={row['fused_ms']:8.2f}ms  "
              f"segmented={row['segmented_ms']:8.2f}ms  unfused={row['unfused_ms']:8.2f}ms  "
              f"| seg/unf={row['segmented_vs_unfused']:5.2f}x  "
              f"fused/unf={row['fused_vs_unfused']:5.2f}x  "
              f"{'exact' if exact else f'DIVERGES {maxdiff:.2e}'}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--counts", type=int, nargs="*", default=[16, 17, 24, 50])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--save")
    args = ap.parse_args()

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print(f"FUS-cap spike — cap={CAP}, res={args.res}, runs={args.runs}")
    out = []
    for dev in devices:
        print(f"\n{dev}:")
        out += bench(args.res, dev, args.counts, args.warmup, args.runs)

    print("\n" + "=" * 74)
    print("  VERDICT INPUT — segmented / unfused at N > cap (the cliff's real cost)")
    print("=" * 74)
    for dev in devices:
        past = [r for r in out if r["device"] == dev and r["n"] > CAP]
        if past:
            g = math.exp(sum(math.log(r["segmented_vs_unfused"]) for r in past) / len(past))
            print(f"  {dev:<5} geomean speedup from segmenting: {g:.2f}x  "
                  f"(N={[r['n'] for r in past]})")
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"cap": CAP, "res": args.res, "rows": out}, f, indent=2)
        print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
