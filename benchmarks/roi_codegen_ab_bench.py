#!/usr/bin/env python3
"""
ROI codegen A/B benchmark (TRK-133)
====================================
Codegen-routed ROI cook vs interpreter-routed ROI cook, at the shape
`docs/roi-spatial-laziness.md` records: `TEX_ROI_CODEGEN` off (interpreter) vs on
(codegen), same ROI window, same program. Modelled on the method a prior lane used to
re-measure that figure on a second box (`measure_codegen_roi.py`, kept outside this
repository in that lane's own worklog) — this commits the METHOD, not that lane's
machine: no local path, box name or embedding-host directory is hard-coded here, so a
third box (or a fourth) starts from this script instead of reconstructing the harness
from prose the way that lane had to (no committed script toggled `TEX_ROI_CODEGEN` at
all before this one).

Method, per `docs/brief-conventions.md`'s three measurement rules plus two more this
family of comparisons has needed every time it was run by hand:
  1. discard the first leg (per-flag warm-up) — each flag gets its OWN `TEX_CACHE_DIR`,
     so the second flag's timing is never reading what the first flag's cook compiled.
  2. import TEX_Wrangle from the WORKTREE's parent, never an installed tree — the same
     `sys.path` shape `roi_scrub_bench.py` already uses.
  3. every printed and saved figure names the box it was taken on (from `torch`, at
     runtime — never a literal).
  4. INTERLEAVE the two flags round-by-round rather than running two sequential
     blocks, so neither flag gets a systematic first-in-round advantage from cache
     warmth or clock ramp.
  5. take a NULL CONTROL: split each flag's own timed samples into two disjoint halves
     (by round parity) and compare them against each other. A ratio smaller than this
     null spread is not a measured difference — it is the box's own noise floor.

The program is the ROI-executable shape `roi_scrub_bench.py` already uses: pointwise +
one INLINE `gauss_blur` (so it is analysed as halo footprint, not blocked) reading a
`$param`, so it is representative of a real comp knob. The blur MUST stay inline —
routing it through a named local makes the program non-ROI-executable in v1 (the name
boundary blocks reach composition) and this benchmark would silently measure
whole-frame cooks instead; `_roi_path_taken` is the guard that catches exactly that.

Usage
-----
    python benchmarks/roi_codegen_ab_bench.py
    python benchmarks/roi_codegen_ab_bench.py --rounds 41 --device cuda
    python benchmarks/roi_codegen_ab_bench.py --save results/roi_codegen_ab.json --tag sitting1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                                # .../TEX_Wrangle
sys.path.insert(0, os.path.dirname(_PKG))                    # .../custom_nodes (package parent)

import torch                                                  # noqa: E402
from TEX_Wrangle import tex_engine, tex_roi                    # noqa: E402

# Same ROI-executable program `roi_scrub_bench.py` uses (pointwise + inline
# `gauss_blur`, scrubbing `$amount`) — kept inline deliberately, see the module
# docstring. `docs/roi-spatial-laziness.md`'s figure used this shape.
_CODE = "@OUT = vec4(mix(@A.rgb, gauss_blur(@A, 2.0).rgb, $amount), 1.0);\n"

#: (roi_side, resolution) pairs — the shape `docs/roi-spatial-laziness.md` records
#: (256²-of-1024², a small scrubbed viewport; 1024²-of-2048², a half-linear window).
SHAPES = [(256, 1024), (1024, 2048)]


def _box_name() -> str:
    """The box this process is running on, derived at runtime — never a literal, so
    this script never carries one machine's name into a tracked file."""
    if torch.cuda.is_available():
        cap = "sm_" + "".join(map(str, torch.cuda.get_device_capability(0)))
        return f"{torch.cuda.get_device_name(0)} ({cap})"
    return f"CPU ({os.cpu_count()} logical cores)"


def _cache_root(base: str | None) -> str:
    return base or os.path.join(tempfile.gettempdir(), "tex_roi_codegen_ab_cache")


def _set_cache_dir(cache_root: str, tag: str) -> tuple:
    d = os.path.join(cache_root, f"cache_{tag}")
    os.makedirs(d, exist_ok=True)
    started_empty = len(os.listdir(d)) == 0
    os.environ["TEX_CACHE_DIR"] = d
    return d, started_empty


def _timed_ms(fn, device: str) -> float:
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _null_spread(xs: list) -> tuple:
    """Split-half self-comparison: same flag, two disjoint halves, against each
    other. The spread this reports is the noise floor a real effect must clear."""
    half = len(xs) // 2
    a, b = xs[:half], xs[half:]
    ma, mb = statistics.median(a), statistics.median(b)
    return ma, mb, (ma / mb if mb else float("nan"))


def run_shape(roi_side: int, res: int, device: str, cache_root: str, tag: str,
             n_rounds: int) -> dict:
    torch.manual_seed(11)
    A = torch.rand(1, res, res, 4, device=device)
    x0 = y0 = max(0, (res - roi_side) // 2)
    roi = (x0, y0, roi_side, roi_side, res, res)

    # Separate cache dirs per flag (measurement rule 1); tag-qualified so each
    # independent sitting (a fresh process invocation) gets a genuinely empty pair,
    # never one left warm by a prior sitting.
    dir0, empty0 = _set_cache_dir(cache_root, f"{tag}_{res}_{roi_side}_interp")
    dir1, empty1 = _set_cache_dir(cache_root, f"{tag}_{res}_{roi_side}_codegen")

    def cook(flag: str, amount: float):
        os.environ["TEX_ROI_CODEGEN"] = flag
        os.environ["TEX_CACHE_DIR"] = dir0 if flag == "0" else dir1
        tex_roi.clear_roi_memo()
        res_ = tex_engine.cook(_CODE, {"A": A, "amount": amount}, device_mode=device,
                                precision="fp32", roi=roi, roi_exec=True)
        if res_.cooked_roi != roi:
            raise RuntimeError(f"flag={flag} did not take the ROI path: "
                                f"{res_.cooked_roi} != {roi}")
        return res_

    # Warm-up leg per flag — DISCARDED (measurement rule 1).
    cook("0", 0.5)
    cook("1", 0.5)

    samples = {"0": [], "1": []}
    for i in range(n_rounds):
        amount = 0.3 + (i % 7) * 0.05          # a moving-but-deterministic knob value
        # INTERLEAVE: alternate which flag goes first each round so neither leg gets
        # a systematic first-in-round advantage.
        order = ("0", "1") if i % 2 == 0 else ("1", "0")
        for flag in order:
            ms = _timed_ms(lambda flag=flag, amount=amount: cook(flag, amount), device)
            if i == 0:
                continue                        # the whole round is the discarded warm-up
            samples[flag].append(ms)

    m0, m1 = statistics.median(samples["0"]), statistics.median(samples["1"])
    n0a, n0b, n0r = _null_spread(samples["0"])
    n1a, n1b, n1r = _null_spread(samples["1"])

    return {
        "device": device, "roi": roi_side, "resolution": res,
        "cache_dir_interp": dir0, "cache_dir_interp_started_empty": empty0,
        "cache_dir_codegen": dir1, "cache_dir_codegen_started_empty": empty1,
        "n_timed_rounds": n_rounds - 1,
        "interp_ms_median": round(m0, 4),
        "codegen_ms_median": round(m1, 4),
        "codegen_vs_interp": round(m1 / m0, 4) if m0 else None,   # >1 = codegen slower
        "interp_vs_codegen_speedup": round(m0 / m1, 4) if m1 else None,
        "null_interp_ratio": round(n0r, 4), "null_codegen_ratio": round(n1r, 4),
        "interp_samples": [round(x, 4) for x in samples["0"]],
        "codegen_samples": [round(x, 4) for x in samples["1"]],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TEX ROI codegen-vs-interpreter A/B benchmark")
    ap.add_argument("--device", default=None, help="cpu|cuda (default: cuda if available)")
    ap.add_argument("--rounds", type=int, default=21,
                    help="rounds per shape; round 0 is the discarded warm-up (default 21)")
    ap.add_argument("--tag", default="default",
                    help="sitting label — qualifies cache dirs and the saved filename")
    ap.add_argument("--cache-root", default=None,
                    help="parent dir for this run's per-flag cache dirs "
                         "(default: a subfolder under the system temp dir)")
    ap.add_argument("--save", default=None, help="write the sitting's numbers as JSON")
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    box = _box_name()
    cache_root = _cache_root(args.cache_root)
    print(f"BOX: {box}  device={device}  torch={torch.__version__}")

    out = {"box": box, "device": device, "shapes": []}
    for roi_side, res in SHAPES:
        print(f"\n=== {roi_side}^2-of-{res}^2  (box: {box}) ===")
        row = run_shape(roi_side, res, device, cache_root, args.tag, args.rounds)
        out["shapes"].append(row)
        print(f"  interp  median  {row['interp_ms_median']:9.4f} ms  "
              f"(null split-half ratio {row['null_interp_ratio']:.3f})")
        print(f"  codegen median  {row['codegen_ms_median']:9.4f} ms  "
              f"(null split-half ratio {row['null_codegen_ratio']:.3f})")
        if row["codegen_vs_interp"] is not None:
            print(f"  codegen/interp  {row['codegen_vs_interp']:.3f}x   "
                  f"(interp/codegen = {row['interp_vs_codegen_speedup']:.3f}x)")
            null_spread = max(abs(row["null_interp_ratio"] - 1.0),
                              abs(row["null_codegen_ratio"] - 1.0))
            real_effect = abs(row["codegen_vs_interp"] - 1.0)
            verdict = "ABOVE null spread (real)" if real_effect > null_spread else \
                      "WITHIN null spread (not distinguishable from noise)"
            print(f"  null spread (|ratio-1|) = {null_spread:.3f}  vs  effect = "
                  f"{real_effect:.3f}  (box: {box})  ->  {verdict}")

    if args.save:
        path = args.save if os.path.isabs(args.save) else os.path.join(_HERE, args.save)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved {path}  (box: {box})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
