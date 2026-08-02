"""ROTO-lang decision spike (doc 42 §2.5 / doc 40 §4.6) — do fusable procedural masks beat
host-rasterized MASK planes?

The gate ROTO-lang has carried for four releases: ship `sdf_bezier`/`spline_mask` as stdlib
ONLY if evaluating a mask per-pixel inside a TEX program is faster than the host rasterizing
it once and wiring it in as an ordinary MASK binding. Either verdict closes the item (doc 40
§4.6's own rule); the measurement is the deliverable.

Two routes, same pixels, at 1080p and 4K on both devices:

  procedural   ONE TEX cook that evaluates an N-segment polygon SDF per pixel and applies it.
               This is what `spline_mask` would compile to — the winding/distance work is a
               bounded loop over segments, which is the cost shape that matters, so a
               hand-written TEX program is a faithful stand-in for the stdlib function that
               does not exist yet.
  rasterized   the host builds the same mask ONCE with torch ops (what a real host's
               rasterizer does — a CPU/GPU scanline or an SDF evaluated outside the cook),
               then a TEX cook applies it as a MASK binding.

The honest framing, stated before the numbers: the rasterized route pays its mask cost ONCE
per parameter change and the procedural route pays it EVERY cook. So the interesting axis is
not one frame — it is what happens while the user is dragging something that is NOT the roto
shape, when the rasterized mask is a cache hit and the procedural one is recomputed. Both are
measured.

    python benchmarks/roto_spike.py
    python benchmarks/roto_spike.py --save results/roto_spike.json
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

#: A closed 8-segment polygon, the shape a roto user actually draws. Points are baked as
#: literals because `spline_mask` would take them as an ARRAY wire (DATA-3) and array indexing
#: in the interpreter is not what this spike is measuring.
_PTS = [(0.30, 0.25), (0.55, 0.20), (0.75, 0.35), (0.80, 0.60),
        (0.65, 0.78), (0.42, 0.80), (0.27, 0.66), (0.24, 0.42)]


def _sdf_expr() -> str:
    """Unrolled per-segment distance + a winding sign, in TEX. `spline_mask` would emit the
    same arithmetic from a bounded loop; unrolled keeps the interpreter's loop overhead out of
    a number that is meant to be about the DISTANCE work."""
    lines = ["vec2 p = vec2(u, v);", "float dmin = 1e9;", "float wind = 0.0;"]
    n = len(_PTS)
    for i in range(n):
        ax, ay = _PTS[i]
        bx, by = _PTS[(i + 1) % n]
        lines += [
            f"vec2 a{i} = vec2({ax:.4f}, {ay:.4f});",
            f"vec2 b{i} = vec2({bx:.4f}, {by:.4f});",
            f"vec2 e{i} = b{i} - a{i};",
            f"vec2 w{i} = p - a{i};",
            f"float t{i} = clamp(dot(w{i}, e{i}) / max(dot(e{i}, e{i}), 1e-9), 0.0, 1.0);",
            f"float d{i} = length(w{i} - e{i} * t{i});",
            f"dmin = min(dmin, d{i});",
            # winding: count upward crossings of the horizontal ray
            f"float c{i} = (((a{i}.y > p.y) != (b{i}.y > p.y)) && "
            f"(p.x < a{i}.x + (p.y - a{i}.y) * e{i}.x / (e{i}.y + 1e-9))) ? 1.0 : 0.0;",
            f"wind = wind + c{i};",
        ]
    lines += [
        # A HARD-EDGED mask, matching what `_rasterize` returns, so the two routes produce the
        # same pixels and the comparison is about cost alone. An earlier cut wrote this as
        # `inside*(1-smoothstep(..)) + inside*smoothstep(..)`, which is algebraically `inside`
        # — so it computed two smoothsteps per pixel to reach the same value and inflated the
        # procedural column against the route that loses. `$feather` went with it: a feather
        # the rasterized side does not apply is not a term this comparison may charge for.
        "float m = mod(wind, 2.0) > 0.5 ? 1.0 : 0.0;",
        "@OUT = vec4(@A.rgb * m, 1.0);",
    ]
    return "\n".join(lines)


#: The same comp, but the mask arrives as an ordinary MASK binding the host built.
_APPLY = "@OUT = vec4(@A.rgb * @M, 1.0);"


def _rasterize(res: int, device: str) -> torch.Tensor:
    """What a host rasterizer produces: the same polygon SDF, evaluated once with torch ops
    outside any cook. Vectorised — a real host would use a scanline fill or its own SDF, and
    either way it is plain tensor work on the host's own schedule."""
    ys = torch.linspace(0, 1, res, device=device).view(res, 1).expand(res, res)
    xs = torch.linspace(0, 1, res, device=device).view(1, res).expand(res, res)
    dmin = torch.full((res, res), 1e9, device=device)
    wind = torch.zeros((res, res), device=device)
    n = len(_PTS)
    for i in range(n):
        ax, ay = _PTS[i]
        bx, by = _PTS[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        wx, wy = xs - ax, ys - ay
        t = ((wx * ex + wy * ey) / max(ex * ex + ey * ey, 1e-9)).clamp(0, 1)
        dmin = torch.minimum(dmin, ((wx - ex * t) ** 2 + (wy - ey * t) ** 2).sqrt())
        cross = ((ay > ys) != (by > ys)) & (xs < ax + (ys - ay) * ex / (ey + 1e-9))
        wind = wind + cross.float()
    inside = (torch.remainder(wind, 2.0) > 0.5).float()
    return inside.view(1, res, res)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


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


def bench(res, device, warmup, runs):
    torch.manual_seed(35)
    A = torch.rand(1, res, res, 4, device=device)
    proc_code = _sdf_expr()

    def procedural():
        tex_engine.cook(proc_code, {"A": A}, device_mode=device, precision="fp32")

    mask = _rasterize(res, device)          # built once, as a host would

    def applied():
        tex_engine.cook(_APPLY, {"A": A, "M": mask}, device_mode=device, precision="fp32")

    row = {
        "res": res, "device": device,
        "procedural_ms": _time(procedural, warmup, runs, device),
        "raster_once_ms": _time(lambda: _rasterize(res, device), warmup, runs, device),
        "apply_ms": _time(applied, warmup, runs, device),
    }
    # A scrub of some OTHER parameter: the rasterized mask is reused (so the frame costs only
    # `apply`), the procedural one is recomputed every frame. This is the axis the decision
    # actually turns on.
    row["scrub_ratio"] = row["procedural_ms"] / max(row["apply_ms"], 1e-9)
    # A first frame, where the host must rasterize before it can apply.
    row["rasterized_first_ms"] = row["raster_once_ms"] + row["apply_ms"]
    row["first_ratio"] = row["procedural_ms"] / max(row["rasterized_first_ms"], 1e-9)
    print(f"  {res:>5}² {device:<5} procedural={row['procedural_ms']:8.2f}ms  "
          f"raster={row['raster_once_ms']:7.2f}ms  apply={row['apply_ms']:7.2f}ms  "
          f"| first={row['first_ratio']:5.2f}x  scrub={row['scrub_ratio']:6.2f}x")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="*", default=[1080, 2160])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--save")
    args = ap.parse_args()

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print(f"ROTO spike — {len(_PTS)}-segment polygon SDF; "
          f"ratios are procedural/rasterized (>1 means procedural is SLOWER)")
    rows = []
    for dev in devices:
        print(f"\n{dev}:")
        for res in args.res:
            rows.append(bench(res, dev, args.warmup, args.runs))

    print("\n" + "=" * 78)
    print("  VERDICT INPUT")
    print("=" * 78)
    for dev in devices:
        rs = [r for r in rows if r["device"] == dev]
        if rs:
            g1 = math.exp(sum(math.log(r["first_ratio"]) for r in rs) / len(rs))
            g2 = math.exp(sum(math.log(r["scrub_ratio"]) for r in rs) / len(rs))
            print(f"  {dev:<5} first frame: {g1:5.2f}x   unrelated-scrub: {g2:6.2f}x "
                  f"(procedural / rasterized)")
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"points": len(_PTS), "rows": rows}, f, indent=2)
        print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
