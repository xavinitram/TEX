#!/usr/bin/env python3
"""PREC-1 — what does it actually cost to STORE a cached preview frame at half precision?

The recorded rejection this item reopens is unqualified: *"Whole-pipeline fp16 & bf16 IMAGE —
accuracy (bf16 err > the 8-bit quantum)"*. That verdict is about **compute and wire dtype** and
it stands. PREC-1 asks a narrower question the register explicitly left open: may a frame that
has ALREADY been cooked in fp32 be *stored* in the interaction-tier cache at half, and upcast
on the way out?

The decision needs three numbers, and this harness measures all three on REAL cooked frames
(not synthetic noise — random data is both incompressible and unrepresentative of the smooth
gradients an image cache actually holds):

  FIDELITY   The metric is NOT max abs error in the abstract. A preview frame's consumer is a
             viewer that quantizes to 8 bits, so the decision metric is `q8_flips`: the
             fraction of pixels whose `round(clamp(x,0,1)*255)` value CHANGES after a storage
             round-trip. A representation whose error is invisible after 8-bit quantization is
             invisible, full stop. maxdiff/meandiff are reported beside it as the honest
             continuous view (an HDR consumer reads those, not q8).

  CAPACITY   Frames held in a fixed byte budget. This is the exit gate's currency.

  COST       The cast each way. `put` pays a downcast; `get` pays an upcast — but `get`
             ALREADY pays a full-frame `clone()` for copy-on-read, and `.to(fp32)` IS a copy,
             so the upcast may be free in the only place it would be felt. Measured, not
             assumed.

CANDIDATES, and why each is here:
  fp32     the control (what ships today).
  fp16     the proposal. 10-bit mantissa; ULP at [0.5,1) is 2^-11 = 4.9e-4, i.e. 8x under the
           8-bit quantum of 1/255 = 3.9e-3.
  bf16     THE NEGATIVE CONTROL. 7-bit mantissa; ULP at [0.5,1) is 2^-8 = 3.9e-3 = exactly the
           8-bit quantum. If this harness is measuring anything real, bf16 must FAIL where fp16
           passes — reproducing the recorded rejection rather than taking it on faith.
  uint16   fixed-point over [0,1]. Same 2 bytes; uniform 1/65535 = 1.5e-5 resolution, so it
           beats fp16 inside the unit interval — and CLIPS outside it, which is the whole
           reason it cannot be the default for a compositor that carries HDR values.

Usage:
    python benchmarks/storage_precision_bench.py
    python benchmarks/storage_precision_bench.py --resolution 2048 --save results/prec1.json
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

import torch                                              # noqa: E402
from run_benchmarks import system_info                    # noqa: E402
from TEX_Wrangle import tex_engine                        # noqa: E402

# Real compositing outputs, chosen to span the value regimes a frame cache actually sees.
# `hdr_glow` and `hdr_exposure` deliberately leave [0,1] — a compositor's working space is
# scene-linear, and a candidate that only works on clamped display values must be seen to
# fail there rather than be measured on a corpus that hides it.
_PROGRAMS = [
    ("grade",        "@OUT = vec4(clamp(@IN.rgb * 1.08 - vec3(0.02), vec3(0.0), vec3(1.0)), 1.0);"),
    ("saturation",   "float y = luma(@IN);\n@OUT = vec4(mix(vec3(y), @IN.rgb, 1.35), 1.0);"),
    ("blur",         "@OUT = gauss_blur(@IN, 3.0);"),
    ("gradient",     "@OUT = vec4(u, v, u * v, 1.0);"),
    ("dark",         "@OUT = vec4(@IN.rgb * 0.02, 1.0);"),
    ("hdr_glow",     "@OUT = vec4(@IN.rgb + 3.0 * gauss_blur(@IN, 6.0).rgb, 1.0);"),
    ("hdr_exposure", "@OUT = vec4(@IN.rgb * 8.0, 1.0);"),
]

_Q8 = 255.0


def _source(res: int, device: str) -> torch.Tensor:
    """A smooth-gradient-plus-detail source. NOT torch.rand: white noise has no spatial
    correlation, so it flatters nothing and misleads every ratio a cache cares about."""
    g = torch.linspace(0.0, 1.0, res, device=device)
    y, x = torch.meshgrid(g, g, indexing="ij")
    base = torch.stack([x, y, (x + y) * 0.5], dim=-1).unsqueeze(0)
    detail = torch.sin(x * 37.0).unsqueeze(-1).unsqueeze(0) * 0.03
    return (base + detail).clamp(0.0, 1.0).contiguous()


def _cook(code: str, src: torch.Tensor, device: str) -> torch.Tensor:
    res = tex_engine.cook(code, {"IN": src}, device_mode=device, precision="fp32")
    out = res.outputs.get("OUT")
    return out if isinstance(out, torch.Tensor) else src


# ── the storage candidates ────────────────────────────────────────────────────
# Each is (name, pack(fp32)->stored, unpack(stored)->fp32, bytes_per_element).

def _pack_uint16(t):
    return (t.clamp(0.0, 1.0) * 65535.0 + 0.5).to(torch.int32).clamp(0, 65535).to(torch.uint16)


def _unpack_uint16(t):
    return t.to(torch.float32) / 65535.0


_CANDIDATES = [
    ("fp32",   lambda t: t,                     lambda t: t,                   4),
    ("fp16",   lambda t: t.to(torch.float16),   lambda t: t.to(torch.float32), 2),
    ("bf16",   lambda t: t.to(torch.bfloat16),  lambda t: t.to(torch.float32), 2),
    ("uint16", _pack_uint16,                    _unpack_uint16,                2),
]


def _q8(t: torch.Tensor) -> torch.Tensor:
    """The 8-bit code a viewer would display. The decision metric's unit."""
    return (t.clamp(0.0, 1.0) * _Q8).round().to(torch.int16)


def _time_ms(fn, *, device: str, reps: int = 7) -> float:
    """Median wall-ms. Synchronizes around every CUDA region — without it this measures
    kernel-launch overhead and nothing else (the standing benchmark rule)."""
    cuda = device.startswith("cuda")
    fn()
    if cuda:
        torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out)


def measure(res: int, device: str) -> list:
    src = _source(res, device)
    rows = []
    for name, code in _PROGRAMS:
        truth = _cook(code, src, device).float().contiguous()
        lo, hi = float(truth.min()), float(truth.max())
        t8 = _q8(truth)
        n = truth.numel()
        for cand, pack, unpack, esz in _CANDIDATES:
            stored = pack(truth)
            back = unpack(stored)
            diff = (back.float() - truth).abs()
            flips = int((_q8(back) != t8).sum())
            rows.append({
                "program": name, "device": device, "resolution": res,
                "candidate": cand,
                "range": [round(lo, 4), round(hi, 4)],
                "bytes": n * esz,
                "ratio": round(4.0 / esz, 3),
                "maxdiff": float(diff.max()),
                "meandiff": float(diff.mean()),
                "q8_flip_frac": flips / n,
                "pack_ms": _time_ms(lambda: pack(truth), device=device),
                "unpack_ms": _time_ms(lambda: unpack(stored), device=device),
                # What `get` pays TODAY on this frame: a full-frame clone for copy-on-read.
                # The upcast replaces it rather than adding to it, so this is the number the
                # unpack cost must be read against.
                "clone_ms": _time_ms(lambda: truth.clone(), device=device),
            })
    return rows


def capacity(res: int, device: str, budget_mb: int) -> list:
    """Frames a fixed budget holds at each storage width — the exit gate's currency."""
    frame_px = res * res * 4
    out = []
    for cand, _pack, _unpack, esz in _CANDIDATES:
        per = frame_px * esz
        out.append({"candidate": cand, "resolution": res, "device": device,
                    "budget_mb": budget_mb, "frame_mb": round(per / (1 << 20), 2),
                    "frames_held": (budget_mb << 20) // per})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--budget-mb", type=int, default=1024)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    devices = ["cpu"]
    if not args.cpu_only and torch.cuda.is_available():
        devices.append("cuda")

    rows, caps = [], []
    for dev in devices:
        rows += measure(args.resolution, dev)
        caps += capacity(args.resolution, dev, args.budget_mb)
        caps += capacity(4096, dev, args.budget_mb)

    print(f"\nPREC-1 storage fidelity @ {args.resolution}^2  "
          f"(q8_flip = fraction of pixels whose 8-bit display code CHANGES)")
    print("-" * 108)
    print(f"{'device':6s} {'program':13s} {'cand':7s} {'range':>16s} {'maxdiff':>10s} "
          f"{'meandiff':>10s} {'q8_flip':>10s} {'pack':>7s} {'unpack':>7s} {'clone':>7s}")
    for r in rows:
        if r["candidate"] == "fp32":
            continue                       # the control: zero by construction
        print(f"{r['device']:6s} {r['program']:13s} {r['candidate']:7s} "
              f"{str(r['range']):>16s} {r['maxdiff']:10.3e} {r['meandiff']:10.3e} "
              f"{r['q8_flip_frac']:10.3e} {r['pack_ms']:7.2f} {r['unpack_ms']:7.2f} "
              f"{r['clone_ms']:7.2f}")

    print(f"\nCapacity at a {args.budget_mb} MB frame budget")
    print("-" * 62)
    print(f"{'device':6s} {'res':>6s} {'cand':7s} {'frame_mb':>10s} {'frames_held':>12s}")
    for c in caps:
        print(f"{c['device']:6s} {c['resolution']:6d} {c['candidate']:7s} "
              f"{c['frame_mb']:10.2f} {c['frames_held']:12d}")

    # The negative control has to actually fire, or the fidelity column proves nothing.
    ldr = [r for r in rows if r["program"] in ("grade", "saturation", "blur", "gradient")]
    f16 = max((r["q8_flip_frac"] for r in ldr if r["candidate"] == "fp16"), default=0.0)
    bf16 = max((r["q8_flip_frac"] for r in ldr if r["candidate"] == "bf16"), default=0.0)
    print(f"\nnegative control: worst LDR q8_flip  fp16={f16:.3e}  bf16={bf16:.3e}  "
          f"-> bf16 is {(bf16 / f16) if f16 else float('inf'):.1f}x worse")

    if args.save:
        p = Path(args.save)
        if not p.is_absolute():
            p = _bench_dir / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"system": system_info(devices[-1]), "fidelity": rows,
                                 "capacity": caps}, indent=1), encoding="utf-8")
        print(f"\nSaved {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
