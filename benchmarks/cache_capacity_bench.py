#!/usr/bin/env python3
"""CACHE-8 — the three measurements that decide what a deep cache tier should actually be.

The Memory report asks for compressed tiers "chosen per tier by measured Pareto (compression
ratio vs decode latency vs interactive feel — run as a benchmark, not a belief)". This is that
benchmark. It answers three questions and is meant to keep answering them, so the verdict in
docs/compressed-cache-tiers.md can be re-checked rather than believed.

  PARETO      For each candidate representation of a frame: size, ratio, encode ms, DECODE ms,
              reconstruction error. Decode is the number that decides it — it is paid on every
              cache hit, where encode is paid once on eviction. Two reference rows sit beside
              the candidates and are the whole point of the table: what it costs to simply
              WRITE the frame to disk uncompressed, and what it costs to move it VRAM->RAM.
              A codec has to beat the thing it replaces, not merely compress.

  CAPACITY    Frames held, and still servable at memory speed, in a fixed byte budget.

  RESIDENCY   The ladder itself, driven through the real ResultCache: how long a demotion
              takes, how long a promotion takes, and what each replaces.

Usage:
    python benchmarks/cache_capacity_bench.py
    python benchmarks/cache_capacity_bench.py --resolution 4096 --save results/cache8.json
"""
from __future__ import annotations

import argparse
import bz2
import io
import json
import lzma
import os
import statistics
import sys
import tempfile
import time
import zlib
from pathlib import Path

_bench_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_bench_dir.parent.parent))
sys.path.insert(0, str(_bench_dir))

os.environ.setdefault(
    "TEX_CACHE_DIR", str(Path(tempfile.gettempdir()) / "tex_bench_cache"))

import torch                                                    # noqa: E402
from run_benchmarks import system_info                          # noqa: E402
from TEX_Wrangle import tex_packing, tex_results                # noqa: E402

#: Above this the entropy coders take minutes rather than seconds. They are reported as skipped
#: rather than silently dropped — a table with a missing row reads as "not applicable", and
#: "too slow to measure at 4K" is the single most decision-relevant fact about them.
_ENTROPY_MAX_BYTES = 32 << 20


def frame(h: int, w: int, c: int = 4, device: str = "cpu") -> torch.Tensor:
    """Realistic image content: smooth gradients plus fine detail. `torch.randn` would be
    incompressible AND unrepresentative, which would answer a question nobody asked — the
    ratios in this table are only meaningful over data shaped like the data being cached."""
    gy = torch.linspace(0, 1, h, device=device).unsqueeze(1)
    gx = torch.linspace(0, 1, w, device=device).unsqueeze(0)
    img = ((gy + gx) * 0.5 + torch.sin(gx * 61.0) * torch.cos(gy * 47.0) * 0.05).clamp(0, 1)
    return torch.stack([img, img * 0.9 + 0.05, img * 0.8 + 0.1, torch.ones_like(img)],
                       dim=-1)[..., :c].unsqueeze(0).contiguous()


def _ms(fn, reps: int = 3, sync: bool = False, warmup: bool = True):
    """Median wall-ms, AND the last result — so a caller that needs what `fn` produced does not
    have to run it a third time. The entropy rows did exactly that (warmup + timed + a call to
    get the bytes), which was ~12 s of pure waste per run at 2048²."""
    if sync:
        torch.cuda.synchronize()
    out = []
    if warmup:
        fn()
        if sync:
            torch.cuda.synchronize()
    val = None
    for _ in range(reps):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        val = fn()
        if sync:
            torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out), val


def _blob(t: torch.Tensor) -> bytes:
    """A tensor as bytes, torch-only — the numpy bridge is banned package-wide (CI's torch has
    no numpy, and LNT-1 lints for it, this docstring included). `torch.save` into a BytesIO is
    the one route that materialises the whole buffer in a single call; its ~500 B of pickle
    framing is noise against a 126 MB frame."""
    b = io.BytesIO()
    torch.save(t, b)
    return b.getvalue()


def pareto(res: int) -> list:
    t = frame(res, res)
    raw = t.numel() * 4
    rows = []

    def add(name, nbytes, enc, dec, err, kind):
        rows.append({"row": name, "kind": kind, "bytes": nbytes,
                     "ratio": raw / nbytes if nbytes else 0.0,
                     "encode_ms": enc, "decode_ms": dec, "maxerr": err})

    # Width reductions — no entropy coding, so "decode" is a cast.
    for name, dtype in (("fp16", tex_packing.FP16), ("uint16 [0,1]", tex_packing.UINT16)):
        p = tex_packing.pack(t, dtype)
        u = tex_packing.unpack(p, torch.float32)
        add(name, p.numel() * p.element_size(),
            _ms(lambda d=dtype: tex_packing.pack(t, d))[0],
            _ms(lambda q=p: tex_packing.unpack(q, torch.float32))[0],
            float((u - t).abs().max()), "width")

    # Entropy coding over the fp32 and fp16 byte images.
    for sname, sv in (("fp32", t), ("fp16", t.to(torch.float16))):
        data = _blob(sv)
        err = 0.0 if sname == "fp32" else float((sv.to(torch.float32) - t).abs().max())
        for cname, comp, decomp in (
                ("zlib-1", lambda d=data: zlib.compress(d, 1), zlib.decompress),
                ("zlib-6", lambda d=data: zlib.compress(d, 6), zlib.decompress),
                ("bz2-1", lambda d=data: bz2.compress(d, 1), bz2.decompress),
                ("lzma-0", lambda d=data: lzma.compress(d, preset=0), lzma.decompress)):
            label = f"{cname} over {sname}"
            if len(data) > _ENTROPY_MAX_BYTES and cname in ("lzma-0", "bz2-1"):
                rows.append({"row": label, "kind": "entropy", "skipped":
                             f"source {len(data) >> 20} MB > {_ENTROPY_MAX_BYTES >> 20} MB "
                             f"cap — minutes per frame"})
                continue
            # `warmup=False`: one encode, timed, and its bytes reused. Warming a 1-rep
            # measurement of a multi-second compressor buys nothing and doubles the run.
            enc, packed = _ms(comp, reps=1, warmup=False)
            add(label, len(packed), enc,
                _ms(lambda p=packed, d=decomp: d(p), reps=1, warmup=False)[0],
                err, "entropy")

    # THE REFERENCE ROWS. Everything above has to beat one of these to be worth having.
    fd, path = tempfile.mkstemp(suffix=".frame")
    os.close(fd)
    try:
        def wr():
            with open(path, "wb") as f:
                torch.save(t, f)

        def rd():
            with open(path, "rb") as f:
                return torch.load(f, weights_only=True)

        wr()
        add("REFERENCE disk write/read", os.path.getsize(path), _ms(wr, 2)[0], _ms(rd, 2)[0],
            0.0, "reference")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if torch.cuda.is_available():
        g = t.to("cuda")
        torch.cuda.synchronize()

        def d2h():
            h = torch.empty(g.shape, dtype=g.dtype, device="cpu")
            h.copy_(g)

        add("REFERENCE VRAM<->RAM", raw, _ms(d2h, 3, sync=True)[0],
            _ms(lambda: t.to("cuda"), 3, sync=True)[0], 0.0, "reference")
    return rows


def capacity(budget_mb: int) -> list:
    out = []
    for res in (1024, 2048, 4096):
        px = res * res * 4
        for name, esz in (("fp32 (v0.32)", 4), ("fp16 (PREC-1)", 2)):
            out.append({"resolution": res, "storage": name, "budget_mb": budget_mb,
                        "frame_mb": round(px * esz / (1 << 20), 2),
                        "frames_held": (budget_mb << 20) // (px * esz)})
    return out


def residency(res: int, budget_mb: int) -> list:
    """The ladder, driven through the shipped ResultCache rather than a mock — so what is
    measured is the code that runs, including the lock discipline and the drain path."""
    if not torch.cuda.is_available():
        return [{"note": "no CUDA on this box — the residency ladder has only one rung here"}]
    rows = []
    with tempfile.TemporaryDirectory() as d:
        f = frame(res, res, device="cuda")
        per_mb = f.numel() * 4 / (1 << 20)

        # Demote: hold two frames, ceiling of one.
        c = tex_results.ResultCache(cache_dir=d)
        c.set_vram_budget(int(per_mb) + 1)
        c.put("a", f)
        c.put("b", f)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c.put("c", f)                       # pushes the coldest out of VRAM
        torch.cuda.synchronize()
        demote_ms = (time.perf_counter() - t0) * 1e3
        st = c.stats()
        rows.append({"row": "put that triggers a demotion", "ms": demote_ms,
                     "demotions": st["demotions"], "demoted_now": st["demoted"],
                     "vram_mb": round(st["vram_bytes"] / (1 << 20), 1)})

        # Promote: hit the demoted frame.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        got = c.get("a")
        torch.cuda.synchronize()
        rows.append({"row": "get that triggers a promotion", "ms":
                     (time.perf_counter() - t0) * 1e3,
                     "promotions": c.stats()["promotions"],
                     "served_on": str(got.device) if got is not None else None})
        c.clear(disk=True)

        # What it replaces: the same pressure with residency OFF goes to disk.
        c2 = tex_results.ResultCache(cache_dir=d, budget_mb=int(per_mb) + 1)
        c2.put("a", f)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c2.put("b", f)                      # evicts "a" to disk
        torch.cuda.synchronize()
        rows.append({"row": "put that triggers a disk spill (v0.32 path)",
                     "ms": (time.perf_counter() - t0) * 1e3, "spills": c2.spills})
        t0 = time.perf_counter()
        c2.get("a")                         # restores from disk
        torch.cuda.synchronize()
        rows.append({"row": "get that triggers a disk restore (v0.32 path)",
                     "ms": (time.perf_counter() - t0) * 1e3, "restores": c2.restores})
        c2.clear(disk=True)
    return rows


def exit_gate(res: int, budget_mb: int, n_frames: int) -> list:
    """THE v0.33 EXIT GATE: "cache capacity at fixed budget measurably up (target >= 2x frames
    held at 4K under Balanced profile vs v0.32)".

    "Capacity" is measured as **frames still servable from a MEMORY tier** after N distinct
    frames have been stored under one byte budget — not as arithmetic on frame sizes. A frame
    that spilled to disk is still *reachable*, so counting entries would flatter every
    configuration equally; what changed in v0.33 is how many stay at memory speed, and that is
    what a scrubbing user feels.

    Rows, all at the same budget and the same frame count:
      v0.32          fp32 storage, residency off       — what shipped last release
      PREC-1         preview-quality storage           — half the bytes per frame
      PREC-1+CACHE-8 preview + residency armed         — CUDA frames demote instead of spilling
    """
    rows = []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for label, quality, vram_mb in (("v0.32 (fp32, no residency)", None, None),
                                    ("PREC-1 (preview storage)", tex_packing.PREVIEW, None),
                                    ("PREC-1 + CACHE-8 residency", tex_packing.PREVIEW,
                                     max(1, budget_mb // 4))):
        with tempfile.TemporaryDirectory() as d:
            c = tex_results.ResultCache(cache_dir=d, budget_mb=budget_mb)
            if vram_mb is not None:
                c.set_vram_budget(vram_mb)
            for i in range(n_frames):
                c.put(f"f{i}", frame(res, res, device=dev) * (1.0 + i * 1e-3),
                      quality=quality)
            st = c.stats()
            # Served from memory = present in `_ram`. A `get` would also count a disk restore
            # as a hit, which is exactly the distinction this row exists to make.
            in_memory = len(c._ram)
            rows.append({"config": label, "resolution": res, "budget_mb": budget_mb,
                         "frames_put": n_frames, "frames_in_memory": in_memory,
                         "spills": c.spills, "demotions": st["demotions"],
                         "ram_mb": round(st["ram_bytes"] / (1 << 20), 1),
                         "vram_mb": round(st["vram_bytes"] / (1 << 20), 1)})
            c.clear(disk=True)
    base = rows[0]["frames_in_memory"] or 1
    for row in rows:
        row["vs_v032"] = round(row["frames_in_memory"] / base, 2)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=2048)
    ap.add_argument("--budget-mb", type=int, default=1024)
    ap.add_argument("--exit-gate", action="store_true",
                    help="run only the v0.33 exit-gate capacity measurement")
    ap.add_argument("--frames", type=int, default=12,
                    help="distinct frames to store, for --exit-gate")
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    if args.exit_gate:
        gate = exit_gate(args.resolution, args.budget_mb, args.frames)
        print(f"\nv0.33 EXIT GATE — frames still servable from a MEMORY tier\n"
              f"  {args.frames} distinct {args.resolution}^2 frames, one {args.budget_mb} MB "
              f"budget, device={'cuda' if torch.cuda.is_available() else 'cpu'}")
        print("-" * 96)
        print(f"{'config':30s} {'in memory':>10s} {'vs v0.32':>9s} {'spills':>7s} "
              f"{'demotes':>8s} {'ram MB':>8s} {'vram MB':>8s}")
        for g in gate:
            print(f"{g['config']:30s} {g['frames_in_memory']:10d} {g['vs_v032']:8.2f}x "
                  f"{g['spills']:7d} {g['demotions']:8d} {g['ram_mb']:8.1f} {g['vram_mb']:8.1f}")
        if args.save:
            p = Path(args.save)
            if not p.is_absolute():
                p = _bench_dir / p
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"system": system_info(
                "cuda" if torch.cuda.is_available() else "cpu"), "exit_gate": gate}, indent=1),
                encoding="utf-8")
            print(f"\nSaved {p}")
        return 0

    par = pareto(args.resolution)
    cap = capacity(args.budget_mb)
    resi = residency(min(args.resolution, 2048), args.budget_mb)

    print(f"\nCACHE-8 Pareto @ {args.resolution}^2 x4 fp32 "
          f"({args.resolution**2*16/(1<<20):.0f} MB/frame)")
    print("-" * 92)
    print(f"{'row':28s} {'MB':>8s} {'ratio':>7s} {'encode ms':>11s} {'DECODE ms':>11s} "
          f"{'maxerr':>10s}")
    for r in par:
        if "skipped" in r:
            print(f"{r['row']:28s} {'skipped: ' + r['skipped']}")
            continue
        print(f"{r['row']:28s} {r['bytes']/(1<<20):8.2f} {r['ratio']:7.2f} "
              f"{r['encode_ms']:11.1f} {r['decode_ms']:11.1f} {r['maxerr']:10.2e}")

    print(f"\nCapacity at a {args.budget_mb} MB frame budget")
    print("-" * 62)
    for c in cap:
        print(f"  {c['resolution']:5d}^2  {c['storage']:15s} {c['frame_mb']:8.2f} MB/frame  "
              f"-> {c['frames_held']:4d} frames")

    print("\nResidency ladder (through the shipped ResultCache)")
    print("-" * 78)
    for row in resi:
        if "note" in row:
            print(f"  {row['note']}")
        else:
            extra = " ".join(f"{k}={v}" for k, v in row.items() if k not in ("row", "ms"))
            print(f"  {row['row']:44s} {row['ms']:8.1f} ms   {extra}")

    if args.save:
        p = Path(args.save)
        if not p.is_absolute():
            p = _bench_dir / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"system": system_info("cuda" if torch.cuda.is_available()
                                                       else "cpu"),
                                 "pareto": par, "capacity": cap, "residency": resi},
                                indent=1), encoding="utf-8")
        print(f"\nSaved {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
