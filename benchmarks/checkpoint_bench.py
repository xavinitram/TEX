#!/usr/bin/env python3
"""CACHE-7 checkpoint benchmark — what a mid-chain edit costs, with and without taps.

The question CACHE-7 exists to answer: on a long fused chain, the user twiddles a param on
stage j. Fusion has spliced the chain into ONE program, so today that recooks stages 0..N —
including everything UPSTREAM of the edit, which cannot have changed. What does a checkpoint
at a cumulative-cost boundary buy, and what does maintaining it cost?

Four rows per (device, chain length, edit position):

  full          the whole fused chain, every tick — today's default, the number to beat
  cache6        the shipped SINGLE-tap suffix splice (`cook_fused_cached`) at the best legal
                cut for this edit: stages 0..k-1 served from the boundary cache, k..N cooked
  cache7        the MULTI-tap path (`cook_checkpointed`), taps placed by measured stage cost
  phase2        what materializing those taps costs once, on idle — CACHE-7's own overhead,
                the number that decides whether the win is real

Run BEFORE any v0.32 change to fix the baseline (`--save results/cache7_baseline.json`),
then again after (`--compare`). CUDA is synced around every timed region (the standing
benchmark discipline — without it these measure kernel-launch time, not work).
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
_pkg_dir = _bench_dir.parent
sys.path.insert(0, str(_pkg_dir.parent))
sys.path.insert(0, str(_bench_dir))

# CACHE-0: never write compiled artifacts into the shipping package's .tex_cache.
os.environ.setdefault(
    "TEX_CACHE_DIR", str(Path(tempfile.gettempdir()) / "tex_bench_cache"))

import torch                                                          # noqa: E402
from run_benchmarks import system_info                                # noqa: E402
from TEX_Wrangle import tex_engine, tex_fusion, tex_results           # noqa: E402
from TEX_Wrangle.tex_runtime import profile as _profile               # noqa: E402

# A chain of real compositing stages with DELIBERATELY uneven cost — the blurs dominate,
# which is the whole premise of placing checkpoints by measured effort rather than by node
# count. Cheap pointwise stages between them are what an even-spacing policy would waste a
# tap on. Every stage reads the chain on @IN and writes @OUT (the linear shape CACHE-6 v1
# covers), and every one is a node a compositor would actually ship.
_STAGE_POOL = [
    ("exposure",   "@OUT = vec4(@IN.rgb * 1.05, 1.0);"),
    ("blackpoint", "@OUT = vec4(max(@IN.rgb - vec3(0.02), vec3(0.0)), 1.0);"),
    ("blur4",      "@OUT = gauss_blur(@IN, 4.0);"),
    ("gamma",      "@OUT = vec4(spow(@IN.rgb, vec3(0.95)), 1.0);"),
    ("saturation", "float y = luma(@IN);\n@OUT = vec4(mix(vec3(y), @IN.rgb, 1.10), 1.0);"),
    ("blur8",      "@OUT = gauss_blur(@IN, 8.0);"),
    ("contrast",   "@OUT = vec4((@IN.rgb - vec3(0.5)) * 1.08 + vec3(0.5), 1.0);"),
    ("tint",       "@OUT = vec4(@IN.rgb * vec3(1.03, 1.0, 0.97), 1.0);"),
]

# The EDITED stage carries a $param, so scrubbing it changes the cook without changing the
# program — the ANIM-1 contract, and the reason a tap upstream of it stays valid.
_EDITED = "@OUT = vec4(@IN.rgb * $knob, 1.0);"


def build_stages(n: int, edit_at: int, src, knob: float) -> list[dict]:
    """An n-stage linear stage list whose stage `edit_at` is the $param-carrying one."""
    out = []
    for i in range(n):
        if i == edit_at:
            code, binds = _EDITED, {"knob": knob}
        else:
            code = _STAGE_POOL[i % len(_STAGE_POOL)][1]
            binds = {}
        # Stage 0 reads the real source on @IN with no chain_input; every later stage chains
        # on @IN, which is what makes this the LINEAR shape CACHE-6/7 cover.
        st = {"code": code, "chain_input": ("IN" if i else None), "bindings": binds}
        if i == 0:
            st["bindings"]["IN"] = src
        out.append(st)
    return out


def _sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _median_ms(fn, reps: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn(-1)
    samples = []
    for i in range(reps):
        t0 = time.perf_counter()
        fn(i)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(samples), 4)


_THRESHOLD_MS = 100.0


def args_threshold() -> float:
    """The placement threshold this run is measuring at (GOV-1 will own it)."""
    return _THRESHOLD_MS


def measure(device: str, res: int, n: int, edit_at: int, reps: int) -> dict:
    src = torch.rand(1, res, res, 3, device=device)
    cache = tex_results.ResultCache()
    up = ("bench-src-v1",)            # the host's content-sensitive source identity (CACHE-1)

    def _stages(i):
        return build_stages(n, edit_at, src, 0.5 + (i % 17) * 0.01)

    # ── row 1: the whole fused chain, every tick (today's default) ───────────────
    def _full(i):
        tex_engine.cook_stage_list(_stages(i), device=device, precision="fp32")
        _sync(device)

    row = {"full": _median_ms(_full, reps)}

    # ── row 2: shipped CACHE-6, single tap at the best legal cut ─────────────────
    # The best single cut for an edit at `edit_at` is exactly `edit_at`: everything below is
    # clean, so the tap caches the largest prefix that survives the scrub.
    k = max(1, min(edit_at, n - 1))

    def _c6(i):
        tex_engine.cook_fused_cached(_stages(i), k, cache, device=device,
                                     precision="fp32", upstream=up)
        _sync(device)

    cache.clear()
    row["cache6"] = _median_ms(_c6, reps)
    row["cut"] = k

    # ── rows 3-4: CACHE-7 multi-tap ──────────────────────────────────────────────
    try:
        from TEX_Wrangle import tex_checkpoint
    except ImportError:
        tex_checkpoint = None                       # pre-implementation baseline run
    if tex_checkpoint is not None:
        try:
            # Placement, from MEASURED per-stage cost — the item's whole thesis. Profile the
            # chain until PROF-1 has settled samples, then let the policy choose the cuts;
            # a hand-picked `cuts=` here would benchmark a benchmark, not the mechanism.
            _profile.reset()
            _profile.enable()
            pkey = _profile.make_key(f"bench-n{n}-{edit_at}", device, "fp32")
            spatial = (1, res, res)
            # Cook until PROF-1's estimate SETTLES. It measures the first 3 cooks of a key and
            # then 1 in 16, so this is ~150 cooks — the honest cost of "effort-based", and a
            # number worth reporting rather than hiding behind a hand-fed cost table. Capped so
            # a profiler change cannot turn the benchmark into an infinite loop.
            warm = 0
            while warm < 400 and not _profile.settled(pkey, spatial,
                                                      need=tex_checkpoint.MIN_SAMPLES):
                with _profile.measure(pkey, spatial, device=device, stages=True):
                    tex_engine.cook_stage_list(_stages(0), device=device, precision="fp32")
                _sync(device)
                warm += 1
            row["cooks_to_settle"] = warm
            costs = _profile.stage_costs(pkey, spatial)
            cuts = tex_checkpoint.plan_checkpoints(
                _stages(0), costs=costs, threshold_ms=args_threshold(), px=res * res,
                settled=_profile.settled(pkey, spatial,
                                         need=tex_checkpoint.MIN_SAMPLES),
                device=device)
            row["stage_costs"] = {str(k): round(v, 3) for k, v in sorted(costs.items())}
            row["taps"] = cuts
            if not cuts:
                row["cache7"] = None                # policy declined — an honest row, not a gap
                row["cache7_why"] = "no cut cleared the threshold + materialization floor"
            else:
                # Phase 2 cost: materialize the taps ONCE, on idle, and time exactly that.
                cache.clear()
                _sync(device)
                t0 = time.perf_counter()
                tex_checkpoint.materialize(_stages(0), cache, device=device,
                                           precision="fp32", upstream=up, cuts=cuts)
                _sync(device)
                row["phase2"] = round((time.perf_counter() - t0) * 1000.0, 4)

                def _c7(i):
                    tex_checkpoint.cook_checkpointed(
                        _stages(i), cache, device=device, precision="fp32",
                        upstream=up, cuts=cuts)
                    _sync(device)

                # PROVE a checkpoint is actually being SERVED before timing anything. A total
                # fallback returns normally and is indistinguishable from "the feature is
                # slower": measured, a populated cache runs 7.5 ms/cook and a wiped one 41.4,
                # and both look like a successful row. Spy on the splice.
                served = []
                _real_suffix = tex_fusion.suffix_stage_list

                def _spy(stages_, k_, boundary_):
                    served.append(k_)
                    return _real_suffix(stages_, k_, boundary_)

                tex_fusion.suffix_stage_list = _spy
                try:
                    _c7(0)
                finally:
                    tex_fusion.suffix_stage_list = _real_suffix
                row["served_from_cut"] = served[-1] if served else None
                if not served:
                    row["cache7"] = None
                    row["cache7_why"] = ("NO CHECKPOINT SERVED — every cut missed and the cook "
                                         "fell back to the whole chain")
                else:
                    row["cache7"] = _median_ms(_c7, reps)

                # ── the row the item actually exists for ────────────────────────────
                # Everything above holds the edit at ONE stage, where `cache6`'s cut is an
                # ORACLE: the bench hands it k = edit_at, i.e. the perfect cut for this exact
                # edit. A real host does not know that, and a grading session does not hold
                # still — the user moves up and down the chain. CACHE-6 has ONE cut, so every
                # move invalidates it and re-materializes a prefix; CACHE-7's taps are
                # edit-agnostic and stand. Measured over a walk across edit positions, which
                # is the comparison that is not rigged for either side.
                positions = [p for p in range(1, n) if p != 0]

                def _scrub_c6(i):
                    p = positions[i % len(positions)]
                    st = build_stages(n, p, src, 0.5 + (i % 17) * 0.01)
                    tex_engine.cook_fused_cached(st, max(1, min(p, n - 1)), c6_cache,
                                                 device=device, precision="fp32", upstream=up)
                    _sync(device)

                def _scrub_c7(i):
                    p = positions[i % len(positions)]
                    st = build_stages(n, p, src, 0.5 + (i % 17) * 0.01)
                    tex_checkpoint.cook_checkpointed(st, cache, device=device,
                                                     precision="fp32", upstream=up, cuts=cuts)
                    _sync(device)

                c6_cache = tex_results.ResultCache()
                row["scrub_cache6"] = _median_ms(_scrub_c6, max(reps, len(positions) * 2))
                row["scrub_cache7"] = _median_ms(_scrub_c7, max(reps, len(positions) * 2))
                if row["scrub_cache6"]:
                    row["scrub_speedup"] = round(row["scrub_cache6"] / row["scrub_cache7"], 3)
        except Exception as exc:                    # a bench must not hide a broken mechanism
            row["cache7_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            _profile.disable()
    else:
        row["cache7"] = None                        # not implemented yet (baseline run)

    for name in ("cache6", "cache7"):
        v = row.get(name)
        if v:
            row[f"{name}_speedup"] = round(row["full"] / v, 3)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--chains", type=str, default="4,8",
                    help="comma-separated chain lengths")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--compare", type=str, default=None)
    ap.add_argument("--threshold-ms", type=float, default=100.0,
                    help="CACHE-7 placement threshold (GOV-1 profile knob)")
    args = ap.parse_args()
    global _THRESHOLD_MS
    _THRESHOLD_MS = args.threshold_ms

    devices = ["cpu"]
    if torch.cuda.is_available() and not args.cpu_only:
        devices.append("cuda")

    # `system_info` stamps tex_version / git_commit / git_dirty as well as the platform and
    # torch build. A saved baseline without a commit is a number nobody can attribute later —
    # and attribution is the whole point of the save/--compare workflow this bench documents.
    out = {"meta": {**system_info("matrix"),
                    "resolution": args.resolution, "reps": args.reps,
                    "threshold_ms": args.threshold_ms},
           "rows": {}}

    for device in devices:
        for n in [int(x) for x in args.chains.split(",")]:
            # Edit LATE in the chain (the interactive case CACHE-6/7 target: the user is
            # twiddling the node they just added) and MID-chain (PM-8's case, where a big
            # clean prefix sits above the edit).
            for label, edit_at in (("late", n - 1), ("mid", n // 2)):
                key = f"{device}/n{n}/{label}"
                r = measure(device, args.resolution, n, edit_at, args.reps)
                out["rows"][key] = r
                c7 = r.get("cache7")
                c7s = (f"{c7:8.3f}ms ({r.get('cache7_speedup')}x, taps={r.get('taps')})"
                       if c7 else f"-- ({r.get('cache7_why') or r.get('cache7_error')})")
                print(f"{key:20s} full={r['full']:8.3f}  "
                      f"c6={r['cache6']:8.3f} ({r.get('cache6_speedup')}x oracle cut={r['cut']})  "
                      f"c7={c7s}", flush=True)
                if r.get("served_from_cut") is not None:
                    print(f"{'':20s}   served from cut {r['served_from_cut']}",
                          flush=True)
                if r.get("scrub_cache7"):
                    print(f"{'':20s}   SCRUB across edit positions: "
                          f"c6={r['scrub_cache6']:8.3f}  c7={r['scrub_cache7']:8.3f}  "
                          f"({r.get('scrub_speedup')}x)   [settled after "
                          f"{r.get('cooks_to_settle')} cooks]", flush=True)

    if args.save:
        p = Path(args.save)
        if not p.is_absolute():
            p = _bench_dir / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nsaved -> {p}")

    if args.compare:
        base = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print("\n=== vs baseline ===")
        for key, r in out["rows"].items():
            b = base["rows"].get(key)
            if not b:
                continue
            for col in ("full", "cache6", "cache7"):
                if r.get(col) and b.get(col):
                    print(f"  {key:22s} {col:8s} {b[col]:9.3f} -> {r[col]:9.3f} "
                          f"({b[col] / r[col]:.3f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
