"""BENCH-47 — representative artist-loop benchmarks, driven through public entry points.

An embedding host's own "compass" (its running commentary on what an artist tick looks like)
describes six loops. This file is one entry point over all six, timed rather than counted —
the counts analogue of scenario (a) already lives in `tests/test_bench2_counts.py`'s
`whole_frame_chain_d*` family (`benchmarks/host_path_counts.py`'s `WholeFrameChainScenario`),
which this file reuses rather than re-deriving.

    (a) whole_frame_tick        whole-frame, node-by-node tick, N=10 stages, D dirty — the
                                host's own per-tick shape (one cook per dirty node, whole
                                frame, cheap nodes uncached). THE yardstick: wave 2's per-cook
                                fixed-overhead cut (TRK-211/TRK-212) is proven against this
                                scenario's counts pin, and this is its timed twin.
    (b) param_drag              a sequence of param edits on one node, edit -> result latency.
                                Thin wrapper over `param_scrub_bench.bench` (ANIM-1's own
                                benchmark) — not re-derived here.
    (c) scrub                   a scrubbed (non-monotonic) time_context on a `tex_provider`-
                                backed source: the playhead jumps, unlike (f)'s steady advance.
    (d) node_insert             a new, never-before-seen node spliced mid-chain: first-result
                                latency including the cold lex/parse/typecheck/compile it
                                forces, against the same chain's steady per-tick cost.
    (e) background_interactive  a paced long cook pre-empted by an edit: drain + edit latency.
                                Thin wrapper over `preempt_drain_bench` (PACE-462's own
                                measurement) — CUDA only, not re-derived here.
    (f) playback                a frame range through a registered `FrameProvider`, steady-
                                state per-frame latency. Reuses `io_playback_bench`'s `_cook`
                                and provider setup (PM-9), timed per frame rather than as one
                                serial/overlapped total.

Every program mix is the same two shapes every scenario in this repo's benches already uses:
cheap per-pixel grades (exposure/gamma/tint-shaped) and blur-heavy stages (`gauss_blur`,
what an embedding host's comps call "comp-shaped"). Metrics, per scenario: time to first
result, p50/p95/p99 of the steady state, and (for (a)) the per-tick structural count context
a reader needs to relate the timing to the pinned counts.

GPU timing is `torch.cuda.synchronize()`-wrapped throughout (invariant #6). This file makes NO
product-code change and asserts nothing; it is a measurement tool, run under a bench lease
(`docs/brief-conventions.md`'s three measurement rules — fresh cache dir, discard the first
leg, name the box beside every figure) and read by a human or folded into a hand-back table.

    python benchmarks/artist_loops_bench.py --device cpu --scenarios whole_frame_tick,param_drag
    python benchmarks/artist_loops_bench.py --device cuda --save results/artist_loops_<box>.json
    python benchmarks/artist_loops_bench.py --scenarios whole_frame_tick --dirty 1,3,5,10
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../TEX_Wrangle
sys.path.insert(0, os.path.dirname(_PKG))           # .../custom_nodes (package parent)

import torch                                        # noqa: E402
from TEX_Wrangle import tex_engine                    # noqa: E402

ALL_SCENARIOS = ("whole_frame_tick", "param_drag", "scrub", "node_insert",
                  "background_interactive", "playback")


# ── shared plumbing ──────────────────────────────────────────────────────────────

def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _percentiles(ms: list) -> dict:
    """p50/p95/p99 (nearest-rank), reusing `compile_modes_bench._pctl` (G6, FIX-GATE) rather
    than a second copy of the same formula -- this file already loads that module's sibling
    `host_path_counts`/`preempt_drain_bench`/`io_playback_bench` by path via `_load()` for
    exactly this reason (R1#3): a "p95" column reported anywhere must mean the same
    arithmetic everywhere it is quoted, not two formulas that happen to agree today."""
    if not ms:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "n": 0}
    xs = sorted(ms)
    cmb = _load("compile_modes_bench", "compile_modes_bench.py")
    return {"p50_ms": round(cmb._pctl(xs, 0.50), 4), "p95_ms": round(cmb._pctl(xs, 0.95), 4),
            "p99_ms": round(cmb._pctl(xs, 0.99), 4),
            "median_ms": round(statistics.median(xs), 4), "n": len(xs)}


def _load(modname: str, filename: str):
    """Load a sibling `benchmarks/*.py` file by path, once per process.

    `benchmarks/` is `.comfyignore`d and is not a package (`tests/helpers.py`'s
    `load_counts_harness` docstring explains why), so a plain `import filename` needs the
    directory on `sys.path` and would still collide on the name if this file and a caller's
    own test both loaded it under the bare module name. Loading by path under a private key
    avoids both — one instance per process, never re-executed, never colliding with a test's
    own `_bench2_host_path_counts` load of the same file."""
    key = f"_artist_loops_{modname}"
    mod = sys.modules.get(key)
    if mod is not None:
        return mod
    path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


# ── (a) whole-frame, node-by-node tick, D of N dirty ─────────────────────────────

def _wfc_scenario_class(hpc, dirty: int):
    """The registered `whole_frame_chain_d{dirty}` class if BENCH-47 pinned it (1, 3, 5, 10
    today), else an ad-hoc subclass of the same `WholeFrameChainScenario` — never a
    reimplementation of its `tick()`, so a D this file measures that the counts file has not
    (yet) pinned still drives the identical `RoiComp.cook(None, stage, use_cache=False)`
    shape the pinned rows describe."""
    name = f"whole_frame_chain_d{dirty}"
    for cls in hpc.SCENARIOS:
        if cls.name == name:
            return cls
    return type(f"WholeFrameChainD{dirty}Adhoc", (hpc.WholeFrameChainScenario,),
                {"name": name, "_DIRTY": dirty})


def bench_whole_frame_tick(res: int, ticks: int, dirties, device: str) -> dict:
    """N=10 (the comp's own `_COMP_STAGES`), D of them dirty per tick, whole-frame
    (`roi=None`), `use_cache=False` — the exact shape `WholeFrameChainScenario` documents.
    `first_result_ms` is `build()`+`prime()`: a fresh comp, warmed (which is where every
    program's first-ever compile is paid) — the cold cost a session pays once. The per-tick
    series that follows is the steady-state edit -> result cost `_DIRTY` stages pay."""
    hpc = _load("host_path_counts", "host_path_counts.py")
    out = {}
    window = min(res, max(1, res // 2))
    for d in dirties:
        cls = _wfc_scenario_class(hpc, d)
        t0 = time.perf_counter()
        scn = cls(res, window, device, ticks=ticks)
        comp = scn.build()
        _sync(device)
        first_result_ms = (time.perf_counter() - t0) * 1000.0
        lat = []
        try:
            for i in range(ticks):
                t0 = time.perf_counter()
                scn.tick(comp, i)
                _sync(device)
                lat.append((time.perf_counter() - t0) * 1000.0)
        finally:
            scn.teardown()
        out[f"d{d}"] = {"n_stages": 10, "dirty": d, "res": res,
                         "first_result_ms": round(first_result_ms, 4),
                         **_percentiles(lat)}
    return out


# ── (b) parameter drag ────────────────────────────────────────────────────────────

def bench_param_drag(res: int, cooks: int, device: str) -> dict:
    """Thin wrapper: `param_scrub_bench.bench` already IS this scenario (ANIM-1's own
    contract benchmark — a $param sweep's cost against a static-param and a full-recook
    control). Reused, not re-derived."""
    m = _load("param_scrub_bench", "param_scrub_bench.py")
    return m.bench(res, cooks, device)


# ── (c) scrub: a non-monotonic time_context on a time-reading source ─────────────

def bench_scrub(res: int, ticks: int, device: str, seed: int = 90) -> dict:
    """A playhead that JUMPS (a scrub, never revisiting the same time twice in a row) against
    (f)'s steady advance — the axis the plan names to keep the two scenarios distinct. Reuses
    `io_playback_bench`'s provider setup and `_cook` (PM-9's own fixtures) rather than
    re-deriving a second `SyntheticFrameProvider` wiring."""
    io = _load("io_playback_bench", "io_playback_bench.py")
    tp = io.tex_provider
    tp.reset_provider()
    p = tp.SyntheticFrameProvider(res=res, rate=1.0, device=device, latency_s=0.0)
    tp.set_provider(p)
    tp.set_media_budget_mb(0.0)          # no pooling: every scrub is a genuine re-fetch
    code = io._code(4)
    rng = random.Random(seed)
    try:
        # warm: compile the grade chain once, outside the counted region.
        io._cook(tp.materialize("plate", 0.0, "fetch"), device, code=code)
        _sync(device)
        lat = []
        prev_t = 0.0
        for _ in range(ticks):
            t = prev_t
            while t == prev_t:                       # a scrub never lands on the same frame
                t = rng.uniform(0.0, 999.0)
            prev_t = t
            t0 = time.perf_counter()
            src = tp.materialize("plate", t, "fetch")
            io._cook(src, device, code=code)
            _sync(device)
            lat.append((time.perf_counter() - t0) * 1000.0)
    finally:
        tp.reset_provider()
        tp.set_media_budget_mb(512.0)
    return {"res": res, "ticks": ticks, **_percentiles(lat)}


# ── (d) node insertion ────────────────────────────────────────────────────────────

_NI_PREFIX = ["@OUT = vec4(@IN.rgb * 1.05, 1.0);",
              "@OUT = vec4(max(@IN.rgb - vec3(0.02), vec3(0.0)), 1.0);",
              "@OUT = gauss_blur(@IN, 1.5);"]
_NI_SUFFIX = ["@OUT = vec4(@IN.rgb + 0.1, 1.0);",
              "@OUT = vec4(clamp(@IN.rgb, vec3(0.0), vec3(1.0)), 1.0);"]


def _ni_chain(codes, first_src, device):
    cur = first_src
    for code in codes:
        r = tex_engine.cook(code, {"IN": cur}, device_mode=device, precision="fp32",
                            compile_mode="none")
        cur = r.outputs["OUT"]
    return cur


def bench_node_insert(res: int, ticks: int, device: str, insert_at: int = 3) -> dict:
    """`insert_at` splits a 5-node chain (`_NI_PREFIX` + `_NI_SUFFIX`) into a warmed prefix and
    suffix. Each insert tick splices a BRAND-NEW node (a fresh source string every tick, never
    seen before — the property that forces the cold lex/parse/typecheck/compile path) between
    them and cooks prefix -> new node -> suffix; `baseline` cooks the unmodified five-node
    chain, already fully warm, for the same tick count — the number the insert is charged
    against."""
    torch.manual_seed(77)
    src = torch.rand(1, res, res, 4, device=device)
    full = _NI_PREFIX + _NI_SUFFIX
    _ni_chain(full, src, device)                     # warm every stage once
    _sync(device)
    baseline = []
    for _ in range(ticks):
        t0 = time.perf_counter()
        _ni_chain(full, src, device)
        _sync(device)
        baseline.append((time.perf_counter() - t0) * 1000.0)
    inserts = []
    for i in range(ticks):
        new_node = (f"@OUT = vec4(mix(@IN.rgb, vec3(luma(@IN)), "
                    f"{0.10 + i * 0.001:.4f}), 1.0);  // BENCH-47 insert {i}\n")
        t0 = time.perf_counter()
        cur = _ni_chain(_NI_PREFIX, src, device)
        cur = tex_engine.cook(new_node, {"IN": cur}, device_mode=device, precision="fp32",
                              compile_mode="none").outputs["OUT"]
        _ni_chain(_NI_SUFFIX, cur, device)
        _sync(device)
        inserts.append((time.perf_counter() - t0) * 1000.0)
    return {"res": res, "insert_at": insert_at,
            "baseline_steady": _percentiles(baseline),
            "insert_first_result": _percentiles(inserts)}


# ── (e) background render + interactive pre-emption ──────────────────────────────

def bench_background_interactive(depths, trials: int, device: str) -> dict:
    """Thin wrapper over `preempt_drain_bench` (PACE-462's own measurement: a heavy paced
    background cook, pre-empted mid-flight by a small interactive one). CUDA only — the
    module hardcodes `device_mode="cuda"` and `.cuda()` bindings, matching the ask's own
    framing ("a paced long cook pre-empted by edits" is a device-timed scenario by
    construction). Skips (returns a note, not a fabricated reading) off CUDA."""
    if device != "cuda" or not torch.cuda.is_available():
        return {"skipped": "CUDA required (preempt_drain_bench hardcodes device_mode='cuda')"}
    m = _load("preempt_drain_bench", "preempt_drain_bench.py")
    full_runtime = m._calibrate()
    return {"cost_unpreempted": m.cost_unpreempted(depths, trials),
            "drain_on_preempt": m.drain_on_preempt(depths, trials, full_runtime),
            "full_runtime_ms": round(full_runtime * 1000.0, 4)}


# ── (f) playback ──────────────────────────────────────────────────────────────────

def bench_playback(res: int, frames: int, device: str) -> dict:
    """A frame range through a registered provider, steady per-frame latency — reuses
    `io_playback_bench`'s `_cook`/provider setup, timed PER FRAME (PM-9's own harness reports
    only the three route totals, not a percentile series)."""
    io = _load("io_playback_bench", "io_playback_bench.py")
    tp = io.tex_provider
    tp.reset_provider()
    p = tp.SyntheticFrameProvider(res=res, rate=1.0, device=device, latency_s=0.0)
    tp.set_provider(p)
    tp.set_media_budget_mb(0.0)
    code = io._code(4)
    try:
        io._cook(tp.materialize("plate", 0.0, "fetch"), device, code=code)   # warm
        _sync(device)
        lat = []
        for i in range(frames):
            t0 = time.perf_counter()
            src = tp.materialize("plate", float(i), "fetch")
            io._cook(src, device, code=code)
            _sync(device)
            lat.append((time.perf_counter() - t0) * 1000.0)
    finally:
        tp.reset_provider()
        tp.set_media_budget_mb(512.0)
    return {"res": res, "frames": frames, **_percentiles(lat)}


# ── driver ─────────────────────────────────────────────────────────────────────────

def run(scenarios, device: str, res: int, ticks: int, dirties, cooks: int, frames: int,
        depths, trials: int) -> dict:
    out = {"device": device, "res": res,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if "whole_frame_tick" in scenarios:
        out["whole_frame_tick"] = bench_whole_frame_tick(res, ticks, dirties, device)
    if "param_drag" in scenarios:
        out["param_drag"] = bench_param_drag(res, cooks, device)
    if "scrub" in scenarios:
        out["scrub"] = bench_scrub(res, ticks, device)
    if "node_insert" in scenarios:
        out["node_insert"] = bench_node_insert(res, ticks, device)
    if "background_interactive" in scenarios:
        out["background_interactive"] = bench_background_interactive(depths, trials, device)
    if "playback" in scenarios:
        out["playback"] = bench_playback(res, frames, device)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=8)
    ap.add_argument("--dirty", default="1,3,5,10", help="comma-separated D values for (a)")
    ap.add_argument("--cooks", type=int, default=300, help="(b)'s cook count")
    ap.add_argument("--frames", type=int, default=60, help="(f)'s frame count")
    ap.add_argument("--depths", default="1,2,4", help="(e)'s pacing depths")
    ap.add_argument("--trials", type=int, default=30, help="(e)'s trial count")
    ap.add_argument("--scenarios", default=",".join(ALL_SCENARIOS),
                    help=f"comma-separated subset of {ALL_SCENARIOS}")
    ap.add_argument("--save")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [s for s in scenarios if s not in ALL_SCENARIOS]
    if unknown:
        print(f"unknown scenario(s): {unknown} — choose from {ALL_SCENARIOS}", file=sys.stderr)
        return 2
    dirties = [int(x) for x in args.dirty.split(",") if x.strip()]
    depths = [int(x) for x in args.depths.split(",") if x.strip()]

    print(f"BENCH-47 artist loops — device={args.device} res={args.res} "
          f"scenarios={scenarios}")
    result = run(scenarios, args.device, args.res, args.ticks, dirties, args.cooks,
                 args.frames, depths, args.trials)
    print(json.dumps(result, indent=2, default=str))

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
