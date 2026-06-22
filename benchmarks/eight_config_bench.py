#!/usr/bin/env python3
"""
TEX 8-Configuration Benchmark
=============================
Measures cook time across the full matrix:

    device {cpu, cuda}  x  compile {off, on}  x  cache {cold, warm}  = 8 configs

  - compile OFF = tree-walking interpreter
  - compile ON  = execute_compiled() (torch.compile, falls back to interpreter)
  - cold        = first run, includes (re)compilation
  - warm        = subsequent run, caches primed

Key differences vs the older bench scripts:
  * Proper torch.cuda.synchronize() around every GPU timing region - without
    this, GPU numbers measure only kernel-launch overhead, not execution.
  * compile-ON runs in a per-program subprocess (crash/TLS-corruption safe) and
    reports an honest STATUS: compiled | fallback | plain | error.
  * Incremental JSON save after every measurement (resumable / crash-proof).
  * --compare baseline.json => per-config geomean speedup + regression flags.

Usage:
    python benchmarks/eight_config_bench.py --save results/baseline.json
    python benchmarks/eight_config_bench.py --resolution 1024 --limit 4   # quick validate
    python benchmarks/eight_config_bench.py --save results/after.json --compare results/baseline.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

_bench_dir = Path(__file__).resolve().parent
_pkg_dir = _bench_dir.parent
_custom_nodes_dir = _pkg_dir.parent
sys.path.insert(0, str(_custom_nodes_dir))
sys.path.insert(0, str(_bench_dir))

import torch
from run_benchmarks import (
    SYNTHETIC_PROGRAMS, load_example_programs, generate_bindings,
    compile_program, run_interpreter, _infer_types, BenchmarkProgram,
    system_info, _fingerprint,
)

CONFIGS = [
    "cpu_off_cold", "cpu_off_warm", "cuda_off_cold", "cuda_off_warm",
    "cpu_on_cold", "cpu_on_warm", "cuda_on_cold", "cuda_on_warm",
]

PRETTY = {
    "cpu_off_cold": "CPU cold",            "cpu_off_warm": "CPU warm",
    "cuda_off_cold": "GPU cold",           "cuda_off_warm": "GPU warm",
    "cpu_on_cold": "CPU cold Compiled",    "cpu_on_warm": "CPU warm Compiled",
    "cuda_on_cold": "GPU cold Compiled",   "cuda_on_warm": "GPU warm Compiled",
}


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def all_programs(limit: int | None = None, synthetic_only: bool = False):
    progs = list(SYNTHETIC_PROGRAMS)
    if not synthetic_only:
        progs += list(load_example_programs())
    progs = [p for p in progs if not p.string_only]  # skip non-spatial string progs
    if limit:
        progs = progs[:limit]
    return progs


# ── compile-OFF (interpreter), in-process, CUDA-synced ───────────────────────

def measure_interp(prog, B, H, W, device, cold,
                   warmup=2, min_runs=5, max_runs=25, target_cv=0.05,
                   budget_sec=8.0) -> dict:
    bindings = generate_bindings(prog, B, H, W, device)
    btypes = _infer_types(bindings)
    program, tm, assigned, used = compile_program(prog.code, btypes)
    out_names = list(assigned.keys()) if assigned and "OUT" not in assigned else None

    prog_t0 = time.perf_counter()
    # Budgeted warmup: bail after the first run if it's already slow (heavy
    # loop-y programs in the tree-walking interpreter cost seconds/run at 1024).
    for _ in range(warmup):
        if cold:
            program, tm, assigned, used = compile_program(prog.code, btypes)
            out_names = list(assigned.keys()) if assigned and "OUT" not in assigned else None
        run_interpreter(program, bindings, tm, device, out_names, used_builtins=used)
        if time.perf_counter() - prog_t0 > budget_sec * 0.5:
            break
    _sync(device)

    times: list[float] = []
    n = 0
    t_timed = time.perf_counter()
    while n < max_runs:
        gc.collect()
        _sync(device)
        t0 = time.perf_counter()
        if cold:
            program, tm, assigned, used = compile_program(prog.code, btypes)
            out_names = list(assigned.keys()) if assigned and "OUT" not in assigned else None
        run_interpreter(program, bindings, tm, device, out_names, used_builtins=used)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)
        n += 1
        if n >= min_runs:
            mu = statistics.mean(times)
            if mu > 0 and statistics.stdev(times) / mu < target_cv:
                break
        # Per-program wall-clock budget keeps the 140-program matrix tractable
        # even when a single program takes seconds per run.
        if time.perf_counter() - t_timed > budget_sec:
            break

    mu = statistics.mean(times)
    cv = (statistics.stdev(times) / mu * 100) if len(times) > 1 and mu > 0 else 0.0
    return {"median": round(statistics.median(times), 4), "n": n,
            "cv": round(cv, 1), "status": "interp"}


# ── compile-ON (torch.compile via execute_compiled), subprocess worker ───────

def worker_main(payload: str) -> None:
    """Run inside an isolated subprocess: measure compile-ON cold+warm for one
    program on one device. Prints a single 'RESULT:{json}' line."""
    args = json.loads(payload)
    prog = next(p for p in all_programs() if p.name == args["prog_name"])
    B, H, W = args["B"], args["H"], args["W"]
    device = args["device"]
    cold_runs, warm_runs, warmup = args["cold_runs"], args["warm_runs"], args["warmup"]

    from TEX_Wrangle.tex_runtime import compiled as C
    from TEX_Wrangle.tex_runtime.compiled import execute_compiled, clear_compiled_cache

    bindings = generate_bindings(prog, B, H, W, device)
    btypes = _infer_types(bindings)
    program, tm, assigned, used = compile_program(prog.code, btypes)
    out_names = list(assigned.keys()) if assigned and "OUT" not in assigned else None
    fp = _fingerprint(prog.code, btypes)
    dtype = torch.device(device).type

    out = {"status": "unknown", "warm": None, "cold": None, "error": None}

    clear_compiled_cache()
    try:
        for _ in range(max(1, warmup)):
            execute_compiled(program, dict(bindings), tm, device, fp, output_names=out_names)
        _sync(device)
    except Exception as e:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {str(e)[:160]}"

    # Honest status detection from the compiled module's caches.
    if out["status"] != "error":
        if (fp, dtype) in C._compiled_cache:
            out["status"] = "compiled"      # torch.compile actually engaged
        elif fp in C._compile_blacklist:
            out["status"] = "fallback"      # compile attempted, failed -> interpreter
        else:
            out["status"] = "plain"         # gated out -> plain interpreter

    def _timed(runs, clear):
        ts = []
        for _ in range(runs):
            if clear:
                clear_compiled_cache()
            gc.collect()
            _sync(device)
            t0 = time.perf_counter()
            try:
                execute_compiled(program, dict(bindings), tm, device, fp, output_names=out_names)
            except Exception as e:  # noqa: BLE001
                out["error"] = out["error"] or f"{type(e).__name__}: {str(e)[:160]}"
            _sync(device)
            ts.append((time.perf_counter() - t0) * 1000)
        return round(statistics.median(ts), 4) if ts else None

    out["warm"] = _timed(warm_runs, clear=False)
    out["cold"] = _timed(cold_runs, clear=True)
    print("RESULT:" + json.dumps(out))


def measure_compiled_subprocess(prog_name, device, B, H, W,
                                cold_runs=2, warm_runs=5, warmup=1,
                                timeout=200) -> dict:
    payload = json.dumps({
        "prog_name": prog_name, "device": device, "B": B, "H": H, "W": W,
        "cold_runs": cold_runs, "warm_runs": warm_runs, "warmup": warmup,
    })
    try:
        r = subprocess.run(
            [sys.executable, str(_bench_dir / "eight_config_bench.py"), "--worker", payload],
            capture_output=True, text=True, timeout=timeout, cwd=str(_bench_dir),
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "warm": None, "cold": None, "error": f"timeout>{timeout}s"}
    if r.returncode != 0:
        tail = (r.stderr.strip().split("\n")[-1][:160] if r.stderr else "unknown")
        return {"status": "crash", "warm": None, "cold": None, "error": tail}
    for line in reversed(r.stdout.strip().split("\n")):
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])
    return {"status": "noresult", "warm": None, "cold": None, "error": "no RESULT line"}


# ── Matrix runner ────────────────────────────────────────────────────────────

def _save(data, path, _attempts=5):
    """Atomic, retry-safe save: write a temp file then rename into place, so a
    concurrent reader can never see a truncated file (Windows shares poorly)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    for i in range(_attempts):
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            tmp.replace(p)
            return
        except OSError:
            if i == _attempts - 1:
                raise
            time.sleep(0.2)


def run_matrix(args):
    B = args.batch
    H = W = args.resolution
    progs = all_programs(limit=args.limit, synthetic_only=args.synthetic_only)
    devices = ["cpu"]
    if torch.cuda.is_available() and not args.cpu_only:
        devices.append("cuda")

    info = system_info("matrix")
    info["resolution"] = H
    info["batch"] = B
    info["num_programs"] = len(progs)
    data = {"system": info, "configs": CONFIGS, "results": {c: {} for c in CONFIGS}}
    res = data["results"]

    save_path = args.save or str(_bench_dir / "results" / "eight_config.json")

    print(f"\n{'='*72}")
    print(f"  TEX 8-Config Benchmark - {H}x{W}, B={B}")
    print(f"  Programs: {len(progs)}  |  Devices: {devices}  |  torch {torch.__version__}")
    if "gpu_name" in info:
        print(f"  GPU: {info['gpu_name']}")
    print(f"  Saving to: {save_path}")
    print(f"{'='*72}\n")

    t_start = time.perf_counter()

    # --- compile-OFF configs (in-process interpreter) ---
    for device in devices:
        for cold in (True, False):
            cfg = f"{device}_off_{'cold' if cold else 'warm'}"
            print(f"--- {PRETTY[cfg]} ({cfg}) - {len(progs)} programs ---")
            for i, prog in enumerate(progs, 1):
                try:
                    m = measure_interp(prog, B, H, W, device, cold)
                except (RuntimeError, MemoryError) as e:
                    m = {"median": None, "status": "error", "error": str(e)[:160]}
                except Exception as e:  # noqa: BLE001
                    m = {"median": None, "status": "skip", "error": str(e)[:160]}
                res[cfg][prog.name] = m
                _save(data, save_path)
                med = m.get("median")
                tag = f"{med:9.3f}ms" if med is not None else f"  {m.get('status','?'):>9}"
                print(f"  [{i:>3}/{len(progs)}] {prog.name:<34} {tag}")
            print()

    # --- compile-ON configs (subprocess per program; yields cold+warm) ---
    for device in devices:
        cold_cfg, warm_cfg = f"{device}_on_cold", f"{device}_on_warm"
        print(f"--- {PRETTY[cold_cfg]} + {PRETTY[warm_cfg]} - {len(progs)} programs (subprocess) ---")
        for i, prog in enumerate(progs, 1):
            r = measure_compiled_subprocess(
                prog.name, device, B, H, W,
                cold_runs=args.cold_runs, warm_runs=args.warm_runs,
                timeout=args.timeout,
            )
            res[cold_cfg][prog.name] = {"median": r.get("cold"), "status": r.get("status"),
                                        "error": r.get("error")}
            res[warm_cfg][prog.name] = {"median": r.get("warm"), "status": r.get("status"),
                                        "error": r.get("error")}
            _save(data, save_path)
            c, w = r.get("cold"), r.get("warm")
            cs = f"{c:8.1f}ms" if c is not None else "    n/a"
            ws = f"{w:8.3f}ms" if w is not None else "    n/a"
            print(f"  [{i:>3}/{len(progs)}] {prog.name:<34} cold={cs} warm={ws}  [{r.get('status')}]")
        print()

    info["elapsed_sec"] = round(time.perf_counter() - t_start, 1)
    _save(data, save_path)
    print(f"\nDone in {info['elapsed_sec']}s. Saved {save_path}")
    summarize(data)
    if args.compare:
        compare(data, args.compare)
    return data


# ── Summary + comparison ─────────────────────────────────────────────────────

def _valid_medians(cfg_map):
    return {k: v["median"] for k, v in cfg_map.items()
            if isinstance(v, dict) and isinstance(v.get("median"), (int, float))}


def _geomean(vals):
    vals = [v for v in vals if v and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def summarize(data):
    res = data["results"]
    print(f"\n{'='*72}")
    print(f"  SUMMARY - geometric mean of per-program median ms")
    print(f"{'='*72}")
    print(f"  {'Config':<22} {'GeoMean':>10} {'Total':>11} {'N':>4}  Status")
    print(f"  {'-'*66}")
    for cfg in CONFIGS:
        medians = _valid_medians(res.get(cfg, {}))
        statuses = {}
        for v in res.get(cfg, {}).values():
            s = (v or {}).get("status", "?")
            statuses[s] = statuses.get(s, 0) + 1
        gm = _geomean(list(medians.values()))
        total = sum(medians.values()) if medians else None
        gm_s = f"{gm:8.3f}ms" if gm else "     n/a"
        tot_s = f"{total:9.1f}ms" if total else "      n/a"
        st = " ".join(f"{k}:{v}" for k, v in sorted(statuses.items()))
        print(f"  {PRETTY[cfg]:<22} {gm_s:>10} {tot_s:>11} {len(medians):>4}  {st}")
    print(f"{'='*72}\n")


def compare(data, baseline_path):
    with open(baseline_path) as f:
        base = json.load(f)
    bres = base.get("results", {})
    cres = data["results"]
    print(f"\n{'='*72}")
    print(f"  REGRESSION CHECK vs {Path(baseline_path).name}")
    print(f"  (speedup = baseline / current; <0.95x = REGRESSION)")
    print(f"{'='*72}")
    print(f"  {'Config':<22} {'GeoMean spd':>12} {'Regressions':>12}  worst")
    print(f"  {'-'*66}")
    any_reg = False
    for cfg in CONFIGS:
        bmap = _valid_medians(bres.get(cfg, {}))
        cmap = _valid_medians(cres.get(cfg, {}))
        speedups = []
        worst = None
        for name, bcur in bmap.items():
            ccur = cmap.get(name)
            if not ccur or ccur <= 0 or bcur <= 0:
                continue
            sp = bcur / ccur
            speedups.append(sp)
            if worst is None or sp < worst[1]:
                worst = (name, sp)
        gm = _geomean(speedups)
        regs = sum(1 for s in speedups if s < 0.95)
        if regs:
            any_reg = True
        gm_s = f"{gm:8.3f}x" if gm else "     n/a"
        worst_s = f"{worst[0]}={worst[1]:.2f}x" if worst else "-"
        flag = "  <-- REGRESSION" if gm and gm < 0.95 else ""
        print(f"  {PRETTY[cfg]:<22} {gm_s:>12} {regs:>6}/{len(speedups):<5} {worst_s}{flag}")
    print(f"{'='*72}")
    print(f"  {'No regressions detected.' if not any_reg else 'Regressions present - see per-config worst offenders.'}\n")


def main():
    ap = argparse.ArgumentParser(description="TEX 8-config benchmark")
    ap.add_argument("--worker", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="limit program count (validation)")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--cold-runs", type=int, default=2)
    ap.add_argument("--warm-runs", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=200, help="per-program compiled subprocess timeout (s)")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--compare", type=str, default=None)
    args = ap.parse_args()

    if args.worker is not None:
        worker_main(args.worker)
        return
    run_matrix(args)


if __name__ == "__main__":
    main()
