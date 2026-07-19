#!/usr/bin/env python3
"""
PM-3 — relaunch cold-start for a 100-program project (v0.25 CACHE-3 exit criterion).

The roadmap's PM-3 gate: "app-relaunch cold start for a 100-program project < 2s to first frame,
no re-trial jank." This makes the claim a reproducible regression check instead of prose.

It measures time-to-first-frame in a FRESH process against a cache a prior session prewarmed
(CACHE-3 warm_state + the .pkl/.cg on disk) versus a cold (empty) cache — the delta is what
prewarm + the persisted artifacts buy. The fixed torch-import cost (identical warm-or-cold, a host
constant outside TEX's budget) is reported separately from the TEX-side cook cost the budget
governs.

Usage (the driver spawns the two child phases, each a fresh process):
    cd TEX_Wrangle
    python benchmarks/cache3_cold_start.py
    python benchmarks/cache3_cold_start.py --n 100 --device cpu
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_bench_dir = Path(__file__).resolve().parent
_pkg_dir = _bench_dir.parent
_custom_nodes_dir = _pkg_dir.parent
sys.path.insert(0, str(_custom_nodes_dir))


def _programs(n: int):
    """`n` distinct TEX programs (distinct source -> distinct fingerprint) — a plausible
    n-node grade/mix project."""
    progs = []
    for i in range(n):
        k = 0.5 + (i % 20) * 0.03
        j = (i * 7) % 11 * 0.05
        if i % 3 == 0:
            progs.append(f"f$s = {k:.3f};\n@OUT = vec4(@A.rgb * $s + {j:.3f}, 1.0);")
        elif i % 3 == 1:
            progs.append(f"@OUT = vec4(@A.rgb * {k:.3f}, @A.a) + vec4({j:.3f}, 0.0, 0.0, 0.0);")
        else:
            progs.append(f"@OUT = mix(@A, vec4({k:.3f}), {j:.3f});")
    return progs


def _phase_warm(cache_dir: str, n: int, device: str) -> None:
    os.environ["TEX_CACHE_DIR"] = cache_dir
    from TEX_Wrangle import tex_api
    from TEX_Wrangle.tex_compiler.types import TEXType
    progs = [(s, {"A": TEXType.VEC4}) for s in _programs(n)]
    t0 = time.perf_counter()
    summary = tex_api.prewarm(progs, shapes=[(1, 512, 512)], device=device, compile_mode="none")
    print(f"WARM  {summary}  in {(time.perf_counter()-t0)*1000:.0f} ms")


def _phase_cold(cache_dir: str, n: int, device: str) -> None:
    os.environ["TEX_CACHE_DIR"] = cache_dir
    t_proc = time.perf_counter()
    import torch
    from TEX_Wrangle import tex_engine
    t_import = time.perf_counter() - t_proc
    srcs = _programs(n)
    A = torch.rand(1, 512, 512, 4)
    t0 = time.perf_counter()
    tex_engine.cook(srcs[0], {"A": A.clone()}, device_mode=device)   # the first frame
    t_first = time.perf_counter() - t0
    t1 = time.perf_counter()
    for s in srcs:
        tex_engine.cook(s, {"A": A.clone()}, device_mode=device)     # all n first-frames
    t_all = time.perf_counter() - t1
    print(f"COLD  torch_import={t_import*1000:.0f}ms  first_frame={t_first*1000:.1f}ms  "
          f"all{n}={t_all*1000:.0f}ms  (TEX-side to-first-frame={t_first*1000:.1f}ms)")


def _driver(n: int, device: str) -> None:
    py = sys.executable
    here = str(Path(__file__).resolve())
    with tempfile.TemporaryDirectory(prefix="tex_pm3_") as tmp:
        warm_dir = os.path.join(tmp, "warm")
        empty_dir = os.path.join(tmp, "empty")
        os.makedirs(warm_dir); os.makedirs(empty_dir)
        env = dict(os.environ)
        subprocess.run([py, "-X", "utf8", here, "--phase", "warm", "--cache", warm_dir,
                        "--n", str(n), "--device", device], env=env, check=True)
        print("\n--- relaunch against a PREWARMED cache ---")
        subprocess.run([py, "-X", "utf8", here, "--phase", "cold", "--cache", warm_dir,
                        "--n", str(n), "--device", device], env=env, check=True)
        print("\n--- baseline: cold against an EMPTY cache ---")
        subprocess.run([py, "-X", "utf8", here, "--phase", "cold", "--cache", empty_dir,
                        "--n", str(n), "--device", device], env=env, check=True)
        print("\nPM-3 target: TEX-side to-first-frame < 2000 ms (torch import is a host constant, "
              "excluded). The prewarmed all-N figure should beat the empty one — the win from the "
              "persisted .pkl/.cg + warm_state.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["driver", "warm", "cold"], default="driver")
    ap.add_argument("--cache", default="")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cpu")   # no-compiler boxes measure the interpreter tier
    a = ap.parse_args()
    if a.phase == "warm":
        _phase_warm(a.cache, a.n, a.device)
    elif a.phase == "cold":
        _phase_cold(a.cache, a.n, a.device)
    else:
        _driver(a.n, a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
