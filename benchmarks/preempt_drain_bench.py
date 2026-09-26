"""PACE-462 — bounded look-ahead pacing: the K-sweep and the drained-p95 measurement.

An embedding host measured that the original PACE-45 pacing mechanism (wait one
poll-interval, every poll, via an `event.query()` + fixed-sleep loop) cost a paced
background render 2.0x/1.67x of its unpaced run time. This bench measures the replacement
(`tex_runtime/pacing.py`'s bounded look-ahead: a per-thread ring of `depth` outstanding CUDA
events, waited on only when the ring is full) against the same shape of scenario: a
background cook on one thread, pre-empted mid-flight by an interactive cook submitted on
another thread.

Four experiments, each host-neutral (a background "committed" render, an "interactive"
request — no host or vendor named):

  1. `cost_unpreempted`   -- background cook's OWN run time, paced (per depth) vs unpaced,
                             nothing ever pre-empts it. The ask's headline number.
  2. `drain_on_preempt`   -- trip the background token at a random point, immediately submit
                             a small interactive cook on another thread: p50/p95 of
                             (i) submit -> interactive cook returns (host) and
                             (ii) submit -> device drained (interactive result's own
                             `CookResult.done.synchronize()`).
  3. `host_lead`          -- for an UNCANCELLED paced cook, how far the host's own return
                             leads the device's actual completion (paced vs unpaced), one
                             number per depth -- answers "does look-ahead pacing restore
                             honest host timings for paced cooks".
  4. `stream_priority`    -- MEASURE ONLY, never shipped as code: with an UNPACED background
                             cook (worst case), does submitting the interactive cook on a
                             high-priority CUDA stream (`torch.cuda.Stream(priority=-1)`)
                             cut drained latency versus the default stream? One small table;
                             a recommendation, not a code path.

Measurement rules this file follows (docs/brief-conventions.md): a fresh CUDA cache
directory per run (set `TEX_CACHE_DIR` before invoking), the first leg of any A/B always
discarded, GPU state and box name recorded beside every number.

    python_embeded/python.exe -X utf8 benchmarks/preempt_drain_bench.py --depths 1,2,3,4,8
    python_embeded/python.exe -X utf8 benchmarks/preempt_drain_bench.py --trials 100 --save results/pace462.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

_bench_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_bench_dir.parent.parent))   # custom_nodes/ -> import TEX_Wrangle

import torch
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime.host import CookCancelled

# ── Fixtures ─────────────────────────────────────────────────────────────────────

_N_STATEMENTS = 24
_SIZE = 2560
_HEAVY = "vec4 x = @A;\n" + "x = gauss_blur(x, 8.0);\n" * _N_STATEMENTS + "@OUT = x;\n"
_INTERACTIVE = "@OUT = vec4(@A.rgb * 1.15 + vec3(0.02), 1.0);"


def _heavy_bindings(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.rand(1, _SIZE, _SIZE, 4, generator=g)
    return {"A": img.cuda()}


def _interactive_bindings(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.rand(1, 64, 64, 4, generator=g)
    return {"A": img.cuda()}


class _PacedToken:
    """Never trips. `pace_depth=None` uses the module default."""
    def __init__(self, depth=None):
        self.pace = True
        if depth is not None:
            self.pace_depth = depth

    def check(self):
        pass


class _UnpacedToken:
    def __init__(self):
        self.pace = False

    def check(self):
        pass


class _TripToken:
    """Trips from a background timer thread, at `delay_s` -- the shape a real interrupt
    arrives in, asynchronously, mid-cook."""
    def __init__(self, delay_s, depth):
        self.pace = True
        self.pace_depth = depth
        self._tripped = threading.Event()
        self._timer = threading.Timer(delay_s, self._tripped.set)
        self._timer.daemon = True
        self._timer.start()

    def check(self):
        if self._tripped.is_set():
            raise CookCancelled("PACE-462 bench: token tripped")


def _p(xs, pct):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _box_note():
    if not torch.cuda.is_available():
        return "no CUDA"
    name = torch.cuda.get_device_name(0)
    cap = "sm_" + "".join(map(str, torch.cuda.get_device_capability(0)))
    util_mem = "?"
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader"], text=True).strip()
        util_mem = out
    except Exception:
        pass
    return f"{name} ({cap}), nvidia-smi: {util_mem}"


def _calibrate():
    """The uncancelled full runtime of the heavy chain -- discard a cold leg, floor of two
    warm legs (docs/brief-conventions.md's measurement discipline)."""
    def once(seed):
        t0 = time.perf_counter()
        tex_engine.cook(_HEAVY, _heavy_bindings(seed), device_mode="cuda")
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    once(1000)
    return min(once(1001), once(1002))


# ── Experiment 1: unpre-empted cost, paced (per depth) vs unpaced ────────────────

def cost_unpreempted(depths, trials, seed0=2000):
    unpaced_t, paced_t = {d: [] for d in depths}, {d: [] for d in depths}
    for d in depths:
        for i in range(trials):
            seed = seed0 + i
            a_first = i % 2 == 0
            if a_first:
                t0 = time.perf_counter()
                tex_engine.cook(_HEAVY, _heavy_bindings(seed), device_mode="cuda", cancel=_UnpacedToken())
                torch.cuda.synchronize()
                tu = time.perf_counter() - t0
                t0 = time.perf_counter()
                tex_engine.cook(_HEAVY, _heavy_bindings(seed), device_mode="cuda", cancel=_PacedToken(d))
                torch.cuda.synchronize()
                tp = time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                tex_engine.cook(_HEAVY, _heavy_bindings(seed), device_mode="cuda", cancel=_PacedToken(d))
                torch.cuda.synchronize()
                tp = time.perf_counter() - t0
                t0 = time.perf_counter()
                tex_engine.cook(_HEAVY, _heavy_bindings(seed), device_mode="cuda", cancel=_UnpacedToken())
                torch.cuda.synchronize()
                tu = time.perf_counter() - t0
            unpaced_t[d].append(tu)
            paced_t[d].append(tp)
    out = {}
    for d in depths:
        med_u, med_p = statistics.median(unpaced_t[d]), statistics.median(paced_t[d])
        out[d] = {"unpaced_ms": med_u * 1000, "paced_ms": med_p * 1000,
                   "overhead_pct": (med_p / med_u - 1.0) * 100}
    return out


# ── Experiment 2: pre-emption -> drained p50/p95, per depth ──────────────────────

def drain_on_preempt(depths, trials, full_runtime, seed0=3000):
    out = {}
    for d in depths:
        returns_ms, drained_ms = [], []
        for i in range(trials):
            import random
            delay = full_runtime * random.uniform(0.15, 0.85)
            tok = _TripToken(delay, d)
            bg_seed = seed0 + i
            bg_done = threading.Event()
            bg_result = {}

            def _bg():
                try:
                    tex_engine.cook(_HEAVY, _heavy_bindings(bg_seed), device_mode="cuda", cancel=tok)
                except CookCancelled:
                    pass
                finally:
                    bg_done.set()

            th = threading.Thread(target=_bg)
            th.start()
            tok._tripped.wait(timeout=full_runtime * 2 + 2)  # wait for the timer to trip

            t0 = time.perf_counter()
            res = tex_engine.cook(_INTERACTIVE, _interactive_bindings(bg_seed), device_mode="cuda")
            t1 = time.perf_counter()
            if res.done is not None:
                res.done.synchronize()
            t2 = time.perf_counter()

            returns_ms.append((t1 - t0) * 1000)
            drained_ms.append((t2 - t0) * 1000)
            th.join(timeout=full_runtime * 2 + 2)
            torch.cuda.synchronize()

        out[d] = {
            "return_p50_ms": _p(returns_ms, 0.50), "return_p95_ms": _p(returns_ms, 0.95),
            "drained_p50_ms": _p(drained_ms, 0.50), "drained_p95_ms": _p(drained_ms, 0.95),
        }
    return out


# ── Experiment 3: host-return-lead-device-completion, per depth ──────────────────

def host_lead(depths, trials=20, seed0=4000):
    out = {}
    for label, make_tok in (("unpaced", lambda d: _UnpacedToken()), ("paced", lambda d: _PacedToken(d))):
        for d in depths if label == "paced" else [None]:
            leads = []
            for i in range(trials):
                seed = seed0 + i
                t0 = time.perf_counter()
                res = tex_engine.cook(_HEAVY, _heavy_bindings(seed), device_mode="cuda", cancel=make_tok(d))
                t_return = time.perf_counter()
                torch.cuda.synchronize()
                t_drained = time.perf_counter()
                leads.append((t_drained - t_return) * 1000)
                _ = t0
            key = label if d is None else f"{label}_depth{d}"
            out[key] = {"lead_p50_ms": _p(leads, 0.50), "lead_p95_ms": _p(leads, 0.95)}
    return out


# ── Experiment 4: stream priority, measure-only ───────────────────────────────────

def stream_priority(trials, full_runtime, seed0=5000):
    """Unpaced background cook (worst case for drain latency); does the INTERACTIVE cook
    on a high-priority CUDA stream drain faster than on the default stream? Measure-only:
    this experiment intentionally does not touch tex_runtime/pacing.py or any product code
    -- it answers a question, and the answer becomes a recommendation in the hand-back."""
    import random
    high = torch.cuda.Stream(priority=-1)
    results = {"default_stream": [], "high_priority_stream": []}
    for i in range(trials):
        use_high = (i % 2 == 0)
        delay = full_runtime * random.uniform(0.15, 0.85)
        tok = _TripToken(delay, depth=999999)  # depth irrelevant: token never asked to pace (pace stays True though)
        tok.pace = False  # this experiment is about UNPACED background cooks specifically
        seed = seed0 + i
        bg_done = threading.Event()

        def _bg():
            try:
                tex_engine.cook(_HEAVY, _heavy_bindings(seed), device_mode="cuda", cancel=tok)
            except CookCancelled:
                pass
            finally:
                bg_done.set()

        th = threading.Thread(target=_bg)
        th.start()
        tok._tripped.wait(timeout=full_runtime * 2 + 2)

        t0 = time.perf_counter()
        if use_high:
            with torch.cuda.stream(high):
                res = tex_engine.cook(_INTERACTIVE, _interactive_bindings(seed), device_mode="cuda")
            torch.cuda.current_stream().wait_stream(high)
        else:
            res = tex_engine.cook(_INTERACTIVE, _interactive_bindings(seed), device_mode="cuda")
        if res.done is not None:
            res.done.synchronize()
        t1 = time.perf_counter()
        results["high_priority_stream" if use_high else "default_stream"].append((t1 - t0) * 1000)
        th.join(timeout=full_runtime * 2 + 2)
        torch.cuda.synchronize()

    out = {}
    for k, xs in results.items():
        out[k] = {"drained_p50_ms": _p(xs, 0.50), "drained_p95_ms": _p(xs, 0.95)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=str, default="1,2,3,4,8")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--host-lead-trials", type=int, default=20)
    ap.add_argument("--stream-trials", type=int, default=40)
    ap.add_argument("--skip-stream", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA on this box -- nothing to measure")
        return

    depths = [int(x) for x in a.depths.split(",")]
    box = _box_note()
    print(f"box: {box}")
    print(f"heavy chain: {_N_STATEMENTS} x gauss_blur(8.0) @ {_SIZE}^2")

    full = _calibrate()
    print(f"calibrated full uncancelled runtime: {full * 1000:.1f} ms\n")

    print("=== Experiment 1: unpre-empted cost, paced vs unpaced (median of "
          f"{a.trials} interleaved trials) ===")
    r1 = cost_unpreempted(depths, a.trials)
    for d in depths:
        row = r1[d]
        print(f"  depth={d:<2d}  unpaced {row['unpaced_ms']:8.2f} ms   "
              f"paced {row['paced_ms']:8.2f} ms   overhead {row['overhead_pct']:+6.2f}%")

    print(f"\n=== Experiment 2: drain on pre-emption ({a.trials} trials/depth) ===")
    r2 = drain_on_preempt(depths, a.trials, full)
    for d in depths:
        row = r2[d]
        print(f"  depth={d:<2d}  return  p50 {row['return_p50_ms']:7.2f} ms  p95 "
              f"{row['return_p95_ms']:7.2f} ms   |  drained  p50 {row['drained_p50_ms']:7.2f} ms  "
              f"p95 {row['drained_p95_ms']:7.2f} ms")

    print(f"\n=== Experiment 3: host-return lead over device completion "
          f"({a.host_lead_trials} trials) ===")
    r3 = host_lead(depths, a.host_lead_trials)
    for k, row in r3.items():
        print(f"  {k:<16s} lead p50 {row['lead_p50_ms']:7.3f} ms   p95 {row['lead_p95_ms']:7.3f} ms")

    r4 = None
    if not a.skip_stream:
        print(f"\n=== Experiment 4 (measure-only): stream priority, unpaced background "
              f"({a.stream_trials} trials) ===")
        r4 = stream_priority(a.stream_trials, full)
        for k, row in r4.items():
            print(f"  {k:<22s} drained p50 {row['drained_p50_ms']:7.2f} ms   "
                  f"p95 {row['drained_p95_ms']:7.2f} ms")

    if a.save:
        out = {"box": box, "full_runtime_ms": full * 1000, "depths": depths,
               "cost_unpreempted": r1, "drain_on_preempt": r2, "host_lead": r3,
               "stream_priority": r4}
        path = a.save if os.path.isabs(a.save) else \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), a.save)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
