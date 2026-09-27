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
  5. `stride_depth_sweep` -- PACE-47: the same cost/drain pair as 1+2, but crossed over
                             FOUR program shapes (a cheap per-pixel chain at two sizes, a
                             mid gauss_blur chain, and the existing heavy chain) and BOTH
                             `pace_depth` and `pace_stride_ms`, to answer whether striding
                             stays a pure cost knob (drained p95 near one statement's own
                             device time at every stride) or leaks into the correctness
                             bound (drained p95 growing with stride) on a given box. Run
                             with `--sweep`.
  6. `interactive_supersede_sweep` -- PACE-47: the host's own acceptance shape verbatim --
                             a small/interactive cook superseded a FIXED delay (default
                             100ms) into its own run, not a random fraction of it -- submit
                             -> drained p50/p95, per depth x stride. Run with `--sweep
                             --sweep-interactive`.

Measurement rules this file follows (docs/brief-conventions.md): a fresh CUDA cache
directory per run (set `TEX_CACHE_DIR` before invoking), the first leg of any A/B always
discarded, GPU state and box name recorded beside every number.

    python_embeded/python.exe -X utf8 benchmarks/preempt_drain_bench.py --depths 1,2,3,4,8
    python_embeded/python.exe -X utf8 benchmarks/preempt_drain_bench.py --trials 100 --save results/pace462.json
    python_embeded/python.exe -X utf8 benchmarks/preempt_drain_bench.py --sweep --save results/pace47_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
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


def _sized_bindings(size, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.rand(1, size, size, 4, generator=g)
    return {"A": img.cuda()}


def _interactive_bindings(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.rand(1, 64, 64, 4, generator=g)
    return {"A": img.cuda()}


class _PacedToken:
    """Never trips. `pace_depth=None`/`stride_ms=None` use the module defaults."""
    def __init__(self, depth=None, stride_ms=None):
        self.pace = True
        if depth is not None:
            self.pace_depth = depth
        if stride_ms is not None:
            self.pace_stride_ms = stride_ms

    def check(self):
        pass


class _UnpacedToken:
    def __init__(self):
        self.pace = False

    def check(self):
        pass


class _TripToken:
    """Trips from a background timer thread, at `delay_s` -- the shape a real interrupt
    arrives in, asynchronously, mid-cook. `stride_ms=None` uses the module default."""
    def __init__(self, delay_s, depth, stride_ms=None):
        self.pace = True
        self.pace_depth = depth
        if stride_ms is not None:
            self.pace_stride_ms = stride_ms
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


class BackgroundCookHungError(RuntimeError):
    """Raised when a benchmark's own background cook thread is still alive after its join
    timeout -- filed as a bug report (BENCH-47's finding on a shared box): a non-daemon
    thread left running here previously kept the whole process (and the GPU) alive well
    past the point the script had printed its results, silently contaminating whichever
    measurement ran next on the same box, with no error, warning or nonzero exit code
    anywhere to flag it. A background cook this benchmark spawns must finish, or raise
    `CookCancelled`, in bounded time -- a thread still alive after twice the shape's own
    calibrated runtime is a hard failure to surface, never a silent continue."""


class BackgroundCookCrashedError(RuntimeError):
    """FIX-GATE G2 (v0.47.0 Phase C, B4#5): raised when a benchmark's background cook
    thread raised something OTHER than `CookCancelled` -- the sibling gap
    `BackgroundCookHungError` above does not cover. A hang is caught by the join-timeout
    check; a genuine bug in the cook path is not a hang at all -- the thread dies almost
    immediately (Python prints "Exception in thread ..." to stderr and the thread simply
    ends), so `is_alive()` reads `False` well within the timeout and, before this fix,
    nothing downstream ever learned the trial did not do what the benchmark thinks it did.
    Worse than a hang: a hang at least stops the run, this produced a plausible-looking but
    corrupted number with no error, warning, or nonzero exit code anywhere."""


def _start_background_cook(fn):
    """Start *fn* on a DAEMON thread (so a hang here can never keep the interpreter alive
    at process shutdown -- the other half of the fix) and return the thread."""
    th = threading.Thread(target=fn, daemon=True)
    th.start()
    return th


def _reap_background_cook(th, timeout, context, error_box=None):
    """Join *th* and raise `BackgroundCookHungError` if it is still alive afterward,
    instead of the previous silent `continue` -- a hung background cook is a benchmark
    result nobody can trust, not a trial to skip quietly.

    G2 (FIX-GATE, B4#5): *error_box*, when given, is the SAME dict a `_bg()` closure's
    `except Exception as exc: error_box["exc"] = exc` stores into -- checked AFTER the
    hang check (a thread that is both still alive AND recorded an error reports the hang,
    the more actionable of the two), and re-raised here as `BackgroundCookCrashedError`
    so a genuine cook-path bug surfaces at the reap point instead of reading as a clean,
    on-time finish."""
    th.join(timeout=timeout)
    if th.is_alive():
        raise BackgroundCookHungError(
            f"{context}: the background cook thread was still alive after a "
            f"{timeout:.2f}s join timeout -- it neither finished nor raised "
            f"CookCancelled in bounded time.")
    if error_box is not None and error_box.get("exc") is not None:
        exc = error_box["exc"]
        raise BackgroundCookCrashedError(
            f"{context}: the background cook thread raised {exc!r} instead of finishing "
            f"or raising CookCancelled -- this trial's result cannot be trusted.") from exc


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
            bg_error: dict = {}   # G2 (FIX-GATE, B4#5): filled iff `_bg()` raises for real

            def _bg():
                try:
                    tex_engine.cook(_HEAVY, _heavy_bindings(bg_seed), device_mode="cuda", cancel=tok)
                except CookCancelled:
                    pass
                except Exception as exc:
                    bg_error["exc"] = exc
                finally:
                    bg_done.set()

            th = _start_background_cook(_bg)
            tok._tripped.wait(timeout=full_runtime * 2 + 2)  # wait for the timer to trip

            t0 = time.perf_counter()
            res = tex_engine.cook(_INTERACTIVE, _interactive_bindings(bg_seed), device_mode="cuda")
            t1 = time.perf_counter()
            if res.done is not None:
                res.done.synchronize()
            t2 = time.perf_counter()

            returns_ms.append((t1 - t0) * 1000)
            drained_ms.append((t2 - t0) * 1000)
            _reap_background_cook(th, full_runtime * 2 + 2,
                                   f"drain_on_preempt depth={d} trial={i} "
                                   f"delay_s={delay:.4f} full_runtime_s={full_runtime:.4f}",
                                   error_box=bg_error)
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
        bg_error: dict = {}   # G2 (FIX-GATE, B4#5): filled iff `_bg()` raises for real

        def _bg():
            try:
                tex_engine.cook(_HEAVY, _heavy_bindings(seed), device_mode="cuda", cancel=tok)
            except CookCancelled:
                pass
            except Exception as exc:
                bg_error["exc"] = exc
            finally:
                bg_done.set()

        th = _start_background_cook(_bg)
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
        _reap_background_cook(th, full_runtime * 2 + 2, f"stream_priority trial={i}",
                              error_box=bg_error)
        torch.cuda.synchronize()

    out = {}
    for k, xs in results.items():
        out[k] = {"drained_p50_ms": _p(xs, 0.50), "drained_p95_ms": _p(xs, 0.95)}
    return out


# ── Experiment 5 (PACE-47): the shape x depth x stride sweep ─────────────────────
#
# Host-neutral replacement for a one-off scratch script this ask was measured with: four
# program shapes (a cheap per-pixel chain at two sizes, a mid gauss_blur chain, and the
# existing heavy chain) crossed with `pace_depth` x `pace_stride_ms`, each cell reporting
# (a) unpre-empted cost (paced vs unpaced, interleaved) and (b) drained p50/p95 after a
# pre-emption. Answers whether PACE-47's fix (the stride skip is honoured only while the
# device has caught up) makes `stride` a pure cost knob: a correct mechanism should show
# drained p95 staying near one statement's own device time at EVERY stride, not growing
# with it, while the unpre-empted cost still falls as stride grows (the knob it is meant
# to be).

_SWEEP_CHEAP_N = 220
_SWEEP_MEDIUM_N = 300

_SWEEP_DEPTHS_DEFAULT = "1,2"
_SWEEP_STRIDES_MS_DEFAULT = "0,0.1,0.25,0.5,1.0"


def _sweep_cheap_code(n):
    # Trivial per-pixel arithmetic -- no gauss_blur, no halo -- so device work per
    # statement is as small as this language can make it: the class of program the
    # stride gate exists to protect.
    return ("vec4 x = @A;\n"
            + "x = x * 1.0001 + vec4(0.00001, 0.00002, 0.00001, 0.0);\n" * n
            + "@OUT = x;\n")


def _sweep_medium_code(n):
    return "vec4 x = @A;\n" + "x = gauss_blur(x, 3.0);\n" * n + "@OUT = x;\n"


def _sweep_shapes():
    """Built lazily (not at import time) so the module stays importable without ever
    constructing the heavy chain's own strings twice; `_HEAVY`/`_SIZE`/`_N_STATEMENTS`
    are reused verbatim so the sweep's "heavy" row is directly comparable to Experiments
    1-3's own heavy-chain numbers."""
    return {
        "cheap256": {"code": _sweep_cheap_code(_SWEEP_CHEAP_N), "size": 256, "n": _SWEEP_CHEAP_N},
        "cheap1024": {"code": _sweep_cheap_code(_SWEEP_CHEAP_N), "size": 1024, "n": _SWEEP_CHEAP_N},
        # PACE-47e: a cheap (point-footprint) chain's own device time scales with PIXELS,
        # not footprint -- cheap2048 is the same trivial-arithmetic code as cheap256/
        # cheap1024, just at a resolution where each statement's device time alone
        # (measured elsewhere: ~0.66ms at 1024^2, so ~2.6ms at 2048^2) is no longer
        # negligible next to a multi-ms stride window.
        "cheap2048": {"code": _sweep_cheap_code(_SWEEP_CHEAP_N), "size": 2048, "n": _SWEEP_CHEAP_N},
        "medium": {"code": _sweep_medium_code(_SWEEP_MEDIUM_N), "size": 1024, "n": _SWEEP_MEDIUM_N},
        "heavy": {"code": _HEAVY, "size": _SIZE, "n": _N_STATEMENTS},
    }


class _SweepTripToken:
    """Deterministic on_progress statement-FRACTION trip (not a wall-clock timer): immune
    to a box's own clock ramp under load, unlike `_TripToken` above, which this sweep does
    not reuse for exactly that reason (a shorter, cheaper-per-statement shape trips too
    fast for a `delay_s` timer calibrated off a separately measured full runtime to stay
    reliable). `on_progress` is a second callback threaded through `tex_engine.cook`,
    called once per top-level interpreter statement -- NOT a method the cancel token
    itself is polled for, so callers must pass `tok.on_progress` explicitly alongside
    `cancel=tok`."""
    def __init__(self, depth, stride_ms, target_frac):
        self.pace = True
        self.pace_depth = depth
        self.pace_stride_ms = stride_ms
        self._target_frac = target_frac
        self._tripped = threading.Event()

    def check(self):
        if self._tripped.is_set():
            raise CookCancelled("PACE-47 sweep: statement-fraction trip")

    def on_progress(self, phase, frac):
        if phase == "stmt" and frac >= self._target_frac:
            self._tripped.set()


def _sweep_calibrate(code, size, trials=2):
    def once(seed):
        t0 = time.perf_counter()
        tex_engine.cook(code, _sized_bindings(size, seed), device_mode="cuda")
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    once(9000)  # discard the cold leg (docs/brief-conventions.md's measurement discipline)
    return min(once(9001 + i) for i in range(trials))


def sweep_cost_unpreempted(code, size, depth, stride_ms, trials, seed0):
    """One cell's (a): median cost when nothing pre-empts, paced vs unpaced, interleaved
    A/B/A/B so box drift cancels rather than compounds into a phantom regression."""
    unpaced_t, paced_t = [], []
    for i in range(trials):
        seed = seed0 + i
        a_first = i % 2 == 0
        order = (("u", "p"), ("p", "u"))[0 if a_first else 1]
        times = {}
        for which in order:
            tok = _UnpacedToken() if which == "u" else _PacedToken(depth, stride_ms)
            t0 = time.perf_counter()
            tex_engine.cook(code, _sized_bindings(size, seed), device_mode="cuda", cancel=tok)
            torch.cuda.synchronize()
            times[which] = time.perf_counter() - t0
        unpaced_t.append(times["u"])
        paced_t.append(times["p"])
    med_u, med_p = statistics.median(unpaced_t), statistics.median(paced_t)
    return {"unpaced_ms": med_u * 1000, "paced_ms": med_p * 1000,
            "overhead_pct": (med_p / med_u - 1.0) * 100}


def sweep_drain_on_preempt(code, size, depth, stride_ms, trials, full_runtime, seed0):
    """One cell's (b): submit -> device-drained p50/p95 after a pre-emption trips at a
    random point in [15%, 85%] of the shape's own uncancelled full runtime."""
    returns_ms, drained_ms = [], []
    for i in range(trials):
        frac = random.uniform(0.15, 0.85)
        tok = _SweepTripToken(depth, stride_ms, frac)
        bg_seed = seed0 + i
        bg_done = threading.Event()
        bg_error: dict = {}   # G2 (FIX-GATE, B4#5): filled iff `_bg()` raises for real

        def _bg():
            try:
                tex_engine.cook(code, _sized_bindings(size, bg_seed), device_mode="cuda",
                                 cancel=tok, on_progress=tok.on_progress)
            except CookCancelled:
                pass
            except Exception as exc:
                bg_error["exc"] = exc
            finally:
                bg_done.set()

        th = _start_background_cook(_bg)
        tok._tripped.wait(timeout=full_runtime * 4 + 5)

        t0 = time.perf_counter()
        res = tex_engine.cook(_INTERACTIVE, _interactive_bindings(bg_seed), device_mode="cuda")
        t1 = time.perf_counter()
        if res.done is not None:
            res.done.synchronize()
        t2 = time.perf_counter()

        returns_ms.append((t1 - t0) * 1000)
        drained_ms.append((t2 - t0) * 1000)
        _reap_background_cook(th, full_runtime * 4 + 5,
                               f"sweep_drain_on_preempt depth={depth} stride_ms={stride_ms} "
                               f"trial={i} trip_frac={frac:.4f}",
                               error_box=bg_error)
        torch.cuda.synchronize()

    return {
        "return_p50_ms": _p(returns_ms, 0.50), "return_p95_ms": _p(returns_ms, 0.95),
        "drained_p50_ms": _p(drained_ms, 0.50), "drained_p95_ms": _p(drained_ms, 0.95),
    }


def stride_depth_sweep(shape_names, depths, strides_ms, cost_trials, drain_trials, verbose=True):
    """The full grid, one dict keyed by shape -> cell (`depth{d}_stride{s}`) -> cost/drain,
    plus each shape's own calibrated uncancelled full runtime."""
    shapes = _sweep_shapes()
    out = {"depths": depths, "strides_ms": strides_ms, "shapes": {}}
    for shape_name in shape_names:
        shp = shapes[shape_name]
        code, size, n = shp["code"], shp["size"], shp["n"]
        full = _sweep_calibrate(code, size)
        if verbose:
            print(f"\n=== shape={shape_name}  n_statements={n}  size={size}  "
                  f"calibrated full={full * 1000:.2f} ms ===")
        shape_out = {"n_statements": n, "size": size, "full_runtime_ms": full * 1000, "cells": {}}
        for depth in depths:
            for stride_ms in strides_ms:
                cell_key = f"depth{depth}_stride{stride_ms}"
                t0 = time.perf_counter()
                cost = sweep_cost_unpreempted(code, size, depth, stride_ms, cost_trials, seed0=10000)
                drain = sweep_drain_on_preempt(code, size, depth, stride_ms, drain_trials, full,
                                               seed0=20000)
                dt = time.perf_counter() - t0
                if verbose:
                    print(f"  depth={depth} stride={stride_ms:>4}ms  "
                          f"unpaced={cost['unpaced_ms']:8.3f}ms paced={cost['paced_ms']:8.3f}ms "
                          f"overhead={cost['overhead_pct']:+6.2f}%   "
                          f"drain p50={drain['drained_p50_ms']:7.3f}ms "
                          f"p95={drain['drained_p95_ms']:7.3f}ms  ({dt:.1f}s)")
                shape_out["cells"][cell_key] = {"depth": depth, "stride_ms": stride_ms,
                                                 "cost": cost, "drain": drain}
        out["shapes"][shape_name] = shape_out
    return out


# ── Experiment 6 (PACE-47): an interactive-shaped path -- a cook superseded ──────
# ── by the next one a FIXED delay in, not a fraction of its own runtime ──────────
#
# The host's own acceptance criterion, verbatim: "a small cook superseded 100 ms in by
# the next one: submit -> drained p95", since the host paces every cancellable cook, not
# only long background renders. Unlike Experiments 2/5 (a trip at a random FRACTION of
# the shape's own full runtime), this fixes the trip at an absolute wall-clock delay --
# the shape a real edit-supersedes-edit interaction has (the second edit doesn't wait for
# any particular fraction of the first to elapse, it lands at a roughly fixed latency
# after the user's own action).

def sweep_interactive_supersede(shape_name, depth, stride_ms, trials, delay_s, seed0=30000):
    shapes = _sweep_shapes()
    shp = shapes[shape_name]
    code, size = shp["code"], shp["size"]
    returns_ms, drained_ms = [], []
    for i in range(trials):
        tok = _TripToken(delay_s, depth, stride_ms)
        bg_seed = seed0 + i
        bg_done = threading.Event()
        bg_error: dict = {}   # G2 (FIX-GATE, B4#5): filled iff `_bg()` raises for real

        def _bg():
            try:
                tex_engine.cook(code, _sized_bindings(size, bg_seed), device_mode="cuda", cancel=tok)
            except CookCancelled:
                pass
            except Exception as exc:
                bg_error["exc"] = exc
            finally:
                bg_done.set()

        th = _start_background_cook(_bg)
        tok._tripped.wait(timeout=delay_s + 5)

        t0 = time.perf_counter()
        res = tex_engine.cook(_INTERACTIVE, _interactive_bindings(bg_seed), device_mode="cuda")
        t1 = time.perf_counter()
        if res.done is not None:
            res.done.synchronize()
        t2 = time.perf_counter()

        returns_ms.append((t1 - t0) * 1000)
        drained_ms.append((t2 - t0) * 1000)
        _reap_background_cook(th, delay_s + 5,
                               f"sweep_interactive_supersede shape={shape_name} depth={depth} "
                               f"stride_ms={stride_ms} trial={i}",
                               error_box=bg_error)
        torch.cuda.synchronize()

    return {
        "return_p50_ms": _p(returns_ms, 0.50), "return_p95_ms": _p(returns_ms, 0.95),
        "drained_p50_ms": _p(drained_ms, 0.50), "drained_p95_ms": _p(drained_ms, 0.95),
    }


def interactive_supersede_sweep(shape_name, depths, strides_ms, trials, delay_s, verbose=True):
    out = {"shape": shape_name, "delay_ms": delay_s * 1000, "depths": depths,
           "strides_ms": strides_ms, "cells": {}}
    if verbose:
        print(f"\n=== Experiment 6: interactive-shaped supersede ({shape_name}, "
              f"delay={delay_s * 1000:.0f}ms, {trials} trials/cell) ===")
    for depth in depths:
        for stride_ms in strides_ms:
            cell_key = f"depth{depth}_stride{stride_ms}"
            row = sweep_interactive_supersede(shape_name, depth, stride_ms, trials, delay_s)
            if verbose:
                print(f"  depth={depth} stride={stride_ms:>4}ms  "
                      f"return p50={row['return_p50_ms']:7.3f}ms p95={row['return_p95_ms']:7.3f}ms  "
                      f"drained p50={row['drained_p50_ms']:7.3f}ms p95={row['drained_p95_ms']:7.3f}ms")
            out["cells"][cell_key] = {"depth": depth, "stride_ms": stride_ms, **row}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=str, default="1,2,3,4,8")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--host-lead-trials", type=int, default=20)
    ap.add_argument("--stream-trials", type=int, default=40)
    ap.add_argument("--skip-stream", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--sweep", action="store_true",
                     help="PACE-47: run the shape x depth x stride sweep (Experiment 5) "
                          "instead of Experiments 1-4.")
    ap.add_argument("--sweep-shapes", type=str, default="cheap256,cheap1024,medium,heavy")
    ap.add_argument("--sweep-depths", type=str, default=_SWEEP_DEPTHS_DEFAULT)
    ap.add_argument("--sweep-strides-ms", type=str, default=_SWEEP_STRIDES_MS_DEFAULT)
    ap.add_argument("--sweep-cost-trials", type=int, default=30)
    ap.add_argument("--sweep-drain-trials", type=int, default=40)
    ap.add_argument("--sweep-interactive", action="store_true",
                     help="PACE-47: also run Experiment 6, an interactive-shaped cook "
                          "superseded a fixed delay in (the host's own acceptance shape).")
    ap.add_argument("--interactive-shape", type=str, default="medium")
    ap.add_argument("--interactive-delay-ms", type=float, default=100.0)
    ap.add_argument("--interactive-trials", type=int, default=40)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA on this box -- nothing to measure")
        return

    box = _box_note()

    if a.sweep:
        print(f"box: {box}")
        shape_names = a.sweep_shapes.split(",")
        depths = [int(x) for x in a.sweep_depths.split(",")]
        strides_ms = [float(x) for x in a.sweep_strides_ms.split(",")]
        print(f"=== Experiment 5 (PACE-47): shape x depth x stride sweep -- "
              f"shapes={shape_names} depths={depths} strides_ms={strides_ms} ===")
        r5 = stride_depth_sweep(shape_names, depths, strides_ms,
                                 a.sweep_cost_trials, a.sweep_drain_trials)
        r6 = None
        if a.sweep_interactive:
            r6 = interactive_supersede_sweep(a.interactive_shape, depths, strides_ms,
                                              a.interactive_trials,
                                              a.interactive_delay_ms / 1000.0)
        if a.save:
            out = {"box": box, **r5}
            if r6 is not None:
                out["interactive_supersede"] = r6
            path = a.save if os.path.isabs(a.save) else \
                os.path.join(os.path.dirname(os.path.abspath(__file__)), a.save)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"\nsaved -> {path}")
        return

    depths = [int(x) for x in a.depths.split(",")]
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
