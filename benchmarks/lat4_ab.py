"""LAT-4 gate: does the coordinate-builtins LRU cost anything on the DEFAULT path?

LAT-4 replaced a single (key, env) slot with a bounded LRU. Two questions, and they
need DIFFERENT instruments — conflating them produces a vacuous result:

  Q1 (invariant #7): on the steady path (same resolution every cook) BOTH versions HIT
      their cache. The entire v0.20->v0.21 delta is the hit lookup itself:
          v0.20:  if cache_key == self._builtins_cache_key:      # tuple __eq__
                      self.env.update(self._builtins_cache_env)
          v0.21:  hit = self._builtins_lru.get(cache_key)        # __hash__ + get
                  if hit is not None:
                      self._builtins_lru.move_to_end(cache_key)  # + relink
                      self.env.update(hit)
      That is tens of nanoseconds against a cook of ~10^5 ns, so WHOLE-COOK timing can
      never resolve it. `hit_path_microbench` times the two lookups directly instead.

      NOTE the trap this file used to fall into: toggling _BUILTINS_LRU_MAX does NOT
      emulate v0.20 here. That global is read at exactly one site — the eviction loop,
      reachable only on a MISS — so on the steady path cap=8 and cap=1 execute the SAME
      code. Comparing them is A-vs-A and prints "neutral" no matter what the change did.
      `steady_null_control` keeps that comparison, but labelled as what it actually is:
      a null control measuring the whole-cook noise floor — i.e. the evidence for why
      Q1 must be a microbenchmark.

  Q2 (the LAT-4 win): under proxy<->full-res alternation the cap IS live — LRU(1)
      evicts and rebuilds every cook, LRU(8) hits. `altern_ab` measures that, paired and
      interleaved so thermal drift hits both arms equally.

Run:  python_embeded/python.exe -X utf8 benchmarks/lat4_ab.py
"""
from __future__ import annotations

import random
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path

_bench_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_bench_dir.parent.parent))   # custom_nodes/ -> import TEX_Wrangle

import torch
from TEX_Wrangle.tex_runtime import interpreter as I
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_marshalling import infer_binding_type as _ibt

CODE = "@OUT = vec4(u, v, 0.0, 1.0);"   # pure coordinate builtins: worst case for
PROG = Parser(Lexer(CODE).tokenize(), source=CODE).parse()   # cache-mechanism overhead
random.seed(1234)


def _tm_for(img):
    ck = TypeChecker(binding_types={"A": _ibt(img)}, source=CODE)
    return ck.check(PROG)


def _cook(interp, img, tm, cap, device):
    I._BUILTINS_LRU_MAX = cap          # only bites on a MISS (see module docstring)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    interp.execute(PROG, {"A": img}, tm, device=device, output_names=["OUT"])
    if device == "cuda":
        torch.cuda.synchronize()       # invariant #6: GPU timing wraps synchronize
    return (time.perf_counter() - t0) * 1e3


# ── Q1: the hit-path lookup, timed directly ──────────────────────────────────
def hit_path_microbench(n=200_000, repeats=7):
    """Time v0.20's slot hit vs v0.21's LRU hit over identical, realistic state.
    min-of-repeats: the minimum is the least noise-contaminated estimator here — we
    want the cost of the code, not of the scheduler."""
    used = frozenset({"u", "v"})
    cache_key = ((1, 512, 512), "cpu", torch.float32, used, 0, None)
    # builtins as the interpreter stores them: expand() VIEWS, not materialized copies
    u = torch.linspace(0, 1, 512).view(1, 1, 512).expand(1, 512, 512)
    v = torch.linspace(0, 1, 512).view(1, 512, 1).expand(1, 512, 512)
    builtins_env = {"u": u, "v": v}

    slot_key, slot_env = cache_key, builtins_env          # v0.20 single slot
    lru = OrderedDict({cache_key: builtins_env})          # v0.21 LRU (steady => 1 entry)
    env_a, env_b = {}, {}

    def v020():
        if cache_key == slot_key:
            env_a.update(slot_env)
            return True
        return False

    def v021():
        hit = lru.get(cache_key)
        if hit is not None:
            lru.move_to_end(cache_key)
            env_b.update(hit)
            return True
        return False

    for f in (v020, v021):                                 # warm
        for _ in range(n // 10):
            f()

    def timeit(f):
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(n):
                f()
            best = min(best, time.perf_counter() - t0)
        return best / n * 1e9                              # ns/op

    ns20, ns21 = timeit(v020), timeit(v021)
    print("\n=== Q1: hit-path lookup, timed directly (n=%d x %d, min-of-N) ===" % (n, repeats))
    print(f"  v0.20 single slot (tuple __eq__)      : {ns20:7.1f} ns/cook")
    print(f"  v0.21 LRU (get + move_to_end)         : {ns21:7.1f} ns/cook")
    print(f"  LAT-4 delta on the steady path        : {ns21 - ns20:+7.1f} ns/cook")
    return ns20, ns21


# ── Q1 (control): the same thing whole-cook, which CANNOT see it ─────────────
def steady_null_control(device, reps=200):
    """A/A NULL CONTROL, not an A/B: at one resolution both arms hit, so the cap is
    never read and the two arms are the SAME code. Whatever spread this prints is the
    whole-cook instrument's noise floor -- the reason Q1 is a microbenchmark."""
    img = torch.rand(1, 512, 512, 3, device=device)
    tm = _tm_for(img)
    a_i, b_i = Interpreter(), Interpreter()
    for _ in range(8):
        _cook(a_i, img, tm, 8, device); _cook(b_i, img, tm, 1, device)
    a_t, b_t = [], []
    for i in range(reps):
        if i % 2 == 0:
            a_t.append(_cook(a_i, img, tm, 8, device)); b_t.append(_cook(b_i, img, tm, 1, device))
        else:
            b_t.append(_cook(b_i, img, tm, 1, device)); a_t.append(_cook(a_i, img, tm, 8, device))
    ma, mb = statistics.median(a_t), statistics.median(b_t)
    cv = 100 * statistics.pstdev(a_t) / statistics.mean(a_t)
    print(f"\n=== Q1 control (A/A): identical code in both arms / {device} (n={reps}) ===")
    print(f"  arm1 median {ma:8.4f} ms | arm2 median {mb:8.4f} ms | CV {cv:4.1f}%")
    print(f"  apparent gap between IDENTICAL arms   : {abs(ma - mb) / ma * 100:5.2f}%  <- noise floor")
    return ma, abs(ma - mb) / ma * 100


def _bootstrap_ci(ratios, n=2000):
    meds = []
    for _ in range(n):
        s = [ratios[random.randrange(len(ratios))] for _ in range(len(ratios))]
        meds.append(statistics.median(s))
    meds.sort()
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


# ── Q2: the real A/B — the cap IS live under alternation ─────────────────────
def altern_ab(device, reps=200):
    sizes = [512, 1024]
    imgs = [torch.rand(1, s, s, 3, device=device) for s in sizes]
    tms = [_tm_for(i) for i in imgs]
    a_i, b_i = Interpreter(), Interpreter()          # A=LRU(8), B=LRU(1)=v0.20 behaviour
    for k in range(len(sizes) * 4):
        j = k % len(sizes)
        _cook(a_i, imgs[j], tms[j], 8, device); _cook(b_i, imgs[j], tms[j], 1, device)

    a_t, b_t = [], []
    for i in range(reps):
        j = i % len(sizes)
        # order must NOT alias with size, or each size only ever sees one arm first
        a_first = (i // len(sizes)) % 2 == 0
        if a_first:
            ta = _cook(a_i, imgs[j], tms[j], 8, device); tb = _cook(b_i, imgs[j], tms[j], 1, device)
        else:
            tb = _cook(b_i, imgs[j], tms[j], 1, device); ta = _cook(a_i, imgs[j], tms[j], 8, device)
        a_t.append(ta); b_t.append(tb)

    ratios = [b / a for a, b in zip(a_t, b_t) if a > 0]     # >1 => LRU(8) faster
    med_r = statistics.median(ratios)
    lo, hi = _bootstrap_ci(ratios)
    wins = sum(1 for r in ratios if r > 1.0)
    print(f"\n=== Q2: proxy<->full alternation / {device} (n={reps}, sizes={sizes}) ===")
    print(f"  LRU(8) v0.21 : median {statistics.median(a_t):8.4f} ms")
    print(f"  LRU(1) v0.20 : median {statistics.median(b_t):8.4f} ms")
    print(f"  paired speedup {med_r:.3f}x  95% CI [{lo:.3f}, {hi:.3f}]  LRU(8) faster in "
          f"{wins}/{len(ratios)} reps ({100 * wins / len(ratios):.0f}%)")
    verdict = ("LRU(8) FASTER" if lo > 1.02 else
               "LRU(8) SLOWER — REGRESSION" if hi < 0.98 else
               "no difference beyond +/-2%")
    print(f"  --> {verdict}")
    return med_r, lo, hi


if __name__ == "__main__":
    ns20, ns21 = hit_path_microbench()
    cook_ms, floor = steady_null_control("cpu")
    delta_pct = (ns21 - ns20) / (cook_ms * 1e6) * 100
    print("\n" + "=" * 68)
    print("INVARIANT #7 (steady default path):")
    print(f"  LAT-4 adds {ns21 - ns20:+.1f} ns to a {cook_ms * 1e6:.0f} ns cook = {delta_pct:+.4f}%")
    print(f"  ...against a whole-cook noise floor of {floor:.2f}% -- ~{floor / max(abs(delta_pct), 1e-9):.0f}x")
    print("  larger than the effect. Neutral by direct measurement of the CHANGED code,")
    print("  not by an A/A comparison of v0.21 with itself.")
    print("=" * 68)
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        altern_ab(dev)
    print("\nQ2 is the LAT-4 win: proxy<->full alternation thrashes the v0.20 single slot.")
