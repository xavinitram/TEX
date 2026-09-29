"""FIX-GATE G6 (v0.47.0 Phase C) -- one shared percentile formula in benchmarks/.

R1#3 found `benchmarks/artist_loops_bench.py._percentiles`'s inner `pct(p)` closure was a
line-for-line reimplementation of `benchmarks/compile_modes_bench._pctl` (same nearest-rank
formula: `idx = round(p * (n - 1))`) in a file that already dynamically loads three OTHER
sibling bench modules (`host_path_counts`, `preempt_drain_bench`, `io_playback_bench`) via
its own `_load()` helper specifically to avoid re-deriving their logic. Two independent
percentile formulas reported under one "p95" label is a plausible source of a false
regression/improvement read -- the exact comparison `artist_loops_bench` exists to enable. This pins that `_percentiles` now goes THROUGH `_load("compile_modes_bench", ...)`
and calls its `_pctl`, rather than recomputing the same arithmetic independently.
"""
import importlib.util
import os
import sys

from helpers import SubTestResult

_HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")


def _load_artist_loops_bench():
    """Fresh load of `artist_loops_bench` under a private module name. The sibling
    `compile_modes_bench` it loads is NOT isolated: it is cached under one shared
    `sys.modules` key, so every load sees the same module object, and the spy below is
    safe only because its `finally` restores `_pctl`."""
    path = os.path.join(_HERE, "artist_loops_bench.py")
    spec = importlib.util.spec_from_file_location("_g6_artist_loops_bench_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


def test_fixgate_g6_percentiles_reuses_compile_modes_bench_pctl(r: SubTestResult):
    print("\n--- G6: artist_loops_bench._percentiles reuses compile_modes_bench._pctl ---")
    try:
        alb = _load_artist_loops_bench()
        cmb = alb._load("compile_modes_bench", "compile_modes_bench.py")

        calls = {"n": 0}
        orig_pctl = cmb._pctl

        def spy_pctl(xs, p):
            calls["n"] += 1
            return orig_pctl(xs, p)

        cmb._pctl = spy_pctl
        try:
            out = alb._percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
        finally:
            cmb._pctl = orig_pctl

        assert calls["n"] == 3, (
            f"_percentiles must compute p50/p95/p99 through the shared compile_modes_bench "
            f"._pctl (one call per percentile), got {calls['n']} calls -- a duplicated "
            f"formula was not reused")
        assert out["p50_ms"] == 3.0 and out["n"] == 5, out
        r.ok("artist_loops_bench._percentiles() calls the shared _pctl, not its own copy")
    except Exception as e:
        r.fail("G6 percentile reuse", str(e))


def test_fixgate_g6_percentiles_still_agrees_with_compile_modes_bench_stats(r: SubTestResult):
    """Behaviour-preserving: since both formulas were already identical (R1#3), sharing one
    must not move any published number -- same nearest-rank percentile on the same series."""
    print("\n--- G6: sharing the formula does not change any percentile value ---")
    try:
        alb = _load_artist_loops_bench()
        cmb = alb._load("compile_modes_bench", "compile_modes_bench.py")
        xs = [12.3, 4.5, 88.1, 9.9, 15.0, 7.7, 30.2, 1.1, 60.0, 5.5]
        got = alb._percentiles(xs)
        want = cmb._stats(xs)
        assert got["p50_ms"] == want["p50"], (got, want)
        assert got["p95_ms"] == want["p95"], (got, want)
        assert got["p99_ms"] == want["p99"], (got, want)
        r.ok("artist_loops_bench and compile_modes_bench agree on every percentile")
    except Exception as e:
        r.fail("G6 percentile parity", str(e))
