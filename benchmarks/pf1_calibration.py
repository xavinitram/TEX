#!/usr/bin/env python3
"""
HW-1 — PF-1 crossover-gate drift canary.

The cuda_graph tier only wins for enough-kernels at low-enough-resolution; that region is
encoded in four hardcoded constants (graphed.py: _GRAPH_MIN_OPS / _GRAPH_HIGH_OPS /
_GRAPH_BASE_PX_CEIL / _GRAPH_HIGH_PX_CEIL). This script re-measures the 4-corner matrix
(low/high op-count x low/high resolution) on THIS box and REPORTS whether the gate's
win/lose prediction still matches measured reality — a drift canary for a torch/driver
bump. **The constants stay the contract**: the script reports (and, under opt-in
TEX_PF1_AUTOCAL=1, writes SUGGESTED capped overrides to results/pf1_autocal.json for a
human to apply) — it never silently changes the gate.

    python benchmarks/pf1_calibration.py            # report fit (CUDA-only)
    TEX_PF1_AUTOCAL=1 python benchmarks/pf1_calibration.py   # + write suggested overrides
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

_b = Path(__file__).resolve().parent
sys.path.insert(0, str(_b.parent.parent))
sys.path.insert(0, str(_b))
import torch
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, _collect_identifiers
from TEX_Wrangle.tex_runtime import graphed as G

# low-op (~3 kernels) and high-op (>=40 kernels) pointwise programs
_LOW = "@OUT = vec4(sin(@A.rgb) * 0.5 + 0.5, 1.0);"
_HIGH = "vec3 c = @A.rgb;" + "".join(f"c = c * 1.01 + sin(c) * 0.001;" for _ in range(20)) + \
        "@OUT = vec4(c, 1.0);"
_CAP_PX = (256 * 256, 2048 * 2048)   # autocal clamp on resolution ceilings
_CAP_OPS = (2, 64)                   # autocal clamp on op thresholds


def _compile(code):
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    tm = TypeChecker(binding_types={"A": TEXType.VEC3}, source=code).check(prog)
    return prog, tm, _collect_identifiers(prog)


def _time(fn, iters=25):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        s.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(s)


def _corner(code, res, label):
    prog, tm, used = _compile(code)
    img = torch.rand(1, res, res, 3, device="cuda")
    G.clear_graph_cache()
    interp = Interpreter()
    t_interp = _time(lambda: interp.execute(prog, {"A": img}, tm, device="cuda",
                                            output_names=["OUT"], used_builtins=used))
    # warm the graph, then time replay
    G.run_graphed(prog, {"A": img}, tm, "cuda", f"pf1_{label}",
                  output_names=["OUT"], used_builtins=used)
    def _replay():
        return G.run_graphed(prog, {"A": img}, tm, "cuda", f"pf1_{label}",
                             output_names=["OUT"], used_builtins=used)
    graphed_out = _replay()
    est_ops = G._capturable(prog)[1]
    predicted_win = G._graph_capture_worthwhile(est_ops, res * res)
    if graphed_out is None:                    # gate declined -> no graph
        return {"corner": label, "res": res, "ops": est_ops, "interp_ms": round(t_interp, 3),
                "graph_ms": None, "measured_win": False, "predicted_win": predicted_win,
                "match": (predicted_win is False)}
    t_graph = _time(_replay)
    measured_win = t_graph < t_interp
    return {"corner": label, "res": res, "ops": est_ops, "interp_ms": round(t_interp, 3),
            "graph_ms": round(t_graph, 3), "speedup": round(t_interp / t_graph, 2),
            "measured_win": measured_win, "predicted_win": predicted_win,
            "match": measured_win == predicted_win}


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable — PF-1 calibration is CUDA-only. SKIP.")
        return
    rows = []
    for code, ops_label in ((_LOW, "low-ops"), (_HIGH, "high-ops")):
        for res in (512, 1024):
            rows.append(_corner(code, res, f"{ops_label}@{res}"))
    print(f"{'corner':>16} | {'ops':>4} | {'interp':>8} | {'graph':>8} | pred/meas | fit")
    print("-" * 70)
    for row in rows:
        g = f"{row['graph_ms']:.3f}" if row["graph_ms"] is not None else "declined"
        print(f"{row['corner']:>16} | {row['ops']:>4} | {row['interp_ms']:>8.3f} | {g:>8} | "
              f"{str(row['predicted_win'])[0]}/{str(row['measured_win'])[0]} | "
              f"{'OK' if row['match'] else 'DRIFT'}")
    drift = [r for r in rows if not r["match"]]
    print(f"\ngate fit: {len(rows) - len(drift)}/{len(rows)} corners match the constants"
          + ("" if not drift else f"  (DRIFT at: {[r['corner'] for r in drift]})"))
    print("Constants stay the contract; a DRIFT is a signal to re-measure + decide, "
          "not an auto-change.")
    if os.environ.get("TEX_PF1_AUTOCAL") == "1":
        # suggest capped overrides for a human to apply (never auto-applied)
        sug = {"note": "SUGGESTED only — apply by hand; constants are the contract",
               "_GRAPH_BASE_PX_CEIL": max(_CAP_PX[0], min(_CAP_PX[1], G._GRAPH_BASE_PX_CEIL)),
               "_GRAPH_HIGH_PX_CEIL": max(_CAP_PX[0], min(_CAP_PX[1], G._GRAPH_HIGH_PX_CEIL)),
               "_GRAPH_MIN_OPS": max(_CAP_OPS[0], min(_CAP_OPS[1], G._GRAPH_MIN_OPS)),
               "_GRAPH_HIGH_OPS": max(_CAP_OPS[0], min(_CAP_OPS[1], G._GRAPH_HIGH_OPS)),
               "corners": rows}
        p = _b / "results" / "pf1_autocal.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(sug, indent=2), encoding="utf-8")
        print(f"TEX_PF1_AUTOCAL: wrote suggested (capped) overrides to {p}")


if __name__ == "__main__":
    main()
