"""
PR-LP5 — the determinism pin (doc 28 §2.2).

TEX is measured **bitwise run-to-run deterministic on CUDA** across every program
class, including scatter `index_put_(accumulate=True)` atomics under heavy collision
stress. Determinism is therefore a FREE, marketable property — forcing strict
determinism would actually *cost* 1.48x on scatter, so TEX already rides the fast
deterministic-in-practice path.

The one genuine exposure is that torch does not *promise* atomic-add ordering, so a
future torch upgrade could break run-to-run scatter determinism. This test pins that:
- CUDA scatter (plain + 1024-way collision stress) must be bitwise identical across 5
  runs. **GATED on a band by default** (A1-4, v0.19: was WARN-first in v0.18): measured
  bitwise-0.0, and a torch atomic-add reorder is a discrete jump well above the 1e-9 band,
  so gating catches a real regression without flapping. Set `TEX_DETERMINISM_SOFT=1` to
  DOWNGRADE to a warning (the old behavior) if a future torch upgrade forces a re-band.
- The CPU is the honest caveat: threaded float accumulation is run-to-run
  nondeterministic at ~5.5e-6. We assert it stays within a loose 1e-5 sanity bound
  (a real red if it blows up), documenting rather than fixing it (pin, don't chase).

CUDA-gated: the CUDA half skips clean on a CPU-only box.
"""
from helpers import *

# plain displacement scatter, and a collision-stress variant (128^2 sources folded
# into a 4x4 target grid => ~1024-way atomic-add collisions per cell).
_SCATTER = (
    "@OUT[ix, iy] = vec3(0.0);"
    "float dx = simplex(u * 3.0, v * 3.0) * 8.0;"
    "float dy = simplex(u * 3.0 + 5.3, v * 3.0 + 7.1) * 8.0;"
    "int tx = int(ix + dx); int ty = int(iy + dy);"
    "@OUT[tx, ty] += vec3(0.1);"
)
_COLLIDE = (
    "@OUT[ix, iy] = vec3(0.0);"
    "int tx = int(u * 4.0); int ty = int(v * 4.0);"
    "@OUT[tx, ty] += vec3(0.01);"
)


def _run(code, device):
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    tm = TypeChecker(binding_types={}, source=code).check(prog)
    return Interpreter().execute(prog, {}, tm, device=device,
                                 output_names=["OUT"], precision="fp32")["OUT"]


def _max_run_to_run(code, device, runs=5):
    ref = _run(code, device)
    worst = 0.0
    for _ in range(runs - 1):
        cur = _run(code, device)
        worst = max(worst, (cur.float() - ref.float()).abs().max().item())
    return worst


# A1-4: CUDA scatter determinism is now GATED on a band (was WARN-first). Measured
# bitwise-0.0; a torch atomic-add reorder is a discrete jump well above this band, so
# gating catches a real regression without flapping. TEX_DETERMINISM_SOFT=1 downgrades
# to WARN (inverse of the old HARD flag) if an environment ever flaps in practice. The
# release gate (TST-8) also reads LAST_CUDA_DET_VAR to fail a publish on out-of-band drift.
_CUDA_DET_BAND = 1e-9
LAST_CUDA_DET_VAR = None   # worst CUDA run-to-run measured; None = not measured (CPU box)


def test_prlp5_determinism_pin(r: SubTestResult):
    print("\n--- PR-LP5: determinism pin (CUDA gated band; CPU documented caveat) ---")
    global LAST_CUDA_DET_VAR
    soft = os.environ.get("TEX_DETERMINISM_SOFT") == "1"

    # CPU caveat: threaded accumulation is nondeterministic — pin the loose bound.
    try:
        cpu_var = _max_run_to_run(_SCATTER, "cpu")
        if cpu_var > 1e-5:
            r.fail("PR-LP5 CPU variance", f"CPU scatter run-to-run {cpu_var:.2e} > 1e-5 "
                   "sanity bound (expected ~5.5e-6)")
        else:
            r.ok(f"CPU scatter run-to-run within caveat bound ({cpu_var:.1e} <= 1e-5)")
    except Exception as e:
        r.fail("PR-LP5 CPU variance", f"{type(e).__name__}: {e}")

    if not torch.cuda.is_available():
        r.ok("PR-LP5 CUDA bitwise determinism (no GPU, SKIPPED)")
        return

    worst_all = 0.0
    for label, code in (("scatter", _SCATTER), ("collision-stress", _COLLIDE)):
        try:
            worst = _max_run_to_run(code, "cuda")
            worst_all = max(worst_all, worst)
            if worst <= _CUDA_DET_BAND:
                r.ok(f"CUDA {label}: run-to-run {worst:.1e} within gated band "
                     f"(<= {_CUDA_DET_BAND:.0e})")
            elif soft:
                r.ok(f"[WARN] CUDA {label} run-to-run {worst:.2e} > band — torch atomic-add "
                     "ordering changed (soft mode; suite stays green). Decide: scoped "
                     "use_deterministic_algorithms (~1.48x) vs re-band.")
            else:
                r.fail(f"PR-LP5 CUDA {label}",
                       f"run-to-run {worst:.2e} > band {_CUDA_DET_BAND:.0e} — atomic-add "
                       "ordering regressed. Fix (scoped use_deterministic_algorithms, "
                       "~1.48x) or consciously re-band; TEX_DETERMINISM_SOFT=1 to downgrade.")
        except Exception as e:
            r.fail(f"PR-LP5 CUDA {label}", f"{type(e).__name__}: {e}")
    LAST_CUDA_DET_VAR = worst_all  # release gate (TST-8) reads this
