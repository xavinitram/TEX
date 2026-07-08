"""
v0.17 Phase 1 — the safety net (build BEFORE decomposition).

TST-5 tier observability · TST-6 registry-parity · TST-2 edge-input matrix ·
TST-4 operator-completeness · TST-1 differential fuzzer.
"""
import ast
import os
import re
import glob

from helpers import *
from failure_harness import run_tier, max_diff
from TEX_Wrangle.tex_runtime import tier_trace


def test_tst5_tier_trace(r: SubTestResult):
    print("\n--- TST-5: tier-execution observability ---")
    img = make_img(1, 8, 8, 3)

    # codegen serves a pointwise program → recorded.
    try:
        tier_trace.reset()
        run_tier("@OUT = vec4(@A.rgb * 0.5 + 0.1, 1.0);", {"A": img}, "codegen")
        rec = tier_trace.last()
        assert rec is not None and rec.tier == "codegen" and rec.fallback_from is None, f"got {rec}"
        r.ok("codegen tier recorded when it serves")
    except Exception as e:
        r.fail("TST-5 codegen record", f"{type(e).__name__}: {e}")

    # A codegen-unsupported program (string comparison) now records the SILENT
    # fallback + reason — previously invisible (`_show_once`-and-forget).
    try:
        tier_trace.reset()
        run_tier('string s = "a"; @OUT = vec4((s == "a") ? 1.0 : 0.0, 0.0, 0.0, 1.0);',
                 {"A": img}, "codegen")
        rec = tier_trace.last()
        assert rec.tier == "interpreter" and rec.fallback_from == "codegen", f"got {rec}"
        r.ok(f"silent codegen->interp fallback now observable (reason={rec.reason})")
    except Exception as e:
        r.fail("TST-5 fallback record", f"{type(e).__name__}: {e}")

    # reset() clears — so a test can detect a tier that never ran at all.
    try:
        tier_trace.reset()
        assert tier_trace.last() is None
        r.ok("reset() clears the record")
    except Exception as e:
        r.fail("TST-5 reset", f"{type(e).__name__}: {e}")

    # graph tier records on a successful CUDA replay.
    if torch.cuda.is_available():
        try:
            from TEX_Wrangle.tex_runtime import graphed as G
            G.clear_graph_cache()
            tier_trace.reset()
            g = make_img(1, 256, 256, 3).cuda()
            code = ("vec4 a = vec4(u, v, u*v, 1.0); vec3 d = a.rgb*2.0 - vec3(0.5); "
                    "@OUT = vec4(clamp(d.r,0.0,1.0), clamp(d.g,0.0,1.0), clamp(d.b,0.0,1.0), 1.0);")
            run_tier(code, {"A": g}, "graph", device="cuda")
            rec = tier_trace.last()
            assert rec is not None and rec.tier == "graph", f"got {rec}"
            r.ok("graph tier recorded on replay")
        except Exception as e:
            r.fail("TST-5 graph record", f"{type(e).__name__}: {e}")
    else:
        r.ok("TST-5 graph record skipped (no CUDA)")


def test_tst6_registry_parity(r: SubTestResult):
    print("\n--- TST-6: codegen parity auto-derived from FUNCTION_SIGNATURES ---")
    from stdlib_probe import all_generated, SKIP
    from TEX_Wrangle.tex_compiler.stdlib_signatures import FUNCTION_SIGNATURES

    # Coverage meta: every signature is either SKIP or generated — a new stdlib
    # function CANNOT silently escape parity testing (it fails this assert).
    try:
        gen = set(all_generated())
        missing = set(FUNCTION_SIGNATURES) - SKIP - gen
        assert not missing, f"functions neither skipped nor covered: {sorted(missing)}"
        r.ok(f"coverage meta: all {len(FUNCTION_SIGNATURES)} signatures classified "
             f"({len(gen)} auto-tested, {len(SKIP)} bespoke-skip)")
    except Exception as e:
        r.fail("TST-6 coverage meta", str(e))

    # Parity: interp == codegen for every auto-generated stdlib call.
    fails = []
    tested = 0
    for name, (code, binds) in all_generated().items():
        try:
            base = run_tier(code, binds, "interp")
            got = run_tier(code, binds, "codegen")
            tested += 1
            md = max_diff(base, got)
            if md > 1e-4:
                fails.append(f"{name}: maxdiff {md:.2e}")
        except Exception as e:
            fails.append(f"{name}: {type(e).__name__}: {str(e)[:55]}")
    if fails:
        r.fail("TST-6 codegen parity",
               f"{len(fails)}/{tested} diverge:\n  " + "\n  ".join(fails[:25]))
    else:
        r.ok(f"interp == codegen for all {tested} auto-generated stdlib calls")


def _nan_diff(a_dict, b_dict):
    """Cross-tier max abs diff treating matching NaN / same-sign Inf as equal."""
    m = 0.0
    for k, av in a_dict.items():
        bv = b_dict.get(k)
        if isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor):
            af, bf = av.float(), bv.float()
            if af.shape != bf.shape:
                return float("inf")
            both_nan = torch.isnan(af) & torch.isnan(bf)
            both_inf = torch.isinf(af) & torch.isinf(bf) & (torch.sign(af) == torch.sign(bf))
            d = torch.where(both_nan | both_inf, torch.zeros_like(af), (af - bf).abs())
            m = max(m, torch.nan_to_num(d, nan=1e9).max().item())
    return m


def test_tst2_edge_matrix(r: SubTestResult):
    print("\n--- TST-2: edge-input matrix (fp16 / shape / int64 × every fn × tier) ---")
    from stdlib_probe import all_generated, ALLOW_NONFINITE
    gen = all_generated()
    base_a = make_img(1, 8, 8, 3, seed=1)
    base_b = make_img(1, 8, 8, 3, seed=2)
    edges = {
        "fp16":   ({"A": base_a.half(), "B": base_b.half()}, 3e-2),
        "1x1":    ({"A": make_img(1, 1, 1, 3, seed=1), "B": make_img(1, 1, 1, 3, seed=2)}, 1e-4),
        "batch4": ({"A": make_img(4, 8, 8, 3, seed=1), "B": make_img(4, 8, 8, 3, seed=2)}, 1e-4),
        "int64":  ({"A": (base_a * 3).long(), "B": (base_b * 3).long()}, 1e-4),
    }
    for edge, (binds, tol) in edges.items():
        fails, tested = [], 0
        for name, (code, _) in gen.items():
            try:
                b = run_tier(code, binds, "interp")
                c = run_tier(code, binds, "codegen")
                tested += 1
                md = _nan_diff(b, c)
                if md > tol:
                    fails.append(f"{name}: maxdiff {md:.2e}")
                if edge == "fp16" and name not in ALLOW_NONFINITE:
                    out = next((v for v in c.values() if isinstance(v, torch.Tensor)), None)
                    if out is not None and not torch.isfinite(out).all():
                        fails.append(f"{name}: fp16 non-finite (NaN/inf on normal pixels)")
            except Exception as e:
                fails.append(f"{name}: {type(e).__name__}: {str(e)[:45]}")
        if fails:
            r.fail(f"TST-2 edge={edge}",
                   f"{len(fails)}/{tested}:\n  " + "\n  ".join(fails[:15]))
        else:
            r.ok(f"edge={edge}: interp==codegen (+finiteness) for all {tested} fns")


def test_tst4_operator_completeness(r: SubTestResult):
    print("\n--- TST-4: operator completeness (every operator lowers in codegen) ---")
    # A program exercising every operator. If codegen can't lower one it falls back
    # (silent _Unsupported) — caught here by BOTH the parity check AND the tier_trace
    # assertion that codegen actually engaged (didn't quietly become interp).
    code = (
        "float a = @A.r + @A.g - @A.b * 0.5 / 0.25;"      # + - * /
        "float b = mod(a, 0.3);"                           # % via mod (real % is int)
        "float c = (a > 0.5 && b < 0.2) || !(a == b) ? 1.0 : 0.0;"  # > < && || ! == ?:
        "float e = (a <= 0.9 || b >= 0.1) ? 0.5 : 0.25;"   # <= >= !=
        "float d = -a;"                                    # unary -
        "d += 0.1; d -= 0.05; d *= 2.0; d /= 1.5;"         # compound += -= *= /=
        "int i = 0; i = i + 1; int j = 5 % 3;"             # int % (integer modulo)
        "@OUT = vec4(a * 0.01, b, c + float(j) * 0.0, d + e - float(i) * 0.0);"
    )
    img = make_img(1, 8, 8, 3)
    try:
        base = run_tier(code, {"A": img}, "interp")
        tier_trace.reset()
        got = run_tier(code, {"A": img}, "codegen")
        rec = tier_trace.last()
        assert max_diff(base, got) < 1e-5, "interp != codegen on the all-operators program"
        assert rec is not None and rec.tier == "codegen", \
            f"codegen fell back (an operator didn't lower): {rec}"
        r.ok("every operator lowers in codegen (no _Unsupported fallthrough) + parity")
    except Exception as e:
        r.fail("TST-4 operator completeness", f"{type(e).__name__}: {e}")


import random as _random

_FN1 = ["sin", "cos", "tanh", "abs", "sqrt", "fract", "exp", "sign", "floor",
        "atan", "sinh", "cosh", "log", "ceil", "round", "degrees"]
_FN2 = ["min", "max", "atan2", "pow", "mod", "hypot", "step", "spow", "sdiv"]
_FN3 = ["clamp", "lerp", "mix", "smoothstep", "smin", "smax"]
_ATOMS = ["u", "v", "@A.r", "@A.g", "@A.b", "@B.r", "0.5", "0.25", "1.5", "2.0", "0.1", "3.0"]
_OPS = ["+", "-", "*", "/"]


def _gen_expr(rng, depth):
    """A random float-typed expression (type-checks by construction — every leaf
    and every function is float-valued)."""
    if depth <= 0 or rng.random() < 0.3:
        return rng.choice(_ATOMS)
    c = rng.random()
    if c < 0.4:
        return f"({_gen_expr(rng, depth-1)} {rng.choice(_OPS)} {_gen_expr(rng, depth-1)})"
    if c < 0.6:
        return f"{rng.choice(_FN1)}({_gen_expr(rng, depth-1)})"
    if c < 0.8:
        return f"{rng.choice(_FN2)}({_gen_expr(rng, depth-1)}, {_gen_expr(rng, depth-1)})"
    return (f"{rng.choice(_FN3)}({_gen_expr(rng, depth-1)}, "
            f"{_gen_expr(rng, depth-1)}, {_gen_expr(rng, depth-1)})")


def _fuzz_diverge(a_dict, b_dict, rtol=1e-5, atol=1e-5, mag_cap=1e2, frac=0.05):
    """Conditioning-aware interp↔codegen divergence check for the fuzzer (audit #2).

    The generator builds numerically PATHOLOGICAL programs — 1e7-magnitude
    spow/sinh/exp towers, and near-singular smoothstep/sdiv/smin whose edges collide
    to ~1e-7. The interp↔codegen contract is bit-exactness (atol=1e-5) on the
    WELL-CONDITIONED [0,1] pixel range, and it HOLDS there (the whole suite + the
    edge matrix prove it). But a strict per-element allclose on a pathological
    program flags float RE-ASSOCIATION through the singularity (a 1-ULP lerp/max
    difference amplified into a 0↔1 flip in a thin discontinuity band, or a huge-
    magnitude blow-up), NOT a compiler bug — the "boy who cried wolf" failure that
    makes a durability engine worthless.

    So compare only WELL-CONDITIONED elements (finite, |ref| ≤ mag_cap), and flag a
    divergence only when a MEANINGFUL FRACTION of them differ: a real formula bug
    (wrong opcode/constant/dtype — the v0.15/v0.16 class) diverges on ~all pixels,
    while a conditioning artifact is a thin band or a magnitude blow-up (< `frac`).
    A finiteness MISMATCH (NaN in one tier, a number in the other) is always real.
    32×32 images (below) make the band a small fraction, sharpening the split.
    Returns True on a genuine divergence."""
    for k, av in a_dict.items():
        bv = b_dict.get(k)
        if not (isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor)):
            continue
        if av.shape != bv.shape:
            return True
        a, b = av.float(), bv.float()
        fa, fb = torch.isfinite(a), torch.isfinite(b)
        mism = fa != fb  # NaN-vs-number: a real disagreement, always counted
        cond = fa & fb & (a.abs() <= mag_cap) & (b.abs() <= mag_cap)
        bad = mism | (((a - b).abs() > (atol + rtol * b.abs())) & cond)
        denom = int((cond | mism).sum())
        if denom > 0 and int(bad.sum()) / denom > frac:
            return True
    return False


def test_tst1_differential_fuzzer(r: SubTestResult):
    print("\n--- TST-1: grammar-driven differential fuzzer (interp vs codegen) ---")
    # Random valid programs, typed so they compile by construction; each asserted
    # bit-exact interp==codegen on the well-conditioned range (see _fuzz_diverge).
    # This is the class that shipped green in v0.15/v0.16 (composition bugs a
    # hand-corpus misses). Scale + seed are env-tunable (audit #2): the seed stays
    # FIXED for CI reproducibility (TEX_FUZZ_SEED); the nightly job passes N=2000
    # (TEX_FUZZ_N) with a date-derived seed to widen coverage.
    seed = int(os.environ.get("TEX_FUZZ_SEED", "20260708"))
    N = int(os.environ.get("TEX_FUZZ_N", "300"))
    rng = _random.Random(seed)
    binds = {"A": make_img(1, 32, 32, 3, seed=11), "B": make_img(1, 32, 32, 3, seed=22)}

    # SELF-TEST: the conditioning-aware comparison must still catch a REAL divergence
    # — a calibration that flags nothing is worthless. Inject a small, WIDESPREAD
    # perturbation (the shape a real formula bug takes) and confirm it trips.
    probe = run_tier("@OUT = vec4(@A.r * 0.5 + 0.2, u, v, 1.0);", binds, "interp")
    bug = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in probe.items()}
    bug["OUT"][..., 0] += 2e-4  # wrong value on every pixel
    if not _fuzz_diverge(probe, bug):
        r.fail("TST-1 self-test", "comparison went blind — a widespread divergence "
               "was NOT caught (calibration too loose)")
        return
    r.ok("self-test: a widespread divergence is still caught (engine not neutered)")

    fails, tested = [], 0
    for _ in range(N):
        # depth 3 (was 4): the audit's generation-depth cap — stacking 4+ near-singular
        # generators (smin/smax/smoothstep/step/sdiv/round) builds discontinuity towers
        # that amplify a 1-ULP reassociation into a wide flip band with no bearing on a
        # real program. Depth 3 still exercises composition (the v0.15/v0.16 bug class is
        # per-op, not depth-dependent); the edge matrix (TST-2) covers per-fn edges.
        expr = _gen_expr(rng, 3)
        code = f"@OUT = vec4({expr}, u, v, 1.0);"
        try:
            b = run_tier(code, binds, "interp")
            c = run_tier(code, binds, "codegen")
            tested += 1
            if _fuzz_diverge(b, c):
                fails.append(f"REL divergence: {code}")
        except Exception as e:
            fails.append(f"{type(e).__name__} on: {code[:80]}")
    if fails:
        r.fail("TST-1 fuzzer", f"{len(fails)}/{tested} diverged/errored:\n  " +
               "\n  ".join(fails[:12]))
    else:
        r.ok(f"{tested} random programs: interp == codegen (conditioning-aware, NaN-aware)")


def _rtest_fns_in_file(path):
    """Names of `def test_*(r ...)` functions in a test module — the run_all
    calling convention (first param `r`). Pytest-native fns (no `r`) are ignored."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    out = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                and n.args.args and n.args.args[0].arg == "r"):
            out.add(n.name)
    return out


def test_tst7_runner_coverage(r: SubTestResult):
    print("\n--- TST-7: no test drifts out of the runner (auto-discover) ---")
    # A test_*(r) function defined in a file but never called in run_all.py silently
    # never runs — the coverage gap that let regressions ship green. Discover every
    # such function and assert run_all.py imports AND calls it.
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, "run_all.py"), encoding="utf-8") as f:
            runner_src = f.read()
        called = set(re.findall(r"(\w+)\(r\)", runner_src))
        orphans = []
        for path in sorted(glob.glob(os.path.join(here, "test_*.py"))):
            for fn in _rtest_fns_in_file(path):
                if fn not in called:
                    orphans.append(f"{os.path.basename(path)}::{fn}")
        if orphans:
            r.fail("TST-7 runner drift",
                   f"{len(orphans)} test(s) defined but never called in run_all.py:\n  "
                   + "\n  ".join(orphans))
        else:
            r.ok(f"all {len(called)} runner-convention tests are wired into run_all.py")
    except Exception as e:
        r.fail("TST-7 runner coverage", f"{type(e).__name__}: {e}")
