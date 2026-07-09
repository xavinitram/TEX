"""
v0.18.0 precision core (PR-LP4 fp16-safe reductions; PR-LP2 auto mode lands here too).
"""
from helpers import *
import random as _random
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib as _S
from TEX_Wrangle.tex_runtime.codegen import try_compile
from TEX_Wrangle.tex_runtime.precision_policy import resolve_auto_precision

_EX = Path(__file__).resolve().parent.parent / "examples"
_FP16_BAR = 3.9e-3  # the 8-bit quantum (doc 22)


def _codegen_src(code, bt):
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    fn = try_compile(prog, tm, fingerprint="lp4probe")
    return getattr(fn, "_tex_src", "") if fn is not None else ""


def test_prlp4_fp16_safe_reductions(r: SubTestResult):
    print("\n--- PR-LP4: fp16-safe reductions (fp32 accumulate; no inf) ---")
    fails = []

    # (a) interp: every reduction on a large-magnitude fp16 image is FINITE and fp32.
    #     ~500-valued pixels summed over 64^2 = ~2e6 >> fp16 max 65504 — an fp16 sum
    #     overflows to inf without the .float() accumulate.
    img16 = torch.full((1, 64, 64, 3), 500.0, dtype=torch.float16)
    fns = {"img_sum": _S.fn_img_sum, "img_mean": _S.fn_img_mean, "img_min": _S.fn_img_min,
           "img_max": _S.fn_img_max, "img_median": _S.fn_img_median}
    for name, fn in fns.items():
        out = fn(img16)
        if not torch.isfinite(out).all():
            fails.append(f"{name}: fp16 result non-finite (overflow to inf)")
        elif out.dtype != torch.float32:
            fails.append(f"{name}: returned {out.dtype}, expected fp32 (would reintroduce inf)")

    # (b) codegen mirror: the emitted reduce must .float() BEFORE the op (invariant #2
    #     — interp and codegen must be bit-identical; a missing mirror would let fp16
    #     codegen overflow where the interp doesn't). Assert on the generated source.
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    for name, op in (("img_sum", "sum"), ("img_mean", "mean"),
                     ("img_min", "amin"), ("img_max", "amax")):
        code = f"float s = {name}(@A).r; @OUT = vec4(@A.rgb * (s * 0.001), 1.0);"
        src = _codegen_src(code, bt)
        if not src:
            fails.append(f"{name}: codegen produced no source (unexpected fallback)")
        elif f".float().{op}(" not in src:
            fails.append(f"{name}: codegen source lacks `.float().{op}(` — fp16 mirror missing")

    if fails:
        r.fail("PR-LP4 fp16 reductions", "; ".join(fails))
    else:
        r.ok(f"all {len(fns)} reductions fp16-finite+fp32 (interp); codegen mirrors .float()")


def test_prlp4_arr_reductions_fp16_safe(r: SubTestResult):
    print("\n--- PR-LP4: arr_* reductions fp32-accumulate (audit) ---")
    from failure_harness import run_tier, max_diff
    binds = {"A": make_img(1, 8, 8, 3, seed=1)}
    # 20-element array of 5000 -> sum 1e5 > fp16 max 65504: arr_sum/arr_avg overflow in
    # fp16 without the .float() accumulate. The auto gate ACCEPTS such a program (arrays
    # are tile-safe, arr_* not fragile), so the overflow must be fixed, not just caught.
    fails = []
    for name in ("arr_sum", "arr_avg", "arr_min", "arr_max", "median"):
        code = (f"float arr[20]; for(int i=0;i<20;i++){{ arr[i]=5000.0; }} "
                f"@OUT = vec4(vec3({name}(arr) * 0.00001), 1.0);")
        for tier in ("interp", "codegen"):
            try:
                out = run_tier(code, binds, tier, precision="fp16")
                t = next(v for v in out.values() if isinstance(v, torch.Tensor))
                if not torch.isfinite(t).all():
                    fails.append(f"{name}/{tier} fp16 non-finite (overflow)")
            except Exception as e:
                fails.append(f"{name}/{tier}: {type(e).__name__}: {e}")
        try:
            a = run_tier(code, binds, "interp", precision="fp32")
            c = run_tier(code, binds, "codegen", precision="fp32")
            if max_diff(a, c) > 1e-5:
                fails.append(f"{name} fp32 interp!=codegen {max_diff(a, c):.2e}")
        except Exception as e:
            fails.append(f"{name} fp32 parity: {e}")
    if fails:
        r.fail("PR-LP4 arr_* fp16", "; ".join(fails))
    else:
        r.ok("all 5 arr_* reductions fp16-finite (interp+codegen) + fp32 bit-exact")


def test_prlp2_node_path_perf(r: SubTestResult):
    print("\n--- PR-LP2: precision=auto node-path speedup (H7 / audit B1) ---")
    if not torch.cuda.is_available():
        r.ok("PR-LP2 node-path perf (no GPU, SKIPPED)")
        return
    import importlib.util
    bench = Path(__file__).resolve().parent.parent / "benchmarks" / "prlp2_node_path.py"
    spec = importlib.util.spec_from_file_location("prlp2_node_path", bench)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    # Substantiates the CHANGELOG claim on the REAL path (TEXWrangleNode.execute), not
    # Interpreter.execute. Thresholds are drift-robust (measured after B1: 1.33x@1024,
    # 1.45x@2048): >=0.9x@1024 catches a re-introduced regression (the B1 bug was 0.79x);
    # >=1.15x@2048 asserts a genuine speedup (20% margin under the measured 1.45x).
    fails = []
    for res, floor in ((1024, 0.9), (2048, 1.15)):
        f32 = mod._time(res, "fp32", iters=15)
        auto = mod._time(res, "auto", iters=15)
        speedup = f32 / auto
        if speedup < floor:
            fails.append(f"{res}²: auto {speedup:.2f}× < {floor}× (node-path B1 regression)")
    if fails:
        r.fail("PR-LP2 node-path perf", "; ".join(fails))
    else:
        r.ok("precision=auto is a node-path speedup (>=0.9x@1024, >=1.15x@2048) — not the "
             "0.79x/1.02x pre-B1 regression")


def _resolve(code, px=2048 * 2048, dev="cuda"):
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    return resolve_auto_precision(prog, px, dev)


def test_prlp2_auto_gate(r: SubTestResult):
    print("\n--- PR-LP2: precision='auto' gate (decline the fragile, accept the smooth) ---")
    fails = []
    # (a) the 4 named programs MUST be declined (doc 28 success test)
    for name in ("lens_distortion", "caustics", "auto_levels", "halftone"):
        p, why = _resolve((_EX / f"{name}.tex").read_text(encoding="utf-8"))
        if p != "fp32":
            fails.append(f"{name}: gate accepted fp16 (must decline) — {why}")
    # (b) smooth pointwise programs MUST be accepted at >=1024^2 CUDA
    accept = {
        "grade": "vec3 c=@A.rgb; c=pow(c,vec3(0.4545)); @OUT=vec4(lerp(c,vec3(0.5),0.2),1.0);",
        "mat3": "mat3 m=mat3(0.4,0.8,0.2,0.3,0.7,0.2,0.3,0.5,0.1); @OUT=vec4(m*@A.rgb,1.0);",
        "invert": "@OUT=vec4(1.0-@A.rgb,1.0);",
        "channel_mix": "@OUT=vec4(@A.r*0.6+@A.g*0.4, @A.g, @A.b, 1.0);",
    }
    for name, code in accept.items():
        p, why = _resolve(code)
        if p != "fp16":
            fails.append(f"{name}: gate declined a smooth pointwise program — {why}")
    # (c) determinism: a fixed program resolves identically twice (tier-trace assertion)
    if _resolve(accept["grade"])[0] != _resolve(accept["grade"])[0]:
        fails.append("gate decision is non-deterministic for a fixed program")
    # (d) region edges: CPU and <1024^2 always decline
    if _resolve(accept["grade"], dev="cpu")[0] != "fp32":
        fails.append("CPU not declined")
    if _resolve(accept["grade"], px=512 * 512)[0] != "fp32":
        fails.append("<1024^2 not declined")
    if fails:
        r.fail("PR-LP2 auto gate", "; ".join(fails))
    else:
        r.ok("gate declines lens/caustics/auto_levels/halftone; accepts smooth pointwise; "
             "deterministic; CPU/<1024^2 -> fp32")


# fp16-safe (smooth) function set — the class the gate accepts. NO floor/sqrt/acos/
# fract/step/smoothstep (those are declined) — this arm checks that ACCEPTED-class
# programs stay within the 8-bit quantum in fp16.
_SAFE_ATOMS = ["u", "v", "@A.r", "@A.g", "@A.b", "0.5", "0.25", "1.5", "0.1"]
_SAFE_OPS = ["+", "-", "*"]
# Expressive, fp16-HOSTILE grammar (floor/sqrt/exp/smin/pow-variable/comparisons). The
# fuzzer runs each program THROUGH THE GATE and only accuracy-tests the ACCEPTED ones —
# directly validating "everything precision=auto accepts is within the 8-bit quantum".
_GEN_FN1 = ["sin", "cos", "tanh", "abs", "sqrt", "exp", "floor", "fract", "sign"]
_GEN_FN2 = ["min", "max", "pow", "hypot", "mod", "smin", "smax", "step"]
_GEN_FN3 = ["clamp", "lerp", "mix", "smoothstep"]


def _gen(rng, depth):
    if depth <= 0 or rng.random() < 0.3:
        return rng.choice(_SAFE_ATOMS)
    c = rng.random()
    if c < 0.4:
        return f"({_gen(rng, depth-1)} {rng.choice(_SAFE_OPS)} {_gen(rng, depth-1)})"
    if c < 0.6:
        return f"{rng.choice(_GEN_FN1)}({_gen(rng, depth-1)})"
    if c < 0.8:
        return f"{rng.choice(_GEN_FN2)}({_gen(rng, depth-1)}, {_gen(rng, depth-1)})"
    return (f"{rng.choice(_GEN_FN3)}({_gen(rng, depth-1)}, "
            f"{_gen(rng, depth-1)}, {_gen(rng, depth-1)})")


def test_prlp2_fp16_accuracy_fuzzer(r: SubTestResult):
    print("\n--- PR-LP2: fp16-auto accuracy fuzzer (gate-filtered; band 3.9e-3) ---")
    if not torch.cuda.is_available():
        r.ok("PR-LP2 fp16 fuzzer (no GPU, SKIPPED)")
        return
    from failure_harness import run_tier
    N = int(os.environ.get("TEX_FP16_FUZZ_N", "500"))
    seed = int(os.environ.get("TEX_FP16_FUZZ_SEED", "20260708"))
    rng = _random.Random(seed)
    binds = {"A": make_img(1, 32, 32, 3, seed=4)}
    fails, accepted, declined, fell_back = [], 0, 0, 0
    for _ in range(N):
        code = f"@OUT = vec4(vec3({_gen(rng, 3)}), 1.0);"
        try:
            prog = Parser(Lexer(code).tokenize(), source=code).parse()
        except Exception:
            continue
        prec, _why = resolve_auto_precision(prog, 2048 * 2048, "cuda")
        if prec != "fp16":
            declined += 1
            continue  # the gate declined it -> runs fp32, not in the fp16 lane
        accepted += 1
        try:
            t32 = run_tier(code, binds, "interp", precision="fp32")["OUT"].float()
            t16 = run_tier(code, binds, "interp", precision="fp16")["OUT"].float()
            if not torch.isfinite(t16).all():
                fell_back += 1  # node's runtime finiteness fallback re-cooks fp32 (safe)
                continue
            md = (t32 - t16).abs().max().item()
            if md > _FP16_BAR:
                fails.append(f"maxdiff {md:.2e}: {code[:72]}")
        except Exception as e:
            fails.append(f"{type(e).__name__}: {code[:60]}")
    if fails:
        r.fail("PR-LP2 fp16 fuzzer", f"{len(fails)}/{accepted} accepted exceeded band:\n  " +
               "\n  ".join(fails[:10]))
    else:
        r.ok(f"{accepted} gate-accepted programs within 3.9e-3 (declined {declined}, "
             f"fp32-fallback {fell_back})")
