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
        # doc 32 (un-masked): arr_sum(arr)=1e5 goes straight into vec3() with NO *0.00001
        # scaling — the vec-constructor now keeps the fp32 value instead of downcasting it
        # to fp16 inf, so the un-scaled program must be finite on interp AND codegen.
        code = (f"float arr[20]; for(int i=0;i<20;i++){{ arr[i]=5000.0; }} "
                f"@OUT = vec4(vec3({name}(arr)), 1.0);")
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
    print("\n--- PR-LP2: precision node-path perf (H7 / doc 32 honesty) ---")
    if not torch.cuda.is_available():
        r.ok("PR-LP2 node-path perf (no GPU, SKIPPED)")
        return
    import importlib.util
    bench = Path(__file__).resolve().parent.parent / "benchmarks" / "prlp2_node_path.py"
    spec = importlib.util.spec_from_file_location("prlp2_node_path", bench)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    # Machine-checks the HONEST claim on the real TEXWrangleNode.execute path (doc 32):
    #   * `auto` is essentially perf-NEUTRAL (measured 0.99x@1024 / 1.08x@2048) — the
    #     per-cook finiteness backstop that keeps it SAFE costs about what fp16 saves. It
    #     must not REGRESS (>=0.85x) and must not be sold as a speedup.
    #   * the raw fp16 win lives in expert `precision="fp16"` (no net): >=1.2x @2048.
    fails = []
    for res in (1024, 2048):
        f32 = mod._time(res, "fp32", iters=15)
        auto = mod._time(res, "auto", iters=15)
        if f32 / auto < 0.85:
            fails.append(f"{res}²: auto {f32/auto:.2f}× < 0.85× (a real regression)")
    f32 = mod._time(2048, "fp32", iters=15)
    fp16 = mod._time(2048, "fp16", iters=15)
    if f32 / fp16 < 1.2:
        fails.append(f"2048²: expert fp16 {f32/fp16:.2f}× < 1.2× (fp16 win vanished)")
    if fails:
        r.fail("PR-LP2 node-path perf", "; ".join(fails))
    else:
        r.ok("precision=auto is perf-neutral+safe (>=0.85x, not sold as a speedup); the "
             "fp16 win (>=1.2x@2048) lives in expert precision='fp16'")


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


# ── C1/C2 (doc 32) adversarial precision gate ────────────────────────────
# Amplification families the gain+magnitude gate MUST decline — each is a MEASURED
# gate-accepted-but-inaccurate hole from doc 32's adversarial sweep (node-path maxerr in
# the comment). The gain analysis assembles sub-threshold amplification (chained products,
# squaring, /const chains, builtin-const, dot/matrix/length/cross fan-in) + peak magnitude
# (additive round-trips) + ill-conditioned fns (atan2/tan/normalize).
_C1_MUST_DECLINE = [
    "@OUT = vec4(vec3(sin(@A.r*3.0*3.0*3.0)),1.0);",                                 # *27 chain 2.7e-2
    "float x=@A.r; x=x*3.0; x=x*3.0; x=x*3.0; @OUT=vec4(vec3(sin(x)),1.0);",         # split reassign
    "@OUT = vec4(vec3(sin(@A.r*PI*PI)),1.0);",                                       # builtin-const chain
    "@OUT = vec4(vec3(sin(@A.r/0.3/0.3)),1.0);",                                     # division chain
    "float x=@A.r*3.0; float y=x*x; float w=y*y; @OUT=vec4(vec3(sin(w)),1.0);",      # squaring tower
    "@OUT = vec4(vec3((@A.r+60000.0)-60000.0),1.0);",                               # additive round-trip
    "@OUT = vec4(vec3(600.0-(600.0-@A.r)),1.0);",
    "@OUT = vec4(vec3(sin(dot(@A.rgb, vec3(3.9,3.9,3.9)))),1.0);",                  # dot fan-in
    "mat3 m=mat3(3.9,3.9,3.9,3.9,3.9,3.9,3.9,3.9,3.9); @OUT=vec4(vec3(sin((m*@A.rgb).r)),1.0);",  # matrix
    "@OUT = vec4(vec3(sin(length(@A.rgb*3.9))),1.0);",                              # length fan-in
    "@OUT = vec4(vec3(cross(@A.rgb*3.9, vec3(1.0,2.0,3.0)).r),1.0);",               # cross
    "float a[3]; a[0]=@A.r*3.9; a[1]=@A.g*3.9; a[2]=@A.b*3.9; @OUT=vec4(vec3(sin(arr_sum(a))),1.0);",  # arr_sum
    "@OUT = vec4(vec3(atan2(@A.r-0.5, @A.g-0.5)),1.0);",                            # ill-conditioned origin
    "@OUT = vec4(vec3(tan(@A.r*3.0+1.5)),1.0);",                                    # poles
    "@OUT = vec4(vec3(vec3(100000.0)),1.0);",                                        # out-of-fp16 literal
    # doc 32 round 2 classes:
    "float s=1.5; mat3 m=mat3(s,s,s,s,s,s,s,s,s); @OUT=vec4(vec3(sin((m*@A.rgb).r)),1.0);",  # matrix-VARIABLE (const-prop)
    "@OUT = vec4(vec3(sin(@A.r*iw)),1.0);",                                          # builtin-dimension amplify
    "@OUT = vec4(vec3(fit(@A.r, 0.49, 0.51, 0.0, 1.0)),1.0);",                       # fit narrow-band 50x
    "@OUT = vec4(fit(@A, vec3(0.49), vec3(0.51), vec3(0.0), vec3(1.0)),1.0);",       # fit VECTOR bounds
    "float a[3]; a[0]=@A.r*3.9; a[1]=@A.g*3.9; a[2]=@A.b*3.9; @OUT=vec4(vec3(sin(arr_max(a))),1.0);",  # array reduction
    "@OUT = vec4(vec3(sin((@A.r+1.0)*3.9)),1.0);",                                   # shift-then-amplify (gain 3.9)
    # doc 33 F1 — amplification inside a user-function body (the gain pass can't see in):
    "float amp(float x){ return x*50.0; } @OUT = vec4(vec3(amp(@A.r)), 1.0);",
    "float amp(float x){ return x*50.0; } @OUT = vec4(vec3(sin(amp(@A.r))), 1.0);",
    "float g(vec3 c){return dot(c,vec3(0.3,0.6,0.1));} @OUT=vec4(vec3(g(@A.rgb)*50.0),1.0);",
]
# Smooth pointwise programs the gate MUST still accept (fp16) — the headline win region.
_C1_MUST_ACCEPT = [
    "vec3 c=@A.rgb; c=pow(c,vec3(0.4545)); @OUT=vec4(lerp(c,vec3(0.5),0.2),1.0);",   # grade
    "mat3 m=mat3(0.4,0.8,0.2,0.3,0.7,0.2,0.3,0.5,0.1); @OUT=vec4(m*@A.rgb,1.0);",
    "@OUT=vec4(1.0-@A.rgb,1.0);",
    "@OUT=vec4(@A.r*0.6+@A.g*0.4, @A.g, @A.b, 1.0);",
    "@OUT=vec4(vec3(sin(@A.r*3.0)*0.5+0.5),1.0);",                                    # single sin, gain 3
    "@OUT=vec4(vec3(dot(@A.rgb, vec3(0.299,0.587,0.114))),1.0);",                    # luma
    "@OUT=vec4(vec3(length(@A.rgb)),1.0);",
    "@OUT=vec4(@A.rgb*1.5,1.0);",                                                     # brightness
]


def test_c1_amplification_gate(r: SubTestResult):
    print("\n--- C1 (doc 32): amplification/condition-number gate ---")
    fails = []
    # (a) gate DECISION (CPU-fine): declines every measured hole, accepts the headline
    for code in _C1_MUST_DECLINE:
        if _resolve(code)[0] != "fp32":
            fails.append(f"MUST-DECLINE accepted fp16: {code[:56]}")
    for code in _C1_MUST_ACCEPT:
        if _resolve(code)[0] != "fp16":
            fails.append(f"MUST-ACCEPT declined: {code[:56]} -> {_resolve(code)[1]}")
    # (b) node-path accuracy: every ACCEPTED program is within the 8-bit quantum (CUDA)
    if torch.cuda.is_available():
        from TEX_Wrangle.tex_node import TEXWrangleNode as _N
        img = torch.rand(1, 1024, 1024, 3, device="cuda")
        for code in _C1_MUST_ACCEPT:
            try:
                a = _N.execute(code=code, A=img, device="cuda", precision="auto")[0]
                b = _N.execute(code=code, A=img, device="cuda", precision="fp32")[0]
                md = (a.float() - b.float()).abs().max().item()
                if md > _FP16_BAR:
                    fails.append(f"accepted but node maxerr {md:.2e} > {_FP16_BAR}: {code[:44]}")
            except Exception as e:
                fails.append(f"{type(e).__name__} on {code[:44]}: {e}")
        note = "gate decisions + node accuracy (CUDA)"
    else:
        note = "gate decisions only (no GPU)"
    if fails:
        r.fail("C1 amplification gate", "; ".join(fails[:12]))
    else:
        r.ok(f"{len(_C1_MUST_DECLINE)} holes declined, {len(_C1_MUST_ACCEPT)} headline "
             f"accepted+accurate ({note})")


def test_c2_data_dependent_nan(r: SubTestResult):
    print("\n--- C2 (doc 32): no silent NaN on a data-dependent input ---")
    fails = []
    # The regression: the B1 finiteness memo (keyed on code+types) trusted the FIRST cook
    # forever, so `1.0/@A.r`-class programs shipped 3.1M NaN on a later input that overflowed
    # fp16. TWO defences now: (a) the div-by-image / big-const-roundtrip repro class is
    # gate-DECLINED outright (image denominator -> inf gain; transient magnitude), and
    # (b) the finiteness net runs on EVERY fp16 cook (never memoized), pinning fp32 on any
    # non-finite. (a) is verifiable without a GPU; (b) needs CUDA.
    repro = [
        "@OUT = vec4(vec3(1.0/@A.r), 1.0);",
        "float b=@A.r/0.0002; b=b/0.0002; @OUT=vec4(vec3(b-b),1.0);",
        "@OUT = vec4(vec3((@A.r+60000.0)-60000.0), 1.0);",
    ]
    for code in repro:
        if _resolve(code)[0] != "fp32":
            fails.append(f"repro accepted fp16 (must decline): {code[:48]}")
    # (b) the user-facing guarantee: cooking the repro on two DIFFERENT inputs never ships a
    # non-finite pixel (the exact failure the audit measured at 3,145,728).
    if torch.cuda.is_available():
        from TEX_Wrangle.tex_node import TEXWrangleNode as _N
        code = "@OUT = vec4(vec3(1.0/@A.r), 1.0);"
        for val in (0.5, 1e-5):
            o = _N.execute(code=code, A=torch.full((1, 1024, 1024, 3), val, device="cuda"),
                           device="cuda", precision="auto")[0]
            nf = int((~torch.isfinite(o)).sum())
            if nf:
                fails.append(f"A={val}: shipped {nf} non-finite (was 3.1M pre-fix)")
        note = "declined + node ships 0 non-finite on two inputs (CUDA)"
    else:
        note = "gate declines the repro class (no GPU)"
    if fails:
        r.fail("C2 data-dependent NaN", "; ".join(fails))
    else:
        r.ok(f"C2: {note}")


def test_c2_finiteness_net_recovers(r: SubTestResult):
    print("\n--- C2/F1: the finiteness net RECOVERS (doesn't crash) when auto-fp16 overflows ---")
    # F1 regression guard: the C1-st extraction moved this block out of execute() without its
    # tier_trace import, so it NameError-crashed exactly when auto accepted fp16 but the output
    # went non-finite (the gate's blind spot). The other C2 test only uses gate-DECLINED
    # programs, so the net returns early and never reaches the crash lines — this drives the
    # net's real recovery body: auto_fp16=True, eff_precision='fp16', a NON-finite output.
    from TEX_Wrangle.tex_node import TEXWrangleNode as N, ExecContext
    from TEX_Wrangle import tex_api
    prog = tex_api.compile("@OUT = vec4(@A.rgb, 1.0);", {"A": TEXType.VEC3, "OUT": TEXType.VEC4})
    ctx = ExecContext(program=prog.ast, bindings={"A": torch.rand(1, 8, 8, 3)},
                      type_map=prog.type_map, device="cpu", code="@OUT = vec4(@A.rgb, 1.0);",
                      latent_channel_count=0, output_names=["OUT"],
                      used_builtins=prog.used_builtins, eff_precision="fp16", fp=None)
    try:
        out = N._fp16_finiteness_net({"OUT": torch.tensor([float("inf")])}, True, ctx, "default")
    except Exception as e:
        r.fail("F1 crash", f"finiteness net raised {type(e).__name__}: {e} (the F1 NameError)")
        return
    t = out.get("OUT")
    ok_unit = isinstance(t, torch.Tensor) and bool(torch.isfinite(t).all())

    # And end-to-end through the REAL execute(): force auto->fp16 on an fp16-OVERFLOWING
    # program (CUDA only — auto is a no-op on CPU); the node must ship finite pixels, not crash.
    e2e = "skipped (no CUDA)"
    if torch.cuda.is_available():
        import TEX_Wrangle.tex_runtime.precision_policy as pp
        orig = pp.resolve_auto_precision
        pp.resolve_auto_precision = lambda *a, **k: ("fp16", "forced (F1 test)")
        try:
            code = "@OUT = vec4(vec3(@A.r * 50000.0 * 50000.0), 1.0);"  # >> fp16 max -> inf
            o = N.execute(code=code, A=torch.full((1, 1024, 1024, 3), 1.0, device="cuda"),
                          device="cuda", precision="auto")[0]
            e2e = "0 non-finite" if int((~torch.isfinite(o)).sum()) == 0 else "SHIPPED non-finite"
        finally:
            pp.resolve_auto_precision = orig

    if ok_unit and e2e in ("0 non-finite", "skipped (no CUDA)"):
        r.ok(f"finiteness net recovers to finite fp32 (no F1 crash); e2e: {e2e}")
    else:
        r.fail("C2 finiteness recovery", f"unit-finite={ok_unit}, e2e={e2e}")
