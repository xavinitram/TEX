"""v0.19.0 Phase 4 - measured perf & the v0.20 seed.

P3 (CUDA mat3/mat4 broadcast): the interpreter and codegen matxvec paths now compute a
device-gated expression - CUDA elementwise broadcast-sum (3.4-3.9x faster for the tiny-
matrix/huge-batch per-pixel case), CPU matmul (~7x faster there). The load-bearing
invariant is interp<->codegen BIT-EXACTNESS on each device; both tiers call the same
device gate, so they match. This is the dedicated cross-device envelope row doc 35 asks for.
"""
import sys
from pathlib import Path
from helpers import *

_PKG = Path(__file__).resolve().parent.parent

# mat3 and mat4 transform programs exercising all four matxvec promotion branches.
_MAT3 = ("mat3 m = mat3(0.6, 0.1, 0.0, -0.1, 0.6, 0.0, 0.2, 0.1, 1.0);\n"
         "@OUT = vec4(m * @A.rgb, 1.0);")                      # mat3 * vec3
_MAT3_V4 = ("mat3 m = mat3(0.6, 0.1, 0.0, -0.1, 0.6, 0.0, 0.2, 0.1, 1.0);\n"
            "vec4 c = vec4(@A.rgb, 0.5);\n@OUT = m * c;")      # mat3 * vec4 (preserve w)
_MAT4_V3 = ("mat4 m = mat4(0.6, 0.1, 0.0, 0.0, -0.1, 0.6, 0.0, 0.0, "
            "0.2, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);\n"
            "vec4 p = m * @A.rgb;\n@OUT = vec4(p.rgb, 1.0);")  # mat4 * vec3 (promote w=1)


def _interp(prog, img, dev):
    from TEX_Wrangle.tex_runtime.interpreter import Interpreter
    return Interpreter().execute(prog.ast, {"A": img}, prog.type_map, device=dev,
                                 output_names=["OUT"], precision="fp32",
                                 used_builtins=prog.used_builtins)["OUT"]


def _codegen(prog, img, dev, fp):
    from TEX_Wrangle.tex_runtime import compiled
    out = compiled.execute_compiled(prog.ast, {"A": img}, prog.type_map, dev, fp,
                                    output_names=["OUT"], used_builtins=prog.used_builtins)
    return out["OUT"] if isinstance(out, dict) else out


def test_p3_matvec_interp_codegen_bit_exact(r: SubTestResult):
    print("\n--- P3: matxvec interp<->codegen bit-exact on each device (the sacred invariant) ---")
    from TEX_Wrangle import tex_api
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    bad = []
    for tag, code in (("mat3*vec3", _MAT3), ("mat3*vec4", _MAT3_V4), ("mat4*vec3", _MAT4_V3)):
        prog = tex_api.compile(code, {"A": TEXType.VEC3, "OUT": TEXType.VEC4})
        for dev in devices:
            img = make_img(1, 64, 64, 3).to(dev)
            it = _interp(prog, img, dev).float()
            cg = _codegen(prog, img, dev, f"p3_{tag}_{dev}").float()
            if not torch.equal(it, cg):
                bad.append(f"{tag}@{dev} maxdiff={float((it - cg).abs().max()):.2e}")
    if bad:
        r.fail("P3 bit-exact", "interp<->codegen diverged: " + "; ".join(bad))
    else:
        r.ok(f"mat3/mat4 x vec interp==codegen bit-exact on {devices} (all 3 promotion branches)")


def test_p3_cuda_matches_matmul_within_ulp(r: SubTestResult):
    print("\n--- P3: CUDA broadcast form stays within 1 fp32 ULP of matmul (envelope) ---")
    if not torch.cuda.is_available():
        r.ok("no CUDA - broadcast path is CUDA-only; CPU keeps matmul (bit-exact)")
        return
    from TEX_Wrangle.tex_runtime.interpreter import _matvec
    M = torch.rand(1, 512, 512, 3, 3, device="cuda")
    v = torch.rand(1, 512, 512, 3, device="cuda")
    bc = _matvec(M, v)                                    # P3 path (CUDA -> broadcast)
    mm = torch.matmul(M, v.unsqueeze(-1)).squeeze(-1)     # reference
    diff = float((bc - mm).abs().max())
    # 1 fp32 ULP near 1.0 is ~1.2e-7; allow a small multiple, and demand << the 8-bit quantum
    if diff > 1e-6:
        r.fail("P3 envelope", f"broadcast vs matmul maxdiff {diff:.2e} > 1e-6")
    else:
        r.ok(f"CUDA broadcast within {diff:.1e} of matmul (<< 3.9e-3 8-bit quantum)")


def test_p3_cpu_keeps_matmul(r: SubTestResult):
    print("\n--- P3: the CPU path is unchanged (still matmul, bit-exact) ---")
    from TEX_Wrangle.tex_runtime.interpreter import _matvec
    M = torch.rand(1, 32, 32, 3, 3)
    v = torch.rand(1, 32, 32, 3)
    if torch.equal(_matvec(M, v), torch.matmul(M, v.unsqueeze(-1)).squeeze(-1)):
        r.ok("CPU _matvec is exactly matmul (no numeric change on the CPU path)")
    else:
        r.fail("P3 CPU", "CPU _matvec diverged from matmul - the CPU path must be untouched")


# ---- P4: is_tile_safe memo -------------------------------------------------

def test_p4_tile_safe_memo(r: SubTestResult):
    print("\n--- P4: is_tile_safe memoized per fingerprint (cached == uncached, both verdicts) ---")
    from TEX_Wrangle import tex_api
    from TEX_Wrangle.tex_memory import is_tile_safe, is_tile_safe_cached, _tile_safe_memo
    _tile_safe_memo.clear()
    safe = tex_api.compile("@OUT = vec4(@A.rgb * 1.2, 1.0);",
                           {"A": TEXType.VEC3, "OUT": TEXType.VEC4})
    unsafe = tex_api.compile("@OUT = vec4(sample(@A, u + 0.01, v).rgb, 1.0);",
                             {"A": TEXType.VEC3, "OUT": TEXType.VEC4})
    bad = []
    for name, prog, fp, want in (("pointwise", safe, "fp_a", True),
                                 ("sample", unsafe, "fp_b", False)):
        direct = is_tile_safe(prog.ast)
        miss = is_tile_safe_cached(prog.ast, fp)   # populate
        hit = is_tile_safe_cached(prog.ast, fp)    # served from memo (esp. the False case)
        if not (direct == miss == hit == want):
            bad.append(f"{name}: direct={direct} miss={miss} hit={hit} want={want}")
    # None fingerprint (fused chain) must fall through to the uncached walk
    if is_tile_safe_cached(safe.ast, None) != is_tile_safe(safe.ast):
        bad.append("None fingerprint did not fall through")
    if bad:
        r.fail("P4 memo", "; ".join(bad))
    else:
        r.ok("cached verdict matches the walk for tile-safe(True) AND non-tile-safe(False); "
             "None falls through")


def test_p4_memo_key_is_cook_fingerprint(r: SubTestResult):
    print("\n--- P4: memo keyed on the cook fingerprint (same key should_stencil_route uses) ---")
    from TEX_Wrangle import tex_api
    from TEX_Wrangle.tex_cache import get_cache
    from TEX_Wrangle.tex_memory import is_tile_safe_cached, _tile_safe_memo
    from TEX_Wrangle.tex_runtime.compiled import should_stencil_route, _stencil_route_memo
    _tile_safe_memo.clear()
    code = "@OUT = vec4(@A.rgb * 0.5, 1.0);"
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    prog = tex_api.compile(code, bt)
    fp = get_cache().fingerprint(code, bt)   # the cook fingerprint execute() threads in
    is_tile_safe_cached(prog.ast, fp)
    should_stencil_route(fp, prog.ast)
    # Both memos must have keyed on the identical cook fingerprint — a mismatch would
    # let one serve a decision for the wrong program.
    if fp in _tile_safe_memo and fp in _stencil_route_memo:
        r.ok("is_tile_safe memo and stencil-route memo share the cook fingerprint key")
    else:
        r.fail("P4 key identity", f"fp {fp!r} not the shared key "
               f"(tile={fp in _tile_safe_memo}, route={fp in _stencil_route_memo})")


# ---- P2: noise shape-recompile fix -----------------------------------------

def test_p2_noise_compile_dynamic(r: SubTestResult):
    print("\n--- P2: noise torch.compile uses dynamic=True (no resolution recompile stall) ---")
    import inspect
    from TEX_Wrangle.tex_runtime import noise
    src = inspect.getsource(noise._compile_noise)
    # dynamic=True gives ONE kernel for all resolutions (measured live this session: 1 graph
    # across 512/1024/2048, ~1 ULP of the static path, ~18x over eager). Full 3-resolution
    # bit-exactness is additionally guarded by the differential fuzzer + TST-6 parity.
    if "dynamic=True" not in src:
        r.fail("P2 dynamic", "_compile_noise must pass dynamic=True (the recompile-stall fix)")
        return
    # The cache key MUST stay shape-UNAWARE: shape-aware keys reset each resolution to the
    # dtype-brittle jit.trace tier (the AssertionError this fix replaced). dynamic=True keeps
    # one entry, so the key carries no `x.shape`.
    fbm_src = inspect.getsource(noise._fbm2d)
    key_line = next((ln for ln in fbm_src.splitlines() if ln.strip().startswith("key = ")), "")
    if "shape" in key_line:
        r.fail("P2 key", f"fbm key went shape-aware ({key_line.strip()!r}) - the fragile re-trace path")
    else:
        r.ok("dynamic=True set; fbm key stays shape-unaware (one kernel, no per-shape re-trace)")


# ---- P6: surface facts (noise compile visibility) --------------------------

def test_p6_noise_compile_visibility(r: SubTestResult):
    print("\n--- P6: noise compiles are recorded + surfaced in the doctor payload ---")
    from TEX_Wrangle.tex_runtime import tier_trace
    from TEX_Wrangle.tex_doctor import collect_doctor_facts
    tier_trace.record_noise_compile("fbm", 1234.5)
    ev = tier_trace.noise_compiles()
    if not (ev and ev[-1]["noise"] == "fbm" and ev[-1]["ms"] == 1234.5):
        r.fail("P6 event", f"record/read round-trip failed: {ev[-3:]}")
        return
    facts = collect_doctor_facts()
    if "noise_compiles" not in facts:
        r.fail("P6 doctor", "doctor payload missing 'noise_compiles' fact")
    else:
        r.ok("noise compile events recorded and surfaced in tex doctor (visible, not a mystery stall)")


# ---- A5-1: parser hints for v0.20-reserved words ---------------------------

def test_a5_1_reserved_word_hints(r: SubTestResult):
    print("\n--- A5-1: pass/stage/kernel/image in block position -> purpose-written hint ---")
    from TEX_Wrangle import tex_api
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

    def compiles(code):
        try:
            tex_api.compile(code, bt); return True, ""
        except Exception as e:
            return False, str(e).split("\n")[0]

    bad = []
    # Block position -> rejected with a hint mentioning multi-pass / chaining.
    for word, code in (("pass", "pass { @OUT = vec4(@A.rgb, 1.0); }"),
                       ("stage", "stage { @OUT = vec4(1.0); }"),
                       ("pass;", "pass;\n@OUT = vec4(@A.rgb, 1.0);")):
        ok, msg = compiles(code)
        if ok:
            bad.append(f"{word} should be rejected in block position")
    # As ordinary variable names -> still compile (no regression).
    for word, code in (("image", "vec3 image = @A.rgb; image = image * 0.5; @OUT = vec4(image, 1.0);"),
                       ("kernel", "float kernel = 0.5; @OUT = vec4(@A.rgb * kernel, 1.0);")):
        ok, msg = compiles(code)
        if not ok:
            bad.append(f"{word} as a variable must still compile, got: {msg}")
    if bad:
        r.fail("A5-1 reserved words", "; ".join(bad))
    else:
        r.ok("pass/stage rejected in block position with a hint; image/kernel usable as variables")


# ---- A5-2: recursive examples ----------------------------------------------

def test_a5_2_recursive_examples(r: SubTestResult):
    print("\n--- A5-2: recursive-function examples compile and produce sane output ---")
    from TEX_Wrangle.tex_node import TEXWrangleNode as N
    img = make_img(1, 32, 32, 3)
    bad = []
    for stem in ("recursive_fractal", "recursive_subdivision"):
        path = _PKG / "examples" / f"{stem}.tex"
        if not path.exists():
            bad.append(f"{stem}.tex missing")
            continue
        code = path.read_text(encoding="utf-8")
        try:
            out = N.execute(code=code, A=img, device="cpu")
            t = (out[0] if isinstance(out, tuple) else out).float()
            if not torch.isfinite(t).all():
                bad.append(f"{stem}: non-finite output")
            elif not (0.0 <= float(t.min()) and float(t.max()) <= 1.0):
                bad.append(f"{stem}: out of [0,1] range [{float(t.min()):.2f},{float(t.max()):.2f}]")
        except Exception as e:
            bad.append(f"{stem}: {type(e).__name__}: {str(e)[:60]}")
    if bad:
        r.fail("A5-2 recursive examples", "; ".join(bad))
    else:
        r.ok("recursive_fractal + recursive_subdivision compile, run, finite, in [0,1]")


# ---- F5: codegen scalar-loop robustness (audit F5 codegen-crash class) ------

def test_f5_codegen_scalar_loop_no_crash(r: SubTestResult):
    print("\n--- F5: a scalar-mode loop calling a non-scalar-emittable stdlib fn stays in "
          "tensor mode (no codegen crash), bit-exact ---")
    # The differential fuzzer, once able to SEE codegen crashes (tier_trace), found that an
    # all-scalar loop body calling a stdlib fn WITHOUT a scalar lowering (spow/degrees/smin)
    # entered codegen's scalar mode, then fell through to the tensor path -> torch.abs(float)
    # crash -> silent interp fallback. `_is_scalar_node` now gates on `_SCALAR_EMITTABLE_FNS`,
    # so such calls run in tensor mode: no crash, and interp==codegen.
    sys.path.insert(0, str(_PKG / "tests")) if str(_PKG / "tests") not in sys.path else None
    from failure_harness import run_tier
    from TEX_Wrangle.tex_runtime import tier_trace
    binds = {"A": make_img(1, 8, 8, 3, seed=1), "B": make_img(1, 8, 8, 3, seed=2)}
    progs = [
        "float a=0.0; for (int i=0;i<3;i=i+1){ a = a + spow(3.0, 0.5); } @OUT=vec4(a,u,v,1.0);",
        "float a=0.0; for (int i=0;i<3;i=i+1){ a = a + degrees(0.5); } @OUT=vec4(a,u,v,1.0);",
        "float a=0.0; for (int i=0;i<2;i=i+1){ a = a + smin(0.3, 0.7, 0.5); } @OUT=vec4(a,u,v,1.0);",
    ]
    bad = []
    for code in progs:
        tier_trace.reset()
        try:
            b = run_tier(code, binds, "interp")
            c = run_tier(code, binds, "codegen")
        except Exception as e:
            bad.append(f"{type(e).__name__}: {code[:40]}")
            continue
        rec = tier_trace.last()
        crashed = rec is not None and rec.fallback_from == "codegen" and rec.reason != "unsupported"
        diverged = float((b["OUT"].float() - c["OUT"].float()).abs().max()) > 1e-5
        if crashed:
            bad.append(f"codegen crash-fallback ({str(rec.reason)[:35]}): {code[:35]}")
        elif diverged:
            bad.append(f"interp!=codegen: {code[:40]}")
    if bad:
        r.fail("F5 codegen scalar loop", "; ".join(bad))
    else:
        r.ok("scalar-loop spow/degrees/smin run codegen in tensor mode: no crash, bit-exact")


def test_f5b_lerp_family_fused_bit_exact(r: SubTestResult):
    print("\n--- F5b: smin/smax/lerp/mix/fit codegen is interp==codegen BIT-EXACT "
          "(nightly-fuzz regression, seed 20260712) ---")
    # The nightly differential fuzzer found a full 0<->1 flip on 22% of pixels for a
    # smin(...) feeding a near-equal-edge smoothstep. Root cause: codegen emitted the
    # lerp inside smin UNFUSED (b + (a-b)*h) while the interpreter uses the fused
    # torch.lerp (one FMA rounding). The ~2e-8 gap is invisible normally, but the
    # singular smoothstep edge (e0~=e1, constant per row) amplified it across whole
    # rows. Codegen now calls the same fused lerp for the whole family, so they match
    # bit-for-bit. This pins the exact canary program plus the general property.
    from failure_harness import run_tier
    binds = {"A": make_img(1, 32, 32, 3, seed=11), "B": make_img(1, 32, 32, 3, seed=22)}
    nightly = ("float lv0 = smin(0.1, 1.5, (v - 0.1)); float lv2 = 0.1; "
               "@OUT = vec4(smoothstep(lv0, lv2, (@B.r - 3.0)), u, v, 1.0);")
    progs = [
        nightly,                                               # the exact canary
        "@OUT = vec4(smin(@A.r, 0.5, v), u, v, 1.0);",         # smin, scalar
        "@OUT = vec4(smax(@A.r, 0.5, v), u, v, 1.0);",         # smax, scalar
        "@OUT = vec4(lerp(@A.rgb, @B.rgb, v), 1.0);",          # lerp vec, scalar weight (unsqueeze)
        "@OUT = vec4(mix(@A.rgb, @B.rgb, @B.bgr), 1.0);",      # mix vec, vec weight
        "@OUT = vec4(fit(@A.r, 0.1, 0.9, 0.0, 1.0), u, v, 1.0);",  # fit
    ]
    bad = []
    for code in progs:
        b = run_tier(code, binds, "interp")["OUT"]
        c = run_tier(code, binds, "codegen")["OUT"]
        finmis = int((torch.isfinite(b) != torch.isfinite(c)).sum())
        mx = float((b - c).abs().max())
        if finmis or mx != 0.0:                                # demand BIT-EXACT, not merely close
            bad.append(f"finmis={finmis} max={mx:.3e}: {code[:45]}")
    if bad:
        r.fail("F5b lerp-family fused fidelity", "; ".join(bad))
    else:
        r.ok("smin/smax/lerp/mix/fit + the nightly canary: interp==codegen bit-exact")


def test_f5c_pow_mod_codegen_fidelity(r: SubTestResult):
    print("\n--- F5c: pow(non-{0,1,2,3}) and mod defer to the interpreter fn "
          "-> interp==codegen (post-audit fidelity) ---")
    # The exhaustive fidelity audit (after the lerp fix) found the NEXT interp!=codegen
    # class: _emit_fn_pow specialized non-folded exponents to expressions that diverge
    # from the interpreter's torch.pow — rsqrt(clamp(x,eps)) for -0.5 and reciprocal(x*x+eps)
    # for -2.0 FLIPPED finiteness on x<=0 / diverged up to 8e-3, and even a plain
    # _torch.pow(x, <python-float>) rounds differently from fn_pow's torch.pow(x,_to_tensor(exp))
    # (~3.7e-9). pow now keeps only x^{0,1,2,3} and defers everything else to _fns['pow'].
    # Separately, codegen's mod used a non-dtype-aware 1e-8 zero-guard that UNDERFLOWS in
    # fp16 (-> NaN) where fn_mod's 6.104e-5 stays finite; mod now defers to _fns['mod'].
    from failure_harness import run_tier
    binds = {"A": make_img(1, 32, 32, 3, seed=11), "B": make_img(1, 32, 32, 3, seed=22)}
    # exponents the optimizer does NOT fold to x*x* -> reach _emit_fn_pow -> must defer.
    # (negative bases give NaN in BOTH tiers -> finiteness matches; that is the point.)
    pow_progs = [f"@OUT = vec4(pow(u - 0.5, {e}), 0, 0, 1);"
                 for e in ("-0.5", "-2.0", "-1.0", "4.0", "0.5", "2.5")]
    pow_progs.append("@OUT = vec4(pow(@A.r, @B.r * 3.0), 0, 0, 1);")   # variable exponent
    mod_progs = ["@OUT = vec4(mod(@A.r * 5.0, u - 0.5), 0, 0, 1);",    # divisor hits 0
                 "@OUT = vec4(mod(@A.r, 0.3), 0, 0, 1);"]
    bad = []
    for code in pow_progs + mod_progs:
        b = run_tier(code, binds, "interp")["OUT"]
        c = run_tier(code, binds, "codegen")["OUT"]
        finmis = int((torch.isfinite(b) != torch.isfinite(c)).sum())
        fin = torch.isfinite(b) & torch.isfinite(c)
        mx = float((b[fin] - c[fin]).abs().max()) if fin.any() else 0.0
        if finmis or mx != 0.0:                                        # demand BIT-EXACT (fp32)
            bad.append(f"finmis={finmis} max={mx:.2e}: {code[:45]}")
    # mod fp16: the zero-guard must be dtype-aware (1e-8 underflows -> NaN in fp16). fp16 is
    # a lossy accuracy band, so assert FINITENESS parity (no NaN-vs-finite), not bit-exactness.
    code16 = "@OUT = vec4(mod(@A.r, u - 0.5), 0, 0, 1);"
    b16 = run_tier(code16, binds, "interp", precision="fp16")["OUT"]
    c16 = run_tier(code16, binds, "codegen", precision="fp16")["OUT"]
    fin16 = int((torch.isfinite(b16) != torch.isfinite(c16)).sum())
    if fin16:
        bad.append(f"mod fp16 finiteness mismatch: {fin16} px (dtype-aware guard regressed)")
    if bad:
        r.fail("F5c pow/mod fidelity", "; ".join(bad))
    else:
        r.ok("pow(non-{0,1,2,3}) + mod defer to interp fn: fp32 bit-exact, fp16 finiteness parity")
