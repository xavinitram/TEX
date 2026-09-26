"""The opt-in routes into codegen place `$param` bindings on the cook device.

A `$param` reaches the generated function as a Python float / int / list (a widget value or a
code-defined default), and the emitted preamble converts it with `as_tensor(value)`, which
lands on the CPU. A 0-dim CPU tensor mixes with CUDA operands only while it stays 0-dim, so the
first broadcast that expanded it (`_bp` against a vec, a vec param, a stdlib bound such as
`clamp`'s) raised a cross-device error and the codegen tier fell back to the interpreter on
every CUDA cook of such a program under `compile_mode="auto"`. The pixels were right (the
interpreter served them); the tier never did. On the opt-in routes a call that raises is now
retried once with those bindings on the cook device, and a generated function that needed it
places them up front from then on (`compiled._codegen_with_params_on_device`); the preamble
passes a device tensor through untouched.

Placement is learned per generated function rather than applied to every opt-in cook, because
it is not free: a host-to-device copy per param per cook, and a device sync wherever the code
reads a param back to the host. Programs codegen already served with CPU params never place
(`test_codegen_param_placement_learned_once`). The default route into codegen (an exact fetch
stencil under `compile_mode="none"`) never places at all
(`test_codegen_param_default_route_unmoved`).

Every row reads the serving tier from `tier_trace` BEFORE comparing pixels: a parity check
against a silent fallback compares the interpreter with itself. Pixels are then compared with
`torch.equal` (invariant 2, bit-exact on the same device). On a CPU-only runner the CPU legs
pin that the CPU tier and the CPU output did not move (placement never runs on the CPU).
"""
import threading

from helpers import *
from failure_harness import run_tier, simulate_restart

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_cache import get_cache
from TEX_Wrangle.tex_runtime import tier_trace

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]

_GAIN = "f$gain = 1.0;\n@OUT = @image * $gain;"

_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def _served_by(what, tier, fallback_from=None):
    rec = tier_trace.last()
    assert rec is not None and rec.tier == tier and rec.fallback_from == fallback_from, \
        f"{what}: served by {rec!r}, expected tier={tier!r} fallback_from={fallback_from!r}"


def _interp_like_engine(code, bindings, device, precision="fp32"):
    """The oracle for an engine cook: the program compiled exactly as the engine compiles it
    (optimizer included), run on the interpreter."""
    bt = {n: _infer_binding_type(v) for n, v in bindings.items()}
    prog, tm, _refs, assigned, _params, used = get_cache().compile_tex(code, bt)
    outs = sorted(assigned.keys())
    return Interpreter().execute(prog, dict(bindings), tm, device=device, output_names=outs,
                                 precision=precision, used_builtins=used)["OUT"]


def _assert_equal(ref, got, what):
    assert ref.shape == got.shape and ref.dtype == got.dtype and ref.device == got.device, \
        (f"{what}: {tuple(got.shape)}/{got.dtype}/{got.device} vs interpreter "
         f"{tuple(ref.shape)}/{ref.dtype}/{ref.device}")
    if not torch.equal(ref, got):
        raise AssertionError(f"{what}: not bit-exact, maxdiff {(ref - got).abs().max().item():.3e}")


def _assert_within_tol(ref, got, what, tol=1e-5):
    """Invariant 2 at its stated tolerance (AGENTS.md, fp32), for a program whose codegen
    lowering reorders a sum the interpreter evaluates term by term."""
    assert ref.shape == got.shape and ref.dtype == got.dtype and ref.device == got.device, \
        (f"{what}: {tuple(got.shape)}/{got.dtype}/{got.device} vs interpreter "
         f"{tuple(ref.shape)}/{ref.dtype}/{ref.device}")
    md = (ref - got).abs().max().item()
    assert md < tol, f"{what}: maxdiff {md:.3e} >= {tol:g}"


def test_codegen_param_served_on_cook_device(r: SubTestResult):
    print("\n--- compile_mode='auto': codegen serves a scalar $param program on every device ---")
    img = make_img(1, 16, 16, 3, seed=5)
    for dev in _DEVICES:
        try:
            binds = {"image": img.to(dev), "gain": 1.2}
            tier_trace.reset()
            res = tex_engine.cook(_GAIN, dict(binds), device_mode=dev, compile_mode="auto")
            _served_by(f"[{dev}] first cook", "codegen")
            ref = _interp_like_engine(_GAIN, binds, dev)
            _assert_equal(ref, res.outputs["OUT"], f"[{dev}] first cook")
            # Class B: the generated fn persists (marshal sidecar) and must still serve once the
            # in-memory tiers are gone, i.e. rematerialized from disk as a fresh process would —
            # which also drops the param names cached on the fn object.
            simulate_restart()
            tier_trace.reset()
            res2 = tex_engine.cook(_GAIN, dict(binds), device_mode=dev, compile_mode="auto")
            _served_by(f"[{dev}] after restart", "codegen")
            _assert_equal(ref, res2.outputs["OUT"], f"[{dev}] after restart")
            r.ok(f"[{dev}] f$gain program served by codegen, bit-exact (fresh and rematerialized)")
        except Exception as e:
            r.fail(f"codegen serves f$gain [{dev}]", f"{type(e).__name__}: {e}")


# name, code, params. Scalar int/float, vec/colour lists, the two mixed, inside loops (body and
# bound), inside a user function, and as a stdlib bound/edge/exponent.
_PARAM_PROGRAMS = [
    ("float", _GAIN, {"gain": 1.2}),
    ("int", "i$n = 3;\n@OUT = @image * $n;", {"n": 3}),
    ("int in / and *", "i$n = 3;\n@OUT = @image / $n + $n * 0.1;", {"n": 7}),
    ("vec3 list", "v3$tint = vec3(1.0, 0.5, 0.25);\n@OUT = @image.rgb * $tint;",
     {"tint": [0.9, 0.5, 0.25]}),
    ("vec3 list x float",
     "f$gain = 1.0;\nv3$tint = vec3(1.0, 0.5, 0.25);\n@OUT = @image.rgb * $tint * $gain;",
     {"gain": 1.3, "tint": [0.9, 0.5, 0.25]}),
    ("colour list x float", "f$gain = 1.0;\nc$col;\n@OUT = mix(@image.rgb, $col, $gain);",
     {"gain": 0.35, "col": [0.2, 0.7, 0.1]}),
    ("float in loop body",
     "f$gain = 0.5;\nvec3 acc = vec3(0.0);\nfor (int i = 0; i < 4; i++) {\n"
     "    acc = acc + @image.rgb * $gain;\n}\n@OUT = acc;", {"gain": 0.37}),
    ("int loop bound x float",
     "i$n = 3;\nf$k = 0.25;\nvec3 acc = vec3(0.0);\nfor (int i = 0; i < $n; i++) {\n"
     "    acc = acc + @image.rgb * $k;\n}\n@OUT = acc;", {"n": 3, "k": 0.3}),
    ("float in user function",
     "f$amt = 1.0;\nvec3 g(vec3 x) { return x * $amt; }\n@OUT = g(@image.rgb);", {"amt": 0.7}),
    ("clamp bounds + pow exponent",
     "f$lo = 0.1;\nf$hi = 0.9;\nf$gam = 2.2;\n@OUT = pow(clamp(@image.rgb, $lo, $hi), vec3($gam));",
     {"lo": 0.1, "hi": 0.8, "gam": 2.2}),
    ("smoothstep edges", "f$e0 = 0.2;\nf$e1 = 0.8;\n@OUT = smoothstep($e0, $e1, @image.rgb);",
     {"e0": 0.2, "e1": 0.7}),
    ("ternary threshold", "f$t = 0.5;\n@OUT = @image.r > $t ? @image.rgb * $t : @image.rgb;",
     {"t": 0.4}),
]


def test_codegen_param_parity_on_every_device(r: SubTestResult):
    print("\n--- run_auto: codegen == interpreter for $param programs, served by codegen ---")
    img = make_img(1, 12, 10, 3, seed=6)
    for name, code, params in _PARAM_PROGRAMS:
        for dev in _DEVICES:
            try:
                binds = {"image": img.to(dev), **params}
                ref = run_tier(code, binds, "interp", device=dev)["OUT"]
                tier_trace.reset()
                got = run_tier(code, binds, "auto", device=dev)["OUT"]
                _served_by(f"[{dev}] {name}", "codegen")
                _assert_equal(ref, got, f"[{dev}] {name}")
                r.ok(f"[{dev}] {name}: codegen served, bit-exact")
            except Exception as e:
                r.fail(f"$param parity [{dev}] {name}", f"{type(e).__name__}: {e}")


def test_codegen_param_precision_requests(r: SubTestResult):
    print("\n--- fp16/auto requests on compile_mode='auto': codegen serves the resolved fp32 ---")
    # prepare() resolves a compiled-tier cook to fp32 (fp16 -> fp32 outright; auto -> fp32
    # "[compiled tier: fp32]"), so fp32 is the precision codegen serves; it must then be
    # bit-exact with the interpreter at that precision. The 1024^2 CUDA leg is the size where
    # `auto` would have picked fp16 for a pointwise program if the tier allowed it.
    code, params = _PARAM_PROGRAMS[4][1], _PARAM_PROGRAMS[4][2]
    legs = [(dev, prec, 12) for dev in _DEVICES for prec in ("fp16", "auto")]
    if _CUDA:
        legs.append(("cuda", "auto", 1024))
    for dev, prec, size in legs:
        try:
            binds = {"image": make_img(1, size, size, 3, seed=7).to(dev), **params}
            tier_trace.reset()
            res = tex_engine.cook(code, dict(binds), device_mode=dev, compile_mode="auto",
                                  precision=prec)
            _served_by(f"[{dev}] precision={prec} {size}px", "codegen")
            assert res.precision == "fp32", \
                f"[{dev}] precision={prec} resolved to {res.precision!r} on a compiled tier"
            ref = _interp_like_engine(code, binds, dev, precision=res.precision)
            _assert_equal(ref, res.outputs["OUT"], f"[{dev}] precision={prec} {size}px")
            r.ok(f"[{dev}] precision={prec} @{size}px -> fp32, codegen served, bit-exact")
        except Exception as e:
            r.fail(f"$param precision request [{dev}] {prec} {size}px", f"{type(e).__name__}: {e}")


def test_codegen_param_placement_learned_once(r: SubTestResult):
    print("\n--- compile_mode='auto': placement is learned once; programs already served never place ---")
    # blur.tex (a kernel radius read back to the host) and sharpen.tex (a lerp weight) are
    # served by codegen with the preamble's CPU params, so they must take no placement step:
    # placing would add a host-to-device copy, and for blur a device sync, to every cook. The
    # probe broadcasts its param: on CUDA its first call raises once, the retry with the param
    # on the device serves, and every later cook places up front — one failed call in total,
    # not one per cook. Counted through spies on the two module-level seams the tier entry
    # calls; each row starts cold so the generated function carries no verdict yet.
    # sharpen.tex's Laplacian lowers to a conv2d, which sums in a different order from the
    # interpreter's term-by-term expression: a few fp32 ulps on every route and device, with
    # or without this change, so that row is held to invariant 2's tolerance, not torch.equal.
    from TEX_Wrangle.tex_runtime import compiled as C
    with open(os.path.join(_EXAMPLES, "blur.tex"), encoding="utf-8") as f:
        blur = f.read()
    with open(os.path.join(_EXAMPLES, "sharpen.tex"), encoding="utf-8") as f:
        sharpen = f.read()
    cooks = 3
    img = make_img(1, 24, 24, 3, seed=9)
    rows = [("blur.tex", blur, {"radius": 2}, _assert_equal),
            ("sharpen.tex", sharpen, {"amount": 1.0}, _assert_within_tol),
            ("f$gain probe", _GAIN, {"gain": 1.2}, _assert_equal)]
    real_place, real_invoke = C._params_on_device, C._invoke_cg
    interactive_thread = threading.current_thread()
    for dev in _DEVICES:
        for name, code, params, same in rows:
            calls = {"placements": 0, "codegen calls": 0}

            # C2 (v0.46 Phase C): count a call only when it comes from THIS (the
            # interactive test) thread. The last cook of a row can submit a background
            # warm job (CC-5); for a program with no real torch.compile cost (the
            # codegen-only eager adapter) that job's own call into `_invoke_cg` can land
            # within microseconds of submission, racing this row's own `finally` (or even
            # a later drain point) with no reliable ordering. That call always runs on a
            # background-pool worker thread, never on this one, so gating on thread
            # identity is a timing-independent way to count "how many INTERACTIVE cooks
            # reached codegen" — the only thing this row's `want` below describes —
            # rather than hoping the race resolves a particular way. The real function is
            # still invoked unconditionally either way; only the count is thread-gated.
            def place(*a, **k):
                if threading.current_thread() is interactive_thread:
                    calls["placements"] += 1
                return real_place(*a, **k)

            def invoke(*a, **k):
                if threading.current_thread() is interactive_thread:
                    calls["codegen calls"] += 1
                return real_invoke(*a, **k)

            try:
                binds = {"image": img.to(dev), **params}
                with cold_engine_state():
                    ref = _interp_like_engine(code, binds, dev)
                    C._params_on_device, C._invoke_cg = place, invoke
                    try:
                        for i in range(cooks):
                            tier_trace.reset()
                            res = tex_engine.cook(code, dict(binds), device_mode=dev,
                                                  compile_mode="auto")
                            _served_by(f"[{dev}] {name} cook {i + 1}", "codegen")
                            same(ref, res.outputs["OUT"], f"[{dev}] {name} cook {i + 1}")
                        # Flush this row's own async work before moving to the next row
                        # (or restoring the real functions) — belt and suspenders on top
                        # of the thread gate above; see cold_engine_state's own drain for
                        # the cross-block leak this closes (B5#1).
                        snapshot = dict(calls)
                        C._drain_bg_for_test()
                    finally:
                        C._params_on_device, C._invoke_cg = real_place, real_invoke
                learns = dev != "cpu" and name == "f$gain probe"
                want = ({"placements": cooks, "codegen calls": cooks + 1} if learns
                        else {"placements": 0, "codegen calls": cooks})
                assert snapshot == want, f"[{dev}] {name}: {snapshot}, expected {want}"
                r.ok(f"[{dev}] {name}: {cooks} cooks served by codegen "
                     f"({'bit-exact' if same is _assert_equal else 'within 1e-5'}), "
                     f"{snapshot['placements']} placements, {snapshot['codegen calls']} codegen calls")
            except Exception as e:
                r.fail(f"placement learned once [{dev}] {name}", f"{type(e).__name__}: {e}")


def test_codegen_param_default_route_unmoved(r: SubTestResult):
    print("\n--- default compile_mode='none' stencil route: unmoved; the opt-in route serves it ---")
    # The default route sends an exact fetch stencil through codegen WITHOUT placing params.
    # blur.tex reads `i$radius` only as a kernel radius (read back to the host, never
    # broadcast), so codegen serves it on every device. Multiplying the result by a float
    # param is the broadcast that falls back on CUDA; on the default route it still does,
    # because placing params there would add a host-to-device copy and a device sync to every
    # default cook — a perf decision of its own. Pixels are bit-exact on every leg, and the
    # same broadcast program under compile_mode="auto" is served by codegen.
    with open(os.path.join(_EXAMPLES, "blur.tex"), encoding="utf-8") as f:
        blur = f.read()
    blur_amt = blur.replace("@OUT = sum / count;", "f$amt = 1.0;\n@OUT = sum / count * $amt;")
    img = make_img(1, 20, 24, 3, seed=8)
    rows = [
        # name, code, params, compile_mode, device -> (tier, fallback_from)
        ("blur.tex i$radius", blur, {"radius": 2}, "none",
         {"cpu": ("codegen", None), "cuda": ("codegen", None)}),
        ("blur.tex x f$amt", blur_amt, {"radius": 1, "amt": 0.8}, "none",
         {"cpu": ("codegen", None), "cuda": ("interpreter", "codegen")}),
        ("blur.tex x f$amt", blur_amt, {"radius": 1, "amt": 0.8}, "auto",
         {"cpu": ("codegen", None), "cuda": ("codegen", None)}),
    ]
    for name, code, params, mode, want in rows:
        for dev in _DEVICES:
            try:
                binds = {"image": img.to(dev), **params}
                tier_trace.reset()
                res = tex_engine.cook(code, dict(binds), device_mode=dev, compile_mode=mode)
                _served_by(f"[{dev}] {name} compile_mode={mode}", *want[dev])
                ref = _interp_like_engine(code, binds, dev)
                _assert_equal(ref, res.outputs["OUT"], f"[{dev}] {name} compile_mode={mode}")
                r.ok(f"[{dev}] {name} compile_mode={mode}: served by {want[dev][0]}"
                     f"{' (fallback)' if want[dev][1] else ''}, bit-exact")
            except Exception as e:
                r.fail(f"stencil route [{dev}] {name} compile_mode={mode}", f"{type(e).__name__}: {e}")
