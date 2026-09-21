"""TRK-18 / FIX-3 — an `@`-bound Python scalar makes codegen fall back on every device.

A host may wire a plain Python number into an `@` slot instead of an image tensor
(a ComfyUI FLOAT/INT primitive plugged into an `@` input is exactly this shape). The
interpreter already copes: its per-cook binding loop mints ANY non-tensor/str/
list-or-tuple value into a tagged 0-dim device tensor (PERF-2). Codegen's emitted
preamble instead reads a wire binding raw (`_bind[name]`, no `as_tensor` guard —
unlike a `$param`'s `_get_param_local`), so the first tensor method a generated
expression calls on it — `_broadcast_pair`'s `.dim()`, reached from `@image * @k`'s
runtime-broadcast path — raised `'float' object has no attribute 'dim'` and the whole
cook fell back to the interpreter, on **every** device (the crash is in Python
dispatch, not a kernel).

Verbatim reproduction at base sha `d6fac03` (`tex_engine.cook`, `compile_mode="auto"`):
    tier_trace.last() == <TierRecord interpreter (fell back from codegen:
    'float' object has no attribute 'dim')>

Fixed at the single invocation seam (`codegen._invoke_cg` -> `_stage_wire_scalars`),
mirroring the interpreter's own conversion (`_tag_host_scalar`, same rounding) rather
than patching the emitted preamble — so no program's generated source moves, and the
fix reaches every codegen-derived tier (codegen-only, compiled/torch.compile-eager,
compiled/torch.compile-flat, the CUDA params-on-device retry) through the one function
they all call. `_stage_wire_scalars` is scoped to `wire` ('@') names ONLY: a `$param`
Python scalar keeps its own, separately-tracked CPU-tensor `as_tensor` staging
(TRK-15/TRK-17) untouched — this fix must not move where a `$param` lands by default.
"""
import torch

from helpers import *
from TEX_Wrangle.tex_cache import parse_and_split
from failure_harness import run_tier, compile_program, clone_bindings

from TEX_Wrangle.tex_runtime.codegen import (_invoke_cg, _wire_names,
                                             _stage_wire_scalars, try_compile)
from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute
from TEX_Wrangle.tex_runtime.stdlib import _HOST_SCALAR_ATTR

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]


def test_trk18_wire_scalar_served_by_codegen_not_fallback(r: SubTestResult):
    """The verbatim TRK-18 reproduction: codegen must now SERVE `@image * @k` with `@k`
    a plain Python float, not fall back to the interpreter."""
    print("\n--- TRK-18: an @-bound Python scalar is served by codegen ---")
    code = "@OUT = @image * @k;"
    bindings = {"image": make_img(1, 4, 4, 3, seed=20), "k": 1.2}
    try:
        prog, tm, outs = compile_program(code, bindings)
        result = _codegen_only_execute(prog, clone_bindings(bindings), tm, "cpu",
                                       output_names=outs, fingerprint="trk18-repro",
                                       time_context=None)
        assert result is not None and "OUT" in result
        r.ok("codegen served @image * @k directly (no exception, no fallback path taken)")
    except Exception as e:
        r.fail("codegen serves an @-bound Python scalar", f"{type(e).__name__}: {e}")


def test_trk18_wire_scalar_matches_interpreter(r: SubTestResult):
    """Interp and codegen must agree BIT-EXACTLY (invariant #2) on a wire scalar, across a
    few operator shapes — not just the one the tracker happened to reproduce with."""
    print("\n--- TRK-18: interp == codegen, bit-exact, for several @-scalar programs ---")
    rows = [
        ("wire scalar multiplied into an image (the tracker's repro)",
         "@OUT = @image * @k;", {"k": 1.2}),
        ("wire scalar added to an image",
         "@OUT = @image + @k;", {"k": 0.3}),
        ("wire scalar as an int (bool/int -> TEXType.INT)",
         "@OUT = @image * float(@k);", {"k": 2}),
        ("wire scalar read through a channel-broadcast ternary",
         "@OUT = (@k > 0.5) ? @image : @image * 0.0;", {"k": 0.9}),
        ("wire scalar feeding a host-scalar-reading builtin (gauss_blur's radius)",
         "@OUT = gauss_blur(@image, @k);", {"k": 0.6}),
        ("two wire scalars, one on each side of the broadcast",
         "@OUT = @image * @k - @j;", {"k": 1.5, "j": 0.1}),
        # Control: @image.r * @k never goes through _bp (channel access already reduces
        # the image side to a scalar-shaped field) — confirms it stays served, unmoved.
        ("wire scalar against a channel access (the tracker's un-broken control)",
         "@OUT = vec3(@image.r * @k);", {"k": 0.4}),
    ]
    for label, code, extra in rows:
        what = f"TRK-18: {label}"
        try:
            bindings = {"image": make_img(1, 4, 5, 3, seed=21), **extra}
            prog, tm, outs = compile_program(code, bindings)
            interp = Interpreter()
            interp_out = interp.execute(prog, clone_bindings(bindings), tm, device="cpu",
                                        output_names=outs)
            cg = try_compile(prog, tm)
            assert cg is not None, "codegen declined the probe program"
            cg_out = _codegen_only_execute(prog, clone_bindings(bindings), tm, "cpu",
                                           output_names=outs,
                                           fingerprint=f"trk18-{abs(hash(code))}",
                                           time_context=None)
            for name in outs:
                a, b = interp_out[name], cg_out[name]
                assert torch.equal(a, b), (
                    f"interp != codegen for '{name}': maxdiff "
                    f"{(a.float() - b.float()).abs().max().item():.3e}")
            r.ok(what)
        except Exception as e:
            r.fail(what, f"{type(e).__name__}: {e}")


def test_trk18_wire_scalar_every_device_and_precision(r: SubTestResult):
    """Same shape as test_codegen_value_parity's device/precision matrix: codegen and
    auto must both stay bit-exact with the interpreter for a wire scalar, everywhere."""
    print("\n--- TRK-18: @-scalar rows across device x precision, real tier entry points ---")
    code = "@OUT = @image * @k;"
    # precision="fp16" is deliberately absent: `tex_engine.prepare()` clamps it to "fp32"
    # for every compile_mode != "none" ("fp16 is an interpreter-only mode for now — the
    # compile/graph paths ... aren't validated for fp16 yet"), so a real cook never hands
    # raw fp16 to codegen/compiled/auto. Calling `run_tier` directly BYPASSES that clamp,
    # and doing so here reproduces a ~3-5e-4 codegen/interp divergence with ZERO wire
    # scalars involved (two plain image bindings) — a pre-existing, ask-unrelated gap in
    # already-documented "not validated" territory, matching test_codegen_value_parity.py's
    # own precision matrix (which tests fp32/auto only, for the same reason).
    for dev in _DEVICES:
        for precision in ("fp32", "auto"):
            what = f"[{dev}/{precision}] @image * @k"
            try:
                bindings = {"image": make_img(1, 4, 5, 3, seed=22).to(dev), "k": 1.2}
                ref = run_tier(code, bindings, "interp", device=dev, precision=precision)["OUT"]
                for tier in ("codegen", "compiled", "auto"):
                    got = run_tier(code, bindings, tier, device=dev, precision=precision)["OUT"]
                    assert ref.shape == got.shape and ref.dtype == got.dtype, (
                        f"{tier}: {tuple(got.shape)}/{got.dtype} vs interpreter "
                        f"{tuple(ref.shape)}/{ref.dtype}")
                    if not torch.equal(ref, got):
                        raise AssertionError(
                            f"{tier}: not bit-exact, maxdiff "
                            f"{(ref.float() - got.float()).abs().max().item():.3e}")
                r.ok(f"{what}: codegen/compiled/auto all bit-exact with the interpreter")
            except Exception as e:
                r.fail(what, f"{type(e).__name__}: {e}")


def test_trk18_wire_names_derivation(r: SubTestResult):
    """`_wire_names` collects exactly the `@`-bound names, never a `$param` name, and
    reaches names nested in a loop/if/function body (the generic `_iter_child_nodes`
    walk, not a shallow top-level-only scan)."""
    print("\n--- TRK-18: _wire_names collects wire names only, everywhere in the AST ---")
    code = (
        "f$g = 1.0;\n"
        "float acc = 0.0;\n"
        "for (int i = 0; i < 2; i++) { if (@cond > 0.5) { acc = acc + @k; } }\n"
        "float f() { return @nested; }\n"
        "@OUT = vec3(acc + f()) * $g + @image * 0.0;"
    )
    bindings = {"image": make_img(1, 4, 4, 3, seed=23), "k": 1.0, "cond": 1.0,
               "nested": 0.5, "g": 1.0}
    try:
        prog, tm, outs = compile_program(code, bindings)
        names = _wire_names(prog)
        # 'OUT' is itself an @-bound (wire) assignment target, so it belongs in the set too.
        assert names == frozenset({"image", "k", "cond", "nested", "OUT"}), (
            f"_wire_names -> {sorted(names)}, expected image/k/cond/nested/OUT (never 'g')")
        r.ok("_wire_names == every @-bound name, nested included, no $param name")
    except Exception as e:
        r.fail("_wire_names derivation", f"{type(e).__name__}: {e}")


def test_trk18_stage_wire_scalars_is_narrow(r: SubTestResult):
    """The seam itself, unit-level, called DIRECTLY (not through `_invoke_cg`'s full
    generated-function call — that would also execute `$g * @image` on a mixed CPU/CUDA
    pair, which is TRK-15/17's own separate, already-tracked device story, not this
    one's): `_stage_wire_scalars` converts a wire Python scalar into the SAME tagged
    0-dim device tensor `_tag_host_scalar` gives the interpreter, touches an already-
    tensor wire binding not at all, and — the hard constraint — leaves a `$param`
    binding completely alone (device, dtype, and Python-vs-tensor identity)."""
    print("\n--- TRK-18: _stage_wire_scalars touches wire scalars and nothing else ---")
    code = "f$g = 1.0;\n@OUT = @image * @k;"
    bindings = {"image": TEXType.VEC3, "k": TEXType.FLOAT, "g": TEXType.FLOAT,
               "OUT": TEXType.VEC3}
    prog = parse_and_split(code, bindings)
    tm = TypeChecker(binding_types=bindings, source=code).check(prog)
    cg = try_compile(prog, tm)
    assert cg is not None, "codegen declined the probe program"
    for dev in _DEVICES:
        for dtype in (None, torch.float32, torch.float16):
            what = f"[{dev}/{dtype}] _stage_wire_scalars stages @k, leaves $g and @image alone"
            try:
                img = make_img(1, 4, 5, 3, seed=24).to(dev).to(dtype or torch.float32)
                binds = {"image": img, "k": 1.2, "g": 0.75, "s_control": "keep"}
                cg._tex_wire_names = None   # fresh per (dev, dtype): don't reuse another
                                            # iteration's cached attribute on the same cg_fn
                _stage_wire_scalars(binds, torch.device(dev), dtype, cg, prog)
                # @k: became a tagged 0-dim tensor on the cook device, in the cook's dtype,
                # matching the value the interpreter would have tagged for the same input.
                k = binds["k"]
                assert torch.is_tensor(k), f"@k stayed {type(k).__name__}"
                assert tuple(k.shape) == (), f"@k staged at rank {k.dim()}, want 0-dim"
                assert k.device.type == torch.device(dev).type, f"@k landed on {k.device}"
                assert k.dtype == (dtype or torch.float32), f"@k dtype {k.dtype}"
                tag = getattr(k, _HOST_SCALAR_ATTR, None)
                assert tag is not None, "@k was minted without the host-scalar tag"
                interp_dtype = dtype or torch.float32
                want_tag = torch.scalar_tensor(1.2, dtype=interp_dtype).item()
                assert tag == want_tag, f"@k's tag {tag} != the interpreter's rounding {want_tag}"
                # @image: already a tensor — untouched (same object, not a copy).
                assert binds["image"] is img, "an already-tensor wire binding was replaced"
                # $g: a $param name must NEVER be staged here — it is not in _wire_names,
                # so it must reach this point exactly as the caller handed it: a raw
                # Python float, not a tensor, not moved to the cook device.
                assert binds["g"] == 0.75 and not torch.is_tensor(binds["g"]), (
                    f"$g was staged by the wire-scalar seam: {binds['g']!r} "
                    f"(type {type(binds['g']).__name__}) — this must stay the caller's own "
                    f"$param staging's job, never this one's")
                assert binds["s_control"] == "keep", "an unrelated string binding was touched"
                r.ok(what)
            except Exception as e:
                r.fail(what, f"{type(e).__name__}: {e}")


def test_trk18_no_program_arg_is_backward_compatible(r: SubTestResult):
    """`_invoke_cg`'s new `program=` parameter defaults to None and must skip the wire
    staging entirely when omitted — the exact shape of any caller that predates this ask
    (there are none left in the shipped tree, but the signature promises it)."""
    print("\n--- TRK-18: _invoke_cg(program=None) is a no-op for wire staging ---")
    code = "@OUT = @image * @k;"
    bindings = {"image": TEXType.VEC3, "k": TEXType.FLOAT, "OUT": TEXType.VEC3}
    prog = parse_and_split(code, bindings)
    tm = TypeChecker(binding_types=bindings, source=code).check(prog)
    try:
        cg = try_compile(prog, tm)
        assert cg is not None
        binds = {"image": make_img(1, 4, 4, 3, seed=25), "k": 1.2}
        try:
            _invoke_cg(cg, {}, binds, TEXStdlib.get_functions(),
                      torch.device("cpu"), (1, 4, 4))   # no program= -> old call shape
            raised = False
        except AttributeError:
            raised = True
        assert raised, ("without `program=`, @k should stay a raw float and crash inside "
                        "_bp exactly as TRK-18 reported — if it did not raise, staging ran "
                        "without being asked to")
        r.ok("omitting program= reproduces the pre-fix crash — the default is a true no-op")
    except Exception as e:
        r.fail("program=None backward compatibility", f"{type(e).__name__}: {e}")
