"""Codegen matches the interpreter on the two values it read at the wrong RANK.

Invariant 2 is bit-exactness, not a tolerance, and the interpreter is the oracle. Two
constructs reached codegen with a rank the interpreter never gives them, and rank is what
every "is this value per-pixel?" test reads — so LANGUAGE.md §7.1's rule ("`break`,
`continue` and `return` under a per-pixel `if` act on **every** pixel") was applied by one
tier and not the other. Both defects returned 4 where the interpreter returned 0:

  * **a vec/colour `$param` component.** The interpreter binds a `v3$tint` widget with
    `vec_list_to_tensor` -> `[1,1,1,3]`, so `$tint.r` is rank 3. The generated preamble
    converted the same list with `as_tensor`, leaving `[3]`, so `$tint.r` came out 0-dim
    and `if ($tint.r > 0.5) { break; }` took the uniform branch. Fixed at the single
    invocation seam (`codegen._invoke_cg` -> `_stage_vec_params`), NOT in the emitter, so
    no program's generated source moves.
  * **`fi` inside a loop.** `codegen._SPATIAL_BUILTINS` — the set that keeps a loop off the
    scalar fast path — listed `u v ix iy` and claimed in a comment that `fi` was 0-dim. It
    is `[B,1,1]`. A loop whose only tensor signal was `fi` compiled scalar and `.item()`-ed
    the frame index. The existing per-pixel-control-flow pin has an `fi` row and stayed
    green because its probe body reads `@A`, which forces the tensor loop.

`test_codegen_spatial_builtins_match_interpreter_ranks` is the derivation that would have
caught the second one: it reads the ranks the interpreter actually binds instead of
trusting a hand-written list, and reds in BOTH directions (a non-0-dim builtin missing
from the set, or a 0-dim one added to it).

`test_codegen_vec_param_staging_leaves_emitted_code_alone` pins the invisibility property
the fix was chosen for: the preamble still converts a `$param` with `as_tensor`, so the
emitted source for every program that was already correct is byte-identical.
"""
import torch

from helpers import *
from failure_harness import run_tier

from TEX_Wrangle.tex_runtime.codegen import _SPATIAL_BUILTINS, _invoke_cg, is_vec_param_list
from TEX_Wrangle.tex_runtime.compiled import _params_on_device
from TEX_Wrangle.tex_runtime.interpreter import _BUILTIN_NAMES

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]


def _both_tiers(code, bindings, B=1, H=4, W=4):
    """(interp OUT, codegen OUT). A decline is a failure: these rows pin BOTH tiers."""
    interp, cg = run_both(code, bindings, B=B, H=H, W=W)
    if cg is None:
        raise AssertionError("codegen declined the program, so only one tier was measured")
    return interp["OUT"], cg["OUT"]


def _uniq(t):
    return sorted(set(round(x, 5) for x in t[..., 0].flatten().tolist()))


# ── the programs ────────────────────────────────────────────────────────────────────
#
# Every `want` is the INTERPRETER's answer, read off the oracle, not a preference. The
# loop rows keep `@A` out of the loop body on purpose: a body that touches a binding
# forces codegen's tensor loop and hides the scalar-classification half of the defect.

_VEC_ROWS = [
    ("$tint.r break in a for loop",
     "v3$tint = vec3(0.0, 0.0, 0.0);\nfloat acc = 0.0;\n"
     "for (int k = 0; k < 4; k++) { if ($tint.r > 0.5) { break; } acc = acc + 1.0; }\n"
     "@OUT = vec3(acc) + @A * 0.0;", {"tint": [0.0, 0.0, 0.0]}, [0.0]),
    ("$tint.b break in a while loop",
     "v3$tint = vec3(0.0, 0.0, 0.0);\nfloat acc = 0.0;\nint i = 0;\n"
     "while (i < 4) { if ($tint.b > 0.5) { break; } acc = acc + 1.0; i = i + 1; }\n"
     "@OUT = vec3(acc) + @A * 0.0;", {"tint": [0.1, 0.2, 0.3]}, [0.0]),
    ("$tint.g continue in a for loop",
     "v3$tint = vec3(0.0, 0.0, 0.0);\nfloat acc = 0.0;\n"
     "for (int k = 0; k < 3; k++) { acc = acc + 1.0; if ($tint.g > 2.0) { continue; }"
     " acc = acc + 10.0; }\n@OUT = vec3(acc) + @A * 0.0;",
     {"tint": [0.4, 0.5, 0.6]}, [3.0]),
    ("$col.r return in a user function (outside any loop)",
     "c$col;\nfloat f() { if ($col.r > 0.5) { return 7.0; } return 3.0; }\n"
     "@OUT = vec3(f()) + @A * 0.0;", {"col": [0.2, 0.7, 0.1]}, [7.0]),
    ("$uv.x in a per-pixel if outside a loop",
     "v2$uv = vec2(0.0, 0.0);\nfloat x = 0.0;\n"
     "if ($uv.x > 0.5) { x = 1.0; } else { x = 2.0; }\n@OUT = vec3(x) + @A * 0.0;",
     {"uv": [0.25, 0.75]}, [2.0]),
    ("$rgba.a in a ternary inside a loop",
     "v4$rgba = vec4(0.0, 0.0, 0.0, 1.0);\nfloat acc = 0.0;\n"
     "for (int k = 0; k < 3; k++) { acc = acc + ($rgba.a > 0.5 ? 2.0 : 1.0); }\n"
     "@OUT = vec3(acc) + @A * 0.0;", {"rgba": [0.1, 0.2, 0.3, 0.9]}, [6.0]),
    # Value rows: the staging must not disturb an ordinary read (these were already green).
    ("$tint components read straight out",
     "v3$tint = vec3(0.0, 0.0, 0.0);\n@OUT = vec3($tint.r, $tint.g, $tint.b) + @A * 0.0;",
     {"tint": [0.25, 0.5, 0.75]}, [0.25]),
    ("$tint multiplied into the image",
     "v3$tint = vec3(1.0, 1.0, 1.0);\n@OUT = @A.rgb * $tint;",
     {"tint": [0.9, 0.5, 0.25]}, None),
    ("$tint mixed with a scalar param",
     "f$g = 0.5;\nv3$tint = vec3(1.0, 1.0, 1.0);\n@OUT = mix(@A.rgb, $tint, $g);",
     {"tint": [0.2, 0.7, 0.1], "g": 0.35}, None),
]

_FI_ROWS = [
    ("fi break in a scalar-bodied for loop",
     "float acc = 0.0;\n"
     "for (int k = 0; k < 4; k++) { if (fi > 5.0) { break; } acc = acc + 1.0; }\n"
     "@OUT = vec3(acc) + @A * 0.0;", {}, [0.0]),
    ("fi continue in a scalar-bodied for loop",
     "float acc = 0.0;\n"
     "for (int k = 0; k < 3; k++) { acc = acc + 1.0; if (fi > 5.0) { continue; }"
     " acc = acc + 10.0; }\n@OUT = vec3(acc) + @A * 0.0;", {}, [3.0]),
    ("fi break in a scalar-bodied while loop",
     "float acc = 0.0;\nint i = 0;\n"
     "while (i < 4) { if (fi > 5.0) { break; } acc = acc + 1.0; i = i + 1; }\n"
     "@OUT = vec3(acc) + @A * 0.0;", {}, [0.0]),
    ("a local initialized from fi, read in a loop",
     "float f = fi;\nfloat acc = 0.0;\n"
     "for (int k = 0; k < 4; k++) { if (f > 5.0) { break; } acc = acc + 1.0; }\n"
     "@OUT = vec3(acc) + @A * 0.0;", {}, [0.0]),
    ("fi in a ternary inside a scalar-bodied loop",
     "float acc = 0.0;\n"
     "for (int k = 0; k < 3; k++) { acc = acc + (fi > 5.0 ? 2.0 : 1.0); }\n"
     "@OUT = vec3(acc) + @A * 0.0;", {}, [3.0]),
    ("fi and a vec param component in the same loop",
     "v3$tint = vec3(0.0, 0.0, 0.0);\nfloat acc = 0.0;\n"
     "for (int k = 0; k < 4; k++) { if (fi + $tint.r > 5.0) { break; } acc = acc + 1.0; }\n"
     "@OUT = vec3(acc) + @A * 0.0;", {"tint": [0.0, 0.0, 0.0]}, [0.0]),
    # Value row: fi outside any loop was always right and must stay right.
    ("fi read outside a loop",
     "@OUT = vec3(fi * 0.25) + @A * 0.0;", {}, None),
]


def _run_rows(r, rows, tag, B=1):
    for label, code, params, want in rows:
        try:
            binds = {"A": make_img(B, 4, 5, 3, seed=11), **params}
            oi, oc = _both_tiers(code, binds, B=B)
            if not torch.equal(oi, oc):
                raise AssertionError(
                    f"interp != codegen, maxdiff {(oi - oc).abs().max().item():.3e}; "
                    f"interp {_uniq(oi)} vs codegen {_uniq(oc)}")
            if want is not None:
                for tier, out in (("interp", oi), ("codegen", oc)):
                    got = _uniq(out)
                    assert got == want, f"{tier}: acc {got}, expected {want}"
            r.ok(f"{tag}: {label} (B={B})")
        except Exception as e:
            r.fail(f"{tag}: {label} (B={B})", f"{type(e).__name__}: {e}")


def test_codegen_vec_param_component_matches_interpreter(r: SubTestResult):
    print("\n--- vec/colour $param components: codegen == interpreter, both tiers ---")
    _run_rows(r, _VEC_ROWS, "vec param")


def test_codegen_fi_in_loop_matches_interpreter(r: SubTestResult):
    print("\n--- fi inside a loop: codegen == interpreter, both tiers ---")
    _run_rows(r, _FI_ROWS, "fi")
    # B>1 is the case the scalar fast path could not even execute (`.item()` on a
    # multi-element frame index), so it fell back and got the right pixels for the wrong
    # reason; the rank fix makes the tier itself right at every batch size.
    _run_rows(r, _FI_ROWS, "fi", B=3)


def test_codegen_value_parity_on_every_device_and_precision(r: SubTestResult):
    print("\n--- the same rows through the engine tiers, per device and precision ---")
    # "codegen" is the plain generated-function tier, "auto" the tiering gate (torch.compile
    # where it wins). precision="auto" is invariant 10's opt-in lever: it may decline fp16
    # and serve fp32, which is fine — the claim is that whatever it serves matches the
    # interpreter under the SAME request.
    rows = [(f"vec/{l}", c, p) for l, c, p, _w in _VEC_ROWS] + \
           [(f"fi/{l}", c, p) for l, c, p, _w in _FI_ROWS]
    for dev in _DEVICES:
        for precision in ("fp32", "auto"):
            for label, code, params in rows:
                what = f"[{dev}/{precision}] {label}"
                try:
                    binds = {"A": make_img(2, 6, 5, 3, seed=12).to(dev), **params}
                    ref = run_tier(code, binds, "interp", device=dev, precision=precision)["OUT"]
                    for tier in ("codegen", "auto"):
                        got = run_tier(code, binds, tier, device=dev, precision=precision)["OUT"]
                        assert ref.shape == got.shape and ref.dtype == got.dtype, \
                            (f"{tier}: {tuple(got.shape)}/{got.dtype} vs interpreter "
                             f"{tuple(ref.shape)}/{ref.dtype}")
                        if not torch.equal(ref, got):
                            raise AssertionError(
                                f"{tier}: not bit-exact, maxdiff "
                                f"{(ref.float() - got.float()).abs().max().item():.3e}")
                    r.ok(f"{what}: codegen and auto both bit-exact with the interpreter")
                except Exception as e:
                    r.fail(what, f"{type(e).__name__}: {e}")


def test_codegen_spatial_builtins_match_interpreter_ranks(r: SubTestResult):
    print("\n--- _SPATIAL_BUILTINS == the builtins the interpreter binds non-0-dim ---")
    try:
        # Read the ranks off the oracle rather than trusting the hand-written set. `fi` was
        # missing from it for exactly as long as the comment beside it said `fi` was 0-dim.
        names = sorted(n for n in _BUILTIN_NAMES if n not in ("frame", "fps", "time"))
        code = "@OUT = vec3(" + " + ".join(names) + ") * 0.0 + @A * 0.0;"
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types={"A": TEXType.VEC3, "OUT": TEXType.VEC3},
                         source=code).check(prog)
        interp = Interpreter()
        interp.execute(prog, {"A": make_img(2, 4, 5, 3, seed=13)}, tm,
                       device="cpu", output_names=["OUT"])
        bound = {n: interp.env[n] for n in names if n in interp.env}
        missing = sorted(n for n in names if n not in bound)
        assert not missing, f"the interpreter bound none of {missing} — the probe is wrong"
        non_scalar = {n for n, t in bound.items() if torch.is_tensor(t) and t.dim() > 0}
        assert non_scalar == set(_SPATIAL_BUILTINS), (
            f"_SPATIAL_BUILTINS is {sorted(_SPATIAL_BUILTINS)} but the interpreter binds "
            f"{sorted(non_scalar)} non-0-dim; a builtin missing from the set compiles loops "
            f"scalar that must stay tensor, one added wrongly costs the fast path")
        r.ok(f"_SPATIAL_BUILTINS == {sorted(non_scalar)}, derived from the interpreter's ranks")
    except Exception as e:
        r.fail("_SPATIAL_BUILTINS derivation", f"{type(e).__name__}: {e}")


def test_codegen_vec_param_staging_is_narrow(r: SubTestResult):
    print("\n--- the staging touches vecN number lists and nothing else ---")
    rows = [
        ("vec2 list", [0.1, 0.2], True), ("vec3 list", [0.1, 0.2, 0.3], True),
        ("vec4 list", [0.1, 0.2, 0.3, 0.4], True), ("vec3 tuple", (0.1, 0.2, 0.3), True),
        ("int-valued vec3", [1, 2, 3], True),
        ("scalar float", 0.5, False), ("string", "abc", False),
        ("1-element list", [0.5], False), ("5-element array param", [1, 2, 3, 4, 5], False),
        ("list of tensors (a host batch list)", [torch.zeros(1, 2, 2, 3)], False),
        ("2 tensors (a host batch list)", [torch.zeros(1, 2, 2, 3)] * 2, False),
        ("list of bools", [True, False, True], False),
        ("nested list", [[0.1, 0.2], [0.3, 0.4]], False),
        ("tensor", torch.zeros(3), False),
    ]
    for label, value, want in rows:
        try:
            assert is_vec_param_list(value) is want, \
                f"is_vec_param_list -> {is_vec_param_list(value)}, expected {want}"
            r.ok(f"is_vec_param_list: {label} -> {want}")
        except Exception as e:
            r.fail(f"is_vec_param_list: {label}", f"{type(e).__name__}: {e}")

    # The seam itself: a vecN list becomes exactly what the interpreter binds, in place,
    # on the cook device and in the cook's dtype; everything else is left for the preamble.
    for dev in _DEVICES:
        for dtype in (None, torch.float32, torch.float16):
            what = f"[{dev}/{dtype}] _invoke_cg stages a vec3 param like the interpreter"
            try:
                code = "v3$tint = vec3(0.0, 0.0, 0.0);\n@OUT = @A.rgb * $tint;"
                prog = Parser(Lexer(code).tokenize(), source=code).parse()
                tm = TypeChecker(binding_types={"A": TEXType.VEC3, "tint": TEXType.VEC3,
                                                "OUT": TEXType.VEC3}, source=code).check(prog)
                cg = try_compile(prog, tm)
                assert cg is not None, "codegen declined the probe program"
                binds = {"A": make_img(1, 4, 5, 3, seed=14).to(dev).to(dtype or torch.float32),
                         "tint": [0.25, 0.5, 0.75], "raw": [1, 2, 3, 4, 5], "s": "keep"}
                _invoke_cg(cg, {}, binds, TEXStdlib.get_functions(),
                           torch.device(dev), (1, 4, 5), dtype)
                t = binds["tint"]
                assert torch.is_tensor(t), f"$tint stayed {type(t).__name__}"
                assert tuple(t.shape) == (1, 1, 1, 3), f"$tint staged {tuple(t.shape)}, want (1,1,1,3)"
                assert t.device.type == torch.device(dev).type, f"$tint on {t.device}"
                assert t.dtype == (dtype or torch.float32), f"$tint dtype {t.dtype}"
                assert binds["raw"] == [1, 2, 3, 4, 5], "a 5-element array param was restaged"
                assert binds["s"] == "keep", "a string param was restaged"
                r.ok(what)
            except Exception as e:
                r.fail(what, f"{type(e).__name__}: {e}")

    # The CUDA placement retry is the OTHER converter of a `$param` on this tier, and it
    # ran `as_tensor` on every non-tensor value — including the vec class the seam has just
    # staged at the right rank, which would put the divergence straight back on the opt-in
    # CUDA route. It must leave that class alone. Unit-level on purpose: reaching the retry
    # for real needs a first call that raises on a non-CPU device, so a pixel test would pin
    # this only on a CUDA runner and only for the programs that happen to raise.
    what = "_params_on_device leaves a vec param to the seam and still places a scalar one"
    try:
        code = ("f$gain = 1.0;\nv3$tint = vec3(0.0, 0.0, 0.0);\n"
                "@OUT = @A.rgb * $tint * $gain;")
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types={"A": TEXType.VEC3, "tint": TEXType.VEC3,
                                        "gain": TEXType.FLOAT, "OUT": TEXType.VEC3},
                         source=code).check(prog)
        cg = try_compile(prog, tm)
        assert cg is not None, "codegen declined the probe program"
        placed = _params_on_device(cg, prog, {"A": make_img(1, 4, 5, 3, seed=15),
                                              "tint": [0.25, 0.5, 0.75], "gain": 1.25},
                                   torch.device("cpu"))
        assert "tint" not in placed, (
            "the placement retry converted a vec param again; it would overwrite the seam's "
            "[1,1,1,C] staging with a rank-1 tensor on the opt-in CUDA route")
        assert "gain" in placed, "the placement retry stopped placing a scalar param"
        r.ok(what)
    except Exception as e:
        r.fail(what, f"{type(e).__name__}: {e}")


def test_codegen_vec_param_staging_leaves_emitted_code_alone(r: SubTestResult):
    print("\n--- the fix is invisible: the preamble still converts a $param with as_tensor ---")
    try:
        code = ("v3$tint = vec3(0.0, 0.0, 0.0);\nfloat acc = 0.0;\n"
                "for (int k = 0; k < 4; k++) { if ($tint.r > 0.5) { break; } acc = acc + 1.0; }\n"
                "@OUT = vec3(acc) + @A * 0.0;")
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types={"A": TEXType.VEC3, "tint": TEXType.VEC3,
                                        "OUT": TEXType.VEC3}, source=code).check(prog)
        src = try_compile(prog, tm, fingerprint="parity-probe")._tex_src
        assert "_bind['tint']" in src, "the $param preamble hoist moved"
        assert "_torch.as_tensor(" in src, (
            "the preamble stopped converting a $param with as_tensor — the staging belongs at "
            "the invocation seam precisely so every already-correct program's emitted source "
            "stays byte-identical")
        assert "vec_list_to_tensor" not in src and "view(1, 1, 1" not in src, \
            "the reshape leaked into the emitted source"
        r.ok("the generated preamble is unchanged; the rank is fixed at the invocation seam")
    except Exception as e:
        r.fail("emitted-code invisibility", f"{type(e).__name__}: {e}")
