"""LANG-L2 — `E3015`: a bare `break`/`continue` inside a function body has no loop of its
own to leave, even when a loop is lexically wrapped around the function's DEFINITION.

The defect (`TRK-28`, `docs/masked-control-flow.md` §3): `type_checker._check_function_def`
pushed a scope for a function body but never saved/reset `_loop_depth`, so a function
defined inside a loop inherited that loop's depth and `_check_break_continue`'s `E3002`
guard never fired. The two tiers then disagreed about which loop a `break` inside such a
function would leave (interpreter: the loop the *call* sits in; codegen: raises a bare
`_CgBreak` past the nested `def`, out of the cook — worse than a value divergence). The
design note rules this construct has no defensible masked meaning at any language level
and refuses it outright, ungated on the `//!tex` pragma, with its own code (`E3015`) rather
than reusing `E3002` — whose text ("outside of a loop") would be false here, since a reader
can see a loop wrapped around their function.

These rows are the kill switch for the mutation-harness entries in
`tests/mutation_check.py`'s LANG-L2 section: removing the `_loop_depth`/`_in_function_body`
save-and-reset in `_check_function_def` must turn every "refused" row below into a silent
pass, and that is exactly what a mutant deleting it produces.
"""
from helpers import *
from TEX_Wrangle import tex_api


# The exact TRK-28 program from docs/masked-control-flow.md §0/§3: a function, defined
# inside a loop, holding a bare `break`.
_TRK28_SRC = """
float total = 0.0;
for (int i = 0; i < 2; i = i + 1) {
  float nudge(float x) { if (x > 0.5) { break; } return x + 1.0; }
  for (int j = 0; j < 3; j = j + 1) { total = total + nudge(@A.r); }
}
@OUT = vec4(total, total, total, 1.0);
"""


def _diag_code(exc):
    """The structured code off a raised TypeCheckError, the way other E30xx rows read it."""
    return getattr(getattr(exc, "diagnostic", None), "code", None)


def test_lang_l2_e3015_trk28_refused_at_compile_time(r: SubTestResult):
    """The TRK-28 program is refused with E3015 before either tier ever sees it — closing
    the invariant-2 divergence by removing the input, not by choosing a tier's answer."""
    print("\n--- LANG-L2: TRK-28 program refused with E3015 ---")

    # 1. tex_api.check() — the editor-lint surface a host calls.
    try:
        diags = tex_api.check(_TRK28_SRC, {"A": TEXType.VEC4})
        codes = [d.code for d in diags]
        assert "E3015" in codes, f"tex_api.check() drew {codes}, not E3015"
        assert "E3002" not in codes, f"E3002 fired too — the two codes should be exclusive: {codes}"
        r.ok("tex_api.check() draws E3015 for the TRK-28 program")
    except Exception as e:
        r.fail("LANG-L2 tex_api.check E3015", f"{type(e).__name__}: {e}")

    # 2. The real compile path (helpers.check_code -> TypeChecker.check(), which raises).
    try:
        raised = None
        try:
            check_code(_TRK28_SRC, {"A": TEXType.VEC4})
        except Exception as e:
            raised = e
        assert raised is not None, "the TRK-28 program type-checked with no error at all"
        code = _diag_code(raised)
        assert code == "E3015", f"wrong error code: {code!r} (raised={raised!r})"
        r.ok("check() raises E3015 for the TRK-28 program (compile-time refusal)")
    except Exception as e:
        r.fail("LANG-L2 check() E3015", f"{type(e).__name__}: {e}")

    # 3. helpers.run_both drives type-checking ONCE, ahead of BOTH the interpreter and
    #    try_compile() — so a refusal here means neither tier is ever asked for an answer.
    #    Before this fix, the interpreter alone could answer (wrongly); after it, neither can.
    try:
        raised = None
        try:
            run_both(_TRK28_SRC, {"A": make_img(1, 1, 4, 4, seed=1)}, B=1, H=1, W=4)
        except Exception as e:
            raised = e
        assert raised is not None, "run_both() produced an answer for a program with no defensible meaning"
        code = _diag_code(raised)
        assert code == "E3015", f"run_both() raised {type(raised).__name__} with code {code!r}, not E3015"
        r.ok("run_both() never reaches either tier: the shared front end refuses first")
    except Exception as e:
        r.fail("LANG-L2 run_both E3015", f"{type(e).__name__}: {e}")


def test_lang_l2_e3015_ungated_on_pragma(r: SubTestResult):
    """The refusal is unconditional: an old pragma, a too-new one, or none at all all draw
    E3015 — this is a breaking change on the 0.23/0.24 surface too, deliberately."""
    print("\n--- LANG-L2: E3015 is ungated on the //!tex pragma ---")
    for pragma in ("", "//!tex 0.23\n", "//!tex 0.24\n", "//!tex 0.25\n", "//!tex 9.9\n"):
        src = pragma + _TRK28_SRC
        try:
            diags = tex_api.check(src, {"A": TEXType.VEC4})
            codes = [d.code for d in diags]
            assert "E3015" in codes, f"pragma={pragma.strip()!r} drew {codes}, not E3015"
            r.ok(f"E3015 fires under pragma={pragma.strip()!r}")
        except Exception as e:
            r.fail(f"LANG-L2 pragma {pragma.strip()!r}", f"{type(e).__name__}: {e}")


def test_lang_l2_e3015_fires_with_no_loop_at_all(r: SubTestResult):
    """Per the design note's implementation shape, E3015 fires for ANY break/continue at
    the top of a function body — whether or not a loop is lexically visible anywhere. A
    function is its own loop scope regardless of what wraps its definition (or doesn't)."""
    print("\n--- LANG-L2: E3015 fires even with no textual loop anywhere ---")
    cases = (
        ("float f() { break; return 1.0; }\n@OUT = vec4(f());", "break, top-level function, no loop"),
        ("float f() { continue; return 1.0; }\n@OUT = vec4(f());", "continue, top-level function, no loop"),
    )
    for src, what in cases:
        try:
            raised = None
            try:
                check_code(src)
            except Exception as e:
                raised = e
            assert raised is not None, f"{what}: type-checked with no error"
            code = _diag_code(raised)
            assert code == "E3015", f"{what}: wrong code {code!r}"
            r.ok(f"E3015: {what}")
        except Exception as e:
            r.fail(f"LANG-L2 no-loop {what}", f"{type(e).__name__}: {e}")


def test_lang_l2_e3002_still_fires_outside_any_function(r: SubTestResult):
    """The family boundary: a bare break/continue with no enclosing loop and no enclosing
    function keeps its original E3002 — this fix must not widen E3015 past function bodies."""
    print("\n--- LANG-L2: E3002 unaffected outside a function ---")
    cases = (
        ("break; @OUT = vec4(1.0);", "break at top level"),
        ("continue; @OUT = vec4(1.0);", "continue at top level"),
        ("if (1.0 > 0.0) { break; } @OUT = vec4(1.0);", "break nested in an if, still top level"),
    )
    for src, what in cases:
        try:
            raised = None
            try:
                check_code(src)
            except Exception as e:
                raised = e
            assert raised is not None, f"{what}: type-checked with no error"
            code = _diag_code(raised)
            assert code == "E3002", f"{what}: got {code!r}, expected E3002 (unchanged)"
            r.ok(f"E3002 (unchanged): {what}")
        except Exception as e:
            r.fail(f"LANG-L2 E3002 {what}", f"{type(e).__name__}: {e}")


def test_lang_l2_message_text_is_true(r: SubTestResult):
    """E3015's message must not claim the program is 'outside of a loop' (E3002's text,
    false here since a loop is visibly wrapped around the definition) and must carry a hint
    that explains the actual rule."""
    print("\n--- LANG-L2: E3015's message and hint ---")
    try:
        diags = tex_api.check(_TRK28_SRC, {"A": TEXType.VEC4})
        hit = [d for d in diags if d.code == "E3015"]
        assert hit, f"no E3015 among {[d.code for d in diags]}"
        msg = hit[0].message
        assert "outside of a loop" not in msg, f"E3015 reused E3002's (false, here) wording: {msg!r}"
        assert "function" in msg.lower(), f"E3015's message doesn't mention the function scope: {msg!r}"
        r.ok(f"E3015 message is scoped to the function, not to 'outside of a loop': {msg!r}")
    except Exception as e:
        r.fail("LANG-L2 message text", f"{type(e).__name__}: {e}")


def test_lang_l2_break_continue_legal_in_functions_own_loop(r: SubTestResult):
    """A break/continue inside a loop the FUNCTION ITSELF declares stays legal — E3015 must
    not overreach past the one construct it exists to refuse."""
    print("\n--- LANG-L2: break/continue in the function's OWN loop stay legal ---")
    cases = (
        ("float f() { float acc = 0.0; for (int i = 0; i < 3; i = i + 1) "
         "{ if (acc > 1.0) { break; } acc = acc + 1.0; } return acc; }\n"
         "for (int k = 0; k < 2; k = k + 1) { @OUT = vec4(f()); }",
         "break in the function's own for-loop, function called from an (unrelated) outer loop"),
        ("float f() { float acc = 0.0; for (int i = 0; i < 3; i = i + 1) "
         "{ if (acc > 1.0) { continue; } acc = acc + 1.0; } return acc; }\n@OUT = vec4(f());",
         "continue in the function's own for-loop, no outer loop at all"),
    )
    for src, what in cases:
        try:
            type_map, checker = check_code(src)
            r.ok(f"legal (no error): {what}")
        except Exception as e:
            r.fail(f"LANG-L2 legal {what}", f"{type(e).__name__}: {e}")


def test_lang_l2_return_inside_function_in_loop_stays_legal(r: SubTestResult):
    """`return` inside a function defined inside a loop is unaffected by E3015 (it stays
    legal at every language level) and both tiers still agree bit-exactly — the codegen
    half of this fix (`_emit_function_def` scoping `_use_native_flow_control`/
    `_scalar_loop`) must not regress the one jump `break`'s refusal leaves reachable."""
    print("\n--- LANG-L2: return inside a function-in-a-loop stays legal, tiers agree ---")
    src = """
        float total = 0.0;
        for (int i = 0; i < 2; i = i + 1) {
          float pick(float x) { if (x > 0.5) { return x * 10.0; } return x * 100.0; }
          total = total + pick(@A.r);
        }
        @OUT = vec4(total, total, total, 1.0);
    """
    try:
        type_map, checker = check_code(src, {"A": TEXType.VEC4})
        r.ok("legal (no error): return inside a function defined inside a loop")
    except Exception as e:
        r.fail("LANG-L2 return legal", f"{type(e).__name__}: {e}")
        return

    try:
        r_img = torch.tensor([0.10, 0.30, 0.70, 0.90]).view(1, 1, 4, 1).repeat(1, 1, 1, 4)
        interp_res, cg_res = run_both(src, {"A": r_img}, B=1, H=1, W=4)
        if cg_res is None:
            r.skip("return-in-function-in-loop", "codegen unsupported (interp only)")
        else:
            max_diff = (interp_res["OUT"].float() - cg_res["OUT"].float()).abs().max().item()
            assert max_diff < 1e-5, f"tiers disagree: max diff={max_diff}"
            r.ok(f"return-in-function-in-loop: interp == codegen (max diff {max_diff:.2e})")
    except Exception as e:
        r.fail("LANG-L2 return codegen equivalence", f"{type(e).__name__}: {e}")


def test_lang_l2_e3015_is_in_the_e301x_family_and_unclaimed_before(r: SubTestResult):
    """E3015 is a genuinely new code in the E301x function-definition family (E3010, E3011,
    E3012, E3013, E3014 are the others) — this just pins that it is wired up and reachable,
    complementing the static family-membership facts in test_simp6_error_codes.py."""
    print("\n--- LANG-L2: E3015 reachable and distinct from its siblings ---")
    try:
        diags = tex_api.check(_TRK28_SRC, {"A": TEXType.VEC4})
        e301x = sorted({d.code for d in diags if d.code.startswith("E301")})
        assert e301x == ["E3015"], f"expected only E3015 from the E301x family, got {e301x}"
        r.ok("E3015 is the only E301x code drawn by the TRK-28 program")
    except Exception as e:
        r.fail("LANG-L2 E301x family", f"{type(e).__name__}: {e}")
