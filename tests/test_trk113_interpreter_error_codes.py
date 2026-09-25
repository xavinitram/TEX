"""TRK-113 — execution-level coverage for the interpreter's `E6xxx` codes.

`InterpreterError` (`tex_runtime/interpreter.py`) is the ONE exception class every
`E6xxx` code is raised through, and reaching one needs an EXECUTION — a compiled program
and real tensors — so it cannot be triggered through `tex_api.check`/`compile` alone, the
surface `test_simp6_error_code_rows.py` uses. That is why `test_simp6_error_codes.py`
still pins all eleven `E6xxx` codes in its untested backlog: paying that part of it down
is this file, one row per code, kept out of the sibling file on purpose (SIMP-6's own
docstring says so).

Of the eleven, seven turn out reachable through the ordinary front door — `check()` still
passes the program, and the failure only shows up once `Interpreter.execute` runs it, which
is a call every one of these rows makes directly (`_run` below), the same three-step
pipeline `tests/helpers.py::compile_and_run` uses, just with `bindings` handed separately
from the declared binding TYPES so a row can omit one at execute() time. One more
(`E6001`) is the multi-output surface asking for a name the program never assigned — also
no bypass needed. One (`E6020`'s OTHER raise site, an undefined variable) needs the type
checker skipped outright, because the checker forecloses that exact mistake from any real
source; `_run_unchecked` below does that the same way existing interpreter tests reach the
tree without going through `TypeChecker.check` first.

The remaining four are genuinely DEFENSIVE: a checker code forecloses the mistake from any
parsed source before the interpreter's branch could ever run, so there is no program to
compile that reaches them — hand-crafting one needs an AST the parser could not have
produced, which tests nothing about a real TEX program. Per the tracker row, each is
EXEMPTED with the checker code that shadows it, and `test_defensive_codes_stay_shadowed`
below keeps that pairing live: it is not a comment asserting the shadow, it is a program
that draws the paired code through `check()`, so a change that widens the checker and
un-shadows one of these four is caught here rather than by nobody. None of the four E6xxx
codes are spelled as string literals anywhere below (only their checker partners are, and
only where triggered) — the comments above name them in prose, which is exactly the kind of
mention `test_simp6_error_codes.py`'s tightened rule (TRK-112) does not count as tested, so
this file cannot accidentally move that ratchet's pin for a code it does not actually test.
"""
from helpers import *
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, InterpreterError


def _run(code, binding_types, bindings, output_names=("OUT",)):
    """check() -> execute(), the production seam, with `bindings` kept separate from the
    declared `binding_types` so a row can hand execute() FEWER bindings than the program
    declares (the "not connected" row needs exactly that gap)."""
    program = parse_and_split(code, binding_types)
    checker = TypeChecker(binding_types=binding_types, source=code)
    type_map = checker.check(program)
    interp = Interpreter()
    return interp.execute(program, bindings, type_map, device="cpu",
                          output_names=list(output_names))


def _run_unchecked(code, binding_types, bindings, output_names=("OUT",)):
    """Front end only — Lexer/Parser via `parse_and_split`, then straight to
    `Interpreter.execute` with an EMPTY type map, `TypeChecker.check` never called. Only
    for a mistake the checker forecloses from any real source, so there is no other way to
    hand the interpreter a program carrying it."""
    program = parse_and_split(code, binding_types)
    interp = Interpreter()
    return interp.execute(program, bindings, {}, device="cpu",
                          output_names=list(output_names))


def _expect_code(r, code, label, fn):
    """Call `fn()`; pass iff it raises `InterpreterError` with exactly `code`."""
    try:
        fn()
    except InterpreterError as e:
        if e._code == code:
            r.ok(f"{code}: {label}")
        else:
            r.fail(f"{code}: {label}", f"raised {e._code!r} instead: {e}")
        return
    except Exception as e:
        r.fail(f"{code}: {label}", f"{type(e).__name__}: {e}")
        return
    r.fail(f"{code}: {label}", "no error was raised")


# ── Reachable through the ordinary front door: check() passes, execute() fails ──

def test_trk113_e6001_requested_output_never_assigned(r: SubTestResult):
    """Multi-output `execute()` asked for a name the program never assigned to."""
    def go():
        code = "@OUT = vec4(1.0); @other = vec4(0.0);"
        _run(code, {}, {}, output_names=("OUT", "other", "missing_out"))
    _expect_code(r, "E6001", "a requested output name the program never assigned", go)


def test_trk113_e6004_channel_assign_target_not_a_variable_or_binding(r: SubTestResult):
    """`.r = value` on a call result: a channel assignment whose base is neither a
    variable nor an `@binding`."""
    def go():
        code = ("vec3 f(vec3 v) { return v; } "
                "f(vec3(1.0, 2.0, 3.0)).r = 9.0; @OUT = vec4(1.0);")
        _run(code, {}, {})
    _expect_code(r, "E6004", "a channel write whose target is a function call", go)


def test_trk113_e6005_array_index_assign_target_not_a_variable_or_binding(r: SubTestResult):
    """`[0] = value` on a call result: an array-index assignment whose base is neither a
    variable nor an `@binding` — `sort()` returns an array, but not a named one."""
    def go():
        code = ("float arr[3] = {1.0, 2.0, 3.0}; sort(arr)[0] = 9.0; "
                "@OUT = vec4(arr[0]);")
        _run(code, {}, {})
    _expect_code(r, "E6005", "an array-index write whose target is a function call", go)


def test_trk113_e6006_scatter_write_channel_count_mismatch(r: SubTestResult):
    """A scatter write's channel count is a RUNTIME property of the buffer a previous
    scatter allocated, not a static one — so the checker cannot see this mismatch:
    `@OUT` becomes a 1-channel mask on its first scatter write, then a second write hands
    it a vec3."""
    def go():
        code = "@OUT[0,0] = 0.5; @OUT[1,1] = vec3(1.0, 0.0, 0.0);"
        _run(code, {}, {})
    _expect_code(r, "E6006", "a vec3 scattered into a mask-shaped @OUT", go)


def test_trk113_e6020_binding_used_with_sample_syntax_but_not_an_image(r: SubTestResult):
    """`@S[x, y]` / `@S(u, v)` type-check against ANY binding type (the checker returns
    the binding's own type, or vec4 for a string) — only `Interpreter._require_image`
    rejects a non-image binding, at runtime."""
    def go():
        code = "@OUT = vec4(@S[0, 0]);"
        _run(code, {"S": TEXType.FLOAT, "OUT": TEXType.VEC4}, {"S": torch.scalar_tensor(0.5)})
    _expect_code(r, "E6020", "@S[0,0] where @S is a scalar, not an image", go)


def test_trk113_e6021_binding_declared_but_not_connected(r: SubTestResult):
    """A binding the program declares (and the checker types) but `execute()` is not
    handed at all — the engine's own forgiving lazy-skip gate (E6003) sits ABOVE this;
    calling `Interpreter.execute` directly, the way this row does, reaches the
    interpreter's OWN "not connected" check instead."""
    def go():
        code = "@OUT = @A;"
        _run(code, {"A": TEXType.VEC4, "OUT": TEXType.VEC4}, {})
    _expect_code(r, "E6021", "a declared binding execute() is not handed", go)


def test_trk113_e6051_stdlib_function_raises_on_a_per_pixel_scalar_argument(r: SubTestResult):
    """`erode`'s radius is typed `float` — the same type a per-pixel expression like
    `@A.r * 5.0` has — so the checker accepts it; `erode`'s host-side `.item()` (needed
    for its footprint/tiling, HALO-ARG) then raises a plain `RuntimeError` on a
    multi-element tensor, which `_eval_function_call`'s catch-all wraps as E6051."""
    def go():
        code = "@OUT = vec4(erode(@A, @A.r * 5.0), 1.0);"
        A = make_img(1, 4, 4, 3)
        _run(code, {"A": TEXType.VEC3, "OUT": TEXType.VEC4}, {"A": A})
    _expect_code(r, "E6051", "erode() given a per-pixel (non-scalar) radius", go)


# ── E6020's OTHER raise site: an undefined variable, which the checker forecloses ──

def test_trk113_e6020_undefined_variable_bypasses_the_checker(r: SubTestResult):
    """`Identifier` lookup's "not defined" branch: unreachable from any checked program
    (the type checker resolves every variable before execution, drawing its own code), so
    this row skips `TypeChecker.check` outright via `_run_unchecked` — a hand-parsed
    program, not a hand-built AST, since the parser itself has nothing against the name."""
    def go():
        code = "@OUT = vec4(undefined_var, 0.0, 0.0, 1.0);"
        _run_unchecked(code, {"OUT": TEXType.VEC4}, {})
    _expect_code(r, "E6020", "a variable no declaration introduces, checker skipped", go)


# ── Defensive: shadowed by a checker code, per the TRK-113 tracker row ──

# (E6xxx code, the checker code that forecloses it, one-line reason, a program that draws
# the checker code). The E6xxx code is deliberately not a string literal that would count
# as "naming" it under TRK-112's tightened rule — see the module docstring.
_DEFENSIVE_PAIRS = (
    ("E4000",
     "the statement-dispatch fallback (\"expected a recognized statement\"): the parser "
     "only ever emits statement node types the interpreter's dispatch table already "
     "covers, so the checker's own dispatch fallback (which every parsed program hits "
     "first) is the only one either side can reach.",
     "vec3(1.0) = vec3(2.0); @OUT = vec4(1.0);"),
    ("E3301",
     "an unrecognized single-channel name (e.g. `.q`) on a channel READ: the checker "
     "validates every channel letter against the same CHANNEL_MAP the interpreter uses, "
     "before execution ever sees the node.",
     "vec3 cc = vec3(1.0); float g = cc.q; @OUT = vec4(g);"),
    ("E3401",
     "an operator strings do not support: the checker rejects every string-string and "
     "string-numeric operator combination the interpreter's string-path fallback exists "
     "to catch, so no checked program reaches that fallback.",
     'string a = "x"; string b = "y"; string c = a - b; @OUT = vec4(1.0);'),
    ("E5001",
     "a function name that resolves to neither a stdlib nor a user function: the checker "
     "looks up every call against the same two tables (`FUNCTION_SIGNATURES` plus "
     "`_user_functions`) before execution, so the interpreter's own \"not recognized\" "
     "branch never sees a name the checker did not already reject.",
     "@OUT = vec4(not_a_real_function(1.0));"),
)


def test_trk113_defensive_codes_stay_shadowed(r: SubTestResult):
    """Each pinned-defensive `E6xxx` code stays exempt only while its checker partner
    really does reject the program first — this asserts that fact on a live program
    rather than trusting a comment. A future change that lets one of these programs
    through `check()` un-shadows the paired `E6xxx` code, and this row reds rather than
    the gap going unnoticed."""
    from TEX_Wrangle import tex_api
    for checker_code, reason, program in _DEFENSIVE_PAIRS:
        try:
            got = [d.code for d in tex_api.check(program, {})]
        except Exception as e:
            r.fail(f"defensive pairing for {checker_code}", f"check() raised {type(e).__name__}: {e}")
            continue
        if checker_code not in got:
            r.fail(f"defensive pairing for {checker_code}",
                   f"check() drew {got}, not {checker_code} — the shadowing program no "
                   f"longer draws its checker code; the paired interpreter code may now "
                   f"be reachable and need a real trigger row instead of an exemption.")
            continue
        r.ok(f"{checker_code} still forecloses its paired interpreter code before execution: {reason}")
