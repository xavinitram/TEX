"""SIMP-6 — one row per error code, triggered through the public surface.

`tests/test_simp6_error_codes.py` counts the codes no test names. This file is where that
count is paid down: for every code with a reachable trigger, the smallest TEX program that
draws it, checked through `tex_api.check` — the same call an editor's live-lint makes, so
a row that passes is evidence a host really can receive the code, not that a private
helper can be talked into constructing one.

Breadth over depth on purpose. A row asserts the code appears and that it appears as an
error with something to show the user; it does not pin the message text (that belongs to
whoever writes the message) or the caret column. What it defends is the thing an editor
binds to: this program, that code.

The programs are grouped the way the codes are — lexer, parser, then the type checker's
own families — so a reader asking what one of them means finds an example beside it. Where
a row's program draws more than one diagnostic (an unused-variable advisory usually tags
along), only the expected code is asserted.

Codes are spelled in the tables and NOWHERE ELSE in this file: the sibling ratchet counts
a code as tested when a test file names it, so naming one in a comment here would retire
it without testing anything.

Codes with no row are codes with no trigger reachable from `check`/`compile`: a constructor
default no call site omits, a guard the grammar forecloses, or a runtime code that needs an
execution. The ratchet in the sibling file still counts them and names them on every run.

No CUDA, no host and no compiler: the front end is pure Python plus torch's presence.
"""
from helpers import SubTestResult


def _codes(source):
    """Every diagnostic `tex_api.check` returns for `source`, as (code, severity, message)."""
    from TEX_Wrangle import tex_api
    return [(d.code, d.severity, d.message) for d in tex_api.check(source, {})]


def _row(r, code, source, what):
    """Assert `source` draws `code` as an error with a message, through `check`."""
    try:
        got = _codes(source)
    except Exception as e:                       # check() is contractually total
        r.fail(f"{code} ({what})", f"check() raised {type(e).__name__}: {e}")
        return
    hit = [g for g in got if g[0] == code]
    if not hit:
        r.fail(f"{code} ({what})",
               f"program {source!r} drew {[g[0] for g in got] or 'nothing'}, not {code}. "
               f"Either the trigger stopped working or the code moved; fix the program or "
               f"retire the row (and put {code} back in the ratchet's backlog).")
        return
    _, severity, message = hit[0]
    if severity != "error":
        r.fail(f"{code} ({what})", f"severity is {severity!r}, expected 'error'")
        return
    if not message.strip():
        r.fail(f"{code} ({what})", "the diagnostic carries no message to show the user")
        return
    r.ok(f"{code}: {what}")


# ── Lexer ─────────────────────────────────────────────────────────────

_LEXER = (
    ("E1002", '@OUT = vec4(0x);', "a hex literal with no digits"),
    ("E1006", '@OUT = vec4(1.0); string s = "abc', "a string literal the file ends inside"),
    ("E1008", "@OUT = vec4(1.0); float a = 1.0 \\ 2.0;", "a character TEX has no meaning for"),
)


def test_simp6_lexer_code_rows(r: SubTestResult):
    """Codes the lexer raises before the parser ever sees a token."""
    for code, source, what in _LEXER:
        _row(r, code, source, what)


# ── Parser ────────────────────────────────────────────────────────────

_PARSER = (
    ("E2001", 'const x = 1.0; @OUT = vec4(x);', "a keyword where a type belongs"),
    ("E2003", '@OUT = vec4(1.0) +', "an operator with nothing after it"),
    ("E2004", 'float arr[0]; @OUT = vec4(1.0);', "an array declared with size zero"),
    ("E2005", 'float arr[]; @OUT = vec4(1.0);', "an array with neither a size nor an initializer"),
    ("E2006", 'float[3] arr; @OUT = vec4(1.0);', "the array size written before the name"),
    ("E2010", '@OUT = vec4(1.0)', "a statement with no `;`"),
    ("E2011", 'f$gain = 1.0 [min: 0, min: 1]; @OUT = vec4($gain);', "a repeated widget metadata key"),
    ("E2020", 'float f(float a, b) { return a; } @OUT = vec4(f(1.0));', "a function parameter with no type"),
)


def test_simp6_parser_code_rows(r: SubTestResult):
    """Codes the parser raises: a program that does not have a shape."""
    for code, source, what in _PARSER:
        _row(r, code, source, what)


# ── Type checker: declarations, functions, control flow ───────────────

_DECLARATIONS = (
    ("E3002", 'break; @OUT = vec4(1.0);', "`break` with no loop around it"),
    ("E3010", 'float f() { return 1.0; } float f() { return 2.0; } @OUT = vec4(f());',
     "one function name defined twice"),
    ("E3012", 'return 1.0; @OUT = vec4(1.0);', "`return` outside any function"),
    ("E3013", 'float f() { return vec3(1.0); } @OUT = vec4(f());',
     "a return whose type is not the declared one"),
    ("E3014", 'float f() { float g() { return 1.0; } return 1.0; } @OUT = vec4(f());',
     "a function defined inside a function"),
    ("E3015",
     'float total = 0.0; for (int i = 0; i < 2; i = i + 1) { '
     'float f(float x) { if (x > 0.5) { break; } return x; } total = total + f(1.0); } '
     '@OUT = vec4(total);',
     "a bare `break` inside a function defined inside a loop"),
)


def test_simp6_declaration_code_rows(r: SubTestResult):
    """Codes about where a declaration or a jump is allowed to appear."""
    for code, source, what in _DECLARATIONS:
        _row(r, code, source, what)


# ── Type checker: arrays ──────────────────────────────────────────────

_ARRAYS = (
    ("E3101", 'float a[2] = {1.0, 2.0}; int b[2] = a; @OUT = vec4(1.0);',
     "copying a float array into an int array"),
    ("E3102", 'float arr[3] = {1.0, 2.0}; @OUT = vec4(arr[0]);',
     "an initializer with fewer elements than the declared size"),
    ("E3103", 'float arr[2000]; @OUT = vec4(arr[0]);', "an array past the size ceiling"),
    ("E3800", 'float x = 1.0; float y = x[0]; @OUT = vec4(y);', "indexing something that is not an array"),
)


def test_simp6_array_code_rows(r: SubTestResult):
    """Codes about arrays: their element type, their size, and indexing them."""
    for code, source, what in _ARRAYS:
        _row(r, code, source, what)


# ── Type checker: bindings and parameters ─────────────────────────────

_BINDINGS = (
    ("E3201", 'f$gain = 1.0; $gain.r = 0.5; @OUT = vec4($gain);',
     "writing to a $parameter, which is a widget input"),
    ("E3202", '@OUT = vec4(@A); f$A = 1.0;', "one name used as both @wire and $parameter"),
    ("E3204", 'const float x = 1.0; x = 2.0; @OUT = vec4(x);', "assigning to a const"),
)


def test_simp6_binding_code_rows(r: SubTestResult):
    """Codes about the @wire / $parameter surface a host wires up."""
    for code, source, what in _BINDINGS:
        _row(r, code, source, what)


# ── Type checker: expressions ─────────────────────────────────────────

_EXPRESSIONS = (
    ("E3400", 'string s = (1.0 > 0.5) ? "a" : 1.0; @OUT = vec4(1.0);',
     "a ternary with a string arm and a numeric arm"),
    ("E3401", 'string a = "x"; string b = "y"; string c = a - b; @OUT = vec4(1.0);',
     "an operator strings do not have"),
    ("E3402", '@OUT = vec4(1.0) * mat3(1.0);', "a matrix multiplied in the wrong order"),
    ("E3500", 'if (vec3(1.0)) { } @OUT = vec4(1.0);', "a vector used as a condition"),
    ("E3501", 'float arr[3] = {1.0, 2.0, 3.0}; if (arr) { } @OUT = vec4(1.0);',
     "an array used as a condition"),
    ("E3600", '@OUT = vec4("a", 1.0, 1.0, 1.0);', "a string handed to a vector constructor"),
    ("E3700", 'string s = string(vec3(1.0)); @OUT = vec4(1.0);', "casting a vector to a string"),
    ("E4000", 'vec3(1.0) = vec3(2.0); @OUT = vec4(1.0);', "an expression used as an assignment target"),
)


def test_simp6_expression_code_rows(r: SubTestResult):
    """Codes about what an expression may be built out of, and what it may be assigned to."""
    for code, source, what in _EXPRESSIONS:
        _row(r, code, source, what)


# ── The other public surface ──────────────────────────────────────────

def test_simp6_compile_carries_the_same_code_as_check(r: SubTestResult):
    """`check` returns diagnostics and `compile` raises them — same code either way.

    A host uses both: the editor lints with `check`, the cook compiles with `compile`. If
    the two disagreed about which code a program draws, a user would be told one thing
    while typing and another on the button press. The rows above all go through `check`,
    so this row ties the other surface to it on one program.
    """
    from TEX_Wrangle import tex_api
    from TEX_Wrangle.tex_compiler.diagnostics import TEXCompileError

    code, source, what = _PARSER[5]          # the missing-semicolon row
    from_check = [c for c, _, _ in _codes(source)]
    try:
        tex_api.compile(source, {})
    except TEXCompileError as e:
        from_compile = [d.code for d in e.diagnostics]
    except Exception as e:
        r.fail("compile() agrees with check()",
               f"compile() raised {type(e).__name__}, not the public TEXCompileError: {e}")
        return
    else:
        r.fail("compile() agrees with check()",
               f"compile() accepted a program check() rejected with {from_check}")
        return

    if code not in from_compile:
        r.fail("compile() agrees with check()",
               f"check() drew {from_check} for {what}, compile() drew {from_compile}")
        return
    r.ok(f"{code} reaches a host through check() and through compile()")
