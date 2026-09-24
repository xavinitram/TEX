"""
v0.42 HOSTAUDIT-4 — a public `.code` on every exception type a host catches.

An embedding host keys its own notification folding on `(speaker, code)` (compass
§3 item 6: "diagnostic codes should be stable enough to fold notifications on"). Each
of `LexerError` / `ParseError` / `TypeCheckError` / `InterpreterError` already carries
the E-code string it was built with — it just lived on the private `_code` a host has
no business reading. This pins the public read-only `.code` property added to each:
same string, no behaviour or message change, no new import cycle (each property reads
an attribute the class already stored on itself).

`TEXCompileError` (ENG-4's public compile-error type) is deliberately NOT given a
`.code` here: it wraps a LIST of `TEXDiagnostic`s (one per error), each of which
already has a public `.code` dataclass field — see `tex_compiler/diagnostics.py`
`TEXDiagnostic.code`. A single `.code` on the wrapper would have to pick one of
several, which is a design question this ask does not open.
"""
from TEX_Wrangle.tex_compiler.ast_nodes import SourceLoc
from TEX_Wrangle.tex_compiler.lexer import LexerError
from TEX_Wrangle.tex_compiler.parser import ParseError
from TEX_Wrangle.tex_compiler.type_checker import TypeCheckError
from TEX_Wrangle.tex_runtime.interpreter import InterpreterError


def test_hostaudit4_lexer_error_exposes_public_code(r):
    e = LexerError("bad token", SourceLoc(1, 1), code="E1042")
    if e.code == "E1042" and e.code == e._code:
        r.ok("LexerError.code mirrors the E-code it was built with")
    else:
        r.fail("LexerError.code", f"got {e.code!r}, want 'E1042' (== ._code)")


def test_hostaudit4_parse_error_exposes_public_code(r):
    e = ParseError("bad syntax", SourceLoc(2, 3), code="E2042")
    if e.code == "E2042" and e.code == e._code:
        r.ok("ParseError.code mirrors the E-code it was built with")
    else:
        r.fail("ParseError.code", f"got {e.code!r}, want 'E2042' (== ._code)")


def test_hostaudit4_type_check_error_exposes_public_code(r):
    e = TypeCheckError("bad type", SourceLoc(4, 5), code="E3042")
    if e.code == "E3042" and e.code == e._code:
        r.ok("TypeCheckError.code mirrors the E-code it was built with")
    else:
        r.fail("TypeCheckError.code", f"got {e.code!r}, want 'E3042' (== ._code)")


def test_hostaudit4_interpreter_error_exposes_public_code(r):
    e = InterpreterError("cook failed", code="E6003")
    if e.code == "E6003" and e.code == e._code:
        r.ok("InterpreterError.code mirrors the E-code it was built with")
    else:
        r.fail("InterpreterError.code", f"got {e.code!r}, want 'E6003' (== ._code)")


def test_hostaudit4_interpreter_error_code_default_unchanged(r):
    # Additive means the default codes every raiser already relies on (E6000 etc.)
    # must not move just because a reader was added.
    e = InterpreterError("cook failed")
    if e.code == "E6000":
        r.ok("InterpreterError's default code (E6000) is unchanged")
    else:
        r.fail("InterpreterError default code", f"got {e.code!r}, want 'E6000'")


def test_hostaudit4_code_is_read_only(r):
    e = InterpreterError("cook failed", code="E6003")
    try:
        e.code = "E9999"
    except AttributeError:
        r.ok("InterpreterError.code refuses assignment (read-only property)")
    else:
        r.fail("InterpreterError.code read-only",
               "assigning e.code did not raise — it is not a read-only property")
