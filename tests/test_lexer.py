"""Lexer tests."""
from helpers import *


def test_lexer(r: SubTestResult):
    print("\n--- Lexer Tests ---")

    # Basic tokens
    try:
        tokens = Lexer("float x = 1.0;").tokenize()
        types = [t.type for t in tokens[:-1]]  # exclude EOF
        assert types == [TokenType.KW_FLOAT, TokenType.IDENT, TokenType.ASSIGN, TokenType.FLOAT_LIT, TokenType.SEMI], f"Got {types}"
        r.ok("basic tokens")
    except Exception as e:
        r.fail("basic tokens", str(e))

    # @ binding
    try:
        tokens = Lexer("@A.r + @B").tokenize()
        types = [t.type for t in tokens[:-1]]
        assert TokenType.AT_BINDING in types
        binding_tokens = [t for t in tokens if t.type == TokenType.AT_BINDING]
        assert binding_tokens[0].value == "A"
        assert binding_tokens[1].value == "B"
        r.ok("@ bindings")
    except Exception as e:
        r.fail("@ bindings", str(e))

    # Comments
    try:
        tokens = Lexer("x + y // comment\n+ z").tokenize()
        values = [t.value for t in tokens if t.type == TokenType.IDENT]
        assert values == ["x", "y", "z"], f"Got {values}"
        r.ok("line comments")
    except Exception as e:
        r.fail("line comments", str(e))

    try:
        tokens = Lexer("x /* block */ + y").tokenize()
        values = [t.value for t in tokens if t.type == TokenType.IDENT]
        assert values == ["x", "y"], f"Got {values}"
        r.ok("block comments")
    except Exception as e:
        r.fail("block comments", str(e))

    # Operators
    try:
        tokens = Lexer("a == b && c != d || e >= f").tokenize()
        ops = [t.type for t in tokens if t.type in (TokenType.EQ, TokenType.AND, TokenType.NEQ, TokenType.OR, TokenType.GTE)]
        assert len(ops) == 5
        r.ok("compound operators")
    except Exception as e:
        r.fail("compound operators", str(e))

    # Numbers
    try:
        tokens = Lexer("42 3.14 0xFF .5 1e3").tokenize()
        num_tokens = [t for t in tokens if t.type in (TokenType.INT_LIT, TokenType.FLOAT_LIT)]
        assert len(num_tokens) == 5, f"Got {len(num_tokens)}: {num_tokens}"
        r.ok("number literals")
    except Exception as e:
        r.fail("number literals", str(e))

    # Error: unexpected character
    try:
        Lexer("x $ y").tokenize()
        r.fail("unexpected char error", "Should have raised LexerError")
    except LexerError:
        r.ok("unexpected char error")
    except Exception as e:
        r.fail("unexpected char error", str(e))


def test_lexer_locations(r: SubTestResult):
    """Exact token/error line:col assertions. The lexer batches per-character
    scan loops (whitespace, comments, numbers, identifiers, strings), so these
    pin the location bookkeeping across every batched run kind."""
    print("\n--- Lexer Location Tests ---")

    def _locs(src):
        return [(t.type, t.value, t.loc.line, t.loc.col) for t in Lexer(src).tokenize()]

    def _error_of(src):
        try:
            Lexer(src).tokenize()
        except LexerError as e:
            e._build_diagnostic()
            return e
        return None

    # Token after a line comment starts on the next line
    try:
        toks = _locs("x // c\ny")
        assert toks[0] == (TokenType.IDENT, "x", 1, 1), toks[0]
        assert toks[1] == (TokenType.IDENT, "y", 2, 1), toks[1]
        assert toks[2] == (TokenType.EOF, "", 2, 2), toks[2]
        r.ok("loc: token after line comment")
    except Exception as e:
        r.fail("loc: token after line comment", str(e))

    # Token after a multi-line block comment
    try:
        toks = _locs("a /* x\nyy */ b")
        assert toks[0] == (TokenType.IDENT, "a", 1, 1), toks[0]
        assert toks[1] == (TokenType.IDENT, "b", 2, 7), toks[1]
        assert toks[2] == (TokenType.EOF, "", 2, 8), toks[2]
        r.ok("loc: token after multi-line block comment")
    except Exception as e:
        r.fail("loc: token after multi-line block comment", str(e))

    # Tokens around a string containing escapes
    try:
        toks = _locs('string s = "a\\tb"; f')
        assert toks[0] == (TokenType.KW_STRING, "string", 1, 1), toks[0]
        assert toks[1] == (TokenType.IDENT, "s", 1, 8), toks[1]
        assert toks[2] == (TokenType.ASSIGN, "=", 1, 10), toks[2]
        assert toks[3] == (TokenType.STRING_LIT, "a\tb", 1, 12), toks[3]
        assert toks[4] == (TokenType.SEMI, ";", 1, 18), toks[4]
        assert toks[5] == (TokenType.IDENT, "f", 1, 20), toks[5]
        assert toks[6] == (TokenType.EOF, "", 1, 21), toks[6]
        r.ok("loc: tokens around string with escapes")
    except Exception as e:
        r.fail("loc: tokens around string with escapes", str(e))

    # EOF location at the end of multi-line source
    try:
        toks = _locs("x\ny\n")
        assert toks[-1] == (TokenType.EOF, "", 3, 1), toks[-1]
        r.ok("loc: EOF at end of multi-line source")
    except Exception as e:
        r.fail("loc: EOF at end of multi-line source", str(e))

    # Two-char operator on a later line
    try:
        toks = _locs("a\n  <= b")
        assert toks[1] == (TokenType.LTE, "<=", 2, 3), toks[1]
        assert toks[2] == (TokenType.IDENT, "b", 2, 6), toks[2]
        r.ok("loc: two-char operator col")
    except Exception as e:
        r.fail("loc: two-char operator col", str(e))

    # Leading-dot float and exponent backtrack keep columns exact
    try:
        toks = _locs("x .5 y")
        assert toks[1] == (TokenType.FLOAT_LIT, ".5", 1, 3), toks[1]
        assert toks[2] == (TokenType.IDENT, "y", 1, 6), toks[2]
        toks = _locs("1e x")
        assert toks[0] == (TokenType.INT_LIT, "1", 1, 1), toks[0]
        assert toks[1] == (TokenType.IDENT, "e", 1, 2), toks[1]
        assert toks[2] == (TokenType.IDENT, "x", 1, 4), toks[2]
        r.ok("loc: dot-float and exponent backtrack cols")
    except Exception as e:
        r.fail("loc: dot-float and exponent backtrack cols", str(e))

    # E1003/E1004 point at the backslash
    try:
        e = _error_of('"ab\\')
        assert e is not None and e.diagnostic.code == "E1003", e
        assert (e.loc.line, e.loc.col) == (1, 4), e.loc
        e = _error_of('"ab\\q"')
        assert e is not None and e.diagnostic.code == "E1004", e
        assert (e.loc.line, e.loc.col) == (1, 4), e.loc
        r.ok("loc: string escape errors at the backslash")
    except Exception as e:
        r.fail("loc: string escape errors at the backslash", str(e))

    # E1005 points at the string start; E1001 at the block comment opener
    try:
        e = _error_of('"abc\ndef"')
        assert e is not None and e.diagnostic.code == "E1005", e
        assert (e.loc.line, e.loc.col) == (1, 1), e.loc
        e = _error_of("x /* y\nz")
        assert e is not None and e.diagnostic.code == "E1001", e
        assert (e.loc.line, e.loc.col) == (1, 4), e.loc
        r.ok("loc: unterminated string/comment error positions")
    except Exception as e:
        r.fail("loc: unterminated string/comment error positions", str(e))


def test_lexer_v11(r: SubTestResult):
    print("\n--- Lexer v1.1 Tests ---")

    # For keyword
    try:
        tokens = Lexer("for").tokenize()
        assert tokens[0].type == TokenType.KW_FOR
        r.ok("lex 'for' keyword")
    except Exception as e:
        r.fail("lex 'for' keyword", str(e))

    # ++ and --
    try:
        tokens = Lexer("i++ j--").tokenize()
        types = [t.type for t in tokens[:-1]]
        assert TokenType.PLUS_PLUS in types
        assert TokenType.MINUS_MINUS in types
        r.ok("lex ++ and --")
    except Exception as e:
        r.fail("lex ++ and --", str(e))

    # Compound assignments
    try:
        tokens = Lexer("a += b -= c *= d /= e").tokenize()
        types = [t.type for t in tokens[:-1]]
        assert TokenType.PLUS_ASSIGN in types
        assert TokenType.MINUS_ASSIGN in types
        assert TokenType.STAR_ASSIGN in types
        assert TokenType.SLASH_ASSIGN in types
        r.ok("lex compound assignments")
    except Exception as e:
        r.fail("lex compound assignments", str(e))
