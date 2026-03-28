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
