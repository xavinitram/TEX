"""Parser tests."""
from helpers import *


def test_parser(r: SubTestResult):
    print("\n--- Parser Tests ---")

    # Variable declaration
    try:
        tokens = Lexer("float x = 1.0;").tokenize()
        prog = Parser(tokens).parse()
        assert len(prog.statements) == 1
        assert prog.statements[0].__class__.__name__ == "VarDecl"
        r.ok("var declaration")
    except Exception as e:
        r.fail("var declaration", str(e))

    # Assignment to @OUT
    try:
        tokens = Lexer("@OUT = vec3(1.0, 0.0, 0.0);").tokenize()
        prog = Parser(tokens).parse()
        assert prog.statements[0].__class__.__name__ == "Assignment"
        r.ok("@OUT assignment")
    except Exception as e:
        r.fail("@OUT assignment", str(e))

    # If/else
    try:
        tokens = Lexer("if (x > 0.5) { @OUT = @A; } else { @OUT = @B; }").tokenize()
        prog = Parser(tokens).parse()
        assert prog.statements[0].__class__.__name__ == "IfElse"
        r.ok("if/else")
    except Exception as e:
        r.fail("if/else", str(e))

    # Operator precedence: a + b * c should parse as a + (b * c)
    try:
        tokens = Lexer("float x = a + b * c;").tokenize()
        prog = Parser(tokens).parse()
        init = prog.statements[0].initializer
        assert init.__class__.__name__ == "BinOp"
        assert init.op == "+"
        assert init.right.__class__.__name__ == "BinOp"
        assert init.right.op == "*"
        r.ok("operator precedence")
    except Exception as e:
        r.fail("operator precedence", str(e))

    # Ternary operator
    try:
        tokens = Lexer("float x = a > 0.5 ? 1.0 : 0.0;").tokenize()
        prog = Parser(tokens).parse()
        init = prog.statements[0].initializer
        assert init.__class__.__name__ == "TernaryOp"
        r.ok("ternary operator")
    except Exception as e:
        r.fail("ternary operator", str(e))

    # Vector constructor
    try:
        tokens = Lexer("vec4 c = vec4(1.0, 0.0, 0.0, 1.0);").tokenize()
        prog = Parser(tokens).parse()
        init = prog.statements[0].initializer
        assert init.__class__.__name__ == "VecConstructor"
        assert init.size == 4
        assert len(init.args) == 4
        r.ok("vec4 constructor")
    except Exception as e:
        r.fail("vec4 constructor", str(e))

    # vec3 constructor
    try:
        tokens = Lexer("vec3 c = vec3(0.5, 0.5, 0.5);").tokenize()
        prog = Parser(tokens).parse()
        init = prog.statements[0].initializer
        assert init.__class__.__name__ == "VecConstructor"
        assert init.size == 3
        r.ok("vec3 constructor")
    except Exception as e:
        r.fail("vec3 constructor", str(e))

    # Channel access
    try:
        tokens = Lexer("float r = @A.r;").tokenize()
        prog = Parser(tokens).parse()
        init = prog.statements[0].initializer
        assert init.__class__.__name__ == "ChannelAccess"
        assert init.channels == "r"
        r.ok("channel access")
    except Exception as e:
        r.fail("channel access", str(e))

    # Function call
    try:
        tokens = Lexer("float y = clamp(x, 0.0, 1.0);").tokenize()
        prog = Parser(tokens).parse()
        init = prog.statements[0].initializer
        assert init.__class__.__name__ == "FunctionCall"
        assert init.name == "clamp"
        assert len(init.args) == 3
        r.ok("function call")
    except Exception as e:
        r.fail("function call", str(e))

    # Cast expression
    try:
        tokens = Lexer("float y = float(x);").tokenize()
        prog = Parser(tokens).parse()
        init = prog.statements[0].initializer
        assert init.__class__.__name__ == "CastExpr"
        r.ok("cast expression")
    except Exception as e:
        r.fail("cast expression", str(e))

    # Parse error: missing semicolon
    try:
        tokens = Lexer("float x = 1.0").tokenize()
        Parser(tokens).parse()
        r.fail("missing semicolon error", "Should have raised ParseError")
    except ParseError:
        r.ok("missing semicolon error")
    except Exception as e:
        r.fail("missing semicolon error", str(e))

    # Nested expressions
    try:
        tokens = Lexer("float x = sin(clamp(@A.r * 2.0 - 1.0, -1.0, 1.0));").tokenize()
        prog = Parser(tokens).parse()
        assert len(prog.statements) == 1
        r.ok("nested expressions")
    except Exception as e:
        r.fail("nested expressions", str(e))


def test_parser_v11(r: SubTestResult):
    print("\n--- Parser v1.1 Tests ---")

    # For loop parsing
    try:
        tokens = Lexer("for (int i = 0; i < 10; i++) { float x = 1.0; }").tokenize()
        prog = Parser(tokens).parse()
        assert len(prog.statements) == 1
        assert prog.statements[0].__class__.__name__ == "ForLoop"
        r.ok("parse for loop")
    except Exception as e:
        r.fail("parse for loop", str(e))

    # Compound assignment parsing
    try:
        tokens = Lexer("x += 1.0;").tokenize()
        prog = Parser(tokens).parse()
        stmt = prog.statements[0]
        assert stmt.__class__.__name__ == "Assignment"
        assert stmt.value.__class__.__name__ == "BinOp"
        assert stmt.value.op == "+"
        r.ok("parse +=")
    except Exception as e:
        r.fail("parse +=", str(e))

    # Increment parsing
    try:
        tokens = Lexer("x++;").tokenize()
        prog = Parser(tokens).parse()
        stmt = prog.statements[0]
        assert stmt.__class__.__name__ == "Assignment"
        assert stmt.value.__class__.__name__ == "BinOp"
        assert stmt.value.op == "+"
        r.ok("parse x++")
    except Exception as e:
        r.fail("parse x++", str(e))

    # Decrement parsing
    try:
        tokens = Lexer("x--;").tokenize()
        prog = Parser(tokens).parse()
        stmt = prog.statements[0]
        assert stmt.__class__.__name__ == "Assignment"
        assert stmt.value.op == "-"
        r.ok("parse x--")
    except Exception as e:
        r.fail("parse x--", str(e))

    # For loop with compound update
    try:
        tokens = Lexer("for (int i = 0; i < 10; i += 2) { @OUT = @A; }").tokenize()
        prog = Parser(tokens).parse()
        loop = prog.statements[0]
        assert loop.__class__.__name__ == "ForLoop"
        assert loop.update.__class__.__name__ == "Assignment"
        r.ok("parse for with += update")
    except Exception as e:
        r.fail("parse for with += update", str(e))
