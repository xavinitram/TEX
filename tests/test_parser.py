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


def test_array_decl_no_hang(r: SubTestResult):
    """Regression: a misplaced-bracket array declaration (`float[5] arr`, the
    C#/Java style) once sent the parser into an unbounded loop that consumed
    gigabytes before being killed. It must now terminate quickly with a single
    clear diagnostic, and the canonical `float arr[5]` form must still parse."""
    print("\n--- Array Declaration Hang Regression ---")
    import threading

    def parse_with_timeout(code, timeout=10.0):
        """Parse in a daemon thread so a regression hangs only this thread, not
        the whole suite. Returns ('timeout', None), ('ok', program), or
        ('raised', exception)."""
        out = {}

        def worker():
            try:
                toks = Lexer(code).tokenize()
                out["value"] = Parser(toks, source=code).parse()
                out["status"] = "ok"
            except BaseException as exc:  # capture any failure for inspection
                out["error"] = exc
                out["status"] = "raised"

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return "timeout", None
        if out.get("status") == "ok":
            return "ok", out.get("value")
        return "raised", out.get("error")

    repro = "float[5] arr = {1.0, 2.0, 3.0, 4.0, 5.0};\n@OUT = vec3(length(arr));"

    # 1. The exact reported repro must terminate (the core regression).
    try:
        status, payload = parse_with_timeout(repro)
        assert status != "timeout", "parser hung on `float[5] arr = {...}`"
        # Malformed syntax: must surface as a clean error, not silent success.
        assert status == "raised", f"expected a parse error, got status={status!r}"
        assert isinstance(payload, (ParseError, TEXMultiError)), \
            f"unexpected error type: {type(payload).__name__}"
        r.ok("array hang: `float[5] arr = {...}` terminates with an error")
    except Exception as e:
        r.fail("array hang: `float[5] arr = {...}` terminates with an error",
               f"{e}\n{traceback.format_exc()}")

    # 2. The diagnostic should explain the actual fix (size after the name).
    try:
        _, payload = parse_with_timeout(repro)
        msg = str(payload)
        assert "after the name" in msg, f"unhelpful message: {msg!r}"
        r.ok("array hang: diagnostic points to `type name[size]`")
    except Exception as e:
        r.fail("array hang: diagnostic points to `type name[size]`",
               f"{e}\n{traceback.format_exc()}")

    # 3. The empty-size variant `float[] arr` must also terminate cleanly.
    try:
        status, _ = parse_with_timeout("float[] arr = {1.0, 2.0}; @OUT = vec3(1.0);")
        assert status == "raised", f"expected error/termination, got status={status!r}"
        r.ok("array hang: `float[] arr = {...}` terminates with an error")
    except Exception as e:
        r.fail("array hang: `float[] arr = {...}` terminates with an error",
               f"{e}\n{traceback.format_exc()}")

    # 4. Canonical TEX syntax `type name[N]` must still parse successfully.
    try:
        status, prog = parse_with_timeout(
            "float arr[5] = {1.0, 2.0, 3.0, 4.0, 5.0};\n@OUT = vec3(length(arr));")
        assert status == "ok", f"canonical array decl failed to parse: {status!r}"
        decl = prog.statements[0]
        assert decl.__class__.__name__ == "ArrayDecl", \
            f"expected ArrayDecl, got {decl.__class__.__name__}"
        assert decl.name == "arr" and decl.size == 5
        r.ok("array hang: canonical `float arr[5] = {...}` still parses")
    except Exception as e:
        r.fail("array hang: canonical `float arr[5] = {...}` still parses",
               f"{e}\n{traceback.format_exc()}")

    # 5. A spread of malformed array-ish inputs must all terminate (no hang),
    #    whether or not they error.
    try:
        nasty = [
            "int[3] a = {1,2,3}; @OUT=vec3(1.0);",
            "vec3[2] a = {vec3(1.0), vec3(2.0)}; @OUT=vec3(1.0);",
            "float[5][3] a;",
            "float[5 arr = {1.0};",
            "{1.0, 2.0};",
        ]
        for code in nasty:
            status, _ = parse_with_timeout(code)
            assert status != "timeout", f"parser hung on: {code!r}"
        r.ok("array hang: assorted malformed inputs all terminate")
    except Exception as e:
        r.fail("array hang: assorted malformed inputs all terminate",
               f"{e}\n{traceback.format_exc()}")
