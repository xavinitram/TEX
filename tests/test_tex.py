"""
TEX Test Suite — comprehensive tests for lexer, parser, type checker, and interpreter.

Run with: python -m pytest tests/test_tex.py -v
Or standalone: python tests/test_tex.py
"""
from __future__ import annotations
import sys
import os
import traceback

# Add custom_nodes dir to path so package-relative imports work
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_custom_nodes_dir = os.path.dirname(_pkg_dir)
sys.path.insert(0, _custom_nodes_dir)

import shutil
import tempfile
import time
import pickle
from pathlib import Path

import torch
from TEX_Wrangle.tex_node import _prepare_output, _unwrap_latent, _infer_binding_type, _map_inferred_type
from TEX_Wrangle.tex_compiler.lexer import Lexer, LexerError, TokenType
from TEX_Wrangle.tex_compiler.parser import Parser, ParseError
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker, TypeCheckError, TEXType
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, InterpreterError
from TEX_Wrangle.tex_cache import TEXCache
from TEX_Wrangle.tex_runtime.compiled import execute_compiled, _plain_execute, clear_compiled_cache


# ── Test Utilities ─────────────────────────────────────────────────────

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def ok(self, name: str):
        self.passed += 1
        print(f"  PASS  {name}")

    def fail(self, name: str, msg: str):
        self.failed += 1
        self.errors.append(f"{name}: {msg}")
        print(f"  FAIL  {name}: {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailures:")
            for e in self.errors:
                print(f"  - {e}")
        print(f"{'='*60}")
        return self.failed == 0


def compile_and_run(code: str, bindings: dict, device: str = "cpu",
                    latent_channel_count: int = 0,
                    out_type: TEXType = TEXType.VEC4) -> torch.Tensor | str | dict:
    """Helper: compile and execute TEX code.

    Returns the @OUT value for single-output programs, or a dict of
    {name: tensor} for multi-output programs (when code assigns to
    names other than @OUT).
    """
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()

    binding_types = {}
    for name, val in bindings.items():
        if isinstance(val, str):
            binding_types[name] = TEXType.STRING
        elif isinstance(val, torch.Tensor):
            if val.dim() == 4:
                c = val.shape[-1]
                binding_types[name] = TEXType.VEC4 if c == 4 else TEXType.VEC3
            elif val.dim() == 3:
                binding_types[name] = TEXType.FLOAT
            else:
                binding_types[name] = TEXType.FLOAT
        else:
            binding_types[name] = TEXType.FLOAT

    checker = TypeChecker(binding_types=binding_types)
    type_map = checker.check(program)

    # Use multi-output path when code assigns to named outputs
    output_names = sorted(checker.assigned_bindings.keys()) if checker.assigned_bindings else None
    if output_names is None or output_names == ["OUT"]:
        # Single-output backward compat
        binding_types["OUT"] = out_type
        checker2 = TypeChecker(binding_types=binding_types)
        type_map = checker2.check(program)
        interp = Interpreter()
        return interp.execute(program, bindings, type_map, device=device,
                              latent_channel_count=latent_channel_count)
    else:
        interp = Interpreter()
        return interp.execute(program, bindings, type_map, device=device,
                              latent_channel_count=latent_channel_count,
                              output_names=output_names)


# ── Lexer Tests ────────────────────────────────────────────────────────

def test_lexer(r: TestResult):
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


# ── Parser Tests ───────────────────────────────────────────────────────

def test_parser(r: TestResult):
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


# ── Type Checker Tests ─────────────────────────────────────────────────

def test_type_checker(r: TestResult):
    print("\n--- Type Checker Tests ---")

    def check_code(code: str, bindings: dict[str, TEXType] | None = None):
        tokens = Lexer(code).tokenize()
        prog = Parser(tokens).parse()
        bt = bindings or {}
        bt.setdefault("OUT", TEXType.VEC4)
        checker = TypeChecker(binding_types=bt)
        return checker.check(prog), checker

    # Basic float variable
    try:
        type_map, checker = check_code("float x = 1.0;")
        r.ok("float variable")
    except Exception as e:
        r.fail("float variable", str(e))

    # Vec4 variable
    try:
        type_map, checker = check_code("vec4 c = vec4(1.0, 0.0, 0.0, 1.0);")
        r.ok("vec4 variable")
    except Exception as e:
        r.fail("vec4 variable", str(e))

    # Channel access type
    try:
        type_map, checker = check_code(
            "float r = @A.r;",
            {"A": TEXType.VEC4}
        )
        r.ok("channel access type")
    except Exception as e:
        r.fail("channel access type", str(e))

    # Binding reference tracking
    try:
        type_map, checker = check_code(
            "@OUT = @A;",
            {"A": TEXType.VEC4}
        )
        assert "A" in checker.referenced_bindings
        assert "OUT" in checker.referenced_bindings
        r.ok("binding reference tracking")
    except Exception as e:
        r.fail("binding reference tracking", str(e))

    # Type error: undefined variable
    try:
        check_code("float x = undefined_var;")
        r.fail("undefined variable error", "Should have raised TypeCheckError")
    except TypeCheckError:
        r.ok("undefined variable error")
    except Exception as e:
        r.fail("undefined variable error", str(e))

    # Swizzle type: .rgb returns vec3
    try:
        type_map, checker = check_code(
            "vec3 rgb = @A.rgb;",
            {"A": TEXType.VEC4}
        )
        r.ok("rgb swizzle")
    except Exception as e:
        r.fail("rgb swizzle", str(e))

    # Promotion: float + vec4
    try:
        type_map, checker = check_code(
            "vec4 result = @A + 0.5;",
            {"A": TEXType.VEC4}
        )
        r.ok("float+vec4 promotion")
    except Exception as e:
        r.fail("float+vec4 promotion", str(e))


# ── Interpreter Tests ──────────────────────────────────────────────────

def test_interpreter(r: TestResult):
    print("\n--- Interpreter Tests ---")

    # Create a test image: 1x4x4 RGB
    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 3)
    test_img4 = torch.rand(B, H, W, 4)
    test_mask = torch.rand(B, H, W)

    # Simple passthrough
    try:
        result = compile_and_run("@OUT = @A;", {"A": test_img4})
        assert result.shape == test_img4.shape, f"Shape mismatch: {result.shape} vs {test_img4.shape}"
        assert torch.allclose(result, test_img4, atol=1e-6)
        r.ok("passthrough")
    except Exception as e:
        r.fail("passthrough", f"{e}\n{traceback.format_exc()}")

    # Grayscale conversion
    try:
        code = """
        float gray = luma(@A);
        @OUT = vec3(gray, gray, gray);
        """
        result = compile_and_run(code, {"A": test_img})
        assert result.shape == (B, H, W, 3), f"Shape: {result.shape}"
        # Check that all channels are equal (grayscale)
        assert torch.allclose(result[..., 0], result[..., 1], atol=1e-5)
        assert torch.allclose(result[..., 1], result[..., 2], atol=1e-5)
        r.ok("grayscale")
    except Exception as e:
        r.fail("grayscale", f"{e}\n{traceback.format_exc()}")

    # Arithmetic
    try:
        code = """
        float x = 2.0;
        float y = 3.0;
        float z = x + y * 2.0;
        @OUT = vec3(z, z, z);
        """
        result = compile_and_run(code, {"A": test_img})
        # z = 2 + 3*2 = 8.0
        expected = 8.0
        assert torch.allclose(result[0, 0, 0, 0], torch.tensor(expected), atol=1e-5), f"Got {result[0,0,0,0].item()}"
        r.ok("arithmetic")
    except Exception as e:
        r.fail("arithmetic", f"{e}\n{traceback.format_exc()}")

    # Channel access
    try:
        code = """
        float r = @A.r;
        float g = @A.g;
        float b = @A.b;
        @OUT = vec3(b, g, r);
        """
        result = compile_and_run(code, {"A": test_img})
        # Red and blue should be swapped
        assert torch.allclose(result[..., 0], test_img[..., 2], atol=1e-5)
        assert torch.allclose(result[..., 2], test_img[..., 0], atol=1e-5)
        r.ok("channel access + swap")
    except Exception as e:
        r.fail("channel access + swap", f"{e}\n{traceback.format_exc()}")

    # Clamp function
    try:
        code = """
        @OUT = vec3(clamp(@A.r * 2.0, 0.0, 1.0), @A.g, @A.b);
        """
        result = compile_and_run(code, {"A": test_img})
        expected_r = torch.clamp(test_img[..., 0] * 2.0, 0.0, 1.0)
        assert torch.allclose(result[..., 0], expected_r, atol=1e-5)
        r.ok("clamp function")
    except Exception as e:
        r.fail("clamp function", f"{e}\n{traceback.format_exc()}")

    # Lerp (mix)
    try:
        img_a = torch.zeros(B, H, W, 3)
        img_b = torch.ones(B, H, W, 3)
        code = "@OUT = lerp(@A, @B, 0.5);"
        result = compile_and_run(code, {"A": img_a, "B": img_b})
        assert torch.allclose(result, torch.full((B, H, W, 3), 0.5), atol=1e-5)
        r.ok("lerp function")
    except Exception as e:
        r.fail("lerp function", f"{e}\n{traceback.format_exc()}")

    # Coordinate variables (u, v)
    try:
        code = """
        @OUT = vec3(u, v, 0.0);
        """
        result = compile_and_run(code, {"A": test_img})
        # u should go from 0 to 1 across width
        assert result[0, 0, 0, 0].item() < 0.01  # left edge
        assert result[0, 0, W-1, 0].item() > 0.99  # right edge
        # v should go from 0 to 1 across height
        assert result[0, 0, 0, 1].item() < 0.01  # top edge
        assert result[0, H-1, 0, 1].item() > 0.99  # bottom edge
        r.ok("coordinate variables (u, v)")
    except Exception as e:
        r.fail("coordinate variables (u, v)", f"{e}\n{traceback.format_exc()}")

    # Ternary operator
    try:
        code = """
        float val = u > 0.5 ? 1.0 : 0.0;
        @OUT = vec3(val, val, val);
        """
        result = compile_and_run(code, {"A": test_img})
        # Left half should be 0, right half should be 1
        assert result[0, 0, 0, 0].item() < 0.5  # left edge = 0
        assert result[0, 0, W-1, 0].item() > 0.5  # right edge = 1
        r.ok("ternary operator")
    except Exception as e:
        r.fail("ternary operator", f"{e}\n{traceback.format_exc()}")

    # If/else (vectorized)
    try:
        code = """
        if (u > 0.5) {
            @OUT = vec3(1.0, 0.0, 0.0);
        } else {
            @OUT = vec3(0.0, 0.0, 1.0);
        }
        """
        result = compile_and_run(code, {"A": test_img})
        # Left half should be blue (0,0,1), right half red (1,0,0)
        assert result[0, 0, 0, 2].item() > 0.5  # left edge = blue
        assert result[0, 0, W-1, 0].item() > 0.5  # right edge = red
        r.ok("if/else (vectorized)")
    except Exception as e:
        r.fail("if/else (vectorized)", f"{e}\n{traceback.format_exc()}")

    # Scalar parameter binding
    try:
        code = """
        @OUT = @A * @B;
        """
        result = compile_and_run(code, {"A": test_img, "B": 0.5})
        expected = test_img * 0.5
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("scalar parameter binding")
    except Exception as e:
        r.fail("scalar parameter binding", f"{e}\n{traceback.format_exc()}")

    # Math functions
    try:
        code = """
        float x = sin(PI * u);
        @OUT = vec3(x, x, x);
        """
        result = compile_and_run(code, {"A": test_img})
        # sin(0) = 0, sin(pi/2) ~= 1, sin(pi) ~= 0
        assert abs(result[0, 0, 0, 0].item()) < 0.1  # sin(0) ~= 0
        r.ok("math functions (sin, PI)")
    except Exception as e:
        r.fail("math functions (sin, PI)", f"{e}\n{traceback.format_exc()}")

    # fit function
    try:
        code = """
        float x = fit(u, 0.0, 1.0, -1.0, 1.0);
        @OUT = vec3(x, x, x);
        """
        result = compile_and_run(code, {"A": test_img})
        # u=0 -> -1, u=1 -> 1
        assert result[0, 0, 0, 0].item() < -0.9
        assert result[0, 0, W-1, 0].item() > 0.9
        r.ok("fit function")
    except Exception as e:
        r.fail("fit function", f"{e}\n{traceback.format_exc()}")

    # Unary negation
    try:
        code = """
        @OUT = -@A + vec3(1.0, 1.0, 1.0);
        """
        result = compile_and_run(code, {"A": test_img})
        expected = -test_img + 1.0
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("unary negation")
    except Exception as e:
        r.fail("unary negation", f"{e}\n{traceback.format_exc()}")

    # Vector broadcast: vec3(scalar)
    try:
        code = """
        @OUT = vec3(0.5);
        """
        result = compile_and_run(code, {"A": test_img})
        assert torch.allclose(result, torch.full((B, H, W, 3), 0.5), atol=1e-5)
        r.ok("vec3 broadcast constructor")
    except Exception as e:
        r.fail("vec3 broadcast constructor", f"{e}\n{traceback.format_exc()}")

    # Error: missing @OUT
    try:
        compile_and_run("float x = 1.0;", {"A": test_img})
        r.fail("missing @OUT error", "Should have raised InterpreterError")
    except InterpreterError as e:
        assert "OUT" in str(e)
        r.ok("missing @OUT error")
    except RuntimeError as e:
        if "OUT" in str(e):
            r.ok("missing @OUT error")
        else:
            r.fail("missing @OUT error", str(e))
    except Exception as e:
        r.fail("missing @OUT error", str(e))

    # Error: unbound input
    try:
        compile_and_run("@OUT = @Z;", {})
        r.fail("unbound input error", "Should have raised error")
    except (InterpreterError, RuntimeError):
        r.ok("unbound input error")
    except Exception as e:
        r.fail("unbound input error", str(e))

    # Smoothstep
    try:
        code = """
        float s = smoothstep(0.25, 0.75, u);
        @OUT = vec3(s, s, s);
        """
        result = compile_and_run(code, {"A": test_img})
        # smoothstep(0.25, 0.75, 0) = 0, smoothstep(0.25, 0.75, 1) = 1
        assert result[0, 0, 0, 0].item() < 0.1
        assert result[0, 0, W-1, 0].item() > 0.9
        r.ok("smoothstep function")
    except Exception as e:
        r.fail("smoothstep function", f"{e}\n{traceback.format_exc()}")

    # Dot product
    try:
        code = """
        vec3 a = vec3(1.0, 0.0, 0.0);
        vec3 b = vec3(0.0, 1.0, 0.0);
        float d = dot(a, b);
        @OUT = vec3(d, d, d);
        """
        result = compile_and_run(code, {"A": test_img})
        # dot([1,0,0], [0,1,0]) = 0
        assert abs(result[0, 0, 0, 0].item()) < 1e-5
        r.ok("dot product")
    except Exception as e:
        r.fail("dot product", f"{e}\n{traceback.format_exc()}")

    # Length function
    try:
        code = """
        vec3 dir = vec3(3.0, 4.0, 0.0);
        float len = length(dir);
        @OUT = vec3(len, len, len);
        """
        result = compile_and_run(code, {"A": test_img})
        # length([3,4,0]) = 5
        assert abs(result[0, 0, 0, 0].item() - 5.0) < 1e-4
        r.ok("length function")
    except Exception as e:
        r.fail("length function", f"{e}\n{traceback.format_exc()}")

    # Division safety (divide by zero)
    try:
        code = """
        float x = 1.0 / 0.0;
        @OUT = vec3(x, x, x);
        """
        result = compile_and_run(code, {"A": test_img})
        # Should not crash (protected division)
        assert not torch.isnan(result).any(), "Got NaN from division by zero"
        r.ok("division by zero safety")
    except Exception as e:
        r.fail("division by zero safety", f"{e}\n{traceback.format_exc()}")

    # Variable 'v' conflict: type checker prevents redeclaring built-in 'v'
    try:
        code = "float v = 0.5; @OUT = vec3(v, v, v);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"A": TEXType.VEC3, "OUT": TEXType.VEC3})
        tc.check(program)
        r.fail("variable v redecl error", "Expected TypeCheckError")
    except TypeCheckError as e:
        assert "already declared" in str(e)
        r.ok("variable v redecl error")
    except Exception as e:
        r.fail("variable v redecl error", f"{e}\n{traceback.format_exc()}")


# ── For Loop Tests ────────────────────────────────────────────────────

def test_for_loops(r: TestResult):
    print("\n--- For Loop Tests ---")

    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 3)

    # Simple accumulator loop
    try:
        code = """
        float sum = 0.0;
        for (int i = 0; i < 5; i++) {
            sum += 1.0;
        }
        @OUT = vec3(sum, sum, sum);
        """
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 0].item() - 5.0) < 1e-4, f"Got {result[0,0,0,0].item()}"
        r.ok("for loop: simple accumulator")
    except Exception as e:
        r.fail("for loop: simple accumulator", f"{e}\n{traceback.format_exc()}")

    # For loop with compound assignment (*=)
    try:
        code = """
        float x = 1.0;
        for (int i = 0; i < 4; i++) {
            x *= 2.0;
        }
        @OUT = vec3(x, x, x);
        """
        result = compile_and_run(code, {"A": test_img})
        # 1.0 * 2^4 = 16.0
        assert abs(result[0, 0, 0, 0].item() - 16.0) < 1e-3, f"Got {result[0,0,0,0].item()}"
        r.ok("for loop: compound *=")
    except Exception as e:
        r.fail("for loop: compound *=", f"{e}\n{traceback.format_exc()}")

    # For loop: decrement
    try:
        code = """
        float sum = 0.0;
        for (int i = 10; i > 0; i--) {
            sum += 1.0;
        }
        @OUT = vec3(sum, sum, sum);
        """
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 0].item() - 10.0) < 1e-4, f"Got {result[0,0,0,0].item()}"
        r.ok("for loop: decrement")
    except Exception as e:
        r.fail("for loop: decrement", f"{e}\n{traceback.format_exc()}")

    # For loop: nested math
    try:
        code = """
        float sum = 0.0;
        for (int i = 1; i < 6; i++) {
            sum += float(i) * float(i);
        }
        @OUT = vec3(sum, sum, sum);
        """
        result = compile_and_run(code, {"A": test_img})
        # 1+4+9+16+25 = 55
        assert abs(result[0, 0, 0, 0].item() - 55.0) < 1e-3, f"Got {result[0,0,0,0].item()}"
        r.ok("for loop: sum of squares")
    except Exception as e:
        r.fail("for loop: sum of squares", f"{e}\n{traceback.format_exc()}")

    # For loop: iteration limit check
    try:
        code = """
        float x = 0.0;
        for (int i = 0; i < 9999; i++) {
            x += 1.0;
        }
        @OUT = vec3(x, x, x);
        """
        compile_and_run(code, {"A": test_img})
        r.fail("for loop: iteration limit", "Should have raised InterpreterError")
    except InterpreterError as e:
        assert "maximum iteration limit" in str(e).lower() or "1024" in str(e)
        r.ok("for loop: iteration limit")
    except Exception as e:
        r.fail("for loop: iteration limit", f"{e}\n{traceback.format_exc()}")

    # For loop: using loop variable in vector ops
    try:
        code = """
        vec3 color = vec3(0.0);
        for (int i = 0; i < 3; i++) {
            color += @A * 0.333;
        }
        @OUT = color;
        """
        result = compile_and_run(code, {"A": test_img})
        expected = test_img * 0.999
        assert torch.allclose(result, expected, atol=1e-3), f"Max diff: {(result - expected).abs().max().item()}"
        r.ok("for loop: vector accumulation")
    except Exception as e:
        r.fail("for loop: vector accumulation", f"{e}\n{traceback.format_exc()}")


# ── Compound Assignment Tests ─────────────────────────────────────────

def test_compound_assignments(r: TestResult):
    print("\n--- Compound Assignment Tests ---")

    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 3)

    # +=
    try:
        code = """
        float x = 5.0;
        x += 3.0;
        @OUT = vec3(x, x, x);
        """
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 0].item() - 8.0) < 1e-5
        r.ok("compound: +=")
    except Exception as e:
        r.fail("compound: +=", f"{e}\n{traceback.format_exc()}")

    # -=
    try:
        code = """
        float x = 10.0;
        x -= 3.0;
        @OUT = vec3(x, x, x);
        """
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 0].item() - 7.0) < 1e-5
        r.ok("compound: -=")
    except Exception as e:
        r.fail("compound: -=", f"{e}\n{traceback.format_exc()}")

    # *=
    try:
        code = """
        float x = 4.0;
        x *= 3.0;
        @OUT = vec3(x, x, x);
        """
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 0].item() - 12.0) < 1e-4
        r.ok("compound: *=")
    except Exception as e:
        r.fail("compound: *=", f"{e}\n{traceback.format_exc()}")

    # /=
    try:
        code = """
        float x = 12.0;
        x /= 4.0;
        @OUT = vec3(x, x, x);
        """
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4
        r.ok("compound: /=")
    except Exception as e:
        r.fail("compound: /=", f"{e}\n{traceback.format_exc()}")

    # ++ (postfix)
    try:
        code = """
        float x = 5.0;
        x++;
        @OUT = vec3(x, x, x);
        """
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 0].item() - 6.0) < 1e-5
        r.ok("compound: x++")
    except Exception as e:
        r.fail("compound: x++", f"{e}\n{traceback.format_exc()}")

    # -- (postfix)
    try:
        code = """
        float x = 5.0;
        x--;
        @OUT = vec3(x, x, x);
        """
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 0].item() - 4.0) < 1e-5
        r.ok("compound: x--")
    except Exception as e:
        r.fail("compound: x--", f"{e}\n{traceback.format_exc()}")


# ── Parser v1.1 Tests ─────────────────────────────────────────────────

def test_parser_v11(r: TestResult):
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


# ── Lexer v1.1 Tests ──────────────────────────────────────────────────

def test_lexer_v11(r: TestResult):
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


# ── Full Example Tests (from examples/) ───────────────────────────────

def test_examples(r: TestResult):
    print("\n--- Example Snippet Tests ---")

    B, H, W = 1, 8, 8
    test_img = torch.rand(B, H, W, 3)

    # Grayscale example
    try:
        code = """
        float gray = luma(@A);
        @OUT = vec3(gray, gray, gray);
        """
        result = compile_and_run(code, {"A": test_img})
        assert result.shape == (B, H, W, 3)
        r.ok("example: grayscale")
    except Exception as e:
        r.fail("example: grayscale", f"{e}\n{traceback.format_exc()}")

    # Threshold mask example
    try:
        code = """
        float gray = luma(@A);
        @OUT = step(@B, gray);
        """
        result = compile_and_run(code, {"A": test_img, "B": 0.5})
        # Result should be binary (0 or 1)
        unique = torch.unique(result)
        assert all(v.item() in (0.0, 1.0) for v in unique), f"Non-binary values: {unique}"
        r.ok("example: threshold mask")
    except Exception as e:
        r.fail("example: threshold mask", f"{e}\n{traceback.format_exc()}")

    # Vignette example
    try:
        code = """
        float cx = u - 0.5;
        float cy = v - 0.5;
        float dist = sqrt(cx * cx + cy * cy);
        float vignette = 1.0 - smoothstep(0.3, 0.7, dist * @B);
        @OUT = @A * vec3(vignette, vignette, vignette);
        """
        result = compile_and_run(code, {"A": test_img, "B": 1.0})
        assert result.shape == (B, H, W, 3)
        # Center should be brighter than corners
        center = result[0, H//2, W//2, 0].item()
        corner = result[0, 0, 0, 0].item()
        # Not checking magnitude since input is random, but shape should be right
        r.ok("example: vignette")
    except Exception as e:
        r.fail("example: vignette", f"{e}\n{traceback.format_exc()}")

    # Color mix example
    try:
        img_a = torch.zeros(B, H, W, 3)
        img_b = torch.ones(B, H, W, 3)
        code = "@OUT = lerp(@A, @B, @C);"
        result = compile_and_run(code, {"A": img_a, "B": img_b, "C": 0.25})
        assert torch.allclose(result, torch.full((B, H, W, 3), 0.25), atol=1e-4)
        r.ok("example: color mix")
    except Exception as e:
        r.fail("example: color mix", f"{e}\n{traceback.format_exc()}")

    # Invert example
    try:
        code = """
        @OUT = vec3(1.0 - @A.r, 1.0 - @A.g, 1.0 - @A.b);
        """
        result = compile_and_run(code, {"A": test_img})
        expected = 1.0 - test_img
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("example: invert")
    except Exception as e:
        r.fail("example: invert", f"{e}\n{traceback.format_exc()}")

    # Conditional example
    try:
        code = """
        float brightness = luma(@A);
        if (brightness < 0.5) {
            float warmth = (0.5 - brightness) * 0.2;
            @OUT = vec3(
                clamp(@A.r + warmth, 0.0, 1.0),
                @A.g,
                clamp(@A.b - warmth, 0.0, 1.0)
            );
        } else {
            float coolness = (brightness - 0.5) * 0.2;
            @OUT = vec3(
                clamp(@A.r - coolness, 0.0, 1.0),
                @A.g,
                clamp(@A.b + coolness, 0.0, 1.0)
            );
        }
        """
        result = compile_and_run(code, {"A": test_img})
        assert result.shape == (B, H, W, 3)
        assert not torch.isnan(result).any()
        r.ok("example: conditional warm/cool")
    except Exception as e:
        r.fail("example: conditional warm/cool", f"{e}\n{traceback.format_exc()}")


# ── Cache Tests ────────────────────────────────────────────────────────

def test_cache(r: TestResult):
    print("\n--- Cache Tests ---")

    # Memory cache hit returns same AST object
    try:
        tmp = tempfile.mkdtemp()
        cache = TEXCache(cache_dir=Path(tmp))
        code = "float g = luma(@A); @OUT = vec3(g, g, g);"
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

        r1 = cache.compile_tex(code, bt)
        r2 = cache.compile_tex(code, bt)
        assert r1[0] is r2[0], "Memory hit should return the exact same program object"
        r.ok("cache: memory hit same object")
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        r.fail("cache: memory hit same object", f"{e}\n{traceback.format_exc()}")

    # Disk cache hit after memory is cleared
    try:
        tmp = tempfile.mkdtemp()
        cache = TEXCache(cache_dir=Path(tmp))
        code = "@OUT = @A * 0.5;"
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

        cache.compile_tex(code, bt)
        # Verify .pkl was written
        pkl_files = list(Path(tmp).glob("*.pkl"))
        assert len(pkl_files) == 1, f"Expected 1 .pkl file, got {len(pkl_files)}"

        cache.clear_memory()
        result = cache.compile_tex(code, bt)
        assert result is not None
        assert len(result) == 5
        r.ok("cache: disk hit after memory clear")
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        r.fail("cache: disk hit after memory clear", f"{e}\n{traceback.format_exc()}")

    # Corrupted disk file handled gracefully
    try:
        tmp = tempfile.mkdtemp()
        cache = TEXCache(cache_dir=Path(tmp))
        code = "@OUT = @A;"
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
        fp = cache.fingerprint(code, bt)

        # Write garbage to the expected disk cache path
        disk_path = Path(tmp) / f"{fp}.pkl"
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        disk_path.write_bytes(b"corrupted data here")

        # compile_tex should handle the corruption and compile fresh
        result = cache.compile_tex(code, bt)
        assert result is not None
        r.ok("cache: corrupted disk handled")
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        r.fail("cache: corrupted disk handled", f"{e}\n{traceback.format_exc()}")

    # Version mismatch triggers recompile
    try:
        import pickle
        tmp = tempfile.mkdtemp()
        cache = TEXCache(cache_dir=Path(tmp))
        code = "@OUT = @A + 1.0;"
        bt = {"A": TEXType.FLOAT, "OUT": TEXType.VEC4}

        cache.compile_tex(code, bt)
        cache.clear_memory()

        # Tamper with the version in the pickled file
        fp = cache.fingerprint(code, bt)
        disk_path = Path(tmp) / f"{fp}.pkl"
        with open(disk_path, "rb") as f:
            data = pickle.load(f)
        data["version"] = "0.0.0-old"
        with open(disk_path, "wb") as f:
            pickle.dump(data, f)

        # Should miss (version mismatch) and recompile fresh
        result = cache.compile_tex(code, bt)
        assert result is not None
        r.ok("cache: version mismatch recompile")
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        r.fail("cache: version mismatch recompile", f"{e}\n{traceback.format_exc()}")

    # Fingerprint stability and differentiation
    try:
        code = "@OUT = @A;"
        bt_a = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        fp1 = TEXCache.fingerprint(code, bt_a)
        fp2 = TEXCache.fingerprint(code, bt_a)
        assert fp1 == fp2, "Same inputs should produce same fingerprint"

        bt_b = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
        fp3 = TEXCache.fingerprint(code, bt_b)
        assert fp1 != fp3, "Different binding types should produce different fingerprint"
        r.ok("cache: fingerprint stability")
    except Exception as e:
        r.fail("cache: fingerprint stability", f"{e}\n{traceback.format_exc()}")


# ── Device Selection Tests ─────────────────────────────────────────────

def test_device_selection(r: TestResult):
    print("\n--- Device Selection Tests ---")

    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 3)

    # Explicit CPU
    try:
        result = compile_and_run("@OUT = @A;", {"A": test_img}, device="cpu")
        assert result.device.type == "cpu"
        assert torch.allclose(result, test_img, atol=1e-6)
        r.ok("device: explicit cpu")
    except Exception as e:
        r.fail("device: explicit cpu", f"{e}\n{traceback.format_exc()}")

    # Auto mode with CPU tensor -> should stay on CPU
    try:
        result = compile_and_run("@OUT = @A * 0.5;", {"A": test_img}, device="cpu")
        expected = test_img * 0.5
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("device: auto cpu tensor")
    except Exception as e:
        r.fail("device: auto cpu tensor", f"{e}\n{traceback.format_exc()}")

    # CUDA if available
    if torch.cuda.is_available():
        try:
            img_cuda = test_img.cuda()
            result = compile_and_run("@OUT = @A;", {"A": img_cuda}, device="cuda")
            # Interpreter returns tensor on the execution device;
            # check the values are correct once moved to CPU
            result_cpu = result.cpu() if result.is_cuda else result
            assert torch.allclose(result_cpu, test_img, atol=1e-5)
            r.ok("device: explicit cuda")
        except Exception as e:
            r.fail("device: explicit cuda", f"{e}\n{traceback.format_exc()}")
    else:
        r.ok("device: cuda skipped (no GPU)")


# ── torch.compile Tests ───────────────────────────────────────────────

def test_torch_compile(r: TestResult):
    print("\n--- torch.compile Tests ---")

    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 4)

    code = "@OUT = @A * 0.5;"
    bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}

    cache = TEXCache()
    program, type_map, refs, *_ = cache.compile_tex(code, bt)
    fp = cache.fingerprint(code, bt)

    # Ensure fresh state
    clear_compiled_cache()

    # Plain execution baseline
    try:
        result_plain = _plain_execute(program, {"A": test_img}, type_map, "cpu")
        expected = test_img * 0.5
        assert torch.allclose(result_plain, expected, atol=1e-5)
        r.ok("torch_compile: plain baseline")
    except Exception as e:
        r.fail("torch_compile: plain baseline", f"{e}\n{traceback.format_exc()}")

    # Compiled execution produces numerically equivalent result
    try:
        result_compiled = execute_compiled(
            program, {"A": test_img}, type_map, "cpu", fp
        )
        expected = test_img * 0.5
        assert torch.allclose(result_compiled, expected, atol=1e-5), (
            f"Max diff: {(result_compiled - expected).abs().max().item()}"
        )
        r.ok("torch_compile: compiled matches plain")
    except Exception as e:
        r.fail("torch_compile: compiled matches plain", f"{e}\n{traceback.format_exc()}")

    # Graceful execution — should never crash regardless of backend availability
    try:
        result = execute_compiled(
            program, {"A": test_img}, type_map, "cpu", fp
        )
        assert result is not None
        assert result.shape == test_img.shape
        r.ok("torch_compile: graceful execution")
    except Exception as e:
        r.fail("torch_compile: graceful execution", f"{e}\n{traceback.format_exc()}")

    # For-loop program still works under compiled path (graph breaks handled)
    try:
        loop_code = """
        float sum = 0.0;
        for (int i = 0; i < 5; i++) {
            sum += 1.0;
        }
        @OUT = @A * (sum / 5.0);
        """
        loop_bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        loop_cache = TEXCache()
        loop_prog, loop_tm, loop_refs, *_ = loop_cache.compile_tex(loop_code, loop_bt)
        loop_fp = loop_cache.fingerprint(loop_code, loop_bt)

        result_loop = execute_compiled(
            loop_prog, {"A": test_img}, loop_tm, "cpu", loop_fp
        )
        # sum = 5, so sum/5 = 1.0, result should equal @A
        assert torch.allclose(result_loop, test_img, atol=1e-4), (
            f"Max diff: {(result_loop - test_img).abs().max().item()}"
        )
        r.ok("torch_compile: for-loop graph breaks")
    except Exception as e:
        r.fail("torch_compile: for-loop graph breaks", f"{e}\n{traceback.format_exc()}")


# ── Sampling Function Tests ────────────────────────────────────────────

def test_sampling(r: TestResult):
    """Test sampling function family: fetch, sample_cubic, sample_lanczos."""
    print("\n--- Sampling Functions ---")

    # Create a small 4x4 test image [B=1, H=4, W=4, C=3]
    # Each pixel has a unique color for verification
    test_img = torch.zeros(1, 4, 4, 3)
    test_img[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])  # (0,0) red
    test_img[0, 0, 1] = torch.tensor([0.0, 1.0, 0.0])  # (1,0) green
    test_img[0, 0, 2] = torch.tensor([0.0, 0.0, 1.0])  # (2,0) blue
    test_img[0, 0, 3] = torch.tensor([1.0, 1.0, 0.0])  # (3,0) yellow
    test_img[0, 1, 0] = torch.tensor([1.0, 0.0, 1.0])  # (0,1) magenta
    test_img[0, 1, 1] = torch.tensor([0.0, 1.0, 1.0])  # (1,1) cyan
    test_img[0, 1, 2] = torch.tensor([0.5, 0.5, 0.5])  # (2,1) gray
    test_img[0, 1, 3] = torch.tensor([1.0, 1.0, 1.0])  # (3,1) white

    # ── fetch: exact pixel access ──────────────────────────────────────

    # Test 1: fetch at integer coordinates returns exact pixel
    try:
        code = "@OUT = fetch(@A, 1.0, 0.0);"  # pixel (1, 0) = green
        result = compile_and_run(code, {"A": test_img})
        # Result is [B, H, W, C] — all pixels should have the green color
        pixel = result[0, 0, 0, :3]  # any pixel, first 3 channels
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(pixel, expected, atol=1e-5), \
            f"Expected green {expected}, got {pixel}"
        r.ok("sampling: fetch integer coords")
    except Exception as e:
        r.fail("sampling: fetch integer coords", f"{e}\n{traceback.format_exc()}")

    # Test 2: fetch with ix/iy neighbor pattern (fetch at ix+1)
    try:
        code = "@OUT = fetch(@A, ix + 1, iy);"
        result = compile_and_run(code, {"A": test_img})
        # At pixel (0,0), fetch(1,0) should give green
        pixel_0_0 = result[0, 0, 0, :3]
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(pixel_0_0, expected, atol=1e-5), \
            f"At (0,0) fetching (1,0): expected {expected}, got {pixel_0_0}"
        # At pixel (2,0), fetch(3,0) should give yellow
        pixel_2_0 = result[0, 0, 2, :3]
        expected_yellow = torch.tensor([1.0, 1.0, 0.0])
        assert torch.allclose(pixel_2_0, expected_yellow, atol=1e-5), \
            f"At (2,0) fetching (3,0): expected {expected_yellow}, got {pixel_2_0}"
        r.ok("sampling: fetch with ix+offset")
    except Exception as e:
        r.fail("sampling: fetch with ix+offset", f"{e}\n{traceback.format_exc()}")

    # Test 3: fetch clamps out-of-bounds
    try:
        code = "@OUT = fetch(@A, -1.0, 0.0);"  # should clamp to (0, 0) = red
        result = compile_and_run(code, {"A": test_img})
        pixel = result[0, 0, 0, :3]
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(pixel, expected, atol=1e-5), \
            f"Out-of-bounds clamp: expected {expected}, got {pixel}"
        r.ok("sampling: fetch clamps OOB")
    except Exception as e:
        r.fail("sampling: fetch clamps OOB", f"{e}\n{traceback.format_exc()}")

    # ── sample (bilinear, already existed) — regression test ───────────

    # Test 4: bilinear sample at center of pixel returns that pixel
    try:
        code = "@OUT = sample(@A, 0.0, 0.0);"  # top-left corner = red
        result = compile_and_run(code, {"A": test_img})
        pixel = result[0, 0, 0, :3]
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(pixel, expected, atol=1e-5), \
            f"Sample (0,0): expected {expected}, got {pixel}"
        r.ok("sampling: bilinear at corner")
    except Exception as e:
        r.fail("sampling: bilinear at corner", f"{e}\n{traceback.format_exc()}")

    # ── sample_cubic: bicubic interpolation ────────────────────────────

    # Test 5: sample_cubic at exact pixel coordinates
    try:
        code = "@OUT = sample_cubic(@A, 0.0, 0.0);"  # top-left = red
        result = compile_and_run(code, {"A": test_img})
        pixel = result[0, 0, 0, :3]
        # Bicubic may have slight overshoot at boundaries but should be close
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(pixel, expected, atol=0.15), \
            f"Cubic at (0,0): expected ~{expected}, got {pixel}"
        r.ok("sampling: cubic at corner")
    except Exception as e:
        r.fail("sampling: cubic at corner", f"{e}\n{traceback.format_exc()}")

    # Test 6: sample_cubic returns [B, H, W, C] with correct shape
    try:
        code = "@OUT = sample_cubic(@A, u, v);"  # identity resample
        result = compile_and_run(code, {"A": test_img})
        assert result.shape == test_img.shape or result.shape[-1] == test_img.shape[-1], \
            f"Shape mismatch: {result.shape} vs expected {test_img.shape}"
        r.ok("sampling: cubic shape preservation")
    except Exception as e:
        r.fail("sampling: cubic shape preservation", f"{e}\n{traceback.format_exc()}")

    # ── sample_lanczos: Lanczos-3 interpolation ────────────────────────

    # Test 7: sample_lanczos at exact pixel coordinates
    try:
        code = "@OUT = sample_lanczos(@A, 0.0, 0.0);"  # top-left
        result = compile_and_run(code, {"A": test_img})
        pixel = result[0, 0, 0, :3]
        # Lanczos may ring slightly but should be close to the source pixel
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(pixel, expected, atol=0.2), \
            f"Lanczos at (0,0): expected ~{expected}, got {pixel}"
        r.ok("sampling: lanczos at corner")
    except Exception as e:
        r.fail("sampling: lanczos at corner", f"{e}\n{traceback.format_exc()}")

    # Test 8: sample_lanczos identity resample preserves shape
    try:
        code = "@OUT = sample_lanczos(@A, u, v);"
        result = compile_and_run(code, {"A": test_img})
        assert result.dim() == 4, f"Expected 4D result, got {result.dim()}D"
        assert result.shape[-1] == test_img.shape[-1], \
            f"Channel mismatch: {result.shape[-1]} vs {test_img.shape[-1]}"
        r.ok("sampling: lanczos shape preservation")
    except Exception as e:
        r.fail("sampling: lanczos shape preservation", f"{e}\n{traceback.format_exc()}")

    # Test 9: identity resample (all methods should roughly reproduce original)
    try:
        # Use a larger smooth image so interpolation is well-conditioned
        H, W = 16, 16
        smooth_img = torch.zeros(1, H, W, 3)
        for y in range(H):
            for x in range(W):
                smooth_img[0, y, x] = torch.tensor([x / (W-1), y / (H-1), 0.5])

        code_bilinear = "@OUT = sample(@A, u, v);"
        code_cubic = "@OUT = sample_cubic(@A, u, v);"
        code_lanczos = "@OUT = sample_lanczos(@A, u, v);"

        res_bilinear = compile_and_run(code_bilinear, {"A": smooth_img})
        res_cubic = compile_and_run(code_cubic, {"A": smooth_img})
        res_lanczos = compile_and_run(code_lanczos, {"A": smooth_img})

        # All should be close to the original for a smooth gradient
        diff_bilinear = (res_bilinear[..., :3] - smooth_img).abs().max().item()
        diff_cubic = (res_cubic[..., :3] - smooth_img).abs().max().item()
        diff_lanczos = (res_lanczos[..., :3] - smooth_img).abs().max().item()

        assert diff_bilinear < 0.05, f"Bilinear identity drift: {diff_bilinear:.4f}"
        assert diff_cubic < 0.1, f"Cubic identity drift: {diff_cubic:.4f}"
        assert diff_lanczos < 0.1, f"Lanczos identity drift: {diff_lanczos:.4f}"

        r.ok(f"sampling: identity resample (bilinear={diff_bilinear:.4f}, cubic={diff_cubic:.4f}, lanczos={diff_lanczos:.4f})")
    except Exception as e:
        r.fail("sampling: identity resample", f"{e}\n{traceback.format_exc()}")

    # Test 10: fetch returns vec4 (type check)
    try:
        code = "vec4 c = fetch(@A, ix, iy);\n@OUT = c;"
        result = compile_and_run(code, {"A": test_img})
        assert result.dim() == 4 and result.shape[-1] >= 3, \
            f"fetch should return vec4-compatible, got shape {result.shape}"
        r.ok("sampling: fetch returns vec4")
    except Exception as e:
        r.fail("sampling: fetch returns vec4", f"{e}\n{traceback.format_exc()}")


# ── Noise Function Tests ───────────────────────────────────────────────

def test_noise(r: TestResult):
    """Test noise functions: perlin, simplex, fbm."""
    print("\n--- Noise Functions ---")

    # Use a 32x32 image for better spatial coverage in noise tests
    noise_img = torch.zeros(1, 32, 32, 3)

    # ── Perlin ─────────────────────────────────────────────────────

    # Test 1: perlin shape
    try:
        code = "float n = perlin(u * 4.0, v * 4.0);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result.dim() == 4 and result.shape[-1] == 3, \
            f"Expected [B,H,W,3], got {result.shape}"
        assert result.shape[1] == 32 and result.shape[2] == 32, \
            f"Expected 32x32, got {result.shape}"
        r.ok("noise: perlin shape")
    except Exception as e:
        r.fail("noise: perlin shape", f"{e}\n{traceback.format_exc()}")

    # Test 2: perlin value range
    try:
        code = "float n = perlin(u * 16.0, v * 16.0);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        vals = result[..., 0]
        assert vals.min() >= -1.5, f"Perlin min {vals.min().item():.3f} out of range"
        assert vals.max() <= 1.5, f"Perlin max {vals.max().item():.3f} out of range"
        r.ok(f"noise: perlin range [{vals.min().item():.3f}, {vals.max().item():.3f}]")
    except Exception as e:
        r.fail("noise: perlin range", f"{e}\n{traceback.format_exc()}")

    # Test 3: perlin determinism
    try:
        code = "float n = perlin(u * 8.0, v * 8.0);\n@OUT = vec3(n, n, n);"
        r1 = compile_and_run(code, {"A": noise_img})
        r2 = compile_and_run(code, {"A": noise_img})
        assert torch.allclose(r1, r2, atol=1e-6), "Perlin not deterministic"
        r.ok("noise: perlin determinism")
    except Exception as e:
        r.fail("noise: perlin determinism", f"{e}\n{traceback.format_exc()}")

    # Test 4: perlin spatial variation (not constant)
    try:
        code = "float n = perlin(u * 8.0, v * 8.0);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        std = result[..., 0].std().item()
        assert std > 0.01, f"Perlin nearly constant (std={std:.6f})"
        r.ok(f"noise: perlin variation (std={std:.4f})")
    except Exception as e:
        r.fail("noise: perlin variation", f"{e}\n{traceback.format_exc()}")

    # Test 5: perlin continuity (nearby coords -> nearby values)
    try:
        code = (
            "float n1 = perlin(u * 4.0, v * 4.0);\n"
            "float n2 = perlin(u * 4.0 + 0.001, v * 4.0);\n"
            "float diff = abs(n1 - n2);\n"
            "@OUT = vec3(diff, diff, diff);"
        )
        result = compile_and_run(code, {"A": noise_img})
        max_diff = result[..., 0].max().item()
        assert max_diff < 0.1, f"Perlin not continuous: max local diff = {max_diff:.6f}"
        r.ok(f"noise: perlin continuity (max_diff={max_diff:.6f})")
    except Exception as e:
        r.fail("noise: perlin continuity", f"{e}\n{traceback.format_exc()}")

    # ── Simplex ────────────────────────────────────────────────────

    # Test 6: simplex shape and range
    try:
        code = "float n = simplex(u * 8.0, v * 8.0);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result.dim() == 4 and result.shape[-1] == 3
        vals = result[..., 0]
        assert vals.min() >= -1.5, f"Simplex min {vals.min().item():.3f} out of range"
        assert vals.max() <= 1.5, f"Simplex max {vals.max().item():.3f} out of range"
        r.ok(f"noise: simplex shape+range [{vals.min().item():.3f}, {vals.max().item():.3f}]")
    except Exception as e:
        r.fail("noise: simplex shape+range", f"{e}\n{traceback.format_exc()}")

    # Test 7: simplex determinism
    try:
        code = "float n = simplex(u * 8.0, v * 8.0);\n@OUT = vec3(n, n, n);"
        r1 = compile_and_run(code, {"A": noise_img})
        r2 = compile_and_run(code, {"A": noise_img})
        assert torch.allclose(r1, r2, atol=1e-6), "Simplex not deterministic"
        r.ok("noise: simplex determinism")
    except Exception as e:
        r.fail("noise: simplex determinism", f"{e}\n{traceback.format_exc()}")

    # Test 8: simplex spatial variation
    try:
        code = "float n = simplex(u * 8.0, v * 8.0);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        std = result[..., 0].std().item()
        assert std > 0.01, f"Simplex nearly constant (std={std:.6f})"
        r.ok(f"noise: simplex variation (std={std:.4f})")
    except Exception as e:
        r.fail("noise: simplex variation", f"{e}\n{traceback.format_exc()}")

    # ── FBM ────────────────────────────────────────────────────────

    # Test 9: fbm shape and range
    try:
        code = "float n = fbm(u * 4.0, v * 4.0, 6);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result.dim() == 4 and result.shape[-1] == 3
        vals = result[..., 0]
        assert vals.min() >= -1.5, f"FBM min {vals.min().item():.3f} out of range"
        assert vals.max() <= 1.5, f"FBM max {vals.max().item():.3f} out of range"
        r.ok(f"noise: fbm shape+range [{vals.min().item():.3f}, {vals.max().item():.3f}]")
    except Exception as e:
        r.fail("noise: fbm shape+range", f"{e}\n{traceback.format_exc()}")

    # Test 10: fbm octave difference (1 octave != 6 octaves)
    try:
        code1 = "float n = fbm(u * 4.0, v * 4.0, 1);\n@OUT = vec3(n, n, n);"
        code6 = "float n = fbm(u * 4.0, v * 4.0, 6);\n@OUT = vec3(n, n, n);"
        r1 = compile_and_run(code1, {"A": noise_img})
        r6 = compile_and_run(code6, {"A": noise_img})
        assert not torch.allclose(r1, r6, atol=1e-3), \
            "FBM with 1 and 6 octaves should produce different output"
        r.ok("noise: fbm octave difference")
    except Exception as e:
        r.fail("noise: fbm octave difference", f"{e}\n{traceback.format_exc()}")


# ── Channel Assignment Tests ──────────────────────────────────────────

def test_channel_assignment(r: TestResult):
    print("\n--- Channel Assignment Tests ---")

    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 4)

    # @OUT.r = 1.0
    try:
        code = "@OUT = vec4(0.0, 0.0, 0.0, 0.0);\n@OUT.r = 1.0;"
        result = compile_and_run(code, {"A": test_img})
        assert torch.allclose(result[..., 0], torch.ones(B, H, W), atol=1e-5)
        assert torch.allclose(result[..., 1], torch.zeros(B, H, W), atol=1e-5)
        assert torch.allclose(result[..., 2], torch.zeros(B, H, W), atol=1e-5)
        r.ok("channel_assign: @OUT.r")
    except Exception as e:
        r.fail("channel_assign: @OUT.r", f"{e}\n{traceback.format_exc()}")

    # @OUT.g = 0.5
    try:
        code = "@OUT = vec4(0.0, 0.0, 0.0, 0.0);\n@OUT.g = 0.5;"
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 1].item() - 0.5) < 1e-5
        assert abs(result[0, 0, 0, 0].item()) < 1e-5
        r.ok("channel_assign: @OUT.g")
    except Exception as e:
        r.fail("channel_assign: @OUT.g", f"{e}\n{traceback.format_exc()}")

    # @OUT.b = 0.75
    try:
        code = "@OUT = vec4(0.0, 0.0, 0.0, 0.0);\n@OUT.b = 0.75;"
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 2].item() - 0.75) < 1e-5
        r.ok("channel_assign: @OUT.b")
    except Exception as e:
        r.fail("channel_assign: @OUT.b", f"{e}\n{traceback.format_exc()}")

    # @OUT.a = 1.0
    try:
        code = "@OUT = vec4(0.0, 0.0, 0.0, 0.0);\n@OUT.a = 1.0;"
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 3].item() - 1.0) < 1e-5
        r.ok("channel_assign: @OUT.a")
    except Exception as e:
        r.fail("channel_assign: @OUT.a", f"{e}\n{traceback.format_exc()}")

    # var.r = 1.0 (Identifier target path)
    try:
        code = "vec4 c = vec4(0.0);\nc.r = 1.0;\n@OUT = c;"
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-5
        assert abs(result[0, 0, 0, 1].item()) < 1e-5
        r.ok("channel_assign: var.r")
    except Exception as e:
        r.fail("channel_assign: var.r", f"{e}\n{traceback.format_exc()}")

    # Preserves other channels
    try:
        code = "@OUT = @A;\n@OUT.r = 0.0;"
        result = compile_and_run(code, {"A": test_img})
        assert torch.allclose(result[..., 0], torch.zeros(B, H, W), atol=1e-5)
        assert torch.allclose(result[..., 1], test_img[..., 1], atol=1e-5)
        assert torch.allclose(result[..., 2], test_img[..., 2], atol=1e-5)
        r.ok("channel_assign: preserves other channels")
    except Exception as e:
        r.fail("channel_assign: preserves other channels", f"{e}\n{traceback.format_exc()}")

    # Multi-channel assign error (caught at type-check or interpreter level)
    try:
        code = "@OUT = vec4(0.0);\n@OUT.rg = 1.0;"
        compile_and_run(code, {"A": test_img})
        r.fail("channel_assign: multi error", "Should have raised error")
    except (InterpreterError, RuntimeError, TypeCheckError) as e:
        assert "single" in str(e).lower() or "swizzle" in str(e).lower()
        r.ok("channel_assign: multi error")
    except Exception as e:
        r.fail("channel_assign: multi error", f"{e}\n{traceback.format_exc()}")

    # Invalid channel error
    try:
        code = "@OUT = vec4(0.0);\n@OUT.q = 1.0;"
        compile_and_run(code, {"A": test_img})
        r.fail("channel_assign: invalid error", "Should have raised error")
    except (InterpreterError, RuntimeError, TypeCheckError) as e:
        r.ok("channel_assign: invalid error")
    except Exception as e:
        r.fail("channel_assign: invalid error", f"{e}\n{traceback.format_exc()}")


# ── Output Type Tests ─────────────────────────────────────────────────

def test_output_types(r: TestResult):
    print("\n--- Output Type Tests ---")

    # IMAGE from vec4 (drops alpha)
    try:
        raw = torch.rand(1, 4, 4, 4)
        result = _prepare_output(raw, "IMAGE")
        assert result.shape == (1, 4, 4, 3), f"Shape: {result.shape}"
        assert torch.allclose(result, raw[..., :3].clamp(0, 1), atol=1e-6)
        r.ok("output: IMAGE from vec4")
    except Exception as e:
        r.fail("output: IMAGE from vec4", f"{e}\n{traceback.format_exc()}")

    # IMAGE from vec3 (passthrough, clamped)
    try:
        raw = torch.rand(1, 4, 4, 3) * 2.0  # values up to 2.0
        result = _prepare_output(raw, "IMAGE")
        assert result.shape == (1, 4, 4, 3)
        assert result.max() <= 1.0
        r.ok("output: IMAGE from vec3")
    except Exception as e:
        r.fail("output: IMAGE from vec3", f"{e}\n{traceback.format_exc()}")

    # IMAGE from mask (3D -> 3-channel)
    try:
        raw = torch.full((1, 4, 4), 0.6)
        result = _prepare_output(raw, "IMAGE")
        assert result.shape == (1, 4, 4, 3), f"Shape: {result.shape}"
        assert torch.allclose(result[..., 0], result[..., 1], atol=1e-6)
        assert abs(result[0, 0, 0, 0].item() - 0.6) < 1e-5
        r.ok("output: IMAGE from mask")
    except Exception as e:
        r.fail("output: IMAGE from mask", f"{e}\n{traceback.format_exc()}")

    # IMAGE from scalar
    try:
        raw = torch.tensor(0.5)
        result = _prepare_output(raw, "IMAGE")
        assert result.shape == (1, 1, 1, 3), f"Shape: {result.shape}"
        assert abs(result[0, 0, 0, 0].item() - 0.5) < 1e-5
        r.ok("output: IMAGE from scalar")
    except Exception as e:
        r.fail("output: IMAGE from scalar", f"{e}\n{traceback.format_exc()}")

    # MASK from vec4 (luma-weighted)
    try:
        raw = torch.zeros(1, 4, 4, 4)
        raw[..., 0] = 1.0  # red only
        result = _prepare_output(raw, "MASK")
        assert result.shape == (1, 4, 4), f"Shape: {result.shape}"
        # Luma: 0.2126 * 1.0 + 0.7152 * 0.0 + 0.0722 * 0.0 = 0.2126
        assert abs(result[0, 0, 0].item() - 0.2126) < 1e-3
        r.ok("output: MASK from vec4")
    except Exception as e:
        r.fail("output: MASK from vec4", f"{e}\n{traceback.format_exc()}")

    # MASK from scalar
    try:
        raw = torch.tensor(0.7)
        result = _prepare_output(raw, "MASK")
        assert result.shape == (1, 1, 1), f"Shape: {result.shape}"
        assert abs(result[0, 0, 0].item() - 0.7) < 1e-5
        r.ok("output: MASK from scalar")
    except Exception as e:
        r.fail("output: MASK from scalar", f"{e}\n{traceback.format_exc()}")

    # FLOAT from spatial
    try:
        raw = torch.full((1, 4, 4, 3), 0.3)
        result = _prepare_output(raw, "FLOAT")
        assert isinstance(result, float), f"Type: {type(result)}"
        assert abs(result - 0.3) < 1e-5
        r.ok("output: FLOAT from spatial")
    except Exception as e:
        r.fail("output: FLOAT from spatial", f"{e}\n{traceback.format_exc()}")

    # INT from spatial
    try:
        raw = torch.full((1, 4, 4, 3), 3.7)
        result = _prepare_output(raw, "INT")
        assert isinstance(result, int), f"Type: {type(result)}"
        assert result == 3
        r.ok("output: INT from spatial")
    except Exception as e:
        r.fail("output: INT from spatial", f"{e}\n{traceback.format_exc()}")


# ── Error Path Tests ──────────────────────────────────────────────────

def test_error_paths(r: TestResult):
    print("\n--- Error Path Tests ---")

    def check_code(code: str, bindings: dict[str, TEXType] | None = None):
        tokens = Lexer(code).tokenize()
        prog = Parser(tokens).parse()
        bt = bindings or {}
        bt.setdefault("OUT", TEXType.VEC4)
        checker = TypeChecker(binding_types=bt)
        return checker.check(prog), checker

    # Unterminated block comment
    try:
        Lexer("x /* no end").tokenize()
        r.fail("error: unterminated block comment", "Should have raised LexerError")
    except LexerError as e:
        assert "unterminated" in str(e).lower() or "Unterminated" in str(e)
        r.ok("error: unterminated block comment")
    except Exception as e:
        r.fail("error: unterminated block comment", f"{e}\n{traceback.format_exc()}")

    # Lone @ symbol
    try:
        Lexer("@").tokenize()
        r.fail("error: lone @ symbol", "Should have raised LexerError")
    except LexerError:
        r.ok("error: lone @ symbol")
    except Exception as e:
        r.fail("error: lone @ symbol", f"{e}\n{traceback.format_exc()}")

    # Variable redeclaration
    try:
        check_code("float x = 1.0;\nfloat x = 2.0;")
        r.fail("error: variable redeclaration", "Should have raised TypeCheckError")
    except TypeCheckError as e:
        assert "already" in str(e).lower() or "redeclar" in str(e).lower()
        r.ok("error: variable redeclaration")
    except Exception as e:
        r.fail("error: variable redeclaration", f"{e}\n{traceback.format_exc()}")

    # Unknown function
    try:
        check_code("float x = bogus(1.0);")
        r.fail("error: unknown function", "Should have raised TypeCheckError")
    except TypeCheckError as e:
        assert "unknown" in str(e).lower() or "Unknown" in str(e)
        r.ok("error: unknown function")
    except Exception as e:
        r.fail("error: unknown function", f"{e}\n{traceback.format_exc()}")

    # Wrong arg count
    try:
        check_code("float x = sin(1.0, 2.0);")
        r.fail("error: wrong arg count", "Should have raised TypeCheckError")
    except TypeCheckError as e:
        assert "expect" in str(e).lower() or "argument" in str(e).lower()
        r.ok("error: wrong arg count")
    except Exception as e:
        r.fail("error: wrong arg count", f"{e}\n{traceback.format_exc()}")

    # 2-component swizzle (no vec2 type)
    try:
        check_code("vec3 v = @A.rg;", {"A": TEXType.VEC4})
        r.fail("error: 2-component swizzle", "Should have raised TypeCheckError")
    except TypeCheckError:
        r.ok("error: 2-component swizzle")
    except Exception as e:
        r.fail("error: 2-component swizzle", f"{e}\n{traceback.format_exc()}")

    # Invalid swizzle .rrr (not in VALID_SWIZZLES)
    try:
        check_code("vec3 v = @A.rrr;", {"A": TEXType.VEC4})
        r.fail("error: invalid swizzle .rrr", "Should have raised TypeCheckError")
    except TypeCheckError as e:
        assert "swizzle" in str(e).lower() or "Invalid" in str(e)
        r.ok("error: invalid swizzle .rrr")
    except Exception as e:
        r.fail("error: invalid swizzle .rrr", f"{e}\n{traceback.format_exc()}")


# ── If Without Else Tests ─────────────────────────────────────────────

def test_if_without_else(r: TestResult):
    print("\n--- If Without Else Tests ---")

    B, H, W = 1, 8, 8
    test_img = torch.rand(B, H, W, 3)

    # Basic if-no-else
    try:
        code = "@OUT = vec3(0.0, 0.0, 0.0);\nif (u > 0.5) {\n    @OUT = vec3(1.0, 1.0, 1.0);\n}"
        result = compile_and_run(code, {"A": test_img})
        # Left half (u <= 0.5) should be 0, right half (u > 0.5) should be 1
        assert result[0, 0, 0, 0].item() < 0.1, "Left should be 0"
        assert result[0, 0, W - 1, 0].item() > 0.9, "Right should be 1"
        r.ok("if-no-else: basic")
    except Exception as e:
        r.fail("if-no-else: basic", f"{e}\n{traceback.format_exc()}")

    # Preserves variable
    try:
        code = "float x = 0.5;\nif (u > 0.5) {\n    x = 1.0;\n}\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 0].item() - 0.5) < 0.1, "Left should be 0.5"
        assert abs(result[0, 0, W - 1, 0].item() - 1.0) < 0.1, "Right should be 1.0"
        r.ok("if-no-else: preserves var")
    except Exception as e:
        r.fail("if-no-else: preserves var", f"{e}\n{traceback.format_exc()}")

    # Preserves binding
    try:
        code = "@OUT = @A;\nif (u > 0.5) {\n    @OUT = vec3(1.0, 0.0, 0.0);\n}"
        result = compile_and_run(code, {"A": test_img})
        # Left half should match @A
        assert torch.allclose(result[0, :, 0, :], test_img[0, :, 0, :], atol=0.1)
        # Right half should be red
        assert result[0, 0, W - 1, 0].item() > 0.9
        assert result[0, 0, W - 1, 1].item() < 0.1
        r.ok("if-no-else: preserves binding")
    except Exception as e:
        r.fail("if-no-else: preserves binding", f"{e}\n{traceback.format_exc()}")


# ── Stdlib Coverage Tests ─────────────────────────────────────────────

def test_stdlib_coverage(r: TestResult):
    print("\n--- Stdlib Coverage Tests ---")

    B, H, W = 1, 2, 2
    test_img = torch.rand(B, H, W, 3)

    # Helper: run code, extract pixel [0,0,0,0]
    def check_val(name, code, expected, atol=1e-3):
        try:
            result = compile_and_run(code, {"A": test_img})
            val = result[0, 0, 0, 0].item()
            assert abs(val - expected) < atol, f"Got {val}, expected {expected}"
            r.ok(f"stdlib: {name}")
        except Exception as e:
            r.fail(f"stdlib: {name}", f"{e}\n{traceback.format_exc()}")

    import math

    check_val("cos", "float x = cos(0.0);\n@OUT = vec3(x, x, x);", 1.0)
    check_val("tan", "float x = tan(0.0);\n@OUT = vec3(x, x, x);", 0.0)
    check_val("asin", "float x = asin(0.5);\n@OUT = vec3(x, x, x);", math.pi / 6)
    check_val("acos", "float x = acos(0.5);\n@OUT = vec3(x, x, x);", math.pi / 3)
    check_val("atan", "float x = atan(1.0);\n@OUT = vec3(x, x, x);", math.pi / 4)
    check_val("atan2", "float x = atan2(1.0, 1.0);\n@OUT = vec3(x, x, x);", math.pi / 4)
    check_val("sqrt", "float x = sqrt(4.0);\n@OUT = vec3(x, x, x);", 2.0)
    check_val("pow", "float x = pow(2.0, 3.0);\n@OUT = vec3(x, x, x);", 8.0)
    check_val("exp", "float x = exp(0.0);\n@OUT = vec3(x, x, x);", 1.0)
    check_val("log", "float x = log(E);\n@OUT = vec3(x, x, x);", 1.0)
    check_val("abs", "float x = abs(-3.0);\n@OUT = vec3(x, x, x);", 3.0)
    check_val("sign", "float x = sign(-5.0);\n@OUT = vec3(x, x, x);", -1.0)
    check_val("floor", "float x = floor(2.7);\n@OUT = vec3(x, x, x);", 2.0)
    check_val("ceil", "float x = ceil(2.3);\n@OUT = vec3(x, x, x);", 3.0)
    check_val("round", "float x = round(2.6);\n@OUT = vec3(x, x, x);", 3.0)
    check_val("fract", "float x = fract(2.7);\n@OUT = vec3(x, x, x);", 0.7)
    check_val("mod", "float x = mod(5.0, 3.0);\n@OUT = vec3(x, x, x);", 2.0)
    check_val("distance", "float x = distance(vec3(0.0, 0.0, 0.0), vec3(3.0, 4.0, 0.0));\n@OUT = vec3(x, x, x);", 5.0)
    check_val("normalize", "float x = length(normalize(vec3(3.0, 4.0, 0.0)));\n@OUT = vec3(x, x, x);", 1.0)
    check_val("cross", "vec3 c = cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));\n@OUT = vec3(c.z, c.z, c.z);", 1.0)


# ── Extended Stdlib Tests (CTL/ACES) ──────────────────────────────────

def test_stdlib_extended(r: TestResult):
    print("\n--- Extended Stdlib Tests (CTL/ACES) ---")

    B, H, W = 1, 2, 2
    test_img = torch.rand(B, H, W, 3)

    def check_val(name, code, expected, atol=1e-3):
        try:
            result = compile_and_run(code, {"A": test_img})
            val = result[0, 0, 0, 0].item()
            assert abs(val - expected) < atol, f"Got {val}, expected {expected}"
            r.ok(f"stdlib: {name}")
        except Exception as e:
            r.fail(f"stdlib: {name}", f"{e}\n{traceback.format_exc()}")

    import math

    check_val("log2", "float x = log2(8.0);\n@OUT = vec3(x, x, x);", 3.0)
    check_val("log10", "float x = log10(1000.0);\n@OUT = vec3(x, x, x);", 3.0)
    check_val("pow2", "float x = pow2(3.0);\n@OUT = vec3(x, x, x);", 8.0)
    check_val("pow10", "float x = pow10(2.0);\n@OUT = vec3(x, x, x);", 100.0)
    check_val("sinh", "float x = sinh(0.0);\n@OUT = vec3(x, x, x);", 0.0)
    check_val("cosh", "float x = cosh(0.0);\n@OUT = vec3(x, x, x);", 1.0)
    check_val("tanh", "float x = tanh(0.0);\n@OUT = vec3(x, x, x);", 0.0)
    check_val("hypot", "float x = hypot(3.0, 4.0);\n@OUT = vec3(x, x, x);", 5.0)
    check_val("degrees", "float x = degrees(PI);\n@OUT = vec3(x, x, x);", 180.0)
    check_val("radians", "float x = radians(180.0);\n@OUT = vec3(x, x, x);", math.pi)

    # isnan / isinf return 0.0 or 1.0
    check_val("isnan", "float x = isnan(sqrt(-1.0));\n@OUT = vec3(x, x, x);", 1.0)
    check_val("isinf", "float x = isinf(log(0.0));\n@OUT = vec3(x, x, x);", 1.0)

    # spow: safe power — sign(x) * pow(abs(x), y)
    check_val("spow", "float x = spow(-2.0, 3.0);\n@OUT = vec3(x, x, x);", -8.0)

    # sdiv: safe division — 0 when b ~= 0
    check_val("sdiv", "float x = sdiv(5.0, 0.0);\n@OUT = vec3(x, x, x);", 0.0)


# ── Numerical Edge Case Tests ─────────────────────────────────────────

def test_numerical_edge_cases(r: TestResult):
    print("\n--- Numerical Edge Case Tests ---")

    B, H, W = 1, 2, 2
    test_img = torch.rand(B, H, W, 3)

    # sqrt(-1) is NaN
    try:
        code = "float x = sqrt(-1.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        assert torch.isnan(result[0, 0, 0, 0]), "sqrt(-1) should be NaN"
        r.ok("edge: sqrt(-1) is NaN")
    except Exception as e:
        r.fail("edge: sqrt(-1) is NaN", f"{e}\n{traceback.format_exc()}")

    # log(0) is -Inf
    try:
        code = "float x = log(0.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        assert torch.isinf(result[0, 0, 0, 0]), "log(0) should be -Inf"
        r.ok("edge: log(0) is -Inf")
    except Exception as e:
        r.fail("edge: log(0) is -Inf", f"{e}\n{traceback.format_exc()}")

    # asin(2) is NaN
    try:
        code = "float x = asin(2.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        assert torch.isnan(result[0, 0, 0, 0]), "asin(2) should be NaN"
        r.ok("edge: asin(2) is NaN")
    except Exception as e:
        r.fail("edge: asin(2) is NaN", f"{e}\n{traceback.format_exc()}")

    # pow(0, -1) is Inf
    try:
        code = "float x = pow(0.0, -1.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        assert torch.isinf(result[0, 0, 0, 0]), "pow(0,-1) should be Inf"
        r.ok("edge: pow(0,-1) is Inf")
    except Exception as e:
        r.fail("edge: pow(0,-1) is Inf", f"{e}\n{traceback.format_exc()}")

    # Modulo operator
    try:
        code = "float x = 5.0 % 3.0;\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[0, 0, 0, 0].item() - 2.0) < 1e-4, f"Got {result[0,0,0,0].item()}"
        r.ok("edge: modulo operator")
    except Exception as e:
        r.fail("edge: modulo operator", f"{e}\n{traceback.format_exc()}")

    # Modulo by zero (epsilon-protected, should not crash)
    try:
        code = "float x = 5.0 % 0.0;\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        # Should not crash — value may be large but not NaN
        assert not torch.isnan(result[0, 0, 0, 0]), "Modulo by zero should be protected"
        r.ok("edge: modulo by zero")
    except Exception as e:
        r.fail("edge: modulo by zero", f"{e}\n{traceback.format_exc()}")

    # Large exp (not NaN/Inf for moderate values)
    try:
        code = "float x = exp(10.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        assert not torch.isnan(result[0, 0, 0, 0]), "exp(10) should not be NaN"
        assert not torch.isinf(result[0, 0, 0, 0]), "exp(10) should not be Inf"
        assert abs(val - 22026.4658) < 1.0, f"exp(10) ~= 22026, got {val}"
        r.ok("edge: large exp")
    except Exception as e:
        r.fail("edge: large exp", f"{e}\n{traceback.format_exc()}")


# ── Swizzle Pattern Tests ─────────────────────────────────────────────

def test_swizzle_patterns(r: TestResult):
    print("\n--- Swizzle Pattern Tests ---")

    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 4)

    # .a (alpha channel)
    try:
        code = "float a = @A.a;\n@OUT = vec4(a, a, a, a);"
        result = compile_and_run(code, {"A": test_img})
        assert torch.allclose(result[..., 0], test_img[..., 3], atol=1e-5)
        r.ok("swizzle: .a (alpha)")
    except Exception as e:
        r.fail("swizzle: .a (alpha)", f"{e}\n{traceback.format_exc()}")

    # .rgba (identity swizzle)
    try:
        code = "vec4 c = @A.rgba;\n@OUT = c;"
        result = compile_and_run(code, {"A": test_img})
        assert torch.allclose(result, test_img, atol=1e-5)
        r.ok("swizzle: .rgba (identity)")
    except Exception as e:
        r.fail("swizzle: .rgba (identity)", f"{e}\n{traceback.format_exc()}")

    # .bgr (reorder)
    try:
        code = "vec3 c = @A.bgr;\n@OUT = vec4(c.r, c.g, c.b, 1.0);"
        result = compile_and_run(code, {"A": test_img})
        assert torch.allclose(result[..., 0], test_img[..., 2], atol=1e-5), "bgr[0] should be blue"
        assert torch.allclose(result[..., 2], test_img[..., 0], atol=1e-5), "bgr[2] should be red"
        r.ok("swizzle: .bgr (reorder)")
    except Exception as e:
        r.fail("swizzle: .bgr (reorder)", f"{e}\n{traceback.format_exc()}")

    # .xyzw (alias for rgba)
    try:
        code = "vec4 c = @A.xyzw;\n@OUT = c;"
        result = compile_and_run(code, {"A": test_img})
        assert torch.allclose(result, test_img, atol=1e-5)
        r.ok("swizzle: .xyzw (alias)")
    except Exception as e:
        r.fail("swizzle: .xyzw (alias)", f"{e}\n{traceback.format_exc()}")

    # .x and .w single access
    try:
        code = "float x = @A.x;\nfloat w = @A.w;\n@OUT = vec4(x, w, 0.0, 0.0);"
        result = compile_and_run(code, {"A": test_img})
        assert torch.allclose(result[..., 0], test_img[..., 0], atol=1e-5), ".x should be red"
        assert torch.allclose(result[..., 1], test_img[..., 3], atol=1e-5), ".w should be alpha"
        r.ok("swizzle: .x .w (singles)")
    except Exception as e:
        r.fail("swizzle: .x .w (singles)", f"{e}\n{traceback.format_exc()}")

    # .abgr (full reverse)
    try:
        code = "vec4 c = @A.abgr;\n@OUT = c;"
        result = compile_and_run(code, {"A": test_img})
        assert torch.allclose(result[..., 0], test_img[..., 3], atol=1e-5), "abgr[0] should be alpha"
        assert torch.allclose(result[..., 3], test_img[..., 0], atol=1e-5), "abgr[3] should be red"
        r.ok("swizzle: .abgr (reverse)")
    except Exception as e:
        r.fail("swizzle: .abgr (reverse)", f"{e}\n{traceback.format_exc()}")


# ── Performance Tests ─────────────────────────────────────────────────

def test_performance(r: TestResult):
    print("\n--- Performance Tests ---")

    B, H, W = 1, 512, 512
    perf_img = torch.rand(B, H, W, 3)

    # Cold compile (fresh cache)
    try:
        tmp_dir = tempfile.mkdtemp()
        cold_cache = TEXCache(cache_dir=Path(tmp_dir))
        code = "float g = luma(@A);\n@OUT = vec3(g, g, g);"

        tokens = Lexer(code).tokenize()
        prog = Parser(tokens).parse()
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
        checker = TypeChecker(binding_types=bt)
        type_map = checker.check(prog)

        start = time.perf_counter()
        interp = Interpreter()
        interp.execute(prog, {"A": perf_img}, type_map, device="cpu")
        cold_ms = (time.perf_counter() - start) * 1000

        shutil.rmtree(tmp_dir, ignore_errors=True)
        assert cold_ms < 2000, f"Cold compile took {cold_ms:.1f}ms (limit: 2000ms)"
        r.ok(f"perf: cold compile ({cold_ms:.1f}ms)")
    except Exception as e:
        r.fail("perf: cold compile", f"{e}\n{traceback.format_exc()}")

    # Hot cached (median of 10)
    try:
        code = "float g = luma(@A);\n@OUT = vec3(g, g, g);"
        # Warm up
        compile_and_run(code, {"A": perf_img})

        times = []
        for _ in range(10):
            start = time.perf_counter()
            compile_and_run(code, {"A": perf_img})
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        median_ms = times[len(times) // 2]
        assert median_ms < 50, f"Hot execution median {median_ms:.1f}ms (limit: 50ms)"
        r.ok(f"perf: hot cached ({median_ms:.1f}ms)")
    except Exception as e:
        r.fail("perf: hot cached", f"{e}\n{traceback.format_exc()}")

    # Complex expression
    try:
        code = "float x = smoothstep(0.0, 1.0, sin(u * PI) * 0.5 + 0.5);\n@OUT = vec3(clamp(x, 0.0, 1.0), lerp(0.0, 1.0, x), x);"
        # Warm up
        compile_and_run(code, {"A": perf_img})

        times = []
        for _ in range(10):
            start = time.perf_counter()
            compile_and_run(code, {"A": perf_img})
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        median_ms = times[len(times) // 2]
        assert median_ms < 100, f"Complex expression median {median_ms:.1f}ms (limit: 100ms)"
        r.ok(f"perf: complex expression ({median_ms:.1f}ms)")
    except Exception as e:
        r.fail("perf: complex expression", f"{e}\n{traceback.format_exc()}")

    # For-loop 10 iterations
    try:
        code = "vec3 sum = vec3(0.0, 0.0, 0.0);\nfor (int i = 0; i < 10; i++) {\n    sum += @A * 0.1;\n}\n@OUT = sum;"
        small_img = torch.rand(1, 256, 256, 3)
        # Warm up
        compile_and_run(code, {"A": small_img})

        times = []
        for _ in range(10):
            start = time.perf_counter()
            compile_and_run(code, {"A": small_img})
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        median_ms = times[len(times) // 2]
        assert median_ms < 500, f"For-loop median {median_ms:.1f}ms (limit: 500ms)"
        r.ok(f"perf: for-loop 10 iters ({median_ms:.1f}ms)")
    except Exception as e:
        r.fail("perf: for-loop 10 iters", f"{e}\n{traceback.format_exc()}")


# ── Cache Eviction Tests ──────────────────────────────────────────────

def test_cache_eviction(r: TestResult):
    print("\n--- Cache Eviction Tests ---")

    # Memory eviction at 128
    try:
        tmp_dir = tempfile.mkdtemp()
        cache = TEXCache(cache_dir=Path(tmp_dir))
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
        for i in range(135):
            code = f"@OUT = @A * {i}.0;"
            cache.compile_tex(code, bt)
        assert len(cache._memory) <= 128, f"Memory has {len(cache._memory)} entries (limit: 128)"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        r.ok(f"cache: memory eviction ({len(cache._memory)} entries)")
    except Exception as e:
        r.fail("cache: memory eviction", f"{e}\n{traceback.format_exc()}")

    # Disk eviction at 512
    try:
        tmp_dir = tempfile.mkdtemp()
        cache = TEXCache(cache_dir=Path(tmp_dir))
        # Create 515 dummy .pkl files
        for i in range(515):
            p = Path(tmp_dir) / f"dummy_{i:04d}.pkl"
            with open(p, "wb") as f:
                pickle.dump({"dummy": i}, f)
            # Stagger access times slightly (on Windows atime may not update,
            # so we use mtime as a proxy — the eviction sorts by atime)
            os.utime(p, (i, i))
        cache._evict_disk_if_needed()
        remaining = len(list(Path(tmp_dir).glob("*.pkl")))
        assert remaining <= 512, f"Disk has {remaining} files (limit: 512)"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        r.ok(f"cache: disk eviction ({remaining} files)")
    except Exception as e:
        r.fail("cache: disk eviction", f"{e}\n{traceback.format_exc()}")


# ── Latent Tests ──────────────────────────────────────────────────────

def _make_latent(B=1, C=4, H=4, W=4) -> dict:
    """Create a fake LATENT dict for testing."""
    return {"samples": torch.randn(B, C, H, W)}


def test_latent(r: TestResult):
    print("\n--- Latent Tests ---")

    # latent: unwrap permute
    try:
        lat = _make_latent(1, 4, 8, 8)
        tensor_cl, meta = _unwrap_latent(lat)
        assert tensor_cl.shape == (1, 8, 8, 4), f"Expected [1,8,8,4] got {tensor_cl.shape}"
        assert isinstance(meta, dict)
        # Verify data is correctly permuted
        assert torch.allclose(tensor_cl[0, 0, 0, :], lat["samples"][0, :, 0, 0])
        r.ok("latent: unwrap permute")
    except Exception as e:
        r.fail("latent: unwrap permute", f"{e}\n{traceback.format_exc()}")

    # latent: infer type dict
    try:
        lat4 = _make_latent(1, 4, 4, 4)
        assert _infer_binding_type(lat4) == TEXType.VEC4
        lat3 = {"samples": torch.randn(1, 3, 4, 4)}
        assert _infer_binding_type(lat3) == TEXType.VEC3
        lat16 = {"samples": torch.randn(1, 16, 4, 4)}
        assert _infer_binding_type(lat16) == TEXType.VEC4  # 16-ch -> VEC4 best-effort
        r.ok("latent: infer type dict")
    except Exception as e:
        r.fail("latent: infer type dict", f"{e}\n{traceback.format_exc()}")

    # latent: passthrough
    try:
        lat = _make_latent(1, 4, 4, 4)
        original = lat["samples"].clone()
        tensor_cl, meta = _unwrap_latent(lat)
        result = compile_and_run("@OUT = @A;", {"A": tensor_cl}, latent_channel_count=4)
        # Permute result back to [B,C,H,W] and compare
        result_cf = result.permute(0, 3, 1, 2)
        assert torch.allclose(result_cf, original, atol=1e-6), "Passthrough mismatch"
        r.ok("latent: passthrough")
    except Exception as e:
        r.fail("latent: passthrough", f"{e}\n{traceback.format_exc()}")

    # latent: scalar gain
    try:
        lat = _make_latent(1, 4, 4, 4)
        tensor_cl, _ = _unwrap_latent(lat)
        result = compile_and_run("@OUT = @A * 0.5;", {"A": tensor_cl}, latent_channel_count=4)
        expected = tensor_cl * 0.5
        assert torch.allclose(result, expected, atol=1e-6), "Scalar gain mismatch"
        r.ok("latent: scalar gain")
    except Exception as e:
        r.fail("latent: scalar gain", f"{e}\n{traceback.format_exc()}")

    # latent: bias
    try:
        lat = _make_latent(1, 4, 4, 4)
        tensor_cl, _ = _unwrap_latent(lat)
        result = compile_and_run("@OUT = @A + 0.1;", {"A": tensor_cl}, latent_channel_count=4)
        expected = tensor_cl + 0.1
        assert torch.allclose(result, expected, atol=1e-6), "Bias mismatch"
        r.ok("latent: bias")
    except Exception as e:
        r.fail("latent: bias", f"{e}\n{traceback.format_exc()}")

    # latent: no clamp
    try:
        # Create latent with values > 1.0
        samples = torch.ones(1, 4, 4, 4) * 2.0  # [B,C,H,W]
        lat = {"samples": samples}
        tensor_cl, _ = _unwrap_latent(lat)
        result = compile_and_run("@OUT = @A * 2.0;", {"A": tensor_cl}, latent_channel_count=4)
        # _prepare_output with LATENT should NOT clamp
        prepared = _prepare_output(result, "LATENT")
        assert prepared.max().item() > 1.0, f"Values were clamped! max={prepared.max().item()}"
        r.ok("latent: no clamp")
    except Exception as e:
        r.fail("latent: no clamp", f"{e}\n{traceback.format_exc()}")

    # latent: metadata preserved
    try:
        noise_mask = torch.ones(1, 1, 4, 4)
        lat = {"samples": torch.randn(1, 4, 4, 4), "noise_mask": noise_mask, "batch_index": [0]}
        tensor_cl, meta = _unwrap_latent(lat)
        assert "noise_mask" in meta, "noise_mask missing from metadata"
        assert "batch_index" in meta, "batch_index missing from metadata"
        assert torch.equal(meta["noise_mask"], noise_mask), "noise_mask value changed"
        assert meta["batch_index"] == [0], "batch_index value changed"
        r.ok("latent: metadata preserved")
    except Exception as e:
        r.fail("latent: metadata preserved", f"{e}\n{traceback.format_exc()}")

    # latent: ic variable
    try:
        lat = _make_latent(1, 4, 4, 4)
        tensor_cl, _ = _unwrap_latent(lat)
        # Use ic to verify it equals channel count
        result = compile_and_run(
            "float c = ic; @OUT = vec4(c, c, c, c);",
            {"A": tensor_cl},
            latent_channel_count=4,
        )
        ic_val = result[0, 0, 0, 0].item()
        assert abs(ic_val - 4.0) < 1e-6, f"ic should be 4.0, got {ic_val}"
        r.ok("latent: ic variable")
    except Exception as e:
        r.fail("latent: ic variable", f"{e}\n{traceback.format_exc()}")

    # latent: lerp two latents
    try:
        lat_a = {"samples": torch.zeros(1, 4, 4, 4)}
        lat_b = {"samples": torch.ones(1, 4, 4, 4) * 2.0}
        tcl_a, _ = _unwrap_latent(lat_a)
        tcl_b, _ = _unwrap_latent(lat_b)
        result = compile_and_run(
            "@OUT = lerp(@A, @B, 0.5);",
            {"A": tcl_a, "B": tcl_b},
            latent_channel_count=4,
        )
        expected_val = 1.0  # midpoint of 0 and 2
        assert torch.allclose(result, torch.ones_like(result) * expected_val, atol=1e-6), \
            f"Lerp midpoint mismatch, got {result[0,0,0,0].item()}"
        r.ok("latent: lerp two latents")
    except Exception as e:
        r.fail("latent: lerp two latents", f"{e}\n{traceback.format_exc()}")

    # latent: channel access
    try:
        # Known values per channel
        samples = torch.zeros(1, 4, 4, 4)
        samples[0, 0, :, :] = 1.0  # channel 0 = 1.0
        samples[0, 1, :, :] = 2.0  # channel 1 = 2.0
        lat = {"samples": samples}
        tensor_cl, _ = _unwrap_latent(lat)
        # .r should be channel 0
        result = compile_and_run(
            "float ch0 = @A.r; @OUT = vec4(ch0, ch0, ch0, ch0);",
            {"A": tensor_cl},
            latent_channel_count=4,
        )
        val = result[0, 0, 0, 0].item()
        assert abs(val - 1.0) < 1e-6, f"Channel 0 should be 1.0, got {val}"
        r.ok("latent: channel access")
    except Exception as e:
        r.fail("latent: channel access", f"{e}\n{traceback.format_exc()}")

    # latent: prepare output
    try:
        # [B,H,W,C] -> [B,C,H,W]
        raw = torch.randn(1, 8, 8, 4)
        prepared = _prepare_output(raw, "LATENT")
        assert prepared.shape == (1, 4, 8, 8), f"Expected [1,4,8,8] got {prepared.shape}"
        # Verify correct permutation
        assert torch.allclose(prepared[0, :, 0, 0], raw[0, 0, 0, :])
        r.ok("latent: prepare output")
    except Exception as e:
        r.fail("latent: prepare output", f"{e}\n{traceback.format_exc()}")

    # latent: IS_CHANGED
    try:
        lat = _make_latent(1, 4, 4, 4)
        from TEX_Wrangle.tex_node import TEXWrangleNode
        # Should not crash with LATENT dict
        h = TEXWrangleNode.IS_CHANGED("@OUT = @A;", A=lat)
        assert isinstance(h, str), f"Expected hash string, got {type(h)}"
        r.ok("latent: IS_CHANGED")
    except Exception as e:
        r.fail("latent: IS_CHANGED", f"{e}\n{traceback.format_exc()}")


# ── String Tests ───────────────────────────────────────────────────────

def test_string(r: TestResult):
    print("\n--- String Tests ---")

    # 1. String literal
    try:
        result = compile_and_run(
            'string s = "hello"; @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello", f"Expected 'hello', got {result!r}"
        r.ok("string: literal")
    except Exception as e:
        r.fail("string: literal", f"{e}\n{traceback.format_exc()}")

    # 2. String concatenation
    try:
        result = compile_and_run(
            'string s = "hello" + " " + "world"; @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello world", f"Expected 'hello world', got {result!r}"
        r.ok("string: concatenation")
    except Exception as e:
        r.fail("string: concatenation", f"{e}\n{traceback.format_exc()}")

    # 3. String escape sequences
    try:
        result = compile_and_run(
            r'string s = "line1\nline2"; @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "line1\nline2", f"Expected newline, got {result!r}"
        r.ok("string: escape sequences")
    except Exception as e:
        r.fail("string: escape sequences", f"{e}\n{traceback.format_exc()}")

    # 4. String equality
    try:
        result = compile_and_run(
            'float eq = ("abc" == "abc"); @OUT = vec4(eq, eq, eq, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 1.0) < 1e-6, f"Expected 1.0, got {val}"
        r.ok("string: equality")
    except Exception as e:
        r.fail("string: equality", f"{e}\n{traceback.format_exc()}")

    # 5. String inequality
    try:
        result = compile_and_run(
            'float neq = ("abc" != "xyz"); @OUT = vec4(neq, neq, neq, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 1.0) < 1e-6, f"Expected 1.0, got {val}"
        r.ok("string: inequality")
    except Exception as e:
        r.fail("string: inequality", f"{e}\n{traceback.format_exc()}")

    # 6. String binding input
    try:
        result = compile_and_run(
            '@OUT = @A + " world";',
            {"A": "hello"}, out_type=TEXType.STRING)
        assert result == "hello world", f"Expected 'hello world', got {result!r}"
        r.ok("string: binding input")
    except Exception as e:
        r.fail("string: binding input", f"{e}\n{traceback.format_exc()}")

    # 7. str() function
    try:
        result = compile_and_run(
            'string s = str(42); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "42", f"Expected '42', got {result!r}"
        r.ok("string: str() function")
    except Exception as e:
        r.fail("string: str() function", f"{e}\n{traceback.format_exc()}")

    # 8. string() cast
    try:
        result = compile_and_run(
            'string s = string(3.14); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert "3.14" in result, f"Expected '3.14' in result, got {result!r}"
        r.ok("string: string() cast")
    except Exception as e:
        r.fail("string: string() cast", f"{e}\n{traceback.format_exc()}")

    # 9. len()
    try:
        result = compile_and_run(
            'float n = len("hello"); @OUT = vec4(n, n, n, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 5.0) < 1e-6, f"Expected 5.0, got {val}"
        r.ok("string: len()")
    except Exception as e:
        r.fail("string: len()", f"{e}\n{traceback.format_exc()}")

    # 10. replace()
    try:
        result = compile_and_run(
            'string s = replace("hello world", "world", "TEX"); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello TEX", f"Expected 'hello TEX', got {result!r}"
        r.ok("string: replace()")
    except Exception as e:
        r.fail("string: replace()", f"{e}\n{traceback.format_exc()}")

    # 11. strip()
    try:
        result = compile_and_run(
            'string s = strip("  hello  "); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello", f"Expected 'hello', got {result!r}"
        r.ok("string: strip()")
    except Exception as e:
        r.fail("string: strip()", f"{e}\n{traceback.format_exc()}")

    # 12. upper()
    try:
        result = compile_and_run(
            'string s = upper("hello"); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "HELLO", f"Expected 'HELLO', got {result!r}"
        r.ok("string: upper()")
    except Exception as e:
        r.fail("string: upper()", f"{e}\n{traceback.format_exc()}")

    # 13. contains()
    try:
        result = compile_and_run(
            'float c = contains("hello world", "world"); @OUT = vec4(c, c, c, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 1.0) < 1e-6, f"Expected 1.0, got {val}"
        r.ok("string: contains()")
    except Exception as e:
        r.fail("string: contains()", f"{e}\n{traceback.format_exc()}")

    # 14. find()
    try:
        result = compile_and_run(
            'float idx = find("hello", "ll"); @OUT = vec4(idx, idx, idx, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 2.0) < 1e-6, f"Expected 2.0 (index of 'll'), got {val}"
        r.ok("string: find()")
    except Exception as e:
        r.fail("string: find()", f"{e}\n{traceback.format_exc()}")

    # 15. find() not found
    try:
        result = compile_and_run(
            'float idx = find("hello", "xyz"); @OUT = vec4(idx, idx, idx, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - (-1.0)) < 1e-6, f"Expected -1.0, got {val}"
        r.ok("string: find() not found")
    except Exception as e:
        r.fail("string: find() not found", f"{e}\n{traceback.format_exc()}")

    # 16. substr()
    try:
        result = compile_and_run(
            'string s = substr("hello world", 6, 5); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "world", f"Expected 'world', got {result!r}"
        r.ok("string: substr()")
    except Exception as e:
        r.fail("string: substr()", f"{e}\n{traceback.format_exc()}")

    # 17. to_float()
    try:
        result = compile_and_run(
            'float n = to_float("3.14"); @OUT = vec4(n, n, n, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 3.14) < 1e-4, f"Expected 3.14, got {val}"
        r.ok("string: to_float()")
    except Exception as e:
        r.fail("string: to_float()", f"{e}\n{traceback.format_exc()}")

    # 18. sanitize_filename()
    try:
        result = compile_and_run(
            'string s = sanitize_filename("my:file<name>.txt"); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert ':' not in result and '<' not in result and '>' not in result, \
            f"Expected sanitized filename, got {result!r}"
        r.ok("string: sanitize_filename()")
    except Exception as e:
        r.fail("string: sanitize_filename()", f"{e}\n{traceback.format_exc()}")

    # 19. String ternary
    try:
        result = compile_and_run(
            'string s = 1.0 > 0.5 ? "yes" : "no"; @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "yes", f"Expected 'yes', got {result!r}"
        r.ok("string: ternary")
    except Exception as e:
        r.fail("string: ternary", f"{e}\n{traceback.format_exc()}")

    # 20. String in if/else
    try:
        result = compile_and_run(
            'string s = "default"; if (1.0 > 0.5) { s = "then"; } else { s = "else"; } @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "then", f"Expected 'then', got {result!r}"
        r.ok("string: if/else")
    except Exception as e:
        r.fail("string: if/else", f"{e}\n{traceback.format_exc()}")

    # 21. String in for loop
    try:
        result = compile_and_run(
            'string s = ""; for (int i = 0; i < 3; i++) { s = s + "x"; } @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "xxx", f"Expected 'xxx', got {result!r}"
        r.ok("string: for loop accumulation")
    except Exception as e:
        r.fail("string: for loop accumulation", f"{e}\n{traceback.format_exc()}")

    # 22. Type error: string + number
    try:
        compile_and_run(
            'string s = "hello" + 1;',
            {}, out_type=TEXType.STRING)
        r.fail("string: type error string+number", "Expected TypeCheckError")
    except (TypeCheckError, InterpreterError):
        r.ok("string: type error string+number")
    except Exception as e:
        r.fail("string: type error string+number", f"Wrong error type: {e}")

    # 23. Type error: string in vec constructor
    try:
        compile_and_run(
            'vec3 v = vec3("hello", 0.0, 0.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        r.fail("string: type error vec constructor", "Expected TypeCheckError")
    except TypeCheckError:
        r.ok("string: type error vec constructor")
    except Exception as e:
        r.fail("string: type error vec constructor", f"Wrong error type: {e}")

    # 24. Type error: channel access on string
    try:
        compile_and_run(
            'string s = "abc"; float x = s.r;',
            {}, out_type=TEXType.STRING)
        r.fail("string: type error channel access", "Expected TypeCheckError")
    except TypeCheckError:
        r.ok("string: type error channel access")
    except Exception as e:
        r.fail("string: type error channel access", f"Wrong error type: {e}")

    # 25. startswith / endswith
    try:
        result = compile_and_run(
            'float sw = startswith("hello", "hel"); float ew = endswith("hello", "llo"); '
            '@OUT = vec4(sw, ew, 0.0, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        sw_val = result[0, 0, 0, 0].item()
        ew_val = result[0, 0, 0, 1].item()
        assert abs(sw_val - 1.0) < 1e-6 and abs(ew_val - 1.0) < 1e-6, \
            f"Expected (1.0, 1.0), got ({sw_val}, {ew_val})"
        r.ok("string: startswith/endswith")
    except Exception as e:
        r.fail("string: startswith/endswith", f"{e}\n{traceback.format_exc()}")


# ── Named Binding Tests ────────────────────────────────────────────────

def test_named_bindings(r: TestResult):
    print("\n--- Named Binding Tests ---")

    B, H, W = 1, 4, 4
    img = torch.rand(B, H, W, 4)

    # 1. Snake_case multi-word name
    try:
        code = "@OUT = @base_image;"
        result = compile_and_run(code, {"base_image": img})
        assert torch.allclose(result, img, atol=1e-6)
        r.ok("named: snake_case binding")
    except Exception as e:
        r.fail("named: snake_case binding", f"{e}\n{traceback.format_exc()}")

    # 2. Single lowercase letter
    try:
        code = "@OUT = @x * 2.0;"
        scalar = torch.tensor(3.0)
        result = compile_and_run(code, {"x": scalar})
        assert abs(result.mean().item() - 6.0) < 1e-5
        r.ok("named: single lowercase letter")
    except Exception as e:
        r.fail("named: single lowercase letter", f"{e}\n{traceback.format_exc()}")

    # 3. Two arbitrary bindings combined
    try:
        code = "@OUT = lerp(@base, @overlay, 0.5);"
        base = torch.full((B, H, W, 4), 0.2)
        overlay = torch.full((B, H, W, 4), 0.8)
        result = compile_and_run(code, {"base": base, "overlay": overlay})
        expected = 0.5
        assert abs(result.mean().item() - expected) < 1e-4
        r.ok("named: two arbitrary bindings")
    except Exception as e:
        r.fail("named: two arbitrary bindings", f"{e}\n{traceback.format_exc()}")

    # 4. Backward compat — uppercase single letter still works
    try:
        code = "@OUT = @A;"
        result = compile_and_run(code, {"A": img})
        assert torch.allclose(result, img, atol=1e-6)
        r.ok("named: backward compat @A")
    except Exception as e:
        r.fail("named: backward compat @A", f"{e}\n{traceback.format_exc()}")

    # 5. Named string binding
    try:
        code = 'string result = @prefix + "_output"; @OUT = result;'
        result = compile_and_run(code, {"prefix": "test"}, out_type=TEXType.STRING)
        assert result == "test_output"
        r.ok("named: string binding")
    except Exception as e:
        r.fail("named: string binding", f"{e}\n{traceback.format_exc()}")

    # 6. Name with underscore prefix
    try:
        code = "@OUT = @_hidden * 2.0;"
        scalar = torch.tensor(5.0)
        result = compile_and_run(code, {"_hidden": scalar})
        assert abs(result.mean().item() - 10.0) < 1e-5
        r.ok("named: underscore prefix")
    except Exception as e:
        r.fail("named: underscore prefix", f"{e}\n{traceback.format_exc()}")

    # 7. Name with digits
    try:
        code = "@OUT = @layer2 + @layer3;"
        l2 = torch.full((B, H, W, 4), 0.3)
        l3 = torch.full((B, H, W, 4), 0.4)
        result = compile_and_run(code, {"layer2": l2, "layer3": l3})
        assert abs(result.mean().item() - 0.7) < 1e-4
        r.ok("named: binding with digits")
    except Exception as e:
        r.fail("named: binding with digits", f"{e}\n{traceback.format_exc()}")

    # 8. Long descriptive name
    try:
        code = "float g = luma(@high_resolution_input_image);\n@OUT = vec4(g, g, g, 1.0);"
        result = compile_and_run(code, {"high_resolution_input_image": img})
        assert result.shape[-1] == 4
        r.ok("named: long descriptive name")
    except Exception as e:
        r.fail("named: long descriptive name", f"{e}\n{traceback.format_exc()}")


# ── Array Tests ────────────────────────────────────────────────────────

def test_arrays(r: TestResult):
    """Tests for fixed-size array type."""
    print("\n--- Array Tests ---")
    B, H, W = 1, 4, 4
    img = torch.rand(B, H, W, 4)

    # 1. Lexer: bracket tokens
    try:
        tokens = Lexer("arr[0]").tokenize()
        types = [t.type for t in tokens[:-1]]
        assert TokenType.LBRACKET in types
        assert TokenType.RBRACKET in types
        r.ok("array: bracket tokens")
    except Exception as e:
        r.fail("array: bracket tokens", f"{e}\n{traceback.format_exc()}")

    # 2. Parser: array declaration with size
    try:
        tokens = Lexer("float arr[3];").tokenize()
        from TEX_Wrangle.tex_compiler.ast_nodes import ArrayDecl
        program = Parser(tokens).parse()
        assert len(program.statements) == 1
        decl = program.statements[0]
        assert isinstance(decl, ArrayDecl)
        assert decl.element_type_name == "float"
        assert decl.name == "arr"
        assert decl.size == 3
        r.ok("array: parse declaration with size")
    except Exception as e:
        r.fail("array: parse declaration with size", f"{e}\n{traceback.format_exc()}")

    # 3. Parser: array with initializer
    try:
        tokens = Lexer("float arr[] = {1.0, 2.0, 3.0};").tokenize()
        from TEX_Wrangle.tex_compiler.ast_nodes import ArrayDecl, ArrayLiteral
        program = Parser(tokens).parse()
        decl = program.statements[0]
        assert isinstance(decl, ArrayDecl)
        assert decl.size is None  # inferred
        assert isinstance(decl.initializer, ArrayLiteral)
        assert len(decl.initializer.elements) == 3
        r.ok("array: parse initializer list")
    except Exception as e:
        r.fail("array: parse initializer list", f"{e}\n{traceback.format_exc()}")

    # 4. Parser: error on empty array without size or initializer
    try:
        tokens = Lexer("float arr[];").tokenize()
        Parser(tokens).parse()
        r.fail("array: missing size error", "Expected ParseError")
    except ParseError:
        r.ok("array: missing size error")
    except Exception as e:
        r.fail("array: missing size error", f"{e}\n{traceback.format_exc()}")

    # 5. Type checker: vec3/vec4/string element types now allowed
    try:
        tokens = Lexer("vec3 arr[5]; @OUT = vec4(0.0);").tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"OUT": TEXType.VEC4})
        tc.check(program)
        r.ok("array: vec3 element type allowed")
    except Exception as e:
        r.fail("array: vec3 element type allowed", f"{e}\n{traceback.format_exc()}")

    # 6. Declaration and element access
    try:
        code = """
float arr[3];
arr[0] = 1.0;
arr[1] = 2.0;
arr[2] = 3.0;
@OUT = vec4(arr[0], arr[1], arr[2], 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 2.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 3.0) < 1e-4
        r.ok("array: declaration and access")
    except Exception as e:
        r.fail("array: declaration and access", f"{e}\n{traceback.format_exc()}")

    # 7. Initializer list
    try:
        code = """
float arr[] = {0.25, 0.5, 0.75};
@OUT = vec4(arr[0], arr[1], arr[2], 1.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 0.25) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 0.50) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 0.75) < 1e-4
        r.ok("array: initializer list")
    except Exception as e:
        r.fail("array: initializer list", f"{e}\n{traceback.format_exc()}")

    # 8. Loop population
    try:
        code = """
float arr[5];
for (int i = 0; i < 5; i++) {
    arr[i] = float(i) * 0.25;
}
@OUT = vec4(arr[2], arr[4], 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 0.5) < 1e-4  # arr[2] = 2*0.25
        assert abs(result[0, 0, 0, 1].item() - 1.0) < 1e-4  # arr[4] = 4*0.25
        r.ok("array: loop population")
    except Exception as e:
        r.fail("array: loop population", f"{e}\n{traceback.format_exc()}")

    # 9. Clamped bounds
    try:
        code = """
float arr[] = {1.0, 2.0, 3.0};
float x = arr[10];
float y = arr[-1];
@OUT = vec4(x, y, 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4  # clamped to arr[2]
        assert abs(result[0, 0, 0, 1].item() - 1.0) < 1e-4  # clamped to arr[0]
        r.ok("array: clamped bounds")
    except Exception as e:
        r.fail("array: clamped bounds", f"{e}\n{traceback.format_exc()}")

    # 10. Array copy
    try:
        code = """
float a[] = {1.0, 2.0, 3.0};
float b[3] = a;
b[0] = 99.0;
@OUT = vec4(a[0], b[0], 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4   # a[0] unchanged
        assert abs(result[0, 0, 0, 1].item() - 99.0) < 1e-4  # b[0] modified
        r.ok("array: copy independence")
    except Exception as e:
        r.fail("array: copy independence", f"{e}\n{traceback.format_exc()}")

    # 11. sort()
    try:
        code = """
float arr[] = {3.0, 1.0, 2.0};
arr = sort(arr);
@OUT = vec4(arr[0], arr[1], arr[2], 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 2.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 3.0) < 1e-4
        r.ok("array: sort")
    except Exception as e:
        r.fail("array: sort", f"{e}\n{traceback.format_exc()}")

    # 12. reverse()
    try:
        code = """
float arr[] = {1.0, 2.0, 3.0};
arr = reverse(arr);
@OUT = vec4(arr[0], arr[1], arr[2], 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 2.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 1.0) < 1e-4
        r.ok("array: reverse")
    except Exception as e:
        r.fail("array: reverse", f"{e}\n{traceback.format_exc()}")

    # 13. arr_sum, arr_min, arr_max, median, arr_avg
    try:
        code = """
float arr[] = {1.0, 2.0, 3.0, 4.0, 5.0};
float s = arr_sum(arr);
float mn = arr_min(arr);
float mx = arr_max(arr);
float md = median(arr);
float av = arr_avg(arr);
@OUT = vec4(s, mn, mx, md);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 15.0) < 1e-4  # sum
        assert abs(result[0, 0, 0, 1].item() - 1.0) < 1e-4   # min
        assert abs(result[0, 0, 0, 2].item() - 5.0) < 1e-4   # max
        assert abs(result[0, 0, 0, 3].item() - 3.0) < 1e-4   # median
        r.ok("array: aggregate functions")
    except Exception as e:
        r.fail("array: aggregate functions", f"{e}\n{traceback.format_exc()}")

    # 14. arr_avg separately
    try:
        code = """
float arr[] = {2.0, 4.0, 6.0};
float av = arr_avg(arr);
@OUT = vec4(av, 0.0, 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 4.0) < 1e-4
        r.ok("array: arr_avg")
    except Exception as e:
        r.fail("array: arr_avg", f"{e}\n{traceback.format_exc()}")

    # 15. len() on arrays
    try:
        code = """
float arr[] = {1.0, 2.0, 3.0};
float n = len(arr);
@OUT = vec4(n, 0.0, 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4
        r.ok("array: len")
    except Exception as e:
        r.fail("array: len", f"{e}\n{traceback.format_exc()}")

    # 16. int arrays
    try:
        code = """
int arr[3];
arr[0] = 10;
arr[1] = 20;
arr[2] = 30;
float total = float(arr[0]) + float(arr[1]) + float(arr[2]);
@OUT = vec4(total, 0.0, 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 60.0) < 1e-4
        r.ok("array: int elements")
    except Exception as e:
        r.fail("array: int elements", f"{e}\n{traceback.format_exc()}")

    # 17. Arrays in if/else (per-pixel merge via torch.where)
    try:
        code = """
float arr[] = {1.0, 2.0, 3.0};
if (u > 0.5) {
    arr[0] = 99.0;
} else {
    arr[0] = 0.0;
}
@OUT = vec4(arr[0], arr[1], arr[2], 0.0);
"""
        result = compile_and_run(code, {"A": img})
        # Left half (u<=0.5): arr[0]=0.0; Right half (u>0.5): arr[0]=99.0
        assert abs(result[0, 0, 0, 0].item() - 0.0) < 1e-4     # u=0
        assert abs(result[0, 0, W-1, 0].item() - 99.0) < 1e-4  # u=1
        # arr[1] and arr[2] unchanged everywhere
        assert abs(result[0, 0, 0, 1].item() - 2.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 3.0) < 1e-4
        r.ok("array: if/else merge")
    except Exception as e:
        r.fail("array: if/else merge", f"{e}\n{traceback.format_exc()}")

    # 18. Per-pixel fetch into array (spatial use case)
    try:
        code = """
float samples[3];
for (int i = 0; i < 3; i++) {
    samples[i] = fetch(@A, ix + i - 1, iy).r;
}
samples = sort(samples);
@OUT = vec4(samples[1], samples[1], samples[1], 1.0);
"""
        result = compile_and_run(code, {"A": img})
        assert result.shape == (B, H, W, 4)
        r.ok("array: spatial fetch+sort")
    except Exception as e:
        r.fail("array: spatial fetch+sort", f"{e}\n{traceback.format_exc()}")

    # 19. Size mismatch error
    try:
        tokens = Lexer("float arr[2] = {1.0, 2.0, 3.0}; @OUT = vec4(0.0);").tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"OUT": TEXType.VEC4})
        tc.check(program)
        r.fail("array: size mismatch error", "Expected TypeCheckError")
    except TypeCheckError as e:
        assert "mismatch" in str(e).lower()
        r.ok("array: size mismatch error")
    except Exception as e:
        r.fail("array: size mismatch error", f"{e}\n{traceback.format_exc()}")

    # 20. Indexing non-array error
    try:
        tokens = Lexer("float x = 5.0; float y = x[0]; @OUT = vec4(y);").tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"OUT": TEXType.VEC4})
        tc.check(program)
        r.fail("array: index non-array error", "Expected TypeCheckError")
    except TypeCheckError as e:
        assert "non-array" in str(e).lower()
        r.ok("array: index non-array error")
    except Exception as e:
        r.fail("array: index non-array error", f"{e}\n{traceback.format_exc()}")


# ── Auto-Inference Tests ──────────────────────────────────────────────

def compile_and_infer(code: str, bindings: dict, device: str = "cpu",
                      latent_channel_count: int = 0) -> tuple:
    """Helper: compile+execute in auto-inference mode. Returns (result, inferred_out_type)."""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()

    binding_types = {}
    for name, val in bindings.items():
        if isinstance(val, str):
            binding_types[name] = TEXType.STRING
        elif isinstance(val, torch.Tensor):
            if val.dim() == 4:
                c = val.shape[-1]
                binding_types[name] = TEXType.VEC4 if c == 4 else TEXType.VEC3
            elif val.dim() == 3:
                binding_types[name] = TEXType.FLOAT
            else:
                binding_types[name] = TEXType.FLOAT
        else:
            binding_types[name] = TEXType.FLOAT
    # OUT is NOT in binding_types -> triggers auto-inference

    checker = TypeChecker(binding_types=binding_types)
    type_map = checker.check(program)
    inferred = checker.inferred_out_type

    interp = Interpreter()
    result = interp.execute(program, bindings, type_map, device=device,
                            latent_channel_count=latent_channel_count)
    return result, inferred


def test_auto_inference(r: TestResult):
    print("\n--- Auto-Inference Tests ---")
    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 4)

    # vec3 output -> IMAGE
    try:
        _, inferred = compile_and_infer(
            "float g = luma(@A); @OUT = vec3(g, g, g);",
            {"A": test_img},
        )
        assert inferred == TEXType.VEC3, f"Expected VEC3, got {inferred}"
        assert _map_inferred_type(inferred, False) == "IMAGE"
        r.ok("auto: vec3 -> IMAGE")
    except Exception as e:
        r.fail("auto: vec3 -> IMAGE", f"{e}\n{traceback.format_exc()}")

    # vec4 output (no latent) -> IMAGE
    try:
        _, inferred = compile_and_infer(
            "@OUT = @A;",
            {"A": test_img},
        )
        assert inferred == TEXType.VEC4, f"Expected VEC4, got {inferred}"
        assert _map_inferred_type(inferred, False) == "IMAGE"
        r.ok("auto: vec4 -> IMAGE")
    except Exception as e:
        r.fail("auto: vec4 -> IMAGE", f"{e}\n{traceback.format_exc()}")

    # vec4 output (with latent) -> LATENT
    try:
        _, inferred = compile_and_infer(
            "@OUT = @A * 1.1;",
            {"A": test_img},
        )
        assert inferred == TEXType.VEC4, f"Expected VEC4, got {inferred}"
        assert _map_inferred_type(inferred, True) == "LATENT"
        r.ok("auto: vec4 + latent -> LATENT")
    except Exception as e:
        r.fail("auto: vec4 + latent -> LATENT", f"{e}\n{traceback.format_exc()}")

    # float output -> MASK
    try:
        _, inferred = compile_and_infer(
            "@OUT = luma(@A);",
            {"A": test_img},
        )
        assert inferred == TEXType.FLOAT, f"Expected FLOAT, got {inferred}"
        assert _map_inferred_type(inferred, False) == "MASK"
        r.ok("auto: float -> MASK")
    except Exception as e:
        r.fail("auto: float -> MASK", f"{e}\n{traceback.format_exc()}")

    # int output -> INT
    try:
        _, inferred = compile_and_infer(
            "int x = 42; @OUT = x;",
            {"A": test_img},
        )
        assert inferred == TEXType.INT, f"Expected INT, got {inferred}"
        assert _map_inferred_type(inferred, False) == "INT"
        r.ok("auto: int -> INT")
    except Exception as e:
        r.fail("auto: int -> INT", f"{e}\n{traceback.format_exc()}")

    # string output -> STRING
    try:
        _, inferred = compile_and_infer(
            '@OUT = "hello";',
            {"A": test_img},
        )
        assert inferred == TEXType.STRING, f"Expected STRING, got {inferred}"
        assert _map_inferred_type(inferred, False) == "STRING"
        r.ok("auto: string -> STRING")
    except Exception as e:
        r.fail("auto: string -> STRING", f"{e}\n{traceback.format_exc()}")

    # channel assignment -> VEC3
    try:
        _, inferred = compile_and_infer(
            "@OUT = @A; @OUT.r = 1.0; @OUT.g = 0.5; @OUT.b = 0.0;",
            {"A": test_img},
        )
        # First assignment @OUT = @A infers VEC4, channel .r/.g/.b don't widen
        # Actually: first assigns VEC4, channel accesses on VEC4 stay VEC4
        assert inferred in (TEXType.VEC3, TEXType.VEC4), f"Expected VEC3/VEC4, got {inferred}"
        assert _map_inferred_type(inferred, False) == "IMAGE"
        r.ok("auto: channel assignment -> IMAGE")
    except Exception as e:
        r.fail("auto: channel assignment -> IMAGE", f"{e}\n{traceback.format_exc()}")

    # channel-only assignment (no direct @OUT = ...) -> VEC3
    try:
        binding_types = {"A": TEXType.VEC4}
        checker = TypeChecker(binding_types=binding_types)
        tokens = Lexer("@OUT.r = @A.r; @OUT.g = @A.g; @OUT.b = @A.b;").tokenize()
        program = Parser(tokens).parse()
        checker.check(program)
        assert checker.inferred_out_type == TEXType.VEC3, f"Expected VEC3, got {checker.inferred_out_type}"
        r.ok("auto: channel-only -> VEC3")
    except Exception as e:
        r.fail("auto: channel-only -> VEC3", f"{e}\n{traceback.format_exc()}")

    # if/else same type
    try:
        _, inferred = compile_and_infer(
            "if (luma(@A) > 0.5) { @OUT = vec3(1.0, 0.0, 0.0); } else { @OUT = vec3(0.0, 0.0, 1.0); }",
            {"A": test_img},
        )
        assert inferred == TEXType.VEC3, f"Expected VEC3, got {inferred}"
        r.ok("auto: if/else same type")
    except Exception as e:
        r.fail("auto: if/else same type", f"{e}\n{traceback.format_exc()}")

    # if/else with promotion (float + vec3 -> vec3)
    try:
        _, inferred = compile_and_infer(
            "if (luma(@A) > 0.5) { @OUT = luma(@A); } else { @OUT = vec3(0.0, 0.0, 1.0); }",
            {"A": test_img},
        )
        assert inferred == TEXType.VEC3, f"Expected VEC3, got {inferred}"
        r.ok("auto: if/else promotion")
    except Exception as e:
        r.fail("auto: if/else promotion", f"{e}\n{traceback.format_exc()}")

    # string vs numeric conflict -> error
    try:
        binding_types = {"A": TEXType.VEC4}
        checker = TypeChecker(binding_types=binding_types)
        tokens = Lexer('if (luma(@A) > 0.5) { @OUT = "yes"; } else { @OUT = vec3(0.0, 0.0, 1.0); }').tokenize()
        program = Parser(tokens).parse()
        try:
            checker.check(program)
            r.fail("auto: string/numeric conflict", "Expected TypeCheckError")
        except TypeCheckError as e:
            assert "string" in str(e).lower() and "numeric" in str(e).lower()
            r.ok("auto: string/numeric conflict")
    except Exception as e:
        r.fail("auto: string/numeric conflict", f"{e}\n{traceback.format_exc()}")

    # explicit output_type still works (backward compat)
    try:
        result = compile_and_run(
            "@OUT = luma(@A);",
            {"A": test_img},
            out_type=TEXType.FLOAT,
        )
        assert isinstance(result, torch.Tensor)
        r.ok("auto: explicit still works")
    except Exception as e:
        r.fail("auto: explicit still works", f"{e}\n{traceback.format_exc()}")

    # _map_inferred_type with None -> IMAGE
    try:
        assert _map_inferred_type(None, False) == "IMAGE"
        assert _map_inferred_type(None, True) == "IMAGE"
        r.ok("auto: None -> IMAGE fallback")
    except Exception as e:
        r.fail("auto: None -> IMAGE fallback", f"{e}\n{traceback.format_exc()}")


def test_batch_temporal(r: TestResult):
    print("\n--- Batch / Temporal Tests ---")

    # ── Built-in variables: fi, fn ─────────────────────────────────────

    # Test 1: fi values correct across frames
    try:
        B, H, W = 4, 2, 2
        img = torch.rand(B, H, W, 3)
        result = compile_and_run("@OUT = vec3(fi, fi, fi);", {"A": img})
        # Each frame b should have fi == b
        for b in range(B):
            expected = float(b)
            actual = result[b, 0, 0, 0].item()
            assert abs(actual - expected) < 1e-5, f"frame {b}: fi={actual}, expected {expected}"
        r.ok("fi values correct (B=4)")
    except Exception as e:
        r.fail("fi values correct (B=4)", f"{e}\n{traceback.format_exc()}")

    # Test 2: fn equals batch size
    try:
        B, H, W = 4, 2, 2
        img = torch.rand(B, H, W, 3)
        result = compile_and_run("@OUT = vec3(fn, fn, fn);", {"A": img})
        for b in range(B):
            actual = result[b, 0, 0, 0].item()
            assert abs(actual - 4.0) < 1e-5, f"frame {b}: fn={actual}, expected 4.0"
        r.ok("fn equals batch size (B=4)")
    except Exception as e:
        r.fail("fn equals batch size (B=4)", f"{e}\n{traceback.format_exc()}")

    # Test 3: fi/fn scalar mode (no image inputs)
    try:
        result = compile_and_run(
            "@OUT = vec4(fi, fn, 0.0, 1.0);",
            {},
        )
        fi_val = result.flatten()[0].item()
        fn_val = result.flatten()[1].item()
        assert abs(fi_val - 0.0) < 1e-5, f"scalar fi={fi_val}"
        assert abs(fn_val - 1.0) < 1e-5, f"scalar fn={fn_val}"
        r.ok("fi/fn scalar mode (no images)")
    except Exception as e:
        r.fail("fi/fn scalar mode (no images)", f"{e}\n{traceback.format_exc()}")

    # Test 4: fade effect — frame 0 is black, last frame is original
    try:
        B, H, W = 4, 2, 2
        img = torch.ones(B, H, W, 4) * 0.8
        result = compile_and_run(
            "@OUT = @A * (fi / max(fn - 1, 1));",
            {"A": img},
        )
        # Frame 0: fi=0, so 0.8 * 0/3 = 0.0
        assert result[0].abs().max().item() < 1e-5, f"frame 0 not black"
        # Frame 3: fi=3, so 0.8 * 3/3 = 0.8
        assert abs(result[3, 0, 0, 0].item() - 0.8) < 1e-4, f"frame 3 not original"
        r.ok("fade effect (fi / max(fn-1, 1))")
    except Exception as e:
        r.fail("fade effect (fi / max(fn-1, 1))", f"{e}\n{traceback.format_exc()}")

    # ── Cross-frame fetch ──────────────────────────────────────────────

    # Create distinct frames for cross-frame tests: R=0.1*b per frame
    B, H, W = 3, 2, 2

    def make_distinct_frames(num_frames):
        """Create image batch where each frame has a unique solid color."""
        frames = torch.zeros(num_frames, H, W, 4)
        for b in range(num_frames):
            frames[b, :, :, 0] = 0.1 * (b + 1)  # R varies by frame
            frames[b, :, :, 1] = 0.5
            frames[b, :, :, 2] = 0.5
            frames[b, :, :, 3] = 1.0
        return frames

    distinct = make_distinct_frames(B)

    # Test 5: fetch_frame from frame 0 — all frames should read frame 0's color
    try:
        result = compile_and_run(
            "@OUT = fetch_frame(@A, 0, ix, iy);",
            {"A": distinct},
        )
        expected_r = 0.1  # frame 0 R = 0.1
        for b in range(B):
            actual_r = result[b, 0, 0, 0].item()
            assert abs(actual_r - expected_r) < 1e-5, \
                f"frame {b}: got R={actual_r}, expected {expected_r}"
        r.ok("fetch_frame(@A, 0, ix, iy) reads frame 0")
    except Exception as e:
        r.fail("fetch_frame(@A, 0, ix, iy) reads frame 0", f"{e}\n{traceback.format_exc()}")

    # Test 6: fetch_frame with fi+1 — each frame reads from next (last clamps)
    try:
        result = compile_and_run(
            "@OUT = fetch_frame(@A, fi + 1, ix, iy);",
            {"A": distinct},
        )
        # frame 0 reads frame 1 (R=0.2), frame 1 reads frame 2 (R=0.3),
        # frame 2 reads frame 2 (clamped, R=0.3)
        expected = [0.2, 0.3, 0.3]
        for b in range(B):
            actual_r = result[b, 0, 0, 0].item()
            assert abs(actual_r - expected[b]) < 1e-5, \
                f"frame {b}: got R={actual_r}, expected {expected[b]}"
        r.ok("fetch_frame(@A, fi+1, ix, iy) next-frame read")
    except Exception as e:
        r.fail("fetch_frame(@A, fi+1, ix, iy) next-frame read", f"{e}\n{traceback.format_exc()}")

    # Test 7: negative frame clamped to 0
    try:
        result = compile_and_run(
            "@OUT = fetch_frame(@A, -1, ix, iy);",
            {"A": distinct},
        )
        expected_r = 0.1  # frame 0
        for b in range(B):
            actual_r = result[b, 0, 0, 0].item()
            assert abs(actual_r - expected_r) < 1e-5, \
                f"frame {b}: got R={actual_r}, expected {expected_r}"
        r.ok("fetch_frame negative frame clamped to 0")
    except Exception as e:
        r.fail("fetch_frame negative frame clamped to 0", f"{e}\n{traceback.format_exc()}")

    # Test 8: oversized frame clamped to B-1
    try:
        result = compile_and_run(
            "@OUT = fetch_frame(@A, 999, ix, iy);",
            {"A": distinct},
        )
        expected_r = 0.3  # frame 2 (last)
        for b in range(B):
            actual_r = result[b, 0, 0, 0].item()
            assert abs(actual_r - expected_r) < 1e-5, \
                f"frame {b}: got R={actual_r}, expected {expected_r}"
        r.ok("fetch_frame oversized frame clamped to B-1")
    except Exception as e:
        r.fail("fetch_frame oversized frame clamped to B-1", f"{e}\n{traceback.format_exc()}")

    # ── Cross-frame sample ─────────────────────────────────────────────

    # Test 9: sample_frame from frame 0 — all frames read frame 0
    try:
        result = compile_and_run(
            "@OUT = sample_frame(@A, 0, u, v);",
            {"A": distinct},
        )
        expected_r = 0.1
        for b in range(B):
            actual_r = result[b, 0, 0, 0].item()
            assert abs(actual_r - expected_r) < 1e-5, \
                f"frame {b}: got R={actual_r}, expected {expected_r}"
        r.ok("sample_frame(@A, 0, u, v) reads frame 0")
    except Exception as e:
        r.fail("sample_frame(@A, 0, u, v) reads frame 0", f"{e}\n{traceback.format_exc()}")

    # Test 10: sample_frame with fi equivalent to regular sample
    try:
        result_sf = compile_and_run(
            "@OUT = sample_frame(@A, fi, u, v);",
            {"A": distinct},
        )
        result_s = compile_and_run(
            "@OUT = sample(@A, u, v);",
            {"A": distinct},
        )
        diff = (result_sf - result_s).abs().max().item()
        assert diff < 1e-5, f"sample_frame(fi) vs sample diff: {diff}"
        r.ok("sample_frame(@A, fi, u, v) == sample(@A, u, v)")
    except Exception as e:
        r.fail("sample_frame(@A, fi, u, v) == sample(@A, u, v)", f"{e}\n{traceback.format_exc()}")

    # ── Integration patterns ───────────────────────────────────────────

    # Test 11: frame difference
    try:
        result = compile_and_run(
            "vec4 curr = fetch_frame(@A, fi, ix, iy);\n"
            "vec4 prev = fetch_frame(@A, max(fi - 1, 0), ix, iy);\n"
            "@OUT = abs(curr - prev);",
            {"A": distinct},
        )
        # Frame 0: diff with self = 0
        assert result[0].abs().max().item() < 1e-5, "frame 0 diff should be 0"
        # Frame 1: R diff = |0.2 - 0.1| = 0.1
        r_diff = result[1, 0, 0, 0].item()
        assert abs(r_diff - 0.1) < 1e-4, f"frame 1 R diff: {r_diff}, expected 0.1"
        r.ok("frame difference pattern")
    except Exception as e:
        r.fail("frame difference pattern", f"{e}\n{traceback.format_exc()}")

    # Test 12: temporal average (3-frame blend)
    try:
        result = compile_and_run(
            "vec4 prev = fetch_frame(@A, fi - 1, ix, iy);\n"
            "vec4 curr = fetch_frame(@A, fi, ix, iy);\n"
            "vec4 next = fetch_frame(@A, fi + 1, ix, iy);\n"
            "@OUT = (prev + curr + next) / 3.0;",
            {"A": distinct},
        )
        # Frame 1: prev=frame0(R=0.1), curr=frame1(R=0.2), next=frame2(R=0.3)
        # avg R = (0.1+0.2+0.3)/3 = 0.2
        avg_r = result[1, 0, 0, 0].item()
        assert abs(avg_r - 0.2) < 1e-4, f"temporal avg R: {avg_r}, expected 0.2"
        r.ok("temporal average (3-frame blend)")
    except Exception as e:
        r.fail("temporal average (3-frame blend)", f"{e}\n{traceback.format_exc()}")

    # Test 13: time-based gradient
    try:
        B2 = 5
        img = torch.rand(B2, 2, 2, 3)
        result = compile_and_run(
            "@OUT = vec3(fi / max(fn - 1, 1), 0.5, 1.0);",
            {"A": img},
        )
        for b in range(B2):
            expected = b / 4.0  # fn=5, fn-1=4
            actual = result[b, 0, 0, 0].item()
            assert abs(actual - expected) < 1e-5, f"frame {b}: R={actual}, expected {expected}"
        r.ok("time-based gradient (fi / max(fn-1, 1))")
    except Exception as e:
        r.fail("time-based gradient (fi / max(fn-1, 1))", f"{e}\n{traceback.format_exc()}")

    # ── Type checking ──────────────────────────────────────────────────

    # Test 14: fi and fn recognized as FLOAT
    try:
        code = "@OUT = vec4(fi, fn, 0.0, 1.0);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        checker = TypeChecker(binding_types=bt)
        checker.check(program)
        # No error = fi and fn are valid FLOAT builtins
        r.ok("fi and fn recognized as FLOAT by type checker")
    except Exception as e:
        r.fail("fi and fn recognized as FLOAT by type checker", f"{e}\n{traceback.format_exc()}")

    # Test 15: fetch_frame and sample_frame accept 4 args, return VEC4
    try:
        code = "vec4 a = fetch_frame(@A, 0, ix, iy); vec4 b = sample_frame(@A, 0, u, v); @OUT = a + b;"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        checker = TypeChecker(binding_types=bt)
        checker.check(program)
        r.ok("fetch_frame/sample_frame type-check OK (4 args, VEC4 return)")
    except Exception as e:
        r.fail("fetch_frame/sample_frame type-check OK (4 args, VEC4 return)", f"{e}\n{traceback.format_exc()}")


# ── Vec Array Tests ───────────────────────────────────────────────────

def test_vec_arrays(r: TestResult):
    """Tests for vec3/vec4 element arrays."""
    print("\n--- Vec Array Tests ---")
    B, H, W = 1, 4, 4
    img = torch.rand(B, H, W, 4)

    # 1. vec3 zero-init, correct shape
    try:
        code = """
vec3 arr[3];
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        # arr[0] should be vec3(0,0,0)
        assert result.shape == (B, H, W, 3), f"Expected shape {(B,H,W,3)}, got {result.shape}"
        assert result.abs().max().item() < 1e-6, "Zero-init vec3 should be all zeros"
        r.ok("vec3 array: zero-init shape")
    except Exception as e:
        r.fail("vec3 array: zero-init shape", f"{e}\n{traceback.format_exc()}")

    # 2. vec4 initializer list
    try:
        code = """
vec4 arr[] = {vec4(1.0, 0.0, 0.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0)};
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4  # R=1
        assert abs(result[0, 0, 0, 1].item() - 0.0) < 1e-4  # G=0
        assert abs(result[0, 0, 0, 3].item() - 1.0) < 1e-4  # A=1
        r.ok("vec4 array: initializer list")
    except Exception as e:
        r.fail("vec4 array: initializer list", f"{e}\n{traceback.format_exc()}")

    # 3. Element access returns correct values
    try:
        code = """
vec4 arr[] = {vec4(1.0, 2.0, 3.0, 4.0), vec4(5.0, 6.0, 7.0, 8.0)};
@OUT = arr[1];
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 5.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 6.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 7.0) < 1e-4
        assert abs(result[0, 0, 0, 3].item() - 8.0) < 1e-4
        r.ok("vec4 array: element access")
    except Exception as e:
        r.fail("vec4 array: element access", f"{e}\n{traceback.format_exc()}")

    # 4. Element assignment
    try:
        code = """
vec4 arr[2];
arr[0] = vec4(1.0, 0.0, 0.0, 1.0);
arr[1] = vec4(0.0, 1.0, 0.0, 1.0);
@OUT = arr[0] + arr[1];
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4  # 1+0
        assert abs(result[0, 0, 0, 1].item() - 1.0) < 1e-4  # 0+1
        assert abs(result[0, 0, 0, 2].item() - 0.0) < 1e-4  # 0+0
        assert abs(result[0, 0, 0, 3].item() - 2.0) < 1e-4  # 1+1
        r.ok("vec4 array: element assignment")
    except Exception as e:
        r.fail("vec4 array: element assignment", f"{e}\n{traceback.format_exc()}")

    # 5. sort() on vec4 array — per-channel sort
    try:
        code = """
vec4 arr[] = {vec4(3.0, 1.0, 2.0, 1.0), vec4(1.0, 3.0, 1.0, 3.0), vec4(2.0, 2.0, 3.0, 2.0)};
arr = sort(arr);
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img})
        # Per-channel sort: R sorted = [1,2,3], G sorted = [1,2,3], etc.
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4  # min R
        assert abs(result[0, 0, 0, 1].item() - 1.0) < 1e-4  # min G
        r.ok("vec4 array: sort per-channel")
    except Exception as e:
        r.fail("vec4 array: sort per-channel", f"{e}\n{traceback.format_exc()}")

    # 6. median() on vec4 array — per-channel median
    try:
        code = """
vec4 arr[] = {vec4(1.0, 10.0, 5.0, 1.0), vec4(5.0, 20.0, 3.0, 2.0), vec4(3.0, 15.0, 1.0, 3.0)};
@OUT = median(arr);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4  # median of [1,5,3]
        assert abs(result[0, 0, 0, 1].item() - 15.0) < 1e-4  # median of [10,20,15]
        assert abs(result[0, 0, 0, 2].item() - 3.0) < 1e-4  # median of [5,3,1]
        r.ok("vec4 array: median per-channel")
    except Exception as e:
        r.fail("vec4 array: median per-channel", f"{e}\n{traceback.format_exc()}")

    # 7. arr_sum on vec3 array
    try:
        code = """
vec3 arr[] = {vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)};
vec3 s = arr_sum(arr);
@OUT = s;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert abs(result[0, 0, 0, 0].item() - 5.0) < 1e-4   # 1+4
        assert abs(result[0, 0, 0, 1].item() - 7.0) < 1e-4   # 2+5
        assert abs(result[0, 0, 0, 2].item() - 9.0) < 1e-4   # 3+6
        r.ok("vec3 array: arr_sum")
    except Exception as e:
        r.fail("vec3 array: arr_sum", f"{e}\n{traceback.format_exc()}")

    # 8. arr_avg on vec4 array
    try:
        code = """
vec4 arr[] = {vec4(2.0, 4.0, 6.0, 8.0), vec4(4.0, 6.0, 8.0, 10.0)};
@OUT = arr_avg(arr);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 5.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 7.0) < 1e-4
        assert abs(result[0, 0, 0, 3].item() - 9.0) < 1e-4
        r.ok("vec4 array: arr_avg")
    except Exception as e:
        r.fail("vec4 array: arr_avg", f"{e}\n{traceback.format_exc()}")

    # 9. arr_min / arr_max on vec4 array
    try:
        code = """
vec4 arr[] = {vec4(1.0, 5.0, 3.0, 2.0), vec4(4.0, 2.0, 6.0, 1.0), vec4(2.0, 3.0, 1.0, 4.0)};
vec4 lo = arr_min(arr);
vec4 hi = arr_max(arr);
@OUT = hi - lo;
"""
        result = compile_and_run(code, {"A": img})
        # R: max=4, min=1, diff=3; G: max=5, min=2, diff=3; B: max=6, min=1, diff=5; A: max=4, min=1, diff=3
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 3.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 5.0) < 1e-4
        assert abs(result[0, 0, 0, 3].item() - 3.0) < 1e-4
        r.ok("vec4 array: arr_min/arr_max")
    except Exception as e:
        r.fail("vec4 array: arr_min/arr_max", f"{e}\n{traceback.format_exc()}")

    # 10. reverse on vec3 array
    try:
        code = """
vec3 arr[] = {vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)};
arr = reverse(arr);
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        # reversed: [blue, green, red], arr[0] = blue = (0,0,1)
        assert abs(result[0, 0, 0, 0].item() - 0.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 0.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 1.0) < 1e-4
        r.ok("vec3 array: reverse")
    except Exception as e:
        r.fail("vec3 array: reverse", f"{e}\n{traceback.format_exc()}")

    # 11. len() on vec4 array returns element count (not channels)
    try:
        code = """
vec4 arr[] = {vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0)};
float n = len(arr);
@OUT = vec4(n, 0.0, 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 5.0) < 1e-4
        r.ok("vec4 array: len returns element count")
    except Exception as e:
        r.fail("vec4 array: len returns element count", f"{e}\n{traceback.format_exc()}")

    # 12. Loop populate vec4 array (spatial fetch pattern)
    try:
        code = """
vec4 samples[3];
for (int i = 0; i < 3; i++) {
    samples[i] = fetch(@A, ix + i - 1, iy);
}
@OUT = samples[1];
"""
        result = compile_and_run(code, {"A": img})
        # samples[1] = fetch(@A, ix, iy) = current pixel
        diff = (result - img).abs().max().item()
        assert diff < 1e-5, f"samples[1] should equal current pixel, diff={diff}"
        r.ok("vec4 array: loop populate with fetch")
    except Exception as e:
        r.fail("vec4 array: loop populate with fetch", f"{e}\n{traceback.format_exc()}")


# ── String Array Tests ──────────────────────────────────────────────

def test_string_arrays(r: TestResult):
    """Tests for string element arrays."""
    print("\n--- String Array Tests ---")
    B, H, W = 1, 2, 2
    img = torch.rand(B, H, W, 4)

    # 1. String array zero-init
    try:
        code = """
string arr[3];
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "", f"Expected empty string, got '{result}'"
        r.ok("string array: zero-init empty strings")
    except Exception as e:
        r.fail("string array: zero-init empty strings", f"{e}\n{traceback.format_exc()}")

    # 2. Initializer list
    try:
        code = """
string arr[] = {"alpha", "beta", "gamma"};
@OUT = arr[1];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "beta", f"Expected 'beta', got '{result}'"
        r.ok("string array: initializer list")
    except Exception as e:
        r.fail("string array: initializer list", f"{e}\n{traceback.format_exc()}")

    # 3. Element assignment
    try:
        code = """
string arr[3];
arr[0] = "hello";
arr[1] = "world";
arr[2] = "!";
@OUT = arr[0] + " " + arr[1] + arr[2];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "hello world!", f"Expected 'hello world!', got '{result}'"
        r.ok("string array: element assignment")
    except Exception as e:
        r.fail("string array: element assignment", f"{e}\n{traceback.format_exc()}")

    # 4. sort() — alphabetical
    try:
        code = """
string arr[] = {"cherry", "apple", "banana"};
arr = sort(arr);
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "apple", f"Expected 'apple', got '{result}'"
        r.ok("string array: sort alphabetical")
    except Exception as e:
        r.fail("string array: sort alphabetical", f"{e}\n{traceback.format_exc()}")

    # 5. reverse()
    try:
        code = """
string arr[] = {"a", "b", "c"};
arr = reverse(arr);
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "c", f"Expected 'c', got '{result}'"
        r.ok("string array: reverse")
    except Exception as e:
        r.fail("string array: reverse", f"{e}\n{traceback.format_exc()}")

    # 6. len() on string array
    try:
        code = """
string arr[] = {"x", "y", "z", "w"};
float n = len(arr);
@OUT = vec4(n, 0.0, 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 4.0) < 1e-4
        r.ok("string array: len")
    except Exception as e:
        r.fail("string array: len", f"{e}\n{traceback.format_exc()}")

    # 7. join() — concatenate with separator
    try:
        code = """
string arr[] = {"apple", "banana", "cherry"};
@OUT = join(arr, ", ");
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "apple, banana, cherry", f"Expected 'apple, banana, cherry', got '{result}'"
        r.ok("string array: join")
    except Exception as e:
        r.fail("string array: join", f"{e}\n{traceback.format_exc()}")

    # 8. join() with empty separator
    try:
        code = """
string parts[] = {"pre", "fix", "_", "name"};
@OUT = join(parts, "");
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "prefix_name", f"Expected 'prefix_name', got '{result}'"
        r.ok("string array: join empty separator")
    except Exception as e:
        r.fail("string array: join empty separator", f"{e}\n{traceback.format_exc()}")


# ── Image Reduction Tests ───────────────────────────────────────────

def test_image_reductions(r: TestResult):
    """Tests for img_sum, img_mean, img_min, img_max, img_median."""
    print("\n--- Image Reduction Tests ---")
    B, H, W = 1, 4, 4

    # Create a test image with known values
    # All pixels same value for easy verification
    uniform = torch.full((B, H, W, 4), 0.5)

    # Gradient image: pixel values vary
    gradient = torch.zeros(B, H, W, 4)
    for y in range(H):
        for x in range(W):
            val = (y * W + x) / (H * W - 1)  # 0 to 1
            gradient[0, y, x, :] = val

    # 1. img_mean on uniform image
    try:
        result = compile_and_run("@OUT = img_mean(@A);", {"A": uniform})
        expected = 0.5
        actual = result[0, 0, 0, 0].item()
        assert abs(actual - expected) < 1e-4, f"Expected {expected}, got {actual}"
        r.ok("img_mean: uniform image")
    except Exception as e:
        r.fail("img_mean: uniform image", f"{e}\n{traceback.format_exc()}")

    # 2. img_min on gradient
    try:
        result = compile_and_run("@OUT = img_min(@A);", {"A": gradient})
        actual = result[0, 0, 0, 0].item()
        assert abs(actual - 0.0) < 1e-4, f"Expected 0.0, got {actual}"
        r.ok("img_min: gradient image")
    except Exception as e:
        r.fail("img_min: gradient image", f"{e}\n{traceback.format_exc()}")

    # 3. img_max on gradient
    try:
        result = compile_and_run("@OUT = img_max(@A);", {"A": gradient})
        actual = result[0, 0, 0, 0].item()
        assert abs(actual - 1.0) < 1e-4, f"Expected 1.0, got {actual}"
        r.ok("img_max: gradient image")
    except Exception as e:
        r.fail("img_max: gradient image", f"{e}\n{traceback.format_exc()}")

    # 4. img_sum on known image
    try:
        result = compile_and_run("@OUT = img_sum(@A);", {"A": uniform})
        expected = 0.5 * H * W  # 0.5 * 16 = 8.0
        actual = result[0, 0, 0, 0].item()
        assert abs(actual - expected) < 1e-3, f"Expected {expected}, got {actual}"
        r.ok("img_sum: uniform image")
    except Exception as e:
        r.fail("img_sum: uniform image", f"{e}\n{traceback.format_exc()}")

    # 5. img_median on known values
    try:
        result = compile_and_run("@OUT = img_median(@A);", {"A": gradient})
        # gradient values are 0/15, 1/15, ..., 15/15
        # median of 16 values = (7/15 + 8/15) / 2 = 0.5
        actual = result[0, 0, 0, 0].item()
        assert abs(actual - 0.5) < 0.1, f"Expected ~0.5, got {actual}"
        r.ok("img_median: gradient image")
    except Exception as e:
        r.fail("img_median: gradient image", f"{e}\n{traceback.format_exc()}")

    # 6. Reductions broadcast in expressions: auto-levels pattern
    try:
        code = """
vec4 lo = img_min(@A);
vec4 hi = img_max(@A);
@OUT = (@A - lo) / max(hi - lo, 0.001);
"""
        result = compile_and_run(code, {"A": gradient})
        # Should normalize to [0, 1] range
        actual_min = result.min().item()
        actual_max = result.max().item()
        assert abs(actual_min - 0.0) < 1e-3, f"Expected min ~0.0, got {actual_min}"
        assert abs(actual_max - 1.0) < 1e-3, f"Expected max ~1.0, got {actual_max}"
        r.ok("img reductions: auto-levels broadcast")
    except Exception as e:
        r.fail("img reductions: auto-levels broadcast", f"{e}\n{traceback.format_exc()}")

    # 7. Mask input (FLOAT, 3D tensor)
    try:
        mask = torch.rand(B, H, W)
        result = compile_and_run(
            "@OUT = img_mean(@A);",
            {"A": mask},
            out_type=TEXType.FLOAT,
        )
        expected = mask.mean().item()
        actual = result[0, 0, 0].item()
        assert abs(actual - expected) < 1e-3, f"Expected {expected:.4f}, got {actual:.4f}"
        r.ok("img_mean: mask (FLOAT) input")
    except Exception as e:
        r.fail("img_mean: mask (FLOAT) input", f"{e}\n{traceback.format_exc()}")

    # 8. Single pixel image: reduction == pixel value
    try:
        single_px = torch.tensor([[[[0.25, 0.5, 0.75, 1.0]]]])  # [1,1,1,4]
        result = compile_and_run("@OUT = img_mean(@A);", {"A": single_px})
        assert abs(result[0, 0, 0, 0].item() - 0.25) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 0.50) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 0.75) < 1e-4
        assert abs(result[0, 0, 0, 3].item() - 1.00) < 1e-4
        r.ok("img_mean: single pixel == pixel value")
    except Exception as e:
        r.fail("img_mean: single pixel == pixel value", f"{e}\n{traceback.format_exc()}")

    # 9. Multi-frame reductions (per-frame independent)
    try:
        B2 = 3
        multi = torch.zeros(B2, 2, 2, 4)
        for b in range(B2):
            multi[b, :, :, :] = (b + 1) * 0.1  # frame 0: 0.1, frame 1: 0.2, frame 2: 0.3
        result = compile_and_run("@OUT = img_mean(@A);", {"A": multi})
        for b in range(B2):
            expected = (b + 1) * 0.1
            actual = result[b, 0, 0, 0].item()
            assert abs(actual - expected) < 1e-4, f"frame {b}: expected {expected}, got {actual}"
        r.ok("img_mean: multi-frame per-frame reduction")
    except Exception as e:
        r.fail("img_mean: multi-frame per-frame reduction", f"{e}\n{traceback.format_exc()}")

    # 10. Type checking: img_mean rejects ARRAY input
    try:
        code = "float arr[] = {1.0, 2.0}; float x = img_mean(arr); @OUT = vec4(x);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"OUT": TEXType.VEC4})
        tc.check(program)
        r.fail("img_mean rejects ARRAY", "Expected TypeCheckError")
    except TypeCheckError:
        r.ok("img_mean rejects ARRAY")
    except Exception as e:
        r.fail("img_mean rejects ARRAY", f"{e}\n{traceback.format_exc()}")


# ── Matrix Tests ──────────────────────────────────────────────────────

def test_matrix_types(r: TestResult):
    print("\n--- Matrix Type Tests ---")
    H, W = 4, 4
    img = torch.rand(1, H, W, 4)

    # 1. mat3 constructor (9 args)
    try:
        code = """
mat3 m = mat3(1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0);
@OUT = m * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3]
        assert torch.allclose(result, expected, atol=1e-4), f"Identity mat3*vec3 failed"
        r.ok("mat3 constructor (9 args)")
    except Exception as e:
        r.fail("mat3 constructor (9 args)", f"{e}\n{traceback.format_exc()}")

    # 2. mat3 broadcast constructor (1 arg -> scaled identity)
    try:
        code = """
mat3 m = mat3(2.0);
@OUT = m * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3] * 2.0
        assert torch.allclose(result, expected, atol=1e-4), f"Scaled identity mat3 failed"
        r.ok("mat3 broadcast constructor (scaled identity)")
    except Exception as e:
        r.fail("mat3 broadcast constructor (scaled identity)", f"{e}\n{traceback.format_exc()}")

    # 3. mat4 constructor (16 args)
    try:
        code = """
mat4 m = mat4(1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 1.0);
@OUT = m * @A;
"""
        result = compile_and_run(code, {"A": img})
        assert torch.allclose(result, img, atol=1e-4), f"Identity mat4*vec4 failed"
        r.ok("mat4 constructor (16 args)")
    except Exception as e:
        r.fail("mat4 constructor (16 args)", f"{e}\n{traceback.format_exc()}")

    # 4. mat4 broadcast constructor
    try:
        code = """
mat4 m = mat4(0.5);
@OUT = m * @A;
"""
        result = compile_and_run(code, {"A": img})
        expected = img * 0.5
        assert torch.allclose(result, expected, atol=1e-4), f"Scaled identity mat4 failed"
        r.ok("mat4 broadcast constructor (scaled identity)")
    except Exception as e:
        r.fail("mat4 broadcast constructor (scaled identity)", f"{e}\n{traceback.format_exc()}")

    # 5. mat3 * vec3 (color transform)
    try:
        code = """
mat3 m = mat3(0.0, 1.0, 0.0,
              0.0, 0.0, 1.0,
              1.0, 0.0, 0.0);
@OUT = m * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        # This matrix cycles channels: r->b, g->r, b->g
        expected = torch.stack([img[..., 1], img[..., 2], img[..., 0]], dim=-1)
        assert torch.allclose(result, expected, atol=1e-4), f"Channel cycle mat3*vec3 failed"
        r.ok("mat3 * vec3 (color transform)")
    except Exception as e:
        r.fail("mat3 * vec3 (color transform)", f"{e}\n{traceback.format_exc()}")

    # 6. mat4 * vec4 (homogeneous transform)
    try:
        code = """
mat4 m = mat4(1.0, 0.0, 0.0, 0.1,
              0.0, 1.0, 0.0, 0.2,
              0.0, 0.0, 1.0, 0.3,
              0.0, 0.0, 0.0, 1.0);
@OUT = m * @A;
"""
        result = compile_and_run(code, {"A": img})
        # Translation: r += 0.1*a, g += 0.2*a, b += 0.3*a
        expected = img.clone()
        expected[..., 0] += 0.1 * img[..., 3]
        expected[..., 1] += 0.2 * img[..., 3]
        expected[..., 2] += 0.3 * img[..., 3]
        assert torch.allclose(result, expected, atol=1e-4), f"Translation mat4*vec4 failed"
        r.ok("mat4 * vec4 (homogeneous transform)")
    except Exception as e:
        r.fail("mat4 * vec4 (homogeneous transform)", f"{e}\n{traceback.format_exc()}")

    # 7. mat3 * mat3 (chain two transforms)
    try:
        code = """
mat3 a = mat3(2.0);
mat3 b = mat3(3.0);
mat3 c = a * b;
@OUT = c * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3] * 6.0  # 2*I * 3*I = 6*I
        assert torch.allclose(result, expected, atol=1e-4), f"mat3*mat3 chain failed"
        r.ok("mat3 * mat3 (chain transforms)")
    except Exception as e:
        r.fail("mat3 * mat3 (chain transforms)", f"{e}\n{traceback.format_exc()}")

    # 8. mat4 * mat4
    try:
        code = """
mat4 a = mat4(2.0);
mat4 b = mat4(0.5);
mat4 c = a * b;
@OUT = c * @A;
"""
        result = compile_and_run(code, {"A": img})
        expected = img.clone()  # 2*I * 0.5*I = I
        assert torch.allclose(result, expected, atol=1e-4), f"mat4*mat4 failed"
        r.ok("mat4 * mat4")
    except Exception as e:
        r.fail("mat4 * mat4", f"{e}\n{traceback.format_exc()}")

    # 9. scalar * mat3 (element-wise scale)
    try:
        code = """
mat3 m = mat3(1.0, 2.0, 3.0,
              4.0, 5.0, 6.0,
              7.0, 8.0, 9.0);
mat3 s = 0.5 * m;
@OUT = s * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        # s = 0.5 * m, then s * vec3
        m = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32) * 0.5
        rgb = img[..., :3]
        expected = torch.matmul(rgb.unsqueeze(-2), m.T).squeeze(-2)
        # Actually mat * vec = matmul(m, v), so expected = matmul(m, v.unsqueeze(-1)).squeeze(-1)
        expected = torch.matmul(m, rgb.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(result, expected, atol=1e-4), f"scalar*mat3 failed"
        r.ok("scalar * mat3 (element-wise scale)")
    except Exception as e:
        r.fail("scalar * mat3 (element-wise scale)", f"{e}\n{traceback.format_exc()}")

    # 10. mat3 * scalar
    try:
        code = """
mat3 m = mat3(1.0);
mat3 s = m * 3.0;
@OUT = s * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3] * 3.0
        assert torch.allclose(result, expected, atol=1e-4)
        r.ok("mat3 * scalar")
    except Exception as e:
        r.fail("mat3 * scalar", f"{e}\n{traceback.format_exc()}")

    # 11. mat3 + mat3 (element-wise add)
    try:
        code = """
mat3 a = mat3(1.0);
mat3 b = mat3(2.0);
mat3 c = a + b;
@OUT = c * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3] * 3.0  # I + 2I = 3I
        assert torch.allclose(result, expected, atol=1e-4)
        r.ok("mat3 + mat3 (element-wise add)")
    except Exception as e:
        r.fail("mat3 + mat3 (element-wise add)", f"{e}\n{traceback.format_exc()}")

    # 12. mat3 - mat3
    try:
        code = """
mat3 a = mat3(3.0);
mat3 b = mat3(1.0);
mat3 c = a - b;
@OUT = c * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3] * 2.0  # 3I - I = 2I
        assert torch.allclose(result, expected, atol=1e-4)
        r.ok("mat3 - mat3 (element-wise subtract)")
    except Exception as e:
        r.fail("mat3 - mat3 (element-wise subtract)", f"{e}\n{traceback.format_exc()}")

    # 13. transpose(mat3)
    try:
        code = """
mat3 m = mat3(1.0, 2.0, 3.0,
              4.0, 5.0, 6.0,
              7.0, 8.0, 9.0);
mat3 t = transpose(m);
@OUT = t * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        m_T = torch.tensor([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype=torch.float32)
        rgb = img[..., :3]
        expected = torch.matmul(m_T, rgb.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(result, expected, atol=1e-4)
        r.ok("transpose(mat3)")
    except Exception as e:
        r.fail("transpose(mat3)", f"{e}\n{traceback.format_exc()}")

    # 14. determinant(mat3) — identity -> 1.0
    try:
        code = """
float d = determinant(mat3(1.0));
@OUT = vec4(d, d, d, 1.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4, f"det(I) should be 1.0"
        r.ok("determinant(mat3) = 1.0 for identity")
    except Exception as e:
        r.fail("determinant(mat3) = 1.0 for identity", f"{e}\n{traceback.format_exc()}")

    # 15. inverse(mat3) — identity -> identity
    try:
        code = """
mat3 m = mat3(1.0);
mat3 inv = inverse(m);
@OUT = inv * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3]
        assert torch.allclose(result, expected, atol=1e-4)
        r.ok("inverse(mat3) identity -> identity")
    except Exception as e:
        r.fail("inverse(mat3) identity -> identity", f"{e}\n{traceback.format_exc()}")

    # 16. inverse: m * inverse(m) ~= identity
    try:
        code = """
mat3 m = mat3(2.0, 1.0, 0.0,
              0.0, 3.0, 1.0,
              1.0, 0.0, 2.0);
mat3 inv = inverse(m);
mat3 prod = m * inv;
@OUT = prod * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3]
        assert torch.allclose(result, expected, atol=1e-3), "m * inverse(m) should ~= identity"
        r.ok("m * inverse(m) ~= identity")
    except Exception as e:
        r.fail("m * inverse(m) ~= identity", f"{e}\n{traceback.format_exc()}")

    # 17. vec3 * mat3 -> type error
    try:
        code = "@OUT = @A.rgb * mat3(1.0);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"A": TEXType.VEC4, "OUT": TEXType.VEC3})
        tc.check(program)
        r.fail("vec * mat type error", "Expected TypeCheckError")
    except TypeCheckError as e:
        assert "transpose" in str(e).lower() or "cannot" in str(e).lower(), f"Unexpected error: {e}"
        r.ok("vec * mat -> type error")
    except Exception as e:
        r.fail("vec * mat type error", f"{e}\n{traceback.format_exc()}")

    # 18. mat3 channel access -> type error
    try:
        code = "mat3 m = mat3(1.0); float x = m.r; @OUT = vec4(x);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"OUT": TEXType.VEC4})
        tc.check(program)
        r.fail("mat3 channel access error", "Expected TypeCheckError")
    except TypeCheckError:
        r.ok("mat3 channel access -> type error")
    except Exception as e:
        r.fail("mat3 channel access error", f"{e}\n{traceback.format_exc()}")

    # 19. mat3 as @OUT -> type error
    try:
        code = "mat3 m = mat3(1.0); @OUT = m;"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"A": TEXType.VEC4, "OUT": TEXType.VEC4})
        tc.check(program)
        r.fail("mat3 as @OUT error", "Expected TypeCheckError")
    except TypeCheckError:
        r.ok("mat3 as @OUT -> type error")
    except Exception as e:
        r.fail("mat3 as @OUT error", f"{e}\n{traceback.format_exc()}")

    # 20. ACES color transform roundtrip (sRGB -> XYZ -> sRGB)
    try:
        code = """
// sRGB to XYZ (D65)
mat3 srgb_to_xyz = mat3(
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041
);
// XYZ to sRGB (D65) — inverse of above
mat3 xyz_to_srgb = inverse(srgb_to_xyz);
// Roundtrip
vec3 xyz = srgb_to_xyz * @A.rgb;
@OUT = xyz_to_srgb * xyz;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3]
        assert torch.allclose(result, expected, atol=1e-3), "sRGB->XYZ->sRGB roundtrip failed"
        r.ok("ACES: sRGB -> XYZ -> sRGB roundtrip")
    except Exception as e:
        r.fail("ACES: sRGB -> XYZ -> sRGB roundtrip", f"{e}\n{traceback.format_exc()}")


def test_matrix_benchmarks(r: TestResult):
    print("\n--- Matrix Benchmark Tests ---")
    H, W = 512, 512
    img3 = torch.rand(1, H, W, 3)
    img4 = torch.rand(1, H, W, 4)

    # 1. mat3 * vec3 at 512×512
    try:
        code = """
mat3 m = mat3(0.4124564, 0.3575761, 0.1804375,
              0.2126729, 0.7151522, 0.0721750,
              0.0193339, 0.1191920, 0.9503041);
@OUT = m * @A.rgb;
"""
        # Warmup
        compile_and_run(code, {"A": img3}, out_type=TEXType.VEC3)
        # Benchmark
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            compile_and_run(code, {"A": img3}, out_type=TEXType.VEC3)
        elapsed = (time.perf_counter() - t0) / N * 1000
        r.ok(f"mat3 * vec3 @ 512x512: {elapsed:.1f}ms")
    except Exception as e:
        r.fail("mat3 * vec3 benchmark", f"{e}\n{traceback.format_exc()}")

    # 2. mat4 * vec4 at 512×512
    try:
        code = """
mat4 m = mat4(1.0, 0.0, 0.0, 0.1,
              0.0, 1.0, 0.0, 0.2,
              0.0, 0.0, 1.0, 0.3,
              0.0, 0.0, 0.0, 1.0);
@OUT = m * @A;
"""
        compile_and_run(code, {"A": img4})
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            compile_and_run(code, {"A": img4})
        elapsed = (time.perf_counter() - t0) / N * 1000
        r.ok(f"mat4 * vec4 @ 512x512: {elapsed:.1f}ms")
    except Exception as e:
        r.fail("mat4 * vec4 benchmark", f"{e}\n{traceback.format_exc()}")

    # 3. chained mat3 * mat3 * vec3 at 512×512
    try:
        code = """
mat3 a = mat3(0.4124564, 0.3575761, 0.1804375,
              0.2126729, 0.7151522, 0.0721750,
              0.0193339, 0.1191920, 0.9503041);
mat3 b = inverse(a);
mat3 c = a * b;
@OUT = c * @A.rgb;
"""
        compile_and_run(code, {"A": img3}, out_type=TEXType.VEC3)
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            compile_and_run(code, {"A": img3}, out_type=TEXType.VEC3)
        elapsed = (time.perf_counter() - t0) / N * 1000
        r.ok(f"chained mat3*mat3*vec3 @ 512x512: {elapsed:.1f}ms")
    except Exception as e:
        r.fail("chained mat3*mat3*vec3 benchmark", f"{e}\n{traceback.format_exc()}")

    # 4. inverse(mat3) at 512×512
    try:
        code = """
mat3 m = mat3(2.0, 1.0, 0.0,
              0.0, 3.0, 1.0,
              1.0, 0.0, 2.0);
mat3 inv = inverse(m);
@OUT = inv * @A.rgb;
"""
        compile_and_run(code, {"A": img3}, out_type=TEXType.VEC3)
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            compile_and_run(code, {"A": img3}, out_type=TEXType.VEC3)
        elapsed = (time.perf_counter() - t0) / N * 1000
        r.ok(f"inverse(mat3) @ 512x512: {elapsed:.1f}ms")
    except Exception as e:
        r.fail("inverse(mat3) benchmark", f"{e}\n{traceback.format_exc()}")


# ── v0.3 Multi-Output, Parameters, Typed Bindings Tests ──────────────

def test_v03_features(r: TestResult):
    """Tests for v0.3: multi-output, $ parameters, typed bindings."""
    print("\n--- v0.3 Feature Tests ---")

    from TEX_Wrangle.tex_compiler.ast_nodes import BindingRef, ParamDecl

    # ── Lexer: $ binding ──
    try:
        tokens = Lexer("$strength").tokenize()
        tok = tokens[0]
        assert tok.type == TokenType.DOLLAR_BINDING, f"Expected DOLLAR_BINDING, got {tok.type}"
        assert tok.value == "strength"
        r.ok("lexer: $ binding")
    except Exception as e:
        r.fail("lexer: $ binding", f"{e}\n{traceback.format_exc()}")

    # ── Lexer: typed @ binding ──
    try:
        tokens = Lexer("f@threshold").tokenize()
        tok = tokens[0]
        assert tok.type == TokenType.TYPED_AT_BINDING, f"Expected TYPED_AT_BINDING, got {tok.type}"
        assert tok.value == "threshold"
        assert tok.prefix == "f"
        r.ok("lexer: typed @ binding (f@threshold)")
    except Exception as e:
        r.fail("lexer: typed @ binding", f"{e}\n{traceback.format_exc()}")

    # ── Lexer: typed $ binding ──
    try:
        tokens = Lexer("i$count").tokenize()
        tok = tokens[0]
        assert tok.type == TokenType.TYPED_DOLLAR_BINDING, f"Expected TYPED_DOLLAR_BINDING, got {tok.type}"
        assert tok.value == "count"
        assert tok.prefix == "i"
        r.ok("lexer: typed $ binding (i$count)")
    except Exception as e:
        r.fail("lexer: typed $ binding", f"{e}\n{traceback.format_exc()}")

    # ── Lexer: prefix vs identifier (f = 5 is NOT a typed binding) ──
    try:
        tokens = Lexer("f = 5;").tokenize()
        tok = tokens[0]
        assert tok.type == TokenType.IDENT, f"Expected IDENT, got {tok.type}"
        assert tok.value == "f"
        r.ok("lexer: prefix vs identifier")
    except Exception as e:
        r.fail("lexer: prefix vs identifier", f"{e}\n{traceback.format_exc()}")

    # ── Lexer: all type prefixes ──
    try:
        for prefix in ["f", "i", "v", "v4", "s", "img", "m", "l"]:
            tokens = Lexer(f"{prefix}@test").tokenize()
            tok = tokens[0]
            assert tok.type == TokenType.TYPED_AT_BINDING, f"prefix '{prefix}': Expected TYPED_AT_BINDING, got {tok.type}"
            assert tok.prefix == prefix, f"prefix '{prefix}': got prefix {tok.prefix!r}"
            assert tok.value == "test"
        r.ok("lexer: all type prefixes")
    except Exception as e:
        r.fail("lexer: all type prefixes", f"{e}\n{traceback.format_exc()}")

    # ── Parser: ParamDecl with default ──
    try:
        tokens = Lexer("f$strength = 0.5;").tokenize()
        program = Parser(tokens).parse()
        stmt = program.statements[0]
        assert isinstance(stmt, ParamDecl), f"Expected ParamDecl, got {type(stmt)}"
        assert stmt.name == "strength"
        assert stmt.type_hint == "f"
        assert stmt.default_expr is not None
        r.ok("parser: ParamDecl with default")
    except Exception as e:
        r.fail("parser: ParamDecl with default", f"{e}\n{traceback.format_exc()}")

    # ── Parser: ParamDecl no default ──
    try:
        tokens = Lexer("i$count;").tokenize()
        program = Parser(tokens).parse()
        stmt = program.statements[0]
        assert isinstance(stmt, ParamDecl), f"Expected ParamDecl, got {type(stmt)}"
        assert stmt.name == "count"
        assert stmt.type_hint == "i"
        assert stmt.default_expr is None
        r.ok("parser: ParamDecl no default")
    except Exception as e:
        r.fail("parser: ParamDecl no default", f"{e}\n{traceback.format_exc()}")

    # ── Parser: $ in expression ──
    try:
        tokens = Lexer("@OUT = @A * $strength;").tokenize()
        program = Parser(tokens).parse()
        # Should parse without error; the assignment RHS contains a $ binding ref
        r.ok("parser: $ in expression")
    except Exception as e:
        r.fail("parser: $ in expression", f"{e}\n{traceback.format_exc()}")

    # ── Parser: typed @ in expression ──
    try:
        tokens = Lexer("img@result = @A * 0.5;").tokenize()
        program = Parser(tokens).parse()
        r.ok("parser: typed @ assignment")
    except Exception as e:
        r.fail("parser: typed @ assignment", f"{e}\n{traceback.format_exc()}")

    # ── Type checker: multi-output ──
    try:
        code = "@result = @A * 0.5;\n@mask = luma(@A);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        img = torch.rand(1, 4, 4, 3)
        checker = TypeChecker(binding_types={"A": TEXType.VEC3})
        checker.check(program)
        assert "result" in checker.assigned_bindings, "Missing 'result' in assigned_bindings"
        assert "mask" in checker.assigned_bindings, "Missing 'mask' in assigned_bindings"
        assert checker.assigned_bindings["result"] == TEXType.VEC3
        assert checker.assigned_bindings["mask"] == TEXType.FLOAT
        r.ok("type checker: multi-output")
    except Exception as e:
        r.fail("type checker: multi-output", f"{e}\n{traceback.format_exc()}")

    # ── Type checker: output type inference ──
    try:
        code = "@OUT = luma(@A);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        checker = TypeChecker(binding_types={"A": TEXType.VEC3})
        checker.check(program)
        assert checker.assigned_bindings["OUT"] == TEXType.FLOAT
        # Backward compat property
        assert checker.inferred_out_type == TEXType.FLOAT
        r.ok("type checker: output type inference")
    except Exception as e:
        r.fail("type checker: output type inference", f"{e}\n{traceback.format_exc()}")

    # ── Type checker: param declaration ──
    try:
        code = "f$strength = 0.5;\n@OUT = @A * $strength;"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        checker = TypeChecker(binding_types={"A": TEXType.VEC3, "strength": TEXType.FLOAT})
        checker.check(program)
        assert "strength" in checker.param_declarations
        assert checker.param_declarations["strength"]["type"] == TEXType.FLOAT
        r.ok("type checker: param declaration")
    except Exception as e:
        r.fail("type checker: param declaration", f"{e}\n{traceback.format_exc()}")

    # ── Type checker: param type mismatch ──
    try:
        code = 'f$x = "hello";'
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        checker = TypeChecker(binding_types={})
        try:
            checker.check(program)
            r.fail("type checker: param type mismatch", "Expected TypeCheckError")
        except TypeCheckError:
            r.ok("type checker: param type mismatch")
    except Exception as e:
        r.fail("type checker: param type mismatch", f"{e}\n{traceback.format_exc()}")

    # ── Interpreter: multi-output ──
    try:
        code = "@result = @A * 0.5;\n@mask = luma(@A);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        img = torch.rand(1, 4, 4, 3)
        checker = TypeChecker(binding_types={"A": TEXType.VEC3})
        type_map = checker.check(program)
        interp = Interpreter()
        out = interp.execute(program, {"A": img}, type_map, device="cpu",
                             output_names=["mask", "result"])
        assert isinstance(out, dict), f"Expected dict, got {type(out)}"
        assert "result" in out, "Missing 'result'"
        assert "mask" in out, "Missing 'mask'"
        assert out["result"].shape[-1] == 3, f"result should be vec3, got shape {out['result'].shape}"
        assert out["mask"].dim() == 3, f"mask should be 3D (float), got dim {out['mask'].dim()}"
        r.ok("interpreter: multi-output")
    except Exception as e:
        r.fail("interpreter: multi-output", f"{e}\n{traceback.format_exc()}")

    # ── Interpreter: param as binding ──
    try:
        code = "f$strength = 0.5;\n@OUT = @A * $strength;"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        img = torch.ones(1, 2, 2, 3)
        checker = TypeChecker(binding_types={"A": TEXType.VEC3, "strength": TEXType.FLOAT})
        type_map = checker.check(program)
        interp = Interpreter()
        out = interp.execute(program, {"A": img, "strength": torch.tensor(0.5)}, type_map,
                             device="cpu")
        assert torch.allclose(out, torch.full_like(img, 0.5), atol=1e-5), f"Expected 0.5, got {out.mean().item()}"
        r.ok("interpreter: param as binding")
    except Exception as e:
        r.fail("interpreter: param as binding", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: multi-output tuple ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        node = TEXWrangleNode()
        img = torch.rand(1, 4, 4, 3)
        result = node.execute(
            "@result = @A * 0.5;\n@mask = luma(@A);",
            A=img, device="cpu", compile_mode="none"
        )
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 8, f"Expected 8 slots, got {len(result)}"
        # Alphabetical order: "mask" at 0, "result" at 1
        assert result[0] is not None, "slot 0 (mask) should not be None"
        assert result[1] is not None, "slot 1 (result) should not be None"
        assert result[2] is None, "slot 2 should be None (unused)"
        r.ok("tex_node: multi-output tuple")
    except Exception as e:
        r.fail("tex_node: multi-output tuple", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: backward compat (single @OUT) ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        node = TEXWrangleNode()
        img = torch.rand(1, 4, 4, 3)
        result = node.execute("@OUT = @A * 0.5;", A=img, device="cpu", compile_mode="none")
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert result[0] is not None, "slot 0 (OUT) should not be None"
        assert result[1] is None, "slot 1 should be None"
        r.ok("tex_node: backward compat (@OUT)")
    except Exception as e:
        r.fail("tex_node: backward compat (@OUT)", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: old output_type ignored ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        node = TEXWrangleNode()
        img = torch.rand(1, 4, 4, 3)
        # Passing output_type="IMAGE" — should be silently ignored
        result = node.execute("@OUT = @A * 0.5;", A=img, device="cpu",
                              compile_mode="none", output_type="IMAGE")
        assert result[0] is not None
        r.ok("tex_node: old output_type ignored")
    except Exception as e:
        r.fail("tex_node: old output_type ignored", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: param widgets flow through kwargs ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        node = TEXWrangleNode()
        img = torch.ones(1, 2, 2, 3)
        # $strength value comes as a kwarg (simulating widget value from ComfyUI)
        result = node.execute(
            "f$strength = 0.5;\n@OUT = @A * $strength;",
            A=img, strength=torch.tensor(0.75), device="cpu", compile_mode="none"
        )
        out = result[0]  # OUT is at slot 0 (only output)
        assert out is not None, "OUT should not be None"
        # $strength = 0.75 (widget overrides declaration default)
        assert abs(out.mean().item() - 0.75) < 0.01, f"Expected ~0.75, got {out.mean().item()}"
        r.ok("tex_node: param kwargs")
    except Exception as e:
        r.fail("tex_node: param kwargs", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: param default fallback ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        node = TEXWrangleNode()
        img = torch.ones(1, 2, 2, 3)
        # No strength kwarg — should use default from code (0.5)
        result = node.execute(
            "f$strength = 0.5;\n@OUT = @A * $strength;",
            A=img, device="cpu", compile_mode="none"
        )
        out = result[0]
        assert out is not None, "OUT should not be None"
        assert abs(out.mean().item() - 0.5) < 0.01, f"Expected ~0.5, got {out.mean().item()}"
        r.ok("tex_node: param default fallback")
    except Exception as e:
        r.fail("tex_node: param default fallback", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: param widget value as kwarg ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        node = TEXWrangleNode()
        img = torch.ones(1, 2, 2, 3)
        # Widget value injected as kwarg by graphToPrompt hook (overrides code default)
        result = node.execute(
            "f$strength = 0.5;\n@OUT = @A * $strength;",
            A=img, strength=0.3,
            device="cpu", compile_mode="none"
        )
        out = result[0]
        assert out is not None, "OUT should not be None"
        assert abs(out.mean().item() - 0.3) < 0.01, f"Expected ~0.3, got {out.mean().item()}"
        r.ok("tex_node: param widget value as kwarg")
    except Exception as e:
        r.fail("tex_node: param widget value as kwarg", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: param widget overrides code default ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        node = TEXWrangleNode()
        img = torch.ones(1, 2, 2, 3)
        # Widget value (scalar kwarg) overrides code default of 0.5
        result = node.execute(
            "f$strength = 0.5;\n@OUT = @A * $strength;",
            A=img, strength=0.8,
            device="cpu", compile_mode="none"
        )
        out = result[0]
        assert out is not None, "OUT should not be None"
        assert abs(out.mean().item() - 0.8) < 0.01, f"Expected ~0.8 (widget), got {out.mean().item()}"
        r.ok("tex_node: param widget overrides code default")
    except Exception as e:
        r.fail("tex_node: param widget overrides code default", f"{e}\n{traceback.format_exc()}")

    # ── type checker: param default_value extraction ──
    try:
        cache = TEXCache(cache_dir=Path(tempfile.mkdtemp()))
        code = "f$strength = 0.005;\ni$count = 3;\ns$label = \"hello\";\n@OUT = @A * $strength;"
        bt = {"A": TEXType.VEC3}
        program, type_map, refs, assigned, params = cache.compile_tex(code, bt)
        assert "strength" in params, "Missing 'strength' in params"
        assert params["strength"]["default_value"] == 0.005, f"Expected 0.005, got {params['strength']['default_value']}"
        assert params["strength"]["type"] == TEXType.FLOAT
        assert "count" in params, "Missing 'count' in params"
        assert params["count"]["default_value"] == 3, f"Expected 3, got {params['count']['default_value']}"
        assert params["count"]["type"] == TEXType.INT
        assert "label" in params, "Missing 'label' in params"
        assert params["label"]["default_value"] == "hello", f"Expected 'hello', got {params['label']['default_value']}"
        assert params["label"]["type"] == TEXType.STRING
        r.ok("type checker: param default_value extraction")
    except Exception as e:
        r.fail("type checker: param default_value extraction", f"{e}\n{traceback.format_exc()}")

    # ── cache: multi-output compile_tex ──
    try:
        cache = TEXCache(cache_dir=Path(tempfile.mkdtemp()))
        code = "@result = @A * 0.5;\n@mask = luma(@A);"
        bt = {"A": TEXType.VEC3}
        program, type_map, refs, assigned, params = cache.compile_tex(code, bt)
        assert "result" in assigned, "Missing 'result' in assigned_bindings"
        assert "mask" in assigned, "Missing 'mask' in assigned_bindings"
        assert assigned["result"] == TEXType.VEC3
        assert assigned["mask"] == TEXType.FLOAT
        r.ok("cache: multi-output compile_tex")
    except Exception as e:
        r.fail("cache: multi-output compile_tex", f"{e}\n{traceback.format_exc()}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TEX Test Suite")
    print("=" * 60)

    r = TestResult()

    test_lexer(r)
    test_lexer_v11(r)
    test_parser(r)
    test_parser_v11(r)
    test_type_checker(r)
    test_interpreter(r)
    test_for_loops(r)
    test_compound_assignments(r)
    test_examples(r)
    test_cache(r)
    test_device_selection(r)
    test_torch_compile(r)
    test_sampling(r)
    test_noise(r)
    test_channel_assignment(r)
    test_output_types(r)
    test_error_paths(r)
    test_if_without_else(r)
    test_stdlib_coverage(r)
    test_stdlib_extended(r)
    test_numerical_edge_cases(r)
    test_swizzle_patterns(r)
    test_performance(r)
    test_cache_eviction(r)
    test_latent(r)
    test_string(r)
    test_named_bindings(r)
    test_arrays(r)
    test_auto_inference(r)
    test_batch_temporal(r)
    test_vec_arrays(r)
    test_string_arrays(r)
    test_image_reductions(r)
    test_matrix_types(r)
    test_matrix_benchmarks(r)
    test_v03_features(r)

    success = r.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
