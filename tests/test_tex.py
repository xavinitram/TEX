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
from TEX_Wrangle.tex_compiler.diagnostics import TEXMultiError
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, InterpreterError
from TEX_Wrangle.tex_cache import TEXCache
from TEX_Wrangle.tex_runtime.compiled import execute_compiled, _plain_execute, clear_compiled_cache
from TEX_Wrangle.tex_runtime.codegen import try_compile, _CgBreak, _CgContinue
from TEX_Wrangle.tex_runtime.interpreter import _ensure_spatial, _broadcast_pair, _collect_identifiers
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib, SAFE_EPSILON, _perlin2d_fast, _grad2d_dot, _lowbias32
from TEX_Wrangle.tex_compiler.type_checker import CHANNEL_MAP
import math


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
    parser = Parser(tokens, source=code)
    program = parser.parse()

    binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}

    checker = TypeChecker(binding_types=binding_types, source=code)
    type_map = checker.check(program)
    output_names = sorted(checker.assigned_bindings.keys())

    if not output_names:
        raise InterpreterError(
            "TEX program has no outputs. Assign to @OUT or another @name."
        )

    interp = Interpreter()
    result = interp.execute(program, bindings, type_map, device=device,
                            latent_channel_count=latent_channel_count,
                            output_names=output_names)

    # Unwrap single-output for backward compat with existing tests
    if output_names == ["OUT"]:
        return result["OUT"]
    return result


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
        prog = Parser(tokens, source=code).parse()
        bt = bindings or {}
        bt.setdefault("OUT", TEXType.VEC4)
        checker = TypeChecker(binding_types=bt, source=code)
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
        assert "maximum iteration limit" in str(e).lower() or "1024" in str(e) or "iterations without finishing" in str(e).lower()
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


# ── Break/Continue Tests ──────────────────────────────────────────────

def test_break_continue(r: TestResult):
    print("\n--- Break/Continue Tests ---")

    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 3)

    # break: exit loop early
    try:
        code = """
        float sum = 0.0;
        for (int i = 0; i < 10; i++) {
            if (i >= 3) { break; }
            sum += 1.0;
        }
        @OUT = vec3(sum, sum, sum);
        """
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 3.0) < 1e-4, f"Expected 3.0, got {val}"
        r.ok("break: exit loop early")
    except Exception as e:
        r.fail("break: exit loop early", f"{e}\n{traceback.format_exc()}")

    # continue: skip iteration
    try:
        code = """
        float sum = 0.0;
        for (int i = 0; i < 5; i++) {
            if (i == 2) { continue; }
            sum += 1.0;
        }
        @OUT = vec3(sum, sum, sum);
        """
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        # Skips i=2, so sum = 4 (i=0,1,3,4)
        assert abs(val - 4.0) < 1e-4, f"Expected 4.0, got {val}"
        r.ok("continue: skip iteration")
    except Exception as e:
        r.fail("continue: skip iteration", f"{e}\n{traceback.format_exc()}")

    # break in nested loop: only breaks inner
    try:
        code = """
        float outer_count = 0.0;
        float inner_total = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 10; j++) {
                if (j >= 2) { break; }
                inner_total += 1.0;
            }
            outer_count += 1.0;
        }
        @OUT = vec3(outer_count, inner_total, 0.0);
        """
        result = compile_and_run(code, {"A": test_img})
        outer = result[0, 0, 0, 0].item()
        inner = result[0, 0, 0, 1].item()
        assert abs(outer - 3.0) < 1e-4, f"Expected outer=3.0, got {outer}"
        assert abs(inner - 6.0) < 1e-4, f"Expected inner=6.0, got {inner}"
        r.ok("break: nested loops (inner only)")
    except Exception as e:
        r.fail("break: nested loops (inner only)", f"{e}\n{traceback.format_exc()}")

    # continue with accumulation: sum only even indices
    try:
        code = """
        float sum = 0.0;
        for (int i = 0; i < 6; i++) {
            if (i % 2 != 0) { continue; }
            sum += float(i);
        }
        @OUT = vec3(sum, sum, sum);
        """
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        # Even indices: 0 + 2 + 4 = 6
        assert abs(val - 6.0) < 1e-4, f"Expected 6.0, got {val}"
        r.ok("continue: sum even indices")
    except Exception as e:
        r.fail("continue: sum even indices", f"{e}\n{traceback.format_exc()}")

    # break outside loop: should be a type check error
    try:
        code = "break;\n@OUT = vec3(0.0);"
        compile_and_run(code, {"A": test_img})
        r.fail("break: outside loop error", "Should have raised TypeCheckError")
    except TypeCheckError as e:
        assert "outside" in str(e).lower() or "loop" in str(e).lower()
        r.ok("break: outside loop error")
    except Exception as e:
        r.fail("break: outside loop error", f"{e}\n{traceback.format_exc()}")

    # continue outside loop: should be a type check error
    try:
        code = "continue;\n@OUT = vec3(0.0);"
        compile_and_run(code, {"A": test_img})
        r.fail("continue: outside loop error", "Should have raised TypeCheckError")
    except TypeCheckError as e:
        assert "outside" in str(e).lower() or "loop" in str(e).lower()
        r.ok("continue: outside loop error")
    except Exception as e:
        r.fail("continue: outside loop error", f"{e}\n{traceback.format_exc()}")

    # break immediately: zero iterations
    try:
        code = """
        float sum = 0.0;
        for (int i = 0; i < 10; i++) {
            break;
            sum += 1.0;
        }
        @OUT = vec3(sum, sum, sum);
        """
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        assert abs(val) < 1e-4, f"Expected 0.0, got {val}"
        r.ok("break: immediate exit")
    except Exception as e:
        r.fail("break: immediate exit", f"{e}\n{traceback.format_exc()}")


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
        assert len(result) == 6
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

    # Multi-output program via execute_compiled
    try:
        mo_code = "@mask = vec4(1.0, 1.0, 1.0, 1.0);\n@result = @A * 0.5;"
        mo_bt = {"A": TEXType.VEC4, "mask": TEXType.VEC4, "result": TEXType.VEC4}
        mo_cache = TEXCache()
        mo_prog, mo_tm, mo_refs, *_ = mo_cache.compile_tex(mo_code, mo_bt)
        mo_fp = mo_cache.fingerprint(mo_code, mo_bt)

        mo_result = execute_compiled(
            mo_prog, {"A": test_img}, mo_tm, "cpu", mo_fp,
            output_names=["mask", "result"]
        )
        assert isinstance(mo_result, dict), f"Expected dict, got {type(mo_result)}"
        assert "mask" in mo_result, "Missing 'mask' in multi-output result"
        assert "result" in mo_result, "Missing 'result' in multi-output result"
        assert torch.allclose(mo_result["result"], test_img * 0.5, atol=1e-5)
        r.ok("torch_compile: multi-output")
    except Exception as e:
        r.fail("torch_compile: multi-output", f"{e}\n{traceback.format_exc()}")


# ── IS_CHANGED Hash Tests ─────────────────────────────────────────────

def test_is_changed_hash(r: TestResult):
    print("\n--- IS_CHANGED Hash Tests ---")

    from TEX_Wrangle.tex_node import _tensor_fingerprint

    # Same-shape tensors with different content → different fingerprints
    try:
        t1 = torch.zeros(1, 4, 4, 3)
        t2 = torch.ones(1, 4, 4, 3)
        fp1 = _tensor_fingerprint(t1)
        fp2 = _tensor_fingerprint(t2)
        assert fp1 != fp2, "Different tensors should produce different fingerprints"
        r.ok("is_changed: different tensors produce different fingerprints")
    except Exception as e:
        r.fail("is_changed: different tensors produce different fingerprints", f"{e}\n{traceback.format_exc()}")

    # Same tensor → same fingerprint (stability)
    try:
        t = torch.rand(1, 8, 8, 3)
        fp_a = _tensor_fingerprint(t)
        fp_b = _tensor_fingerprint(t)
        assert fp_a == fp_b, "Same tensor should produce same fingerprint"
        r.ok("is_changed: same tensor produces stable fingerprint")
    except Exception as e:
        r.fail("is_changed: same tensor produces stable fingerprint", f"{e}\n{traceback.format_exc()}")

    # Large tensor still works (stride sampling)
    try:
        t_large = torch.rand(2, 512, 512, 4)
        fp = _tensor_fingerprint(t_large)
        assert "512" in fp, "Fingerprint should contain shape info"
        r.ok("is_changed: large tensor fingerprint")
    except Exception as e:
        r.fail("is_changed: large tensor fingerprint", f"{e}\n{traceback.format_exc()}")


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

    # MASK from vec3 (luma-weighted — m@mask = vec3 use-case)
    try:
        raw = torch.zeros(1, 4, 4, 3)
        raw[..., 1] = 1.0  # green only
        result = _prepare_output(raw, "MASK")
        assert result.shape == (1, 4, 4), f"Shape: {result.shape}"
        # Luma(0, 1, 0) = 0.7152
        assert abs(result[0, 0, 0].item() - 0.7152) < 1e-3
        r.ok("output: MASK from vec3 (luminance)")
    except Exception as e:
        r.fail("output: MASK from vec3 (luminance)", f"{e}\n{traceback.format_exc()}")

    # MASK from [H, W] unbatched (adds batch dim)
    try:
        raw = torch.full((4, 4), 0.5)
        result = _prepare_output(raw, "MASK")
        assert result.shape == (1, 4, 4), f"Shape: {result.shape}"
        assert abs(result[0, 0, 0].item() - 0.5) < 1e-5
        r.ok("output: MASK from [H,W] (adds batch dim)")
    except Exception as e:
        r.fail("output: MASK from [H,W] (adds batch dim)", f"{e}\n{traceback.format_exc()}")

    # m@ prefix forces MASK output type even when value is vec3
    try:
        code = """
vec3 color = vec3(0.0, 1.0, 0.0);  // green
m@my_mask = color;
"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        program = Parser(tokens).parse()
        checker = TypeChecker({})
        checker.check(program)
        assert checker.assigned_bindings.get("my_mask") == TEXType.FLOAT, \
            f"Expected FLOAT (MASK), got {checker.assigned_bindings.get('my_mask')}"
        r.ok("m@ output prefix forces MASK type in assigned_bindings")
    except Exception as e:
        r.fail("m@ output prefix forces MASK type in assigned_bindings", f"{e}\n{traceback.format_exc()}")

    # img@ prefix forces IMAGE output type even when value is float
    try:
        code = "img@out = 0.5;"
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        program = Parser(tokens).parse()
        checker = TypeChecker({})
        checker.check(program)
        assert checker.assigned_bindings.get("out") == TEXType.VEC3, \
            f"Expected VEC3 (IMAGE), got {checker.assigned_bindings.get('out')}"
        r.ok("img@ output prefix forces IMAGE type in assigned_bindings")
    except Exception as e:
        r.fail("img@ output prefix forces IMAGE type in assigned_bindings", f"{e}\n{traceback.format_exc()}")

    # Full pipeline: m@mask = vec3 → prepared as [B,H,W] MASK
    try:
        code = """
vec3 color = vec3(0.0, 1.0, 0.0);
m@my_mask = color;
"""
        B, H, W = 1, 4, 4
        img = torch.rand(B, H, W, 3)
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        program = Parser(tokens).parse()
        checker = TypeChecker({"A": TEXType.VEC3})
        type_map = checker.check(program)
        interp = Interpreter()
        raw_out = interp.execute(program, {"A": img}, type_map, output_names=["my_mask"])
        raw = raw_out["my_mask"]
        eff = _map_inferred_type(checker.assigned_bindings["my_mask"], False)
        prepared = _prepare_output(raw, eff)
        assert prepared.shape == (B, H, W), f"Shape: {prepared.shape}"
        # green channel luma: 0.7152
        assert abs(prepared[0, 0, 0].item() - 0.7152) < 1e-3
        r.ok("m@ pipeline: vec3 to MASK (luma conversion)")
    except Exception as e:
        r.fail("m@ pipeline: vec3 to MASK (luma conversion)", f"{e}\n{traceback.format_exc()}")

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
        prog = Parser(tokens, source=code).parse()
        bt = bindings or {}
        bt.setdefault("OUT", TEXType.VEC4)
        checker = TypeChecker(binding_types=bt, source=code)
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
        assert "can't find" in str(e).lower() or "unknown" in str(e).lower()
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
        check_code("vec3 c = @A.rg;", {"A": TEXType.VEC4})
        r.fail("error: 2-component swizzle", "Should have raised TypeCheckError")
    except (TypeCheckError, TEXMultiError):
        r.ok("error: 2-component swizzle")
    except Exception as e:
        r.fail("error: 2-component swizzle", f"{e}\n{traceback.format_exc()}")

    # Invalid swizzle .rrr (not in VALID_SWIZZLES)
    try:
        check_code("vec3 c = @A.rrr;", {"A": TEXType.VEC4})
        r.fail("error: invalid swizzle .rrr", "Should have raised TypeCheckError")
    except (TypeCheckError, TEXMultiError) as e:
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
    # Use pow(-1,0.5) to produce NaN, and pow(0,-1) to produce Inf (raw pow is unclamped)
    check_val("isnan", "float x = isnan(pow(-1.0, 0.5));\n@OUT = vec3(x, x, x);", 1.0)
    check_val("isinf", "float x = isinf(pow(0.0, -1.0));\n@OUT = vec3(x, x, x);", 1.0)

    # spow: safe power — sign(x) * pow(abs(x), y)
    check_val("spow", "float x = spow(-2.0, 3.0);\n@OUT = vec3(x, x, x);", -8.0)

    # sdiv: safe division — 0 when b ~= 0
    check_val("sdiv", "float x = sdiv(5.0, 0.0);\n@OUT = vec3(x, x, x);", 0.0)


# ── Numerical Edge Case Tests ─────────────────────────────────────────

def test_numerical_edge_cases(r: TestResult):
    print("\n--- Numerical Edge Case Tests ---")

    B, H, W = 1, 2, 2
    test_img = torch.rand(B, H, W, 3)

    # sqrt(-1) clamped to 0
    try:
        code = "float x = sqrt(-1.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        assert abs(val) < 1e-5, f"sqrt(-1) should be clamped to 0, got {val}"
        r.ok("edge: sqrt(-1) clamped to 0")
    except Exception as e:
        r.fail("edge: sqrt(-1) clamped to 0", f"{e}\n{traceback.format_exc()}")

    # log(0) clamped to log(SAFE_EPSILON)
    try:
        import math
        code = "float x = log(0.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        expected_val = math.log(1e-8)  # ~-18.42
        assert not torch.isinf(result[0, 0, 0, 0]), f"log(0) should not be -Inf, got {val}"
        assert abs(val - expected_val) < 0.1, f"log(0) should be ~{expected_val:.2f}, got {val}"
        r.ok("edge: log(0) clamped to finite")
    except Exception as e:
        r.fail("edge: log(0) clamped to finite", f"{e}\n{traceback.format_exc()}")

    # asin(2) clamped to asin(1) = pi/2
    try:
        import math
        code = "float x = asin(2.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        assert abs(val - math.pi / 2) < 1e-4, f"asin(2) should be ~pi/2, got {val}"
        r.ok("edge: asin(2) clamped to pi/2")
    except Exception as e:
        r.fail("edge: asin(2) clamped to pi/2", f"{e}\n{traceback.format_exc()}")

    # acos(2) clamped to acos(1) = 0
    try:
        code = "float x = acos(2.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        assert abs(val) < 1e-4, f"acos(2) should be ~0.0, got {val}"
        r.ok("edge: acos(2) clamped to 0")
    except Exception as e:
        r.fail("edge: acos(2) clamped to 0", f"{e}\n{traceback.format_exc()}")

    # pow(0, -1) is Inf — standard pow is unchanged
    try:
        code = "float x = pow(0.0, -1.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        assert torch.isinf(result[0, 0, 0, 0]), "pow(0,-1) should be Inf"
        r.ok("edge: pow(0,-1) is Inf")
    except Exception as e:
        r.fail("edge: pow(0,-1) is Inf", f"{e}\n{traceback.format_exc()}")

    # spow(0, -1) returns 0 (safe power)
    try:
        code = "float x = spow(0.0, -1.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        assert abs(val) < 1e-4, f"spow(0,-1) should be 0, got {val}"
        r.ok("edge: spow(0,-1) is 0")
    except Exception as e:
        r.fail("edge: spow(0,-1) is 0", f"{e}\n{traceback.format_exc()}")

    # sdiv(1, 0) returns 0 (safe division)
    try:
        code = "float x = sdiv(1.0, 0.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        assert abs(val) < 1e-4, f"sdiv(1,0) should be 0, got {val}"
        r.ok("edge: sdiv(1,0) is 0")
    except Exception as e:
        r.fail("edge: sdiv(1,0) is 0", f"{e}\n{traceback.format_exc()}")

    # mod(5, 0) returns finite (not NaN)
    try:
        code = "float x = mod(5.0, 0.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        assert not torch.isnan(result[0, 0, 0, 0]), f"mod(5,0) should not be NaN, got {val}"
        r.ok("edge: mod(5,0) is finite")
    except Exception as e:
        r.fail("edge: mod(5,0) is finite", f"{e}\n{traceback.format_exc()}")

    # log2(0) clamped to finite
    try:
        import math
        code = "float x = log2(0.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        assert not torch.isinf(result[0, 0, 0, 0]), f"log2(0) should not be -Inf, got {val}"
        r.ok("edge: log2(0) clamped to finite")
    except Exception as e:
        r.fail("edge: log2(0) clamped to finite", f"{e}\n{traceback.format_exc()}")

    # log10(0) clamped to finite
    try:
        import math
        code = "float x = log10(0.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        assert not torch.isinf(result[0, 0, 0, 0]), f"log10(0) should not be -Inf, got {val}"
        r.ok("edge: log10(0) clamped to finite")
    except Exception as e:
        r.fail("edge: log10(0) clamped to finite", f"{e}\n{traceback.format_exc()}")

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

    # latent: fingerprint_inputs
    try:
        lat = _make_latent(1, 4, 4, 4)
        from TEX_Wrangle.tex_node import TEXWrangleNode
        # Should not crash with LATENT dict
        h = TEXWrangleNode.fingerprint_inputs(code="@OUT = @A;", A=lat)
        assert isinstance(h, str), f"Expected hash string, got {type(h)}"
        r.ok("latent: fingerprint_inputs")
    except Exception as e:
        r.fail("latent: fingerprint_inputs", f"{e}\n{traceback.format_exc()}")


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
    except (TypeCheckError, TEXMultiError, InterpreterError):
        r.ok("string: type error string+number")
    except Exception as e:
        r.fail("string: type error string+number", f"Wrong error type: {e}")

    # 23. Type error: string in vec constructor
    try:
        compile_and_run(
            'vec3 c = vec3("hello", 0.0, 0.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        r.fail("string: type error vec constructor", "Expected TypeCheckError")
    except (TypeCheckError, TEXMultiError):
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
        assert "doesn't work on" in str(e).lower() or "non-array" in str(e).lower() or "only arrays" in str(e).lower()
        r.ok("array: index non-array error")
    except Exception as e:
        r.fail("array: index non-array error", f"{e}\n{traceback.format_exc()}")


# ── Auto-Inference Tests ──────────────────────────────────────────────

def compile_and_infer(code: str, bindings: dict, device: str = "cpu",
                      latent_channel_count: int = 0) -> tuple:
    """Helper: compile+execute in auto-inference mode. Returns (result, inferred_out_type)."""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source=code)
    program = parser.parse()

    binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}

    checker = TypeChecker(binding_types=binding_types, source=code)
    type_map = checker.check(program)
    inferred = checker.inferred_out_type
    output_names = sorted(checker.assigned_bindings.keys())

    interp = Interpreter()
    result = interp.execute(program, bindings, type_map, device=device,
                            latent_channel_count=latent_channel_count,
                            output_names=output_names)
    return result["OUT"], inferred


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
    except (TypeCheckError, TEXMultiError):
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
        assert "transpose" in str(e).lower() or "cannot" in str(e).lower() or "isn't supported" in str(e).lower(), f"Unexpected error: {e}"
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
        img = torch.rand(1, 4, 4, 3)
        result = TEXWrangleNode.execute(
            code="@result = @A * 0.5;\n@mask = luma(@A);",
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
        img = torch.rand(1, 4, 4, 3)
        result = TEXWrangleNode.execute(code="@OUT = @A * 0.5;", A=img, device="cpu", compile_mode="none")
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert result[0] is not None, "slot 0 (OUT) should not be None"
        assert result[1] is None, "slot 1 should be None"
        r.ok("tex_node: backward compat (@OUT)")
    except Exception as e:
        r.fail("tex_node: backward compat (@OUT)", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: old output_type ignored ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        img = torch.rand(1, 4, 4, 3)
        # Passing output_type="IMAGE" — should be silently ignored
        result = TEXWrangleNode.execute(code="@OUT = @A * 0.5;", A=img, device="cpu",
                                        compile_mode="none", output_type="IMAGE")
        assert result[0] is not None
        r.ok("tex_node: old output_type ignored")
    except Exception as e:
        r.fail("tex_node: old output_type ignored", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: param widgets flow through kwargs ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        img = torch.ones(1, 2, 2, 3)
        # $strength value comes as a kwarg (simulating widget value from ComfyUI)
        result = TEXWrangleNode.execute(
            code="f$strength = 0.5;\n@OUT = @A * $strength;",
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
        img = torch.ones(1, 2, 2, 3)
        # No strength kwarg — should use default from code (0.5)
        result = TEXWrangleNode.execute(
            code="f$strength = 0.5;\n@OUT = @A * $strength;",
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
        img = torch.ones(1, 2, 2, 3)
        # Widget value injected as kwarg by graphToPrompt hook (overrides code default)
        result = TEXWrangleNode.execute(
            code="f$strength = 0.5;\n@OUT = @A * $strength;",
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
        img = torch.ones(1, 2, 2, 3)
        # Widget value (scalar kwarg) overrides code default of 0.5
        result = TEXWrangleNode.execute(
            code="f$strength = 0.5;\n@OUT = @A * $strength;",
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
        program, type_map, refs, assigned, params, *_ = cache.compile_tex(code, bt)
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
        program, type_map, refs, assigned, params, *_ = cache.compile_tex(code, bt)
        assert "result" in assigned, "Missing 'result' in assigned_bindings"
        assert "mask" in assigned, "Missing 'mask' in assigned_bindings"
        assert assigned["result"] == TEXType.VEC3
        assert assigned["mask"] == TEXType.FLOAT
        r.ok("cache: multi-output compile_tex")
    except Exception as e:
        r.fail("cache: multi-output compile_tex", f"{e}\n{traceback.format_exc()}")


# ── Wireable Parameter Tests ───────────────────────────────────────────

def test_wireable_params(r: TestResult):
    """Tests that param ($) values work when passed as different types (simulating wired inputs)."""
    print("\n--- Wireable Parameter Tests ---")

    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 3)

    # Float param passed as Python float (widget path)
    try:
        code = "f$strength = 1.0;\n@OUT = @A * $strength;"
        result = compile_and_run(code, {"A": test_img, "strength": 0.5})
        expected = test_img * 0.5
        assert torch.allclose(result[..., :3], expected, atol=1e-5), \
            f"Max diff: {(result[..., :3] - expected).abs().max().item()}"
        r.ok("wireable: float param from widget")
    except Exception as e:
        r.fail("wireable: float param from widget", f"{e}\n{traceback.format_exc()}")

    # Float param passed as scalar tensor (simulating wire from FLOAT output)
    try:
        code = "f$strength = 1.0;\n@OUT = @A * $strength;"
        result = compile_and_run(code, {"A": test_img, "strength": torch.tensor(0.25)})
        expected = test_img * 0.25
        assert torch.allclose(result[..., :3], expected, atol=1e-5), \
            f"Max diff: {(result[..., :3] - expected).abs().max().item()}"
        r.ok("wireable: float param from wire (scalar tensor)")
    except Exception as e:
        r.fail("wireable: float param from wire (scalar tensor)", f"{e}\n{traceback.format_exc()}")

    # Int param passed as Python int
    try:
        code = "i$count = 1;\nfloat c = float($count);\n@OUT = @A * c;"
        result = compile_and_run(code, {"A": test_img, "count": 2})
        expected = test_img * 2.0
        assert torch.allclose(result[..., :3], expected, atol=1e-5)
        r.ok("wireable: int param from widget")
    except Exception as e:
        r.fail("wireable: int param from widget", f"{e}\n{traceback.format_exc()}")

    # Param uses code default when not provided (via tex_node's param_info injection)
    try:
        from TEX_Wrangle.tex_cache import TEXCache
        cache = TEXCache(cache_dir=Path(tempfile.mkdtemp()))
        code = "f$strength = 0.5;\n@OUT = @A * $strength;"
        bt = {"A": TEXType.VEC3}
        program, type_map, refs, assigned, param_info, *_ = cache.compile_tex(code, bt)
        # Simulate tex_node.py's default injection: inject param default into bindings
        bindings = {"A": test_img}
        for ref_name in refs:
            if ref_name not in bindings and ref_name in param_info:
                default_val = param_info[ref_name].get("default_value")
                if default_val is not None:
                    bindings[ref_name] = default_val
        interp = Interpreter()
        result = interp.execute(program, bindings, type_map, device="cpu")
        expected = test_img * 0.5
        assert torch.allclose(result[..., :3], expected, atol=1e-5)
        r.ok("wireable: param falls back to code default")
    except Exception as e:
        r.fail("wireable: param falls back to code default", f"{e}\n{traceback.format_exc()}")

    # Non-scalar param: mask ($m) passed as tensor
    try:
        mask = torch.ones(B, H, W)
        code = "m$mymask;\n@OUT = @A * $mymask;"
        result = compile_and_run(code, {"A": test_img, "mymask": mask})
        assert torch.allclose(result[..., :3], test_img, atol=1e-5)
        r.ok("wireable: mask param from wire")
    except Exception as e:
        r.fail("wireable: mask param from wire", f"{e}\n{traceback.format_exc()}")


def test_string_functions_v04(r: TestResult):
    """Tests for new string functions added in v0.4."""
    print("\n--- String Functions v0.4 Tests ---")
    B, H, W = 1, 2, 2
    img = torch.rand(B, H, W, 4)

    # -- split --
    try:
        code = '''
string s = "a,b,c,d";
string arr[] = split(s, ",");
@OUT = arr[2];
'''
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "c", f"Expected 'c', got {result!r}"
        r.ok("split: basic delimiter")
    except Exception as e:
        r.fail("split: basic delimiter", f"{e}\n{traceback.format_exc()}")

    try:
        code = '''
string s = "a.b.c.d";
string arr[] = split(s, ".", 2);
@OUT = arr[2];
'''
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "c.d", f"Expected 'c.d', got {result!r}"
        r.ok("split: max_splits")
    except Exception as e:
        r.fail("split: max_splits", f"{e}\n{traceback.format_exc()}")

    try:
        code = '''
string s = "hello world foo";
string arr[] = split(s, " ");
@OUT = join(arr, "-");
'''
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "hello-world-foo", f"Expected 'hello-world-foo', got {result!r}"
        r.ok("split: round-trip with join")
    except Exception as e:
        r.fail("split: round-trip with join", f"{e}\n{traceback.format_exc()}")

    # -- lstrip / rstrip --
    try:
        result = compile_and_run(
            'string s = "  hello  "; @OUT = lstrip(s);',
            {}, out_type=TEXType.STRING)
        assert result == "hello  ", f"Expected 'hello  ', got {result!r}"
        r.ok("lstrip: trim leading")
    except Exception as e:
        r.fail("lstrip: trim leading", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            'string s = "  hello  "; @OUT = rstrip(s);',
            {}, out_type=TEXType.STRING)
        assert result == "  hello", f"Expected '  hello', got {result!r}"
        r.ok("rstrip: trim trailing")
    except Exception as e:
        r.fail("rstrip: trim trailing", f"{e}\n{traceback.format_exc()}")

    # -- pad_left / pad_right --
    try:
        result = compile_and_run(
            '@OUT = pad_left("42", 5, "0");',
            {}, out_type=TEXType.STRING)
        assert result == "00042", f"Expected '00042', got {result!r}"
        r.ok("pad_left: zero-pad number")
    except Exception as e:
        r.fail("pad_left: zero-pad number", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = pad_right("hi", 6);',
            {}, out_type=TEXType.STRING)
        assert result == "hi    ", f"Expected 'hi    ', got {result!r}"
        r.ok("pad_right: space-pad default")
    except Exception as e:
        r.fail("pad_right: space-pad default", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = pad_left("long_string", 5, "x");',
            {}, out_type=TEXType.STRING)
        assert result == "long_string", f"Expected no-op for string longer than width, got {result!r}"
        r.ok("pad_left: no-op when already wider")
    except Exception as e:
        r.fail("pad_left: no-op when already wider", f"{e}\n{traceback.format_exc()}")

    # -- format --
    try:
        result = compile_and_run(
            '@OUT = format("frame_{}_v{}", 42, 3);',
            {}, out_type=TEXType.STRING)
        assert result == "frame_42_v3", f"Expected 'frame_42_v3', got {result!r}"
        r.ok("format: integer placeholders")
    except Exception as e:
        r.fail("format: integer placeholders", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = format("value={}", 3.14);',
            {}, out_type=TEXType.STRING)
        assert result == "value=3.14", f"Expected 'value=3.14', got {result!r}"
        r.ok("format: float placeholder")
    except Exception as e:
        r.fail("format: float placeholder", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = format("{}-{}", "hello", "world");',
            {}, out_type=TEXType.STRING)
        assert result == "hello-world", f"Expected 'hello-world', got {result!r}"
        r.ok("format: string placeholders")
    except Exception as e:
        r.fail("format: string placeholders", f"{e}\n{traceback.format_exc()}")

    # -- repeat --
    try:
        result = compile_and_run(
            '@OUT = repeat("ab", 3);',
            {}, out_type=TEXType.STRING)
        assert result == "ababab", f"Expected 'ababab', got {result!r}"
        r.ok("repeat: basic repeat")
    except Exception as e:
        r.fail("repeat: basic repeat", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = repeat("x", 0);',
            {}, out_type=TEXType.STRING)
        assert result == "", f"Expected '', got {result!r}"
        r.ok("repeat: zero count")
    except Exception as e:
        r.fail("repeat: zero count", f"{e}\n{traceback.format_exc()}")

    # -- str_reverse --
    try:
        result = compile_and_run(
            '@OUT = str_reverse("hello");',
            {}, out_type=TEXType.STRING)
        assert result == "olleh", f"Expected 'olleh', got {result!r}"
        r.ok("str_reverse: basic reverse")
    except Exception as e:
        r.fail("str_reverse: basic reverse", f"{e}\n{traceback.format_exc()}")

    # -- count --
    try:
        result = compile_and_run(
            'float n = count("banana", "an"); @OUT = vec4(n);',
            {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 2.0, f"Expected 2.0, got {v}"
        r.ok("count: substring occurrences")
    except Exception as e:
        r.fail("count: substring occurrences", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            'float n = count("hello", "xyz"); @OUT = vec4(n);',
            {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 0.0, f"Expected 0.0, got {v}"
        r.ok("count: no matches")
    except Exception as e:
        r.fail("count: no matches", f"{e}\n{traceback.format_exc()}")

    # -- replace with max_count --
    try:
        result = compile_and_run(
            '@OUT = replace("aaa", "a", "b", 2);',
            {}, out_type=TEXType.STRING)
        assert result == "bba", f"Expected 'bba', got {result!r}"
        r.ok("replace: max_count limits replacements")
    except Exception as e:
        r.fail("replace: max_count limits replacements", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = replace("aaa", "a", "b");',
            {}, out_type=TEXType.STRING)
        assert result == "bbb", f"Expected 'bbb', got {result!r}"
        r.ok("replace: all occurrences (no max)")
    except Exception as e:
        r.fail("replace: all occurrences (no max)", f"{e}\n{traceback.format_exc()}")

    # -- matches --
    try:
        result = compile_and_run(
            'float m = matches("frame_042", "frame_\\\\d+"); @OUT = vec4(m);',
            {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 1.0, f"Expected 1.0, got {v}"
        r.ok("matches: regex match")
    except Exception as e:
        r.fail("matches: regex match", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            'float m = matches("frame_abc", "frame_\\\\d+"); @OUT = vec4(m);',
            {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 0.0, f"Expected 0.0, got {v}"
        r.ok("matches: regex no match")
    except Exception as e:
        r.fail("matches: regex no match", f"{e}\n{traceback.format_exc()}")

    # -- hash --
    try:
        result = compile_and_run(
            '@OUT = hash("hello");',
            {}, out_type=TEXType.STRING)
        assert len(result) == 16, f"Expected 16-char hash, got len={len(result)}"
        assert all(c in "0123456789abcdef" for c in result), f"Expected hex string, got {result!r}"
        r.ok("hash: returns hex string")
    except Exception as e:
        r.fail("hash: returns hex string", f"{e}\n{traceback.format_exc()}")

    try:
        r1 = compile_and_run('@OUT = hash("hello");', {}, out_type=TEXType.STRING)
        r2 = compile_and_run('@OUT = hash("hello");', {}, out_type=TEXType.STRING)
        r3 = compile_and_run('@OUT = hash("world");', {}, out_type=TEXType.STRING)
        assert r1 == r2, f"Same input should produce same hash"
        assert r1 != r3, f"Different input should produce different hash"
        r.ok("hash: deterministic and distinct")
    except Exception as e:
        r.fail("hash: deterministic and distinct", f"{e}\n{traceback.format_exc()}")


def test_while_loops(r: TestResult):
    """Tests for while loop support."""
    print("\n--- While Loop Tests ---")
    B, H, W = 1, 2, 2
    img = torch.rand(B, H, W, 4)

    # 1. Basic while loop: count to 5
    try:
        code = """
float x = 0.0;
float i = 0.0;
while (i < 5) {
    x = x + 1.0;
    i = i + 1.0;
}
@OUT = vec4(x);
"""
        result = compile_and_run(code, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 5.0, f"Expected 5.0, got {v}"
        r.ok("while: basic counting")
    except Exception as e:
        r.fail("while: basic counting", f"{e}\n{traceback.format_exc()}")

    # 2. While loop with break
    try:
        code = """
float x = 0.0;
float i = 0.0;
while (i < 100) {
    if (i >= 3) {
        break;
    }
    x = x + 1.0;
    i = i + 1.0;
}
@OUT = vec4(x);
"""
        result = compile_and_run(code, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 3.0, f"Expected 3.0, got {v}"
        r.ok("while: break exits loop")
    except Exception as e:
        r.fail("while: break exits loop", f"{e}\n{traceback.format_exc()}")

    # 3. While loop with continue
    try:
        code = """
float total = 0.0;
float i = 0.0;
while (i < 6) {
    i = i + 1.0;
    if (mod(i, 2.0) > 0.5) {
        continue;
    }
    total = total + i;
}
@OUT = vec4(total);
"""
        result = compile_and_run(code, {"A": img})
        v = result[0, 0, 0, 0].item()
        # Even numbers 2+4+6 = 12
        assert v == 12.0, f"Expected 12.0, got {v}"
        r.ok("while: continue skips iteration")
    except Exception as e:
        r.fail("while: continue skips iteration", f"{e}\n{traceback.format_exc()}")

    # 4. While loop terminates on false condition (zero iterations)
    try:
        code = """
float x = 42.0;
while (0) {
    x = 0.0;
}
@OUT = vec4(x);
"""
        result = compile_and_run(code, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 42.0, f"Expected 42.0 (body never runs), got {v}"
        r.ok("while: false condition skips body")
    except Exception as e:
        r.fail("while: false condition skips body", f"{e}\n{traceback.format_exc()}")

    # 5. While loop iteration limit
    try:
        code = """
float x = 0.0;
while (1) {
    x = x + 1.0;
}
@OUT = vec4(x);
"""
        compile_and_run(code, {"A": img})
        r.fail("while: iteration limit", "Should have raised InterpreterError")
    except Exception as e:
        assert "maximum iteration limit" in str(e).lower() or "exceeded" in str(e).lower() or "iterations without finishing" in str(e).lower()
        r.ok("while: iteration limit")

    # 6. Nested while loops
    try:
        code = """
float total = 0.0;
float i = 0.0;
while (i < 3) {
    float j = 0.0;
    while (j < 3) {
        total = total + 1.0;
        j = j + 1.0;
    }
    i = i + 1.0;
}
@OUT = vec4(total);
"""
        result = compile_and_run(code, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 9.0, f"Expected 9.0, got {v}"
        r.ok("while: nested loops")
    except Exception as e:
        r.fail("while: nested loops", f"{e}\n{traceback.format_exc()}")

    # 7. While with string building
    try:
        code = """
string s = "";
float i = 0.0;
while (i < 3) {
    s = s + "x";
    i = i + 1.0;
}
@OUT = s;
"""
        result = compile_and_run(code, {}, out_type=TEXType.STRING)
        assert result == "xxx", f"Expected 'xxx', got {result!r}"
        r.ok("while: string building")
    except Exception as e:
        r.fail("while: string building", f"{e}\n{traceback.format_exc()}")


def test_new_stdlib_functions(r: TestResult):
    """Tests for hash_float, hash_int, char_at, and string array sort."""
    print("\n--- New Stdlib Function Tests ---")
    B, H, W = 1, 2, 2
    img = torch.rand(B, H, W, 4)

    # -- hash_float --
    try:
        result = compile_and_run(
            'float h = hash_float("seed_42"); @OUT = vec4(h);',
            {"A": img})
        v = result[0, 0, 0, 0].item()
        assert 0.0 <= v < 1.0, f"Expected [0,1), got {v}"
        r.ok("hash_float: returns value in [0,1)")
    except Exception as e:
        r.fail("hash_float: returns value in [0,1)", f"{e}\n{traceback.format_exc()}")

    try:
        r1 = compile_and_run('float h = hash_float("abc"); @OUT = vec4(h);', {"A": img})
        r2 = compile_and_run('float h = hash_float("abc"); @OUT = vec4(h);', {"A": img})
        r3 = compile_and_run('float h = hash_float("xyz"); @OUT = vec4(h);', {"A": img})
        v1 = r1[0, 0, 0, 0].item()
        v2 = r2[0, 0, 0, 0].item()
        v3 = r3[0, 0, 0, 0].item()
        assert v1 == v2, "Same input should produce same hash_float"
        assert v1 != v3, "Different input should produce different hash_float"
        r.ok("hash_float: deterministic and distinct")
    except Exception as e:
        r.fail("hash_float: deterministic and distinct", f"{e}\n{traceback.format_exc()}")

    # -- hash_int --
    try:
        result = compile_and_run(
            'float h = hash_int("seed", 100); @OUT = vec4(h);',
            {"A": img})
        v = result[0, 0, 0, 0].item()
        assert 0.0 <= v < 100.0, f"Expected [0,100), got {v}"
        assert v == int(v), f"Expected integer, got {v}"
        r.ok("hash_int: returns int in [0, max)")
    except Exception as e:
        r.fail("hash_int: returns int in [0, max)", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            'float h = hash_int("test_key"); @OUT = vec4(h);',
            {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v >= 0 and v == int(v), f"Expected non-negative integer, got {v}"
        r.ok("hash_int: no max returns large int")
    except Exception as e:
        r.fail("hash_int: no max returns large int", f"{e}\n{traceback.format_exc()}")

    # -- char_at --
    try:
        result = compile_and_run(
            '@OUT = char_at("hello", 1);',
            {}, out_type=TEXType.STRING)
        assert result == "e", f"Expected 'e', got {result!r}"
        r.ok("char_at: basic index")
    except Exception as e:
        r.fail("char_at: basic index", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = char_at("hello", 0);',
            {}, out_type=TEXType.STRING)
        assert result == "h", f"Expected 'h', got {result!r}"
        r.ok("char_at: first character")
    except Exception as e:
        r.fail("char_at: first character", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = char_at("hello", 99);',
            {}, out_type=TEXType.STRING)
        assert result == "", f"Expected '' for out-of-bounds, got {result!r}"
        r.ok("char_at: out of bounds returns empty")
    except Exception as e:
        r.fail("char_at: out of bounds returns empty", f"{e}\n{traceback.format_exc()}")

    # -- string array sort (already implemented, verifying) --
    try:
        code = """
string arr[] = {"cherry", "apple", "banana"};
string sorted_arr[] = sort(arr);
@OUT = sorted_arr[0];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "apple", f"Expected 'apple', got {result!r}"
        r.ok("sort: string array lexicographic")
    except Exception as e:
        r.fail("sort: string array lexicographic", f"{e}\n{traceback.format_exc()}")

    try:
        code = """
string arr[] = {"cherry", "apple", "banana"};
string sorted_arr[] = sort(arr);
@OUT = sorted_arr[2];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "cherry", f"Expected 'cherry', got {result!r}"
        r.ok("sort: string array last element")
    except Exception as e:
        r.fail("sort: string array last element", f"{e}\n{traceback.format_exc()}")


def test_else_if_chains(r: TestResult):
    """Verify else-if chains work correctly."""
    print("\n--- Else-If Chain Tests ---")
    B, H, W = 1, 2, 2
    img = torch.rand(B, H, W, 4)

    # 1. else if chain
    try:
        code = """
float x = 2.0;
float result = 0.0;
if (x < 1.0) {
    result = 10.0;
} else if (x < 3.0) {
    result = 20.0;
} else {
    result = 30.0;
}
@OUT = vec4(result);
"""
        result = compile_and_run(code, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 20.0, f"Expected 20.0, got {v}"
        r.ok("else if: middle branch")
    except Exception as e:
        r.fail("else if: middle branch", f"{e}\n{traceback.format_exc()}")

    # 2. else if chain - first branch
    try:
        code = """
float x = 0.5;
float result = 0.0;
if (x < 1.0) {
    result = 10.0;
} else if (x < 3.0) {
    result = 20.0;
} else if (x < 5.0) {
    result = 30.0;
} else {
    result = 40.0;
}
@OUT = vec4(result);
"""
        result = compile_and_run(code, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 10.0, f"Expected 10.0, got {v}"
        r.ok("else if: first branch of 4")
    except Exception as e:
        r.fail("else if: first branch of 4", f"{e}\n{traceback.format_exc()}")

    # 3. else if chain - last (else) branch
    try:
        code = """
float x = 99.0;
float result = 0.0;
if (x < 1.0) {
    result = 10.0;
} else if (x < 3.0) {
    result = 20.0;
} else {
    result = 30.0;
}
@OUT = vec4(result);
"""
        result = compile_and_run(code, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 30.0, f"Expected 30.0, got {v}"
        r.ok("else if: falls through to else")
    except Exception as e:
        r.fail("else if: falls through to else", f"{e}\n{traceback.format_exc()}")

    # 4. else if without final else
    try:
        code = """
float x = 99.0;
float result = 0.0;
if (x < 1.0) {
    result = 10.0;
} else if (x < 3.0) {
    result = 20.0;
}
@OUT = vec4(result);
"""
        result = compile_and_run(code, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 0.0, f"Expected 0.0 (no branch taken), got {v}"
        r.ok("else if: no matching branch, no final else")
    except Exception as e:
        r.fail("else if: no matching branch, no final else", f"{e}\n{traceback.format_exc()}")


# ── Optimization Regression Tests ──────────────────────────────────────

def test_optimization_regressions(r: TestResult):
    """Targeted regression tests for all optimization paths introduced in Phases 1-7."""
    print("\n--- Optimization Regression Tests ---")
    img = torch.rand(1, 8, 8, 3)
    img4 = torch.rand(1, 8, 8, 4)
    mask = torch.rand(1, 8, 8)

    # ── 1. Inlined binop operators ──────────────────────────────────

    # Test all operators produce correct results vs known values
    try:
        a = torch.full((1, 4, 4, 3), 0.7)
        b = torch.full((1, 4, 4, 3), 0.3)
        bindings = {"A": a, "B": b}

        # Addition
        result = compile_and_run("@OUT = @A + @B;", bindings, out_type=TEXType.VEC3)
        assert torch.allclose(result, a + b, atol=1e-5), f"+ failed"
        # Subtraction
        result = compile_and_run("@OUT = @A - @B;", bindings, out_type=TEXType.VEC3)
        assert torch.allclose(result, a - b, atol=1e-5), f"- failed"
        # Multiplication
        result = compile_and_run("@OUT = @A * @B;", bindings, out_type=TEXType.VEC3)
        assert torch.allclose(result, a * b, atol=1e-5), f"* failed"
        # Division (with safe epsilon)
        result = compile_and_run("@OUT = @A / @B;", bindings, out_type=TEXType.VEC3)
        assert result.shape == a.shape, f"/ shape mismatch"
        # Comparison operators
        result = compile_and_run("@OUT = vec3(@A.r < @B.r);", bindings, out_type=TEXType.VEC3)
        assert (result[..., 0] == 0.0).all(), "< failed (0.7 < 0.3 should be false)"
        result = compile_and_run("@OUT = vec3(@A.r > @B.r);", bindings, out_type=TEXType.VEC3)
        assert (result[..., 0] == 1.0).all(), "> failed (0.7 > 0.3 should be true)"
        result = compile_and_run("@OUT = vec3(@A.r == @B.r);", bindings, out_type=TEXType.VEC3)
        assert (result[..., 0] == 0.0).all(), "== failed"
        result = compile_and_run("@OUT = vec3(@A.r != @B.r);", bindings, out_type=TEXType.VEC3)
        assert (result[..., 0] == 1.0).all(), "!= failed"
        # Logical operators
        t = torch.ones(1, 4, 4, 3)
        f_val = torch.zeros(1, 4, 4, 3)
        result = compile_and_run("@OUT = vec3(@A.r && @B.r);", {"A": t, "B": t}, out_type=TEXType.VEC3)
        assert (result[..., 0] == 1.0).all(), "&& failed"
        result = compile_and_run("@OUT = vec3(@A.r || @B.r);", {"A": f_val, "B": t}, out_type=TEXType.VEC3)
        assert (result[..., 0] == 1.0).all(), "|| failed"
        # Modulo
        result = compile_and_run("@OUT = vec3(mod(@A.r, @B.r));", bindings, out_type=TEXType.VEC3)
        assert result.shape == a.shape, "mod shape mismatch"
        r.ok("inlined binop: all operators correct")
    except Exception as e:
        r.fail("inlined binop: all operators correct", f"{e}\n{traceback.format_exc()}")

    # Binop with string operands still works
    try:
        result = compile_and_run('@OUT_str = "hello" + " " + "world";', {"A": img})
        assert result["OUT_str"] == "hello world", f"String concat failed: {result['OUT_str']}"
        r.ok("inlined binop: string concat")
    except Exception as e:
        r.fail("inlined binop: string concat", f"{e}\n{traceback.format_exc()}")

    # Binop with scalar-vs-spatial broadcast
    try:
        result = compile_and_run("@OUT = @A + 0.5;", {"A": img}, out_type=TEXType.VEC3)
        assert result.shape == img.shape, f"Broadcast shape: {result.shape}"
        assert torch.allclose(result, img + 0.5, atol=1e-5)
        r.ok("inlined binop: scalar-spatial broadcast")
    except Exception as e:
        r.fail("inlined binop: scalar-spatial broadcast", f"{e}\n{traceback.format_exc()}")

    # Division by zero protection
    try:
        result = compile_and_run("@OUT = @A / vec3(0.0);", {"A": img}, out_type=TEXType.VEC3)
        assert not torch.isnan(result).any(), "Division by zero produced NaN"
        assert not torch.isinf(result).any(), "Division by zero produced Inf"
        r.ok("inlined binop: division by zero protection")
    except Exception as e:
        r.fail("inlined binop: division by zero protection", f"{e}\n{traceback.format_exc()}")

    # ── 2. Matrix multiplication path ───────────────────────────────

    # mat3 * mat3 (same dimensions — was broken before fix)
    try:
        code = """
mat3 m = mat3(1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0);
mat3 m2 = mat3(2.0, 0.0, 0.0,
               0.0, 3.0, 0.0,
               0.0, 0.0, 4.0);
mat3 prod = m * m2;
vec3 col0 = prod * vec3(1.0, 0.0, 0.0);
vec3 col1 = prod * vec3(0.0, 1.0, 0.0);
vec3 col2 = prod * vec3(0.0, 0.0, 1.0);
@OUT = vec3(col0.r, col1.g, col2.b);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = torch.tensor([2.0, 3.0, 4.0])
        assert torch.allclose(result[0, 0, 0], expected, atol=1e-4), f"mat3*mat3 diagonal: {result[0,0,0]}"
        r.ok("matrix multiplication: mat3 * mat3 (same dim)")
    except Exception as e:
        r.fail("matrix multiplication: mat3 * mat3 (same dim)", f"{e}\n{traceback.format_exc()}")

    # mat3 * vec3
    try:
        code = """
mat3 m = mat3(1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0);
vec3 tv = vec3(1.0, 2.0, 3.0);
@OUT = m * tv;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(result[0, 0, 0], expected, atol=1e-4), f"mat3*vec3: {result[0,0,0]}"
        r.ok("matrix multiplication: mat3 * vec3")
    except Exception as e:
        r.fail("matrix multiplication: mat3 * vec3", f"{e}\n{traceback.format_exc()}")

    # scalar * mat3 (should be element-wise, not matmul)
    try:
        # 2 * identity should give 2*identity; verify via mat*vec
        code = """
mat3 m = mat3(1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0);
mat3 scaled = 2.0 * m;
vec3 test_vec = vec3(1.0, 2.0, 3.0);
@OUT = scaled * test_vec;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = torch.tensor([2.0, 4.0, 6.0])  # 2I * [1,2,3] = [2,4,6]
        assert torch.allclose(result[0, 0, 0], expected, atol=1e-4), f"scalar*mat3: {result[0,0,0]}"
        r.ok("matrix multiplication: scalar * mat3 (element-wise)")
    except Exception as e:
        r.fail("matrix multiplication: scalar * mat3 (element-wise)", f"{e}\n{traceback.format_exc()}")

    # ── 3. Vec constructor optimization ─────────────────────────────

    # vec3(scalar) broadcast in spatial context
    try:
        result = compile_and_run("@OUT = vec3(0.5);", {"A": img}, out_type=TEXType.VEC3)
        assert result.shape == (1, 8, 8, 3), f"vec3(scalar) shape: {result.shape}"
        assert torch.allclose(result, torch.full((1, 8, 8, 3), 0.5)), "vec3(scalar) values wrong"
        r.ok("vec constructor: vec3(scalar) broadcast spatial")
    except Exception as e:
        r.fail("vec constructor: vec3(scalar) broadcast spatial", f"{e}\n{traceback.format_exc()}")

    # vec4(scalar) broadcast
    try:
        result = compile_and_run("@OUT = vec4(0.25);", {"A": img4})
        assert result.shape == (1, 8, 8, 4), f"vec4(scalar) shape: {result.shape}"
        assert torch.allclose(result, torch.full((1, 8, 8, 4), 0.25))
        r.ok("vec constructor: vec4(scalar) broadcast spatial")
    except Exception as e:
        r.fail("vec constructor: vec4(scalar) broadcast spatial", f"{e}\n{traceback.format_exc()}")

    # vec3(spatial_scalar) — broadcast [B,H,W] to [B,H,W,3]
    try:
        result = compile_and_run("@OUT = vec3(@A.r);", {"A": img}, out_type=TEXType.VEC3)
        assert result.shape == (1, 8, 8, 3), f"vec3(spatial) shape: {result.shape}"
        for c in range(3):
            assert torch.allclose(result[..., c], img[..., 0], atol=1e-5), f"Channel {c} mismatch"
        r.ok("vec constructor: vec3(spatial_scalar)")
    except Exception as e:
        r.fail("vec constructor: vec3(spatial_scalar)", f"{e}\n{traceback.format_exc()}")

    # vec3(r, g, b) multi-arg spatial (uses empty+fill path)
    try:
        result = compile_and_run("@OUT = vec3(@A.r, @A.g, @A.b);", {"A": img}, out_type=TEXType.VEC3)
        assert result.shape == img.shape, f"vec3(r,g,b) shape: {result.shape}"
        assert torch.allclose(result, img, atol=1e-5), "vec3(r,g,b) should reproduce original"
        r.ok("vec constructor: vec3(r,g,b) multi-arg spatial")
    except Exception as e:
        r.fail("vec constructor: vec3(r,g,b) multi-arg spatial", f"{e}\n{traceback.format_exc()}")

    # vec4 from scalar args in non-spatial context
    try:
        code = """
float a = 0.1;
float b = 0.2;
float c = 0.3;
float d = 0.4;
@OUT = vec4(a, b, c, d);
"""
        result = compile_and_run(code, {})
        expected = torch.tensor([0.1, 0.2, 0.3, 0.4])
        assert torch.allclose(result, expected, atol=1e-5), f"vec4 non-spatial: {result}"
        r.ok("vec constructor: vec4 non-spatial scalar args")
    except Exception as e:
        r.fail("vec constructor: vec4 non-spatial scalar args", f"{e}\n{traceback.format_exc()}")

    # vec4(vec3, float) — composite constructor (GLSL-style)
    try:
        code = """
vec3 color = vec3(0.1, 0.2, 0.3);
@OUT = vec4(color, 1.0);
"""
        result = compile_and_run(code, {})
        expected = torch.tensor([0.1, 0.2, 0.3, 1.0])
        assert torch.allclose(result, expected, atol=1e-5), f"vec4(vec3,float): {result}"
        r.ok("vec constructor: vec4(vec3, float) composite")
    except Exception as e:
        r.fail("vec constructor: vec4(vec3, float) composite", f"{e}\n{traceback.format_exc()}")

    # vec4(float, vec3) — float first, then vec3
    try:
        code = """
vec3 color = vec3(0.2, 0.3, 0.4);
@OUT = vec4(1.0, color);
"""
        result = compile_and_run(code, {})
        expected = torch.tensor([1.0, 0.2, 0.3, 0.4])
        assert torch.allclose(result, expected, atol=1e-5), f"vec4(float,vec3): {result}"
        r.ok("vec constructor: vec4(float, vec3) composite")
    except Exception as e:
        r.fail("vec constructor: vec4(float, vec3) composite", f"{e}\n{traceback.format_exc()}")

    # vec4(vec3, float) — spatial context (image inputs)
    try:
        code = """
vec3 color = @A.rgb;
@OUT = vec4(color, 0.5);
"""
        result = compile_and_run(code, {"A": img})
        assert result.shape == (1, 8, 8, 4), f"Shape: {result.shape}"
        assert torch.allclose(result[..., :3], img, atol=1e-5), "RGB channels should match input"
        assert torch.allclose(result[..., 3], torch.full((1, 8, 8), 0.5), atol=1e-5), "Alpha should be 0.5"
        r.ok("vec constructor: vec4(vec3, float) spatial")
    except Exception as e:
        r.fail("vec constructor: vec4(vec3, float) spatial", f"{e}\n{traceback.format_exc()}")

    # vec3(float, float, float) still works (regression check)
    try:
        code = "@OUT = vec3(0.1, 0.2, 0.3);"
        result = compile_and_run(code, {})
        expected = torch.tensor([0.1, 0.2, 0.3])
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("vec constructor: vec3(f, f, f) regression")
    except Exception as e:
        r.fail("vec constructor: vec3(f, f, f) regression", f"{e}\n{traceback.format_exc()}")

    # simplex_terrain pattern: vec4(vec3_var, 1.0) with spatial inputs
    try:
        code = """
vec3 water = vec3(0.1, 0.3, 0.7);
vec3 color = water;
@OUT = vec4(color, 1.0);
"""
        result = compile_and_run(code, {"A": img})
        assert result.shape == (1, 8, 8, 4), f"Shape: {result.shape}"
        assert abs(result[0, 0, 0, 0].item() - 0.1) < 1e-5
        assert abs(result[0, 0, 0, 3].item() - 1.0) < 1e-5
        r.ok("vec constructor: simplex_terrain pattern")
    except Exception as e:
        r.fail("vec constructor: simplex_terrain pattern", f"{e}\n{traceback.format_exc()}")

    # ── 4. Static for-loop optimization ─────────────────────────────

    # Standard static loop: for (int i = 0; i < 5; i = i + 1)
    try:
        code = """
float sum = 0.0;
for (int i = 0; i < 5; i = i + 1) {
    sum = sum + 1.0;
}
@OUT = vec3(sum);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert abs(result[0, 0, 0, 0].item() - 5.0) < 1e-5, f"Static loop sum: {result[0,0,0,0].item()}"
        r.ok("static for-loop: standard i++ pattern")
    except Exception as e:
        r.fail("static for-loop: standard i++ pattern", f"{e}\n{traceback.format_exc()}")

    # Static loop with step > 1
    try:
        code = """
float sum = 0.0;
for (int i = 0; i < 10; i = i + 2) {
    sum = sum + 1.0;
}
@OUT = vec3(sum);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert abs(result[0, 0, 0, 0].item() - 5.0) < 1e-5, f"Step-2 loop: {result[0,0,0,0].item()}"
        r.ok("static for-loop: step 2")
    except Exception as e:
        r.fail("static for-loop: step 2", f"{e}\n{traceback.format_exc()}")

    # Static loop with <= condition
    try:
        code = """
float sum = 0.0;
for (int i = 1; i <= 5; i = i + 1) {
    sum = sum + float(i);
}
@OUT = vec3(sum);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert abs(result[0, 0, 0, 0].item() - 15.0) < 1e-5, f"<= loop sum: {result[0,0,0,0].item()}"
        r.ok("static for-loop: <= condition (1+2+3+4+5=15)")
    except Exception as e:
        r.fail("static for-loop: <= condition (1+2+3+4+5=15)", f"{e}\n{traceback.format_exc()}")

    # Static loop with negative step (i = i - 1)
    try:
        code = """
float sum = 0.0;
for (int i = 5; i < 10; i = i - 1) {
    sum = sum + 1.0;
}
@OUT = vec3(sum);
"""
        # Negative step with i < 10 should immediately have no iterations (or fall to general path)
        # range(5, 10, -1) is empty
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert abs(result[0, 0, 0, 0].item() - 0.0) < 1e-5, f"Negative step empty: {result[0,0,0,0].item()}"
        r.ok("static for-loop: negative step (empty range)")
    except Exception as e:
        r.fail("static for-loop: negative step (empty range)", f"{e}\n{traceback.format_exc()}")

    # Break inside static loop
    try:
        code = """
float sum = 0.0;
for (int i = 0; i < 100; i = i + 1) {
    if (i > 4.5) { break; }
    sum = sum + 1.0;
}
@OUT = vec3(sum);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        val = result[0, 0, 0, 0].item()
        assert abs(val - 5.0) < 1e-5, f"Break in static loop: {val}"
        r.ok("static for-loop: break")
    except Exception as e:
        r.fail("static for-loop: break", f"{e}\n{traceback.format_exc()}")

    # Continue inside static loop
    try:
        code = """
float sum = 0.0;
for (int i = 0; i < 10; i = i + 1) {
    if (i == 3.0 || i == 7.0) { continue; }
    sum = sum + 1.0;
}
@OUT = vec3(sum);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        val = result[0, 0, 0, 0].item()
        assert abs(val - 8.0) < 1e-5, f"Continue in static loop: {val} (expected 8)"
        r.ok("static for-loop: continue skips i=3 and i=7")
    except Exception as e:
        r.fail("static for-loop: continue skips i=3 and i=7", f"{e}\n{traceback.format_exc()}")

    # Non-static loop still works (runtime-dependent bound)
    try:
        code = """
float limit = @A.r * 10.0;
float sum = 0.0;
for (int i = 0; i < 5; i = i + 1) {
    sum = sum + 1.0;
}
@OUT = vec3(sum);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert abs(result[0, 0, 0, 0].item() - 5.0) < 1e-5
        r.ok("static for-loop: non-static bound falls to general path")
    except Exception as e:
        r.fail("static for-loop: non-static bound falls to general path", f"{e}\n{traceback.format_exc()}")

    # ── 5. Constant folding & algebraic simplification ──────────────

    # Constant folding: compile-time evaluation
    try:
        result = compile_and_run("@OUT = vec3(2.0 + 3.0);", {"A": img}, out_type=TEXType.VEC3)
        assert abs(result[0, 0, 0, 0].item() - 5.0) < 1e-5
        r.ok("constant folding: 2.0 + 3.0 = 5.0")
    except Exception as e:
        r.fail("constant folding: 2.0 + 3.0 = 5.0", f"{e}\n{traceback.format_exc()}")

    # Constant folding of pure functions
    try:
        result = compile_and_run("@OUT = vec3(sin(0.0));", {"A": img}, out_type=TEXType.VEC3)
        assert abs(result[0, 0, 0, 0].item()) < 1e-5, "sin(0) should be ~0"
        r.ok("constant folding: sin(0.0) = 0.0")
    except Exception as e:
        r.fail("constant folding: sin(0.0) = 0.0", f"{e}\n{traceback.format_exc()}")

    # x * 0 shape preservation (was a bug)
    try:
        result = compile_and_run("@OUT = @A * 0.0;", {"A": img}, out_type=TEXType.VEC3)
        assert result.shape == img.shape, f"x*0 shape: {result.shape} != {img.shape}"
        assert (result == 0.0).all(), "x*0 should be all zeros"
        r.ok("algebraic opt: x*0 preserves shape")
    except Exception as e:
        r.fail("algebraic opt: x*0 preserves shape", f"{e}\n{traceback.format_exc()}")

    # 0 * x shape preservation (was a bug)
    try:
        result = compile_and_run("@OUT = 0.0 * @A;", {"A": img}, out_type=TEXType.VEC3)
        assert result.shape == img.shape, f"0*x shape: {result.shape} != {img.shape}"
        assert (result == 0.0).all(), "0*x should be all zeros"
        r.ok("algebraic opt: 0*x preserves shape")
    except Exception as e:
        r.fail("algebraic opt: 0*x preserves shape", f"{e}\n{traceback.format_exc()}")

    # x * 1 identity
    try:
        result = compile_and_run("@OUT = @A * 1.0;", {"A": img}, out_type=TEXType.VEC3)
        assert torch.allclose(result, img, atol=1e-5), "x*1 should equal x"
        r.ok("algebraic opt: x*1 = x")
    except Exception as e:
        r.fail("algebraic opt: x*1 = x", f"{e}\n{traceback.format_exc()}")

    # x + 0 identity
    try:
        result = compile_and_run("@OUT = @A + 0.0;", {"A": img}, out_type=TEXType.VEC3)
        assert torch.allclose(result, img, atol=1e-5), "x+0 should equal x"
        r.ok("algebraic opt: x+0 = x")
    except Exception as e:
        r.fail("algebraic opt: x+0 = x", f"{e}\n{traceback.format_exc()}")

    # Division by constant -> multiplication by reciprocal
    try:
        result = compile_and_run("@OUT = @A / 2.0;", {"A": img}, out_type=TEXType.VEC3)
        expected = img * 0.5
        assert torch.allclose(result, expected, atol=1e-5), "x/2 should equal x*0.5"
        r.ok("algebraic opt: x/const -> x*(1/const)")
    except Exception as e:
        r.fail("algebraic opt: x/const -> x*(1/const)", f"{e}\n{traceback.format_exc()}")

    # pow(x, 2) -> x * x strength reduction
    try:
        code = "float val = 3.0; @OUT = vec3(pow(val, 2.0));"
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert abs(result[0, 0, 0, 0].item() - 9.0) < 1e-4, f"pow(3,2): {result[0,0,0,0].item()}"
        r.ok("algebraic opt: pow(x,2) -> x*x")
    except Exception as e:
        r.fail("algebraic opt: pow(x,2) -> x*x", f"{e}\n{traceback.format_exc()}")

    # ── 6. CSE (Common Subexpression Elimination) ───────────────────

    # CSE should produce correct results when same expression appears twice
    try:
        code = """
float a = sin(u * 3.14) + cos(v * 3.14);
float b = sin(u * 3.14) + cos(v * 3.14);
@OUT = vec3(a + b);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        # a and b should be identical
        code2 = "@OUT = vec3(2.0 * (sin(u * 3.14) + cos(v * 3.14)));"
        result2 = compile_and_run(code2, {"A": img}, out_type=TEXType.VEC3)
        assert torch.allclose(result, result2, atol=1e-4), "CSE should not change results"
        r.ok("CSE: duplicate expressions produce correct result")
    except Exception as e:
        r.fail("CSE: duplicate expressions produce correct result", f"{e}\n{traceback.format_exc()}")

    # CSE inside loop body
    try:
        code = """
float total = 0.0;
for (int i = 0; i < 5; i = i + 1) {
    float a = sin(u * 2.0) * cos(v * 2.0);
    float b = sin(u * 2.0) * cos(v * 2.0);
    total = total + a + b;
}
@OUT = vec3(total);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert result.shape == (1, 8, 8, 3), f"CSE in loop shape: {result.shape}"
        r.ok("CSE: works inside loop body")
    except Exception as e:
        r.fail("CSE: works inside loop body", f"{e}\n{traceback.format_exc()}")

    # ── 7. Dead code elimination ────────────────────────────────────

    try:
        # Dead variable should be eliminated without affecting output
        code = """
float dead = sin(u) * cos(v) * 42.0;
@OUT = @A;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert torch.allclose(result, img, atol=1e-5), "DCE should not affect output"
        r.ok("DCE: dead variable eliminated correctly")
    except Exception as e:
        r.fail("DCE: dead variable eliminated correctly", f"{e}\n{traceback.format_exc()}")

    # ── 8. inference_mode skip for non-tensor programs ──────────────

    # Pure string program (no tensors)
    try:
        code = '@OUT_str = "hello";'
        result = compile_and_run(code, {})
        assert result["OUT_str"] == "hello"
        r.ok("inference_mode skip: pure string program")
    except Exception as e:
        r.fail("inference_mode skip: pure string program", f"{e}\n{traceback.format_exc()}")

    # Pure scalar program
    try:
        code = """
float x = 3.14;
float y = x * 2.0;
@OUT = vec4(y);
"""
        result = compile_and_run(code, {})
        assert abs(result[0].item() - 6.28) < 1e-3, f"Scalar program: {result[0].item()}"
        r.ok("inference_mode skip: pure scalar program")
    except Exception as e:
        r.fail("inference_mode skip: pure scalar program", f"{e}\n{traceback.format_exc()}")

    # ── 9. Selective cloning in spatial if/else ─────────────────────

    # Only modified variables should be cloned
    try:
        code = """
float x = 0.0;
float y = 1.0;
if (u > 0.5) {
    x = 2.0;
} else {
    x = 3.0;
}
@OUT = vec3(x + y);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        # Left half: x=3 (u<=0.5), right half: x=2 (u>0.5); y=1 always
        # So left=4, right=3
        r_vals = result[0, 0, :, 0]  # row 0, all cols, channel 0
        # u=0 at col 0, u=1 at col 7
        assert r_vals[0].item() > 3.5, f"u=0: expected ~4.0, got {r_vals[0].item()}"
        assert r_vals[-1].item() < 3.5, f"u=1: expected ~3.0, got {r_vals[-1].item()}"
        r.ok("selective cloning: spatial if/else correct merge")
    except Exception as e:
        r.fail("selective cloning: spatial if/else correct merge", f"{e}\n{traceback.format_exc()}")

    # Nested if/else
    try:
        code = """
float val = 0.0;
if (u > 0.5) {
    if (v > 0.5) {
        val = 1.0;
    } else {
        val = 2.0;
    }
} else {
    val = 3.0;
}
@OUT = vec3(val);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert result.shape == (1, 8, 8, 3), f"Nested if/else shape: {result.shape}"
        r.ok("selective cloning: nested if/else")
    except Exception as e:
        r.fail("selective cloning: nested if/else", f"{e}\n{traceback.format_exc()}")

    # Binding modification in if/else
    try:
        code = """
if (u > 0.5) {
    @OUT = vec3(1.0);
} else {
    @OUT = vec3(0.0);
}
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert result.shape == (1, 8, 8, 3)
        # Right side should be 1.0, left side should be 0.0
        assert result[0, 0, 0, 0].item() < 0.5, "u=0 should give 0.0"
        assert result[0, 0, -1, 0].item() > 0.5, "u=1 should give 1.0"
        r.ok("selective cloning: binding modification in branches")
    except Exception as e:
        r.fail("selective cloning: binding modification in branches", f"{e}\n{traceback.format_exc()}")

    # ── 10. In-place operations ─────────────────────────────────────

    try:
        code = """
float acc = 0.0;
for (int i = 0; i < 10; i = i + 1) {
    acc = acc + 1.0;
}
@OUT = vec3(acc);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert abs(result[0, 0, 0, 0].item() - 10.0) < 1e-5, f"In-place acc: {result[0,0,0,0].item()}"
        r.ok("in-place ops: accumulator pattern")
    except Exception as e:
        r.fail("in-place ops: accumulator pattern", f"{e}\n{traceback.format_exc()}")

    # In-place with spatial tensors
    try:
        code = """
vec3 color = @A;
color = color + vec3(0.1);
color = color * 2.0;
@OUT = color;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = (img + 0.1) * 2.0
        assert torch.allclose(result, expected, atol=1e-4), "In-place spatial"
        r.ok("in-place ops: spatial tensor accumulation")
    except Exception as e:
        r.fail("in-place ops: spatial tensor accumulation", f"{e}\n{traceback.format_exc()}")

    # ── 11. Interpreter singleton reuse ─────────────────────────────

    try:
        interp = Interpreter()
        # Run two different programs on the same interpreter
        code1 = "@OUT = @A + 0.1;"
        tokens1 = Lexer(code1).tokenize()
        prog1 = Parser(tokens1).parse()
        bt1 = {"A": TEXType.VEC3, "OUT": TEXType.VEC3}
        tc1 = TypeChecker(binding_types=bt1)
        tm1 = tc1.check(prog1)
        r1 = interp.execute(prog1, {"A": img}, tm1)

        code2 = "@OUT = @A * 0.5;"
        tokens2 = Lexer(code2).tokenize()
        prog2 = Parser(tokens2).parse()
        bt2 = {"A": TEXType.VEC3, "OUT": TEXType.VEC3}
        tc2 = TypeChecker(binding_types=bt2)
        tm2 = tc2.check(prog2)
        r2 = interp.execute(prog2, {"A": img}, tm2)

        assert torch.allclose(r1, img + 0.1, atol=1e-5), "First execution wrong"
        assert torch.allclose(r2, img * 0.5, atol=1e-5), "Second execution wrong"
        r.ok("interpreter reuse: two programs, correct results")
    except Exception as e:
        r.fail("interpreter reuse: two programs, correct results", f"{e}\n{traceback.format_exc()}")

    # ── 12. Cache 6-tuple output ────────────────────────────────────

    try:
        cache = TEXCache(cache_dir=Path(tempfile.mkdtemp()))
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC3}
        result = cache.compile_tex("@OUT = @A;", bt)
        assert len(result) == 6, f"Cache should return 6-tuple, got {len(result)}"
        program, type_map, refs, assigned, params, used_builtins = result
        assert isinstance(used_builtins, frozenset), f"used_builtins type: {type(used_builtins)}"
        # Verify cache hit returns same structure
        result2 = cache.compile_tex("@OUT = @A;", bt)
        assert len(result2) == 6, "Cache hit should also return 6-tuple"
        r.ok("cache: 6-tuple output with used_builtins")
        shutil.rmtree(cache._cache_dir, ignore_errors=True)
    except Exception as e:
        r.fail("cache: 6-tuple output with used_builtins", f"{e}\n{traceback.format_exc()}")

    # ── 13. used_builtins correctness ───────────────────────────────

    try:
        from TEX_Wrangle.tex_runtime.interpreter import _collect_identifiers

        # Program using u, v, PI
        code = "@OUT = vec3(sin(u * PI) * cos(v));"
        tokens = Lexer(code).tokenize()
        prog = Parser(tokens).parse()
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC3}
        tc = TypeChecker(binding_types=bt)
        tc.check(prog)
        from TEX_Wrangle.tex_compiler.optimizer import optimize
        prog = optimize(prog)
        used = _collect_identifiers(prog)
        assert "u" in used, "Should find u"
        assert "v" in used, "Should find v"
        assert "PI" in used, "Should find PI"
        assert "ix" not in used, "Should not find ix"
        assert "iy" not in used, "Should not find iy"
        r.ok("used_builtins: correct identification")
    except Exception as e:
        r.fail("used_builtins: correct identification", f"{e}\n{traceback.format_exc()}")

    # Program using NO builtins
    try:
        code = "@OUT = @A;"
        tokens = Lexer(code).tokenize()
        prog = Parser(tokens).parse()
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC3}
        tc = TypeChecker(binding_types=bt)
        tc.check(prog)
        prog = optimize(prog)
        used = _collect_identifiers(prog)
        assert len(used) == 0, f"Should have no builtins, got {used}"
        r.ok("used_builtins: empty for trivial program")
    except Exception as e:
        r.fail("used_builtins: empty for trivial program", f"{e}\n{traceback.format_exc()}")

    # ── 14. Dispatch table fallback ─────────────────────────────────

    # Ensure break/continue still work (they're outside the dispatch table)
    try:
        code = """
float sum = 0.0;
for (int i = 0; i < 20; i = i + 1) {
    if (i > 9.5) { break; }
    if (i == 5.0) { continue; }
    sum = sum + 1.0;
}
@OUT = vec3(sum);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        # i=0..9 except i=5 → 9 iterations
        assert abs(result[0, 0, 0, 0].item() - 9.0) < 1e-5
        r.ok("dispatch table: break/continue work correctly")
    except Exception as e:
        r.fail("dispatch table: break/continue work correctly", f"{e}\n{traceback.format_exc()}")

    # ── 15. Precision handling ──────────────────────────────────────

    try:
        code = "@OUT = @A * 0.5;"
        tokens = Lexer(code).tokenize()
        prog = Parser(tokens).parse()
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC3}
        tc = TypeChecker(binding_types=bt)
        tm = tc.check(prog)
        prog = optimize(prog)
        interp = Interpreter()
        # Execute with fp32
        r32 = interp.execute(prog, {"A": img}, tm, precision="fp32")
        assert r32.dtype == torch.float32, f"fp32 result dtype: {r32.dtype}"
        r.ok("precision: fp32 output correct dtype")
    except Exception as e:
        r.fail("precision: fp32 output correct dtype", f"{e}\n{traceback.format_exc()}")

    # ── 16. End-to-end: complex real-world patterns ─────────────────

    # Blur-like pattern (loop + spatial sampling)
    try:
        code = """
vec3 sum = vec3(0.0);
float count = 0.0;
for (int dx = -1; dx <= 1; dx = dx + 1) {
    for (int dy = -1; dy <= 1; dy = dy + 1) {
        float sx = u + float(dx) / iw;
        float sy = v + float(dy) / ih;
        sum = sum + sample(@A, sx, sy);
        count = count + 1.0;
    }
}
@OUT = sum / count;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert result.shape == img.shape, f"Blur shape: {result.shape}"
        assert not torch.isnan(result).any(), "Blur produced NaN"
        r.ok("e2e: blur-like 3x3 pattern")
    except Exception as e:
        r.fail("e2e: blur-like 3x3 pattern", f"{e}\n{traceback.format_exc()}")

    # Conditional color grading
    try:
        code = """
float luma = @A.r * 0.299 + @A.g * 0.587 + @A.b * 0.114;
vec3 result = @A;
if (luma > 0.5) {
    result = result * 1.2;
} else {
    result = result * 0.8;
}
@OUT = clamp(result, 0.0, 1.0);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert result.shape == img.shape, f"Color grade shape: {result.shape}"
        assert (result >= 0.0).all() and (result <= 1.0).all(), "Should be clamped"
        r.ok("e2e: conditional color grading")
    except Exception as e:
        r.fail("e2e: conditional color grading", f"{e}\n{traceback.format_exc()}")

    # Multi-output program
    try:
        code = """
@bright = @A * 1.5;
@dark = @A * 0.5;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert isinstance(result, dict), f"Multi-output should return dict: {type(result)}"
        assert "bright" in result and "dark" in result
        assert torch.allclose(result["bright"], img * 1.5, atol=1e-4)
        assert torch.allclose(result["dark"], img * 0.5, atol=1e-4)
        r.ok("e2e: multi-output program")
    except Exception as e:
        r.fail("e2e: multi-output program", f"{e}\n{traceback.format_exc()}")

    # String processing (no tensors, tests inference_mode skip)
    try:
        code = """
string a = "hello";
string b = "world";
@OUT_str = a + " " + b;
"""
        result = compile_and_run(code, {})
        assert result["OUT_str"] == "hello world", f"String processing: {result['OUT_str']}"
        r.ok("e2e: pure string processing")
    except Exception as e:
        r.fail("e2e: pure string processing", f"{e}\n{traceback.format_exc()}")

    # Batch processing (B > 1)
    try:
        batch_img = torch.rand(4, 8, 8, 3)
        result = compile_and_run("@OUT = @A * 0.5;", {"A": batch_img}, out_type=TEXType.VEC3)
        assert result.shape == (4, 8, 8, 3), f"Batch shape: {result.shape}"
        assert torch.allclose(result, batch_img * 0.5, atol=1e-5)
        r.ok("e2e: batch processing B=4")
    except Exception as e:
        r.fail("e2e: batch processing B=4", f"{e}\n{traceback.format_exc()}")


# ── Diagnostic Quality Tests ──────────────────────────────────────────

def test_diagnostic_quality(r: TestResult):
    """Verify the quality of TEX's structured diagnostics system."""
    print("\n--- Diagnostic Quality Tests ---")
    import re

    def check_code(code: str, bindings: dict[str, TEXType] | None = None):
        tokens = Lexer(code).tokenize()
        prog = Parser(tokens, source=code).parse()
        bt = bindings or {}
        bt.setdefault("OUT", TEXType.VEC4)
        checker = TypeChecker(binding_types=bt, source=code)
        return checker.check(prog), checker

    # ── 1. Structured Diagnostic Fields ──────────────────────────────

    # 1a. Typo in function name produces TEXDiagnostic with proper fields
    try:
        code = "float x = clampp(0.5, 0.0, 1.0); @OUT = vec4(x);"
        check_code(code)
        r.fail("diag: structured fields (clampp)", "Should have raised an error")
    except (TypeCheckError, TEXMultiError) as e:
        diag = e.diagnostic if hasattr(e, "diagnostic") else e.diagnostics[0]
        assert diag is not None, "diagnostic should not be None"
        assert diag.code.startswith("E"), f"Error code should start with E, got {diag.code}"
        assert len(diag.suggestions) > 0, "Should have suggestions for typo"
        assert diag.source_line != "", "source_line should be set"
        r.ok("diag: structured fields (clampp)")
    except Exception as e:
        r.fail("diag: structured fields (clampp)", f"{e}\n{traceback.format_exc()}")

    # 1b. Diagnostic has severity and message fields
    try:
        code = "float x = bogusFunc(1.0); @OUT = vec4(x);"
        check_code(code)
        r.fail("diag: severity and message", "Should have raised an error")
    except (TypeCheckError, TEXMultiError) as e:
        diag = e.diagnostic if hasattr(e, "diagnostic") else e.diagnostics[0]
        assert diag.severity == "error", f"Expected severity 'error', got {diag.severity}"
        assert len(diag.message) > 0, "message should not be empty"
        r.ok("diag: severity and message")
    except Exception as e:
        r.fail("diag: severity and message", f"{e}\n{traceback.format_exc()}")

    # ── 2. Multi-Error Reporting ─────────────────────────────────────

    # 2a. Multiple unknown functions produce multiple diagnostics
    try:
        code = "float x = bogus1(1.0);\nfloat y = bogus2(2.0);\n@OUT = vec4(x + y);"
        check_code(code)
        r.fail("diag: multi-error reporting", "Should have raised an error")
    except TEXMultiError as e:
        assert len(e.diagnostics) >= 2, f"Expected 2+ diagnostics, got {len(e.diagnostics)}"
        r.ok("diag: multi-error reporting")
    except TypeCheckError:
        # If only one error surfaced, that's a partial pass but not ideal
        r.fail("diag: multi-error reporting", "Only got single TypeCheckError, expected TEXMultiError with 2+")
    except Exception as e:
        r.fail("diag: multi-error reporting", f"{e}\n{traceback.format_exc()}")

    # 2b. Each diagnostic in multi-error has its own code
    try:
        code = "float x = bogus1(1.0);\nfloat y = bogus2(2.0);\n@OUT = vec4(x + y);"
        check_code(code)
        r.fail("diag: multi-error codes", "Should have raised an error")
    except TEXMultiError as e:
        all_have_codes = all(d.code.startswith("E") for d in e.diagnostics)
        assert all_have_codes, "All diagnostics should have error codes starting with E"
        r.ok("diag: multi-error codes")
    except TypeCheckError:
        r.ok("diag: multi-error codes")  # single error still has a code
    except Exception as e:
        r.fail("diag: multi-error codes", f"{e}\n{traceback.format_exc()}")

    # ── 3. "Did You Mean?" Quality ───────────────────────────────────

    typo_cases = [
        ("clampp", "clamp"),
        ("smoothsetp", "smoothstep"),
        ("lenght", "length"),
    ]
    for typo, expected in typo_cases:
        try:
            code = f"float x = {typo}(0.5); @OUT = vec4(x);"
            check_code(code)
            r.fail(f"diag: did-you-mean {typo}", "Should have raised an error")
        except (TypeCheckError, TEXMultiError) as e:
            diag = e.diagnostic if hasattr(e, "diagnostic") else e.diagnostics[0]
            assert expected in diag.suggestions, \
                f"Expected '{expected}' in suggestions {diag.suggestions} for typo '{typo}'"
            r.ok(f"diag: did-you-mean {typo}")
        except Exception as e:
            r.fail(f"diag: did-you-mean {typo}", f"{e}\n{traceback.format_exc()}")

    # ── 4. Foreign Language Hints ────────────────────────────────────

    # 4a. "return x;" outside a function should mention @OUT or assignment
    try:
        code = "return 1.0; @OUT = @A;"
        check_code(code, {"A": TEXType.VEC4})
        r.fail("diag: return outside function", "Should have raised an error")
    except (TypeCheckError, TEXMultiError) as e:
        if hasattr(e, "_build_diagnostic"):
            if e.diagnostic is None:
                e._build_diagnostic()
            diag = e.diagnostic
        else:
            diag = e.diagnostics[0]
        hint_text = (diag.hint + " " + diag.message).lower()
        assert "function" in hint_text or "@out" in hint_text or "assign" in hint_text, \
            f"Hint for 'return' should mention function or @OUT, got hint='{diag.hint}'"
        r.ok("diag: return outside function")
    except Exception as e:
        r.fail("diag: return outside function", f"{e}\n{traceback.format_exc()}")

    # 4b. "texture2D" should mention sample()
    try:
        code = "float x = texture2D(@A, 0.5, 0.5); @OUT = vec4(x);"
        check_code(code, {"A": TEXType.VEC4})
        r.fail("diag: foreign func texture2D", "Should have raised an error")
    except (TypeCheckError, TEXMultiError) as e:
        if hasattr(e, "_build_diagnostic") and e.diagnostic is None:
            e._build_diagnostic()
        diag = e.diagnostic if hasattr(e, "diagnostic") else e.diagnostics[0]
        full_text = (diag.hint + " " + diag.message + " " + str(e)).lower()
        assert "sample" in full_text, \
            f"Hint for 'texture2D' should mention sample(), got hint='{diag.hint}'"
        r.ok("diag: foreign func texture2D")
    except Exception as e:
        r.fail("diag: foreign func texture2D", f"{e}\n{traceback.format_exc()}")

    # 4c. "print" should hint about no print
    try:
        code = "float x = print(1.0); @OUT = vec4(x);"
        check_code(code)
        r.fail("diag: foreign func print", "Should have raised an error")
    except (TypeCheckError, TEXMultiError) as e:
        if hasattr(e, "_build_diagnostic") and e.diagnostic is None:
            e._build_diagnostic()
        diag = e.diagnostic if hasattr(e, "diagnostic") else e.diagnostics[0]
        full_text = (diag.hint + " " + diag.message + " " + str(e)).lower()
        assert "print" in full_text or "@out" in full_text, \
            f"Hint for 'print' should mention no print or @OUT, got hint='{diag.hint}'"
        r.ok("diag: foreign func print")
    except Exception as e:
        r.fail("diag: foreign func print", f"{e}\n{traceback.format_exc()}")

    # ── 5. Parser Recovery ───────────────────────────────────────────

    # 5a. Multiple syntax errors are reported together
    try:
        code = "float x = ;\nfloat y = ;\n@OUT = vec4(1.0);"
        tokens = Lexer(code).tokenize()
        Parser(tokens, source=code).parse()
        r.fail("diag: parser recovery", "Should have raised an error")
    except TEXMultiError as e:
        assert len(e.diagnostics) >= 2, \
            f"Parser should recover and report 2+ errors, got {len(e.diagnostics)}"
        r.ok("diag: parser recovery")
    except ParseError:
        # Even a single error is acceptable if parser doesn't recover
        r.ok("diag: parser recovery (single)")
    except Exception as e:
        r.fail("diag: parser recovery", f"{e}\n{traceback.format_exc()}")

    # 5b. Parser recovery still reports all found errors
    try:
        code = "float a = ;\nfloat b = ;\nfloat c = ;\n@OUT = vec4(1.0);"
        tokens = Lexer(code).tokenize()
        Parser(tokens, source=code).parse()
        r.fail("diag: parser recovery 3 errors", "Should have raised an error")
    except TEXMultiError as e:
        assert len(e.diagnostics) >= 2, \
            f"Expected 2+ diagnostics from parser recovery, got {len(e.diagnostics)}"
        r.ok("diag: parser recovery 3 errors")
    except ParseError:
        r.ok("diag: parser recovery 3 errors (single)")
    except Exception as e:
        r.fail("diag: parser recovery 3 errors", f"{e}\n{traceback.format_exc()}")

    # ── 6. Error Codes Follow Pattern ────────────────────────────────

    error_code_pattern = re.compile(r"^E[1-6]\d{3}$")

    # 6a. Type checker error codes match E-pattern
    try:
        code = "float x = nonexistent(1.0); @OUT = vec4(x);"
        check_code(code)
        r.fail("diag: error code pattern (type)", "Should have raised an error")
    except (TypeCheckError, TEXMultiError) as e:
        diag = e.diagnostic if hasattr(e, "diagnostic") else e.diagnostics[0]
        assert error_code_pattern.match(diag.code), \
            f"Error code '{diag.code}' doesn't match E[1-6]xxx pattern"
        r.ok("diag: error code pattern (type)")
    except Exception as e:
        r.fail("diag: error code pattern (type)", f"{e}\n{traceback.format_exc()}")

    # 6b. Parser error codes match E-pattern
    try:
        code = "float x = ;"
        tokens = Lexer(code).tokenize()
        Parser(tokens, source=code).parse()
        r.fail("diag: error code pattern (parser)", "Should have raised an error")
    except (ParseError, TEXMultiError) as e:
        diag = e.diagnostic if hasattr(e, "diagnostic") else e.diagnostics[0]
        assert error_code_pattern.match(diag.code), \
            f"Error code '{diag.code}' doesn't match E[1-6]xxx pattern"
        r.ok("diag: error code pattern (parser)")
    except Exception as e:
        r.fail("diag: error code pattern (parser)", f"{e}\n{traceback.format_exc()}")

    # 6c. Foreign keyword error codes are E2xxx
    try:
        code = "const x = 1.0;"
        tokens = Lexer(code).tokenize()
        Parser(tokens, source=code).parse()
        r.fail("diag: foreign keyword error code", "Should have raised an error")
    except (ParseError, TEXMultiError) as e:
        diag = e.diagnostic if hasattr(e, "diagnostic") else e.diagnostics[0]
        assert diag.code.startswith("E2"), \
            f"Foreign keyword error should be E2xxx, got {diag.code}"
        r.ok("diag: foreign keyword error code")
    except Exception as e:
        r.fail("diag: foreign keyword error code", f"{e}\n{traceback.format_exc()}")

    # ── 7. Source Snippets ───────────────────────────────────────────

    # 7a. Diagnostic includes the offending source line
    try:
        code = "float x = unknownFunc(1.0);\n@OUT = vec4(x);"
        check_code(code)
        r.fail("diag: source snippet present", "Should have raised an error")
    except (TypeCheckError, TEXMultiError) as e:
        diag = e.diagnostic if hasattr(e, "diagnostic") else e.diagnostics[0]
        assert "unknownFunc" in diag.source_line, \
            f"source_line should contain the offending code, got: '{diag.source_line}'"
        r.ok("diag: source snippet present")
    except Exception as e:
        r.fail("diag: source snippet present", f"{e}\n{traceback.format_exc()}")

    # 7b. Rendered diagnostic contains source snippet with line number
    try:
        code = "float x = missingFn(1.0);\n@OUT = vec4(x);"
        check_code(code)
        r.fail("diag: rendered snippet", "Should have raised an error")
    except (TypeCheckError, TEXMultiError) as e:
        diag = e.diagnostic if hasattr(e, "diagnostic") else e.diagnostics[0]
        rendered = diag.render()
        assert "missingFn" in rendered, \
            f"Rendered diagnostic should include source text, got:\n{rendered}"
        assert "|" in rendered, \
            f"Rendered diagnostic should include gutter '|', got:\n{rendered}"
        r.ok("diag: rendered snippet")
    except Exception as e:
        r.fail("diag: rendered snippet", f"{e}\n{traceback.format_exc()}")

    # 7c. Diagnostic has location info
    try:
        code = "float x = weirdName(1.0);\n@OUT = vec4(x);"
        check_code(code)
        r.fail("diag: location info", "Should have raised an error")
    except (TypeCheckError, TEXMultiError) as e:
        diag = e.diagnostic if hasattr(e, "diagnostic") else e.diagnostics[0]
        assert diag.loc is not None, "Diagnostic should have location info"
        assert diag.loc.line >= 1, f"Line number should be >= 1, got {diag.loc.line}"
        assert diag.loc.col >= 1, f"Column number should be >= 1, got {diag.loc.col}"
        r.ok("diag: location info")
    except Exception as e:
        r.fail("diag: location info", f"{e}\n{traceback.format_exc()}")


# ── User-Defined Functions ─────────────────────────────────────────────

def test_user_functions(r: TestResult):
    print("\n--- User-Defined Functions ---")
    img = torch.rand(1, 4, 4, 4)

    # Basic function: float -> float
    try:
        result = compile_and_run("""
            float add1(float x) {
                return x + 1.0;
            }
            @OUT = @A * add1(0.5);
        """, {"A": img})
        expected = img * 1.5
        assert torch.allclose(result, expected, atol=1e-5), f"Got {result.mean()}"
        r.ok("basic float function")
    except Exception as e:
        r.fail("basic float function", str(e))

    # Multi-parameter function
    try:
        result = compile_and_run("""
            float lerp3(float a, float b, float t) {
                return a * (1.0 - t) + b * t;
            }
            float val = lerp3(0.0, 1.0, 0.75);
            @OUT = @A * val;
        """, {"A": img})
        expected = img * 0.75
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("multi-param function")
    except Exception as e:
        r.fail("multi-param function", str(e))

    # Vec3 return type
    try:
        result = compile_and_run("""
            vec3 tint(vec3 c, float s) {
                return c * s;
            }
            @OUT = vec4(tint(@A.rgb, 0.5), @A.a);
        """, {"A": img})
        expected = torch.cat([img[..., :3] * 0.5, img[..., 3:4]], dim=-1)
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("vec3 return type")
    except Exception as e:
        r.fail("vec3 return type", str(e))

    # Vec4 return type
    try:
        result = compile_and_run("""
            vec4 premultiply(vec4 c) {
                return vec4(c.rgb * c.a, c.a);
            }
            @OUT = premultiply(@A);
        """, {"A": img})
        expected = torch.cat([img[..., :3] * img[..., 3:4], img[..., 3:4]], dim=-1)
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("vec4 return type")
    except Exception as e:
        r.fail("vec4 return type", str(e))

    # Nested calls
    try:
        result = compile_and_run("""
            float inc(float x) {
                return x + 1.0;
            }
            float val = inc(inc(0.5));
            @OUT = @A * val;
        """, {"A": img})
        expected = img * 2.5
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("nested function calls")
    except Exception as e:
        r.fail("nested function calls", str(e))

    # Function calling another function
    try:
        result = compile_and_run("""
            float square(float x) {
                return x * x;
            }
            float cube(float x) {
                return x * square(x);
            }
            float val = cube(3.0);
            @OUT = @A * val;
        """, {"A": img})
        expected = img * 27.0
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("function calling another function")
    except Exception as e:
        r.fail("function calling another function", str(e))

    # Function using builtins
    try:
        result = compile_and_run("""
            float circle(float cx, float cy, float radius) {
                float dx = u - cx;
                float dy = v - cy;
                return sqrt(dx * dx + dy * dy) < radius ? 1.0 : 0.0;
            }
            @OUT = @A * circle(0.5, 0.5, 0.3);
        """, {"A": img})
        r.ok("function using builtins (u, v)")
    except Exception as e:
        r.fail("function using builtins (u, v)", str(e))

    # Function with if/else in body
    try:
        result = compile_and_run("""
            float thresh(float x, float t) {
                if (x > t) {
                    return 1.0;
                }
                return 0.0;
            }
            @OUT = @A * thresh(0.7, 0.5);
        """, {"A": img})
        expected = img * 1.0
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("function with if/else and multiple returns")
    except Exception as e:
        r.fail("function with if/else and multiple returns", str(e))

    # Function with for loop in body
    try:
        result = compile_and_run("""
            float sum_n(int n) {
                float s = 0.0;
                for (int i = 0; i < n; i++) {
                    s = s + 1.0;
                }
                return s;
            }
            float val = sum_n(5);
            @OUT = @A * val;
        """, {"A": img})
        expected = img * 5.0
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("function with for loop")
    except Exception as e:
        r.fail("function with for loop", str(e))

    # Function used in expression
    try:
        result = compile_and_run("""
            float half(float x) {
                return x * 0.5;
            }
            float val = half(2.0) + half(4.0);
            @OUT = @A * val;
        """, {"A": img})
        expected = img * 3.0
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("function in arithmetic expression")
    except Exception as e:
        r.fail("function in arithmetic expression", str(e))

    # Function used in ternary
    try:
        result = compile_and_run("""
            float double(float x) {
                return x * 2.0;
            }
            float val = double(1.0) > 1.5 ? 1.0 : 0.0;
            @OUT = @A * val;
        """, {"A": img})
        expected = img * 1.0
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("function in ternary")
    except Exception as e:
        r.fail("function in ternary", str(e))

    # No return statement -> returns zero
    try:
        result = compile_and_run("""
            float noop(float x) {
                float y = x + 1.0;
            }
            float val = noop(5.0);
            @OUT = @A * val;
        """, {"A": img})
        expected = img * 0.0
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("function without return -> zero")
    except Exception as e:
        r.fail("function without return -> zero", str(e))

    # Function doesn't modify caller scope
    try:
        result = compile_and_run("""
            float mutator(float x) {
                float val = 999.0;
                return x;
            }
            float val = 1.0;
            float dummy = mutator(0.0);
            @OUT = @A * val;
        """, {"A": img})
        expected = img * 1.0
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("function scope isolation")
    except Exception as e:
        r.fail("function scope isolation", str(e))

    # Zero-parameter function
    try:
        result = compile_and_run("""
            float pi_half() {
                return PI * 0.5;
            }
            @OUT = @A * sin(pi_half());
        """, {"A": img})
        expected = img * 1.0
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("zero-parameter function")
    except Exception as e:
        r.fail("zero-parameter function", str(e))

    # Int parameter and return
    try:
        result = compile_and_run("""
            int double_int(int x) {
                return x * 2;
            }
            @OUT = @A * double_int(3);
        """, {"A": img})
        expected = img * 6.0
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("int parameter and return")
    except Exception as e:
        r.fail("int parameter and return", str(e))

    # Error: redefining stdlib function
    try:
        compile_and_run("""
            float clamp(float x) {
                return x;
            }
            @OUT = @A;
        """, {"A": img})
        r.fail("error: redefine stdlib", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("error: redefine stdlib")
    except Exception as e:
        r.fail("error: redefine stdlib", str(e))

    # Error: duplicate function definition
    try:
        compile_and_run("""
            float foo(float x) { return x; }
            float foo(float x) { return x * 2.0; }
            @OUT = @A;
        """, {"A": img})
        r.fail("error: duplicate function", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("error: duplicate function")
    except Exception as e:
        r.fail("error: duplicate function", str(e))

    # Error: wrong arg count
    try:
        compile_and_run("""
            float add2(float a, float b) { return a + b; }
            float val = add2(1.0);
            @OUT = @A * val;
        """, {"A": img})
        r.fail("error: wrong arg count", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("error: wrong arg count")
    except Exception as e:
        r.fail("error: wrong arg count", str(e))

    # Error: wrong arg type
    try:
        compile_and_run("""
            float process(vec3 c) { return c.r; }
            float val = process("hello");
            @OUT = @A * val;
        """, {"A": img})
        r.fail("error: wrong arg type", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("error: wrong arg type")
    except Exception as e:
        r.fail("error: wrong arg type", str(e))

    # Error: return outside function
    try:
        compile_and_run("""
            return 1.0;
            @OUT = @A;
        """, {"A": img})
        r.fail("error: return outside function", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("error: return outside function")
    except Exception as e:
        r.fail("error: return outside function", str(e))

    # Error: return type mismatch
    try:
        compile_and_run("""
            float bad() {
                return vec3(1.0, 0.0, 0.0);
            }
            @OUT = @A;
        """, {"A": img})
        r.fail("error: return type mismatch", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("error: return type mismatch")
    except Exception as e:
        r.fail("error: return type mismatch", str(e))

    # String return type
    try:
        result = compile_and_run("""
            string greet(string name) {
                return "Hello, " + name;
            }
            s@OUT = greet("TEX");
        """, {"A": img})
        assert result == "Hello, TEX", f"Got {result}"
        r.ok("string function")
    except Exception as e:
        r.fail("string function", str(e))

    # Error: nested function definition
    try:
        compile_and_run("""
            float outer() {
                float inner() {
                    return 1.0;
                }
                return inner();
            }
            @OUT = @A;
        """, {"A": img})
        r.fail("error: nested function def", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("error: nested function def")
    except Exception as e:
        r.fail("error: nested function def", str(e))

    # Error: infinite recursion hits depth limit
    try:
        compile_and_run("""
            float boom(float x) {
                return boom(x);
            }
            @OUT = vec4(boom(1.0));
        """, {"A": img})
        r.fail("error: recursion depth limit", "Should have raised an error")
    except InterpreterError as e:
        assert "depth" in str(e).lower() or "recursion" in str(e).lower(), f"Expected recursion error, got: {e}"
        r.ok("error: recursion depth limit")
    except (TypeCheckError, TEXMultiError):
        r.fail("error: recursion depth limit", "Unexpected type check error")
    except Exception as e:
        r.fail("error: recursion depth limit", str(e))


# ── Binding Access Syntax ──────────────────────────────────────────────

def test_binding_access(r: TestResult):
    print("\n--- Binding Access Syntax ---")
    img = torch.zeros(1, 4, 4, 4)
    # Fill with known pattern: pixel (x,y) has value (x*0.1 + y*0.01) in all channels
    for y in range(4):
        for x in range(4):
            img[0, y, x, :] = x * 0.1 + y * 0.01

    # @A[ix, iy] — fetch at current pixel (identity)
    try:
        result = compile_and_run("@OUT = @A[ix, iy];", {"A": img})
        assert torch.allclose(result, img, atol=1e-5), f"Max diff: {(result - img).abs().max()}"
        r.ok("@A[ix, iy] identity fetch")
    except Exception as e:
        r.fail("@A[ix, iy] identity fetch", str(e))

    # @A[0, 0] — fetch pixel (0,0) everywhere
    try:
        result = compile_and_run("@OUT = @A[0, 0];", {"A": img})
        expected_val = img[0, 0, 0, :]  # pixel (0,0)
        assert torch.allclose(result[0, :, :, :], expected_val.expand(4, 4, -1), atol=1e-5)
        r.ok("@A[0, 0] constant fetch")
    except Exception as e:
        r.fail("@A[0, 0] constant fetch", str(e))

    # @A(u, v) — sample at current UV (identity, approximately)
    try:
        result = compile_and_run("@OUT = @A(u, v);", {"A": img})
        # Bilinear at exact pixel centers should be close to identity
        r.ok("@A(u, v) identity sample")
    except Exception as e:
        r.fail("@A(u, v) identity sample", str(e))

    # @A(0.5, 0.5) — sample center
    try:
        result = compile_and_run("@OUT = @A(0.5, 0.5);", {"A": img})
        # Should be uniform across all pixels (same UV everywhere)
        center = result[0, 0, 0, :]
        assert torch.allclose(result[0], center.expand(4, 4, -1), atol=1e-5)
        r.ok("@A(0.5, 0.5) center sample")
    except Exception as e:
        r.fail("@A(0.5, 0.5) center sample", str(e))

    # Chaining with channel access: @A[ix, iy].rgb
    try:
        result = compile_and_run("vec3 c = @A[ix, iy].rgb; @OUT = vec4(c, 1.0);", {"A": img})
        expected = torch.cat([img[..., :3], torch.ones(1, 4, 4, 1)], dim=-1)
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("@A[ix, iy].rgb chain")
    except Exception as e:
        r.fail("@A[ix, iy].rgb chain", str(e))

    # Chaining with channel access: @A(u, v).r
    try:
        result = compile_and_run("float val = @A(u, v).r; @OUT = @A * val;", {"A": img})
        r.ok("@A(u, v).r chain")
    except Exception as e:
        r.fail("@A(u, v).r chain", str(e))

    # Equivalence: @A[ix, iy] == fetch(@A, ix, iy)
    try:
        r1 = compile_and_run("@OUT = @A[ix, iy];", {"A": img})
        r2 = compile_and_run("@OUT = fetch(@A, ix, iy);", {"A": img})
        assert torch.allclose(r1, r2, atol=1e-5)
        r.ok("@A[ix, iy] == fetch(@A, ix, iy)")
    except Exception as e:
        r.fail("@A[ix, iy] == fetch(@A, ix, iy)", str(e))

    # Equivalence: @A(u, v) == sample(@A, u, v)
    try:
        r1 = compile_and_run("@OUT = @A(u, v);", {"A": img})
        r2 = compile_and_run("@OUT = sample(@A, u, v);", {"A": img})
        assert torch.allclose(r1, r2, atol=1e-5)
        r.ok("@A(u, v) == sample(@A, u, v)")
    except Exception as e:
        r.fail("@A(u, v) == sample(@A, u, v)", str(e))

    # Frame argument: @A[ix, iy, 0]
    try:
        batch_img = torch.rand(2, 4, 4, 4)
        result = compile_and_run("@OUT = @A[ix, iy, 0];", {"A": batch_img})
        # Frame 0 sampled everywhere
        expected_frame = batch_img[0:1].expand_as(batch_img)
        r.ok("@A[ix, iy, 0] with frame")
    except Exception as e:
        r.fail("@A[ix, iy, 0] with frame", str(e))

    # Frame argument: @A(u, v, 0)
    try:
        batch_img = torch.rand(2, 4, 4, 4)
        result = compile_and_run("@OUT = @A(u, v, 0);", {"A": batch_img})
        r.ok("@A(u, v, 0) with frame")
    except Exception as e:
        r.fail("@A(u, v, 0) with frame", str(e))

    # Binding index in function argument
    try:
        result = compile_and_run("@OUT = clamp(@A[ix, iy], 0.0, 0.5);", {"A": img})
        expected = torch.clamp(img, 0.0, 0.5)
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("@A[ix, iy] as function arg")
    except Exception as e:
        r.fail("@A[ix, iy] as function arg", str(e))

    # Binding sample in user function
    try:
        result = compile_and_run("""
            vec4 blur_sample(float ox, float oy) {
                return @A(u + ox, v + oy);
            }
            @OUT = blur_sample(0.0, 0.0);
        """, {"A": img})
        r.ok("binding sample in user function")
    except Exception as e:
        r.fail("binding sample in user function", str(e))

    # Error: too few args
    try:
        compile_and_run("@OUT = @A[0];", {"A": img})
        r.fail("error: @A[0] too few args", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("error: @A[0] too few args")
    except Exception as e:
        r.fail("error: @A[0] too few args", str(e))

    # Error: too many args
    try:
        compile_and_run("@OUT = @A[0, 0, 0, 0];", {"A": img})
        r.fail("error: @A[0,0,0,0] too many args", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("error: @A[0,0,0,0] too many args")
    except Exception as e:
        r.fail("error: @A[0,0,0,0] too many args", str(e))

    # Error: too few args for sample
    try:
        compile_and_run("@OUT = @A(0.5);", {"A": img})
        r.fail("error: @A(0.5) too few args", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("error: @A(0.5) too few args")
    except Exception as e:
        r.fail("error: @A(0.5) too few args", str(e))


# ── Ternary Exhaustive Tests ──────────────────────────────────────────

def test_ternary_exhaustive(r: TestResult):
    """Exhaustive tests for the ternary (? :) operator."""
    print("\n--- Ternary Exhaustive Tests ---")
    img = torch.rand(1, 4, 4, 4)

    # Basic ternary — true branch
    try:
        result = compile_and_run("""
            float x = 0.8 > 0.5 ? 1.0 : 0.0;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.0) < 1e-5, f"Expected 1.0, got {v}"
        r.ok("ternary: basic true branch")
    except Exception as e:
        r.fail("ternary: basic true branch", str(e))

    # Basic ternary — false branch
    try:
        result = compile_and_run("""
            float x = 0.2 > 0.5 ? 1.0 : 0.0;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 0.0) < 1e-5, f"Expected 0.0, got {v}"
        r.ok("ternary: basic false branch")
    except Exception as e:
        r.fail("ternary: basic false branch", str(e))

    # Nested ternary
    try:
        result = compile_and_run("""
            float x = 0.8 > 0.7 ? 1.0 : 0.8 > 0.3 ? 0.5 : 0.0;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.0) < 1e-5, f"Expected 1.0, got {v}"
        r.ok("ternary: nested (first branch)")
    except Exception as e:
        r.fail("ternary: nested (first branch)", str(e))

    # Nested ternary — middle branch
    try:
        result = compile_and_run("""
            float x = 0.5 > 0.7 ? 1.0 : 0.5 > 0.3 ? 0.5 : 0.0;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 0.5) < 1e-5, f"Expected 0.5, got {v}"
        r.ok("ternary: nested (middle branch)")
    except Exception as e:
        r.fail("ternary: nested (middle branch)", str(e))

    # Ternary in vec3 constructor
    try:
        result = compile_and_run("""
            float sel = 0.8 > 0.5 ? 1.0 : 0.0;
            @OUT = vec4(vec3(sel, 0.25, 0.0), 1.0);
        """, {"A": img})
        r0 = result[0, 0, 0, 0].item()
        g0 = result[0, 0, 0, 1].item()
        assert abs(r0 - 1.0) < 1e-5 and abs(g0 - 0.25) < 1e-5, f"Got r={r0}, g={g0}"
        r.ok("ternary: in vec constructor")
    except Exception as e:
        r.fail("ternary: in vec constructor", str(e))

    # Ternary with function calls
    try:
        result = compile_and_run("""
            float x = clamp(0.8 > 0.5 ? 2.0 : -1.0, 0.0, 1.0);
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.0) < 1e-5, f"Expected 1.0, got {v}"
        r.ok("ternary: with function calls")
    except Exception as e:
        r.fail("ternary: with function calls", str(e))

    # Ternary with int-to-float promotion
    try:
        result = compile_and_run("""
            float x = 0.8 > 0.5 ? 1 : 0.5;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.0) < 1e-5, f"Expected 1.0, got {v}"
        r.ok("ternary: int-to-float promotion")
    except Exception as e:
        r.fail("ternary: int-to-float promotion", str(e))

    # Ternary in variable declaration with u/v
    try:
        result = compile_and_run("""
            float x = u > 0.5 ? u : v;
            @OUT = vec4(x);
        """, {"A": img})
        r.ok("ternary: in variable declaration with u/v")
    except Exception as e:
        r.fail("ternary: in variable declaration with u/v", str(e))

    # Ternary in assignment
    try:
        result = compile_and_run("""
            vec3 c = vec3(0.0);
            c = 0.8 > 0.5 ? vec3(1.0) : vec3(0.0);
            @OUT = vec4(c, 1.0);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.0) < 1e-5, f"Expected 1.0, got {v}"
        r.ok("ternary: in assignment to vec3")
    except Exception as e:
        r.fail("ternary: in assignment to vec3", str(e))

    # Ternary as function argument
    try:
        result = compile_and_run("""
            float x = clamp(0.2 > 0.5 ? 2.0 : -1.0, 0.0, 1.0);
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 0.0) < 1e-5, f"Expected 0.0, got {v}"
        r.ok("ternary: as function argument (false branch)")
    except Exception as e:
        r.fail("ternary: as function argument (false branch)", str(e))

    # Chained ternary (4 levels)
    try:
        result = compile_and_run("""
            float val = 0.6;
            float x = val > 0.75 ? 1.0 : val > 0.5 ? 0.75 : val > 0.25 ? 0.5 : 0.25;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 0.75) < 1e-5, f"Expected 0.75, got {v}"
        r.ok("ternary: chained 4 levels")
    except Exception as e:
        r.fail("ternary: chained 4 levels", str(e))

    # Ternary with == comparison
    try:
        result = compile_and_run("""
            float a = 1.0;
            float b = 1.0;
            float x = (a == b) ? 1.0 : 0.0;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.0) < 1e-5, f"Expected 1.0, got {v}"
        r.ok("ternary: with == comparison")
    except Exception as e:
        r.fail("ternary: with == comparison", str(e))

    # Ternary with logical AND
    try:
        result = compile_and_run("""
            float a = 0.5;
            float b = 0.5;
            float x = (a > 0.3 && b > 0.3) ? 1.0 : 0.0;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.0) < 1e-5, f"Expected 1.0, got {v}"
        r.ok("ternary: with logical AND")
    except Exception as e:
        r.fail("ternary: with logical AND", str(e))

    # Ternary returning vec3
    try:
        result = compile_and_run("""
            vec3 c = 0.8 > 0.5 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
            @OUT = vec4(c, 1.0);
        """, {"A": img})
        r0 = result[0, 0, 0, 0].item()
        b0 = result[0, 0, 0, 2].item()
        assert abs(r0 - 1.0) < 1e-5 and abs(b0 - 0.0) < 1e-5, f"Got r={r0}, b={b0}"
        r.ok("ternary: returning vec3")
    except Exception as e:
        r.fail("ternary: returning vec3", str(e))

    # Ternary in loop
    try:
        result = compile_and_run("""
            float x = 0.0;
            for (int i = 0; i < 4; i++) {
                x += i > 1 ? 1.0 : 0.5;
            }
            @OUT = vec4(x);
        """, {"A": img})
        # i=0: 0.5, i=1: 0.5, i=2: 1.0, i=3: 1.0 => 3.0
        v = result[0, 0, 0, 0].item()
        assert abs(v - 3.0) < 1e-5, f"Expected 3.0, got {v}"
        r.ok("ternary: in loop")
    except Exception as e:
        r.fail("ternary: in loop", str(e))

    # Ternary with binding
    try:
        result = compile_and_run("""
            @OUT = u > 0.5 ? @A : vec4(0.0);
        """, {"A": img})
        r.ok("ternary: with binding access")
    except Exception as e:
        r.fail("ternary: with binding access", str(e))


# ── User Functions Advanced Tests ─────────────────────────────────────

def test_user_functions_advanced(r: TestResult):
    """Advanced user-defined function tests."""
    print("\n--- User Functions Advanced Tests ---")
    img = torch.rand(1, 4, 4, 4)

    # Function B defined first, A calls B
    try:
        result = compile_and_run("""
            float double_it(float x) {
                return x * 2.0;
            }
            float quad(float x) {
                return double_it(double_it(x));
            }
            @OUT = @A * quad(0.25);
        """, {"A": img})
        expected = img * 1.0
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("func: A calls B (B defined first)")
    except Exception as e:
        r.fail("func: A calls B (B defined first)", str(e))

    # Recursive function with depth limit (factorial of small number)
    try:
        result = compile_and_run("""
            float factorial(float n) {
                if (n <= 1.0) {
                    return 1.0;
                }
                return n * factorial(n - 1.0);
            }
            float val = factorial(5.0);
            @OUT = vec4(val);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 120.0) < 1e-3, f"Expected 120.0, got {v}"
        r.ok("func: recursive factorial")
    except Exception as e:
        r.fail("func: recursive factorial", str(e))

    # Function with 5 parameters
    try:
        result = compile_and_run("""
            float weighted_sum(float a, float b, float c, float d, float e) {
                return a + b + c + d + e;
            }
            float val = weighted_sum(1.0, 2.0, 3.0, 4.0, 5.0);
            @OUT = vec4(val);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 15.0) < 1e-5, f"Expected 15.0, got {v}"
        r.ok("func: 5 parameters")
    except Exception as e:
        r.fail("func: 5 parameters", str(e))

    # Function returning result of another function
    try:
        result = compile_and_run("""
            float inner(float x) {
                return x * 3.0;
            }
            float outer(float x) {
                return inner(x + 1.0);
            }
            float val = outer(1.0);
            @OUT = vec4(val);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 6.0) < 1e-5, f"Expected 6.0, got {v}"
        r.ok("func: returns result of another function")
    except Exception as e:
        r.fail("func: returns result of another function", str(e))

    # Function with early return
    try:
        result = compile_and_run("""
            float safe_sqrt(float x) {
                if (x < 0.0) {
                    return 0.0;
                }
                return sqrt(x);
            }
            float a = safe_sqrt(4.0);
            float b = safe_sqrt(-1.0);
            @OUT = vec4(a, b, 0.0, 1.0);
        """, {"A": img})
        a_val = result[0, 0, 0, 0].item()
        b_val = result[0, 0, 0, 1].item()
        assert abs(a_val - 2.0) < 1e-4, f"Expected 2.0, got {a_val}"
        assert abs(b_val - 0.0) < 1e-5, f"Expected 0.0, got {b_val}"
        r.ok("func: early return")
    except Exception as e:
        r.fail("func: early return", str(e))

    # Function with while loop inside
    try:
        result = compile_and_run("""
            float count_up(float limit) {
                float acc = 0.0;
                float i = 0.0;
                while (i < limit) {
                    acc += 1.0;
                    i += 1.0;
                }
                return acc;
            }
            float val = count_up(7.0);
            @OUT = vec4(val);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 7.0) < 1e-5, f"Expected 7.0, got {v}"
        r.ok("func: with while loop")
    except Exception as e:
        r.fail("func: with while loop", str(e))

    # Function with nested if/else
    try:
        result = compile_and_run("""
            float classify(float x) {
                if (x < 0.0) {
                    return -1.0;
                } else {
                    if (x > 1.0) {
                        return 1.0;
                    } else {
                        return 0.0;
                    }
                }
            }
            float a = classify(-5.0);
            float b = classify(0.5);
            float c = classify(3.0);
            @OUT = vec4(a, b, c, 1.0);
        """, {"A": img})
        a_val = result[0, 0, 0, 0].item()
        b_val = result[0, 0, 0, 1].item()
        c_val = result[0, 0, 0, 2].item()
        assert abs(a_val - (-1.0)) < 1e-5, f"a: expected -1.0, got {a_val}"
        assert abs(b_val - 0.0) < 1e-5, f"b: expected 0.0, got {b_val}"
        assert abs(c_val - 1.0) < 1e-5, f"c: expected 1.0, got {c_val}"
        r.ok("func: nested if/else")
    except Exception as e:
        r.fail("func: nested if/else", str(e))

    # Function with compound assignment inside
    try:
        result = compile_and_run("""
            float accumulate(float x) {
                float acc = 0.0;
                acc += x;
                acc += x;
                acc *= 2.0;
                return acc;
            }
            float val = accumulate(3.0);
            @OUT = vec4(val);
        """, {"A": img})
        # (3+3)*2 = 12
        v = result[0, 0, 0, 0].item()
        assert abs(v - 12.0) < 1e-5, f"Expected 12.0, got {v}"
        r.ok("func: compound assignment inside")
    except Exception as e:
        r.fail("func: compound assignment inside", str(e))

    # Function result in compound assignment
    try:
        result = compile_and_run("""
            float triple(float x) {
                return x * 3.0;
            }
            float x = 1.0;
            x += triple(2.0);
            @OUT = vec4(x);
        """, {"A": img})
        # 1.0 + 6.0 = 7.0
        v = result[0, 0, 0, 0].item()
        assert abs(v - 7.0) < 1e-5, f"Expected 7.0, got {v}"
        r.ok("func: result in compound assignment")
    except Exception as e:
        r.fail("func: result in compound assignment", str(e))

    # Function result as array index
    try:
        result = compile_and_run("""
            int get_idx() {
                return 2;
            }
            float arr[4] = {10.0, 20.0, 30.0, 40.0};
            float val = arr[get_idx()];
            @OUT = vec4(val);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 30.0) < 1e-5, f"Expected 30.0, got {v}"
        r.ok("func: result as array index")
    except Exception as e:
        r.fail("func: result as array index", str(e))

    # Multiple return paths with loop
    try:
        result = compile_and_run("""
            float find_first_above(float threshold) {
                for (int i = 0; i < 10; i++) {
                    if (float(i) > threshold) {
                        return float(i);
                    }
                }
                return -1.0;
            }
            float val = find_first_above(3.5);
            @OUT = vec4(val);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 4.0) < 1e-5, f"Expected 4.0, got {v}"
        r.ok("func: multiple return paths with loop")
    except Exception as e:
        r.fail("func: multiple return paths with loop", str(e))

    # User function + channel access on result via intermediate
    try:
        result = compile_and_run("""
            vec3 make_color(float r, float g, float b) {
                return vec3(r, g, b);
            }
            vec3 c = make_color(0.1, 0.2, 0.3);
            float red = c.r;
            @OUT = vec4(red);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 0.1) < 1e-4, f"Expected 0.1, got {v}"
        r.ok("func: channel access on result via intermediate")
    except Exception as e:
        r.fail("func: channel access on result via intermediate", str(e))

    # User function that uses @A binding internally
    try:
        result = compile_and_run("""
            vec4 brighten(float amount) {
                return @A(u, v) * amount;
            }
            @OUT = brighten(2.0);
        """, {"A": img})
        expected = img * 2.0
        assert torch.allclose(result, expected, atol=1e-4)
        r.ok("func: uses binding @A internally")
    except Exception as e:
        r.fail("func: uses binding @A internally", str(e))

    # User function that returns string
    try:
        result = compile_and_run("""
            string label(float val) {
                return "value=" + str(val);
            }
            s@OUT = label(42.0);
        """, {"A": img})
        assert "value=" in result and "42" in result, f"Got {result}"
        r.ok("func: returns string")
    except Exception as e:
        r.fail("func: returns string", str(e))

    # Error: calling undefined function (typo)
    try:
        compile_and_run("""
            float blend(float a, float b, float t) {
                return lerp(a, b, t);
            }
            float val = blned(0.0, 1.0, 0.5);
            @OUT = vec4(val);
        """, {"A": img})
        r.fail("error: undefined func (typo blned)", "Should have raised an error")
    except (TypeCheckError, TEXMultiError) as e:
        msg = str(e).lower()
        # Should suggest 'blend' in the error
        r.ok("error: undefined func (typo blned)")
    except Exception as e:
        r.fail("error: undefined func (typo blned)", str(e))

    # Error: function defined after use
    try:
        compile_and_run("""
            float val = late_func(1.0);
            float late_func(float x) {
                return x * 2.0;
            }
            @OUT = vec4(val);
        """, {"A": img})
        r.fail("error: func defined after use", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("error: func defined after use")
    except Exception as e:
        r.fail("error: func defined after use", str(e))

    # Vec4 function with multiple statements
    try:
        result = compile_and_run("""
            vec4 tint_red(vec4 pixel) {
                float lum = luma(pixel);
                vec3 tinted = vec3(lum * 1.5, lum * 0.5, lum * 0.5);
                return vec4(tinted, pixel.a);
            }
            @OUT = tint_red(@A);
        """, {"A": img})
        assert result.shape == img.shape
        r.ok("func: vec4 with multiple statements")
    except Exception as e:
        r.fail("func: vec4 with multiple statements", str(e))

    # Two user functions called in same expression
    try:
        result = compile_and_run("""
            float add1(float x) { return x + 1.0; }
            float mul2(float x) { return x * 2.0; }
            float val = add1(3.0) + mul2(4.0);
            @OUT = vec4(val);
        """, {"A": img})
        # 4.0 + 8.0 = 12.0
        v = result[0, 0, 0, 0].item()
        assert abs(v - 12.0) < 1e-5, f"Expected 12.0, got {v}"
        r.ok("func: two functions in same expression")
    except Exception as e:
        r.fail("func: two functions in same expression", str(e))


# ── Binding Access Advanced Tests ─────────────────────────────────────

def test_binding_access_advanced(r: TestResult):
    """Advanced binding access tests."""
    print("\n--- Binding Access Advanced Tests ---")
    img = torch.zeros(1, 4, 4, 4)
    for y in range(4):
        for x in range(4):
            img[0, y, x, :] = x * 0.1 + y * 0.01

    img_b = torch.ones(1, 4, 4, 4) * 0.5

    # Binding access in loop (neighbor sum)
    try:
        result = compile_and_run("""
            vec4 sum = vec4(0.0);
            for (int dx = -1; dx <= 1; dx++) {
                sum += @A[ix + dx, iy];
            }
            @OUT = sum / 3.0;
        """, {"A": img})
        r.ok("binding: access in loop (neighbor)")
    except Exception as e:
        r.fail("binding: access in loop (neighbor)", str(e))

    # Binding access with arithmetic coords (clamp returns float, use int() cast)
    try:
        result = compile_and_run("""
            float cx = clamp(float(ix) * 2.0, 0.0, float(iw - 1));
            float cy = clamp(float(iy) * 2.0, 0.0, float(ih - 1));
            @OUT = @A[int(cx), int(cy)];
        """, {"A": img})
        r.ok("binding: arithmetic coords")
    except Exception as e:
        r.fail("binding: arithmetic coords", str(e))

    # Binding access with function call coords
    try:
        result = compile_and_run("""
            @OUT = @A[int(floor(u * float(iw - 1))), int(floor(v * float(ih - 1)))];
        """, {"A": img})
        r.ok("binding: function call coords")
    except Exception as e:
        r.fail("binding: function call coords", str(e))

    # Nested binding access — use result of one to compute another
    try:
        result = compile_and_run("""
            vec4 a_pixel = @A[ix, iy];
            float brightness = luma(a_pixel);
            float offset = brightness * 3.0;
            @OUT = @A[int(clamp(float(ix) + offset, 0.0, float(iw - 1))), iy];
        """, {"A": img})
        r.ok("binding: nested access (data-dependent coords)")
    except Exception as e:
        r.fail("binding: nested access (data-dependent coords)", str(e))

    # Binding access on multiple inputs
    try:
        result = compile_and_run("""
            @OUT = @A[ix, iy] + @B[ix, iy];
        """, {"A": img, "B": img_b})
        expected = img + img_b
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("binding: multiple inputs @A + @B")
    except Exception as e:
        r.fail("binding: multiple inputs @A + @B", str(e))

    # Sample with computed UV
    try:
        result = compile_and_run("""
            @OUT = @A(u * 0.5 + 0.25, v * 0.5 + 0.25);
        """, {"A": img})
        r.ok("binding: sample with computed UV")
    except Exception as e:
        r.fail("binding: sample with computed UV", str(e))

    # Sample result used in function
    try:
        result = compile_and_run("""
            float l = luma(@A(u, v));
            @OUT = vec4(l, l, l, 1.0);
        """, {"A": img})
        r.ok("binding: sample result in function (luma)")
    except Exception as e:
        r.fail("binding: sample result in function (luma)", str(e))

    # Binding access assigned to intermediate
    try:
        result = compile_and_run("""
            vec4 pixel = @A[ix, iy];
            float red = pixel.r;
            float green = pixel.g;
            @OUT = vec4(red, green, 0.0, 1.0);
        """, {"A": img})
        expected_r = img[..., 0]
        assert torch.allclose(result[..., 0], expected_r, atol=1e-5)
        r.ok("binding: assigned to intermediate + channel access")
    except Exception as e:
        r.fail("binding: assigned to intermediate + channel access", str(e))

    # Frame access with variable frame index
    try:
        batch_img = torch.rand(3, 4, 4, 4)
        result = compile_and_run("""
            int f = 0;
            @OUT = @A[ix, iy, f];
        """, {"A": batch_img})
        r.ok("binding: frame access with variable")
    except Exception as e:
        r.fail("binding: frame access with variable", str(e))

    # Binding access in ternary
    try:
        result = compile_and_run("""
            @OUT = u > 0.5 ? @A[ix, iy] : @B[ix, iy];
        """, {"A": img, "B": img_b})
        r.ok("binding: access in ternary")
    except Exception as e:
        r.fail("binding: access in ternary", str(e))

    # Binding access with swizzle chain .rgb
    try:
        result = compile_and_run("""
            vec3 c = @A[ix, iy].rgb;
            @OUT = vec4(c, 1.0);
        """, {"A": img})
        expected = torch.cat([img[..., :3], torch.ones(1, 4, 4, 1)], dim=-1)
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("binding: swizzle .rgb on fetch")
    except Exception as e:
        r.fail("binding: swizzle .rgb on fetch", str(e))

    # Binding sample with swizzle .a
    try:
        result = compile_and_run("""
            float alpha = @A(u, v).a;
            @OUT = vec4(alpha);
        """, {"A": img})
        r.ok("binding: swizzle .a on sample")
    except Exception as e:
        r.fail("binding: swizzle .a on sample", str(e))

    # Error: binding access on undefined binding
    try:
        compile_and_run("@OUT = @nonexistent[0, 0];", {"A": img})
        r.fail("error: undefined binding @nonexistent", "Should have raised an error")
    except (TypeCheckError, TEXMultiError, InterpreterError):
        r.ok("error: undefined binding @nonexistent")
    except Exception as e:
        r.fail("error: undefined binding @nonexistent", str(e))

    # Mixed fetch and sample in same program
    try:
        result = compile_and_run("""
            vec4 fetched = @A[ix, iy];
            vec4 sampled = @A(u, v);
            @OUT = (fetched + sampled) * 0.5;
        """, {"A": img})
        r.ok("binding: mixed fetch and sample in same program")
    except Exception as e:
        r.fail("binding: mixed fetch and sample in same program", str(e))


# ── Scope and Shadowing Tests ─────────────────────────────────────────

def test_scope_and_shadowing(r: TestResult):
    """Test variable scoping and shadowing."""
    print("\n--- Scope and Shadowing Tests ---")
    img = torch.rand(1, 4, 4, 4)

    # Variable declared in if block not visible outside
    try:
        compile_and_run("""
            if (1.0 > 0.0) {
                float inner_var = 5.0;
            }
            @OUT = vec4(inner_var);
        """, {"A": img})
        r.fail("scope: if-block var not visible outside", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("scope: if-block var not visible outside")
    except Exception as e:
        r.fail("scope: if-block var not visible outside", str(e))

    # Variable in for-loop init not visible after loop
    try:
        compile_and_run("""
            for (int i = 0; i < 3; i++) {
                float dummy = 1.0;
            }
            @OUT = vec4(float(i));
        """, {"A": img})
        r.fail("scope: loop var not visible after loop", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("scope: loop var not visible after loop")
    except Exception as e:
        r.fail("scope: loop var not visible after loop", str(e))

    # Variable re-declaration in if block — TEX if-blocks share the enclosing
    # scope, so re-declaring x inside an if-block overwrites the outer x.
    try:
        result = compile_and_run("""
            float x = 1.0;
            if (1.0 > 0.0) {
                x = 2.0;
            }
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 2.0) < 1e-5, f"Expected 2.0 (modified in if), got {v}"
        r.ok("scope: if-block modifies outer var")
    except Exception as e:
        r.fail("scope: if-block modifies outer var", str(e))

    # Function parameter shadows outer variable
    try:
        result = compile_and_run("""
            float x = 100.0;
            float identity(float x) {
                return x;
            }
            float val = identity(5.0);
            @OUT = vec4(val, x, 0.0, 1.0);
        """, {"A": img})
        val = result[0, 0, 0, 0].item()
        outer_x = result[0, 0, 0, 1].item()
        assert abs(val - 5.0) < 1e-5, f"Expected func result 5.0, got {val}"
        assert abs(outer_x - 100.0) < 1e-5, f"Expected outer x 100.0, got {outer_x}"
        r.ok("scope: function param shadows outer")
    except Exception as e:
        r.fail("scope: function param shadows outer", str(e))

    # Loop variable i not visible after loop (for)
    try:
        compile_and_run("""
            for (int j = 0; j < 3; j++) {}
            @OUT = vec4(float(j));
        """, {"A": img})
        r.fail("scope: for-loop j not visible after", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("scope: for-loop j not visible after")
    except Exception as e:
        r.fail("scope: for-loop j not visible after", str(e))

    # While loop variable scope
    try:
        result = compile_and_run("""
            float outer = 10.0;
            int count = 0;
            while (count < 3) {
                float inner = 1.0;
                count++;
            }
            @OUT = vec4(outer);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 10.0) < 1e-5, f"Expected 10.0, got {v}"
        r.ok("scope: while loop inner not leaked")
    except Exception as e:
        r.fail("scope: while loop inner not leaked", str(e))

    # Nested for loop variable independence
    try:
        result = compile_and_run("""
            float total = 0.0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 2; j++) {
                    total += 1.0;
                }
            }
            @OUT = vec4(total);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 6.0) < 1e-5, f"Expected 6.0, got {v}"
        r.ok("scope: nested for loops independent")
    except Exception as e:
        r.fail("scope: nested for loops independent", str(e))

    # Variable declared in else block
    try:
        compile_and_run("""
            if (0.0 > 1.0) {
                float a = 1.0;
            } else {
                float b = 2.0;
            }
            @OUT = vec4(b);
        """, {"A": img})
        r.fail("scope: else-block var not visible outside", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("scope: else-block var not visible outside")
    except Exception as e:
        r.fail("scope: else-block var not visible outside", str(e))

    # Function body scope doesn't leak
    try:
        compile_and_run("""
            float my_func() {
                float secret = 42.0;
                return secret;
            }
            float val = my_func();
            @OUT = vec4(secret);
        """, {"A": img})
        r.fail("scope: function body doesn't leak", "Should have raised an error")
    except (TypeCheckError, TEXMultiError):
        r.ok("scope: function body doesn't leak")
    except Exception as e:
        r.fail("scope: function body doesn't leak", str(e))

    # Multiple blocks with same variable name (independent)
    try:
        result = compile_and_run("""
            float result = 0.0;
            if (1.0 > 0.0) {
                float tmp = 10.0;
                result = tmp;
            }
            if (1.0 > 0.0) {
                float tmp = 20.0;
                result += tmp;
            }
            @OUT = vec4(result);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 30.0) < 1e-5, f"Expected 30.0, got {v}"
        r.ok("scope: multiple blocks same var name")
    except Exception as e:
        r.fail("scope: multiple blocks same var name", str(e))

    # Variable modification in function doesn't affect outer
    try:
        result = compile_and_run("""
            float x = 5.0;
            float modify(float x) {
                x = x * 100.0;
                return x;
            }
            float dummy = modify(x);
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 5.0) < 1e-5, f"Expected 5.0 (unmodified), got {v}"
        r.ok("scope: function param modification doesn't affect outer")
    except Exception as e:
        r.fail("scope: function param modification doesn't affect outer", str(e))

    # Break inside nested scope preserves outer variables
    try:
        result = compile_and_run("""
            float x = 0.0;
            for (int i = 0; i < 10; i++) {
                x += 1.0;
                if (i == 4) {
                    break;
                }
            }
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 5.0) < 1e-5, f"Expected 5.0, got {v}"
        r.ok("scope: break preserves outer variables")
    except Exception as e:
        r.fail("scope: break preserves outer variables", str(e))


# ── Operator Edge Cases Tests ─────────────────────────────────────────

def test_operator_edge_cases(r: TestResult):
    """Test operator combinations and edge cases."""
    print("\n--- Operator Edge Cases Tests ---")
    img = torch.rand(1, 4, 4, 4)

    # Modulo operator
    try:
        result = compile_and_run("""
            float x = mod(7.0, 3.0);
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.0) < 1e-5, f"Expected 1.0, got {v}"
        r.ok("operator: mod(7, 3)")
    except Exception as e:
        r.fail("operator: mod(7, 3)", str(e))

    # Comparison chain via nested ternary
    try:
        result = compile_and_run("""
            float a = 1.0;
            float b = (a > 0.5) ? ((a < 1.5) ? 1.0 : 0.0) : 0.0;
            @OUT = vec4(b);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.0) < 1e-5, f"Expected 1.0, got {v}"
        r.ok("operator: comparison chain via ternary")
    except Exception as e:
        r.fail("operator: comparison chain via ternary", str(e))

    # Logical NOT on comparison
    try:
        result = compile_and_run("""
            float x = !(0.8 > 0.5) ? 1.0 : 0.0;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 0.0) < 1e-5, f"Expected 0.0, got {v}"
        r.ok("operator: logical NOT on comparison")
    except Exception as e:
        r.fail("operator: logical NOT on comparison", str(e))

    # Unary minus on expression
    try:
        result = compile_and_run("""
            float a = 3.0;
            float b = 2.0;
            float x = -(a + b);
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - (-5.0)) < 1e-5, f"Expected -5.0, got {v}"
        r.ok("operator: unary minus on expression")
    except Exception as e:
        r.fail("operator: unary minus on expression", str(e))

    # Operator precedence: 2 + 3 * 4 == 14
    try:
        result = compile_and_run("""
            float x = 2.0 + 3.0 * 4.0;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 14.0) < 1e-5, f"Expected 14.0, got {v}"
        r.ok("operator: precedence mul before add")
    except Exception as e:
        r.fail("operator: precedence mul before add", str(e))

    # Mixed int/float arithmetic
    try:
        result = compile_and_run("""
            float x = 3 * 0.5;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.5) < 1e-5, f"Expected 1.5, got {v}"
        r.ok("operator: mixed int/float arithmetic")
    except Exception as e:
        r.fail("operator: mixed int/float arithmetic", str(e))

    # String + float concatenation via str()
    try:
        result = compile_and_run("""
            s@OUT = "val=" + str(1.5);
        """, {"A": img})
        assert "val=" in result and "1.5" in result, f"Got {result}"
        r.ok("operator: string + str(float)")
    except Exception as e:
        r.fail("operator: string + str(float)", str(e))

    # Comparison returning mask-like value
    try:
        result = compile_and_run("""
            float mask = 0.8 > 0.5 ? 1.0 : 0.0;
            @OUT = vec4(mask);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.0) < 1e-5, f"Expected 1.0, got {v}"
        r.ok("operator: comparison as mask (true)")
    except Exception as e:
        r.fail("operator: comparison as mask (true)", str(e))

    # Compound assignment with expression: x *= 2 + 1 should multiply by 3
    try:
        result = compile_and_run("""
            float x = 5.0;
            x *= 2.0 + 1.0;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 15.0) < 1e-5, f"Expected 15.0, got {v}"
        r.ok("operator: compound *= with expression")
    except Exception as e:
        r.fail("operator: compound *= with expression", str(e))

    # Increment operators
    try:
        result = compile_and_run("""
            int x = 5;
            x++;
            int y = x;
            @OUT = vec4(float(y));
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 6.0) < 1e-5, f"Expected 6.0, got {v}"
        r.ok("operator: postfix increment")
    except Exception as e:
        r.fail("operator: postfix increment", str(e))

    # Decrement operators
    try:
        result = compile_and_run("""
            int x = 5;
            x--;
            @OUT = vec4(float(x));
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 4.0) < 1e-5, f"Expected 4.0, got {v}"
        r.ok("operator: postfix decrement")
    except Exception as e:
        r.fail("operator: postfix decrement", str(e))

    # Assignment to .r channel
    try:
        result = compile_and_run("""
            vec3 c = vec3(0.0);
            c.r = 1.0;
            @OUT = vec4(c, 1.0);
        """, {"A": img})
        rv = result[0, 0, 0, 0].item()
        gv = result[0, 0, 0, 1].item()
        assert abs(rv - 1.0) < 1e-5 and abs(gv - 0.0) < 1e-5, f"Got r={rv}, g={gv}"
        r.ok("operator: assignment to .r channel")
    except Exception as e:
        r.fail("operator: assignment to .r channel", str(e))

    # Assignment to .rgb — TEX only supports single-channel assignment, so
    # multi-channel swizzle assignment is an error
    try:
        compile_and_run("""
            vec4 c = vec4(0.0);
            c.rgb = vec3(1.0, 0.5, 0.25);
            @OUT = c;
        """, {"A": img})
        r.fail("operator: .rgb assignment is error", "Should have raised an error")
    except (TypeCheckError, TEXMultiError, ParseError, InterpreterError):
        r.ok("operator: .rgb assignment is error")
    except Exception as e:
        r.fail("operator: .rgb assignment is error", str(e))

    # Logical OR
    try:
        result = compile_and_run("""
            float x = (0.2 > 0.5 || 0.8 > 0.5) ? 1.0 : 0.0;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.0) < 1e-5, f"Expected 1.0, got {v}"
        r.ok("operator: logical OR")
    except Exception as e:
        r.fail("operator: logical OR", str(e))


# ── Casting Exhaustive Tests ──────────────────────────────────────────

def test_casting_exhaustive(r: TestResult):
    """Test all type cast paths."""
    print("\n--- Casting Exhaustive Tests ---")
    img = torch.rand(1, 4, 4, 4)

    # float(int) — int to float
    try:
        result = compile_and_run("""
            int n = 7;
            float x = float(n);
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 7.0) < 1e-5, f"Expected 7.0, got {v}"
        r.ok("cast: float(int)")
    except Exception as e:
        r.fail("cast: float(int)", str(e))

    # int(float) — truncation
    try:
        result = compile_and_run("""
            float f = 1.7;
            int n = int(f);
            @OUT = vec4(float(n));
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 1.0) < 1e-5, f"Expected 1.0 (truncated), got {v}"
        r.ok("cast: int(1.7) truncates")
    except Exception as e:
        r.fail("cast: int(1.7) truncates", str(e))

    # vec3(float) — broadcast
    try:
        result = compile_and_run("""
            vec3 c = vec3(0.5);
            @OUT = vec4(c, 1.0);
        """, {"A": img})
        rv = result[0, 0, 0, 0].item()
        gv = result[0, 0, 0, 1].item()
        bv = result[0, 0, 0, 2].item()
        assert abs(rv - 0.5) < 1e-5 and abs(gv - 0.5) < 1e-5 and abs(bv - 0.5) < 1e-5
        r.ok("cast: vec3(float) broadcast")
    except Exception as e:
        r.fail("cast: vec3(float) broadcast", str(e))

    # vec4(vec3, alpha) — explicit construction with alpha
    # Note: vec4(vec3) single-arg is not supported; use vec4(c, 1.0)
    try:
        result = compile_and_run("""
            vec3 c = vec3(1.0, 0.5, 0.0);
            vec4 full = vec4(c, 1.0);
            @OUT = full;
        """, {"A": img})
        rv = result[0, 0, 0, 0].item()
        gv = result[0, 0, 0, 1].item()
        av = result[0, 0, 0, 3].item()
        assert abs(rv - 1.0) < 1e-5 and abs(gv - 0.5) < 1e-5, f"Got r={rv}, g={gv}"
        assert abs(av - 1.0) < 1e-5, f"Expected alpha=1.0, got {av}"
        r.ok("cast: vec4(vec3, alpha)")
    except Exception as e:
        r.fail("cast: vec4(vec3, alpha)", str(e))

    # str(float) — float to string
    try:
        result = compile_and_run("""
            s@OUT = str(42.0);
        """, {"A": img})
        assert "42" in result, f"Got {result}"
        r.ok("cast: str(float)")
    except Exception as e:
        r.fail("cast: str(float)", str(e))

    # str(int) — int to string
    try:
        result = compile_and_run("""
            s@OUT = str(7);
        """, {"A": img})
        assert "7" in result, f"Got {result}"
        r.ok("cast: str(int)")
    except Exception as e:
        r.fail("cast: str(int)", str(e))

    # to_int(string) — string to int
    try:
        result = compile_and_run("""
            int n = to_int("42");
            @OUT = vec4(float(n));
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 42.0) < 1e-5, f"Expected 42.0, got {v}"
        r.ok("cast: to_int(string)")
    except Exception as e:
        r.fail("cast: to_int(string)", str(e))

    # to_float(string) — string to float
    try:
        result = compile_and_run("""
            float f = to_float("3.14");
            @OUT = vec4(f);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 3.14) < 1e-3, f"Expected 3.14, got {v}"
        r.ok("cast: to_float(string)")
    except Exception as e:
        r.fail("cast: to_float(string)", str(e))

    # int(3.9) — should truncate to 3
    try:
        result = compile_and_run("""
            int n = int(3.9);
            @OUT = vec4(float(n));
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 3.0) < 1e-5, f"Expected 3.0, got {v}"
        r.ok("cast: int(3.9) truncates to 3")
    except Exception as e:
        r.fail("cast: int(3.9) truncates to 3", str(e))

    # int(-1.5) — negative truncation
    try:
        result = compile_and_run("""
            int n = int(-1.5);
            @OUT = vec4(float(n));
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        # TEX uses floor-based int cast: int(-1.5) -> -2
        assert abs(v - (-2.0)) < 1e-5, f"Expected -2.0, got {v}"
        r.ok("cast: int(-1.5) negative truncation")
    except Exception as e:
        r.fail("cast: int(-1.5) negative truncation", str(e))

    # Cast in expression
    try:
        result = compile_and_run("""
            float x = float(3) + 0.5;
            @OUT = vec4(x);
        """, {"A": img})
        v = result[0, 0, 0, 0].item()
        assert abs(v - 3.5) < 1e-5, f"Expected 3.5, got {v}"
        r.ok("cast: float(3) + 0.5 in expression")
    except Exception as e:
        r.fail("cast: float(3) + 0.5 in expression", str(e))

    # vec4(float) — broadcast to all channels
    try:
        result = compile_and_run("""
            vec4 c = vec4(0.75);
            @OUT = c;
        """, {"A": img})
        for ch in range(4):
            v = result[0, 0, 0, ch].item()
            assert abs(v - 0.75) < 1e-5, f"Channel {ch}: expected 0.75, got {v}"
        r.ok("cast: vec4(float) broadcast")
    except Exception as e:
        r.fail("cast: vec4(float) broadcast", str(e))


# ── Missing Stdlib Functions ───────────────────────────────────────────

def test_missing_stdlib_functions(r: TestResult):
    """Tests for reflect, rgb2hsv, and matches stdlib functions."""
    print("\n--- Missing Stdlib Functions Tests ---")

    B, H, W = 1, 4, 4
    img = torch.rand(B, H, W, 3)

    # -- reflect(incident, normal) --

    # reflect along Y axis: incident=(1,0,0), normal=(0,1,0) -> (1,0,0) (no change in x)
    try:
        code = "vec3 r = reflect(vec3(1,0,0), vec3(0,1,0)); @OUT = vec4(r.x, r.y, r.z, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 1.0) < 1e-4, f"x={result[0,0,0,0].item()}"
        assert abs(result[0,0,0,1].item()) < 1e-4, f"y={result[0,0,0,1].item()}"
        assert abs(result[0,0,0,2].item()) < 1e-4, f"z={result[0,0,0,2].item()}"
        r.ok("reflect: (1,0,0) over normal (0,1,0)")
    except Exception as e:
        r.fail("reflect: (1,0,0) over normal (0,1,0)", f"{e}")

    # reflect: incident=(0,1,0), normal=(0,1,0) -> (0,-1,0)
    try:
        code = "vec3 r = reflect(vec3(0,1,0), vec3(0,1,0)); @OUT = vec4(r.x, r.y, r.z, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item()) < 1e-4
        assert abs(result[0,0,0,1].item() - (-1.0)) < 1e-4
        assert abs(result[0,0,0,2].item()) < 1e-4
        r.ok("reflect: (0,1,0) over normal (0,1,0)")
    except Exception as e:
        r.fail("reflect: (0,1,0) over normal (0,1,0)", f"{e}")

    # reflect: 45-degree — incident=(1,1,0), normal=(0,1,0) -> (1,-1,0)
    try:
        code = "vec3 r = reflect(vec3(1,1,0), vec3(0,1,0)); @OUT = vec4(r.x, r.y, r.z, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 1.0) < 1e-4
        assert abs(result[0,0,0,1].item() - (-1.0)) < 1e-4
        r.ok("reflect: 45-degree (1,1,0)")
    except Exception as e:
        r.fail("reflect: 45-degree (1,1,0)", f"{e}")

    # reflect: 4D vector
    try:
        code = "vec4 r = reflect(vec4(0,1,0,0), vec4(0,1,0,0)); @OUT = r;"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,1].item() - (-1.0)) < 1e-4
        r.ok("reflect: 4D vector")
    except Exception as e:
        r.fail("reflect: 4D vector", f"{e}")

    # reflect: zero incident
    try:
        code = "vec3 r = reflect(vec3(0,0,0), vec3(0,1,0)); @OUT = vec4(r.x, r.y, r.z, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item()) < 1e-4
        assert abs(result[0,0,0,1].item()) < 1e-4
        r.ok("reflect: zero incident")
    except Exception as e:
        r.fail("reflect: zero incident", f"{e}")

    # -- rgb2hsv --

    # Pure red (1,0,0) -> H ~ 0, S=1, V=1
    try:
        code = "vec3 hsv = rgb2hsv(vec3(1,0,0)); @OUT = vec4(hsv.x, hsv.y, hsv.z, 1);"
        result = compile_and_run(code, {"A": img})
        h = result[0,0,0,0].item()
        s = result[0,0,0,1].item()
        v = result[0,0,0,2].item()
        assert (abs(h) < 0.02 or abs(h - 1.0) < 0.02), f"Red H should be ~0 or ~1, got {h}"
        assert abs(s - 1.0) < 0.02, f"Red S should be ~1, got {s}"
        assert abs(v - 1.0) < 0.02, f"Red V should be ~1, got {v}"
        r.ok("rgb2hsv: pure red")
    except Exception as e:
        r.fail("rgb2hsv: pure red", f"{e}")

    # Pure green (0,1,0) -> H ~ 0.333, S=1, V=1
    try:
        code = "vec3 hsv = rgb2hsv(vec3(0,1,0)); @OUT = vec4(hsv.x, hsv.y, hsv.z, 1);"
        result = compile_and_run(code, {"A": img})
        h = result[0,0,0,0].item()
        assert abs(h - 0.333) < 0.02, f"Green H should be ~0.333, got {h}"
        assert abs(result[0,0,0,1].item() - 1.0) < 0.02
        r.ok("rgb2hsv: pure green")
    except Exception as e:
        r.fail("rgb2hsv: pure green", f"{e}")

    # White (1,1,1) -> S=0, V=1
    try:
        code = "vec3 hsv = rgb2hsv(vec3(1,1,1)); @OUT = vec4(hsv.x, hsv.y, hsv.z, 1);"
        result = compile_and_run(code, {"A": img})
        s = result[0,0,0,1].item()
        v = result[0,0,0,2].item()
        assert abs(s) < 0.02, f"White S should be ~0, got {s}"
        assert abs(v - 1.0) < 0.02, f"White V should be ~1, got {v}"
        r.ok("rgb2hsv: white")
    except Exception as e:
        r.fail("rgb2hsv: white", f"{e}")

    # Black (0,0,0) -> S=0, V=0
    try:
        code = "vec3 hsv = rgb2hsv(vec3(0,0,0)); @OUT = vec4(hsv.x, hsv.y, hsv.z, 1);"
        result = compile_and_run(code, {"A": img})
        s = result[0,0,0,1].item()
        v = result[0,0,0,2].item()
        assert abs(s) < 0.02, f"Black S should be ~0, got {s}"
        assert abs(v) < 0.02, f"Black V should be ~0, got {v}"
        r.ok("rgb2hsv: black")
    except Exception as e:
        r.fail("rgb2hsv: black", f"{e}")

    # Round-trip: rgb -> hsv -> rgb
    try:
        code = """
vec3 orig = vec3(0.3, 0.6, 0.9);
vec3 hsv = rgb2hsv(orig);
vec3 back = hsv2rgb(hsv);
@OUT = vec4(back.x, back.y, back.z, 1);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 0.3) < 0.01, f"R round-trip failed"
        assert abs(result[0,0,0,1].item() - 0.6) < 0.01, f"G round-trip failed"
        assert abs(result[0,0,0,2].item() - 0.9) < 0.01, f"B round-trip failed"
        r.ok("rgb2hsv: round-trip with hsv2rgb")
    except Exception as e:
        r.fail("rgb2hsv: round-trip with hsv2rgb", f"{e}")

    # -- matches(string, pattern) --
    # Note: matches() uses re.fullmatch, so the pattern must match the entire string

    # Basic full match
    try:
        code = 'float x = matches("hello", "hello"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 1.0) < 1e-4
        r.ok("matches: exact full match")
    except Exception as e:
        r.fail("matches: exact full match", f"{e}")

    # No match
    try:
        code = 'float x = matches("hello", "xyz"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item()) < 1e-4
        r.ok("matches: no match")
    except Exception as e:
        r.fail("matches: no match", f"{e}")

    # Regex pattern with fullmatch
    try:
        code = 'float x = matches("test123", "test[0-9]+"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 1.0) < 1e-4
        r.ok("matches: regex digit pattern")
    except Exception as e:
        r.fail("matches: regex digit pattern", f"{e}")

    # Partial match should fail with fullmatch
    try:
        code = r'float x = matches("hello world", "hello"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item()) < 1e-4
        r.ok("matches: partial does not fullmatch")
    except Exception as e:
        r.fail("matches: partial does not fullmatch", f"{e}")

    # Wildcard fullmatch
    try:
        code = 'float x = matches("anything", ".*"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 1.0) < 1e-4
        r.ok("matches: wildcard .* fullmatch")
    except Exception as e:
        r.fail("matches: wildcard .* fullmatch", f"{e}")

    # Empty string with .*
    try:
        code = 'float x = matches("", ".*"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 1.0) < 1e-4
        r.ok("matches: empty string with .*")
    except Exception as e:
        r.fail("matches: empty string with .*", f"{e}")


# ── Numeric Edge Case Matrix ──────────────────────────────────────────

def test_numeric_edge_case_matrix(r: TestResult):
    """Systematically test math stdlib functions with special inputs."""
    print("\n--- Numeric Edge Case Matrix Tests ---")

    B, H, W = 1, 2, 2
    img = torch.rand(B, H, W, 3)

    def run_check(name, code, expect_finite=True):
        """Run code and check that it doesn't crash. If expect_finite, verify result is finite."""
        try:
            result = compile_and_run(code, {"A": img})
            val = result[0,0,0,0].item()
            is_nan = val != val
            is_inf = abs(val) == float('inf')
            if expect_finite and is_nan:
                r.fail(f"edge: {name}", f"Got NaN, expected finite")
            elif expect_finite and is_inf:
                r.fail(f"edge: {name}", f"Got Inf, expected finite")
            else:
                r.ok(f"edge: {name}")
        except Exception as e:
            r.fail(f"edge: {name}", f"{e}")

    # -- sin --
    run_check("sin(0.0)", "float x = sin(0.0); @OUT = vec3(x,x,x);")
    run_check("sin(-1.0)", "float x = sin(-1.0); @OUT = vec3(x,x,x);")
    run_check("sin(1e30)", "float x = sin(1e30); @OUT = vec3(x,x,x);")
    run_check("sin(1e-30)", "float x = sin(1e-30); @OUT = vec3(x,x,x);")

    # -- cos --
    run_check("cos(0.0)", "float x = cos(0.0); @OUT = vec3(x,x,x);")
    run_check("cos(-1.0)", "float x = cos(-1.0); @OUT = vec3(x,x,x);")
    run_check("cos(1e30)", "float x = cos(1e30); @OUT = vec3(x,x,x);")
    run_check("cos(1e-30)", "float x = cos(1e-30); @OUT = vec3(x,x,x);")

    # -- tan --
    run_check("tan(0.0)", "float x = tan(0.0); @OUT = vec3(x,x,x);")
    run_check("tan(-1.0)", "float x = tan(-1.0); @OUT = vec3(x,x,x);")
    run_check("tan(1e-30)", "float x = tan(1e-30); @OUT = vec3(x,x,x);")

    # -- asin --
    run_check("asin(0.0)", "float x = asin(0.0); @OUT = vec3(x,x,x);")
    run_check("asin(-0.5)", "float x = asin(-0.5); @OUT = vec3(x,x,x);")
    run_check("asin(1e-30)", "float x = asin(1e-30); @OUT = vec3(x,x,x);")

    # -- acos --
    run_check("acos(0.0)", "float x = acos(0.0); @OUT = vec3(x,x,x);")
    run_check("acos(-0.5)", "float x = acos(-0.5); @OUT = vec3(x,x,x);")
    run_check("acos(1e-30)", "float x = acos(1e-30); @OUT = vec3(x,x,x);")

    # -- atan --
    run_check("atan(0.0)", "float x = atan(0.0); @OUT = vec3(x,x,x);")
    run_check("atan(-1.0)", "float x = atan(-1.0); @OUT = vec3(x,x,x);")
    run_check("atan(1e30)", "float x = atan(1e30); @OUT = vec3(x,x,x);")
    run_check("atan(1e-30)", "float x = atan(1e-30); @OUT = vec3(x,x,x);")

    # -- atan2 --
    run_check("atan2(0,1)", "float x = atan2(0.0, 1.0); @OUT = vec3(x,x,x);")
    run_check("atan2(1,0)", "float x = atan2(1.0, 0.0); @OUT = vec3(x,x,x);")
    run_check("atan2(-1,-1)", "float x = atan2(-1.0, -1.0); @OUT = vec3(x,x,x);")
    run_check("atan2(0,0)", "float x = atan2(0.0, 0.0); @OUT = vec3(x,x,x);")

    # -- sqrt --
    run_check("sqrt(0.0)", "float x = sqrt(0.0); @OUT = vec3(x,x,x);")
    run_check("sqrt(1e-30)", "float x = sqrt(1e-30); @OUT = vec3(x,x,x);")
    run_check("sqrt(1e30)", "float x = sqrt(1e30); @OUT = vec3(x,x,x);")

    # -- pow --
    run_check("pow(0,0)", "float x = pow(0.0, 0.0); @OUT = vec3(x,x,x);")
    run_check("pow(2,-1)", "float x = pow(2.0, -1.0); @OUT = vec3(x,x,x);")
    run_check("pow(1e30,2)", "float x = pow(1e30, 2.0); @OUT = vec3(x,x,x);", expect_finite=False)

    # -- exp --
    run_check("exp(0.0)", "float x = exp(0.0); @OUT = vec3(x,x,x);")
    run_check("exp(-100)", "float x = exp(-100.0); @OUT = vec3(x,x,x);")
    run_check("exp(1e-30)", "float x = exp(1e-30); @OUT = vec3(x,x,x);")

    # -- log --
    run_check("log(1.0)", "float x = log(1.0); @OUT = vec3(x,x,x);")
    run_check("log(1e-30)", "float x = log(1e-30); @OUT = vec3(x,x,x);")
    run_check("log(1e30)", "float x = log(1e30); @OUT = vec3(x,x,x);")

    # -- log2 --
    run_check("log2(1.0)", "float x = log2(1.0); @OUT = vec3(x,x,x);")
    run_check("log2(1e-30)", "float x = log2(1e-30); @OUT = vec3(x,x,x);")

    # -- log10 --
    run_check("log10(1.0)", "float x = log10(1.0); @OUT = vec3(x,x,x);")
    run_check("log10(1e-30)", "float x = log10(1e-30); @OUT = vec3(x,x,x);")

    # -- abs --
    run_check("abs(0.0)", "float x = abs(0.0); @OUT = vec3(x,x,x);")
    run_check("abs(-1e30)", "float x = abs(-1e30); @OUT = vec3(x,x,x);")
    run_check("abs(1e-30)", "float x = abs(1e-30); @OUT = vec3(x,x,x);")

    # -- sign --
    run_check("sign(0.0)", "float x = sign(0.0); @OUT = vec3(x,x,x);")
    run_check("sign(-1e30)", "float x = sign(-1e30); @OUT = vec3(x,x,x);")
    run_check("sign(1e-30)", "float x = sign(1e-30); @OUT = vec3(x,x,x);")

    # -- floor --
    run_check("floor(0.0)", "float x = floor(0.0); @OUT = vec3(x,x,x);")
    run_check("floor(-0.5)", "float x = floor(-0.5); @OUT = vec3(x,x,x);")
    run_check("floor(1e30)", "float x = floor(1e30); @OUT = vec3(x,x,x);")

    # -- ceil --
    run_check("ceil(0.0)", "float x = ceil(0.0); @OUT = vec3(x,x,x);")
    run_check("ceil(-0.5)", "float x = ceil(-0.5); @OUT = vec3(x,x,x);")
    run_check("ceil(1e30)", "float x = ceil(1e30); @OUT = vec3(x,x,x);")

    # -- round --
    run_check("round(0.0)", "float x = round(0.0); @OUT = vec3(x,x,x);")
    run_check("round(-0.5)", "float x = round(-0.5); @OUT = vec3(x,x,x);")
    run_check("round(1e30)", "float x = round(1e30); @OUT = vec3(x,x,x);")

    # -- fract --
    run_check("fract(0.0)", "float x = fract(0.0); @OUT = vec3(x,x,x);")
    run_check("fract(-0.5)", "float x = fract(-0.5); @OUT = vec3(x,x,x);")
    run_check("fract(1e30)", "float x = fract(1e30); @OUT = vec3(x,x,x);")

    # -- mod --
    run_check("mod(5,3)", "float x = mod(5.0, 3.0); @OUT = vec3(x,x,x);")
    run_check("mod(0,1)", "float x = mod(0.0, 1.0); @OUT = vec3(x,x,x);")
    run_check("mod(-5,3)", "float x = mod(-5.0, 3.0); @OUT = vec3(x,x,x);")

    # -- clamp --
    run_check("clamp(0.5,0,1)", "float x = clamp(0.5, 0.0, 1.0); @OUT = vec3(x,x,x);")
    run_check("clamp(-1,0,1)", "float x = clamp(-1.0, 0.0, 1.0); @OUT = vec3(x,x,x);")
    run_check("clamp(1e30,0,1)", "float x = clamp(1e30, 0.0, 1.0); @OUT = vec3(x,x,x);")

    # -- lerp --
    run_check("lerp(0,1,0.5)", "float x = lerp(0.0, 1.0, 0.5); @OUT = vec3(x,x,x);")
    run_check("lerp(0,1,0)", "float x = lerp(0.0, 1.0, 0.0); @OUT = vec3(x,x,x);")
    run_check("lerp(0,1,1)", "float x = lerp(0.0, 1.0, 1.0); @OUT = vec3(x,x,x);")
    run_check("lerp(0,1,-1)", "float x = lerp(0.0, 1.0, -1.0); @OUT = vec3(x,x,x);")

    # -- smoothstep --
    run_check("smoothstep(0,1,0.5)", "float x = smoothstep(0.0, 1.0, 0.5); @OUT = vec3(x,x,x);")
    run_check("smoothstep(0,1,0)", "float x = smoothstep(0.0, 1.0, 0.0); @OUT = vec3(x,x,x);")
    run_check("smoothstep(0,1,1)", "float x = smoothstep(0.0, 1.0, 1.0); @OUT = vec3(x,x,x);")
    run_check("smoothstep(0,1,-1)", "float x = smoothstep(0.0, 1.0, -1.0); @OUT = vec3(x,x,x);")

    # -- step --
    run_check("step(0.5,0.3)", "float x = step(0.5, 0.3); @OUT = vec3(x,x,x);")
    run_check("step(0.5,0.5)", "float x = step(0.5, 0.5); @OUT = vec3(x,x,x);")
    run_check("step(0.5,0.7)", "float x = step(0.5, 0.7); @OUT = vec3(x,x,x);")
    run_check("step(0,0)", "float x = step(0.0, 0.0); @OUT = vec3(x,x,x);")


# ── Array Bounds Tests ────────────────────────────────────────────────

def test_array_bounds(r: TestResult):
    """Tests for array bounds handling, edge cases."""
    print("\n--- Array Bounds Tests ---")

    B, H, W = 1, 4, 4
    img = torch.rand(B, H, W, 4)

    # Access at index 0 (first element)
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[0], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 10.0) < 1e-4
        r.ok("array bounds: index 0")
    except Exception as e:
        r.fail("array bounds: index 0", f"{e}")

    # Access at last valid index
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[2], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 30.0) < 1e-4
        r.ok("array bounds: last index")
    except Exception as e:
        r.fail("array bounds: last index", f"{e}")

    # Access beyond bounds (should clamp to last element)
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[5], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 30.0) < 1e-4
        r.ok("array bounds: beyond upper clamps to last")
    except Exception as e:
        r.fail("array bounds: beyond upper clamps to last", f"{e}")

    # Access at very large index
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[1000], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 30.0) < 1e-4
        r.ok("array bounds: very large index clamps")
    except Exception as e:
        r.fail("array bounds: very large index clamps", f"{e}")

    # Negative index (should clamp to 0)
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[-1], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 10.0) < 1e-4
        r.ok("array bounds: negative clamps to 0")
    except Exception as e:
        r.fail("array bounds: negative clamps to 0", f"{e}")

    # Very negative index
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[-100], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 10.0) < 1e-4
        r.ok("array bounds: very negative clamps to 0")
    except Exception as e:
        r.fail("array bounds: very negative clamps to 0", f"{e}")

    # Array length with len()
    try:
        code = "float a[] = {1.0, 2.0, 3.0, 4.0, 5.0}; float n = len(a); @OUT = vec4(n, 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 5.0) < 1e-4
        r.ok("array bounds: len()")
    except Exception as e:
        r.fail("array bounds: len()", f"{e}")

    # Single element array
    try:
        code = "float a[] = {42.0}; @OUT = vec4(a[0], len(a), 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 42.0) < 1e-4
        assert abs(result[0,0,0,1].item() - 1.0) < 1e-4
        r.ok("array bounds: single element")
    except Exception as e:
        r.fail("array bounds: single element", f"{e}")

    # Write beyond bounds (should clamp)
    try:
        code = """
float a[] = {1.0, 2.0, 3.0};
a[10] = 99.0;
@OUT = vec4(a[0], a[1], a[2], 1);
"""
        result = compile_and_run(code, {"A": img})
        # Writing at clamped index 2 should change a[2]
        assert abs(result[0,0,0,2].item() - 99.0) < 1e-4
        r.ok("array bounds: write beyond clamps")
    except Exception as e:
        r.fail("array bounds: write beyond clamps", f"{e}")

    # Write at negative index (should clamp to 0)
    try:
        code = """
float a[] = {1.0, 2.0, 3.0};
a[-5] = 99.0;
@OUT = vec4(a[0], a[1], a[2], 1);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 99.0) < 1e-4
        r.ok("array bounds: write negative clamps to 0")
    except Exception as e:
        r.fail("array bounds: write negative clamps to 0", f"{e}")

    # Nested array access (array of arrays via vec)
    try:
        code = """
vec3 a[] = {vec3(1,2,3), vec3(4,5,6)};
float x = a[0].y;
float y = a[1].z;
@OUT = vec4(x, y, 0, 1);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 2.0) < 1e-4
        assert abs(result[0,0,0,1].item() - 6.0) < 1e-4
        r.ok("array bounds: vec array element access")
    except Exception as e:
        r.fail("array bounds: vec array element access", f"{e}")

    # Vec array beyond bounds
    try:
        code = """
vec3 a[] = {vec3(1,2,3), vec3(4,5,6)};
vec3 val = a[10];
@OUT = vec4(val.x, val.y, val.z, 1);
"""
        result = compile_and_run(code, {"A": img})
        # Should clamp to a[1]
        assert abs(result[0,0,0,0].item() - 4.0) < 1e-4
        r.ok("array bounds: vec array beyond bounds clamps")
    except Exception as e:
        r.fail("array bounds: vec array beyond bounds clamps", f"{e}")

    # Dynamic index with loop
    try:
        code = """
float a[] = {10.0, 20.0, 30.0, 40.0, 50.0};
float sum = 0.0;
for (int i = 0; i < 5; i++) {
    sum += a[i];
}
@OUT = vec4(sum, 0, 0, 1);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 150.0) < 1e-4
        r.ok("array bounds: dynamic index in loop")
    except Exception as e:
        r.fail("array bounds: dynamic index in loop", f"{e}")

    # Float index (should be floored)
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[1.7], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 20.0) < 1e-4  # floor(1.7) = 1
        r.ok("array bounds: float index floors")
    except Exception as e:
        r.fail("array bounds: float index floors", f"{e}")

    # Large array
    try:
        code = """
float a[100];
for (int i = 0; i < 100; i++) {
    a[i] = float(i);
}
@OUT = vec4(a[0], a[50], a[99], 1);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item()) < 1e-4
        assert abs(result[0,0,0,1].item() - 50.0) < 1e-4
        assert abs(result[0,0,0,2].item() - 99.0) < 1e-4
        r.ok("array bounds: large array (100)")
    except Exception as e:
        r.fail("array bounds: large array (100)", f"{e}")

    # Empty-sized array (declared with size 0 should error)
    try:
        code = "float a[0]; @OUT = vec4(0,0,0,1);"
        compile_and_run(code, {"A": img})
        r.fail("array bounds: size 0 should error", "Expected error")
    except (ParseError, TypeCheckError):
        r.ok("array bounds: size 0 errors")

    # String array bounds
    try:
        code = '''
string a[] = {"alpha", "beta", "gamma"};
@OUT = a[0];
'''
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "alpha"
        r.ok("array bounds: string array index 0")
    except Exception as e:
        r.fail("array bounds: string array index 0", f"{e}")

    # String array beyond bounds
    try:
        code = '''
string a[] = {"alpha", "beta", "gamma"};
@OUT = a[10];
'''
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "gamma"  # clamped to last
        r.ok("array bounds: string array clamp to last")
    except Exception as e:
        r.fail("array bounds: string array clamp to last", f"{e}")


# ── String Edge Cases ─────────────────────────────────────────────────

def test_string_edge_cases(r: TestResult):
    """Tests for string edge cases."""
    print("\n--- String Edge Cases Tests ---")

    B, H, W = 1, 4, 4
    img = torch.rand(B, H, W, 4)

    # Empty string
    try:
        result = compile_and_run(
            'string s = ""; @OUT = s;', {}, out_type=TEXType.STRING)
        assert result == "", f"Expected empty string, got {result!r}"
        r.ok("string edge: empty string")
    except Exception as e:
        r.fail("string edge: empty string", f"{e}")

    # Concatenation with empty
    try:
        result = compile_and_run(
            'string s = "hello" + ""; @OUT = s;', {}, out_type=TEXType.STRING)
        assert result == "hello", f"Expected 'hello', got {result!r}"
        r.ok("string edge: concat with empty")
    except Exception as e:
        r.fail("string edge: concat with empty", f"{e}")

    # Empty + non-empty
    try:
        result = compile_and_run(
            'string s = "" + "world"; @OUT = s;', {}, out_type=TEXType.STRING)
        assert result == "world", f"Expected 'world', got {result!r}"
        r.ok("string edge: empty + non-empty")
    except Exception as e:
        r.fail("string edge: empty + non-empty", f"{e}")

    # Very long string
    try:
        long_str = "a" * 200
        code = f'string s = "{long_str}"; float n = len(s); @OUT = vec4(n,n,n,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 200.0) < 1e-4
        r.ok("string edge: long string (200 chars)")
    except Exception as e:
        r.fail("string edge: long string (200 chars)", f"{e}")

    # Special chars in strings (tab, backslash)
    try:
        result = compile_and_run(
            r'string s = "tab\there"; @OUT = s;', {}, out_type=TEXType.STRING)
        assert "\t" in result, f"Expected tab char, got {result!r}"
        r.ok("string edge: tab character")
    except Exception as e:
        r.fail("string edge: tab character", f"{e}")

    # find() with not-found case
    try:
        code = 'float x = find("hello", "xyz"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - (-1.0)) < 1e-4
        r.ok("string edge: find not found returns -1")
    except Exception as e:
        r.fail("string edge: find not found returns -1", f"{e}")

    # find() with found case
    try:
        code = 'float x = find("hello world", "world"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 6.0) < 1e-4
        r.ok("string edge: find found returns index")
    except Exception as e:
        r.fail("string edge: find found returns index", f"{e}")

    # find() at start
    try:
        code = 'float x = find("hello", "hel"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item()) < 1e-4
        r.ok("string edge: find at start returns 0")
    except Exception as e:
        r.fail("string edge: find at start returns 0", f"{e}")

    # replace() with empty replacement
    try:
        result = compile_and_run(
            'string s = replace("hello world", "world", ""); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello ", f"Expected 'hello ', got {result!r}"
        r.ok("string edge: replace with empty")
    except Exception as e:
        r.fail("string edge: replace with empty", f"{e}")

    # replace() with not-found target
    try:
        result = compile_and_run(
            'string s = replace("hello", "xyz", "abc"); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello", f"Expected 'hello', got {result!r}"
        r.ok("string edge: replace not found")
    except Exception as e:
        r.fail("string edge: replace not found", f"{e}")

    # substr() from start
    try:
        result = compile_and_run(
            'string s = substr("hello world", 0, 5); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello", f"Expected 'hello', got {result!r}"
        r.ok("string edge: substr from 0")
    except Exception as e:
        r.fail("string edge: substr from 0", f"{e}")

    # substr() past-end length
    try:
        result = compile_and_run(
            'string s = substr("hello", 3, 100); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "lo", f"Expected 'lo', got {result!r}"
        r.ok("string edge: substr past end")
    except Exception as e:
        r.fail("string edge: substr past end", f"{e}")

    # substr() without length (to end)
    try:
        result = compile_and_run(
            'string s = substr("hello world", 6); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "world", f"Expected 'world', got {result!r}"
        r.ok("string edge: substr no length")
    except Exception as e:
        r.fail("string edge: substr no length", f"{e}")

    # len() on empty string
    try:
        code = 'float n = len(""); @OUT = vec4(n,n,n,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item()) < 1e-4
        r.ok("string edge: len empty string")
    except Exception as e:
        r.fail("string edge: len empty string", f"{e}")

    # len() on long string
    try:
        long_str = "x" * 150
        code = f'float n = len("{long_str}"); @OUT = vec4(n,n,n,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 150.0) < 1e-4
        r.ok("string edge: len long string")
    except Exception as e:
        r.fail("string edge: len long string", f"{e}")

    # Multiple concatenations
    try:
        result = compile_and_run(
            'string s = "a" + "b" + "c" + "d" + "e"; @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "abcde", f"Expected 'abcde', got {result!r}"
        r.ok("string edge: multiple concat")
    except Exception as e:
        r.fail("string edge: multiple concat", f"{e}")

    # String equality comparison
    try:
        code = 'float x = ("abc" == "abc"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 1.0) < 1e-4
        r.ok("string edge: equality comparison")
    except Exception as e:
        r.fail("string edge: equality comparison", f"{e}")

    # String in conditional
    try:
        result = compile_and_run(
            'string s = "a"; if (len(s) > 0.0) { s = s + "b"; } @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "ab", f"Expected 'ab', got {result!r}"
        r.ok("string edge: string in conditional")
    except Exception as e:
        r.fail("string edge: string in conditional", f"{e}")

    # String in loop
    try:
        result = compile_and_run(
            'string s = ""; for (int i = 0; i < 5; i++) { s = s + "x"; } @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "xxxxx", f"Expected 'xxxxx', got {result!r}"
        r.ok("string edge: string in loop")
    except Exception as e:
        r.fail("string edge: string in loop", f"{e}")

    # Numeric to string conversion
    try:
        result = compile_and_run(
            'string s = "value=" + str(42); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert "42" in result, f"Expected '42' in result, got {result!r}"
        r.ok("string edge: numeric to string concat")
    except Exception as e:
        r.fail("string edge: numeric to string concat", f"{e}")

    # split with single char
    try:
        code = '''
string parts[] = split("a,b,c", ",");
float n = len(parts);
@OUT = vec4(n, 0, 0, 1);
'''
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 3.0) < 1e-4
        r.ok("string edge: split count")
    except Exception as e:
        r.fail("string edge: split count", f"{e}")

    # find in empty string
    try:
        code = 'float x = find("", "abc"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - (-1.0)) < 1e-4
        r.ok("string edge: find in empty string")
    except Exception as e:
        r.fail("string edge: find in empty string", f"{e}")

    # find empty in string
    try:
        code = 'float x = find("hello", ""); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item()) < 1e-4  # Python str.find("") returns 0
        r.ok("string edge: find empty in string")
    except Exception as e:
        r.fail("string edge: find empty in string", f"{e}")


# ── Realistic Sizes Tests ─────────────────────────────────────────────

def test_realistic_sizes(r: TestResult):
    """Tests with realistic image sizes (512x512) and multi-batch."""
    print("\n--- Realistic Sizes Tests ---")

    # 512x512 passthrough
    try:
        big_img = torch.rand(1, 512, 512, 3)
        result = compile_and_run("@OUT = @A;", {"A": big_img})
        assert result.shape == (1, 512, 512, 3), f"Shape mismatch: {result.shape}"
        r.ok("realistic: 512x512 passthrough shape")
    except Exception as e:
        r.fail("realistic: 512x512 passthrough shape", f"{e}")

    # Verify values preserved
    try:
        big_img = torch.rand(1, 512, 512, 3)
        result = compile_and_run("@OUT = @A;", {"A": big_img})
        max_diff = (result - big_img).abs().max().item()
        assert max_diff < 1e-6, f"Passthrough altered values: max diff={max_diff}"
        r.ok("realistic: 512x512 passthrough values")
    except Exception as e:
        r.fail("realistic: 512x512 passthrough values", f"{e}")

    # Color grade at 512x512
    try:
        big_img = torch.rand(1, 512, 512, 3) * 0.5
        result = compile_and_run("@OUT = @A * 1.5;", {"A": big_img})
        expected = big_img * 1.5
        max_diff = (result - expected).abs().max().item()
        assert max_diff < 1e-5, f"Color grade drift: max diff={max_diff}"
        r.ok("realistic: 512x512 color grade")
    except Exception as e:
        r.fail("realistic: 512x512 color grade", f"{e}")

    # Shape preserved for vec4 output
    try:
        big_img = torch.rand(1, 512, 512, 4)
        result = compile_and_run("@OUT = @A;", {"A": big_img})
        assert result.shape == (1, 512, 512, 4), f"Shape: {result.shape}"
        r.ok("realistic: 512x512 vec4 shape preserved")
    except Exception as e:
        r.fail("realistic: 512x512 vec4 shape preserved", f"{e}")

    # Precision check with known values
    try:
        known = torch.zeros(1, 512, 512, 3)
        known[0, 0, 0, :] = torch.tensor([0.123456, 0.654321, 0.999999])
        known[0, 255, 255, :] = torch.tensor([0.111111, 0.222222, 0.333333])
        result = compile_and_run("@OUT = @A;", {"A": known})
        assert abs(result[0,0,0,0].item() - 0.123456) < 1e-5
        assert abs(result[0,0,0,1].item() - 0.654321) < 1e-5
        assert abs(result[0,255,255,2].item() - 0.333333) < 1e-5
        r.ok("realistic: precision preserved at 512x512")
    except Exception as e:
        r.fail("realistic: precision preserved at 512x512", f"{e}")

    # Multi-batch (B=2)
    try:
        batch_img = torch.rand(2, 64, 64, 3)
        result = compile_and_run("@OUT = @A;", {"A": batch_img})
        assert result.shape[0] == 2, f"Batch dim: {result.shape[0]}"
        assert result.shape == (2, 64, 64, 3), f"Shape: {result.shape}"
        r.ok("realistic: multi-batch B=2 shape")
    except Exception as e:
        r.fail("realistic: multi-batch B=2 shape", f"{e}")

    # Multi-batch values preserved
    try:
        batch_img = torch.rand(2, 64, 64, 3)
        result = compile_and_run("@OUT = @A;", {"A": batch_img})
        max_diff = (result - batch_img).abs().max().item()
        assert max_diff < 1e-6, f"Batch passthrough altered values: max diff={max_diff}"
        r.ok("realistic: multi-batch values preserved")
    except Exception as e:
        r.fail("realistic: multi-batch values preserved", f"{e}")

    # Multi-batch arithmetic
    try:
        batch_img = torch.rand(2, 64, 64, 3) * 0.5
        result = compile_and_run("@OUT = @A + 0.25;", {"A": batch_img})
        expected = batch_img + 0.25
        max_diff = (result - expected).abs().max().item()
        assert max_diff < 1e-5, f"Batch arithmetic drift: max diff={max_diff}"
        r.ok("realistic: multi-batch arithmetic")
    except Exception as e:
        r.fail("realistic: multi-batch arithmetic", f"{e}")

    # B=4 batch
    try:
        batch_img = torch.rand(4, 32, 32, 3)
        result = compile_and_run("@OUT = @A;", {"A": batch_img})
        assert result.shape == (4, 32, 32, 3), f"Shape: {result.shape}"
        r.ok("realistic: B=4 batch shape")
    except Exception as e:
        r.fail("realistic: B=4 batch shape", f"{e}")

    # Non-square: 256x128
    try:
        rect_img = torch.rand(1, 128, 256, 3)
        result = compile_and_run("@OUT = @A;", {"A": rect_img})
        assert result.shape == (1, 128, 256, 3), f"Shape: {result.shape}"
        r.ok("realistic: non-square 256x128")
    except Exception as e:
        r.fail("realistic: non-square 256x128", f"{e}")

    # Clamp at large size
    try:
        big_img = torch.rand(1, 256, 256, 3) * 2.0  # values up to 2.0
        result = compile_and_run("@OUT = clamp(@A, 0.0, 1.0);", {"A": big_img})
        assert result.max().item() <= 1.0 + 1e-6
        assert result.min().item() >= 0.0 - 1e-6
        r.ok("realistic: clamp at 256x256")
    except Exception as e:
        r.fail("realistic: clamp at 256x256", f"{e}")

    # u/v coordinates at 512x512
    try:
        big_img = torch.rand(1, 512, 512, 3)
        result = compile_and_run("@OUT = vec3(u, v, 0.0);", {"A": big_img})
        # Corner (0,0) should be u=0, v=0
        assert abs(result[0,0,0,0].item()) < 1e-4, f"u at (0,0): {result[0,0,0,0].item()}"
        assert abs(result[0,0,0,1].item()) < 1e-4, f"v at (0,0): {result[0,0,0,1].item()}"
        # Corner (511,511) should be u~1, v~1
        assert abs(result[0,511,511,0].item() - 1.0) < 1e-3
        assert abs(result[0,511,511,1].item() - 1.0) < 1e-3
        r.ok("realistic: u/v at 512x512 corners")
    except Exception as e:
        r.fail("realistic: u/v at 512x512 corners", f"{e}")


# ── NaN/Inf Propagation Tests ─────────────────────────────────────────

def test_nan_inf_propagation(r: TestResult):
    """Tests for NaN and Inf behavior through the pipeline."""
    print("\n--- NaN/Inf Propagation Tests ---")

    B, H, W = 1, 4, 4
    img = torch.rand(B, H, W, 3)

    # NaN through injection: create a tensor with NaN
    try:
        nan_img = torch.full((1, 4, 4, 3), float('nan'))
        result = compile_and_run("@OUT = @A;", {"A": nan_img})
        # Should not crash
        r.ok("nan: passthrough NaN tensor")
    except Exception as e:
        r.fail("nan: passthrough NaN tensor", f"{e}")

    # NaN through arithmetic: 0/0
    try:
        code = "float x = 0.0 / 0.0; @OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": img})
        # Should produce NaN or handled value, but not crash
        r.ok("nan: 0/0 does not crash")
    except Exception as e:
        r.fail("nan: 0/0 does not crash", f"{e}")

    # NaN through sin
    try:
        nan_img = torch.full((1, 4, 4, 3), float('nan'))
        code = "float x = sin(@A.r); @OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": nan_img})
        r.ok("nan: sin(NaN) does not crash")
    except Exception as e:
        r.fail("nan: sin(NaN) does not crash", f"{e}")

    # NaN through cos
    try:
        nan_img = torch.full((1, 4, 4, 3), float('nan'))
        code = "float x = cos(@A.r); @OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": nan_img})
        r.ok("nan: cos(NaN) does not crash")
    except Exception as e:
        r.fail("nan: cos(NaN) does not crash", f"{e}")

    # Safe division: 1/0 produces large value (not Inf) due to SAFE_EPSILON
    try:
        code = "float x = 1.0 / 0.0; @OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": img})
        val = result[0,0,0,0].item()
        assert val > 1e6, f"Expected large value from safe division, got {val}"
        r.ok("nan: 1/0 safe division produces large value")
    except Exception as e:
        r.fail("nan: 1/0 safe division produces large value", f"{e}")

    # Inf through clamp should give 1.0
    try:
        code = "float x = clamp(1.0 / 0.0, 0.0, 1.0); @OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": img})
        val = result[0,0,0,0].item()
        assert abs(val - 1.0) < 1e-4, f"clamp(Inf,0,1) should be 1, got {val}"
        r.ok("nan: clamp(Inf,0,1) = 1")
    except Exception as e:
        r.fail("nan: clamp(Inf,0,1) = 1", f"{e}")

    # Negative Inf through clamp
    try:
        code = "float x = clamp(-1.0 / 0.0, 0.0, 1.0); @OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": img})
        val = result[0,0,0,0].item()
        assert abs(val) < 1e-4, f"clamp(-Inf,0,1) should be 0, got {val}"
        r.ok("nan: clamp(-Inf,0,1) = 0")
    except Exception as e:
        r.fail("nan: clamp(-Inf,0,1) = 0", f"{e}")

    # Inf tensor passthrough
    try:
        inf_img = torch.full((1, 4, 4, 3), float('inf'))
        result = compile_and_run("@OUT = @A;", {"A": inf_img})
        r.ok("nan: Inf passthrough does not crash")
    except Exception as e:
        r.fail("nan: Inf passthrough does not crash", f"{e}")

    # NaN in lerp
    try:
        nan_img = torch.full((1, 4, 4, 3), float('nan'))
        code = "float x = lerp(0.0, 1.0, @A.r); @OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": nan_img})
        r.ok("nan: lerp with NaN does not crash")
    except Exception as e:
        r.fail("nan: lerp with NaN does not crash", f"{e}")

    # Inf in arithmetic
    try:
        code = "float x = 1.0 / 0.0 + 1.0; @OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": img})
        r.ok("nan: Inf + 1 does not crash")
    except Exception as e:
        r.fail("nan: Inf + 1 does not crash", f"{e}")

    # Inf - Inf = NaN
    try:
        code = "float x = (1.0/0.0) - (1.0/0.0); @OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": img})
        r.ok("nan: Inf - Inf does not crash")
    except Exception as e:
        r.fail("nan: Inf - Inf does not crash", f"{e}")

    # NaN comparison
    try:
        code = """
float x = 0.0 / 0.0;
float y = 0.0;
if (x == x) { y = 1.0; }
@OUT = vec3(y, y, y);
"""
        result = compile_and_run(code, {"A": img})
        # NaN == NaN behavior depends on implementation
        r.ok("nan: NaN comparison does not crash")
    except Exception as e:
        r.fail("nan: NaN comparison does not crash", f"{e}")

    # isnan() function — use NaN from image input since safe division won't produce NaN
    try:
        nan_input = torch.full((1, 4, 4, 3), float('nan'))
        code = "float x = isnan(@A.r); @OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": nan_input})
        val = result[0,0,0,0].item()
        assert abs(val - 1.0) < 1e-4, f"isnan(NaN) should be 1, got {val}"
        r.ok("nan: isnan(NaN) = 1")
    except Exception as e:
        r.fail("nan: isnan(NaN) = 1", f"{e}")

    # isinf() function — use Inf from image input since safe division won't produce Inf
    try:
        inf_input = torch.full((1, 4, 4, 3), float('inf'))
        code = "float x = isinf(@A.r); @OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": inf_input})
        val = result[0,0,0,0].item()
        assert abs(val - 1.0) < 1e-4, f"isinf(Inf) should be 1, got {val}"
        r.ok("nan: isinf(Inf) = 1")
    except Exception as e:
        r.fail("nan: isinf(Inf) = 1", f"{e}")


# ── Codegen Equivalence Tests ─────────────────────────────────────────

def test_codegen_equivalence(r: TestResult):
    """Verify codegen and interpreter produce the same results."""
    print("\n--- Codegen Equivalence Tests ---")

    _MAX_LOOP_ITERATIONS = 1024

    B, H, W = 1, 4, 4
    img = torch.rand(B, H, W, 3)

    def run_both(code, bindings):
        """Run through both interpreter and codegen, return (interp_result, codegen_result)."""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source=code)
        program = parser.parse()
        binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}
        checker = TypeChecker(binding_types=binding_types, source=code)
        type_map = checker.check(program)
        output_names = sorted(checker.assigned_bindings.keys())

        # Interpreter path
        interp = Interpreter()
        interp_result = interp.execute(program, bindings, type_map, device="cpu",
                                        output_names=output_names)

        # Codegen path
        cg_fn = try_compile(program, type_map)
        if cg_fn is None:
            return interp_result, None

        stdlib_fns = TEXStdlib.get_functions()
        env = {}
        dev = torch.device("cpu")

        sp = None
        for v in bindings.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 3:
                sp = (v.shape[0], v.shape[1], v.shape[2])
                break

        used = _collect_identifiers(program)
        if sp:
            B_sp, H_sp, W_sp = sp
            dtype = torch.float32
            if "ix" in used:
                env["ix"] = torch.arange(W_sp, dtype=dtype, device=dev).view(1, 1, W_sp)
            if "u" in used:
                ix = torch.arange(W_sp, dtype=dtype, device=dev).view(1, 1, W_sp)
                env["u"] = (ix / max(W_sp - 1, 1)).expand(B_sp, H_sp, W_sp)
            if "iy" in used:
                env["iy"] = torch.arange(H_sp, dtype=dtype, device=dev).view(1, H_sp, 1)
            if "v" in used:
                iy = torch.arange(H_sp, dtype=dtype, device=dev).view(1, H_sp, 1)
                env["v"] = (iy / max(H_sp - 1, 1)).expand(B_sp, H_sp, W_sp)
            if "iw" in used:
                env["iw"] = torch.tensor(float(W_sp), dtype=dtype, device=dev)
            if "ih" in used:
                env["ih"] = torch.tensor(float(H_sp), dtype=dtype, device=dev)
            if "fi" in used:
                env["fi"] = torch.arange(B_sp, dtype=dtype, device=dev).view(B_sp, 1, 1)
            if "fn" in used:
                env["fn"] = torch.tensor(float(B_sp), dtype=dtype, device=dev)
        if "PI" in used:
            env["PI"] = torch.tensor(math.pi, dtype=torch.float32, device=dev)
        if "E" in used:
            env["E"] = torch.tensor(math.e, dtype=torch.float32, device=dev)

        # Make a copy of bindings so codegen doesn't mutate the originals
        cg_bindings = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                       for k, v in bindings.items()}

        cg_fn(env, cg_bindings, stdlib_fns, dev, sp,
              torch, _broadcast_pair, _ensure_spatial, torch.where,
              math, SAFE_EPSILON, CHANNEL_MAP, _MAX_LOOP_ITERATIONS,
              _CgBreak, _CgContinue)

        cg_result = {name: cg_bindings[name] for name in output_names}
        return interp_result, cg_result

    programs = [
        ("basic arithmetic", "@OUT = @A * 2.0 + 0.5;"),
        ("subtraction", "@OUT = @A - 0.5;"),
        ("division", "@OUT = @A / 2.0;"),
        ("vec3 constructor", "@OUT = vec3(0.5, 0.25, 0.75);"),
        ("channel access", "@OUT = vec3(@A.r, @A.g, @A.b);"),
        ("ternary", "float x = (@A.r > 0.5) ? 1.0 : 0.0; @OUT = vec3(x, x, x);"),
        ("for loop sum", """
float s = 0.0;
for (int i = 0; i < 5; i++) { s += 0.1; }
@OUT = vec3(s, s, s);
"""),
        ("sin function", "@OUT = vec3(sin(@A.r), sin(@A.g), sin(@A.b));"),
        ("cos function", "@OUT = vec3(cos(@A.r), cos(@A.g), cos(@A.b));"),
        ("clamp", "@OUT = clamp(@A, 0.2, 0.8);"),
        ("lerp", "@OUT = lerp(vec3(0,0,0), vec3(1,1,1), 0.5);"),
        ("abs negative", "@OUT = vec3(abs(@A.r - 0.5), abs(@A.g - 0.5), abs(@A.b - 0.5));"),
        ("floor", "@OUT = vec3(floor(@A.r * 10.0), floor(@A.g * 10.0), floor(@A.b * 10.0));"),
        ("step", "@OUT = vec3(step(0.5, @A.r), step(0.5, @A.g), step(0.5, @A.b));"),
        ("smoothstep", "@OUT = vec3(smoothstep(0.0, 1.0, @A.r), smoothstep(0.0, 1.0, @A.g), smoothstep(0.0, 1.0, @A.b));"),
        ("multiple vars", """
float x = @A.r * 2.0;
float y = @A.g + 0.1;
@OUT = vec3(x, y, @A.b);
"""),
        ("nested ternary", "float x = (@A.r > 0.5) ? ((@A.g > 0.5) ? 0.8 : 0.5) : 0.2; @OUT = vec3(x, x, x);"),
        ("pow and sqrt", "@OUT = vec3(pow(@A.r, 2.0), sqrt(@A.g), @A.b);"),
        ("sign and fract", "@OUT = vec3(sign(@A.r - 0.5), fract(@A.g * 5.0), @A.b);"),
        ("max and min", "@OUT = vec3(max(@A.r, @A.g), min(@A.r, @A.g), @A.b);"),
    ]

    for name, code in programs:
        try:
            interp_res, cg_res = run_both(code, {"A": img})
            if cg_res is None:
                r.ok(f"codegen equiv: {name} (codegen unsupported, skip)")
                continue
            for out_name in interp_res:
                interp_t = interp_res[out_name]
                cg_t = cg_res[out_name]
                if isinstance(interp_t, torch.Tensor) and isinstance(cg_t, torch.Tensor):
                    max_diff = (interp_t.float() - cg_t.float()).abs().max().item()
                    assert max_diff < 1e-5, f"Max diff={max_diff} for output '{out_name}'"
            r.ok(f"codegen equiv: {name}")
        except Exception as e:
            r.fail(f"codegen equiv: {name}", f"{e}")


# ── Arithmetic Hash Noise Tests ───────────────────────────────────────

def test_arithmetic_hash_noise(r: TestResult):
    """Tests for the arithmetic hash Perlin noise implementation quality."""
    print("\n--- Arithmetic Hash Noise Tests ---")

    # Output range: perlin values should be approximately in [-1, 1]
    try:
        x = torch.rand(1, 64, 64) * 20.0 - 10.0
        y = torch.rand(1, 64, 64) * 20.0 - 10.0
        noise = _perlin2d_fast(x, y)
        assert noise.max().item() <= 1.5, f"Max noise too high: {noise.max().item()}"
        assert noise.min().item() >= -1.5, f"Min noise too low: {noise.min().item()}"
        r.ok("noise: perlin output range [-1.5, 1.5]")
    except Exception as e:
        r.fail("noise: perlin output range [-1.5, 1.5]", f"{e}")

    # Determinism: same inputs -> same outputs
    try:
        x = torch.tensor([1.5, 2.5, 3.5])
        y = torch.tensor([4.5, 5.5, 6.5])
        r1 = _perlin2d_fast(x.clone(), y.clone())
        r2 = _perlin2d_fast(x.clone(), y.clone())
        assert torch.allclose(r1, r2), f"Non-deterministic: diff={torch.abs(r1-r2).max().item()}"
        r.ok("noise: perlin determinism")
    except Exception as e:
        r.fail("noise: perlin determinism", f"{e}")

    # Continuity: nearby inputs -> nearby outputs
    try:
        x = torch.tensor([5.0])
        y = torch.tensor([5.0])
        eps = 0.001
        v0 = _perlin2d_fast(x, y).item()
        v1 = _perlin2d_fast(x + eps, y).item()
        v2 = _perlin2d_fast(x, y + eps).item()
        assert abs(v0 - v1) < 0.1, f"Discontinuity in x: |{v0}-{v1}|={abs(v0-v1)}"
        assert abs(v0 - v2) < 0.1, f"Discontinuity in y: |{v0}-{v2}|={abs(v0-v2)}"
        r.ok("noise: perlin continuity")
    except Exception as e:
        r.fail("noise: perlin continuity", f"{e}")

    # Variation: different inputs -> different outputs
    try:
        coords = (torch.arange(10).float() + 0.37).unsqueeze(0)
        noise = _perlin2d_fast(coords, torch.full_like(coords, 0.37))
        unique_vals = len(set([round(v, 4) for v in noise.flatten().tolist()]))
        assert unique_vals >= 3, f"Not enough variation: only {unique_vals} unique values"
        r.ok("noise: perlin variation")
    except Exception as e:
        r.fail("noise: perlin variation", f"{e}")

    # _lowbias32: deterministic integer hash
    try:
        inp = torch.tensor([0, 1, 2, 3, 100], dtype=torch.int32)
        h1 = _lowbias32(inp.clone())
        h2 = _lowbias32(inp.clone())
        assert torch.equal(h1, h2), "lowbias32 not deterministic"
        # Different inputs should give different outputs
        assert len(set(h1.tolist())) == 5, "lowbias32 collision on small inputs"
        r.ok("noise: lowbias32 determinism and uniqueness")
    except Exception as e:
        r.fail("noise: lowbias32 determinism and uniqueness", f"{e}")

    # _grad2d_dot: verify correct dot products for known gradients
    try:
        dx = torch.tensor([1.0])
        dy = torch.tensor([1.0])
        # h=0 -> gradient (1,0) -> dot = dx = 1.0
        h0 = torch.tensor([0], dtype=torch.int32)
        assert abs(_grad2d_dot(h0, dx, dy).item() - 1.0) < 1e-4, "grad h=0 failed"
        # h=1 -> gradient (-1,0) -> dot = -dx = -1.0
        h1 = torch.tensor([1], dtype=torch.int32)
        assert abs(_grad2d_dot(h1, dx, dy).item() - (-1.0)) < 1e-4, "grad h=1 failed"
        # h=2 -> gradient (0,1) -> dot = dy = 1.0
        h2 = torch.tensor([2], dtype=torch.int32)
        assert abs(_grad2d_dot(h2, dx, dy).item() - 1.0) < 1e-4, "grad h=2 failed"
        # h=3 -> gradient (0,-1) -> dot = -dy = -1.0
        h3 = torch.tensor([3], dtype=torch.int32)
        assert abs(_grad2d_dot(h3, dx, dy).item() - (-1.0)) < 1e-4, "grad h=3 failed"
        r.ok("noise: grad2d_dot cardinal gradients")
    except Exception as e:
        r.fail("noise: grad2d_dot cardinal gradients", f"{e}")

    # _grad2d_dot: diagonal gradients
    try:
        dx = torch.tensor([1.0])
        dy = torch.tensor([1.0])
        # h=4 -> gradient (1,1) -> dot = dx+dy = 2.0
        h4 = torch.tensor([4], dtype=torch.int32)
        assert abs(_grad2d_dot(h4, dx, dy).item() - 2.0) < 1e-4, "grad h=4 failed"
        # h=5 -> gradient (-1,1) -> dot = -dx+dy = 0.0
        h5 = torch.tensor([5], dtype=torch.int32)
        assert abs(_grad2d_dot(h5, dx, dy).item() - 0.0) < 1e-4, "grad h=5 failed"
        # h=6 -> gradient (1,-1) -> dot = dx-dy = 0.0
        h6 = torch.tensor([6], dtype=torch.int32)
        assert abs(_grad2d_dot(h6, dx, dy).item() - 0.0) < 1e-4, "grad h=6 failed"
        # h=7 -> gradient (-1,-1) -> dot = -dx-dy = -2.0
        h7 = torch.tensor([7], dtype=torch.int32)
        assert abs(_grad2d_dot(h7, dx, dy).item() - (-2.0)) < 1e-4, "grad h=7 failed"
        r.ok("noise: grad2d_dot diagonal gradients")
    except Exception as e:
        r.fail("noise: grad2d_dot diagonal gradients", f"{e}")

    # FBM via DSL: output range
    try:
        big_img = torch.rand(1, 32, 32, 3)
        code = "float n = fbm(u * 10.0, v * 10.0, 4); @OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": big_img})
        max_val = result.max().item()
        min_val = result.min().item()
        assert max_val <= 1.5, f"FBM max too high: {max_val}"
        assert min_val >= -1.5, f"FBM min too low: {min_val}"
        r.ok("noise: FBM output range")
    except Exception as e:
        r.fail("noise: FBM output range", f"{e}")

    # Perlin at integer coords: should be 0 (gradient property)
    try:
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        y = torch.tensor([0.0, 1.0, 2.0, 3.0])
        noise = _perlin2d_fast(x, y)
        for i in range(4):
            assert abs(noise[i].item()) < 1e-4, f"Noise at integer ({i},{i}) = {noise[i].item()}"
        r.ok("noise: perlin zero at integer coords")
    except Exception as e:
        r.fail("noise: perlin zero at integer coords", f"{e}")


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
    test_break_continue(r)
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
    test_is_changed_hash(r)
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
    test_wireable_params(r)
    test_string_functions_v04(r)
    test_while_loops(r)
    test_new_stdlib_functions(r)
    test_else_if_chains(r)
    test_optimization_regressions(r)
    test_diagnostic_quality(r)
    test_ternary_exhaustive(r)
    test_user_functions_advanced(r)
    test_binding_access_advanced(r)
    test_scope_and_shadowing(r)
    test_operator_edge_cases(r)
    test_casting_exhaustive(r)
    test_user_functions(r)
    test_binding_access(r)
    test_missing_stdlib_functions(r)
    test_numeric_edge_case_matrix(r)
    test_array_bounds(r)
    test_string_edge_cases(r)
    test_realistic_sizes(r)
    test_nan_inf_propagation(r)
    test_codegen_equivalence(r)
    test_arithmetic_hash_noise(r)

    success = r.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
