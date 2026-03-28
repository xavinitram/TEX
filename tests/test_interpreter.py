"""Interpreter tests — execution, loops, control flow, compound assignments."""
from helpers import *


def test_interpreter(r: SubTestResult):
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

def test_for_loops(r: SubTestResult):
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

def test_break_continue(r: SubTestResult):
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

def test_compound_assignments(r: SubTestResult):
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


# ── While Loop Tests ─────────────────────────────────────────────────

def test_while_loops(r: SubTestResult):
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
