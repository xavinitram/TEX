"""Language feature tests — conditionals, scoping, operators, casting, types, swizzles."""
from helpers import *


def test_channel_assignment(r: SubTestResult):
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

    # Multi-channel assign (.rg, .rgb, .xyz, etc.) is now supported
    try:
        code = "@OUT = vec4(0.0);\n@OUT.rg = vec2(0.5, 0.8);"
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[..., 0].mean().item() - 0.5) < 1e-5
        assert abs(result[..., 1].mean().item() - 0.8) < 1e-5
        assert abs(result[..., 2].mean().item() - 0.0) < 1e-5  # unchanged
        r.ok("channel_assign: multi .rg")
    except Exception as e:
        r.fail("channel_assign: multi .rg", f"{e}\n{traceback.format_exc()}")

    # Multi-channel .rgb assign preserves alpha
    try:
        code = "@OUT = vec4(0.0, 0.0, 0.0, 0.9);\n@OUT.rgb = vec3(1.0, 0.5, 0.25);"
        result = compile_and_run(code, {"A": test_img})
        assert abs(result[..., 0].mean().item() - 1.0) < 1e-5
        assert abs(result[..., 1].mean().item() - 0.5) < 1e-5
        assert abs(result[..., 2].mean().item() - 0.25) < 1e-5
        assert abs(result[..., 3].mean().item() - 0.9) < 1e-5  # alpha preserved
        r.ok("channel_assign: multi .rgb preserves alpha")
    except Exception as e:
        r.fail("channel_assign: multi .rgb preserves alpha", f"{e}\n{traceback.format_exc()}")

    # Invalid channel error
    try:
        code = "@OUT = vec4(0.0);\n@OUT.q = 1.0;"
        compile_and_run(code, {"A": test_img})
        r.fail("channel_assign: invalid error", "Should have raised error")
    except (InterpreterError, RuntimeError, TypeCheckError) as e:
        r.ok("channel_assign: invalid error")
    except Exception as e:
        r.fail("channel_assign: invalid error", f"{e}\n{traceback.format_exc()}")


def test_output_types(r: SubTestResult):
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


def test_if_without_else(r: SubTestResult):
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


def test_swizzle_patterns(r: SubTestResult):
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


def test_else_if_chains(r: SubTestResult):
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


def test_ternary_exhaustive(r: SubTestResult):
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


def test_scope_and_shadowing(r: SubTestResult):
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


def test_operator_edge_cases(r: SubTestResult):
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

    # Assignment to .rgb — multi-channel swizzle assignment is now supported
    try:
        result = compile_and_run("""
            vec4 c = vec4(0.0, 0.0, 0.0, 0.9);
            c.rgb = vec3(1.0, 0.5, 0.25);
            @OUT = c;
        """, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-5
        assert abs(result[0, 0, 0, 1].item() - 0.5) < 1e-5
        assert abs(result[0, 0, 0, 2].item() - 0.25) < 1e-5
        assert abs(result[0, 0, 0, 3].item() - 0.9) < 1e-5  # alpha preserved
        r.ok("operator: .rgb assignment")
    except Exception as e:
        r.fail("operator: .rgb assignment", str(e))

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


def test_casting_exhaustive(r: SubTestResult):
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


def test_vec2_type(r: SubTestResult):
    """Tests for the vec2 first-class type."""
    print("\n--- Vec2 Type Tests ---")

    B, H, W = 1, 4, 4
    img4 = torch.rand(B, H, W, 4)
    img3 = torch.rand(B, H, W, 3)

    # Constructor: vec2(scalar) broadcast
    try:
        result = compile_and_run("@OUT = vec2(0.5);", {})
        assert result.shape[-1] == 2, f"Expected 2 channels, got {result.shape[-1]}"
        assert torch.allclose(result[..., 0], result[..., 1])
        r.ok("vec2: scalar broadcast constructor")
    except Exception as e:
        r.fail("vec2: scalar broadcast constructor", f"{e}\n{traceback.format_exc()}")

    # Constructor: vec2(a, b)
    try:
        result = compile_and_run("@OUT = vec2(0.25, 0.75);", {})
        assert result.shape[-1] == 2
        assert abs(result[..., 0].item() - 0.25) < 1e-5
        assert abs(result[..., 1].item() - 0.75) < 1e-5
        r.ok("vec2: two-arg constructor")
    except Exception as e:
        r.fail("vec2: two-arg constructor", f"{e}\n{traceback.format_exc()}")

    # Variable declaration
    try:
        result = compile_and_run("vec2 p = vec2(0.1, 0.9); @OUT = p;", {})
        assert result.shape[-1] == 2
        r.ok("vec2: variable declaration")
    except Exception as e:
        r.fail("vec2: variable declaration", f"{e}\n{traceback.format_exc()}")

    # Swizzle .xy / .rg on vec4 returns vec2
    try:
        result = compile_and_run("@OUT = @A.xy;", {"A": img4})
        assert result.shape[-1] == 2
        assert torch.allclose(result[..., 0], img4[..., 0])
        assert torch.allclose(result[..., 1], img4[..., 1])
        r.ok("vec2: .xy swizzle on vec4")
    except Exception as e:
        r.fail("vec2: .xy swizzle on vec4", f"{e}\n{traceback.format_exc()}")

    # Swizzle .rg on vec3
    try:
        result = compile_and_run("@OUT = @A.rg;", {"A": img3})
        assert result.shape[-1] == 2
        assert torch.allclose(result[..., 0], img3[..., 0])
        assert torch.allclose(result[..., 1], img3[..., 1])
        r.ok("vec2: .rg swizzle on vec3")
    except Exception as e:
        r.fail("vec2: .rg swizzle on vec3", f"{e}\n{traceback.format_exc()}")

    # vec2 + float -> vec2 (broadcast)
    try:
        result = compile_and_run("@OUT = vec2(0.1, 0.2) + 0.5;", {})
        assert result.shape[-1] == 2
        assert abs(result[..., 0].item() - 0.6) < 1e-5
        assert abs(result[..., 1].item() - 0.7) < 1e-5
        r.ok("vec2: vec2 + float promotion")
    except Exception as e:
        r.fail("vec2: vec2 + float promotion", f"{e}\n{traceback.format_exc()}")

    # vec2 + vec3 -> vec3 (channel padding)
    try:
        result = compile_and_run("@OUT = vec2(1.0, 2.0) + vec3(0.1, 0.2, 0.3);", {})
        assert result.shape[-1] == 3, f"Expected 3 channels, got {result.shape[-1]}"
        assert abs(result[..., 0].item() - 1.1) < 1e-5
        assert abs(result[..., 1].item() - 2.2) < 1e-5
        assert abs(result[..., 2].item() - 0.3) < 1e-5  # 0 + 0.3
        r.ok("vec2: vec2 + vec3 promotion")
    except Exception as e:
        r.fail("vec2: vec2 + vec3 promotion", f"{e}\n{traceback.format_exc()}")

    # vec2 * vec2 -> vec2
    try:
        result = compile_and_run("@OUT = vec2(2.0, 3.0) * vec2(0.5, 0.25);", {})
        assert result.shape[-1] == 2
        assert abs(result[..., 0].item() - 1.0) < 1e-5
        assert abs(result[..., 1].item() - 0.75) < 1e-5
        r.ok("vec2: vec2 * vec2")
    except Exception as e:
        r.fail("vec2: vec2 * vec2", f"{e}\n{traceback.format_exc()}")

    # Channel assign .x on vec2
    try:
        result = compile_and_run("vec2 p = vec2(0.0, 0.0); p.x = 1.0; @OUT = p;", {})
        assert result.shape[-1] == 2
        assert abs(result[..., 0].item() - 1.0) < 1e-5
        assert abs(result[..., 1].item() - 0.0) < 1e-5
        r.ok("vec2: channel assign .x")
    except Exception as e:
        r.fail("vec2: channel assign .x", f"{e}\n{traceback.format_exc()}")

    # Output vec2 pads to IMAGE (3-channel) via _prepare_output
    try:
        raw = torch.tensor([[[[0.5, 0.8]]]])  # [1, 1, 1, 2]
        out = _prepare_output(raw, "IMAGE")
        assert out.shape[-1] == 3, f"Expected 3 channels after padding, got {out.shape[-1]}"
        assert abs(out[..., 0].item() - 0.5) < 1e-5
        assert abs(out[..., 1].item() - 0.8) < 1e-5
        assert abs(out[..., 2].item() - 0.0) < 1e-5  # zero-padded
        r.ok("vec2: output padding to IMAGE")
    except Exception as e:
        r.fail("vec2: output padding to IMAGE", f"{e}\n{traceback.format_exc()}")

    # vec2 in ternary
    try:
        result = compile_and_run("float cond = 1.0; @OUT = cond > 0.5 ? vec2(1.0, 0.0) : vec2(0.0, 1.0);", {})
        assert result.shape[-1] == 2
        assert abs(result[..., 0].item() - 1.0) < 1e-5
        assert abs(result[..., 1].item() - 0.0) < 1e-5
        r.ok("vec2: ternary expression")
    except Exception as e:
        r.fail("vec2: ternary expression", f"{e}\n{traceback.format_exc()}")

    # vec2 array
    try:
        result = compile_and_run(
            "vec2 arr[2]; arr[0] = vec2(0.1, 0.2); arr[1] = vec2(0.3, 0.4); @OUT = arr[1];",
            {}
        )
        assert result.shape[-1] == 2
        assert abs(result[..., 0].item() - 0.3) < 1e-5
        assert abs(result[..., 1].item() - 0.4) < 1e-5
        r.ok("vec2: array of vec2")
    except Exception as e:
        r.fail("vec2: array of vec2", f"{e}\n{traceback.format_exc()}")

    # _map_inferred_type maps VEC2 to IMAGE
    try:
        assert _map_inferred_type(TEXType.VEC2, False) == "IMAGE"
        r.ok("vec2: _map_inferred_type -> IMAGE")
    except Exception as e:
        r.fail("vec2: _map_inferred_type -> IMAGE", f"{e}\n{traceback.format_exc()}")
