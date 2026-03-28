"""Codegen equivalence and optimizer tests."""
from helpers import *


def test_codegen_equivalence(r: SubTestResult):
    """Verify codegen and interpreter produce the same results."""
    print("\n--- Codegen Equivalence Tests ---")

    B, H, W = 1, 4, 4
    torch.manual_seed(42)
    img = torch.rand(B, H, W, 3)

    # ── Single-input programs (binding A) ──
    programs = [
        # Basic arithmetic
        ("basic arithmetic", "@OUT = @A * 2.0 + 0.5;"),
        ("subtraction", "@OUT = @A - 0.5;"),
        ("division", "@OUT = @A / 2.0;"),
        ("modulo", "@OUT = vec3(fract(@A.r * 3.0), fract(@A.g * 3.0), fract(@A.b * 3.0));"),

        # Constructors
        ("vec3 constructor", "@OUT = vec3(0.5, 0.25, 0.75);"),
        ("vec3 broadcast", "@OUT = vec3(0.5);"),
        ("vec4 constructor", "@OUT = vec4(@A.r, @A.g, @A.b, 1.0);"),
        ("vec2 constructor", "@OUT = vec2(0.3, 0.7);"),
        ("vec2 broadcast", "@OUT = vec2(@A.r);"),
        ("vec2 arithmetic", "@OUT = vec2(@A.r, @A.g) * 2.0;"),

        # Channel access
        ("channel access", "@OUT = vec3(@A.r, @A.g, @A.b);"),
        (".rgb swizzle", "@OUT = @A.rgb;"),
        (".bgr swizzle", "@OUT = @A.bgr;"),
        (".xy swizzle", "@OUT = vec3(@A.xy, 0.0);"),
        ("single channel .r", "m@OUT = @A.r;"),
        ("single channel .g", "m@OUT = @A.g;"),

        # Ternary
        ("ternary", "float x = (@A.r > 0.5) ? 1.0 : 0.0; @OUT = vec3(x, x, x);"),
        ("nested ternary", "float x = (@A.r > 0.5) ? ((@A.g > 0.5) ? 0.8 : 0.5) : 0.2; @OUT = vec3(x, x, x);"),
        ("vec ternary", "@OUT = vec3((@A.r > 0.5) ? 1.0 : 0.0, (@A.g > 0.5) ? 1.0 : 0.0, @A.b);"),

        # For loops
        ("for loop sum", """
float s = 0.0;
for (int i = 0; i < 5; i++) { s += 0.1; }
@OUT = vec3(s, s, s);
"""),
        ("for loop accumulator", """
vec3 acc = vec3(0.0);
for (int i = 0; i < 4; i++) { acc += @A * 0.25; }
@OUT = acc;
"""),
        ("for loop nested", """
float s = 0.0;
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        s += 0.1;
    }
}
@OUT = vec3(s, s, s);
"""),

        # While loops
        ("while loop", """
float x = 0.0;
while (x < 0.5) { x += 0.1; }
@OUT = vec3(x, x, x);
"""),

        # Break/continue
        ("for break", """
float s = 0.0;
for (int i = 0; i < 10; i++) {
    if (s > 0.4) { break; }
    s += 0.1;
}
@OUT = vec3(s, s, s);
"""),
        ("for continue", """
float s = 0.0;
for (int i = 0; i < 5; i++) {
    if (i == 2) { continue; }
    s += 0.1;
}
@OUT = vec3(s, s, s);
"""),

        # Compound assignments
        ("compound +=", "vec3 c = @A; c += vec3(0.1); @OUT = c;"),
        ("compound -=", "vec3 c = @A; c -= vec3(0.1); @OUT = c;"),
        ("compound *=", "vec3 c = @A; c *= 2.0; @OUT = c;"),
        ("compound /=", "vec3 c = @A; c /= 2.0; @OUT = c;"),

        # Stdlib math
        ("sin function", "@OUT = vec3(sin(@A.r), sin(@A.g), sin(@A.b));"),
        ("cos function", "@OUT = vec3(cos(@A.r), cos(@A.g), cos(@A.b));"),
        ("tan function", "float x = @A.r * 0.5; @OUT = vec3(tan(x), 0.0, 0.0);"),
        ("asin function", "@OUT = vec3(asin(@A.r), 0.0, 0.0);"),
        ("acos function", "@OUT = vec3(acos(@A.r), 0.0, 0.0);"),
        ("atan function", "@OUT = vec3(atan(@A.r), 0.0, 0.0);"),
        ("atan2 function", "@OUT = vec3(atan2(@A.r, @A.g), 0.0, 0.0);"),
        ("exp function", "@OUT = vec3(exp(@A.r), 0.0, 0.0);"),
        ("abs negative", "@OUT = vec3(abs(@A.r - 0.5), abs(@A.g - 0.5), abs(@A.b - 0.5));"),
        ("floor", "@OUT = vec3(floor(@A.r * 10.0), floor(@A.g * 10.0), floor(@A.b * 10.0));"),
        ("ceil", "@OUT = vec3(ceil(@A.r * 10.0), ceil(@A.g * 10.0), ceil(@A.b * 10.0));"),
        ("round", "@OUT = vec3(round(@A.r * 10.0), round(@A.g * 10.0), round(@A.b * 10.0));"),
        ("trunc", "@OUT = vec3(trunc(@A.r * 10.0), trunc(@A.g * 10.0), trunc(@A.b * 10.0));"),
        ("sign and fract", "@OUT = vec3(sign(@A.r - 0.5), fract(@A.g * 5.0), @A.b);"),
        ("clamp", "@OUT = clamp(@A, 0.2, 0.8);"),
        ("lerp", "@OUT = lerp(vec3(0,0,0), vec3(1,1,1), 0.5);"),
        ("lerp spatial", "@OUT = lerp(vec3(0,0,0), @A, 0.5);"),
        ("step", "@OUT = vec3(step(0.5, @A.r), step(0.5, @A.g), step(0.5, @A.b));"),
        ("smoothstep", "@OUT = vec3(smoothstep(0.0, 1.0, @A.r), smoothstep(0.0, 1.0, @A.g), smoothstep(0.0, 1.0, @A.b));"),
        ("pow and sqrt", "@OUT = vec3(pow(@A.r, 2.0), sqrt(@A.g), @A.b);"),
        ("max and min", "@OUT = vec3(max(@A.r, @A.g), min(@A.r, @A.g), @A.b);"),
        ("dot product", "float d = dot(@A, vec3(0.2126, 0.7152, 0.0722)); @OUT = vec3(d, d, d);"),
        ("length", "float l = length(@A); @OUT = vec3(l, l, l);"),
        ("normalize", "@OUT = normalize(@A);"),
        ("distance", "float d = distance(@A, vec3(0.5)); @OUT = vec3(d, d, d);"),
        ("sincos", "vec2 sc = sincos(@A.r); @OUT = vec3(sc.x, sc.y, 0.0);"),
        ("luma", "float l = luma(@A); @OUT = vec3(l, l, l);"),
        ("degrees radians", "@OUT = vec3(degrees(@A.r), radians(@A.g * 360.0), 0.0);"),
        ("hypot", "@OUT = vec3(hypot(@A.r, @A.g), 0.0, 0.0);"),
        ("sinh cosh tanh", "@OUT = vec3(sinh(@A.r * 0.5), cosh(@A.g * 0.5), tanh(@A.b));"),

        # Newly inlined stdlib functions
        ("sqrt", "@OUT = vec3(sqrt(@A.r), sqrt(@A.g), sqrt(@A.b));"),
        ("log", "float x = log(@A.r + 0.1); @OUT = vec3(x, x, x);"),
        ("log2", "float x = log2(@A.r + 0.1); @OUT = vec3(x, x, x);"),
        ("log10", "float x = log10(@A.r + 0.1); @OUT = vec3(x, x, x);"),
        ("fract", "@OUT = vec3(fract(@A.r * 5.0), fract(@A.g * 5.0), fract(@A.b * 5.0));"),
        ("pow2", "float x = pow2(@A.r); @OUT = vec3(x, x, x);"),
        ("pow10", "float x = pow10(@A.r * 0.5); @OUT = vec3(x, x, x);"),
        ("isnan", "float x = isnan(@A.r); @OUT = vec3(x, x, x);"),
        ("isinf", "float x = isinf(@A.r); @OUT = vec3(x, x, x);"),
        ("smoothstep inline", "@OUT = vec3(smoothstep(0.2, 0.8, @A.r), smoothstep(0.2, 0.8, @A.g), smoothstep(0.2, 0.8, @A.b));"),
        ("step inline", "@OUT = vec3(step(0.3, @A.r), step(0.5, @A.g), step(0.7, @A.b));"),
        ("fit", "@OUT = vec3(fit(@A.r, 0.0, 1.0, 0.2, 0.8), fit(@A.g, 0.0, 1.0, 0.2, 0.8), @A.b);"),
        ("mod", "@OUT = vec3(mod(@A.r * 10.0, 3.0), mod(@A.g * 10.0, 3.0), @A.b);"),
        ("reflect", "@OUT = reflect(@A, normalize(vec3(0.0, 1.0, 0.0)));"),
        ("spow", "@OUT = vec3(spow(@A.r - 0.5, 2.0), spow(@A.g - 0.5, 2.0), @A.b);"),
        ("sdiv", "@OUT = vec3(sdiv(@A.r, @A.g), sdiv(@A.g, @A.b), 0.0);"),

        # Variables
        ("multiple vars", """
float x = @A.r * 2.0;
float y = @A.g + 0.1;
@OUT = vec3(x, y, @A.b);
"""),
        ("var reassignment", """
float x = @A.r;
x = x * 2.0;
x = x + 0.1;
@OUT = vec3(x, x, x);
"""),

        # If/else
        ("scalar if/else", """
float mode = 0.0;
vec3 c = @A;
if (mode > 0.5) {
    c = vec3(1.0, 0.0, 0.0);
} else {
    c = @A * 0.5;
}
@OUT = c;
"""),
        ("spatial if/else", """
float val = 0.0;
if (@A.r > 0.5) {
    val = 1.0;
} else {
    val = 0.0;
}
@OUT = vec3(val, val, val);
"""),

        # Casting
        ("int cast", "float x = float(int(@A.r * 10.0)); @OUT = vec3(x / 10.0, 0.0, 0.0);"),

        # Arrays
        ("float array basic", """
float arr[3];
arr[0] = @A.r;
arr[1] = @A.g;
arr[2] = @A.b;
@OUT = vec3(arr[0], arr[1], arr[2]);
"""),
        ("float array literal", """
float arr[] = {0.1, 0.2, 0.3};
@OUT = vec3(arr[0], arr[1], arr[2]);
"""),
        ("array in loop", """
float arr[5];
for (int i = 0; i < 5; i++) { arr[i] = 0.1 * float(i); }
@OUT = vec3(arr[0], arr[2], arr[4]);
"""),

        # User functions
        ("user function simple", """
float double_it(float x) { return x * 2.0; }
@OUT = vec3(double_it(@A.r), double_it(@A.g), double_it(@A.b));
"""),
        ("user function multi-param", """
float blend(float a, float b, float t) { return a * (1.0 - t) + b * t; }
@OUT = vec3(blend(@A.r, 1.0, 0.5), blend(@A.g, 0.0, 0.3), @A.b);
"""),
        ("user function with vec", """
vec3 tint(vec3 c, float t) { return c * t; }
@OUT = tint(@A, 0.5);
"""),

        # Channel assignment
        ("multi-channel .rgb assign", """
vec4 c = vec4(@A.r, @A.g, @A.b, 1.0);
c.rgb = vec3(0.5, 0.25, 0.1);
@OUT = c;
"""),
        ("multi-channel .xy assign", """
vec3 c = vec3(0.0);
c.xy = vec2(@A.r, @A.g);
@OUT = c;
"""),
        ("single channel assign", """
vec3 c = @A;
c.r = 1.0;
@OUT = c;
"""),

        # Builtins
        ("uv builtins", "@OUT = vec3(u, v, 0.0);"),
        ("pixel builtins", "@OUT = vec3(ix * px, iy * py, 0.0);"),
        ("size builtins", "@OUT = vec3(iw / 1000.0, ih / 1000.0, 0.0);"),
        ("PI constant", "@OUT = vec3(PI / 10.0, 0.0, 0.0);"),
        ("TAU constant", "@OUT = vec3(TAU / 10.0, 0.0, 0.0);"),
        ("E constant", "@OUT = vec3(E / 10.0, 0.0, 0.0);"),

        # Comparison operators
        ("eq operator", "@OUT = vec3((@A.r == @A.r) ? 1.0 : 0.0, 0.0, 0.0);"),
        ("neq operator", "@OUT = vec3((@A.r != @A.g) ? 1.0 : 0.0, 0.0, 0.0);"),
        ("lt gt operators", "@OUT = vec3((@A.r < 0.5) ? 1.0 : 0.0, (@A.g > 0.5) ? 1.0 : 0.0, 0.0);"),
        ("lte gte operators", "@OUT = vec3((@A.r <= 0.5) ? 1.0 : 0.0, (@A.g >= 0.5) ? 1.0 : 0.0, 0.0);"),
        ("logical and", "float x = (@A.r > 0.3 && @A.g > 0.3) ? 1.0 : 0.0; @OUT = vec3(x, x, x);"),
        ("logical or", "float x = (@A.r > 0.7 || @A.g > 0.7) ? 1.0 : 0.0; @OUT = vec3(x, x, x);"),
        ("logical not", "float x = !(@A.r > 0.5) ? 1.0 : 0.0; @OUT = vec3(x, x, x);"),

        # Unary
        ("unary negate", "@OUT = -@A;"),

        # Binding index access (fetch)
        ("fetch pixel", "@OUT = @A[0, 0];"),
        ("fetch with vars", "@OUT = @A[ix, iy];"),

        # Multi-output
        ("multi-output", """
@OUT = @A;
@mask = vec3(luma(@A));
"""),

        # Matte output
        ("matte output", "m@OUT = luma(@A);"),

        # Complex expressions
        ("chained math", "@OUT = clamp((@A - 0.5) * 2.0 + 0.5, 0.0, 1.0);"),
        ("nested functions", "@OUT = vec3(sqrt(abs(sin(@A.r))), sqrt(abs(cos(@A.g))), @A.b);"),
    ]

    # ── Two-input programs (bindings A + B) ──
    img_b = torch.rand(B, H, W, 3)
    two_input_programs = [
        ("two input add", "@OUT = @A + @B;"),
        ("two input lerp", "@OUT = lerp(@A, @B, 0.5);"),
        ("two input max", "@OUT = vec3(max(@A.r, @B.r), max(@A.g, @B.g), max(@A.b, @B.b));"),
        ("two input diff", "@OUT = vec3(abs(@A.r - @B.r), abs(@A.g - @B.g), abs(@A.b - @B.b));"),
        ("cross product", "@OUT = cross(@A, @B);"),
    ]

    for name, code in programs:
        assert_equiv(r, name, code, {"A": img}, B=B, H=H, W=W)

    for name, code in two_input_programs:
        assert_equiv(r, name, code, {"A": img, "B": img_b}, B=B, H=H, W=W)


# ── Optimization Regression Tests ──────────────────────────────────────

def test_optimization_regressions(r: SubTestResult):
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

    # ── Nested loop unrolling — outer var visible in inner loop body ──
    # Regression: _subst_stmt didn't handle ForLoop/WhileLoop, so when the
    # outer loop was unrolled the inner loop body still referenced the outer
    # variable by name but it was never declared → E6020 "not defined".
    try:
        code = """
vec3 acc = vec3(0.0);
for (int dy = -1; dy <= 1; dy = dy + 1) {
    for (int dx = -1; dx <= 1; dx = dx + 1) {
        float su = u + float(dx) / iw;
        float sv = v + float(dy) / ih;
        acc = acc + sample(@A, su, sv);
    }
}
@OUT = acc / 9.0;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert result.shape == (1, 8, 8, 3), f"Shape: {result.shape}"
        r.ok("nested loop unroll: outer var in inner body")
    except Exception as e:
        r.fail("nested loop unroll: outer var in inner body", f"{e}\n{traceback.format_exc()}")


# ── LICM Tests ─────────────────────────────────────────────────────────

def test_licm(r: SubTestResult):
    """Test Loop-Invariant Code Motion optimizer pass."""
    from TEX_Wrangle.tex_compiler.optimizer import optimize
    img = torch.rand(1, 8, 8, 3)

    # 1. LICM hoists invariant expression out of for loop
    try:
        code = """
float total = 0.0;
float base = 3.0;
float scale = sin(base * 2.0);
for (int i = 0; i < 5; i = i + 1) {
    total = total + scale;
}
@OUT = vec3(total);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = math.sin(3.0 * 2.0) * 5.0
        assert abs(result[0, 0, 0, 0].item() - expected) < 1e-4, \
            f"Expected {expected}, got {result[0, 0, 0, 0].item()}"
        r.ok("licm: basic invariant expression hoisted correctly")
    except Exception as e:
        r.fail("licm: basic invariant expression hoisted correctly", f"{e}\n{traceback.format_exc()}")

    # 2. LICM with loop-variant expression (should NOT be hoisted)
    try:
        code = """
float total = 0.0;
for (int i = 0; i < 10; i = i + 1) {
    total = total + float(i);
}
@OUT = vec3(total);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = sum(range(10))  # 45
        assert abs(result[0, 0, 0, 0].item() - expected) < 1e-4, \
            f"Expected {expected}, got {result[0, 0, 0, 0].item()}"
        r.ok("licm: loop-variant expression not broken")
    except Exception as e:
        r.fail("licm: loop-variant expression not broken", f"{e}\n{traceback.format_exc()}")

    # 3. LICM with nested loops — inner invariant w.r.t. inner loop
    try:
        code = """
float total = 0.0;
for (int i = 0; i < 3; i = i + 1) {
    float outer_val = float(i) * 10.0;
    for (int j = 0; j < 4; j = j + 1) {
        total = total + outer_val;
    }
}
@OUT = vec3(total);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = sum(i * 10.0 * 4 for i in range(3))  # 0 + 40 + 80 = 120
        assert abs(result[0, 0, 0, 0].item() - expected) < 1e-4, \
            f"Expected {expected}, got {result[0, 0, 0, 0].item()}"
        r.ok("licm: nested loops correct")
    except Exception as e:
        r.fail("licm: nested loops correct", f"{e}\n{traceback.format_exc()}")

    # 4. Verify LICM actually hoists at AST level
    try:
        # Use @A binding so expression can't be constant-folded
        # Use 10 iterations (> unroll threshold of 8) so loop is preserved
        code = """
float total = 0.0;
float base = @A.r;
for (int i = 0; i < 10; i = i + 1) {
    total = total + sin(base * 2.0);
}
@OUT = vec3(total);
"""
        tokens = Lexer(code).tokenize()
        prog = Parser(tokens, source="test").parse()
        tc = TypeChecker(binding_types={"A": TEXType.VEC3}, source="test")
        tc.check(prog)
        prog = optimize(prog)
        # After LICM, sin(base * 2.0) should be hoisted as _licm0 before the for loop
        from TEX_Wrangle.tex_compiler.ast_nodes import VarDecl, ForLoop
        found_licm_var = False
        found_for_after = False
        for stmt in prog.statements:
            if isinstance(stmt, VarDecl) and stmt.name.startswith("_licm"):
                found_licm_var = True
            if isinstance(stmt, ForLoop) and found_licm_var:
                found_for_after = True
        assert found_licm_var, "LICM should create a _licm temp variable"
        assert found_for_after, "ForLoop should appear after hoisted variable"
        r.ok("licm: AST-level hoisting verified")
    except Exception as e:
        r.fail("licm: AST-level hoisting verified", f"{e}\n{traceback.format_exc()}")

    # 5. LICM with while loop
    try:
        code = """
float total = 0.0;
float base = 2.0;
int count = 0;
while (count < 5) {
    total = total + cos(base * 3.14);
    count = count + 1;
}
@OUT = vec3(total);
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = math.cos(2.0 * 3.14) * 5.0
        assert abs(result[0, 0, 0, 0].item() - expected) < 1e-3, \
            f"Expected {expected}, got {result[0, 0, 0, 0].item()}"
        r.ok("licm: while loop invariant hoisted correctly")
    except Exception as e:
        r.fail("licm: while loop invariant hoisted correctly", f"{e}\n{traceback.format_exc()}")


def test_optimizer_passes(r: SubTestResult):
    """Test each optimizer pass at the AST level."""
    print("\n--- Optimizer Pass AST-Level Tests ---")
    from TEX_Wrangle.tex_compiler.optimizer import (
        optimize, _opt_stmt, _eliminate_dead_code,
        _eliminate_common_subexpressions, _hoist_loop_invariants,
    )
    from TEX_Wrangle.tex_compiler.ast_nodes import (
        NumberLiteral, BinOp, Identifier, VarDecl, ForLoop, FunctionCall,
        Assignment, Program,
    )

    def _parse(code):
        tokens = Lexer(code).tokenize()
        prog = Parser(tokens).parse()
        bt = {"OUT": TEXType.VEC3}
        tc = TypeChecker(binding_types=bt)
        tc.check(prog)
        return prog

    def _find_nodes(node, cls, results=None):
        """Recursively collect all AST nodes of a given class."""
        if results is None:
            results = []
        if isinstance(node, cls):
            results.append(node)
        for attr in vars(node).values():
            if isinstance(attr, list):
                for item in attr:
                    if hasattr(item, '__dict__'):
                        _find_nodes(item, cls, results)
            elif hasattr(attr, '__dict__'):
                _find_nodes(attr, cls, results)
        return results

    # ── 1. Constant folding ──────────────────────────────────────────
    try:
        prog = _parse("float x = 2.0 + 3.0; @OUT = vec3(x);")
        # Apply only the constant folding / algebraic simplification pass
        for i, stmt in enumerate(prog.statements):
            prog.statements[i] = _opt_stmt(stmt)

        # The VarDecl for x should have a NumberLiteral(5.0) initializer, not a BinOp
        x_decl = None
        for stmt in prog.statements:
            if isinstance(stmt, VarDecl) and stmt.name == "x":
                x_decl = stmt
                break
        assert x_decl is not None, "VarDecl for x not found"
        assert isinstance(x_decl.initializer, NumberLiteral), \
            f"Expected NumberLiteral after constant folding, got {type(x_decl.initializer).__name__}"
        assert abs(x_decl.initializer.value - 5.0) < 1e-9, \
            f"Expected 5.0, got {x_decl.initializer.value}"
        r.ok("pass: constant folding (2.0 + 3.0 -> 5.0)")
    except Exception as e:
        r.fail("pass: constant folding (2.0 + 3.0 -> 5.0)", f"{e}\n{traceback.format_exc()}")

    # ── 2. Algebraic simplification: x * 1.0 -> x ───────────────────
    try:
        prog = _parse("float x = u * 1.0; @OUT = vec3(x);")
        for i, stmt in enumerate(prog.statements):
            prog.statements[i] = _opt_stmt(stmt)

        x_decl = None
        for stmt in prog.statements:
            if isinstance(stmt, VarDecl) and stmt.name == "x":
                x_decl = stmt
                break
        assert x_decl is not None, "VarDecl for x not found"
        assert isinstance(x_decl.initializer, Identifier), \
            f"Expected Identifier (u) after x * 1.0 simplification, got {type(x_decl.initializer).__name__}"
        assert x_decl.initializer.name == "u", \
            f"Expected identifier 'u', got '{x_decl.initializer.name}'"
        r.ok("pass: algebraic simplification (u * 1.0 -> u)")
    except Exception as e:
        r.fail("pass: algebraic simplification (u * 1.0 -> u)", f"{e}\n{traceback.format_exc()}")

    # ── 3. Algebraic simplification: x + 0.0 -> x ───────────────────
    try:
        prog = _parse("float x = u + 0.0; @OUT = vec3(x);")
        for i, stmt in enumerate(prog.statements):
            prog.statements[i] = _opt_stmt(stmt)

        x_decl = None
        for stmt in prog.statements:
            if isinstance(stmt, VarDecl) and stmt.name == "x":
                x_decl = stmt
                break
        assert x_decl is not None, "VarDecl for x not found"
        assert isinstance(x_decl.initializer, Identifier), \
            f"Expected Identifier (u) after x + 0.0 simplification, got {type(x_decl.initializer).__name__}"
        assert x_decl.initializer.name == "u", \
            f"Expected identifier 'u', got '{x_decl.initializer.name}'"
        r.ok("pass: algebraic simplification (u + 0.0 -> u)")
    except Exception as e:
        r.fail("pass: algebraic simplification (u + 0.0 -> u)", f"{e}\n{traceback.format_exc()}")

    # ── 4. Algebraic simplification: x * 0.0 ─────────────────────────
    # NOTE: The optimizer intentionally does NOT fold x * 0.0 -> 0.0 when x
    # could be a spatial tensor (shape [B,H,W,C]).  Only literal*literal is folded.
    # We test that 2.0 * 0.0 (both literals) DOES fold to 0.0.
    try:
        prog = _parse("float x = 2.0 * 0.0; @OUT = vec3(x);")
        for i, stmt in enumerate(prog.statements):
            prog.statements[i] = _opt_stmt(stmt)

        x_decl = None
        for stmt in prog.statements:
            if isinstance(stmt, VarDecl) and stmt.name == "x":
                x_decl = stmt
                break
        assert x_decl is not None, "VarDecl for x not found"
        assert isinstance(x_decl.initializer, NumberLiteral), \
            f"Expected NumberLiteral(0.0) after 2.0 * 0.0, got {type(x_decl.initializer).__name__}"
        assert abs(x_decl.initializer.value - 0.0) < 1e-9, \
            f"Expected 0.0, got {x_decl.initializer.value}"
        r.ok("pass: algebraic simplification (2.0 * 0.0 -> 0.0)")
    except Exception as e:
        r.fail("pass: algebraic simplification (2.0 * 0.0 -> 0.0)", f"{e}\n{traceback.format_exc()}")

    # ── 5. Dead code elimination ─────────────────────────────────────
    try:
        prog = _parse("float unused = 1.0; float used = 2.0; @OUT = vec3(used);")
        # Apply constant folding first (as optimize() does), then DCE
        for i, stmt in enumerate(prog.statements):
            prog.statements[i] = _opt_stmt(stmt)
        prog.statements = _eliminate_dead_code(prog.statements)

        var_names = [s.name for s in prog.statements if isinstance(s, VarDecl)]
        assert "unused" not in var_names, \
            f"'unused' should be eliminated, but found vars: {var_names}"
        assert "used" in var_names, \
            f"'used' should be kept, but found vars: {var_names}"
        r.ok("pass: dead code elimination (unused var removed)")
    except Exception as e:
        r.fail("pass: dead code elimination (unused var removed)", f"{e}\n{traceback.format_exc()}")

    # ── 6. CSE (Common Subexpression Elimination) ────────────────────
    try:
        prog = _parse("float a = sin(u); float b = sin(u); @OUT = vec3(a + b);")
        # Apply constant folding first, then CSE
        for i, stmt in enumerate(prog.statements):
            prog.statements[i] = _opt_stmt(stmt)
        prog.statements = _eliminate_common_subexpressions(prog.statements)

        # After CSE, there should be a _cse variable and sin(u) should appear
        # only in the CSE temp decl, not in both a and b declarations.
        cse_vars = [s for s in prog.statements
                    if isinstance(s, VarDecl) and s.name.startswith("_cse")]
        # Check that at least one CSE temp was created
        if len(cse_vars) > 0:
            # Verify the CSE temp holds a sin() call
            cse_init = cse_vars[0].initializer
            assert isinstance(cse_init, FunctionCall) and cse_init.name == "sin", \
                f"Expected CSE temp to hold sin() call, got {type(cse_init).__name__}"
            r.ok("pass: CSE (sin(u) deduplicated)")
        else:
            # CSE might not trigger if sin(u) depth < _CSE_MIN_DEPTH (2).
            # sin(u) has depth 1 (FunctionCall with Identifier leaf), which is
            # below the threshold. This is expected behavior.
            # Verify with a deeper expression: sin(u + 1.0) which has depth 2.
            prog2 = _parse("float a = sin(u + 1.0); float b = sin(u + 1.0); @OUT = vec3(a + b);")
            for i, stmt in enumerate(prog2.statements):
                prog2.statements[i] = _opt_stmt(stmt)
            prog2.statements = _eliminate_common_subexpressions(prog2.statements)
            cse_vars2 = [s for s in prog2.statements
                         if isinstance(s, VarDecl) and s.name.startswith("_cse")]
            assert len(cse_vars2) > 0, \
                "CSE should deduplicate sin(u + 1.0) (depth >= 2)"
            cse_init2 = cse_vars2[0].initializer
            assert isinstance(cse_init2, FunctionCall) and cse_init2.name == "sin", \
                f"Expected CSE temp to hold sin() call, got {type(cse_init2).__name__}"
            r.ok("pass: CSE (sin(u+1.0) deduplicated, sin(u) below depth threshold)")
    except Exception as e:
        r.fail("pass: CSE (common subexpression eliminated)", f"{e}\n{traceback.format_exc()}")

    # ── 7. LICM (Loop-Invariant Code Motion) ─────────────────────────
    try:
        # cos(2.0) would be constant-folded, and cos(base) has depth 1 which
        # is below _LICM_MIN_DEPTH=2.  Use cos(base * 3.14) which has depth 2
        # (FunctionCall > BinOp > leaves) and won't be fully folded since
        # base is a variable.
        prog = _parse("""
float base = 2.0;
for(int i=0; i<4; i++) {
    float y = cos(base * 3.14);
    @OUT = vec3(y);
}
""")
        # Apply constant folding pass first (as optimize() does)
        for i, stmt in enumerate(prog.statements):
            prog.statements[i] = _opt_stmt(stmt)
        # Apply LICM pass
        prog.statements = _hoist_loop_invariants(prog.statements)

        # After LICM, cos(base) should be hoisted before the for loop.
        # Look for a _licm variable before the ForLoop.
        licm_vars = [s for s in prog.statements
                     if isinstance(s, VarDecl) and s.name.startswith("_licm")]
        for_loops = [s for s in prog.statements if isinstance(s, ForLoop)]

        assert len(licm_vars) > 0, \
            "LICM should hoist cos(base * 3.14) into a _licm temp variable"
        assert len(for_loops) > 0, "ForLoop should still exist"

        # Verify the hoisted var appears before the for loop in the statement list
        licm_idx = prog.statements.index(licm_vars[0])
        for_idx = prog.statements.index(for_loops[0])
        assert licm_idx < for_idx, \
            f"Hoisted _licm var (idx {licm_idx}) should appear before ForLoop (idx {for_idx})"

        # Verify the hoisted expression is cos(...) — it is the deepest
        # invariant subtree that meets the depth threshold.
        licm_init = licm_vars[0].initializer
        assert isinstance(licm_init, FunctionCall) and licm_init.name == "cos", \
            f"Expected hoisted expression to be cos(), got {type(licm_init).__name__}"
        r.ok("pass: LICM (cos(base*3.14) hoisted before loop)")
    except Exception as e:
        r.fail("pass: LICM (cos(base*3.14) hoisted before loop)", f"{e}\n{traceback.format_exc()}")
