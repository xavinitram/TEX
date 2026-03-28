"""Stdlib function tests — math, color, SDF, reductions, edge cases."""
from helpers import *


def test_stdlib_coverage(r: SubTestResult):
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

def test_stdlib_extended(r: SubTestResult):
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

def test_numerical_edge_cases(r: SubTestResult):
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
    
        code = "float x = log2(0.0);\n@OUT = vec3(x, x, x);"
        result = compile_and_run(code, {"A": test_img})
        val = result[0, 0, 0, 0].item()
        assert not torch.isinf(result[0, 0, 0, 0]), f"log2(0) should not be -Inf, got {val}"
        r.ok("edge: log2(0) clamped to finite")
    except Exception as e:
        r.fail("edge: log2(0) clamped to finite", f"{e}\n{traceback.format_exc()}")

    # log10(0) clamped to finite
    try:
    
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


def test_new_stdlib_functions(r: SubTestResult):
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


def test_missing_stdlib_functions(r: SubTestResult):
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

def test_numeric_edge_case_matrix(r: SubTestResult):
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


# ── NaN/Inf Propagation Tests ─────────────────────────────────────────

def test_nan_inf_propagation(r: SubTestResult):
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


def test_image_reductions(r: SubTestResult):
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


def test_sdf_functions(r: SubTestResult):
    """Tests for SDF primitives."""
    print("\n--- SDF Function Tests ---")

    sdf_img = torch.zeros(1, 32, 32, 3)

    # sdf_circle — negative inside, positive outside
    try:
        code = "float d = sdf_circle(u - 0.5, v - 0.5, 0.3);\n@OUT = vec3(d, d, d);"
        result = compile_and_run(code, {"A": sdf_img})
        center_val = result[0, 16, 16, 0].item()
        corner_val = result[0, 0, 0, 0].item()
        assert center_val < 0, f"Circle center should be negative (inside), got {center_val}"
        assert corner_val > 0, f"Circle corner should be positive (outside), got {corner_val}"
        r.ok("sdf: circle inside/outside")
    except Exception as e:
        r.fail("sdf: circle inside/outside", f"{e}\n{traceback.format_exc()}")

    # sdf_box — negative inside, positive outside
    try:
        code = "float d = sdf_box(u - 0.5, v - 0.5, 0.3, 0.2);\n@OUT = vec3(d, d, d);"
        result = compile_and_run(code, {"A": sdf_img})
        center_val = result[0, 16, 16, 0].item()
        corner_val = result[0, 0, 0, 0].item()
        assert center_val < 0, f"Box center should be negative, got {center_val}"
        assert corner_val > 0, f"Box corner should be positive, got {corner_val}"
        r.ok("sdf: box inside/outside")
    except Exception as e:
        r.fail("sdf: box inside/outside", f"{e}\n{traceback.format_exc()}")

    # sdf_line — distance to line segment, always >= 0
    try:
        code = "float d = sdf_line(u, v, 0.2, 0.2, 0.8, 0.8);\n@OUT = vec3(d, d, d);"
        result = compile_and_run(code, {"A": sdf_img})
        vals = result[..., 0]
        assert vals.min().item() >= -0.01, f"Line distance should be >= 0, got {vals.min().item()}"
        assert vals.std().item() > 0.001, "Line has no spatial variation"
        r.ok("sdf: line distance")
    except Exception as e:
        r.fail("sdf: line distance", f"{e}\n{traceback.format_exc()}")

    # sdf_polygon — hexagon (6 sides)
    try:
        code = "float d = sdf_polygon(u - 0.5, v - 0.5, 0.3, 6);\n@OUT = vec3(d, d, d);"
        result = compile_and_run(code, {"A": sdf_img})
        center_val = result[0, 16, 16, 0].item()
        corner_val = result[0, 0, 0, 0].item()
        assert center_val < 0, f"Polygon center should be negative, got {center_val}"
        assert corner_val > 0, f"Polygon corner should be positive, got {corner_val}"
        r.ok("sdf: polygon hexagon")
    except Exception as e:
        r.fail("sdf: polygon hexagon", f"{e}\n{traceback.format_exc()}")

    # sdf_polygon — triangle (3 sides, minimum)
    try:
        code = "float d = sdf_polygon(u - 0.5, v - 0.5, 0.3, 3);\n@OUT = vec3(d, d, d);"
        result = compile_and_run(code, {"A": sdf_img})
        assert result.dim() == 4
        r.ok("sdf: polygon triangle")
    except Exception as e:
        r.fail("sdf: polygon triangle", f"{e}\n{traceback.format_exc()}")

    # smin — smooth blending of two SDFs
    try:
        code = """
float d1 = sdf_circle(u - 0.3, v - 0.5, 0.15);
float d2 = sdf_circle(u - 0.7, v - 0.5, 0.15);
float hard = min(d1, d2);
float soft = smin(d1, d2, 0.1);
@OUT = vec3(hard, soft, soft - hard);
"""
        result = compile_and_run(code, {"A": sdf_img})
        # smin should be <= min (smoother)
        diff = result[..., 2]
        assert diff.min().item() <= 0.01, "smin should be <= min"
        r.ok("sdf: smin blending")
    except Exception as e:
        r.fail("sdf: smin blending", f"{e}\n{traceback.format_exc()}")

    # smax
    try:
        code = """
float a = u - 0.5;
float b = v - 0.5;
float hard = max(a, b);
float soft = smax(a, b, 0.1);
@OUT = vec3(hard, soft, soft - hard);
"""
        result = compile_and_run(code, {"A": sdf_img})
        diff = result[..., 2]
        assert diff.max().item() >= -0.01, "smax should be >= max"
        r.ok("sdf: smax blending")
    except Exception as e:
        r.fail("sdf: smax blending", f"{e}\n{traceback.format_exc()}")

    # sample_grad — gradient of a known image
    try:
        # Create a horizontal gradient image: pixel value increases with x
        grad_img = torch.zeros(1, 32, 32, 3)
        for x in range(32):
            grad_img[0, :, x, :] = x / 31.0
        code = "vec2 g = sample_grad(@A, u, v);\n@OUT = g;"
        result = compile_and_run(code, {"A": grad_img})
        assert result.shape[-1] == 2, f"sample_grad should return vec2, got shape {result.shape}"
        # Horizontal gradient should be positive, vertical near zero
        gx = result[..., 0].mean().item()
        gy = result[..., 1].mean().item()
        assert gx > 0.001, f"Horizontal gradient should be positive, got {gx}"
        assert abs(gy) < 0.05, f"Vertical gradient should be near zero, got {gy}"
        r.ok("sdf: sample_grad horizontal")
    except Exception as e:
        r.fail("sdf: sample_grad horizontal", f"{e}\n{traceback.format_exc()}")

    # sample_grad — vertical gradient
    try:
        grad_img = torch.zeros(1, 32, 32, 3)
        for y in range(32):
            grad_img[0, y, :, :] = y / 31.0
        code = "vec2 g = sample_grad(@A, u, v);\n@OUT = g;"
        result = compile_and_run(code, {"A": grad_img})
        gx = result[..., 0].mean().item()
        gy = result[..., 1].mean().item()
        assert abs(gx) < 0.05, f"Horizontal gradient should be near zero, got {gx}"
        assert gy > 0.001, f"Vertical gradient should be positive, got {gy}"
        r.ok("sdf: sample_grad vertical")
    except Exception as e:
        r.fail("sdf: sample_grad vertical", f"{e}\n{traceback.format_exc()}")


def test_new_builtins_and_fixes(r: SubTestResult):
    """Test TAU constant, px/py pixel step, sincos(), and distance(vec2) fix."""

    def run(code, bindings=None, check_tc=True):
        tokens = Lexer(code).tokenize()
        program = Parser(tokens, source="test").parse()
        raw_bindings = {}
        bt = {}
        if bindings:
            for k, v in bindings.items():
                # Strip leading @ for internal representation
                name = k.lstrip("@")
                raw_bindings[name] = v
                if isinstance(v, torch.Tensor) and v.dim() == 4 and v.shape[-1] in (2, 3, 4):
                    bt[name] = TEXType.VEC4
                elif isinstance(v, torch.Tensor):
                    bt[name] = TEXType.FLOAT
        tc = TypeChecker(binding_types=bt, source="test")
        type_map = tc.check(program)
        if check_tc:
            assert not tc.errors, f"Type errors: {[e.message for e in tc.errors]}"
        output_names = sorted(tc.assigned_bindings.keys())
        interp = Interpreter()
        return interp.execute(program, raw_bindings, type_map,
                              output_names=output_names, source="test")

    dummy = {"A": torch.ones(1, 4, 4, 3)}

    # 1. TAU constant
    try:
        result = run("@OUT = vec3(TAU, TAU, TAU);", dummy)
        val = result["OUT"][0, 0, 0, 0].item()
        assert abs(val - math.tau) < 1e-4, f"TAU should be {math.tau}, got {val}"
        r.ok("TAU constant")
    except Exception as e:
        r.fail("TAU constant", e)

    # 2. px / py pixel step variables
    try:
        img = torch.ones(1, 10, 20, 3)
        result = run("@OUT = vec3(px, py, 0.0);", {"@A": img})
        px_val = result["OUT"][0, 0, 0, 0].item()
        py_val = result["OUT"][0, 0, 0, 1].item()
        assert abs(px_val - 1.0/20) < 1e-5, f"px should be 1/20={1/20:.5f}, got {px_val}"
        assert abs(py_val - 1.0/10) < 1e-5, f"py should be 1/10={1/10:.5f}, got {py_val}"
        r.ok("px/py pixel step variables")
    except Exception as e:
        r.fail("px/py pixel step variables", e)

    # 3. sincos() returns vec2(sin, cos)
    try:
        result = run("vec2 sc = sincos(PI * 0.5);\n@OUT = vec3(sc.x, sc.y, 0.0);", dummy)
        sin_val = result["OUT"][0, 0, 0, 0].item()
        cos_val = result["OUT"][0, 0, 0, 1].item()
        assert abs(sin_val - 1.0) < 1e-4, f"sin(PI/2) should be 1.0, got {sin_val}"
        assert abs(cos_val - 0.0) < 1e-4, f"cos(PI/2) should be 0.0, got {cos_val}"
        r.ok("sincos() function")
    except Exception as e:
        r.fail("sincos() function", e)

    # 4. distance(vec2, vec2) returns float (was returning vec2)
    try:
        result = run("float d = distance(vec2(0.0, 0.0), vec2(3.0, 4.0));\n@OUT = vec3(d, d, d);", dummy)
        val = result["OUT"][0, 0, 0, 0].item()
        assert abs(val - 5.0) < 1e-4, f"distance(vec2) should be 5.0, got {val}"
        r.ok("distance(vec2, vec2) returns float")
    except Exception as e:
        r.fail("distance(vec2, vec2) returns float", e)

    # 5. length(vec2) returns float (same fix)
    try:
        result = run("float l = length(vec2(3.0, 4.0));\n@OUT = vec3(l, l, l);", dummy)
        val = result["OUT"][0, 0, 0, 0].item()
        assert abs(val - 5.0) < 1e-4, f"length(vec2) should be 5.0, got {val}"
        r.ok("length(vec2) returns float")
    except Exception as e:
        r.fail("length(vec2) returns float", e)

    # 6. normalize(vec2) returns unit vec2
    try:
        result = run("vec2 n = normalize(vec2(3.0, 4.0));\nfloat l = length(n);\n@OUT = vec3(l, n.x, n.y);", dummy)
        l_val = result["OUT"][0, 0, 0, 0].item()
        assert abs(l_val - 1.0) < 1e-4, f"normalize(vec2) length should be 1.0, got {l_val}"
        r.ok("normalize(vec2) returns unit vec2")
    except Exception as e:
        r.fail("normalize(vec2) returns unit vec2", e)


# ── Stdlib Edge Cases ─────────────────────────────────────────────────

def test_stdlib_edge_cases(r: SubTestResult):
    """Test edge cases for stdlib functions — boundary inputs, negative radii, OOB UVs, etc."""
    print("\n--- Stdlib Edge Cases ---")

    bindings = {"A": torch.rand(1, 4, 4, 3)}

    def run_ok(name, code):
        """Run code and verify it doesn't crash."""
        try:
            compile_and_run(code, bindings)
            r.ok(f"edge: {name}")
        except Exception as e:
            r.fail(f"edge: {name}", f"{e}\n{traceback.format_exc()}")

    def run_val(name, code, expected, atol=1e-4):
        """Run code, extract [0,0,0,0], compare to expected."""
        try:
            result = compile_and_run(code, bindings)
            val = result[0, 0, 0, 0].item()
            assert abs(val - expected) < atol, f"Got {val}, expected {expected}"
            r.ok(f"edge: {name}")
        except Exception as e:
            r.fail(f"edge: {name}", f"{e}\n{traceback.format_exc()}")

    # -- Math edge cases --
    run_val("sqrt(-1) clamps to 0",
            "float x = sqrt(-1.0);\n@OUT = vec3(x, x, x);", 0.0)

    run_ok("log(0.0) no crash",
           "float x = log(0.0);\n@OUT = vec3(x, x, x);")

    run_val("pow(0.0, 0.0) is 1.0",
            "float x = pow(0.0, 0.0);\n@OUT = vec3(x, x, x);", 1.0)

    run_val("atan2(0.0, 0.0) is 0.0",
            "float x = atan2(0.0, 0.0);\n@OUT = vec3(x, x, x);", 0.0)

    run_val("sdiv(1.0, 0.0) returns 0",
            "float x = sdiv(1.0, 0.0);\n@OUT = vec3(x, x, x);", 0.0)

    run_val("sdiv(0.0, 0.0) returns 0",
            "float x = sdiv(0.0, 0.0);\n@OUT = vec3(x, x, x);", 0.0)

    run_ok("spow(-2.0, 0.5) no crash",
           "float x = spow(-2.0, 0.5);\n@OUT = vec3(x, x, x);")

    # -- Sampling edge cases --
    run_ok("sample(@A, -0.5, -0.5) OOB below",
           "vec3 c = sample(@A, -0.5, -0.5);\n@OUT = vec3(c.x, c.y, c.z);")

    run_ok("sample(@A, 2.0, 2.0) OOB above",
           "vec3 c = sample(@A, 2.0, 2.0);\n@OUT = vec3(c.x, c.y, c.z);")

    run_ok("gauss_blur(@A, 0.0) zero sigma",
           "@OUT = gauss_blur(@A, 0.0);")

    run_ok("gauss_blur(@A, 10.0) large sigma",
           "@OUT = gauss_blur(@A, 10.0);")

    # -- SDF edge cases --
    run_ok("sdf_circle negative radius",
           "float d = sdf_circle(0.0, 0.0, -1.0);\n@OUT = vec3(d, d, d);")

    run_ok("sdf_box zero dimensions",
           "float d = sdf_box(0.0, 0.0, 0.0, 0.0);\n@OUT = vec3(d, d, d);")

    # -- Array edge cases: single-element arrays --
    run_val("arr_sum single element",
            "float a[] = {5.0};\nfloat x = arr_sum(a);\n@OUT = vec3(x, x, x);", 5.0)

    run_ok("sort single element",
           "float a[] = {5.0};\nsort(a);\n@OUT = vec3(a[0], a[0], a[0]);")

    run_ok("sort already-sorted array",
           "float a[] = {1.0, 2.0, 3.0, 4.0};\nsort(a);\n"
           "@OUT = vec3(a[0], a[1], a[2]);")

    run_ok("reverse single element",
           "float a[] = {5.0};\nreverse(a);\n@OUT = vec3(a[0], a[0], a[0]);")

    run_val("median single element",
            "float a[] = {5.0};\nfloat x = median(a);\n@OUT = vec3(x, x, x);", 5.0)


# ── NaN / Inf Robustness ─────────────────────────────────────────────

def test_stdlib_nan_inf(r: SubTestResult):
    """Test that major math functions handle NaN and Inf inputs without crashing."""
    print("\n--- Stdlib NaN/Inf Robustness ---")

    bindings = {"A": torch.rand(1, 4, 4, 3)}

    def run_ok(name, code):
        """Run code and verify it doesn't crash."""
        try:
            compile_and_run(code, bindings)
            r.ok(f"nan_inf: {name}")
        except Exception as e:
            r.fail(f"nan_inf: {name}", f"{e}\n{traceback.format_exc()}")

    # -- NaN input (generated via pow(-1.0, 0.5)) --
    nan_funcs = ["sin", "cos", "tan", "sqrt", "abs", "floor", "ceil", "round",
                 "fract", "exp", "log", "sign", "clamp", "smoothstep"]
    for fn in nan_funcs:
        if fn == "clamp":
            code = ("float nan_val = pow(-1.0, 0.5);\n"
                    "float result = clamp(nan_val, 0.0, 1.0);\n"
                    "@OUT = vec3(isnan(result) + 0.0);")
        elif fn == "smoothstep":
            code = ("float nan_val = pow(-1.0, 0.5);\n"
                    "float result = smoothstep(0.0, 1.0, nan_val);\n"
                    "@OUT = vec3(isnan(result) + 0.0);")
        else:
            code = (f"float nan_val = pow(-1.0, 0.5);\n"
                    f"float result = {fn}(nan_val);\n"
                    f"@OUT = vec3(isnan(result) + 0.0);")
        run_ok(f"{fn}(NaN)", code)

    # -- Inf input (generated via pow(0.0, -1.0)) --
    inf_funcs = ["sin", "cos", "tan", "sqrt", "abs", "floor", "ceil", "round",
                 "fract", "exp", "log", "sign", "clamp"]
    for fn in inf_funcs:
        if fn == "clamp":
            code = ("float inf_val = pow(0.0, -1.0);\n"
                    "float result = clamp(inf_val, 0.0, 1.0);\n"
                    "@OUT = vec3(isinf(result) + 0.0);")
        else:
            code = (f"float inf_val = pow(0.0, -1.0);\n"
                    f"float result = {fn}(inf_val);\n"
                    f"@OUT = vec3(isinf(result) + 0.0);")
        run_ok(f"{fn}(Inf)", code)

    # -- sdiv special cases --
    run_ok("sdiv(0.0, 0.0)",
           "float result = sdiv(0.0, 0.0);\n@OUT = vec3(result, result, result);")

    run_ok("sdiv(Inf, Inf)",
           "float inf1 = pow(0.0, -1.0);\nfloat inf2 = pow(0.0, -1.0);\n"
           "float result = sdiv(inf1, inf2);\n@OUT = vec3(result, result, result);")

    # -- Arithmetic edge cases --
    run_ok("inf - inf",
           "float inf1 = pow(0.0, -1.0);\nfloat inf2 = pow(0.0, -1.0);\n"
           "float x = inf1 - inf2;\n@OUT = vec3(x, x, x);")

    run_ok("inf * 0.0",
           "float inf_val = pow(0.0, -1.0);\nfloat x = inf_val * 0.0;\n"
           "@OUT = vec3(x, x, x);")
