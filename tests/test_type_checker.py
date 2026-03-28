from helpers import *


def test_type_checker(r: SubTestResult):
    print("\n--- Type Checker Tests ---")

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
