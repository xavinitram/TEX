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

    # H1 regression: `mat * array` must error cleanly, not crash. It used to
    # return None, then crash with AttributeError on .is_string in _is_assignable.
    try:
        check_code("mat3 m = mat3(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0);"
                   " float arr[2] = {1.0, 2.0}; mat3 z = m * arr; @OUT = vec4(0.0);")
        r.fail("mat * array errors cleanly", "Should have raised a type error")
    except TypeCheckError:
        r.ok("mat * array errors cleanly")
    except Exception as e:
        r.fail("mat * array errors cleanly", f"crashed with {type(e).__name__}: {e}")

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


def test_stdlib_promote_typing(r: SubTestResult):
    """Elementwise builtins (step/smoothstep/pow/mod/atan2/hypot/spow/sdiv/
    clamp/fit) broadcast a vector in ANY argument position at runtime, so the
    checker promotes across all args instead of passing through the first
    arg's type. Pins the promoted inference and its user-visible effects
    (IMAGE-not-MASK output, new E3200 on scalar-typed vars, no spurious
    E5003 on vector-only builtins)."""
    print("\n--- Stdlib Promoted Typing Tests ---")

    vec_cases = [
        ("smoothstep", "smoothstep(0.0, 1.0, @A)"),
        ("step", "step(0.5, @A)"),
        ("atan2", "atan2(1.0, @A)"),
        ("pow", "pow(2.0, @A)"),
        ("mod", "mod(1.0, @A)"),
        ("hypot", "hypot(1.0, @A)"),
        ("spow", "spow(2.0, @A)"),
        ("sdiv", "sdiv(1.0, @A)"),
        ("clamp", "clamp(0.5, @A, @A)"),
        ("fit", "fit(0.5, 0.0, 1.0, vec3(0.0, 0.0, 0.0), @A)"),
    ]
    for name, expr in vec_cases:
        try:
            _, checker = check_code(f"@OUT = {expr};", {"A": TEXType.VEC3})
            got = checker.inferred_out_type
            assert got == TEXType.VEC3, f"inferred {got}, expected VEC3"
            r.ok(f"promote typing: {name}(..., vec3) -> vec3")
        except Exception as e:
            r.fail(f"promote typing: {name}(..., vec3) -> vec3",
                   f"{e}\n{traceback.format_exc()}")

    # All-scalar args keep FLOAT/MASK typing (mask pipelines unchanged)
    try:
        _, checker = check_code("@OUT = smoothstep(0.0, 1.0, @A);",
                                {"A": TEXType.FLOAT})
        assert checker.inferred_out_type == TEXType.FLOAT
        assert _map_inferred_type(checker.inferred_out_type, False) == "MASK"
        r.ok("promote typing: all-scalar args stay FLOAT/MASK")
    except Exception as e:
        r.fail("promote typing: all-scalar args stay FLOAT/MASK",
               f"{e}\n{traceback.format_exc()}")

    # float var holding a vec-arg call result is now a compile error
    # (previously compiled while the variable silently held a vec3 tensor)
    try:
        check_code("float f = step(0.5, @A);\n@OUT = vec4(f);", {"A": TEXType.VEC3})
        r.fail("promote typing: float var from vec call errors (E3200)",
               "Should have raised TypeCheckError")
    except (TypeCheckError, TEXMultiError):
        r.ok("promote typing: float var from vec call errors (E3200)")
    except Exception as e:
        r.fail("promote typing: float var from vec call errors (E3200)", str(e))

    # int/float mixes promote to FLOAT (matches torch dtype promotion)
    try:
        check_code("int i = 3;\nint j = mod(i, 2.5);\n@OUT = vec4(float(j));")
        r.fail("promote typing: int var from mod(int, float) errors (E3200)",
               "Should have raised TypeCheckError")
    except (TypeCheckError, TEXMultiError):
        r.ok("promote typing: int var from mod(int, float) errors (E3200)")
    except Exception as e:
        r.fail("promote typing: int var from mod(int, float) errors (E3200)", str(e))

    # Previously-spurious E5003 rejection now passes: the call result is a
    # vector, so vector-only builtins accept it
    try:
        check_code("float l = length(smoothstep(0.0, 1.0, @A));\n@OUT = vec4(l);",
                   {"A": TEXType.VEC3})
        r.ok("promote typing: length(smoothstep(vec)) accepted")
    except Exception as e:
        r.fail("promote typing: length(smoothstep(vec)) accepted",
               f"{e}\n{traceback.format_exc()}")

    # vec3(smoothstep(vec3), 0.0, 0.0) is now a clean compile error (3+1+1=5
    # components) instead of a runtime tensor-shape crash
    try:
        check_code("@OUT = vec3(smoothstep(0.0, 1.0, @A), 0.0, 0.0);",
                   {"A": TEXType.VEC3})
        r.fail("promote typing: vec3(smoothstep(vec), 0, 0) errors (E3601)",
               "Should have raised TypeCheckError")
    except (TypeCheckError, TEXMultiError):
        r.ok("promote typing: vec3(smoothstep(vec), 0, 0) errors (E3601)")
    except Exception as e:
        r.fail("promote typing: vec3(smoothstep(vec), 0, 0) errors (E3601)", str(e))

    # m@OUT hint prefix still forces MASK output
    try:
        _, checker = check_code("m@OUT = smoothstep(0.0, 1.0, @A);",
                                {"A": TEXType.VEC3})
        assert checker.assigned_bindings["OUT"] == TEXType.FLOAT
        r.ok("promote typing: m@OUT hint forces MASK")
    except Exception as e:
        r.fail("promote typing: m@OUT hint forces MASK", f"{e}\n{traceback.format_exc()}")

    # End-to-end: inferred output is IMAGE and the payload stays per-channel
    try:
        torch.manual_seed(5)
        img = torch.rand(1, 4, 4, 3)
        out, inferred = compile_and_infer("@OUT = smoothstep(0.0, 1.0, @A);",
                                          {"A": img})
        assert inferred == TEXType.VEC3, f"inferred {inferred}"
        assert _map_inferred_type(inferred, False) == "IMAGE"
        assert out.shape == (1, 4, 4, 3), f"shape {out.shape}"
        r.ok("promote typing: smoothstep(vec) infers IMAGE with 3 channels")
    except Exception as e:
        r.fail("promote typing: smoothstep(vec) infers IMAGE with 3 channels",
               f"{e}\n{traceback.format_exc()}")

    # End-to-end: vec4(smoothstep(vec3), 1.0) now counts 3+1 components
    try:
        torch.manual_seed(6)
        img = torch.rand(1, 4, 4, 3)
        out = compile_and_run("@OUT = vec4(smoothstep(0.0, 1.0, @A), 1.0);",
                              {"A": img})
        assert out.shape == (1, 4, 4, 4), f"shape {out.shape}"
        r.ok("promote typing: vec4(smoothstep(vec), 1.0) compiles and runs")
    except Exception as e:
        r.fail("promote typing: vec4(smoothstep(vec), 1.0) compiles and runs",
               f"{e}\n{traceback.format_exc()}")
