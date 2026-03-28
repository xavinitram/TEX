"""Binding, parameter, and user function tests."""
from helpers import *


def test_named_bindings(r: SubTestResult):
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


def test_binding_access(r: SubTestResult):
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


def test_binding_access_advanced(r: SubTestResult):
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


def test_scatter_writes(r: SubTestResult):
    """Tests for scatter write (@OUT[px, py] = value) functionality."""
    print("\n--- Scatter Write Tests ---")

    B, H, W = 1, 8, 8
    img = torch.rand(B, H, W, 3)

    # Identity scatter: @OUT[ix, iy] = @A; (should reproduce input)
    try:
        result = compile_and_run("@OUT[ix, iy] = @A;", {"A": img})
        assert result.shape == img.shape, f"Shape mismatch: {result.shape} vs {img.shape}"
        max_diff = (result - img).abs().max().item()
        assert max_diff < 1e-5, f"Identity scatter failed, max_diff={max_diff}"
        r.ok("scatter: identity")
    except Exception as e:
        r.fail("scatter: identity", f"{e}\n{traceback.format_exc()}")

    # Offset scatter: shift right by 2 pixels
    try:
        result = compile_and_run("@OUT[ix + 2, iy] = @A;", {"A": img})
        assert result.shape == img.shape
        # Pixel at (2,0) in output should match (0,0) in input
        assert (result[0, 0, 2] - img[0, 0, 0]).abs().max().item() < 1e-5
        r.ok("scatter: offset right")
    except Exception as e:
        r.fail("scatter: offset right", f"{e}\n{traceback.format_exc()}")

    # Vertical flip via scatter
    try:
        result = compile_and_run("@OUT[ix, ih - 1.0 - float(iy)] = @A;", {"A": img})
        assert result.shape == img.shape
        # First row of output should match last row of input
        assert (result[0, 0, :] - img[0, H-1, :]).abs().max().item() < 1e-5
        r.ok("scatter: vertical flip")
    except Exception as e:
        r.fail("scatter: vertical flip", f"{e}\n{traceback.format_exc()}")

    # Out-of-bounds clamp: scatter to negative coords should clamp to 0
    try:
        result = compile_and_run("@OUT[ix - 100, iy] = @A;", {"A": img})
        # All pixels land at x=0 (clamped), last-write-wins
        r.ok("scatter: out-of-bounds clamp")
    except Exception as e:
        r.fail("scatter: out-of-bounds clamp", f"{e}\n{traceback.format_exc()}")

    # Mixed: regular assignment then scatter on top
    try:
        result = compile_and_run("@OUT = vec3(0.0); @OUT[0, 0] = vec3(1.0, 0.0, 0.0);", {"A": img})
        # Pixel (0,0) should be red, rest should be black
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-5
        assert abs(result[0, 0, 0, 1].item() - 0.0) < 1e-5
        r.ok("scatter: regular then scatter")
    except Exception as e:
        r.fail("scatter: regular then scatter", f"{e}\n{traceback.format_exc()}")

    # Compound += scatter (per-pixel: each pixel adds to itself)
    try:
        result = compile_and_run(
            "@OUT = vec3(0.5); @OUT[ix, iy] += vec3(0.1, 0.2, 0.3);",
            {"A": img}
        )
        # Each pixel adds once to itself
        assert abs(result[0, 0, 0, 0].item() - 0.6) < 1e-5
        assert abs(result[0, 0, 0, 1].item() - 0.7) < 1e-5
        assert abs(result[0, 0, 0, 2].item() - 0.8) < 1e-5
        r.ok("scatter: compound +=")
    except Exception as e:
        r.fail("scatter: compound +=", f"{e}\n{traceback.format_exc()}")

    # Compound -= scatter (per-pixel: every pixel subtracts at (0,0))
    try:
        result = compile_and_run(
            "@OUT = vec3(1.0); @OUT[ix, iy] -= vec3(0.1, 0.1, 0.1);",
            {"A": img}
        )
        # Each pixel subtracts from itself, so all pixels should be 0.9
        assert abs(result[0, 0, 0, 0].item() - 0.9) < 1e-5
        r.ok("scatter: compound -=")
    except Exception as e:
        r.fail("scatter: compound -=", f"{e}\n{traceback.format_exc()}")

    # Scalar scatter
    try:
        result = compile_and_run("@OUT = 0.0; @OUT[0, 0] = 1.0;", {"A": img})
        assert abs(result[0, 0, 0].item() - 1.0) < 1e-5
        r.ok("scatter: scalar value")
    except Exception as e:
        r.fail("scatter: scalar value", f"{e}\n{traceback.format_exc()}")


def test_wireable_params(r: SubTestResult):
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


def test_new_param_types(r: SubTestResult):
    """Tests for new parameter types: boolean (b$), color (c$), vec2 (v2$), vec3 (v3$)."""
    print("\n--- New Parameter Type Tests ---")

    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 3)

    # Boolean param — widget value (Python bool → float tensor)
    try:
        code = "b$enabled = 1;\n@OUT = @A * float($enabled);"
        result = compile_and_run(code, {"A": test_img, "enabled": True})
        assert torch.allclose(result[..., :3], test_img, atol=1e-5)
        r.ok("param: bool enabled=True → 1.0")
    except Exception as e:
        r.fail("param: bool enabled=True → 1.0", f"{e}\n{traceback.format_exc()}")

    try:
        code = "b$enabled = 1;\n@OUT = @A * float($enabled);"
        result = compile_and_run(code, {"A": test_img, "enabled": False})
        assert torch.allclose(result[..., :3], torch.zeros_like(test_img), atol=1e-5)
        r.ok("param: bool enabled=False → 0.0")
    except Exception as e:
        r.fail("param: bool enabled=False → 0.0", f"{e}\n{traceback.format_exc()}")

    # Boolean param — code default
    try:
        code = "b$flag = 0;\n@OUT = @A * float($flag);"
        result = compile_and_run(code, {"A": test_img, "flag": 0})
        assert torch.allclose(result[..., :3], torch.zeros_like(test_img), atol=1e-5)
        r.ok("param: bool flag=0 default")
    except Exception as e:
        r.fail("param: bool flag=0 default", f"{e}\n{traceback.format_exc()}")

    # Vec2 param — list value (simulating converted widget)
    try:
        code = "v2$offset;\n@OUT = @A + vec3($offset.x, $offset.y, 0.0);"
        result = compile_and_run(code, {"A": test_img, "offset": [0.1, 0.2]})
        expected = test_img.clone()
        expected[..., 0] += 0.1
        expected[..., 1] += 0.2
        assert torch.allclose(result[..., :3], expected, atol=1e-5)
        r.ok("param: vec2 list value")
    except Exception as e:
        r.fail("param: vec2 list value", f"{e}\n{traceback.format_exc()}")

    # Vec3 param — list value (simulating converted widget)
    try:
        code = "v3$tint;\n@OUT = @A * $tint;"
        tint = [0.5, 0.8, 0.3]
        result = compile_and_run(code, {"A": test_img, "tint": tint})
        expected = test_img * torch.tensor(tint)
        assert torch.allclose(result[..., :3], expected, atol=1e-5)
        r.ok("param: vec3 list value")
    except Exception as e:
        r.fail("param: vec3 list value", f"{e}\n{traceback.format_exc()}")

    # Vec3 param — with vec constructor default
    try:
        code = "v3$color = vec3(1.0, 0.5, 0.0);\n@OUT = @A * $color;"
        result = compile_and_run(code, {"A": test_img, "color": [1.0, 0.5, 0.0]})
        expected = test_img * torch.tensor([1.0, 0.5, 0.0])
        assert torch.allclose(result[..., :3], expected, atol=1e-5)
        r.ok("param: vec3 with vec constructor default")
    except Exception as e:
        r.fail("param: vec3 with vec constructor default", f"{e}\n{traceback.format_exc()}")

    # Color param — hex string conversion (test _hex_to_rgb via tex_node)
    try:
        from TEX_Wrangle.tex_node import _hex_to_rgb
        rgb = _hex_to_rgb("#FF8000")
        assert abs(rgb[0] - 1.0) < 1e-3, f"R={rgb[0]}"
        assert abs(rgb[1] - 0.502) < 1e-2, f"G={rgb[1]}"
        assert abs(rgb[2] - 0.0) < 1e-3, f"B={rgb[2]}"
        r.ok("param: _hex_to_rgb conversion")
    except Exception as e:
        r.fail("param: _hex_to_rgb conversion", f"{e}\n{traceback.format_exc()}")

    # Color param — hex short form
    try:
        from TEX_Wrangle.tex_node import _hex_to_rgb
        rgb = _hex_to_rgb("#FFF")
        assert abs(rgb[0] - 1.0) < 1e-3 and abs(rgb[1] - 1.0) < 1e-3 and abs(rgb[2] - 1.0) < 1e-3
        r.ok("param: _hex_to_rgb short form #FFF")
    except Exception as e:
        r.fail("param: _hex_to_rgb short form #FFF", f"{e}\n{traceback.format_exc()}")

    # Color param — runtime flow (list value from converted hex)
    try:
        code = 'c$tint;\n@OUT = @A * $tint;'
        result = compile_and_run(code, {"A": test_img, "tint": [1.0, 0.0, 0.0]})
        expected = test_img.clone()
        expected[..., 1] = 0.0
        expected[..., 2] = 0.0
        assert torch.allclose(result[..., :3], expected, atol=1e-5)
        r.ok("param: color as vec3 list (red)")
    except Exception as e:
        r.fail("param: color as vec3 list (red)", f"{e}\n{traceback.format_exc()}")

    # Vec param wired with spatial tensor (overrides widget)
    try:
        code = "v3$tint;\n@OUT = @A * $tint;"
        tint_img = torch.ones(B, H, W, 3) * 0.5
        result = compile_and_run(code, {"A": test_img, "tint": tint_img})
        expected = test_img * 0.5
        assert torch.allclose(result[..., :3], expected, atol=1e-5)
        r.ok("param: vec3 wired with spatial tensor")
    except Exception as e:
        r.fail("param: vec3 wired with spatial tensor", f"{e}\n{traceback.format_exc()}")

    # Param default extraction for vec constructor
    try:
        from TEX_Wrangle.tex_cache import TEXCache
        cache = TEXCache(cache_dir=Path(tempfile.mkdtemp()))
        code = "v3$color = vec3(0.2, 0.4, 0.6);\n@OUT = @A * $color;"
        bt = {"A": TEXType.VEC3}
        _, _, refs, _, param_info, *_ = cache.compile_tex(code, bt)
        assert "color" in param_info
        dv = param_info["color"]["default_value"]
        assert isinstance(dv, list) and len(dv) == 3
        assert abs(dv[0] - 0.2) < 1e-6 and abs(dv[1] - 0.4) < 1e-6 and abs(dv[2] - 0.6) < 1e-6
        r.ok("param: vec3 default extraction from vec constructor")
    except Exception as e:
        r.fail("param: vec3 default extraction from vec constructor", f"{e}\n{traceback.format_exc()}")

    # _convert_param_value tests
    try:
        from TEX_Wrangle.tex_node import _convert_param_value
        # Bool conversion
        assert _convert_param_value(True, {"type_hint": "b"}) == 1.0
        assert _convert_param_value(False, {"type_hint": "b"}) == 0.0
        # Color conversion
        rgb = _convert_param_value("#00FF00", {"type_hint": "c"})
        assert isinstance(rgb, list) and abs(rgb[1] - 1.0) < 1e-3
        # Vec2 string conversion
        v2 = _convert_param_value("0.5, 0.3", {"type_hint": "v2"})
        assert isinstance(v2, list) and abs(v2[0] - 0.5) < 1e-6 and abs(v2[1] - 0.3) < 1e-6
        # Vec3 string conversion
        v3 = _convert_param_value("1.0, 2.0, 3.0", {"type_hint": "v3"})
        assert isinstance(v3, list) and len(v3) == 3
        # Passthrough for non-matching types
        assert _convert_param_value(0.5, {"type_hint": "f"}) == 0.5
        r.ok("param: _convert_param_value helper")
    except Exception as e:
        r.fail("param: _convert_param_value helper", f"{e}\n{traceback.format_exc()}")


def test_user_functions(r: SubTestResult):
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


def test_user_functions_advanced(r: SubTestResult):
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
