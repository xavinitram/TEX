"""Noise and sampling tests."""
from helpers import *


def test_sampling(r: SubTestResult):
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

def test_noise(r: SubTestResult):
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


def test_arithmetic_hash_noise(r: SubTestResult):
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


# ── New Noise Function Tests ──────────────────────────────────────────

def test_new_noise_functions(r: SubTestResult):
    """Tests for worley, curl, ridged, billow, turbulence, flow, alligator noise."""
    print("\n--- New Noise Functions ---")

    noise_img = torch.zeros(1, 32, 32, 3)

    # Worley F1 — basic shape and range
    try:
        code = "float n = worley_f1(u * 4.0, v * 4.0);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result.dim() == 4 and result.shape[-1] == 3
        vals = result[..., 0]
        assert vals.min().item() >= -0.01, f"F1 min too low: {vals.min().item()}"
        assert vals.std().item() > 0.01, "F1 has no spatial variation"
        r.ok("noise: worley_f1 shape and range")
    except Exception as e:
        r.fail("noise: worley_f1 shape and range", f"{e}\n{traceback.format_exc()}")

    # Worley F2 — should be >= F1 everywhere
    try:
        code = """
float f1 = worley_f1(u * 4.0, v * 4.0);
float f2 = worley_f2(u * 4.0, v * 4.0);
@OUT = vec3(f1, f2, f2 - f1);
"""
        result = compile_and_run(code, {"A": noise_img})
        f2_minus_f1 = result[..., 2]
        assert f2_minus_f1.min().item() >= -1e-5, f"F2 < F1 somewhere: {f2_minus_f1.min().item()}"
        r.ok("noise: worley_f2 >= f1")
    except Exception as e:
        r.fail("noise: worley_f2 >= f1", f"{e}\n{traceback.format_exc()}")

    # Voronoi alias — should equal F1
    try:
        code = """
float a = voronoi(u * 4.0, v * 4.0);
float b = worley_f1(u * 4.0, v * 4.0);
@OUT = vec3(abs(a - b), 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": noise_img})
        assert result[..., 0].max().item() < 1e-5, "voronoi != worley_f1"
        r.ok("noise: voronoi alias")
    except Exception as e:
        r.fail("noise: voronoi alias", f"{e}\n{traceback.format_exc()}")

    # Curl — returns vec2, has spatial variation
    try:
        code = "vec2 c = curl(u * 4.0, v * 4.0);\n@OUT = c;"
        result = compile_and_run(code, {"A": noise_img})
        assert result.shape[-1] == 2, f"Curl should return vec2, got shape {result.shape}"
        assert result[..., 0].std().item() > 0.01, "Curl x has no variation"
        assert result[..., 1].std().item() > 0.01, "Curl y has no variation"
        r.ok("noise: curl returns vec2")
    except Exception as e:
        r.fail("noise: curl returns vec2", f"{e}\n{traceback.format_exc()}")

    # Curl 3D — returns vec3
    try:
        code = "vec3 c = curl(u * 4.0, v * 4.0, 0.5);\n@OUT = c;"
        result = compile_and_run(code, {"A": noise_img})
        assert result.shape[-1] == 3, f"3D curl should return vec3, got shape {result.shape}"
        assert result[..., 0].std().item() > 0.01, "3D curl x has no variation"
        r.ok("noise: curl 3D returns vec3")
    except Exception as e:
        r.fail("noise: curl 3D returns vec3", f"{e}\n{traceback.format_exc()}")

    # Ridged — shape, range, spatial variation
    try:
        code = "float n = ridged(u * 4.0, v * 4.0, 4);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        vals = result[..., 0]
        assert vals.min().item() >= -0.1, f"Ridged min too low: {vals.min().item()}"
        assert vals.std().item() > 0.01, "Ridged has no variation"
        r.ok("noise: ridged")
    except Exception as e:
        r.fail("noise: ridged", f"{e}\n{traceback.format_exc()}")

    # Billow — shape and range
    try:
        code = "float n = billow(u * 4.0, v * 4.0, 4);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        vals = result[..., 0]
        assert vals.std().item() > 0.01, "Billow has no variation"
        r.ok("noise: billow")
    except Exception as e:
        r.fail("noise: billow", f"{e}\n{traceback.format_exc()}")

    # Turbulence — should be in [0, 1]
    try:
        code = "float n = turbulence(u * 4.0, v * 4.0, 4);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        vals = result[..., 0]
        assert vals.min().item() >= -0.01, f"Turbulence min: {vals.min().item()}"
        assert vals.max().item() <= 1.01, f"Turbulence max: {vals.max().item()}"
        assert vals.std().item() > 0.01, "Turbulence has no variation"
        r.ok("noise: turbulence")
    except Exception as e:
        r.fail("noise: turbulence", f"{e}\n{traceback.format_exc()}")

    # Flow — varies with time
    try:
        code_t0 = "float n = flow(u * 4.0, v * 4.0, 0.0);\n@OUT = vec3(n, n, n);"
        code_t1 = "float n = flow(u * 4.0, v * 4.0, 1.0);\n@OUT = vec3(n, n, n);"
        r0 = compile_and_run(code_t0, {"A": noise_img})
        r1 = compile_and_run(code_t1, {"A": noise_img})
        diff = (r0 - r1).abs().mean().item()
        assert diff > 0.01, f"Flow should vary with time, diff={diff}"
        r.ok("noise: flow time variation")
    except Exception as e:
        r.fail("noise: flow time variation", f"{e}\n{traceback.format_exc()}")

    # Alligator — shape and spatial variation
    try:
        code = "float n = alligator(u * 4.0, v * 4.0, 4);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        vals = result[..., 0]
        assert vals.min().item() >= -0.01, f"Alligator min: {vals.min().item()}"
        assert vals.std().item() > 0.01, "Alligator has no variation"
        r.ok("noise: alligator")
    except Exception as e:
        r.fail("noise: alligator", f"{e}\n{traceback.format_exc()}")

    # Alligator with default octaves (no 3rd arg)
    try:
        code = "float n = alligator(u * 4.0, v * 4.0);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result.dim() == 4
        r.ok("noise: alligator default octaves")
    except Exception as e:
        r.fail("noise: alligator default octaves", f"{e}\n{traceback.format_exc()}")

    # Determinism — worley should be reproducible
    try:
        code = "float n = worley_f1(u * 4.0, v * 4.0);\n@OUT = vec3(n, n, n);"
        r1 = compile_and_run(code, {"A": noise_img})
        r2 = compile_and_run(code, {"A": noise_img})
        assert torch.allclose(r1, r2), "Worley not deterministic"
        r.ok("noise: worley determinism")
    except Exception as e:
        r.fail("noise: worley determinism", f"{e}\n{traceback.format_exc()}")


# ── 3D Noise Tests ────────────────────────────────────────────────────

def test_3d_noise(r: SubTestResult):
    """Tests for 3D noise functions."""
    print("\n--- 3D Noise Tests ---")

    noise_img = torch.zeros(1, 32, 32, 3)

    # 3D perlin — basic shape and variation
    try:
        code = "float n = perlin(u * 4.0, v * 4.0, 0.5);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result.dim() == 4 and result.shape[-1] == 3
        vals = result[..., 0]
        assert vals.std().item() > 0.01, "3D perlin has no variation"
        assert vals.max().item() <= 1.5 and vals.min().item() >= -1.5
        r.ok("noise: 3D perlin")
    except Exception as e:
        r.fail("noise: 3D perlin", f"{e}\n{traceback.format_exc()}")

    # 3D perlin varies with z
    try:
        code_z0 = "float n = perlin(u * 4.0, v * 4.0, 0.0);\n@OUT = vec3(n, n, n);"
        code_z1 = "float n = perlin(u * 4.0, v * 4.0, 1.0);\n@OUT = vec3(n, n, n);"
        r0 = compile_and_run(code_z0, {"A": noise_img})
        r1 = compile_and_run(code_z1, {"A": noise_img})
        diff = (r0 - r1).abs().mean().item()
        assert diff > 0.01, f"3D perlin should vary with z, diff={diff}"
        r.ok("noise: 3D perlin z variation")
    except Exception as e:
        r.fail("noise: 3D perlin z variation", f"{e}\n{traceback.format_exc()}")

    # 2D perlin backward compat (2 args still works)
    try:
        code = "float n = perlin(u * 4.0, v * 4.0);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result.dim() == 4
        r.ok("noise: 2D perlin backward compat")
    except Exception as e:
        r.fail("noise: 2D perlin backward compat", f"{e}\n{traceback.format_exc()}")

    # 3D FBM — 4 args
    try:
        code = "float n = fbm(u * 4.0, v * 4.0, 0.5, 4);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result[..., 0].std().item() > 0.01, "3D FBM has no variation"
        r.ok("noise: 3D FBM")
    except Exception as e:
        r.fail("noise: 3D FBM", f"{e}\n{traceback.format_exc()}")

    # 2D FBM backward compat (3 args)
    try:
        code = "float n = fbm(u * 4.0, v * 4.0, 4);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result.dim() == 4
        r.ok("noise: 2D FBM backward compat")
    except Exception as e:
        r.fail("noise: 2D FBM backward compat", f"{e}\n{traceback.format_exc()}")

    # 3D worley
    try:
        code = "float n = worley_f1(u * 4.0, v * 4.0, 0.5);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        vals = result[..., 0]
        assert vals.min().item() >= -0.01, f"3D worley min too low: {vals.min().item()}"
        assert vals.std().item() > 0.01, "3D worley has no variation"
        r.ok("noise: 3D worley_f1")
    except Exception as e:
        r.fail("noise: 3D worley_f1", f"{e}\n{traceback.format_exc()}")

    # 3D ridged
    try:
        code = "float n = ridged(u * 4.0, v * 4.0, 0.5, 4);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result[..., 0].std().item() > 0.01, "3D ridged has no variation"
        r.ok("noise: 3D ridged")
    except Exception as e:
        r.fail("noise: 3D ridged", f"{e}\n{traceback.format_exc()}")

    # 3D billow
    try:
        code = "float n = billow(u * 4.0, v * 4.0, 0.5, 4);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result[..., 0].std().item() > 0.01, "3D billow has no variation"
        r.ok("noise: 3D billow")
    except Exception as e:
        r.fail("noise: 3D billow", f"{e}\n{traceback.format_exc()}")

    # 3D turbulence
    try:
        code = "float n = turbulence(u * 4.0, v * 4.0, 0.5, 4);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        vals = result[..., 0]
        assert vals.min().item() >= -0.01, f"3D turbulence min: {vals.min().item()}"
        assert vals.std().item() > 0.01, "3D turbulence has no variation"
        r.ok("noise: 3D turbulence")
    except Exception as e:
        r.fail("noise: 3D turbulence", f"{e}\n{traceback.format_exc()}")

    # 3D flow
    try:
        code = "float n = flow(u * 4.0, v * 4.0, 0.5, 1.0);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result[..., 0].std().item() > 0.01, "3D flow has no variation"
        r.ok("noise: 3D flow")
    except Exception as e:
        r.fail("noise: 3D flow", f"{e}\n{traceback.format_exc()}")

    # 3D alligator
    try:
        code = "float n = alligator(u * 4.0, v * 4.0, 0.5, 4);\n@OUT = vec3(n, n, n);"
        result = compile_and_run(code, {"A": noise_img})
        assert result[..., 0].std().item() > 0.01, "3D alligator has no variation"
        r.ok("noise: 3D alligator")
    except Exception as e:
        r.fail("noise: 3D alligator", f"{e}\n{traceback.format_exc()}")

    # 3D perlin determinism
    try:
        code = "float n = perlin(u * 4.0, v * 4.0, 0.5);\n@OUT = vec3(n, n, n);"
        r1 = compile_and_run(code, {"A": noise_img})
        r2 = compile_and_run(code, {"A": noise_img})
        assert torch.allclose(r1, r2), "3D perlin not deterministic"
        r.ok("noise: 3D perlin determinism")
    except Exception as e:
        r.fail("noise: 3D perlin determinism", f"{e}\n{traceback.format_exc()}")


def test_sample_mip(r: SubTestResult):
    """Tests for sample_mip() mipmap sampling."""
    print("\n--- Mipmap Sampling Tests ---")

    B, H, W = 1, 32, 32
    # Create a checkerboard pattern — high frequency content that
    # should blur to gray (0.5) at higher mip levels
    checker = torch.zeros(B, H, W, 3)
    for y in range(H):
        for x in range(W):
            if (x + y) % 2 == 0:
                checker[0, y, x] = 1.0

    # LOD 0: should reproduce the input (full resolution)
    try:
        result = compile_and_run("@OUT = sample_mip(@A, u, v, 0.0);", {"A": checker})
        max_diff = (result - checker).abs().max().item()
        assert max_diff < 0.05, f"LOD 0 should match input, max_diff={max_diff}"
        r.ok("mip: LOD 0 identity")
    except Exception as e:
        r.fail("mip: LOD 0 identity", f"{e}\n{traceback.format_exc()}")

    # High LOD: checkerboard should average to ~0.5 (gray)
    try:
        result = compile_and_run("@OUT = sample_mip(@A, u, v, 4.0);", {"A": checker})
        mean = result.mean().item()
        assert abs(mean - 0.5) < 0.15, f"High LOD should average to ~0.5, got {mean}"
        r.ok("mip: high LOD averages")
    except Exception as e:
        r.fail("mip: high LOD averages", f"{e}\n{traceback.format_exc()}")

    # Integer LOD fast path: LOD 2.0 should work and return lower-res sample
    try:
        result = compile_and_run("@OUT = sample_mip(@A, u, v, 2.0);", {"A": checker})
        assert result.shape == checker.shape, f"Shape mismatch: {result.shape}"
        # Should be blurrier than LOD 0 — check variance is lower
        var_0 = checker.var().item()
        var_2 = result.var().item()
        assert var_2 < var_0, f"LOD 2 should be blurrier: var_0={var_0:.4f}, var_2={var_2:.4f}"
        r.ok("mip: integer LOD fast path")
    except Exception as e:
        r.fail("mip: integer LOD fast path", f"{e}\n{traceback.format_exc()}")

    # Fractional LOD: should interpolate between two levels
    try:
        result = compile_and_run("@OUT = sample_mip(@A, u, v, 1.5);", {"A": checker})
        # Should be between LOD 1 and LOD 2 variance
        r1 = compile_and_run("@OUT = sample_mip(@A, u, v, 1.0);", {"A": checker})
        r2 = compile_and_run("@OUT = sample_mip(@A, u, v, 2.0);", {"A": checker})
        var_1 = r1.var().item()
        var_15 = result.var().item()
        var_2 = r2.var().item()
        # Variance at 1.5 should be between 1 and 2 (or close)
        assert var_15 <= var_1 + 0.01, f"LOD 1.5 should be blurrier than LOD 1"
        r.ok("mip: fractional LOD trilinear")
    except Exception as e:
        r.fail("mip: fractional LOD trilinear", f"{e}\n{traceback.format_exc()}")

    # Per-pixel LOD: distance from center controls blur (tilt-shift)
    try:
        code = """
float dx = u - 0.5;
float dy = v - 0.5;
float d = sqrt(dx * dx + dy * dy);
float lod = d * 8.0;
@OUT = sample_mip(@A, u, v, lod);
"""
        result = compile_and_run(code, {"A": checker})
        # Center should be sharp (close to original), edges blurry
        center_var = result[0, 14:18, 14:18].var().item()
        edge_var = result[0, 0:4, 0:4].var().item()
        assert center_var > edge_var, "Center should be sharper than edges in tilt-shift"
        r.ok("mip: per-pixel LOD (tilt-shift)")
    except Exception as e:
        r.fail("mip: per-pixel LOD (tilt-shift)", f"{e}\n{traceback.format_exc()}")

    # Clamping: negative LOD should clamp to 0
    try:
        result = compile_and_run("@OUT = sample_mip(@A, u, v, -5.0);", {"A": checker})
        r0 = compile_and_run("@OUT = sample_mip(@A, u, v, 0.0);", {"A": checker})
        diff = (result - r0).abs().max().item()
        assert diff < 1e-5, f"Negative LOD should clamp to 0, diff={diff}"
        r.ok("mip: negative LOD clamp")
    except Exception as e:
        r.fail("mip: negative LOD clamp", f"{e}\n{traceback.format_exc()}")

    # Very high LOD: should clamp to max level (1x1), all pixels same color
    try:
        result = compile_and_run("@OUT = sample_mip(@A, u, v, 100.0);", {"A": checker})
        # All pixels should be nearly identical at max LOD
        pixel_range = result.max().item() - result.min().item()
        assert pixel_range < 0.1, f"Max LOD should produce near-uniform output, range={pixel_range}"
        r.ok("mip: max LOD clamp")
    except Exception as e:
        r.fail("mip: max LOD clamp", f"{e}\n{traceback.format_exc()}")

    # 4-channel image
    try:
        img4 = torch.rand(1, 16, 16, 4)
        result = compile_and_run("@OUT = sample_mip(@A, u, v, 1.0);", {"A": img4})
        assert result.shape[-1] == 4, f"Should preserve 4 channels, got {result.shape[-1]}"
        r.ok("mip: 4-channel image")
    except Exception as e:
        r.fail("mip: 4-channel image", f"{e}\n{traceback.format_exc()}")


def test_gauss_blur_and_mip_gauss(r: SubTestResult):
    """Tests for gauss_blur() and sample_mip_gauss()."""
    print("\n--- Gaussian Blur & Mip Gauss Tests ---")

    B, H, W = 1, 32, 32
    # Checkerboard — high frequency that blurs to gray
    checker = torch.zeros(B, H, W, 3)
    for y in range(H):
        for x in range(W):
            if (x + y) % 2 == 0:
                checker[0, y, x] = 1.0

    # gauss_blur: sigma=0 returns input unchanged
    try:
        result = compile_and_run("@OUT = gauss_blur(@A, 0.0);", {"A": checker})
        assert torch.allclose(result[..., :3], checker, atol=1e-5), "sigma=0 should be identity"
        r.ok("gauss_blur: sigma=0 identity")
    except Exception as e:
        r.fail("gauss_blur: sigma=0 identity", f"{e}\n{traceback.format_exc()}")

    # gauss_blur: positive sigma reduces variance (blurs)
    try:
        result = compile_and_run("@OUT = gauss_blur(@A, 2.0);", {"A": checker})
        var_orig = checker.var().item()
        var_blurred = result[..., :3].var().item()
        assert var_blurred < var_orig * 0.5, \
            f"sigma=2 should significantly blur: orig_var={var_orig:.4f}, blurred_var={var_blurred:.4f}"
        r.ok("gauss_blur: reduces variance")
    except Exception as e:
        r.fail("gauss_blur: reduces variance", f"{e}\n{traceback.format_exc()}")

    # gauss_blur: large sigma approaches uniform mean
    try:
        result = compile_and_run("@OUT = gauss_blur(@A, 10.0);", {"A": checker})
        mean = result[..., :3].mean().item()
        assert abs(mean - 0.5) < 0.1, f"Large sigma should average to ~0.5, got {mean}"
        r.ok("gauss_blur: large sigma averages")
    except Exception as e:
        r.fail("gauss_blur: large sigma averages", f"{e}\n{traceback.format_exc()}")

    # gauss_blur: preserves energy (mean should stay ~0.5 for checkerboard)
    try:
        result = compile_and_run("@OUT = gauss_blur(@A, 3.0);", {"A": checker})
        orig_mean = checker.mean().item()
        blur_mean = result[..., :3].mean().item()
        assert abs(orig_mean - blur_mean) < 0.05, \
            f"Blur should preserve mean: orig={orig_mean:.4f}, blur={blur_mean:.4f}"
        r.ok("gauss_blur: energy preservation")
    except Exception as e:
        r.fail("gauss_blur: energy preservation", f"{e}\n{traceback.format_exc()}")

    # gauss_blur: 4-channel image
    try:
        img4 = torch.rand(1, 16, 16, 4)
        result = compile_and_run("@OUT = gauss_blur(@A, 1.5);", {"A": img4})
        assert result.shape[-1] == 4, f"Should preserve 4 channels, got {result.shape[-1]}"
        r.ok("gauss_blur: 4-channel")
    except Exception as e:
        r.fail("gauss_blur: 4-channel", f"{e}\n{traceback.format_exc()}")

    # sample_mip_gauss: LOD 0 identity
    try:
        result = compile_and_run("@OUT = sample_mip_gauss(@A, u, v, 0.0);", {"A": checker})
        max_diff = (result[..., :3] - checker).abs().max().item()
        assert max_diff < 0.05, f"LOD 0 should match input, max_diff={max_diff}"
        r.ok("mip_gauss: LOD 0 identity")
    except Exception as e:
        r.fail("mip_gauss: LOD 0 identity", f"{e}\n{traceback.format_exc()}")

    # sample_mip_gauss: high LOD averages to ~0.5
    try:
        result = compile_and_run("@OUT = sample_mip_gauss(@A, u, v, 4.0);", {"A": checker})
        mean = result[..., :3].mean().item()
        assert abs(mean - 0.5) < 0.15, f"High LOD should average to ~0.5, got {mean}"
        r.ok("mip_gauss: high LOD averages")
    except Exception as e:
        r.fail("mip_gauss: high LOD averages", f"{e}\n{traceback.format_exc()}")

    # sample_mip_gauss: produces different (smoother) results than box pyramid
    try:
        # Use a non-pathological image; at fractional LOD the two pyramids should differ
        natural = torch.rand(B, H, W, 3)
        r_box = compile_and_run("@OUT = sample_mip(@A, u, v, 0.5);", {"A": natural})
        r_gauss = compile_and_run("@OUT = sample_mip_gauss(@A, u, v, 0.5);", {"A": natural})
        diff = (r_box[..., :3] - r_gauss[..., :3]).abs().max().item()
        assert diff > 1e-4, f"Gauss and box pyramids should differ: max_diff={diff}"
        r.ok("mip_gauss: differs from box pyramid")
    except Exception as e:
        r.fail("mip_gauss: differs from box pyramid", f"{e}\n{traceback.format_exc()}")

    # sample_mip_gauss: fractional LOD trilinear works
    try:
        result = compile_and_run("@OUT = sample_mip_gauss(@A, u, v, 1.5);", {"A": checker})
        r1 = compile_and_run("@OUT = sample_mip_gauss(@A, u, v, 1.0);", {"A": checker})
        var_1 = r1[..., :3].var().item()
        var_15 = result[..., :3].var().item()
        assert var_15 <= var_1 + 0.01, "LOD 1.5 should be blurrier than LOD 1"
        r.ok("mip_gauss: fractional LOD trilinear")
    except Exception as e:
        r.fail("mip_gauss: fractional LOD trilinear", f"{e}\n{traceback.format_exc()}")

    # Exponential blur integration test: 6-tap with Gaussian pyramid
    try:
        code = """
f$radius = 16.0;
float offsets[] = {-3.5416, -2.1090, -1.0466, -0.1620, 0.6150, 1.3337};
float weights[] = {0.00063, 0.00802, 0.05349, 0.21820, 0.45467, 0.26499};
float base = log2(max($radius, 1.0) / 0.825);
vec3 acc = vec3(0.0);
for (int i = 0; i < 6; i++) {
    float lod = max(base + offsets[i], 0.0);
    acc += sample_mip_gauss(@A, u, v, lod) * weights[i];
}
@OUT = acc;
"""
        smooth = torch.rand(B, H, W, 3)
        result = compile_and_run(code, {"A": smooth, "radius": 16.0})
        # Result should be blurred but preserve mean
        orig_mean = smooth.mean().item()
        blur_mean = result[..., :3].mean().item()
        assert abs(orig_mean - blur_mean) < 0.1, \
            f"6-tap exp blur should preserve mean: orig={orig_mean:.3f}, blur={blur_mean:.3f}"
        r.ok("mip_gauss: 6-tap exponential blur integration")
    except Exception as e:
        r.fail("mip_gauss: 6-tap exponential blur integration", f"{e}\n{traceback.format_exc()}")
