"""
v0.15.0 Phase 3 regression tests — memory.
M-1: peak estimator + free_tensor_caches.
"""
from helpers import *
from TEX_Wrangle.tex_memory import (
    estimate_peak_bytes, free_tensor_caches, enforce_cache_budget,
    cache_budget_bytes,
)
import os


def _compile(code, bt):
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    return prog


def test_m1_peak_estimator(r: SubTestResult):
    print("\n--- M-1: peak-memory estimator ---")
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    shape = (1, 512, 512)
    frame4 = 1 * 512 * 512 * 4 * 4  # [B,H,W,4] fp32 bytes

    # Pointwise program → K=1.
    try:
        prog = _compile("@OUT = vec4(@A * 0.5 + 0.1, 1.0);", bt)
        est = estimate_peak_bytes(prog, shape)
        assert abs(est - frame4) < frame4 * 0.01, f"pointwise K!=1 (est {est} vs {frame4})"
        r.ok("pointwise program → K=1")
    except Exception as e:
        r.fail("pointwise program → K=1", str(e))

    # Sample-family call inside a loop → K=4.
    try:
        prog = _compile(
            "i$r=2; vec3 a=vec3(0.0); for(int d=-$r; d<=$r; d=d+1){ a=a+sample(@A,u,v).rgb; } @OUT=vec4(a,1.0);",
            {"A": TEXType.VEC3, "r": TEXType.INT, "OUT": TEXType.VEC4})
        est = estimate_peak_bytes(prog, shape)
        assert est >= frame4 * 3.9, f"sample-in-loop K should be ~4 (est {est} vs 4x {frame4})"
        r.ok("sample-in-loop program → K≈4")
    except Exception as e:
        r.fail("sample-in-loop program → K≈4", str(e))

    # A large vec4 array contributes its static bytes (the ~17x term).
    try:
        prog = _compile(
            "vec4 arr[25]; arr[0]=vec4(@A, 1.0); @OUT = arr[0];",
            {"A": TEXType.VEC3, "OUT": TEXType.VEC4})
        est = estimate_peak_bytes(prog, shape)
        array_bytes = 25 * 4 * (512 * 512) * 4  # 25 vec4 elements per pixel
        assert est >= array_bytes, f"array term missing (est {est} < arrays {array_bytes})"
        r.ok("vec4 array adds its static bytes")
    except Exception as e:
        r.fail("vec4 array adds its static bytes", str(e))

    # No spatial shape → 0.
    try:
        assert estimate_peak_bytes(_compile("@OUT = vec4(1.0);", bt), None) == 0
        r.ok("no spatial shape → 0")
    except Exception as e:
        r.fail("no spatial shape → 0", str(e))


def test_m2_cache_budget(r: SubTestResult):
    print("\n--- M-2: byte-budgeted cache eviction ---")
    from TEX_Wrangle.tex_runtime import stdlib as _sl

    # A tiny env budget evicts oldest mip entries down to (near) empty.
    try:
        free_tensor_caches()
        # Populate the mip cache with several distinct images large enough that
        # 6 entries (~3 MB each) exceed the 1 MB test budget.
        for s in range(6):
            img = make_img(1, 512, 512, 3, seed=s)
            compile_and_run("@OUT = vec4(sample_mip(@A, u, v, 1.0).rgb, 1.0);", {"A": img})
        n_before = len(_sl._mip_cache)
        saved = os.environ.get("TEX_CACHE_BUDGET_MB")
        os.environ["TEX_CACHE_BUDGET_MB"] = "1"  # 1 MB — forces eviction
        try:
            enforce_cache_budget("cpu")
        finally:
            if saved is None:
                os.environ.pop("TEX_CACHE_BUDGET_MB", None)
            else:
                os.environ["TEX_CACHE_BUDGET_MB"] = saved
        n_after = len(_sl._mip_cache)
        assert n_after <= max(1, n_before), "budget did not evict"
        assert n_after < n_before or n_before <= 1, f"no eviction under 1MB budget ({n_before}->{n_after})"
        r.ok(f"tiny budget evicts mip entries ({n_before}->{n_after})")
    except Exception as e:
        r.fail("tiny budget evicts mip entries", str(e))

    # A generous budget keeps entries; env override is honored.
    try:
        assert cache_budget_bytes.__call__ is not None
        os.environ["TEX_CACHE_BUDGET_MB"] = "4096"
        try:
            assert cache_budget_bytes("cpu") == 4096 * 1024 * 1024
        finally:
            os.environ.pop("TEX_CACHE_BUDGET_MB", None)
        # default CPU budget is 512 MB
        assert cache_budget_bytes("cpu") == 512 * 1024 * 1024
        r.ok("cache budget env override + CPU default")
    except Exception as e:
        r.fail("cache budget env override + CPU default", str(e))
    finally:
        free_tensor_caches()


def test_m3_fp16_mode(r: SubTestResult):
    print("\n--- M-3: fp16 image-data mode (fp32 coordinates) ---")
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

    def _run(code, bindings, precision, H, W):
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        used = _collect_identifiers(prog)
        return Interpreter().execute(prog, bindings, tm, device="cpu",
                                     output_names=["OUT"], precision=precision,
                                     used_builtins=used)["OUT"]

    # Coordinates stay fp32: u at 2048² has full distinct-column resolution even
    # in fp16 mode (fp16 u would collapse to ~4097 values).
    try:
        img = torch.rand(1, 2048, 2048, 3)
        r16 = _run("@OUT = vec4(vec3(u), 1.0);", {"A": img}, "fp16", 2048, 2048)
        distinct = len(torch.unique(r16[0, 0, :, 0]))
        assert distinct == 2048, f"fp16 coordinates collapsed ({distinct} distinct, want 2048)"
        r.ok(f"coordinates stay fp32 in fp16 mode ({distinct} distinct @2048)")
    except Exception as e:
        r.fail("coordinates stay fp32 in fp16 mode", str(e))

    # Pointwise fp16 accuracy within the 8-bit quantum.
    try:
        img = torch.rand(1, 256, 256, 3)
        code = "vec3 c=@A.rgb; float l=luma(c); @OUT=vec4(mix(c, vec3(l), 0.4)*1.1 + 0.05, 1.0);"
        a16 = _run(code, {"A": img}, "fp16", 256, 256)
        a32 = _run(code, {"A": img}, "fp32", 256, 256)
        md = (a16[..., :3].float() - a32[..., :3].float()).abs().max().item()
        assert md < 4e-3, f"fp16 accuracy {md} exceeds 8-bit quantum"
        r.ok(f"pointwise fp16 accuracy {md:.1e} < 4e-3")
    except Exception as e:
        r.fail("pointwise fp16 accuracy", str(e))

    # Sampling works under fp16 (grid_sample dtype reconciled; grid stays fp32).
    try:
        img = torch.rand(1, 128, 128, 3)
        s16 = _run("@OUT = vec4(sample(@A, u, v).rgb, 1.0);", {"A": img}, "fp16", 128, 128)
        assert torch.isfinite(s16).all()
        r.ok("sample() works in fp16 (dtype reconciled)")
    except Exception as e:
        r.fail("sample() works in fp16 (dtype reconciled)", str(e))

    # Output is upcast to fp32 (the IMAGE contract is unchanged).
    try:
        img = torch.rand(1, 16, 16, 3)
        out = _run("@OUT = vec4(@A * 0.5, 1.0);", {"A": img}, "fp16", 16, 16)
        assert out.dtype == torch.float32, f"fp16 output not upcast (got {out.dtype})"
        r.ok("fp16 output upcast to fp32 (IMAGE contract preserved)")
    except Exception as e:
        r.fail("fp16 output upcast to fp32", str(e))


def test_m4_tiling(r: SubTestResult):
    print("\n--- M-4: tiled execution driver ---")
    from TEX_Wrangle.tex_memory import is_tile_safe, run_tiled

    def _compile_full(code, bt):
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        return prog, tm, _collect_identifiers(prog)

    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

    # Tile-safety analysis.
    try:
        assert is_tile_safe(_compile_full("@OUT = vec4(@A.rgb * v + u, 1.0);", bt)[0]) is True
        assert is_tile_safe(_compile_full("@OUT = vec4(sample(@A, u, v).rgb, 1.0);", bt)[0]) is False
        assert is_tile_safe(_compile_full("@OUT = vec4(vec3(img_mean(@A)), 1.0);", bt)[0]) is False
        assert is_tile_safe(_compile_full("@OUT = @A; @OUT[ix, iy] = vec3(1.0);", bt)[0]) is False
        r.ok("tile-safety: pointwise safe; sample/img_mean/scatter unsafe")
    except Exception as e:
        r.fail("tile-safety analysis", str(e))

    # Seam correctness: run_tiled output == untiled, bitwise, for a coordinate-
    # heavy program across strip counts.
    try:
        code = "vec3 c=@A.rgb; float m = sin(u*6.28)*cos(v*6.28) + v*0.5 + float(iy)/ih; @OUT = vec4(c*m + vec3(u,v,0.5)*0.1, 1.0);"
        prog, tm, used = _compile_full(code, bt)
        torch.manual_seed(0)
        img = torch.rand(1, 256, 192, 3)
        interp = Interpreter()
        full = interp.execute(prog, {"A": img}, tm, device="cpu",
                              output_names=["OUT"], used_builtins=used)["OUT"]
        ok = True
        for n in (2, 3, 7):
            tiled = run_tiled(interp, prog, {"A": img}, tm, "cpu", 0, ["OUT"], used, "fp32", n)["OUT"]
            md = (tiled - full).abs().max().item()
            ok = ok and md == 0.0
        assert ok, "tiled output not bitwise-identical to untiled"
        r.ok("run_tiled seam correctness bitwise (2/3/7 strips)")
    except Exception as e:
        r.fail("run_tiled seam correctness", str(e))

    # Multi-output tiling assembles each output correctly.
    try:
        code = "@a = @A * 0.5; @b = vec3(v);"
        prog, tm, used = _compile_full(code, {"A": TEXType.VEC3, "a": TEXType.VEC3, "b": TEXType.VEC3})
        img = torch.rand(1, 128, 64, 3)
        interp = Interpreter()
        full = interp.execute(prog, {"A": img}, tm, device="cpu",
                              output_names=["a", "b"], used_builtins=used)
        tiled = run_tiled(interp, prog, {"A": img}, tm, "cpu", 0, ["a", "b"], used, "fp32", 4)
        assert torch.equal(tiled["a"], full["a"]) and torch.equal(tiled["b"], full["b"])
        r.ok("run_tiled multi-output assembly")
    except Exception as e:
        r.fail("run_tiled multi-output assembly", str(e))


def test_m5_out_reuse(r: SubTestResult):
    print("\n--- M-5: codegen out= temp reuse ---")
    from TEX_Wrangle.tex_runtime import codegen as cgmod
    from TEX_Wrangle.tex_runtime import compiled as cmod
    from TEX_Wrangle.tex_compiler.optimizer import optimize

    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

    def _emit_src(code, enabled):
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        prog = optimize(prog, tm)
        saved = cgmod._OUT_REUSE_ENABLED
        cgmod._OUT_REUSE_ENABLED = enabled
        try:
            gen = cgmod._CodeGen(tm)
            gen.emit_program(prog)
            return "\n".join(gen._lines)
        finally:
            cgmod._OUT_REUSE_ENABLED = saved

    def _run(code, img, enabled, fp):
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        saved = cgmod._OUT_REUSE_ENABLED
        cgmod._OUT_REUSE_ENABLED = enabled
        try:
            return cmod._codegen_only_execute(prog, {"A": img}, tm, "cpu",
                                              output_names=["OUT"], fingerprint=fp, time_context=None)["OUT"]
        finally:
            cgmod._OUT_REUSE_ENABLED = saved

    grade = ("vec3 c=@A.rgb; c = c*1.2 - 0.1; c = c*0.9 + 0.05; "
             "c = (c - 0.5)*1.3 + 0.5; c = c*1.05 + 0.02; @OUT=vec4(c,1.0);")

    # The reuse fires on the const-arithmetic chain; the kill switch removes it.
    try:
        src_on = _emit_src(grade, True)
        src_off = _emit_src(grade, False)
        assert src_on.count("out=") >= 4, f"expected >=4 out= reuses, got {src_on.count('out=')}"
        assert "out=" not in src_off, "kill switch did not disable out= reuse"
        r.ok(f"out= reuse fires on grade chain ({src_on.count('out=')}x); kill switch disables")
    except Exception as e:
        r.fail("out= reuse fires / kill switch", str(e))

    # Bit-exact: reuse ON vs reuse OFF, and vs the tree-walking interpreter.
    try:
        torch.manual_seed(1)
        img = torch.rand(1, 64, 48, 3)
        on = _run(grade, img, True, "m5_on")
        off = _run(grade, img, False, "m5_off")
        prog = Parser(Lexer(grade).tokenize(), source=grade).parse()
        tm = TypeChecker(binding_types=bt, source=grade).check(prog)
        ref = Interpreter().execute(prog, {"A": img}, tm, device="cpu",
                                    output_names=["OUT"])["OUT"]
        assert (on - off).abs().max().item() == 0.0, "reuse != no-reuse"
        assert (on - ref).abs().max().item() < 1e-6, "reuse != interpreter"
        r.ok("out= reuse bit-exact vs no-reuse and vs interpreter")
    except Exception as e:
        r.fail("out= reuse bit-exact", str(e))

    # Shape-safety hazard: a spatial FLOAT mixed with a scalar-typed operand must
    # NOT be reused into the smaller buffer. Non-constant operands never reuse;
    # verify the chain stays correct (would crash 'out shape mismatch' if unsafe).
    try:
        torch.manual_seed(2)
        img = torch.rand(1, 32, 32, 3)
        # 'm' is a scalar-derived FLOAT; c is a spatial vec3 — mixed operands.
        code = ("vec3 c=@A.rgb; float m = luma(c); m = m*2.0 - 0.3; "
                "c = c + vec3(m)*0.1; @OUT=vec4(c,1.0);")
        b2 = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=b2, source=code).check(prog)
        saved = cgmod._OUT_REUSE_ENABLED
        cgmod._OUT_REUSE_ENABLED = True
        try:
            got = cmod._codegen_only_execute(prog, {"A": img}, tm, "cpu",
                                             output_names=["OUT"], fingerprint="m5_hz", time_context=None)["OUT"]
        finally:
            cgmod._OUT_REUSE_ENABLED = saved
        ref = Interpreter().execute(prog, {"A": img}, tm, device="cpu",
                                    output_names=["OUT"])["OUT"]
        assert (got - ref).abs().max().item() < 1e-6
        r.ok("spatial/scalar mixed chain stays correct under reuse")
    except Exception as e:
        r.fail("spatial/scalar mixed chain", str(e))


def test_m1_free_caches(r: SubTestResult):
    print("\n--- M-1: free_tensor_caches ---")
    from TEX_Wrangle.tex_runtime import stdlib as _sl
    try:
        # Populate a couple caches, then free.
        img = make_img(1, 16, 16, 3)
        compile_and_run("@OUT = vec4(sample(@A, u, v).rgb, 1.0);", {"A": img})
        free_tensor_caches()
        assert len(_sl._sampler_cache) == 0 and len(_sl._grid_buf) == 0
        r.ok("free_tensor_caches clears stdlib tensor caches")
    except Exception as e:
        r.fail("free_tensor_caches clears stdlib tensor caches", str(e))
