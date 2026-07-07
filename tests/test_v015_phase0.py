"""
v0.15.0 Phase 0 regression tests — hygiene / capture-enabler / bug fixes.

Covers: PC-1 (inductor cache-dir wiring), OOM-unwrap detection, the sample_mip
inference-tensor cache-key fix, and UC-5 (literal array-index without a sync).
"""
from helpers import *
import os
from TEX_Wrangle.tex_runtime import stdlib as _stdlib_mod
from TEX_Wrangle.tex_runtime import compiled as _compiled_mod
from TEX_Wrangle.tex_cache import get_cache


def test_pc1_inductor_cache_dir(r: SubTestResult):
    print("\n--- PC-1: inductor cache-dir wiring ---")
    # The old wiring assigned torch._inductor.config.cache_dir, which does not
    # exist on torch 2.10 (raised, was swallowed). Assert our env-var fix works.
    saved_env = os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
    try:
        _compiled_mod._ensure_inductor_cache_dir()
        got = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        parent = str(get_cache().torch_compile_cache_dir)
        # PC-2 versions the dir under torch_compile/ by cache-version + torch build.
        assert got and got.startswith(parent), f"env var {got!r} not under {parent!r}"
        assert os.path.isdir(got), "cache dir was not created"
        r.ok("inductor cache dir set via env var")
    except Exception as e:
        r.fail("inductor cache dir set via env var", str(e))

    # A pre-set env var is respected, not overwritten (idempotent).
    try:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "Z:/preset_by_user"
        _compiled_mod._ensure_inductor_cache_dir()
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "Z:/preset_by_user"
        r.ok("pre-set cache dir respected")
    except Exception as e:
        r.fail("pre-set cache dir respected", str(e))
    finally:
        if saved_env is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = saved_env


def test_oom_detection(r: SubTestResult):
    print("\n--- OOM-unwrap: detection ---")
    from TEX_Wrangle.tex_node import _is_oom_error
    try:
        oom_t = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_t is not None:
            assert _is_oom_error(oom_t("out of memory")) is True
        assert _is_oom_error(RuntimeError("size of tensor a must match")) is False
        assert _is_oom_error(ValueError("bad value")) is False
        r.ok("OOM error detection (torch fallback)")
    except Exception as e:
        r.fail("OOM error detection (torch fallback)", str(e))


def test_sample_mip_inference_tensor(r: SubTestResult):
    print("\n--- sample_mip: inference-tensor cache key ---")
    # _safe_version must not raise on an inference tensor (the _version read does)
    try:
        with torch.inference_mode():
            t = torch.rand(1, 16, 16, 3)
            assert t.is_inference()
            assert _stdlib_mod._safe_version(t) == 0
        t2 = torch.rand(1, 16, 16, 3)  # normal tensor
        assert _stdlib_mod._safe_version(t2) == t2._version
        r.ok("_safe_version handles inference tensors")
    except Exception as e:
        r.fail("_safe_version handles inference tensors", str(e))

    # sample_mip end-to-end under inference_mode (the interpreter's real regime)
    try:
        img = make_img(1, 32, 32, 3)
        result = compile_and_run("@OUT = sample_mip(@A, u, v, 1.0);", {"A": img})
        assert result.shape[-1] == 3 and torch.isfinite(result).all()
        r.ok("sample_mip runs (inference-tensor path)")
    except Exception as e:
        r.fail("sample_mip runs (inference-tensor path)", str(e))

    # CUDA + CPU-resident binding: the exact today-bug the fix targets.
    try:
        if torch.cuda.is_available():
            img_cpu = make_img(1, 32, 32, 3)  # stays on CPU; interpreter moves it on-device
            result = compile_and_run("@OUT = sample_mip(@A, u, v, 1.0);",
                                     {"A": img_cpu}, device="cuda")
            assert torch.isfinite(result).all()
            r.ok("sample_mip CUDA with CPU-resident binding")
        else:
            r.ok("sample_mip CUDA with CPU-resident binding (no GPU, SKIPPED)")
    except Exception as e:
        r.fail("sample_mip CUDA with CPU-resident binding", str(e))


def test_uc5_literal_array_index(r: SubTestResult):
    print("\n--- UC-5: literal array index (no sync) ---")
    img = make_img(1, 4, 4, 3)

    # Read: literal index returns the right element (float + vec arrays)
    try:
        result = compile_and_run("""
            float arr[4];
            arr[0] = 10.0; arr[1] = 20.0; arr[2] = 30.0; arr[3] = 40.0;
            @OUT = vec3(arr[2] / 100.0);
        """, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 0.30) < 1e-6
        r.ok("float array literal index read")
    except Exception as e:
        r.fail("float array literal index read", str(e))

    try:
        result = compile_and_run("""
            vec3 arr[3];
            arr[0] = vec3(0.1, 0.0, 0.0);
            arr[1] = vec3(0.0, 0.2, 0.0);
            arr[2] = vec3(0.0, 0.0, 0.3);
            @OUT = arr[1];
        """, {"A": img})
        v = result[0, 0, 0]
        assert abs(v[1].item() - 0.2) < 1e-6 and abs(v[0].item()) < 1e-6
        r.ok("vec array literal index read")
    except Exception as e:
        r.fail("vec array literal index read", str(e))

    # Out-of-bounds literal indices clamp exactly like the tensor path
    try:
        result = compile_and_run("""
            float arr[3];
            arr[0] = 1.0; arr[1] = 2.0; arr[2] = 3.0;
            @OUT = vec3(arr[99] * 0.1, arr[0] * 0.1, 0.0);
        """, {"A": img})
        v = result[0, 0, 0]
        assert abs(v[0].item() - 0.3) < 1e-6, "arr[99] should clamp to arr[2]=3.0"
        r.ok("literal index out-of-bounds clamps")
    except Exception as e:
        r.fail("literal index out-of-bounds clamps", str(e))

    # Interpreter vs codegen equivalence for a literal-index fill+read
    assert_equiv(r, "literal-index fill/read", """
        float arr[4];
        for (int i = 0; i < 4; i = i + 1) { arr[i] = float(i); }
        @OUT = vec3(arr[0], arr[2], arr[3]) * 0.1;
    """, {"A": img})
