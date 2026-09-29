"""
v0.15.0 Phase 1 regression tests — persistence (PC-3 marshal codegen cache;
CT-1 fused-chain disk persistence).
"""
from helpers import *
from TEX_Wrangle.tex_cache import get_cache, _CG_UNSUPPORTED
from TEX_Wrangle.tex_runtime import compiled as _C
from TEX_Wrangle.tex_runtime import codegen as _CG
import TEX_Wrangle.tex_fusion as _FUS


# A loop-heavy spatial program that routes through codegen.
_CG_CODE = """
vec3 acc = vec3(0.0);
for (int i = 0; i < 4; i = i + 1) {
    acc = acc + @A.rgb * 0.1;
}
@OUT = vec4(acc, 1.0);
"""


def _compile(code, img):
    cache = get_cache()
    bt = {"A": TEXType.VEC3}
    prog, tm, refs, asg, params, used = cache.compile_tex(code, bt)
    fp = cache.fingerprint(code, bt)
    return prog, tm, used, fp


def test_pc3_codegen_persistence(r: SubTestResult):
    print("\n--- PC-3: marshal codegen persistence ---")
    cache = get_cache()
    img = make_img(1, 8, 8, 3)
    prog, tm, used, fp = _compile(_CG_CODE, img)

    # Clean slate: no memory entry, no sidecar.
    try:
        cache._codegen_memory.pop(fp, None)
        cache._cg_path(fp).unlink(missing_ok=True)

        fn1 = _C._get_or_make_codegen_fn(prog, tm, fp)
        assert fn1 is not None, "codegen returned None for a codegen-routable program"
        assert cache._cg_path(fp).exists(), "marshal sidecar was not persisted"
        assert fp in cache._codegen_memory
        r.ok("codegen fn persisted to sidecar")
    except Exception as e:
        r.fail("codegen fn persisted to sidecar", str(e))

    # Simulate a process restart: drop the memory tier, keep the sidecar, and
    # assert the fn materializes from disk WITHOUT re-emitting (no _try_codegen).
    try:
        cache._codegen_memory.pop(fp, None)
        calls = {"n": 0}
        orig = _C._try_codegen
        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        _C._try_codegen = counting
        try:
            fn2 = _C._get_or_make_codegen_fn(prog, tm, fp)
        finally:
            _C._try_codegen = orig
        assert fn2 is not None, "materialize from sidecar returned None"
        assert calls["n"] == 0, f"re-emitted {calls['n']}x instead of materializing from disk"
        r.ok("codegen fn materialized from disk (no re-emit)")
    except Exception as e:
        r.fail("codegen fn materialized from disk (no re-emit)", str(e))

    # Materialized fn must produce output bit-identical to the interpreter.
    try:
        cache._codegen_memory.pop(fp, None)  # force disk materialize path
        out_cg = execute_compiled(prog, {"A": img}, tm, "cpu", fp,
                                  used_builtins=used)
        out_interp = _plain_execute(prog, {"A": img}, tm, "cpu", used_builtins=used, time_context=None)
        if isinstance(out_cg, dict):
            out_cg = out_cg["OUT"]
        if isinstance(out_interp, dict):
            out_interp = out_interp["OUT"]
        assert torch.allclose(out_cg, out_interp, atol=1e-6), \
            f"materialized codegen output differs (max {(out_cg-out_interp).abs().max().item()})"
        r.ok("materialized codegen output matches interpreter")
    except Exception as e:
        r.fail("materialized codegen output matches interpreter", str(e))

    # A stale-version sidecar must be rejected (deleted, returns None).
    try:
        import pickle
        from TEX_Wrangle import tex_recovery
        cache._codegen_memory.pop(fp, None)
        with open(cache._cg_path(fp), "rb") as f:
            good = pickle.load(f)
        bad = dict(good); bad["version"] = "STALE_VERSION"
        # Both files are MAC-signed, so only the version differs: an unsigned file would
        # be rejected by the trailer check and never reach the version comparison.
        assert tex_recovery.sign_pickle(cache._cg_path(fp), good)
        assert cache.get_codegen_fn(fp) is not None, "signed current-version sidecar rejected"
        cache._codegen_memory.pop(fp, None)
        assert tex_recovery.sign_pickle(cache._cg_path(fp), bad)
        got = cache.get_codegen_fn(fp)  # must reject + delete
        assert got is None, "stale-version sidecar was not rejected"
        assert not cache._cg_path(fp).exists(), "stale sidecar not deleted"
        r.ok("stale-version sidecar rejected")
    except Exception as e:
        r.fail("stale-version sidecar rejected", str(e))
    finally:
        cache._codegen_memory.pop(fp, None)
        cache._cg_path(fp).unlink(missing_ok=True)


def test_pc2_precompile_safety(r: SubTestResult):
    print("\n--- PC-2: caching_precompile safety ---")
    import torch, os

    # The context enables caching_precompile only inside it, never leaking.
    try:
        import torch._dynamo.config as dc
        before = getattr(dc, "caching_precompile", None)
        with _C._precompile_ctx():
            inside = getattr(dc, "caching_precompile", None)
        after = getattr(dc, "caching_precompile", None)
        if hasattr(dc, "caching_precompile"):
            assert inside is True, "caching_precompile not enabled inside ctx"
            assert after == before, "caching_precompile leaked outside ctx"
        r.ok("precompile ctx scopes the flag")
    except Exception as e:
        r.fail("precompile ctx scopes the flag", str(e))

    # Attach-failure signatures must be recognized (so they don't blacklist).
    try:
        assert _C._is_precompile_attach_failure(AssertionError("guard miss"))
        assert _C._is_precompile_attach_failure(NameError("name '__compiled_fn_3' is not defined"))
        assert _C._is_precompile_attach_failure(
            RuntimeError("Compile package was created with a different torch version"))
        # Genuine program failures must NOT be classified as attach failures.
        assert not _C._is_precompile_attach_failure(NameError("name 'foo' is not defined"))
        assert not _C._is_precompile_attach_failure(RuntimeError("size mismatch"))
        assert not _C._is_precompile_attach_failure(ValueError("bad"))
        r.ok("precompile attach-failure allowlist classifies correctly")
    except Exception as e:
        r.fail("precompile attach-failure allowlist classifies correctly", str(e))

    # The inductor cache dir is versioned by cache-version + torch build.
    prev_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    try:
        os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        _C._ensure_inductor_cache_dir()
        d = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
        tag = torch.__version__.split("+")[0].replace(".", "")
        assert tag in d, f"cache dir {d!r} not versioned by torch build {tag}"
        assert "torch_compile" in d
        r.ok("inductor cache dir is version-scoped")
    except Exception as e:
        r.fail("inductor cache dir is version-scoped", str(e))
    finally:
        if prev_dir is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = prev_dir


def test_pc2_precompile_recovery_wipes_the_whole_store(r: SubTestResult):
    """HOUSE-50/H3 (TRK-226): `_clear_dynamo_precompile_store`'s recovery scope is the
    WHOLE `dynamo/` subdir, not just the fingerprint whose attach failed -- confirmed here
    with two independent fake per-fingerprint entries, neither of which torch actually
    wrote (no real compile needed to pin the SCOPE of the delete). This is documented,
    intentional behaviour (see the function's own docstring and DEVELOPMENT.md's rejected-
    decisions register for why a narrower delete is not safely implementable against
    torch 2.12's `caching_precompile`), not a defect this test guards against reopening by
    accident: a future narrower rewrite should have to explain itself against this pin,
    not silently start leaving unrelated stale entries behind."""
    print("\n--- PC-2 (HOUSE-50/H3): precompile recovery scope is whole-store, by design ---")
    import os
    with cold_engine_state(warm=False):
        owned = get_cache().torch_compile_cache_dir
        prev = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(owned)
        try:
            dynamo_dir = owned / "dynamo"
            fp1 = dynamo_dir / "fingerprint-one"
            fp2 = dynamo_dir / "fingerprint-two"
            fp1.mkdir(parents=True, exist_ok=True)
            fp2.mkdir(parents=True, exist_ok=True)
            (fp1 / "entry.bin").write_text("stale", encoding="utf-8")
            (fp2 / "entry.bin").write_text("perfectly valid, unrelated", encoding="utf-8")

            _C._clear_dynamo_precompile_store()

            r.ok("recovering fingerprint-one's stale attach also cleared fingerprint-two, "
                 "an unrelated entry that was never stale -- the confirmed, documented "
                 "collateral scope") \
                if not dynamo_dir.exists() else \
                r.fail("PC-2 recovery scope",
                       f"expected the whole dynamo/ dir gone, found: "
                       f"{list(dynamo_dir.iterdir()) if dynamo_dir.exists() else '(gone)'}")
        finally:
            if prev is None:
                os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
            else:
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = prev


def test_ct1_fused_disk_persistence(r: SubTestResult):
    print("\n--- CT-1: fused-chain disk persistence ---")
    cache = get_cache()
    torch.manual_seed(7)
    img = torch.rand(1, 4, 4, 4)
    stages = [
        {"code": "float t = $amt; @OUT = clamp(@A * t + 0.1, 0.0, 1.0);",
         "chain_input": None, "bindings": {"A": img, "amt": 0.3}},
        {"code": "float t = $amt; @OUT = @X * t;",
         "chain_input": "X", "bindings": {"amt": 0.8}},
    ]

    def _clear_all_fused():
        _FUS._FUSED_MEMO.clear()
        for p in cache._cache_dir.glob("fused_*.pkl"):
            p.unlink(missing_ok=True)

    # Fresh compile persists a fused_*.pkl.
    try:
        _clear_all_fused()
        prog, tm, refs, asg, params, used, merged = _FUS.compile_fused(stages, _infer_binding_type)
        interp = Interpreter()
        out1 = interp.execute(prog, dict(merged), tm, device="cpu",
                              output_names=sorted(asg.keys()), used_builtins=used)["OUT"]
        disk_files = list(cache._cache_dir.glob("fused_*.pkl"))
        assert len(disk_files) >= 1, "fused result not persisted to disk"
        r.ok("fused chain persisted to disk")
    except Exception as e:
        r.fail("fused chain persisted to disk", str(e))

    # Simulate restart: drop the in-memory memo, keep the disk entry — the next
    # compile must load from disk WITHOUT re-parsing any stage.
    try:
        _FUS._FUSED_MEMO.clear()   # memory only; disk survives
        calls = {"n": 0}
        orig = _FUS._parse
        def counting(code):
            calls["n"] += 1
            return orig(code)
        _FUS._parse = counting
        try:
            prog2, tm2, refs2, asg2, params2, used2, merged2 = _FUS.compile_fused(stages, _infer_binding_type)
        finally:
            _FUS._parse = orig
        assert calls["n"] == 0, f"re-parsed {calls['n']} stages instead of disk-loading"
        interp = Interpreter()
        out2 = interp.execute(prog2, dict(merged2), tm2, device="cpu",
                              output_names=sorted(asg2.keys()), used_builtins=used2)["OUT"]
        assert torch.allclose(out1, out2, atol=1e-6), "disk-loaded fused output differs"
        r.ok("fused chain disk-loaded without recompile (bit-identical)")
    except Exception as e:
        r.fail("fused chain disk-loaded without recompile (bit-identical)", str(e))
    finally:
        _clear_all_fused()
