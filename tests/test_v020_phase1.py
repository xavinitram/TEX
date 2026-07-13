"""
v0.20 Phase 1 — XPU transfer scheduling + tier honesty.

XPU-1..4: pinned egress (prepare_output / unwrap_latent write page-locked CPU
tensors when CUDA exists and the tensor clears _PIN_MIN_BYTES) + non-blocking
ingestion (a pinned CPU binding headed to CUDA rides an async DMA; stream
ordering keeps results bit-identical).

G-1: post-commit verification — a committed torch.compile artifact that
measures slower than _DEMOTE_RATIO x the interpreter is evicted + blacklisted.
"""
from helpers import *
import torch

from TEX_Wrangle.tex_marshalling import (
    prepare_output, unwrap_latent, to_fp32_if_int_image, _PIN_MIN_BYTES,
)
from TEX_Wrangle.tex_runtime.stdlib import LUMA_R, LUMA_G, LUMA_B

_CUDA = torch.cuda.is_available()


def test_xpu1_pinned_egress(r: SubTestResult):
    print("\n--- XPU-1: prepare_output pins large CPU outputs (CUDA present) ---")
    if not _CUDA:
        r.skip("XPU-1", "no CUDA: pinning is gated off — nothing to DMA to")
        return
    try:
        # IMAGE (4ch → drop alpha + clamp): 512²x4 fp32 = 4MB ≥ _PIN_MIN_BYTES
        raw = torch.rand(1, 512, 512, 4) * 1.4 - 0.2
        out = prepare_output(raw, "IMAGE")
        assert out.is_pinned(), "large CPU IMAGE egress should be page-locked"
        assert torch.equal(out, raw[..., :3].clamp(0, 1)), "pinned clamp changed values"

        # MASK (luma + clamp)
        mask = prepare_output(raw, "MASK")
        want = (LUMA_R * raw[..., 0] + LUMA_G * raw[..., 1] + LUMA_B * raw[..., 2]).clamp(0, 1)
        assert mask.is_pinned(), "large CPU MASK egress should be page-locked"
        assert torch.equal(mask, want), "pinned MASK changed values"

        # LATENT (permute materialization)
        lat_raw = torch.randn(1, 512, 512, 4)
        lat = prepare_output(lat_raw, "LATENT")
        assert lat.is_pinned(), "large CPU LATENT egress should be page-locked"
        assert torch.equal(lat, lat_raw.permute(0, 3, 1, 2).contiguous()), \
            "pinned LATENT permute changed values"

        # Size gate: 64²x4 fp32 = 64KB < 1MB → pageable (nothing worth hiding)
        small = prepare_output(torch.rand(1, 64, 64, 4), "IMAGE")
        assert not small.is_pinned(), "sub-threshold output should stay pageable"

        # CUDA-resident output: untouched (GPU stays GPU)
        g = prepare_output(torch.rand(1, 512, 512, 4, device="cuda"), "IMAGE")
        assert g.device.type == "cuda", "CUDA egress must stay on CUDA"
        r.ok("IMAGE/MASK/LATENT >=1MB pinned + value-identical; small stays pageable; GPU stays GPU")
    except Exception as e:
        r.fail("XPU-1 pinned egress", f"{type(e).__name__}: {e}")


def test_xpu2_unwrap_latent_pinned(r: SubTestResult):
    print("\n--- XPU-2: unwrap_latent keeps latent chains pinned ---")
    if not _CUDA:
        r.skip("XPU-2", "no CUDA")
        return
    try:
        samples = torch.randn(1, 4, 512, 512)  # BCHW, 4MB
        t, meta = unwrap_latent({"samples": samples, "noise_mask": "m"})
        assert t.is_pinned(), "unwrap materialization should be page-locked"
        assert torch.equal(t, samples.permute(0, 2, 3, 1).contiguous())
        assert meta == {"noise_mask": "m"}
        r.ok("BCHW-BHWC unwrap copy is pinned + value-identical (meta preserved)")
    except Exception as e:
        r.fail("XPU-2 unwrap pinned", f"{type(e).__name__}: {e}")


def test_xpu3_nonblocking_ingestion_bitexact(r: SubTestResult):
    print("\n--- XPU-3: pinned-to-CUDA ingestion is async-safe and bit-identical ---")
    if not _CUDA:
        r.skip("XPU-3", "no CUDA")
        return
    try:
        from TEX_Wrangle.tex_cache import TEXCache
        from TEX_Wrangle.tex_runtime.interpreter import Interpreter

        dev = torch.device("cuda", torch.cuda.current_device())
        pinned = torch.rand(1, 512, 512, 4).pin_memory()
        pageable = pinned.clone()
        assert pinned.is_pinned() and not pageable.is_pinned()

        # Unit: the single-source ingestion helper moves both to CUDA, same values
        a = to_fp32_if_int_image(pinned, device=dev)
        b = to_fp32_if_int_image(pageable, device=dev)
        assert a.device == dev and b.device == dev
        assert torch.equal(a, b), "pinned (non_blocking) vs pageable move diverged"

        # End-to-end: a full interpreter cook from a pinned vs pageable binding
        code = ("vec3 g = @A.rgb * vec3(0.9, 1.1, 1.0) + 0.02 * sin(u * 6.28318);\n"
                "@OUT = vec4(clamp(g.r,0.0,1.0), clamp(g.g,0.0,1.0), clamp(g.b,0.0,1.0), 1.0);")
        bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        prog, tm, refs, asg, params, used = TEXCache().compile_tex(code, bt)
        out_p = Interpreter().execute(prog, {"A": pinned}, tm, "cuda")
        out_g = Interpreter().execute(prog, {"A": pageable}, tm, "cuda")
        assert torch.equal(out_p, out_g), "cook from pinned input diverged (stream-order bug?)"
        r.ok("pinned-to-CUDA non_blocking ingestion bit-identical (unit + full cook)")
    except Exception as e:
        r.fail("XPU-3 non-blocking ingestion", f"{type(e).__name__}: {e}")


def test_xpu4_egress_ingest_roundtrip(r: SubTestResult):
    print("\n--- XPU-4: CPU cook - pinned egress - CUDA cook (mixed-device chain) ---")
    if not _CUDA:
        r.skip("XPU-4", "no CUDA")
        return
    try:
        from TEX_Wrangle.tex_cache import TEXCache
        from TEX_Wrangle.tex_runtime.interpreter import Interpreter

        code_a = "@OUT = vec4(@A.rgb * 0.8 + 0.1, 1.0);"
        code_b = "@OUT = vec4(sqrt(@A.rgb), 1.0);"
        prog_a, tm_a, *_ = TEXCache().compile_tex(
            code_a, {"A": TEXType.VEC4, "OUT": TEXType.VEC4})
        # Node B consumes A's 3-channel IMAGE egress DIRECTLY (no re-wrap that
        # would strip pinned-ness) — the honest ComfyUI handoff shape.
        prog_b, tm_b, *_ = TEXCache().compile_tex(
            code_b, {"A": TEXType.VEC3, "OUT": TEXType.VEC4})

        src = torch.rand(1, 512, 512, 4)
        # Node A cooks on CPU; its IMAGE egress should come out pinned
        raw_a = Interpreter().execute(prog_a, {"A": src}, tm_a, "cpu")
        img_a = prepare_output(raw_a, "IMAGE")
        assert img_a.device.type == "cpu" and img_a.is_pinned(), \
            "CPU cook egress should be pinned on a CUDA box"
        # Node B force-cooks on CUDA from A's still-pinned handoff
        out_mixed = Interpreter().execute(prog_b, {"A": img_a}, tm_b, "cuda")
        out_ref = Interpreter().execute(prog_b, {"A": img_a.clone()}, tm_b, "cuda")
        assert out_mixed.device.type == "cuda"
        assert torch.equal(out_mixed, out_ref), "mixed-device chain diverged"
        r.ok("CPU-to-CUDA chain: egress pinned, ingestion async, results bit-identical")
    except Exception as e:
        r.fail("XPU-4 roundtrip", f"{type(e).__name__}: {e}")


def test_f1_fused_compile_tiers(r: SubTestResult):
    print("\n--- F-1: fused chains route to the compile tiers (keyed by fused_fp) ---")
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        S = TEXWrangleNode.select_tier
        # Routing: a fused chain WITH a fingerprint takes the requested tier...
        assert S("torch_compile", "cpu", True, True) == "torch_compile"
        assert S("auto", "cuda:0", True, True) == "auto"
        # ...and falls to default without one (no key to memoize under).
        assert S("torch_compile", "cpu", True, False) == "default"
        assert S("auto", "cuda:0", True, False) == "default"
        # Single-node routing unchanged.
        assert S("auto", "cpu", False, False) == "auto"
        assert S("none", "cuda:0", True, True) == "default"

        # End-to-end: a chain-shaped program through execute_compiled under a
        # chain-style fingerprint stays equivalent to the interpreter.
        import TEX_Wrangle.tex_runtime.compiled as C
        from TEX_Wrangle.tex_cache import TEXCache
        code = ("vec3 s1 = @A.rgb * vec3(1.05, 0.98, 0.92) + 0.01;\n"
                "vec3 s2 = clamp(s1 * s1 * (3.0 - 2.0 * s1), 0.0, 1.0);\n"
                "@OUT = vec4(lerp(s1, s2, 0.5), 1.0);")
        bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        prog, tm, *_ = TEXCache().compile_tex(code, bt)
        fused_fp = "fused:" + TEXCache.fingerprint(code, bt)
        dev = "cuda" if _CUDA else "cpu"
        img = torch.rand(1, 128, 128, 4, device=dev)
        C.clear_compiled_cache()
        got = C.execute_compiled(prog, {"A": img}, tm, dev, fused_fp)
        ref = C._plain_execute(prog, {"A": img}, tm, dev)
        md = (got - ref).abs().max().item()
        assert md < 1e-5, f"fused-fp compiled cook diverged from interpreter: {md:.2e}"
        C.clear_compiled_cache()
        r.ok(f"fused routing to compile tiers + execute_compiled equivalent on {dev} (maxdiff {md:.1e})")
    except Exception as e:
        r.fail("F-1 fused compile tiers", f"{type(e).__name__}: {e}")


def test_a2_env_cache_scatter_cow(r: SubTestResult):
    print("\n--- A-2: codegen scatter COW protects the cross-cook env tensor cache ---")
    try:
        import TEX_Wrangle.tex_runtime.compiled as C
        from TEX_Wrangle.tex_cache import TEXCache

        C.clear_compiled_cache()
        # THE regression shape (workflow-verified): '@OUT = ix' emits a bare
        # alias; without the COW guard the scatter writes IN PLACE into the
        # process-global cached arange, corrupting every later cook at that W.
        code = "@OUT = ix;\n@OUT[1.0, 0.0] = 99.0;"
        bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        prog, tm, *_ = TEXCache().compile_tex(code, bt)
        fp = TEXCache.fingerprint(code, bt)
        img = torch.rand(1, 8, 8, 4)

        out1 = C._codegen_only_execute(prog, {"A": img}, tm, "cpu", fingerprint=fp)
        ref = C._plain_execute(prog, {"A": img}, tm, "cpu")
        assert torch.equal(out1, ref), "codegen scatter-through-alias diverged from interpreter"
        # The cached ix must be pristine: cook a pure '@OUT = ix' and compare.
        code2 = "@OUT = ix;"
        prog2, tm2, *_ = TEXCache().compile_tex(code2, bt)
        fp2 = TEXCache.fingerprint(code2, bt)
        out2 = C._codegen_only_execute(prog2, {"A": img}, tm2, "cpu", fingerprint=fp2)
        ref2 = C._plain_execute(prog2, {"A": img}, tm2, "cpu")  # interpreter = pristine oracle
        assert torch.equal(out2, ref2), \
            "cached ix corrupted by a previous cook's scatter (COW guard broken)"
        assert float(out2.reshape(-1).max()) < 99.0, \
            "scatter value leaked into the cached builtin grid"
        # Cache hygiene: env tensors are cached across cooks + cleared on reset.
        assert len(C._ENV_TENSOR_CACHE) > 0, "env tensor cache never populated"
        C.clear_compiled_cache()
        assert len(C._ENV_TENSOR_CACHE) == 0, "clear_compiled_cache must clear env tensors"
        r.ok("scatter into a builtin alias clones (interp-equal); cached ix pristine; cache clears")
    except Exception as e:
        r.fail("A-2 env-cache scatter COW", f"{type(e).__name__}: {e}")


def test_c1_gate_profiles_sane(r: SubTestResult):
    print("\n--- C-1: per-arch gate profiles are sane + Turing entry unchanged ---")
    try:
        from TEX_Wrangle.tex_runtime import arch_support as A
        assert set(A.VERIFIED_ARCHS) == set(A._GATE_PROFILES), \
            "VERIFIED_ARCHS must be exactly the measured-profile arches"
        assert A.CALIB_ARCH in A._GATE_PROFILES, "Turing calibration entry missing"
        turing = A._GATE_PROFILES[A.CALIB_ARCH]
        assert turing == {"graph_base_px_ceil": 512 * 512,
                          "graph_high_px_ceil": 1024 * 1024,
                          "min_fp16_px": 1024 * 1024}, \
            "Turing profile drifted from the original calibration"
        for arch, prof in A._GATE_PROFILES.items():
            assert prof["graph_base_px_ceil"] <= prof["graph_high_px_ceil"], \
                f"{arch}: base ceiling above high ceiling"
            assert prof["min_fp16_px"] >= 1024 * 1024, \
                f"{arch}: fp16 floor below the smallest measured win region"
        prof = A.gate_profile()
        assert set(prof) == set(turing), "gate_profile() must expose every gate key"
        r.ok(f"{len(A._GATE_PROFILES)} profiles sane; Turing pinned; gate_profile() complete")
    except Exception as e:
        r.fail("C-1 gate profiles", f"{type(e).__name__}: {e}")


def test_g2_verify_arming(r: SubTestResult):
    print("\n--- G-2: post-commit verification arms on real commits only ---")
    try:
        import TEX_Wrangle.tex_runtime.compiled as C
        from TEX_Wrangle.tex_cache import TEXCache

        C.clear_compiled_cache()
        # >=8 ops, codegen-supported, no stdlib graph-breaks → torch.compile path.
        code = ("@OUT = @A * 0.5 + @A * 0.25 + @A * 0.125 + @A * 0.0625 "
                "+ @A * 0.03125 + 0.01;")
        bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        prog, tm, *_ = TEXCache().compile_tex(code, bt)
        fp = TEXCache.fingerprint(code, bt)
        img = torch.rand(1, 8, 8, 4)
        cache_key = (fp, "cpu", "fp32")

        orig_compile = torch.compile
        torch.compile = lambda fn, **kw: fn   # identity: commit succeeds instantly
        try:
            C.execute_compiled(prog, {"A": img}, tm, "cpu", fp)
        finally:
            torch.compile = orig_compile
        if cache_key in C._compiled_cache:
            entry_backend = C._compiled_cache[cache_key][1]
            if entry_backend is not None:
                assert cache_key in C._verify_state, \
                    "real torch.compile commit must arm a verification window"
                assert C._verify_state[cache_key]["samples"] == [], \
                    "commit cook must not contribute a sample"
                r.ok("commit armed a px-scoped verification window (no commit-cook sample)")
            else:
                assert cache_key not in C._verify_state, \
                    "codegen-eager entry (backend None) must NOT arm verification"
                r.ok("codegen-eager entry did not arm verification (nothing to churn)")
        else:
            r.ok("[note] compile route gated out on this box — arming path not reachable")
        C.clear_compiled_cache()
    except Exception as e:
        r.fail("G-2 verify arming", f"{type(e).__name__}: {e}")


def test_g1_compile_demotion(r: SubTestResult):
    print("\n--- G-1: post-commit verification demotes a slower-than-interp artifact ---")
    try:
        import TEX_Wrangle.tex_runtime.compiled as C
        from TEX_Wrangle.tex_cache import TEXCache

        C.clear_compiled_cache()
        code = "@OUT = sin(@A) * 0.5 + cos(@A) * 0.25 + @A * @A * 0.1 + 0.01;"
        bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        prog, tm, refs, asg, params, used = TEXCache().compile_tex(code, bt)
        fp = TEXCache.fingerprint(code, bt)
        img = torch.rand(1, 4, 4, 4)
        cache_key = (fp, "cpu", "fp32")

        # Plant a fake committed artifact whose verification window is already
        # full of terrible samples (the guard-churn signature), then cook once:
        # the verdict must evict it and blacklist the fingerprint.
        fake = lambda program, bindings, type_map, device, lcc=0, out_names=None: \
            torch.zeros(1, 4, 4, 4)
        C._compiled_cache[cache_key] = (fake, "inductor")
        C._verify_state[cache_key] = {"px": 16, "samples": [1000.0] * C._VERIFY_COOKS}
        C.execute_compiled(prog, {"A": img}, tm, "cpu", fp)
        assert cache_key not in C._compiled_cache, "slow artifact not evicted"
        assert fp in C._compile_blacklist, "slow artifact not blacklisted"
        # And the next cook takes the honest interpreter path (values correct)
        out = C.execute_compiled(prog, {"A": img}, tm, "cpu", fp)
        ref = C._plain_execute(prog, {"A": img}, tm, "cpu")
        assert torch.equal(out, ref), "post-demotion cook diverged from interpreter"
        C.clear_compiled_cache()
        r.ok(f"artifact at 1000ms vs interp demoted (ratio gate {C._DEMOTE_RATIO}x) + blacklisted")
    except Exception as e:
        r.fail("G-1 demotion", f"{type(e).__name__}: {e}")
