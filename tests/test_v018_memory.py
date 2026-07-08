"""
v0.18.0 memory management (MEM-2 pool trim, MEM-3 fp16 estimator, MEM-4 per-device budget).
"""
from helpers import *
from TEX_Wrangle import tex_memory as MEM


def test_mem2_pool_trim_gating(r: SubTestResult):
    print("\n--- MEM-2: threshold-gated reserved-pool trim ---")
    fails = []
    # CPU is always a no-op (never raises).
    try:
        MEM.trim_reserved_pool("cpu")
    except Exception as e:
        fails.append(f"cpu path raised: {e}")

    # Gating logic — monkeypatch the CUDA memory stats so we don't allocate GBs.
    import types
    calls = {"n": 0}
    real = {k: getattr(torch.cuda, k, None) for k in
            ("memory_reserved", "memory_allocated", "get_device_properties",
             "empty_cache", "current_device")}
    try:
        torch.cuda.current_device = lambda: 0
        torch.cuda.get_device_properties = lambda i: types.SimpleNamespace(total_memory=8 * 1024**3)
        torch.cuda.empty_cache = lambda: calls.__setitem__("n", calls["n"] + 1)

        def _set(reserved_gb, alloc_gb):
            torch.cuda.memory_reserved = lambda i=0: int(reserved_gb * 1024**3)
            torch.cuda.memory_allocated = lambda i=0: int(alloc_gb * 1024**3)

        # under threshold (reserved-allocated = 0.5 GB < max(1GB,12.5%*8GB=1GB)) -> no trim
        calls["n"] = 0; _set(1.0, 0.5); MEM.trim_reserved_pool(torch.device("cuda:0"))
        if calls["n"] != 0:
            fails.append("trimmed under threshold (should be gated off)")
        # over threshold (reserved-allocated = 1.5 GB > 1 GB) -> trim fires
        calls["n"] = 0; _set(1.7, 0.2); MEM.trim_reserved_pool(torch.device("cuda:0"))
        if calls["n"] != 1:
            fails.append("did NOT trim when stranded > threshold")
        # kill switch
        import os
        calls["n"] = 0; os.environ["TEX_NO_POOL_TRIM"] = "1"
        _set(1.7, 0.2); MEM.trim_reserved_pool(torch.device("cuda:0"))
        os.environ.pop("TEX_NO_POOL_TRIM", None)
        if calls["n"] != 0:
            fails.append("TEX_NO_POOL_TRIM=1 did not disable the trim")
    finally:
        for k, v in real.items():
            if v is not None:
                setattr(torch.cuda, k, v)

    if fails:
        r.fail("MEM-2 pool trim", "; ".join(fails))
    else:
        r.ok("trim gated off under threshold, fires when stranded > max(1GB,12.5%), "
             "kill switch honored, CPU no-op")


def test_mem3_fp16_estimator(r: SubTestResult):
    print("\n--- MEM-3: fp16-aware peak estimator (2 bytes, not 4) ---")
    code = "vec3 c = @A.rgb; c = c * 1.2 + 0.05; @OUT = vec4(c, 1.0);"
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    spatial = (1, 1024, 1024)
    est4 = MEM.estimate_peak_bytes(prog, spatial, 4)
    est2 = MEM.estimate_peak_bytes(prog, spatial, 2)
    if est4 <= 0 or est2 != est4 // 2:
        r.fail("MEM-3 fp16 estimator",
               f"fp16 estimate {est2} != half of fp32 {est4} for a pointwise program")
    else:
        r.ok(f"fp16 peak estimate is exactly half fp32 ({est2} = {est4}//2)")


def test_mem4_per_device_budget(r: SubTestResult):
    print("\n--- MEM-4: per-device cache-budget accounting ---")
    from TEX_Wrangle.tex_runtime import stdlib as SL
    import os
    fails = []
    if not torch.cuda.is_available():
        r.ok("MEM-4 cross-device (no GPU, SKIPPED)")
        return
    try:
        MEM.free_tensor_caches()
        # one CUDA mip entry + several CPU mip entries
        SL._mip_cache[("cuda", 0)] = ((1, 512, 512),
                                      torch.zeros(512, 512, 3, device="cuda"),
                                      [torch.zeros(256, 256, 3, device="cuda")])
        for s in range(4):
            SL._mip_cache[("cpu", s)] = ((1, 512, 512), torch.zeros(512, 512, 3),
                                         [torch.zeros(256, 256, 3)])
        saved = os.environ.get("TEX_CACHE_BUDGET_MB")
        os.environ["TEX_CACHE_BUDGET_MB"] = "1"
        try:
            MEM.enforce_cache_budget("cpu")   # CPU enforcement must NOT touch CUDA entry
        finally:
            if saved is None: os.environ.pop("TEX_CACHE_BUDGET_MB", None)
            else: os.environ["TEX_CACHE_BUDGET_MB"] = saved
        if ("cuda", 0) not in SL._mip_cache:
            fails.append("CPU enforcement evicted a CUDA-resident entry (MEM-4 broken)")
        cpu_left = sum(1 for k in SL._mip_cache if k[0] == "cpu")
        if cpu_left > 1:
            fails.append(f"CPU entries not evicted under CPU budget ({cpu_left} left)")
    except Exception as e:
        fails.append(f"{type(e).__name__}: {e}")
    finally:
        MEM.free_tensor_caches()
    if fails:
        r.fail("MEM-4 per-device budget", "; ".join(fails))
    else:
        r.ok("CPU enforcement leaves CUDA entries untouched; evicts only CPU entries")
