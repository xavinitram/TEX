"""
v0.18.0 memory management (MEM-2 pool trim, MEM-3 fp16 estimator, MEM-4 per-device budget).
"""
from helpers import *
from TEX_Wrangle import tex_memory as MEM


def test_mem2_pool_trim_gating(r: SubTestResult):
    print("\n--- MEM-2: downshift-gated reserved-pool trim (B2) ---")
    import types
    import os
    fails = []
    try:
        MEM.trim_reserved_pool("cpu", 512 * 512)  # CPU is always a no-op
    except Exception as e:
        fails.append(f"cpu path raised: {e}")

    # Count BOTH allocator queries and empty_cache — the B2 fix is that same-size steady
    # state does ZERO allocator queries (the +48% tax was the always-on query).
    q = {"reserved": 0, "empty": 0}
    real = {k: getattr(torch.cuda, k, None) for k in
            ("memory_reserved", "memory_allocated", "get_device_properties",
             "empty_cache", "current_device")}
    try:
        torch.cuda.current_device = lambda: 0
        torch.cuda.get_device_properties = lambda i: types.SimpleNamespace(total_memory=8 * 1024**3)
        torch.cuda.empty_cache = lambda: q.__setitem__("empty", q["empty"] + 1)

        def _set(reserved_gb, alloc_gb):
            def _mr(i=0):
                q["reserved"] += 1
                return int(reserved_gb * 1024**3)
            torch.cuda.memory_reserved = _mr
            torch.cuda.memory_allocated = lambda i=0: int(alloc_gb * 1024**3)

        MEM._last_trim_px.clear(); MEM._total_mem_cache.clear()
        _set(1.7, 0.2)  # 1.5 GB stranded (> the 1 GB threshold) — always over

        # (1) first cook @512² (prev=0, not a downshift) -> NO allocator query
        q["reserved"] = q["empty"] = 0
        MEM.trim_reserved_pool(torch.device("cuda:0"), 512 * 512)
        if q["reserved"] != 0 or q["empty"] != 0:
            fails.append("first (non-downshift) cook queried/trimmed")
        # (2) same-size steady state @512² -> STILL no allocator query (the B2 win)
        q["reserved"] = q["empty"] = 0
        MEM.trim_reserved_pool(torch.device("cuda:0"), 512 * 512)
        if q["reserved"] != 0:
            fails.append("same-size cook queried the allocator (B2 regression not fixed)")
        # (3) DOWNSHIFT @256² -> queries + trims (stranded > threshold)
        q["reserved"] = q["empty"] = 0
        MEM.trim_reserved_pool(torch.device("cuda:0"), 256 * 256)
        if q["reserved"] == 0:
            fails.append("downshift did not query the allocator")
        if q["empty"] != 1:
            fails.append("downshift over-threshold did not trim")
        # (4) downshift but UNDER threshold -> query, no trim
        MEM._last_trim_px[0] = 512 * 512
        _set(1.0, 0.5)  # 0.5 GB stranded < 1 GB
        q["empty"] = 0
        MEM.trim_reserved_pool(torch.device("cuda:0"), 256 * 256)
        if q["empty"] != 0:
            fails.append("trimmed under threshold")
        # (5) kill switch
        MEM._last_trim_px[0] = 512 * 512
        _set(1.7, 0.2)
        os.environ["TEX_NO_POOL_TRIM"] = "1"; q["empty"] = 0
        MEM.trim_reserved_pool(torch.device("cuda:0"), 256 * 256)
        os.environ.pop("TEX_NO_POOL_TRIM", None)
        if q["empty"] != 0:
            fails.append("TEX_NO_POOL_TRIM=1 did not disable the trim")
    finally:
        for k, v in real.items():
            if v is not None:
                setattr(torch.cuda, k, v)
        MEM._last_trim_px.clear(); MEM._total_mem_cache.clear()

    if fails:
        r.fail("MEM-2 pool trim", "; ".join(fails))
    else:
        r.ok("same-size steady state does 0 allocator queries; downshift queries + trims "
             "over threshold; under-threshold + kill-switch honored; CPU no-op")


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

        # skip-newest scoping (audit): with the CUDA entry inserted LAST (the GLOBAL
        # newest), a CPU sweep must still protect its OWN newest CPU entry — the old
        # global `newest` let it evict ("cpu", 3), defeating the isolation.
        MEM.free_tensor_caches()
        for s in range(4):
            SL._mip_cache[("cpu", s)] = ((1, 512, 512), torch.zeros(512, 512, 3),
                                         [torch.zeros(256, 256, 3)])
        SL._mip_cache[("cuda_newest",)] = ((1, 512, 512),
                                           torch.zeros(512, 512, 3, device="cuda"),
                                           [torch.zeros(256, 256, 3, device="cuda")])
        os.environ["TEX_CACHE_BUDGET_MB"] = "1"
        try:
            MEM.enforce_cache_budget("cpu")
        finally:
            if saved is None: os.environ.pop("TEX_CACHE_BUDGET_MB", None)
            else: os.environ["TEX_CACHE_BUDGET_MB"] = saved
        if ("cpu", 3) not in SL._mip_cache:
            fails.append("CPU sweep evicted its OWN newest entry (skip-newest not dev-scoped)")
        if ("cuda_newest",) not in SL._mip_cache:
            fails.append("CPU sweep evicted the CUDA entry")
    except Exception as e:
        fails.append(f"{type(e).__name__}: {e}")
    finally:
        MEM.free_tensor_caches()
    if fails:
        r.fail("MEM-4 per-device budget", "; ".join(fails))
    else:
        r.ok("CPU enforcement leaves CUDA entries untouched; evicts only CPU entries")
