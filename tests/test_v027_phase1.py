"""
v0.27 Phase 1 — "Big frames, placed well".

SCHED-3  cancellation + progress: a CancelToken checked at cook yield points (tier attempts,
         fp16/OOM re-cooks, per strip, per top-level interpreter statement) raising CookCancelled;
         on_progress(phase, frac). Pure value channel (like ENG-7 time_context) — never keyed.
SCHED-2  tex_scheduler.py: topological device placement (Viterbi DP for chains, exact enumeration
         for small DAGs, greedy fallback = correctness baseline), user pins, per-device budgets,
         boundary transfers, hysteresis / frozen-per-range. Distinct from PF-4 (where, not tier).
CACHE-5  the global cache governor: a CacheRegistry arbitrating the per-device VRAM/RAM pools
         (stdlib / graphs / frame cache) against ONE budget, preserving the graph-address safety
         (pin-skip / free_graphs_only, NEVER clear_graph_cache). Disk tiers keep separate caps.
ROI-5    halo-aware tiling: the blur/morphology class `is_tile_safe` refuses now tiles with a
         grown strip — differential oracle (halo-tiled == whole-frame), + the WDDM TDR time cap.
CACHE-6  fusion <-> caching: a stage-boundary tap (cache the fp32 handoff) + suffix splice (recook
         only stages k..N while a downstream knob is hot) — bit-exact vs the full fused cook.

Everything here ships OFF the default ComfyUI cook path (invariant #7): the scheduler + governor
+ tap are host-armed, halo tiling engages only under memory pressure, and cancel/progress are
None unless a host passes them. CPU-pinned; CUDA looped when present.
"""
import os
import tempfile

from helpers import *  # noqa: F401,F403  (SubTestResult, torch, make_img)
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib  # noqa: F401  (populates REGISTRY)
from TEX_Wrangle import tex_engine, tex_api, tex_fusion, tex_scheduler, tex_memory, tex_roi
from TEX_Wrangle.tex_runtime.host import CookCancelled
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_results import ResultCache, lineage_key
from TEX_Wrangle.tex_scheduler import SchedNode, plan_placement, graph_from_spec, Scheduler
from TEX_Wrangle.tex_compiler.types import TEXType

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]


class _TripToken:
    """A CancelToken that raises CookCancelled on its Nth check()."""
    def __init__(self, n):
        self.n = n
        self.calls = 0

    def check(self):
        self.calls += 1
        if self.calls >= self.n:
            raise CookCancelled("stop")


# ── SCHED-3 ───────────────────────────────────────────────────────────────────

def test_sched3_cancellation(r: SubTestResult):
    print("\n--- SCHED-3: cancellation + progress ---")
    code = "@OUT = vec4(@A.rgb * 1.2, 1.0);"
    img = make_img(1, 64, 64, 3, seed=1)

    # default path: no token/sink -> normal cook, byte-identical shape
    res = tex_engine.cook(code, {"A": img.clone()}, device_mode="cpu")
    if res.outputs["OUT"].shape != (1, 64, 64, 4):
        r.fail("SCHED-3 default", "default cook perturbed")
    else:
        r.ok("default cook unaffected (no token/sink)")

    # progress bookends
    events = []
    tex_engine.cook(code, {"A": img.clone()}, device_mode="cpu",
                    on_progress=lambda p, f: events.append((p, f)))
    r.ok("tier progress bookends") if ("tier", 0.0) in events and ("tier", 1.0) in events \
        else r.fail("SCHED-3 progress", f"missing tier bookends: {events}")

    # cancel at yield A -> CookCancelled out of cook()
    try:
        tex_engine.cook(code, {"A": img.clone()}, device_mode="cpu", cancel=_TripToken(1))
        r.fail("SCHED-3 cancel", "cook did not raise CookCancelled")
    except CookCancelled:
        r.ok("cook aborts at yield A (CookCancelled)")

    # per-statement cancel via tex_api.execute (3 statements, trip on 2nd)
    prog = tex_api.compile("float a = @A.r; float b = a*2.0; @OUT = vec4(b,b,b,1.0);",
                           {"A": TEXType.VEC3})
    tok = _TripToken(2)
    try:
        tex_api.execute(prog, {"A": img.clone()}, device="cpu", cancel=tok)
        r.fail("SCHED-3 per-stmt", "execute did not raise")
    except CookCancelled:
        r.ok("per-statement cancel") if tok.calls == 2 else r.fail("SCHED-3 per-stmt", f"calls={tok.calls}")

    # per-strip cancel + tiled==untiled bit-exact (drive run_tiled directly on CPU)
    interp = Interpreter()
    prog2 = tex_api.compile(code, {"A": TEXType.VEC3})
    sp = []
    try:
        tex_memory.run_tiled(interp, prog2.ast, {"A": img.clone()}, prog2.type_map, "cpu", 0,
                             ["OUT"], prog2.used_builtins, "fp32", 4,
                             cancel=_TripToken(3), on_progress=lambda p, f: sp.append((p, f)))
        r.fail("SCHED-3 per-strip", "run_tiled did not raise")
    except CookCancelled:
        r.ok("per-strip cancel after 1 strip") if sp == [("strip", 0.25)] \
            else r.fail("SCHED-3 per-strip", f"progress {sp}")
    out = tex_memory.run_tiled(interp, prog2.ast, {"A": img.clone()}, prog2.type_map, "cpu", 0,
                               ["OUT"], prog2.used_builtins, "fp32", 4)["OUT"]
    ref = interp.execute(prog2.ast, {"A": img.clone()}, prog2.type_map, device="cpu",
                         output_names=["OUT"], used_builtins=prog2.used_builtins, precision="fp32")["OUT"]
    r.ok("tiled == untiled bit-exact") if torch.equal(out, ref) else r.fail("SCHED-3 tiled", "perturbed pixels")


# ── SCHED-2 ───────────────────────────────────────────────────────────────────

def test_sched2_placement(r: SubTestResult):
    print("\n--- SCHED-2: device-placement scheduler ---")
    MB = 1 << 20
    DEVS = ["cpu", "cuda"]

    def chain(n):
        return [SchedNode(id=i, out_nbytes=MB, inputs=((i - 1,) if i > 0 else ())) for i in range(n)]

    def cook(cpu, cuda):
        return lambda node, dev: (cuda if str(dev).startswith("cuda") else cpu)

    def xfer(c):
        return lambda nb, s, d: (0.0 if str(s).startswith("cuda") == str(d).startswith("cuda") else c)

    p = plan_placement(chain(3), devices=DEVS, default_device="cpu",
                       cook_cost=cook(10, 1), transfer_cost=xfer(5))
    r.ok("DP chain -> all cuda (GPU-favorable)") if p.method == "dp" and all(
        v == "cuda" for v in p.devices.values()) and abs(p.est_cost_ms - 13.0) < 1e-6 \
        else r.fail("SCHED-2 dp", f"{p.devices} {p.est_cost_ms} {p.method}")

    p = plan_placement(chain(3), devices=DEVS, default_device="cpu",
                       cook_cost=cook(2, 1.5), transfer_cost=xfer(10))
    r.ok("boundary transfers -> all cpu (CPU-favorable)") if all(
        v == "cpu" for v in p.devices.values()) else r.fail("SCHED-2 boundary", f"{p.devices}")

    nodes = chain(3)
    nodes[1] = SchedNode(id=1, out_nbytes=MB, inputs=(0,), pin="cpu")
    p = plan_placement(nodes, devices=DEVS, default_device="cpu",
                       cook_cost=cook(10, 1), transfer_cost=xfer(2))
    r.ok("user pin forces device, DP works around") if p.devices[1] == "cpu" and p.devices[0] == "cuda" \
        else r.fail("SCHED-2 pin", f"{p.devices}")

    diamond = [SchedNode(id=0, out_nbytes=MB, inputs=()), SchedNode(id=1, out_nbytes=MB, inputs=(0,)),
               SchedNode(id=2, out_nbytes=MB, inputs=(0,)), SchedNode(id=3, out_nbytes=MB, inputs=(1, 2))]
    p = plan_placement(diamond, devices=DEVS, default_device="cpu",
                       cook_cost=cook(10, 1), transfer_cost=xfer(1))
    r.ok("small DAG solved by enumeration") if p.method == "enumerate" else r.fail("SCHED-2 enum", p.method)

    p = plan_placement(chain(3), devices=DEVS, default_device="cpu")
    r.ok("unmeasured -> greedy anchor (not piled)") if all(
        v == "cpu" for v in p.devices.values()) else r.fail("SCHED-2 greedy", f"{p.devices}")

    prev = tex_scheduler.Placement({0: "cpu", 1: "cpu", 2: "cpu"}, 0.0, "dp", "")
    p = plan_placement(chain(3), devices=DEVS, default_device="cpu",
                       cook_cost=cook(10, 9.8), transfer_cost=xfer(0),
                       previous=prev, hysteresis_ms=5.0)
    r.ok("hysteresis holds previous plan") if all(v == "cpu" for v in p.devices.values()) and "hysteresis" in p.note \
        else r.fail("SCHED-2 hysteresis", f"{p.devices} {p.note}")

    cyc = [SchedNode(id=0, inputs=(1,)), SchedNode(id=1, inputs=(0,))]
    p = plan_placement(cyc, devices=DEVS, default_device="cpu")
    r.ok("cycle -> greedy fallback, no crash") if p.method == "greedy" else r.fail("SCHED-2 cycle", p.method)

    # forest (2 disconnected chains -> 2 sources) must NOT be treated as linear: the Viterbi
    # reconstruction assumes one source and would KeyError on the second (bug-hunt MED).
    forest = [SchedNode(id=0, out_nbytes=MB, inputs=()), SchedNode(id=1, out_nbytes=MB, inputs=(0,)),
              SchedNode(id=2, out_nbytes=MB, inputs=()), SchedNode(id=3, out_nbytes=MB, inputs=(2,))]
    try:
        p = plan_placement(forest, devices=DEVS, default_device="cpu",
                           cook_cost=cook(10, 1), transfer_cost=xfer(1))
        r.ok(f"forest -> no crash, planned ({p.method})") if set(p.devices) == {0, 1, 2, 3} \
            else r.fail("SCHED-2 forest", f"{p.devices}")
    except Exception as e:
        r.fail("SCHED-2 forest", f"raised {type(e).__name__}: {e}")

    # greedy device-class match (bug-hunt): a 'cuda:0'-pinned source's consumer follows the GPU
    # CLASS (an exact-string `in` check would spuriously drop it to CPU with a needless D2H).
    from TEX_Wrangle.tex_scheduler import _greedy, _toposort
    gn = [SchedNode(id=0, inputs=(), pin="cuda:0"), SchedNode(id=1, inputs=(0,))]
    gdev = _greedy(_toposort(gn), {n.id: n for n in gn}, {0: ["cuda:0"], 1: ["cpu", "cuda"]}, "cpu")
    r.ok("greedy: cuda:0 pin -> consumer follows GPU class") if gdev[1] == "cuda" \
        else r.fail("SCHED-2 dev-class", f"{gdev}")
    # greedy any-GPU fan-in (resolve_device rule): a node with one cpu + one cuda input -> GPU.
    fn = [SchedNode(id=0, inputs=(), pin="cpu"), SchedNode(id=1, inputs=(), pin="cuda"),
          SchedNode(id=2, inputs=(0, 1))]
    fdev = _greedy(_toposort(fn), {n.id: n for n in fn},
                   {0: ["cpu"], 1: ["cuda"], 2: ["cpu", "cuda"]}, "cpu")
    r.ok("greedy: fan-in with any GPU input -> GPU") if fdev[2] == "cuda" \
        else r.fail("SCHED-2 any-gpu", f"{fdev}")

    spec = {"schema": 1, "dag": True,
            "stages": [{"code": "a"}, {"code": "b", "chain_inputs": {"X": [0, "OUT"]}, "device": "cpu"}],
            "terminal_chain_inputs": {"Y": [1, "OUT"]}, "terminal_device": "cuda"}
    g = {n.id: n for n in graph_from_spec(spec)}
    r.ok("graph_from_spec: DAG edges + device pins") if g[1].pin == "cpu" and g["terminal"].pin == "cuda" \
        and g["terminal"].inputs == (1,) else r.fail("SCHED-2 adapter", f"{g[1].pin} {g['terminal'].pin}")

    # cook_ms accessor: unmeasured key -> None (never 0)
    from TEX_Wrangle.tex_runtime import autotier
    r.ok("autotier.cook_ms unmeasured -> None") if autotier.cook_ms(
        autotier.make_key("nope_fp", "cuda", "fp32", (1, 256, 256))) is None \
        else r.fail("SCHED-2 cook_ms", "unmeasured should be None")


# ── CACHE-5 ───────────────────────────────────────────────────────────────────

def test_cache5_governor(r: SubTestResult):
    print("\n--- CACHE-5: global cache governor ---")
    reg = tex_memory.get_cache_registry()
    st = reg.stats("cpu")
    r.ok("default pools registered (stdlib, graphs)") if "stdlib" in st and "graphs" in st \
        else r.fail("CACHE-5 pools", f"{list(st)}")

    r.ok("graph pool CPU no-op (CUDA-only)") if tex_memory._graph_pool_bytes("cpu") == 0 \
        and tex_memory._evict_graphs("cpu", 1 << 30) == 0 else r.fail("CACHE-5 graphs", "cpu graph pool nonzero")

    d = tempfile.mkdtemp(prefix="tex_c5_")
    rc = ResultCache(budget_mb=1000, cache_dir=d)   # high self-budget: only the governor evicts
    for i in range(20):
        rc.put(lineage_key(program_fp=f"p{i}", device="cpu", precision="fp32"),
               make_img(1, 128, 128, 3, seed=i), canvas={"shape": [1, 128, 128, 3]})
    before = rc.governed_bytes("cpu")
    tex_memory.register_result_cache(rc, name="results_v027")
    try:
        r.ok("frame cache folded into governor") if reg.stats("cpu").get("results_v027") == before \
            else r.fail("CACHE-5 fold", "frame bytes not visible")

        budget = 5 * 128 * 128 * 3 * 4
        freed = reg.arbitrate("cpu", budget=budget)
        after = rc.governed_bytes("cpu")
        spilled = len([n for n in os.listdir(os.path.join(d, "results")) if n.endswith(".frame")]) \
            if os.path.exists(os.path.join(d, "results")) else 0
        r.ok(f"arbitrate evicts to budget (freed {freed})") if after <= budget else r.fail("CACHE-5 arb", f"{after}>{budget}")
        r.ok(f"evicted frames SPILLED not dropped ({spilled})") if spilled >= 14 else r.fail("CACHE-5 spill", f"{spilled}")

        k0 = lineage_key(program_fp="p0", device="cpu", precision="fp32")
        got = rc.get(k0)
        r.ok("spilled frame restores bit-exact") if got is not None and torch.equal(
            got, make_img(1, 128, 128, 3, seed=0)) else r.fail("CACHE-5 restore", "not bit-exact")

        r.ok("under-budget arbitrate is a no-op") if reg.arbitrate("cpu", budget=1 << 40) == 0 \
            else r.fail("CACHE-5 earlyout", "evicted under budget")
    finally:
        reg.unregister("results_v027")
    r.ok("unregister removes pool") if "results_v027" not in reg.stats("cpu") else r.fail("CACHE-5 unreg", "still present")

    # enforce_cache_budget default path intact (unchanged per-cook call)
    tex_memory.enforce_cache_budget("cpu")
    r.ok("enforce_cache_budget default path intact")


# ── ROI-5 ─────────────────────────────────────────────────────────────────────

def test_roi5_halo_tiling(r: SubTestResult):
    print("\n--- ROI-5: halo-aware tiling (differential oracle) ---")
    interp = Interpreter()

    prog = tex_api.compile("@OUT = vec4(gauss_blur(@A, 2.0).rgb, 1.0);", {"A": TEXType.VEC3})
    r.ok("is_tile_safe still refuses gauss_blur (unchanged)") if not tex_memory.is_tile_safe(prog.ast) \
        else r.fail("ROI-5 is_tile_safe", "should refuse blur")

    def oracle(code, H=96, W=80, dev="cpu", strips=(2, 3, 4, 5), exact=False):
        prog = tex_api.compile(code, {"A": TEXType.VEC3})
        plan = tex_roi.roi_plan(code, {})
        img = make_img(1, H, W, 3, seed=7)
        ref = interp.execute(prog.ast, {"A": img.clone()}, prog.type_map, device=dev,
                             output_names=["OUT"], used_builtins=prog.used_builtins, precision="fp32")["OUT"]
        for n in strips:
            out = tex_memory.run_tiled_halo(interp, prog.ast, {"A": img.clone()}, prog.type_map, dev, 0,
                                            ["OUT"], prog.used_builtins, "fp32", n, plan.narrow, plan.halo)["OUT"]
            md = (out.float() - ref.float()).abs().max().item()
            if not (torch.equal(out, ref) if exact else md < 1e-5):
                return False, f"n={n} maxdiff={md}"
        return True, f"halo={plan.halo}"

    for dev in _DEVICES:
        g, msg = oracle("@OUT = vec4(gauss_blur(@A, 2.0).rgb, 1.0);", dev=dev)
        r.ok(f"[{dev}] gauss_blur halo-tiled == whole-frame ({msg})") if g else r.fail("ROI-5 blur", f"[{dev}] {msg}")
    e, msg = oracle("@OUT = vec4(erode(@A, 2.0).rgb, 1.0);", exact=True)
    r.ok(f"erode halo-tiled bit-exact ({msg})") if e else r.fail("ROI-5 erode", msg)
    d, msg = oracle("@OUT = vec4(dilate(@A, 3.0).rgb, 1.0);", exact=True)
    r.ok(f"dilate halo-tiled bit-exact ({msg})") if d else r.fail("ROI-5 dilate", msg)
    g2, msg = oracle("@OUT = vec4(gauss_blur(@A, 4.0).rgb, 1.0);", H=128, W=96, strips=(2, 4, 6))
    r.ok(f"gauss_blur(4) larger halo == whole-frame ({msg})") if g2 else r.fail("ROI-5 blur4", msg)

    # planner cuda-only
    code = "@OUT = vec4(gauss_blur(@A, 2.0).rgb, 1.0);"
    prog = tex_api.compile(code, {"A": TEXType.VEC3})
    p = tex_engine._halo_tile_plan(prog.ast, code, {"A": make_img(1, 256, 256, 3)}, "cpu", 0, 4,
                                   None, None, "fp32")
    r.ok("halo_tile_plan CPU -> None (cuda-only)") if p is None else r.fail("ROI-5 plan", "cpu returned a plan")

    if _CUDA:
        dev = "cuda:%d" % torch.cuda.current_device()
        # Cheap gate (invariant #7): a SMALL cook returns None BEFORE the free-VRAM query, even
        # with a tiny free_hint — a program a small fraction of total VRAM can't need tiling.
        small = tex_engine._halo_tile_plan(prog.ast, code, {"A": make_img(1, 256, 256, 3)},
                                           dev, 0, 4, None, float(2 << 20), "fp32")
        r.ok("cheap gate: small cook -> None (no free query)") if small is None \
            else r.fail("ROI-5 gate", f"small cook should gate out, got {small}")
        # Force the gate open with a mocked-tiny total VRAM so a modest frame reads as large; the
        # tiny free_hint then drives the strip count. Restore the cache after.
        idx = torch.cuda.current_device()
        saved = tex_memory._total_mem_cache.get(idx)
        tex_memory._total_mem_cache[idx] = 32 << 20     # 32 MB total -> total//8 = 4 MB
        try:
            plan = tex_engine._halo_tile_plan(prog.ast, code, {"A": make_img(1, 512, 512, 3)},
                                              dev, 0, 4, None, float(2 << 20), "fp32")
        finally:
            if saved is None:
                tex_memory._total_mem_cache.pop(idx, None)
            else:
                tex_memory._total_mem_cache[idx] = saved
        r.ok(f"halo_tile_plan under pressure -> strips ({plan[0] if plan else None})") \
            if plan and plan[0] >= 2 and plan[2] > 0 else r.fail("ROI-5 cuda plan", f"{plan}")


# ── CACHE-6 ───────────────────────────────────────────────────────────────────

def _c6_stages(mul=1.5, off=0.1, g=0.9, src=None):
    src = make_img(1, 48, 40, 3, seed=3) if src is None else src
    return src, [
        {"code": "@OUT = vec4(@A.rgb * $mul, 1.0);", "chain_input": None, "bindings": {"A": src, "mul": mul}},
        {"code": "@OUT = vec4(@X.rgb + $off, 1.0);", "chain_input": "X", "bindings": {"off": off}},
        {"code": "@OUT = vec4(clamp(@Y.rgb * $g, 0.0, 1.0), 1.0);", "chain_input": "Y", "bindings": {"g": g}},
    ]


def test_cache6_fusion_recook(r: SubTestResult):
    print("\n--- CACHE-6: fusion <-> caching reconciliation ---")

    def md(a, b):
        return (a.float() - b.float()).abs().max().item()

    src, S = _c6_stages()
    r.ok("linear stage list detected") if tex_fusion.is_linear_stage_list(S) else r.fail("CACHE-6 linear", "not linear")
    suf = tex_fusion.suffix_stage_list(S, 2, make_img(1, 48, 40, 3))
    r.ok("suffix(k=2) rewrites head as source") if len(suf) == 1 and suf[0]["chain_input"] is None \
        and "Y" in suf[0]["bindings"] else r.fail("CACHE-6 suffix", f"{suf}")

    # A host arms CACHE-6 with the source's CONTENT-sensitive CACHE-1 identity via `upstream`.
    UP = ("srcA",)   # the source A's lineage key (same across cooks that reuse source A)

    # ORACLE: cached suffix-splice == full fused cook, CPU+CUDA, cut-points 1 and 2
    for dev in _DEVICES:
        src, S = _c6_stages()
        full = tex_engine.cook_stage_list(S, device=dev, precision="fp32")["OUT"]
        for k in (1, 2):
            rc = ResultCache(budget_mb=100, cache_dir=tempfile.mkdtemp(prefix="tex_c6_"))
            out = tex_engine.cook_fused_cached(S, k, rc, device=dev, precision="fp32", upstream=UP)["OUT"]
            r.ok(f"[{dev}] suffix-splice k={k} == full (maxdiff {md(out, full):.1e})") if md(out, full) < 1e-5 \
                else r.fail("CACHE-6 oracle", f"[{dev}] k={k} maxdiff {md(out, full)}")

    # boundary cache HIT on repeat; hot downstream param reuses boundary; upstream busts it
    src, S = _c6_stages(g=0.9)
    rc = ResultCache(budget_mb=100, cache_dir=tempfile.mkdtemp(prefix="tex_c6h_"))
    _ = tex_engine.cook_fused_cached(S, 2, rc, device="cpu", upstream=UP)["OUT"]
    hits0, miss0 = rc.hits, rc.misses
    tex_engine.cook_fused_cached(S, 2, rc, device="cpu", upstream=UP)
    r.ok("boundary cache HIT on repeat") if rc.hits > hits0 and rc.misses == miss0 \
        else r.fail("CACHE-6 hit", f"hits {hits0}->{rc.hits} miss {miss0}->{rc.misses}")

    _, S2 = _c6_stages(g=0.5, src=src)   # same source (same UP), new downstream g
    hits1, miss1 = rc.hits, rc.misses
    out = tex_engine.cook_fused_cached(S2, 2, rc, device="cpu", upstream=UP)["OUT"]
    full2 = tex_engine.cook_stage_list(S2, device="cpu")["OUT"]
    r.ok("hot downstream param reuses boundary (hit)") if rc.hits > hits1 and rc.misses == miss1 \
        else r.fail("CACHE-6 hot", f"miss {miss1}->{rc.misses}")
    r.ok("hot-param cook == full") if md(out, full2) < 1e-5 else r.fail("CACHE-6 hot-eq", f"{md(out, full2)}")

    _, S3 = _c6_stages(mul=2.0, src=src)   # upstream PREFIX param -> boundary key changes (same UP)
    miss2 = rc.misses
    out = tex_engine.cook_fused_cached(S3, 2, rc, device="cpu", upstream=UP)["OUT"]
    full3 = tex_engine.cook_stage_list(S3, device="cpu")["OUT"]
    r.ok("upstream param change busts boundary (miss)") if rc.misses > miss2 \
        else r.fail("CACHE-6 upstream", "no re-materialize")
    r.ok("upstream-changed cook == full") if md(out, full3) < 1e-5 else r.fail("CACHE-6 up-eq", f"{md(out, full3)}")

    # gate: no cache -> correct full cook
    src, S = _c6_stages()
    full = tex_engine.cook_stage_list(S, device="cpu")["OUT"]
    out = tex_engine.cook_fused_cached(S, 2, None, device="cpu", upstream=UP)["OUT"]
    r.ok("no-cache gate falls back to full cook") if md(out, full) < 1e-5 else r.fail("CACHE-6 gate", f"{md(out, full)}")

    # HIGH-bug regression (bug-hunt): a DIFFERENT source image with a DIFFERENT upstream key must
    # BUST the boundary (never a stale serve); and NO upstream must fall back to a full cook (a raw
    # data_ptr is unsafe — a reused/overwritten frame buffer keeps its address).
    A, Bimg = make_img(1, 32, 32, 3, seed=11), make_img(1, 32, 32, 3, seed=22)
    rc2 = ResultCache(budget_mb=100, cache_dir=tempfile.mkdtemp(prefix="tex_c6src_"))
    _, sA = _c6_stages(src=A)
    _, sB = _c6_stages(src=Bimg)
    tex_engine.cook_fused_cached(sA, 2, rc2, device="cpu", upstream=("srcA",))
    m0 = rc2.misses
    outB = tex_engine.cook_fused_cached(sB, 2, rc2, device="cpu", upstream=("srcB",))["OUT"]  # new key
    fullB = tex_engine.cook_stage_list(sB, device="cpu")["OUT"]
    r.ok("different source (upstream key) busts the boundary — no stale serve") \
        if rc2.misses > m0 and md(outB, fullB) < 1e-5 \
        else r.fail("CACHE-6 source-key", f"stale boundary served (maxdiff {md(outB, fullB)}, miss {m0}->{rc2.misses})")
    # in-place overwrite of the SAME source buffer without a new upstream key would be a stale
    # serve under a data_ptr key — the no-upstream default refuses to cache, so it can't happen.
    rc3 = ResultCache(budget_mb=100, cache_dir=tempfile.mkdtemp(prefix="tex_c6np_"))
    Csrc = make_img(1, 32, 32, 3, seed=33)
    _, sC = _c6_stages(src=Csrc)
    tex_engine.cook_fused_cached(sC, 2, rc3, device="cpu")           # no upstream -> not cached
    Csrc.copy_(make_img(1, 32, 32, 3, seed=44))                       # overwrite buffer in place
    outC = tex_engine.cook_fused_cached(sC, 2, rc3, device="cpu")["OUT"]
    fullC = tex_engine.cook_stage_list(sC, device="cpu")["OUT"]
    r.ok("no-upstream refuses to cache (in-place overwrite is safe)") \
        if rc3.hits == 0 and md(outC, fullC) < 1e-5 \
        else r.fail("CACHE-6 no-upstream", f"hits {rc3.hits} maxdiff {md(outC, fullC)}")

    # defensive (bug-hunt): a malformed cut (head stage lacks a chain_input to rebind) raises
    # FusionError from suffix_stage_list — cook_fused_cached wraps it to a full cook, not a crash.
    try:
        tex_fusion.suffix_stage_list([{"code": "a", "chain_input": None, "bindings": {}},
                                      {"code": "b", "chain_input": None, "bindings": {}}],
                                     1, make_img(1, 8, 8, 3))
        r.fail("CACHE-6 suffix-guard", "expected FusionError on a headless cut")
    except tex_fusion.FusionError:
        r.ok("suffix_stage_list raises FusionError on a headless cut (wrapped -> full cook)")

    fp1 = tex_fusion.prefix_fingerprint(S, 1, tex_engine._infer_binding_type)
    fp2 = tex_fusion.prefix_fingerprint(S, 2, tex_engine._infer_binding_type)
    r.ok("prefix_fingerprint stable + cut-distinct") if fp1 != fp2 and fp1 == tex_fusion.prefix_fingerprint(
        S, 1, tex_engine._infer_binding_type) else r.fail("CACHE-6 fp", "unstable or collides")


if __name__ == "__main__":
    _r = SubTestResult()
    for _t in (test_sched3_cancellation, test_sched2_placement, test_cache5_governor,
               test_roi5_halo_tiling, test_cache6_fusion_recook):
        _t(_r)
    _r.summary()
