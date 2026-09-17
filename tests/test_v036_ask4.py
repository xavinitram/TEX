"""ASK-4 — `img_width`/`img_height`, a binding's own width and height.

This file carries ASK-4's semantics rows (values on both tiers/devices, the uniform
=1 reading, the graph-tier capture, and the reserved-name/argument-type errors); the
other rows land as edits to the existing test files that own each property:
test_v017_phase2.py (`_looks_spatial` TST-3 heuristic), test_v021_phase1.py (fusion
equivalence), test_v024_phase1.py (ROI reach pin), test_codegen_optimizer.py
(equivalence corpus + the differently-shaped-binding tier story), stdlib_probe.py
(fuzzer/edge-matrix coverage), test_v017_phase1.py (fuzzer-grammar exclusion pin),
test_v019_phase1.py (precision="auto" decline), test_v023_phase1.py
(_NON_LOCAL_SINCE_V022 literal + footprint classification).

Each row runs on both devices and both tiers where meaningful — `_DEVICES` below
follows the same convention as test_v024_phase1.py / test_v035_ask13.py.
"""
from helpers import *
from failure_harness import run_tier, TierUnavailable

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]


def test_ask4_t1_width_height_values(r: SubTestResult):
    print("\n--- ASK-4 T1: img_width/img_height read the BINDING's own extent, "
          "not the cook grid ---")
    # @K is [1,17,33,3] (H=17, W=33) beside a differently-shaped @A [1,48,64,3]
    # (H=48, W=64) — the cook grid is @A's shape, so iw/ih must read @A's extent
    # while img_width(@K)/img_height(@K) read @K's own, independently.
    K = make_img(1, 17, 33, 3, seed=1)
    A = make_img(1, 48, 64, 3, seed=2)
    # @A is read (times 0.0) so it — not @K — anchors the cook grid (CF-6 consensus);
    # iw/ih must then read @A's extent while img_width(@K)/img_height(@K) still read
    # @K's own, independently of which binding anchors the grid.
    code = "@OUT = @A * 0.0 + vec4(img_width(@K), img_height(@K), iw, ih);"
    for dev in _DEVICES:
        for tier in ("interp", "codegen"):
            try:
                got = run_tier(code, {"K": K.to(dev), "A": A.to(dev)}, tier, device=dev)["OUT"]
                vals = got.reshape(-1, 4)[0].tolist()
                assert vals == [33.0, 17.0, 64.0, 48.0], f"got {vals}"
                r.ok(f"[{dev}/{tier}] img_width(K)=33, img_height(K)=17, iw=64, ih=48 "
                     f"(K's own extent independent of the cook grid)")
            except Exception as e:
                r.fail(f"ASK-4 T1 values [{dev}/{tier}]", f"{type(e).__name__}: {e}")


def test_ask4_t2_mask_binding(r: SubTestResult):
    print("\n--- ASK-4 T2: a MASK binding's own extent (rank-3 [B,H,W], no channel dim) ---")
    M = torch.rand(1, 9, 5)
    code = "@OUT = vec4(img_width(@M), img_height(@M), 0.0, 1.0);"
    for dev in _DEVICES:
        for tier in ("interp", "codegen"):
            try:
                got = run_tier(code, {"M": M.to(dev)}, tier, device=dev)["OUT"]
                vals = got.reshape(-1, 4)[0].tolist()
                assert vals[0] == 5.0 and vals[1] == 9.0, f"got {vals}"
                r.ok(f"[{dev}/{tier}] mask [1,9,5]: img_width=5, img_height=9")
            except Exception as e:
                r.fail(f"ASK-4 T2 mask [{dev}/{tier}]", f"{type(e).__name__}: {e}")


def test_ask4_t3_uniform_reads_one(r: SubTestResult):
    print("\n--- ASK-4 T3: a uniform (rank < 3) reads 1.0, not an error and not iw/ih ---")
    # img_width(0.5): a bare literal never carries a spatial dimension at all.
    # img_height(ix): ix is a broadcast VIEW, [1,1,W] — its own H is genuinely 1
    # (it never varies by row); img_width(ix) would correctly read W (not probed
    # here — the help text's caveat is about ix specifically, not a general rule).
    cases = [
        ("img_width(0.5) [bare literal]", "float w = img_width(0.5); @OUT = vec4(w,0.0,0.0,1.0);", 1.0),
        ("img_height(ix) [row-broadcast view]", "float h = img_height(ix); @OUT = vec4(h,0.0,0.0,1.0);", 1.0),
    ]
    for label, code, want in cases:
        for dev in _DEVICES:
            for tier in ("interp", "codegen"):
                try:
                    got = run_tier(code, {}, tier, device=dev)["OUT"]
                    val = got.reshape(-1, 4)[0, 0].item()
                    assert abs(val - want) < 1e-6, f"got {val}, want {want}"
                    r.ok(f"[{dev}/{tier}] {label}: reads {want}")
                except Exception as e:
                    r.fail(f"ASK-4 T3 uniform [{dev}/{tier}] {label}", f"{type(e).__name__}: {e}")


def test_ask4_t4_fp16_precision_exact_at_4095(r: SubTestResult):
    print("\n--- ASK-4 T4: a 4095-wide binding under precision='fp16' reads exactly "
          "4095, dtype fp32 (invariant #4: forced fp32, never the cook dtype) ---")
    img = torch.rand(1, 3, 4095, 3)
    code = "@OUT = vec4(img_width(@A), 0.0, 0.0, 1.0);"
    for dev in _DEVICES:
        for tier in ("interp", "codegen"):
            try:
                got = run_tier(code, {"A": img.to(dev)}, tier, device=dev, precision="fp16")["OUT"]
                val = got.reshape(-1, 4)[0, 0]
                assert val.item() == 4095.0, f"got {val.item()!r}, want exactly 4095.0"
                assert got.dtype == torch.float32, f"dtype {got.dtype} != fp32"
                r.ok(f"[{dev}/{tier}] precision='fp16': img_width reads exactly 4095.0, "
                     f"dtype {got.dtype}")
            except Exception as e:
                r.fail(f"ASK-4 T4 fp16-exact [{dev}/{tier}]", f"{type(e).__name__}: {e}")


def test_ask4_t5_graph_tier(r: SubTestResult):
    print("\n--- ASK-4 T5: the graph tier captures (not declined) and matches interp ---")
    if not _CUDA:
        r.ok("ASK-4 T5 graph tier SKIPPED (no CUDA)")
        return
    K = make_img(1, 17, 33, 3, seed=3).cuda()
    A = make_img(1, 48, 64, 3, seed=4).cuda()
    code = "@OUT = @A * 0.0 + vec4(img_width(@K), img_height(@K), iw, ih);"
    try:
        interp = run_tier(code, {"K": K, "A": A}, "interp", device="cuda")["OUT"]
        graph = run_tier(code, {"K": K.clone(), "A": A.clone()}, "graph", device="cuda")["OUT"]
        md = (interp.float() - graph.float()).abs().max().item()
        assert md < 1e-5, f"maxdiff {md}"
        r.ok(f"graph tier captured (not declined), matches interp (maxdiff {md:.1e})")
    except TierUnavailable as e:
        r.fail("ASK-4 T5 graph tier", f"unexpectedly declined: {e}")
    except Exception as e:
        r.fail("ASK-4 T5 graph tier", f"{type(e).__name__}: {e}")


def test_ask4_t6_e3011_reserved_names(r: SubTestResult):
    print("\n--- ASK-4 T6: img_width and img_height are reserved builtin names (E3011) ---")
    for name in ("img_width", "img_height"):
        try:
            raised = None
            try:
                check_code(f"float {name}(float x){{ return x; }}\n@OUT = vec4(0.0);")
            except Exception as e:
                raised = e
            assert raised is not None, f"redefining {name} as a user function did not raise"
            code = getattr(getattr(raised, "diagnostic", None), "code", None)
            assert code == "E3011", f"wrong error code: {code!r} (raised={raised!r})"
            r.ok(f"`float {name}(...)` user function is refused as E3011 (reserved builtin)")
        except Exception as e:
            r.fail(f"ASK-4 T6 E3011 {name}", f"{type(e).__name__}: {e}")


def test_ask4_t7_e5003_argument_type(r: SubTestResult):
    print("\n--- ASK-4 T7: a non-numeric (string) argument is refused as E5003 ---")
    for name in ("img_width", "img_height"):
        try:
            raised = None
            try:
                check_code(f'float w = {name}("hi"); @OUT = vec4(0.0);')
            except Exception as e:
                raised = e
            assert raised is not None, f"{name}(\"hi\") did not raise"
            code = getattr(getattr(raised, "diagnostic", None), "code", None)
            assert code == "E5003", f"wrong error code: {code!r} (raised={raised!r})"
            r.ok(f"{name}(\"hi\") (string argument) is refused as E5003")
        except Exception as e:
            r.fail(f"ASK-4 T7 E5003 {name}", f"{type(e).__name__}: {e}")
