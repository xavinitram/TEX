"""v0.31 — NOISE-TIER: the cold frame must render what every later frame renders.

    The FIRST cook of a process and the Nth cook of the identical program with
    identical inputs must be bit-identical.

`_TieredCache` (tex_runtime/noise.py) runs eager → jit.trace → torch.compile. Its cold
frame returned the EAGER result and cached a trace for every call after it, so call #1 of
a process ran a *different tier* than calls #2+. On CUDA those tiers are not bit-identical:
the TorchScript fuser reassociates the pointwise chain, and `simplex` at u*800 measured
5.3e-4 apart from eager over 56% of pixels — `x0 = x - X0` cancels ~800 against ~800 where
the fp32 quantum is ~6e-5, and the (0.5-d²)⁴ falloff amplifies what survives. Measured on
an RTX 2080 SUPER / torch 2.5.0+cu118, `simplex` and `fbm` each rendered 2 distinct images
across 6 cooks; `examples/film_grain.tex` was non-reproducible on first cook. CPU never
showed it (its two tiers agree exactly), which inverts ARCHITECTURE.md's marketed
"bitwise run-to-run deterministic on CUDA … the CPU is the honest caveat".

The defect predates v0.30.0 — it is not a release regression, and the fix does not move
the steady state: cook #1 now returns what cook #2+ always returned.

What each row pins:
  * cold_frame_parity      — NEVER-SEVER. A FRESH PROCESS cooks simplex 5x; all 5 must be
                             bit-equal. Fresh-process is the honest form: the effect is
                             per-process first-use, and nothing in `cold_engine_state`
                             resets the module-level tier caches. 5 cooks also crosses
                             `_COMPILE_AFTER_CALLS`, so on a box where Inductor actually
                             engages this row additionally answers whether the tier 2 → 3
                             promotion is bit-safe — a boundary that could NOT be measured
                             where this was fixed (no Triton for CUDA, and the CPU Inductor
                             build fails with CppCompileError), and which try_upgrade's
                             docstring records as knowingly open.
  * resolution_dance       — NEVER-SEVER. The SECOND half of the same defect, and the one
                             a fixed-shape test cannot see. A traced module is not one
                             numeric object: torch's profiling executor runs
                             `_jit_get_num_profiled_runs()` (=1) UNOPTIMIZED passes per new
                             (shape, dtype) and only then installs the fused plan. Measured:
                             a module traced at 24x32 and called at 48x64 returned the EAGER
                             value on that shape's first call (4d40b133) and the fused value
                             forever after (952dbe87). So pinning the cold frame alone fixes
                             the process-first cook and silently reopens the identical bug
                             at every new resolution — 512→1024→512 is ordinary ComfyUI use.
                             This row cooks 24x32 → 48x64 → back, twice, and demands one
                             image per resolution.
  * cold_equals_warm       — the same claim as a direct unit, for all three _TieredCache
                             users (simplex, fbm, worley) on every available device.
  * cold_path_shape        — CANARY. The cold path must ROUTE THROUGH the shared callable
                             rather than compute its own result. The previous fix here
                             tried to hold parity by making the eager body textually mirror
                             the traced body; identical source is exactly what the fuser
                             reassociates, so re-introducing a separate eager result is a
                             regression even when it looks bit-for-bit right.
"""
import inspect
import subprocess

from helpers import *

from TEX_Wrangle.tex_runtime.noise import _COMPILE_AFTER_CALLS as _COMPILE_AFTER

_CUDA = torch.cuda.is_available()

# Large coords on purpose: the divergence lives in the `x - X0` cancellation, so a
# small-coordinate probe would pass on a build that still has the bug.
_SIMPLEX_PROG = "@OUT = vec4(vec3(simplex(u*800.0, v*800.0)), 1.0);"
_N_COOKS = 5           # > _COMPILE_AFTER_CALLS (3), so the row spans the tier-3 promotion


_PARITY_CHILD = r'''
import os, sys
sys.path.insert(0, sys.argv[1])                       # .../custom_nodes
os.environ["TEX_CACHE_DIR"] = sys.argv[2]             # before any TEX import
import hashlib, struct, torch
from TEX_Wrangle import tex_engine

dev, n, prog = sys.argv[3], int(sys.argv[4]), sys.argv[5]
torch.manual_seed(5)
img = torch.rand(1, 24, 32, 4, device=dev)

def digest(t):
    # Whole-tensor and numpy-free: .tolist() gives Python doubles, struct.pack("f")
    # narrows back to the exact fp32 bits. tensor_fingerprint would NOT do here — it
    # samples 256 strided elements, so a localized divergence can slip through.
    v = t.detach().float().cpu().flatten().tolist()
    return hashlib.sha256(struct.pack(str(len(v)) + "f", *v)).hexdigest()[:16]

for _ in range(n):
    out = tex_engine.cook(prog, {"A": img}, device_mode=dev, precision="fp32").outputs["OUT"]
    print("DIGEST", digest(out))
'''


def _cook_digests_in_fresh_process(dev, cache_dir, n=_N_COOKS, prog=_SIMPLEX_PROG):
    """Return (digests, error). A fresh interpreter per call — the defect is first-use."""
    custom_nodes = str(Path(__file__).resolve().parents[2])
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PARITY_CHILD, custom_nodes, cache_dir, dev, str(n), prog],
            capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, "the child never finished"
    if proc.returncode != 0:
        return None, f"child exited {proc.returncode}: {(proc.stderr or '')[-400:]}"
    digests = [l.split()[1] for l in proc.stdout.splitlines() if l.startswith("DIGEST ")]
    if len(digests) != n:
        return None, f"child emitted {len(digests)} digests, expected {n}"
    return digests, None


def test_v031_noise_cold_frame_parity(r: SubTestResult):
    """NEVER-SEVER: N cooks of one program in one fresh process, all bit-identical."""
    print("\n--- v0.31 NOISE-TIER: the first cook must equal every later cook ---")
    devices = ["cpu"] + (["cuda"] if _CUDA else [])
    if not _CUDA:
        r.skip("cold-frame parity on CUDA",
               "no CUDA on this box — CPU's tiers agree bitwise, so only the CUDA row has teeth")

    for dev in devices:
        with cold_engine_state() as cold:
            digests, err = _cook_digests_in_fresh_process(dev, cold.dir)
        if err:
            r.fail(f"cold-frame parity ({dev})", err)
            continue
        if len(set(digests)) == 1:
            r.ok(f"simplex renders one image across {_N_COOKS} cooks on {dev} "
                 f"(cold frame == warm frame, {digests[0]})")
            # …and a SECOND independent process must land on the same image: the settled
            # fused value has to be a property of the program, not of one session.
            with cold_engine_state() as cold2:
                again, err2 = _cook_digests_in_fresh_process(dev, cold2.dir, n=2)
            if err2:
                r.fail(f"cross-process parity ({dev})", err2)
            elif again[0] == digests[0]:
                r.ok(f"a second process renders the identical image on {dev} ({again[0]})")
            else:
                r.fail(f"cross-process parity ({dev})",
                       f"process A rendered {digests[0]}, process B rendered {again[0]}")
            continue
        # Name WHICH boundary broke — tier 1→2 shows up at cook 2, tier 2→3 at cook 4.
        first_change = next(i for i in range(1, len(digests)) if digests[i] != digests[i - 1])
        boundary = ("tier 1 (eager) -> tier 2 (jit.trace)" if first_change == 1
                    else f"tier promotion at cook #{first_change + 1} "
                         f"(_COMPILE_AFTER_CALLS={_COMPILE_AFTER}) — the torch.compile tier")
        r.fail(f"cold-frame parity ({dev})",
               f"{len(set(digests))} distinct images across {_N_COOKS} cooks; first change "
               f"at cook #{first_change + 1} => {boundary}. digests={digests}")


_DANCE_CHILD = r'''
import os, sys
sys.path.insert(0, sys.argv[1])
os.environ["TEX_CACHE_DIR"] = sys.argv[2]
import hashlib, struct, torch
from TEX_Wrangle import tex_engine

dev, prog = sys.argv[3], sys.argv[4]

def digest(t):
    v = t.detach().float().cpu().flatten().tolist()
    return hashlib.sha256(struct.pack(str(len(v)) + "f", *v)).hexdigest()[:16]

# Two passes over the same two resolutions. Pass 2 proves a settled signature STAYS
# settled after the other resolution has run through the same traced module.
for _pass in range(2):
    for (h, w) in [(24, 32), (48, 64), (24, 32)]:
        torch.manual_seed(5)
        img = torch.rand(1, h, w, 4, device=dev)
        out = tex_engine.cook(prog, {"A": img}, device_mode=dev,
                              precision="fp32").outputs["OUT"]
        print("DIGEST", h, w, digest(out))
'''


def test_v031_noise_resolution_dance(r: SubTestResult):
    """NEVER-SEVER: each resolution must render exactly one image, across a size dance.

    A fixed-shape parity test cannot see this — the profiling window is per (shape, dtype),
    so it reopens at every new size even when the cold frame is correctly pinned.
    """
    print("\n--- v0.31 NOISE-TIER: 24x32 -> 48x64 -> back renders one image per size ---")
    if not _CUDA:
        r.skip("resolution dance", "no CUDA — the profiling-window gap is a CUDA fuser effect")
        return

    custom_nodes = str(Path(__file__).resolve().parents[2])
    with cold_engine_state() as cold:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _DANCE_CHILD, custom_nodes, cold.dir, "cuda",
                 _SIMPLEX_PROG], capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            r.fail("resolution dance", "the child never finished")
            return
    if proc.returncode != 0:
        r.fail("resolution dance", f"child exited {proc.returncode}: {(proc.stderr or '')[-400:]}")
        return

    by_res = {}
    for line in proc.stdout.splitlines():
        if line.startswith("DIGEST "):
            _, h, w, d = line.split()
            by_res.setdefault(f"{h}x{w}", []).append(d)
    if not by_res:
        r.fail("resolution dance", f"child emitted no digests: {(proc.stdout or '')[-300:]}")
        return

    bad = {res: ds for res, ds in by_res.items() if len(set(ds)) != 1}
    if bad:
        r.fail("resolution dance",
               "a resolution rendered more than one image — the profiling window reopened "
               "per shape: " + "; ".join(f"{res}={ds}" for res, ds in bad.items()))
    else:
        r.ok(f"one image per resolution across the dance "
             f"({', '.join(f'{res}={ds[0]}' for res, ds in sorted(by_res.items()))})")


def test_v031_noise_cold_equals_warm(r: SubTestResult):
    """The same claim as a direct unit, across all three _TieredCache users.

    In-process, so it clears the module-level tier caches by hand — `cold_engine_state`
    does not reach them (they are plain dicts on `noise._simplex_cache` &c., not engine
    state). Clearing is safe mid-suite: they are pure caches and rebuild on next touch.
    """
    print("\n--- v0.31 NOISE-TIER: cold frame == warm frame (simplex / fbm / worley) ---")
    from TEX_Wrangle.tex_runtime import noise

    devices = ["cpu"] + (["cuda"] if _CUDA else [])
    cases = [
        ("simplex", noise._simplex_cache, lambda x, y: noise._simplex2d(x, y)),
        ("fbm",     noise._fbm_cache,     lambda x, y: noise._fbm2d(x, y, 4)),
        ("worley",  noise._worley_cache,  lambda x, y: noise._worley2d(x, y)),
    ]
    bad = []
    for dev in devices:
        H, W = 24, 32
        yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
        # simplex/fbm want the cancelling large coords; worley's grid is unit-cell.
        for name, cache, fn in cases:
            scale = 8.0 if name == "worley" else 800.0
            x, y = (xx / W) * scale, (yy / H) * scale
            cache.cache.clear()
            cache._compile_attempted.clear()
            cache._call_count.clear()
            cold = fn(x, y)
            warm = fn(x, y)
            if not torch.equal(cold.float(), warm.float()):
                bad.append(f"{name}@{dev} maxdiff="
                           f"{float((cold.float() - warm.float()).abs().max()):.2e}")
    if bad:
        r.fail("cold==warm", "the cold frame ran a different tier: " + "; ".join(bad))
    else:
        r.ok(f"simplex/fbm/worley cold frame == warm frame on {devices}")


def test_v031_noise_stride_signature(r: SubTestResult):
    """The settle signature must key on STRIDES, not just shape+dtype.

    torch's profiling guards are stride-aware, so two tensors a shape-only signature
    calls identical can be different guard classes. Measured before this was keyed:
    after a contiguous 64x64 had settled, a TRANSPOSED 64x64 view (same shape, same
    dtype, stride (1,64)) returned ff73847f then b45a711d — the hole reopened on a
    tensor the cache believed was already settled.
    """
    print("\n--- v0.31 NOISE-TIER: the settle signature is stride-aware ---")
    if not _CUDA:
        r.skip("stride signature", "no CUDA — the profiling-window gap is a CUDA fuser effect")
        return
    from TEX_Wrangle.tex_runtime import noise

    for c in (noise._simplex_cache,):
        c.cache.clear(); c._settled.clear()
        c._compile_attempted.clear(); c._call_count.clear()

    n = 64
    base = (torch.arange(n * n, device="cuda", dtype=torch.float32).reshape(n, n)
            / (n * n)) * 800.0
    bad = []
    for label, t in (("contiguous", base.contiguous()),
                     ("transposed", base.t()),              # same shape+dtype, stride (1,64)
                     ("expanded", base[0:1, :].expand(n, n))):  # stride (0,1)
        outs = [noise._simplex2d(t, t) for _ in range(4)]
        if not all(torch.equal(outs[0], o) for o in outs[1:]):
            bad.append(f"{label} stride={t.stride()} "
                       f"maxdiff={float((outs[0] - outs[-1]).abs().max()):.2e}")
    if bad:
        r.fail("stride signature",
               "a strided layout reopened the profiling gap: " + "; ".join(bad))
    else:
        r.ok("contiguous / transposed / expanded layouts each settle independently")


def test_v031_noise_cold_path_shape(r: SubTestResult):
    """CANARY: the cold path must ROUTE through the shared callable, not recompute.

    Textual mirroring of the eager body against the traced body is what this replaced,
    and it cannot hold — the fuser reassociates the very source being mirrored. So a
    re-introduced standalone eager result is a regression by shape, not by value.
    """
    print("\n--- v0.31 NOISE-TIER: every call routes through _TieredCache.call ---")
    from TEX_Wrangle.tex_runtime import noise

    missing = [name for name, fn in (("_simplex2d", noise._simplex2d),
                                     ("_fbm2d", noise._fbm2d),
                                     ("_worley2d", noise._worley2d))
               if "_cache.call(" not in inspect.getsource(fn)]
    if missing:
        r.fail("cold-path shape",
               f"{', '.join(missing)}: no longer routed through _TieredCache.call — the "
               f"cold frame and each new (shape, dtype) can diverge from the settled tier")
    else:
        r.ok("_simplex2d / _fbm2d / _worley2d all route every call through _TieredCache.call")

    # A failed trace must fall back to eager, not propagate the False sentinel store()
    # writes — a raising noise fn would take down every cook.
    def _boom():
        raise RuntimeError("trace unavailable")
    probe = noise._TieredCache("probe-trace-failure")
    got = probe.call("k", (torch.ones(3), torch.ones(3)), device=torch.device("cpu"),
                     trace_fn=_boom, compile_fn=_boom, eager_fn=lambda a, b: a + b)
    if probe.cache.get("k") is False and torch.equal(got, torch.full((3,), 2.0)):
        r.ok("a failed trace falls back to the eager body instead of raising")
    else:
        r.fail("trace-failure fallback",
               f"expected the False sentinel and an eager result, got "
               f"cache={probe.cache.get('k')!r} result={got!r}")

    # A signature that never settles must demote to eager — never spin, and never serve a
    # value that depends on the call index.
    calls = {"n": 0}
    def _never_settles(a):
        calls["n"] += 1
        return a * float(calls["n"])
    probe2 = noise._TieredCache("probe-no-settle")
    probe2.cache["k2"] = _never_settles
    out = probe2._settle("k2", _never_settles, lambda a: a * -1.0, (torch.ones(3),))
    if probe2.cache["k2"] is False and torch.equal(out, torch.full((3,), -1.0)):
        r.ok(f"a signature that will not settle in {noise._SETTLE_MAX_RUNS} runs demotes "
             f"the key to eager")
    else:
        r.fail("settle non-convergence",
               f"expected a permanent eager demotion, got "
               f"cache={probe2.cache['k2']!r} result={out!r}")

    # _bitwise_same must call NaN equal to itself (so a NaN-producing program can settle)
    # and +0.0 unequal to -0.0 (reproducibility is a claim about bits). It must also
    # survive a 0-DIM tensor: `Tensor.view(dtype)` rejects those outright, and scalar
    # coordinates are ordinary TEX — `simplex(2.0, 3.0)` produces exactly that shape.
    nan = torch.tensor([float("nan")])
    checks = [(noise._bitwise_same(nan, nan.clone()), True, "NaN == NaN"),
              (noise._bitwise_same(torch.tensor([0.0]), torch.tensor([-0.0])), False,
               "+0.0 != -0.0"),
              (noise._bitwise_same(torch.tensor(1.5), torch.tensor(1.5)), True, "0-dim =="),
              (noise._bitwise_same(torch.tensor(1.5), torch.tensor(2.5)), False, "0-dim !="),
              (noise._bitwise_same(torch.zeros(0), torch.zeros(0)), True, "empty ==")]
    wrong = [lbl for got_, want, lbl in checks if got_ is not want]
    r.ok("_bitwise_same: NaN settles, signed zero does not, 0-dim and empty are safe") \
        if not wrong else \
        r.fail("_bitwise_same", "wrong verdict for " + ", ".join(wrong))

    # End-to-end: a scalar-coordinate noise call must still cook. This is the shape that
    # broke seven example programs when _bitwise_same viewed a 0-dim tensor as bytes.
    from TEX_Wrangle import tex_engine
    devices = ["cpu"] + (["cuda"] if _CUDA else [])
    broke = []
    for dev in devices:
        img = torch.rand(1, 16, 16, 4, device=dev)
        for fn_call in ("simplex(2.0, 3.0)", "fbm(2.0, 3.0, 4)", "worley_f1(2.0, 3.0)"):
            try:
                tex_engine.cook(f"float n = {fn_call};\n@OUT = vec4(vec3(n), 1.0);",
                                {"A": img}, device_mode=dev)
            except Exception as e:
                broke.append(f"{fn_call}@{dev}: {type(e).__name__} {str(e)[:80]}")
    if broke:
        r.fail("scalar coords", "a 0-dim noise call failed to cook: " + "; ".join(broke))
    else:
        r.ok(f"scalar-coordinate simplex/fbm/worley_f1 cook on {devices}")
