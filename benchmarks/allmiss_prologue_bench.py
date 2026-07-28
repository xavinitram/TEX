"""The ALL-MISS prologue cost of `cook_checkpointed` (CACHE-7), warm and fresh-process.

The docstring on `cook_checkpointed` used to assert that arming CACHE-7 on a cache that can
serve nothing was FREE. It is not, and P0-8 measured how much: this bench is what keeps that
claim honest release to release. `docs/effort-based-checkpoints.md` §13 quotes it.

    warm  : same process, a FRESH empty cache per sample, `cook_checkpointed` against
            `cook_stage_list`, medians of 9.
    fresh : a new interpreter per sample, ONE cook each, timed AFTER the imports — the
            process-scope first-call costs (the CUDA context, the default-budget probe) are
            exactly what a memo inside the process cannot amortize away.

`cuts` is supplied explicitly. Left to PROF-1, placement returns `[]` until the profiler has
settled (147 cooks) and `cook_checkpointed` short-circuits to a plain cook before any prologue
runs — measuring nothing, and reporting 1.00x while doing it.

Run FOREGROUND. Nothing else may run while this measures.

    python benchmarks/allmiss_prologue_bench.py [--res 2048] [--n 12]
"""
import argparse, json, os, pathlib, statistics, subprocess, sys, tempfile, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# CACHE-0: never compile into the shipping package. Every sibling cache bench sets this, and
# here it is load-bearing for the MEASUREMENT too, not just hygiene: the fresh-process arm times
# the process-scope first-call costs, which include the program-cache probe — so without the
# redirect the numbers depend on whatever happens to be sitting in the repo's `.tex_cache`, and
# two developers cannot reproduce each other.
os.environ.setdefault("TEX_CACHE_DIR", str(pathlib.Path(tempfile.gettempdir()) / "tex_bench_cache"))

CHAIN = [
    "@OUT = vec4(@IN.rgb * vec3(1.06, 1.0, 0.97), 1.0);",
    "@OUT = vec4((@IN.rgb - vec3(0.5)) * 1.08 + vec3(0.5), 1.0);",
    "@OUT = vec4(@IN.rgb + 0.12 * gauss_blur(@IN, 4.0).rgb, 1.0);",
]


def stages(n):
    # Stage 0 reads the source on @IN; every later stage CHAINS on @IN - the linear shape
    # CACHE-6/7 cover (mirrors benchmarks/checkpoint_bench.py::build_stages).
    return [{"code": CHAIN[i % len(CHAIN)], "chain_input": ("IN" if i else None),
             "bindings": {}} for i in range(n)]


def cuts(n):
    return [i for i in (n // 3, 2 * n // 3) if 0 < i < n]


def _sync(dev):
    # PREFIX, not equality: `--device cuda:0` is accepted by argparse and passed straight to
    # `torch.rand(..., device=dev)`. An `== "cuda"` test skips the sync for it and reports
    # kernel-launch time, which is the one failure this project's bench discipline exists to
    # prevent (`benchmarks/checkpoint_bench.py::_sync` is the form being matched).
    if str(dev).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def one_cook(dev, n, st, armed):
    """THE timed operation, spelled once. Both arms and both scopes (warm and fresh) go through
    here — two spellings is how a bench drifts from what its docs claim it measures.

    `st` is passed in rather than built here so the caller can keep the source-tensor allocation
    outside its timer; a FRESH `ResultCache()` per call is load-bearing, not incidental — the
    all-miss case is what is being measured."""
    from TEX_Wrangle import tex_engine, tex_checkpoint, tex_results
    if armed:
        return tex_checkpoint.cook_checkpointed(st, result_cache=tex_results.ResultCache(),
                                                device=dev, precision="fp32", cuts=cuts(n))
    return tex_engine.cook_stage_list(st, device=dev, precision="fp32")


def warm(dev, res, n, reps):
    import torch
    st = stages(n)
    st[0]["bindings"] = {"IN": torch.rand(1, res, res, 3, device=dev)}

    def full():
        return one_cook(dev, n, st, False)

    def armed():
        return one_cook(dev, n, st, True)

    out = {}
    for name, fn in (("full", full), ("armed", armed)):
        for _ in range(2):
            fn()
        _sync(dev)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            _sync(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        out[name] = statistics.median(ts)
    out["ratio"] = round(out["armed"] / out["full"], 3)
    return out


def fresh(dev, res, n, reps):
    here = os.path.abspath(__file__)
    out = {}
    for name, flag in (("full", "0"), ("armed", "1")):
        ts = []
        for _ in range(reps):
            p = subprocess.run([sys.executable, here, "--child", flag, "--device", dev,
                                "--res", str(res), "--n", str(n)],
                               capture_output=True, text=True)
            ts.append(float(p.stdout.strip().splitlines()[-1]))
        out[name] = statistics.median(ts)
    out["ratio"] = round(out["armed"] / out["full"], 3)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--fresh-reps", type=int, default=5)
    a = ap.parse_args()
    if a.child is not None:
        # Imports and the source tensor are OUTSIDE the timer; the process-scope first-call
        # costs the memo exists to amortize (the CUDA context, the default-budget probe) are
        # inside it, because that is exactly what a fresh process still pays.
        import torch
        from TEX_Wrangle import tex_engine, tex_checkpoint, tex_results       # noqa: F401
        st = stages(a.n)
        st[0]["bindings"] = {"IN": torch.rand(1, a.res, a.res, 3, device=a.device)}
        t0 = time.perf_counter()
        one_cook(a.device, a.n, st, a.child == "1")     # ONE definition of the timed operation
        _sync(a.device)
        print((time.perf_counter() - t0) * 1e3)
        raise SystemExit(0)
    import torch
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = {}
    for d in devs:
        rows[d] = {"warm": warm(d, a.res, a.n, a.reps),
                   "fresh": fresh(d, a.res, a.n, a.fresh_reps)}
        print(d, json.dumps(rows[d]))
    print(json.dumps(rows, indent=2))
