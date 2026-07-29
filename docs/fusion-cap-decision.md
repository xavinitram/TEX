# FUS-cap — the 16-stage fusion cap, decided on a measurement

*Decision doc for doc 41 §3.5's spike. Verdict: **reject-with-the-number** — the cap stays at
16, and the reason it stays is not the reason the code gave.*

---

## 1. The question

`_MAX_FUSED_REGION_STAGES = 16`, and `_grow_region` returns `None` the moment a region
exceeds it (`tex_fusion.py:1106`). So a 17-stage linear graph gets **zero** fusion, not a
partial one: N=16 → 1 region, N=17 → 0, N=50 → 0. A cliff, not a taper.

Nothing had ever measured it. The comment justifying the cap offered two reasons, and by
2026-07 both were unusable: "CACHE-6 hasn't landed" (it landed in v0.27, and CACHE-7
generalized it in v0.32), and torch.compile trace/capture size (plausible, never measured,
and irrelevant on the interpreter tier where most cooks actually run).

Doc 41 §3.5 asks for one of three verdicts: **segment** a long chain into chained ≤16-stage
regions, **raise** the cap, or **reject with the number**.

## 2. What was measured

`benchmarks/fus_cap_bench.py`, sm_75 / RTX 2080 SUPER, torch 2.10+cu130, medians of 7–9 runs
with `torch.cuda.synchronize()` around every timed region. Three routes over the same linear
chain of pointwise stages, verified `torch.equal`-identical before timing:

* **fused** — one region of N stages (the planner refuses this past 16; the splicer does not,
  so it is measurable as the upper bound);
* **segmented** — `ceil(N/16)` fused regions chained, each re-based;
* **unfused** — N single-stage cooks chained. **This is what ships at N>16.**

### Time (ms, median)

| res | N | device | fused | segmented | unfused | seg/unf |
|-----|---|--------|-------|-----------|---------|---------|
| 512 | 17 | cpu | 16.02 | 17.95 | 18.35 | 1.02× |
| 512 | 50 | cpu | 51.64 | 48.88 | 57.04 | 1.17× |
| 512 | 17 | cuda | 2.93 | 2.91 | 3.55 | 1.22× |
| 512 | 50 | cuda | 8.41 | 8.29 | 10.37 | 1.25× |
| 1024 | 17 | cpu | 29.88 | 30.34 | **19.08** | **0.63×** |
| 1024 | 50 | cpu | 88.74 | 86.62 | **57.76** | **0.67×** |
| 1024 | 17 | cuda | 10.66 | 10.77 | 10.74 | 1.00× |
| 1024 | 50 | cuda | 31.64 | 31.51 | 31.50 | 1.00× |

Geomeans past the cap: **512 → 1.09× (cpu) / 1.23× (cuda)** for segmenting;
**1024 → 0.64–0.67× (cpu) / 1.00× (cuda)**.

### Peak memory (1024², N=50, CUDA)

```
fused-50        peak =  828.0 MiB
segmented-16    peak =  816.0 MiB
unfused-1       peak =   64.0 MiB      <- 12.9x less
```

## 3. The verdict, and the mechanism

**Reject with the number. The cap stays at 16.**

The 512² result looks like a win and is a red herring. Fusion's benefit in this engine is
amortizing **per-cook overhead** — 50 unfused cooks pay ~1.1 ms of Python each, which is why
the unfused row barely moves between 512² (57.04 ms) and 1024² (57.76 ms): at 512² it is
overhead-bound, not pixel-bound. Fusion's *cost* is that a spliced region materializes every
stage's full-res intermediate and holds them live for the whole cook, and that cost scales
with pixels. At 512² the saved overhead still wins; by 1024² it does not, and on CPU fusing
is **1.54× slower** than not fusing.

The memory number is the mechanism, and it is the finding worth keeping: **12.9× peak**. The
N=17 cliff is therefore *protective*. It is the thing standing between a user with a 50-node
comp and a 13× VRAM spike for no speed at all.

**Segmenting does not help, and the measurement is unambiguous about why.** It tracks the
fused numbers on both axes (0.67× CPU time, 816 MiB peak) rather than the unfused ones. Each
region still materializes its own stages' intermediates; chopping the chain into four pieces
reproduces the cost four times in smaller pieces instead of avoiding it. Segmenting would
have bought the 512²-scale overhead win at the price of the 1024²-scale memory and CPU loss —
a trade that gets worse as frames get bigger, which is the direction frames go.

**Raising the cap** makes every number above worse and is rejected on the same measurement.
**Lowering it** would forfeit the real small-resolution win (1.10×/1.23× at 512²) that the
cap currently allows below 16.

## 4. What this changes

* `tex_fusion.py`'s justifying comment is rewritten to the measured reasons (doc 41 §3.5
  requires this whichever verdict won). The spent "CACHE-6 hasn't landed" clause is gone and
  the unmeasured trace-size claim is replaced by the peak-memory numbers.
* No code change. The cap, the early-out in `_grow_region`, and the cliff all stay exactly
  as they are — now for a reason a reader can check.
* The `>7-taps` / `MAX_OUTPUTS` rejection chain that segmenting would have reopened **stays
  closed**: nothing here asks to reopen it.
* Doc 41 §3.5's own alternative reading is the one that survives: *"the cliff is fine because
  CACHE-9 owns big comps"* — PM-8's split. A 50-stage comp is a caching problem
  (CACHE-7 checkpoints, CACHE-9 region recook), not a fusion problem.

## 5. The reopen gate

One measurement reopens this: **fusion that does not materialize per-stage intermediates.**
Every number above is a consequence of the tree-walking interpreter evaluating each spliced
statement into its own full-res tensor. A backend that fuses at the *kernel* level — GRAPH-2's
territory, or a working Triton path — changes the cost model completely, and the cap should
be re-measured the day one exists rather than reasoned about.

Secondary, cheaper reopen: if per-cook overhead ever falls far enough that the unfused route
stops being overhead-bound at 512², the small-resolution win disappears too and the cap
becomes irrelevant rather than protective.

*Numbers are sm_75 (RTX 2080 SUPER), the standing hardware caveat. Raw rows:
`benchmarks/fus_cap_bench.py --save`.*
