# Host-path counts — design note (BENCH-2, v0.37.0 shipped / timing tier deferred)

*What an embedding host pays per interactive tick, measured as integers rather than as
milliseconds. The harness is `benchmarks/host_path_counts.py`; the gate is
`tests/test_bench2_counts.py`; the timing tier that this note DEFERS is §7.*

## 1. Why counts, and not times

`docs/roadmap.md` §10 item 3 records what a timing bench does on this class of machine when it
is pointed at a tree and asked to compare it with **itself**:

| null control (identical code, same box) | per-config geomean | worst row |
|---|---|---|
| v0.32, `eight_config_bench` at 512², same tree twice | 0.984 – 1.027 | 0.83 – 0.96 |
| v0.33, three-leg sequence, `cpu_off_warm` | **0.949** | 0.70 – 1.08 |
| v0.33, `cpu_on_cold` | 1.105 | 0.88 – 2.32 |

A byte-identical tree tripped the 0.95 stop-ship threshold against itself, and individual rows
spanned 0.70–2.32. The roadmap's own conclusion is that **a geomean below 0.95 opens an
investigation, it does not close one** — which is another way of saying wall-clock cannot be a
gate here. It is a release-sitting activity on a quiet box (§7).

Counts can be a gate. A structural trace of an embedding host driving a ten-stage comp through
TEX's ROI/results-cache pattern found the per-tick counts to be **exact integers across steady
ticks** (min == max over 29 ticks) while wall-clock over the same ticks varied 16–26 %. This
note's §4 reproduces that finding from TEX's own side, and §5 shows a same-tree null run of the
whole matrix moving **zero rows**.

So: counts gate in CI, times are measured once per release by hand. That split is the whole
design.

## 2. What is counted

`benchmarks/host_path_counts.py` drives **TEX's own `examples/host_demo.py::RoiComp`** — the
ten-stage `_COMP_STAGES` comp with a persistent per-stage canvas, a CACHE-2 `ResultCache` armed
by the host, and CACHE-1 lineage keys that carry the upstream chain. That is the pattern an
embedding host ports, so a count that moves here is a count that moves in the host.

Three passes per scenario, each from a freshly built comp:

| pass | instrument | what it sees |
|---|---|---|
| **A. API** | monkeypatch spies over a list of dotted targets, generalised from `tests/test_v031_anim_contract.py::_Spies`; installed and restored as a context manager | `tex_engine.cook/prepare/run`, `TEXCache.compile_ast/compile_tex/fingerprint`, `Lexer.tokenize`, `Parser.parse`, `TypeChecker.check/check_collect`, `tex_roi._fold_program/roi_plan/stage_halo/chain_windows`, `tex_results.lineage_key`, `ResultCache.get/put`, `tex_memory.run_roi`, `Interpreter._exec_stmt`, `profile.record`, `torch.cuda.synchronize` **attributed by caller file**, plus the per-cook fixed-pipeline rows of §6. Also `torch.cuda.memory_stats()` deltas (`allocation.all.allocated`, `num_device_alloc`, `num_alloc_retries`) and the host results cache's entry growth |
| **B. frames** | `sys.setprofile`, filtered to files under the package directory | TEX Python frames per `module:function`, with per-module subtotals and a top-N |
| **C. CUDA** | `torch.profiler` chrome trace | kernel launches and memcpy H2D/D2H/D2D, attributed to a tick by the **launch's CPU timestamp** (the device clock does not share an origin with the tick annotation; the `correlation` id bridges them) |

Each pass runs a **warm-up tick, reported separately**, then N steady ticks reported as
min / median / max / total with a `stable` flag (`min == max`). **Only stable rows are
gateable**, and `--compare` refuses to compare an unstable one — a row that disagrees with
itself says nothing about a tree.

Two measurement hazards were found by readings disagreeing with each other, and both are now
structural properties of the harness (`Scenario._seq`):

* **Global memos outlive the comp.** Each pass rebuilds the comp, so canvases and the results
  cache are genuinely fresh — but `tex_roi`'s fold memo keys on the param VALUES and is
  module-global, as is the program cache. Replaying the same slider values in pass B served
  every fold from the memo pass A had just filled: the frame pass reported 448 TEX frames for
  a terminal tick whose API pass had already counted a full re-lex and re-parse.
* **The same trap across devices.** Running both legs in one process, the CUDA leg reported
  `Lexer.tokenize = 0` per terminal tick where the CPU leg — same code, same host, same tick —
  reported 1, because the CPU leg had already folded those exact values.

A per-instance salt on every scrubbed value fixes both. The pan walk has its own bounded
ordinal (`_pan_seq`), for the same reason in the other direction: a repeated window is a cache
hit, not a pan, and reusing pass A's positions in pass C served the coordinate tensors from the
LAT-4 builtin LRU and reported 22 CUDA kernels for a pan tick that really costs 26.

## 3. The seven scenarios

| scenario | what a host is doing |
|---|---|
| `prewarm` | project load: `tex_api.prewarm` over the ten programs, each tick in its own cold cache dir |
| `source_edit` | the first whole-frame cook after a source edit of one stage |
| `terminal` | terminal-knob scrub: viewport window open, `dirty_from` = the last stage |
| `midgraph` | the same drag five nodes up, so the dirty suffix is five stages |
| `pan` | the window moves 16 px per tick, parameters constant |
| `all_dirty` | a source-side knob each tick, whole frame, so the results cache misses all ten |
| `lint` | `tex_api.check` with a one-character edit per tick — no cook at all |

## 4. The per-tick signature at head

Measured by `benchmarks/host_path_counts.py --res 1024 --window 512 --ticks 6`, CPU and CUDA
legs, on the dev laptop — first at `v0.37.0`, then re-derived after PERF-1 took the ROI
planner's re-lex out of the interactive tick (§6 item 1). **These are counts. Box noise is
irrelevant to them** — that is the entire point of the instrument, and the null run in §5 is
the proof. A cell reading `a..b~` is an unstable row (reported, never gated).

### 4.1 Rows that read the same on both devices

Every row in this table was measured identically on the CPU leg and the CUDA leg, which is what
makes it CI-gateable on a machine with no GPU.

| per tick | prewarm | source_edit | terminal | midgraph | pan | all_dirty | lint |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tex_engine.cook` | 0 | 7 | **1** | **5** | **1** | **10** | **0** |
| `tex_engine.prepare` / `run` | 0 | 7 | 1 | 5 | 1 | 10 | 0 |
| `TEXCache.compile_ast` | 10 | 0 | **0** | **0** | **0** | **0** | **0** |
| `TEXCache.compile_tex` | 10 | 7 | **1** | **5** | **1** | **10** | **0** |
| `Lexer.tokenize` | 10 | 1 | **0** | **0** | **0** | **0** | **1** |
| `Parser.parse` | 10 | 0 | **0** | **0** | **0** | **0** | **1** |
| `TypeChecker.check` | 20 | 1 | 0 | 0 | 0 | 0 | **0** |
| `TypeChecker.check_collect` | 0 | 0 | **0** | 0 | 0 | 0 | **1** |
| `tex_roi._fold_program` | 0 | 0 | **1** | **1** | **0** | **0** | **0** |
| `tex_roi.roi_plan` | 0 | 0 | **2** | **6** | **1** | **0** | **0** |
| `tex_roi.chain_windows` | 0 | 0 | **1** | **1** | **1** | **0** | **0** |
| `tex_results.lineage_key` | 0 | 10 | **11** | **15** | **11** | **10** | **0** |
| `TEXCache.fingerprint` | 30 | 14 | **2** | **10** | **2** | **20** | **0** |
| `tex_marshalling.param_only_names` | 30 | 14 | 2 | 10 | 2 | 20 | 0 |
| `ResultCache.get` | 0 | 10 | **1** | **5** | **1** | **10** | **0** |
| `ResultCache.put` | 0 | 7 | **1** | **5** | **1** | **10** | **0** |
| results-cache entries added | 0 | 7 | 1 | 5 | 1 | 10 | 0 |
| `tex_memory.run_roi` | 0 | 0 | **1** | **5** | **1** | **0** | **0** |
| `Interpreter._exec_stmt` | 0 | 9 | **2** | **6** | **2** | **12** | **0** |
| `_tile_plan` | 0 | 7 | 0 | 0 | 0 | 10 | 0 |
| `_halo_tile_plan` | 0 | 7 | 0 | 0 | 0 | 10 | 0 |
| `enforce_cache_budget` | 0 | 7 | 1 | 5 | 1 | 10 | 0 |
| `trim_reserved_pool` | 0 | 7 | 1 | 5 | 1 | 10 | 0 |
| `_disown_inputs` | 0 | 7 | 1 | 5 | 1 | 10 | 0 |
| `profile.record` (PROF-1 off) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **`torch.cuda.synchronize` from inside TEX** | 0 | 0 | **0** | **0** | **0** | **0** | **0** |

The **bold** cells are the ones `tests/test_bench2_counts.py` pins as exact integer literals
(measured there at 96²/48²/4 ticks, where every one of them reads the same).

`Lexer.tokenize` and `Parser.parse` read **1** on `terminal` and `midgraph` at `v0.37.0`; PERF-1
took both to 0 by memoizing the ROI fold's parse per SOURCE and handing each fold its own
`ast_nodes.clone_tree` copy. `tex_roi._fold_program` stays at 1: the fold is the part that
genuinely depends on the parameter values, so it is the parse that was cached and not the walk.

`source_edit`'s `Lexer.tokenize` is 1 on a cold leg and 0 on a leg whose program cache already
holds the edited sources; it is reported and not pinned for that reason.

### 4.2 Device rows (CUDA leg, 1024² with a 512² window, sm_120)

| per tick | prewarm | source_edit | terminal | midgraph | pan | all_dirty | lint |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUDA kernel launches | 0 | 0..86~ | **22** | **74** | **26** | **112** | 0 |
| memcpy D2H | 0 | 0 | **0** | **0** | **0** | **0** | 0 |
| memcpy D2H bytes | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| memcpy D2H at `v0.37.0`, before PERF-2 | 0 | 0..3~ | 0 | 2 | 0 | 3 | 0 |
| memcpy D2D | 0 | 4..10~ | 0 | 2 | 0 | 1 | 0 |
| memcpy H2D | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| allocator allocations | 0 | 72 | **18** | **56** | **22** | **86** | 0 |
| `num_device_alloc` | 0 | 7 | 0 | 0 | 0 | 0..10~ | 0 |
| `num_alloc_retries` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `torch.cuda.mem_get_info` | 10 | 4 | 0 | 0 | 0 | 7 | 0 |
| `_preflight_memory` | 0 | 7 | 1 | 5 | 1 | 10 | 0 |
| host's own end-of-frame sync | 0 | 1 | 1 | 1 | 1 | 1 | 0 |

Before PERF-2 every D2H was **4 bytes** (8 over two blur stages, 12 over three): they were
`.item()` drains, not image traffic — `gauss_blur` resolving its kernel radius. PERF-2 carries
the host reading of a literal / `$param` / folded constant on the 0-dim tensor it is minted
into, so the whole column is 0 and the `source_edit` row became stable at 0 with it. A sigma
genuinely computed on the device still drains, correctly: this comp has no such stage. There is
no H2D on any interactive tick either — the canvases stay resident.

### 4.3 TEX Python frames

| frames per tick | prewarm | source_edit | terminal | midgraph | pan | all_dirty | lint |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU leg | 13292 | 301..1351~ | 1552 | 2211 | 434 | 1760 | 1371 |
| CUDA leg | 13587 | 301..1505~ | 1560 | 2251 | 442 | 1968 | 1371 |
| CPU leg at `v0.37.0`, before PERF-1 | 13277 | 301..1341~ | 2744 | 3271 | 433 | 1747 | 1371 |

PERF-2 moved the frame rows UP by 1 to 13 per tick (one `_host_scalar` call per blur, one
`_tag_host_scalar` per scalar binding per cook) while taking the D2H column to zero. That is
the trade this note exists to make legible: a Python frame is host work that pipelines, a
`cudaStreamSynchronize` is host work that stops the pipeline, and the counts are not
interchangeable. §7's timing tier is where the trade is priced.

The terminal tick's 1551 frames break down (CPU leg, per-module subtotals), with the
`v0.37.0` column beside them so what moved is readable without a second document:

| module | frames | at `v0.37.0` | |
|---|---:|---:|---|
| `tex_compiler.ast_nodes` | 592 | 656 | 565 of these are `iter_child_nodes` — the analysis traversals, not the parse |
| `tex_compiler.parser` | 0 | 590 | } the re-lex and re-parse PERF-1 removed: |
| `tex_compiler.lexer` | 0 | 528 | } the source is parsed once and the fold copies it |
| `tex_compiler.optimizer` | 106 | 106 | the constant-fold, which still runs per VALUE |
| `tex_api` (`_ControlFlowLint._walk`) | 255 | 255 | the region-dependence walk, on the same memo miss |
| `tex_results` | 209 | 209 | the eleven lineage keys |
| `tex_roi` | 159 | 158 | |
| `tex_runtime.interpreter` | 61 | 61 | the actual cook |
| `examples.host_demo` | 46 | 46 | the host |
| everything else | 123 | 135 | |

At `v0.37.0`, **two thirds of an interactive tick was the compiler re-reading a program that
had not changed**; `pan` — the same cook with the parameters held constant, so the walk memo
hits — costs 433 frames, and that difference was the whole of §6 item 1. PERF-1 closed the
front-end half of it. What is left above `pan` is the part that is a function of the VALUES:
the fold (`optimizer`), the region-dependence walk (`tex_api`, plus most of the
`iter_child_nodes` frames) and the reach accumulation (`tex_roi`).

Counted across ALL modules rather than only the package — the honest denominator, because a
fix that moves work into `copy.deepcopy` or into torch would be invisible to the table above —
a terminal tick went **3267 → 1897** Python frames and a mid-graph tick **4109 → 2896**, while
`pan` stayed at 518. Nothing moved out of the package; it stopped being done.

## 5. The null run

The same tree, measured twice in sequence at 1024²/512²/6 ticks, both devices, all seven
scenarios, `--compare` between them:

```
  baseline: TEX 0.37.0 @ dfe7c3890c71
  current : TEX 0.37.0 @ dfe7c3890c71

  0 stable row(s) moved; 0 unstable row(s) differ (ignored).
```

Compare that with the timing null controls in §1, taken on the same laptop. The CPU pass of
the gate at 96²/48²/4 ticks runs in about 2.6 s.

## 6. Avoidable per tick — the candidate follow-ups

**None of these is fixed in this lane.** Each is listed with the file:line re-verified at head
and, crucially, **the counter that would show it fixed** — so a future lane has its acceptance
test before it starts, and cannot claim a win the instrument would not see.

1. **`_walk`'s memo keys on the param VALUES.** **The front-end half is FIXED (PERF-1); the
   fold and the region-dependence walk are not.** `_walk`'s key is
   `(sha256(code), _param_key(param_values), _string_wire_key(binding_types))`, and
   `_param_key` (`tex_lazy.py`) folds every scalar's fp32 bit pattern, so a slider misses the
   memo on every tick. That key is right — the walk's answer really can depend on a value (a
   `$sigma` in a halo radius; `mix(@A, @B, $k)` with `k = 0` folding `@B` away, which is what
   `fold_erased` exists to report) — so what PERF-1 changed is not the key but what a miss
   COSTS: the parse is memoized per source (`tex_roi._pristine_program`) and each fold works on
   an `ast_nodes.clone_tree` copy of it, so a miss re-folds a reused parse instead of re-lexing
   and re-parsing. `Lexer.tokenize` and `Parser.parse` read **0** on `terminal` and `midgraph`,
   `tex_roi._fold_program` still reads 1, and `frames.total` on `terminal` fell 2744 → 1551.
   *What is still avoidable, and shows fixed as:* `tex_roi._fold_program` **1 → 0** on
   `terminal` and `midgraph`, with `frames.total` falling from 1551 toward `pan`'s 433. That
   needs a value-INDEPENDENT key, which needs a proof that the walk cannot depend on a value —
   and `mix(@A, @B, $k)` shows that the whole five-tuple has no such proof in general. The
   region-dependence component alone plausibly does (`region_dependent`'s own docstring calls
   it "a pure function of (source, binding types)", and `region_dependent_cached` already
   memoizes it on exactly that), and it is worth roughly half of what a miss now costs.
2. **The host mints every chain lineage key every tick.** `RoiComp.cook` carries the
   whole-frame key forward for the clean prefix so stage *i+1* can link to it
   (`examples/host_demo.py`, the `_key` docstring explains why the link is load-bearing). That
   is 10 SHA-256 keys for one dirty stage. **This is host policy, not an engine cost** — the
   engine offers no "the prefix did not change" handle, which is what a follow-up would design.
   *Shows fixed as:* `tex_results.lineage_key` going **11 → 2** on `terminal`, **15 → 6** on
   `midgraph`, with `ResultCache.get/put` unchanged.
3. **`gauss_blur` reads `sigma` back with `.item()`.** **FIXED (PERF-2).** It was
   `sigma_val = max(sigma_t.item(), 0.0)` — one 4-byte D2H plus the stream synchronisation that
   copy implies, per blur stage per cook. The old comment claimed a constant sigma made it fire
   once; it fired once per COOK, because the value arrived as a tensor every time: a literal is
   minted into a 0-dim device tensor by the interpreter's literal cache and by codegen's hoisted
   constants, and a `$param` float by the interpreter's binding setup. Those mint sites now
   record the host reading ON the tensor (`stdlib._tag_host_scalar`) and the builtins take it
   from there (`stdlib._host_scalar`); a sigma genuinely computed on the device carries no tag
   and still reads back, which is correct and stays. The same reader serves `bilateral_filter`'s
   two sigmas, `convolve`'s `normalize` flag, `erode`/`dilate`'s radius (through `_to_float`) and
   `patch_dist`'s uniform-or-raise check. `cuda.memcpy_DtoH` is **2 → 0** on `midgraph`, **3 → 0**
   on `all_dirty` and stable-0 on `source_edit`, with `cuda.memcpy_DtoH_bytes` following and the
   kernel / allocation / engine-sync rows unmoved; the price is 1 to 13 more TEX Python frames
   per tick (§4.3).
   *What is still avoidable, and shows fixed as:* `sample_mip`'s LOD is read back from a tensor
   that has already been through `.clamp(0, max_level)`, so the tag is gone by the time it is
   read — hoisting that clamp to the host would take `cuda.memcpy_DtoH` to 0 on a comp that
   mip-samples, which this one does not, so no row here would move.
4. **PROF-1 costs four device syncs per sampled cook.** `tex_runtime/profile.py:408`
   (`measure._sync`), called at `:421` and `:433` — twice per `measure` block, and the engine
   arms a nested one. Measured with `--prof1 on` on a terminal tick: the **sampled** tick reads
   `torch.cuda.synchronize[engine]` = **4** and `profile.record` = **1**, and the unsampled
   ticks read 0 — the row is `~unstable` by construction, because `should_sample` measures
   every cook of an unseen key until it has three samples and then one in sixteen. This is why
   the profiler is disarmed by default (invariant #7) and why the gate asserts **0 engine-side
   syncs per interactive tick** with it off.
   *Shows fixed as:* `torch.cuda.synchronize[engine]` going **4 → 2** (or 0) on a sampled tick
   under `--prof1 on`, with `profile.record` unchanged at 1 — i.e. the same sample taken with
   fewer barriers, not fewer samples.
5. **The results cache grows by one entry per cook, forever, on interactive ticks.** The
   `results_cache.entries_added` row tracks `ResultCache.put` exactly on every scenario. **Host
   policy** — a scrub visits values it will never revisit, and nothing tells the cache so.
   *Shows fixed as:* `results_cache.entries_added` going to **0** on `terminal` and `pan` while
   `ResultCache.get`'s hit behaviour on a revisited value is unchanged.
6. **The per-cook fixed pipeline, paid ten times on a whole-frame cook.** Per cook, at head:
   two tile plans (`tex_tiling._tile_plan:38` and `_halo_tile_plan:142`, both re-exported into
   `tex_engine` at `tex_engine.py:101`), `enforce_cache_budget` (`tex_memory.py:327`),
   `trim_reserved_pool` (`tex_memory.py:766`) and `_disown_inputs` (`tex_buffers.py:158`) —
   called from `tex_engine.py:1318-1322` — plus `fingerprint` **twice**. On CUDA,
   `torch.cuda.mem_get_info` fires on **7 of the 10** stages of an `all_dirty` frame and on
   **none** of the interactive ticks.
   *Shows fixed as:* `TEXCache.fingerprint` going **20 → 10** on `all_dirty` (one fingerprint
   per cook instead of two), `_halo_tile_plan` going **10 → 0** on the stages whose pixel-local
   plan already answered, and `torch.cuda.mem_get_info` going **7 → 1** per whole frame.
7. **A window move rebuilds the coordinate builtins.** `pan` costs **26** CUDA kernels and
   **22** allocations against `terminal`'s 22 and 18 — exactly +4 and +4 for the same cook with
   a moved window. (This is also the LAT-4 LRU the harness had to defeat to measure the row
   honestly; see §2.)
   *Shows fixed as:* `cuda.kernels` on `pan` going **26 → 22** and `alloc.allocated`
   **22 → 18**.
8. **Two lexes for a never-seen program.** `TEXCache.fingerprint` (`tex_cache.py:344`) calls
   `param_only_names` (`tex_marshalling.py:830`), which tokenizes; the compile then tokenizes
   again. The counts track exactly — `param_only_names` equals `fingerprint` in every column of
   §4.1 — and `fingerprint` itself is called **twice per cook**.
   *Shows fixed as:* `TEXCache.fingerprint` and `param_only_names` both going **2 → 1** per
   cook, and the `prewarm` warm-up tick's `Lexer.tokenize` falling from **20 to 10** (one lex
   per never-seen program instead of two — the steady prewarm ticks already read 10 because
   `param_only_names` memoizes on the source after first sight).
9. **A full `TypeChecker.check` on every disk-cache reload.** `tex_cache.py:618` re-runs the
   checker to regenerate a `type_map` with valid `id()` keys after unpickling an already
   optimized program. `prewarm` reads **20** checks for ten programs.
   *Shows fixed as:* `TypeChecker.check` on `prewarm` going **20 → 10**.

## 7. The deferred tier: timing, at a release sitting

Timing is not deleted, it is **scheduled**. The existing benches stay the instrument; what this
note fixes is when and how they are run:

| bench | what it times |
|---|---|
| `benchmarks/eight_config_bench.py` | the release gate: 8 configs × the program corpus, `--save` / `--compare` |
| `benchmarks/roi_scrub_bench.py` | whole / fixed-window / panning / pan+param — the four ROI costs |
| `benchmarks/param_scrub_bench.py` | a slider sweep on one program |
| `benchmarks/region_recook_bench.py` | a partial re-cook against a whole one |
| `benchmarks/cookqueue_bench.py` | the SCHED-4 queue |

The sitting's rules, all of them already paid for by a false regression (`docs/roadmap.md` §10
item 3):

* **Three legs, not two**: base → after → base. The third leg distinguishes "the code changed
  something" from "that leg was disturbed", and costs one more run.
* **A same-tree null control in the same sequence**, because the threshold is inside this box's
  noise floor for at least one config.
* **Read geomeans, not rows.** Identical code produces rows at 0.70 and at 2.32.
* **Machine idle includes your own tooling** — background tasks, test suites and diagnostics.
  A timing run is foreground; if it is backgrounded, the only permitted concurrent activity is
  reading.
* **The `TEX_Wrangle` symlink gotcha** (`docs/bench1-v020-v028.md` §"The critical gotcha"): both
  bench scripts import the package as `TEX_Wrangle`, which is a link in `custom_nodes/` pointing
  at the *current* checkout. Which code gets measured is decided by **where `TEX_Wrangle`
  resolves, not by which worktree's bench you launched** — a naive `git worktree add` plus
  run-its-bench benchmarks the same build twice. Point a fresh `TEX_Wrangle` name at the other
  checkout instead. The counts harness has the same property and says so in `load_host_demo`.

## 8. What is NOT claimed

* The counts do not measure speed. A change that halves the kernel count can be slower, and the
  harness will happily report the halving. It answers "did the structure of the tick change",
  which is the question a CI gate can answer honestly on shared hardware.
* The device rows are device rows. Kernel counts, memcpys and allocations are properties of a
  fuser, a tier and an allocator; they are pinned for one device class and **skip** — not pass —
  without CUDA.
* `Interpreter._exec_stmt` is pinned on the CPU interpreter tier. A compiled tier bypasses it by
  design; that is a tier change, and a tier change is a CHANGELOG entry.
* The harness measures TEX through one host pattern. It is the pattern `examples/host_demo.py`
  ships and an embedding host ports, and it is the only one claimed.
