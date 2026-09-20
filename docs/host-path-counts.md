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

Measured at `v0.37.0` by `benchmarks/host_path_counts.py --res 1024 --window 512 --ticks 6`,
CPU and CUDA legs, on the dev laptop. **These are counts. Box noise is irrelevant to them** —
that is the entire point of the instrument, and the null run in §5 is the proof. A cell reading
`a..b~` is an unstable row (reported, never gated).

### 4.1 Rows that read the same on both devices

Every row in this table was measured identically on the CPU leg and the CUDA leg, which is what
makes it CI-gateable on a machine with no GPU.

| per tick | prewarm | source_edit | terminal | midgraph | pan | all_dirty | lint |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tex_engine.cook` | 0 | 7 | **1** | **5** | **1** | **10** | **0** |
| `tex_engine.prepare` / `run` | 0 | 7 | 1 | 5 | 1 | 10 | 0 |
| `TEXCache.compile_ast` | 10 | 0 | **0** | **0** | **0** | **0** | **0** |
| `TEXCache.compile_tex` | 10 | 7 | **1** | **5** | **1** | **10** | **0** |
| `Lexer.tokenize` | 10 | 1 | **1** | **1** | **0** | **0** | **1** |
| `Parser.parse` | 10 | 0 | **1** | **1** | **0** | **0** | **1** |
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

`source_edit`'s `Lexer.tokenize` is 1 on a cold leg and 0 on a leg whose program cache already
holds the edited sources; it is reported and not pinned for that reason.

### 4.2 Device rows (CUDA leg, 1024² with a 512² window, sm_120)

| per tick | prewarm | source_edit | terminal | midgraph | pan | all_dirty | lint |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUDA kernel launches | 0 | 0..86~ | **22** | **74** | **26** | **112** | 0 |
| memcpy D2H | 0 | 0..3~ | **0** | **2** | **0** | **3** | 0 |
| memcpy D2H bytes | 0 | 0..12~ | 0 | 8 | 0 | 12 | 0 |
| memcpy D2D | 0 | 4..10~ | 0 | 2 | 0 | 1 | 0 |
| memcpy H2D | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| allocator allocations | 0 | 72 | **18** | **56** | **22** | **86** | 0 |
| `num_device_alloc` | 0 | 7 | 0 | 0 | 0 | 0..10~ | 0 |
| `num_alloc_retries` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `torch.cuda.mem_get_info` | 10 | 4 | 0 | 0 | 0 | 7 | 0 |
| `_preflight_memory` | 0 | 7 | 1 | 5 | 1 | 10 | 0 |
| host's own end-of-frame sync | 0 | 1 | 1 | 1 | 1 | 1 | 0 |

Every D2H is **4 bytes** (8 bytes over two blur stages, 12 over three): these are `.item()`
drains, not image traffic. There is no H2D on any interactive tick — the canvases stay resident.

### 4.3 TEX Python frames

| frames per tick | prewarm | source_edit | terminal | midgraph | pan | all_dirty | lint |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU leg | 13277 | 301..1341~ | 2744 | 3271 | 433 | 1747 | 1371 |
| CUDA leg | 13572 | 301..1530~ | 2757 | 3336 | 446 | 2005 | 1371 |

The terminal tick's 2744 frames break down (CPU leg, per-module subtotals):

| module | frames | |
|---|---:|---|
| `tex_compiler.ast_nodes` | 656 | } |
| `tex_compiler.parser` | 590 | } **1880 of 2744 — 69 %** — one re-lex, |
| `tex_compiler.lexer` | 528 | } one re-parse and one constant-fold of |
| `tex_compiler.optimizer` | 106 | } the scrubbed stage's source |
| `tex_api` (`_ControlFlowLint._walk`) | 255 | the region-dependence walk, on the same memo miss |
| `tex_results` | 209 | the eleven lineage keys |
| `tex_roi` | 158 | |
| `tex_runtime.interpreter` | 61 | the actual cook |
| `examples.host_demo` | 46 | the host |
| everything else | 135 | |

**Two thirds of an interactive tick is the compiler re-reading a program that did not change.**
`pan` — the same cook with the parameters held constant — costs 433 frames. That difference is
the whole of §6 item 1.

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

1. **`_walk`'s memo keys on the param VALUES.** `tex_roi.py:660` builds the key as
   `(sha256(code), _param_key(param_values), _string_wire_key(binding_types))`, and
   `_param_key` (`tex_lazy.py:140`) folds every scalar's fp32 bit pattern. A slider therefore
   misses the memo on every tick and re-runs `_fold_program` (`tex_roi.py:583`) — a full re-lex,
   re-parse, constant-fold and, on the same miss, a `region_dependent` walk (`tex_roi.py:688`).
   *Shows fixed as:* `Lexer.tokenize`, `Parser.parse` and `tex_roi._fold_program` going
   **1 → 0** on `terminal` and `midgraph` (they are already 0 on `pan`, which is the control
   that proves the cost is the key and not the cook), and `frames.total` on `terminal` falling
   from 2744 toward `pan`'s 433.
2. **The host mints every chain lineage key every tick.** `RoiComp.cook` carries the
   whole-frame key forward for the clean prefix so stage *i+1* can link to it
   (`examples/host_demo.py`, the `_key` docstring explains why the link is load-bearing). That
   is 10 SHA-256 keys for one dirty stage. **This is host policy, not an engine cost** — the
   engine offers no "the prefix did not change" handle, which is what a follow-up would design.
   *Shows fixed as:* `tex_results.lineage_key` going **11 → 2** on `terminal`, **15 → 6** on
   `midgraph`, with `ResultCache.get/put` unchanged.
3. **`gauss_blur` reads `sigma` back with `.item()`.** `tex_runtime/stdlib.py:1470` —
   `sigma_val = max(sigma_t.item(), 0.0)`, one 4-byte D2H plus a stream sync per blur stage per
   cook. The comment there already says a constant sigma makes it fire once; it does not, because
   the value arrives as a tensor binding each cook.
   *Shows fixed as:* `cuda.memcpy_DtoH` going **2 → 0** on `midgraph` and **3 → 0** on
   `all_dirty`, with `cuda.memcpy_DtoH_bytes` following.
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
