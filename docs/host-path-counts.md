# Host-path counts — design note (BENCH-2, shipped v0.37.0 / timing tier sat v0.38.0)

*What an embedding host pays per interactive tick, measured as integers rather than as
milliseconds. The harness is `benchmarks/host_path_counts.py`; the gate is
`tests/test_bench2_counts.py`; the timing tier this note used to DEFER is §7, and it is no
longer deferred — the first sitting happened, and §7 records it.*

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

## 3. The seven scenarios, and the eighth

| scenario | what a host is doing |
|---|---|
| `prewarm` | project load: `tex_api.prewarm` over the ten programs, each tick in its own cold cache dir |
| `source_edit` | the first whole-frame cook after a source edit of one stage |
| `terminal` | terminal-knob scrub: viewport window open, `dirty_from` = the last stage |
| `midgraph` | the same drag five nodes up, so the dirty suffix is five stages |
| `pan` | the window moves 16 px per tick, parameters constant |
| `all_dirty` | a source-side knob each tick, whole frame, so the results cache misses all ten |
| `lint` | `tex_api.check` with a one-character edit per tick — no cook at all |
| `node_scrub` | the ComfyUI **node**: two `check_lazy_status` rounds then `execute`, one `$param` moving |

**Why the eighth is not a comp scenario (BENCH-3).** The seven above drive `tex_api` /
`tex_engine` directly, and `tex_engine.prepare` consults the lazy analysis only when its caller
passes `forgive_dead_refs`, which defaults to `False`. The one caller that passes it is
`tex_node.execute` (`forgive_dead_refs=bool(slot_entries)` — the ComfyUI lazy input pool), so
`tex_lazy.lazy_required_bindings` reads **0 per tick on all seven**, `all_dirty` included, which
enters `prepare` ten times a tick.

**That 0 is a property of this harness, not of every host.** A second embedding host calls
`tex_lazy.lazy_required_bindings` **directly** from its own planner, outside `prepare()`, to drop
image bindings a program cannot read at the current parameter values (reported 2026-09-21). PERF-4's
and PERF-8's fixes reach that host through that call, and nothing in the table below would have
shown it. Read a 0 here as *this harness does not reach the code*, never as *no host pays for it*.

The first-class host's own per-tick cost was therefore
invisible to this instrument: PERF-4 measured a slider tick at **2 lexes and 2 parses** before
its fix and **0** after (one lex and one parse in total, paid by the first tick), and no row
here moved either way. `node_scrub` drives what a user's slider drives — round 1 of
`check_lazy_status` with the wired scalar still `None`, round 2 once it has cooked (the T4-lite
round), then `execute`, whose E6003 gate is the analysis's third consumer — and its
device-independent rows, measured on both legs at 96²/48²/4 ticks and identical on each:

| per tick, `node_scrub` | | | |
|---|---:|---|---:|
| `lazy_required_bindings` | **3** | `TEXCache.compile_tex` / `fingerprint` | **1** / **1** |
| `Lexer.tokenize` / `Parser.parse` | **0** / **0** | `tex_engine.prepare` / `run` | **1** / **1** |
| `TEXCache.compile_ast` | **0** | `tex_engine.cook` | **0** |
| `Interpreter._exec_stmt` | **2** | every `tex_roi` / `tex_results` / `ResultCache` row | **0** |

`tex_engine.cook` reads 0 and `prepare`/`run` read 1 because the node calls the two halves
itself — it needs the plan between them for the Q-4 stage attribution — and the whole ROI /
results-cache tier reads 0 because ComfyUI has no viewport window to cook. That is the point
of the scenario: it is the other half of what an embedding host pays, not a second reading of
the half the comp already covers. `tests/test_bench2_counts.py` pins the rows above, so the
PERF-4 class is gated from now on.

**The ninth, named and not built.** No scenario here drives `tex_checkpoint.cook_checkpointed`.
The eight above reach the ROI, node and whole-frame shapes, and the checkpointed cook is reached by
none of them. An embedding host reports (2026-09-21) that on its tree the checkpointed cook is on
the **interactive** path, not the render path: its router sends the commonest interactive edit — a
linear fused chain with a settled cost table and a non-empty cut plan, asking for no window — to
`cook_checkpointed` for the whole frame, and that route also takes one `boundary_lineage_key` probe
per planned cut. `tex_engine.cook_fused_cached` and `cook_stage_list` sit on that host's render
route instead. So a checkpoint-serve tick is an interactive shape this instrument cannot see, and
saying so is the honest form: it is a gap in the harness, not a claim that the route costs nothing.
The host that runs it has undertaken to contribute the scenario once it re-pins to a tree carrying
the harness, on the ground that a scenario it cannot run against its own pin is a guess.

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
| `torch.cuda.mem_get_info` at `v0.37.0`, before PERF-6 | 10 | 4 | 0 | 0 | 0 | 7 | 0 |
| `host.get_free_memory` | **10** | 0 | 0 | 0 | 0 | 0 | 0 |
| `torch.cuda.mem_get_info` | **10** | 0 | 0 | 0 | 0 | 0 | 0 |
| `_preflight_memory` | 0 | 7 | 1 | 5 | 1 | 10 | 0 |
| host's own end-of-frame sync | 0 | 1 | 1 | 1 | 1 | 1 | 0 |

Before PERF-2 every D2H was **4 bytes** (8 over two blur stages, 12 over three): they were
`.item()` drains, not image traffic — `gauss_blur` resolving its kernel radius. PERF-2 carries
the host reading of a literal / `$param` / folded constant on the 0-dim tensor it is minted
into, so the whole column is 0 and the `source_edit` row became stable at 0 with it. A sigma
genuinely computed on the device still drains, correctly: this comp has no such stage. There is
no H2D on any interactive tick either — the canvases stay resident.

**The two free-VRAM rows, and whose queries the `prewarm` 10 are (PERF-6, F1).** Both rows
read the same integers here, but they are not the same measurement: the driver call is the
inner 13–17 µs of a 90–112 µs host call, which is why the seam is counted as well as the
driver (§6 item 6). PERF-6 took the tile planner's queries to **0** on every cooking scenario
by sharing one live reading across a frame — and left `prewarm`'s **10** exactly where they
were, because they are a DIFFERENT CALLER asking a different question:
`tex_runtime/compiled.py::_cuda_headroom_ok` (`compiled.py:1006`), once per program, deciding
whether there is comfortable VRAM headroom (`free > 2 GB`) to submit a BACKGROUND compile. It
wants a live reading precisely because it is about to start something that allocates, and a
prewarm is a once-per-project cost rather than a per-frame one — so this row is 10 by design,
not by omission. It is also the NON-INERT witness for `tests/test_bench2_counts.py`'s
free-memory pins: five of the seven scenarios pin it at 0, and a lane that takes the
`_cuda_headroom_ok` query (10 → 1, with `TEXCache.compile_ast` unmoved at 10) owes that test
another non-zero row first.

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

  0 stable counter row(s) moved; 0 frame row(s) moved (not gated); 0 unstable row(s) differ (ignored).
```

**What the verdict counts, and what it only reports.** `--compare` returns rc 1 on a moved
**api** or **cuda** row and never on a `frames.*` row. The frame census counts Python frames
per `module:function`, so it moves for every lawful change that adds a call, renames a helper,
splits a module or adds a scenario: the chain split moved thirteen `frames.mod.*` rows on each
device with `frames.total` and `sum(frames.mod.*)` conserved to the unit on all seven
scenarios, and a scenario ADDITION moves them too. A gate that counted them returned 1 for
every such change, so its exit code said nothing and the reader had to reason past it by hand.
The frame rows are therefore printed under their own heading with those two sums beside them —
the sums are the check — and `--counters-only` names the rule for a caller that relies on it
(`tools/gate.py --tier full --counts-baseline …` passes it). The gate in `tools/gate.py` is the
single entry point: one command, one verdict line, the known-red allowlist applied from
`tests/known_reds.json` rather than by eye.

Compare that with the timing null controls in §1, taken on the same laptop. The CPU pass of
the gate at 96²/48²/4 ticks runs in about 2.6 s.

The same check is what an added scenario owes the ones already there: BENCH-3 saved the seven
at the gate shape (96²/48²/4, both devices) before adding `node_scrub` and `--compare`d the
same seven after, reading **0 stable rows moved** — a new scenario must not move an old row,
and the `--scenario` filter makes that provable rather than asserted. Give each leg its own
cold `TEX_CACHE_DIR`: a warm one reports six `source_edit` rows moving that are the scenario's
own cold/warm program cache and not the change (PERF-4 hit exactly that). `--save` now records
the directory and whether it was empty at start, and the sha carries a `-dirty` suffix when the
measured tree is not the commit it names, so a comparison between two legs that were not both
cold — or between a commit and an edited copy of it — says so in its own header instead of
being reconstructed from memory afterwards.

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

   **An embedding host has confirmed the pattern and wants the handle** (2026-09-21): it carries
   the whole-frame key forward across the clean prefix exactly as `RoiComp` does, with its own
   refusal of the quadratic alternative measured beside it, and it asks for the counter above to
   be the handle's acceptance test. It is a **hook, not a defect**, and it is not urgent on that
   host's account: that host also mints one key per non-routing node over its **whole graph** every
   tick, in addition to the chain's, which is the larger count and is its own to fix first.
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

   **This is a work item, not a curiosity: PROF-1 is ARMED in an embedding host's shipped
   default** (reported 2026-09-21), deliberately, with the sync cost named and budgeted. That
   host's checkpoint planner returns an empty plan forever without the per-stage breakdown, so
   it arms the profiler at bring-up, re-arms it after a reset, and drops a stored *off* from an
   older settings file on upgrade. Two consequences. First, *"the profiler is disarmed by
   default"* above is **TEX's** default and not the deployed state, so the four syncs are on a
   real interactive tick today. Second, any fix inherits a contract: the per-stage breakdown must
   survive, and `should_sample`'s shape — every cook of an unseen key until three samples, then
   one in sixteen — is load-bearing for that host's cost table. Changing the sampling rule is a
   contract change that is named in a hand-back before the tag, never a tuning.

   **Attempted, measured, and DECLINED (PERF-9, 2026-09-21). The four syncs stay.** The obvious
   fix is to drop the two inner barriers on the ground that the outer pair already serialises.
   Built and interleaved against its own base, it does not error and does not empty the per-stage
   table — it fills the table with plausible, badly wrong numbers while the whole-cook total, still
   bracketed by the untouched outer pair, stays roughly right:

   | | stage 0 | the heavy stage | stage 2 | sum of stages | whole cook |
   |---|---:|---:|---:|---:|---:|
   | base | 6.32 ms | **50.51 ms** | 11.91 ms | 68.74 ms, tracks the total | 68.9 ms |
   | inner barriers removed | 0.21 | **0.68** | 0.22 | ~1.12 ms, **1.6 % of the total** | 64.8–84.8 |

   A checkpoint planner fed the second row would never place a tap on that 50 ms stage. That is the
   *present but wrong* failure this item's constraint exists to prevent, and it is worse than the
   barriers. **4 → 0 is structurally unavailable** as well: some tier routes never reach the
   interpreter's inner syncs at all, so the outer bracket is their only barrier. Even 4 → 3 fails —
   merging the outer enter with the inner pre-loop is falsified by real GPU dispatch in the
   binding-cast and coordinate-builtin preamble, and merging the inner close with the outer exit is
   safe only at fp32.

   **One premise died usefully.** The cook queue's own completion bracket does *not* provide a
   barrier that would make any of the four redundant: it feeds the profiler from a bare wall-clock
   delta with no device synchronisation, pricing job admission rather than the cook.

   *What would reopen it:* a per-stage timing that does not need a barrier at all — device events
   recorded into the stream and read once at the end of the cook, rather than a synchronise per
   stage boundary. That is a different mechanism, not a tuning of this one.
5. **The results cache grows by one entry per cook, forever, on interactive ticks.** The
   `results_cache.entries_added` row tracks `ResultCache.put` exactly on every scenario. **Host
   policy** — a scrub visits values it will never revisit, and nothing tells the cache so.
   *Shows fixed as:* `results_cache.entries_added` going to **0** on `terminal` and `pan` while
   `ResultCache.get`'s hit behaviour on a revisited value is unchanged.

   **Answered, and closed on the engine's side** (2026-09-21). An embedding host already withholds
   every interactive write at its own door: it stamps each routing decision with a spill flag that
   is false for the interactive quality class, and its cache arm returns early from both the write
   and the patch, counting the withheld write. Reads stay ungated, so a scrub still hits what a
   render or an export left behind — which is why `restores` reads 0 on this harness and is the
   same fact seen from the other side. That host measured this cliff independently, on its own box
   and for the same reason, and asks for **no engine change**: not a `transient=True` entry class,
   not a host-bumped generation, because one boolean at one host-side door is already counted and a
   second mechanism upstream would be a second place to get it wrong. So retention is **host
   policy**; what was owed here is this paragraph, not a new row.

   **What it costs when the budget is reached, measured (BENCH-3, from PERF-6's F3).** The row
   reads a tidy integer the whole way down the cliff, so here is the cliff. Driven at 1024²
   on a quiet box (GPU 0 %, 0 MiB either side), timing each tick and reading `ResultCache.stats()`
   beside it (drive a scenario's `tick` in a loop and print `comp.cache.stats()` each time —
   the harness's own scenario classes are importable, so this is a dozen lines):

   | | `terminal` (4 MB/tick) | `all_dirty` (10 × 16 MB/tick) |
   |---|---:|---:|
   | RAM budget (default, this box) | 2048 MB | 2048 MB |
   | tick the budget is reached on | **471** | **11** |
   | median before / after | 1.09 ms / 6.47 ms | 6.5 ms / 165 ms |
   | ratio | **5.9×** | **25–31×** |
   | per tick past the knee | 1 eviction + 1 spill | 10 evictions + 10 spills |
   | `restores` over the whole run | **0** | 10 (all from priming) |

   So an interactive scrub does reach it, and a `terminal` drag is not exempt — it is 470 ticks
   away rather than 11, which is seconds of dragging one slider, and `pan` fills at the same
   rate. **The 40-tick run PERF-6 saw the cliff on was `all_dirty`; `terminal` at 40 ticks is
   flat** (48 ticks: median 0.87 ms, 0 evictions, 356 MB held) and only turns over at ~471.

   **Where the time goes: the SPILL, not the eviction and not the copy-on-read.** Neutering
   `ResultCache._spill` and re-running the same twenty `all_dirty` ticks interleaved between two
   shipped legs (interleaved, per §7's rule) leaves the same **92 evictions**
   and takes the ratio from 26.4× / 35.0× to **0.96×** — flat. Eviction itself is bookkeeping
   under the lock; what costs is `_drain_spills` writing each victim out (a D2H for a CUDA
   frame plus a pickle), ~14.6 ms per 16 MB frame ≈ 1 GB/s. Copy-on-read is not in it at all:
   a scrub never revisits a value, so `hits` stays 0 and `get`'s copy never runs. Those frames
   are written and never read — `restores` is 0 over 620 `terminal` ticks and 119 spills.

   **Which counter would show it, and why the gate cannot.** `ResultCache.stats()` already
   counts `evictions`, `spills` and `restores`; a harness row over `spills` is the honest
   instrument (it is 0 on every short run and non-zero exactly at the knee, while
   `entries_added` is 1 either side). It is NOT added here, because at the gate's 96²/4-tick
   shape a canvas is 147 KB and `all_dirty` would need ~14 000 ticks to reach the knee — a row
   that can only ever read 0 in CI is decoration. What would make it gateable is a scenario
   that arms the cache with a small explicit budget (`ResultCache(budget_mb=…)`), which crosses
   the knee in a handful of ticks at any resolution; that is a scenario design decision and
   belongs with whoever takes item 5's retention policy, since the two share an acceptance test.
6. **The per-cook fixed pipeline, paid ten times on a whole-frame cook.** Per cook, at head:
   two tile plans (`tex_tiling._tile_plan:38` and `_halo_tile_plan:142`, both re-exported into
   `tex_engine` at `tex_engine.py:113`), `enforce_cache_budget` (`tex_memory.py:327`),
   `trim_reserved_pool` (`tex_memory.py:780`) and `_disown_inputs` (`tex_buffers.py:158`) —
   all called from `tex_engine.run` (`tex_engine.py:1243-1253`) — plus `fingerprint`, **once**
   since the per-cook key became one string handed down from `prepare`.

   **Who buys the free-VRAM reading, corrected.** An earlier reading of this item attributed
   the **7 of 10** stages that issue `torch.cuda.mem_get_info` on an `all_dirty` frame to the
   M-1 preflight. They are the **TILE PLANNER's**: on those stages LAT-2's cheap path fires,
   `_preflight_memory` returns `free_hint = None` without querying, and `_tile_plan` then buys
   its own reading — so the `free_hint` hand-off is inert on exactly the cook it was written
   for. The other three stages are the blur/morphology ones, which are not tile-safe and leave
   through `is_tile_safe_cached` before any query. The interactive ticks issue **none**.

   **Measure it at the seam that costs the money, not at the driver.** The `host.get_free_memory`
   row is the one to read: the driver call is only the inner **13-17 µs** of a **90-112 µs**
   host call (the host folds allocator statistics in on top of `mem_get_info`), so a fix
   measured on `torch.cuda.mem_get_info` alone would claim a seventh of what it actually saved.
   Both rows read 7 on `all_dirty`, 4 on `source_edit` and 0 on every interactive tick **when
   this item was written; PERF-6 has since landed it and at head both read 0 on all three**
   (`tests/test_bench2_counts.py` pins them there and §4.2 carries the current readings), while
   `prewarm`'s 10 are a different caller entirely (`tex_runtime/compiled.py::_cuda_headroom_ok`,
   once per program before a background compile is submitted) and stayed at 10.
   *Shows fixed as:* `host.get_free_memory` **and** `torch.cuda.mem_get_info` going **7 → 1 or
   0** per `all_dirty` frame with `prewarm`'s 10 unmoved, and `_halo_tile_plan` going
   **10 → 0** on the stages whose pixel-local plan already answered.
7. **A window move rebuilds the coordinate builtins.** `pan` costs **26** CUDA kernels and
   **22** allocations against `terminal`'s 22 and 18 — exactly +4 and +4 for the same cook with
   a moved window. (This is also the LAT-4 LRU the harness had to defeat to measure the row
   honestly; see §2.)
   *Shows fixed as:* `cuda.kernels` on `pan` going **26 → 22** and `alloc.allocated`
   **22 → 18**.
8. **Two lexes for a never-seen program.** `TEXCache.fingerprint` (`tex_cache.py:344`) calls
   `param_only_names` (`tex_marshalling.py:837`), which tokenizes; the compile then tokenizes
   again. The counts track exactly — `param_only_names` equals `fingerprint` in every column of
   §4.1 — and `fingerprint` itself is called **twice per cook**.
   *Shows fixed as:* `TEXCache.fingerprint` and `param_only_names` both going **2 → 1** per
   cook, and the `prewarm` warm-up tick's `Lexer.tokenize` falling from **20 to 10** (one lex
   per never-seen program instead of two — the steady prewarm ticks already read 10 because
   `param_only_names` memoizes on the source after first sight).
9. **A full `TypeChecker.check` on every disk-cache reload.** `TEXCache._load_from_disk`
   (`tex_cache.py:643-649`) re-runs the checker to regenerate a `type_map` with valid `id()`
   keys after unpickling an already optimized program. `prewarm` reads **20** checks for ten
   programs.
   *Shows fixed as:* `TypeChecker.check` on `prewarm` going **20 → 10**.

## 7. The timing tier: a release sitting, and the first one

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

### 7.1 The first sitting: 2026-09-20, the sm_75 reference box (v0.38.0)

**The tier is no longer deferred.** It sat, on a quiet **sm_75 desktop** — deliberately not the
sm_120 development laptop, which is shared with other work. Ten legs, each running all five
benches above plus the counts harness, each with its own cache directory emptied before the leg
started. The whole record — every leg's JSON, every leg-to-leg comparison, the per-program
breakdown and the reproduction commands — is archived at
`benchmarks/results/sitting_2026-09-20_sm75/`, with its own README. What follows is the reading.

**The closing read is the one to quote**, because it is the only one with its null control in
the same sequence: `base_a4` → `after3_e` (the round-3 tree, `8f38f82`, on its second run) →
`base_a5`. Ratios below are `base_a4 / leg`, so >1 means the leg is faster, and the **third
column is identical code**:

| measurement | `after3_e` | `base_a5` (the null) |
|---|---|---|
| eight-config corpus, per-config geomean, all eight | 0.996 – 1.027 | **1.003 – 1.047** |
| `roi_scrub`, panning window **plus** a moving parameter | **1.215** | 1.013 |
| `roi_scrub`, fixed window / panning window | 1.041 / 1.040 | 1.000 / 1.001 |
| `roi_scrub`, whole frame | 1.014 | 0.998 |
| `region_recook`, CUDA 2048², mid region | 1.028 | 0.999 |
| `region_recook`, CUDA 2048², whole frame | 1.008 | 1.002 |
| `param_scrub`, recook median | 1.030 | 1.009 |
| `param_scrub`, scrub / static median | 1.022 / 1.023 | 1.005 / 1.008 |
| `param_scrub`, worst tick after warm-up | 1.334 | 0.988 |
| structural counter rows moved | **23**, every one predicted | **0** |

**Read the first row and the last row together; they are the whole argument.** On the
eight-config corpus the null leg's spread (1.003–1.047) is **wider than the claim's**
(0.996–1.027), so the default whole-frame cook path is **neutral** — which is what invariant 7
asks for, and is not a speedup claim. On the four benches that isolate an interactive cost the
null leg sits at 0.99–1.01 and the release does not; that gap is what makes a 1.04 mean
something there when a 1.04 on the corpus would mean nothing. And the counters — which are not
a timing instrument at all — moved 23 rows for the release and **zero** for identical code.

An earlier null pairing in the same sitting agrees: two `v0.37.0` base legs returned per-config
geomeans of 0.967 – 1.033 with individual rows from 0.40 to 2.49. The per-row spread is why §1
says what it says.

Every moved counter row was predicted before it was timed — `Lexer.tokenize` and `Parser.parse`
1 → 0 on `terminal` and `midgraph` (PERF-1/4), `TEXCache.fingerprint` and `param_only_names`
halved or better on every scenario (PERF-5), `cuda.memcpy_DtoH` 2 → 0 and 3 → 0 (PERF-2),
`torch.cuda.mem_get_info` 7 → 0 on `all_dirty` and 4 → 0 on `source_edit` (PERF-6) — and the
timing rows that moved are the ones those counters sit on. That correspondence, not either
number alone, is what this note exists to make routine.

### 7.2 Two artefacts the first sitting measured, and the protocol they bought

**A tree's FIRST run is not a measurement.** One leg read
`region_recook cpu/n50/2048/whole_all` at **1996.74 ms** where every other leg of the same
sitting — including the same tree run immediately afterwards — read ~900 ms. A 2.2× phantom, on
one row, from one tree's first run. Nothing in the diff explains it and nothing needed to: it is
the first-touch cost of a freshly materialised tree. **So each newly materialised tree gets a
discard leg**: run it, throw it away, report from the second. The discarded leg is archived
beside the reported one so the protocol is visible rather than described.

**A shared artifact cache is not a comparison.** Each leg gets its own cache directory, emptied
before it starts, because a second leg sharing one reads what the first wrote and produces rows
that look structural. Two lanes lost a measurement to exactly this before it became a rule.

Both are now standing law in `docs/brief-conventions.md` §"Two measurement rules that are not
negotiable", and `--save` records the cache directory, whether it started empty, and a `-dirty`
suffix when the measured tree is not the commit it names — so a leg that broke either rule says
so in its own header instead of being reconstructed from memory afterwards.

**One more thing the sitting settled, and it is the reason this note exists.** A regression
reported against the cold compiled path did not survive its own null control: three of five legs
agreed to within 0.7 %, the other two agreed with each other and were ~8 % faster, and it was an
accident of which of them was used as the denominator. Pick the other base leg and the same
after legs read 0.999 and 0.993. The null pairing — the same tree against itself — read **1.082**
on that config, i.e. **larger than the claim**. A counts pin shipped instead of a fix, because
there was nothing to fix (`tests/test_perf7_compiled_cold.py`).

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
