# Effort-based checkpoints — design note (CACHE-7, v0.32.0)

*Companion to `docs/results-caching.md` (ENG-12/CACHE-1/2/3/4), which this extends, and to
`docs/cook-queue-scheduling.md` (SCHED-4/PROF-1/PRED-1), which supplies the idle time and the
cost signal. Roadmap: `docs/roadmap.md` §4 P2, and report 39 §v0.32.0.*

The reports' flagship idea is short: **place cache points by cumulative measured cook time,
not by node count.** This note is about what that sentence costs, given that the only
materialization point TEX ships (CACHE-6) is one cut, chosen by the host, on a linear chain,
at fp32, keyed without a resolution.

---

## 1. The problem

Fusion is why a ten-node grade chain cooks as one program instead of ten. It is also why
twiddling the *last* node's slider recooks all ten stages: a fused program has no interior
cut points, so every tick pays for nine stages that provably did not change. Fusion makes
interactivity worse exactly where it matters most, and CACHE-6 exists to cut the chain.

CACHE-6 cuts it **once**, at a cut point `k` the *host* supplies. That is enough for the case
it was built for — "the user is twiddling the node they just added", where the right cut is
"just below the terminal". It is not enough for the case v0.32 is named after:

- **The cut is a guess.** Nothing in the engine knows what a stage costs, so `k` is a host's
  hunch. A cut one stage too high leaves the expensive blur below the tap and buys nothing.
- **One cut serves one edit.** Tap at `k`, then edit a stage *above* `k`: the tap's prefix
  changed, the boundary is invalid, and the next edit pays a full recook *and* a re-tap. A
  user moving up and down a chain — which is what grading is — thrashes.
- **Measured, on this box** (`benchmarks/checkpoint_bench.py`, 12-stage chain, 2048²): a
  *late* edit takes the single tap from 41.4 ms to 2.5 ms on CUDA (**16.3×**) — but a
  *mid-chain* edit only from 42.7 ms to 18.9 ms (**2.26×**), and that is with the benchmark
  handing CACHE-6 an **oracle** cut (`k = edit_at`, the perfect cut for that exact edit) which
  no host actually knows. The second number, and the oracle, are the item.

The report's model — checkpoints at cumulative-cost boundaries, so any edit costs at most
(threshold + suffix) — needs *N* taps, placed by *measured* cost, maintained without making
the first cook slower. All three of those are the work.

---

## 2. What already exists (and what that leaves as the work)

The load-bearing discovery, and the reason an L item fits in one release beside two others:
**the engine can already export the interior handoffs of a fused chain in a single cook.**

`compile_fused` accepts a per-stage `tap: True` and appends `@_tap_s{i} = <that stage's
handoff local>` to the spliced program (`tex_fusion.py:520`, `:574`). The tap is an assignment
of a local the program already computed, so arming it costs no arithmetic. Verified on this
box, CPU and CUDA: a 6-stage chain tapped at stages 1 and 3 returns
`['OUT', '_tap_s1', '_tap_s3']`, each tap **bit-exact** (`torch.equal`) against the standalone
prefix cook of those stages, with `@OUT` itself unchanged by the tapping.

That turns phase 2 from "re-cook the chain in N segments" into "cook it once, keep what falls
out" — **up to seven of them**. `MAX_OUTPUTS` is 8, so a fused program carries at most 7 taps
beside its own `@OUT`, and `compile_fused` drops the overflow in *ascending stage order*:
measured, asking for 8 taps returns stages 1–7, and asking for 12 also returns 1–7. It keeps
the shallowest and discards the deepest — exactly inverting §4's deepest-first serve, which is
where the entire win comes from. §8 says what CACHE-7 does about it.

The rest of the substrate:

| Piece | Where | What it gives |
|---|---|---|
| Stage-list surgery | `tex_fusion.is_linear_stage_list` / `prefix_fingerprint` / `suffix_stage_list` (`:895`–`:924`) | The linear-chain test, a value-independent fingerprint per cut, and the rebind of stage `k`'s `chain_input` to an injected boundary |
| Multi-tap export | `compile_fused` `tap:` (`tex_fusion.py:520`) | Every checkpoint from one cook |
| The tap key | `tex_engine.boundary_lineage_key` (`:1723`) | Already namespaced by the cut (`flags=['tap:s{k-1}']`), so N cuts mint N keys with no key-scheme change |
| The single-tap driver | `tex_engine.cook_fused_cached` (`:1754`) | The gate, the `FusionError` fallback, and the "correct-but-not-incremental full cook" posture |
| The store | `tex_results.ResultCache.put/get` (`:239`/`:269`) | Freeze, slice compaction, byte accounting, LRU + disk spill |
| Per-stage cost | `tex_runtime.profile.stage_costs(key, spatial)` (`:284`) | `{stage_index: EWMA ms}` — CACHE-7's declared input |
| Idle time | `tex_cookqueue.CookQueue.submit(..., klass=SPECULATIVE)` (`:274`) | A single worker thread that yields to interactive work |

**So the work is:** a placement policy over `stage_costs`; a multi-tap cook/serve driver; the
phase-2 harvest; and four correctness repairs the generalization forces (§5, §6, §7, §8).

---

## 3. Placement: cumulative measured cost

Walk the stages from the source. Keep a running sum of per-stage cost. When the sum since the
last checkpoint crosses the threshold, put a checkpoint at that boundary and reset the sum.

```
plan_checkpoints(stages, costs, threshold_ms) -> [k1, k2, ...]        # cut indices, ascending
```

Three properties this buys, and one it deliberately does not:

- **Bounded recook.** With checkpoints every ≤ threshold of work, an edit at stage *j* recooks
  from the nearest checkpoint at or below *j* — at most `threshold` of upstream work plus the
  suffix. That is the report's model, stated as an invariant rather than a hope.
- **Cheap stages never get a tap.** Six pointwise grades costing 0.2 ms each do not earn a
  checkpoint between them; the blur does. An even-spacing or every-N-nodes policy spends taps
  where they buy nothing, which is the exact failure "effort-based" is named against.
- **A tap is never placed where it cannot pay.** A checkpoint whose own materialization cost
  (a `put`: freeze + contiguous copy + key) exceeds what it saves is refused. This floor is
  not a formality, and it is **device-dependent by more than an order of magnitude** (medians
  of 30, 4-channel fp32, torch 2.10 / RTX 2080 SUPER):

  | | 512² | 1024² |
  |---|---|---|
  | cpu | 1.117 ms | 3.501 ms |
  | cuda | 0.062 ms | 0.957 ms |

  On CPU that is several whole pointwise stages, so a policy ignoring it would place taps
  costing more than the recook they save. A single CPU-calibrated constant would make the
  opposite error on CUDA — overstating the floor ~18× at 512² and refusing checkpoints that
  comfortably pay, on the device interactive hosts actually run on. The floor is also a
  *multiple* of the `put` (2×), so a checkpoint that merely breaks even is refused too: the
  saving repeats on every later edit, but so does the memory, and a whole frame of RAM should
  buy more than a wash.
- **It is not a global optimum.** A greedy prefix walk is not the dynamic program that would
  minimize expected recook cost over a distribution of edit positions. It is chosen because
  the input is an EWMA with a handful of samples, and a DP over noisy costs optimizes noise.
  Recorded as deferred with its gate (§11).

**The cost input is honest about its own youth, and the number is not small.** PROF-1's first
samples include the cold cook, and the resulting attribution is not merely noisy — for a while
it is *inverted*. Measured on a 3-stage chain whose stage 0 is a multiply and stage 1 a blur:

```
 3 samples (warmup):  stage0=3.096  stage1=3.411  stage2=0.665
15 samples (settled): stage0=0.318  stage1=2.073  stage2=0.633
standalone truth:              0.301         1.928
```

At 3 samples the cheap stage is over-attributed ~10× and outranks nothing correctly; by 12 it
is settled and matches standalone cooks closely. So `MIN_SAMPLES = 12`, and the honest cost of
that is stated rather than buried: PROF-1 measures the first 3 cooks of a key and then 1 in
16, so **12 samples is roughly 150 cooks**. Until then a host gets *no checkpoints* — today's
cook, correct and not incremental. A host that knows its own chain need not wait: `cuts=` is
accepted everywhere placement is and skips the estimator entirely.

Placement on young numbers is worse than no placement: it would checkpoint the cheap stage and
skip the expensive one — the exact failure "effort-based" is named against — while reporting
success.

**`stage_costs` now falls back across resolution buckets.** It used to return `{}` at any
resolution the session had not already cooked, while `predict` answered happily for the same
key, so a host that changed resolution lost placement entirely until it re-measured from
scratch. It now scales the nearest measured bucket by the pixel ratio, exactly as `predict`
does — the deferral `predict`'s own docstring recorded ("left for CACHE-7, whose placement
decision is the one that would actually be wrong") come due.

---

## 4. Multi-tap: serve from the deepest valid checkpoint

On a cook, walk the planned cuts from the **deepest** down. The first one whose boundary is in
the cache wins: splice the suffix from there and cook only `k..N`. Nothing is cached → cook
the whole chain, exactly as today.

Deepest-first is the whole point: it is what makes an edit's cost depend on the distance to
the nearest checkpoint rather than on the length of the chain. It also means an edit high in
the chain silently invalidates every checkpoint below it — correctly, because each of those
boundaries is keyed by the fingerprint *and the param values* of its own prefix
(`boundary_lineage_key`), so a changed upstream param mints a different key and simply misses.
There is no invalidation protocol to get wrong; the key scheme already is one.

---


> **CORRECTED (v0.33.1).** This section narrated LIFTING the fp32 gate on a 22-row measurement.
> The gate is closed again. The measurement was real; its conclusion did not follow, because
> every row in its matrix produced an fp16-REPRESENTABLE boundary. M-3 keeps coordinate builtins
> fp32, so `u * 1000.0 + 0.123` is an fp32 local through a straight-through fused cook while the
> checkpointed path materializes it and the suffix downcasts at ingest — **maxdiff 6.58e-01**,
> both devices, fp32 control bit-exact. The precondition was never "fp16", it was "this boundary
> happens to hold values fp16 can represent", which is a property of the VALUES that a precision
> label cannot carry. Reopening requires a representability check on the actual tensor and
> belongs to PREC-1 (`docs/preview-tier-precision.md`), not to this gate.

## 5. Precision: the fp32 gate is lifted, and the measurement that lifted it

CACHE-6 gates taps to fp32, on the stated belief that "the boundary MUST be the exact fp32
handoff or the FUS-3 oracle breaks". Under `precision="auto"` a chain may cook fp16, which
makes the whole mechanism **inert on the ComfyUI default**.

Measured instead of inherited (`tests/test_v032_checkpoint.py::fp16 rows`; probe reproduced on
CPU and CUDA, 6-stage chains, cuts at 1/3/5, pointwise and with a `gauss_blur`): a prefix/suffix
split at fp16 is **bit-exact** against the straight-through fp16 cook — 22 of 22 rows
`torch.equal`, maxdiff 0.0.

The mechanism, which is why this is a property and not a coincidence: the interpreter upcasts
outputs to fp32 on egress, so the harvested boundary is an fp32 tensor holding exactly
representable fp16 values; feeding it back downcasts to fp16 losslessly. **So the gate is
lifted, and the property is pinned by a differential row rather than asserted** — if the
egress upcast ever changes, that row reds.

Three consequences stated plainly:

- **A tap costs fp32 bytes even for an fp16 cook** (the boundary is fp32 on egress, measured).
  That is GOV-1's budget arithmetic, not a correctness issue, and it is what CACHE-8's
  half/uint16 packing addresses in v0.33.
- **A LATENT still forces fp32 and still refuses taps** (M-3 forces the precision; the
  `latent_channel_count` term stays in the gate untouched).
- **`precision` must be RESOLVED — `"auto"` is refused.** `prepare()` is where auto resolves,
  and this path never calls it: `cook_stage_list` hands the string to the interpreter, which
  cooks `"auto"` as fp32. That is not a wrong pixel, but it would key a boundary under the
  label `"auto"` while the tensor is fp32 — a frame the same host could never find once it
  resolved, and a label meaning different dtypes on different boxes. Refusing is one line; a
  mislabelled cache entry is not. A host under `precision="auto"` resolves first (the value is
  on `CookResult.precision`) and passes the answer.

---

## 6. Identity: the resolution hole the single-tap key had

`cook_fused_cached` mints the boundary key **without** `canvas=` (`tex_engine.py:1787`), so a
tap's identity carries no shape. `ResultCache.get` validates neither shape nor device — its
`canvas` field is write-only metadata. Resolution identity therefore rides entirely on the
host's `upstream` string, and nothing documents that it must encode one.

Reproduced on this box: with one host source key `("host-source-A",)`, a 64² cook and a 128²
cook mint the **same** boundary key, and the 128² cook is served the 64² boundary — it returns
a `(1, 64, 64, 4)` frame for a 128² request. Silent, wrong-size output, no error.

CACHE-7 closes it, because a multi-tap scheme multiplies the exposure by the number of taps:
the boundary key now carries the **spatial shape of every tensor binding in the prefix**,
sorted by name. That is derivable before the cook (a TEX program's output canvas equals its
input canvas until LANG-6 lands), costs one tuple per cut, and needs no key-scheme change.

The fix lives in **`boundary_lineage_key`'s own default**, not in the new call site: the hole
was in the key minter's contract, not in a caller's diligence, so fixing it where CACHE-7
happened to notice would have left shipped CACHE-6 wrong. Both paths are pinned by the
regression row, which asserts the 64²/128² keys differ *and* that a 128² cook through
`cook_fused_cached` now comes back 128².

**A second range hole, found the same way and fixed the same way.** Python slicing clamps, so
`stages[:k]` for any `k ≥ len` is the whole list — and `prefix_fingerprint(S, 99)` silently
returned the *whole chain's* fingerprint, which is the string PROF-1 keys its per-stage table
on and CACHE-1 keys frames on. The one shipped caller passed a hand-audited constant, so it
never fired; a multi-tap planner generates `k` programmatically, and one off-by-one would have
written tap tensors into the profiler's identity. It now raises `FusionError`.

**What is NOT fixed, stated plainly.** `upstream` remains an *arity* check, not an identity
check: the gate counts keys against the prefix's tensor bindings but TEX never inspects what a
key means. A host passing a stable but content-*insensitive* key (a node id, a path it later
overwrites) still gets a stale boundary — measured at maxdiff 0.91 on a swapped source. An
engine-side content check would mean hashing every source tensor on every cook, which is the
cost caching exists to avoid. The contract is the host's to keep, and it is documented on
`boundary_lineage_key` where a host implementer will meet it.

---

## 7. Threading: the phase-2 cook is a second writer

`ResultCache` is documented as not thread-safe: "a host that shares one across threads guards
it" (`tex_results.py:194`, the class docstring). Until now that was a host's problem, because every writer was the
host's own cook. Phase 2 makes the **engine** a writer, on the SCHED-4 worker thread, while
the host's interactive cook may be `get`-ing on the main thread — a concurrent `move_to_end`
and `popitem` on one `OrderedDict`, which is a corrupted LRU or a `RuntimeError`, not a stale
read.

Two options were weighed. Documenting "submit your cooks through the queue too" pushes a
correctness precondition onto every host and is unenforceable. Instead `ResultCache` grows an
internal `RLock` around its mutating sections. It is never on the default ComfyUI path
(invariant #7 is untouched — `tex_node.py` has no reference to `tex_results`), an uncontended
acquire measures 220 ns against a `put` of 1.13 ms — 0.02% — and it removes a whole class of
bug from every host at once rather than from the one that read the docstring. Re-entrant
because `get` → `_restore` → `put` is a real call chain.

---

## 8. Phase 1 / phase 2: the first cook must not get slower

**Phase 1 is unchanged.** The first cook of a chain runs straight through, fused, with no taps
armed and no checkpoint lookups on the path. This is the report's own sequencing and it is
also invariant #7: whatever CACHE-7 does, a host that never arms it must not be able to tell.

**Phase 2 runs on idle** — the host submits it at `SPECULATIVE` with reason
`IDLE_CHECKPOINT` (the constant `tex_cookqueue.py:734` has been reserving since v0.31 with no
producer). It re-cooks the chain with `tap: True` on the planned stages and `put`s each
harvested boundary. Because it is speculative it is preemptible and sheddable: an interactive
cook arriving mid-harvest trips the token, the harvest yields at the next SCHED-3 point and
re-queues, and no partial state is published — the `put`s happen after the cook returns, so a
preempted harvest simply caches nothing.

**The 7-tap ceiling is handled by batching, deepest first.** `MAX_OUTPUTS` is a *host socket
count* — `tex_node.py` binds output slots to it — so raising it to suit an internal harvest
would change the node's surface to buy an engine convenience, which is the wrong trade. Nor is
silently capping acceptable, because the drop order keeps precisely the wrong taps. So
`materialize` splits the planned cuts into passes that fit, and puts the **deepest cuts in the
first pass**. On any chain needing ≤ 7 checkpoints — every shape a host can currently produce,
since fusion caps a region at 16 stages — there is exactly one pass and this costs nothing.
Beyond that each extra pass is one more cook on idle time, and if a later pass is preempted
the checkpoints already cached are the valuable ones.

Cost of the harvest, measured: one ordinary cook per pass plus one `put` per tap. Arming the
taps does not change `@OUT` and does not change the cook's arithmetic (verified bit-exact).

---

## 9. DAG regions: the cut set, and what v0.32 actually ships

CACHE-6 refuses a `chain_inputs` DAG (`is_linear_stage_list`), and the roadmap records DAG
suffix-splitting as the deferred half. CACHE-7's design owns the generalization, so here it is:

A cut at `k` on a linear chain crosses exactly **one** edge — stage `k-1` → stage `k`. On a
DAG it crosses a **set**: every edge from a stage below the cut to a stage at or above it. So
the general checkpoint is not a tensor, it is a *cut set* of tensors, and the general suffix
rebind must inject one boundary per crossing edge and rewrite each consumer's positional
`chain_inputs` entry to read it.

**v0.32 ships the analysis, not the execution.** `cut_set(stages, k)` computes the crossing
edges and is tested; `plan_checkpoints` refuses any cut whose set is not a single edge, so a
DAG region gets no checkpoints and recooks whole — correct, just not incremental, exactly the
posture CACHE-6 v1 took. The reason is not effort but evidence: the shipped region detector
(`detect_fusable_regions`) only admits regions fed by **one** external image edge, so the DAG
shapes that reach a stage list at all are internal fan-out that rejoins — and the measured
interactive case, on both in-repo hosts, is a linear chain. Shipping a multi-tensor rebind
with no consumer to exercise it is how a mechanism rots. The gate that reopens it is in §11.

---

## 10. Not on the default path

Every clause of invariant #7, stated so a reviewer can check them one at a time:

- `tex_node.py` gains nothing. The ComfyUI path never constructs a `ResultCache`, so it cannot
  reach a checkpoint.
- `cook_checkpointed` is a **new** entry point beside `cook_fused_cached`. No existing caller
  changes behaviour; `cook_fused_cached` keeps working and keeps its tests.
- Placement reads PROF-1, which is **disarmed by default** (32.9 ns/cook when off). With the
  profiler off, `stage_costs` is `{}`, placement refuses, and the cook is today's cook.
- The taps are only armed by the phase-2 harvest, which only runs when a host submits it.
- The one change to a shipped code path is the `ResultCache` lock (§7) and the boundary-key
  `canvas` (§6) — both inside a class the default path never instantiates. The key change is a
  deliberate cache-identity break for existing tap entries: they miss once and re-materialize,
  which is the correct outcome for keys that were under-specified.

Verified by the standing lane: `eight_config_bench.py --compare` against the v0.31.0 baseline,
warm **and** cold (the cold lane exists because v0.31's first invariant-#7 measurement covered
steady state only and missed a +44% cold-compile regression).

---

## 11. Deferred, with the gate that would reopen each

- **DAG cut-set execution.** Ships as analysis only (§9). *Gate:* a producer that emits a
  genuine multi-external-edge region — i.e. FUS-1's multi-injection — plus one measured
  interactive case where the whole-region recook is the bottleneck.
- **DP placement over an edit distribution.** Greedy prefix walk today (§3). *Gate:* a
  measured edit-position distribution from a real session, and a PROF-1 key with enough
  settled samples that a DP is optimizing signal rather than EWMA noise.
- **Persisted checkpoints across launches.** Boundaries live in `ResultCache`, which already
  spills to disk and restores under `env_epoch`; nothing persists the *placement*.
  `profile.snapshot()` is the seam its own docstring names. *Gate:* CACHE-8's compressed tiers,
  so a relaunch's disk read is cheaper than a re-cook.
- **Half-precision checkpoint storage.** A tap costs fp32 bytes even for an fp16 cook (§5).
  *Gate:* CACHE-8/PREC-1 in v0.33, which own storage precision as a decision.
- **Checkpoints under ROI.** `roi=` is refused on a fused chain (`tex_engine.py:1261`), so
  CACHE-7 (fused) and CACHE-9 (per-stage ROI) serve two different host shapes and do not
  compose. *Gate:* ROI execution on a fused program, which needs the reach analysis to see
  through fusion's local variables — a LANG/ROI item, not a caching one.
- **More than 7 checkpoints in one cook.** Handled by batching (§7), not by raising
  `MAX_OUTPUTS`. *Gate:* a chain that genuinely needs more than 7 — which today cannot exist,
  because fusion caps a region at 16 stages and a 16-stage chain wants at most a handful of
  100 ms checkpoints. If `_MAX_FUSED_REGION_STAGES` ever rises, this does too.
- **An identity check on `upstream`.** Arity only today (§6). *Gate:* a content-addressing
  scheme cheaper than hashing every source tensor per cook — GRAPH-1's version counters are
  the intended answer, since a host that stamps a key per produced value gets this for free.
- **O(N²) key minting in the deepest-first serve loop.** Each candidate cut mints a full
  `boundary_lineage_key`, and each mint re-walks `stages[:k]` — measured at 301 µs for 7 keys
  (36 µs at k=1 rising to 50 µs at k=7). On an ALL-MISS cook (phase 1, before any harvest) that
  is **+6–17%** at small frames; on the normal path a hit at the deepest cut mints exactly one
  key. The probe half is already cheap — a spilled-key set turned each miss from a 19 µs
  `stat` syscall into a dict lookup. *Gate:* a measured host where phase-1 cooks dominate.
  The alternative is known and cheap — key_k depends only on `stages[:k]`, so a rolling
  chain-hash computes all N in one ascending pass (**19.5 µs vs 361.5 µs**, 18×) — but it
  changes the key scheme, and a key-scheme change on a mechanism whose last two bugs were both
  key bugs wants its own release.
- **A measured, probed `put_cost_ms`.** The floor is a committed constant table for this box's
  CPU and one CUDA arch (§3), where `tex_runtime/xfer.py` already owns the machinery for a
  probed, persisted, per-machine cost model (`_fit_line`, arch-keyed persistence, coarse
  fallbacks). *Gate:* a second box whose measured floor disagrees enough to change a placement
  — at which point the table becomes `xfer`'s fallback constants and the probe becomes a
  `copy_ms(nbytes, device)` lane beside `transfer_ms`.

---

## 12. A note on the shape PM-8 asks for

PM-8 names "a 50-node 4K comp". **A 50-node comp has no fused chain at all**, so it cannot
exercise CACHE-7 as written. Measured on synthetic linear graphs through the shipped detector:

```
N=8 -> 1 region (8 stages)    N=16 -> 1 region (16)    N=17 -> 0 regions
N=20 -> 0 regions             N=50 -> 0 regions
```

`_MAX_FUSED_REGION_STAGES` is 16 and `_grow_region` returns `None` past it rather than
truncating, so a chain longer than 16 falls off a cliff into *no fusion whatsoever*. That is a
FUS-side question this item does not reopen (the cap exists to bound one `torch.compile` trace
and the live-intermediate count), but it decides what PM-8 can honestly measure:

- **CACHE-7 is measured on the shape hosts can produce** — fused regions of ≤ 16 stages, which
  is what `benchmarks/checkpoint_bench.py` drives.
- **CACHE-9 is measured on the 50-node comp**, because past 16 nodes a comp is entirely
  unfused per-stage cooks, which is exactly the shape region-granular recook serves.

Reporting one blended number for both would hide which mechanism produced it, so PM-8 is
recorded as two measurements against the same v0.31 baseline. The stale half of the record is
noted too: `tex_fusion.py:31-33` still justifies the cap partly because "CACHE-6 hasn't
landed" — it landed in v0.27, and this item is the other half of that sentence.

---

## 13. Measured (this box: RTX 2080 SUPER, sm_75, no Triton)

`benchmarks/checkpoint_bench.py`, 12-stage chain, 2048², medians of 15.

**The comparison that is not rigged for either side** — a scrub that MOVES across edit
positions, because one cut serves one position and grading does not hold still:

| | full recook | CACHE-6 | CACHE-7 | |
|---|---|---|---|---|
| CPU, scrub (late) | 182.2 ms | 189.7 ms | **117.3 ms** | **1.62×** |
| CPU, scrub (mid) | 198.4 ms | 172.2 ms | **122.4 ms** | **1.41×** |
| CUDA, scrub (late) | 41.4 ms | 36.8 ms | **23.2 ms** | **1.58×** |
| CUDA, scrub (mid) | 42.7 ms | 36.8 ms | **20.1 ms** | **1.83×** |

**Where CACHE-6 wins, stated because it does.** On a *fixed* edit position the benchmark hands
CACHE-6 an oracle cut and it beats CACHE-7 outright (CUDA late: 2.5 ms / 16.3× against 7.6 ms /
5.5×). That is the correct result for a host that knows exactly where the next edit will be and
re-cuts for it. CACHE-7's taps are placed once, from measurement, and serve every position.

**Two numbers that decide how this is used:**

- **Settling takes 147 cooks.** PROF-1 measures 3 warmup cooks then 1 in 16, and `MIN_SAMPLES`
  is 12 (§3). Until then placement returns `[]` — today's cook.
- **The report's ~100 ms default places NOTHING on chains this box cooks.** A 12-stage 2048²
  chain is 42 ms on CUDA; the rows above run at a 10 ms threshold. This is the calibration
  report 39 §4 already lists as host-gated ("CACHE-7's default threshold — a design-doc
  constant until real comps calibrate it"), and GOV-1 is where it will be re-cut. The default
  is left at 100 ms rather than tuned to this hardware, because a constant fitted to one box is
  how a policy stops being portable.
