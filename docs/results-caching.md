# Results become first-class (v0.25 design note)

*"Remember frames."* TEX persists **programs** superbly (the two-tier Mega-Cache, the
codegen `.cg` sidecars, the autotier verdicts) and **results** not at all: the standalone
engine ships with no way to hold a cooked frame. v0.25 builds that half — the ownership
contract a cached frame needs (ENG-12), the identity it is keyed by (CACHE-1), the cache
itself (CACHE-2, `tex_results.py`), warm-tier persistence + prewarm (CACHE-3), and layered
cache epochs so a codegen-only edit stops cold-starting the whole cache (CACHE-4).

Like ROI-3 in v0.24, the frame cache ships **armed by an engine host, not by the ComfyUI
node** — the ComfyUI host already caches node outputs, so wiring a second cache under it
would only double the memory. It is measured, tested, and dormant until a host asks. The
default ComfyUI cook path is byte-identical (invariant #7): none of the five items touches
a watched compiler/runtime file, and every new hook is off the default path.

Dependency order (roadmap §6): **ENG-12 + CACHE-1 → CACHE-2**; ENG-12 lands first, before
any frame is cached. CACHE-3 and CACHE-4 are independent and ride alongside.

---

## Shipped state before v0.25

- **Programs cached, results not.** `tex_cache.py` persists compiled ASTs (`.pkl`) and
  marshalled codegen (`.cg`), keyed by `fingerprint(code, binding_types)`; `autotier.json`
  persists tier verdicts. Nothing writes a *tensor* to disk or holds one in a keyed RAM store.
- **The sampling hash is the only result identity, and it is the wrong one.**
  `tensor_fingerprint` (tex_marshalling.py) hashes 256 strided samples + shape + sum/mean;
  its own docstring admits a collision class on localized edits. That is fine for ComfyUI's
  `IS_CHANGED` cache-*busting* (a false "changed" just recooks) and deterministically wrong
  for cache-*reuse* (a false "same" serves a stale frame).
- **Ownership is a static-analysis invariant, unwritten as a contract.** Codegen's M-5 `out=`
  reuse and the scatter clone-before-write (COW) each keep the cook from scribbling a buffer
  it doesn't own, but "who may write a produced tensor" was never stated — and undefined write
  discipline is the bug class that killed Natron's engine.
- **One mono-hash keys every artifact.** `_CACHE_VERSION` = sha256 over 12 compiler/runtime
  files; a comment-only stdlib edit flips it and cold-starts *every* store (`.pkl`, `.cg`,
  inductor dir, verdicts) for every user — fatal at monthly app-update cadence.
- **Warm decisions die at process exit.** Graph-capturability verdicts, the capture blacklist,
  the torch.compile blacklist, and the backend probes are in-memory only; a relaunch re-trials
  every program from scratch.

---

## ENG-12 — buffer ownership & immutability contract

*Who may write a cooked tensor after it is produced.* The contract, and its enforcement,
live in `tex_engine.py`'s ownership section (beside `_owned_copy` / `_disown_inputs` / DLPack).

**Frozen is a signal, not a fence.** A cook output is usually born an *inference tensor* (every
tier runs under `torch.inference_mode()`), and torch raises on an in-place op on one — **but that
raise is not a rollback**: on torch 2.12 (verified, CPU + CUDA) the op *lands the write and then
raises*. So `frame.clamp_(...)` on a frozen frame corrupts it even though it "failed", and a
version stamp cannot catch it (`frame_version` of an inference tensor is a constant 0). Freezing is
a loud tripwire for a cache-*internal* mistake, never a guarantee against a *consumer* write.
(My first probe checked only that it *raised* — false confidence; the fourth-round audit caught it.)

**The real guarantee is copy-on-read** (mirroring `to_dlpack(copy=True)`):

1. `put()` stores the frame **frozen** — the canonical master; freezing keeps the cache's own code
   from scribbling it and makes any such bug loud.
2. `get()` / `_restore` return an **owned copy by default** (`frame.clone()` — a normal, mutable,
   independent buffer), so a consumer's in-place write can never reach the stored master. A hit
   costs one frame copy — far cheaper than the cook it saves. `get(key, copy=False)` is the opt-in
   zero-copy path for a read-only consumer that promises not to mutate (same posture as
   `to_dlpack(copy=False)`).
3. **Version-stamp-and-verify** (`verify_unmutated`, torch's `t._version`) stays a live mutation
   detector — but only for a **normal** (host-supplied, non-frozen) entry (a bumped counter drops
   it); it is inert for a frozen master, which is exactly why copy-on-read, not the stamp, protects
   the master. `freeze()` / `frozen_copy()` produce the frozen master; `_owned_copy` / `.clone()`
   the mutable copy handed out.

**Why M-5 `out=` reuse can never scribble a cached frame.** Its reuse set is codegen `_tN`
arithmetic temps only — never a binding — and a cached frame can only re-enter a cook *as an
input binding*. The one residual write into a binding, a scatter, is already COW-guarded
(interpreter `_scatter_owned` / codegen `_scat_owned`, clone-before-first-write). ENG-12 does
not touch either DO-NOT-TOUCH mechanism; it states the contract they already satisfy and gives
CACHE-2 the freeze/verify primitives (`is_frozen`, `frame_version`, `verify_unmutated`,
`frozen_copy`, `freeze`). GRAPH-2 cross-thread edges and XPU-2 frame handles inherit these rules.

**ENG-6 (DLPack) already upholds it.** `to_dlpack(copy=True)` is the default and hands out an
owned copy; `copy=False` is documented as voiding the cache if the consumer mutates the view.

---

## CACHE-1 — lineage keys (`tex_results.lineage_key`)

Every cooked output can carry the key that produced it:

```
lineage_key = H(program_fp × params × upstream_keys × frame × device × precision/quality
                × env_epoch × flags × canvas/ROI)
```

This is Nuke's op-hash. The fused memo key (`tex_fusion._fused_fp`) already proved the
value-**independent** half; CACHE-1 adds the value-**dependent** half — and, crucially, does it
*without content-hashing pixels*:

- **`program_fp`** — the fingerprint (`fp`, or `fused_fp` on a chain). Value-independent.
- **`params`** — the non-tensor bindings (widget `$params`) by value.
- **`upstream_keys`** — a tensor input contributes its **upstream lineage key**, never its
  pixels. Under ComfyUI there is no TEX-internal upstream edge yet, so this is empty and two
  cooks of the same program+params produce the same key regardless of input content: a lineage
  key *identifies the cook that produced a frame*, it does not re-sample the frame. A GRAPH-1
  host threads real upstream keys. (This is the property that retires the collision-prone
  sampling hash for reuse.)
- **`device` + `precision` are MANDATORY** — a `None` on either raises, it is never a wildcard.
  Invariant #9's up-to-6.1e-2 CPU↔GPU envelope makes placement *visible*, so ENG-2's CPU-retry
  and a future SCHED-2 placement change must mint a **new** key and recook, never serve a
  cross-device hit. `env_epoch` (torch version + GPU name + compute capability + the code epoch)
  keeps a disk-spilled frame from surviving a torch/driver/GPU/code change a recook would no
  longer reproduce.
- **`frame` / `canvas`** — the whole ENG-7 playhead (every builtin — `frame`/`fps`/`time` — by
  exact value, so a `time`- or `fps`-animation at a fixed frame is a distinct result) and, per
  output, the **produced frame's full shape** (batch + every dim, so a batch-N or a BCHW-latent
  cook can't collide with a batch-1/BHWC one) plus the ROI rect (two same-size sub-windows at
  different offsets key apart, tying into ROI-3).
- **precision is the ACTUAL cooked precision** — if the fp16 finiteness net re-cooks fp32, the key
  (and `CookResult.precision`) say fp32, matching the pixels, so a re-cooked frame doesn't false-
  miss against the later auto-pinned-fp32 cook of the same inputs.

The engine attaches these **only when asked**: `prepare(want_lineage=True)` → `run()` computes
`CookResult.lineage = {output_name: key}` (per-output, keyed by that output's own shape). Default
`False`, so the ComfyUI cook pays nothing (invariant #7).

Scope under ComfyUI (roadmap): CACHE-1's reach is TEX-internal edges (fused-stage handoffs,
CACHE-6) — full lineage arrives with GRAPH-1, whose in-session **version counters** are the
dirty signal while lineage keys are the persistence/disk identity. Complementary, not competing.

---

## CACHE-2 — the frame cache (`tex_results.ResultCache`)

A keyed store of cooked frames, RAM-tier byte-budgeted with a disk-spill tail, every entry
frozen per ENG-12 and keyed by CACHE-1.

**API (host-facing):** `get(key) → tensor|None`, `put(key, tensor, *, canvas=None)`, plus
`stats()`. A host that has a `CookResult.lineage` calls `put(res.lineage[name], res.outputs[name])`
and later `get(key)`. The ComfyUI node does not call any of it.

**RAM tier** — an `OrderedDict` LRU with its own byte budget (`TEX_RESULTS_BUDGET_MB`, default
a modest slice of VRAM), accounted with the same `untyped_storage().nbytes()` primitive the
tex_memory governor uses. Kept **separate** from the stdlib tensor budget: frames are large and
long-lived, and evicting a mip pyramid to hold a frame (or vice-versa) would thrash. CACHE-5
(future) is the governor that arbitrates the two pools under one budget; until then the frame
budget self-evicts and exposes its byte total so CACHE-5 can fold it in.

**Freeze on insert + copy-on-read (ENG-12).** `put` stores `freeze(tensor)` (the frozen master) and
records `frame_version`. `get` returns an **owned copy by default** — a consumer's in-place write
can never reach the master (a frozen frame is not write-proof on torch 2.12; see ENG-12 above) —
and `get(key, copy=False)` is the opt-in zero-copy path. It also runs `verify_unmutated` before
serving, which drops a **normal** (host-supplied) entry that was written through; the differential
test mutates a served frame and asserts the next `get` is still clean.

**Disk spill** — when the RAM budget is exceeded, the LRU victim is spilled instead of dropped:
staged GPU→host as a plain contiguous CPU copy and atomically pickled (`tex_results._atomic_pickle`)
under `get_cache()._cache_dir`, filename = the lineage key (so a stale-epoch frame simply has a
different key and is never found). Page-locking is applied on the RESTORE side, not the spill: a
pinned buffer would not survive pickle anyway, so `get` stages the loaded pageable tensor into a
page-locked buffer when worthwhile and issues the H2D `non_blocking` (pillar 5 — a prefetching host
can overlap the copy with cook time), then re-admits the frame to RAM. A
separate byte cap bounds the spill directory (extending CACHE-0's census posture), tracked by a
running byte total so the cap is O(1) per spill and only rescans the dir on a budget crossing.
The page-lock decision uses `tex_marshalling._pin_worthwhile`'s size band; consulting the measured
`xfer.transfer_ms` (ENG-8) for the overlap/crossover decision is **deferred** to when the 3-stream
pipeline (XPU-1) needs it — a spilled frame is always worth restoring, so a size threshold suffices
here.

**Auto-placement (deferred).** The roadmap scoped a Nuke-style heuristic to CACHE-2 — *cache at
fan-out points and immediately upstream of the node being edited*. That is **deferred to the arming
host (GRAPH-1)**: it needs graph topology (which node fans out, which is being edited), and no such
graph exists under ComfyUI, so there is nothing to place against yet. `ResultCache` is a pure keyed
store; *what* to cache is the host's call.

**Correctness gate (differential oracle).** A frame served from the cache must equal a freshly
cooked one, bit-exact; a spill→restore round-trip must be bit-exact; eviction must stay under
budget; a canvas/ROI change must miss. These are pinned in `test_v025_phase1.py`.

`tex_results` is the **19th** module-level cache (ARCHITECTURE.md / AGENTS.md counts bumped). It
self-evicts against its own byte budget; folding it into the global governor (so `free_tensor_caches`
and memory-pressure eviction see frames too) is **CACHE-5**'s job — until then a host arming it must
size `TEX_RESULTS_BUDGET_MB` conservatively against VRAM.

---

## CACHE-3 — warm-tier persistence + prewarm (`tex_runtime/warm_state.py`)

Generalizes the `autotier.json` pattern into `warm_state.json`, persisting the graph-capturability
verdict (`graphed._capturable_memo`: fp → (capturable, op_count)) — the static AST capture-gate
result, a deterministic function of the program AST + arch — so a relaunch skips re-walking the
gate. CUDA graphs themselves cannot serialize; we persist the *decision* and re-capture off the
hot path.

Deliberately NOT persisted: **backend probes** (`_select_backend` treats a known-True the same as
an unknown — it only skips a known-False — so persisting positives is *inert*, and persisting a
False would harden a one-off runtime failure into a permanent skip); and the **compile / capture
blacklists** (both mix a stable verdict with a transient runtime/OOM crash, which must not become a
permanent cross-launch demotion — DEVELOPMENT.md's rejected-decisions). The version tag is the
CACHE-4 **VERDICT epoch × GPU identity** (device name + torch): a tier-policy or codegen change
invalidates a stale verdict, and a warm_state from another GPU is ignored. Atomic tmp-file +
`os.replace`, under `get_cache()._cache_dir`, loaded once (a latch) on first use.

`tex_api.prewarm(programs, shapes, *, device, precision, compile_mode)` drives the LAT-1a
machinery for project-load / idle warm: per program it compiles, warms + persists the codegen fn
(`compiled._get_or_make_codegen_fn`), submits a background compile (`compiled._submit_bg_compile`),
and seeds the capturability verdict, so the first scrub after relaunch replays instead of
trialling. (`shapes` is accepted for forward compatibility with per-resolution timing warm; the
warming above is shape-independent — codegen emission, the compile artifact, and the static
capturability verdict don't depend on resolution.)

---

## CACHE-4 — layered cache epochs (`tex_cache.py`)

The mono-hash is split into a **nested lattice** of epochs, each gating exactly the artifacts it
can affect:

```
AST_EPOCH      = H(ast_nodes, lexer, parser, type_checker, optimizer, stdlib_signatures)
                 → gates the compiled-program .pkl tier
CODEGEN_EPOCH  = H(AST_EPOCH, codegen, codegen_stdfns, interpreter, stdlib, noise, tex_fusion,
                   <cgreuse env>)                     → gates .cg sidecars + the inductor dir
VERDICT_EPOCH  = H(CODEGEN_EPOCH, precision_policy, autotier, compiled, graphed)
                 → gates autotier.json + warm_state.json
```

The nesting is **load-bearing**: a `.cg` blob is emitted from the compiled program, so it depends
on the AST pipeline *and* the codegen files — `CODEGEN_EPOCH` includes `AST_EPOCH`, or an
AST-file edit would leave a stale `.cg` passing its version check (the "codegen drifts from the
interpreter" failure). The **win**: a codegen-only edit (touch `stdlib.py`) bumps `CODEGEN_EPOCH`
and invalidates the `.cg` + verdicts, but **`AST_EPOCH` is unchanged so the parsed-program `.pkl`
survives** — the parse/typecheck/optimize work is no longer thrown away on every stdlib edit.

The full mono-hash is **demoted to a completeness tripwire**: a test asserts the union of the
three partition file-sets equals the documented watched set and that AST/CODEGEN are disjoint —
so adding a compiler file without assigning it to an epoch reds the suite (it can no longer
silently fall out of every key). A **fail-safe** spot-check recompiles one persisted artifact per
epoch and asserts codegen==interpreter oracle equivalence, so a wrong partition (an edit that
changes output but lands in the wrong epoch) is caught loudly rather than shipping drift.

`codegen_epoch()` is exported for CACHE-1's `env_epoch` (a result is only reproducible under the
codegen epoch that made it). Introducing the epoch scheme is a one-time cache reset at the v0.25
upgrade (the `version` field format changes); every subsequent codegen-only edit then spares the
`.pkl` tier.

---

## The gate that would change the verdict

The frame cache flips from "armed by a host" to "on under ComfyUI" only if a future host owns
its downstream consumers (so a cached frame's ownership is guaranteed, not hoped) **and** there
is a demand signal a node graph can't already answer more cheaply — i.e. GRAPH-1's version
counters. Until then, doubling the cache under ComfyUI's own node cache is pure memory cost, so
CACHE-2 stays host-armed. The exit measurement for v0.25 is **PM-3**: relaunch cold-start for a
100-program project, dry-run measured (the warm_state + prewarm path is what that budget buys).
