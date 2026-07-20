# TEX Roadmap — from ComfyUI node to compositing engine

*Drafted at v0.20.0 (July 2026). This is a living document: horizons shift, items get
re-cut, but the strategic bets and the decision register are meant to be durable.*

TEX's long arc: become the **core engine of a node-based 2D compositing application**
(the Nuke / Fusion / Flame / Copernicus family). Every node in that app runs TEX under
the hood; groups of nodes bundle into tools. The ten pillars that engine must deliver:

| # | Pillar | One line |
|---|--------|----------|
| 1 | Compilation | node groups execute as one fused block — faster cooks, fewer intermediates |
| 2 | XPU | one source runs on CPU, GPU, or both |
| 3 | Caching | compiled code *and results* cached; subsequent runs fast |
| 4 | Lazy inputs | bottom-up demand: only cook what the outputs need |
| 5 | Unified memory | GPU/RAM offload; copy time overlaps cook time |
| 6 | Wrangling | stock nodes are TEX programs; users write their own |
| 7 | AI-ready | TEX data hands to ML models with zero copies |
| 8 | Low overhead | fast interpret, fast compile, fast execute |
| 9 | Easy to learn | good errors, good docs, smooth curve |
| 10 | Versatile | images, video, strings, numbers, rich stdlib |

How to read this: three horizons — **near** (v0.21–v0.23, inside ComfyUI), **mid**
(v0.24–v0.3x, the engine grows its own surfaces while ComfyUI stays first-class), and
**far** (the engine-extraction era: a second host exists). Every item respects the
AGENTS.md invariants and the DEVELOPMENT.md rejected-decisions register; where an item
*borders* a rejected decision, §7 says explicitly why it does or does not reopen it.

---

## 1. Strategic bets (what the field evidence says)

These come from studying how Nuke/Blink, Houdini Copernicus, Fusion/Flame/DCTL,
Blender's GPU compositor, Natron, and GEGL each solved — or failed to solve — the same
ten pillars. They are the load-bearing assumptions of everything below.

1. **One language, one source, every device.** Copernicus's biggest self-inflicted
   wound is the two-language split: VEX wrangles are easy but CPU-only (round-trip
   poison); OpenCL is fast but hostile. Fusion repeats it (Lua Fuses vs embedded DCTL
   kernels). TEX's single-source XPU — same text runs interpreter-on-CPU and
   codegen-on-GPU — is the clearest competitive differentiator we have. **Never fork
   the language into a "fast variant."**

2. **Interpreter-first, compile-behind, hot-swap.** BlinkScript's field reputation is
   compile stalls, driver timeouts, and crashes with no pre-validation; Copernicus
   users still fear "edit a node → everything recompiles → app unresponsive." TEX's
   dual backend makes the cure structural: *always* cook the current frame on the
   interpreter/codegen tier while torch.compile / graph capture happens on a worker,
   then swap. No shipping compositor can do this; it must stay our default posture.

3. **Full-frame planar tensors, never a scanline/tile-granular core.** Blender deleted
   its tile compositor (full-frame rewrite: "often several times faster", bounded
   memory via refcounted buffer lifetimes). GEGL's per-tile dual CPU/GPU residency kept
   OpenCL broken for 15 years. Nuke's row engine needed the PlanarIop retrofit for GPU
   work. TEX starts planar — keep it that way. Stripes/halos are an *out-of-VRAM
   streaming strategy* (M-4's lineage), not a scheduling model.

4. **Plan lazily, execute statically.** Nuke's pull model (validate → request →
   engine) is the canonical laziness design, but its per-row dynamic pulling needed
   heavy thread sync — Foundry themselves added a push ("top-down") mode in 13.2. The
   synthesis for TEX: pull semantics at *plan* time (demand, ROI, frame ranges), then
   compile the resolved cook into a static push plan — exactly what torch.compile and
   CUDA graphs want. Natron is the cautionary tale for recursive pull with per-node
   locking: its own developers cite engine race conditions as what killed it.

5. **One hash family keys everything.** Nuke's per-Op content hash simultaneously
   answers "is this cached result valid," "are these two subgraphs the same
   computation" (op unification = graph CSE), and "what's the disk key." TEX already
   has the value-independent *program* fingerprint; the missing half is the *lineage*
   (value) key — see CACHE-1. Code hash keys compiled artifacts (param tweaks never
   recompile); code hash × lineage keys frames.

6. **A tool is a compilation unit.** Gizmos, macros, and HDAs all bundle UI-wise, but
   in every shipping compositor a tool's internals still cook node-by-node — a macro
   is never faster than its parts. TEX fusion can make **tool-ification a performance
   feature**: publish = pre-fuse the subgraph into one compiled block with promoted
   params as runtime arguments. Nobody else offers this.

7. **Memory budget before feature breadth.** The #1 field complaint against
   Copernicus: ~50 nodes at 4K fp32 filled 16 GB of VRAM (every node caches its
   output); Fusion's answer is still "call PurgeCache and pray." On our 12 GB-class
   target, the global cache governor, narrow storage, and RAM spill are prerequisites
   for *shipping*, not polish.

8. **Parity comes from sharing the execution substrate.** Flame Matchbox is beloved
   because user GLSL runs on the same path as stock tools — Autodesk ships stock
   effects in the user format. Fusion's Fuses are structurally equal but perf-tiered,
   and users learn which tools are "just fuses." Rule for TEX: **any stock node that
   can be expressed in TEX must be** — dogfooding is also the best regression suite.

9. **Params-from-code, layout optional.** DCTL proves one annotated declaration can
   yield storage + UI + marshalling; Matchbox layers an optional XML sidecar for
   layout; Fusion shows why *imperative* UI code is a mistake (schema not inspectable
   without execution). TEX param declarations should be data (LANG-1), never callbacks.

10. **Sequence the product the way Copernicus did.** Production-ready *texturing*
    shipped years before production-ready *compositing*. TEX's app should climb the
    same ladder: stills/procedural texturing → slap comp → video comp (§8).

---

## 2. Near term (v0.21–v0.23): finish what exists, harden the seams

Work TEX can ship *inside ComfyUI* that is simultaneously the compositor's foundation.
Four workstreams. Effort tags: S/M/L.

### Workstream A — finish fusion (pillar 1)

- **FUS-0 (S, v0.20.1 bug).** Fused chains cannot reach `torch_compile`/`auto` in
  production: `select_tier` requires `fused_fp_present` (tex_node.py:571-574) but
  `execute()` computes `fused_fp` only under `cuda_graph` (tex_node.py:808, stale
  comment). One-line gate fix + a node-path regression test (the existing F-1 test
  bypasses the gate with a synthetic fingerprint).
- **FUS-1 (M).** DAG-region fusion producer. The Q-3 splicer already accepts arbitrary
  DAG edges (`chain_inputs`), multi-output (`exports`), and observed intermediates
  (`tap`) — but the only production producer emits linear chains
  (tex_fusion.py:623-652), and the frontend detector breaks on any fan-out
  (js/tex_extension.js:2354-2390). Generalize detection to single-terminal fusable
  *regions* — and implement the detector **once, in pure Python**
  (`detect_fusable_regions(nodes, edges)` beside tex_fusion), with the JS demoted to
  a thin caller from day one. Fusion legality is a TEX-semantics analysis;
  host-side reimplementations drift (this pre-empts the porting half of SCHED-1).
  In a real comp graph (merges, masks, branches) linear-only fusion fuses almost
  nothing; this is pillar 1's core promise.
- **FUS-2 (M).** Fused-chain lazy composition: `fused_required_bindings(spec, params)`
  walking stages terminal-first with per-stage folded params, so dead upstream
  branches of a fused chain stop cooking (today fusion requests everything —
  tex_node.py:493-494). One memoized analysis must feed **both**
  `check_lazy_status` and execute()'s E6003 gate for fused chains — the same
  dual-consumer discipline invariant #11 already mandates for tex_lazy._memo —
  and the never-sever rows extend to fused chains as the gate.
- **FUS-3 (M).** Fusion-region equivalence suite: branching/fan-out regions vs
  unfused execution, bit-exact per invariant #2's discipline. Gate for FUS-1.

### Workstream B — interactive latency (pillar 8)

- **LAT-1 (L).** Compile-latency masking everywhere, in two halves. **(a, M)** forced
  `torch_compile` currently blocks the cook on `future.result()` (~28 s cold —
  compiled.py:655); route it through the background-compile machinery `auto` already
  uses (serve codegen/interpreter meanwhile, swap on ready), plus queue-time
  speculative warm: pre-materialize codegen fns and submit background compiles for
  every TEX program in the prompt. **(b, M, own mini-design)** asynchronous
  CUDA-graph capture (3 warmups + capture currently synchronous in-cook) — capture
  has thread-local capture-mode and stream-ordering constraints, so (b) is the hard
  half and must not ride (a)'s coattails.
- **LAT-2 (M).** `PreparedProgram`: a per-(fingerprint, binding-type signature,
  device, precision) bound-cook object caching the ~8 O(#bindings) static scans,
  env-key lists, and capture-key skeletons that `execute()` re-derives every cook.
  Keep `has_spatial` per-cook (value-dependent — DO-NOT-TOUCH register).
- **LAT-3 (S).** Deferred timing readback: MEASURING/TRIAL/verify cooks currently
  force a CUDA event sync per cook; record the event pair, read `elapsed_time` on the
  *next* cook (stream-ordered; medians don't care), cap measurement frequency.
  Invariant #6 intact — deferral changes *when* the sync happens, never whether.
- **LAT-4 (S).** Interpreter coordinate-builtins cache is single-entry
  (interpreter.py:141-142); proxy/full-res alternation rebuilds ~64 MB of u/v tensors
  every flip. Small LRU (4–8), registered in `_build_keepalive` + the tex_memory
  budget, mirroring codegen's `_ENV_TENSOR_CACHE`.

### Workstream C — engine-seam hardening (pillars 2, 5, 7 + embedding)

- **ENG-1 (L, flagship).** PORT-2b: extract the cook orchestrator out of
  `tex_node.py` into `tex_engine.cook(program_or_chain, bindings, *, device,
  compile_mode, precision, ...)`. Today the full engine (tier selection, fallbacks,
  OOM ladder, tiling, precision-auto) is only reachable through a ComfyUI v3 node
  classmethod — even `tex run` calls the node facade (tex_cli.py:101).
  `tex_node.execute` becomes marshal-in → `engine.cook` → marshal-out. Mechanical
  move (STR-2/3 already isolated the seams); gate: full suite + benchmark-neutral
  (invariant #7).
- **ENG-2 (M).** Standalone memory authority: `NullHostServices.get_free_memory`
  returns None so the Null host can neither tile nor retry — the same 8K cook that
  survives under ComfyUI just OOMs under tex_api. Use `torch.cuda.mem_get_info` in
  host.py + an engine-side OOM retry ladder (drop caches → retry tiled → optional CPU).
- **ENG-3 (S).** Egress profiles: `'comfy'` (today's clamp/alpha-drop/gray-expand,
  byte-identical, default) vs `'engine'` (raw fp32 BHWC, alpha kept, unclamped —
  scene-linear values survive node hops). Host-set, never per-node. The cheapest
  change with compositor-grade payoff; canary-pin the comfy profile.
- **ENG-4 (S).** Structured diagnostics at the API: `tex_api` wraps compiler errors in
  a public `TEXCompileError(diagnostics=[TEXDiagnostic])` instead of raw raises; the
  node's `TEX_DIAG:` string suffix stays for the existing JS. Canary-pin
  `TEXDiagnostic.to_dict`'s key set (it is already a de-facto frontend contract).
- **ENG-5 (S).** Embedding-contract canaries: pin the ui-payload keys, chain-spec
  fields (+ add a `schema` version field), `HostServices` method set; document
  fingerprint *instability* across TEX versions (hosts must not persist them). Add an
  "API stability tiers" table to DEVELOPMENT.md.
- **ENG-6 (S).** Zero-copy AI handoff contract (pillar 7): document + canary-test
  device residency / contiguous BHWC / fp32; `to_dlpack`/`from_dlpack` helpers;
  `layout='bchw'` zero-copy view option; document the `inference_mode` autograd caveat.
  Differentiable cooking stays out of scope until the engine era.
- **ENG-7 (S).** Host time context: `frame`/`fps`/`time` builtins fed from an
  ExecContext field (ComfyUI adapter: 0 or a wired INT; engine host: playhead).
  Must enter as *builtins*, not `$params`, or the lazy memo churns every frame.
- **ENG-8 (S).** Measured transfer-cost model: a once-per-process, disk-persisted
  PCIe bandwidth probe (pinned/pageable × H2D/D2H) exposing
  `transfer_ms(nbytes, pinned, direction)`. Feeds SCHED-2's edge costs and turns the
  v0.21 3-stream crossover gate (docs/xpu-transfer-scheduling.md) into arithmetic.
- **ENG-9 (M).** Concurrency audit of module-level runtime state: classify the 15
  caches (immutable-after-insert / device-keyed / per-worker), lock inserts only,
  keep the single-threaded fast path lock-free, add a two-thread CPU smoke test.
  Prerequisite for GRAPH-2 and XPU-1; finding this out during the engine build is
  the expensive way.
- **ENG-10 (S, docs).** Record **split-frame dual-device cooking** as a rejected
  decision (§7): the CPU↔GPU envelope (invariant #9) would put up-to-6.1e-2
  divergence on a visible mid-frame seam, output would depend on the split ratio
  (breaking run-to-run determinism), and the ceiling is a few percent. "Both" is
  honestly satisfied by branch parallelism + copy/compute overlap.
- **ENG-12 (M, design doc + canaries).** Buffer ownership & immutability contract:
  who may write a tensor after it's produced. Cached frames (CACHE-2) must be
  frozen — copy-on-write or version-stamp-and-verify at cache re-entry (codegen's
  M-5 `out=` reuse, a DO-NOT-TOUCH, must never scribble on a cached buffer);
  ENG-6's zero-copy views document that in-place mutation by an ML consumer voids
  the cache (or hand out cloned views by default); GRAPH-2's cross-thread edges and
  XPU-2's frame handles inherit the same rules. Undefined write discipline is the
  bug class that killed Natron's engine — this lands *before* CACHE-2 and GRAPH-2.

### Workstream D — language & authoring (pillars 6, 9)

- **LANG-1 (M).** Param UI metadata, additive grammar:
  `f$strength = 0.5 [min: 0, max: 2, label: "Strength"];` — literals only, parsed
  into new optional ParamDecl fields; type checker ignores them; the frontend
  auto-widget builder and (later) tool manifests consume them. The DCTL floor with
  Matchbox's layering to come.
- **LANG-2 (M).** Compile-only diagnostics API: `tex_api.check(source, types) ->
  list[TEXDiagnostic]` (never raises), a `/tex_wrangle/check` route, debounced
  live-lint in the editor, `end_line` on SourceLoc, and the first real W7xxx
  warnings (unused variable / unused input / param shadow). This is 90% of an LSP's
  backend; a `tex_lsp.py` stdio wrapper becomes a thin follow-up.
- **LANG-3 (M).** Language versioning as a mechanism, not a promise:
  `LANGUAGE_VERSION` constant exported via tex_api; optional `//!tex 0.20` pragma;
  a frozen compat corpus (the 116 examples + adversarial grammar programs with
  golden hashes, CPU-pinned) run in CI; `LANGUAGE.md` at the root (grammar summary,
  promotion rules, reserved words, written compat policy).
- **LANG-4 (S).** Finish DOC-4: the registry already carries `doc=`/`ex=` on
  143/144 functions (since v0.18) — the remaining work is generating
  `TEX_HELP_DATA` / a help JSON *from* the registry so the JS stops being
  hand-kept, and flipping the drift test to assert JS == registry view. This also
  gives the CLI and any future editor `tex help fn` for free.
- **LANG-5 (S).** Server-side user snippet store (JSON in a user dir via a
  `get_user_dir()` host seam); localStorage becomes an offline cache. Snippets are
  the sanctioned sub-tool reuse path — make them a real, git-versionable asset.
- **CACHE-0 (S, hygiene).** Disk-cache census: orphan `.cg` sidecars are invisible to
  eviction (~2,900 benchmark-harness orphans in the live cache dir today); add an
  orphan pass + total-size cap, and point the test/bench harness at a scratch
  `cache_dir` so it stops polluting the production cache.
- **ROI-1 (M, enabling substrate).** Replace the boolean `non_local` with an
  access-*footprint* descriptor in the `@stdlib` registry: `point` (default),
  `('halo', r)`, `('halo_arg', i)`, `'image'`, `('frame', i)`. Derive `_NON_LOCAL_FNS`
  as `footprint != 'point'` (existing consumers unchanged); extend TST-3 + the
  name-prefix heuristic to demand a classified footprint on sample*/blur/morphology
  names. Everything in §3's ROI program stands on this.

---

## 3. Mid term (v0.24–v0.3x): the five programs

### P1 — ROI / DoD (pillar 4's spatial dimension)

The single biggest interactivity lever a compositor has: cook only the pixels the
viewer requests. TEX is unusually close — the 1-D `tile=(y0, H_total)` machinery
already proves seam-exact coordinates and zero-copy view narrowing.

- **ROI-2 (L).** `tex_roi.py` — the spatial sibling of tex_lazy: per-binding
  footprint analysis on the lattice *identity ⊑ halo(dy0,dy1,dx0,dx1) ⊑ whole-image*,
  composing the existing pieces ($param folding from tex_lazy, affine offset/radius
  extraction from codegen_stencil, ROI-1's taxonomy). `where` branches **union**
  (both evaluate — same discipline invariant #11 codifies); anything unresolved →
  whole-image.
- **ROI-3 (L).** ROI execution: generalize `tile=` to `roi=(x0,y0,w,h,W,H)` in the
  interpreter; the executor slices each input to output-ROI ⊕ footprint; border-clamp
  only at true image edges. **Precondition (v0.23 audit):** ROI-1's `('halo_arg', i)`
  descriptors name the *argument*, not the true reach — `gauss_blur`'s halo is
  `ceil(3·sigma)`, ~3× the sigma the descriptor points at. Before the executor reads any
  radius, add the reach multiplier + a TST-3-style test pinning each `halo`/`halo_arg`
  descriptor to its impl's actual neighbour reach, or the slice under-approximates and
  seams appear at every tile edge (invariant #11 under-approximation). Exposed host-agnostically as `execute(..., roi=)` on
  PORT-2. Interpreter-tier first; compiled tiers fall back until ROI-bucketed keys
  exist (quantize rects to buckets so compile/graph caches don't explode).
- **ROI-4 (M, gate).** The fail-loud discipline does not carry over automatically: a
  wrongly-shrunk ROI ships wrong pixels *silently* — the worst failure class in the
  codebase. Three-part port: whitelist posture (unknown → whole-image), a
  differential ROI oracle lane (fuzz programs × random ROIs, assert ROI-assembled ==
  full-frame interpreter cook), pinned spatial never-sever rows mirroring
  test_lazy_cooking. **ROI ships nothing until this lane is green.**
- **ROI-5 (L).** Halo-aware tiling: today `is_tile_safe` refuses *any* non-local
  program, so an 8K `gauss_blur` cannot tile at all — the exact programs that
  dominate compositing VRAM. Strip tiling grows each input `narrow()` by the
  footprint radius; global reductions go two-pass or stay whole-frame. Route tiled
  cooks through codegen too (today interpreter-only). Windows/WDDM note: the strip
  planner must also cap estimated per-kernel *time* (~2 s TDR watchdog on display
  GPUs — reuse autotier's persisted medians), not just bytes; BlinkScript's driver
  timeouts are this failure, un-planned-for.
- **ROI-6 (M).** Temporal laziness groundwork: frame-window analysis (`fi - 1` →
  window [-1,0]) via ROI-1's `('frame', i)` footprints; interpreter batch-slice
  execution `(f0, B_total)` mirroring `tile=`'s seam-exactness (note
  `fetch_frame`'s clamp must clamp against B_total). Near-term payoff inside
  ComfyUI: batch-strip memory tiling for video cooks.

### P2 — results & caching (pillar 3's second half)

TEX persists *programs* superbly and *results* not at all — the standalone engine
ships with no way to hold a cooked frame.

- **CACHE-1 (M, keystone).** Lineage keys: stop content-hashing tensors mid-graph
  (the sampling hash has an admitted collision class — fine for cache-*busting*,
  deterministically wrong for cache-*reuse*). Every cooked output carries the key
  that produced it: `H(program fp × param values × upstream keys × frame ×
  device × precision/quality × env-epoch × flags)`. **Device and precision are
  mandatory key components** — invariant #9's envelope (up to 6.1e-2 cross-device)
  means ENG-2's CPU-retry and SCHED-2's placement changes must mint *new* keys and
  recook, never serve a cross-device hit; and the env-epoch (torch version +
  driver/GPU identity) keeps disk-spilled frames from surviving an update a recook
  would no longer reproduce. Content hashing survives only at graph ingress (file
  path + mtime; small stills). This is Nuke's op-hash, and the fused memo key
  already proves the pattern. Scope honesty: under ComfyUI its reach is
  TEX-internal edges (fused-stage handoffs, CACHE-6) — full lineage arrives with
  GRAPH-1, whose in-session *version counters* are the dirty signal while lineage
  keys are the persistence/disk identity; complementary, not competing.
- **CACHE-2 (L).** `tex_results.py` — the engine frame cache: RAM tier byte-budgeted
  through the tex_memory seam, disk spill staged through the existing pinned-egress
  helpers (spill/restore copies overlap cook time — pillar 5's mechanism, already
  written). Keyed by CACHE-1; keys carry a canvas/ROI descriptor from day one. The
  ComfyUI node does *not* enable it (the host already caches); armed by the engine
  host. Auto-placement heuristic from Nuke: cache at fan-out points and immediately
  upstream of the node being edited.
- **CACHE-3 (M).** Warm-tier persistence + prewarm: generalize the autotier.json
  pattern into `warm_state.json` (graph-capture verdicts/blacklists, backend
  probes); `tex_api.prewarm(programs, shapes)` for project-load/idle warm so first
  scrub after relaunch doesn't jank. CUDA graphs themselves can't serialize —
  persist the *decisions*, re-capture off the hot path.
- **CACHE-4 (M).** Layered cache epochs: today one mono-hash over 12 source files
  (plus the codegen-reuse env flag) keys every artifact — a comment-only stdlib edit
  cold-starts every user (fatal at monthly app updates). Split into AST_EPOCH /
  CODEGEN_EPOCH / VERDICT_EPOCH with the mono-hash demoted to a CI tripwire (any
  watched-file change without a recorded epoch decision is a red test). Because
  epochs move invalidation from over-approximate to human-judged, add a fail-safe:
  a CI/startup spot-check recompiles one persisted artifact per epoch and asserts
  oracle equivalence, so a wrong "no-bump" call is caught loudly instead of
  shipping stale codegen that drifts from the interpreter.
- **CACHE-5 (L).** The global cache governor: four independent budgets exist (stdlib
  tensors, CUDA graphs, compiled callables, inductor disk) and none sees the others.
  A `CacheRegistry` in tex_memory arbitrates the *per-device VRAM/RAM pools* against
  one budget with host-supplied priority hints (live-graph keys, playhead distance);
  disk tiers get separate size caps (extending CACHE-0), not per-device arbitration.
  Keys and lifecycles stay per-cache (the "15 caches are non-redundant" register) —
  only eviction *arbitration* centralizes. Registry-level correctness contract: any
  arbitrated eviction of a pool that captured CUDA graphs may reference must trigger
  `clear_graph_cache()` exactly as tex_memory does today (the stale-address safety
  in the DO-NOT-TOUCH register).
- **CACHE-6 (L).** Fusion ↔ caching reconciliation: a fused chain has no interior
  cut-points, so twiddling the last node's param recooks all N stages per tick —
  fusion currently makes interactivity *worse* exactly where it matters. Two levers:
  opt-in stage-boundary taps (materialize + memoize a designated handoff, keyed by
  the upstream sub-chain fingerprint), and suffix splicing (compile stages k..N
  reading the stage-(k−1) handoff from cache while a knob is hot; recook the full
  fused program on idle). The boundary tensor must be the exact fp32 handoff or the
  oracle contract breaks.

### P3 — tools (pillar 6's bundling half)

- **TOOL-1 (L).** The `.textool` manifest: `{name, tool_version, tex_language pin,
  stages (the exact Q-3 shape compile_fused consumes), promoted_params (stage,
  internal name, widget metadata from LANG-1), doc, category, author}` — all TEX
  code carried **inline** (§7 on why this is not the rejected import system).
  Loader validates via the existing chain_preflight; execution reuses
  prepare_fused + the fused disk cache unchanged. v1 constraints: internal edges
  image/mask-typed; unfusable constructs are tool-authoring errors (no sequential
  fallback mode); **no by-name tool nesting**.
- **TOOL-2 (M).** Publish flow: "collapse selection to tool" in the frontend writes
  the manifest; promoted-params picker (the gizmo/macro drag-promote UX); instancing
  renders one node whose widgets are the promoted params. Sealed vs open stays a
  serialization detail with cheap reversibility (Fusion's one-keyword lesson).
- **TOOL-3 (M).** Tool = compilation unit (strategic bet #6): publishing pre-fuses
  and warm-compiles the artifact keyed by the promoted-param signature. Warm keys
  are derived **at install time by re-fingerprinting the inline stage code** (the
  loader already runs chain_preflight, so sources are in hand) — never carried in
  the file, because fingerprints are deliberately unstable across TEX versions
  (ENG-5's persistence rule).
- **TOOL-4 (S).** `tex build tool.textool` CLI (Matchbox shader_builder's role):
  validate, type-check, report diagnostics, emit/refresh the manifest; context tags
  (generator / filter / transition / keyer) so hosts know where to surface a tool;
  min-engine-version so tools fail at install, not at cook.
- **TOOL-5 (S, design note — gates TOOL-4's install flow).** Threat model for
  shared tools: TEX codegen emits Python source from a user AST, so a downloaded
  .textool is untrusted input to a code generator. Required: an identifier/string
  sanitization audit + an adversarial-AST fuzz lane on the emitter, manifest schema
  validation before any compile, validate-only default at install (no
  compile-on-install without consent), and documented resource limits (a hostile
  tool can OOM/TDR a machine even without an emitter escape).

### P4 — scheduling & the graph IR (pillar 2's brain)

- **SCHED-1 (S).** Promote the `_tex_chain` spec to a documented, schema-versioned
  **GraphSpec** (add a `schema` field, document the stage shape hosts must emit).
  The detection half already moved to Python in FUS-1 — any host with a node graph
  gets fusion for free by emitting GraphSpec and calling the shared detector.
- **SCHED-2 (L).** Device-placement scheduler (`tex_scheduler.py`): topological DP
  minimizing Σ cook_cost(node, dev) + Σ transfer_cost(edge), subject to user pins
  and per-device memory budgets. Cook costs come from autotier's persisted medians
  (exactly the right shape, already on disk); transfer costs from ENG-8. Greedy
  stays the fallback and the correctness baseline. Output stability rules
  (invariant #9's envelope makes placement *visible*): device is part of every
  result key (CACHE-1), placement is **frozen per render range** — re-planning
  happens only at interactive/idle boundaries, never mid-sequence (§7 records
  mid-sequence placement migration as rejected) — and hysteresis damps interactive
  flapping. Distinct from rejected PF-4: this never changes *which tier* runs,
  only *where*.
- **SCHED-3 (M).** Cancellation + progress: a CancelToken checked at the natural
  yield points (per-strip, between tier attempts, before fp16 re-cook, per
  top-level statement in the interpreter), raising a typed `CookCancelled`;
  `on_progress(phase, frac)` on the same seam. Honest limit documented: an
  in-flight kernel can't be preempted — granularity is per-statement/per-strip.
  Aborting the stale cook when a newer edit arrives is *the* most-used code path in
  an interactive viewer.

### P5 — data model & sessions (pillars 7, 10)

- **DATA-1 (M).** Buffer metadata sidecar: `{colorspace: srgb|linear|oklab|unknown,
  premult: premultiplied|unassociated|opaque|unknown, frame, host-opaque keys}`
  carried per binding at the marshalling seam, re-attached at egress with an
  explicit merge policy (conflict → `unknown`, never a silent pick). Phase 2:
  W7xxx advisories when e.g. `gauss_blur` cooks an srgb-tagged buffer — the halo
  hazard the stdlib docstring already warns about in prose. Tags only; **no
  transforms** (§7 vs the OCIO rejection).
- **DATA-2 (L).** Storage-format descriptor + EXR: a `BufferDesc` (storage dtype,
  encoded range, transfer hint) resolved *only* at ingestion/egress — half/uint16
  are storage dtypes cast to fp32 exactly like uint8 today. New `tex_io/exr.py`:
  pure-torch scanline EXR (struct + zlib + `torch.frombuffer`; NONE/ZIP only, tiled
  EXR out of scope) because numpy-based OpenEXR bindings are banned by invariant #1.
  uint16 PNG write lands at the tex_cli seam PORT-4 already names.
- **DATA-3 (M).** ARRAY wire data under the engine profile: marshal float/int/vec
  arrays as `[N]`/`[N,C]` tensors with an explicit `a@name` ingress hint; the
  compiler-side rejection ("not representable in ComfyUI" — a host convention baked
  into tex_compiler) becomes profile-conditional. Curves, histograms, palettes flow
  between tools.
- **DATA-4 (M).** EngineSession phase 1: an object owning {TEXCache, Interpreter,
  decision caches, host services} with module-level singletons as views of a default
  session (ComfyUI byte-identical); a *written* thread-safety contract in
  DEVELOPMENT.md. Phase 2 (threading the session through engine.cook) follows ENG-1
  naturally. Companion CI lane: a **soak test** (thousands of cooks across tier
  transitions + session create/destroy, asserting flat RSS/VRAM watermarks) —
  ComfyUI process lifetimes hide slow leaks; a compositor runs for days.
- **LANG-7 (S).** Learnability past the near term: `tex_lsp.py` — a thin stdio
  JSON-RPC wrapper over LANG-2's `check()` + registry completion data (the backend
  is already built by then), and shipped offline reference docs (generated
  Error-Codes/Function-Reference served by a local route; `wiki_url_for_code`
  returns the local route when the frontend consumes it). An air-gapped box gets
  error-code docs; a standalone editor gets hover/completion/squiggles.
- **PORT-5 (M, milestone).** The proof artifact: `examples/host_demo.py` (~200 LOC,
  stdlib-only — an `http.server` viewer with a browser repaint loop, since the
  Windows embedded CPython ships **no tkinter**): a hand-built 3-stage GraphSpec
  (grade → blur → vignette), compiled via prepare_fused directly (no ComfyUI, no
  JS), one slider bound to a promoted param. Acceptance: **engine-side cook
  <50 ms/frame warm at 1024² on the sm_120 box, zero comfy imports** (run under
  the S-1 import blocker; display transport is excluded from the budget). The demo
  also *arms CACHE-2* (param-history scrub) so the frame cache gains a real
  consumer and acceptance test a horizon before GRAPH-1. This converts "the tests
  pass with comfy blocked" into "a second host exists," and it regression-guards
  ENG-1/SCHED-1/SCHED-3 forever.

---

## 4. Far horizon (the engine era): the standalone compositor host

Everything here presumes ENG-1/SCHED-1 shipped and PORT-5 proved the seam. These are
programs, not tasks — each needs its own design doc when its time comes.

- **GRAPH-1.** The pull-planner / push-executor (`tex_graph.py` / `engine/`): node
  DAG in; demand (outputs × ROI × frame range) propagated bottom-up using tex_lazy
  per node and ROI-2 footprints per edge; then the resolved cook compiles into a
  static execution plan (strategic bet #4). Dirty tracking by **version counters**
  the engine stamps on values it produced — no mid-graph hashing, no GPU syncs
  (the content-hash model exists *because* ComfyUI can't guarantee ownership; the
  engine can). Per-node result cache = CACHE-2 keyed by CACHE-1.
- **GRAPH-2.** Branch-parallel executor: ready DAG nodes dispatched to a small
  worker pool — CPU cooks on threads (torch CPU kernels release the GIL), GPU cooks
  serialized on one stream initially, cross-device joins via the existing
  ingest-event pattern generalized to edge events. Prerequisites ENG-9 and
  per-worker Interpreter instances land near/mid at low cost.
- **XPU-1.** Out-of-core tile executor + the 3-stream pipeline (the
  docs/xpu-transfer-scheduling.md v0.21 design, generalized): frame-of-record in a
  budgeted pinned host-RAM staging ring (replacing the 256 MB per-tensor pin cap
  that goes silent exactly at 8K scale), VRAM holds 2–3 strips,
  H2D(i+1) ∥ compute(i) ∥ D2H(i−1) with event chaining. Gated on ENG-8's measured
  crossover *plus* memory pressure. Its cross-node domain is the unfusable
  boundaries — for linear chains, fusion already wins by not transferring at all.
- **XPU-2.** Engine-owned async D2H egress: frame handles carrying CUDA events; any
  CPU consumer waits on the handle before first touch (the exact inverse of the
  shipped ingest fence). This deliberately reopens the doc's shelved
  "return-time transfer" idea — both recorded objections (ComfyUI API fragility,
  custody loss) are properties of the ComfyUI host and vanish when the engine owns
  every consumer. Unsafe under ComfyUI custody; engine-only.
- **DATA-5.** The Domain model (Blender's, essentially): a `TexImage` carrier =
  (tensor, DoD origin, canvas size, 3×3 transform, border policy). Transform nodes
  mutate the domain transform (cheap); realization is an auto-inserted TEX
  resampling program — so it fuses; transforms **concatenate** across consecutive
  nodes and resample once (Nuke's concatenation is the quality bar; Blender 4.2's
  destructive per-node realization is the documented artist complaint to avoid).
  Mostly-empty layers (text, roto, corner-pins) stop paying full-canvas cost.
  Strictly engine-internal until the standalone host exists; the ComfyUI adapter
  always flattens to dense IMAGE.
- **LANG-6.** Output-canvas declaration (`canvas(iw*2, ih*2);`): output resolution
  is currently welded to the first spatial input, so Transform-with-scale / Crop /
  Reformat — a compositor's fourth workhorse — are inexpressible in TEX and pillar 6
  caps out at "filters only." Threads through parser → type checker (must fold to
  scalars) → interpreter shape derivation; canvas-changing stages are initially a
  fusion break. Touches invariants #4/#11 and the tiling planner: design doc first.
- **DATA-6.** Named multi-plane buffers (AOVs, cryptomatte): do **not** widen
  TEXType past VEC4 (a VECN touches every module). A PLANES binding at the
  marshalling seam — `{plane_name: [B,H,W,C≤4]}` — expands one wire into ordinary
  per-plane bindings (`@src.N`, `@src.Z`) before compile, the same namespacing trick
  fusion already uses. Channel-demand laziness (only cook planes a program reads)
  falls out of tex_lazy unchanged. `layer.channel` naming verbatim for EXR interop.
- **DATA-7.** FrameProvider: out-of-batch temporal sampling
  (`fetch_time`/`sample_time`) through a host protocol beside HostServices, LRU'd in
  tex_memory, registered with `footprint=('frame', i)` (or `'image'` if unbounded)
  plus `sync=True` in ROI-1's post-boolean taxonomy — the derivations (graph-capture
  bar, tiling refusal) then follow from the footprint with no new machinery. Motion
  blur, temporal median, flow-warp; checkpoint-rate caching for stateful loops (the
  Copernicus Simulate lesson).
- **STOCK-1.** The stock-node library in TEX — the actual *content* of the
  compositor, and the largest body of work this doc implies (strategic bet #8:
  every expressible stock node MUST be TEX). Starts mid as `.textool` exemplars
  shipped with TOOL-1 (Grade, Blur, Merge, Vignette work today); the full first-20
  set (Transform/Crop/Reformat need LANG-6; premult-correct Merge wants DATA-1;
  keyers/AOV tools want DATA-6) is enumerated in its own design doc with each
  tool's blockers named. Dogfooding the compile path through the defaults is also
  the optimizer's best regression suite.
- **ML-1.** In-graph inference nodes (pillar 7's end state): a torch module as a
  graph node — weights-hash feeding CACHE-1 keys, tensors crossing on ENG-6's
  zero-copy contract, placement visible to SCHED-2. TEX's structural advantage
  over every cited competitor (Copernicus round-trips through geometry; Pybox
  round-trips through files): the buffers already *are* the model's input tensors.
  Differentiable cooking (training through TEX) remains out of scope until here.
- **ROTO-1 (design-doc-first).** Mask sources: a compositor without roto/paint/text
  can't do its core job, and in the standalone host there is no upstream ComfyUI
  graph to supply masks. v1 bet: host-side spline/paint rasterizers emitting MASK
  planes; investigated alternative: TEX-expressible SDF evaluation (which would
  fuse). The answer shapes DATA-5 and STOCK-1 — decide before §8's rung 3.
- **COLOR-1.** The OCIO decision, properly reopened: a shipping standalone
  compositor cannot avoid studio color management — the *engine-era* answer is the
  OCIO-v2-as-codegen-seam pattern (GpuShaderDesc's op list / baked LUTs translated
  into TEX IR: 3D LUT = grid_sample, matrices/curves = pointwise code — so viewer
  transforms *fuse*), with dynamic properties as uniforms so exposure/gamma tweaks
  never recompile. Until then, DATA-1 tags are the whole color story (§7).
- **ENG-11.** Engine lifecycle: `engine.warmup(device)` (splash-screen warm),
  `engine.shutdown()` (drain the compile pool, persist verdicts, free caches),
  cache-dir resolution (`TEX_CACHE_DIR` → user cache dir → package fallback) for
  pip-installed deployments.
- **PREC-1 (deferred decision).** Preview-tier fp16 *storage*: Blender ships half
  working precision on GPU for interactivity. The recorded rejection is unqualified
  — "Whole-pipeline fp16 & bf16 IMAGE — accuracy (bf16 err > the 8-bit quantum)" —
  so any carve-out must be argued in its own future decision doc, on its own
  merits, not pre-narrowed here. The shape that doc would examine: a quality-tagged
  preview tier (CACHE-2 keys already carry quality) storing color planes half /
  data planes fp32 during interaction only, full fp32 on idle re-cook. Not
  committed; the final-render contract is untouchable either way.

---

## 5. Pillar scorecard

| Pillar | Today (v0.20) | Near | Mid | Far |
|--------|---------------|------|-----|-----|
| 1 Compilation | linear chains fuse; compile tiers verified (but FUS-0 bug) | DAG regions, fused lazy | tools as compile units, suffix recook | whole-plan compilation in tex_graph |
| 2 XPU | per-node greedy placement; single-source language | transfer probe, concurrency audit | placement scheduler, GraphSpec | branch-parallel executor |
| 3 Caching | program caches persistent & layered; zero result caching | speculative warm (LAT-1), hygiene (CACHE-0) | lineage keys, frame cache, epochs, governor, prewarm (CACHE-3) | version-counter dirty tracking |
| 4 Lazy | input-level, over-approximate, dual-consumer | footprint registry, fused-chain lazy | ROI/DoD analysis + execution + oracle | demand-driven pull planner |
| 5 Unified memory | pinned egress + async ingest, budget ladder | Null-host authority | halo tiling, spill-through-pinned | 3-stream out-of-core, async D2H handles |
| 6 Wrangling | one node, snippets, fusion substrate | param metadata, snippet store | .textool + publish flow + tex build + first exemplars | canvas decl, stock-node library (STOCK-1), roto/paint sources (ROTO-1) |
| 7 AI-ready | torch-native, device-resident | DLPack contract (ENG-6) | ARRAY wires (DATA-3) | in-graph inference nodes (ML-1) |
| 8 Low overhead | 4 measured tiers, ~15 memos | bg compile everywhere, PreparedProgram, deferred timing | ROI cooks, result reuse | version-counter invalidation, trusted interior edges |
| 9 Learnability | error codes, hints, HUD, doctor | check() API, W7xxx, live lint, DOC-4 | LSP + offline docs (LANG-7) | in-app manual on LANG-7's pages; generated-code inspector (an emitted-source viewer over codegen's flat fn — Copernicus's best UX idea) |
| 10 Versatile | IMAGE/MASK/LATENT/STRING/INT/FLOAT, 144 fns | time builtins | metadata tags, EXR/half storage, arrays | planes/AOVs, FrameProvider, Domain |

---

## 6. Dependency spine

The critical path, in order — most items off it can float. Note the ROI chain
distinguishes *build* order from *ship* order: ROI-3 is built first (behind a flag,
interpreter-tier) because ROI-4's differential oracle needs ROI execution to exist
before it can gate it.

```
FUS-0 → FUS-1/FUS-3 (regions proven) → TOOL-2 (collapse selection = a region)
ROI-1 → ROI-2 → ROI-3 (built, flagged off) → ROI-4 (oracle gate) → ship ROI-3 + ROI-5
ENG-1 (engine.cook) → SCHED-1 (GraphSpec) → PORT-5 (demo) → GRAPH-1 (executor)
ENG-12 (ownership) + CACHE-1 (lineage) → CACHE-2 (frames) → CACHE-6 (suffix recook)
ENG-8 + ENG-9 → SCHED-2 (placement) → GRAPH-2 / XPU-1     ROI-5 → XPU-1
LANG-1 + LANG-3 (version pin) → TOOL-1 → TOOL-2/3 (TOOL-3 also needs CACHE-3)
DATA-4 (sessions) → GRAPH-2         ROI-2 + CACHE-2 → GRAPH-1
```

Rules of thumb: nothing in P1 ships before ROI-4's oracle lane is green; nothing in
§4 starts before PORT-5 passes; every item lands behind invariant #7 (full suite +
benchmark-neutral on the default path).

---

## 7. Decision-register deltas

**New rejections to record in DEVELOPMENT.md** (this roadmap is their provenance):
- *Split-frame dual-device cooking* (ENG-10): envelope divergence on a visible seam,
  split-ratio-dependent output vs the determinism pin, single-digit ceiling.
- *Mid-sequence placement migration* (SCHED-2's temporal twin of ENG-10): moving a
  node between devices between frames of one render range changes pixels by up to
  the invariant-#9 envelope — temporal popping. Placement freezes per render range;
  re-planning happens at interactive/idle boundaries only.
- *A second "fast" kernel language / dialect* (strategic bet #1): the Copernicus
  VEX/OpenCL trap. Device placement is a scheduler concern, never an authoring one.
- *Scanline or tile-granular execution core* (bet #3): stripes are a memory-streaming
  tactic under the full-frame model, not a scheduling model.
- *Recursive pull executor with per-node locking* (Natron post-mortem): plan lazily,
  execute a static compiled plan.

**GPU vendor strategy (decided now so the seams don't bake CUDA in):** engine v1
targets **CUDA + CPU**. The interpreter and codegen tiers are the portable floor —
torch MPS/ROCm inherit them for free the day someone runs there — while the
cuda_graph tier, pinned 3-stream pipeline (XPU-1), and event-carrying frame handles
(XPU-2) are CUDA-only *accelerators*, never correctness dependencies. Concretely:
ENG-2/ENG-8 route every device query through host.py so a second backend is
additive; nothing outside the accelerator tiers may import a `torch.cuda.*` API
without a capability check. A cross-vendor port is a sanctioned future project, not
a v1 promise.

**Adjacent to rejected decisions, explicitly NOT reopening them:**
- **Tools ≠ the cross-node import system.** A `.textool` carries every stage's code
  inline; sharing a tool is sharing one self-contained plaintext file; no program's
  compilation ever resolves an external name. The line that *would* reopen the
  rejection — tools referencing other tools by name — is excluded from v1 by design.
- **Metadata tags ≠ ACES/OCIO.** DATA-1 is tagging only: no transforms, no configs,
  no LUTs; conversions remain the user-called srgb/oklab functions. The tag seam is
  what lets the OCIO rejection *stand* without closing the compositor door.
- **Half/uint16 storage ≠ whole-pipeline fp16 IMAGE.** Storage dtypes convert to
  fp32 at ingestion exactly like uint8 today; compute and wire stay fp32.
- **Suffix splicing / stage taps ≠ the extra fusion wire.** No new user-visible
  topology; the frontend collapse is untouched — only backend recompile granularity
  and an internal materialization point are added.
- **SCHED-2 ≠ PF-4.** PF-4 was *tier* choice on CPU; the scheduler chooses *node
  placement* and never changes which tier runs.

**Sanctioned future reopenings** (each requires its own decision doc at the time):
- *Engine-owned async D2H* (XPU-2) — the shelved return-time-transfer objections are
  ComfyUI-host properties; reopen only under engine custody.
- *OCIO integration* (COLOR-1) — engine era, seam pattern above.
- *Preview-tier fp16 storage* (PREC-1) — the recorded rejection is unqualified; any
  carve-out is argued fresh.
- *By-name tool nesting* (TOOL-1's excluded v2 feature) — tools referencing other
  tools by name is by-reference composition, i.e. the rejected import system's
  ethos question re-asked at the tool layer. It may never happen; if it does, it
  goes through this door, not around it.

**Doc placement:** this file lives in `docs/`, which DOC-6's two-layer policy does
not name — but practice already put design docs there (xpu-transfer-scheduling.md,
live-session-checklist.md). Landing this roadmap should ride with a one-line DOC-6
amendment naming `docs/` as the third layer: internal design/planning documents
that would bloat the root.

**Invariants: all eleven stand.** Two get *extended*, not weakened: invariant #5's
tag-derivation discipline generalizes to footprints (ROI-1), and invariant #11's
over-approximation discipline is ported to the spatial lattice with a new oracle
lane because ROI failures are silent where lazy failures were loud (ROI-4).

---

## 8. Product sequencing & proof milestones

Climb Copernicus's ladder (bet #10) — each rung is shippable and self-funding:

1. **Better ComfyUI node** (near term): everything in §2 makes today's product
   faster and the seams honest. Users see: fused DAGs, fewer compile hitches, live lint.
   (v0.21 ships the fused DAGs; the compile-hitch work is only partly landed — LAT-1a
   is deferred, since `torch.compile` is lazy and a background *wrap* doesn't hide the
   first-*execution* stall. See §9.)
2. **Stills / procedural texturing host** (PORT-5 → a real minimal app): single
   frame, ROI viewport, tools. This is where Copernicus won first.
3. **Slap comp**: frame cache + scrubbing + metadata tags + EXR I/O.
4. **Video comp**: FrameProvider, temporal laziness, out-of-core, OCIO — plus the
   headless final-quality mode (engine.cook over frame ranges with pinned placement,
   the same call a render farm would make).

Proof milestones (each converts a claim into a regression test):
- **PM-1**: FUS-3 equivalence suite green on branching regions (pillar 1 is real).
- **PM-2**: PORT-5 demo cooks <50 ms/frame warm at 1024², zero comfy imports, and
  scrubs param history from CACHE-2 (a second host exists, and the frame cache has
  a consumer).
- **PM-3**: app-relaunch cold start for a 100-program project < 2 s to first frame
  (CACHE-3), no re-trial jank.
- **PM-4**: compat corpus green across a minor version bump (LANG-3 — "v0.25 runs a
  v0.20 tool" is a mechanism, not a promise).
- **PM-5**: an 8K `gauss_blur` cooks on the 12 GB box via halo tiling (ROI-5), and a
  50-node 4K graph stays under budget with the governor (CACHE-5).

---

## 9. Version pencil (tentative — re-cut at each release)

Penciled against the repo's release conventions: one **theme** per release, items
grouped so each release is independently shippable and benchmark-gated, phase test
files (`tests/test_vXXX_phaseN.py`), and the live-session checklist for anything
touching the frontend. FUS-0 ships as **v0.20.1** (hotfix, in flight).

| Version | Theme | Items | New modules |
|---------|-------|-------|-------------|
| v0.21.0 | **Fuse the graph** — fusion real on internally-branching regions | FUS-1, FUS-3, FUS-2 (mechanism), ~~LAT-1a~~ (deferred: compile is lazy), LAT-3, LAT-4, CACHE-0, ENG-8, ENG-10 (doc) | `tex_runtime/xfer.py` |
| v0.21.1 | **Multi-injection** — an external producer may feed >1 region member (`Load → [blur, sharpen] → merge`), the canonical comp split v0.21 leaves unfused | FUS-1b (source spec becomes a list: splicer + JS transport), FUS-1c (see below) | — |
| v0.22.0 | **The engine seam** — the cook engine stops being a ComfyUI classmethod | ENG-1, ~~LAT-2~~ (deferred: measured at 2% of a cook; see below), ENG-2, ENG-3, ENG-4, ENG-5, ENG-7, SCHED-1 | `tex_engine.py` |
| v0.23.0 | **Authoring** — the language grows its tool-era surface | LANG-1, LANG-2, LANG-3, LANG-4, LANG-5, ENG-6, ENG-9, ROI-1, ~~LAT-1b~~ (deferred: one-time capture stall; see below) | `tex_snippets.py`, `LANGUAGE.md`, `tex_help.json` |
| v0.24.0 | **See less, cook less** — spatial laziness | ROI-2, ROI-3 (flagged), ROI-4 (ship gate), ROI-6 | `tex_roi.py` |
| v0.25.0 | **Remember frames** — results become first-class | ENG-12, CACHE-1, CACHE-2, CACHE-3, CACHE-4 | `tex_results.py` |
| v0.26.0 | **Tools** — the bundling promise | TOOL-1..5, first STOCK exemplars, LANG-7 | `tex_tool.py`, `tex_lsp.py` |
| v0.27.0 | **Big frames, placed well** | ROI-5, CACHE-5, CACHE-6, SCHED-2, SCHED-3 | `tex_scheduler.py` |
| v0.28.0 | **Second host** — the proof release | DATA-1, DATA-2, DATA-3, DATA-4, PORT-5 (PM-2) | `tex_io/exr.py`, `examples/host_demo.py` |
| v0.3x → v1.0 | **Engine era** | §4 programs (GRAPH/XPU/DATA-5..7/STOCK/ML/ROTO/COLOR), each behind its own design doc | `tex_graph.py`, `tex_runtime/streams.py` |

Per-release notes:

- **v0.21.0** honors the standing promise in `docs/xpu-transfer-scheduling.md` (the
  "v0.21 follow-up") the honest way: ENG-8 ships the *measurement* the 3-stream
  design demands, and the release records the go/no-go verdict in that doc. The
  pipeline itself stays gated (it is XPU-1, engine era) unless the numbers say
  otherwise. FUS-1's JS work makes this a live-checklist release. Exit: PM-1
  (FUS-3 equivalence suite green on branching regions) + the fused-chain compile
  speedup reachable end-to-end from a real graph.
- **v0.21.1 (FUS-1c)** collects the fusion limits v0.21 knowingly shipped, none of
  which risk a wrong pixel — each is coverage or hygiene, and all were confirmed by
  reading the code, not assumed:
  - *Linear-first interaction* (larger than it looks — traced through the code, not
    assumed): `graphToPrompt` runs the proven linear collapse first, and the two passes
    do not coordinate. `_texSerializeGraph` reads **litegraph**, which still holds the
    nodes the linear pass just deleted from the prompt, so the detector happily returns
    a plan naming them — and `_texCollapseRegion`'s "whole region present?" check then
    rejects that plan **wholesale**. So in `S→A→B→{C,D}→E` only the `A→B` head fuses;
    C, D and E cook separately — the region does not partially fuse, it does not fuse at
    all. Any linear run of ≥2 leading from a region's external source into its first
    fan-out defeats that region's fusion entirely. An *interior* run (`S→A→{B1→B2, C}→D`)
    forms no collapsible chain and region-fuses normally. This is lost coverage, never a
    wrong pixel, and not a regression (v0.20 fused exactly the same head). Keeping the
    proven pass first is the safe order; unifying them means the region pass subsumes
    linear, which needs the linear path's live-checklist evidence re-earned.
  - *Preflight cost*: `_region_compiles` compiles each region once per candidate source
    shape (2 for IMAGE) and persists artifacts for shapes that may never cook. Bounded
    and reclaimed by CACHE-0's orphan census, but a shape-keyed structural memo would
    skip the repeat. Note this is per **queue**, not per param tweak — region detection
    is only reachable from `graphToPrompt` — so it is nowhere near the interactive path.
  - *Cache hygiene*: orphan `.cg` eviction ranks by write-time rather than last-use, and
    the 600 s orphan grace exceeds the 300 s census throttle (transient bloat that
    self-heals). `TEX_CACHE_DIR` is resolved once at first `get_cache()`, so a host that
    re-points it mid-process keeps the old dir — fine for the test/bench harness, worth
    revisiting when ENG-11 defines a real resolution order.
  - *Multi-GPU*: `xfer`'s `_MODEL` is not device-keyed, so a second GPU with different
    PCIe characteristics would read the first one's fit.
- **v0.22.0** is deliberately *one big mechanical move plus small seams*: ENG-1 is
  the release. Exit: the S-1 import-blocker suite green, `tex run` rerouted through
  `engine.cook`, benchmark strictly neutral (invariant #7 — a refactor release must be
  invisible). **All met.** ENG-1 measured at +1.3 µs/cook (O(1), +0.19% at 1024² CUDA)
  and the whole-suite PASS set is identical to v0.21.0's.
  - **LAT-2 was DEFERRED on measurement, and the number re-cuts the item.** The plan was
    for LAT-2 to ride ENG-1 (PreparedProgram building on the freshly-moved ExecContext
    seam). Measured after the move, on a 256² CUDA cook (263 µs): `prepare()` is 66.3 µs,
    of which **the M-1 preflight's `get_free_memory` is 61.0 µs — 92%** — and everything
    LAT-2 proposed to cache is **5.27 µs, 2.0% of the cook**. So LAT-2 would mint a new
    (fingerprint × binding-signature × device × precision) cache — the silent-staleness
    class ENG-7 had to defend against in four separate places — to chase 2%, standing
    next to a 61 µs query it doesn't touch. Not measured, so not claimed: the compile
    tiers' per-cook env/capture-key building (LAT-2's other half), which sits off the
    default path invariant #7 governs. Re-open LAT-2 only with a measurement showing the
    scans matter on a path someone actually uses.
  - **LAT-5 (S/M, NEW — the target LAT-2 was aiming past).** The M-1 memory preflight
    queries free VRAM on *every* CUDA cook, and that query is 92% of `prepare()`
    (ComfyUI's own `get_free_memory` ≈ 44–61 µs; TEX's own probe is ~8 µs since ENG-2).
    **The 61 µs above still undercounts the shipped path, and the review that caught it is
    the reason this item exists at its current size:** `_tile_plan` independently bought a
    *second* free-memory reading in `run()`, microseconds after `prepare()`'s, so a default
    CUDA cook of a tile-safe program paid **2 × ~68 µs = ~42% of a 345 µs 256² cook**
    (independently re-measured at 48.3% by stubbing the query). v0.22 fixed the redundancy
    — the plan carries the preflight's reading and `_tile_plan` reuses it, guarded by a
    `preflight_freed` flag because asking the host to unload *raises* true free and makes
    the number stale-low. Measured after: `get_free_memory` 2 → 1 per cook.
    Do NOT extend that to the *estimate*, which looks like the same redundancy and is not:
    the two sites pass different `dtype_bytes` (the preflight runs before `auto` resolves,
    so it passes 4; `_tile_plan` passes 2 once auto→fp16), and reusing it hands `_tile_plan`
    a 2× inflated peak that over-tiles precisely the cooks `auto` accepted. It is ~6% of
    the cost anyway.
    What remains for LAT-5 is the *last* query. It is live state, so it cannot be memoized
    — the lever is *skipping* it: the estimate is usually orders under free VRAM, and the
    query only decides whether to ask the host to unload models. A cheap "obviously fits"
    pre-check (against a cached device total, say) would drop it from the interactive path
    entirely. Needs its own design note: the failure mode of skipping wrongly is an OOM a
    preflight would have prevented, so the cheap check must be conservative in the right
    direction. Scope honestly: this is CPU-side latency on a launch-bound async cook — at
    2048² real GPU work dominates and the fixed cost falls to ~7%, so the win concentrates
    at ≤1024², which is exactly the interactive/proxy path (and the small animated cooks
    ENG-7 just enabled).
  - **ENG-4's split is worth re-cutting before LANG-2 lands** (raised by the v0.22
    review). This doc put `TEXCompileError` at *tex_api*, but `tex_node` and `tex_cli`
    each catch the raw per-phase set independently, so three modules now know that tuple
    — and `tex_engine.cook`, the advertised host-agnostic entry, still raises raw
    internal compiler types. The class was landed in `tex_compiler/diagnostics.py`
    (shared) precisely so the *raiser* can move down to the engine later without the type
    moving. Decide in v0.23, when LANG-2's `check()` gives the second consumer.
- **v0.23.0** shipped the authoring surface: LANG-1 param metadata, LANG-2 `check()` +
  the first W7xxx warnings + live lint, LANG-3 language versioning + the frozen compat
  corpus + LANGUAGE.md, LANG-4 registry-sourced function help (+ `tex help` CLI), LANG-5
  server-backed snippets, ENG-6 DLPack handoff, ENG-9 per-thread interpreters, and ROI-1
  (pure registry metadata, zero behaviour change) so v0.24 starts on a settled substrate.
  **Exit met:** PM-4 — the compat corpus is green against v0.23-frozen goldens (129
  programs: 116 examples + adversarial grammar), the mechanism that keeps a v0.22 program
  computing the same pixels on later versions.
  - **LAT-1b was DEFERRED on measurement** (the LAT-1a / LAT-2 precedent). The synchronous
    CUDA-graph capture stall is a **one-time ~72 ms** on the first cook of a graph key
    (1024², sm_120); the capture key excludes `$param` values, so an interactive scrub
    captures once and every later frame replays at 0.34 ms. Async capture's real cost is
    serializing *all* foreground device cooks against the capture window (CUDA capture
    forbids concurrent device work) — a whole-cook lock through the engine seam and a
    DO-NOT-TOUCH tier, plus a GPU soak test — which a one-time-per-key stall does not
    justify. The design + the reopen gate (repeated capture stalls on an interactive path)
    are in `docs/lat1b-async-graph-capture.md`. Landing it as a mini-design is the
    roadmap's own §10.1 "design note first for L/XL items" discipline.
- **v0.24.0** shipped spatial laziness: ROI-2 (`tex_roi.py` footprint analysis on the
  point ⊑ halo ⊑ image lattice), ROI-3 (`roi=(x0,y0,w,h,W,H)` execution — the M-4
  `tile=` machinery generalized to a 2-D window and normalized to one seam-exact path;
  `tex_memory.run_roi` narrows to `ROI ⊕ H` + crops), ROI-4 (the differential oracle
  ship-gate: reach-pinning + spatial never-sever rows + fuzz `maxdiff < 1e-5`), and ROI-6
  (temporal groundwork: `frame_window`/`batch_sliceable` + the interpreter `batch_slice=`
  + `run_batch_strips`). **Exit met:** the differential ROI fuzz lane is green (CPU + CUDA)
  and no ROI cook ships wrong pixels silently.
  - ROI-3 lands **flagged off** (`TEX_ROI_EXEC`, default off; no ComfyUI cook passes
    `roi=`). The flag flip to a production viewport is a later release, gated on the fuzz
    lane green across a nightly run *and* a host that asks for a sub-region.
  - **The reach-multiplier precondition** (v0.23 audit) is closed: `gauss_blur`'s
    `('halo_arg', 1, 3.0)` descriptor carries the `3·sigma` reach; the ROI-4 reach-pinning
    test pins every halo descriptor to its impl's measured neighbour reach.
  - v1 scope honestly recorded (`docs/roi-spatial-laziness.md`): ROI executes the point +
    direct-tensor-halo (blur/morphology) class; gathers/reductions cook whole-frame because
    `fn_sample`/`fn_fetch` size their output from the input image, not the coordinate grid —
    a decoupled gather output grid is **ROI-5**. Cross-frame temporal reads are the batch
    twin of that limitation, also ROI-5-era. The bit-exactness is exact for pointwise +
    integer morphology, ~1 ulp for conv/bilateral (size-dependent kernel dispatch).
- **v0.25.0** shipped results as first-class: ENG-12 (the buffer ownership & immutability
  contract — a cook output is born frozen; a frame cache stores it frozen or version-stamps a
  normal one and re-verifies at re-entry) landed first, before any frame is cached; CACHE-1
  (`tex_results.lineage_key` — device/precision mandatory, tensor inputs by upstream key not
  pixels, env_epoch carries the CACHE-4 codegen epoch); CACHE-2 (`tex_results.ResultCache` — RAM
  byte-budget + disk spill through the pinned helpers, frames frozen, keyed by CACHE-1, **armed
  by an engine host, not the ComfyUI node** — the ROI-3 posture); CACHE-3 (`warm_state.json` +
  `tex_api.prewarm` — the graph-capturability verdict persists so a relaunch skips re-walking the
  gate; backend probes / blacklists are deliberately not persisted as inert or transient); CACHE-4
  (the mono-hash split into the nested AST ⊑ CODEGEN ⊑ VERDICT
  epoch lattice — a codegen-only edit no longer cold-starts the `.pkl` tier). New modules
  `tex_results.py`, `tex_runtime/warm_state.py`; design in `docs/results-caching.md`.
  **Exit met — PM-3 dry-run measured:** on the sm_120 box, relaunching a 100-program project
  against a prewarmed cache cooks all 100 first-frames in ~116 ms (vs ~226 ms cold/empty — the
  persisted `.pkl`/`.cg` + warm_state roughly halve the project warm cost), with a to-first-frame
  cook of ~13 ms; the ~4 s of wall-clock is torch's own import, a fixed host constant identical
  warm-or-cold and outside TEX's budget. Well under the 2 s TEX-side target, no re-trial jank.
- The default ComfyUI cook path is byte-identical (invariant #7): no watched compiler/runtime
  file was touched, so existing program disk caches survive the upgrade apart from the one-time
  CACHE-4 version-format reset; the lineage attach is gated on `want_lineage` (off by default),
  the epoch split runs once at import, and the warm-state hooks live only on the compile/graph
  tiers. The frame-cache flag flip to on-under-ComfyUI is a later release, gated on a host that
  owns its downstream consumers (ENG-12) + GRAPH-1's demand signal.
- **v0.26.0** shipped the bundling promise: TOOL-1 (`tex_tool.py` — the `.textool` manifest +
  loader, schema-validated before any compile, promoted-param → per-stage mapping, single-stage
  + fused cook paths), TOOL-2 (the publish backend — `write_tool` on the `get_user_dir` seam,
  `/tex_wrangle/publish_tool` + `/tex_wrangle/list_tools`, and a "Publish as TEX tool…" node
  command), TOOL-3 (warm keys re-fingerprinted at install from the inline code, never stored —
  ENG-5; `install_tool` validate-only by default), TOOL-4 (`tex build`), TOOL-5 (the threat
  model: validate-only install, schema-first validation, the emitter injection audit + the
  adversarial-AST fuzz lane, resource limits), the first STOCK exemplars (Grade/Blur/Merge/
  Vignette + a fused `GradeVignette` under `stock/`), and LANG-7 (`tex_lsp.py` — the stdio LSP
  over `check()` + the registry, plus the offline-docs route). New modules `tex_tool.py`,
  `tex_lsp.py`; design + threat model in `docs/tools.md`. **Exit met:** a `.textool` round-trips
  author → publish → install → cook, bit-identical to the unfused graph
  (`test_tool_roundtrip_unfused`, maxdiff 0.0 CPU + CUDA).
  - The TOOL-5 audit found the emitter was **already** injection-safe (lexer is ASCII-only so
    confusables can't tokenise; identifiers are namespace-prefixed; strings are `repr()`'d;
    the type checker rejects unknown functions before codegen; `_torch`/`_math` dispatch a
    whitelist) — so **no watched compiler file was touched**, and invariant #7 holds. The fuzz
    lane pins it: 6 hostile programs rejected pre-codegen, 6 benign-hostile emit no dangerous
    call/attribute/import (verified by `ast`-walking the generated source — string data is
    `ast.Constant`, so only real code trips a finding).
  - The frontend **collapse-selection picker + instanced-tool-node rendering** land as pure JS
    and are verified in a running ComfyUI (the live-session checklist); the manifest they
    exchange, the publish route, and the tool cook are backend-proven here. Recorded honestly
    in `docs/tools.md` §8 as the follow-up.
- **v0.27.0** shipped "big frames, placed well": ROI-5 (`tex_memory.run_tiled_halo` — the
  blur/morphology class `is_tile_safe` refuses now tiles with a halo-grown strip, driven over
  `run_roi`'s ROI-4-validated grow-cook-crop, so an 8K `gauss_blur` tiles bit-exact; `_halo_tile_plan`
  gates it on memory pressure OR a `_tdr_strip_floor` WDDM ~1.8 s per-strip time cap from autotier
  medians; reductions/gathers stay whole-frame), CACHE-5 (`tex_memory.CacheRegistry` arbitrating the
  stdlib/graph/frame pools against one `governor_budget`, cheapest-to-rebuild first, preserving the
  graph-address safety via pin-skip / `free_graphs_only` — NOT `clear_graph_cache`, the stale note
  now corrected; `ResultCache.governed_bytes`/`evict_bytes` fold the frame cache in, host-armed),
  CACHE-6 (`tex_fusion.prefix_fingerprint`/`suffix_stage_list` + `tex_engine.cook_stage_list`/
  `boundary_lineage_key`/`cook_fused_cached` — a stage-boundary tap caches the fp32 handoff keyed by
  the upstream sub-chain fp × prefix param values × the host-supplied `upstream` source key, a
  suffix splice recooks only stages k..N while a downstream knob is hot; fp32-gated, requires a
  source key to cache (else a correct full cook), linear-v1, DAG deferred), SCHED-2 (`tex_scheduler.py` — a
  Viterbi-DP-for-chains / exact-enumeration-for-small-DAGs / greedy-fallback placement planner over
  autotier `cook_ms` + `xfer.transfer_ms` with boundary transfers, user pins, per-device budgets,
  hysteresis, frozen-per-range; PORT-1-clean, dormant), and SCHED-3 (a `CancelToken`→`CookCancelled`
  polled at every cook yield point + `on_progress`, on the ENG-7 value channel, never keyed). New
  module `tex_scheduler.py`. **Exit met:** ROI-5's halo oracle (tiled == whole-frame, CPU+CUDA) and
  CACHE-6's oracle (suffix-splice == full fused, maxdiff 0.0) are green.
  - The default ComfyUI cook path is byte-identical (invariant #7): no watched compiler/runtime file
    changed the default path. Halo tiling is a new branch reached only under pressure on programs
    that could not tile before; `enforce_cache_budget` is byte-identical (the governor is a separate
    opt-in layer); the scheduler, tap, and cancel/progress are opt-in / host-armed / None-by-default.
  - **Routing tiled cooks through codegen is DEFERRED** (honest scope, the ROI-3 posture): tiled +
    ROI cooks run on the interpreter tier, compiled tiers fall back. The memory-correctness win (an
    8K blur tiles at all) lands here; a codegen tile/ROI path threading the strip offset through the
    coordinate-env builder is the measured follow-up.
  - **CACHE-6 v1 is LINEAR-only.** A `chain_inputs` DAG is not suffix-split (positional chain-input
    rebasing across a cut is the deferred half); it recooks whole — correct, just not incremental.
    The interactive case CACHE-6 targets ("twiddle the last node's param") is a linear chain. It
    also REQUIRES a host-supplied `upstream` source key to cache (a raw `data_ptr` is unsafe under
    buffer reuse) — no key → correct full cook.
  - **Scope notes shipped honestly, for symmetry with the above deferrals:** CACHE-5's
    "live-graph-key" priority hint is realized by MECHANISM, not a hint parameter — graph pools
    evict LAST (`evict_order=90`) and any graph-pinned storage is pin-skipped (`pinned_storages`),
    so a live capture's memory is preserved without a separate key; only `playhead` is a passed
    hint. The WDDM TDR per-strip time-cap is on the halo planner only — the pointwise `_tile_plan`
    stays byte-only (a per-cook median is a no-op on the first, TDR-prone cook, so a code cap there
    would buy nothing; the halo class is the one that both tiles new AND runs long). SCHED-2's
    per-device memory budget is CUDA-only (CPU is the unbudgeted fallback), and SCHED-3
    cancellation is reachable via `tex_engine.cook(cancel=)` only — the ComfyUI node does not yet
    bridge the host's own interrupt (a pure-frontend follow-up).
- **v0.28.0** shipped "second host": DATA-1 (`tex_marshalling.BufferMeta` — colour/alpha/frame tags
  on the `ExecContext` value channel, merge-to-`unknown` on conflict, never keyed; the W7005
  gamma-halo advisory `tex_api.color_advisories`), DATA-2 (`tex_io` — `BufferDesc` + a pure-torch
  scanline EXR reader/writer and a 16-bit PNG writer, half/uint16 as storage dtypes cast to fp32 at
  the seam), DATA-3 (ARRAY host wires under the engine profile — `a@name` inputs + array outputs as
  `[N]`/`[N,C]`, gated by a set-once flag with the comfy egress as the always-on guard), DATA-4
  (`tex_session.EngineSession` phase 1 — one handle over the module singletons as views, + the soak
  lane), and PORT-5 (`examples/host_demo.py`). New package `tex_io/`, new module `tex_session.py`.
  **Exit met — PM-2:** the demo cooks **~1.4 ms/frame warm at 1024²** on the sm_120 box (target
  <50 ms; display excluded), zero comfy imports (green under the S-1 blocker), scrubbing param history
  from a host-armed CACHE-2 frame cache.
  - **DATA-3 v1 is output + builtin-consumption + pass-through only.** Direct element indexing
    `a@name[i]` (the parser routes a `BindingRef` index to image-fetch, not array-index) and
    per-element vec typing of an INPUT array are the deferred half — v1 already lets curves/palettes
    flow between tools (produce an array, or receive one and `arr_avg`/`sort`/`len` it). The flag is
    deliberately NOT fingerprinted (the comfy egress rejects ARRAY regardless, so caches survive).
  - **DATA-4 phase 2 is deferred** (it needs the ENG-1 cook-signature change): threading a session
    through `engine.cook(session=…)` so an ISOLATED session owns its own caches. Phase 1 is one
    process-default session whose caches ARE the module singletons — the MUT-cache sharding for a
    parallel executor (GRAPH-2) inherits ENG-9's boundary, which the session handle names, not moves.
  - **EXR scope (honest):** scanline only — NONE/ZIPS/ZIP, HALF/FLOAT (UINT read-only). Tiled,
    multipart, deep, and the lossy codecs (PIZ/PXR24/B44/DWA) raise a clean error rather than
    mis-decode. Little-endian hosts (every target). Cross-validated zero-error against OpenCV.

## 10. How an item lands (the implementation mapping)

Every roadmap item follows the same lifecycle, which is the repo's existing working
method made explicit:

1. **Design note first for L/XL items** — a `docs/<topic>.md` in the
  xpu-transfer-scheduling.md mold (shipped-state + deferred-state + the measured
  gate that would change the verdict). S/M items design in the PR/CHANGELOG entry.
2. **Module placement is pre-decided** — the table above names the home module for
  each new mechanism; nothing lands in `tex_node.py` unless it is ComfyUI
  adaptation (S-1), nothing imports comfy outside the three adapter files (PORT-1),
  and new modules start under the 1500-LOC soft budget (REG-2).
3. **Baseline before, compare after** — `eight_config_bench.py --save` before the
  first change of a release, `--compare` before tagging; per-config geomean < 0.95
  is a stop-ship (invariant #7). Machine idle, on AC.
4. **Every mechanism ships with its pinning test** — the repo's four proven shapes:
  *canary* (contract key-sets: ENG-4/5/6), *derivation* (registry tags → consumer
  sets, TST-3 style: ROI-1), *differential oracle* (new execution path vs the
  interpreter: FUS-3, ROI-4), *never-sever rows* (analysis over-approximation:
  FUS-2, ROI-4). An item without an obvious pinning shape is under-designed — send
  it back to step 1.
5. **Adversarial verify before release** — the multi-agent review workflow (the
  v0.20 pattern that caught the scatter-COW bug) runs on the release diff; findings
  triage into fix-now vs roadmap.
6. **Release exit** — full suite green (1823+ on the dev box, CI green), benchmark
  compare clean, live-session checklist for frontend-touching releases
  (screenshots into the build log), CHANGELOG entry, version bump in
  `pyproject.toml` + `__init__.py`, and any new rejected decision recorded in
  DEVELOPMENT.md the same day it is decided (§7's lists are the queue).

---

*Provenance: this roadmap synthesizes a 7-agent gap analysis of the v0.20.0 codebase
against the ten pillars plus a 4-agent study of Nuke/Blink, Houdini Copernicus,
Fusion/Flame/DCTL, and Blender/Natron/GEGL/OCIO engine architectures (July 2026).
File:line evidence for every gap lives in the analysis; the doc above keeps only the
anchors needed to locate each seam.*
