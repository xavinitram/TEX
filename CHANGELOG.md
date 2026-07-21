# Changelog

All notable changes to TEX Wrangle will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.29.0] - 2026-07-20

**Close the register — the consolidation the v0.28 audit ordered.** No new mechanisms: v0.29
closes the deferral register §9 of the roadmap accumulated across v0.21–v0.28. The compile-error
raiser finally moves down to the engine (ENG-4), the ComfyUI Stop button reaches a running cook
(SCHED-3 bridge), one producer can feed several members of a fused region (FUS-1b) and a linear
run no longer defeats a region's fusion (FUS-1c), and a sweep of small correctness/hygiene items
lands. **Invariant #7, stated precisely (measured, not asserted):** the CLI / standalone / test
path (`cancel=None`) is byte-identical — the interpreter's statement-loop body is untouched. The
**ComfyUI** cook is deliberately *not* byte-identical, because the node now always arms an
interrupt token: it moves to the cancel-polling branch, measured at **0.00 %–0.37 %** of a real
cook (worst at 64², nil by 512²) — perf-neutral, not unchanged. ENG-4 is an error-path refactor
(~0.6 µs/cook), the fusion changes sit behind the fusion flag, and the channel-access guards add
~44 ns per single-channel access (the cheap rank test short-circuits `@image.r` before any type
lookup). Full suite green apart from the standing S-4 cp1252-console env gap.

### ENG-4 — `tex_engine._compile_or_raise` is the one `TEXCompileError` raiser

- The raw per-phase compiler exceptions (`LexerError`/`ParseError`/`TypeCheckError`/`TEXMultiError`)
  are now caught in **exactly one place** — `tex_engine._compile_or_raise`, reached by `prepare()`,
  `cook_stage_list`, and (by delegation) `tex_api.compile`. `tex_node` and `tex_cli` catch the public
  `TEXCompileError` instead of the raw tuple: the "three modules know the tuple" state collapses to
  one raiser + one type. The `_diag_from_exc` materializer moved to `tex_compiler.diagnostics`
  (`diagnostic_from_exc`) so the raiser and `check()` share it without an import cycle.
- The node's `TEX_DIAG:` JSON suffix and the `to_dict` 12-key contract are byte-identical (built from
  `.diagnostics`); Q-4 fused-stage attribution now rides the runtime `InterpreterError` path only (a
  compile error never has a running stage to blame). The ENG-5 canary confirms the surfaces held.
- **Bug-hunt fix (fused path):** `compile_fused`'s per-stage parse/type-check never ran through the
  raiser, so a typo in a fused upstream node escaped as a raw per-phase type into the node's
  bug-report catch-all (losing the `TEX_DIAG`). Per-stage errors now wrap to `TEXCompileError` (the
  clean diagnostic); the final *spliced* compile catches `TypeCheckError` AND its sibling
  `TEXMultiError` (the second audit's find — a ≥2-error chain no longer escapes) → a clean
  `FusionError`; a genuine fusion problem (missing `@OUT`) stays a `FusionError`. Pinned in
  `test_v029_phase1.test_eng4_recut_single_raiser`.

### SCHED-3 bridge — the ComfyUI Stop button reaches a running cook

- `tex_node.execute` now passes `cancel=get_host_services().cancel_token()` into `prepare()`. The token
  (`tex_runtime.host._ComfyInterruptToken`) reads ComfyUI's interrupt flag READ-ONLY and raises
  `CookCancelled` at the next cook yield point (per-strip / per-statement / between tiers); the node's
  `except CookCancelled` re-surfaces the host's own clean `InterruptProcessingException` (a
  BaseException the executor treats as a clean stop, not a red node error). Off ComfyUI the token is
  `None` — the CLI/standalone/default path is byte-identical. `HostServices` gains `cancel_token()` /
  `raise_if_interrupted()` (the ENG-5 method-set canary updated 5 → 7). Live end-to-end verify is LIVE-1.

### FUS-1b / FUS-1c — multi-injection + linear/region coordination (the unshipped v0.21.1)

- **FUS-1b:** one external producer may now feed >1 region member (`Load → [blur, sharpen] → merge`).
  The detector groups external edges by producer (a genuine two-producer merge still stays unfused) and
  the source spec carries an `injections` list; the splicer already handled multi-injection, proven
  bit-exact (CPU+CUDA) vs sequential. GraphSpec **schema 1 → 2** (`source_injections`); a single-injection
  region stays schema 1 / byte-identical, and an older build refuses schema 2 rather than mis-splice.
- **FUS-1c:** the frontend now DETECTS regions before the linear collapse pass and the linear pass
  DEFERS to any node a region will claim — so a linear run ≥2 into a region's fan-out no longer defeats
  that region's fusion (`S→A→B→{C,D}→E` now fuses whole, not just `A→B`). Pure-linear chains are
  unchanged; a route miss falls back to the proven v0.20 linear behaviour. The JS pass reordering is
  verified in the LIVE-1 live-canvas session (it touches the linear pass that runs on every graph).

### The small-item sweep

- **count_var** (open since v0.22): the stencil route now DECLINES an outer-loop counter (it emits
  `kH*kW` regardless of which loop the counter lived in — right for an inner counter, wrong by ×kW for
  an outer one). Codegen falls to a generic loop that matches the interpreter bit-for-bit. **Audit fix:**
  the decline now fires whenever ANY outer counter is present (not only the outer-*only* case) — a nest
  with both counters still deletes the outer one when the box lowers with the inner count.
- **spatial-scalar `.r`/`.x`**: `@mask.r` on a `[B,H,W]` mask used to slice a pixel COLUMN. The
  interpreter now makes it identity (the scalar is its own red), keyed on the base's STATIC TYPE so
  it matches codegen (which bails a single-channel access on a non-vector base to the interpreter) —
  invariant #2 preserved, the common `@image.r` vector fast-path untouched. The write twin
  (`m.r = v` → `m = v`) is fixed to match. **Bug-hunt fix:** the first cut keyed on runtime RANK,
  which diverged from codegen for a vector-TYPED value that is channel-less at runtime (`vec3 cc =
  @mask`, truncate-coerced to `[B,H,W]`); the type-based gate makes both tiers agree there too. The
  type checker still rejects `.g/.b/.a` on a scalar (E3301); the compat corpus (PM-4) is unchanged.
- **v0.27 test-hygiene:** the CACHE-6 oracle asserts `torch.equal` (the real bit-exact claim, not `<1e-5`);
  the morphology halo oracle now runs on CUDA too.

### BENCH-1 / DOCS-1 / LIVE-1

- **BENCH-1:** a many-frame governor soak (PM-5's second half — 50 frames incl. a 4K result held under
  budget across repeated arbitrations, flat RSS). The canonical suite total is confirmed 2232-class (the
  ±1 is the `example compiled: merge` 30 s timeout flake under GPU contention; the one persistent failure
  is the S-4 cp1252 console env gap). A cumulative v0.20↔v0.28 eight-config compare + the durable baseline
  JSON ride the idle-box bench sitting (the roadmap's "machine idle, on AC" requirement).
- **DOCS-1:** the §5 pillar scorecard's "Today" column refreshed to v0.28 reality; the DOC-6 doc-layering
  policy amended in AGENTS.md to name `docs/` as the third layer (seven design docs live there).
- **LIVE-1:** `docs/live-session-checklist.md` extended with the retro-audit rows for v0.23 live-lint/W7xxx
  and v0.26 publish-flow, plus the new v0.29 frontend behaviours (FUS-1c/1b, SCHED-3 Stop) and the
  pre-launch `TEX_Wrangle` junction-removal step. The live run + screenshots is the release's tag gate.

### Adversarial audits (three passes) and the consolidation they forced

- A 4-dimension bug-hunt caught **2 invariant-critical bugs this release introduced**: the
  spatial-scalar rank-vs-type tier divergence (a `vec3 cc = @mask` local read/wrote differently on the
  two tiers) and the ENG-4 fused-path escape (a fused compile typo bypassed the clean `TEX_DIAG`). A
  second independent 17-agent audit caught **2 more one-liners**: `count_var` declined only the
  outer-*only* case, and the fused splice caught `TypeCheckError` but not its sibling `TEXMultiError`.
- A final `/simplify` (4 angles over the whole diff) then removed the duplication those bugs grew in:
  the raw per-phase exception tuple is now spelled **once** (`diagnostics.raw_compile_errors()` +
  `compile_error_from`), and the interpreter/codegen tier-agreement signal is **one predicate**
  (`types.base_is_vector`) instead of four copies — the two seams that had each already drifted into a
  bug. It also replaced FUS-1c's predicted skip-set with region-first + observation (fixing a coverage
  regression where a detected-but-un-appliable region left nodes fused by *neither* pass), collapsed
  `count_var` to a single `return None` in the scanner, and made `injections` the sole in-memory
  source representation.
- The bug-hunt over *those* refactors caught the release's most severe defect: the `m.r = v` identity
  write stored the RHS tensor **itself** (or a stride-0 expanded view) while claiming ownership, so a
  later in-place op could corrupt another local, an already-stored `@OUT`, or the **caller's input
  tensor** — the ComfyUI wire shared with every downstream consumer — and a 0-dim RHS crashed outright.
  Codegen bails on these, so the interpreter was the sole executor: silent wrong pixels, not a
  divergence. Fixed by cloning (matching every sibling branch) and pinned with three regression checks.
- Perf tidies aligned with the compositor loop: the SCHED-3 token is built once per host, the
  interpreter skips the progress arithmetic on the cancel-only path, the channel guards test cheap rank
  before the type lookup (−68 % of their overhead), and the PM-5 soak no longer spills ~600 MB per run.

## [0.28.0] - 2026-07-20

**Second host — the proof release.** v0.28 makes good on the roadmap's PM-2 milestone: a
standalone image viewer (`examples/host_demo.py`) that cooks a fused grade→blur→vignette pipeline
with **zero ComfyUI imports**, scrubs one slider live off the CACHE-2 frame cache, and holds it all
through one engine session — the first time "the tests pass with comfy blocked" becomes "a second
host exists." Around it, four data-model items give the engine the buffer vocabulary a compositor
needs: colour/alpha metadata tags (DATA-1), a storage-format seam with pure-torch EXR + 16-bit PNG
(DATA-2), ARRAY values on a host wire (DATA-3), and the session handle (DATA-4). New package
`tex_io/` (`exr.py`, `png.py`) and module `tex_session.py`; new proof artifact `examples/host_demo.py`.
Everything ships **off the default ComfyUI cook path** (invariant #7): the tags and session are
host-armed / views of the singletons, array wires need the engine profile, and EXR/PNG are opt-in
I/O — the full suite's only failure is the pre-existing S-4 cp1252 console env gap, identical on
committed v0.27.0.

### DATA-1 — buffer metadata sidecar (`BufferMeta`)

- A per-binding `{colorspace, premult, frame, extra}` tag (colorspace ∈ srgb/linear/oklab/unknown,
  premult ∈ premultiplied/unassociated/opaque/unknown) rides the cook on the **value channel** —
  a new `ExecContext.binding_meta` field passed via `tex_engine.cook(binding_meta=…)`, surfacing on
  the new `CookResult.out_meta`. Like `time_context`/`cancel`, it is a pure VALUE: verified absent
  from `fp`, `auto_ckey`, and the lineage key (a tag does not move a pixel). A host supplies the tags;
  the default ComfyUI cook supplies none, so `out_meta` is `None` and the path is byte-identical.
- **Merge on conflict is `unknown`, never a silent pick** (`merge_buffer_meta`): an output derived
  from an srgb and a linear input is tagged `unknown` — the honest answer. `extra` keeps only keys
  every input carries with an equal value.
- **W7005** (`tex_api.color_advisories`): the gamma-space halo hazard — a spatial op (blur/morphology)
  reading a buffer a host tagged srgb/oklab, since averaging a neighbourhood in a non-linear space
  darkens edges. Pure analysis over `tex_roi.binding_footprints` × the tags; off the cook path.
- Tags ONLY, never transforms (roadmap §7) — this is exactly what lets the ACES/OCIO rejection stand.

### DATA-2 — storage-format descriptor + `tex_io` (EXR + 16-bit PNG)

- `tex_io.BufferDesc` (storage dtype ∈ uint8/uint16/float16/float32) resolves the storage envelope at
  the ONE seam that matters — ingestion (`decode_to_fp32`) and egress (`encode_from_fp32`). half and
  uint16 are storage dtypes cast to fp32 exactly like uint8; compute and the wire stay fp32 (NOT
  whole-pipeline fp16 — roadmap §7). The CLI's uint8 load/save now routes through this seam.
- `tex_io/exr.py` — a **pure-torch scanline OpenEXR reader/writer** (struct + zlib + torch, no numpy
  bindings; invariant #1). NONE / ZIPS / ZIP compression, HALF or FLOAT (UINT read-only); the ZIP
  interleave + delta predictor are vectorized (a cumsum, not a per-byte loop). FLOAT round-trips
  bit-exact and preserves HDR (negatives, >1); HALF to ~1e-3. **Cross-validated zero-error against
  OpenCV** both directions. Tiled / multipart / deep / lossy codecs raise a clean `EXRError`.
- `tex_io/png.py` — a 16-bit PNG writer (torchvision's encoder is uint8-only); `tex run --bit-depth 16`.
  `tex run … out.exr` writes a float/half EXR under the value-preserving `engine` egress (DATA-2 lands
  the HDR sink the CLI's 8-bit PNG never had), `--half` for HALF storage. EXR is also a CLI **ingest**
  format now (`tex run --in scene.exr` reads scene-linear fp32, HDR + alpha kept), so
  `tex run --in a.exr --out b.exr` round-trips values the PNG path would clamp/drop. Ingest is
  RGB/RGBA-only; a depth/AOV/multi-plane EXR is refused with a clear message (named planes are a
  future host feature), never silently reinterpreted.

### DATA-3 — ARRAY values on a host wire (engine profile)

- Curves / palettes / histograms flow between tools: an `a@name` input hint and array **outputs** are
  now legal under the engine profile — marshalled as `[N]` (scalar) / `[N,C]` (vec) tensors. An array
  input is consumable via the array builtins (`arr_avg`/`sort`/`median`/`len`/…); a non-spatial
  array pass-through preserves `[N,C]` bit-for-bit.
- Profile-gated by a set-once host flag (`tex_compiler.types.array_wires_enabled`, driven by
  `set_egress_profile`), deliberately **not** in the fingerprint: the always-on guard is the **comfy
  egress**, which refuses an ARRAY output regardless — so existing disk caches survive an upgrade and
  the default path is byte-identical (the E3203 compile-time rejection still fires under comfy).
- **v1 scope, honestly recorded:** direct element indexing `a@name[i]` (the parser routes a
  `BindingRef` index to image-fetch) and per-element vec typing of an input array are the follow-up;
  v1 delivers array output + builtin consumption + pass-through, which is "flows between tools."

### DATA-4 — the engine session (`tex_session.EngineSession`)

- One handle over the process's cook state (program cache, CACHE-5 governor, host services, the
  per-thread interpreter) with `reset()` / `close()` / `stats()`, instead of six imports and six reset
  functions. **Phase 1**: exactly one session (the process default); its `.cache`/`.registry`/`.host`
  ARE the module singletons (views), so ComfyUI is byte-identical. A written thread-safety contract
  extends DEVELOPMENT.md's ENG-9 section (the session adds no lock; it names the single-cook-thread
  boundary a parallel executor must shard).
- Companion **soak lane** (`tests/test_v028_phase1`): hundreds of cooks across shapes + `reset()`
  cycles hold flat RSS and VRAM watermarks (measured +0.0 MB) — the slow leak a days-long compositor
  process would otherwise hide. Isolated sessions threaded through `engine.cook` are phase 2 (ENG-1).

### PORT-5 — the standalone host demo (`examples/host_demo.py`) — PM-2

- ~250 LOC, stdlib + torch + the TEX package, no ComfyUI, no JS: an `http.server` viewer (the Windows
  embedded CPython has no tkinter) that builds a 3-stage GraphSpec, compiles it fused via
  `tex_engine.cook(chain_payload=…)`, hands back raw fp32 (ENG-3 `engine` profile), and scrubs a
  vignette-strength slider. **CACHE-2 is armed by the host** (a `ResultCache` keyed by CACHE-1 lineage):
  scrub back to a visited strength and the frame is a cache HIT, no recook. Each cook takes a
  SCHED-3 cancel token, so a fast drag abandons the stale cook.
- **PM-2 met:** the engine-side cook is **~1.4 ms/frame warm at 1024² on the sm_120 box** (target
  <50 ms; display transport excluded), and the demo runs green under the S-1 comfy-import blocker.
  This regression-guards ENG-1 / SCHED-1 / SCHED-3 / the engine profile forever.

### Post-implementation audit fixes (all OFF the default ComfyUI cook path, invariant #7)

- Audit (3 finder dims × adversarial verify) confirmed the hand-rolled EXR/PNG codec is **bit-exact
  vs OpenCV both directions**, and found three shape/plumbing bugs at the codec's boundary plus one
  off-budget perf win — each reproduced, fixed, and mutation-pinned in `test_v028_phase1`:
  - **B1 (major):** `tex run --out x.exr` on a MASK/scalar program crashed (or, at widths 1–4,
    *silently transposed*). The engine `MASK` egress is `[1,H,W]`, which `write_exr`'s `[H,W,C]`
    dim-ladder read as `[H=1,W,C]`. `write_exr` now collapses a dim-3 `[1,H,W]` to `[H,W,1]`,
    mirroring `tex_cli.save_image`'s branch; a MASK-output program now round-trips through EXR.
  - **B2 (major):** DATA-1 tags were silently dropped on **every fused cook** (`out_meta` always
    `None`) — `_prepare_fused` renames external bindings to `_s{i}_u_<name>`, so `egress_meta`'s
    `binding_meta` match (keyed by the original name) never hit. It now strips the `^_s\d+_u_`
    prefix (the same one `tex_engine` strips for E6003), so the sidecar works on the flagship path.
  - **B3 (minor):** `read_exr` leaked a raw `ValueError` from `torch.frombuffer(b'')` on a zero-length
    NONE block; `ValueError` is now in the caught tuple, honouring its "never a raw error" contract.
  - **P1 (perf):** the EXR/PNG writers marshalled every pixel via `.tolist()` + `struct.pack(*millions)`
    (measured 105 ms→4.2 ms at 1024²×4). They now reinterpret the contiguous tensor's raw bytes via
    `ctypes.string_at` (EXR is little-endian == torch native order; PNG needs a vectorized
    `view(uint8)…flip(-1)` byteswap) — verified **byte-identical** to the old output (whole-file,
    across float/half/HDR/inf and 1–4 channels) and still bit-exact vs OpenCV. Stdlib only (no numpy,
    invariant #1).
- Also: `read_exr` now tags its `BufferDesc.transfer="linear"` (EXR's scene-linear convention — the
  field's first producer, previously declared-but-dead); dropped a redundant `.contiguous()` on the
  EXR read path; new ENG-5 key-set canaries pin `BufferDesc` (STORAGE_DTYPES + fields/defaults) and
  `EngineSession`'s surface. Full suite **2223/2224** (S-4 cp1252 console the only fail, env-only,
  identical on committed v0.27.0); v028 phase-1 40/40.

- **Follow-up root fixes — the C∉{3,4} 4-D-image family + scalar swizzle + `$param` wire count**
  (surfaced by the EXR-ingest work, then hardened by an adversarial-review pass; each mutation-pinned
  in `test_v028_phase1.test_root_channel_and_swizzle_fixes`). Five latent, pre-existing defects — none
  blocked v0.28 (the 3-channel IMAGE wire + `load_image`'s {3,4} EXR pin shielded them), each a genuine
  root defect in how the engine handles a 4-D `[B,H,W,C]` with C∉{3,4}, a scalar base, or an array base:
  - **`_to_mask_shape` channel guard:** a 1-/2-channel 4-D image at a MASK egress IndexError'd on the
    unguarded `raw[...,2]`. Now channel-count-guarded (missing channels read as 0, matching the IMAGE
    2ch zero-pad), so a mask egress of any width is defined.
  - **`infer_binding_type` deliberate C-policy, unified across the LATENT-dict + tensor branches**
    (new `_spatial_channels_to_type`): C=1 and C≥5 both fell to `else → FLOAT`, so a >4-channel
    passthrough *silently collapsed to 1-channel luma* at the MASK egress. Now C=1→FLOAT (lossless
    mask) and C=0/C≥5 raise. **Reachable via the mainstream LATENT wire** — the node unwraps a
    >4-channel latent (SD3/Flux/Wan 16ch, LTX-2 128ch) into a `[B,H,W,C]` tensor *before* inference —
    so a wide-latent TEX node now fails **loud** with a clear message ("VAE-decode it to an image
    first") instead of the old silent luma-MASK that couldn't even re-wire to a latent consumer. The
    dict branch (which had kept collapsing) was reconciled to the same policy; `test_integration`'s
    stale "16-ch → FLOAT" assertion was corrected to expect the refusal.
  - **Scalar swizzle bounds-checked:** the swizzle range check was gated on `is_vector`, so `.rgb`/
    `.rgba`/`.g`/`.a` on a FLOAT/INT (scalar, C=1) base was neither errored nor expanded — the
    interpreter then sliced the *spatial* axis, not channels (`@in.a` on a `[1,H,W]` mask returned
    column 3). The check now covers a scalar base (E3301); `vec3(x)` is the correct scalar→vector
    broadcast. `.r`/`.x` (index 0) stays valid — a count-preserving reduction no-op relied on by
    `arr_avg(a@pal).r`. A FUS-1 `[15]` assertion that assumed a scalar `.a` was legal-and-fusable off
    a MASK was corrected (such a region now correctly does not fuse).
  - **ARRAY-base channel access gated** (closes the ARRAY tracked follow-up this entry flagged): a
    swizzle on an `a@name` array wire (DATA-3) hit *none* of the guards — the single/multi bounds
    checks gate on `is_vector or is_scalar`, so `a@pal.rgb`/`a@pal.g` fell straight through, claimed a
    VECn from the swizzle width, and the interpreter then sliced an `[…,N,C]`/`[…,N]` array by a
    channel index (undefined/silently-wrong). An early `is_array` guard now rejects it (E3300),
    mirroring the string/matrix guards; under the engine profile the cook now *aborts* cleanly instead
    of mis-slicing. Swizzling a reduction's SCALAR result (`arr_avg(a@pal).r`) is a FLOAT base that
    never reaches the guard, so the load-bearing pattern is untouched. Only reachable under the engine
    egress profile (array wires off by default in ComfyUI), so low-severity — but now closed.
  - **`tex run` `$param` wire-count:** `run_program`'s `referenced − assigned` counted `$param` names
    as wire inputs (they are in `referenced`; param_info is empty at that compile site), so a single-
    image program with a `$param` (`examples/vignette.tex`'s `f$strength`) was falsely rejected as
    ">1 input". Now excludes `set(prog.params)`; the >1-wire guard is preserved.
  `load_image`'s `.exr` C∈{3,4} guard is deliberately kept strict (`test_data2_storage_exr` pins a
  C=1/C=5 EXR refusal at ingest). One same-family gap remains a tracked follow-up: the narrow
  spatial-scalar `.r`/`.x` runtime mis-slice can't be distinguished at the type layer from the
  load-bearing `arr_avg(...).r`. Full suite **2231/2232** (S-4 cp1252 console the only fail, env-only);
  v028 phase-1 **48/48**.

## [0.27.0] - 2026-07-20

**Big frames, placed well.** This release makes the engine survive scale — an 8K blur that
could not tile at all, a graph whose caches sum past VRAM, a scrub that recooked a whole fused
chain per tick — and adds the substrate a standalone host needs to place work across devices and
abandon a stale cook. Five items: halo-aware tiling (ROI-5), the global cache governor (CACHE-5),
fusion↔caching reconciliation (CACHE-6), the device-placement scheduler (SCHED-2), and
cancellation + progress (SCHED-3). New module `tex_scheduler.py`. Everything ships **off the
default ComfyUI cook path** (invariant #7): the scheduler, governor, and stage-boundary tap are
host-armed (the ROI-3 / CACHE-2 dormant-by-default posture), halo tiling engages only under
memory pressure on programs that could not tile before, and cancel/progress are `None` unless a
host passes them. Full suite green (the one failure is the pre-existing S-4 cp1252 console, an
environment gap confirmed identical on committed v0.26.0).

### ROI-5 — halo-aware tiling (`tex_memory.run_tiled_halo`)

- `is_tile_safe` refuses **any** non-pixel-local program, so an 8K `gauss_blur` — exactly the
  program that dominates compositing VRAM — could not tile at all. ROI-5 tiles the bounded
  neighbourhood class (blur / erode / dilate) with a **grown strip**: each horizontal strip is
  cooked over `ROI ⊕ halo` so an interior pixel reads the same neighbours as the whole-image
  cook. It is driven entirely over `run_roi`'s proven 2-D grow-cook-crop (a vertical strip is
  `run_roi` with `x0=0, w=W`), so the **ROI-4 differential-oracle bit-exactness carries over
  unchanged**. `is_tile_safe` itself is untouched (still the point-only fast path read by the
  OOM ladder + TST-3); the halo class routes through a separate `_halo_tile_plan`.
- The reach comes from `tex_roi.roi_plan` (`gauss_blur`'s halo is `ceil(3·sigma)`); **global
  reductions and gathers are not halo-tileable** (`roi_plan` is not executable for them) and
  cook whole-frame — the conservative correct choice.
- **WDDM TDR time cap:** the strip planner caps estimated per-**strip** cook time (~1.8 s, under
  the ~2 s display-driver watchdog) using autotier's persisted whole-frame median, not just
  bytes — BlinkScript's driver timeouts are this failure, un-planned-for. The OOM ladder gains a
  halo-tiling backstop for a blur that OOMs whole-frame.
- Honest scope: tiled cooks run on the **interpreter tier** (compiled tiers fall back, exactly
  as ROI-3 ships); routing tiled cooks through codegen is the documented follow-up. Exit gate:
  the halo-tiled cook equals the whole-frame cook (CPU + CUDA, bit-exact morphology / <1e-5 blur).

### CACHE-5 — the global cache governor (`tex_memory.CacheRegistry`)

- Four cache families held per-device VRAM/RAM and **none saw the others** (stdlib env tensors,
  CUDA graphs, the CACHE-2 frame cache; their independent caps sum past 37.5% of VRAM on a 12 GB
  box). `CacheRegistry` arbitrates them against **one** `governor_budget` (~40% of free VRAM, a
  single pressure-responsive cap), evicting cheapest-to-rebuild first: stdlib pyramids → frame
  cache → CUDA graphs.
- **The graph-address safety is preserved, not the stale function.** The register's note said
  eviction calls `clear_graph_cache()`; it has not since MEM-1 — tex_memory uses **pin-and-skip**
  (`pinned_storages()` + `free_graphs_only()`). The governor keeps that exactly: it never frees a
  storage a live graph baked, and to reclaim graph VRAM it tears graphs down with
  `free_graphs_only()` (which keeps the capture blacklist + RNG-poison kill switch —
  `clear_graph_cache()` would re-arm doomed captures and regress MEM-1). The stale note is
  corrected in AGENTS.md.
- Keys and lifecycles stay **per-cache** (the 19-cache register); only eviction **arbitration**
  centralizes. Disk tiers keep their own size caps (CACHE-0 / ResultCache disk), never per-device
  VRAM arbitration. The frame cache folds in via `ResultCache.governed_bytes` / `evict_bytes`
  (host-armed with `register_result_cache`; spill, don't drop). The per-cook
  `enforce_cache_budget` is byte-identical — the governor is a separate opt-in layer.

### CACHE-6 — fusion ↔ caching reconciliation

- A fused chain has no interior cut-points, so twiddling the last node's param recooked all N
  stages per tick — fusion made interactivity **worse** exactly where it matters. CACHE-6 cuts
  the chain at a stage boundary `k`: a **stage-boundary tap** caches the stage-(k-1) fp32 handoff
  (keyed by the upstream *sub-chain* fingerprint `tex_fusion.prefix_fingerprint` × the prefix
  param VALUES × the host-supplied source identity `upstream` × device × precision × playhead ×
  canvas), and a **suffix splice** recooks only
  stages k..N reading that cached boundary while a downstream knob is hot; the full program
  recooks on idle.
- The boundary **must be the exact fp32 handoff** or the oracle breaks (an interior handoff is
  fp16 under auto→fp16), so `cook_fused_cached` gates taps to fp32 and falls back to a whole-chain
  cook otherwise. `cook_stage_list` replicates prepare()'s param default-inject + convert so a
  sub-chain cooks bit-identically to those stages inside the full program. Exit gate: the
  suffix-spliced cook equals the full fused cook (CPU + CUDA, maxdiff 0.0); a hot downstream param
  reuses the cached boundary, an upstream param busts its key.
- OPT-IN and dormant (invariant #7): a host passes the `ResultCache` + cut-point; nothing on the
  default path calls it. v1 covers **linear** chains (the interactive grade→blur→vignette case); a
  `chain_inputs` DAG recooks whole (a documented follow-up — correct, just not incremental).

### SCHED-2 — device-placement scheduler (`tex_scheduler.py`, NEW)

- Where should each node cook — CPU or CUDA? Minimize `Σ cook_cost + Σ transfer_cost` subject to
  user pins and per-device memory budgets. Cook costs from autotier's persisted medians (new
  public `autotier.cook_ms`), transfer costs from the ENG-8 probe (`xfer.transfer_ms`), with
  **boundary transfers** (the external input's upload, the result's download) so the CPU-vs-GPU
  trade is real. Exact where it can be (a **Viterbi DP** for a linear chain, exact **enumeration**
  for a small DAG), **greedy** where it must be — and greedy ("a node runs where its input is,
  else the default device") is `resolve_device`'s own auto rule, so it is the correctness baseline
  the plan defers to when cost data is missing or a node is pinned.
- Device is part of every result key (CACHE-1), so a placement is visible: the plan is **frozen
  per render range**, re-solved only at an interactive/idle boundary (mid-sequence migration is a
  rejected decision — it pops pixels by the invariant-#9 envelope), with **hysteresis** damping
  interactive flapping. Distinct from the rejected PF-4: it chooses *where* a node runs, never
  *which tier*. CUDA + CPU only. Pure planner — no `comfy` import (PORT-1), no bare `torch.cuda.*`;
  cost hooks are injectable so tests drive it with explicit numbers. Ships measured, tested, and
  **dormant** (its consumer is a GraphSpec host — PORT-5 / GRAPH-1); an additive per-stage
  `device` pin in GraphSpec (`graph_from_spec`) an older reader ignores harmlessly.

### SCHED-3 — cancellation + progress

- A `CancelToken` (`.check()` raising `CookCancelled`) is polled at every natural cook yield
  point — before the first tier attempt, before each interpreter fallback, before the fp16 and
  OOM re-cooks, per tile/halo strip, and per top-level interpreter statement — and an
  `on_progress(phase, frac)` callback reports (`tier` bookends, `strip` per strip, `stmt` per
  statement). Aborting a stale cook the instant a newer edit arrives is *the* most-used path in an
  interactive viewer.
- Both ride the **value channel** exactly like ENG-7's `time_context` (threaded host → prepare →
  ExecContext → every tier + `interp.execute`) and are **never** part of any fingerprint / cache /
  lineage key. The default path is byte-identical: the interpreter takes one `is-None` branch per
  cook and runs the unchanged plain loop when no token/sink is supplied. `CookCancelled` is a
  plain exception unrelated to any OOM type (an explicit guard keeps the OOM ladder from retrying
  it). Honest granularity: an in-flight kernel and a `for`-loop body are not preempted — the floor
  is per-statement / per-strip. Exposed on `tex_api.execute` + re-exported from `tex_api`.

### Gates

- v0.27 phase-1: SCHED-3 cancellation (yield points, per-statement, per-strip, `tiled == untiled`
  bit-exact, default path unaffected), SCHED-2 placement (DP / enumeration / greedy / pins /
  hysteresis / adapter / `cook_ms`), CACHE-5 governor (pool register, frame-cache fold-in, evict +
  spill, graph-safety no-op, early-out), ROI-5 halo oracle (blur/morphology tiled == whole-frame,
  CPU + CUDA), CACHE-6 oracle (suffix-splice == full fused, boundary hit/miss/reuse). Compat corpus
  + release gates green; version 0.27.0 consistent.
- **Invariant #7 holds by construction** on the default ComfyUI cook: no watched compiler/runtime
  file changed the default path — halo tiling is a new branch reached only under pressure on
  programs that could not tile before; the governor's `enforce_cache_budget` is byte-identical; the
  scheduler, tap, and cancel/progress are opt-in/host-armed/None-by-default. The cache register
  count is unchanged (the governor holds pools, adds no store). Measured neutral: the interpreter's
  per-statement loop is byte-identical when no cancel/progress is set (one `is-None` branch/cook).
- **A /simplify pass** caught one real invariant-#7 regression: the ROI-5 halo probe ran
  `roi_plan` + `estimate` + `get_free_memory` (the ~61 µs driver query) + `cook_ms` on *every*
  non-tiled CUDA blur/morphology cook. `_halo_tile_plan` now cheap-gates on a memoized estimate +
  cached total VRAM + a dict `cook_ms` lookup (`est < total//8` skip, mirroring `_preflight_memory`),
  so a small default cook returns before any driver query; genuine pressure on a small cook is still
  caught by the OOM ladder's new halo backstop.
- **An adversarial bug-hunt** (5 find+verify agent pairs on the diff) caught three real bugs, all
  fixed and pinned (a SCHED-3 "swallowed cancel" candidate was verified refuted — the
  `except CookCancelled: raise` guards are correct):
  - (HIGH) CACHE-6's `boundary_lineage_key` omitted the source identity, so a different source image
    of the same shape + params served the prior image's cached boundary (silent-wrong). The source
    identity must be **content-sensitive** — a raw `data_ptr` is not (a reused/overwritten frame
    buffer keeps its address; `src.copy_(next_frame)` in a video pipeline, or allocator address
    reuse, both reproduce a stale serve). `cook_fused_cached` now **requires** a host-supplied
    `upstream` CACHE-1 source key to cache a boundary (the "tensor enters by upstream key"
    contract); with no `upstream` it falls back to a correct full cook.
  - (MED) `tex_scheduler._is_linear_chain` accepted a **forest** (≥2 disconnected chains — a dead
    stage in a GraphSpec produces one); the Viterbi reconstruction assumes one source, so a second
    source's `None` back-pointer raised `KeyError`, breaking the "never raises → greedy" contract on
    a ≥2-device box. It now requires exactly one source; a forest routes to enumeration/greedy.
  - (LOW-MED) `_halo_tile_plan` sized the cook off a first-match scan that didn't skip broadcast
    singletons, so a `[B,H,1]` companion bound first under-sized `est` and diverged the autotier
    px-bucket, silently zeroing the WDDM TDR floor (no wrong pixels). It now derives spatial from
    `shared_tile_height`/`shared_tile_width` (both skip singletons).
- **A second, larger audit** (8 finders + adversarial verify) over the full diff: 28 findings
  survived, 3 refuted. One genuine bug — `_oom_retry`'s outer `except Exception` reclassified a
  `CookCancelled` (raised at the OOM yield point or inside the tiled/halo OOM re-cook) as the
  original OOM and retried an abandoned cook — now re-raises cancel first, completing the
  `_run_default` cancel-passthrough. Plus dormant/defensive hardenings: CACHE-6 requires `upstream`
  to key EVERY prefix tensor (not just be non-empty) and wraps `suffix_stage_list`'s `FusionError`
  to the whole-chain fallback; SCHED-2's greedy matches device by CLASS (a `cuda:0`-pinned node's
  consumer no longer drops to CPU) and applies the any-input-on-GPU rule for fan-in; `_halo_tile_plan`
  bails on `is_tile_safe_cached` first so a pointwise CUDA cook skips its scans. Honest scope notes
  added (CACHE-5 live-graph-key realized by evict-order + pin-skip; TDR cap halo-only; SCHED-2
  budget CUDA-only; SCHED-3 cancel not yet node-wired; halo strip-overlap cost).

## [0.26.0] - 2026-07-19

**Tools** — the bundling promise. A tool is the compositor's gizmo / macro / HDA: a named,
self-contained bundle of TEX code with a UI. This release ships the `.textool` format
(TOOL-1), the publish backend (TOOL-2), tool-as-compilation-unit warm keys (TOOL-3), the
`tex build` CLI (TOOL-4), the threat model for sharing tools (TOOL-5), the first stock-node
exemplars (Grade, Blur, Merge, Vignette), and a standalone LSP (LANG-7). A tool carries every
stage's source **inline** — sharing one is sharing one plaintext file, and no program's
compilation ever resolves an external name (this is *not* the rejected cross-node import
system; no by-name tool nesting in v1). The fused tool cook reuses the FUS-3-proven fused
path, so a `.textool` round-trips author → publish → install → cook **bit-identical to the
unfused graph** (the release exit gate). The default ComfyUI cook path is byte-identical
(invariant #7): no watched compiler/runtime file was touched — the tool loader, CLI, and LSP
are new modules beside the engine, exercised only when a tool is built or cooked. New modules
`tex_tool.py`, `tex_lsp.py`; stock tools under `stock/`; design + threat model in `docs/tools.md`.

### TOOL-1 — the `.textool` manifest + loader (`tex_tool.py`)

- A UTF-8 JSON manifest: `{manifest_schema, name, tool_version, tex_language, min_engine,
  category, context, doc, author, inputs, promoted_params}` + one of `code` (single-stage)
  or `graphspec`+`terminal_code` (fused). `TEXTOOL_SCHEMA = 1` versions the format; a newer
  manifest is rejected, never mis-read.
- **Promoted params** map an external widget name → a `$param` in one stage, carrying the
  LANG-1 `ParamDecl.metadata` (`min`/`max`/`step`/`precision`/`label`) so an instanced tool
  reconstructs the source node's widget. Applied uniformly into the target stage's bindings.
- Two cook shapes: **single-stage** tools (all four stock exemplars, and multi-input nodes
  like Merge) cook as a plain program; **fused** tools store the GraphSpec `region_to_collapse_plan`
  emits and `engine.cook(chain_payload=)` consumes — a thin pass-through over the fused path.
- Loader order is TOOL-5-safe: parse → **schema validation (before any TEX is parsed)** →
  language-pin advisory → engine-version gate (fails at install, not at cook). `preflight_tool`
  type-checks without cooking (`tex_api.check` / `tex_fusion.chain_preflight`, both total).

### TOOL-2 — publish flow (backend + JS)

- Backend: `tex_tool.write_tool` (atomic write to `<user_dir>/tex_wrangle/tools/`, the LANG-5
  `get_user_dir` seam), `load_all_tools`/`tool_summary` (the palette), and routes
  `/tex_wrangle/publish_tool` (validate-first, never compiles) + `/tex_wrangle/list_tools`.
- Frontend: a "Publish as TEX tool…" node command that collapses a node's program into a
  single-stage `.textool` with every `$param` promoted (metadata from the code). The richer
  multi-node collapse + drag-promote picker and the instanced-tool-node rendering are the
  documented live-checklist follow-up (pure frontend; the manifest they exchange is
  backend-proven).

### TOOL-3 — tool = compilation unit

- `tool_warm_keys` re-derives a tool's value-independent fingerprint **at install, from the
  inline code** (`fused_fingerprint` / `TEXCache.fingerprint`) — never carried in the file
  (ENG-5: fingerprints are unstable across TEX versions; a stored one would be wrong after any
  update). `warm_tool` / `install_tool(warm=True)` compile the tool at its promoted-param
  signature and warm the codegen/compile tiers off the hot path — single-stage OR the **real
  fused program** (via `prepare_fused`, not `terminal_code` in isolation). **Validate-only by
  default** (no compile on install without consent, TOOL-5).

### TOOL-4 — the `tex build` CLI

- `tex build <tool.textool>` validates + type-checks + reports diagnostics (`--json` for a
  machine report, `--emit` to refresh a normalized manifest, `--warm` to opt into compile).
  Exits non-zero on a preflight error so it can gate a CI/install step. Context tags
  (generator/filter/transition/keyer) and `min_engine` surface in the report.

### TOOL-5 — threat model for shared tools (design note + fuzz lane)

- A downloaded `.textool` is untrusted input to a code generator (`codegen.py` emits Python
  from a user AST). The posture: **validate-only install**, **schema validation before any
  compile**, an **emitter injection audit**, and **documented resource limits**
  (`MAX_TOOL_BYTES`/`MAX_STAGES`/`MAX_PROMOTED_PARAMS`/`MAX_STAGE_CODE_BYTES` — they bound
  parse/compile, *not* cook). The audit confirmed and pinned the two structural safeties: the
  lexer is ASCII-only (`_is_ascii_alpha` rejects Unicode confusables) and identifiers are
  namespaced (`_lv_`/`_p_`/`_uf_lv_`), so no source identifier becomes an un-prefixed Python
  name; string literals are emitted with `repr()`; the type checker rejects unknown functions
  (E5002/E5003) *before* codegen, so `__import__`/`eval` never reach the emitter; and
  `_torch.{name}`/`_math.{name}` dispatch a whitelist, never a raw name. **No change to the
  watched `codegen.py`** — the safety was already there; this release proves it. The
  adversarial-AST fuzz lane (`test_tool_emitter_fuzz`) drives hostile programs through the
  compiler and `ast`-walks the generated source: 6 hostile programs are rejected pre-codegen,
  6 benign-but-hostile (dunder/keyword identifiers, pathological strings) emit **zero**
  dangerous call/attribute/import nodes.

### STOCK — first stock-node exemplars (`stock/`)

- Grade, Blur, Merge, Vignette shipped as `.textool` (single-stage), plus `GradeVignette`
  (fused 2-stage) — the strategic-bet-#8 dogfood. Generated + drift-checked by
  `tools/gen_stock_tools.py` (each manifest is validated + preflighted before it is written).

### LANG-7 — the TEX LSP (`tex_lsp.py`) + offline docs

- A thin stdio JSON-RPC Language Server over LANG-2's `check()` (diagnostics/squiggles) and the
  REG-1 registry (`help_entries`/`help_lookup` → completion + hover). Dispatch is split from
  I/O (`LSPServer.handle` is pure, unit-testable). `python -m TEX_Wrangle.tex_lsp` speaks LSP.
- Offline reference: a `/tex_wrangle/docs/{page}` route serves the shipped `Function-Reference.md`,
  `Error-Codes.md`, and `LANGUAGE.md`, and `wiki_url_for_code` returns the local route when
  `TEX_DOCS_LOCAL` is set *and the page exists*. `Error-Codes.md` now ships at the package root
  (generated by `tools/gen_error_codes.py`, like `Function-Reference.md`) so the offline route
  serves real content — an air-gapped editor gets error-code docs, not a 404.

### Gates

- Full suite green (+9 v0.26 sub-tests: the fused-tool↔unfused-graph exit-gate oracle CPU+CUDA,
  the stock-exemplar canary, the manifest/promoted-param key canaries, the promoted-param
  derivation test, the TOOL-3 warm-key/validate-only test, the TOOL-5 schema-reject + emitter
  fuzz lanes, the LANG-7 LSP smoke, and the `tex build` CLI test).
- **Exit gate met:** a `.textool` round-trips author → publish → install → cook, bit-identical
  to the unfused graph (`test_tool_roundtrip_unfused`, maxdiff 0.0 on CPU and CUDA).
- Invariant #7 holds by construction and by the byte-level check: **no watched compiler/runtime
  file was touched** (the change set is docs + `__init__` routes + the `tex build` CLI arm +
  `diagnostics.wiki_url_for_code` + JS + the new `tex_tool.py`/`tex_lsp.py`/stock assets — none in
  `_AST_FILES`/`_CODEGEN_FILES`/`_VERDICT_FILES`), and **codegen source is byte-identical across
  `PYTHONHASHSEED`** (`test_c1st_execute_line_budget`/the codegen-determinism canary). The tool
  loader, CLI, and LSP execute only on a tool build/cook, never on the default ComfyUI node cook.
  (`eight_config_bench` gave an unstable partial read this session — only the noisy cold-CPU
  config completed, cv up to ~60% — so the byte-identical-codegen check is the load-bearing
  evidence, as in prior refactor-neutral releases.)
- Adversarial bug-hunt (3 parallel skeptics — tool-correctness, security, LSP/JS/invariant —
  each finding reproduced): the emitter injection/RCE claims held under attack (no escape found),
  but the *robustness* posture had gaps, all fixed: a schema-valid-but-malformed fused manifest
  could crash the "never raises" preflight (an out-of-range/negative promoted `stage`, a non-dict
  `terminal_params`, a `code`+non-dict-`graphspec` that `is_fused` and the validator disagreed on)
  — `validate_manifest` now bound-checks every promoted `stage`, requires one external input per
  fused tool with `terminal_image_input` among the inputs, rejects internal/input name collisions,
  and keys fused-ness on presence so `is_fused` can't diverge; the single-stage warm key coerced
  every default to float (mis-keying Merge's int `operation` so the warm key never matched the
  cook) → now inferred from the real default; `TEX_DOCS_LOCAL` is gated on the page existing (no
  dead 404); the publish command handles generators (no inputs) and filters param metadata to the
  validator's key set. The emitter fuzz lane was hardened to the escape-vector blocklist it
  documents (added user-function programs exercising the `_uf_{name}`/depth-guard sites, a fused
  chain, and pickle/`load`/`save` attr vectors); 17 malformed manifests + 6 hostile programs are
  now rejected, 8 benign-hostile + a fused chain emit no dangerous code.
- Maintainer audit (full suite 2128/2129, the 1 fail being the known S-4 cp1252 console) caught
  defects clustered in the warm/preflight/instancing/docs seams — the parts least exercised by the
  exit gate — all fixed and pinned (`test_tool_audit5_fixes`): **TOOL-3's warm was dead** —
  `install_tool(warm=True)` passed bare strings to `prewarm` (which wants `(source, binding_types)`
  pairs), swallowed into a warning, and the fused branch warmed `terminal_code` in isolation (a
  different program than the fused chain) → now `warm_tool` compiles the tool (single-stage or the
  real spliced fused program) and warms the tiers, and `tex build --warm` actually warms;
  **single-stage preflight was type-blind** (`check(code, {})` typed every input VEC4, so `@image.a`
  on a VEC3 input false-passed then failed at cook) → now type-checks against the tool's real
  binding types, and the fused preflight/warm placeholder is shaped by the declared input type
  (MASK/LATENT no longer forced to VEC3); **the Merge exemplar was unfaithful** (ops 4/5 fell
  through as a no-op; the doc claimed alpha "over") → all ten modes implemented, doc corrected;
  plus a `KeyError`→`TEXToolError` on a missing fused source, JSON `true` rejected as a promoted
  `stage`, and an LSP stdio guard so one bad frame can't kill the server. Manifests now declare
  **`outputs`** (a host needs them to wire an instanced tool — the multi-output Vignette
  especially), and the fused-cook per-frame `deepcopy` is a structure-share + completion is memoized.
- Independent audit#6 (24-agent adversarial review, each finding reproduced; full suite 2130/2131,
  the 1 fail the known S-4 cp1252 console) — no critical/high, the emitter/RCE surface held under
  attack; the survivors clustered again in the warm/preflight/robustness seams and are fixed +
  pinned (`test_tool_audit6_fixes`): **warm keys mis-typed a whole class of params** — `_hint_value`
  keyed `b`→FLOAT (not INT) and vector/color hints via scalar lists (which infer FLOAT), so the
  TOOL-3 warm artifact was stored under a fingerprint the cook never asks for → silent warm-miss
  for every bool/vector/color param; now every hint infers to `BINDING_HINT_TYPES` (bool→INT,
  vectors→4-D tensors), the fused warm path types params by hint (not raw default), and warm keys
  cover both the RGB and **RGBA** channel variant a cook may hand an IMAGE input. **Two schema-valid
  manifests could still crash the cook** (violating `validate_manifest`'s own "never a raw KeyError"
  invariant, reachable because publish is validate-only): a linear fused stage missing `image_input`
  (now a validate error) and a graphspec omitting `terminal_image_input` (which `region_to_collapse_plan`
  does) — `_fused_cook_inputs` now makes the manifest field authoritative on the copy. **The 512 KB
  `MAX_TOOL_BYTES` cap was advisory** (only on file-load, so the `/publish_tool` route was bounded
  only by ComfyUI's ~100 MB upload cap) → enforced on the whole serialized manifest at every entry
  point, and `_read_capped` bounds the read (no stat-then-slurp TOCTOU). Publish now records the true
  socket type (`m@`→MASK, `l@`→LATENT) instead of flattening to IMAGE; the LSP no longer busy-spins
  on a closed pipe or exits on a headerless frame, and skips redundant re-analysis of unchanged text;
  preflight cross-checks declared vs assigned outputs; `validate_manifest` returns its parsed lists
  (one validation pass per load); `load_all_tools` caches summaries by mtime; the docs route caches by
  mtime; and the CLI error prefix names the actual subcommand.

## [0.25.0] - 2026-07-19

**Remember frames** — results become first-class. TEX persisted *programs* superbly and
*results* not at all; the standalone engine shipped with no way to hold a cooked frame. This
release builds that half: the ownership contract a cached frame needs (ENG-12), the identity it
is keyed by (CACHE-1), the frame cache itself (CACHE-2, `tex_results.py`), warm-tier persistence
+ prewarm (CACHE-3), and layered cache epochs so a codegen-only edit stops cold-starting the
whole cache (CACHE-4). Like ROI-3 in v0.24, the frame cache ships **armed by an engine host, not
by the ComfyUI node** — the ComfyUI host already caches node outputs, so a second cache under it
is pure memory cost. It is measured, tested, and dormant until a host asks. The default ComfyUI
cook path is byte-identical (invariant #7): none of the five items touches a watched
compiler/runtime file, and every new hook is off the default path. New modules `tex_results.py`
and `tex_runtime/warm_state.py`; design in `docs/results-caching.md`.

### ENG-12 — buffer ownership & immutability contract (lands first, before any frame is cached)
- *Who may write a cooked tensor after it is produced.* Undefined write discipline is the bug
  class that killed Natron's engine, so it lands before CACHE-2 and every later cross-owner edge
  (GRAPH-2 threads, XPU-2 handles) inherits it. THE FLOOR: a cook output is **born frozen** —
  every tier runs under `torch.inference_mode()`, so its outputs are inference tensors and torch
  itself raises on any in-place write. A buffer another party holds cannot be silently scribbled.
- Two enforcement strata for a cached frame: (1) frozen (inference) frames are immutable by
  torch, version stamp a constant 0; (2) normal frames carry `t._version` and are
  version-stamped at insert / re-verified at re-entry (the proven mip-cache pattern) — a bumped
  counter drops the entry, never serves it. New helpers in `tex_engine.py`: `is_frozen`,
  `frame_version`, `verify_unmutated`, `frozen_copy` (a hard-freeze — a `.clone()` made *inside*
  inference_mode), `freeze` (idempotent). M-5 `out=` reuse can never target a cached frame (its
  reuse set is codegen `_tN` temps, never a binding); the one residual write into a binding, a
  scatter, is already COW-guarded. No DO-NOT-TOUCH mechanism was altered.

### CACHE-1 — lineage keys (`tex_results.lineage_key` / `env_epoch`)
- Every cooked output can carry the key that produced it: `H(program_fp × params × upstream_keys
  × frame × device × precision/quality × env_epoch × flags × canvas/ROI)` — Nuke's op-hash. The
  fused memo key already proved the value-independent half; CACHE-1 adds the value-dependent half
  **without content-hashing pixels**: a tensor input enters by its *upstream lineage key*, never
  its content (the sampling hash's admitted collision class is fine for cache-BUST, wrong for
  cache-REUSE).
- **Device and precision are MANDATORY** (a `None` raises, never a wildcard): invariant #9's
  up-to-6.1e-2 CPU↔GPU envelope makes placement visible, so ENG-2's CPU-retry and a future
  SCHED-2 placement change must mint a NEW key and recook, never serve a cross-device hit.
  `env_epoch` (torch version + GPU name + compute capability + the CACHE-4 codegen epoch) keeps a
  spilled frame from surviving a torch/driver/GPU/code change. The engine attaches
  `CookResult.lineage` (per-output) only under `prepare(want_lineage=True)` — off the default
  path, so a ComfyUI cook pays nothing.

### CACHE-2 — the engine frame cache (`tex_results.ResultCache`, the 19th cache)
- RAM-tier byte-budgeted (`TEX_RESULTS_BUDGET_MB`, a conservative default slice of VRAM) with a
  disk-spill tail: an LRU victim is staged GPU→host as a plain contiguous CPU copy and atomically
  pickled under `results/` (page-locking is applied on RESTORE — pinning doesn't survive pickle —
  where the loaded tensor is staged into a page-locked buffer so the H2D can overlap cook time),
  filename = the lineage key; `get` restores host→device (`non_blocking` when pinned) and
  re-admits to RAM. Every entry frozen per ENG-12; `get` re-verifies unmutated before serving.
- Keyed by CACHE-1, keys carry a canvas/ROI descriptor from day one (ties into ROI-3). The
  ComfyUI node does not use it — armed by an engine host. Differential gate (test_v025_phase1): a
  served frame == a freshly cooked one bit-exact; a spill→restore round-trip is bit-exact (CPU +
  CUDA); eviction stays under budget; a mutated normal frame is dropped, not served.

### CACHE-3 — warm-tier persistence + prewarm (`tex_runtime/warm_state.py`, `tex_api.prewarm`)
- Generalizes the `autotier.json` pattern into `warm_state.json`, persisting the graph-capturability
  verdict (`graphed._capturable_memo`) — the static AST capture-gate result, deterministic per
  program AST + arch — so a relaunch skips re-walking the gate. CUDA graphs can't serialize — we
  persist the *decision* and re-capture off the hot path. Version-tagged by the CACHE-4 **VERDICT
  epoch × GPU identity** (device name + torch): a tier-policy/codegen change invalidates a stale
  verdict, and a warm_state from another GPU is ignored. Backend probes and the compile/capture
  blacklists are deliberately NOT persisted — a persisted positive backend probe is inert
  (`_select_backend` only skips a known-False), and persisting any negative would harden a one-off
  runtime/OOM failure into a permanent cross-launch demotion. The load hook is best-effort and off
  the interpreter path.
- `tex_api.prewarm(programs, shapes, ...)` drives the LAT-1a machinery for project-load / idle
  warm: per program it materializes + persists the codegen fn (writes the `.cg`), submits a
  background `torch.compile`, and seeds the capturability verdict, so the first scrub after
  relaunch replays instead of trialling.

### CACHE-4 — layered cache epochs (`tex_cache.py`)
- The one compiler mono-hash that versioned every artifact (a comment-only stdlib edit
  cold-started `.pkl` + `.cg` + inductor + verdicts for every user) is split into a NESTED epoch
  lattice: `AST_EPOCH` gates the parsed-program `.pkl`; `CODEGEN_EPOCH` (= H(AST_EPOCH, codegen
  files, cgreuse)) gates the `.cg` sidecars + inductor dir; `VERDICT_EPOCH` (= H(CODEGEN_EPOCH,
  tier-policy files)) gates `autotier.json` + `warm_state.json`. The nesting is load-bearing (a
  `.cg` is emitted from the compiled program, so it depends on the AST pipeline too). THE WIN: a
  codegen-only edit bumps CODEGEN_EPOCH while AST_EPOCH is unchanged, so the
  parse/typecheck/optimize `.pkl` **survives**.
- The mono-hash is demoted to a completeness tripwire (a test asserts the partition file-sets
  union to the watched set and AST/CODEGEN are disjoint); a fail-safe oracle spot-check asserts a
  reloaded persisted artifact cooks identically to a fresh compile. `codegen_epoch()` is exported
  for CACHE-1's `env_epoch`. Introducing the scheme is a one-time cache reset at the v0.25
  upgrade; every subsequent codegen-only edit then spares the `.pkl` tier. Also strictly safer:
  a tier-policy edit (`compiled.py`/`graphed.py`/…) now invalidates stale verdicts, which the old
  12-file mono-hash never watched.

### Gates
- Full suite **2118/2119** (only the S-4 subprocess-import env artifact; +43 v0.25 sub-tests:
  ENG-12 ownership canaries, CACHE-1 key discrimination + not-a-content-hash + cross-device +
  the playhead never-collide regression, CACHE-2 bit-exact hit / spill-restore / mutation-drop,
  CACHE-3 warm_state round-trip + version-tag guard + prewarm, CACHE-4 tripwire + layering +
  codegen-spares-pkl + fail-safe oracle — CPU + CUDA).
- Invariant #7 (default cook path unchanged) holds by construction (the lineage attach is gated
  on `want_lineage`, off by default; the epoch split runs once at import; the warm-state hooks
  are only on the compile/graph tiers) and by measurement — a same-environment A/B (git-stash to
  v0.24 vs v0.25 on the identical box) is statistically indistinguishable, and a direct micro-bench
  of the added default-path work measures ~50 ns/cook (the `want_lineage` branch + two frozen-
  dataclass fields). No watched compiler/runtime file was touched, so existing program disk caches
  survive the upgrade apart from the one-time CACHE-4 version-format reset.
- Adversarial bug-hunt: a 7-dimension multi-agent review (ENG-12 / CACHE-1..4 / invariant-#7 /
  simplify), each finding reproduced or refuted by an independent skeptic — **13 confirmed and
  fixed**, headlined by a CRITICAL: `lineage_key` keyed only `frame` (int-truncated), so a
  `time`/`fps`-animation or a fractional frame collided onto one key and would serve a stale frame;
  the whole normalized playhead now keys by exact value (pinned by a never-collide test). Also:
  `warm_state.json` now gated by the VERDICT epoch; transient compile failures no longer persist
  across launches; layout-exact spill/restore; several doc-precision corrections. A focused second
  review pass over the fixes found 3 more (all fixed) — headlined by a Mapping-playhead asymmetry:
  the interpreter reads the playhead by duck-typing while the keyer type-checked it, so a
  read-only `MappingProxyType` playhead drove pixels but fell out of the key; `time_context` is now
  normalized to a plain dict at the `prepare()` boundary and the keyer duck-types (pinned by the
  same never-collide test).
- A third (maintainer) audit — 22 agents, 6 hunters → adversarial verify, 7 candidates refuted —
  caught two more keystone stale-serve classes the CPU/CUDA suite structurally couldn't reach
  (they only bite the host-armed engine path), plus a tautological test and quality/perf items,
  all fixed: (1) when the fp16 finiteness net re-cooks fp32, the frame is now labelled fp32 and
  keyed fp32 (was left "fp16" → a false miss vs the auto-pinned-fp32 cook and a lie to a host);
  (2) the lineage canvas now carries the produced frame's FULL shape (batch + every dim), so a
  batch-N or BCHW-latent cook can't collide with a batch-1/BHWC one (was keyed by (W,H) off an
  input, dropping batch); (3) the CACHE-4 fail-safe oracle now compares the reloaded `.pkl`
  against SOURCE-independent ground truth (it previously compiled both sides through the same
  `.pkl` — a tautology that could never detect drift); plus a frozen-sub-view byte over-count/pin,
  an honest scoping of `warm_state` to capturability-only (a persisted backend probe is inert), a
  committed PM-3 benchmark (`benchmarks/cache3_cold_start.py`), an O(N²)→O(N) disk-budget sweep,
  and doc-precision corrections.
- A fourth (independent) audit — 18 agents, 3 hunters → adversarial verify — found **two more
  stale-serve classes** in the same family the prior passes kept surfacing (a pixel-moving input
  that falls out of the key), both fixed and pinned by `test_cache1_pixel_moving_flags` (CPU+CUDA):
  (1) `debug_nan_highlight` — the DBG-3 magenta / C4-ux cyan overlay repaints `raw_output` *before*
  it is keyed, so a debug-overlaid frame could be served for a non-debug cook (or vice-versa); the
  toggle now keys via a `dbg:nan` flag, mirroring the node cache which already folds it in.
  (2) `latent_channel_count` — it materializes the program-readable `ic` builtin, so a program
  reading `ic` produces channel-count-dependent pixels even at a channel-count-*independent* output
  shape; it now keys via an `ic:N` flag, mirroring `graphed._capture_key`. Both are off the default
  path (want_lineage) and latent (no in-tree consumer yet), so no shipped cook was affected. Plus
  perf/robustness on the spill path: the disk-budget check is now O(1)/spill via a running byte
  total (full scan only on a budget crossing); spill does one host copy (was `to()`+`clone()`);
  restore is born frozen so `put()`'s freeze is a no-op (no redundant VRAM re-clone); `prewarm`'s
  background compiles are gated on VRAM headroom + not-capturing (matching the interactive auto
  path); a `warm_state` `atexit` flush keeps last-throttle-window verdicts; and doc-precision
  corrections (auto-placement deferral, restore-overlap threshold, the CACHE-5 governor scope).
- A fifth (final-polish) pass — a `/simplify` (3 `put()` dedups) then an 18-agent bug-hunt — found
  a **MAJOR** that re-bases the ENG-12 model: the "born-frozen = immutable" floor is empirically
  **false on torch 2.12** — an in-place op on an inference tensor LANDS the write and *then* raises,
  so the raise is a loud tripwire, not a rollback, and `frame_version` (constant 0 for a frozen
  tensor) can't catch it. Since `get()` returned the stored buffer, a consumer's `frame.clamp_(...)`
  silently corrupted the cache and the next `get()` served the corrupted frame. Fixed with
  **copy-on-read**: `get()` returns an owned copy by default (mirroring `to_dlpack(copy=True)`);
  `get(key, copy=False)` is the opt-in zero-copy path; the frozen master is never handed out. The
  ENG-12 contract, helper docstrings, design note, and the `_raises_inplace` test (which had only
  checked that the op *raised*, never that the value survived — the false confidence that hid this)
  are corrected, and `test_cache2_hit_is_bit_exact` now mutates a served frame and asserts the cache
  stays clean. Also fixed: a bare-`"cuda"` device string + a process-wide `env_epoch` memo could
  collide two GPUs on a multi-GPU host (now keyed by the produced frame's concrete device + a
  per-device epoch memo), and `clear(disk=True)` zeroing the disk-byte total when a file failed to
  delete (now forces a reconciling scan). A same-key-double-spill counter over-count was also fixed.

## [0.24.0] - 2026-07-18

**See less, cook less** — spatial laziness. The single biggest interactivity lever a
compositor has is to cook only the pixels the viewer asks for. This release builds the
substrate: a per-binding spatial **footprint** analysis, a sub-region (`roi=`) executor that
generalizes the M-4 height-strip to a 2-D window, the differential oracle that gates it, and
the batch-axis (temporal) twin. ROI execution ships **flagged off** — measured, tested, and
dormant behind `TEX_ROI_EXEC` until a viewport host consumes it. `tex_roi.py` is the new
module; the design lives in `docs/roi-spatial-laziness.md`.

### Added

- **ROI-2 — `tex_roi.py`, spatial footprint analysis (the spatial sibling of `tex_lazy`).**
  Per-binding access footprints on the lattice `point ⊑ halo(up,down,left,right) ⊑ image`,
  composing `$param` folding (reused from `tex_lazy`), the ROI-1 registry taxonomy + the
  **reach model**, and affine offset extraction (reused from `codegen_stencil`). `binding_
  footprints()` returns the substrate; `roi_plan()` the ROI-3 execution plan. `where`/`if`
  branches union; anything unresolved → whole-image. Never raises, over-approximates (a
  too-large footprint is a missed optimisation, never a wrong pixel — invariant #11 ported
  to the spatial lattice).
  - **The reach multiplier** closes the trap the v0.23 audit flagged: `('halo_arg', i)` names
    *which argument* carries the radius, not a pixel reach — `gauss_blur`'s kernel radius is
    `ceil(3·sigma)`, **not** `sigma`. The descriptor gains an optional third element `mult`
    (`('halo_arg', 1, 3.0)` for `gauss_blur`; default `1.0` for `erode`/`dilate`, whose
    radius already is pixels), so a wrong reach can't silently under-pad. `_valid_footprint`
    validates the new form and still fails loud on a malformed one.
- **ROI-3 — `roi=(x0,y0,w,h,W,H)` sub-region execution (interpreter tier, flagged off).**
  The M-4 `tile=(y0,H_total)` machinery generalized to a 2-D window: `ix`/`iy` offset by
  `x0`/`y0`, `u`/`v`/`iw`/`ih`/`px`/`py` against the full `W`/`H` — seam-exact. `tile` is now
  the 1-D special case `roi=(0,y0,W,H,W,H_total)`, **normalized to `roi` so there is one
  coordinate path**. `tex_memory.run_roi` narrows every spatial input uniformly to the cook
  region `ROI ⊕ H` (a zero-copy view), cooks, and crops each output to the ROI. Bit-exact
  for pointwise + integer morphology; conv/bilateral within ~1 ulp (size-dependent kernel
  dispatch). v1 executes the point + direct-tensor-halo (blur/morphology) class — the
  dominant compositing ops; gathers/reductions fall to whole-frame (a decoupled gather output
  grid is ROI-5). Wired through `tex_engine.cook(roi=)` → `prepare` → `_run_default`, gated by
  `tex_roi.roi_exec_enabled()` (env `TEX_ROI_EXEC`, default off) and the interpreter tier
  (compiled tiers don't thread `roi`, so they cook whole-frame — same posture as tiling).
- **ROI-4 — the differential ROI oracle (the ship gate; `tests/test_v024_phase1.py`).** ROI
  ships nothing until this is green: a **reach-pinning** test (a one-pixel impulse's output
  spread ≤ the descriptor's declared reach — a wrong multiplier reds here), **spatial
  never-sever rows** (any gather / reduction / symbolic radius / scatter → whole-frame, never
  a shrunk ROI), and the **fuzz oracle** (seeded random ROI-executable programs × random ROIs,
  ROI-assembled == whole-frame cook to `maxdiff < 1e-5`, the FUS-3 convention) plus a
  partition-assembly test. CPU-pinned; CUDA looped. *The oracle earned its keep: it caught a
  real bug during development — the cook halo must be the max reach of **any** halo op,
  including one wrapping a generated (binding-free) expression like `erode(vec4(u,v),3)`, not
  just those over bindings.*
- **ROI-6 — temporal laziness groundwork (the batch-axis twin of ROI-3).** `tex_roi.frame_
  window()` computes the program's temporal footprint `(min,max)` frame offset relative to
  `fi` (via ROI-1's `('frame', i)` descriptors + affine frame-index extraction —
  `fetch_frame(@A, fi-1, …)` → `(-1, 0)`); `batch_sliceable()` is the per-frame whitelist.
  The interpreter gains `batch_slice=(f0, B_total)` — `fi` starts at `f0`, `fn` reports
  `B_total`, seam-exact — and `tex_memory.run_batch_strips` cooks a per-frame program's batch
  in frame-strips (narrowing dim 0, so a LATENT batch-strips safely) to bound VRAM on video
  cooks. Cross-frame reads (a strip lacks the neighbour frames) stay whole-batch — the
  temporal analog of a spatial gather, deferred with the same absolute-index limitation.

### Changed

- `interpreter.execute` / `_execute_inner` / `_create_builtins` gain sibling `roi=` and
  `batch_slice=` parameters; `tile=` is normalized into `roi=` internally (one seam-exact
  coordinate path). The builtins LRU keys on the raw `(tile, roi, batch_slice)` so a
  sub-region / frame-strip cook can't alias a whole cook of the same `spatial_shape`; the
  `tile→roi` normalization runs only after an LRU miss, so the warm-hit default path
  (`tile=roi=batch_slice=None`) allocates nothing new and every coordinate is byte-identical
  to before (invariant #7; measured +2.6 ns/cook, effectively neutral).
- `gauss_blur`'s registry footprint is now `('halo_arg', 1, 3.0)` (was `('halo_arg', 1)`) —
  a metadata-only change; `non_local`, `is_tile_safe`, and every other consumer are unchanged
  (the classifier reads `footprint[0]`).

### Fixed (pre-release adversarial bug-hunt — 6 agents on the ROI diff)

Seven confirmed defects, all in the (flagged-off) ROI path — none reachable by a production
ComfyUI cook, but each would make the ship gate dishonest. Each reproduced, fixed, and now
covered by a new oracle/never-sever row so the ROI-4 gate would catch a regression:

- **`roi_plan` executability made presence-based, not attribution-based.** A gather /
  reduction reaching a binding through a **local-variable alias** (`vec4 x=@A; sample(x,…)`),
  a **user-function parameter**, or a **bindless generated image** (`sample(vec4(u,v,…),…)`)
  was wrongly declared ROI-executable → silent wrong pixels. Now ANY gather / reduction /
  index-access present → whole-frame (the design doc's own gate, enforced on presence).
- **Ungrounded halos block.** A blur/morphology chain split across a local variable
  (`vec4 b = gauss_blur(@A,2); @OUT = gauss_blur(b,2)`) under-sized the cook halo (6, not the
  true 12) → ROI-edge contamination. A halo op behind a local var / function / loop now cooks
  whole-frame (v1 has no local-variable dataflow model).
- **`run_roi` crops each spatial dim independently.** A companion output broadcast in exactly
  one spatial dim (a gradient row `@ROW = iy/ih`, a broadcast strip passthrough) skipped the
  all-or-nothing crop and returned at cook-region extent, wrong size and offset. Now cropped
  per dim, mirroring the per-dim narrow.
- **`batch_sliceable` is now false for ANY frame op** — including offset-0
  `fetch_frame(@A, fi, …)` (an absolute-index gather that a strip clamps to a frozen edge
  frame) and the 3-arg cross-frame sugar `@A[ix,iy,frame]` / `@A(u,v,frame)` (which
  `_frame_ops` missed, as it parses to an index/sample access with the frame in the *last*
  arg). Only pointwise-over-batch programs batch-slice in v1.
- **The analysis honors its "never raises" contract** — the memo key (which could throw on a
  non-`str` program or an unsortable param dict) moved inside the `try`; `binding_footprints`
  and `roi_plan` were also consolidated onto one memoized `_walk` (the 18th cache).

A **second audit** (19 findings) surfaced a further hole in the executability guard and
coverage gaps — fixed before the flag can ever flip:

- **A halo result flowing through a NAME** — a bare `t = gauss_blur(@A,2)`, a reassignment
  `x = gauss_blur(x,2)`, or an intermediate output `@T = gauss_blur(@A,2)` — then blurred
  again slipped past the first ungrounded-halo guard (which only caught VarDecl *initializers*)
  → under-sized cook halo, ROI-edge contamination. The guard now blocks whenever a name
  assigned a halo-containing value is read elsewhere; a name carrying a mere INPUT
  (`vec4 x=@A; gauss_blur(x,2)`) and multi-output programs stay executable.
- A **purely generative** program (no image input) given `roi=` cooked a fabricated gradient
  the whole-frame cook (scalar mode) never produces → `run_roi` now falls back to whole-frame
  when no spatial binding is present.
- The ROI cook is **clamped to fp32** even under `precision="fp16"` (the ~1-ulp conv slack
  scales at fp16; the oracle validates fp32).
- **Coverage**: the differential oracle now fuzzes multi-statement programs with locals and a
  partial-broadcast companion output (the gap that let the above through) and reports all
  mismatches instead of breaking on the first; a new test drives the real
  `cook(roi=)`→`prepare`→`_run_default` engine path (flag gate, fp32 clamp, non-executable
  fallback); and the `tile≡roi` test now pins coordinates against an independent literal grid.

A **third bug-hunt** (post-`/simplify`, 4 finders) confirmed the diff clean and surfaced two
low/latent gaps in `roi_plan`, now closed: a **channel/array-wrapped scatter target**
(`@OUT[x,y].r = …`, `@OUT[x,y][0] = …` — a `ChannelAccess`/`ArrayIndexAccess` around the
`BindingIndexAccess`) escaped the scatter gate (a pre-existing `_has_scatter` gap, not a
regression); and a degenerate zero-arg spatial call (`img_mean()`) was reported executable.
Both now block; a plain swizzle write (`@OUT.rgb = …`, not a scatter) stays executable.

A **third audit** (4-finder × per-finding verify sweep) closed the last executability hole
and two coverage/doc gaps — again, all before the flag can flip:

- **A halo chain through a name NESTED in an `if`/loop body** — `if (u>0.5) { @T =
  gauss_blur(@A,2); @OUT = gauss_blur(@T,2); }` — escaped BOTH ungrounded-halo detectors: the
  block-scan didn't treat `if` as a reach boundary (correctly — a single grounded blur in a
  branch composes fine), and the halo-through-name collector only scanned *top-level*
  statements. So the ±12 chain cooked on `ROI ⊕ 6` → seam contamination (measured maxdiff
  3.2e-3, ~320× the ship tolerance). The collector now recurses into every block, mirroring
  the read-side walk; a single grounded blur in an `if` stays executable (new positive row).
- **`run_batch_strips` mis-stitched a batch-broadcast companion output** (`@OUT=@A; @AUX=@B`
  with `@B` a `[1,H,W]` passthrough): the stitch keyed on `shape[0]==(f1-f0)`, so a `[1,…]`
  output was left `torch.empty` garbage on multi-frame strips and wrongly replicated to
  `[B_total,…]` when every strip was size 1. Now classified per output (per-frame → stitched,
  frame-independent → stored once as `[1,…]`, matching the whole-batch cook), with a
  multi-output/all-strip-size regression row (the ROI-6 oracle previously cooked only
  single-output fully-batched programs, so the bug was invisible).
- **Coverage + docs**: `bilateral_filter` — the 4th whitelist op, named in the oracle's
  "conv/bilateral ~1 ulp" tolerance — is now generated by the differential fuzzer (was never
  cooked through `run_roi`); and the `('halo_arg', i, mult)` reach-multiplier form is now in
  AGENTS.md invariant #5 + the stdlib recipe and DEVELOPMENT.md (it lived only in the
  registry/CHANGELOG, so a maintainer adding a `k·arg` blur could have re-introduced the 3×
  under-pad trap).

Full suite green after all three rounds (2074/2075; the one failure is the pre-existing
`S-4 console` env-only case — an embedded-Python subprocess can't import `TEX_Wrangle`).

### Notes

- **Flagged off by design.** No ComfyUI cook passes `roi=`/`batch_slice=`; the oracle drives
  `run_roi`/`run_batch_strips` directly. The flag flips to a production path when a viewport
  host consumes a sub-region and the fuzz lane is green across a nightly run — a later release.
- Exit gate (roadmap §9 v0.24): the differential ROI fuzz lane is green (CPU + CUDA); no ROI
  cook ships wrong pixels silently.

## [0.23.0] - 2026-07-18

**Authoring** — the language grows its tool-era surface. Param widgets get metadata, the
compiler gets a non-raising lint API with the first real warnings, the language gets a
version and a frozen compatibility corpus, function help moves into the registry, user
snippets become server-backed, and the engine gets a documented zero-copy AI-handoff and a
per-thread interpreter. One roadmap item (LAT-1b) lands as a measured design note.

### Added

- **LANG-1 — parameter UI metadata.** A declaration may carry an optional, literal-only
  metadata block the frontend uses to build a widget: `f$strength = 0.5 [min: 0, max: 2,
  label: "Strength"];` (metadata without a default is allowed too: `i$count [min: 1, max:
  16];`). Values are literals only — a binding ref or an expression inside the block is a
  syntax error. The type checker **ignores** the metadata (a nonsensical range is not a
  compile error); it rides through the optimizer on the AST for tool manifests later.
- **LANG-2 — `tex_api.check()` + the first W7xxx warnings + live lint.** A new
  `check(source, types) -> list[TEXDiagnostic]` that lexes, parses and type-checks and
  **never raises** (errors *and* warnings in one pass, empty when clean). Backs a new
  `POST /tex_wrangle/check` route and debounced editor squiggles. The first three
  advisories: **W7001** unused variable, **W7002** unused input, **W7003** shadowing a
  built-in / an outer-scope variable. `SourceLoc` gains `end_line` (surfaced in
  `to_dict`) for multi-line spans.
- **LANG-3 — language versioning + a frozen compat corpus.** `tex_api.LANGUAGE_VERSION`
  (`0.23`, versioned separately from the package); an optional `//!tex X.Y` pragma
  (`check()` advises **W7004** when a program targets a newer language). A frozen
  **compatibility corpus** (`tests/compat_corpus.py`) runs every example plus adversarial
  grammar programs on the CPU interpreter and hashes their quantized outputs against
  committed goldens, so a language change that alters what an existing program computes
  fails the suite (PM-4). New root `LANGUAGE.md` (grammar, promotion rules, reserved words,
  the compatibility policy).
- **LANG-4 — function help from the registry.** The per-function signature and category
  moved out of the hand-kept JS into the stdlib registry (which already held doc/example),
  so help is single-sourced. New `tex_help.json` (generated), a `tex help <fn>` CLI, and
  a flipped drift test that pins the editor's `TEX_HELP_DATA` sigs to the registry.
- **LANG-5 — server-backed user snippets.** User snippets now persist as JSON in the
  host's user directory via a new `get_user_dir()` host seam; the editor's localStorage
  becomes an offline cache (synced from the server on menu-open, written through on save).
  New `GET/POST /tex_wrangle/user_snippets` routes and `tex_snippets.py` store.
- **ENG-6 — zero-copy AI handoff (DLPack).** `tex_engine.to_dlpack` / `from_dlpack` hand a
  cooked output to another framework over DLPack. The pinned contract: an output is a
  device-resident, fp32, channels-last `[B,H,W,C]` tensor. `layout='bchw'` returns an
  NCHW view; `copy=True` (default) hands back an owned, contiguous, grad-ready tensor
  (materialized out of `inference_mode`) — the safe posture, since codegen reuses output
  buffers; `copy=False` is a genuine zero-copy view.
- **ROI-1 — access-footprint registry.** The boolean `non_local` stdlib tag became a
  footprint descriptor (`point` | `('halo', r)` | `('halo_arg', i)` | `image` | `('frame',
  i)`); the M-4 tiling set `_NON_LOCAL_FNS` is now *derived* from it (`footprint !=
  'point'`), so it can't drift from the impls. Zero behaviour change — the substrate the
  §3 ROI program stands on.

### Changed

- **ENG-9 — the interpreter is per-thread.** `tex_engine._get_interpreter()` and
  `compiled._get_plain_interp()` return a `threading.local` instance. The interpreter
  carries per-instance execution state (scope stack, literal/builtin caches); a shared
  instance mixes up programs across threads (a two-thread cook was proven to corrupt).
  Single-cook ComfyUI is unchanged (one thread → one instance). DEVELOPMENT.md gains the
  first written thread-safety contract + the module-cache classification; a two-thread CPU
  smoke test pins it. Prerequisite for the future branch-parallel executor (GRAPH-2).

### Fixed

- **LANG-5 — two user-snippet data-loss bugs (server-truth vs. read-failure).** The sync
  trusted the server unconditionally while the server could not tell "empty" from "read
  failed", so two paths silently destroyed saved snippets:
  - *An offline / rejected save was discarded on the next menu-open.* `_saveUserSnippets`
    POSTed best-effort and never checked `resp.ok` or the `{"ok": false}` body, so a write
    that never reached disk (read-only dir, unresolved path) was treated as durable; the
    next `_syncUserSnippetsFromServer` then overwrote the cache with the server map,
    dropping the edit. Saves now mark their names *pending* (persisted in localStorage);
    the POST clears a name only on a confirmed-durable write (`resp.ok` **and** body
    `{"ok": true}`), and pending names are re-pushed on the next sync.
  - *A transient server read error wiped everything.* `load_user_snippets` returned `{}`
    on **any** exception, so a briefly-locked / unreadable `user_snippets.json` (e.g. AV
    scanning it right after another tab's `os.replace`) was indistinguishable from a
    genuinely empty store; the frontend synced that `{}` over its cache and the next save
    POSTed the near-empty map as the whole new store. `load_user_snippets` now raises
    `SnippetStoreError` when the store exists but is unreadable / not a JSON object (an
    absent or explicitly-empty store still reads as `{}`); the GET route returns **HTTP 503
    + `{"read_error": true}`** instead of an empty 200, and `_syncUserSnippetsFromServer`
    is non-destructive — it overwrites the cache only on a signalled-successful read and
    otherwise merges server truth with locally-pending edits (last-write-wins per key).
  - Route policy (`user_snippets_get_payload` / `user_snippets_post_payload`) moved into
    `tex_snippets.py` (pure `os`/`json`, PORT-1/S-1) so the read-error 503 and failed-save
    503 are unit-tested off the event loop; a malformed POST body is now a 400 that leaves
    the store untouched (it can no longer wipe snippets). New route test covers the
    round-trip, the read-error path, and the failed-save-preserves-store guarantee.
  - *Concurrency hardening (from an adversarial review of the fix).* `load_user_snippets`
    no longer gates presence on `os.path.isfile` — that swallows the `OSError` of a locked /
    permission-denied stat and would mis-report a present-but-unreadable store as empty (the
    same read-vs-empty confusion, moved onto the stat path); it now opens directly and splits
    `FileNotFoundError` (absent → `{}`) from any other read fault (→ `SnippetStoreError`). The
    client sync merge is now CACHE-BASED and preservation-favoring, so a snippet saved
    concurrently (present in the cache but absent from a possibly-stale GET) is never dropped
    by the re-POST; and whole-map POSTs are serialized through a single-flight chain so rapid
    saves land in issue order instead of racing. Residual: full cross-TAB / cross-machine
    convergence *including delete propagation* needs a versioned or per-key server store — the
    whole-map-replace design predates this fix — and stays a follow-up, out of scope here.

### Deferred

- **LAT-1b — asynchronous CUDA-graph capture — deferred, with the measurement** (the
  LAT-1a / LAT-2 precedent). The synchronous capture stall is a **one-time ~72 ms** on the
  first cook of a graph key (1024², sm_120); the capture key excludes `$param` values, so
  an interactive scrub captures once and replays every later frame at 0.34 ms. Weighed
  against the async machinery's real cost — serializing *all* foreground device cooks
  against the capture window (CUDA capture forbids concurrent device work), through the
  engine seam and a DO-NOT-TOUCH tier, plus a GPU soak test — the one-time stall does not
  justify it yet. The design, the crux, and the reopen gate are recorded in
  `docs/lat1b-async-graph-capture.md`.

### Notes

- Frontend-touching (LANG-1 widgets, LANG-2 live-lint, LANG-5 snippet sync + the
  pending-set / non-destructive-merge data-loss fix above): the JS ships again on this box,
  which has no JS runtime — the CM6 bundle rebuild (LANG-2 multi-line squiggles) and the
  live-session checklist (now including the LANG-5 snippet-sync section) remain to be walked.

## [0.22.0] - 2026-07-16

**The engine seam** — the cook engine stops being a ComfyUI classmethod. This is a
refactor release: it should be invisible from inside ComfyUI, and the interesting part
is what it makes possible outside it.

### Added

- **ENG-1 — `tex_engine`, the host-agnostic cook engine.** Tier selection, the
  interpreter fallbacks, strip tiling, the OOM ladder and the `precision="auto"` gate
  were reachable only through `TEXWrangleNode.execute`, a ComfyUI v3 node classmethod —
  so even `tex run`, the CLI whose job is proving TEX is host-agnostic, imported the node
  to cook one frame. All of it now lives in `tex_engine`:
  - `cook(code, bindings, **opts) -> CookResult` — the one-shot call, for a host with
    nothing to marshal. Returns RAW tensors (no clamp / alpha-drop).
  - `prepare(...) -> CookPlan` / `run(plan) -> CookResult` — the two-step. The split is
    load-bearing, not stylistic: a `prepare` failure means *nothing cooked*, which is
    what lets the node tell "this chain never spliced" from "a linked node failed
    mid-cook" (the Q-4 stage attribution).
  - `tex_node.execute` is now marshal-in → engine → marshal-out: **1207 → ~600 LOC**,
    `execute()` **376 → 205 lines** (its budget ratchets 385 → 240 to hold that).
  - `tex run` cooks through `engine.cook`. Byte-identical output, verified against the
    node path across clamped / gray / alpha-bearing / MASK outputs.
- **ENG-3 — egress profiles.** `comfy` (default) is the shipped conversion, byte-identical
  and canary-pinned forever. `engine` is value-preserving: no clamp, alpha kept, channels
  kept, fp32 BHWC — so scene-linear values survive a node hop. Host-set, never per-node;
  a per-node toggle would let two TEX nodes disagree about what an IMAGE *is*.
- **ENG-4 — `TEXCompileError(diagnostics=[TEXDiagnostic])`.** One public exception type
  for a failed compile, instead of making an embedding host import four internal
  per-phase errors. Raised by `tex_api.compile`; the node keeps its `TEX_DIAG:` suffix,
  which the shipped JS parses. **This is a breaking change to `tex_api.compile` — see
  below.**
- **ENG-2 — a standalone host can see its own VRAM.** `NullHostServices.get_free_memory`
  returned `None`, which disabled preflight, tiling and retry for every non-ComfyUI host:
  the same 8K cook that tiled happily under ComfyUI just OOMed under `tex_api`. It now
  measures the driver + allocator slack. Plus an engine-side OOM ladder (drop TEX's
  caches → re-cook in strips), strictly additive: if it cannot recover it re-raises **the
  original OOM**, so ComfyUI's own `unload_all_models` + retry still fires.
- **ENG-7 — host time builtins `frame` / `fps` / `time`.** Fed per cook from the host's
  playhead (`_tex_time` on the node; `time_context=` on the engine). Builtins, not
  `$params` — a param is part of the lazy memo and the compile fingerprint, so an
  animating value would churn both every frame.
- **ENG-5 / SCHED-1 — the embedding contracts are pinned.** The `_tex_chain` payload is
  now a schema-versioned **GraphSpec** (`GRAPHSPEC_SCHEMA = 1`; an absent field means a
  pre-v0.22 emitter and reads as 1, so every saved workflow keeps working; a *newer*
  schema is refused with an actionable message rather than mis-spliced). Canaries pin
  `TEXDiagnostic.to_dict`, the `ui=` payload keys, `HostServices`' method set and the
  GraphSpec. `DEVELOPMENT.md` gains an **API stability tiers** table and the rule that
  fingerprints are deliberately unstable across versions — a host must never persist one.

### Changed — BREAKING

- **`frame`, `fps` and `time` are now reserved built-in names** (ENG-7). A program that
  declares its own `float time = ...;` will fail to compile; rename the variable. The
  error names the collision and the fix. `$time` — the *parameter* — is unaffected: the
  `$` sigil keeps the namespaces apart, so `examples/caustics.tex` and anything like it
  still works. The README's reserved list and the editor's help panel name all three.

- **`tex_api.compile` now raises `TEXCompileError`, not the per-phase types** (ENG-4).
  v0.21's docstring told hosts to catch `LexerError` / `ParseError` / `TypeCheckError`;
  `TEXCompileError` subclasses only `Exception`, so that tuple silently stops catching —
  the failure is an uncaught exception, not a wrong result, but it is a real break on a
  Tier-1 Public surface and it is filed here rather than under "Added" because that is
  what it is. Our own `tex run` needed exactly this fix, which is how we found it.

  ```python
  # before (v0.21)
  try:    prog = tex_api.compile(src, types)
  except (LexerError, ParseError, TypeCheckError) as e: ...

  # after (v0.22) — one type, and the structured payload comes with it
  try:    prog = tex_api.compile(src, types)
  except tex_api.TEXCompileError as e:
      for d in e.diagnostics: show(d.to_dict())
  ```

  A shared base class was considered and rejected: for the old tuple to keep working,
  `TEXCompileError` would have to subclass all three phase errors, which would make the
  type a lie about what it is. A clean break with a migration note beats an inheritance
  graph nobody can read. `__cause__` is preserved, so the original error is still there.

### Fixed

- **A weighted stencil tap silently computed the wrong answer on the shipped default
  path** (pre-existing since v0.17.0, `649c195`). `acc = acc + @A[ix+dx, iy+dy] * 0.5`
  lowered to an **unweighted** `avg_pool2d`: the `* 0.5` never reached the emitted
  kernel, so the result was exactly 2x, with shapes intact and no diagnostic. maxdiff
  0.42 against invariant #2's 1e-5 — ~42,000x over — at `compile_mode="none"`, no
  opt-in required.
  The matcher did a **search where it needed a match**. `_is_sum_accum` documents itself
  as matching `acc = acc + fetch(...)`, but resolved the tap with `_find_fetch_call`,
  which recursed into `BinOp` and returned a fetch found *anywhere* in the subtree — so
  a composed term matched as if the accumulated term *were* the bare tap it contained.
  The pool lowerings (`avg_pool2d` / `max_pool2d` / `unfold`) can only express an
  unweighted neighbourhood, so the tap must BE the whole term. It is now `_match_tap`:
  strict, no `BinOp` descent, and no `ChannelAccess` descent either (every caller peels
  the swizzle itself to record `channels`; a residual one meant the swizzle sat
  somewhere unattributable and was being dropped too). A composed tap now DECLINES the
  route and runs on the interpreter — correct, just not accelerated. Lowering the weight
  into the kernel is a separate, deliberate feature; the inline `conv2d` path already
  does it and was never affected.
  **All three matchers shared the helper, and all three were wrong.** `_is_minmax_accum`
  dropped weights the same way — and worse: for `max(m, @A[...].bgr * 0.5)` the
  `ChannelAccess` sits *under* the `BinOp`, so `channels` stayed `None` and codegen
  corrupted channel **order** as well as magnitude. `_is_array_collect_assign` (the
  shape behind `examples/median_filter.tex`) funnelled a `BinOp` into the search from
  all three of its branches.
- **The median gate accepted loop bodies it could not account for.** Box and min/max
  refuse a body containing anything they did not match (`not has_unknown`); median did
  not. Lowering REPLACES the whole loop nest, so an array-collect sitting beside any
  other statement silently dropped that statement (maxdiff 1.29). This also gated the
  fix above: while a weighted accumulator still (wrongly) matched, `accum_info` was set
  and `not accum_info` kept median out — making the matcher strict *opened* the gate and
  merely converted a box mis-lowering into a median one of the same magnitude. The two
  ship together because neither is sufficient alone.
- **Only one lowering may claim a loop.** Two accumulators of different kinds in one
  body (`acc = acc + tap; m = max(m, tap);`) hit the box-then-minmax-then-median
  preference order: box won, the nest was replaced, and `m` silently kept its init.
  Detection now declines when more than one kind is present (two of the *same* kind
  already did).
- **A single-channel stencil tap came back one rank too high.** `_stencil_to_bchw`
  slices one channel as `bchw[:, i:i+1]` — the pool ops need that axis to stay 4-D — so
  permuting back gave `[B,H,W,1]` where the interpreter holds a `float` as `[B,H,W]`.
  A plain `.r` box blur (no weight anywhere) returned rank-5 output or raised inside a
  downstream `stack()`. Affected box, min/max **and** the inline `conv2d` emitter;
  median was immune (it is built with `channels=None`). Multi-channel swizzles were
  always right (`.rg` -> vec2, `.rgb` -> vec3), so only `len == 1` is squeezed.
  This one was load-bearing in an unpleasant way: the `stack()` error it raised is what
  *masked* the multi-accumulator bug above, by crashing codegen into a silent
  interpreter fallback that produced the right answer by accident. Fixing the rank
  unmasks the claim bug, which is why they land in the same release.
- **TST-1 could not have caught any of this.** Its atom alphabet held no tap at all —
  no `fetch`, no `sample`, no `@A[...]` — and the only loop it emitted was a flat
  `0..n`, never the nested symmetric `-R..R` that `_try_extract_symmetric_range`
  requires. Measured over 3000 generated programs: **0 stencil routes**. That is how a
  default-path 2x error sat under 1866 green tests. The generator now emits stencil
  nests (both tap spellings, box / min-max / array-collect, ~55% composed taps), and
  folds the result into `@OUT` rather than leaving it to chance — a stencil whose atom
  never reaches an output is dead code the parity check cannot see, which measured only
  ~23% live. Same 3000 programs now: 762 carry a tap, 404 route, all three lowerings
  exercised. Restoring the old matcher makes the fuzzer fail 26/300.
  One known defect is deliberately *not* generated yet, because it would flag instead
  of the tap grammar: a count variable in the outer loop (`count_var` collapses which
  loop it lived in and is always emitted as `kH*kW`, so an outer counter is 3x low). It
  is pre-existing, survives this release, and needs a semantics decision rather than a
  mechanical fix. (The sibling seed defect that shipped beside it — a non-identity
  accumulator seed — turned out to need no semantics decision at all and is now fixed;
  see the next entry.)
- **A non-identity accumulator seed was overwritten instead of accounted for.**
  `_emit_box_stencil` and `_emit_minmax_stencil` replace the loop nest with a single
  pool, then assigned that pool straight into the accumulator — discarding the value it
  held at loop entry. But the interpreter runs `acc = acc + tap` / `m = max/min(m, tap)`
  FROM that entry value, so any non-identity seed diverged on the shipped default path
  (`compile_mode="none"`, no opt-in): `vec3 acc = vec3(0.5)` box came back low by the
  seed (maxdiff 0.0556), and — worse, because min/max have **no finite identity** —
  `vec3 m = vec3(0.0)` max/min over **signed** data (latents, the core domain) dropped
  the seed's clamp entirely (maxdiff up to ~0.6). The two lowerings needed different
  treatment, because their folds differ in kind:
  - **min/max now fold** via `torch.maximum` / `torch.minimum` — the exact ops the
    interpreter's `max()` / `min()` go through. A max/min fold is a *selection*, so it is
    order-independent and stays **bit-exact for any seed**, including `+/-inf`, `NaN`, and
    signed latents (verified on CPU and CUDA sm_120). min/max have no finite identity, so
    folding is the only way to accelerate a seeded pool at all.
  - **box-sum now declines a provably non-zero constant seed** instead of folding it.
    `avg_pool2d` sums the taps in the interpreter's own left-to-right order, so a zero
    seed is bit-exact *even at large magnitude*; but a non-identity additive seed folded
    onto that sum sits at the opposite end of the accumulation from the interpreter's
    seed-first left-fold, and the FP reassociation error there scales with `|seed|`
    (measured: `vec3(100.0)` over `[0,1)` data → 2.3e-5, over the 1e-5 bound; ~1.5e-5 for
    a seed comparable to latent-magnitude taps). So a provably non-zero constant seed
    (`vec3(0.5)`, `vec3(1000.0)`) declines to the **bit-exact static unroll** (which
    accumulates in the interpreter's order), matching how the weighted-tap fix already
    declines what the pool cannot express exactly. A zero/default seed keeps the pool
    (byte-identical bare-box output, no fast-path regression); a *runtime* seed
    (undecidable at emit time) still folds — correct up to the same pool reassociation,
    and strictly better than the old overwrite, which dropped the seed outright.
  Pinned by `test_codegen_audit_fixes` (box seeded `0.5` and a large `1000.0` that reds
  without the decline, the single-channel `[B,H,W,1]->[B,H,W]` squeeze, signed max/min
  seeded `0.0`, min seeded `0.5` over a deterministic ramp, and a routing control that a
  non-zero-constant box declines while a zero/default box and every seeded min/max lower).
- Programs reading the new time builtins are declined by codegen and by CUDA-graph
  capture, and run on the interpreter. Not a limitation for its own sake: everything
  between a playhead and a pixel is keyed on something that does *not* move when the
  playhead does (the codegen executor is a per-fingerprint closure; `_env_cached` and a
  captured graph both replay the value they captured). Each would have failed the same
  way — the cook succeeds and the animation sits still. Declining is the only variant
  that cannot be silently wrong, and it costs nothing today, since ComfyUI has no
  timeline and these read 0 there.
  **Seven routes needed teaching, and only four were found by design.** M-4 strip tiling
  re-enters the interpreter per strip (review); `compiled._plain_execute` is where a
  declined program actually *lands* — the decline happens inside the tier and returns
  rather than raising, so the engine's own fallback never sees it, which froze the
  playhead at 0 on the **shipped default path** (an exact stencil at `compile_mode="none"`);
  and `run_auto`'s internal codegen closure was a third. Both of the latter were found by
  the pre-commit bug hunt — and an **eighth** was found by the release audit, after the
  other seven were pinned: `execute_compiled`'s deep-loop branch (post-optimizer
  `loop_depth > 2`, reachable at `compile_mode="torch_compile"` with e.g. a raymarcher or
  an iterated fbm) handed off to `_codegen_only_execute` without the playhead, while both
  of its siblings six lines away forwarded it. Measured: deep-nest at `compile_mode="none"`
  → 0.9000, the same program at `"torch_compile"` → 0.0000, cook successful, no
  diagnostic. The suite could not see it because the routing needs all of a deep nest, an
  op count over the compile threshold, and trip counts over `_UNROLL_MAX_ITERS`, and the
  existing rows had none of them: `simple` has no loops, and `stencil`'s two unroll away.
  All nine routes are now cooked end-to-end by `test_eng7_time_barred_from_frozen_tiers`
  (which also pins the deep nest's post-optimizer *shape*, so an unroller change cannot
  quietly retire the coverage) — the old test only asserted that `try_compile` returned
  `None`, which proved the bar existed but never that the path it forced you onto was
  correct. `time_context` is now **keyword-only and required** on the two functions those
  declines land on: `None` and "forgotten" are indistinguishable at the interpreter (both
  read 0.0 and cook a still image), so a default could not be safe, and an omission is now
  a `TypeError` at the call instead of a frozen playhead.
- `frame`/`fps`/`time` are built in **fp32**, so the builtin's own value is exact (fp16
  holds integers exactly only to 2048; frame 2049 read as 2048, and every later frame
  rounded to even). **The first version of this note went further and was wrong**: it
  claimed the fp32 made `@A.rgb * frame` exact under fp16 "exactly as it does for `fi`".
  It does not. These are 0-dim tensors, and torch does not let a 0-dim operand lift the
  result dtype — `fp16_image * fp32_scalar -> fp16`, so 2049 rounds back to 2048 at the
  multiply. `fi` escapes only because it is `[B,1,1]`, i.e. DIMENSIONED, a shape these
  cannot borrow (it mis-aligns against `[B,H,W,C]`). What protects users is the
  amplification gate above, not this line; expert `precision="fp16"` has no gate by
  definition, and there the multiply is fp16 like everything else.
- **`precision="auto"` now knows the playhead exists.** ENG-7 added three builtins and
  never registered them in the C1 amplification gate's magnitude table — and they are the
  only *unbounded* ones in the language, where every other entry is capped by an image
  dimension or a batch length. So `sin(@A.r * frame)` was ACCEPTED for fp16 while
  `sin(@A.r * iw)`, the identical shape with a registered builtin, was correctly declined.
  Measured on CUDA at 2048²: frame=500 shipped maxdiff **0.2443 — 63× the 3.9e-3 budget**;
  frame=5000 shipped **2.0 (513×)**, every pixel finite so the C2 net never fired.
  AGENTS.md invariant #10 is explicit — "Any accepted program exceeding 3.9e-3 is a gate
  bug" — and this was one, new in v0.22, on a mode a user selects from a dropdown.
  `frame`/`time` are now pinned at fp16's max (not an estimate of their magnitude, but a
  statement that none can be assumed) and `fps` at 240. After: maxdiff **0.000000**.
- **…and a zero-arg user function could still launder one straight past that gate** (F2).
  Registering `frame`/`time` fixed the direct expression; wrapping it did not.
  `float f(){ return frame; } @OUT = vec4(@A.rgb * f(), 1.0);` was ACCEPTED for fp16 and
  shipped maxdiff **3.2163 at frame=5000 — 825× the 3.9e-3 budget**, every pixel finite, so
  the C2 net never fired and the gate reported "gate-verified accurate (smooth, bounded
  condition number)" while doing it. The mechanism is a `default=`: the gain pass scores an
  unknown call FROM ITS ARGS (`max(..., default=0.0)` / `max(..., default=1.0)`), so a call
  with *no* args is handed gain 0 and magnitude 1 whatever its body assembled. F1 had
  declined user-fn calls since v0.19 and read as though it closed the class; it closed only
  the half that arrives through the interface, and a zero-arg call has no interface. The
  same hole was open for image lineage the whole time — `float f(){ return @A.r*50.0; }`
  measured **0.0278 (7×)** and its squared twin **0.0440 (11×)** — so both halves are now
  declined: any user-fn call whose ARGS carry image lineage (F1) *or* whose BODY reads image
  lineage or a magnitude builtin (F2), followed transitively through calls. The walk is
  cycle-guarded because TEX permits self-recursion (a shipped feature — an unguarded scan
  would hang a cook on a program that merely parses) and depth-capped at the interpreter's
  own `MAX_CALL_DEPTH`, so a chain too deep to model is one it would refuse to run anyway.
  Deliberately as blunt as F1, and over-declining by the same rule #10 already states —
  `float f(){ return @A.r*1.1; }` is fp16-safe at 0.0011 and is now fp32. Measured cost:
  no shipped example changes verdict, no accepted-case test uses a user fn, and the A1-1
  fuzzer's fp16-taken count is unchanged at 5/150. After: every one **fp32**.
- **Two stencil gate holes** (pre-existing; both silently DROP statements on the shipped
  default `compile_mode="none"`). The `local_defs` guard validated against a set it
  populated itself: `_collect_local_defs` records every identifier assignment so taps can
  resolve through temporaries, so reusing it to decide what may be *discarded* accepted
  everything — and the lowering, which replaces the whole nest, then deleted it. It now
  requires the name to be DECLARED inside the nest, which provably cannot outlive it.
  Separately the outer-loop scan had no `has_unknown` mechanism at all: a non-count
  assignment fell through the `elif` chain and was accepted by silence, which also defeated
  the one-lowering-per-loop check (that only ever inspects the inner body). Neither fix
  cost a lowering — UC-2 routing and the codegen-equivalence oracle are unchanged.
- **Inline conv2d applied the first tap's swizzle to every tap** (pre-existing). The
  collector compared each tap's *binding* but never its *channels*, so `@A.r` followed by
  `@A.g` lowered as if both read `.r`. The emitter has one channel selection to give; a
  nest needing two is not a conv2d, so it now declines.
- The codegen decline reports itself honestly on the channel users actually see. The first
  fix landed on the `_show_once` log line; the HUD tooltip reads `tier_trace`, which still
  said "unsupported" — the wild-goose chase the fix was meant to remove. The message would
  have been false either way: the emitter handles these fine, and the decline is a caching
  policy, so a program reading `frame`/`fps`/`time` now says exactly that.
- The OOM ladder clears the failed cook's traceback frames before retrying. The
  interpreter holds whole `[B,H,W,C]` tensors as frame locals (~127 MB apiece at 4K
  fp32), and the OOM's traceback pins one set per nesting level — measured at 160 MB
  still reachable inside the retry. Without this, rung 1 freed the mip caches and rung 2
  re-cooked straight back into the peak that had just failed.
- The `precision="fp16"` × compiled-tier clamp moved from the node into the engine. It is
  engine policy — which precisions a tier supports — and leaving half of it in the host
  meant `engine.cook(precision="fp16", compile_mode="torch_compile")` violated the
  engine's own documented contract.
- `tex run` still fails like a CLI. `tex_api.compile` raising the new `TEXCompileError`
  (ENG-4) put it outside `tex_cli.main`'s except tuple, so every syntax error dumped a
  Python traceback instead of a one-line message — breaking the F3 contract that function
  exists to keep. Found by the bug hunt.
- **A cooked output never aliases an input binding.** `@OUT = @A;` binds the output name
  straight to the input tensor — and const-folding widens that past literal identity, so
  `@OUT = @A * 1.0;` does it too. The ComfyUI node never noticed, because its egress clamp
  materializes a fresh tensor on the way out; ENG-3's `engine` profile removed the clamp
  and, with it, that accidental copy — so a host recycling frame buffers would have had its
  input silently rewritten by its own output. `tex_engine.run` now clones an output that
  shares storage with a binding — and "storage" is literal, which the first version of this
  fix got wrong twice over. Not object identity (a reshape returns a new object over the
  same buffer), and **not `.data_ptr()`**, which is the address of the first ELEMENT: a
  view at a non-zero offset compares unequal and sails through. `@X = @A.rgb;` starts at
  offset 0 and looked caught, which is exactly what made that spelling dangerous — `@X =
  @A.a;` starts at offset 3 and did not. `untyped_storage().data_ptr()` is the buffer.
  Conditional, so a genuinely computed output is never copied. **Ownership is a property of
  the CALL, not of a process global:** `prepare(disown=True)` (the default) is what makes
  `tex_api`'s published promise — "`cook()` guarantees its outputs do not alias your input
  bindings" — true for the host that does exactly what the docs say and never touches
  `set_egress_profile`. Gating it on the process-wide egress profile instead, as the first
  version of this fix did, answered a question nobody asked: the global says what a host
  *set*, not what *this call* will do with the result, and `cook()`'s whole contract is to
  return RAW tensors with no profile applied at all. Under the default (`comfy`) that made
  the guarantee silently false — `@OUT = @A;` handed back the caller's own buffer — with
  zero in-tree impact (the node clamps, the CLI pins `comfy`), which is precisely why
  nothing went red: a false guarantee on a Tier-1 public surface is invisible from inside
  the tree. A caller that can PROVE its own egress materializes passes `disown=False`;
  `tex_node` is the only one, and it derives that from the profile it will actually format
  through (`egress_materializes()`) rather than hardcoding it — under `comfy` the clamp
  allocates regardless, so cloning first cost a full-frame copy immediately thrown away
  (measured 1.349 ms vs 0.019 ms on a 2048² passthrough: the clone WAS the cook), on the
  default path, in a release whose budget is +1.3 µs/cook; under `engine`, the profile that
  REMOVES the clamp, that proof evaporates and the node buys the guarantee instead.
  `tex_api.execute` — the raw interpreter call — makes no ownership promise, and now says so.
- The engine egress profile honours its own "never pinned" claim: LATENT was delegating to
  the comfy body and inheriting its page-locking.
- The C2 fp16→fp32 pin actually reads back. Adding `compile_mode` to the memo key (below)
  updated the reader and left the writer at the old 3-tuple — and a 3-tuple can never match
  a 4-tuple lookup, so "pin the program to fp32 thereafter" was write-only. Correctness
  never depended on it (the net re-runs every cook), only cost: every such program
  double-cooked forever. The net is now handed the key `prepare()` looked up, deleting the
  second construction site rather than re-synchronising it.
- The `precision="auto"` decision memo keys on `compile_mode`. It caches a
  compile-mode-*adjusted* verdict (`[compiled tier: fp32]`) under a key that omitted the
  mode, so whichever mode cooked a program first decided the precision for every later
  mode — a `torch_compile` cook would pin fp32 and the next `compile_mode="none"` cook
  silently lost fp16. Pre-existing; `cook()` newly makes it public API.
- The OOM ladder clears the traceback that actually holds the tensors. `_oom_in_chain`
  digs the OOM out of the `__cause__` chain, but M-1 re-wraps a stdlib OOM as
  `InterpreterError` — the likeliest OOM there is — and in that shape the big tree-walk
  frames hang off the **wrapper**, while the inner OOM carries only the innermost two.
  Measured: clearing the inner one freed 0 of 24 MB. It also clears the doomed attempt's
  `debug_print` probes, which were otherwise duplicated into the HUD — and the re-cook is
  tiled, so its per-strip probes reported the wrong pixel.
- The codegen decline reports itself honestly on the path users actually see. The v0.22
  fix landed on the `_show_once` log line; the channel that reaches the HUD tooltip is
  `tier_trace`, which still said "unsupported" — the wild-goose chase this was supposed
  to have removed.
- `tex_memory` no longer imports `tex_node`. The documented `tex_node ↔ tex_memory` cycle
  is now `tex_engine ↔ tex_memory` — moved, not removed.

### Performance

Invariant #7 says a refactor release must be **invisible**. That is the claim, and here
is exactly what backs it — including which of the usual gates does *not*:

- **ENG-1's move costs +1.3 µs/cook, O(1)** — two dataclass constructions and a call hop,
  against a normalized structural diff of the old and new cook bodies that reduces to
  three additions and nothing else. **+0.19%** of a 1024² CUDA cook, under the jitter.
- **The whole-suite result set is identical to v0.21.0's, program for program** (1855
  PASS / 1 known-env FAIL, diffed line by line; only embedded timings differ). For a
  behaviour-preserving move this is the strongest evidence available, and it is the gate
  that actually bit during development.
- `tex run` through `engine.cook` is byte-identical to the old node path.
- **`eight_config_bench --compare` does NOT gate this release, and saying otherwise would
  be false.** The harness drives `Interpreter.execute` directly (`run_benchmarks.run_
  interpreter`) and imports neither `tex_node` nor `tex_engine` — so it never executes the
  cook path ENG-1 moved. Run anyway on an idle box, it reports both a 1.35× *gain* and a
  0.91× *loss* across configs of code this release did not touch, which is the shape of
  noise, not of a regression (CVs on this box are 16–26% median). It does cover ENG-7's
  interpreter change, measured at **+66 ns/cook** — ~0.002% of a 4 ms cook, i.e. invisible.
  The gate for a change this small is a direct microbenchmark on the real path, which is
  where the +1.3 µs above comes from.

### Deferred, with the measurement

- **LAT-2 (`PreparedProgram`) — deferred, because it aims at the wrong 5 microseconds.**
  Measured on a 256² CUDA cook (263 µs): `prepare()` is 66.3 µs, of which the M-1
  preflight's free-VRAM query is **61.0 µs (92%)** and *everything LAT-2 proposes to
  cache* is **5.27 µs — 2.0% of the cook**. A perfect PreparedProgram would win that 2%
  in exchange for a new (fingerprint × binding-signature × device × precision) cache —
  the same silent-staleness class ENG-7 had to defend against in four separate places.
  Not measured, and so not claimed either way: the compile tiers' per-cook env/capture
  key building, which is LAT-2's other half but sits off the default path. **The real
  target, found while measuring:** that 61 µs preflight query, on every CUDA cook. It is
  live state, so it cannot be cached — it needs to be *skippable*, which is its own
  design. Recorded in the roadmap.

## [0.21.0] - 2026-07-16

**Fuse the graph** — the first release on the compositor-engine roadmap
(`docs/roadmap.md`). v0.20 let a *linear* TEX chain compile as one kernel set; a real
comp graph is a DAG — merges, masks, A/B branches — where linear-only fusion fuses
almost nothing. This release makes fusion real on internally-branching regions (see the
scope note below for what that does and doesn't reach), sharpens the compile tier's
timing, and lands the measurement + hygiene the roadmap's next steps stand on. Suite green except the one known dev-box-only S-4 subprocess-import artifact of the
embedded Python (`._pth` ignores PYTHONPATH); ~24 new v0.21 sub-tests across FUS-1/2/3 +
LAT/CACHE/ENG, including a fusion-region preflight and interp↔codegen DAG parity.

**Performance (invariant #7, measured).** LAT-4 changes the *default* interpreter path,
so it gets a real gate: `benchmarks/lat4_ab.py`. The two questions need different
instruments, and conflating them yields a vacuous answer.

*Steady path (same resolution — where most cooks live).* Both versions HIT their cache,
so the entire delta is the lookup: v0.20's `cache_key == self._builtins_cache_key`
(one tuple compare) versus v0.21's `lru.get(...)` + `move_to_end(...)`. Timed directly
(200k×7, min-of-N): **65.1 ns → 106.2 ns, i.e. LAT-4 costs +41 ns per cook**. Against a
171 µs cook that is **+0.024%**, roughly 15× below this box's own 0.35% whole-cook noise
floor (established by an A/A null control in the same script). So the cost is real, named,
and far below observability — invariant #7 holds, and the honest claim is "+0.024%", not
"zero".

*The win it was built for.* Under proxy↔full-res alternation the v0.20 single slot
thrashes (rebuild every cook) while the LRU hits — paired and interleaved in one process
so thermal drift hits both arms:

| proxy↔full alternation | LRU(8) v0.21 | single slot v0.20 | paired speedup (95% CI) |
|---|---|---|---|
| cpu | 0.2636 ms | 0.3119 ms | **1.16×** [1.14, 1.19], wins 82% of reps |
| cuda | 0.0785 ms | 0.1309 ms | **1.73×** [1.58, 1.80], wins 100% of reps |

LAT-3 *removes* a per-cook `synchronize`, and cache entries are `expand()` views (~64 KB
even at 4K, measured).

The eight-config `--compare` was **run and is reported as inconclusive, not as a pass**:
on this laptop its per-program CV is 16–26% median (max 223%; e.g. `matrix_heavy` "gained"
11.9x at CV 101.8%), which is far wider than any LAT-4-scale effect — its ±6% geomean
flags are noise in both directions, and the v0.20 baseline was itself captured on a loaded
box. That instrument cannot answer this question, so the paired A/B above is the gate that
does. Re-running the eight-config on a quiet machine remains worthwhile for gross
regressions.

### Fusion on DAGs (FUS-1, FUS-3)
- **`detect_fusable_regions(nodes, edges)`** — one pure-Python detector is now the
  fusion authority: single-terminal regions over `@OUT` (slot-0) handoffs, fed by
  exactly one external image edge (a node folds iff *every* consumer is an in-region
  TEX node — the linear "sole consumer" rule generalized). The `compile_fused` backend
  already consumed DAG specs (Q-3 `chain_inputs`); this supplies the producer it never had.
- **Scope — internal vs external fan-out (read this before expecting a win).** Covered:
  fan-out *inside* a region, where a member branches and the branches rejoin at the
  terminal (`src → A → [B, C] → D`) — the branch point is in-region, so there's still
  one external edge. **Not yet covered:** one *external* producer feeding two members
  (`Load → [blur, sharpen] → merge`) — a canonical comp shape, but two external edges,
  which needs multi-injection (one source spec per edge: splicer + transport work),
  deferred to **v0.21.1**. Two-source merges are likewise unfused. So on a real comp
  graph this release fuses strictly more than v0.20, but a shared-source split still
  runs unfused — always safe, just not collapsed.
- **`POST /tex_wrangle/detect_regions`** exposes the detector so every host performs
  the *same* fusion (legality can't drift per host); it **preflight-compiles** each
  region before trusting it — the exact tensor is unknown at detect time, so the frontend
  now sends the producer's socket type and the preflight tests that family (IMAGE → 3-
  *and* 4-channel; MASK → the `[B,H,W]` float; LATENT → its 4-channel unwrapped form).
  A socket type names a family, not a channel count, so each family is checked at its
  dominant shape: a 16-channel SD3/Flux/Wan-class latent (which infers as a *float*) and
  an exotic 1-/2-channel image are knowingly not covered — see `_preflight_samples` for
  why widening it would cost more fusion than it buys. The preflight drops any region
  that passes topology but trips a `compile_fused` guard
  (`@OUT` in a loop/function, a scatter-write, an extra output, a channel mismatch), so
  a region's upstream nodes are never deleted from the prompt only for the cook to then
  hard-fail. Detection runs in a thread executor so it never blocks the aiohttp event
  loop. The graphToPrompt hook adds an **additive, fail-safe** region pass (memoized by
  graph signature): the proven linear collapse is untouched, purely-linear regions are
  left to it, and the collapse verifies the source node still exists (the linear pass or
  a bypass/mute can remove it) before rewiring — so a region never produces a
  dangling-reference prompt that ComfyUI rejects. A wired-`$param` member, a zero-input
  generator, a wired-`code` terminal, and a >16-stage region are all left unfused
  (value severing / resolution divergence / no static code / oversized compile).
- **FUS-3 (the release gate, PM-1):** a fused DAG region is **bit-exact** to running
  its nodes sequentially, AND the merged program runs through **codegen** identically to
  the interpreter (invariant #2 for the DAG-spliced program). Verified on CPU and CUDA
  across linear / diamond / fan-out tree / spatial-`sample` regions (maxdiff 0.0).
- **FUS-2 mechanism** (`fused_required_bindings`): terminal-first lazy composition over
  a fused chain, tested and ready. Its check_lazy_status/E6003 *wiring* is deliberately
  deferred — in this release's single-external-source fusion scope the source is always
  the R1 shape anchor (never prunable), so the "dead upstream branch" win needs
  multi-source regions and lands with them.

### Latency & cache (LAT-3, LAT-4, LAT-1a, CACHE-0)
- **Deferred timing readback** (LAT-3): the frequent MEASURING baseline no longer forces
  a per-cook `torch.cuda.synchronize()` — it records the event pair and reads the prior
  same-resolution cook's elapsed time only if already complete (a non-blocking `query()`),
  so the sync leaves the interactive path; the autotier deque tolerates the sparse
  samples. The slot is keyed by resolution so a proxy↔full-res flip can't fold a timing
  into the wrong px bucket. Invariant #6 holds (a reading is still fenced — deferral
  changes *when*, not *whether*). The TRIAL cook and the one-shot post-commit verify
  window stay **synchronous** — the verify window is bounded (3 cooks) so its sync
  doesn't hurt interactivity, and deferring it would let a resolution-flip-heavy session
  never accumulate its samples (a genuinely slow artifact would escape demotion).
- **Interpreter coordinate-builtin LRU** (LAT-4): the single-slot cache that rebuilt
  fp32 `u/v/ix/iy` on every proxy↔full-res flip is now a small (8-entry) LRU, the
  interpreter counterpart of the codegen path's `_ENV_TENSOR_CACHE` (`u`/`v` are
  `.expand()` views, so a full entry is ~64 KB even at 4K). The interpreter-FALLBACK
  path (`_plain_execute`, the never-cached no-backend path) now reuses ONE persistent
  interpreter instead of a fresh one per cook, so the LRU actually persists there — the
  path that needed it most. Swept by `free_tensor_caches`; invariant #4 untouched.
- **LAT-1a is deferred.** torch.compile is lazy — the ~28 s inductor cost is at first
  *execution*, not at wrap — so a background *wrap* doesn't hide it, and a naive
  non-blocking gate moved compile-failure handling (blacklist / backend-status / Triton
  hint) off the synchronous path (caught by the compiled-audit tests). Truly hiding the
  stall needs a background *execution* pass; deferred to a follow-up with that design.
  The forced `torch_compile` path keeps its proven synchronous behavior unchanged.
- **Orphan `.cg` census** (CACHE-0): `store_codegen_fn` writes a `.cg` sidecar for any
  fingerprint, paired with a `.pkl` or not (the bench harness mints thousands). The
  `.pkl`-only eviction never reclaimed the unpaired ones — they leaked without bound.
  A grace-gated census now reclaims orphan `.cg` oldest-first past a cap. A
  **`TEX_CACHE_DIR`** env override points a test/bench harness at a scratch dir so its
  artifacts never land in the shipping cache (also the first rung of ENG-11's order).

### Engine seams (ENG-8, ENG-10)
- **`tex_runtime/xfer.py`** — a measured host↔device transfer-cost model
  (`transfer_ms(nbytes, pinned, direction)`): a once-per-process, disk-persisted PCIe
  probe fits latency + inverse-bandwidth per (direction, pinned) lane and surfaces in
  `tex doctor`. This is the measurement the v0.21 note in `docs/xpu-transfer-scheduling.md`
  promised; it feeds the mid-term placement scheduler (SCHED-2) and the 3-stream
  crossover gate. Torch-only (no numpy), imports no ComfyUI surface.
- **Rejected decisions recorded** (ENG-10): `DEVELOPMENT.md` gains the compositor-roadmap
  §7 entries — split-frame dual-device cooking, mid-sequence device-placement migration,
  a second "fast" kernel dialect, a scanline/tile-granular core, and a recursive pull
  executor — each with its rationale, so they aren't re-derived.

### Fixed (pre-existing, found auditing this release)
- **A fused terminal that read-modify-writes a wired input** (`@b += …`, `@b = @b*2`,
  `@b.rgb = …`) raised `E6021` at cook. Such a binding is both read and assigned, so it
  was treated purely as an output and its READ never resolved to the upstream handoff —
  and because the failure is at cook, the preflight couldn't see it: a broken prompt.
  The terminal's output is now seeded from the upstream handoff (post-transform, so the
  local isn't re-prefixed); a write-first terminal just overwrites the seed. Hit both the
  DAG path and v0.20's linear path. Gate: `test_fus3_terminal_rmw` (4 forms × CPU/CUDA,
  maxdiff 0.0).
- **A linear chain whose source is bypassed/muted produced a dangling prompt** (v0.20).
  The chain source is read from litegraph but spliced into the *serialized* prompt, and
  ComfyUI omits mode 2/4 nodes from it — so the rewrite could name a node that isn't
  there and ComfyUI rejected the *whole* prompt. The linear collapse now verifies the
  source survived, mirroring the region path; unfused is always correct.
- **The linear pass baked a stale widget for a wire-driven `$param`** (v0.20) — silent
  wrong output. It now mirrors the region pass's `param_wired` guard.
- **Variable-resolution sessions never promoted to the compiled tier**: the LAT-3
  deferred-timing slot keyed on exact shape while autotier buckets by pixel-octave, so
  no bucket ever accumulated the samples to promote. The slot now uses the bucketed key.
- **Region preflight assumed its source was an IMAGE.** It tested 3- and 4-channel
  tensors only, but a MASK infers as a *float*. To be precise about what that cost —
  it was never a wrong pixel, since a program invalid off a mask is invalid fused or
  not: (a) *lost fusion* — `@in.a` is legal on a float but not a vec3, so valid
  mask-fed regions were rejected; (b) *a misleading error* — vector-only calls
  (`length`, `normalize`) pass as vec3/vec4 but reject a float, so a mask-fed one
  false-passed, fused, and died as "couldn't fuse this chain, turn off TEX Fusion"
  instead of the true `length() needs a vector, but argument 1 is float` at the node.
  (Scalars broadcast through operators *and* swizzles, so vector-only calls are the
  only shape that actually diverges.)

### Notes
- The FUS-1 frontend (region serialize/collapse in `js/tex_extension.js`) can't be
  render-verified headlessly — it is on the release live-session checklist. The
  automated gate (FUS-3) does not depend on it.
- `TEX_CACHE_DIR` is now actually set by `tests/run_all.py` and `benchmarks/eight_config_bench.py`,
  so a test or bench run no longer writes compiled artifacts into the shipping package's
  `.tex_cache` (the knob shipped in CACHE-0 but nothing used it).
- v0.20.1 (fused chains reach `torch_compile`/`auto` from the node path) is folded in.

## [0.20.0] - 2026-07-13

The **hardware-honesty** release — the perf gates meet their second GPU. Every constant in the
engine was calibrated on one Turing card (RTX 2080 SUPER); this release runs the whole
measurement kit on Blackwell (RTX 5070 Ti Laptop, sm_120, torch 2.12+cu130 + working Triton)
and ships what the numbers actually said — including the ones that said "don't".

Headline (eight-config, 24 programs @1024², vs v0.19.1 on the same box): **GPU warm Compiled
geomean 2.2–3.0x** (matrix total 1682ms → 100–139ms), CPU cold Compiled 1.1–1.3x, interpreter
configs at parity (verified tree-for-tree under identical measurement). Suite 1814 → **1823/1824**
(the one remaining failure is a dev-box-only subprocess-import artifact of the embedded Python).

### Per-arch gate profiles (S-5 grows teeth)
- `arch_support` gains `_GATE_PROFILES` + `gate_profile()`: **verified** architectures get their
  measured gate constants; unverified ones keep the Turing calibration untouched (never silent
  per-box auto-tuning — profiles are repo-committed measurements, recorded inline in the
  profile table and in this entry).
- **sm_120 profile** (first non-Turing entry): CUDA-graph low-op ceiling **512² → 1024²**
  (measured 1.66x win at 1024² where the Turing gate declined; still correctly declines at
  2048², 0.94x) and fp16 floor **1024² → 2048²** (node-path fp16-auto measured **0.80x at
  1024²** — a real end-to-end loss eaten by cast-in/cast-out; kernel-level 2.0x at 2048² is
  real). Fixes the shipped PR-LP2 node-path perf gate on Blackwell.
- Gate-consuming tests now assert against the live module constants instead of Turing literals,
  so they hold on every verified arch; the fp16 decline reason states the live floor.

### torch.compile tier — narrowed, verified, honest
- **Compile ONLY the flat codegen fn** (G'): wrapping the whole adapter put AST-walking/dict
  Python inside the traced region; dynamo guard-churned (CLASS_MATCH on interpreter classes,
  unserializable under caching_precompile) and re-traced EVERY cook — measured **125ms/cook vs
  0.9ms plain interpreter** on `for_10`@1024². With env construction eager, `for_100` goes from
  60x slower than the interpreter to **2.96x faster**.
- **Post-commit verification** (G): `torch_compile` mode committed blind (unlike the measured
  `auto` tier). Each fresh artifact's first 3 warm cooks are timed (px-scoped — a resolution
  change resets the window); anything slower than **1.5x one timed interpreter cook** is
  evicted + blacklisted with a visible demotion notice. Codegen-eager entries (backend None —
  no dynamo to churn) don't arm a window.
- **Interpreter-wrap removed**: measured as never beneficial in ANY configuration (CPU inductor
  needs MSVC; CUDA-with-Triton is the 100x guard-churn case above; CUDA-without-Triton fails at
  first call). Codegen-rejected programs now go straight to the plain interpreter.
- **cudagraph output-escape fix**: under `reduce-overhead`, inductor's cudagraph trees own the
  output storage — the next cook overwrites it. Outputs escaping to ComfyUI are now cloned at
  the graph boundary (same stage→replay→clone contract as the graph tier).

### Fused chains meet the compile tiers
- `select_tier` now routes a fused chain into **`torch_compile` and `auto`** (keyed by its chain
  fingerprint — the same pattern `cuda_graph` has used since v0.17). Fused chains were
  interpreter-only on these modes; measured on a fused-chain-shaped program at 1024²
  (sm_120 + Triton): **inductor 2.63x vs interpreter**. Toolchain-less boxes self-fall-back,
  and `auto` measures-then-rejects, so enabling them is never a regression.

### XPU — staggered copy/compute for mixed-device chains
- **Pinned egress**: a CPU cook (with CUDA present) writes IMAGE/MASK/LATENT outputs in the
  1MB–256MB band into page-locked memory — same write, and torch's caching host allocator
  amortizes the lock. `unwrap_latent`'s BCHW→BHWC copy is pinned too, so latent chains keep
  pinned-ness. (The 256MB cap exists because pinned pages are unswappable and the allocator
  retains freed blocks for the process lifetime — an uncapped video-scale egress would
  permanently page-lock GBs; above the cap nothing hides a seconds-scale copy anyway.)
- **Non-blocking ingestion**: a pinned CPU binding headed to CUDA rides the DMA engine in the
  background while Python keeps working (remaining bindings, coordinate builtins, dispatch);
  stream ordering makes the first consuming kernel wait exactly until the copy lands — no
  events, no manual sync, bit-identical (the standard DataLoader pattern). An ingest fence
  (event recorded at the copy point, synchronized before the cook returns — ~free by then)
  guarantees the DMA has landed before the output escapes, so no downstream host-side writer
  can race an in-flight copy.
- Measured on the mixed CPU→CUDA chain: **1.10x at 1024²** (16MB), **1.41x at 2048²** (64MB,
  3.5ms/cook saved). All-CPU and all-GPU chains are untouched (zero copies either way).
- Cross-device cooks stopped burning a codegen attempt: `_contiguous_bindings` now co-locates
  off-device bindings on every tier incl. the auto-tier artifact paths (fused with the M5-INT
  cast — one copy, single-sourced in `to_fp32_if_int_image`), instead of raising at the first
  mixed op and retrying on the interpreter.
- Design notes for the v0.21 follow-ups (3-stream tiled pipeline; speculative return-time
  upload) live in `docs/xpu-transfer-scheduling.md`.

### Codegen warm-cook cost + scatter copy-on-write
- Builtin env tensors (`u`/`v`/`ix`/…) are now cached across cooks (`_ENV_TENSOR_CACHE`, keyed
  shape/device/dtype, keepalive-registered for graph safety, device-canonicalized so an
  index-less "cuda" can't split keys or serve wrong-GPU tensors) — the codegen path was
  re-allocating every one of them per frame while the interpreter has cached its equivalents
  for ages.
- **Codegen scatter COW** (caught by this release's 28-agent adversarial verification pass
  before it shipped): `@OUT = ix; @OUT[x,y] = v;` wrote in place through the bare alias — with
  the env cache that would have corrupted the cached coordinate grid for every later cook.
  Codegen now mirrors the interpreter's `_scatter_owned` clone-before-first-write exactly
  (including disown-on-rebind), which also stops scatters from mutating caller-owned input
  bindings — restoring interpreter parity on that side effect.

### Measured and declined (the honest column)
- **TF32**: ~1.0x on sm_120 — TEX's op mix has no TF32 surface (elementwise ops don't matmul,
  the CUDA matvec is broadcast-sum by design, depthwise conv never picks tensor-core kernels;
  timing AND bits identical with TF32 on). `apply_tf32_profile` stays an unwired opt-in; the
  generic "1.5–2.5x on Ampere+" claim does not transfer to TEX workloads.
- **channels_last conv stencils**: 0.65–0.78x on CUDA (depthwise conv prefers the current
  NCHW-materialized path), wash on CPU. Not shipped.
- **CUDA graphs above 1024²**: 0.80–0.99x at 2048²+ on sm_120 for every program class — the
  hard cap stays.

Suite **1823/1824** on the dev box (the 1 failure + 2 skips are dev-box env artifacts;
9 new v0.20 tests incl. the scatter-COW and XPU regression pins).

## [0.19.1] - 2026-07-12

A correctness + CI patch. The nightly differential fuzzer earned its keep — it caught a real
interpreter↔codegen divergence, and an exhaustive follow-up audit found two more of the same
class. **No behaviour change on the default `compile_mode="none"` (interpreter) path**; the
fixes matter only on the opt-in codegen tiers, where they restore bit-exactness with the
interpreter.

### Codegen ↔ interpreter fidelity (the bit-exactness contract)
- **`smin` / `smax` / `lerp` / `mix` / `fit`** emitted an *unfused* lerp (`b + (a-b)*h`) while the
  interpreter uses the fused `torch.lerp` (one FMA rounding). The ~2e-8 gap is normally invisible,
  but a near-equal-edge `smoothstep` amplified it into a full **0↔1 flip on 22 % of pixels**
  (found by the nightly fuzzer, seed 20260712). All five now route through the fused helper.
- **`pow`** specialized non-folded constant exponents (`-0.5`, `-2.0`, `-1.0`, `4.0`) to
  `rsqrt`/`reciprocal` forms that **flipped finiteness on x ≤ 0** and diverged up to 8e-3 from the
  interpreter's `torch.pow`. It now keeps only the bit-exact `x^{0,1,2,3}` and defers every other
  exponent to the interpreter's `fn_pow`.
- **`mod`** used a non-dtype-aware `1e-8` zero-guard that underflows to `NaN` in fp16 (where the
  interpreter's dtype-aware `6.104e-5` stays finite); it now defers to `fn_mod`.
- Verified **0 divergences across 80,000 fuzzed programs** (40 seeds incl. the next 30 nightly
  dates); pinned by two new regression tests. `codegen_stdfns.py` is now folded into the
  compiler hash, so an emit change always invalidates the persisted codegen cache.

### CI
- Three tests only began running on CI once v0.19.0's `path: TEX_Wrangle` checkout fix let pytest
  import the package. They assumed a CUDA device (`test_cc1_triton_hint`) or the separate
  gitignored `wiki/` checkout (`Error-Codes` / `LLM-Cheatsheet` drift); all three now **skip**
  gracefully via a new `SubTestResult.skip()` instead of failing on a CPU-only, wiki-less runner.
- The Tests matrix gained `fail-fast: false`, so one Python version's failure no longer cancels
  and hides the others.

Suite **1818/1818**.

## [0.19.0] - 2026-07-10

The **"Prove It"** release — validation catches up to velocity. After six releases in seven
days, v0.19 hardens the durability net, *proves* the machine is host-agnostic and its perf
gates measurable, opens the first external-adoption loop, and lands three measured perf wins
— every change shipping with the test that guards it. Suite **1761 → 1813/1813**.

### Harden the net (durability)
- **The differential fuzzer now generates real programs** — user-function defs/calls, bounded
  loops, multi-statement locals — not just single expressions, and adds an fp16-`auto`
  tolerance arm. It immediately found **two real accuracy holes** that shipped green in three
  prior releases: `degrees()` amplified image lineage ×57 past the fp16 gate, and `log10` was
  fp16-fragile but unclassified. Both fixed; the fuzzer is now clean at **0 divergences /
  6000 programs**.
- **fp16 precision taxonomy federated into the stdlib registry** — a new fp16-fragile stdlib
  function can no longer silently default to fp16-eligible; an unclassified one fails the suite
  loudly.
- `execute()` re-extracted behind a per-function line budget; a combined fusion × lazy ×
  precision × tier end-to-end test closes the highest-risk untested seam.

### Prove the machine (host-agnostic + measurable)
- **ComfyUI-free core, machine-enforced.** A package-level boundary lint keeps the compiler +
  runtime free of any ComfyUI import (the host layer is exactly three files), and a smoke test
  drives the node with ComfyUI fully blocked — the whole test suite is already the standalone
  lane. *Scope note:* this **enforces** the tex_core boundary but does **not** yet ship a
  physical `pip install ./tex_core` package — the doc-30 file reroot is deferred to a live
  session (it needs live import-path verification). Portability is *proven and guarded*, not
  *packaged* — that packaging is a v0.20 follow-up.
- **`tex validate-hw`** — a new CLI subcommand that measures whether TEX's Turing-calibrated
  perf gates hold on *your* GPU (fp16 crossover, the CUDA-graph gate, TF32, determinism) and
  emits a shareable report; an issue template invites Ampere/Ada/Blackwell reports. On the
  calibration box: fp16 gate sound, graph gate 4/4, scatter determinism 0.0.
- **Per-architecture honesty** — on any unmeasured GPU, `tex doctor` now says the gates were
  calibrated on Turing and points at `validate-hw`. Behavior unchanged; the caveat is honest.

### Show the truth & open the loop (adoption)
- **Perf HUD on a DOM dual-path** designed to render under both classic and Vue node modes,
  now showing `debug_print` probes and a hover tooltip with the tier/precision reasons.
  *(The Vue/Nodes-2.0 render path ships as code with a headless guard; final in-canvas
  verification is pending the live-ComfyUI session — see `docs/live-session-checklist.md`.)*
- **`tex doctor` modal** in the node's right-click menu; the default node code points at the
  snippet browser (116 examples).
- **Near-singularity diagnostic** — with `debug_nan_highlight` on, a guarded division that hits
  the epsilon branch (e.g. `1/(x-x)`) paints **cyan**, distinct from a magenta NaN, and a count
  is surfaced. Zero-cost when the toggle is off.
- **Adoption artifacts** — a registry-generated **LLM-authoring cheatsheet** (paste into any
  model to have it write valid TEX) and **8 drag-and-drop example workflows**, both drift-tested.

### Measured performance
- **CUDA mat3/mat4 × vector is 3.9× faster** (op-level) via an elementwise broadcast instead of
  `matmul`; CPU keeps `matmul` (7× faster there). The interpreter and codegen emit the identical
  device-gated expression, so they stay bit-exact per device.
- **The noise resolution-recompile stall is gone.** `torch.compile(dynamic=True)` gives one
  kernel for every resolution, eliminating a measured **134× / 5.6 s** recompile when a program's
  resolution changes — while keeping the full compile speedup.
- **`is_tile_safe` is memoized** per program fingerprint — a 22 µs AST walk that ran every CUDA
  cook now runs once (43× faster on a hit).
- Noise compiles are surfaced in `tex doctor`; a reach for the future `pass { }` multi-pass
  syntax gets a helpful hint (chain nodes today); two recursive examples added; the v0.20
  multi-pass execution model is scoped in a design spike.

### Correctness hardening (post-audit fixes)
These closed findings from the release-readiness audits — all on the opt-in `precision="auto"`
path; the default `fp32` path was clean throughout.
- **Fixed a crash in the `precision="auto"` safety net.** An extraction had dropped a needed
  import, so when the fp16 fast-path overflowed, the automatic recovery to `fp32`
  *crashed* instead of recovering. It now recovers end-to-end (verified on a real overflow),
  with a regression test that exercises the actual recovery path.
- **Completed the fp16 accuracy gate.** `pow2`/`pow10` (2^x/10^x) and six alpha-dividing
  compositing functions (`over`, `under`, `unpremultiply`, `color_dodge`, `color_burn`,
  `vivid_light`) were unclassified and could take the fp16 path when they shouldn't; they now
  correctly force `fp32`. The guard that catches this is now implementation-based (it scans for
  data-dependent division, plus one level of delegation), not just name-based.
- **Made the differential fuzzer honest about codegen.** It now reports when a program falls
  back from codegen to the interpreter (previously invisible), and the fp16-accuracy fuzzer now
  counts a NaN-vs-finite mismatch as a failure. A codegen scalar-loop crash class this surfaced
  was fixed (such calls stay on the tensor path).
- **`tex validate-hw` no longer crashes on a Windows (cp1252) console** when printing its report
  glyphs — the terminal echo degrades to ASCII; the written report files keep the glyphs.

## [0.18.0] - 2026-07-09

The "make it visible, make it honest, make it portable" release — it converts two cycles
of built-but-unwired infrastructure into user-visible value, ships the one *measured*
precision lever, fixes a memory-path safety bug, and lays the first stones of a
host-agnostic core. Suite **1691 → 1761/1761**.

### Lazy input cooking
- **Wired inputs the code cannot use are never cooked** — their whole upstream
  subgraphs are pruned from execution. Covers: inputs never referenced (T1),
  references inside statically-dead flow (T2), and — the sweet spot — branches
  disabled by a `$param` value (T3): `if ($mode > 0.5) { @OUT = @B; } else
  { @OUT = @A; }` cooks only the taken side's upstream. **Wired scalar params
  cook first and fold on the next round** (iterative `check_lazy_status`), so a
  param fed by a Primitive node still prunes image branches ("T4-lite").
- Mechanism: ComfyUI only honours `lazy` on schema-declared input names, so the
  schema declares a hidden pool of lazy AnyType slots (`in_0..in_15`); the
  frontend maps wired user inputs onto them **in the queued prompt only**
  (`_tex_slot_map`) — workflows, slots, and labels keep user names. New module
  `tex_lazy.py` (analysis: substitute $params as fp32 literals → fold →
  propagate → fold → prune literal-condition flow → collect survivors; memoized,
  cache #14). Setting: **TEX Lazy: skip cooking unused inputs** (default on).
- Deliberately NEVER severed (correctness): `@A * 0.0` (NaN·0 = NaN), `&&`/`||`
  operands (both sides always evaluate), spatial per-pixel conditions
  (torch.where computes both branches), string-param conditions (never fold).
  Safety rails: the first spatial wire (the first-wins shape anchor) must
  survive or nothing is skipped; LATENT wires always cook (they flip output
  typing/fp32); any analysis failure cooks everything. Fused chains skip
  nothing in v1. Full T4 (conditions on *image* values) is deferred by design.
- Behaviour notes: a `@ref` that is statically dead under the current params no
  longer raises E6003 *when queued through the lazy path* (it is dead code);
  the legacy path (no slot map — old prompts, direct API calls) is byte-for-byte
  unchanged. `fingerprint_inputs` hashes the slot map, so remapping busts the
  cache correctly.

### Precision
- **`precision="auto"`** — an **experimental, conservative fp16 mode** (default stays fp32).
  It runs fp16 only on CUDA, ≥1024×1024, for a smooth pointwise program that a
  **condition-number gate proves won't amplify fp16's ~1e-3 input error past the 8-bit
  quantum** — a flow-sensitive image-**gain + magnitude** analysis that declines
  amplification assembled from sub-threshold steps (`sin(@A.r*3*3)`), squaring, `/const`
  chains, builtin-dimension products (`@A.r*iw`), dot/matrix/length/cross fan-in, `fit`
  remaps, additive round-trips, array reductions, ill-conditioned fns (tan/atan2/normalize/
  hypot/sdiv), and **any user-function call touching image lineage** (the gain pass doesn't
  model `FunctionDef` bodies, so it declines them). **Verified 0 accuracy
  violations across 225 *direct-expression* adversarial programs (two independent red-team
  rounds) + a fuzzer** — but it is a heuristic, not a proof, so a per-cook finiteness net
  re-cooks fp32 on any non-finite (**runs every cook**; the earlier
  "check once then trust the fingerprint" shipped 3.1M NaN silently when a program met a
  new input — that regression is fixed).
  **Honest perf:** through the node (`TEXWrangleNode.execute`) `auto` is essentially
  **perf-neutral (~0.99× @1024² / ~1.08× @2048²)** — the safety net costs about what fp16
  saves. (An earlier "≈1.45×" was a repeated-input microbenchmark measured off
  `Interpreter.execute`, not the user path; this corrects it.) The raw fp16 win
  (~1.35–1.45×) is available, **without the safety net**, via expert `precision="fp16"`.
- **fp16-safe reductions** — `img_sum`/`mean`/`min`/`max`/`median` and `arr_sum`/`arr_avg`
  now accumulate in fp32 (an fp16 sum overflowed to inf at ≥1024²); a large-value
  `vec()`/literal also stays fp32, so interp == codegen. Bit-identical on fp32.
- **TF32 profile** (`apply_tf32_profile`) — opt-in, default OFF, no-op on Turing.

### Debugging / UX
- **Per-node tier/timing HUD** — a badge under each TEX node shows which acceleration tier
  served the cook, the time, and the precision (amber on a tier fallback). Renders on the
  classic (Nodes 1.0) canvas; the Nodes-2.0/Vue render path lands in v0.19.
- **`tex doctor`** — a `/tex_wrangle/doctor` route reporting torch/CUDA, Triton presence,
  MSVC, cache size, and which tiers are actually reachable on your box (queryable via the
  route; a one-click UI panel lands in v0.19).
- **Hover docs** in the code editor (signature + description).
- **NaN/Inf overlay** — `debug_nan_highlight` paints non-finite pixels magenta.
- **`debug_print(label, value[, x, y])`** — a value-at-pixel probe (returns the value
  unchanged); the probed values are returned in the node's `ui` payload (on-node display
  lands in v0.19).
- **Better diagnostics** — declaring a variable named `v`/`u`/`ix`/… now explains it's a
  built-in and to rename; `float3`/`texture2D`-style mistakes point to the TEX name.
- **Honest tooltips** — compile-mode tooltips state the Triton reality.

### Portability
- **`tex run` CLI** — run a `.tex` program on an image file with **no ComfyUI**
  (`python -m TEX_Wrangle.tex_cli run prog.tex --in a.png --out b.png`); torchvision-only I/O.
- **Public API** — `tex_api.compile()` / `execute()` + a stable `Program` dataclass.
- **Host seam** — `comfy.model_management` is now behind a single `HostServices` interface,
  pinned by a lint; TEX runs host-agnostic (a Null host when ComfyUI is absent).

### Memory / hardware / safety
- **Graph-safe cache eviction (safety fix)** — cache-budget eviction no longer tears down
  every captured CUDA graph (nor resets the RNG-poison kill switch) on an unrelated
  eviction; it pins the graph's baked storages and evicts only what's actually free.
- **Reserved-pool trim** — reclaims stranded VRAM after a big→small resolution downshift
  (measured ~1.5 GB back for ~3.4 ms), threshold-gated so it never fires at steady state.
- **Per-device cache budget** — a CPU cook no longer evicts CUDA-resident cache entries.
- **Multi-GPU correctness** — CUDA-graph capture/recovery pin the cook's device index.
- **`TEX_CPU_THREADS`** — opt-in CPU-thread override (never auto-set).

### Internal / honesty
- Cross-device parity pinned as a *characterization envelope* (there is no CPU↔GPU bit
  parity to sell — it's already 1.8e-7…6.1e-2); determinism pinned (TEX is bitwise
  run-to-run deterministic on CUDA — a free property). Both machine-checked.
- All 143 stdlib functions (144 callable names incl. the `mix`→`lerp` alias) carry inline
  `doc=`/`ex=`; the reference is generated from them. AGENTS.md map-drift canary. No
  user-facing behavior change from these.

## [0.17.0] - 2026-07-08

Longevity / LLM-coding / structure release. **No user-facing behavior change** — the
interp↔codegen bit-exactness contract holds across every refactor (verified: the full
suite stays green, 1683→1689 sub-tests, plus a live-GPU cuda_graph parity pass). The
whole cycle makes the codebase navigable, its invariants machine-enforced, and its
biggest modules decomposed.

### The LLM map + machine-enforced invariants
- **`AGENTS.md`** — the entry point: pipeline map, a MUST-NOT-BREAK invariant table (each
  naming its enforcing test), the corrected stdlib recipe, the DO-NOT-TOUCH register, a
  module LOC-budget policy, and a doc-layering policy.
- **`ARCHITECTURE.md`** — single module-graph/layering source; **`Function-Reference.md`**
  and **`examples/INDEX.md`** are now *generated* views (drift-tested).
- New safety net: a **grammar-driven differential fuzzer** (interp↔codegen; nightly at
  N=2000), an **edge-input matrix** (fp16/1×1/batch/int64 × every fn × tier), **tier-
  execution observability**, taxonomy-consistency checks (closes the `_NON_LOCAL_FNS`
  wrong-when-tiled trap), operator-completeness, a release gate (version-consistency +
  hash-seed determinism), a numpy-ban lint, coverage tooling, and a runner-drift guard.

### Single-source spine
- **`TEXType` → `tex_compiler/types.py`** (a dependency-free leaf; breaks the
  stdlib_signatures↔type_checker cycle; checker fan-in 9→~2).
- **Single-source stdlib registry** — one `@stdlib(...)` decorator per function replaces
  the hand-maintained 143-entry map + the parallel taxonomy tables; `codegen`'s emit
  dispatch self-registers likewise. Documentation drift closed (8 previously-undocumented
  functions added; 47 dispatch methods + 140 stdlib return-type hints).

### Decomposition (bit-exact code motion)
- **`codegen.py` split 4092→~2730** across `codegen_stdfns` / `codegen_stencil` /
  `codegen_persist` (strict DAG, gated cluster-by-cluster on the fuzzer + edge matrix).
- **`execute()` 388→277** — the tier cascade is now a pure, CPU-testable `select_tier`
  classifier + a strategy registry; the duplicated recovery path is single-sourced.
- Optimizer pipeline is a data-driven `PASSES` list; a CPython-style `NodeVisitor` base
  backs the pure-traversal optimizer walks; fusion reuses a shared `compile_ast`.

## [0.16.0] - 2026-07-07

Correctness-and-honesty release driven by the v0.16 roadmap, with a full
per-item build log. Every performance claim is a
**measured, same-session interleaved A/B** (this box drifts 10–30 %/hour, so
full-suite deltas are noise-dominated). Several roadmap items were **measured and
then NOT adopted** because the measurement refuted their premise — that is the
headline of this release. 1649 sub-tests pass.

### ⚠️ Breaking — new reserved built-in names
The stdlib additions below reserve their names; TEX forbids a user function from
redefining a built-in. If your program defines a function named any of
`over` `under` `atop` `premultiply` `unpremultiply` `srgb_to_linear`
`linear_to_srgb` `oklab_from_rgb` `oklab_to_rgb` `screen` `overlay` `hard_light`
`soft_light` `color_dodge` `color_burn` `linear_light` `vivid_light`
`erode` `dilate`, rename it (e.g. prefix `my_`).

### Added
- **Color management** — `srgb_to_linear` / `linear_to_srgb` (piecewise sRGB
  EOTF/OETF) and `oklab_from_rgb` / `oklab_to_rgb` (Ottosson OKLab). Blur/blend in
  linear-light to avoid gamma-space halos; mix in OKLab for perceptually-even
  gradients.
- **Compositing** — `over` / `under` / `atop` / `premultiply` / `unpremultiply`
  (Porter-Duff on straight-alpha RGBA, ComfyUI's un-premultiplied convention).
- **Blend modes** — `screen` `overlay` `hard_light` `soft_light` `color_dodge`
  `color_burn` `linear_light` `vivid_light` (curated set).
- **Morphology** — `erode` / `dilate` (iterative separable min/max; O(1) extra
  memory in the radius). All 19 new stdlib functions run bit-identically on the
  codegen tier and appear in the editor autocomplete.
- **`const` arrays** — `const float lut[3] = {…};` is now accepted (was a parser
  error); const arrays reject reassignment and element writes.
- **Failure-mode test harness (PROC-1)** — a reusable `tests/failure_harness.py`
  covering the five classes (async lifecycle, restart/persistence, real
  entry-point, cross-tier equivalence, full-surface sweep) that the v0.15 suite
  was structurally blind to; every v0.16 fix plugs its regression test into it.

### Fixed
- **CUDA-graph tier is never worse than eager (PF-1/PF-2).** The tier captured
  unconditionally and was a *measured loss* above ~1024² for low-kernel programs
  and at every resolution for ~0-kernel programs. A crossover gate now captures
  only in the measured win region (256²/512² for kernel-bearing programs),
  preserving the 3.5–6.5× wins while removing the 0.7–0.96× losses.
- **Octave-noise no longer eats doomed graph captures (P1-UC1-STATIC-GATE).** The
  octave/count noise family (`fbm`/`ridged`/`billow`/`turbulence`/`flow`/
  `alligator`) is excluded from capture (its count resolves via a capture-illegal
  `.item()`); single-eval noise (`perlin`/`simplex`/`worley`/`voronoi`/`curl`)
  stays capturable and wins ~6× at 256².
- **Negative-literal constants fold (P2-UC4-NEG).** `float k = -0.5;` is now
  constant-propagated (it parses as a unary-minus and was previously missed).
- **int64 tensor bindings keep the codegen path (P2-M5-INT).** A wired int tensor
  binding no longer forces a silent fallback to the interpreter on the M-5 `out=`
  reuse (integer image tensors are cast to fp32 at codegen ingestion).
- **CPU cache budget (P1-M2-CPU).** The mip/grid/sampler byte budget is now
  enforced on CPU cooks too (was CUDA-only → unbounded growth on CPU installs).
- **OOM frees TEX's caches (P1-M1-FREERETRY).** On OOM the node drops its own
  tensor caches before re-raising, so ComfyUI's unload+retry has that memory.

### Changed / measured-not-adopted
- **`fp16` stays experimental.** A full re-measure (88 programs) shows fp16 is
  accurate for smooth-pointwise programs (median ~1e-3) but diverges badly on
  threshold/quantize/branch programs (up to 1.0) and can NaN — it must not be a
  default.
- **Default CPU codegen routing was evaluated and NOT adopted (PF-4).** On the
  post-v0.15 codebase, the interpreter's own optimizations closed codegen's lead;
  a blanket route would *regress* the dominant color-grade shape ~30%. Measurement
  prevented a regression.
- **`auto` tier hardening, torch.compile persistence, and DAG-fusion widening are
  deferred** — unreproducible/unvalidatable on this no-Triton box or frontend-
  heavy.

## [0.15.0] - 2026-07-07

Optimization-roadmap release: all 24 proposals from the 2026-07 TEX Optimization
Roadmap, implemented in priority order, then hardened by a pre-push audit.
Every item ships with a regression test, and every
performance claim below is a **measured, same-session interleaved A/B** on the
affected programs (this box drifts 10–30 %/hour, so full-suite deltas are
noise-dominated and were not used). 1590 sub-tests pass.

### Fixed — pre-push audit
- **UC-3** — uniform-range loop resolution now only fires on integer-valued
  bounds; a fractional `for(float …)` bound/step (or a bound reading a
  body-mutated binding) falls back to the exact per-iteration path instead of
  silently changing the loop values vs v0.14.1.
- **UC-4** — an array shadowing a same-named literal local no longer corrupts the
  const-propagation pass (was a spurious compile error on legal code).
- **Q-5** — the chain-preflight endpoint now succeeds on valid chains (it seeds a
  placeholder terminal binding); previously every fused chain showed a false-red
  "not fusable" bubble.
- **M-1** — an out-of-memory error raised *inside* a stdlib call (sample_mip,
  gauss_blur) is re-raised unwrapped so ComfyUI's OOM handling (memory summary +
  model unload) fires, instead of being masked as a generic error.
- **M-3** — fp16 mode no longer hard-fails on `mix`/`lerp`/`fit`/`smin`/`smax`/
  `sample_mip`/`gauss_blur` (dtypes are reconciled around the strict `torch.lerp`
  / `conv2d` ops); ~5e-4 vs fp32.
- **UC-1** — `cuda_graph` now stages vec/color params as `[1,1,1,C]` tensors
  (were collapsed to the R component → silent wrong output); a cache-budget
  eviction that could free a graph-baked tensor now tears down the graph cache.
- **UC-2** — the codegen/`auto` stencil path refuses sample-based (bilinear)
  stencils so it stays bit-exact with the interpreter (fetch stencils keep the
  fast avg_pool lowering).
- **M-4** — tiled execution refuses to tile a LATENT (`[B,C,H,W]`) or bindings of
  disagreeing height, falling back to an untiled cook.
- **CC-1** — the Triton-on-Windows install hint now fires at the point the failure
  actually occurs (first compiled call), and a missing Triton marks the backend
  unavailable instead of blacklisting the program.
- **Q-3** — preview-tap exports are capped at `MAX_OUTPUTS` (8), dropping excess
  taps with a log instead of overflowing the node's output slots.
- **Hygiene** — codegen in-memory cache is LRU-bounded again; the `out=`-reuse
  kill switch is folded into the cache key; the dead "last-cook ms" HUD field was
  removed; `compile_mode="auto"` is labelled experimental (its background-compile
  timing is still being hardened — it only ever falls back to a correct path).

### Added — compiled-code persistence (restarts are free)
- **PC-3 — persisted codegen objects**: the generated Python code object is
  marshalled to a `.cg` sidecar (validated by cache version + CPython bytecode
  magic + SHA-256). A warm restart materializes from disk instead of re-emitting.
  Measured **2.2–6.4× faster** first-cook-after-restart on the codegen path.
- **PC-2 — precompile safety**: `caching_precompile` entries attach with a
  crash-signature allowlist; a stale/corrupt entry clears the dynamo store and
  recompiles fresh instead of blacklisting.
- **CT-1 — fused-chain disk persistence**: a fused chain's compiled artifact
  survives restart (skips the ~2.5 ms splice+double-typecheck+optimize).
  Measured **7.5×** on the restart path.

### Added — CUDA-graph & codegen routing
- **UC-1 — CUDA-graph replay** (`compile_mode="cuda_graph"`): captures and
  replays the unmodified interpreter per (program × input-signature). Measured
  **up to ~6.2× aggregate** on small launch-bound GPU programs; per-graph pools,
  RNG-poison recovery, bytes-aware LRU.
- **Q-1 — fused chain as the capture unit**: a fused chain gets a first-class
  fingerprint and can route through the graph tier as one capture region.
- **UC-2 — stencil-codegen routing**: exact fetch/conv stencils default-route
  through the codegen tier (avg_pool2d/conv2d/unfold lowering).
- **UC-3 — uniform-range loop analysis** and **UC-5 — literal array indexing**
  broaden what the graph/codegen tiers accept.
- **CC-2 — measured auto-tier** (`compile_mode="auto"`, opt-in): starts on the
  always-safe codegen path, compiles in the **background without ever stalling
  the cook** (the 28 s foreground compile is gone), trials the compiled fn timed,
  and switches only on a measured >10 % win over codegen-only. Verdicts persist.
- **CC-1**: a Triton-on-Windows install hint when CUDA inductor is unavailable.

### Added — memory cooperation
- **M-1/M-2 — peak estimator + byte-budgeted cache eviction**: preflight OOM
  and cap TEX's tensor-cache residency at a VRAM/CPU byte budget (env override
  `TEX_CACHE_BUDGET_MB`).
- **M-4 — tiled (strip) execution**: under GPU memory pressure a tile-safe
  program runs in horizontal strips with seam-exact coordinates. Measured 4096²:
  peak **1074 → 612 MB** at 8 strips (transient 0.24×); seams bitwise-identical.
- **M-3 — fp16 image-data mode** (`precision="fp16"`, opt-in): fp16 image data,
  fp32 coordinates (a fp16 `u` would collapse at high res). Peak/churn win is
  **conditional** on an fp16-native input (0.69×/0.63×); neutral-to-worse when
  fed fp32 — tooltip states this. Output stays fp32 (IMAGE contract).
- **M-5 — codegen `out=` temp reuse**: reuses a dead fresh arithmetic temp as
  the `out=` target for constant-scalar ops. Measured **26 % fewer allocator
  calls** on the color-grade chain, bit-exact, timing-neutral (kill switch
  `TEX_CODEGEN_NO_OUT_REUSE=1`).

### Added — chained-node QOL
- **Q-3 — fusion coverage widening (backend)**: `compile_fused` generalized from
  a linear chain to a **DAG** — stages can read any earlier stage's output
  (`chain_inputs`), export multiple outputs (`exports`), or expose an observed
  intermediate as a `@_tap_s{i}` output (`tap`). All bit-exact vs unfused; the
  legacy linear path is byte-identical. (The frontend maximal-component collapse
  is pending a live-ComfyUI validation pass; the backend accepts payloads now.)
- **Q-5 — chain preflight + perf HUD**: `POST /tex_wrangle/chain_preflight`
  validates a drawn chain's fusability as you edit — the fusion bubble turns
  **red** with the blocking node/reason before you queue, instead of failing the
  queued prompt. Green bubble shows stage count / op estimate. Setting
  `TEX Fusion: Preflight chains + perf HUD` (default on).
- **Q-4 — fused-chain error attribution**: a runtime error inside a fused chain
  names the originating linked node (stage-tagged `SourceLoc`).
- **Q-6 — low-res live preview (scaffold)**: a preview downscale primitive +
  `_tex_preview` kwarg; the live-preview orchestration remains gated on ComfyUI's
  unverified `partial_execution` API (exploratory).

### Changed — compile times
- **CT-2 — offset-based lazy source locations**: `SourceLoc` is now offset-backed
  and resolves line/col only when a diagnostic renders, dropping the lexer's
  per-token bookkeeping. Measured **~8–9 % faster lexing** (first-compile only).
  Reconciled with Q-4 (loc stays a mutable object carrying `stage`).
- **UC-4 — const-propagation pass** and **Q-2 — purity-aware DCE** in the
  optimizer.

### Notes
- New `compile_mode` options: `auto` (measured auto-tier) and `cuda_graph`.
- New `precision` option: `fp16` (interpreter-only; forced to fp32 under any
  compile mode).
- `fp16` and the graph/compile tiers are opt-in; the default path
  (`compile_mode="none"`, `precision="fp32"`) is unchanged.

## [0.14.1] - 2026-07-06

### Fixed (correctness)
- **Codegen scalar-loop misclassification after a spatial `if`** — a spatial
  `if`/`if-else` that assigns a scalar-typed variable turns it into a
  `torch.where`-merged `[B,H,W]` tensor at runtime, but the merge left the
  codegen's `_spatial_vars` set reflecting only the last-emitted branch. A
  subsequent static `for`-loop then classified the variable as scalar and
  emitted `.item()` on the tensor, raising at runtime and silently falling back
  to the interpreter (a correctness-preserving but slow path). The merge now
  marks every possibly-spatial merged variable so the following loop stays in
  tensor mode. (Follow-up to the 0.14.0 CG-P4 if/else single-emit change, which
  the audit had deferred.)

## [0.14.0] - 2026-07-02

Full-engine optimization and cleanliness audit: every layer was audited by
independent reviewers, every finding adversarially verified before
implementation, and every performance change proven (or reverted) with
before/after benchmarks on CPU and CUDA.

### Fixed (correctness)
- **In-place aliasing holes in the interpreter** — `@OUT = c; c = c + 1.0;`
  no longer mutates the already-stored output; user-function parameters can no
  longer mutate the caller's variable (or a cached literal) through the
  in-place fast path; function passthrough returns (`return p;`) and
  view-returning stdlib calls now correctly invalidate in-place readiness
- **Scatter writes no longer corrupt shared buffers** — `@OUT = @A;
  @OUT[x,y] = v;` used to write straight into the *upstream node's* image;
  scatter now clones any buffer this execution doesn't own (also protects
  cached literals and builtin coordinate grids reached through views)
- **Codegen conv2d stencil ordering** — the inline-stencil fast path could
  emit the convolution *before* statements that textually precede it (e.g. a
  binding reassignment), silently diverging from the interpreter; emission is
  now order-preserving, with hardened pattern detection and hoisted-BCHW
  invalidation
- **Optimizer dropped `is_int` on negation** — `-3` folded to a float literal,
  which could fail the cache's re-type-check with a hard E3200 on valid
  programs (e.g. negative constants reaching array indices)
- **`for (…; @X[a,b] += v)` mis-desugared** — compound scatter assignment in a
  for-loop header parsed to a plain write, dropping the accumulate
- **Elementwise builtins mistyped** — `step`, `smoothstep`, `clamp`, `fit`,
  `pow`, `mod`, `atan2`, `hypot`, `spow`, `sdiv` now type as the promotion of
  their arguments instead of the first argument's type, so
  `@OUT = smoothstep(0.0, 1.0, @A);` correctly infers an IMAGE output (was
  MASK). Inferred output socket types can change for affected programs —
  matching what the runtime always produced
- **Stencil-tap AST walker drift** — array-literal-referenced taps could be
  consumed by the stencil detector and crash the generated code at runtime
  (silent fallback + spurious blacklist); the three duplicated AST walkers are
  now one shared traversal
- **Fused-path error messages** — errors in fused chains showed the internal
  `u_` prefix (`@u_amt` instead of `@amt`)
- **`fbm`/`ridged`/`billow`/`turbulence` hard-errored on CUDA without Triton**
  (since 0.13.0) — the noise compile tier gated on MSVC and warmed on a CPU
  dummy, so a CUDA-poisoned compiled callable was cached and raised at every
  later call; the gate is now per-device (CUDA requires Triton) and warmup
  happens on the target device, falling back to the traced tier

### Changed (performance)
- **Fused chains compile once** — the fusion path now memoizes spliced
  programs (LRU, keyed by per-stage code/topology/binding types); previously
  every queue execution re-parsed, re-spliced, re-type-checked (twice) and
  re-optimized the whole chain, which could cost more than fusion saved
- **Channel/array writes are copy-on-first-write** — `c.r = …` / `arr[i] = …`
  cloned the full tensor on *every* write; now only the first write clones
  (guarded by AST alias analysis plus a runtime storage-overlap check), so an
  N-element array fill loop does O(N) work instead of O(N²)
- **Literal tensors persist across executions** — constants are no longer
  re-uploaded every frame (bounded cache; matters most on the launch-bound
  CUDA path)
- **`clamp()` is sync-free on CUDA** — 0-dim tensor bounds previously forced
  two `.item()` pipeline flushes per call; bounds now pass through as tensors
  on CUDA while CPU keeps the faster scalar-overload kernel
- **codegen-only executions are cached** — deep-loop programs routed to the
  codegen backend under `compile_mode="torch_compile"` regenerated and
  re-`exec`'d their source every frame; the adapter and its gate verdicts are
  now cached per fingerprint (the compiled-cache key also gains `precision`)
- **$params hoisted in generated code** — one `as_tensor` per parameter per
  execution instead of one per use-site per loop iteration
- **Scalar-loop if/else emits once** — nested if/else in scalar loops emitted
  both vectorized and scalar bodies (2× per nesting level of dead generated
  source)
- **torch.compile backend failures classified** — a Triton-less CUDA box now
  marks the backend unavailable per-device after the first failure instead of
  paying a full failed inductor compile for every new program (and no longer
  disables CPU compilation because CUDA failed)
- **Compiler frontend** — batched lexer scanning (no per-character method
  calls), `slots=True` tokens, two-char-operator table, memoized CSE/LICM
  subexpression info (O(n²) → O(n) on fused-sized programs), exact-class
  dispatch in the type checker, hoisted per-node imports
- **CSE/LICM pure-function list fixed** — the list named seven functions that
  don't exist (`rgb_to_hsv`, `luminance`, `saturate`, …) while the real ones
  (`rgb2hsv`, `hsv2rgb`, `luma`, …) were missing, so common color conversions
  were never deduplicated or hoisted
- **Interpreter micro-hots** — unbound-method in-place ops, hoisted
  fetch/sample dispatch, cached flat batch index for scatter writes,
  device-object cache keys, true-LRU sampler caches
- **fp16-safe zero-divisor guards** — division/mod guards use a
  dtype-aware epsilon (1e-8 underflows to zero in fp16)
- **Frontend (JS)** — the help-overlay RAF loop no longer does per-frame
  `querySelector` calls in Nodes 1.0 or redundant style writes in Nodes 2.0;
  CodeMirror editors are destroyed on node removal (leak); param schema
  updates are refcounted instead of last-writer-wins on the shared node type
- Benchmarks: `run_benchmarks.py` now sync-brackets CUDA timing (previously
  measured kernel-launch enqueue only); `eight_config_bench.py` probes the new
  compiled-cache key

### Removed
- Dead `tile_offset` plumbing (never wired to any caller), dead `_STDLIB`
  fusion constant, dead `source_key` legacy payload, dead `cache_key`
  parameter, unreachable `prepare_output` CPU fallback, `TypeChecker.get_type`
  (no callers), stale/contradictory comments and duplicated branches
  throughout

### Documented
- Two measured CPU performance traps now carry explanatory comments so they
  can't be "fixed" back: the sample-grid buffer deliberately allocates fresh
  while holding the previous allocation (true reuse measured ~30% slower on
  CPU), and `clamp()` bounds are device-split
- Version strings aligned (`__init__.py`/`pyproject.toml` said 0.12.0 while
  the changelog shipped 0.13.0)

## [0.13.0] - 2026-06-27

### Added
- **Cross-node fusion** — a chain of linked TEX nodes compiles into a single program so only the terminal node cooks; intermediate nodes never materialize or cache an image. On by default — opt out via ComfyUI Settings (`TEX Fusion: Compile linked TEX nodes together`). At queue time the frontend collapses the chain into its terminal node, marking the fused region with a faint Houdini-style bubble (a rounded convex hull, or a plain rectangle via `TEX Fusion: Bubble as convex hull`; toggle the whole bubble with `TEX Fusion: Show grouping bubble`) that reveals a `TEX fused` label on hover; the backend (`tex_fusion.py`) splices each stage's `@OUT` into the next as one re-type-checked, re-optimized program — bit-equivalent to running the nodes separately. A chain breaks (and runs unfused) at a Preview/Save tap, a fan-out, a multi-input or multi-output node, a scatter write to `@OUT`, or `@OUT` used inside a loop.
- **GPU: batched noise octaves** — `curl`, `fbm`, `ridged`, `billow`, and `turbulence` evaluate all Perlin octaves in a single batched call on CUDA (~3.2x faster at 512², bit-exact; CPU keeps the per-octave path)
- **GPU: faster `dot()`** — uses `mul + sum` on CUDA instead of `einsum` (~9.8x for vec3, ~5.8x for vec4; numerically equivalent), speeding up `dot`/`luma`/`normalize`/lighting math
- **On-device output** — IMAGE and MASK node outputs stay on their compute device (a GPU output stays on the GPU), so chained TEX nodes avoid CPU↔GPU round-trips; terminal Save/Preview nodes move to CPU themselves
- **Worley-3D offset cache** — the 27-neighbour offset meshgrid is cached per device instead of rebuilt on every call
- **Error code `E6051`** — a function's *runtime* failure now has its own code, distinct from `E6050` (unknown function)

### Changed
- **Faster input fingerprinting** — `tensor_fingerprint` hashes the raw sample bytes instead of formatting 256 floats into the key (~2x faster; runs for every input every frame)
- **Friendlier diagnostics** — runtime (E6xxx) errors now render the source-line caret; a nested error keeps its own code/location/hint instead of being re-wrapped at the wrong place; ordinary user/config errors (device, output type, input-size mismatch) are shown cleanly instead of as TEX bugs with a bug-report link

### Fixed
- **`fetch()` shape at batch=1** — a mixed spatial+scalar fetch such as `@A[ix, ih-1.0]` no longer collapses a spatial axis; output shape now matches the input across batch sizes
- **Error attribution** — a nested error raised while evaluating a function argument is no longer re-labelled at the wrong location under the wrong function
- **`inverse()`** — reports "singular matrix" only for a genuine `LinAlgError`; other failures (e.g. CUDA OOM) keep their real cause

## [0.12.0] - 2026-04-06

### Added
- **Sample/fetch inline emission** -- codegen now emits `grid_sample` directly for sample() calls inside loops, bypassing the stdlib wrapper. Pre-allocated grid buffers eliminate `torch.stack` allocation overhead (~0.1ms per call)
- **Sample/fetch dispatch table** -- `sample()` and `fetch()` function calls use inline emitters when bindings are hoisted, eliminating Python function call overhead
- **`bilateral_filter(@img, sigma_s, sigma_r)`** -- new stdlib function for edge-preserving blur using `Tensor.unfold`. Optimal for small kernels (3x3)
- **Sample-to-fetch conversion** -- detects `sample(@img, u + expr*px, v + expr*py)` patterns and emits direct tensor indexing instead of grid_sample

### Changed
- **Benchmark binding detection** -- replaced fragile regex-based binding detection with proper 2-pass type checking. Fixes 20 previously erroring example programs
- **Graph break reduction** -- ternary string guard skipped for numeric types, param `isinstance` guard skipped for known numeric types, fetch dict lookup hoisted to preamble

### Fixed
- **Benchmark harness: 20 programs erroring** -- `generate_bindings` now uses AST-based type checking to discover ALL referenced bindings, not just hardcoded names like `@image`/`@A`
- **film_exponential_blur codegen** -- fixed channel mismatch (was creating 4-channel bindings for 3-channel programs)
- **Codegen `_has_fn_calls` metadata** -- attached to compiled functions for future torch.compile optimization routing

### Documented
- **Parameter widget types** -- README, tutorial, and wiki updated with `b$` (boolean), `c$` (color hex), `v2$`/`v3$` (vector) parameter examples
- **`bilateral_filter`** -- added to Function Reference wiki and README stdlib section
- **Wiki Unicode fix** -- replaced em-dashes and special characters with ASCII equivalents, fixing 6 broken wiki pages

## [0.11.0] - 2026-03-31

### Added
- **Stencil specialization** — nested for-loops matching spatial filter patterns are replaced with bulk PyTorch ops: `avg_pool2d` (box blur), `max_pool2d` (min/max reduction), `Tensor.unfold` (median/rank filters), depthwise `conv2d` (inline weighted stencils like sharpen)
- **Codegen-only execution path** — programs with deep loop nesting (>2 levels) bypass `torch.compile` and run the codegen flat function directly, avoiding tracing overhead on loop-heavy programs
- **Codegen function specializations** — `pow()` constant-exponent cases (-0.5, 4.0, -2.0), `clamp()`/`step()` with float literals, `luma()` as direct channel arithmetic instead of einsum
- **Pre-resolved stdlib locals** — codegen hoists `_fn_X = _fns['X']` to preamble, replacing per-call dict lookups
- **While-loop native break/continue** — while loops now emit Python-native flow control instead of exception-based `_CgBreak`/`_CgContinue`
- **Smart execution routing** — programs are routed to the fastest path based on analysis: plain interpreter (trivial or no-image programs), codegen-only (deep loops), or codegen+torch.compile (spatial tensor chains)
- **Cross-validation in compiled tests** — `test_example_files_compiled` now verifies codegen output matches interpreter output (max_diff < 0.01) for every program with codegen support

### Changed
- **Codegen dispatch table** — replaced 385-line `_emit_function_call()` with a dispatch table mapping ~40 function names to 11 handler methods, reducing branching and improving maintainability
- **Codegen for-loop split** — broke 191-line `_emit_for_loop()` into `_emit_static_for_loop()`, `_setup_scalar_loop()`, `_setup_tensor_loop()`, and `_emit_general_for_loop()`
- **Interpreter if/else extraction** — extracted `_exec_spatial_if()`, `_snapshot_vars()`, and `_merge_branch_vars()` from `_exec_if_else()` for clarity
- **Type checker array flattening** — replaced 89-line 8-level nested `_check_array_decl()` with `_check_array_initializer()`, `_check_array_literal()`, `_check_array_copy()`, and `_resolve_array_size()`
- **Marshalling extraction** — moved marshalling and type inference utilities from `tex_node.py` (547→360 lines) into new `tex_marshalling.py` module
- **Noise 3-tier cache** — replaced per-noise-type boilerplate (simplex, FBM, Worley each with separate cache/lock/counter dicts) with a shared `_TieredCache` class (~70 lines of duplication removed)
- **Noise table cleanup** — removed unused `_GRAD2_SIMPLEX` table and simplified `_get_noise_tables()` from 5-tuple to 3-tuple

### Fixed
- **Type-aware fetch/sample return** — `fetch()`, `sample()`, and all sampling variants now return the binding's actual type (VEC3 for IMAGE, FLOAT for MASK) instead of hardcoded VEC4; prevents type mismatches and enables better codegen optimization
- **Codegen inference_mode conflict** — removed in-place accumulation optimization (`add_`, `mul_`) that was incompatible with `torch.inference_mode()`, fixing "Inplace update to inference tensor" errors that blocked 21 programs from using codegen
- **Sampling grid buffer cache** — `_grid_buf` now detects inference-mode tensors and recreates them, preventing cross-context mutation errors when interpreter and codegen run in the same session
- **Optimizer stencil preservation** — nested for-loops are no longer unrolled by the optimizer, preserving the structure needed for stencil pattern detection
- **Inline stencil tap variable leaking** — tap variables referenced in later statements (e.g., `center` in sharpen.tex) are no longer consumed by the stencil optimization, preventing NoneType errors
- **Unary string type error fallthrough** — `_check_unary()` in type_checker.py now returns a fallback type after reporting an error for unary operators on strings, preventing cascade type errors
- **Memory leak: unbounded `_grid_buf` cache** — sampling grid buffers (~16 MB at 1080p per entry) now use OrderedDict with LRU eviction (max 16 entries) instead of an unbounded dict
- **Memory leak: unbounded `_sampler_cache`** — batch index and Lanczos tap tensors now use OrderedDict with LRU eviction (max 32 entries)
- **Memory leak: unbounded `_compiled_cache`** — torch.compile compiled callables (~30-60 MB of Inductor kernels per entry) now use OrderedDict with LRU eviction (max 16 entries, ~0.5-1 GB ceiling) with `torch._dynamo.reset()` on eviction to reclaim kernel memory
- **Memory leak: unbounded `linecache` growth** — codegen now prunes old `<tex_codegen_N>` entries, keeping only the most recent 64
- **Race condition in noise `_TieredCache.try_upgrade()`** — call counter increment moved inside the lock to prevent duplicate torch.compile attempts under concurrent access

### Documented
- **`type_hint` contract** — documented valid values for `ParamDecl` ("f", "i", "s", "b", "c", "v2", "v3", "v4") and `BindingRef` ("f", "i", "v", "v2", "v3", "v4", "img", "m", "l", "s", "") in `ast_nodes.py`

## [0.10.0] - 2026-03-26

### Added
- **`vec2` first-class type** — 2-component vectors with constructors (`vec2(a, b)`), swizzles (`.xy`, `.rg`), arithmetic, and promotion chain `int → float → vec2 → vec3 → vec4`; vec2 outputs auto-pad to 3-channel IMAGE
- **Array codegen** — programs using arrays now run through the codegen path instead of falling back to the tree-walking interpreter; supports array declarations, literals, constant/dynamic index access, and vec/string arrays
- **User function codegen** — user-defined functions (`float foo(float x) { return x * 2.0; }`) now compile to nested Python `def`s in codegen with depth-limited recursion, matching interpreter semantics
- **Scatter writes** — `@OUT[px, py] = value;` writes to arbitrary pixel positions with last-write-wins semantics; compound assignments (`+=`, `-=`, `*=`, `/=`) use `index_put_` with accumulation; optional 3rd frame argument (`@OUT[x, y, frame]`)
- **Multi-channel assignment** — `c.rgb = vec3(1.0, 0.5, 0.25);` and `c.xy = vec2(0.5, 0.8);` now work in both interpreter and codegen; alpha channel preserved when assigning `.rgb` on a vec4
- **`const` qualifier** — `const float PI2 = 6.28;` declares read-only variables; type checker rejects reassignment or channel modification of const variables (E3204)
- **`trunc()` function** — truncates toward zero (`trunc(-2.7)` → `-2.0`), with codegen fast-path inlining
- **8 new noise functions** — `worley_f1`/`worley_f2`/`voronoi` (cell-based distance noise), `curl` (divergence-free flow field → vec2), `ridged`/`billow`/`turbulence` (FBM variants), `flow` (time-varying domain-warped Perlin), `alligator` (layered cell noise with ridge accumulation); all use arithmetic hash for TorchInductor compatibility
- **3D noise** — all noise functions now accept an optional `z` parameter for 3D evaluation: `perlin(x,y,z)`, `fbm(x,y,z,octaves)`, `worley_f1(x,y,z)`, etc.; 3D Perlin uses the classic 12-gradient set with arithmetic hash; 3D Worley searches 27 cells; `curl(x,y,z)` returns `vec3` (divergence-free 3D flow); 2D calls remain backward compatible
- **SDF primitives** — `sdf_circle(px,py,radius)`, `sdf_box(px,py,half_w,half_h)`, `sdf_line(px,py,ax,ay,bx,by)`, `sdf_polygon(px,py,radius,sides)` for signed distance fields (negative inside, positive outside); polygon supports any side count ≥ 3
- **`smin`/`smax`** — polynomial smooth minimum and maximum with smoothing radius `k`, for organic blending of SDF shapes
- **`sample_grad(@A, u, v)`** — samples the luminance gradient of an image at UV coordinates, returning `vec2` (horizontal, vertical) via central finite differences
- **`sample_mip(@A, u, v, lod)`** — mipmap sampling with explicit level of detail; builds a cached pyramid per input with area downsampling; trilinear filtering between levels; fast path skips interpolation when LOD is a uniform integer; per-pixel LOD supported for effects like tilt-shift
- **`TAU` constant** — `TAU = 6.28318…` (2π) available alongside PI and E
- **`px` / `py` built-in variables** — pixel step in UV space (`1.0 / iw`, `1.0 / ih`), eliminating boilerplate in sampling kernels
- **`sincos(x)` function** — returns `vec2(sin(x), cos(x))` in a single call
- **29 new example programs** — grade (Nuke-style), STMap, turbulent displacement, simple/film lens distortion, 2D transform, corner pin, distortion map, image gradient, convolve, directional blur, vector blur, temporal median, luma keyer, erode/dilate, premultiply, frame blend, time echo, normalize mask, soft clamp, tilt-shift, plus 8 film-quality examples: film vignette (cos⁴ + optical + mechanical), film grain (density-domain), grain (simplified), film chromatic aberration (spectral N-band), denoise (NLM in YCoCg), chroma keyer (Vlahos color-difference), optical flow (Lucas-Kanade), ZDefocus (scatter-as-gather spiral)

### Changed
- **Promotion chain expanded** — type promotion now includes vec2: `float → vec2 → vec3 → vec4` with zero-padding for channel promotion between vector sizes
- **2-component swizzle** — `.xy`/`.rg` swizzles now return `vec2` instead of raising error E3303
- **Noise extracted to `noise.py`** — all procedural noise functions moved from stdlib.py to a dedicated module; stdlib.py imports the public API; no user-facing changes
- **Test suite restructured** — split monolithic test_tex.py (11,408 lines) into 13 domain-specific files with shared helpers module; 77 test functions containing ~1,215 sub-tests (was 61 functions); new coverage for optimizer passes, node helpers, stdlib edge cases, NaN/Inf propagation, and all 114 example files run through the full pipeline (interpreter and torch.compile paths); dual runner support (pytest and standalone); added `@pytest.mark.slow` for timing tests

### Fixed
- **`distance()`, `length()`, `normalize()` with vec2** — these functions rejected 2-component inputs due to a `shape[-1] in (3, 4)` guard; now accepts vec2 inputs correctly

## [0.9.0] - 2026-03-26

### Added
- **Arithmetic hash Perlin noise** — `perlin` and `fbm` now use a purely arithmetic gradient hash instead of permutation table lookups, enabling TorchInductor fusion and **9x faster FBM noise** (284ms → 22ms at 512x512)
- **Codegen program-level locals** — all env variables are pre-registered as Python locals (`_lv_{name}`), eliminating dict lookups on every read/write and producing cleaner FX graphs for TorchInductor
- **Four-scenario benchmark** (`benchmarks/four_scenario_bench.py`) — measures cook times across compile off/on × cold/warm start for 8 representative programs

### Changed
- **`torch.scalar_tensor` optimization** — all scalar constant creation (`torch.tensor(scalar)`) replaced with `torch.scalar_tensor(scalar)` across interpreter, codegen, and stdlib for ~1.5x faster 0-D tensor allocation
- **Spatial if/else safety** — codegen's `_emit_spatial_if_else` now guards snapshot/merge against `None` locals for variables declared only inside one branch (prevents runtime crash on asymmetric branches)
- **Overall 3.4x faster** than v0.6.0 across all 4 benchmark scenarios (compile off/on × cold/warm); **2.6x faster** than v0.8.0
- **Test suite expanded** — 61 test functions covering stdlib coverage, numeric edge cases, array bounds, string edge cases, realistic tensor sizes, NaN/Inf propagation, codegen-interpreter equivalence, and arithmetic hash noise quality

### Fixed
- 3 missed `torch.scalar_tensor` conversions in interpreter builtin defaults (PI, E, scalar builtins)
- Codegen `_collect_all_env_vars` trivial wrapper inlined
- Codegen snapshot clone crashing on `None` locals in spatial if/else branches

## [0.8.0] - 2026-03-24

### Added
- **Structured diagnostics system** — new `tex_compiler/diagnostics.py` module with `TEXDiagnostic` dataclass carrying error code, source snippet, suggestions, and contextual hints
- **Multi-error reporting** — type checker and parser now report ALL errors at once via `TEXMultiError`, not just the first; parser uses panic-mode recovery (synchronizes on semicolons) to continue after syntax errors
- **"Did you mean?" suggestions** — unknown functions and undefined variables now suggest similar names using fuzzy matching (e.g., `clampp` → `clamp`)
- **Contextual hints for beginners** — 40+ foreign keyword/function/variable patterns detected with helpful hints (GLSL, HLSL, JavaScript, Python, Houdini VEX)
- **Error codes** — every error has a stable code (E1xxx lexer, E2xxx parser, E3xxx–E5xxx type checker, E6xxx runtime) for searchability and documentation
- **Source snippets** — errors include the offending line of code with the line number, rendered Rust-style in the error overlay
- **Structured JSON transport** — errors sent to frontend as `TEX_DIAG:` JSON payloads with full diagnostic metadata (suggestions, hints, error codes)
- **Empathetic error voice** — error messages rewritten in friendly, non-accusatory tone ("I can't find a function named 'clampp'" instead of "Unknown function: 'clampp'")
- **`ErrorNode` AST placeholder** — parser recovery inserts `ErrorNode` for failed statements; type checker silently skips them, preventing cascade errors
- **Snippet system** — right-click context menu with cascade submenus for browsing and inserting 114 built-in example snippets organized by category (Color, Compositing, Effects, Filter, Generate, Mask, Distortion, Latent, String, Video, Educational)
- **User snippets** — save selections as named snippets with `/` folder paths (stored in localStorage); manage dialog for renaming and deleting
- **Backend snippet API** (`/tex_wrangle/snippets`) — serves example `.tex` files from the `examples/` directory at runtime; eliminates ~650 lines of duplicated client-side template literals

### Changed
- **Error overlay** — now renders structured diagnostics with error codes, source snippets, suggestions (amber), and hints; supports multiple errors stacked
- **CM6 lint bridge** (`tex_lint.mjs`) — parses `TEX_DIAG:` JSON for multi-diagnostic CM6 integration with per-diagnostic severity; falls back to legacy regex for backward compat
- Error classes (`LexerError`, `ParseError`, `TypeCheckError`, `InterpreterError`) now carry a `.diagnostic` attribute with full `TEXDiagnostic` metadata
- Source text threaded through entire pipeline (Lexer → Parser → TypeChecker → Interpreter) for snippet rendering
- **Context menu** — reordered to Cut, Copy, Paste, Select All, separator, TEX Help (renamed from "TEX Reference"), Snippets
- **Cascade submenu hover** — shared per-level timeout prevents submenus from getting stuck when moving the mouse quickly between categories

## [0.7.0] - 2026-03-22

### Changed
- **Nodes v3 API migration** — `TEXWrangleNode` now inherits from `IO.ComfyNode` (via `comfy_api.latest`); `define_schema()` replaces `INPUT_TYPES` / `RETURN_TYPES` / `CATEGORY` / `FUNCTION` / `DESCRIPTION` / `OUTPUT_TOOLTIPS`; `fingerprint_inputs()` replaces `IS_CHANGED()`; `execute()` is now a classmethod returning `IO.NodeOutput`
- **Wireable parameters** — `$param` widgets now support drag-to-wire connections via ComfyUI's widget-input duality (`input.widget = { name }` linking), eliminating the v1 `convertToInput` workaround
- **`accept_all_inputs=True`** replaces `ContainsAnyDict` for dynamic input passthrough
- **Test helper cleanup** — `compile_and_run()` and `compile_and_infer()` simplified to single-pass type checking using `_infer_binding_type()`; removed redundant double type-check

### Removed
- `ANY_TYPE = "*"` module constant (replaced by `IO.AnyType`)
- `ContainsAnyDict` class (replaced by `accept_all_inputs=True`)
- All v1 class attributes: `CATEGORY`, `FUNCTION`, `RETURN_TYPES`, `RETURN_NAMES`, `OUTPUT_TOOLTIPS`, `DESCRIPTION`, `INPUT_TYPES()`, `IS_CHANGED()`
- `output_type` backward-compat code (`_SYSTEM_KWARGS` entry and `kwargs.pop`)
- `_infer_out` dead field from type checker
- Dead latent-dict check in `_resolve_device()` (latents already unwrapped before device resolution)

### Fixed
- **`loadCM6()` runtime error** — autocomplete toggle setting called non-existent `loadCM6()` function; replaced with synchronous `getCM6()` used everywhere else
- Stale cache docstrings (claimed 3-tuple return, actually 6-tuple)

## [0.6.0] - 2026-03-21

### Added
- **Nodes 2.0 compatibility** — TEX Wrangle now renders correctly in both legacy and Nodes 2.0 (Vue) rendering modes; original code textarea is spliced from the widget array (prevents Vue ComponentWidget rendering) and cleaned up via polling (prevents legacy DOM lingering)
- **Universal search panel visibility** — TEX Wrangle now appears when dragging a wire of any type, via a wildcard (`*`) input registration (same mechanism as PreviewAny)
- **DEVELOPMENT.md** — architecture, compilation pipeline internals, and how-to guides for adding functions/types/operators split out from README

### Changed
- **README.md rewrite** — hero image, badges, feature table, collapsible troubleshooting, categorized examples; reduced from 1008 to 244 lines for a cleaner GitHub landing page
- GitHub Actions workflow now declares `permissions: contents: read` (resolves CodeQL security alert)

## [0.5.0] - 2026-03-20

### Added
- **AST optimizer** — constant folding and algebraic simplification passes run before interpretation, reducing runtime work
- **Codegen backend** — compiles TEX AST to Python functions for zero-dispatch-overhead execution (automatic fallback to tree-walking for unsupported patterns)
- **Benchmark suite** (`benchmarks/run_benchmarks.py`) — reproducible cross-system performance measurement with adaptive run counts, multi-resolution sweeps, and JSON comparison workflow
- **7 new example programs:** `bilateral_approx`, `color_grade`, `lens_distortion`, `levels`, `normal_map`, `tone_map`, `unsharp_mask`
- Interpreter support for `precision` parameter (`fp32`/`fp16`/`bf16`) and `used_builtins` pre-scan for faster stdlib dispatch
- `inference_mode()` wrapper for pure tensor programs (no gradient tracking overhead)

### Changed
- **Interpreter rewrite** — 1.4x–2.0x faster across all benchmarks vs v0.4.0 (geometric mean):
  - Statement/expression dispatch via type-keyed tables instead of if/elif chains
  - Literal tensor cache eliminates repeated constant allocation
  - Branching rewired to fused `torch.where` fast-path (branch_nested: **34x** faster at 1024px)
  - Loop bodies pre-compiled, break/continue via lightweight exceptions
  - Spatial context (`u`, `v`, `ix`, `iy`) lazily allocated only when referenced
- **stdlib improvements** — sampler cache for batch indices and Lanczos taps; tighter safety epsilon handling
- **AST nodes** now use `__slots__` throughout for lower memory and faster attribute access
- **Cache version** bumped to `2.4.0` (existing disk caches auto-invalidate)

### Fixed
- **Parameter widget interaction** — promoted `$param` widgets now register in `nodeData.input.optional` so ComfyUI's Vue overlay enables click/drag/arrow interaction
- **DOM overlay blocking** — editor container and hidden textarea marked `pointer-events: none` so canvas-rendered widgets (params, compile_mode, device) remain interactive
- INT widget step corrected to 10 (was 1, causing 0.1 increments); FLOAT step corrected to 0.1 (was 0.01)
- Default font size changed to 10px; settings slider range narrowed to 4–20

## [0.4.0] - 2026-03-19

### Added
- **`while` loops** — `while (condition) { body }` with break/continue support, 1024 iteration cap
- **`else if` chains** — `if (...) {} else if (...) {} else {}` (already parsed, now documented and tested)
- **15 new string functions:**
  - `split(s, delim, max?)` — split string into array
  - `lstrip(s)`, `rstrip(s)` — directional whitespace trimming
  - `pad_left(s, width, char?)`, `pad_right(s, width, char?)` — padding with optional fill character
  - `format(template, ...args)` — `{}` placeholder interpolation (up to 15 args)
  - `repeat(s, count)` — repeat string N times
  - `str_reverse(s)` — reverse a string
  - `count(s, sub)` — count non-overlapping substring occurrences
  - `matches(s, pattern)` — full regex match (returns 1.0/0.0)
  - `hash(s)` — deterministic SHA-256 hex prefix (16 chars)
  - `hash_float(s)` — deterministic hash to float in [0, 1)
  - `hash_int(s, max?)` — deterministic hash to non-negative integer
  - `char_at(s, i)` — character at index (empty string if out of bounds)
- **`replace()` now accepts optional 4th argument** `max_count` to limit replacements
- Dynamic-size string arrays from `split()` — `string arr[] = split(s, ",");`

### Changed
- break/continue now work in both `for` and `while` loops
- `format()` rounds float32 values to 6 significant digits to avoid precision noise

## [0.3.1] - 2026-03-19

### Added
- GitHub Actions CI pipeline (Python 3.10/3.11/3.12, CPU-only torch)
- This changelog

### Changed
- Math functions (`sqrt`, `log`, `log2`, `log10`, `asin`, `acos`, `mod`) now clamp inputs to valid domains instead of producing NaN/Inf
- Standardized safety epsilon to `SAFE_EPSILON = 1e-8` across stdlib and interpreter

### Fixed
- `sdiv` eager evaluation bug — `torch.where` evaluates both branches, causing warnings on zero divisors
- `spow(0, negative)` producing NaN instead of 0
- `IS_CHANGED` not detecting tensor/list content changes (only hashed metadata like shape/dtype)
- `torch.compile` dropping multi-output results (`output_names` not threaded through compiled path)

## [0.3.0] - 2026-03-10

### Added
- Multi-output programs (`@mask`, `@result`, etc.)
- Parameter system with `param_*` inputs
- Auto-inferred output types
- ComfyUI settings panel
- `spow` (signed power) and `sdiv` (safe division) functions

### Fixed
- Parameter type prefix in expression context

## [0.2.0] - 2026-03-09

### Added
- CodeMirror 6 editor with syntax highlighting
- Monaspace Neon font
- Autocomplete for functions, bindings, and swizzles
- `mat3`/`mat4` matrix types with `transpose`, `determinant`, `inverse`
- 14 CTL/ACES math functions (`log2`, `log10`, `atan2`, `mod`, etc.)
- Performance optimizations

## [0.1.0] - 2026-03-06

### Added
- Initial release
- TEX DSL: lexer, parser, type checker, tree-walking interpreter
- 60+ built-in functions (math, color, noise, sampling)
- Optional `torch.compile` acceleration with backend cascade
- Disk-backed compilation cache
- 355+ test suite

### Fixed
- `KeyError` on dynamic inputs with newer ComfyUI versions
