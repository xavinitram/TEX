# ROI / spatial laziness (v0.24 design note)

*The v0.24.0 "See less, cook less" release. In the xpu-transfer-scheduling.md /
lat1b mold: the shipped state, the analysis model, the execution model and the one
crux that makes it correct, the reach trap the roadmap flagged, and the oracle gate
that lets the flag flip. Covers ROI-2 (analysis), ROI-3 (execution, flagged off),
ROI-4 (the differential oracle ship-gate), and ROI-6 (temporal groundwork).*

The single biggest interactivity lever a compositor has is to **cook only the pixels
the viewer asks for**. TEX is unusually close: the M-4 `tile=(y0, H_total)` machinery
already proves seam-exact coordinates and zero-copy view narrowing along the height
axis. ROI generalizes that from a 1-D strip to a 2-D (and, ROI-6, a temporal) window.

---

## Shipped state before v0.24

- `tex_lazy.py` — *input*-level laziness: which `@/$` bindings a program can reference,
  `$param`-folded (the analysis ROI-2 mirrors on the spatial axis).
- `tex_memory.run_tiled` + `_tile_plan` — height-strip execution for **pointwise**
  (`is_tile_safe`) programs only. `is_tile_safe` refuses *any* non-`point` footprint,
  so a `gauss_blur` cannot tile at all.
- ROI-1 (v0.23) — the per-function access **footprint** in the stdlib registry
  (`'point' | 'image' | ('halo', r) | ('halo_arg', i) | ('frame', i)`), from which
  `tex_memory._NON_LOCAL_FNS` is derived. Pure metadata, zero behaviour change — the
  substrate this release stands on.
- The interpreter's seam-exact coordinate builtins already accept `tile=(y0, H_total)`
  (`interpreter._create_builtins`): under a strip, `iy` starts at `y0`, `v`/`ih`/`py`
  use `H_total`, so a strip's coordinates match the untiled cook exactly.

Egress (`tex_marshalling`, the node) is fully H/W-agnostic — nothing between
`interpreter.execute()` and egress assumes a full-frame shape. The *only* place that
materializes full-frame dimensions is `run_tiled`'s output preallocation +
`buf[:, y0:y1] = strip_out`.

---

## ROI-2 — the footprint analysis (`tex_roi.py`)

The spatial sibling of `tex_lazy`. For a program + folded params, compute a **per-binding
footprint** on the lattice

```
point  ⊑  halo(up, down, left, right)  ⊑  image
```

- `point` — the output pixel reads only the same input pixel (pure pointwise / elementwise).
- `halo(u,d,l,r)` — the output pixel reads a bounded neighbourhood; the four extents are
  non-negative pixel counts. `point` is `halo(0,0,0,0)`.
- `image` — whole-image or data-dependent gather (unbounded); the top of the lattice.

**Composition (over-approximating, like tex_lazy):**

1. `$param` folding (reused from tex_lazy: `_substitute_params` + `_fold_all`) resolves
   `gauss_blur(@A, $sigma)` radii to literals when the widget value is known. A radius
   that stays symbolic (a wired param) → `image` (conservative).
2. ROI-1's registry footprint + the **reach model** (below) turns a call's descriptor into
   a pixel halo on its image argument.
3. Affine offset extraction (reused from `codegen_stencil`: `_extract_pixel_offset` /
   `_extract_uv_offset`) turns a constant-offset `fetch(@A, ix+3, iy)` / hand-written
   stencil loop into `halo(0,0,0,3)` — the analysis is complete here as ROI-5/GRAPH-1
   substrate, even though ROI-3 v1 does not *execute* the narrowed gather (see the crux).
4. Reads under an `if` / `where` **union** both branches (both may evaluate — the same
   discipline invariant #11 codifies for the lazy analysis; `where` evaluates both sides
   on tensors). A binding's aggregate footprint is the least-upper-bound over all its read
   sites.
5. **Anything unresolved → `image`.** Never raises; a too-large footprint is a missed
   optimisation, never a wrong pixel.

### The reach model (the trap the roadmap flagged)

ROI-1's `('halo_arg', i)` names *which argument* carries the radius — **not** a pixel
reach, and the argument is not always in pixels:

| fn | descriptor | arg | true pixel reach |
|----|-----------|-----|------------------|
| `erode` / `dilate` | `('halo_arg', 1)` | `radius` | `ceil(radius)` (window `(2r+1)²`) |
| `bilateral_filter` | `('halo', 3)` | — | fixed `3` (`min(ceil(3σ_s), 3)` in impl) |
| `gauss_blur` | `('halo_arg', 1, 3.0)` | `sigma` | **`ceil(3·sigma)`** |

`gauss_blur(@A, 2.0)` reads **±6 px**, not ±2. A reach resolver that treats the
`halo_arg` value as the pixel reach under-pads by 3× and produces silent wrong pixels at
every ROI seam — the worst failure class in the codebase. The fix is a per-function
**multiplier** carried in the descriptor itself (single source, invariant #5): the
`('halo_arg', i)` form gains an optional third element `m` (default `1.0`), and the reach
is `ceil(m · |arg_i|)`. `gauss_blur` becomes `('halo_arg', 1, 3.0)`. ROI-4's reach-pinning
test cooks a one-pixel impulse through each halo fn and asserts the *measured* pixel
spread ≤ the descriptor's declared reach, so a wrong multiplier fails loudly at CI.

---

## ROI-3 — ROI execution (flagged off), the crux

Generalize `tile=(y0, H_total)` to `roi=(x0, y0, w, h, W, H)` in the interpreter: cook an
output window `[x0, x0+w) × [y0, y0+h)` of a full `W×H` image, with seam-exact 2-D
coordinate builtins (`ix`/`iy` offset by `x0`/`y0`; `u`/`v`/`iw`/`ih`/`px`/`py` against
the full `W`/`H`). `tile=(y0, H_total)` is exactly the special case
`roi=(0, y0, W, h, W, H_total)`, so the existing strip path is preserved unchanged.

### The one model that composes cleanly

A naïve "narrow each input to output-ROI ⊕ its own footprint" runs aground: a single cook
has **one** output grid, and different bindings would want different narrowed sizes;
worse, `sample`/`fetch` address their input tensor through `u,v`/`ix,iy` that are
**full-image** coordinates, so a narrowed tensor sampled with full-image coordinates reads
the wrong absolute position. That is exactly why `sample`/`fetch` are `image` footprint.

The model that *is* correct (and, for the pointwise/morphology core, bit-exact):

> **Cook region = ROI ⊕ H** where `H` is the program's single **maximum** halo reach
> (the max over all bindings' halo extents), clamped to the image. Narrow **every**
> spatial input uniformly to the cook region, cook at the cook-region grid with seam-exact
> coordinates, then **crop** each output to the ROI sub-window.

Everything in the cook is one uniform `(ROI ⊕ H)` grid, so pointwise reads, the
direct-tensor halo ops, and the coordinate builtins all align by construction. Why it is
correct:

- **Pointwise** (`H = 0`): identical to `run_tiled`, in 2-D — **bit-exact**. The narrowed
  input view and the coordinate grid share the ROI shape.
- **A direct-tensor halo op** (`gauss_blur`/`erode`/`dilate`/`bilateral_filter`) operates
  on the tensor it is handed and returns the same shape. On the `(ROI ⊕ H)`-narrowed
  input, every ROI-interior output pixel is `H` pixels from the narrowed edge, so its
  convolution/window reads the *same real neighbours* as the full-image cook — a conv's
  interior output is independent of tensor extent. Where the cook region is clamped at a
  true image edge, the op's replicate-pad at the clamped (= true) edge equals the full
  cook's pad. Nested blurs compose additively (`H = Σ radii`); the contaminated outer
  margin is always outside the ROI and discarded by the crop.
  - **Integer morphology** (`erode`/`dilate`, min/max pool) is **bit-exact** — no fp
    accumulation. **Conv/bilateral** match to **~1 ulp** (measured peak `1.8e-7` at a
    clamped corner): `torch`'s reductions dispatch different SIMD blocking for a `6×7`
    cook region than a `32×24` frame, so `exp`/weighted-sum/division can differ in the
    last bit — far below the 8-bit quantum (`3.9e-3`), invisible in a viewport. The ROI-4
    oracle therefore asserts `maxdiff < 1e-5` (the FUS-3 convention), not `torch.equal`.

### Executability gate (whitelist posture)

ROI execution engages only when the program's non-pointwise work is confined to the
**direct-tensor halo whitelist** `{gauss_blur, erode, dilate, bilateral_filter}` (ops that
transform the tensor in place of a per-pixel gather). The gate blocks (→ whole-frame) on the
**presence** of any of these, checked in the walk rather than by attributing a read to a
binding — the whitelist posture, unknown → whole-image, never a shrunk ROI:

- **Any gather / reduction** — `sample`/`fetch`/`sample_*`/`img_*` or a
  `BindingIndexAccess`/`BindingSampleAccess`. Blocking on presence (not on resolving the
  image to a wire binding) is load-bearing: a gather over a **local-variable alias**
  (`vec4 x = @A; sample(x, …)`), a **user-function parameter**, or a **bindless generated
  image** (`sample(vec4(u,v,…), …)`) would otherwise escape a binding-attribution check and
  silently ROI-shrink.
- **A symbolic-radius halo** (a wired blur radius that doesn't fold) — the halo is unknown.
- **An *ungrounded* halo** — a blur/morphology op whose reach can't be composed by the
  single-expression-tree walk. Two shapes: (a) a halo **inside** a VarDecl initializer, a
  function body, or a loop body; and (b) a halo **result that flows through a NAME** — a
  local variable or intermediate `@binding` assigned a halo-containing value (via a VarDecl,
  a bare/reassigning `Assignment`, OR an `@T = …` output) that is then read elsewhere. The
  double blur `b = gauss_blur(@A,2); @OUT = gauss_blur(b,2)` reads `@A` ±12, not the ±6 the
  walk infers across the name boundary, so cooking on `ROI ⊕ 6` would contaminate the
  ROI-edge band. Reading a name that carries a mere INPUT (no halo) is fine
  (`vec4 x = @A; gauss_blur(x,2)` and multi-output `@OUT = gauss_blur(@A,2); @MASK = @A*0.5`
  stay executable) — only a name carrying a halo *result* blocks. v1 has no local-variable
  dataflow model (`_propagate_literal_locals` inlines only literals), so these cook
  whole-frame. Precise composition (inline non-literal locals before analysis) is ROI-5.
- **A scatter** (`@OUT[x,y] = …`, including a channel/array-wrapped target `@OUT[x,y].r`,
  `@OUT[x,y].rgb`, `@OUT[x,y][0]`) — an absolute write can't land in a sub-region buffer.

For a program clear of all of these, every binding footprint is `point` or a bounded
grounded `halo`, `H` is finite, and the narrow-cook-crop is correct (bit-exact / ~1 ulp as
above).

Why v1 excludes gathers entirely (not just narrows them): `fn_sample`/`fn_fetch` size
their output from the **input image** shape (`grid = _get_grid_buf(B, H, W)` off
`img.shape`), so a whole-passed gather driven by an ROI-sized coordinate grid produces a
*whole*-sized output — a shape mismatch, not an ROI. Passing gather inputs whole while the
output stays the ROI would need the gather output grid **decoupled** from its input, a
broad change to every `sample*`/`fetch*` impl — that is ROI-5. So affine `fetch`/`sample`
at a constant offset are *analysed* by ROI-2 (a bounded, non-narrowable halo — substrate)
but the program falls to whole-frame. Documented scope, not a silent gap; the dominant
compositing ops (grade, blur, vignette, mask shrink/grow) are pointwise + blur/morphology
and *are* ROI-executable.

### Flagged off

The `roi=` parameter threads through `interpreter.execute` / `_execute_inner` /
`_create_builtins` (a sibling of `tile=`) and the engine's `cook`/`prepare`. It is
**off by default**: no production caller passes `roi=`, and the engine-level auto-narrow
path is gated behind `tex_roi.roi_exec_enabled()` (env `TEX_ROI_EXEC`, default off), the
same "interpreter-only feature, clamp elsewhere" pattern `prepare()` uses to force
compiled modes to fp32. The oracle lane (ROI-4) exercises the mechanism directly; the flag
flip to a production viewport is a later release, gated on that lane being green.

Compiled tiers (`torch_compile` / `auto` / `cuda_graph`) do not thread `tile`/`roi` and
have no ROI seam today: a cook that would ROI-narrow runs on the interpreter tier, exactly
as tiling already does.

---

## ROI-4 — the differential oracle (the ship gate)

ROI ships nothing until this lane is green. Three parts, mirroring the repo's proven
shapes (§10.4):

1. **Whitelist posture** — the executability gate above; unknown → whole-image. Pinned by
   *spatial never-sever rows* mirroring `test_lazy_cooking`: a program that reads a binding
   through an `image`-footprint call must fall back (not ROI-shrink), asserted case by case.
2. **The reach-pinning derivation test** — cook a one-pixel impulse through each halo fn at
   a known radius and assert the measured non-zero spread ≤ the descriptor's declared reach
   (`ceil(m·arg)`), so a wrong multiplier (the `gauss_blur` 3× trap) reds at CI. Extends
   `test_v023_phase1`'s ROI-1 footprint family.
3. **The differential ROI oracle** — a seeded generator (TST-1's precedent) emits random
   ROI-executable programs (pointwise arithmetic + blur/morphology); for each, over a set of
   random ROIs tiling the frame, assert the ROI-assembled full frame (cook each ROI, place
   into a buffer) matches the whole-frame interpreter cook to `maxdiff < 1e-5` (the FUS-3
   convention — pointwise/morphology land bit-exact, conv/bilateral within ~1 ulp of
   size-dependent kernel dispatch). CPU-pinned for determinism; CUDA looped when present.

---

## ROI-6 — temporal laziness groundwork

The batch-axis twin of ROI-3. `tex_roi.frame_window` computes the program's temporal
footprint — the `(min, max)` frame offset relative to `fi` — from ROI-1's `('frame', i)`
footprints (`fetch_frame`/`sample_frame`) **and** the 3-arg cross-frame sugar
`@A[ix,iy,frame]` / `@A(u,v,frame)` (which parse to `BindingIndex`/`SampleAccess` with the
frame as the **last** arg, not `args[1]` like the direct-call footprint). Interpreter
batch-slice execution `batch_slice=(f0, B_total)` narrows dim 0 exactly as `tile` narrows
dim 1: `fi` starts at `f0` and `fn` reports `B_total`, seam-exact, so a per-frame program's
frame builtins match the whole-batch cook. `tex_memory.run_batch_strips` narrows dim 0 (so a
LATENT batch-strips safely) and stitches — the near-term payoff is batch-strip memory tiling
for video cooks.

**`batch_sliceable` is True only when the program has NO frame op at all.** A
`fetch_frame`/`sample_frame` (or the 3-arg sugar) is an *absolute* frame-index gather into
the batch: under a strip the global `fi` indexes a strip-local tensor and its impl clamps
against the strip's `B`, not `B_total`, so it reads the wrong (frozen) frame at **every**
offset — including offset 0 (`fetch_frame(@A, fi, …)`, which reads the current frame at
full-batch but a clamped edge frame under a strip). So any frame op → whole-batch in v1, the
temporal analog of a spatial gather, deferred with the same absolute-index limitation. Only
pointwise-over-batch programs (which may still do per-frame spatial gathers/blurs, since
those don't cross frames) batch-slice. `frame_window` remains the substrate for the
ROI-5-era executor that would plumb `f0`/`B_total` into `fetch_frame`/`sample_frame`.

---

## The gate that would change the verdict

ROI-3's flag flips to a production path (a viewport, a slap-comp host) when: (a) ROI-4's
differential fuzz lane is green across a nightly run, and (b) a real host consumes
`roi=` — i.e., PORT-5's demo or the standalone viewport exists to *ask* for a sub-region.
Until a consumer asks, whole-frame is the honest default and ROI is measured, tested, and
dormant. The ROI-5 reopen items, named here so they are not re-derived:

- **Correctness/coverage extensions**: compiled-tier ROI (ROI-bucketed graph/compile keys so
  a scrubbed ROI does not explode the cache); affine-gather narrowing (a decoupled gather
  output grid so `sample`/`fetch` can be narrowed instead of whole-passed); precise
  local-variable dataflow (inline non-literal locals before analysis, so a blur through a
  named value composes its reach instead of blocking); and plumbing `f0`/`B_total` into
  `fetch_frame`/`sample_frame` so cross-frame temporal reads can batch-slice.
- **Performance (only matters once a viewport actually pans)**: the builtins LRU keys on the
  full `roi`, so a *panning* viewport misses the LAT-4 cache every frame — bucket ROI rects,
  or cache full-frame coords once and take affine `.narrow()` views. A deterministic
  `run_roi` failure re-cooks (falls back) every frame — a fingerprint-keyed circuit breaker
  would skip the doomed narrow. `roi_plan`'s three AST walks (accumulate / ungrounded / scatter)
  could fuse into one pass. None of these touch the default (`roi=None`) cook path.
```
