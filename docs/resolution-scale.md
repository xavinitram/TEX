# Resolution scale (SCALE-47b implementation note)

*What `scale=` does, what it covers, and what it deliberately does not. Written for an
embedding host deciding whether and how to adopt it. Not user-facing (a TEX author never
writes `scale` in source) — this is engine-integration documentation, in the `docs/`
internal-design layer (DOC-6).*

## What `scale` is

A per-cook resolution multiplier a host passes to `tex_engine.prepare()`/`cook()` (and to the
CACHE-6/7 stage-list family: `cook_stage_list`, `cook_checkpointed`, `materialize`,
`boundary_lineage_key`). It multiplies every pixel-unit argument of a tagged stdlib builtin
(`gauss_blur`'s sigma, `erode`/`dilate`'s radius, `bilateral_filter`'s spatial_sigma) and the
halo margin those builtins derive, so a program cooked on a smaller canvas produces
proportionally smaller blur/morphology kernels instead of over-blurring relative to a
full-resolution cook of the same program.

**`scale=None` (the default — no host has to opt in) is byte-identical and cost-free**: no
runtime multiply is emitted or evaluated, tier selection is unaffected, and every
lineage/checkpoint key is exactly as it was before this feature existed. `scale=1.0` is the
degenerate, EXACT case — mathematically a no-op multiply, forced onto the interpreter tier
(see "What tier a scale-active cook runs on" below) but bit-identical in VALUE to a plain
cook of the same program on the same canvas.

**The engine never picks a scale, and never picks a proxy.** Both are the host's own
decision (its own timing, its own held proxy media/mip level); the engine only makes the
result of that decision — a cook on a smaller canvas, with proportionally smaller
pixel-unit magnitudes — correct.

## The API

- `tex_engine.prepare(code, bindings, ..., scale=0.5)` / `tex_engine.cook(..., scale=0.5)`.
- `tex_engine.cook_stage_list(stages, ..., scale=0.5)`,
  `tex_checkpoint.cook_checkpointed(stages, cache, ..., scale=0.5)`,
  `tex_checkpoint.materialize(stages, cache, ..., scale=0.5)`.
- `tex_engine.boundary_lineage_key(...)` / `tex_results.lineage_key(...)` accept `scale=` too,
  so a coarse-scale result or checkpoint boundary never collides with a full-scale one — a
  cache that never mentions scale keys exactly as it did before this feature (`scale=None`
  keys as `"n"`, the same shape `quality`/`frame` already use).
- `tex_api.scale_verdict(source, param_values=None) -> ScaleVerdict(safe, code, source)` — a
  cheap, memoized, PRE-COOK query: would a non-trivial scale be refused for this program?
  Cheap enough to call on every drag tick after a program's first lookup, and reads the exact
  same memoized answer `prepare()`'s own refusal does, so the two can never disagree.
- `tex_api.check_proxy_scale(bindings, full_hw, scale)` — an OFFERED, never-enforced sanity
  check: does a bound proxy image's shape actually agree with the `scale` claimed for it? The
  engine never calls this itself; a mismatched proxy still cooks (whatever that produces). A
  host that wants the belt-and-braces call opts in explicitly.

## What is covered

Only the stdlib functions carrying the registry's `pixel_args=` tag:

| Builtin | Pixel-unit argument | Registered footprint |
|---|---|---|
| `gauss_blur(img, sigma)` | `sigma` (arg 1) | `('halo_arg', 1, 3.0)` |
| `erode(img, radius)` | `radius` (arg 1) | `('halo_arg', 1)` |
| `dilate(img, radius)` | `radius` (arg 1) | `('halo_arg', 1)` |
| `bilateral_filter(img, spatial_sigma, range_sigma)` | `spatial_sigma` (arg 1) ONLY | `('halo', 3)` |

`bilateral_filter`'s `range_sigma` (a colour-similarity threshold, not a pixel distance) is
deliberately NOT scaled. The halo margin `stage_halo`/`roi_plan`/`chain_windows` derive from
these builtins' resolved arguments scales the same way, ceiling up (never under-pads).

`u`/`v`/`sample*`/`fetch*` and `px`/`py`/`ix`/`iy`/`img_width`/`img_height` already track
whatever canvas the host's bindings actually are — a program built purely from those already
renders "the same picture, downscaled" for free, with no engine change at all. `scale=` exists
only to close the narrower gap: an argument whose contract is "a distance in pixels" that does
not otherwise shrink with the canvas.

## What is NOT covered

- **`sdf_*`/`fbm`/`worley_*`** take generic float coordinates with no fixed unit contract — the
  same call is legitimately written in normalized `u,v` space (already resolution-independent)
  or pixel space, and the registry cannot tell which from the call site. These are never
  auto-scaled.
- **Hand-written pixel arithmetic** (`fetch(@A, ix + 5, iy)`, `img_width`/`img_height` used in
  ordinary expressions) is the author's own, and is not scaled.
- **The classifier declines these on sight** (see below) rather than silently mis-scaling
  them: a cook that asks for a non-1.0, non-None scale on a program the classifier cannot
  prove safe REFUSES with a structured, stable reason code (`EngineRefusal(code=
  "scale-unsafe")`), never silently substituting full scale (the engine never picks a scale
  for the host) and never silently cooking a wrong picture.
- **ROI narrowing and `scale` are not yet reconciled.** A cook that passes both `roi=` and a
  non-None `scale` declines the ROI window (cooks whole-frame at the requested scale) rather
  than risk the two interacting incorrectly.
- **Only the interpreter tier honours `scale`.** A scale-active cook (`scale` not `None`) is
  routed to the plain interpreter unconditionally, ahead of tier selection — bypassing
  `torch_compile`/`auto`/`cuda_graph` and the default tier's own internal codegen/stencil/
  tiling shortcuts. None of those tiers currently thread a runtime scale multiplier through
  their emitted or captured code, so extending each one individually was deferred rather than
  risk a silently wrong replay (a CUDA-graph capture taken at one scale must never replay under
  another). This costs real acceleration for a scale-active cook today; it does not cost
  correctness, and it does not touch the default (`scale=None`) path's tier selection at all.
  Reported via `tier_trace` exactly like every other tier decline (`tier_trace.last().tier ==
  "interpreter"`, reason names scale) — never a silent fallback.

## The classifier and the override comment

`tex_roi.scale_safe(code)` (memoized as `tex_roi.scale_verdict(code)`, mirrored publicly as
`tex_api.scale_verdict(source)`) conservatively declares a program unsafe when it reads
`ix`/`iy`/`img_width`/`img_height` anywhere OTHER than the whitelisted `fetch`/`fetch_frame`
coordinate arguments or the `@A[x,y]`/`@A(u,v)` sugar's own coordinate position — those already
read the real, correctly-scaled canvas by construction. The walk is over-approximating by
design: a program it cannot prove safe is declared unsafe, never the reverse, so it is wrong
only in the safe direction (a missed optimisation, never a wrong picture). Any analysis
failure (a program the walk cannot parse or walk) also declares unsafe.

An author who knows better can override the verdict in either direction with a leading
comment, parsed the same never-becomes-a-token way as the existing `//!tex X.Y` language
pragma, but NOT the identical header-scan: a leading `/* ... */` block comment (a common
file-header style) is skipped over here rather than ending the scan — only real code does,
unlike the language pragma, which stops at a block comment on purpose (FIX-SCALE S5).

```
//!tex scale: safe
```
forces a program the classifier declines to be treated as scale-safe (the author vouching for
a case the walk cannot model — an `sdf_*` call it can prove is fed from `u,v` by a path the
walk doesn't follow, for instance).

```
//!tex scale: never
```
forces a program the classifier would otherwise accept to always refuse a non-trivial scale
request. When a program's header declares BOTH directions, `never` wins regardless of which
line comes first — the conservative, fail-closed direction takes precedence over the
optimistic one, never a silent "whichever the scan saw first."

## The R1 promise: a measured envelope, not equality

Bit-exactness is not on offer even for a plain resize: a Gaussian kernel discretizes
`ceil(3·sigma·scale)` to a different integer radius per resolution, and morphology's
structuring element is likewise an integer radius. The promise mirrors invariant #9's
cross-device envelope: a `scale=s` cook, upsampled, compared against a `scale=None` cook of
the SAME program, downsampled to the same size — a maxdiff BAND, pinned per builtin family
(`tests/test_scale47b_r1_envelope.py`), not a claim of equality. A regression past a pinned
band is a loud decision to re-measure and re-band, never a silently tightened or loosened
tolerance. `scale=1.0` on the SAME canvas remains the exact, non-approximate case (a bit-exact
identity multiply).

## Precision under scale

A coarse cook (`scale` neither `None` nor `1.0`) whose caller left `precision` at its literal
default promotes to `"auto"` — the existing invariant #10 accuracy net, unchanged, already
reasons about data amplification independent of canvas resolution, so "reduced precision under
scale's envelope, never surfaced as a new decision" needs no new mechanism. An explicit
`precision=` on a coarse cook is still honoured unchanged.
