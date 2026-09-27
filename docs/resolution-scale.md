# Resolution scale (SCALE-47b implementation note)

*What `scale=` does, what it covers, and what it deliberately does not. Written for an
embedding host deciding whether and how to adopt it. Not user-facing (a TEX author never
writes `scale` in source) — this is engine-integration documentation, in the `docs/`
internal-design layer (DOC-6).*

## What `scale` is

A per-cook resolution multiplier a host passes to `tex_engine.prepare()`/`cook()` (and to the
CACHE-6/7 stage-list family: `cook_stage_list`, `cook_fused_cached`, `cook_checkpointed`,
`materialize`, `boundary_lineage_key`). It multiplies every pixel-unit argument of a tagged stdlib builtin
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
  `tex_chain.cook_fused_cached(stages, k, cache, ..., scale=0.5)`,
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

## The declared-fallback query (TIERQ-48)

Both gaps above — "only the interpreter tier honours `scale`" and ROI's own `tier_id ==
"default"` requirement (`docs/roi-spatial-laziness.md`) — mean a host cannot learn, short
of timing a cook and noticing it was slow, that a `torch_compile`/`auto`/`cuda_graph`-
eligible program silently downgrades to the interpreter the moment `scale`/`roi` is
requested. `tex_api.tier_verdict` (delegating to `tex_engine_tiers.tier_verdict`) makes
that fact QUERYABLE instead of only documented in prose:

    from TEX_Wrangle import tex_api
    v = tex_api.tier_verdict(source, compile_mode="torch_compile", device="cuda:0",
                             roi=(x0, y0, w, h, full_w, full_h), roi_exec=True)
    # v.tier == "torch_compile"       (the coarse tier select_tier would pick)
    # v.roi_armed == False            (that tier never threads roi — whole-frame)
    # v.roi_reason == "roi-declined-tier-not-default"

It returns a `TierVerdict(tier, reason, roi_armed, roi_reason)`:

- `tier` is one of `"torch_compile"` / `"auto"` / `"cuda_graph"` / `"default"` /
  `"interpreter"` — the same five strings the real dispatch (`tex_engine_tiers._run_tier`)
  can actually produce — or `None` when the cook itself would REFUSE (a non-1.0 `scale`
  the classifier cannot prove safe): the query never guesses what an exception-raising
  cook "would have" run on.
- `roi_armed`/`roi_reason` answer the SEPARATE question of whether a requested `roi`
  window actually narrows the cook — `False` even on an eligible tier when that tier
  never threads ROI at all, or on the `"default"` tier itself for any of the ordinary
  reasons (`roi_exec` not armed, a malformed or whole-frame window, a non-ROI-executable
  program, a non-fp32 effective precision).
- Both `reason` and `roi_reason` are STABLE string constants
  (`tex_engine_tiers.TIER_REASON_*` / `ROI_REASON_*`) a host may branch on; they do not
  change shape across a release without a CHANGELOG entry.

It is side-effect-free (no compile, no cache write, no cook) and read-only over tier
selection: it calls the exact same `select_tier` plus the exact same `tex_roi` predicates
(`scale_safe`/`roi_exec_enabled`/`validate_roi`/`canonical_roi`/`roi_plan`) `tex_engine.
prepare()`'s own tier/ROI gates call, in the same order — so the query and a real cook's
plan can never disagree by construction. `precision` must be the cook's already-resolved
EFFECTIVE precision (`"fp32"`/`"fp16"`; `None` means `"fp32"`) — the query does not
resolve `precision="auto"` itself, because that resolution needs a real cook's bindings/
resolution to size the pixel-count gate; a caller predicting an `"auto"` cook resolves it
first, exactly as `prepare()` does before this same gate runs.

**Measured, not merely designed: whether codegen-ROI (`TEX_ROI_CODEGEN=1`) changes this
query's answer.** The query reports `roi_armed`/`tier` for the tier a cook actually runs
on; `TEX_ROI_CODEGEN` is an orthogonal, `"default"`-tier-internal routing choice (codegen
vs. the tree-walking interpreter for an already-armed ROI window) that does not change
which of the five `tier` values the query reports. See the codegen-ROI re-measurement
below for whether that internal choice is worth flipping.

## The classifier and the override comment

`tex_roi.scale_safe(code)` (memoized as `tex_roi.scale_verdict(code)`, mirrored publicly as
`tex_api.scale_verdict(source)`) conservatively declares a program unsafe when it reads
`ix`/`iy`/`img_width`/`img_height` anywhere OTHER than the whitelisted `fetch`/`fetch_frame`
coordinate arguments or the `@A[x,y]`/`@A(u,v)` sugar's own coordinate position — those already
read the real, correctly-scaled canvas by construction. The walk is over-approximating by
design: a program it cannot prove safe is declared unsafe, never the reverse, so it is wrong
only in the safe direction (a missed optimisation, never a wrong picture). Any analysis
failure (a program the walk cannot parse or walk) also declares unsafe.

**Known over-refusal, left as-is (FIX-SCALE S9):** the whitelist does not survive nesting —
`fetch(@A, int(ix), int(iy))` (a benign, canvas-relative whole-pixel fetch behind a
defensive type cast) is declared unsafe, because walking into the `int(...)` cast loses the
whitelist the outer `fetch` call granted. This can only ever LOSE the whitelist, never wrongly
grant one, so it is over-refusal only (a missed optimisation), never a wrong picture — but it
does needlessly decline a plausible idiom. Left conservative rather than threading the
whitelist through every intervening node; `//!tex scale: safe` is the documented workaround
for a program that hits this.

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

**The bands get worse at the coarser rungs, and the published numbers above (0.10 / 0.05 /
0.05 / 0.08) hold at `scale=0.5` only** — the Bible's own ladder is `{½, ¼, ⅛}`, and this is
the integer-radius/kernel discretization named above, not a canvas-size artifact (measured
near-identical at 256² and 512²; size does not move these numbers, only scale does). Metric:
maxdiff of a `scale=s` cook (upsampled) against a `scale=None` cook of the same program
(downsampled to the same size), on TWO inputs — an adversarial 8-pixel-period checker (worst
case for high-frequency content) and a realistic "smooth gradient plus a few hard-edged
rectangles" image (most of a real comp is smooth; only isolated boundaries are sharp) — the
band pins whichever measures WORSE, since neither input is reliably the worse one across
every family (erode's `scale=0.125` divergence is worse on the smooth+edges image, 0.43–0.48,
than on the checker, 0.25).

| family | scale=½ (pinned, `test_scale47b_r1_envelope_*`) | scale=¼ | scale=⅛ |
|---|---:|---:|---:|
| `gauss_blur` | 0.10 | 0.20 | 0.40 |
| `bilateral_filter` | 0.08 | 0.06 | 0.05 |
| `erode` | 0.05 | 0.30 | **not recommended** — measured 0.25–0.48 |
| `dilate` | 0.05 | 0.30 | **not recommended** — measured 0.53–0.75 |

**`erode`/`dilate` are not recommended below `scale=¼`.** At `scale=⅛` the measured maxdiff
(0.25–0.75, on a `[0,1]` channel range) is more than half the value range — past any band
that could be called "the same picture, downscaled." The mechanism: `erode`/`dilate`'s scaled
radius is truncated to an int downstream (`stdlib_sample._morph`), unlike `gauss_blur`'s
continuous sigma (whose own `ceil` — `stdlib_core.py`'s `radius = int(math.ceil(3.0 *
sigma))` — never reaches zero for a positive sigma); at `scale=0.125` a radius of `4.0`
truncates to a scaled radius of `0`, a total no-op. Investigated whether switching `_morph`
to `ceil` (matching `gauss_blur`'s own convention) helps: measured, it does NOT reliably — on
the checker pattern the maxdiff is IDENTICAL either way (a radius-1 pass and a radius-0
no-op are just two different discretizations of a true radius of 0.5, neither closer to it
in general), and on the smooth+edges image `ceil` is worse for one family and better for the
other. Left unchanged (`int()`, matching the pinned `scale=None`/`scale=1.0` default-path
behaviour exactly either way — invariant #7) rather than trade one silently-worse case for
another; a host choosing `¼`/`⅛` under memory or latency pressure should not select `⅛` for
these two builtins.

## Precision under scale

A coarse cook (`scale` neither `None` nor `1.0`) whose caller left `precision` at its literal
default promotes to `"auto"` — the existing invariant #10 accuracy net, unchanged, already
reasons about data amplification independent of canvas resolution, so "reduced precision under
scale's envelope, never surfaced as a new decision" needs no new mechanism. An explicit
`precision=` on a coarse cook is still honoured unchanged.
