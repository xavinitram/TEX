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
| `bilateral_filter(img, spatial_sigma, range_sigma)` | `spatial_sigma` (arg 1) ONLY | `('halo_arg', 1, 8.0)` |

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
- **`torch_compile`/`auto`/`cuda_graph` are scale-aware mechanisms with, as of this writing,
  NO real-builtin speedup to show for it.** A scale-active cook whose tier selection names
  one of those three is dispatched to that tier directly instead of being forced onto the
  plain interpreter, and `scale` genuinely reaches each tier's own execution path. But every
  one of today's four registered `pixel_args=` builtins (`gauss_blur`/`erode`/`dilate`/
  `bilateral_filter`) is ALSO registered `sync=True` in `graphed._SYNC_STDLIB` (its pixel-unit
  argument resolves via a capture-illegal `.item()`), so **`cuda_graph` never captures a real
  scale-active program calling one — `_capturable()` declines it before a `_capture_key` is
  ever computed, on any box, CPU or CUDA.** And none of the four inlines in codegen, so
  **`torch_compile`/`auto` never reach real Inductor tracing for one either — `_try_compile`
  returns the codegen-only eager adapter (backend unreached), not a compiled kernel.** Both
  declines are unconditional and box-independent, not a quirk of any particular hardware.
  `tier_verdict` (see below) reports the tier that ACTUALLY runs — `"codegen"` for the
  `torch_compile`/`auto` decline, `"interpreter"` for the `cuda_graph` decline — rather than
  the tier `select_tier` merely selected, so a host reading the query never expects a speedup
  that will not materialize. The mechanism is kept, not removed: it is exercised end-to-end
  by a synthetic, non-sync-gated builtin (`mix`, tagged as `pixel_args=` for the test's own
  scope only) proving the underlying dispatch and cache-key logic are correct, ready for the
  day a `pixel_args=` builtin ships without a capture-illegal sync or a codegen graph-break.
  Making one of the four builtins reach a real compiled/captured tier is future work, not
  shipped here.

  When a future builtin DOES reach one of these tiers: `execute_compiled`/`run_auto` share
  ONE compiled artifact across every scale value (`scale` is a runtime call argument, never
  a cache-key component — the callable rebuilds its codegen environment fresh every cook, so
  there is nothing stale to key against). `run_graphed`'s `_capture_key` is the one exception:
  a captured graph replays a fixed sequence of kernel launches against fixed buffer shapes
  recorded once at capture time, and a `pixel_args=` builtin's scaled radius is baked in as
  exactly such a shape — so `_capture_key` keeps an explicit, trailing `scale` component (a
  distinct scale value gets its own capture, reused on every repeat of that value, bounded by
  the same VRAM budget that already bounds every other distinct canvas shape). `scale=1.0`
  keys/buckets identically to `scale=None` everywhere (the documented byte-identical no-op).
  ROI stays out of scope on every tier regardless (the bullet above).

  **The `"default"` tier's own internal codegen shortcut honours `scale` — but only for
  the narrow class of program that shortcut already accelerates.** `_should_stencil_route`
  recognizes exactly one shape: a *hand-written*, nested-loop, exact-fetch stencil (the
  UC-2 pattern, `fetch(@A, ix + dx, iy + dy)` inside a fixed-radius loop) written directly
  in the TEX source. It has nothing to do with whether the program calls a
  `pixel_args=`-tagged builtin. A scale-active cook whose program is a *plain*
  `gauss_blur`/`erode`/`dilate`/`bilateral_filter` call with no coincidental hand-written
  stencil loop still runs on the plain interpreter — the single most common program shape
  the `pixel_args=` mechanism exists for is **not** sped up by this route either. Only a
  program that independently contains the UC-2 stencil shape gets routed to codegen, and
  that routing carries any `pixel_args=` call sites in the SAME program along for the ride
  (their scale multiplier is emitted as a runtime value read from the cook's own
  environment, never a folded literal, so one cached codegen fn serves every scale value
  without recompiling). M-4/ROI-5 *tiling* is still out of scope on either route — a
  scale-active cook always cooks whole-frame (see the ROI bullet above). Reported via
  `tier_trace` exactly like every other tier decision (`tier_trace.last().tier ==
  "interpreter"` or `== "codegen"`, reason names scale either way) — never a silent
  fallback. `tier_verdict`'s own `TIER_REASON_SCALE_ACTIVE_CODEGEN` reason code names this
  precisely: it is returned only when `_should_stencil_route` itself says yes for THIS
  program, never as a general "codegen now supports scale" signal.

## The declared-fallback query (TIERQ-48)

ROI's own `tier_id == "default"` requirement (`docs/roi-spatial-laziness.md`) means a host
cannot learn, short of timing a cook and noticing it was slow, that a `torch_compile`/
`auto`/`cuda_graph`-eligible program silently cooks whole-frame the moment `roi` is
requested. `scale` has the analogous gap, one level deeper: even where a tier is dispatched
directly (see above), the tier can itself self-decline past that point for a program the
`pixel_args=` mechanism actually targets. `tex_api.tier_verdict` (delegating to
`tex_engine_tiers.tier_verdict`) makes every one of these facts QUERYABLE instead of only
documented in prose — including the self-decline, by reusing the SAME predicates the real
dispatch checks (`graphed._capturable`, codegen's `_has_fn_calls`), never a parallel guess:

    from TEX_Wrangle import tex_api
    v = tex_api.tier_verdict(source, compile_mode="torch_compile", device="cuda:0",
                             roi=(x0, y0, w, h, full_w, full_h), roi_exec=True)
    # v.tier == "torch_compile"       (the coarse tier select_tier would pick)
    # v.roi_armed == False            (that tier never threads roi — whole-frame)
    # v.roi_reason == "roi-declined-tier-not-default"

It returns a `TierVerdict(tier, reason, roi_armed, roi_reason)`:

- `tier` is one of `"torch_compile"` / `"auto"` / `"cuda_graph"` / `"default"` /
  `"interpreter"` / `"codegen"` (SCALE-CG-48's UC-2 stencil route for a scale-active
  `"default"`-tier cook — precise only when `binding_types` lets the query compile
  `source`; see `tier_verdict`'s own docstring for the conservative fallback) — the same
  six strings the real dispatch (`tex_engine_tiers._run_tier`) can actually produce — or
  `None` when the cook itself would REFUSE (a non-1.0 `scale` the classifier cannot prove
  safe): the query never guesses what an exception-raising cook "would have" run on.
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
which `tier` value the query reports. Re-measured at realistic interactive-viewport shapes
(a small window against a 1920x1080 canvas, and against a 3840x2160/"4k" canvas) on a
current box: codegen is measurably SLOWER than the interpreter at both shapes (reproduced
across two independent, interleaved A/B sittings), so `TEX_ROI_CODEGEN` stays flagged OFF
by default — this internal choice is not worth flipping today.

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

## `gauss_blur` past the exact threshold (GAUSSPYR-50)

`gauss_blur(img, sigma)`'s signature and registry entry are unchanged — this is an ENGINE
POLICY, not a new argument, in the same shape `precision="auto"`'s own gate already is
(invariant #10): an internal, automatic, measured-safe decision the engine makes, never a
knob a TEX author writes.

**Below `GAUSS_BLUR_PYRAMID_THRESHOLD_SIGMA` (256.0), nothing changed: `gauss_blur` runs
today's exact separable convolution, unconditionally.** Proven with `torch.equal` (not a
tolerance) across a sigma sweep from 0 up to the threshold, CPU and CUDA
(`tests/test_gausspyr50_engine_policy.py`) — the same call, the same kernel, the same
padding and conv passes as every release before this policy existed.

**Above the threshold, an automatic downscale-pyramid approximation runs instead** ("Nuke
quality": reduce the image in ONE `interpolate(mode='area')` call to a size sized by a
downsample `factor` computed from `sigma`/`GAUSS_BLUR_PYRAMID_QUALITY_CAP` alone — never from
the image's own dimensions, fixed by FIX-APPROX A2 after B2's bug hunt found the original
per-level `avg_pool2d` cascade's stopping condition could revert to `O(sigma)` cost on a fixed
image size — blur EXACTLY at that residual sigma via the same exact code path used below the
threshold, upsample back bilinear). This exists because the exact convolution's cost grows
with sigma (kernel width is `2*ceil(3*sigma)+1`) without bound, to the point of being unusable
at the radii an "arbitrarily large blur" request implies — a single 4k call already costs over
a second past sigma≈256 on a measured box, and grows into the tens of seconds by the low
thousands. The pyramid path's cost is flat instead: `O(image size)` once, independent of
sigma, confirmed flat from sigma=1e4 to sigma=1e9 on a 1080p image after A2's fix (~2ms
throughout; the pre-A2 shape rose to 707.5ms at sigma=1e9).

**A windowed, tiled, or DAG-joined cook of a call past the threshold is served whole-frame,
never narrowed (FIX-APPROX A1).** The pyramid's resample grid is anchored to whatever crop it
is handed, not the frame's absolute coordinates, so a window that has not grown to the whole
frame would sample on a different phase than a whole-frame cook of the same program — a
genuine, silently-wrong divergence (measured up to ~8e-5 maxdiff), not a rounding footnote.
`gauss_blur`'s footprint declares this threshold to the ROI/tiling/`cook_stage_dag` planner
(`tex_roi._reach_of`'s `approx_above`), which declines to narrow past it — the identical
decline a symbolic (non-foldable) sigma already got. Below the threshold this is unaffected:
the window still narrows and still matches a whole-frame crop exactly.

**Both constants were picked by measurement** (a fuzzer sweep over an 8-pixel-period checker
and a smooth-gradient-plus-hard-edged-rectangles corpus — the same two-input protocol this
document's own R1 promise below uses — at 1080p and 4k, CPU): the threshold is set well above
any sigma that still runs in a bounded, tolerable time exactly, and `quality_cap=8.0` keeps
the worst-measured maxdiff on the realistic (smooth+edges) corpus image at or under ~0.09 for
sigma up to the low thousands, and under ~0.13 on the adversarial checker (which saturates to
a near-flat 0.5 under either method once sigma exceeds its own period — the checker is not
reliably the worse case, the same finding this document's R1 table below already records for
`erode`/`dilate` under `scale`). Both sit inside the 0.05–0.10 band this document's own R1
promise already accepts for this same builtin family.

**No accuracy claim below the threshold, and no correctness claim changes past it either**: a
program that asked for `sigma > 256` before this policy existed got the exact answer, just
slowly (never silently wrong, unlike `erode`/`dilate`'s pre-existing 256-radius clamp or
`bilateral_filter`'s pre-existing 7×7 window cap, which are silently WRONG past their own
caps). Past the threshold, a call that used to be exact but slow now returns a close,
measured-bounded approximation quickly — a performance trade with a disclosed accuracy cost
at the extreme end, not a bug fix.

**Scale composes for free.** `scale=` (above) multiplies `gauss_blur`'s sigma BEFORE this
policy's threshold check ever runs (the same call-site multiply described under "What is
covered"), so a coarse-scale cook simply tends to land in the cheap/exact regime more often —
no new scale-awareness was needed in this policy, and nothing here changes `scale=`'s own R1
bands or its ROI/tier-decline behaviour.

## `bilateral_filter` past the exact window (BILAT-50)

`bilateral_filter(img, spatial_sigma, range_sigma)`'s signature and registry entry are
unchanged. The old `radius = min(ceil(3·spatial_sigma), 3)` silently clamped every
`spatial_sigma` past ~1.0 to whatever a 7×7 window gives; the clamp is gone.

**At or below `spatial_sigma ≈ 1.0` (radius ≤ 3), nothing changed: byte-for-byte the same
math as every release before this ask.** From radius 4 up to `_BILATERAL_EXACT_RADIUS_MAX`
(24, i.e. `spatial_sigma` up to ~8.0) the filter runs the SAME exact weighted-average formula,
row-tiled to keep peak memory bounded independent of resolution (proven bit-identical to an
untiled pass at any tile size). Past that — the exact filter's own `O(radius²)` memory blowup
makes even a tiled exact pass too slow — a downscale + detail-transfer approximation takes
over: reduce until the residual `spatial_sigma` lands back inside the ORIGINAL exact 7×7
window, filter there exactly, upsample, and add back the full-resolution high-frequency detail
the downscale discarded.

**A windowed, tiled, or DAG-joined cook past the detail-transfer threshold is served
whole-frame, never narrowed (FIX-APPROX A1)** — the identical mechanism and the identical
reason as `gauss_blur`'s own decline above: the detail-transfer resample grid is anchored to
whatever crop it is handed, so a non-saturating window would otherwise sample on a different
phase than a whole-frame cook (measured up to 0.0265 maxdiff — the larger of the two builtins'
divergences). Below the threshold this is unaffected.

**The declared footprint matches the exact tiers' own true reach exactly (FIX-APPROX A4)**:
the reach multiplier is `3.0` (`radius = ceil(3·spatial_sigma)`), not a larger number picked to
conservatively cover the approximate tier too — A1's decline already handles that tier by
refusing to narrow at all, so nothing needs a conservative cover here.

## Perceptual accuracy past the approximation thresholds (SSIMULACRA2, FIX-APPROX A6)

The R1 promise above and GAUSSPYR-50/BILAT-50's own bands are all max-abs on a `[0,1]` channel
range — a mathematically precise but perceptually opaque number. This table adds
[SSIMULACRA2](https://github.com/cloudinary/ssimulacra2) (0–100, higher is better; ~90 is
"visually lossless," ~70 is "hard to notice artifacts without comparison to the original"),
measured on ONE shared corpus for both approximations (unifying GAUSSPYR-50's and BILAT-50's
previously independent checker/smooth+edges helpers, R1#4): an 8-pixel-period checker
(adversarial), a smooth-gradient-plus-hard-edged-rectangles image, and a new "realistic" plate
(a sinusoidal+linear gradient base, six hard-edged rectangles, and band-limited synthetic
noise — closer to a real comp's natural-statistics content than either of the other two). All
three are sRGB-gamma-encoded to 8-bit PNG before scoring (SSIMULACRA2 expects display-referred
input); the approximation is compared against the best available exact reference — the exact
convolution below `gauss_blur`'s own threshold, the exact tier's own boundary filter
(radius=24) for `bilateral_filter` past its threshold, since a true exact reference is not
computable there at all (§ above). 1080p (1080×1080), CPU (an RTX 5070 Ti Laptop workstation).

| builtin | corpus | magnitude | max-abs | SSIMULACRA2 |
|---|---|---:|---:|---:|
| `gauss_blur` | checker | sigma=260 (just past) | 0.128 | 77.6 |
| `gauss_blur` | checker | sigma=1024 (well past) | 0.209 | 39.0 |

**`bilateral_filter`'s row is corrected below (BILAT-51), not repeated here** — its original
reading in this table compared every past-threshold `spatial_sigma` against ONE fixed anchor
(the exact filter at `ss=8.0`/`radius=24`), which understated the detail-transfer path's own
error once `ss` grew large (see "Correcting the method" below). The checker rows above are
unaffected by GAUSS-51 (out of scope — an 8-pixel-period adversarial corpus was never the
gauss_blur worst case; see GAUSS8-51 immediately below for the corpus and metric this wave's
work is measured against).

## The display-8 bar for `gauss_blur` past sigma 256 (GAUSS8-51)

**GAUSS8-51's own criterion (the smooth+edges/realistic max-abs rows above are STALE and
replaced by this section): does the approximation change what the artist sees at all**, not
how far apart two floats are in scene-linear space. The plate is blurred with the shipped
pyramid path and with the exact reference, both mapped through the ACES 1.x RRT + sRGB ODT
(Hill's fitted form) and rounded to 8 bits — every pixel within 1 code (worst channel) is the
bar. Two plates (`tools/display8.py`'s `plate_day`/`plate_night` — a daylight exterior with
hard highlights, and a near-black interior with forty small, very bright practical lights, the
harder case for a highlight-compressing tone curve under a large blur), 1080×1080, CPU (an RTX
5070 Ti Laptop workstation); a 3840×3840 spot check at sigma=1024 confirmed the same bound
holds at 4K.

**The first reading found the bar badly missed, and not by noise — by a bias.** At the
original `GAUSS_BLUR_PYRAMID_QUALITY_CAP` (8.0), the night plate showed a mean SIGNED shift up
to +12 codes at sigma=1024 (worst pixel +23 in the centre half of the frame alone), the day
plate up to +4.6 at sigma=2048 — most of the error a shift, not lost detail.

**The mechanism, proved red-first.** `_gauss_blur_pyramid_approx` downsamples with one 2-D
`interpolate(mode='area')` call, then blurs the reduced image with `_gauss_blur_bchw`'s own
replicate padding — which repeats the REDUCED image's own edge pixel. That edge pixel is a
`factor`x`factor` 2-D block average taken INWARD from the border in both axes at once, so it
has already mixed interior content into what a replicate pad should treat as a pure boundary
constant. The exact convolution never does this: its pad always repeats the single, un-mixed
border row/column. Once sigma is large relative to the image — routine here, since this path
only runs past the threshold, and the kernel's 3-sigma reach often exceeds the whole frame —
the exact result is ITSELF dominated by that one replicated row, so any mismatch in what gets
replicated becomes a systematic, image-wide bias, not a local edge artifact (matching the
measured shift showing up in the CENTRE half of the frame too, not only near the border). The
area-down/bilinear-up resample pair was also checked for a half-pixel phase shift and cleared:
`tests/test_gauss8_display_bar.py::test_gauss8_mechanism_red_first_naive_pad_is_worse`
reproduces the pre-fix mechanism inline and proves it is measurably worse than the fix against
the same exact reference, isolating the border-mixing mechanism as the cause.

**The fix (`_gauss_blur_bchw_edge_pad`, `tex_runtime/stdlib_core.py`).** Downsample the border
strip ONLY along the axis parallel to it (`interpolate` on a 1-pixel-thick slice, never mixing
in the perpendicular, into-the-image direction the 2-D reduction does) to get the true edge
value at the coarse resolution, and pad the residual blur with THAT instead of the reduced
image's own edge. The vertical pass's pad is the HORIZONTALLY-BLURRED top/bottom edge (matching
`_gauss_blur_bchw`'s own pad-then-conv order, where the vertical pad reads the already
horizontally-blurred image) — skipping this made the day plate's corners visibly worse even
after the boundary-dominated night-plate case was fixed. Cost is O(H)+O(W) extra, not O(sigma)
or O(image area), so the flat-cost promise holds.

**`GAUSS_BLUR_PYRAMID_QUALITY_CAP` raised 8.0 -> 32.0 (author-approved).** At cap=8.0 the
reduced level shrinks fast enough, relative to the residual blur's own kernel radius
(`3 * min(sigma, cap)`), that even a correctly-seeded pad still lets the border dominate a
reduced image only a few pixels wide once sigma runs into the thousands. cap=32.0 keeps the
reduced level wide enough, at the sigmas below, that the bar holds; the one real blur still
runs at a bounded `sigma / factor <= quality_cap`, so cost stays flat regardless of sigma
(measured 4K, this box: ~2-10ms at both cap=8 and cap=32, sigma=260..4096, CPU and CUDA;
CUDA warm-call timings at sigma=260/1024/4096 were 1.9-2.8ms at cap=32 vs. 2.7-9.8ms at
cap=8 — no regression from the higher cap).

**The bar, before and after, at 1080x1080.** Both columns come from ONE run of the same script
(plates generated on the GPU, both paths run on an RTX 5070 Ti Laptop); "before" is the previous
release's shipped path (cap=8, no edge fix), "after" is cap=32 plus the edge fix. Cell = max
codes (worst channel) / mean signed codes / % of pixels off by >= 2 codes. SSIMULACRA2 is the
after path's, measured separately on CPU.

| sigma | plate | before | after | after SSIMULACRA2 |
|---:|---|---|---|---:|
| 260 | day | 6 / +0.34 / 25.3% | 2 / +0.08 / 0.01% | 92.6 |
| 260 | night | 18 / +3.42 / 82.1% | 1 / -0.04 / 0% | 92.5 |
| 512 | day | 6 / +1.00 / 43.8% | 1 / +0.17 / 0% | 90.9 |
| 512 | night | 13 / +5.21 / 100% | 1 / -0.00 / 0% | 94.5 |
| 1024 | day | 9 / +3.19 / 96.4% | 1 / +0.10 / 0% | 91.6 |
| 1024 | night | 27 / +11.45 / 100% | 1 / +0.02 / 0% | 95.4 |
| 2048 | day | 43 / +26.30 / 100% | 1 / +0.03 / 0% | 92.8 |
| 2048 | night | 20 / +11.89 / 100% | 1 / +0.01 / 0% | 95.9 |
| 1024 (3840x3840) | day | — | 1 / -0.00 / 0% | — |
| 1024 (3840x3840) | night | — | 1 / -0.00 / 0% | — |

**The bar (<=1 code) is met on every row but one:** sigma=260 on the day plate leaves 0.01% of
pixels at 2 codes, all near the frame border (the centre half stays within 1). Before the fix the
previous release missed it everywhere, by up to 43 codes, almost all of it a brightness bias. The
edge fix is what closes it; raising the cap alone does not (cap=32 without the fix still left a
+3 code bias on the night plate at sigma=1024). Cost stays flat in sigma. The fast-row regression
tests (the mechanism proof and the scaled-down bar check) live in
`tests/test_gauss8_display_bar.py`.

**This changes results for an existing program that already called `gauss_blur` with
sigma > 256** (the pyramid path was always documented as an approximation past the threshold,
never exact — see above — so this is not a new invariant violation, but a program's own past
sigma>256 output is not bit-identical to what it was before this fix; every below-threshold
call is untouched, proven by `torch.equal` in
`tests/test_gauss8_display_bar.py::test_gauss8_below_threshold_still_bitexact`).

## `bilateral_filter` past radius 24 (BILAT-50/BILAT-51): measured against the SAME-σ exact reference

**Correcting the method.** The original v0.50 table (above) scored every past-threshold
`spatial_sigma` against ONE fixed anchor — the exact filter at `ss=8.0`/`radius=24`, reused for
every larger `ss` because a same-σ exact reference wasn't available at the time. That anchor
conflates two different things: the picture legitimately getting blurrier as `ss` grows, and the
approximation's OWN error. Every number below instead compares against the TRUE exact filter AT
THE SAME `spatial_sigma` — the same tiled math the exact tier already runs, extended purely for
this measurement (confirmed feasible through `radius=96`, i.e. `ss=32`; `radius=192`/`ss=64` was
not attempted for the full corpus sweep — the measured `O(r²)` growth puts it at several minutes
per reference call). Three methods, one shared corpus (checker / smooth+edges / realistic, §
above), 1080p, CPU:

| method | mechanism | corpus | ss=8.5 | ss=16 | ss=32 |
|---|---|---|---:|---:|---:|
| v0.49 (fixed 7×7 clamp) | today's exact math, window frozen at radius=3 regardless of `ss` | checker | 100 / 0.0 | 100 / 0.0 | 100 / 0.0 |
| v0.49 | " | smooth+edges | 69.6 / 0.052 | 52.0 / 0.066 | 33.7 / 0.075 |
| v0.49 | " | realistic | 34.6 / 0.063 | 23.9 / 0.083 | 18.5 / 0.092 |
| v0.50 (BILAT-50 detail-transfer) | downscale + residual add-back | checker | 97.7 / 0.0004 | 97.1 / 0.001 | 97.6 / 0.0002 |
| v0.50 | " | smooth+edges | 46.8 / 0.080 | 32.1 / 0.079 | 6.0 / 0.080 |
| v0.50 | " | realistic | 8.3 / 0.095 | 3.3 / 0.093 | -11.6 / 0.116 |
| **v0.51 (BILAT-51, shipped)** | separable (row-then-column) bilateral pass | checker | 100 / 0.0 | 100 / 0.0 | 100 / 0.0 |
| **v0.51** | " | smooth+edges | 93.3 / 0.027 | 91.8 / 0.029 | 87.8 / 0.031 |
| **v0.51** | " | realistic | 91.9 / 0.026 | 91.4 / 0.026 | 89.1 / 0.033 |

(cell = SSIMULACRA2 / max-abs.)

**Reading the corrected table**: measured correctly, v0.50's detail-transfer path was WORSE
than the v0.49 clamp it replaced on realistic content at every one of these radii (8.3 vs 34.6
at ss=8.5, and actually negative — perceptually worse than doing nothing differently — by
ss=32), not merely "lower than its own max-abs band would suggest" as the original table's
prose put it; the fixed-anchor methodology had been hiding this. The separable pass shipped in
BILAT-51 clears SSIMULACRA2 ≥ 80 on every corpus at every one of these radii, beating both
priors by a wide margin. Two other candidates were also built and measured this round (a
Kopf-style joint bilateral upsample, including a per-level range-sigma-scheduled variant; a
bilateral grid with true trilinear splat/slice and a separable Gaussian space+range blur, tried
at a few grid resolutions) and neither beat the separable pass or even the v0.49 clamp on
realistic content. The summary is that a naive hand-built JBU or bilateral grid is a real
build-and-tune project, not a quick win, and this round's separable result made further tuning
of either moot once it cleared the bar outright.

**Cost trade, stated plainly**: the separable pass is NOT flat-cost like detail-transfer was —
it grows with `radius` (`O(image size × radius)`). `radius ≤ 24` (spatial_sigma ≤ ~8.0) is
untouched — byte-for-byte the same exact math as every release before this one.

## Time + memory: separable vs detail-transfer vs exact-tiled, r=25/32/64/128/256

Best-of-5 wall time (best-of-2/3 at the two most expensive exact-tiled cells, named below),
realistic corpus; CPU on an RTX 5070 Ti Laptop workstation, CUDA on an RTX 2080 SUPER (sm_75),
each measured with the GPU otherwise idle.
`exact_tiled` past radius=64 is NOT run directly (its own `O(r²)` cost — confirmed by the
measured cells below — makes the remaining ones multi-minute-to-multi-hour; the trend is
extrapolated from the measured points, not asserted from a new run):

| radius | size | device | detail_transfer | separable | exact_tiled |
|---:|---|---|---:|---:|---:|
| 25 | 1080p | CPU | 0.005 s | 0.401 s | 7.33 s |
| 32 | 1080p | CPU | 0.005 s | 0.524 s | 10.9 s |
| 64 | 1080p | CPU | 0.004 s | 1.00 s | 54.9 s (2 reps) |
| 128 | 1080p | CPU | 0.003 s | 1.85 s | not run — extrapolated ≈220 s |
| 256 | 1080p | CPU | 0.004 s | 3.79 s | not run — extrapolated ≈880 s |
| 25 | 4K | CPU | 0.061 s | 9.22 s | 103 s |
| 32 | 4K | CPU | 0.062 s | 13.1 s | 182 s |
| 64 | 4K | CPU | 0.060 s | 25.9 s | not run — extrapolated ≈700 s |
| 128 | 4K | CPU | 0.053 s | 51.2 s | not run — extrapolated ≈2900 s |
| 256 | 4K | CPU | 0.054 s | **106 s** | not run — extrapolated ≈3.2 h |
| 25 | 1080p | CUDA | 0.0008 s | 0.062 s | 0.89 s |
| 32 | 1080p | CUDA | 0.0008 s | 0.079 s | 1.41 s |
| 64 | 1080p | CUDA | 0.0009 s | 0.156 s | 5.47 s (3 reps) |
| 128 | 1080p | CUDA | 0.0008 s | 0.310 s | not run — extrapolated ≈22 s |
| 256 | 1080p | CUDA | 0.0009 s | 0.615 s | not run — extrapolated ≈88 s |
| 25 | 4K | CUDA | 0.005 s | 0.739 s | 11.1 s |
| 32 | 4K | CUDA | 0.005 s | 0.940 s | 17.5 s |
| 64 | 4K | CUDA | 0.005 s | 1.86 s | 68.5 s (3 reps) |
| 128 | 4K | CUDA | 0.006 s | 3.71 s | not run — extrapolated ≈274 s |
| 256 | 4K | CUDA | 0.006 s | 7.39 s | not run — extrapolated ≈18 min |

**Memory**: CUDA's own `torch.cuda.max_memory_allocated` (the only precise reading available —
CPU peak was estimated from `psutil` RSS deltas, which are noisy on Windows and reported for
context only, not as a load-bearing number) shows `separable` FLAT at ~122.5 MB (1080p) /
~1475.7 MB (4K) across every radius 25→256 — confirming the O(image size)-only memory design
(one `[B,C,H,W]` tap tensor alive at a time, never an `O(radius²)` window) — while
`exact_tiled` grows with radius even under tiling (1080p: 247→209→695 MB at r=25/32/64; 4K:
896→1122→2846 MB). `detail_transfer` is flat too, as always (~74–75 MB / ~886 MB).

**Where separable's cost exceeds a sane interactive budget** (~200 ms, this document's own
working assumption — no builtin in this codebase has ever been "interactive" much past its
exact tier's own boundary, so this is about disclosure, not a new promise): on CPU, separable
is already past 200 ms at the SMALLEST measured radius (25) at both resolutions — CPU was never
interactive here, matching the exact tier's own pre-existing 7.3 s at its radius=24 boundary.
On CUDA at 1080p, it crosses 200 ms between radius=64 (156 ms) and radius=128 (310 ms). On CUDA
at 4K, it is already past 200 ms at radius=25 (739 ms). Separable was never meant to compete
with detail-transfer's near-zero cost on interactivity; its job is to be the accurate default
for a committed (non-preview) cook, at a cost in the same order the exact tier's own boundary
case already pays.

**The crossover moved: 256 → 96, red-first (`tests/test_bilat50_radius.py::test_bilat51_
separable_ceiling_matches_measured_evidence`).** Two independent things point the same way.
First, this ask's own SSIMULACRA2 sweep (the table above) only measured separable beating both
priors through `spatial_sigma=32` (radius=96) — the first-shipped ceiling of 256 silently
extended the regime into a radius this ask never scored for accuracy. Second, the cost sweep
above found the worst measured cell (radius=256, 4K, CPU) at **106 seconds** — cost that keeps
growing (measured, not merely extrapolated, up to radius=128) with no accuracy evidence past
radius=96 to justify paying it. `_BILATERAL_SEPARABLE_RADIUS_MAX` is now `96`: every radius this
ask actually measured stays exactly as scored above; radius=96 itself was not run directly but
interpolates to ≈39 s at 4K/CPU and ≈2.8 s at 4K/CUDA from the measured radius=64/128 points —
both well under half the radius=256 worst case they replace, and the same order as the exact
tier's own already-accepted boundary cost (54.9 s at radius=64, 1080p CPU); a
`spatial_sigma` past that now falls back to detail-transfer (flat, ms-scale) rather than an
unvalidated extension of separable's measured range. `radius ≤ 24` is unaffected either way.

**CUDA correctness**: `_bilateral_separable_bchw` on CUDA (the second workstation's RTX 2080
SUPER) matches its own CPU output within `8.3e-7` max-abs across radius 25/64/128 on two canvas
sizes — well inside invariant 2's `1e-5` tolerance (this is a plain tensor-op implementation
with no device-shaped branch, so bit-exactness modulo float rounding is the expected reading,
not a surprise). `tests/test_bilat50_radius.py` (19 cases) and `tests/test_fixapprox_a1_window_
decline.py` both pass there with CUDA present.

## Precision under scale

A coarse cook (`scale` neither `None` nor `1.0`) whose caller left `precision` at its literal
default promotes to `"auto"` — the existing invariant #10 accuracy net, unchanged, already
reasons about data amplification independent of canvas resolution, so "reduced precision under
scale's envelope, never surfaced as a new decision" needs no new mechanism. An explicit
`precision=` on a coarse cook is still honoured unchanged.
