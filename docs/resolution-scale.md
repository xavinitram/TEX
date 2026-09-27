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
computable there at all (§ above). 1080p (1080×1080), CPU, this box.

| builtin | corpus | magnitude | max-abs | SSIMULACRA2 |
|---|---|---:|---:|---:|
| `gauss_blur` | checker | sigma=260 (just past) | 0.128 | 77.6 |
| `gauss_blur` | checker | sigma=1024 (well past) | 0.209 | 39.0 |
| `gauss_blur` | smooth+edges | sigma=260 | 0.034 | 87.2 |
| `gauss_blur` | smooth+edges | sigma=1024 | 0.049 | 84.9 |
| `gauss_blur` | realistic | sigma=260 | 0.036 | 85.6 |
| `gauss_blur` | realistic | sigma=1024 | 0.041 | 84.9 |
| `bilateral_filter` | checker | ss=8.5 (just past) | 0.0004 | 97.7 |
| `bilateral_filter` | checker | ss=64 (well past) | 0.0001 | 100.0 |
| `bilateral_filter` | smooth+edges | ss=8.5 | 0.078 | 56.2 |
| `bilateral_filter` | smooth+edges | ss=64 | 0.083 | 50.4 |
| `bilateral_filter` | realistic | ss=8.5 | 0.087 | 43.4 |
| `bilateral_filter` | realistic | ss=64 | 0.110 | 25.0 |

**Reading the table**: `gauss_blur`'s pyramid holds up well on both realistic corpora
(smooth+edges and realistic both stay in the "high quality" 80s regardless of how far past the
threshold sigma goes) and only degrades on the adversarial checker at a very large sigma — the
same "checker isn't reliably the worse case until it is" pattern this document's R1 table
already records elsewhere. `bilateral_filter`'s detail-transfer path scores much lower on both
realistic corpora (40s-50s, "low-to-medium quality") than its own max-abs band alone would
suggest — the max-abs number (0.08-0.11) sits inside BILAT-50's own accepted band, but the
edge-preserving filter's whole PURPOSE is to keep hard boundaries crisp, and SSIMULACRA2 is
more sensitive to boundary/detail mismatches than a uniform per-pixel max-abs is. This is a
real, disclosed quality gap on top of BILAT-50's own numbers, not a new bug: the mechanism
(add back the full-resolution detail a downscale discarded) is a bounded-cost STAND-IN for a
true joint bilateral upsample (RADIUS-50a-design.md D3, option 2 — not built, no evidence yet
that a real workflow needs it), and this table is the disclosure that stand-in owes past its
own max-abs band.

## Precision under scale

A coarse cook (`scale` neither `None` nor `1.0`) whose caller left `precision` at its literal
default promotes to `"auto"` — the existing invariant #10 accuracy net, unchanged, already
reasons about data amplification independent of canvas resolution, so "reduced precision under
scale's envelope, never surfaced as a new decision" needs no new mechanism. An explicit
`precision=` on a coarse cook is still honoured unchanged.
