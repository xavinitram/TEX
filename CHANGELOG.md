# Changelog

All notable changes to TEX Wrangle will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
