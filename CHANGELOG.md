# Changelog

All notable changes to TEX Wrangle will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Snippet system** — right-click context menu with cascade submenus for browsing and inserting 36 built-in example snippets organized by category (Color, Effects, Filter, Generate, Mask, Latent, String, Video)
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
