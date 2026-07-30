# TEX Wrangle — Developer Guide

This document covers the internals of TEX Wrangle for developers who want to understand, extend, or contribute to the project.

## Architecture

```
TEX_Wrangle/
  __init__.py              # ComfyUI entry point, version
  tex_node.py              # ComfyUI node class (device/compile_mode params, UI integration)
  tex_marshalling.py       # Input marshalling, type inference, output preparation
  tex_cache.py             # Two-tier compilation cache (memory LRU + disk pickle)
  tex_fusion.py            # Cross-node fusion: splice a linked TEX chain into one program
  tex_tool.py              # TOOL-1..5: the .textool bundling format (loader, publish, cook, warm keys)
  tex_lsp.py               # LANG-7: stdio JSON-RPC Language Server (diagnostics/completion/hover)
  tex_compiler/
    lexer.py               # Tokenizer
    parser.py              # Recursive-descent parser -> AST
    ast_nodes.py           # AST node definitions (with __slots__)
    type_checker.py        # Static type analysis
    optimizer.py           # Const-fold, algebraic simplify, DCE, CSE, LICM, small-loop unroll
    diagnostics.py         # Structured error diagnostics (codes, carets, suggestions, hints)
    stdlib_signatures.py   # Function signatures for type checking
  tex_runtime/
    interpreter.py         # Tree-walking tensor evaluator
    stdlib.py              # Built-in function implementations (math, color, sampling, SDF, string, array)
    noise.py               # Procedural noise library (Perlin, Worley, FBM, curl, etc.)
    compiled.py            # torch.compile wrapper with backend cascade
    codegen.py             # AST -> Python function compiler
  js/
    tex_extension.js       # Frontend: auto-socket, CodeMirror 6 editor, help popup
    tex_cm6_bundle.js      # Pre-built CodeMirror 6 bundle (Rollup)
  tests/
    README.md              # Test suite structure and how to add tests
    helpers.py             # Shared test utilities (SubTestResult, compile helpers, fixtures)
    conftest.py            # pytest fixture wiring
    run_all.py             # Standalone runner (no pytest dependency)
    test_lexer.py          # Lexer token tests
    test_parser.py         # Parser AST tests
    test_type_checker.py   # Type checker tests
    test_interpreter.py    # Interpreter execution tests
    test_language.py       # Language feature tests (if/else, loops, ternary, scoping)
    test_stdlib.py         # Stdlib function tests (math, color, SDF, edge cases)
    test_strings_arrays.py # String and array operation tests
    test_noise_sampling.py # Noise generation and sampling tests
    test_bindings_params.py # Binding access, params, user functions
    test_codegen_optimizer.py # Codegen equivalence and optimizer pass tests
    test_integration.py    # End-to-end: cache, device, torch.compile, node helpers
    test_diagnostics.py    # Error message and diagnostic quality tests
    test_performance.py    # Timing benchmarks (@pytest.mark.slow)
  benchmarks/
    run_benchmarks.py      # Reproducible performance benchmarks
    README.md              # Benchmark usage and result format docs
  examples/                # Example TEX snippets (114 files)
  .tex_cache/              # Disk cache directory (auto-created, gitignored)
```

## Compilation Pipeline

```
Source Code
    |
    v
+---------+     +--------+     +-------------+     +-----------+
|  Lexer  |---->| Parser |---->| TypeChecker |---->| Optimizer |
| lexer.py|     |parser.py|    |type_checker |     |optimizer  |
+---------+     +--------+     +-------------+     +-----------+
  tokens          AST            type_map           optimized AST
                                                         |
                                            +------------+------------+
                                            |                         |
                                            v                         v
                                    +-------------+           +-----------+
                                    | Interpreter |           |  Codegen  |
                                    |interpreter.py|          |codegen.py |
                                    +-------------+           +-----------+
                                     tensor result             Python fn
                                    (tree-walking)          (exec'd callable)
```

**Lexer** (`tex_compiler/lexer.py`): Converts source text into a token stream. Each token has a type (e.g. `NUMBER`, `IDENTIFIER`, `PLUS`), a value, and a source location `(line, column)`.

**Parser** (`tex_compiler/parser.py`): Consumes tokens and builds an AST (Abstract Syntax Tree). Uses recursive descent with explicit operator precedence levels. AST nodes are defined as dataclasses in `ast_nodes.py`.

**TypeChecker** (`tex_compiler/type_checker.py`): Walks the AST and assigns a `TEXType` to every expression node. Enforces type compatibility rules, validates function signatures, and manages variable scopes. Produces a `type_map` dict mapping AST node `id()` -> `TEXType`. Fetch/sample calls return the binding's actual type (VEC3 for IMAGE, FLOAT for MASK) rather than a hardcoded VEC4.

**Optimizer** (`tex_compiler/optimizer.py`): Constant folding and algebraic simplification (`x * 1.0` -> `x`, strength reduction like `x / c` -> `x * (1/c)`, pre-evaluating constant sub-expressions), plus dead-code elimination (DCE), common-subexpression elimination (CSE), loop-invariant code motion (LICM), and small-loop unrolling. Optimizer-synthesized nodes are re-type-checked from scratch so the type map stays complete.

**Interpreter** (`tex_runtime/interpreter.py`): Tree-walking evaluator that executes the AST using PyTorch tensor operations. Reads types from `type_map` to guide evaluation (e.g. choosing `torch.where` for if/else). Produces output tensors/strings for all assigned `@name` bindings. Used as the default execution path.

**Codegen** (`tex_runtime/codegen.py`): Compiles the AST into a Python function string, then `exec()`s it into a callable. Eliminates per-node dispatch overhead. All env variables are pre-registered as Python locals (`_lv_{name}`) to avoid dict lookups and produce cleaner FX graphs for TorchInductor. Falls back to the interpreter for unsupported patterns (string operations). Includes **stencil specialization**: nested for-loops matching spatial filter patterns are detected and replaced with bulk PyTorch ops (`avg_pool2d` for box blur, `max_pool2d` for min/max reduction, `Tensor.unfold` for median/rank filters, depthwise `conv2d` for weighted stencils). Also detects inline (non-loop) stencil patterns from hand-unrolled fetch sequences. For loops with sample/fetch calls, BCHW conversion and grid buffers are hoisted outside the loop, and sample() calls are inlined as direct `grid_sample` operations (bypassing the stdlib wrapper).

**Compiled** (`tex_runtime/compiled.py`): Execution routing and optional `torch.compile` wrapper. Routes programs to the fastest path: plain interpreter (trivial programs or no spatial tensors), codegen-only (deep loop nesting where torch.compile overhead exceeds benefit), or codegen + torch.compile (spatial tensor chains that benefit from kernel fusion). Manages a bounded LRU cache of compiled callables with `dynamo.reset()` on eviction to reclaim Inductor kernel memory.

**Noise** (`tex_runtime/noise.py`): Procedural noise library with 2D/3D implementations. Contains Perlin gradient noise (arithmetic hash for TorchInductor compatibility), simplex noise, FBM with tiered compilation (eager → jit.trace → torch.compile), Worley/Voronoi cell noise, curl (divergence-free flow fields), and FBM variants (ridged, billow, turbulence, flow, alligator). All functions accept optional `z` parameter for 3D evaluation.

**tex_node.py** + **tex_marshalling.py**: ComfyUI integration layer. `tex_node.py` defines the node class with device/compile_mode parameters and orchestrates execution. `tex_marshalling.py` handles input marshalling (converting ComfyUI types to TEX tensors), type inference for bindings, and output preparation (converting results back to ComfyUI IMAGE/MASK/LATENT/STRING).

**Caching** (`tex_cache.py`): Sits between source input and type checking. Caches `(AST, type_map)` tuples keyed by `SHA256(code + binding_types)`. Two tiers: in-memory LRU (128 entries) and on-disk pickle (512 entries, versioned).

## Tensor Layout Conventions

TEX uses **channel-last** layout throughout, matching ComfyUI's convention:

| Data Type | Shape | Description |
|-----------|-------|-------------|
| Image | `[B, H, W, C]` | B=batch, H=height, W=width, C=channels (3 or 4) |
| Mask | `[B, H, W]` | Single-channel spatial data |
| Scalar array | `[B, H, W, N]` | N elements per pixel |
| Vector array | `[B, H, W, N, C]` | N elements x C channels per pixel |
| Matrix (mat3) | `[B, H, W, 3, 3]` | 3x3 matrix per pixel (or `[3, 3]` for constants) |
| Matrix (mat4) | `[B, H, W, 4, 4]` | 4x4 matrix per pixel (or `[4, 4]` for constants) |
| String array | Python `list[str]` | Not a tensor |
| Latent (input) | `[B, C, H, W]` | Channel-first (ComfyUI convention) |
| Latent (internal) | `[B, H, W, C]` | Permuted to channel-last for processing |

Latent tensors are permuted from `[B,C,H,W]` to `[B,H,W,C]` on input and back on output. This permutation is handled in `tex_node.py`, transparent to TEX code.

## Per-Pixel Vectorization

TEX achieves per-pixel semantics without explicit pixel loops by representing all values as tensors with spatial dimensions. A TEX expression like:

```c
@OUT = @A * 0.5 + vec4(u, v, 0.0, 1.0);
```

becomes:
```python
# @A is [B, H, W, 4], u is [B, H, W], 0.5 is scalar
result = A_tensor * 0.5 + torch.stack([u_tensor, v_tensor, zeros, ones], dim=-1)
# result is [B, H, W, 4] -- all pixels computed simultaneously
```

**Built-in variables** are pre-created as broadcast-ready tensors:

| Variable | Shape | Creation |
|----------|-------|----------|
| `ix` | `[1, 1, W]` | `torch.arange(W)` expanded |
| `iy` | `[1, H, 1]` | `torch.arange(H)` expanded |
| `u` | `[1, 1, W]` | `ix / (W - 1)` |
| `v` | `[1, H, 1]` | `iy / (H - 1)` |
| `fi` | `[B, 1, 1]` | `torch.arange(B)` expanded |
| `fn` | scalar | `float(B)` |
| `iw`, `ih` | scalar | `float(W)`, `float(H)` |

PyTorch broadcasting automatically expands these to full `[B, H, W]` when combined in expressions.

## Vectorized if/else

TEX's `if/else` uses `torch.where()` for per-pixel branching. **Both branches execute fully** on all pixels:

1. Saves current environment state
2. Evaluates the condition -> boolean mask `[B, H, W]`
3. Executes then-branch -> captures modified variables
4. Restores environment, executes else-branch -> captures modified variables
5. Merges results: `result = torch.where(condition, then_value, else_value)` per variable

Side effects in branches (array assignments, etc.) are merged using the same `torch.where` pattern.

## Loops

For and while loops execute sequentially -- each iteration runs the body as vectorized tensor operations. The loop variable is a scalar (not a per-pixel tensor). Each iteration computes the body across all pixels simultaneously. The iteration limit is 1024 (`MAX_LOOP_ITERATIONS`). Both `break` and `continue` are supported.

## Type System

```
VOID -> INT -> FLOAT -> VEC2 -> VEC3 -> VEC4
                        MAT3    MAT4 (internal only, no @OUT)
                               STRING (no numeric promotion)
                               ARRAY (container type)
```

**Promotion rules** (automatic):
- `INT` + `FLOAT` -> `FLOAT`
- `FLOAT` + `VEC2` -> `VEC2` (broadcast scalar to all channels)
- `VEC2` + `VEC3` -> `VEC3` (zero-pad missing channels)
- `VEC3` + `VEC4` -> `VEC4` (alpha = 1.0)
- `STRING` does NOT coerce to/from numeric types

**Auto-inference** for outputs: The type checker tracks all assignments to `@name` bindings and infers output types. Results are stored in `checker.assigned_bindings` (dict mapping name -> TEXType).

**Array type tracking**: `TEXArrayType` stores element type and size. The `_array_meta` dict in the interpreter tracks array sizes at runtime for bounds clamping.

## Two-Tier Cache

```
compile_and_run(code, bindings)
    |
    v
+-------------------------------------+
|  fingerprint = SHA256(code + types)  |
+--------------------+-----------------+
                     |
            +--------v--------+
            | Memory LRU hit? |--yes--> return cached (AST, type_map)
            +--------+--------+
                     | no
            +--------v--------+
            |  Disk .pkl hit? |--yes--> re-run TypeChecker (regenerate id()-based type_map)
            +--------+--------+         promote to memory cache, return
                     | no
            +--------v--------+
            |  Full compile   |--> Lexer -> Parser -> TypeChecker
            +--------+--------+
                     |
         Store in memory cache + disk cache
                     |
                     v
              return (AST, type_map)
```

**Memory cache**: `OrderedDict` with LRU eviction (128 entries). Keys are SHA256 fingerprints. Values are `(program_ast, type_map, referenced_bindings)` tuples.

**Disk cache**: Pickle files in `.tex_cache/` directory (512 max). Stores `(program_ast, binding_types, cache_version)`. On load, the TypeChecker must re-run because `type_map` keys are `id()` values that change between sessions.

**Cache version (CACHE-4 layered epochs)**: as of v0.25 the one mono-hash is split into a nested epoch lattice — `AST_EPOCH` gates the `.pkl` (parse/typecheck/optimize files), `CODEGEN_EPOCH = H(AST_EPOCH, codegen/interpreter files, cgreuse)` gates the `.cg` sidecars + inductor dir, `VERDICT_EPOCH = H(CODEGEN_EPOCH, tier-policy files)` gates `autotier.json`/`warm_state.json`. A codegen-only edit no longer cold-starts the `.pkl` tier. `_CACHE_VERSION` remains as a back-compat alias of `codegen_epoch()`; the full mono-hash is demoted to a completeness tripwire. All still cause a graceful cache miss, not a crash. See `docs/results-caching.md`.

## String vs Tensor Execution

**Spatial mode** (tensor): When any input is an image/mask/latent tensor, the interpreter creates built-in spatial variables (ix, iy, u, v, etc.) and all operations are vectorized across `[B, H, W]`.

**Scalar mode** (string-only): When all inputs are strings/scalars and the output is a string, `spatial_shape` is `None`. Built-in variables are not created. String operations execute once (not per-pixel).

Mixed programs (tensor inputs + string variables) work naturally -- string operations are scalar, tensor operations are spatial, and they can coexist via `str()` / `to_float()` conversion functions.

## Frontend Extension

The JavaScript frontend (`js/tex_extension.js`) provides:

**Auto-socket creation**: A regex parser scans TEX code for `@name` references and `$name` parameter declarations. For each `@name`, a LiteGraph input/output slot is created dynamically. For each `$name`, a typed widget (FLOAT/INT/STRING) is created on the node. Sockets are updated on a 400ms debounce.

**CodeMirror 6 editor**: Bundled CM6 editor providing syntax highlighting, autocompletion, error squiggles, and bracket matching. The original ComfyUI textarea is spliced from the widget array and replaced with a DOM widget hosting the CM6 EditorView (compatible with both legacy and Nodes 2.0 rendering). Uses the Monaspace Neon font with ligatures.

**Error display**: Listens for ComfyUI's WebSocket `execution_error` events. Errors are rendered above the node title bar and as inline diagnostics in the editor.

---

## Cross-Node Fusion

`tex_fusion.py` lets a linear chain of linked TEX nodes compile and run as **one** program, so only the terminal node executes — intermediate nodes never materialize or cache an image. It is opt-in via the `TEX Fusion` ComfyUI settings.

- **Frontend** (`js/tex_extension.js`): at `graphToPrompt` time, a maximal TEX-only chain is detected and collapsed into its terminal node. The upstream nodes are **deleted from the submitted prompt** (so the executor never schedules them), their `{code, params}` are serialized into the terminal's `_tex_chain` input, and a faint Houdini-style bubble is drawn behind the fused region. An integrity scan guarantees the rewritten prompt never contains a dangling link.
- **Backend** (`tex_fusion.py`): `prepare_fused` / `compile_fused` splice the stages — each non-terminal `@OUT` becomes a typed local, the next stage's chain input reads that local, and every user identifier (locals, `@`-inputs, `$`-params, user functions) is namespaced per stage (`_s{i}_`) so independently-authored stages never collide. The merged program is re-type-checked and re-optimized through the normal pipeline, making it **bit-equivalent** to running the stages sequentially.
- **Chain breaks** raise `FusionError` (caught in `tex_node.execute` and shown cleanly, never as a bug): a scatter write to `@OUT`, more than one output, `@OUT` used inside a loop, or a non-image/mask handoff type. The frontend additionally stops a chain at a Preview/Save tap, a fan-out, or a multi-input node, leaving those nodes to run normally.

Validated by `benchmarks/fusion_splice_test.py` and `benchmarks/fusion_regression_test.py` (fused output == sequential, including cross-stage name collisions and every chain-break case).

## How-To Guides

### Adding a New Stdlib Function

**Example: adding `saturate(x)` that clamps to [0, 1].**

1. **`tex_runtime/stdlib.py`** -- implement the function (or `tex_runtime/noise.py` for noise functions) and register it with the co-located `@stdlib(...)` decorator (REG-1 — there is no central `get_functions()` dict; it derives from the decorators). `sig=`/`category=` carry the help data (LANG-4), `doc=`/`ex=` the description/example:
```python
@stdlib("saturate", doc="Clamp value to [0,1].", ex="@OUT = vec4(saturate(@A.rgb), 1.0);",
        sig="saturate(x) → float", category="Math")
@staticmethod
def fn_saturate(x):
    """Clamp value to [0, 1] range."""
    return torch.clamp(_to_tensor(x), 0.0, 1.0)
```

2. **`tex_compiler/stdlib_signatures.py`** -- add the type signature:
```python
"saturate": {"args": (1, 1), "return": _passthrough_type},
# (1, 1) = exactly 1 argument; _passthrough_type = returns same type as input
```
Return type options: `TEXType.FLOAT`, `TEXType.VEC3`, `TEXType.VEC4`, `TEXType.STRING`, `TEXType.INT`, or `_passthrough_type` (callable that returns the first arg's type).

3. **`tex_compiler/type_checker.py`** -- add validation if needed (optional):
```python
# Only needed for special validation beyond signature checking.
if node.name == "saturate":
    if arg_types and arg_types[0] == TEXType.STRING:
        self._error("saturate() expects a numeric argument", node.loc)
```

4. **`js/tex_extension.js`** -- add a `TEX_HELP_DATA` entry (autocomplete + help popup):
```javascript
{ name: "saturate", sig: "saturate(x) → float", desc: "Clamp value to [0,1].", example: "@OUT = vec4(saturate(@A.rgb), 1.0);" },
```

**⚠️ 4b. If your function reads NEIGHBOURING pixels or the whole image**
(sample/fetch/blur/morphology/reduction), you MUST classify it **in the `@stdlib(...)`
decorator** — the taxonomy sets DERIVE from the tags (TST-3 fails a mismatch); do NOT
edit the sets by hand. Leaving a footprint unset is **silently wrong when tiled** and/or
fails CUDA-graph capture:
- `footprint=` (ROI-1) — one of `('halo', r)`, `('halo_arg', i[, mult])`, `'image'`,
  `('frame', i)`; derives `tex_memory._NON_LOCAL_FNS` (default `'point'` → wrong output when
  tiled). The optional `mult` (default `1.0`) turns a non-pixel arg into a pixel reach —
  `gauss_blur` is `('halo_arg', 1, 3.0)` (radius `= 3·sigma`); a missing `mult` under-pads the
  ROI cook halo (ROI-4's reach-pinning test catches it).
- `sync=True` — if it does an internal `.item()`/sync (derives `graphed._SYNC_STDLIB`).
- `spatial=True` — if codegen should lower it as a stencil (derives `codegen._SPATIAL_STDLIB`).

> **Canonical recipe: `AGENTS.md`** (this repo root). The invariants and the full
> table list live there; keep this section in sync with it.

5. **`tests/test_stdlib.py`** -- add a test (see `tests/README.md` for the full pattern):
```python
try:
    result = compile_and_run("@OUT = vec4(saturate(1.5), saturate(-0.5), saturate(0.5), 1.0);", {"A": img})
    assert abs(result[0,0,0,0].item() - 1.0) < 1e-4  # clamped to 1
    assert abs(result[0,0,0,1].item() - 0.0) < 1e-4  # clamped to 0
    assert abs(result[0,0,0,2].item() - 0.5) < 1e-4  # unchanged
    r.ok("saturate function")
except Exception as e:
    r.fail("saturate function", f"{e}\n{traceback.format_exc()}")
```

### Adding a New Built-in Variable

**Example: adding `aspect` (image aspect ratio `iw / ih`).**

1. **`tex_runtime/interpreter.py`** -- add in `_create_builtins()`:
```python
self.env["aspect"] = torch.tensor(float(W) / float(H), dtype=torch.float32, device=self.device)
```

2. **`tex_compiler/type_checker.py`** -- add to the `builtins` dict:
```python
builtins = {
    ...,
    "aspect": TEXType.FLOAT,
}
```

3. **`js/tex_extension.js`** -- add to `TEX_COORD_VARS` for syntax highlighting:
```javascript
const TEX_COORD_VARS = new Set([..., "aspect"]);
```
Update `TEX_HELP_HTML` to document it.

### Adding a New Type

Adding a new TEX type requires changes across the entire pipeline:

1. **`tex_compiler/type_checker.py`** -- add to `TEXType` enum. Add promotion rules in `_promote()` and compatibility in `_is_compatible()`.
2. **`tex_compiler/parser.py`** -- add to `TYPE_NAME_MAP` dict if it can be used in declarations.
3. **`tex_runtime/interpreter.py`** -- handle the new type in `_eval()`, `_exec_assignment()`, and any type-specific evaluation paths.
4. **`tex_marshalling.py`** -- add input/output handling in `infer_binding_type()`, `prepare_output()`, and `map_inferred_type()`.

### Adding a New AST Node

1. **`tex_compiler/ast_nodes.py`** -- define the dataclass with `__slots__`.
2. **`tex_compiler/parser.py`** -- add parsing logic that creates the node. Use `self._loc()` to capture source location.
3. **`tex_compiler/type_checker.py`** -- add a check method and dispatch from `_check_statement()` or `_check_expression()`.
4. **`tex_runtime/interpreter.py`** -- add an exec/eval method and dispatch from `_exec_stmt()` or `_eval()`.

### Adding a New Operator

1. **`tex_compiler/lexer.py`** -- add the token type and recognition logic.
2. **`tex_compiler/parser.py`** -- add to the appropriate precedence level in the expression parser.
3. **`tex_compiler/type_checker.py`** -- add type checking in `_check_binary_op()` or `_check_unary_op()`.
4. **`tex_runtime/interpreter.py`** -- add evaluation in `_eval_binary_op()` or `_eval_unary_op()`.

## Running Tests

```bash
cd custom_nodes/TEX_Wrangle

# Full suite (~79 test functions, ~1358 sub-tests)
python -m pytest tests/ -v

# Skip slow timing tests
python -m pytest tests/ -v -m 'not slow'

# Single domain
python -m pytest tests/test_stdlib.py -v

# Standalone runner (no pytest dependency)
python tests/run_all.py
```

See `tests/README.md` for the test suite structure, sub-test pattern, and how to add new tests.

## Benchmarks

See `benchmarks/README.md` for full documentation. Quick start:

```bash
# Device x cache x compile matrix (CUDA-synchronized) — the canonical perf harness
python benchmarks/eight_config_bench.py

# GPU resolution-scaling + kernel-launch profile (launch-bound vs compute-bound)
python benchmarks/gpu_profile.py

# 4-scenario benchmark (compile off/on × cold/warm)
python benchmarks/four_scenario_bench.py

# Synthetic + example program benchmarks
python benchmarks/run_benchmarks.py

# Measure whether the Turing-calibrated perf gates hold on YOUR GPU (S-4)
python -m TEX_Wrangle.tex_cli validate-hw
```

### Performance facts worth knowing (P6)

These are measured, non-obvious behaviours a maintainer will otherwise rediscover the hard way:

- **Codegen is NOT a universal win.** A forced-codegen sweep across the 116 examples regresses
  ~61/100 (median 0.94×) — widening the codegen route naively is *slower* on most programs
  (color_grade is 0.43–0.57× vs the interpreter). So the default cook stays on the interpreter
  and only routes to codegen where a measured win exists (the UC-2 stencil gate). The honest fix
  for broad codegen routing is a cost model — **PROF-1 (v0.31) is now that substrate**, and
  wiring it to routing is unscheduled rather than blocked (this line read "deferred to v0.20"
  for twelve releases). Until it lands, **~4% of programs
  benefit from codegen routing** — treat "just codegen everything" as a known trap.
- **Noise `torch.compile` is `dynamic=True` (P2).** One compiled kernel serves every resolution,
  so a resolution dance (512→1024→512) no longer thrashes torch.compile's shape guards — that was
  a measured **134× / 5.6 s** recompile stall. CPU compile is ~13× faster than the jit.trace tier
  (so we do NOT cap CPU at trace), and dynamic stays within ~1 fp32 ULP of the static kernel.
  Any noise compile (or future recompile) is now surfaced in `tex doctor` (`noise_compiles`, P6).
- **CUDA mat×vec uses an elementwise broadcast, not `matmul` (P3).** For TEX's tiny-matrix /
  huge-per-pixel-batch shape, `(m * v.unsqueeze(-2)).sum(-1)` is 3.4–3.9× faster than `matmul` on
  CUDA; CPU keeps `matmul` (7× faster there). Both the interpreter and codegen emit the identical
  device-gated expression, so interp↔codegen stays bit-exact per device.
- **`is_tile_safe` is memoized per fingerprint (P4)** — the tile-safety AST walk (~22 µs) ran every
  CUDA cook; it's a static property, so it's cached on the same key `should_stencil_route` uses.

## Error Reporting Guidelines

TEX uses structured diagnostics (`tex_compiler/diagnostics.py`) to produce clear, helpful error messages. Every error carries an error code, source snippet, optional fuzzy-match suggestions, and a contextual hint.

### Error Code Ranges

| Range | Phase | Examples |
|-------|-------|----------|
| `E1xxx` | Lexer | Unterminated strings, invalid characters, malformed numbers |
| `E2xxx` | Parser | Unexpected tokens, missing semicolons, foreign keywords |
| `E3xxx` | Type checker — names, scope, types & coercions | Undefined variables, duplicate declarations, type mismatches, failed promotions |
| `E4xxx` | Type checker — unrecognized construct (catch-all) | A construct the type checker doesn't recognize |
| `E5xxx` | Type checker — function signatures | Wrong argument count, argument type errors |
| `E6xxx` | Runtime (interpreter) | Loop limit, division by zero, out-of-bounds; `E6050` unknown function, `E6051` a function's runtime failure |
| `W7xxx` | Warnings (reserved range) | Non-fatal advisories |

### Voice and Tone

TEX errors are written in an empathetic, first-person voice. The compiler speaks as a helpful assistant, never as an authority scolding the user.

**Do:**
- Use active voice, present tense: *"I can't find a function named `clampp`."*
- Explain what went wrong, then what to do: *"I expected `;` after this expression. Add a semicolon to end the statement."*
- Use: `"I found..."`, `"I expected..."`, `"This ... isn't supported"`

**Don't:**
- Use blame-laden words: ~~`fatal`~~, ~~`illegal`~~, ~~`invalid`~~, ~~`user error`~~
- Use passive/impersonal phrasing: ~~`"Unknown identifier"`~~ (use `"I can't find a variable named ..."`)
- Omit actionable guidance — always tell the user what to try next

### Adding a New Error

1. **Pick a code** from the appropriate range (e.g. `E3012` for a new scope error). Check existing codes in the codebase to avoid collisions.

2. **Write the message** in empathetic voice:
   ```python
   # Good
   "I can't find a variable named `{name}`."
   # Bad
   "Unknown identifier: {name}"
   ```

3. **Call `make_diagnostic()`** with `code=` and `hint=`:
   ```python
   from .diagnostics import make_diagnostic, suggest_similar

   diag = make_diagnostic(
       code="E3012",
       message=f"I can't find a variable named `{name}`.",
       loc=node.loc,
       source=self.source,
       hint="Check your spelling, or make sure the variable is declared before this line.",
       phase="type_checker",
   )
   ```

4. **Add fuzzy suggestions** when the error involves a name the user may have mistyped:
   ```python
   from .diagnostics import suggest_similar

   candidates = list(self.env.keys())
   diag.suggestions = suggest_similar(name, candidates)
   ```

5. **Thread `source=`** so the diagnostic can render a source snippet with a caret underline. The source string is the full program text; `make_diagnostic` extracts the relevant line automatically via `get_source_line()`.

### Foreign Hint Maps

Three dictionaries in `diagnostics.py` provide contextual hints when users try syntax from other languages:

| Dict | Purpose | Example |
|------|---------|---------|
| `_FOREIGN_FUNCTION_HINTS` | GLSL/HLSL/JS/VEX function names | `"texture2D"` -> tells user to use `sample()` |
| `_FOREIGN_VARIABLE_HINTS` | Shader/JS built-in variable names | `"iResolution"` -> tells user to use `iw`, `ih` |
| `_FOREIGN_KEYWORD_HINTS` | Keywords from other languages | `"let"` -> tells user to use explicit types |

To add a new entry, add a key-value pair to the appropriate dict. The key is the foreign name (as the user would type it), and the value is a short, helpful string explaining the TEX equivalent. Set the value to `None` if the keyword is actually valid in TEX (no hint needed).

```python
# Example: adding a Unity ShaderLab hint
_FOREIGN_FUNCTION_HINTS["UnpackNormal"] = (
    "TEX doesn't have UnpackNormal. Use x * 2.0 - 1.0 to unpack normal maps."
)
```

### Testing Errors

When testing that the compiler produces the right error for bad input:

1. **Catch the right exception type** — `TEXMultiError` for compile-time errors, standard exceptions for runtime errors:
   ```python
   from tex_compiler.diagnostics import TEXMultiError

   try:
       compile_and_run("@OUT = unknownfunc(1.0);", {"A": img})
       r.fail("expected error for unknown function")
   except TEXMultiError as e:
       assert "I can't find" in str(e)
       assert len(e.diagnostics) >= 1
       assert e.diagnostics[0].code.startswith("E")
       r.ok("unknown function error")
   ```

2. **Assert on key phrases, not exact strings.** Error messages may be refined over time. Check for stable fragments like `"I can't find"` or `"I expected"`, not the full sentence.

3. **Verify the error code prefix** matches the expected phase (e.g. `E3` for type checker name errors, `E5` for signature errors).

4. **Check suggestions** when testing fuzzy matching:
   ```python
   assert "clamp" in e.diagnostics[0].suggestions  # typo "clampp" -> "clamp"
   ```

### Bug Reports

If you encounter an error message that is confusing, unhelpful, or missing a hint, please file an issue: https://github.com/xavinitram/TEX/issues

---

## Snippet System

The snippet system lets users browse, insert, save, and manage TEX code snippets via the right-click context menu.

### Architecture

```
Backend (Python)                          Frontend (JavaScript)
────────────────                          ────────────────────
__init__.py                               tex_extension.js
  _EXAMPLE_CATEGORIES dict                  _fetchBuiltinSnippets()
  _load_example_snippets()                    ↓ fetches once, caches
  /tex_wrangle/snippets route               _buildSnippetTree()
        ↓ reads                               ↓ merges with user snippets
  examples/*.tex files                      _createCascadeSubmenu()
                                              ↓ renders nested menus
                                            Save / Manage dialogs
                                              ↓ persists to localStorage
```

### Backend: Snippet API

- **Route**: `GET /tex_wrangle/snippets` (registered in `__init__.py`)
- **Source**: reads all `.tex` files from the `examples/` directory
- **Category mapping**: the `_EXAMPLE_CATEGORIES` dict maps filename stems (e.g. `"auto_levels"`) to display paths (e.g. `"Color/Auto Levels"`). Files not in the dict get auto-categorized under `"Uncategorized/"`.
- **Caching**: `_snippets_cache` is built once on first request and never invalidated (examples are static assets)
- **Response**: JSON object where keys are paths like `"Examples/Color/Auto Levels"` and values are the full `.tex` source code

### Frontend: Cascade Menu

- **Fetch**: `_fetchBuiltinSnippets()` calls the API once and caches in `_builtinSnippetsCache`. On failure, returns `{}` without caching so the next hover retries.
- **Tree building**: `_buildSnippetTree()` splits `/`-separated paths into a nested object tree, then merges built-in and user snippets.
- **Cascade rendering**: `_createCascadeSubmenu()` recursively builds nested DOM menus. A shared per-level `pendingTimeout` prevents hover races when the mouse moves quickly between categories.
- **Cleanup**: `_closeAllSubmenus()` tears down all open cascade levels; the dismiss handler checks both the main menu and all submenus.

### User Snippets

- **Storage**: localStorage key `tex_wrangle_snippets`, JSON object of `{"path/name": "code", ...}`
- **Paths**: use `/` as folder separator (e.g. `"My Snippets/Color/warm tint"`)
- **Save dialog** (`_showSaveSnippetDialog`): modal with name input (with `/` folder hint), 3-line code preview, Enter/Escape/click-outside dismissal
- **Manage dialog** (`_showManageSnippetsDialog`): scrollable list of user snippets with Rename and Delete buttons; live re-renders on changes

### Adding a New Built-in Example

1. Create a `.tex` file in the `examples/` directory
2. Add the filename stem to `_EXAMPLE_CATEGORIES` in `__init__.py` with a `"Category/Display Name"` value
3. The snippet will appear automatically in the cascade menu under `Examples/Category/Display Name`

## API stability tiers (ENG-5)

TEX is embeddable (`tex_api`, `tex_engine`, `tex_cli` all run with ComfyUI absent), so
some of its surface is other people's problem when it moves. This table says which.
Everything in **Tier 1** has a canary test whose whole job is to fail when the shape
changes — the point is not that these can never change, but that changing one is a
decision someone makes on purpose, in a release note, rather than a rename that quietly
breaks a host.

| Tier | Surface | Contract | Pinned by |
|------|---------|----------|-----------|
| **1 — Public** | `tex_api.compile` / `execute` / `Program` field names | Names + shape are stable; additive only | `test_port2_program_shape`, `test_port2_facade` |
| **1 — Public** | `tex_api.TEXCompileError` + `.diagnostics` (ENG-4) | The ONE exception type a host catches for a bad compile | `test_eng4_structured_compile_error` |
| **1 — Public** | `TEXDiagnostic.to_dict()` key set | A de-facto frontend contract since v0.15 | `test_eng5_embedding_canaries` |
| **1 — Public** | `tex_engine.cook` / `prepare` / `run`, `CookResult` fields (ENG-1) | The host-agnostic cook entry point | `test_eng1_engine_cooks_without_the_node` |
| **1 — Public** | The `ui=` HUD payload (`tex_perf` / `tex_probes` keys) | Read by the shipped JS | `test_eng5_embedding_canaries` |
| **1 — Public** | `HostServices` method set (PORT-1) | What a host must implement | `test_port1_host_services`, `test_eng5_embedding_canaries` |
| **1 — Public** | GraphSpec (`_tex_chain`) + `GRAPHSPEC_SCHEMA` (SCHED-1) | Versioned; absent == 1; a newer schema is REFUSED, never guessed | `test_eng5_embedding_canaries` |
| **1 — Public** | The `.textool` manifest + `TEXTOOL_SCHEMA` + `promoted_params` key set (TOOL-1) | Versioned; a newer `manifest_schema` is REFUSED; validated before any compile | `test_tool_manifest_keys`, `test_tool_schema_rejects` |
| **1 — Public** | Egress profiles `comfy` / `engine` (ENG-3) | `comfy` is byte-identical, forever | `test_eng3_comfy_profile_canary` |
| **1 — Public** | `tex_marshalling.BufferMeta` fields + `merge_buffer_meta` + `CookResult.out_meta` (DATA-1) | The colour/alpha/frame tag vocabularies + the merge-to-`unknown` policy; a value channel, never keyed | `test_v028_phase1` |
| **1 — Public** | `tex_io.BufferDesc` + `tex_io.exr` / `tex_io.png` (DATA-2) | Storage dtypes; EXR is the OpenEXR format (NONE/ZIP scanline, HALF/FLOAT) — the file bytes are the standard's contract | `test_v028_phase1` |
| **1 — Public** | `tex_session.EngineSession` / `default_session` (DATA-4) | The session handle; phase-1 `.cache`/`.registry`/`.host` view the module singletons | `test_v028_phase1` |
| **2 — Semi** | `a@name` ARRAY wire + array outputs (DATA-3) | Engine-profile only; `a` is now a RESERVED binding prefix; comfy rejects array outputs (E3203 + egress guard) | `test_v028_phase1` |
| **2 — Semi** | TEX the language | Additive; new builtin/function names are RESERVED, so adding one is a minor breaking change — note it in the CHANGELOG (v0.22 reserved `frame`/`fps`/`time`) | the compat corpus (LANG-3, planned) |
| **2 — Semi** | Error codes (E1xxx–E6xxx) | Codes are stable; message TEXT is not | `test_c3ux_error_codes_resolve` |
| **3 — Internal** | Everything else — `tex_runtime.*`, `tex_compiler.*`, `tex_fusion` internals, `tex_engine._*` | No promise. Import at your own risk | — |

**Fingerprints are NOT stable — never persist one.** `TEXCache.fingerprint` /
`fused_fingerprint` are value-independent keys for TEX's own caches, deliberately derived
from a mono-hash over the compiler sources (and the codegen-reuse env flag). They are
*designed* to change when TEX changes: that is what invalidates stale compiled artifacts
across an upgrade. A host that writes one to disk and expects a hit after upgrading TEX
has stored a number whose whole purpose is to become different. Key your own storage on
your own identity (the roadmap's CACHE-1 lineage key is the durable one); TEX's cache
directory is TEX's business, and `TEX_CACHE_DIR` tells it where to live.

## Concurrency & thread-safety (ENG-9)

The written contract for TEX's module-level runtime state. It exists because a
branch-parallel executor (GRAPH-2) will cook on more than one thread, and finding out
where that races during the engine build is the expensive way.

**Today's model is single-cook-thread.** Under ComfyUI's one-cook-at-a-time executor
(and the engine's own default), exactly one thread cooks; the compile worker pool is
`max_workers=1`. Everything below is written so the single-threaded fast path stays
**lock-free** — locks guard *inserts*, never reads.

**The interpreter is per-thread, not a process singleton (ENG-9).** `tex_engine._get_interpreter()`
and `compiled._get_plain_interp()` return a `threading.local` instance. The interpreter
carries per-instance mutable execution state (the scope stack, `_literal_cache`,
`_builtins_lru`); a *shared* instance corrupts under concurrent cooks — a two-thread cook
mixes up programs (proven, then pinned by `test_eng9_two_thread_cpu_cook`). One cook thread
still gets exactly one instance, so ComfyUI is unchanged. A small locked registry of every
instance created lets `free_tensor_caches` sweep them all (its clear runs on whatever thread
called it, which thread-local storage would otherwise miss).

**Cache classification** (the register lives in AGENTS.md / ARCHITECTURE.md; this is the
concurrency lens on it):

- **Immutable-after-insert (IAI)** — value never mutated once stored; only the LRU
  container is managed. `_FINGERPRINT_MEMO`, `_tile_safe_memo`, `_peak_static_memo`,
  `tex_lazy._memo`, `tex_roi._walk_memo` (ROI-2, keyed on `(code-hash, param-key)` exactly
  like `tex_lazy._memo`; shared by `binding_footprints`/`roi_plan`), `_stencil_route_memo`,
  `_route_memo`, `_ENV_TENSOR_CACHE`, `_gauss_kernel_cache`, the worley-offset caches,
  `_AUTO_DECISION`, `_FUSED_MEMO`. A
  concurrent insert of the same key recomputes the same value (harmless); a racy eviction
  at worst *over-evicts* (a later recompute). Safe for concurrent CPU cooks under CPython's
  GIL (verified under eviction pressure). A future non-GIL/free-threaded runtime should
  lock the insert+evict sequence — never the `get` hit.
- **Device-keyed** — the device/index is part of the key, so different devices never
  collide: `_grid_buf`, `_total_mem_cache`, `_last_trim_px`, `_backend_status`,
  `_compiled_cache`, `_graph_cache` (keys include `dev.index`), the noise tiered caches,
  `xfer._MODEL`.
- **Per-worker / lifecycle-coupled** — the per-thread interpreters above, and the
  `_COMPILE_POOL` background-compile futures (`_bg_futures`) which are owned by the single
  compile worker.
- **Mutable-after-insert (MUT) — still single-cook-thread only.** The compile/graph-tier
  state machines mutate their entries in place and are *not* yet safe for a parallel
  executor: `autotier._STATE`, `graphed._graph_cache`/`_blacklist`/`_CAPTURING`/`_graph_bytes`,
  `compiled._verify_state`, `compiled._deferred_ev`. A pure-interpreter CPU cook never
  touches these; a branch-parallel executor that drives the compile tiers must serialize
  or shard them (GRAPH-2's work). The single genuine multi-writer today —
  `_compiled_cache`, written by both the foreground and background compile — is race-free
  *only* because both run on the one `max_workers=1` worker.

The single existing data lock (`noise._TieredCache._lock`) predates ENG-9 and stays.

**The engine session (DATA-4).** `tex_session.EngineSession` is a *handle* over the state
classified above — the program cache, the CACHE-5 governor, host services, and the per-thread
interpreter pool — not a new owner of it. Phase 1 has exactly one session (the process default);
its `.cache` / `.registry` / `.host` / `.interpreter` **are** those module singletons (views),
so ComfyUI is byte-identical and this section's classification is unchanged. The session adds no
lock: `session.reset()` runs the existing `free_tensor_caches()` sweep, which is single-cook-thread
safe and, exactly like calling it directly today, must **not** run concurrently with a live cook
(it clears the IAI/device-keyed tensor caches other threads may be reading). The `.interpreter`
property returns *this thread's* instance (the ENG-9 per-thread pool), never a shared one. What a
session does NOT yet provide is isolation: a second `EngineSession` is still a view of the same
globals (`isolated == False`). Threading a session through `engine.cook(session=…)` so an isolated
session owns its own caches is **phase 2** (it needs the ENG-1 cook-signature change) and inherits
the MUT-cache sharding this section already flags for a parallel executor — the session handle
*names* that boundary, it does not move it. The `tests/test_v028_phase1` soak lane (thousands of
cooks across tiers + `reset()` cycles, flat RSS/VRAM watermarks) guards the single-thread lifecycle
against the slow leaks a days-long compositor process would otherwise hide.

## Rejected design decisions (don't re-propose)

Settled calls, kept here so they're not re-derived:
- **Raising `MAX_OUTPUTS` so a fused chain can carry more CACHE-7 taps** (v0.32) — rejected.
  It is a *host output-slot count* (`tex_node.py` binds its node outputs to it), so raising it
  changes the node's published surface to buy an engine convenience. `materialize` batches the
  harvest deepest-first and retries on what actually came back instead. Reopen only if a chain
  can genuinely need more than 7 checkpoints — which today it cannot, since fusion caps a
  region at 16 stages.
- **An engine-side content check on CACHE-6/7's `upstream` source key** (v0.32) — rejected on
  cost. The gate is an ARITY check; a host passing a content-*insensitive* key still gets a
  stale boundary (measured maxdiff 0.91 on a swapped source). Verifying it engine-side means
  hashing every source tensor on every cook, which is the cost caching exists to avoid.
  GRAPH-1's version counters are the intended answer — a host that stamps a key per produced
  value gets this for free.
- **GOV-1 adaptive tier switching** (v0.32) — deferred, not built. Report 39 asks the governor
  to shift emphasis automatically when a session changes shape. It collides with S-5 ("never
  silently auto-tune a box we haven't measured"), and there is no session-shape classifier
  anywhere in the engine to drive it. What shipped is the honest half: named, repo-committed,
  `tex doctor`-reportable presets. Reopen when PRED-1's reason stream has been calibrated
  against a real session — the mechanism it would need, not the policy.
- **A lossless entropy codec on any cache tier** (v0.33 CACHE-8) — rejected on measurement, and
  the measurement is committed (`benchmarks/cache_capacity_bench.py`). At 4K, zlib-1 costs
  6710 ms to encode and **935 ms to decode** a frame that can simply be written to disk in
  335 ms and read back in 60 ms; bz2 and lzma are an order worse; LZ4/zstd/blosc do not exist in
  a torch-only package and a new dependency is out of scope (the same rule that bans numpy).
  Decode is paid on every cache hit, so this is not close. Break-even is derived rather than
  asserted: compression only pays below **~200 MB/s** of storage bandwidth. What buys capacity
  is *width* (PREC-1's fp16, exactly 2×) and *residency*, both of which shipped. A test greps
  `tex_results.py` for codec names so a future addition has to argue with the number.
  Reopen only for a storage medium under that bandwidth, and only for the disk tier.
- **`_learn_spilled` can re-walk the spill dir once per miss while spilling continues**
  (v0.33.2 H4) — DEFERRED, and it is a REGRESSION this release knowingly ships. Before H4 the
  scan always ended in a definite set, so it ran once; now a scan that raced a spill correctly
  leaves membership UNKNOWN, and `_restore` calls `_learn_spilled()` on every miss where
  `_spilled is None`. While membership is unknown `_spill` records nothing into it, so under a
  sustained spilling workload (the CACHE-7 shape: a worker spilling while the main thread
  probes) every scan can race and the walk repeats — O(entries) syscalls per miss on the exact
  path the `_spilled` set exists to keep syscall-free. It self-corrects the moment spilling
  pauses. Shipped anyway because the alternative is the orphaned-frame bug H4 closes, and a slow
  cache beats an unserveable frame. *Gate:* have `_spill` record into a small `_spilled_since`
  set even when membership is unknown, so `_learn_spilled` can return `found | _spilled_since`
  and be definite — which removes the race from all THREE unlocked walks rather than papering
  over one. That touches `_spill`'s index block, which is the most-audited code in the file, so
  it wants its own change and its own repro rather than a patch-release edit.
- **`patch_region` propagates the packing LICENCE, not the tier tag** (v0.33.2 H3) — DEFERRED.
  The ratchet is gated on `src.orig_dtype is not None`, which is right for the pack decision but
  means a patch of a preview-tagged-but-UNPACKED base (a MASK, or an out-of-fp16-range frame) is
  stored tagged final. Nothing reads the tag today except `choose_storage`, so there is no live
  defect — but A4's own justification names a second consumer ("requalify-on-idle has to
  enumerate preview entries"), and that pass would under-count. *Gate:* keep `propagate_quality`
  unconditional and instead pass `storage="fp32"` to the nested `put` when the base was not
  reduced — `choose_storage`'s pin arm means exactly "store as cooked, at any tier". Needs a
  decision about clashing with a caller-supplied `storage=`, which is design, not a patch.
- **Three unlocked directory walks hand-roll the same capture-walk-compare** (v0.33.2) —
  DEFERRED. `_learn_spilled`, `reindex_disk` and `clear` each spell "capture the witness under
  the lock, walk unlocked, compare under the lock", and `clear` additionally maintains
  `_purge_depth` by hand. *Gate:* one `_unlocked_walk()` context manager owning both the witness
  and the depth, which would make the depth exception-safe BY CONSTRUCTION rather than by a
  `finally` that has to be remembered. Three audited call sites — worth doing, not worth doing
  the week of a tag.
- **The audit tests hand-roll two gadgets three times each** (v0.33.2) — DEFERRED. The
  racing-`scandir` fixture appears in `test_v0331_audit.py` twice and `test_v0332_audit.py` once;
  the "two Events plus a trip flag" parked-subclass appears three times. *Gate:* a
  `spill_during_scan(cache, ...)` context manager and a `park(obj, method)` helper in
  `tests/helpers.py`. Test-only churn across two audit files; not the week of a tag.
- **Frames spilled by v0.33.1 restore UNTAGGED** (v0.33.2 A4) — ACCEPTED, not fixable. `quality`
  is a format-2 field, so `rec.get("quality")` is `None` for every `.frame` a v0.33.1 process
  wrote, and `env_epoch()` is unchanged by the upgrade so those records are still served. The
  viral-quality rule therefore does not fire on a pre-upgrade preview frame. There is nothing to
  recover — the tag was never written — and inventing one would be worse than admitting it. The
  exposure is bounded and self-healing: it lasts until each key is next cooked and re-spilled,
  and it can only under-protect a frame the host already asked to be stored at preview fidelity.
  A host that cares can `clear(disk=True)` once after upgrading.
- **`evict_bytes` can over-report to the CACHE-5 governor when the tier disarms mid-flight**
  (v0.33.2 A5) — DEFERRED, low. `_queue_demotions` charges bytes at QUEUE time on the stated
  invariant "the drain is unconditional from here", and A5 makes the drain conditional from both
  ends (`set_vram_budget(None)` empties the queue; the commit re-check drops one already in
  flight). So a governor that disarms residency in the same window as an `arbitrate()` can be
  told bytes were freed that were not. It self-corrects on the next call — `governed_bytes()` is
  recomputed from `_bytes_by_dev`, which stays consistent throughout (verified by recount) — so
  the error is one arbitration round, not a permanent skew. *Gate:* charge at COMMIT time
  instead, which means `evict_bytes` can no longer answer synchronously and the CACHE-5 hook
  contract changes shape. That is a v0.34 conversation, not a patch.
- **A lock-depth early-out in `ResultCache._promote`** (v0.33.2 A5(d)) — WITHDRAWN after being
  written, because it traded a latency problem for a correctness one. The idea was sound in
  isolation: the drains refuse to run a full-frame copy while a composite holds `_lock`, and
  `_promote` is an 11.1 ms H2D at 4K that `patch_region` reaches at depth 1. What it missed is
  that `_promote` is not a drain. A drain DEFERS — the queue survives and `patch_region` runs it
  on release — whereas this DEGRADED, handing back the host copy with no `_pending_promotes` to
  make good on it. Measured: base demoted (`home=cuda:0`) -> `patch_region` -> result
  `device=cpu`, `home=cpu`, `promotions=0`. Nothing raises, because `frame[...] = patch` accepts
  a CUDA source into a CPU destination, so the patched frame just leaves the residency ladder
  permanently and every stage downstream inherits a CPU home — the "one-way trip to the CPU"
  `_Entry.home` exists to prevent. *Gate to reopen:* a `_pending_promotes` queue drained like
  the other two, PLUS home propagation through `_patch_region_locked`'s nested `put` — the
  second is what the first alone does not fix, since the result is composed while the base is
  still on the host. Both belong with v0.34's async-write work, not in a patch. The invariant
  the withdrawal restores is pinned
  (`test_v0332_a5_promote_keeps_a_patched_frame_on_its_home_device`).
- **Pruning `_spill_seq` / `_spill_locks`** (v0.33.2 A1) — DEFERRED, quantified rather than
  waved at. Both dicts gain an entry per distinct key ever spilled and lose it never: ~240 B
  per key (a `threading.Lock`, its OS mutex block, the dict entry, and the retained 64-hex key
  string). A 50-node comp scrubbed over 200 frames is ~10k keys / ~2.4 MB; the v0.40 headless
  shape (10k frames x 12 stages) is ~120k keys / ~29 MB and 120k live lock objects. Small
  against a 512 MB-2 GB frame budget, and the tier is opt-in, so it does not gate this release.
  **Do not "fix" it by clearing `_spill_locks` in `clear()`** — the comment there explains why
  that re-opens A1. *Gate:* fold both into one refcounted `_writers[key]` entry popped when the
  last in-flight spill of that key leaves, which bounds the map by CONCURRENT spills instead of
  by keys ever seen.
- **`examples/host_demo.py::RoiComp` still hand-rolls the window composition** (v0.33.2, P0-4a)
  — DEFERRED, and worth stating because it makes a CHANGELOG line narrower than it sounds. The
  `declined=` migration reached `tex_roi.chain_windows`, the bench host and the tests; it did
  NOT reach the shipped example, which carries its own `_needed_windows` backward walk, its own
  `_WHOLE_FRAME` sentinel and its own halo inversion, and tracks neither `valid` nor `declined`.
  So the artifact a host is most likely to copy still demonstrates the pre-P0-4a pattern whose
  error this release quotes at 2.10e-01. *Gate:* `RoiComp` is load-bearing for PM-2/PM-6 and is
  imported and asserted on by `test_v028_phase1` and `test_v030_phase1`, so rewriting its cook
  loop means re-running those measurements — a v0.34 job, not a patch-release one.
- **Hoisting `patch_region`'s base fetch out of the atomicity lock** (v0.33.2) — DEFERRED.
  `_patch_region_locked` calls `self.get(base_key)` while `patch_region` holds `_lock`, and on
  a RAM miss that reaches `_restore` — a pickle load plus an H2D, i.e. 327-496 ms of disk I/O
  inside the critical section the lock rule exists to keep I/O out of. The lock is not
  removable (an earlier draft measured 200/200 lost updates without it), so the fix is to
  resolve the base — fetch, restore, promote — BEFORE acquiring the lock and re-verify entry
  identity under it. *Gate:* that re-verification needs its own design and its own lost-update
  measurement; it also subsumes the `_promote` item above. Recorded here so the gap is a known
  deferral rather than something rediscovered by the next audit.
- **Frequency- or playhead-weighted residency victims** (v0.33 CACHE-8, v0.33.2) — DEFERRED,
  recorded here because two APIs already have the slot and neither uses it, which reads as an
  oversight unless it is written down. `_enforce_residency` picks demotion victims by **LRU**,
  and `evict_bytes(..., playhead=)` ACCEPTS a playhead and ignores it. The report asks for an
  access-FREQUENCY policy; the argument for shipping recency anyway is that a frame cache's
  access pattern is a playhead — a scrub touches near-frames both most recently and most often,
  so the two orderings largely coincide, and LRU is the one that needs no per-entry counter.
  *Gate to reopen:* a measured workload where they diverge — a comp with a hot reference frame
  far from the playhead is the obvious shape. `stats()` already reports demotion/promotion
  counts so the case can be made with a number rather than an opinion, and a frequency-weighted
  choice is a change to `_queue_demotions` alone (it is the single victim-choosing walk, which
  is why that consolidation was worth doing before the policy question was settled).
- **Asynchronous D2H for the CACHE-8 demote path** (v0.33 XPU-2) — rejected on lifetime, not
  speed. Async egress requires a page-locked destination, and a demoted frame's host buffer is
  *retained* for as long as the entry lives; pinned pages are unswappable and torch's caching
  host allocator holds freed blocks for the process lifetime, so retaining them is a slow leak.
  Copying into pinned and cloning to pageable to release the lock costs a second full host
  memcpy of the frame — and the asynchrony it would buy back is MEASURED at 1.01-1.06x over
  a pinned blocking copy, i.e. noise (`benchmarks/`-adjacent measurement, v0.33 review).
  `egress(retained=True)` states the distinction rather than hiding it (spelled `staging=`
  in the first draft; the rename outlived this line for one release). The transient spill
  buffer *is* async, which is the case the mode exists to separate.
- **Disk→GPU direct paths (GPUDirect / cuFile)** (v0.33) — investigated, measured, **no-go**.
  No first-party torch API; Linux-only in practice (this project's primary platform is Windows);
  and it removes only the H2D half of a 46.8 ms path that CACHE-8's residency tier removes
  *entirely*. Revisit condition: a torch-native GDS API **and** a working set that genuinely
  exceeds host RAM. See `docs/compressed-cache-tiers.md` §4.
- **Inferring a colour-vs-data plane role from pixels** (v0.33 PREC-1) — rejected on S-5. The
  roadmap's shape is "colour planes half / data planes fp32", and TEX cannot tell them apart:
  DATA-1's vocabulary has no role field and named planes are DATA-6 (v0.35). Sniffing content to
  guess is exactly the silent auto-tuning the discipline forbids. Shipped instead: an explicit
  `storage="fp32"` pin, and an *exact* value-range gate — both failing toward fp32. When DATA-6
  lands, `choose_storage` grows a role arm and no caller changes.
- **Doc 41 §2.4's three CACHE-7/9 hardening items** (v0.33) — DEFERRED, not built, recorded the
  same day per §10.6. Each has a measured trigger already on record, which is what makes the
  deferral checkable rather than open-ended:
  (a) *all-dirty routing as an engine answer* — the 0.21×/0.04× cliff is still fenced by prose
  ("a host must route an all-dirty recook whole-frame") instead of by a `not-worth-it` return
  from the serviceability API. **Reopen gate:** a host that actually drives region recooks at
  scale, i.e. the compositor; PROF-1 already prices both sides, so this is wiring, not research.
  (b) *fast-settle for placement* — 147 cooks to settle (3 warmup + 1-in-16 to `MIN_SAMPLES=12`)
  is the recorded adoption tax. **Reopen gate:** any host reporting that checkpoints never
  appear; the burst mode is ~10 lines in `profile.py` and cuts it to ~12 cooks.
  (c) *per-device checkpoint thresholds* — the 100 ms default places NOTHING on CUDA on this box
  (12-stage 2048² ≈ 43 ms) while the materialization floor is device-dependent by 18×. **Reopen
  gate:** the first calibration run on real comps, which doc 39 §4 already marks host-gated.
- **Doc 41 §2.5's ROTO-lang decision doc** (v0.33) — DEFERRED, and this one is overdue by three
  releases (doc 40 §5 dates it "written v0.32–33"). It is document-only and it SHAPES DATA-5
  (v0.37), so the cost of leaving it is that DATA-5 starts without knowing whether fusable
  procedural masks beat host-rasterized MASK planes. **Reopen gate:** before DATA-5's design doc
  opens, not after. The recipe is written down (doc 41 §2.5): a throwaway `spline_mask`
  prototype interpreter-side, benchmarked against rasterized masks wired as bindings at 1080/4K
  on both devices, with the go/no-go recorded either way — a "not shipped" verdict is a
  completed item under doc 40 §4.6's own rule.
- **Doc 41 §B5b/c** (v0.33.1) — two decisions doc 41 assigned that remain unrecorded: the PROF-1
  `snapshot()` / persisted-placement cross-launch question (its stated gate, CACHE-8's tiers, has
  now shipped, so it is unblocked), and XPU-2 × CUDA-graph capture — either add the
  `is_current_stream_capturing()` guard (precedent: `noise.py`) or write the non-interaction
  argument into `docs/async-egress.md`. **Note for whoever takes the second:** `egress`'s
  `_blocking` fallback currently sits OUTSIDE the `try`, so it would raise uncaught mid-capture.
- **A cross-node include/import system** — rejected on ethos grounds; self-containment
  ("five lines of self-contained plaintext") is a deliberate shareability feature.
- **An extra fusion wire** — the frontend collapses a chain into the terminal node
  transparently instead.
- **Whole-pipeline fp16 & bf16 IMAGE** — accuracy (bf16 err > the 8-bit quantum).
- **Full ACES/OCIO color management** — scope creep (sRGB↔linear + OKLab only).
- **Merging the interpreter and codegen stdlib implementations** — the duplication is
  the bit-exactness safety margin.
- **Default CPU codegen routing (PF-4)** — measured to regress the dominant color-grade
  shape; the interpreter's own optimizations closed codegen's lead.

Recorded by the compositor-engine roadmap (`docs/roadmap.md` §7 is the provenance):
- **Split-frame dual-device cooking (ENG-10)** — cooking one frame half-on-CPU,
  half-on-GPU. The CPU↔GPU envelope (invariant #9) is characterized, not bit-parity
  (up to 6.1e-2 on scatter), so a row-split would put that divergence on a visible
  mid-frame seam; the output would depend on the split ratio, breaking the pinned
  run-to-run determinism; and a laptop CPU adds only single-digit % of GPU pointwise
  throughput while contending for the same memory bus. "Both CPU and GPU" is honestly
  satisfied by branch-parallel execution + copy/compute overlap, not frame-splitting.
- **Mid-sequence device-placement migration (SCHED-2's temporal twin of ENG-10)** —
  moving a node between devices between frames of one render range changes pixels by
  up to the same envelope (temporal popping). Placement is part of the result key and
  freezes per render range; re-planning happens only at interactive/idle boundaries.
- **A second, "fast" kernel language or dialect** — the Copernicus VEX(CPU)/OpenCL(GPU)
  and Fusion Lua/DCTL trap: an easy-but-slow language and a fast-but-hostile one. TEX's
  single source runs interpreter-on-CPU and codegen-on-GPU; device placement is a
  scheduler concern, never an authoring one.
- **A scanline- or tile-granular execution core** — Blender deleted its tile compositor
  (full-frame rewrite: bounded memory, "often several times faster"); GEGL's per-tile
  dual CPU/GPU residency kept OpenCL broken for 15 years. TEX is full-frame planar;
  stripes/halos are a memory-streaming tactic under that model, not a scheduling model.
- **A recursive pull executor with per-node locking** — the Natron post-mortem (its own
  developers cite engine race conditions as what killed it). Plan lazily (demand, ROI,
  frame ranges), then compile the resolved cook into a static push plan.

Recorded by v0.25 "Remember frames" (`docs/results-caching.md` is the provenance):
- **Enabling the frame cache (CACHE-2) under the ComfyUI node** — the ComfyUI host already
  caches every node's output, so a second TEX-owned frame cache under it only doubles the
  memory for no reuse it doesn't already have. `ResultCache` is host-instantiated and armed
  by an engine host; the flip to on-under-ComfyUI needs a host that owns its downstream
  consumers (ownership guaranteed, not hoped — ENG-12) *and* a demand signal a node graph
  can't answer more cheaply, i.e. GRAPH-1's version counters.
- **Persisting the runtime CUDA-graph capture blacklist (CACHE-3)** — the *capturability*
  verdict (a deterministic function of the program AST + arch) IS persisted; the runtime
  *capture-failed* blacklist is not. Its key is a shape/driver-specific opaque capture
  signature, and a transient failure (a one-off memory-pressure/OOM during capture) must not
  harden into a permanent cross-restart "never capture this" verdict — the capturability memo
  already keeps the expensive path away from the programs that genuinely can't capture.
- **Content-hashing tensors for cache REUSE (CACHE-1)** — the sampling hash
  (`tensor_fingerprint`) stays only for ComfyUI's `IS_CHANGED` cache-*busting* (a false
  "changed" just recooks). A result cache keys on a *lineage* key (the cook that produced a
  frame), never re-sampling its pixels: the sampling hash has an admitted collision class that
  is harmless for busting and a silent stale-serve for reuse.
- **A worker POOL for the cook queue (SCHED-4, v0.31)** — one worker thread, deliberately.
  The tree-walking interpreter is Python-heavy and holds the GIL between kernels, so a
  concurrent Tier-B cook does not overlap Tier A; it interleaves two Python interpreters onto
  one core and slows Tier A down, which is the one outcome the item exists to prevent. A second
  worker would buy overlap only for programs that are almost entirely kernel time, and those
  are exactly the programs whose Tier-A latency is already GPU-bound. Reopens with GRAPH-2's
  parallel region executor, which needs the ENG-9 MUT-cache sharding anyway.
- **Resumable cooks (SCHED-4)** — a preempted job re-cooks from the top. Resuming needs a
  serializable interpreter state, which the tree-walker does not have and codegen actively does
  not (its locals are Python frame state). The cost is bounded by the yield granularity (one
  statement / one strip) and mitigated by CACHE-2 on the retry. Reopens only if PM-7 ever
  measures preemption cost dominated by re-cook rather than by yield latency — at v0.31 it is
  0.130 ms (CUDA) / 0.414 ms (CPU) to first kernel.
- **`CookQueue` arming the in-engine profiler (PROF-1, v0.31)** — it does not call `enable()`.
  The queue already brackets every job, so it feeds `profile.record()` from that bracket: no
  sampling gate needed, and it cannot put a CUDA sync into a cook the queue does not own. The
  in-engine sampler exists for the per-STAGE breakdown and for hosts cooking outside the queue,
  and stays disarmed until a host asks (invariant #7 applies to the profiler itself).
- **Persisting PROF-1's cost table across processes (v0.31)** — autotier persists because
  re-deriving a compile verdict costs a background compile; a PROF-1 estimate costs the few
  cooks a host was going to run anyway. The `snapshot()` seam is there for CACHE-7 if
  cross-launch placement stability turns out to matter — that is its design doc's call.
- **A journal for `autotier.json` (ENG-13, v0.31)** — `warm_state` got one because its writes
  were throttled; `autotier._persist` already runs on EVERY terminal verdict, so its loss
  window is the single verdict being written, which is already inside ENG-13's bound. It gained
  the shared durable write and nothing else.
- **A per-pixel or per-batch-element `t` for `fetch_time`/`sample_time` (DATA-7, v0.34)** —
  refused as E7003. A per-PIXEL time is a retime map and is unbounded I/O from one statement:
  as many source frames as there are distinct values, at cook resolution. There is no honest
  cap to pick, and the failure mode of guessing one (silently servicing 8 of 4096 requested
  frames) is wrong pixels. A per-BATCH-ELEMENT time (`time + fi/fps` over a B=100 batch) is
  the bounded half — at most B frames — and is the natural next step; it needs the pool to
  admit B frames at once and a scatter assembly over the batch axis. Reopens when a host
  actually cooks playback as one batch rather than one frame per playhead, which is not the
  shape the CookQueue and `time_context` have today.
- **Deriving the source read-set for lineage keys automatically (DATA-7, v0.34)** — a cook
  that read `plate@v3` and one that read `plate@v4` mint the same `lineage_key`, so a host
  must stamp `flags=tex_provider.source_flags(...)` itself. The structural fix needs the
  read-set BEFORE the cook and it is only known after: a source key can be a `$param`, a
  string expression, or a loop variable. An AST derivation would close the common case and
  silently miss the rest, which is worse than a stated obligation with a helper. Reopens if
  the compiler ever grows a "constant string arguments to fn X" analysis for another reason
  (COLOR-1's space names are the likely one, v0.36).
- **Per-device residency for the DATA-7 media pool (v0.34)** — the pool caches whatever
  device the provider returned, and `_provider_read` moves the frame to the cook's device on
  every call. A CPU provider feeding a CUDA cook therefore pays one H2D per CALL rather than
  per pixel (the pool still amortizes the decode). Caching a second per-device copy doubles
  the pool's bytes to serve a win only a mismatched host sees, and a host that wants device
  residency can return device tensors today. Reopens on a measurement from a host that cannot.
- **A disk tier for the DATA-7 media pool (v0.34)** — deliberately absent, not missing.
  CACHE-8's ladder exists because a cooked frame is expensive and has nowhere else to live; a
  source frame's disk tier is *the source file*, which the host already has and can already
  decode. Spilling a decoded copy beside it buys a faster decode at the price of a second copy
  of the user's media on their disk. Reopens only for a source whose decode is so expensive
  that re-decoding beats reading a cached copy — a measurement, not an assumption.
- **A promise -> jobs index in the cook queue (IO-1, v0.34)** — `_on_input_landed` scans the
  three class deques instead. The waiting set is bounded by the queue's own depth and a
  landing is a once-per-frame event, so an index would be a second structure to keep
  consistent with the deques — and the deques are already the single home that lets shed,
  cancel, close and snapshot see waiting jobs without knowing they exist. Reopens if a host
  ever parks hundreds of jobs on promises, which the `max_pending` shed policy currently
  prevents.
- **A host-side readiness gate instead of `Promise` (IO-1, v0.34)** — REFUSED, not deferred,
  and the argument is worth keeping: a gate object needs no engine change, and it does not
  solve the actual defect. A host gating `submit()` behind its own check still eventually
  calls `cook(bindings)`, and the day someone passes an unlanded object `infer_binding_type`
  types it FLOAT and compiles a program that does not match the pixels. The gate moves the
  hazard; the declaration removes it.
- **A provider-polled cancel token mid-read (IO-1, v0.34)** — cancellation is defined as
  drop-on-landing: a read already inside the provider cannot be stopped from outside, and
  what TEX guarantees is that its result is never installed. A provider MAY accept the token
  and poll it; one that ignores it is correct, just slower. Making it mandatory would put a
  requirement on every host decoder to buy back time only a slow network source loses.
- **The FUS-cap 12.9x peak-memory number now has committed instrumentation (v0.34.1)** —
  `benchmarks/fus_cap_bench.py` records `torch.cuda.max_memory_allocated` per route and saves
  it, so `docs/fusion-cap-decision.md`'s headline is reproducible rather than a number in prose.
  The decision itself is unchanged: the cap stays at 16, and the reopen gate is a backend that
  fuses at the kernel level instead of materializing per-stage intermediates.
- **`frames_are_owned` is opt-in, and the default costs a copy (v0.34.1 P0-E)** — the media
  pool copies every frame at its boundary because ownership is a promise about the FUTURE, not
  a property readable off a tensor: a provider decoding into a reusable buffer hands back a
  contiguous tensor that owns its storage and then overwrites it. A host that genuinely yields
  a fresh tensor per call declares `frames_are_owned = True` and pays nothing. Measured ~17 ms
  per 4K frame, reported in `stats()["copy_ms"]`. Reopens if a host reports the copy dominating
  a real decode, which would mean the decode is unusually cheap rather than the copy expensive.
- **Two v0.34.1 mutation rows retired with reasons, not left surviving** — the provider
  read-ORDER row (the fix stopped depending on order; both reads are under one lock, so there
  is no order left to mutate) and the wake-path `shed_requested` row (every setter of that flag
  also removes the job from its deque under the same lock, so the branch is unreachable today;
  it stays in the source as documented belt-and-braces because the flag and the removal are set
  by different call sites). Reasons are recorded in `tests/mutation_check.py` beside the rows.
- **The mutation harness was blind to v0.34 (v0.34.1)** — its RUNNER imported a hardcoded test
  module list ending at v0.33, so any v0.34 row would have reported SURVIVED with "0 failing
  rows" whatever the guard did. That, not simple omission, is why v0.34 shipped with none. The
  list now reaches v0.34.1, and a new release's modules must be added to it or its rows are
  decorative — the same trap `test_v0331_audit`'s A5 row fell into a different way.

