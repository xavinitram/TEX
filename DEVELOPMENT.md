# TEX Wrangle — Developer Guide

This document covers the internals of TEX Wrangle for developers who want to understand, extend, or contribute to the project.

## Architecture

```
TEX_Wrangle/
  __init__.py              # ComfyUI entry point, version
  tex_node.py              # ComfyUI node class (device/compile_mode params, UI integration)
  tex_marshalling.py       # Input marshalling, type inference, output preparation
  tex_cache.py             # Two-tier compilation cache (memory LRU + disk pickle)
  tex_compiler/
    lexer.py               # Tokenizer
    parser.py              # Recursive-descent parser -> AST
    ast_nodes.py           # AST node definitions (with __slots__)
    type_checker.py        # Static type analysis
    optimizer.py           # Constant folding + algebraic simplification
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

**Optimizer** (`tex_compiler/optimizer.py`): Constant folding and algebraic simplification pass. Reduces expressions like `x * 1.0` -> `x` and pre-evaluates constant sub-expressions.

**Interpreter** (`tex_runtime/interpreter.py`): Tree-walking evaluator that executes the AST using PyTorch tensor operations. Reads types from `type_map` to guide evaluation (e.g. choosing `torch.where` for if/else). Produces output tensors/strings for all assigned `@name` bindings. Used as the default execution path.

**Codegen** (`tex_runtime/codegen.py`): Compiles the AST into a Python function string, then `exec()`s it into a callable. Eliminates per-node dispatch overhead. All env variables are pre-registered as Python locals (`_lv_{name}`) to avoid dict lookups and produce cleaner FX graphs for TorchInductor. Falls back to the interpreter for unsupported patterns (string operations). Includes **stencil specialization**: nested for-loops matching spatial filter patterns are detected and replaced with bulk PyTorch ops (`avg_pool2d` for box blur, `max_pool2d` for min/max reduction, `Tensor.unfold` for median/rank filters, depthwise `conv2d` for weighted stencils). Also detects inline (non-loop) stencil patterns from hand-unrolled fetch sequences.

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

**Cache version** (`_CACHE_VERSION`): Bumped when AST structure or type checker changes would make existing `.pkl` files invalid. Causes graceful cache miss, not crash.

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

## How-To Guides

### Adding a New Stdlib Function

**Example: adding `saturate(x)` that clamps to [0, 1].**

1. **`tex_runtime/stdlib.py`** -- implement the function (or `tex_runtime/noise.py` for noise functions):
```python
@staticmethod
def fn_saturate(x):
    """Clamp value to [0, 1] range."""
    return torch.clamp(_to_tensor(x), 0.0, 1.0)
```
Register it in `get_functions()`:
```python
"saturate": TEXStdlib.fn_saturate,
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

4. **`js/tex_extension.js`** -- add to `TEX_BUILTINS` set for syntax highlighting:
```javascript
"saturate",  // in the TEX_BUILTINS Set
```
Update `TEX_HELP_HTML` to document it in the help popup.

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

# Full suite via pytest (77 test functions, ~1215 sub-tests)
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
# 4-scenario benchmark (compile off/on × cold/warm)
python benchmarks/four_scenario_bench.py

# Legacy synthetic benchmarks
python benchmarks/run_benchmarks.py
```

## Error Reporting Guidelines

TEX uses structured diagnostics (`tex_compiler/diagnostics.py`) to produce clear, helpful error messages. Every error carries an error code, source snippet, optional fuzzy-match suggestions, and a contextual hint.

### Error Code Ranges

| Range | Phase | Examples |
|-------|-------|----------|
| `E1xxx` | Lexer | Unterminated strings, invalid characters, malformed numbers |
| `E2xxx` | Parser | Unexpected tokens, missing semicolons, foreign keywords |
| `E3xxx` | Type checker — names & scope | Undefined variables, undefined functions, duplicate declarations |
| `E4xxx` | Type checker — types & coercions | Type mismatches, incompatible operands, failed promotions |
| `E5xxx` | Type checker — function signatures | Wrong argument count, argument type errors |
| `E6xxx` | Runtime (interpreter) | Loop iteration limit, division by zero, out-of-bounds access |
| `W7xxx` | Warnings | Unused variables, redundant casts, shadowed names |

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
