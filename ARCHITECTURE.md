# TEX Wrangle — Architecture

The single source for the module graph and dataflow. `AGENTS.md`, `DEVELOPMENT.md`,
and the memory file link here rather than restating it (so it can't drift).

## Pipeline (dataflow)

```
source text
  → lexer          (tex_compiler/lexer.py)      tokens
  → parser         (tex_compiler/parser.py)     AST (tex_compiler/ast_nodes.py)
  → type_checker   (tex_compiler/type_checker.py) type_map: id(node) → TEXType
  → optimizer      (tex_compiler/optimizer.py)  const-fold/DCE/CSE/LICM/unroll  (re-typechecks the result)
  → EXECUTION TIER (one of):
       interpreter  (tex_runtime/interpreter.py)   tree-walking tensor eval — the ORACLE
       codegen      (tex_runtime/codegen.py)        AST → flat Python fn (no torch.compile)
       compiled     (tex_runtime/compiled.py)       codegen + torch.compile lifecycle, in a disposable thread
       graphed      (tex_runtime/graphed.py)         CUDA-graph replay of the interpreter
  → marshalling    (tex_marshalling.py)          IMAGE/MASK/LATENT/string in & out
```

**Fusion** (`tex_fusion.py`) splices a linked chain of TEX nodes into ONE program
*before* compilation, so only the terminal node cooks. **Caching** (`tex_cache.py`)
keys compiled artifacts and fused programs. **Memory** (`tex_memory.py`) does OOM
preflight, byte-budgeted cache eviction, and horizontal-strip tiling.
**Autotier** (`tex_runtime/autotier.py`) is the `auto` mode's measure→trial→commit
state machine. **Noise** (`tex_runtime/noise.py`) is the arithmetic-hash procedural
noise family. The ComfyUI node itself is `tex_node.py`.

## Module layers (the import rule)

Three layers; **imports point downward only**. The `tex_compiler` package has
**zero** edges into `tex_runtime` — this is a load-bearing invariant.

| Layer | Modules | Depends on |
|-------|---------|-----------|
| **Types** (leaf vocabulary) | `types` (`TEXType`, `CHANNEL_MAP`, `TYPE_NAME_MAP`, swizzles) | nothing (stdlib `enum`/`dataclasses` only) |
| **Compiler** (IR + front end) | `ast_nodes`, `lexer`, `parser`, `type_checker`, `optimizer`, `stdlib_signatures`, `diagnostics` | types + itself |
| **Runtime** (execution tiers) | `interpreter`, `codegen` (+ `codegen_stdfns`/`codegen_stencil`/`codegen_persist`, STR-7), `compiled`, `graphed`, `stdlib` (+ `stdlib_registry`), `precision_policy` (PR-LP2 `auto` gate), `tier_trace`, `noise`, `autotier`, `tex_cache`, `tex_marshalling`, `tex_memory`, `host` (PORT-1 seam — the ONLY `comfy.model_management` consumer) | types + compiler + itself |
| **Public API** (host-agnostic) | `tex_api` (`compile`/`execute`/`Program`, PORT-2), `tex_cli` (`tex run`, PORT-3) | runtime + compiler |
| **Orchestration** | `tex_node` (+ `tex_doctor` route, DBG-4), `tex_fusion` | all of the above |

**Coupling hubs** (Appendix A, doc 24): `tex_node` fan-out 12; `ast_nodes` fan-in 11
(pure data — fine). `type_checker` was fan-in 9 (the `TEXType` leak — 9 modules imported
the *checker* just to know what a `vec3` is); **STR-1 relocated `TEXType`/`TEXArrayType`/
`CHANNEL_MAP`/`TYPE_NAME_MAP`/`VALID_SWIZZLES` to the dependency-free `tex_compiler/types.py`
leaf**, dropping the checker's fan-in to ~2 (the runtime now depends on the IR + `types`,
never the checker). `type_checker` re-imports them for its own use, so legacy
`from .type_checker import TEXType` still resolves.

**Two logical import cycles** remain, broken by hand via *function-local* imports
(deliberate — see §5/§6 of doc 24, and `AGENTS.md`): a 7-node runtime SCC
(`codegen/graphed/compiled/noise/stdlib/interpreter/tex_cache`); `tex_node ↔ tex_memory`.
(STR-1 removed the third — `stdlib_signatures ↔ type_checker` — by moving `TEXType` to the
leaf; `type_checker`'s lazy `FUNCTION_SIGNATURES` bind is now defensive, not cycle-breaking.)
Do **not** hoist the remaining two to top-level imports — the cycles are logical; the fix is
to remove the *reason* (REG-1), not to move the import.

## The stdlib taxonomy tables (why an LLM edit here is dangerous)

A stdlib function is classified along several axes. **REG-1 single-sourced the
classification into one `@stdlib(...)` decorator** co-located with each impl in
`tex_runtime/stdlib.py`; `get_functions()` is now a view of that registry, and the
`spatial`/`sync`/`non_local` tags let **TST-3 derive and machine-check** the tables
below, so they can no longer silently drift.

| Axis | Source of truth | Consumed by | Default when absent |
|-------|------|------|---------------------|
| name → impl | `@stdlib("name")` decorator → `get_functions()` view | interpreter, codegen | (function unusable) |
| name → arg-count + return rule | `FUNCTION_SIGNATURES` (compiler; **not** derived — Option-1, TST-3 checks parity) | `type_checker` | (type error) |
| `spatial=` | `@stdlib` tag → `_SPATIAL_STDLIB` (`codegen.py`) | codegen stencil path | non-spatial |
| `sync=` | `@stdlib` tag → `_SYNC_STDLIB` (`graphed.py`) | CUDA-graph capture gate | capturable (may fail loudly) |
| `non_local=` | `@stdlib` tag → `_NON_LOCAL_FNS` (`tex_memory.py`) | tiling planner | **pixel-local → WRONG OUTPUT WHEN TILED** ⚠️ |
| help/autocomplete | `TEX_HELP_DATA` (`js/tex_extension.js`) | editor | (no autocomplete) |

⚠️ `non_local` defaults to "pixel-local": forgetting the tag on a
`sample`/`blur`/morphology fn produces **silently wrong output only when the program
is tiled under memory pressure** — it passes every non-tiled parity test. TST-3's
name-prefix heuristic (`sample*`/`fetch*`/`blur`/`erode`/`dilate`/`*_filter` must be
`non_local`) now turns that forgotten tag into a **red CI test**, not a field bug.

## Acceleration tiers (`compile_mode`)

`none` (default: interpreter, or codegen for exact stencils) · `auto` (experimental
measure→trial→commit) · `torch_compile` · `cuda_graph` (GPU replay). Every tier
**falls back to the interpreter** on any failure; the interpreter is the universal
oracle and the bit-exactness reference.

## Precision & parity (PR-LP1)

The **only** exactness contract is **interp↔codegen on the SAME device** (`tol=1e-5`
fp32) — invariant #2. **CPU↔GPU bit-parity does not exist and is not sold**: measured
fp32 cross-device divergence is already 1.8e-7 (pointwise) → 7.9e-5 (grid_sample) → 6.1e-2
(scatter, where a coordinate-rounding ULP legally moves a whole deposit to a neighbouring
pixel — structurally benign, numerically large). Instead a **characterization envelope**
(`tests/test_cross_device_envelope.py`) pins each program class inside its measured band,
so a torch/driver bump that blows a band is a loud, recorded decision, not silent drift.

**Determinism is a free, marketed property**: TEX is bitwise run-to-run deterministic on
CUDA across every class incl. scatter atomics under collision stress; forcing strict
determinism would *cost* 1.48× on scatter, so TEX already rides the fast path
(`tests/test_determinism_pin.py`). The CPU is the honest caveat (~5.5e-6 threaded-accumulation
variance).

**`precision="auto"`** (PR-LP2) resolves to fp16 only in the measured win region — CUDA,
≥1024², a smooth pointwise program (no sampling/scatter/reduction, no discontinuous/domain
function, no data branch, no image-derived threshold; `tex_runtime/precision_policy.py`).
The decision is memoized per (fingerprint, resolution-bucket, device); a first-cook
finiteness check re-cooks + pins fp32 on any NaN, then trusts the verified fingerprint — so
the win (~1.45× at 2048²) isn't eaten by a per-cook CUDA-sync. fp16 stays out of the
compiled/graph tiers this cycle. Reductions (`img_*`, `arr_*`) accumulate in fp32 (an fp16
sum overflows to inf at ≥1024²).

## The 13-cache architecture

Non-redundant by design — each store keys on a different thing (source-hash vs
`id()`-type_map vs device/precision tuple vs AST-fingerprint vs resolution-bucket)
with a distinct lifecycle (persist-across-restart vs per-run-clear vs LRU). Do not
"consolidate" them (doc 24 §6).
