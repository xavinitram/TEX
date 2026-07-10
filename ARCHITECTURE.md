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

**`tex_core` boundary (S-1, doc 30/35).** The ComfyUI-adapter layer is exactly three
files — `tex_node.py`, `__init__.py`, `tex_runtime/host.py`; **every other module is
`tex_core`** and imports no ComfyUI surface (`comfy*`/`server`/`folder_paths`/`nodes`).
This is machine-enforced by `test_s1_core_no_comfy` (a package-level lint) and exercised
by `test_s1_comfyui_free_execution` (blocks every comfy import, then drives
`TEXWrangleNode.execute` + `tex_api.compile`). The whole suite already runs ComfyUI-free
(comfy is not on the CI path), so CI **is** the standalone lane — the payoff of a physical
`tex_core/` package split without the churn. The physical `git mv` reroot is deferred to a
live-ComfyUI session (import-path verification can't be done headlessly); the manifest above
is the split's contents.

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

## Lazy input cooking (`tex_lazy.py` + the `in_N` slot pool)

ComfyUI decides input laziness **per schema-declared input name at graph-build time**
(`comfy_execution/graph.py::get_input_info` — an unknown name is never lazy), so TEX's
dynamic user-named inputs cannot be lazy directly. The mechanism instead:

1. **Schema** declares a fixed pool of lazy AnyType slots (`in_0..in_15`, hidden —
   the frontend removes all initial slots in `onNodeCreated`).
2. **Frontend** (`_texLazyRename`, after fusion collapse in the `graphToPrompt`
   wrapper) renames wired user inputs onto pool slots **in the queued prompt only**
   (graph/workflow/labels keep user names) and adds the `_tex_slot_map` constant.
   Fail-safe: pool-name collision or >16 wires → the node stays fully eager.
3. **`check_lazy_status`** (tex_node.py) runs `tex_lazy.lazy_required_bindings(code,
   scalar_params)` — parse → substitute $params as fp32 literals → fold → propagate →
   fold → prune literal-condition flow → collect surviving `@/$` references — and
   requests only the needed slots. Uncooked slots arrive as `None`; ComfyUI re-invokes
   iteratively, so **wired scalar $params cook first** and fold on the next round
   (the "T4-lite" round). Unrequested slots' upstream subgraphs are **never cooked**.
4. **`execute()`** maps cooked `in_N` values back to user names and forgives E6003
   for references that are statically dead under the current params (same memoized
   analysis — the two callers cannot disagree).

**Safety rules (all fail toward "cook everything"):** R1 — if any spatial-capable wire
exists, the *first* one (first-wins shape derivation, `_determine_spatial_shape`) must
be needed and of known tensor type (IMAGE/MASK); R2 — LATENT wires always cook (they
flip output typing/fp32); R3 — analysis failure keeps all. **Never severed:** `@A*0`
(NaN·0=NaN; the optimizer refuses x*0→0 for shape safety), `&&`/`||` operands (the
interpreter evaluates both sides), spatial-condition branches (torch.where computes
both) — invariant #11. Fused chains (`_tex_chain`) request everything in v1.
Setting: `TEX Lazy: skip cooking unused inputs` (default on).

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

**`precision="auto"`** (PR-LP2, `tex_runtime/precision_policy.py`) is an **experimental,
conservative fp16 gate**. It resolves to fp16 only on CUDA, ≥1024², for a smooth pointwise
program, and — the doc-32 rework — only when a **condition-number heuristic proves the
program won't amplify fp16's ~1e-3 input error past the 8-bit quantum**. That heuristic
(`_amplification_hazard`) is a flow-sensitive image-**gain + magnitude** analysis: it tracks
amplification assembled from sub-threshold steps (`sin(@A.r*3*3)`), image×image squaring,
`/const` chains, builtin-const/dimension products (`@A.r*iw`), dot/matrix-row/length/cross
fan-in, `fit` remaps, additive round-trips (`(@A.r+60000)-60000`), and array reductions,
with scalar const-propagation; it declines ill-conditioned fns (tan/atan2/normalize/hypot/
sdiv) and **any user-function call touching image lineage** (the gain pass does not recurse
into `FunctionDef` bodies — doc 33 F1 — so it declines rather than model them). Verified
**0 accuracy violations across 225 direct-expression adversarial programs (two independent
red-team rounds) + the fuzzer** — but it is a **heuristic, not a proof** (each round found
new classes), so a per-cook finiteness net re-cooks + pins fp32 on any non-finite
(runs EVERY fp16 cook — doc 32 C2 — never memoized). The decision (not the finiteness
verdict) is memoized per (fingerprint, resolution-bucket, device).

**Honest perf:** on the real `TEXWrangleNode.execute` path `auto` is essentially
**neutral (~0.99×@1024 / ~1.08×@2048)** — the finiteness backstop costs about what fp16
saves. The earlier "1.45×" was a repeated-input microbenchmark off `Interpreter.execute`;
the raw fp16 win (~1.35–1.45×) is available, without the safety net, via expert
`precision="fp16"`. fp16 stays out of the compiled/graph tiers this cycle. Reductions
(`img_*`, `arr_*`) accumulate in fp32 (an fp16 sum overflows to inf at ≥1024²); an
out-of-fp16-range literal / a large-value `vec()` also stays fp32 (interp==codegen).

## The 15-cache architecture

Non-redundant by design — each store keys on a different thing (source-hash vs
`id()`-type_map vs device/precision tuple vs AST-fingerprint vs resolution-bucket)
with a distinct lifecycle (persist-across-restart vs per-run-clear vs LRU). Do not
"consolidate" them (doc 24 §6). #14 is `tex_lazy._memo` (code-hash × fp32 param bits →
required-binding set; shared by `check_lazy_status` and `execute()`). #15 is
`tex_node._AUTO_DECISION` (fingerprint × resolution-bucket × device → the `precision="auto"`
fp16/fp32 gate decision — the *decision*, not the per-cook finiteness verdict; bounded LRU,
cleared at 512 entries). The count here must match AGENTS.md (a DOC-7b check enforces it).
