# AGENTS.md — start here

You are editing **TEX Wrangle**, a per-pixel DSL for ComfyUI (a GLSL/VEX-like
language: lexer → parser → type-checker → optimizer → tensor execution on PyTorch).
This file is pointers + invariants, not a re-doc. Read `ARCHITECTURE.md` for the
module graph. Read this before touching anything.

## The pipeline (one line)

`lexer → parser → ast_nodes → type_checker → optimizer → { interpreter | codegen | compiled/graphed }`
— fusion (`tex_fusion.py`) splices linked nodes *before* compile; the interpreter is
the **oracle** every other tier must match bit-for-bit. Full map: `ARCHITECTURE.md`.

## Invariants — MUST NOT BREAK (each names its enforcing check)

| # | Invariant | Why | Enforced by |
|---|-----------|-----|-------------|
| 1 | **No numpy.** Never `import numpy` or `.numpy()`. | CI runs torch **without** numpy; `.numpy()` raises there. Has already bitten the project. Use `struct.pack`/`.tolist()`. | `tests/test_no_numpy_ban.py` (LNT-1) |
| 2 | **interp ↔ codegen are bit-exact** (`tol=1e-5` fp32). | The crown-jewel contract; codegen falls back to interp. | `test_codegen_equivalence` + the differential fuzzer (TST-1) |
| 3 | **The optimizer re-typechecks its output AST.** | CSE/LICM synthesize nodes outside the `id()`-keyed `type_map`; dropping the re-check silently corrupts types. | `tex_cache` calls it; optimizer/type_map contract (doc 22 §5) |
| 4 | **Coordinate/spatial builtins are forced fp32** (never `self._dtype`). | The M-3 fp16 contract; fp16 coords mis-address rows at large H. | `interpreter.py` (~line 368) — do not "unify" the dtype |
| 5 | **A non-pixel-local stdlib fn MUST be tagged** `non_local=True` in its `@stdlib(...)` decorator. | The tag *derives* `_NON_LOCAL_FNS` (default pixel-local); a missing tag is **wrong output only when tiled** (passes non-tiled tests). | TST-3 taxonomy test (derivation + name-prefix heuristic); see recipe below |
| 6 | **GPU timing wraps `torch.cuda.synchronize()`.** | Unsynced CUDA timing measures only kernel-launch enqueue. Has bitten benchmarks before. | benchmark harness convention |
| 7 | **Every change is behavior-preserving + perf-neutral** on the DEFAULT path. | v0.18 adds one opt-in perf lever (`precision="auto"`, default fp32); everything else is still structure/UX. A refactor that risks bit-exactness or a hot path is a **bad trade** — see §"Trades to refuse". | full suite green + benchmark neutral |
| 8 | **`comfy.model_management` is imported ONLY in `tex_runtime/host.py`** (PORT-1). | The host seam keeps TEX runnable host-agnostic (CLI, tests, future hosts); re-scattering the import re-couples it. | `test_port1_import_lint` |
| 9 | **CPU↔GPU is a *characterization envelope*, not bit-parity** (same-device interp↔codegen is the only exactness contract). | Cross-device divergence is already 1.8e-7…6.1e-2; a torch/driver bump that blows a class band must be a loud decision, not silent drift. | `test_cross_device_envelope` (PR-LP1); determinism `test_determinism_pin` (PR-LP5) |
| 10 | **`precision="auto"`'s gate is a CONDITION-NUMBER heuristic, not a proof** (doc 32/33). It declines fp16-fragile fns, unsafe pow, data branches, image comparisons, out-of-range literals, image-lineage amplification (a flow-sensitive gain+magnitude analysis, `precision_policy._amplification_hazard`, catching chained/squaring/fan-in (dot/matrix/length/cross)/round-trip/builtin-dimension/`fit`/array amplification, with scalar const-propagation), **and any user-function call touching image lineage** (doc 33 F1: the gain pass does NOT recurse into `FunctionDef` bodies, so it declines rather than model them). Verified **0 accuracy violations across 225 direct-expression adversarial programs (2 red-team rounds) + the fuzzer**, headline preserved. STRONG, not COMPLETE — over-declines rather than risk accuracy; a per-cook finiteness net re-cooks fp32 on any non-finite (never memoized — doc 32 C2). auto is **experimental + ~perf-neutral** (the net costs ~what fp16 saves); the raw fp16 win is expert `precision="fp16"`. Any *accepted* program exceeding 3.9e-3 is a gate bug. | `test_c1_amplification_gate` (+ user-fn cases) + `test_c2_data_dependent_nan` + gate-filtered fuzzer |
| 11 | **The lazy analysis (`tex_lazy.py`) may only OVER-approximate.** `@A*0` (NaN·0=NaN), `&&`/`||` operands, and spatial-condition branches must never sever a dependency; only conditions that fold to a compile-time literal prune. All safety rules (R1 shape-anchor / R2 LATENT / R3 analysis-failure) fail toward "cook everything". | An under-approximation skips an input the cook needs — loud ("Input '@X' is not connected"), never silent, but must never ship. The same memoized analysis drives `check_lazy_status` AND `execute()`'s E6003 gate, so they cannot disagree. | `tests/test_lazy_cooking.py` (the `*0`/`&&`/string-param rows pin the never-sever set) |

## Adding a stdlib function (the current recipe — REG-1 registry)

> REG-1 collapsed the parallel name→impl / taxonomy tables into **one `@stdlib(...)`
> decorator co-located with the impl**. `get_functions()` is now a registry view;
> the `spatial`/`sync`/`non_local` tags *derive* the taxonomy sets (TST-3 checks it).
> The signature stays compiler-side by design (Option 1 — preserves the layering
> invariant; TST-3 machine-checks registry↔`FUNCTION_SIGNATURES` parity).

For a **pixel-local** function (output at a pixel depends only on that pixel):
1. `tex_runtime/stdlib.py` — add the impl with a co-located decorator:
   `@stdlib("NAME")` / `@staticmethod` / `def fn_NAME(...)`. (This one line replaces
   the old `get_functions()` row **and** the taxonomy-table edits.)
2. `tex_compiler/stdlib_signatures.py` — add `"NAME": {"args": (lo,hi), "return": <rule>}`
   to `FUNCTION_SIGNATURES` (compiler-side type contract; keep `return` a **named**
   helper, not a lambda).
3. Add a codegen-equivalence test (`tests/test_codegen_optimizer.py`); the fuzzer/edge
   matrix (TST-1/2/6) auto-cover it from the signature the moment it's registered.

If your function **reads neighbouring pixels or the whole image** (sample/fetch/
blur/morphology/reduction), set the tags **in the same decorator** — do NOT edit the
downstream sets by hand (they derive from these tags; TST-3 fails a mismatch):
- `non_local=True` — **required**, or it is *silently wrong when tiled* (TST-3's
  name-prefix heuristic catches a forgotten tag on `sample*`/`blur`/morphology names).
- `sync=True` — if it does an internal `.item()`/sync (radius/octave count), else CUDA-graph capture fails.
- `spatial=True` — if codegen should lower it as a stencil.

`js/tex_extension.js` `TEX_HELP_DATA` is still hand-kept until DOC-4 generates it
from the registry's `doc=`/`ex=` slots.

New built-in names are **reserved** — a user program may not define a function of
the same name. Adding a name is a (minor) breaking change; note it in the CHANGELOG.

Test: `cd tests && python run_all.py` (or `pytest` from the repo root). numpy is not
installed in CI — do not add it to a test.

## Trades to REFUSE (things that look like cleanups and are bugs)

- **Do NOT merge the interpreter and codegen stdlib implementations.** The ~40-fn
  duplication *is* the safety margin (codegen falls back to interp). Shared
  tables/constants are already factored — that's the correct boundary.
- **Do NOT hoist the function-local imports to top-level** to "remove the cycles."
  The cycles are logical; forcing load-time reintroduces an ordering crash.
- **Do NOT specialize `pow(x, 0.5)` → `sqrt`** — it preserves codegen↔interp NaN
  equivalence.
- **Do NOT "improve" codegen's M-5 `out=` reuse / ownership tracking** during any
  code move — mechanical moves only.

## The DO-NOT-TOUCH register (load-bearing code that looks redundant/dead)

Each of these looks removable and is not. Simplifying = a measured slowdown or a
silent-wrong result.

**Perf traps (simplifying = slower):**
- `stdlib.py` `_get_grid_buf` **allocate-and-hold** — reuse measured ~30% SLOWER on CPU. Comment says "PERF TRAP."
- `interpreter.py` inline dtype zero-guards (`where(x==0, EPS, x)`, on the `_eval_binop` hot path) — a shared helper adds a per-divide call.
- Interpreter `fn_*` vs codegen `_emit_fn_*` — deliberate tree-walk-vs-emit split.

**Correctness contracts (removing = silent-wrong):**
- `tex_cache` post-optimize re-typecheck (invariant #3).
- `interpreter.py` coordinate builtins forced fp32 (invariant #4).
- `tex_memory` `clear_graph_cache()` after cache eviction — stale-CUDA-graph-address safety.
- `graphed.py` `_graph_mode_disabled` kill-switch, `capture_error_mode="thread_local"`, `_build_keepalive` refs, per-graph pool.
- `compiled.py` dynamo reset on the **calling** thread (dynamo state is process-global); `has_spatial` deliberately NOT memoized (depends on binding *values*).
- `tex_marshalling` numpy-free `struct.pack` fingerprint (invariant #1).
- `optimizer.py` noise excluded from the CSE-pure whitelist (pending a determinism audit).
- `interpreter.py` `_literal_cache` persists across executions; spatial-if short-circuit (break/continue correctness).

**Config escape hatches (keep even though never set normally):**
`TEX_CODEGEN_NO_OUT_REUSE` / `_OUT_REUSE_ENABLED`, `TEX_CACHE_BUDGET_MB`, the
`TORCHINDUCTOR_CACHE_DIR` ownership-check, the `_tex_any` phantom search-panel slot,
the `bf16` bench plumbing (deliberately dev/bench-only, not user-exposed).

**The 13 caches are non-redundant** — each keys on a different thing with a distinct
lifecycle. Do not consolidate them.

## Module size budget (REG-2 — soft policy)

Keep a module under **~1500 LOC (soft)** / **2000 LOC (hard)**. Past the hard line,
split by *domain*, not by aesthetics — a module that needs splitting has usually
grown a second responsibility. This is a policy, not a gate: `test_reg2_loc_budget`
*reports* over-budget modules and **ratchets** (a **new** module crossing 2000 is a
red test; the known-over baseline below is grandfathered pending its planned split).

Currently over the hard budget — status as of v0.18.0 (drift-checked by
`test_doc7b_map_drift`, which reds if any `~LOC` here diverges >20% from `wc -l`):

| Module | LOC | Status |
|--------|-----|--------|
| `tex_runtime/codegen.py` | ~2730 | STR-7 split **shipped** (4092→2731: `codegen_stdfns.py` / `codegen_stencil.py` / `codegen_persist.py` extracted). Docs 27/28 verdict: **stop here** — the remainder is one cohesive emitter; further splitting is aesthetic, not domain-driven |
| `tex_runtime/stdlib.py` | ~2330 | per-domain `stdlib_*.py` still planned — unblocked by REG-1 (registration is per-decorator, not one central dict) |
| `tex_runtime/interpreter.py` | ~2140 | STR-3/STR-4 **shipped** (`ExecContext` + shared `NodeVisitor` extracted); the residual is the core tree-walk |

`tex_compiler/optimizer.py` (~1520) is over *soft*; STR-5 (the `PASSES` list) **shipped**.

## Doc-layering policy (DOC-6)

Three layers, each with one audience — put a doc where its reader looks:
- **Repo root** = agent/developer facing — `AGENTS.md` (this file), `ARCHITECTURE.md`,
  `DEVELOPMENT.md`, `Function-Reference.md` (generated), `CHANGELOG.md`.
- **`wiki/`** = end-user facing — tutorials + reference (`Learn-TEX-in-5-Minutes.md`
  is canonical; the root copy is a redirect stub). Don't duplicate a wiki page at
  the root — link to it.
- **`TEX_research/`** (outside the repo) = deep rationale — design specs, audits,
  build logs. The *why*, not the *how*.

## Where the deep rationale lives

`TEX_research/` (outside the repo) holds 26 design/audit/build-log docs. In-repo
index: `DEVELOPMENT.md` → "Research index". Before reversing a design decision
(e.g. "add an import system" — rejected on ethos grounds), check it there.
