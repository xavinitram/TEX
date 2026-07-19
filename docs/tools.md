# Tools — the `.textool` bundling format (TOOL-1..5)

*Design note for v0.26.0 "Tools". Shipped-state + deferred-state + the pinning tests, in
the `xpu-transfer-scheduling.md` / `results-caching.md` mould. Provenance:
`docs/roadmap.md` §3 P3 (TOOL-1..5) and §9 (v0.26.0).*

A **tool** is a named, self-contained bundle of TEX code with a UI: the compositor's
gizmo / macro / HDA. TOOL-1 defines the file format (`.textool`), TOOL-2 the publish flow
(collapse a node selection → a tool), TOOL-3 makes a published tool a *compilation unit*
(warm keys re-derived at install), TOOL-4 the `tex build` CLI, TOOL-5 the threat model for
sharing tools. The four stock exemplars (Grade, Blur, Merge, Vignette) ship as `.textool`
files — dogfooding strategic bet #8 ("any stock node that can be expressed in TEX must be").

The load-bearing promise (the release exit gate, roadmap §9 line 863): **a `.textool`
round-trips author → publish → fresh-install → cook, bit-identical to the unfused graph.**

---

## 1. What a tool is (and is *not*)

A tool carries **every stage's TEX code inline** — sharing a tool is sharing one
self-contained plaintext file, and no program's compilation ever resolves an external
name. This is deliberately *not* the rejected cross-node import system (roadmap §7): the
line that would reopen that rejection — tools referencing other tools by name — is
**excluded from v1** (no by-name tool nesting).

Two shapes, both `.textool`:

- **Single-stage tool** — one TEX program with promoted params. All four stock exemplars
  are this shape. May have several external inputs (Merge reads `@fg`, `@bg`, `m@mask`).
  Cooks as a plain program: `tex_engine.cook(code, bindings)`.
- **Fused (multi-stage) tool** — a linear or DAG chain of stages with **exactly one
  external image source** (the fusion-region model: `tex_fusion.detect_fusable_regions`
  admits one external image edge). Internal handoff edges are image/mask-typed. Cooks
  through the *same* fused path a collapsed region does — the manifest stores the
  `GraphSpec` that `region_to_collapse_plan` emits and `engine.cook(chain_payload=)`
  consumes, so the tool layer is a thin pass-through over machinery FUS-3 already proves
  bit-exact against the unfused graph.

Unfusable constructs in a multi-stage tool are **authoring errors** caught at build/preflight
(`chain_preflight` never raises — it returns `{ok: False, stage_of_error, error}`); there is
**no sequential fallback mode**. A tool either fuses or it is rejected at `tex build`.

---

## 2. The manifest schema

`.textool` is UTF-8 JSON. `tex_tool.TEXTOOL_SCHEMA = 1` versions the *manifest* shape
(distinct from `tex_fusion.GRAPHSPEC_SCHEMA`, which versions the embedded fused payload).

```jsonc
{
  "manifest_schema": 1,              // TOOL manifest format version
  "name": "Grade",                   // display / node name (required)
  "tool_version": "1.0.0",           // the TOOL author's version (semver string)
  "tex_language": "0.23",            // LANG-3 language pin the stages were authored against
  "min_engine": "0.26.0",           // minimum TEX package __version__; fails at INSTALL, not cook
  "category": "Color",               // help-panel / palette grouping
  "context": "filter",               // generator | filter | transition | keyer (where a host surfaces it)
  "doc": "Nuke-style lift/gamma/gain grade.",
  "author": "TEX",

  // ── one of `code` (single-stage) OR `graphspec`+`terminal_code` (fused) ──
  "code": "f$gamma = 1.0; ...\n@OUT = ...;",        // single-stage program source

  // fused form (absent for single-stage):
  //   "graphspec": { ...tex_fusion GraphSpec (schema 1): stages[], dag, source_stage,
  //                  source_binding, terminal_chain_inputs... },
  //   "terminal_code": "...",                       // the terminal stage's source
  //   "terminal_image_input": "image",              // socket binding carrying the source

  "inputs": [                        // external @-bindings the tool exposes
    {"name": "image", "type": "IMAGE"}
  ],
  "outputs": [                       // output ports a host wires when instancing the tool
    {"name": "OUT", "type": "IMAGE"} //   (Vignette declares darkened + vignette_mask)
  ],

  "promoted_params": [               // the tool's widgets (from LANG-1 ParamDecl.metadata)
    {"name": "gamma",                //   external widget name
     "internal": "gamma",            //   the $param name inside its stage
     "stage": null,                  //   null = single-stage / terminal; int = graphspec.stages index; "terminal"
     "type": "f",                    //   LANG-1 type_hint: f i s b c v2 v3 v4
     "default": 1.0,
     "metadata": {"min": 0.0, "max": 4.0, "label": "Gamma"}}
  ]
}
```

Rules:

- `manifest_schema`, `name`, `tex_language`, and exactly one of `code` / (`graphspec` +
  `terminal_code`) are **required**. Everything else has a documented default.
- `promoted_params[*].metadata` is a plain `{str: float|int|str}` dict of **literal
  scalars** — the same shape `ParamDecl.metadata` carries (LANG-1). Recognised keys today:
  `min`, `max`, `step`, `precision` (numeric) + `label` (string). Publish (TOOL-2) copies
  it straight off the promoted param's `ParamDecl`; the host auto-widget builder consumes it.
- A promoted param's `default` and `type` mirror its `ParamDecl` default/type_hint, so an
  instanced tool node reconstructs the exact widget the source node had.
- **No fingerprint is ever stored** (ENG-5): the fused warm key is re-derived at install
  from the inline code (see §5). A `.textool` from a different TEX version still installs.

The manifest **inputs/promoted lists are the tool's contract**; the pinning canary
`test_tool_manifest_keys` locks the required key set so a host can rely on it.

---

## 3. Loading & validation — schema *before* compile (TOOL-5)

`tex_tool.load_tool(path_or_dict) -> ToolManifest` runs a strict pipeline, and **every
structural check happens before any TEX source is parsed or any code is generated** —
because a downloaded `.textool` is untrusted input to a code generator (§6):

1. **Parse JSON** (size-capped, §6).
2. **Schema validation** — `validate_manifest(raw)`: required keys present, types correct,
   `manifest_schema <= TEXTOOL_SCHEMA` (a newer manifest is rejected with a legible message,
   never mis-read), `promoted_params` well-formed, metadata values are literal scalars only,
   exactly one of the code/graphspec forms present. Raises `TEXToolError` on any violation.
   **Nothing is compiled yet.**
3. **Language-pin advisory** — if `tex_language` is newer than `LANGUAGE_VERSION`, attach a
   warning (mirrors LANG-3's W7004); does not block.
4. **Engine-version gate** — if `min_engine` > package `__version__`, raise `TEXToolError`
   ("tool needs TEX ≥ X"). Fails at install, not at cook (roadmap TOOL-4).

`preflight_tool(manifest) -> dict` then type-checks the stages *without cooking*: a
single-stage tool runs `tex_api.check(code, {})`; a fused tool runs
`tex_fusion.chain_preflight(stages, infer_binding_type)`. Both are total (never raise) and
return structured diagnostics — the `tex build` reporter (§ TOOL-4) renders them.

---

## 4. Cooking a tool

`tex_tool.cook_tool(manifest, inputs, params=None, **cook_kwargs) -> CookResult`

- `inputs`: `{binding_name: tensor}` for the external image/mask inputs.
- `params`: `{promoted_name: value}`; anything omitted uses the promoted param's `default`.

Promoted values are written into their target stage's param dict **uniformly** (single-stage
→ the program's bindings; fused → `graphspec.stages[i].params` or the terminal params),
then:

- **Single-stage**: `tex_engine.cook(code, {**inputs, **promoted_bindings}, **cook_kwargs)`.
- **Fused**: `tex_engine.cook(terminal_code, {terminal_image_input: source, **terminal_params},
  chain_payload=graphspec, **cook_kwargs)` — the real engine path (tiers, OOM ladder,
  precision-auto), identical to how a host cooks a collapsed region.

Because the fused path *is* the region path, the tool cook equals the unfused stage-by-stage
cook by the same construction FUS-3 pins (the round-trip oracle in §7 asserts it for a tool).

---

## 5. Tool = compilation unit (TOOL-3)

Publishing a fused tool should make it *faster* than its parts (strategic bet #6): the
promoted-param signature is fixed, so the fused artifact can be warm-compiled once.

`tex_tool.tool_warm_keys(manifest) -> list[str]` derives the fused chain's value-independent
fingerprint (`tex_fusion.fused_fingerprint`) **at install time, by re-fingerprinting the
inline stage code** — the loader already has the sources in hand. The key is **never carried
in the file** (ENG-5: fingerprints are deliberately unstable across TEX versions; a stored
one would be wrong after any TEX update). A single-stage tool's key is `TEXCache.fingerprint`.

`tex_tool.install_tool(manifest, *, warm=False, device=..., ...)` writes the manifest into
the host user dir (§ TOOL-2) and, **only with explicit `warm=True` consent** (TOOL-5:
validate-only default), drives the LAT-1a machinery via `tex_api.prewarm` at the promoted-param
signature: materialise + persist the codegen fn, submit a background `torch.compile`, seed the
capturability verdict. Warm-compile lives entirely off the cook hot path.

---

## 6. Threat model for shared tools (TOOL-5) — gates the install flow

TEX codegen **emits Python source from a user AST** (`tex_runtime/codegen.py`), so a
downloaded `.textool` is untrusted input to a code generator. A hostile tool must not be
able to escape the emitter, and even a non-escaping one can exhaust a machine. The posture:

**A. Validate-only default.** `install_tool` and `tex build` **never compile on install
without explicit consent** (`warm=True` / `--warm`). Installing a tool parses + schema-checks
+ type-checks it; it does not run codegen or `torch.compile` unless the user asks. So merely
adding a tool to a library cannot execute generated code.

**B. Manifest schema validation before any compile** (§3 step 2). No stage source reaches the
parser until the JSON shape, sizes, and promoted-param metadata are proven well-formed.

**C. Emitter injection audit + adversarial-AST fuzz lane.** The codegen emitter is the trust
boundary. Two structural facts make it safe, and both are now pinned:
  - **Identifiers cannot inject.** Every user identifier reaching codegen came through the
    lexer, which only admits `[A-Za-z_][A-Za-z0-9_]*`, and codegen namespaces each as
    `_s{i}_u_{name}` / `_tN` — there is no path from a source identifier to an un-prefixed
    Python name, keyword, dunder, or attribute access. A tool cannot name `__globals__`,
    `import`, `eval`, etc., because the lexer would never have tokenised it as one identifier.
  - **String literals cannot inject.** TEX string literals reach the emitter as Python
    `str` values and are emitted with `repr()` (never f-string-interpolated into the
    generated source), so quotes/newlines/backslashes in a tool string cannot break out of
    the literal.
  The **adversarial-AST fuzz lane** (`test_tool_emitter_fuzz`) has three arms: (1) programs
  naming dangerous/unknown functions (`__import__`, `eval`, `system`) or unicode-confusable
  identifiers must be **rejected before codegen** (the type-checker/lexer gate); (2) valid-but-
  hostile programs — dunder/keyword identifiers, hostile-named user *functions* (exercising the
  `_uf_{name}` emission + the depth-guard `raise RuntimeError('… in {name}()')` string site),
  pathological string literals (quotes, `\n`, `"""`, `#{}`) — are compiled and their generated
  Python is **`ast`-walked**: a repr'd string is an `ast.Constant` (safe DATA), so a finding
  fires only on real CODE — an `Import`, an `Attribute` to a dunder/`system`/pickle vector
  (`load`/`save`/`jit`/…), or a `Call` to a blocklisted name. (3) a fused chain is scanned the
  same way (the splice adds more prefixing, so it is strictly safer). The check is a **blocklist
  of escape vectors**, not an allowlist — the emitter legitimately emits many sanctioned
  helpers (`_bp`, `RuntimeError`, `int`) an allowlist would have to enumerate — backstopped by
  the real gate above. Any escape is a red test, not a shipped tool.

**D. Documented resource limits.** Schema validation caps the attack surface a *valid* tool
can present without an emitter escape: `MAX_TOOL_BYTES` (manifest size), `MAX_STAGES`
(mirrors `tex_fusion._MAX_FUSED_REGION_STAGES = 16`), `MAX_PROMOTED_PARAMS`, `MAX_STAGE_CODE_BYTES`.
These bound parse/compile cost; they do **not** bound *cook* cost — a valid tool can still
request an 8K `gauss_blur` and OOM/TDR a machine, exactly as a hand-written program can. That
residual is stated, not silently "handled": a host that installs third-party tools owns the
same memory-budget / TDR-watchdog duty it owes any user program (CACHE-5 / ROI-5 territory).

TOOL-5 is a design note + a test lane + the audit above; it records no new *rejected*
decision (the by-name-nesting exclusion is already in the §7 register).

---

## 7. Pinning tests (roadmap §10.4)

| Shape | Test | Asserts |
|-------|------|---------|
| **differential oracle** | `test_tool_roundtrip_unfused` | a fused tool cooked via `cook_tool` == the stage-by-stage unfused interpreter cook, bit-exact (CPU + CUDA) — the release exit gate |
| **canary** | `test_tool_manifest_keys` | the required manifest key set + `promoted_params[*]` key set (a host contract) |
| **derivation** | `test_tool_promoted_params` | promoted values land in the right stage's bindings; omitted ones fall back to `default` |
| **canary** | `test_tool_stock_exemplars` | every shipped `.textool` loads, preflights clean, and cooks |
| **security** | `test_tool_emitter_fuzz` | the §6-C adversarial-AST lane |
| **canary** | `test_tool_schema_rejects` | malformed manifests / newer `manifest_schema` / newer `min_engine` are rejected with `TEXToolError`, before any compile |

---

## 8. v1 scope, honestly recorded

- **Single external image source per fused tool** (the fusion-region constraint). Multi-input
  merges are expressible as *single-stage* tools (Merge reads `@fg`/`@bg`), not as fused chains.
- **No by-name tool nesting** (roadmap §7 rejection stands). A tool's stages are all inline.
- **No sequential fallback** — an unfusable multi-stage tool is a build error.
- **Warm-compile is opt-in** (TOOL-5-A). Install is validate-only by default.
- **Cook-cost is not sandboxed** (§6-D). Resource limits bound compile, not cook.
- **The frontend publish UI (TOOL-2) is host-visual**; its live-session checklist is verified
  in a running ComfyUI. The manifest writer, publish route, and instanced-tool cook are
  backend-tested here; the collapse-selection picker is JS that lands with the release and is
  visually verified by the maintainer in-session (screenshots into the build log), the standing
  practice for frontend-touching releases (roadmap §10.6).
- **Instancing (dropping an installed tool as a node) is the deferred frontend half** of the
  bundling promise, and the one deliberate scope call in v0.26. Its backend is complete and tested:
  `list_tools`/`tool_summary` enumerate installed tools with their `inputs`, **`outputs`**, and
  promoted-param `widgets`; `cook_tool` runs one; the manifest declares outputs so a host knows what
  to wire (the multi-output Vignette especially). What remains is purely a ComfyUI node that renders
  those widgets and cooks through `cook_tool`, verified live. Until it lands, a published tool is
  usable from the CLI/engine API but not yet droppable in the ComfyUI graph.
