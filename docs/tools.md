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
- **Fused (multi-stage) tool** — a linear or DAG chain of stages with **one external image
  source** (the fusion-region model: `tex_fusion.detect_fusable_regions` admits one external
  producer) and, **opt-in**, further external inputs, each naming the stage bindings it is
  written into (`inputs[*].feeds`, §2) and required to be co-extent with the source at cook
  (§4). Internal handoff edges are image/mask-typed. Cooks
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
  //   and, opt-in, on any input OTHER than the source:
  //   {"name": "plate", "type": "IMAGE", "feeds": [["terminal", "plate"], [0, "ref"]]}

  "inputs": [                        // external @-bindings the tool exposes
    {"name": "image", "type": "IMAGE"},
    {"name": "mask", "type": "MASK", "optional": true}   // may be left unwired (host UI hint)
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
     "metadata": {"min": 0.0, "max": 4.0, "label": "Gamma",
                  "tooltip": "Power curve applied to normalized pixel values."}},
    {"name": "channel",              //   a labelled-choice ("combo") widget
     "internal": "channel", "stage": null, "type": "i", "default": 0,
     "metadata": {"min": 0, "max": 2, "step": 1, "label": "Channel",
                  "options": ["Red", "Green", "Blue"]}}
  ]
}
```

Rules:

- `manifest_schema`, `name`, `tex_language`, and exactly one of `code` / (`graphspec` +
  `terminal_code`) are **required**. Everything else has a documented default.
- `promoted_params[*].metadata` is a plain `{str: float|int|str}` dict of **literal
  scalars** — the same shape `ParamDecl.metadata` carries (LANG-1) — plus two structured
  keys, each capped to bound parse cost the way the numeric ones already are (§6-D):
  `tooltip` (a `str`, at most `MAX_TOOLTIP_CHARS` = 1024 characters) and `options` (a
  labelled-choice list, valid only on an `"i"`-typed param: 1..`MAX_OPTIONS` = 256
  non-empty strings, each at most `MAX_OPTION_CHARS` = 128 characters). When `options` is
  present, `min`/`max`/`step` — if given at all — must read `0` / `len(options)-1` / `1`
  (one source of truth instead of two that could disagree), and a given `default` must be
  an `int` index into the list. Recognised keys today: `min`, `max`, `step`, `precision`
  (numeric) + `label`, `tooltip` (string) + `options` (list). Publish (TOOL-2) copies
  metadata straight off the promoted param's `ParamDecl`; a host auto-widget builder
  consumes whatever subset it recognises, and a host that doesn't recognise a key ignores it.
- `inputs[*].optional` (`bool`, default absent/`false`) marks an external input a host may
  leave unwired — an optional mask or reference plate, say. It is a host UI hint only: TEX
  binds nothing for an absent input either way, and a program that actually reads an
  unwired one still fails the ordinary E6021 "not connected" gate at cook, never a silently
  wrong pixel. A **fused** tool's source input may never be `optional: true` — the engine
  requires it to splice the chain — and neither may a fed input (next rule).
- `inputs[*].feeds` (**fused tools only**; absent by default) routes an external input other
  than the source into the chain: a list of 1..`MAX_STAGES` `[stage, binding]` pairs, `stage` a
  `graphspec.stages` index or `"terminal"` (the `promoted_params[*].stage` vocabulary),
  `binding` an identifier. At cook the input's tensor is written into every binding it names,
  exactly where a promoted value goes (§4), so the fused program is the ordinary GraphSpec one.
  Once any input declares `feeds`, every input except the source (`terminal_image_input`) must
  declare it and the source may not. A fed input is `IMAGE` or `MASK` and never `optional`. A
  feed targets a **whole wire**: `binding` is a bare identifier, so a fused tool cannot feed one
  plane of a PLANES wire (`beauty.diffuse`) in this release — the manifest validator refuses the
  dot rather than binding a port no host can wire; it reopens with a host wire type that carries
  a plane set (v0.37.0, language 0.24, `LANGUAGE.md` §5.3). A
  feed may not target a binding its stage already binds — its chain input, a source injection
  point, a baked param, a promoted param's `internal`, or another feed — because one of the two
  writes would silently win. On a DAG spec every stage must read a chain, the source or a feed:
  a stage that reads none would adopt the fused program's extent instead of cooking at its own,
  which is why the graph fusion detector never folds such a node either. A fused tool with no
  `feeds` anywhere is exactly the one-input shape it always was. `feeds` needs no
  `manifest_schema` bump: a tool with fed inputs declares at least two inputs, which a build
  without `feeds` refuses at load rather than misreads.
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

**Refusal codes.** The fused-input refusals carry a stable `TEXToolError.code` — a host keys on
it, for instance to word the refusal in its own language; the message beside it may be
reworded, the code may not — and `TEXToolError.input` names the offending input when one is to
blame. Every other `TEXToolError` has `code = None`, and `str(e)` is the message alone, as it
always was.

| `code` | Raised by | Meaning |
|--------|-----------|---------|
| `fused-input-count` | `validate_manifest` | a fused tool that declares no `feeds` declares more than one input |
| `fused-input-unrouted` | `validate_manifest` | a fused tool with `feeds` has a non-source input that declares none |
| `fused-feed-invalid` | `validate_manifest` | `feeds` is malformed, on the source, on a non-`IMAGE`/`MASK` or `optional` input, targets a stage out of range, or sits on a single-stage tool |
| `fused-feed-collision` | `validate_manifest` | a feed targets a binding its stage already binds (§2) |
| `fused-stage-unanchored` | `validate_manifest` | a DAG stage reads no chain, no source and no feed |
| `fused-input-missing` | `cook_tool` | a declared input of a fused tool was not passed |
| `fused-input-extent` | `cook_tool` | an input's `[B,H,W]` differs from the source's, or cannot be read |

The constants are `tex_tool.REFUSE_FUSED_*`.

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
- **Fused with fed inputs**: first every declared input must be passed (`fused-input-missing`)
  and every input's `shape[:3]` must equal the source's exactly — batch, height and width; no
  batch broadcast, no singleton axis (`fused-input-extent`). A promised input is resolved before
  that check (E7007 if it has not landed). Then each fed tensor is written into the bindings it
  names — a copy of `graphspec.stages[i].params`, or the terminal bindings — and the cook is the
  unchanged fused call above.

Because the fused path *is* the region path, the tool cook equals the unfused stage-by-stage
cook by the same construction FUS-3 pins (the round-trip oracle in §7 asserts it for a tool).

**Why co-extent is a refusal, not a convenience.** Fused, a fed input joins the one grid the
whole program cooks on (the per-axis max over the bindings it reads); cooked stage by stage,
each stage grids on its own inputs. A batch or singleton axis that broadcasts harmlessly unfused
therefore changes an *upstream* stage's pixels fused — silently, with the same output shape and
no error. Measured on random inputs: a `B=4` input beside a `B=1` source moved a stage reading
`fi` by maxdiff 2.99; a one-row `[1,1,W,C]` source beside a full frame moved a stage reading `v`
by 1.00. A tool without fed inputs cannot meet this, because every stage anchors on its one source.

---

## 5. Tool = compilation unit (TOOL-3)

Publishing a fused tool should make it *faster* than its parts (strategic bet #6): the
promoted-param signature is fixed, so the fused artifact can be warm-compiled once.

`tex_tool.tool_warm_keys(manifest) -> list[str]` derives the fused chain's value-independent
fingerprint (`tex_fusion.fused_fingerprint`) **at install time, by re-fingerprinting the
inline stage code** — the loader already has the sources in hand. The key is **never carried
in the file** (ENG-5: fingerprints are deliberately unstable across TEX versions; a stored
one would be wrong after any TEX update). A single-stage tool's key is `TEXCache.fingerprint`.
A fused tool with fed inputs derives its keys with every `IMAGE` input sampled at the same
channel count — one RGB key and one RGBA key, not every combination; a cook that mixes RGB and
RGBA inputs misses the warm cache and compiles.

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
`tooltip` and `options` (§2) are capped literal strings — never parsed as TEX and never
reaching the emitter — so a host that renders one (a tooltip, a combo widget) owns the same
output-encoding duty it owes any other untrusted string field; TEX's guarantee here is the
size/shape cap alone.

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
(mirrors `tex_fusion._MAX_FUSED_REGION_STAGES = 16`), `MAX_PROMOTED_PARAMS`, `MAX_STAGE_CODE_BYTES`,
and, for the two structured metadata keys (§2), `MAX_TOOLTIP_CHARS`, `MAX_OPTIONS`,
`MAX_OPTION_CHARS`; an input's `feeds` is at most `MAX_STAGES` pairs. These bound parse/compile cost; they do **not** bound *cook* cost — a valid tool can still
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
| **derivation** | `test_tool_metadata_tooltip_options` | `tooltip` + a labelled-choice `options` list accepted and round-tripped through `to_dict()`/`tool_summary()` |
| **derivation** | `test_tool_input_optional` | `inputs[*].optional` round-trips true/false; an absent key and an unrecognised one stay out of the dict |
| **canary** | `test_tool_manifest_byte_identity` | every stock `.textool` + two representative older manifests still give byte-identical `to_dict()`/`tool_summary()`/written bytes |
| **canary** | `test_tool_js_publish_filter_pin` | the publish-menu JS still forwards only the original five metadata keys (forwarding the new ones is a separate, later frontend change) |
| **differential oracle** | `test_tool_fused_feeds_roundtrip_unfused` | a fused tool with a fed input — into the terminal (Blur → Merge), an upstream linear stage, a DAG stage — cooks == its stages cooked one by one (CPU + CUDA), and the cook leaves the manifest unchanged |
| **differential oracle** | `test_tool_fused_feeds_codegen_parity` | invariant #2 on those fused programs: interpreter == codegen, asserting codegen actually served |
| **refusal + negative control** | `test_tool_fused_feeds_extent_refusal` | a `B=4` input beside `B=1`, a strip source, an H mismatch → `fused-input-extent` with the engine never called; the first two, handed to the engine anyway, cook silently wrong; missing / non-tensor / promised inputs |
| **canary** | `test_tool_fused_feeds_rejects` | every malformed, colliding or unanchored `feeds` manifest is refused with its code and offending input |
| **canary** | `test_tool_fused_input_refusals_unchanged` | a fused tool without `feeds`: the one-input refusals keep their text and type; `fused-input-count` and `fused-input-missing` ride along as codes |
| **derivation** | `test_tool_fused_feeds_manifest_roundtrip` | `feeds` round-trips `load_tool`/`to_dict()`/`tool_summary()`/`write_tool`; preflight ok; the warm keys include an RGB and an RGBA cook's keys |
| **derivation** | `test_tool_fused_feeds_rekey` | an RGB then an RGBA fed input compile under distinct fused keys, each cook == stage-by-stage |
| **canary** | `test_tool_no_feeds_is_pre_feeds_identical` | a tool without `feeds` (the stock tools, the tooltip/options/optional manifests, two older shapes) serialises, hands `tex_engine.cook` the same call, cooks the same pixels and derives the same warm keys as before `feeds` existed |

---

## 8. v1 scope, honestly recorded

- **One external image source per fused tool, plus opt-in fed inputs** (§2 `feeds`) that must be
  co-extent with it at cook (§4). A merge that needs no chain is still a *single-stage* tool
  (Merge reads `@A`/`@B`). Declined, each with what reopens it:
  - *Several producers in graph fusion* — `tex_fusion`'s region detector keeps its one-producer
    rule; the reasons are recorded in `DEVELOPMENT.md` §"Rejected design decisions". A tool author
    routes a second input explicitly; the ComfyUI graph does not fuse one.
  - *Broadcasting a fed input* over a batch or singleton axis — refused, because fused it changes
    pixels silently (§4). Reopens with the gather-source deferral in `DEVELOPMENT.md` (a binding
    read only through sampling is not a co-extent participant).
  - *`LATENT` fed inputs* — no fused tool exercises the latent axis; reopens when a host names one.
  - *A GraphSpec key for extra inputs* — a `GRAPHSPEC_SCHEMA` bump for what stage params already carry.
  - *A code on every `TEXToolError`* — only the fused-input refusals carry one; the rest is its own item.
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
