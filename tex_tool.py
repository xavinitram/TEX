"""TOOL-1..5 — the `.textool` bundling format.

A `.textool` is a self-contained UTF-8 JSON bundle of TEX code plus a UI (promoted
params): the compositor's gizmo / macro / HDA. It carries EVERY stage's source inline —
sharing a tool is sharing one plaintext file, and no program's compilation ever resolves
an external name (this is deliberately NOT the rejected cross-node import system; no
by-name tool nesting in v1). Design + threat model: docs/tools.md.

Two shapes, both `.textool`:
  * single-stage — one TEX program with promoted params (the four stock exemplars, and any
    multi-input node like Merge). Cooks as a plain program.
  * fused — a linear/DAG chain with ONE external image source (the fusion-region model),
    plus — opt-in — further external inputs, each naming the stage bindings it is written
    into (`inputs[*].feeds`) and required to share the source's [B,H,W] at cook. Stores the
    GraphSpec `region_to_collapse_plan` emits and `engine.cook` consumes, so the tool layer
    is a thin pass-through over the FUS-3-proven fused path.

This module is host-agnostic (no comfy imports); heavy siblings (tex_engine, torch,
tex_fusion, tex_marshalling) are imported lazily so `import tex_tool` stays cheap.

TOOL-5: a downloaded `.textool` is untrusted input to a code generator, so EVERY structural
check (schema, sizes, metadata shape) happens BEFORE any TEX source is parsed or any code
generated, and install is validate-only by default (no compile without explicit consent).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any

# The manifest FORMAT version (distinct from tex_fusion.GRAPHSPEC_SCHEMA, which versions
# the embedded fused payload). Bump only when an older reader would MISREAD a manifest.
TEXTOOL_SCHEMA = 1

# Stable reason codes carried as `TEXToolError.code` by the fused-input refusals (None on every
# other TEXToolError). A host keys on these strings — to word a refusal in its own language, say;
# the message beside a code may be reworded, a code may not (tex_checkpoint's REFUSE_* contract).
REFUSE_FUSED_INPUT_COUNT = "fused-input-count"            # >1 input, and none declares `feeds`
REFUSE_FUSED_INPUT_UNROUTED = "fused-input-unrouted"      # a non-source input declares no `feeds`
REFUSE_FUSED_FEED_INVALID = "fused-feed-invalid"          # a malformed or unsupported `feeds`
REFUSE_FUSED_FEED_COLLISION = "fused-feed-collision"      # a feed lands on a name its stage binds
REFUSE_FUSED_STAGE_UNANCHORED = "fused-stage-unanchored"  # a DAG stage reads no chain/source/feed
REFUSE_FUSED_INPUT_MISSING = "fused-input-missing"        # cook: a declared input was not passed
REFUSE_FUSED_INPUT_EXTENT = "fused-input-extent"          # cook: an input's [B,H,W] != the source's

# TOOL-5 resource limits — they bound PARSE/COMPILE cost, never COOK cost (a valid tool can
# still request an 8K gauss_blur; that residual is the host's memory-budget duty, docs/tools.md §6-D).
MAX_TOOL_BYTES = 512 * 1024          # a manifest larger than this is rejected unparsed
MAX_STAGES = 16                      # mirrors tex_fusion._MAX_FUSED_REGION_STAGES
MAX_PROMOTED_PARAMS = 128
MAX_STAGE_CODE_BYTES = 64 * 1024
MAX_TOOLTIP_CHARS = 1024             # metadata['tooltip'] string length cap
MAX_OPTIONS = 256                    # metadata['options'] entry-count cap
MAX_OPTION_CHARS = 128               # each metadata['options'][i] string length cap

_TYPE_HINTS = {"f", "i", "s", "b", "c", "v2", "v3", "v4"}   # LANG-1 ParamDecl type_hints
_CONTEXTS = {"generator", "filter", "transition", "keyer"}  # TOOL-4 context tags
_META_KEYS = {"min", "max", "step", "precision", "label",   # LANG-1 recognised widget keys
              "tooltip", "options"}    # host-rendered hint text + a labelled-choice list
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TOOL_STORE = "tools"                # <user_dir>/tex_wrangle/tools/


class TEXToolError(ValueError):
    """A `.textool` is malformed, unsupported, or fails to install/preflight.

    A ValueError subclass so the CLI's existing except-tuple catches it (tex_cli.main).
    `code` is one of the REFUSE_FUSED_* constants on a fused-input refusal and None on every
    other one; `input` names the offending input when one input is to blame. Both are DATA
    beside the message: `str(e)` and `e.args` are exactly what they were before either existed."""

    def __init__(self, *args, code: str | None = None, input: str | None = None):
        super().__init__(*args)
        self.code = code
        self.input = input


# ── the manifest ──────────────────────────────────────────────────────────────
@dataclass
class PromotedParam:
    """One tool widget: an external name bound to a `$param` inside one stage."""
    name: str                    # external widget name
    internal: str                # the $param name inside its stage
    stage: Any = None            # None/"terminal" = single-stage prog / fused terminal; int = graphspec.stages index
    type: str = "f"              # LANG-1 type_hint
    default: Any = 0.0
    metadata: dict = field(default_factory=dict)   # {min,max,step,precision,label} literals

    def to_dict(self) -> dict:
        return {"name": self.name, "internal": self.internal, "stage": self.stage,
                "type": self.type, "default": self.default, "metadata": dict(self.metadata)}


@dataclass
class ToolManifest:
    name: str
    tool_version: str
    tex_language: str
    min_engine: str
    category: str
    context: str
    doc: str
    author: str
    inputs: list                 # [{"name","type"}] external image/mask inputs
    promoted_params: list        # [PromotedParam]
    outputs: list = field(default_factory=list)   # [{"name","type"}] a host wires these when instancing
    code: str | None = None      # single-stage program source
    graphspec: dict | None = None            # fused: the prepare_fused GraphSpec
    terminal_code: str | None = None         # fused: terminal stage source
    terminal_image_input: str | None = None  # fused: socket binding carrying the source
    terminal_params: dict | None = None      # fused: baked terminal knob values
    manifest_schema: int = TEXTOOL_SCHEMA
    warnings: list = field(default_factory=list)   # non-fatal advisories (language pin, ...)

    @property
    def is_fused(self) -> bool:
        return self.graphspec is not None

    def to_dict(self) -> dict:
        """Serialise back to the on-disk JSON shape (round-trips load_tool)."""
        d = {
            "manifest_schema": self.manifest_schema,
            "name": self.name,
            "tool_version": self.tool_version,
            "tex_language": self.tex_language,
            "min_engine": self.min_engine,
            "category": self.category,
            "context": self.context,
            "doc": self.doc,
            "author": self.author,
            "inputs": [dict(i) for i in self.inputs],
            "outputs": [dict(o) for o in self.outputs],
            "promoted_params": [p.to_dict() for p in self.promoted_params],
        }
        if self.is_fused:
            d["graphspec"] = self.graphspec
            d["terminal_code"] = self.terminal_code
            d["terminal_image_input"] = self.terminal_image_input
            d["terminal_params"] = dict(self.terminal_params or {})
        else:
            d["code"] = self.code
        return d


# ── schema validation (TOOL-5-B: BEFORE any compile) ────────────────────────────
def _req(raw: dict, key: str, typ):
    if key not in raw:
        raise TEXToolError(f".textool is missing required field '{key}'")
    v = raw[key]
    if not isinstance(v, typ):
        raise TEXToolError(f".textool field '{key}' must be {typ.__name__}, got {type(v).__name__}")
    return v


def _validate_promoted(raw_list) -> list[PromotedParam]:
    if not isinstance(raw_list, list):
        raise TEXToolError("'promoted_params' must be a list")
    if len(raw_list) > MAX_PROMOTED_PARAMS:
        raise TEXToolError(f"too many promoted params ({len(raw_list)} > {MAX_PROMOTED_PARAMS})")
    out = []
    seen = set()
    for i, p in enumerate(raw_list):
        if not isinstance(p, dict):
            raise TEXToolError(f"promoted_params[{i}] must be an object")
        name = p.get("name")
        internal = p.get("internal", name)
        if not isinstance(name, str) or not _IDENT_RE.match(name):
            raise TEXToolError(f"promoted_params[{i}].name must be a valid identifier")
        if not isinstance(internal, str) or not _IDENT_RE.match(internal):
            raise TEXToolError(f"promoted_params[{i}].internal must be a valid identifier")
        if name in seen:
            raise TEXToolError(f"duplicate promoted param name '{name}'")
        seen.add(name)
        thint = p.get("type", "f")
        if thint not in _TYPE_HINTS:
            raise TEXToolError(f"promoted_params[{i}].type '{thint}' is not a valid LANG-1 type hint")
        stage = p.get("stage")
        # bool is an int subclass in Python -- reject JSON true/false so it can't index a stage.
        if not (stage is None or stage == "terminal"
                or (isinstance(stage, int) and not isinstance(stage, bool))):
            raise TEXToolError(f"promoted_params[{i}].stage must be null, an int, or 'terminal'")
        meta = p.get("metadata", {}) or {}
        if not isinstance(meta, dict):
            raise TEXToolError(f"promoted_params[{i}].metadata must be an object")
        for mk, mv in meta.items():
            if mk not in _META_KEYS:
                raise TEXToolError(f"promoted_params[{i}].metadata has unknown key '{mk}'")
            if mk == "tooltip":
                if not isinstance(mv, str):
                    raise TEXToolError(f"promoted_params[{i}].metadata['tooltip'] must be a string")
                if len(mv) > MAX_TOOLTIP_CHARS:
                    raise TEXToolError(f"promoted_params[{i}].metadata['tooltip'] exceeds "
                                       f"{MAX_TOOLTIP_CHARS} characters")
            elif mk == "options":
                if thint != "i":
                    raise TEXToolError(f"promoted_params[{i}].metadata['options'] is only valid "
                                       f"on type 'i' (got '{thint}')")
                if not isinstance(mv, list):
                    raise TEXToolError(f"promoted_params[{i}].metadata['options'] must be a list")
                if not (1 <= len(mv) <= MAX_OPTIONS):
                    raise TEXToolError(f"promoted_params[{i}].metadata['options'] must have "
                                       f"between 1 and {MAX_OPTIONS} entries")
                for oi, ov in enumerate(mv):
                    if not isinstance(ov, str) or not ov:
                        raise TEXToolError(f"promoted_params[{i}].metadata['options'][{oi}] "
                                           f"must be a non-empty string")
                    if len(ov) > MAX_OPTION_CHARS:
                        raise TEXToolError(f"promoted_params[{i}].metadata['options'][{oi}] "
                                           f"exceeds {MAX_OPTION_CHARS} characters")
            elif not isinstance(mv, (int, float, str)) or isinstance(mv, bool):
                raise TEXToolError(f"promoted_params[{i}].metadata['{mk}'] must be a literal scalar")
        if "options" in meta:
            # One source of truth: a labelled-choice widget's min/max/step, if given at all,
            # must agree with the option list instead of racing it, and a given default must
            # be a valid index into it. All of it optional -- an old manifest with no
            # 'options' never reaches this block, so it cannot change what it validates.
            n = len(meta["options"])
            if "min" in meta and meta["min"] != 0:
                raise TEXToolError(f"promoted_params[{i}].metadata['min'] must be 0 to match "
                                   f"'options'")
            if "max" in meta and meta["max"] != n - 1:
                raise TEXToolError(f"promoted_params[{i}].metadata['max'] must be {n - 1} to "
                                   f"match 'options'")
            if "step" in meta and meta["step"] != 1:
                raise TEXToolError(f"promoted_params[{i}].metadata['step'] must be 1 to match "
                                   f"'options'")
            if "default" in p:
                dv = p["default"]
                if isinstance(dv, bool) or not isinstance(dv, int) or not (0 <= dv < n):
                    raise TEXToolError(f"promoted_params[{i}].default must be an int in "
                                       f"[0, {n}) to match 'options'")
        out.append(PromotedParam(name=name, internal=internal, stage=stage, type=thint,
                                 default=p.get("default", 0.0), metadata=dict(meta)))
    return out


def _validate_inputs(raw_list) -> list:
    # An empty list is legal for a single-stage GENERATOR (a program that reads no @inputs,
    # e.g. `@OUT = vec4(u, v, 0, 1);`). A fused tool's one-source requirement is enforced
    # separately in validate_manifest's cross-field checks.
    if not isinstance(raw_list, list):
        raise TEXToolError("'inputs' must be a list")
    out = []
    for i, inp in enumerate(raw_list):
        if not isinstance(inp, dict) or "name" not in inp:
            raise TEXToolError(f"inputs[{i}] must be an object with a 'name'")
        nm = inp["name"]
        if not isinstance(nm, str) or not _IDENT_RE.match(nm):
            raise TEXToolError(f"inputs[{i}].name must be a valid identifier")
        entry = {"name": nm, "type": str(inp.get("type", "IMAGE"))}
        # "optional" is host UI advice only (which inputs may go unwired) -- TEX binds
        # nothing for an absent input either way (docs/tools.md §2), so it's copied in ONLY
        # when present: an old manifest with no such key rebuilds this entry byte-identically.
        # Any OTHER unknown inputs[*] key keeps silently dropping here, same as before.
        if "optional" in inp:
            opt = inp["optional"]
            if not isinstance(opt, bool):
                raise TEXToolError(f"inputs[{i}].optional must be a bool")
            entry["optional"] = opt
        # "feeds" routes an extra input of a FUSED tool into named stage bindings. Copied only
        # when present, like "optional", so a manifest that never declares it rebuilds each entry
        # exactly as before. Shape only here; which stages exist and what each already binds is
        # validate_manifest's cross-field check.
        if "feeds" in inp:
            entry["feeds"] = _validate_feeds_shape(i, nm, inp["feeds"])
        out.append(entry)
    return out


def _validate_feeds_shape(i: int, name: str, raw) -> list:
    """`inputs[i].feeds`: 1..MAX_STAGES `[stage, binding]` pairs — `stage` an int (a
    graphspec.stages index) or "terminal" (PromotedParam.stage's vocabulary), `binding` an
    identifier. Returned as a list of lists, the JSON shape."""
    def bad(msg):
        return TEXToolError(msg, code=REFUSE_FUSED_FEED_INVALID, input=name)
    if not isinstance(raw, (list, tuple)) or not (1 <= len(raw) <= MAX_STAGES):
        raise bad(f"inputs[{i}].feeds must be a list of 1 to {MAX_STAGES} [stage, binding] pairs")
    out = []
    for k, pair in enumerate(raw):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise bad(f"inputs[{i}].feeds[{k}] must be a [stage, binding] pair")
        stage, binding = pair
        # bool is an int subclass -- reject JSON true/false so it can't index a stage.
        if not (stage == "terminal" or (isinstance(stage, int) and not isinstance(stage, bool))):
            raise bad(f"inputs[{i}].feeds[{k}] stage must be an int or 'terminal'")
        if not isinstance(binding, str) or not _IDENT_RE.match(binding):
            raise bad(f"inputs[{i}].feeds[{k}] binding must be a valid identifier")
        out.append([stage, binding])
    return out


def _validate_outputs(raw_list) -> list:
    """Output ports a host wires when instancing the tool. May be empty (unusual). Each is
    {name, type}; names need not be identifiers (`@OUT` is the common single output, but a
    multi-output tool like Vignette uses `darkened`/`vignette_mask`)."""
    if raw_list is None:
        return []
    if not isinstance(raw_list, list):
        raise TEXToolError("'outputs' must be a list")
    out = []
    for i, o in enumerate(raw_list):
        if not isinstance(o, dict) or not isinstance(o.get("name"), str):
            raise TEXToolError(f"outputs[{i}] must be an object with a string 'name'")
        out.append({"name": o["name"], "type": str(o.get("type", "IMAGE"))})
    return out


def _source_injection_points(gs: dict, n_stages: int):
    """Where `tex_fusion._stages_from_spec` writes the source on a DAG spec, as a set of
    `(stage | "terminal", binding)` — read the way it reads them: schema-2 `source_injections`
    over the schema-1 `source_stage`/`source_binding` scalars, a falsy binding injecting nothing,
    index `n_stages` meaning the terminal. A linear spec injects into stage 0's `image_input`,
    which is already that stage's own binding, so it contributes nothing here. None when the
    injection list cannot be read: the cook raises on it anyway, so the checks that need it
    stand down rather than guess."""
    if not gs.get("dag"):
        return set()
    raw_inj = gs.get("source_injections")
    try:
        points = ([(int(s), b) for (s, b) in raw_inj] if raw_inj
                  else [(gs.get("source_stage", 0), gs.get("source_binding"))])
    except (TypeError, ValueError):
        return None
    out = set()
    for s, b in points:
        if not b:
            continue
        if not isinstance(b, str):
            return None
        out.update((j, b) for j in range(n_stages) if j == s)
        if s == n_stages:
            out.add(("terminal", b))
    return out


def _stage_bound_names(raw: dict, promoted: list, injections) -> dict:
    """`{stage | "terminal": {binding: what already binds it}}` for a fused manifest — every name
    a feed may not take, because the engine (or the promoted-param writer) already writes that
    binding and one of the two writes would silently win."""
    gs = raw["graphspec"]
    stages = gs["stages"]
    dag = bool(gs.get("dag"))
    bound: dict = {j: {} for j in range(len(stages))}
    bound["terminal"] = {raw["terminal_image_input"]: "the source socket (terminal_image_input)"}
    for j, st in enumerate(stages):
        if dag:
            if isinstance(st.get("chain_inputs"), dict):
                for b in st["chain_inputs"]:
                    bound[j].setdefault(b, "its chain input")
        else:
            bound[j].setdefault(st["image_input"], "its chain input" if j else "the source")
        if isinstance(st.get("params"), dict):
            for b in st["params"]:
                bound[j].setdefault(b, "a baked param")
    if dag and isinstance(gs.get("terminal_chain_inputs"), dict):
        for b in gs["terminal_chain_inputs"]:
            bound["terminal"].setdefault(b, "its chain input")
    for key, b in injections or ():
        bound[key].setdefault(b, "a source injection point")
    if isinstance(raw.get("terminal_params"), dict):
        for b in raw["terminal_params"]:
            bound["terminal"].setdefault(b, "a baked param")
    for pp in promoted:                      # stage range was validated before this runs
        key = "terminal" if pp.stage in (None, "terminal") else pp.stage
        bound[key].setdefault(pp.internal, f"promoted param '{pp.name}'")
    return bound


def _validate_fused_feeds(raw: dict, inputs: list, promoted: list) -> None:
    """The cross-field half of `inputs[*].feeds` — a fused tool with more than one external input.

    The source (`terminal_image_input`) still travels the one way the engine splices it; every
    OTHER input names the stage bindings it is written into, the same place a promoted value
    goes, so the fused program cooks through the unchanged GraphSpec path. What can only be
    known at cook — that the inputs share one [B,H,W] — is cook_tool's refusal. Shape-only,
    before any TEX is parsed (TOOL-5)."""
    gs = raw["graphspec"]
    stages = gs["stages"]
    n = len(stages)
    tii = raw["terminal_image_input"]
    if tii not in {i["name"] for i in inputs}:
        raise TEXToolError(f"terminal_image_input '{tii}' is not a declared input")
    injections = _source_injection_points(gs, n)
    bound = _stage_bound_names(raw, promoted, injections)
    targets: dict = {}                       # (stage | "terminal", binding) -> feeding input
    for inp in inputs:
        nm = inp["name"]
        if nm == tii:
            if "feeds" in inp:
                raise TEXToolError(f"input '{nm}' is the fused tool's source (terminal_image_input) "
                                   f"and may not declare 'feeds'",
                                   code=REFUSE_FUSED_FEED_INVALID, input=nm)
            if inp.get("optional"):
                raise TEXToolError(f"a fused tool's source input '{nm}' may not be marked "
                                   f"'optional' (the engine requires it to splice the chain)")
            continue
        if "feeds" not in inp:
            raise TEXToolError(f"input '{nm}' is not routed into the fused chain: every input of a "
                               f"fused tool other than its source must declare 'feeds'",
                               code=REFUSE_FUSED_INPUT_UNROUTED, input=nm)
        if inp["type"].upper() not in ("IMAGE", "MASK"):
            raise TEXToolError(f"input '{nm}' is {inp['type']}; a fed input must be IMAGE or MASK",
                               code=REFUSE_FUSED_FEED_INVALID, input=nm)
        if inp.get("optional"):
            raise TEXToolError(f"fed input '{nm}' may not be marked 'optional' (a tool with fed "
                               f"inputs cooks only when every one is passed)",
                               code=REFUSE_FUSED_FEED_INVALID, input=nm)
        for stage, binding in inp["feeds"]:
            if stage != "terminal" and not (0 <= stage < n):
                raise TEXToolError(f"input '{nm}' feeds stage {stage}, out of range for {n} "
                                   f"upstream stage(s)", code=REFUSE_FUSED_FEED_INVALID, input=nm)
            where = "the terminal" if stage == "terminal" else f"stage {stage}"
            if (stage, binding) in targets:
                raise TEXToolError(f"input '{nm}' feeds @{binding} on {where}, which input "
                                   f"'{targets[(stage, binding)]}' already feeds",
                                   code=REFUSE_FUSED_FEED_COLLISION, input=nm)
            if binding in bound[stage]:
                raise TEXToolError(f"input '{nm}' feeds @{binding} on {where}, which is already "
                                   f"{bound[stage][binding]}",
                                   code=REFUSE_FUSED_FEED_COLLISION, input=nm)
            targets[(stage, binding)] = nm
    # tex_fusion's region detector never folds a node with no image input into a region: such a
    # stage would adopt the fused program's extent instead of cooking at its own. A DAG spec can
    # still spell one, so on this branch it is refused here; the terminal always reads a chain.
    _check_dag_stages_anchored(gs, stages, injections, targets)


def _check_dag_stages_anchored(gs: dict, stages: list, injections, fed_targets) -> None:
    """TRK-9: refuse a DAG stage that reads no chain, no source injection and no fed input.

    `tex_fusion`'s region detector (`_grow_region`) never folds a node with no image input
    into a region — rule ③ — so a LIVE ComfyUI graph can never produce this shape. A
    hand-authored DAG manifest can still spell one (nothing stopped it): standalone, such a
    stage has no shared grid to size its output by, so its spatial builtins default to 0-dim
    and it cooks to a bare rank-1 vector instead of `[B,H,W,C]`; spliced into the terminal
    alongside a real image, that vector hits a plain torch broadcast `RuntimeError` — not a
    named TEX check. Originally only reachable from `_validate_fused_feeds` (the multi-input
    `feeds` path), so a single-source manifest (schema-1 `source_stage`/`source_binding`, or
    `source_injections`, with no OTHER input feeding anything) skipped it entirely — exactly
    the shape this row's own minimal repro used. `fed_targets` iterates `(stage, binding)`
    pairs, exactly like `injections` — a dict's keys (`_validate_fused_feeds`'s `targets`) or an
    empty container (the no-`feeds` path) both work. `injections` is `None` when
    `_source_injection_points` could not read them, in which case cook raises on that shape
    anyway and this check stands down rather than guess."""
    if not gs.get("dag") or injections is None:
        return
    read = {s for s, _ in injections} | {s for s, _ in fed_targets}
    for j, st in enumerate(stages):
        if not st.get("chain_inputs") and j not in read:
            raise TEXToolError(f"graphspec.stages[{j}] reads no chain, no source and no fed "
                               f"input, so it would adopt the fused program's extent instead "
                               f"of cooking at its own", code=REFUSE_FUSED_STAGE_UNANCHORED)


def validate_manifest(raw: dict) -> dict:
    """TOOL-5-B: prove the manifest shape well-formed BEFORE any TEX source is parsed.

    Raises TEXToolError on any structural violation. Does NOT compile, parse TEX, or run
    codegen — a hostile manifest is rejected here, on shape alone. Returns the parsed
    {inputs, outputs, promoted} lists so load_tool need not re-validate; write_tool ignores it."""
    if not isinstance(raw, dict):
        raise TEXToolError(".textool must be a JSON object")
    # TOOL-5: bound the WHOLE manifest, not only per-stage code. _read_capped caps the file-load
    # path, but the dict/route path (POST /tex_wrangle/publish_tool) reaches here directly, so a
    # manifest with a huge NON-code field — millions of graphspec param keys, a giant dag — would
    # otherwise be unbounded (per-stage-code + stage-count limits don't see it). One serialized-size
    # check bounds total node/param count for every entry point (route, dict, and file alike).
    try:
        total_bytes = len(json.dumps(raw).encode("utf-8"))
    except (TypeError, ValueError):
        raise TEXToolError(".textool is not JSON-serialisable")
    if total_bytes > MAX_TOOL_BYTES:
        raise TEXToolError(f".textool is {total_bytes} bytes (> {MAX_TOOL_BYTES} limit)")
    ms = raw.get("manifest_schema", TEXTOOL_SCHEMA)
    if not isinstance(ms, int) or ms < 1:
        raise TEXToolError(f".textool has an invalid manifest_schema ({ms!r})")
    if ms > TEXTOOL_SCHEMA:
        raise TEXToolError(f"this .textool was built by a newer TEX (manifest_schema {ms}; "
                           f"this build reads {TEXTOOL_SCHEMA}). Update TEX to install it.")
    _req(raw, "name", str)
    _req(raw, "tex_language", str)

    # Key fused-ness on PRESENCE (not type) so validation and ToolManifest.is_fused
    # (graphspec is not None) can never disagree — a `code` + non-dict `graphspec` must not
    # slip through as single-stage and then take the fused path on a non-dict graphspec.
    has_code = isinstance(raw.get("code"), str)
    gs_present = raw.get("graphspec") is not None
    if has_code and gs_present:
        raise TEXToolError("a .textool must carry exactly one of 'code' (single-stage) "
                           "or 'graphspec'+'terminal_code' (fused), not both")
    if not has_code and not gs_present:
        raise TEXToolError("a .textool must carry either 'code' (single-stage) "
                           "or 'graphspec'+'terminal_code' (fused)")
    if gs_present and not isinstance(raw["graphspec"], dict):
        raise TEXToolError("'graphspec' must be an object")
    has_fused = gs_present
    if has_code:
        if len(raw["code"].encode("utf-8")) > MAX_STAGE_CODE_BYTES:
            raise TEXToolError("single-stage 'code' exceeds the size limit")
    else:
        gs = raw["graphspec"]
        stages = gs.get("stages")
        if not isinstance(stages, list) or not stages:
            raise TEXToolError("fused 'graphspec.stages' must be a non-empty list")
        if len(stages) + 1 > MAX_STAGES:      # +1 for the terminal
            raise TEXToolError(f"fused tool has too many stages (> {MAX_STAGES})")
        tc = raw.get("terminal_code")
        if not isinstance(tc, str):
            raise TEXToolError("fused tool requires a string 'terminal_code'")
        if not isinstance(raw.get("terminal_image_input"), str):
            raise TEXToolError("fused tool requires a string 'terminal_image_input'")
        is_dag = bool(gs.get("dag"))
        for j, st in enumerate(stages):
            if not isinstance(st, dict) or not isinstance(st.get("code"), str):
                raise TEXToolError(f"graphspec.stages[{j}] must be an object with string 'code'")
            if len(st["code"].encode("utf-8")) > MAX_STAGE_CODE_BYTES:
                raise TEXToolError(f"graphspec.stages[{j}].code exceeds the size limit")
            # A LINEAR spec's _stages_from_spec reads st['image_input'] UNCONDITIONALLY (the
            # socket each stage reads as the chain) — a missing one is a raw KeyError at cook, the
            # exact class of crash the cross-field note below promises a schema-valid manifest can
            # never reach. (A DAG spec routes via chain_inputs instead, so image_input is optional.)
            if not is_dag and not isinstance(st.get("image_input"), str):
                raise TEXToolError(f"graphspec.stages[{j}] must declare a string 'image_input' "
                                   f"(the socket it reads as the chain)")
        if len(tc.encode("utf-8")) > MAX_STAGE_CODE_BYTES:
            raise TEXToolError("'terminal_code' exceeds the size limit")

    ctx = raw.get("context", "filter")
    if ctx not in _CONTEXTS:
        raise TEXToolError(f"context '{ctx}' is not one of {sorted(_CONTEXTS)}")
    inputs = _validate_inputs(raw.get("inputs", [{"name": "image", "type": "IMAGE"}]))
    outputs = _validate_outputs(raw.get("outputs"))
    promoted = _validate_promoted(raw.get("promoted_params", []))

    # Cross-field checks — so a schema-valid manifest can never crash cook/preflight with a
    # raw IndexError/KeyError/ValueError (or, worse, silently write a promoted value into the
    # WRONG stage via a negative index). Everything below is shape-only, still before compile.
    input_names = {i["name"] for i in inputs}
    fed = [i for i in inputs if "feeds" in i]
    if has_fused:
        n_stages = len(raw["graphspec"]["stages"])
        for pp in promoted:
            if isinstance(pp.stage, int) and not (0 <= pp.stage < n_stages):
                raise TEXToolError(f"promoted param '{pp.name}' targets stage {pp.stage}, "
                                   f"out of range for {n_stages} upstream stage(s)")
        tp = raw.get("terminal_params")
        if tp is not None and not isinstance(tp, dict):
            raise TEXToolError("'terminal_params' must be an object (or absent)")
        if not fed:
            # The engine pops the source from terminal_bindings[terminal_image_input]; a fused tool
            # that routes no other input therefore has exactly ONE external image source, and its
            # socket must be a declared input.
            if len(inputs) != 1:
                raise TEXToolError("a fused tool must declare exactly one external input "
                                   "(the single fusion source)", code=REFUSE_FUSED_INPUT_COUNT)
            if inputs[0].get("optional"):
                raise TEXToolError("a fused tool's sole external input may not be marked "
                                   "'optional' (the engine requires it to splice the chain)")
            if raw["terminal_image_input"] not in input_names:
                raise TEXToolError(f"terminal_image_input '{raw['terminal_image_input']}' is not a "
                                   f"declared input")
            # TRK-9: a DAG spec's OTHER route to this same check — no `feeds`-declared input
            # exists on this branch (that's what put us here), so `_validate_fused_feeds`
            # (the only caller of `_check_dag_stages_anchored` before this row) never runs;
            # a hand-authored single-source DAG manifest with an unanchored stage sailed
            # through validation uncaught.
            _check_dag_stages_anchored(gs, stages, _source_injection_points(gs, n_stages), set())
        else:
            _validate_fused_feeds(raw, inputs, promoted)
        gs_tii = raw["graphspec"].get("terminal_image_input")
        if gs_tii is not None and gs_tii != raw["terminal_image_input"]:
            raise TEXToolError("graphspec.terminal_image_input disagrees with the manifest's")
    else:
        if fed:
            nm = fed[0]["name"]
            raise TEXToolError(f"input '{nm}' declares 'feeds', which only a fused tool reads "
                               f"(a single-stage tool binds every input by its name)",
                               code=REFUSE_FUSED_FEED_INVALID, input=nm)
        # single-stage: all promoted params share the program's flat binding namespace, so an
        # internal that duplicates another (last-write-wins) or collides with an input tensor
        # (the scalar would clobber the image) is a manifest error.
        seen_internal = set()
        for pp in promoted:
            if pp.internal in input_names:
                raise TEXToolError(f"promoted param internal '{pp.internal}' collides with an "
                                   f"input name")
            if pp.internal in seen_internal:
                raise TEXToolError(f"duplicate promoted internal '{pp.internal}' in a "
                                   f"single-stage tool")
            seen_internal.add(pp.internal)
    # Return the parsed lists so load_tool consumes them instead of re-running all three
    # validators (each does per-name/per-internal regex work over up to MAX_PROMOTED_PARAMS).
    # write_tool ignores the return; the TOOL-5 "validate before compile" ordering is unchanged.
    return {"inputs": inputs, "outputs": outputs, "promoted": promoted}


# ── loading ─────────────────────────────────────────────────────────────────────
def _package_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "0.0.0"


def load_tool(path_or_dict) -> ToolManifest:
    """Load + validate a `.textool` into a ToolManifest. TOOL-5 order: parse -> schema
    validation -> language-pin advisory -> engine-version gate. NOTHING is compiled here."""
    if isinstance(path_or_dict, dict):
        raw = path_or_dict
    else:
        data = _read_capped(path_or_dict)
        try:
            raw = json.loads(data)
        except json.JSONDecodeError as e:
            raise TEXToolError(f"'{path_or_dict}' is not valid JSON: {e}")

    parsed = validate_manifest(raw)   # TOOL-5-B: before any TEX source is touched; returns the
                                      # already-parsed inputs/outputs/promoted (validated once).
    from .tex_api import _ver_tuple   # TRK-144: the one definition, not a redefinition here;
                                      # imported unconditionally (not inside the try below) so
                                      # the engine-version gate past it still has a name bound
                                      # even if the LANGUAGE_VERSION import in that try fails.

    warnings = []
    # LANG-3 language-pin advisory (mirrors W7004): does not block.
    try:
        from .tex_api import LANGUAGE_VERSION
        if _ver_tuple(raw["tex_language"]) > _ver_tuple(LANGUAGE_VERSION):
            warnings.append(f"tool targets TEX language {raw['tex_language']} but this build "
                            f"speaks {LANGUAGE_VERSION}; newer constructs may not type-check")
    except Exception:
        pass

    # Engine-version gate: fail at INSTALL, not at cook (roadmap TOOL-4).
    min_engine = str(raw.get("min_engine", "0.0.0"))
    if _ver_tuple(min_engine) > _ver_tuple(_package_version()):
        raise TEXToolError(f"tool '{raw.get('name')}' needs TEX >= {min_engine} "
                           f"(this build is {_package_version()})")

    return ToolManifest(
        name=raw["name"],
        tool_version=str(raw.get("tool_version", "1.0.0")),
        tex_language=raw["tex_language"],
        min_engine=min_engine,
        category=str(raw.get("category", "Uncategorized")),
        context=str(raw.get("context", "filter")),
        doc=str(raw.get("doc", "")),
        author=str(raw.get("author", "")),
        inputs=parsed["inputs"],
        outputs=parsed["outputs"],
        promoted_params=parsed["promoted"],
        code=raw.get("code"),
        graphspec=raw.get("graphspec"),
        terminal_code=raw.get("terminal_code"),
        terminal_image_input=raw.get("terminal_image_input"),
        terminal_params=raw.get("terminal_params") or {},
        manifest_schema=raw.get("manifest_schema", TEXTOOL_SCHEMA),
        warnings=warnings,
    )


def _read_capped(path: str) -> str:
    # Read at most the cap (+1 to detect overflow) instead of stat-then-read: the pre-stat is a
    # TOCTOU (a symlinked/swapped/grown file could report small then read huge) and read() would
    # slurp the whole file regardless. Bounding the read makes MAX_TOOL_BYTES actually enforced.
    with open(path, "rb") as fh:
        raw = fh.read(MAX_TOOL_BYTES + 1)
    if len(raw) > MAX_TOOL_BYTES:
        raise TEXToolError(f"'{path}' exceeds the {MAX_TOOL_BYTES}-byte limit")
    return raw.decode("utf-8")


# ── type inference for a tool's bindings (shared by preflight + warm keys) ─────────
def _zeros_bhwc(c: int):
    """A [1,1,1,C] tensor — the ONLY shape infer_binding_type maps to a vector TEXType
    (a Python list collapses to its element type, i.e. FLOAT)."""
    import torch
    return torch.zeros(1, 1, 1, c)


def _hint_value(hint: str, default):
    """A representative value whose infer_binding_type() matches the promoted param's LANG-1
    type hint, so a promoted param keys/type-checks as the type its widget produces (an int
    slider -> INT, not the FLOAT a coerced default would give).

    This MUST agree with the compiler's canonical hint->type map (type_checker.BINDING_HINT_TYPES):
    `b`->INT (a boolean binds as 0/1, not a float), and `c`/`v2`/`v3`/`v4`->VEC*. Because
    infer_binding_type collapses a Python list to its first element's type (FLOAT), a scalar
    list can NEVER represent a vector binding here — only a 4-D [B,H,W,C] tensor does. Getting
    this wrong silently mis-keys the TOOL-3 warm artifact so the cook never finds it."""
    if hint == "i" or hint == "b":
        return 0                                 # INT (a bool binds as an int, BINDING_HINT_TYPES["b"])
    if hint == "s":
        return str(default) if isinstance(default, str) else ""
    if hint in ("c", "v3"):
        return _zeros_bhwc(3)                     # VEC3
    if hint == "v2":
        return _zeros_bhwc(2)                     # VEC2
    if hint == "v4":
        return _zeros_bhwc(4)                     # VEC4
    return 0.0                                    # f -> FLOAT


def _repr_params(manifest: ToolManifest, *, warm: bool = False) -> dict:
    """{promoted_name: representative value} a WARM/PREFLIGHT derivation feeds so the fused
    fingerprint/preflight infers each param's type.

    PREFLIGHT wants the param's SEMANTIC type (a `c`/`v*` param must type-check as a vector for
    `$col.rgb`), so it uses `_hint_value`. WARM must instead match what the COOK fingerprints:
    the engine keys on `infer_binding_type(RAW value)` BEFORE `_convert_param_value` runs
    (tex_engine.prepare ~874/884/955), so a cook feeds the raw default (a hex/comma STRING for
    `c`/`v*`, and an INT for an `f` param whose JSON default serialized as an int). Using the raw
    default here makes the warm key match the cook's; `_hint_value`'s VEC3/FLOAT would miss."""
    if warm:
        return {pp.name: pp.default for pp in manifest.promoted_params}
    return {pp.name: _hint_value(pp.type, pp.default) for pp in manifest.promoted_params}


def _single_binding_types(manifest: ToolManifest, image_channels: int = 3, *, warm: bool = False) -> dict:
    """{name: TEXType} for a single-stage tool. Inputs by declared socket type. Params: PREFLIGHT
    uses the semantic hint (`_hint_value`, so `$col.rgb` type-checks); WARM uses the RAW default so
    the key matches the cook's ingress-type fingerprint (see _repr_params). `image_channels` picks
    the channel count an IMAGE input is sampled at (3=RGB, 4=RGBA); MASK/LATENT ignore it."""
    from .tex_marshalling import infer_binding_type
    bt = {}
    for inp in manifest.inputs:
        bt[inp["name"]] = infer_binding_type(_placeholder_tensor(inp.get("type"), image_channels))
    for pp in manifest.promoted_params:
        val = pp.default if warm else _hint_value(pp.type, pp.default)
        bt[pp.internal] = infer_binding_type(val)
    return bt


# ── preflight (type-check, never cook) ───────────────────────────────────────────
def _declared_output_mismatch(manifest: ToolManifest, assigned_names) -> list[str]:
    """Advisory(s) when the manifest declares an output the program never assigns. Outputs are the
    host's wiring contract (docs/tools.md), so an unproduced port would wire to nothing. Non-fatal
    and one-directional: a code may legitimately assign MORE than it declares, so only a
    declared-but-unassigned name warns."""
    if not manifest.outputs:
        return []
    assigned = set(assigned_names or [])
    missing = [o["name"] for o in manifest.outputs if o.get("name") not in assigned]
    if not missing:
        return []
    return [f"declared output(s) {missing} are not assigned by the tool's code "
            f"(the code assigns {sorted(assigned)})"]


def preflight_tool(manifest: ToolManifest) -> dict:
    """Type-check the stages WITHOUT cooking. Total (never raises). Returns
    {ok, diagnostics: [dict], stats, output_warnings}. Single-stage -> tex_api.check; fused ->
    tex_fusion.chain_preflight against placeholder shapes."""
    if manifest.is_fused:
        return _preflight_fused(manifest)
    from .tex_api import check, compile as _compile
    # Type-check against the tool's REAL binding types (inputs by socket type, params by hint),
    # not an empty map -- else every @input resolves to VEC4 and a `@image.a` on a VEC3 input
    # false-passes preflight and then fails at cook.
    bt = _single_binding_types(manifest)
    diags = check(manifest.code, bt)
    errs = [d for d in diags if getattr(d, "severity", "error") == "error"]
    ow = []
    if not errs:                                 # cross-check outputs only once the code type-checks
        try:
            ow = _declared_output_mismatch(manifest, _compile(manifest.code, bt).assigned.keys())
        except Exception:
            pass
    return {"ok": not errs, "diagnostics": [d.to_dict() for d in diags],
            "stats": {"stages": 1}, "output_warnings": ow}


def _preflight_fused(manifest: ToolManifest) -> dict:
    """Assemble the fused stages (with placeholder image tensors) and run chain_preflight."""
    try:
        import torch
        from .tex_marshalling import infer_binding_type
        from .tex_fusion import chain_preflight
    except Exception as e:                       # pragma: no cover - torch always present
        return {"ok": False, "diagnostics": [{"code": "E0000", "severity": "error",
                "message": f"preflight unavailable: {e}"}], "stats": None}
    # chain_preflight is total, but the GraphSpec ASSEMBLY before it (_stages_from_spec) can
    # still raise on a graphspec whose internals validate_manifest doesn't fully model (e.g. a
    # linear stage missing image_input). Keep the "never raises" contract by catching it here.
    # Placeholder shaped by the tool's declared source type (a MASK/LATENT-fed fused tool must
    # not preflight against VEC3).
    try:
        stages = _assemble_fused_stages(manifest, _fused_source_placeholder(manifest), _repr_params(manifest),
                                        _fed_placeholders(manifest))
    except Exception as e:
        return {"ok": False, "diagnostics": [{"code": "E9001", "severity": "error",
                "message": f"fused tool graphspec is malformed: {e}"}], "stats": None}
    res = chain_preflight(stages, infer_binding_type)
    diags = []
    ow = []
    if not res.get("ok"):
        diags.append({"code": "E9001", "severity": "error",
                      "message": res.get("error", "fused tool failed to compile"),
                      "stage": res.get("stage_of_error")})
    else:
        ow = _declared_output_mismatch(manifest, (res.get("stats") or {}).get("outputs"))
    return {"ok": res.get("ok", False), "diagnostics": diags, "stats": res.get("stats"),
            "output_warnings": ow}


# ── cooking ──────────────────────────────────────────────────────────────────────
def _resolve_params(manifest: ToolManifest, params: dict | None) -> dict:
    """Map promoted values (or their defaults) onto the flat {internal: value} for a
    single-stage tool. Fused tools use _assemble_fused_stages instead."""
    params = params or {}
    out = {}
    for pp in manifest.promoted_params:
        out[pp.internal] = params.get(pp.name, pp.default)
    return out


def _fused_cook_inputs(manifest: ToolManifest, source, params: dict | None,
                       extras: dict | None = None):
    """The one place that applies promoted values to a fused tool: copy the GraphSpec, write
    each promoted value (or its default) into its target stage, and build the terminal bindings
    (baked terminal params + the source under terminal_image_input). Shared by preflight, cook,
    and warm-key derivation so all three see byte-identical inputs. Structure-shares the graphspec
    (shallow top-level + per-stage dict, params copied only for a written stage) instead of a full
    deepcopy -- this runs every frame of a fused-tool slider drag, and only stages[i]['params'] is
    ever mutated.

    `extras` ({input name: value}) carries the inputs a tool routes with `inputs[*].feeds`; each
    value is written into every binding its input feeds, exactly where a promoted value would go.
    None (every tool that declares no `feeds`) skips that loop entirely."""
    src = manifest.graphspec
    gs = dict(src)
    gs["stages"] = [dict(s) for s in src.get("stages", [])]
    # The engine's _stages_from_spec pops the source via spec['terminal_image_input'] — it reads
    # that key from the GRAPHSPEC, not from the manifest. region_to_collapse_plan omits it from the
    # graphspec (its docstring says so), so make the manifest field authoritative on this copy;
    # otherwise a schema-valid fused tool cooks to a FusionError ("source image reached the
    # terminal empty"). Set on the copy only — manifest.graphspec is never mutated.
    gs["terminal_image_input"] = manifest.terminal_image_input
    term_params = dict(manifest.terminal_params or {})
    written = set()      # stage indices whose params dict this call already copied (owns → mutate in place)
    for pp in manifest.promoted_params:
        val = (params or {}).get(pp.name, pp.default)
        if pp.stage in (None, "terminal"):
            term_params[pp.internal] = val
        else:
            st = gs["stages"][pp.stage]
            if pp.stage not in written:          # copy the original params ONCE (never mutate src's)
                st["params"] = dict(st.get("params") or {})
                written.add(pp.stage)
            st["params"][pp.internal] = val      # then O(1) writes, not an O(K) rebuild per param
    if extras:
        for inp in manifest.inputs:
            if "feeds" not in inp:
                continue
            val = extras[inp["name"]]
            for stage, binding in inp["feeds"]:
                if stage == "terminal":
                    term_params[binding] = val
                else:
                    st = gs["stages"][stage]
                    if stage not in written:
                        st["params"] = dict(st.get("params") or {})
                        written.add(stage)
                    st["params"][binding] = val
    return gs, {**term_params, manifest.terminal_image_input: source}


def _assemble_fused_stages(manifest: ToolManifest, source, params: dict,
                           extras: dict | None = None) -> list:
    """The compile_fused stages list for a fused tool, run through the SAME _stages_from_spec
    the engine uses, so preflight and cook see byte-identical stages."""
    from .tex_fusion import _stages_from_spec
    gs, term_bindings = _fused_cook_inputs(manifest, source, params, extras)
    return _stages_from_spec(gs, manifest.terminal_code, term_bindings)


def _fed_inputs(manifest: ToolManifest) -> list:
    """The inputs a fused tool routes with `feeds`, in declaration order ([] for every other tool)."""
    return [inp for inp in manifest.inputs if "feeds" in inp]


def _fed_cook_values(manifest: ToolManifest, inputs: dict, fed: list):
    """cook_tool's gate for a tool with fed inputs: (source, {fed name: value}), or a coded refusal.

    Every declared input must be passed, and every one must share the source's [B,H,W] EXACTLY.
    Fused, a fed input joins the one grid the whole program cooks on (the per-axis max over the
    bindings it reads), while the same stages cooked one by one each grid on their own inputs —
    so a batch or singleton axis that would broadcast unfused changes an UPSTREAM stage's pixels
    fused, silently, with the same output shape and no error (measured on random inputs: a B=4
    input beside a B=1 source moved a `fi`-reading stage by maxdiff 2.99; a one-row source beside
    a full frame moved a `v`-reading stage by 1.00). A tool without fed inputs cannot meet this:
    every stage anchors on the one source.

    A promised input is resolved first (E7007 if unlanded, as `tex_engine.prepare` does), because
    the extent is read off the landed tensor and a fed value travels inside the chain payload,
    where the engine's own resolution never looks."""
    import torch
    from .tex_marshalling import resolve_promise_bindings
    tii = manifest.terminal_image_input
    for inp in fed:
        if inp["name"] not in inputs:
            raise TEXToolError(f"fused tool '{manifest.name}' needs its fed input '{inp['name']}'",
                               code=REFUSE_FUSED_INPUT_MISSING, input=inp["name"])
    vals = resolve_promise_bindings(
        {tii: inputs[tii], **{inp["name"]: inputs[inp["name"]] for inp in fed}})

    def extent(v):
        return tuple(v.shape[:3]) if isinstance(v, torch.Tensor) and v.dim() >= 3 else None

    want = extent(vals[tii])
    if want is None:
        raise TEXToolError(f"fused tool '{manifest.name}' cannot read the extent of its source "
                           f"'{tii}': a tool with fed inputs cooks [B,H,W] or [B,H,W,C] tensors",
                           code=REFUSE_FUSED_INPUT_EXTENT, input=tii)
    for inp in fed:
        got = extent(vals[inp["name"]])
        if got != want:
            have = "no readable [B,H,W]" if got is None else f"[B,H,W] = {list(got)}"
            raise TEXToolError(f"fused tool '{manifest.name}': input '{inp['name']}' has {have} but "
                               f"its source '{tii}' has [B,H,W] = {list(want)}; every input of a "
                               f"tool with fed inputs must match the source exactly (no batch or "
                               f"singleton axis is broadcast)",
                               code=REFUSE_FUSED_INPUT_EXTENT, input=inp["name"])
    return vals[tii], {inp["name"]: vals[inp["name"]] for inp in fed}


def cook_tool(manifest: ToolManifest, inputs: dict, params: dict | None = None, **cook_kwargs):
    """Cook a tool. `inputs`={binding: tensor} (external image/mask), `params`={promoted:
    value} (omitted -> default). Returns a tex_engine.CookResult. Fused tools take the real
    engine path (tiers, OOM, precision-auto), identical to a collapsed region; a fused tool with
    fed inputs is first refused (TEXToolError, `code` REFUSE_FUSED_INPUT_MISSING /
    REFUSE_FUSED_INPUT_EXTENT) unless every declared input is passed and co-extent."""
    from . import tex_engine
    if manifest.is_fused:
        if manifest.terminal_image_input not in inputs:
            raise TEXToolError(f"fused tool '{manifest.name}' needs its source input "
                               f"'{manifest.terminal_image_input}'",
                               code=REFUSE_FUSED_INPUT_MISSING, input=manifest.terminal_image_input)
        fed = _fed_inputs(manifest)
        if fed:
            source, extras = _fed_cook_values(manifest, inputs, fed)
            gs, term_bindings = _fused_cook_inputs(manifest, source, params, extras)
        else:
            gs, term_bindings = _fused_cook_inputs(manifest, inputs[manifest.terminal_image_input], params)
        return tex_engine.cook(manifest.terminal_code, term_bindings, chain_payload=gs, **cook_kwargs)
    bindings = {**inputs, **_resolve_params(manifest, params)}
    return tex_engine.cook(manifest.code, bindings, **cook_kwargs)


# ── TOOL-3: warm keys (re-fingerprinted at install, NEVER stored) ─────────────────
def _image_channel_variants(manifest: ToolManifest) -> list[int]:
    """The IMAGE channel counts a cook may hand this tool's source(s): [3, 4] if any input is an
    IMAGE (RGB or RGBA — the socket names a family, not a channel count), else [3] (MASK/LATENT
    have fixed shapes so the value is irrelevant). Warm keys are derived for each so an RGBA cook
    still hits the warm cache — mirrors tex_fusion._preflight_samples, which tries both."""
    for inp in manifest.inputs:
        if str(inp.get("type", "IMAGE")).upper() == "IMAGE":
            return [3, 4]
    return [3]


def tool_warm_keys(manifest: ToolManifest) -> list[str]:
    """Derive the value-independent fingerprint(s) of a tool's compiled artifact, AT
    INSTALL TIME, from the inline code. Never carried in the file (ENG-5: fingerprints are
    deliberately unstable across TEX versions). One key per IMAGE channel-count variant (an RGB
    and an RGBA cook fingerprint differently), deduped. Returns [] if none can be derived."""
    try:
        import torch  # noqa: F401
        from .tex_marshalling import infer_binding_type
    except Exception:                            # pragma: no cover
        return []
    keys: list[str] = []
    if manifest.is_fused:
        from .tex_fusion import fused_fingerprint
        rp = _repr_params(manifest, warm=True)   # match the cook's ingress types (raw defaults)
        for ch in _image_channel_variants(manifest):
            gs, term_bindings = _fused_cook_inputs(manifest, _fused_source_placeholder(manifest, ch), rp,
                                                   _fed_placeholders(manifest, ch))
            fp = fused_fingerprint(gs, manifest.terminal_code, term_bindings, infer_binding_type)
            if fp and fp not in keys:
                keys.append(fp)
        return keys
    from .tex_cache import TEXCache
    # Infer types the SAME way the cook does: inputs by socket type + channel count, params from
    # their RAW default (the ingress form the engine fingerprints, warm=True) -- _hint_value's
    # semantic type would mis-key a `c`/`v*` param (STRING at cook) or an int-defaulted `f`.
    for ch in _image_channel_variants(manifest):
        try:
            fp = TEXCache.fingerprint(manifest.code, _single_binding_types(manifest, ch, warm=True))
        except Exception:
            continue
        if fp not in keys:
            keys.append(fp)
    return keys


def _placeholder_tensor(type_str, image_channels: int = 3):
    """A zero tensor matching a declared input socket type, for type inference (warm keys /
    fused preflight) — mirrors tex_fusion._preflight_samples' shapes. `image_channels` (3 or 4)
    selects the channel count an IMAGE source is sampled at, so warm keys can cover both the
    VEC3 and VEC4 shapes a cook may hand it; MASK/LATENT ignore it (their shapes are fixed)."""
    import torch
    t = (type_str or "IMAGE").upper()
    if t == "MASK":
        return torch.zeros(1, 8, 8)          # -> FLOAT
    if t == "LATENT":
        return torch.zeros(1, 8, 8, 4)       # -> VEC4 (dominant)
    return torch.zeros(1, 8, 8, image_channels)   # IMAGE -> VEC3 (3) / VEC4 (4)


def _fused_source_placeholder(manifest: ToolManifest, image_channels: int = 3):
    """The placeholder source tensor a fused tool's external source is sampled at, shaped by its
    declared type. The source is the input named by terminal_image_input: the only input a tool
    without `feeds` declares (validate_manifest), and one of several — not necessarily the first —
    on a tool with them. The fallbacks are defensive (a shaped zero, so assembly still type-infers
    rather than seeing None)."""
    for inp in manifest.inputs:
        if inp.get("name") == manifest.terminal_image_input:
            return _placeholder_tensor(inp.get("type"), image_channels)
    if manifest.inputs:
        return _placeholder_tensor(manifest.inputs[0].get("type"), image_channels)
    return _zeros_bhwc(image_channels)


def _fed_placeholders(manifest: ToolManifest, image_channels: int = 3):
    """{name: placeholder} for each input a fused tool routes with `feeds`, or None when it routes
    none — so a tool without `feeds` hands `_fused_cook_inputs` exactly what it did before they
    existed. The channel variant is ALIGNED with the source's: an RGB warm key samples every IMAGE
    input at 3 channels, an RGBA one at 4 (two keys, not every combination); a cook that mixes
    them is a warm-cache miss, which compiles."""
    fed = {inp["name"]: _placeholder_tensor(inp.get("type"), image_channels)
           for inp in manifest.inputs if "feeds" in inp}
    return fed or None


# ── the user tool store (TOOL-2 backend: get_user_dir seam, LANG-5 pattern) ───────
def tools_dir() -> str | None:
    """<user_dir>/tex_wrangle/tools/, falling back under the TEX cache dir for CLI/tests.
    Mirrors tex_snippets._snippets_path()."""
    base = None
    try:
        from .tex_runtime.host import get_host_services
        base = get_host_services().get_user_dir()
    except Exception:
        base = None
    if not base:
        try:
            from .tex_cache import get_cache
            base = os.path.join(get_cache()._cache_dir, "user")
        except Exception:
            return None
    return os.path.join(base, "tex_wrangle", _TOOL_STORE)


def _safe_tool_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "tool"
    return stem + ".textool"


def write_tool(manifest_or_dict, dest_dir: str | None = None) -> str:
    """Publish (TOOL-2 backend): validate then atomically write a manifest to the tool
    store (or dest_dir). Returns the written path. Validates first so a malformed manifest
    is never persisted."""
    raw = manifest_or_dict.to_dict() if isinstance(manifest_or_dict, ToolManifest) else manifest_or_dict
    validate_manifest(raw)
    dest_dir = dest_dir or tools_dir()
    if not dest_dir:
        raise TEXToolError("no tool directory available (no host user dir and no cache dir)")
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, _safe_tool_filename(raw["name"]))
    # Two DIFFERENT tool names can sanitize to the same filename ("My Grade" / "My/Grade").
    # Re-publishing the SAME tool overwrites (an update); a collision with a DIFFERENT name
    # must not silently clobber it — fail loud so the publisher renames.
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
            if existing.get("name") != raw["name"]:
                raise TEXToolError(f"tool file '{os.path.basename(path)}' already holds a "
                                   f"different tool ('{existing.get('name')}'); rename this one")
        except (OSError, json.JSONDecodeError):
            pass                                # unreadable/corrupt existing file: overwrite it
    # BOUNDED (P0-7): see `tex_recovery.bounded_mkstemp` — an ACL-denied directory makes a
    # bare `mkstemp` retry ~2.1 billion times rather than raise.
    from .tex_recovery import bounded_mkstemp
    fd, tmp = bounded_mkstemp(dir=dest_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def list_tools(dir: str | None = None) -> list[str]:
    d = dir or tools_dir()
    if not d or not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".textool"))


def tool_summary(manifest: ToolManifest) -> dict:
    """A palette/instancing summary of a tool (TOOL-2): name, doc, inputs, and the promoted
    widgets a host renders. Carries no source — enough to list and instance a tool node."""
    return {
        "name": manifest.name,
        "tool_version": manifest.tool_version,
        "category": manifest.category,
        "context": manifest.context,
        "doc": manifest.doc,
        "shape": "fused" if manifest.is_fused else "single-stage",
        "inputs": [dict(i) for i in manifest.inputs],
        "outputs": [dict(o) for o in manifest.outputs],
        "widgets": [p.to_dict() for p in manifest.promoted_params],
    }


_SUMMARY_CACHE: dict = {}   # path -> ((mtime_ns, size), summary_or_error_dict)


def load_all_tools(dir: str | None = None) -> list[dict]:
    """Load + summarize every installed tool (TOOL-2 palette). Cached per (path, mtime, size), so
    an unchanged tool is not re-read/re-parsed/re-validated on every /list_tools call. A tool that
    fails to load is cached as an error entry rather than aborting the whole listing."""
    out = []
    live = set()
    for path in list_tools(dir):
        live.add(path)
        try:
            st = os.stat(path)
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            key = None
        cached = _SUMMARY_CACHE.get(path)
        if key is not None and cached is not None and cached[0] == key:
            out.append(cached[1])
            continue
        try:
            summary = tool_summary(load_tool(path))
        except TEXToolError as e:
            summary = {"name": os.path.basename(path), "error": str(e)}
        if key is not None:
            _SUMMARY_CACHE[path] = (key, summary)
        out.append(summary)
    for stale in [p for p in _SUMMARY_CACHE if p not in live]:   # forget deleted tools
        _SUMMARY_CACHE.pop(stale, None)
    return out


def _compile_tool_program(manifest: ToolManifest, image_channels: int = 3):
    """Compile a tool to (program_ast, type_map, used_builtins, fingerprint) for warming, at the
    given IMAGE channel count. Single-stage via compile(); FUSED via prepare_fused -- the real
    spliced program keyed by the fused fingerprint, NOT terminal_code in isolation (a different
    program than what cooks). Promoted params are typed by their LANG-1 hint (_repr_params), so
    the warm fingerprint matches the cook's (the fused path used to key off raw defaults)."""
    from .tex_marshalling import infer_binding_type
    if manifest.is_fused:
        from .tex_fusion import prepare_fused, fused_fingerprint
        gs, term_bindings = _fused_cook_inputs(
            manifest, _fused_source_placeholder(manifest, image_channels), _repr_params(manifest, warm=True),
            _fed_placeholders(manifest, image_channels))
        prog, type_map, _referenced, _assigned, _pinfo, used_builtins, _merged = \
            prepare_fused(gs, manifest.terminal_code, term_bindings, infer_binding_type)
        fp = fused_fingerprint(gs, manifest.terminal_code, term_bindings, infer_binding_type)
        return prog, type_map, used_builtins, fp
    from .tex_api import compile as _compile
    from .tex_cache import TEXCache
    bt = _single_binding_types(manifest, image_channels, warm=True)
    prog = _compile(manifest.code, bt)
    return prog.ast, prog.type_map, prog.used_builtins, TEXCache.fingerprint(manifest.code, bt)


def _warm_compiled(prog_ast, type_map, fp, used_builtins, *, device: str = "cuda",
                   precision: str = "fp32") -> dict:
    """Warm the compile/codegen tiers for an ALREADY-COMPILED program (the same three steps
    tex_api.prewarm runs per program, but for a program object rather than a source string, so a
    fused tool can be warmed by its real spliced program). Best-effort; off the hot path."""
    import torch
    from .tex_runtime import compiled, graphed, warm_state
    warm_state.ensure_loaded()
    dev_type = "cuda" if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    summary = {"codegen": 0, "bg_compile": 0, "capturable": 0}
    try:
        if compiled._get_or_make_codegen_fn(prog_ast, type_map, fp) is not None:
            summary["codegen"] = 1
    except Exception:
        pass
    if dev_type == "cuda":
        try:
            if compiled._cuda_headroom_ok(device) and not compiled._capture_in_flight():
                if compiled._submit_bg_compile((fp, dev_type, precision), prog_ast, type_map,
                                               dev_type, used_builtins, precision, fp):
                    summary["bg_compile"] = 1
        except Exception:
            pass
        try:
            graphed._capturable_memo[fp] = graphed._capturable(prog_ast)
            summary["capturable"] = 1
        except Exception:
            pass
    warm_state.persist(force=True)
    return summary


def warm_tool(manifest: ToolManifest, *, device: str = "cuda", precision: str = "fp32") -> dict:
    """TOOL-3: warm-compile a tool at its promoted-param signature -- single-stage OR the real
    fused program, for each IMAGE channel-count variant (so both an RGB and an RGBA cook find a
    warmed artifact). Off the cook hot path; raises only on a genuine compile error (callers
    treat it best-effort). Returns a {codegen, bg_compile, capturable, variants} summary."""
    summary = {"codegen": 0, "bg_compile": 0, "capturable": 0, "variants": 0}
    seen: set[str] = set()
    for ch in _image_channel_variants(manifest):
        prog_ast, type_map, used_builtins, fp = _compile_tool_program(manifest, ch)
        if fp in seen:          # a tool that doesn't depend on channel count keys identically → warm once
            continue
        seen.add(fp)
        s = _warm_compiled(prog_ast, type_map, fp, used_builtins, device=device, precision=precision)
        for k in ("codegen", "bg_compile", "capturable"):
            summary[k] += s.get(k, 0)
        summary["variants"] += 1
    return summary


def install_tool(manifest_or_path, dest_dir: str | None = None, *, warm: bool = False,
                 device: str = "cuda", precision: str = "fp32") -> dict:
    """Install a tool into the store. VALIDATE-ONLY by default (TOOL-5-A): parses, schema-
    checks, type-checks, writes the manifest. Compiles NOTHING unless warm=True (explicit
    consent), in which case it re-derives warm keys and drives tex_api.prewarm off the hot
    path. Returns {path, ok, warnings, warm_keys, preflight}."""
    manifest = manifest_or_path if isinstance(manifest_or_path, ToolManifest) else load_tool(manifest_or_path)
    pf = preflight_tool(manifest)
    if not pf["ok"]:
        raise TEXToolError(f"tool '{manifest.name}' failed preflight: "
                           f"{pf['diagnostics'][0].get('message') if pf['diagnostics'] else 'unknown'}")
    path = write_tool(manifest, dest_dir)
    result = {"path": path, "ok": True,
              "warnings": list(manifest.warnings) + list(pf.get("output_warnings") or []),
              "warm_keys": [], "preflight": pf}
    if warm:                                     # TOOL-3, opt-in only
        result["warm_keys"] = tool_warm_keys(manifest)
        try:
            result["warmed"] = warm_tool(manifest, device=device, precision=precision)
        except Exception as e:                   # warming is best-effort, never fatal
            result["warnings"].append(f"warm-compile skipped: {e}")
    return result
