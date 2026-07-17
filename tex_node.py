"""
TEX Wrangle Node — ComfyUI custom node for the TEX (Tensor Expression Language).

Provides a single node where users write compact TEX scripts to process
images, masks, and scalar values using per-pixel tensor operations.

Requires ComfyUI with Nodes v3 API support (comfy_api.latest).

**ENG-1 (v0.22): this file is the ComfyUI ADAPTER, not the engine.** The cook itself
— tier selection, fallbacks, the OOM ladder, tiling, the `precision="auto"` gate —
lives in `tex_engine`. `execute()` is marshal-in → `engine.prepare`/`engine.run` →
marshal-out, and what stays here is exactly what ComfyUI imposes: the kwargs +
lazy-slot-pool protocol, LATENT dict wrap/unwrap, the `ui=` HUD payload, and the
`TEX_DIAG:`-suffixed RuntimeError the frontend parses. Anything host-agnostic that
lands here is in the wrong module (S-1).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import torch
import traceback
from typing import Any

logger = logging.getLogger("TEX")

from .tex_compiler.lexer import LexerError
from .tex_compiler.parser import ParseError
from .tex_compiler.type_checker import TypeCheckError
from .tex_compiler.diagnostics import TEXMultiError, TEX_BUG_REPORT_URL
from .tex_runtime.interpreter import InterpreterError
from .tex_fusion import FusionError

# ENG-1: the cook engine (host-agnostic). This node is one of its callers.
from . import tex_engine as _engine
from .tex_engine import _oom_in_chain, _drop_tex_caches_on_oom

# Marshalling and type inference utilities (extracted to tex_marshalling.py)
from .tex_marshalling import (
    tensor_fingerprint as _tensor_fingerprint,
    unwrap_latent as _unwrap_latent,
    prepare_output as _prepare_output,
    map_inferred_type as _map_inferred_type,
    egress_materializes as _egress_materializes,
)

# ── v3 API import ──
# Falls back to a stub base class when running outside ComfyUI (e.g. standalone tests).
try:
    from comfy_api.latest import IO
    _V3_AVAILABLE = True
except ImportError:
    IO = None
    _V3_AVAILABLE = False

from . import tex_lazy as _tex_lazy

# Matches the schema's lazy input-pool slot names (in_0..in_{MAX_LAZY_INPUTS-1}).
_LAZY_SLOT_RE = re.compile(r"^in_\d+$")


def _downscale_for_preview(img: torch.Tensor, max_dim: int = 256) -> torch.Tensor:
    """Q-6: downscale a [B,H,W,C] image so its largest spatial dim ≤ max_dim
    (aspect preserved), for the debounced low-res live preview. Returns the
    input unchanged when it already fits. Bilinear, antialias-free (a preview)."""
    if not (isinstance(img, torch.Tensor) and img.dim() == 4):
        return img
    _b, h, w, _c = img.shape
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / m
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    bchw = img.permute(0, 3, 1, 2)
    out = torch.nn.functional.interpolate(bchw, size=(nh, nw), mode="bilinear",
                                          align_corners=False)
    return out.permute(0, 2, 3, 1).contiguous()


# ── Node class ──
# When running inside ComfyUI, inherit from IO.ComfyNode (v3 API).
# When running standalone (tests), fall back to plain object.
_BaseClass = IO.ComfyNode if _V3_AVAILABLE else object


class TEXWrangleNode(_BaseClass):
    """
    TEX Wrangle — per-pixel tensor expression processor.

    Write compact TEX scripts to process images, masks, and scalars.
    Reference inputs with @name (e.g. @A, @base_image, @strength).
    Write output to @OUT.

    Example:
        float gray = luma(@A);
        @OUT = vec4(gray, gray, gray, 1.0);
    """

    # Maximum number of output slots (pre-allocated for dynamic outputs). Single-sourced
    # from the engine's ceiling (ENG-1) — the socket count and the cook's E6002 gate are
    # the same number, and the node's schema is what makes it 8.
    MAX_OUTPUTS = _engine.MAX_OUTPUTS

    # Lazy input cooking: ComfyUI decides laziness per *schema-declared* input
    # name at graph-build time (comfy_execution/graph.py get_input_info), so
    # TEX's dynamic user-named inputs can never be lazy directly. Instead the
    # schema declares this fixed pool of lazy AnyType slots (in_0..in_N); the
    # frontend maps wired user inputs onto them IN THE QUEUED PROMPT ONLY
    # (workflow JSON / slots / labels keep user names) and passes the mapping
    # as the `_tex_slot_map` constant. check_lazy_status() then tells ComfyUI
    # which slots the program actually needs; upstream subgraphs feeding the
    # rest are never cooked.
    MAX_LAZY_INPUTS = 16

    # System kwargs that are NOT TEX bindings
    _SYSTEM_KWARGS = {"code", "device", "compile_mode", "precision", "_tex_any",
                      "_tex_chain", "_tex_preview", "debug_nan_highlight",
                      "_tex_slot_map", "_tex_time"}

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TEX_Wrangle",
            display_name="TEX Wrangle",
            category="TEX",
            description=(
                "TEX Wrangle: per-pixel tensor expressions for images, masks, and scalars.\n"
                "Reference inputs with @name (e.g. @A, @base_image).\n"
                "Write outputs with @name = expr (e.g. @OUT, @mask, @result).\n"
                "Add parameter widgets with $name (e.g. f$strength = 0.5).\n"
                "Supports float, int, vec2/vec3/vec4, mat3/mat4, string, and array types "
                "with if/else, for/while loops, and 100+ stdlib functions.\n"
                "Click the ? icon for a quick reference."
            ),
            inputs=[
                IO.String.Input(
                    "code",
                    multiline=True,
                    dynamic_prompts=False,
                    default=(
                        "// TEX Wrangle\n"
                        "// Read inputs with @A, @B, etc.\n"
                        "// Write output to @OUT\n"
                        "// Right-click → TEX Snippets for 116 examples\n\n"
                        "float gray = luma(@IN);\n"
                        "@OUT = vec3(gray);\n"
                    ),
                    placeholder="// TEX code here...",
                    tooltip="TEX source code. Use @name for inputs, @name = expr for outputs. Use $name for parameter widgets.",
                ),
                IO.Combo.Input(
                    "device",
                    options=["auto", "cpu", "cuda"],
                    default="auto",
                    tooltip="Execution device. auto: follows input tensors. cpu/cuda: force a specific device.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "compile_mode",
                    options=["none", "auto", "torch_compile", "cuda_graph"],
                    default="none",
                    tooltip="none: standard interpreter (recommended default). auto: EXPERIMENTAL measured auto-tier — runs the fast codegen path and trials torch.compile in the background, committing only on a measured win. torch_compile: force JIT-compile via torch.compile. NOTE: auto/torch_compile need Triton for any GPU speedup — Triton is absent on most Windows installs, where they simply fall back to the interpreter on CUDA (CPU torch.compile works but is often slower for small programs). Run `tex doctor` to see whether Triton is present. cuda_graph: CUDA-graph replay (GPU only, needs NO Triton; big win for small launch-bound programs). All tiers fall back to the interpreter on failure.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "precision",
                    options=["fp32", "auto", "fp16"],
                    default="fp32",
                    tooltip="fp32: full precision (default, recommended). auto: EXPERIMENTAL — runs fp16 ONLY where a condition-number gate proves it stays accurate (CUDA, >=1024x1024, smooth pointwise, no amplification/ill-conditioning; verified across 225 adversarial programs); everything else runs fp32. A per-cook finiteness net makes auto ~perf-NEUTRAL — it's an accuracy-safe convenience, not a speedup. fp16: EXPERT — force half-precision IMAGE temps for the raw ~1.35-1.45x win with NO safety net (coordinates & sampling stay fp32; ~1e-3 accuracy, diverges on threshold/branch/amplifying programs). LATENT stays fp32.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "debug_nan_highlight",
                    default=False,
                    tooltip="DBG-3: paint any pixel that is NaN or Inf bright magenta so "
                            "non-finite output is visible at a glance (a 0/0, a log of a "
                            "negative, an fp16 overflow). Off by default and zero-cost when "
                            "off. Sits above all tiers, so it works on every execution path.",
                    optional=True,
                ),
                # Wildcard input so the node appears in the search panel
                # when dragging a wire of any type. The frontend extension
                # removes all initial input slots in onNodeCreated and
                # manages them dynamically, so this slot never renders.
                IO.AnyType.Input(
                    "_tex_any",
                    optional=True,
                    tooltip="TEX accepts any input type. Use @name in code to reference it.",
                ),
            ] + [
                # Lazy slot pool (see MAX_LAZY_INPUTS above). Never rendered:
                # the frontend removes all initial input slots in onNodeCreated.
                IO.AnyType.Input(
                    f"in_{i}",
                    optional=True,
                    lazy=True,
                    tooltip="Internal lazy input slot — mapped from wired user inputs at queue time.",
                )
                for i in range(cls.MAX_LAZY_INPUTS)
            ],
            outputs=[
                IO.AnyType.Output(
                    id=f"out_{i}",
                    display_name=f"out_{i}",
                    tooltip="TEX output. Type auto-inferred from code.",
                )
                for i in range(cls.MAX_OUTPUTS)
            ],
            accept_all_inputs=True,
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        """Cache-busting: re-execute if code, inputs, device, or compile_mode change."""
        code = kwargs.get("code", "")
        parts = [code]
        # Include device and compile_mode in the hash
        parts.append(kwargs.get("device", "auto"))
        parts.append(kwargs.get("compile_mode", "none"))
        parts.append(kwargs.get("precision", "fp32"))
        # DBG-3 (audit): the NaN-overlay toggle changes the OUTPUT, so it must bust the
        # cache — else toggling it on serves the stale (un-overlaid) cook and the debug
        # aid silently does nothing on first toggle.
        parts.append(str(kwargs.get("debug_nan_highlight", False)))
        # LOAD-BEARING: the fused-chain payload (upstream code + params) must
        # be hashed explicitly. It sits in _SYSTEM_KWARGS so the loop below
        # never treats it as a TEX binding — without this line, editing an
        # upstream node's code would no longer recook the fused terminal.
        chain_payload = kwargs.get("_tex_chain")
        if chain_payload is not None:
            parts.append(f"_tex_chain:{chain_payload}")
        # ENG-7: the playhead CHANGES THE PIXELS, so it must bust the cache — and being a
        # _SYSTEM_KWARG it is skipped by the binding loop below, exactly like _tex_chain.
        # Without this line the host would serve frame 1's cached result for frame 2 and
        # the animation would sit still: the same freeze the builtins LRU, the codegen
        # closure and CUDA-graph capture each had to be taught about separately. This is
        # the last of the four caches between a playhead and a pixel.
        time_payload = kwargs.get("_tex_time")
        if time_payload is not None:
            parts.append(f"_tex_time:{time_payload}")
        # Lazy slot mapping: renaming is semantics-relevant (it decides which
        # user name each wired value binds to), so the map must bust the cache.
        slot_map = kwargs.get("_tex_slot_map")
        if slot_map is not None:
            parts.append(f"_tex_slot_map:{slot_map}")
        # Hash all binding inputs (everything except system kwargs)
        for name in sorted(kwargs.keys()):
            if name in cls._SYSTEM_KWARGS:
                continue
            val = kwargs[name]
            if val is not None:
                if isinstance(val, list):
                    list_parts = []
                    for item in val:
                        if isinstance(item, torch.Tensor):
                            list_parts.append(_tensor_fingerprint(item))
                        else:
                            list_parts.append(str(item))
                    parts.append(f"{name}:LIST:{','.join(list_parts)}")
                elif isinstance(val, dict) and "samples" in val:
                    lat = f"{name}:LATENT:{_tensor_fingerprint(val['samples'])}"
                    # Include noise_mask + scalar metadata so a SetLatentNoiseMask
                    # change (same samples, new mask) still busts the cache.
                    mask = val.get("noise_mask")
                    if isinstance(mask, torch.Tensor):
                        lat += f":mask:{_tensor_fingerprint(mask)}"
                    for mk in sorted(k for k in val if k not in ("samples", "noise_mask")):
                        mv = val[mk]
                        if isinstance(mv, (int, float, str, bool)):
                            lat += f":{mk}={mv}"
                        elif isinstance(mv, torch.Tensor):
                            lat += f":{mk}:{_tensor_fingerprint(mv)}"
                        else:
                            lat += f":{mk}={mv!r}"
                    parts.append(lat)
                elif isinstance(val, torch.Tensor):
                    parts.append(f"{name}:{_tensor_fingerprint(val)}")
                else:
                    parts.append(f"{name}:{val}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    @staticmethod
    def _parse_time_context(raw) -> dict | None:
        """ENG-7: normalize the `_tex_time` payload into {"frame","fps","time"} floats.

        Tolerates a JSON string or a dict (same shape as `_tex_chain`); anything else, or
        a payload with no usable key, yields None — the engine then reads zeros. Never
        raises: a malformed time hint must not fail a cook that would otherwise render."""
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                return None
        if not isinstance(raw, dict):
            return None
        out = {}
        for k in ("frame", "fps", "time"):
            v = raw.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = float(v)
        return out or None

    @staticmethod
    def _parse_slot_map(slot_map) -> list[dict]:
        """Normalize the frontend's `_tex_slot_map` constant into an ordered
        list of {"name", "slot", "type"} dicts. Tolerates a JSON string or a
        list; anything else (or malformed entries) yields []."""
        if isinstance(slot_map, str):
            try:
                slot_map = json.loads(slot_map)
            except Exception:
                return []
        if not isinstance(slot_map, list):
            return []
        out = []
        for e in slot_map:
            if (isinstance(e, dict) and isinstance(e.get("name"), str)
                    and isinstance(e.get("slot"), str)):
                out.append({"name": e["name"], "slot": e["slot"],
                            "type": e.get("type", "*")})
        return out

    @classmethod
    def check_lazy_status(cls, **kwargs):
        """Lazy input cooking: name the `in_N` pool slots this cook actually
        needs; ComfyUI never cooks the upstream subgraphs of the rest.

        Protocol (execution.py): wired-but-uncooked lazy inputs arrive as None;
        the method is re-invoked as requested inputs become available, so wired
        scalar $params can be cooked FIRST and folded like widget values (the
        iterative "T4-lite" round). Analysis is tex_lazy.lazy_required_bindings
        (memoized; shared with execute()'s E6003 gate — same inputs, same
        result, so the two can never disagree).

        Safety rules (semantics preservation — see tex_lazy docstring):
          R1  if any spatial-capable wire exists, the FIRST one (slot order,
              first-wins shape derivation) must be needed and of a known
              tensor type (IMAGE/MASK) — else keep everything.
          R2  LATENT wires are always cooked (they flip output typing/fp32).
          R3  analysis failure -> keep everything.
        Fused chains (_tex_chain) keep everything in v1. Never raises: any
        internal error degrades to "cook everything wired".
        """
        entries = cls._parse_slot_map(kwargs.get("_tex_slot_map"))
        if not entries:
            return []

        def _pending(subset=None):
            src = entries if subset is None else subset
            return [e["slot"] for e in src if kwargs.get(e["slot"]) is None]

        try:
            if kwargs.get("_tex_chain"):
                return _pending()
            code = kwargs.get("code", "")
            # Foldable values: scalar widget constants + already-cooked wired
            # scalars (bool/int/float). Pool-slot keys are never params.
            params: dict[str, Any] = {}
            for name, val in kwargs.items():
                if name in cls._SYSTEM_KWARGS or _LAZY_SLOT_RE.match(name):
                    continue
                if isinstance(val, (bool, int, float)):
                    params[name] = val
            scalar_pending = []
            for e in entries:
                v = kwargs.get(e["slot"])
                if v is None:
                    if e["type"] in _tex_lazy.SCALAR_WIRE_TYPES:
                        scalar_pending.append(e)
                elif isinstance(v, (bool, int, float)):
                    params[e["name"]] = v
            needed = _tex_lazy.lazy_required_bindings(code, params)
            if needed is None:
                return _pending()  # R3
            # T4-lite: cook referenced wired scalars first; the next round
            # folds their values and may prune whole image branches.
            first = [e["slot"] for e in scalar_pending if e["name"] in needed]
            if first:
                return first
            # R1: first-wins shape anchor must survive and be a known tensor.
            spatial = [e for e in entries
                       if e["type"] in _tex_lazy.SPATIAL_WIRE_TYPES]
            if spatial and (spatial[0]["name"] not in needed
                            or spatial[0]["type"] not in _tex_lazy.SHAPE_ANCHOR_TYPES):
                return _pending()
            return [e["slot"] for e in entries
                    if kwargs.get(e["slot"]) is None
                    and (e["type"] == "LATENT" or e["name"] in needed)]
        except Exception:
            logger.warning("[TEX] lazy analysis failed; cooking all wired inputs.",
                           exc_info=True)
            return _pending()

    @classmethod
    def _build_ui_payload(cls, elapsed_ms, device, eff_precision, near_singularities):
        """DBG-1/C1-ux: assemble the ADDITIVE `ui=` HUD payload — tier/timing/precision +
        C7-ux reasons + debug_print probes + the C4-ux near-singularity count (present only
        with the debug toggle). The result tuple is byte-identical whether or not this
        payload is attached. ENG-1: the count now arrives on the CookResult (the engine
        reads it before disarming the trace) instead of being re-read from guard_trace here.
        ENG-5 canary-pins this key set — it is a frontend contract."""
        from .tex_runtime import tier_trace
        tr = tier_trace.last()
        pr = tier_trace.last_precision()
        perf = {
            "tier": tr.tier if tr is not None else "interpreter",
            "fallback_from": tr.fallback_from if tr is not None else None,
            "reason": tr.reason if tr is not None else None,
            "elapsed_ms": round(elapsed_ms, 2),
            "device": str(device),
            "precision": pr[0] if pr is not None else eff_precision,
            "precision_reason": pr[1] if pr is not None else None,
        }
        if near_singularities is not None:  # C4-ux: count is armed with the toggle
            perf["near_singularities"] = near_singularities
        ui_payload = {"tex_perf": [perf]}
        probes = tier_trace.get_probes()  # LX-5: debug_print value-at-pixel taps
        if probes:
            ui_payload["tex_probes"] = probes
        return ui_payload

    @classmethod
    def execute(cls, **kwargs):
        """Execute TEX code with the provided inputs."""
        start_time = time.perf_counter()

        # Pop system parameters so they don't get treated as bindings
        code = kwargs.pop("code", "")
        device_mode = kwargs.pop("device", "auto")
        compile_mode = kwargs.pop("compile_mode", "none")
        precision = kwargs.pop("precision", "fp32")
        # (the fp16 x compiled-tier clamp is engine policy — tex_engine.prepare, ENG-1)
        debug_nan_highlight = bool(kwargs.pop("debug_nan_highlight", False))  # DBG-3
        kwargs.pop("_tex_any", None)  # search-panel wildcard slot, unused
        kwargs.pop("_tex_preview", None)  # Q-6 preview hint; pop so it never
        # becomes a phantom @_tex_preview binding (it is a _SYSTEM_KWARG).
        # ENG-7: the host's playhead for the frame/fps/time builtins. ComfyUI has no
        # timeline, so this is normally absent -> the engine reads zeros. Carried as an
        # underscore-prefixed SYSTEM kwarg (like _tex_chain) rather than a plain `frame`
        # input on purpose: a bare name would collide with any user binding called
        # `frame`, and _SYSTEM_KWARGS members are excluded from bindings. In ComfyUI,
        # position within an image BATCH is `fi`/`fn`, which is the axis that exists here.
        time_context = cls._parse_time_context(kwargs.pop("_tex_time", None))
        # Lazy slot pool: map cooked in_N values back to their user names.
        # Slots the lazy analysis skipped arrive as None and simply stay
        # absent — identical to an unwired input (the E6003 gate below
        # forgives their dead references). Unmapped in_N keys (no entry —
        # defensive) are dropped so they never become phantom bindings.
        slot_entries = cls._parse_slot_map(kwargs.pop("_tex_slot_map", None))
        for entry in slot_entries:
            val = kwargs.pop(entry["slot"], None)
            if val is not None:
                kwargs[entry["name"]] = val
        for stray in [k for k in kwargs if _LAZY_SLOT_RE.match(k)]:
            kwargs.pop(stray)
        # Cross-node fusion: when the frontend collapses a linked TEX chain into
        # this (terminal) node, it passes the upstream stages here. Absent → the
        # node behaves exactly as a single program (no behaviour change).
        chain_payload = kwargs.pop("_tex_chain", None)

        # Everything remaining in kwargs is a TEX binding (inputs + params)
        bindings: dict[str, Any] = {
            name: val for name, val in kwargs.items() if val is not None
        }

        # Unwrap lists — use first element (TEX doesn't support per-image indexing)
        for name, val in list(bindings.items()):
            if isinstance(val, list):
                if len(val) > 0:
                    bindings[name] = val[0]
                else:
                    bindings[name] = torch.tensor(0.0)

        # Unwrap LATENT dicts into channel-last tensors, preserving metadata
        latent_meta: dict[str, Any] = {}
        has_latent_input = False
        latent_channel_count = 0
        for name, val in list(bindings.items()):
            if isinstance(val, str):
                continue  # strings pass through as-is
            if isinstance(val, dict) and "samples" in val:
                tensor_cl, meta = _unwrap_latent(val)
                bindings[name] = tensor_cl
                has_latent_input = True
                if not latent_meta and meta:
                    latent_meta = meta
                if not latent_channel_count:
                    latent_channel_count = tensor_cl.shape[-1]

        fused_chain = False
        try:
            # ENG-1: the cook itself is the engine's. The two-step (prepare then run) is
            # deliberate: `fused_chain` must be set only once a chain has actually
            # ASSEMBLED, because the Q-4 stage attribution below distinguishes "a linked
            # node failed while running" from "the chain never spliced" (a splice-time
            # type error carries a stage tag too, but has no running stage to blame).
            plan = _engine.prepare(
                code, bindings, chain_payload=chain_payload, device_mode=device_mode,
                compile_mode=compile_mode, precision=precision,
                has_latent_input=has_latent_input,
                latent_channel_count=latent_channel_count,
                # The lazy pool may have skipped statically-dead inputs, so E6003 must
                # forgive references the optimizer will drop (invariant #11's gate).
                forgive_dead_refs=bool(slot_entries),
                debug_nan_highlight=debug_nan_highlight,
                time_context=time_context,
                # ENG-1/ENG-3: skip the engine's no-alias clone only while THIS node's own
                # egress is guaranteed to allocate anyway. `_prepare_output` below pins no
                # profile, so it formats through the process-wide one — which makes the
                # process-wide answer the honest answer for this call, and is why the node
                # may ask. Under 'comfy' (every shipping user) the clamp materializes and
                # the clone would be pure waste: +0.236 ms at 2048² passthrough, on the
                # default path (invariant #7). Under 'engine' there is no clamp — that
                # profile exists to remove the lossy steps — so the guarantee has to be
                # bought, and this asks for it. Deriving it beats hardcoding False: that
                # would hand ComfyUI a live view of an input buffer the moment a host
                # flipped the profile, which is the bug one layer down.
                disown=not _egress_materializes(),
                max_outputs=cls.MAX_OUTPUTS)
            fused_chain = plan.fused_chain
            cooked = _engine.run(plan)

            # Format each output based on inferred type
            results = []
            output_types_log = []
            for name in cooked.output_names:
                inferred_type = cooked.assigned[name]
                effective_type = _map_inferred_type(inferred_type, has_latent_input)
                result = _prepare_output(cooked.outputs[name], effective_type)
                if effective_type == "LATENT":
                    result = {"samples": result, **latent_meta}
                results.append(result)
                output_types_log.append(f"{name}:{effective_type}")

            # Pad with None for unused output slots
            while len(results) < cls.MAX_OUTPUTS:
                results.append(None)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            compile_tag = " [compiled]" if compile_mode == "torch_compile" else ""
            logger.info(
                "Executed in %.1fms%s | device: %s | outputs: [%s] | bindings: %s",
                elapsed_ms, compile_tag, cooked.device,
                ", ".join(output_types_log), cooked.binding_names,
            )

            # DBG-1/C1-ux: per-cook tier/timing/probe facts for the frontend HUD, on the v3
            # `ui=` channel — ADDITIVE (the result tuple is byte-identical without it).
            ui_payload = cls._build_ui_payload(elapsed_ms, cooked.device, cooked.precision,
                                               cooked.near_singularities)

            # Return NodeOutput for v3 (with the HUD payload), or tuple for standalone/test.
            if _V3_AVAILABLE:
                return IO.NodeOutput(*results, ui=ui_payload)
            return tuple(results)

        except TEXMultiError as e:
            # Multiple errors — human-readable first, JSON at end for frontend
            readable = "\n\n".join(d.render() for d in e.diagnostics)
            payload = json.dumps([d.to_dict() for d in e.diagnostics])
            raise RuntimeError(f"{readable}\nTEX_DIAG:{payload}") from e
        except (LexerError, ParseError, TypeCheckError, InterpreterError) as e:
            # M-1: an OOM raised inside a stdlib call arrives here wrapped as an
            # InterpreterError — re-raise the underlying OOM UNWRAPPED so ComfyUI's
            # OOM handling (memory summary + unload_all_models) still fires. Must
            # precede the diagnostic re-wrap, which would otherwise mask it.
            _oom = _oom_in_chain(e)
            if _oom is not None:
                _drop_tex_caches_on_oom()   # P1-M1-FREERETRY
                raise _oom
            # Q-4: attribute a fused-chain runtime error to its originating linked
            # node (the stage tag survives the splice; the frontend collapsed the
            # nodes so line numbers alone wouldn't map back).
            stage_note = ""
            _loc = getattr(e, "loc", None)
            _stage = getattr(_loc, "stage", None) if _loc is not None else None
            if fused_chain and _stage is not None:
                stage_note = (f"\n\nThis error came from linked TEX node #{_stage + 1} "
                              f"in the fused chain (counting from the source image). "
                              f"Break the chain there to debug it in isolation.")
            # Single error — human-readable first, JSON at end for frontend
            if hasattr(e, '_build_diagnostic'):
                e._build_diagnostic()
            if hasattr(e, 'diagnostic') and e.diagnostic:
                readable = e.diagnostic.render()
                payload = json.dumps([e.diagnostic.to_dict()])
                raise RuntimeError(f"{readable}{stage_note}\nTEX_DIAG:{payload}") from e
            # Fallback for errors without diagnostic (shouldn't happen)
            raise RuntimeError(f"TEX Error: {e}{stage_note}") from e
        except FusionError as e:
            # A chain couldn't be fused (e.g. an upstream node scatter-writes @OUT,
            # or has multiple outputs). A structural/config condition — show it
            # cleanly, no traceback or bug-report link. We must still raise: the
            # upstream nodes were collapsed out of the prompt, so running this
            # terminal alone would be wrong.
            raise RuntimeError(
                f"TEX couldn't fuse this chain of nodes:\n  {e}\n\n"
                f"Turn off 'TEX Fusion: Compile linked TEX nodes together' in "
                f"settings, or break the chain at the node mentioned above."
            ) from e
        except (ValueError, KeyError, TypeError, IndexError) as e:
            # Usually a value or type a built-in didn't expect — a user-side
            # problem, not a TEX bug. Show it cleanly, without a traceback.
            raise RuntimeError(
                f"TEX couldn't finish running your program:\n  {e}\n\n"
                f"This usually points to an input or parameter value that a built-in "
                f"didn't expect — check the inputs feeding this node."
            ) from e
        except RuntimeError as e:
            # OutOfMemoryError subclasses RuntimeError — re-raise it UNWRAPPED so
            # ComfyUI's own OOM handling (execution.py: is_oom → memory summary +
            # unload_all_models + user tips) still fires; the generic re-wrap
            # below would hide it. (Preflight/retry recovery is M-1, Phase 3.)
            _oom = _oom_in_chain(e)
            if _oom is not None:
                _drop_tex_caches_on_oom()   # P1-M1-FREERETRY
                raise _oom
            msg = str(e)
            # Ordinary user/config RuntimeErrors — a device selection issue, a
            # string output without the s@ prefix, or inputs of mismatched
            # resolution — are NOT TEX bugs. Show them cleanly (no traceback / URL).
            if "TEX Error:" in msg or "size of tensor" in msg or "must match" in msg:
                raise RuntimeError(
                    f"TEX couldn't run your program:\n  {msg}\n\n"
                    f"This is usually a device, output-type, or input-size mismatch "
                    f"— check the inputs and settings feeding this node."
                ) from e
            tb = traceback.format_exc()
            raise RuntimeError(
                f"TEX hit an unexpected problem while running your code:\n  {e}\n\n"
                f"If your code looks correct, this may be a TEX bug worth reporting:\n  "
                f"{TEX_BUG_REPORT_URL}\n{tb}"
            ) from e
        except Exception as e:
            # OOM can also surface here as torch.AcceleratorError — re-raise it
            # unwrapped for ComfyUI's OOM handling (see the RuntimeError branch).
            _oom = _oom_in_chain(e)
            if _oom is not None:
                _drop_tex_caches_on_oom()   # P1-M1-FREERETRY
                raise _oom
            # Genuinely unexpected — this may be a TEX bug. Include the traceback
            # and a report link so it can be diagnosed.
            tb = traceback.format_exc()
            raise RuntimeError(
                f"TEX hit an unexpected problem while running your code:\n  {e}\n\n"
                f"If your code looks correct, this may be a TEX bug worth reporting:\n  "
                f"{TEX_BUG_REPORT_URL}\n{tb}"
            ) from e

