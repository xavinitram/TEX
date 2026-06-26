"""
TEX Wrangle Node — ComfyUI custom node for the TEX (Tensor Expression Language).

Provides a single node where users write compact TEX scripts to process
images, masks, and scalar values using per-pixel tensor operations.

Requires ComfyUI with Nodes v3 API support (comfy_api.latest).
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
from .tex_compiler.ast_nodes import SourceLoc
from .tex_runtime.interpreter import Interpreter, InterpreterError
from .tex_cache import get_cache
from .tex_runtime.compiled import execute_compiled
from .tex_fusion import prepare_fused as _prepare_fused, FusionError

# Marshalling and type inference utilities (extracted to tex_marshalling.py)
from .tex_marshalling import (
    tensor_fingerprint as _tensor_fingerprint,
    unwrap_latent as _unwrap_latent,
    convert_param_value as _convert_param_value,
    infer_binding_type as _infer_binding_type,
    prepare_output as _prepare_output,
    map_inferred_type as _map_inferred_type,
)

# ── v3 API import ──
# Falls back to a stub base class when running outside ComfyUI (e.g. standalone tests).
try:
    from comfy_api.latest import IO
    _V3_AVAILABLE = True
except ImportError:
    IO = None
    _V3_AVAILABLE = False


# ── Cached interpreter (reused across executions to avoid rebuild overhead) ──
_interpreter: Interpreter | None = None


def _get_interpreter() -> Interpreter:
    """Get or create the cached Interpreter singleton."""
    global _interpreter
    if _interpreter is None:
        _interpreter = Interpreter()
    return _interpreter


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

    # Maximum number of output slots (pre-allocated for dynamic outputs)
    MAX_OUTPUTS = 8

    # System kwargs that are NOT TEX bindings
    _SYSTEM_KWARGS = {"code", "device", "compile_mode", "_tex_any"}

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
                "Supports float, int, vec3, vec4, string types with if/else, for loops, and 100+ stdlib functions.\n"
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
                        "// Write output to @OUT\n\n"
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
                    options=["none", "torch_compile"],
                    default="none",
                    tooltip="none: standard interpreter. torch_compile: JIT-compile via torch.compile (falls back on failure).",
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
            ],
            outputs=[
                IO.AnyType.Output(
                    id=f"out_{i}",
                    display_name=f"out_{i}",
                    tooltip="TEX output. Type auto-inferred from code.",
                )
                for i in range(8)
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
                    parts.append(lat)
                elif isinstance(val, torch.Tensor):
                    parts.append(f"{name}:{_tensor_fingerprint(val)}")
                else:
                    parts.append(f"{name}:{val}")
        return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()

    @classmethod
    def execute(cls, **kwargs):
        """Execute TEX code with the provided inputs."""
        start_time = time.perf_counter()

        # Pop system parameters so they don't get treated as bindings
        code = kwargs.pop("code", "")
        device_mode = kwargs.pop("device", "auto")
        compile_mode = kwargs.pop("compile_mode", "none")
        kwargs.pop("_tex_any", None)  # search-panel wildcard slot, unused
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

        try:
            fused_chain = False
            if chain_payload:
                # Fused path: splice the whole linked chain into one program.
                # Per-stage validation already happened in compile_fused, so a
                # failure here raises (we must NOT silently run the terminal
                # alone — the upstream nodes were collapsed away in the prompt).
                spec = json.loads(chain_payload) if isinstance(chain_payload, str) else chain_payload
                (program, type_map, referenced, assigned_bindings, param_info,
                 used_builtins, bindings) = _prepare_fused(spec, code, bindings, _infer_binding_type)
                fused_chain = True
            else:
                # Infer binding types for inputs
                binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}

                # Compile (uses two-tier Mega-Cache: memory LRU + disk persistence)
                cache = get_cache()
                program, type_map, referenced, assigned_bindings, param_info, used_builtins = cache.compile_tex(code, binding_types)

            # Resolve target device (from the effective bindings — merged for a chain)
            device = cls._resolve_device(device_mode, bindings)

            # Determine output names (sorted alphabetically for stable ordering)
            output_names = sorted(assigned_bindings.keys())
            if not output_names:
                raise InterpreterError(
                    "TEX program has no outputs. Assign to @OUT or another @name.",
                    loc=SourceLoc(1, 1), source=code, code="E6001",
                )
            if len(output_names) > cls.MAX_OUTPUTS:
                raise InterpreterError(
                    f"TEX program has {len(output_names)} outputs, "
                    f"exceeding the maximum ({cls.MAX_OUTPUTS}).",
                    loc=SourceLoc(1, 1), source=code, code="E6002",
                )

            # Check that referenced input bindings are available.
            # For $params with code-defined defaults, inject the default as a
            # fallback when no widget value or wire connection is provided.
            for ref_name in referenced:
                if ref_name not in assigned_bindings and ref_name not in bindings:
                    # Try param default before erroring
                    if ref_name in param_info:
                        default_val = param_info[ref_name].get("default_value")
                        if default_val is not None:
                            bindings[ref_name] = default_val
                            continue
                    sigil = "$" if ref_name in param_info else "@"
                    # On the fused path bindings are stage-prefixed (_s0_amt); show
                    # the user's original name, not the synthetic one.
                    disp = re.sub(r"^_s\d+_", "", ref_name) if fused_chain else ref_name
                    raise InterpreterError(
                        f"TEX code references {sigil}{disp} but no input is connected to slot '{disp}'.",
                        loc=SourceLoc(1, 1), source=code, code="E6003",
                    )

            # Convert param widget values to appropriate types for the interpreter
            # (e.g. hex color strings → RGB lists, comma vec strings → float lists)
            for pname, pinfo in param_info.items():
                if pname in bindings:
                    bindings[pname] = _convert_param_value(bindings[pname], pinfo, pname)

            # Execute — optionally with torch.compile acceleration.
            # Fused chains always use the interpreter (the spliced program isn't
            # keyed in the torch_compile fingerprint cache).
            if compile_mode == "torch_compile" and not fused_chain:
                fp = cache.fingerprint(code, binding_types)
                try:
                    raw_output = execute_compiled(program, bindings, type_map, device, fp,
                                                  latent_channel_count=latent_channel_count,
                                                  output_names=output_names,
                                                  used_builtins=used_builtins)
                except Exception as compile_exc:
                    # Defense in depth: selecting torch_compile must NEVER hard-fail
                    # the node. If execute_compiled's own fallback still raised,
                    # reset dynamo on THIS thread and run the plain interpreter
                    # (dynamo state is process-global — see compiled.py).
                    logger.warning(
                        "[TEX] torch_compile path failed (%s); using interpreter.",
                        compile_exc,
                    )
                    try:
                        torch._dynamo.reset()
                    except Exception:
                        pass
                    interp = _get_interpreter()
                    raw_output = interp.execute(program, bindings, type_map, device=device,
                                                source=code,
                                                latent_channel_count=latent_channel_count,
                                                output_names=output_names,
                                                used_builtins=used_builtins)
                # torch_compile returns single value — wrap for compat
                if not isinstance(raw_output, dict):
                    raw_output = {output_names[0]: raw_output}
            else:
                interp = _get_interpreter()
                # Pass source so runtime (E6xxx) errors render a source-line caret.
                # Fused chains splice many sources, so their line numbers wouldn't
                # map to `code` — leave source empty there (errors stay message-only).
                raw_output = interp.execute(program, bindings, type_map, device=device,
                                            source=("" if fused_chain else code),
                                            latent_channel_count=latent_channel_count,
                                            output_names=output_names,
                                            used_builtins=used_builtins)

            # Format each output based on inferred type
            results = []
            output_types_log = []
            for name in output_names:
                raw = raw_output[name]
                inferred_type = assigned_bindings[name]
                effective_type = _map_inferred_type(inferred_type, has_latent_input)
                result = _prepare_output(raw, effective_type)
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
                f"Executed in {elapsed_ms:.1f}ms{compile_tag} | "
                f"device: {device} | outputs: [{', '.join(output_types_log)}] | "
                f"bindings: {list(bindings.keys())}"
            )

            # Return NodeOutput for v3, or tuple for standalone/test usage
            if _V3_AVAILABLE:
                return IO.NodeOutput(*results)
            return tuple(results)

        except TEXMultiError as e:
            # Multiple errors — human-readable first, JSON at end for frontend
            readable = "\n\n".join(d.render() for d in e.diagnostics)
            payload = json.dumps([d.to_dict() for d in e.diagnostics])
            raise RuntimeError(f"{readable}\nTEX_DIAG:{payload}") from e
        except (LexerError, ParseError, TypeCheckError, InterpreterError) as e:
            # Single error — human-readable first, JSON at end for frontend
            if hasattr(e, '_build_diagnostic'):
                e._build_diagnostic()
            if hasattr(e, 'diagnostic') and e.diagnostic:
                readable = e.diagnostic.render()
                payload = json.dumps([e.diagnostic.to_dict()])
                raise RuntimeError(f"{readable}\nTEX_DIAG:{payload}") from e
            # Fallback for errors without diagnostic (shouldn't happen)
            raise RuntimeError(f"TEX Error: {e}") from e
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
            # Genuinely unexpected — this may be a TEX bug. Include the traceback
            # and a report link so it can be diagnosed.
            tb = traceback.format_exc()
            raise RuntimeError(
                f"TEX hit an unexpected problem while running your code:\n  {e}\n\n"
                f"If your code looks correct, this may be a TEX bug worth reporting:\n  "
                f"{TEX_BUG_REPORT_URL}\n{tb}"
            ) from e

    @staticmethod
    def _resolve_device(device_mode: str, bindings: dict[str, Any]) -> str:
        """
        Resolve the target execution device.

        "auto" — infer from inputs (prefer GPU if any input is on GPU, else CPU)
        "cpu"  — force CPU execution
        "cuda" — force GPU execution (raises if CUDA unavailable)
        """
        if device_mode == "cpu":
            return "cpu"

        if device_mode == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "TEX Error: device='cuda' selected but CUDA is not available. "
                    "Select 'auto' or 'cpu' instead."
                )
            return "cuda"

        # "auto" mode: prefer GPU if any input tensor is on GPU
        # (latent dicts are already unwrapped to tensors before this is called)
        for val in bindings.values():
            if isinstance(val, torch.Tensor) and val.is_cuda:
                return str(val.device)
        return "cpu"
