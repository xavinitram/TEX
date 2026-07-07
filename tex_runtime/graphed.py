"""
UC-1 — CUDA-graph replay engine.

Caches a torch.cuda.CUDAGraph per (program x input-signature), captured around
the UNMODIFIED tree-walking interpreter. After the first two cooks, a cook is:
copy inputs into static staging buffers -> graph.replay() -> clone outputs out.
All Python dispatch and kernel launches are baked into the graph; replay is a
single launch. Measured 7-12x at 256 for launch-bound programs.

Safety is layered:
  * a static AST gate excludes anything that syncs during execution (while
    loops, non-static-range for loops, string ops, the sync-bearing stdlib
    calls) so a doomed capture is never attempted;
  * capture runs under capture_error_mode="thread_local"; ANY .item()/sync that
    slips through fails the capture loudly (never a silent wrong replay);
  * a failed capture triggers RNG-poison recovery, blacklists the key, and the
    caller falls back to the plain interpreter — execution never fails because
    of this feature;
  * keepalive refs to every module-level tensor cache keep captured addresses
    stable; the graph cache is bytes-aware and frees under memory pressure.

Entry point: run_graphed(...). Returns the output, or None to signal "not
graphable — run the interpreter".
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import torch

try:
    import comfy.model_management as _mm
except Exception:
    _mm = None

from ..tex_compiler.ast_nodes import (
    Program, ForLoop, WhileLoop, FunctionCall,
    try_extract_static_range,
    iter_child_nodes as _iter_child_nodes,
)
from .interpreter import Interpreter, _collect_identifiers

logger = logging.getLogger("TEX.graphed")

# CC-2: set while a CUDA-graph capture is running so the auto-tier defers
# submitting a background torch.compile that would fight it for the allocator.
_CAPTURING = False


def is_capturing() -> bool:
    return _CAPTURING


# stdlib calls that .item()/sync internally (blur sigma, mip lod, bilateral) —
# capturing a program that calls them is doomed, so gate them out statically.
_SYNC_STDLIB = frozenset({
    "blur", "gauss_blur", "bilateral_filter",
    "sample_mip", "sample_mip_gauss", "sample_lod",
})

# fingerprint-signature -> GraphedProgram (LRU, bytes-aware).
_graph_cache: "OrderedDict[tuple, GraphedProgram]" = OrderedDict()
# Signatures that failed capture — never retried this session.
_blacklist: set[tuple] = set()
# fingerprint -> static capturability (the AST gate is a full walk; memoize it so
# cache-hit replays don't re-walk the program every cook).
_capturable_memo: dict[str, bool] = {}

# Bytes budget for the shared pool's reserved growth. Freed wholesale when over.
_GRAPH_BYTES_BUDGET = 512 * 1024 * 1024
_graph_bytes = 0

# Process-wide kill switch: set if RNG recovery ever exhausts its retries — a
# still-poisoned generator is a hazard beyond TEX.
_graph_mode_disabled = False

# Strong refs to every module-level tensor cache, so a captured graph's baked-in
# reads never point at freed/reallocated storage (verifier fix 3).
_keepalive: list = []

# Last capture error message (diagnostics only).
_last_capture_error: list = [None]


def _build_keepalive() -> None:
    if _keepalive:
        return
    try:
        from . import stdlib as _sl
        from . import noise as _ns
        _keepalive.extend([
            _sl._sampler_cache, _sl._grid_buf, _sl._mip_cache,
            _sl._gauss_mip_cache, _sl._gauss_kernel_cache,
            _ns._worley_offsets_cache, _ns._worley3d_offsets_cache,
        ])
    except Exception:
        pass


# ── Static capturability gate ─────────────────────────────────────────

def _capturable(program: Program) -> bool:
    """True if the program has no statically-detectable sync. Excludes while
    loops, non-static-range for loops (param/uniform bounds resolve via .item()
    at loop entry — a capture blocker), and sync-bearing stdlib calls. Anything
    that slips through still fails capture loudly (never silent-wrong)."""
    stack = list(program.statements)
    while stack:
        n = stack.pop()
        cls = n.__class__
        if cls is WhileLoop:
            return False
        if cls is ForLoop and try_extract_static_range(n) is None:
            return False
        if cls is FunctionCall and n.name in _SYNC_STDLIB:
            return False
        stack.extend(_iter_child_nodes(n))
    return True


# ── RNG-poison recovery (verified protocol V4) ────────────────────────

def _recover_from_capture_failure() -> bool:
    """A failed capture leaves the CUDA generator's capture flag stuck.
    set_rng_state does NOT clear it; the working protocol is to run a trivial
    successful 1-op capture whose generator epilogue clears the flag. Returns
    True if torch.rand works afterward. Caps at 6 iterations."""
    for _ in range(6):
        try:
            torch.cuda.synchronize()
            _ = (torch.zeros(1, device="cuda") + 1)  # swallow a sticky error
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.cuda.graph(g):
                    _ = torch.zeros(1, device="cuda") + 1
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            _ = torch.rand(1, device="cuda")  # verify generator is clean
            return True
        except Exception:
            continue
    return False


# ── The captured program ──────────────────────────────────────────────

class GraphedProgram:
    def __init__(self, key: tuple):
        self.key = key
        self.graph: Any = None
        self.interp = Interpreter()          # dedicated (private builtins cache)
        self.static_bindings: dict[str, Any] = {}
        self.static_outputs: Any = None      # tensor or dict of tensors
        self.output_names: list[str] | None = None
        self.bytes = 0

    def _stage(self, bindings: dict[str, Any]) -> None:
        """Copy the current cook's inputs into the static staging buffers."""
        for name, buf in self.static_bindings.items():
            src = bindings.get(name)
            if src is None:
                continue
            if isinstance(src, list):
                src = _list_to_static(src, getattr(buf, "device", None),
                                      getattr(buf, "dtype", None))
            if isinstance(buf, torch.Tensor):
                if isinstance(src, torch.Tensor):
                    buf.copy_(src, non_blocking=True)
                else:
                    buf.fill_(float(src))

    @staticmethod
    def _make_static(value, device, dtype):
        """A stable device buffer mirroring a binding value."""
        if isinstance(value, list):
            # UC-1: a vec/color param ([r,g,b]) tensorizes to [1,1,1,C] exactly as
            # the interpreter does — collapsing to value[0] made the graph compute
            # every channel from the R component (silent wrong output). A batch
            # list (of tensors) is unwrapped upstream; if one slips through,
            # _list_to_static returns element 0.
            value = _list_to_static(value, device, dtype)
        if isinstance(value, torch.Tensor):
            b = torch.empty_like(value, device=device)
            b.copy_(value)
            return b
        if isinstance(value, str):
            return None  # strings can't be staged — program is gated out anyway
        return torch.tensor(float(value), dtype=dtype, device=device)

    def capture(self, program, bindings, type_map, device, latent_channel_count,
                output_names, precision, used_builtins) -> bool:
        """Warm up, then capture. Returns True on success."""
        global _CAPTURING
        _CAPTURING = True
        try:
            return self._capture_inner(program, bindings, type_map, device,
                                       latent_channel_count, output_names,
                                       precision, used_builtins)
        finally:
            _CAPTURING = False

    def _capture_inner(self, program, bindings, type_map, device, latent_channel_count,
                       output_names, precision, used_builtins) -> bool:
        dtype = self.interp._PRECISION_DTYPES.get(precision, torch.float32) \
            if hasattr(self.interp, "_PRECISION_DTYPES") else torch.float32
        # Build static staging buffers for every binding.
        self.static_bindings = {}
        for name, val in bindings.items():
            b = self._make_static(val, device, dtype)
            if b is None:
                return False
            self.static_bindings[name] = b
        self.output_names = output_names

        def _run():
            return self.interp.execute(
                program, self.static_bindings, type_map, device=device,
                latent_channel_count=latent_channel_count,
                output_names=output_names, precision=precision,
                used_builtins=used_builtins)

        # Warm up on a side stream (≥3 runs; UC-5 already gated fn_pow's probe).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _run()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        before = torch.cuda.memory_stats().get("reserved_bytes.all.current", 0)
        g = torch.cuda.CUDAGraph()
        # Per-graph pool (no shared pool): sharing one pool across captures of
        # different programs triggers the allocator's "use_count > 0" assert when
        # a prior graph's still-live output buffer occupies the pool, and the
        # failure cascades. Isolated pools cost a little more reserved memory
        # (bounded by _GRAPH_BYTES_BUDGET) but capture reliably.
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            out = _run()
        self.graph = g
        self.static_outputs = out
        after = torch.cuda.memory_stats().get("reserved_bytes.all.current", 0)
        self.bytes = max(0, after - before)
        return True

    def replay(self, bindings: dict[str, Any]):
        """Stage inputs, replay, and clone outputs out (ComfyUI caches node
        outputs while the next replay overwrites the static buffer)."""
        self._stage(bindings)
        self.graph.replay()
        out = self.static_outputs
        if isinstance(out, dict):
            return {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                    for k, v in out.items()}
        return out.clone() if isinstance(out, torch.Tensor) else out


# ── Capture key + entry point ─────────────────────────────────────────

def _list_to_static(value, device, dtype):
    """UC-1: shape a list binding for the graph the way the interpreter does.

    A vec/color param (list of numbers) becomes a `[1,1,1,C]` channel-last tensor
    (len 2/3/4) so all channels are staged — NOT collapsed to component 0. A
    ComfyUI batch list (of tensors, unwrapped upstream) falls back to element 0."""
    dtype = dtype or torch.float32
    if not value:
        return torch.scalar_tensor(0.0, dtype=dtype, device=device)
    if isinstance(value[0], torch.Tensor):
        return value[0]
    from .interpreter import vec_list_to_tensor
    return vec_list_to_tensor(value, dtype, device)


def _capture_key(fingerprint, device, precision, bindings, output_names,
                 latent_channel_count) -> tuple:
    dev = torch.device(device)
    tensor_sig = []
    scalar_names = []
    for name, v in bindings.items():
        if isinstance(v, list):
            # UC-1: key a vec/color param by its component count so a vec3 and a
            # vec4 param (or a scalar) never collide on the same captured graph.
            if v and isinstance(v[0], torch.Tensor):
                v = v[0]
            else:
                tensor_sig.append((name, ("list", len(v)), "param_list"))
                continue
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            tensor_sig.append((name, tuple(v.shape), str(v.dtype)))
        else:
            scalar_names.append(name)
    return (fingerprint, dev.index if dev.index is not None else torch.cuda.current_device(),
            precision, tuple(sorted(tensor_sig)), tuple(sorted(scalar_names)),
            tuple(output_names) if output_names else (), latent_channel_count)


def _free_all_graphs() -> None:
    global _graph_bytes
    _graph_cache.clear()
    _graph_bytes = 0
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _under_memory_pressure() -> bool:
    if _mm is None or not hasattr(_mm, "get_free_memory"):
        return False
    try:
        free = _mm.get_free_memory(torch.device("cuda"))
        return free is not None and free < 512 * 1024 * 1024
    except Exception:
        return False


def run_graphed(program, bindings, type_map, device, fingerprint,
                latent_channel_count=0, output_names=None, precision="fp32",
                used_builtins=None):
    """Execute via a cached CUDA graph, or return None to fall back to the
    interpreter. cuda-only; every failure path returns None."""
    global _graph_bytes
    if _graph_mode_disabled:
        return None
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return None
    cap = _capturable_memo.get(fingerprint)
    if cap is None:
        cap = _capturable(program)
        _capturable_memo[fingerprint] = cap
    if not cap:
        return None
    if used_builtins is None:
        used_builtins = _collect_identifiers(program)

    key = _capture_key(fingerprint, dev, precision, bindings, output_names,
                       latent_channel_count)
    if key in _blacklist:
        return None

    # Hot path: a captured graph exists — stage + replay, nothing else.
    gp = _graph_cache.get(key)
    if gp is not None:
        _graph_cache.move_to_end(key)
        try:
            return gp.replay(bindings)
        except Exception as e:
            logger.warning("[TEX] graph replay failed (%s); disabling this key.", e)
            if _graph_cache.pop(key, None) is not None:
                _graph_bytes -= gp.bytes  # keep the byte budget honest on eviction
            _blacklist.add(key)
            return None

    # First sight of this key: check pressure, then attempt capture.
    if _under_memory_pressure():
        _free_all_graphs()
    _build_keepalive()
    gp = GraphedProgram(key)
    try:
        ok = gp.capture(program, bindings, type_map, dev, latent_channel_count,
                        output_names, precision, used_builtins)
    except Exception as e:
        ok = False
        logger.info("[TEX] CUDA-graph capture failed (%s); using interpreter.", e)
        _last_capture_error[0] = f"{type(e).__name__}: {e}"
        if not _recover_from_capture_failure():
            _disable_graph_mode()
    if not ok:
        _blacklist.add(key)
        return None

    _graph_cache[key] = gp
    _graph_bytes += gp.bytes
    while _graph_bytes > _GRAPH_BYTES_BUDGET and len(_graph_cache) > 1:
        _, old = _graph_cache.popitem(last=False)
        _graph_bytes -= old.bytes
    try:
        return gp.replay(bindings)
    except Exception as e:
        logger.warning("[TEX] first graph replay failed (%s); using interpreter.", e)
        if _graph_cache.pop(key, None) is not None:
            _graph_bytes -= gp.bytes  # undo the += above so the budget stays honest
        _blacklist.add(key)
        return None


def _disable_graph_mode() -> None:
    global _graph_mode_disabled
    _graph_mode_disabled = True
    _free_all_graphs()
    logger.warning("[TEX] CUDA-graph mode disabled for this session "
                   "(capture-failure recovery exhausted).")


def clear_graph_cache() -> None:
    """Testing / free-memory hook."""
    global _graph_mode_disabled
    _free_all_graphs()
    _blacklist.clear()
    _graph_mode_disabled = False
