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

from .host import get_host_services  # PORT-1: the ONE seam to comfy.model_management

from ..tex_compiler.ast_nodes import (
    Program, ForLoop, WhileLoop, FunctionCall, Identifier,
    try_extract_static_range,
    iter_child_nodes as _iter_child_nodes,
)
from .interpreter import Interpreter, _collect_identifiers
from . import tier_trace  # leaf module (imports only threading) — no cycle

logger = logging.getLogger("TEX.graphed")

# CC-2: set while a CUDA-graph capture is running so the auto-tier defers
# submitting a background torch.compile that would fight it for the allocator.
_CAPTURING = False


def is_capturing() -> bool:
    return _CAPTURING


# stdlib calls that .item()/sync internally — capturing a program that calls
# them is doomed, so gate them out statically (never attempt the capture).
_SYNC_STDLIB = frozenset({
    # sample-space sync: blur sigma / mip lod / bilateral read a scalar via .item()
    "blur", "gauss_blur", "bilateral_filter",
    "sample_mip", "sample_mip_gauss", "sample_lod",
    # P1-UC1-STATIC-GATE: the octave/iteration-count noise family resolves its
    # loop count via int(_to_float(count)) → a capture-illegal .item() sync, so
    # every fresh key ate a doomed capture + RNG-poison recovery. Single-eval
    # noise (perlin/simplex/worley/voronoi/curl) has no such sync and stays
    # capturable — measured 6.4× at 256². (Verified empirically, 2026-07; the
    # doc-22 PF-3 "compile-upgrade mid-capture" hypothesis doesn't apply on a
    # no-Triton box, where the compile tier never fires — the sync is the blocker.)
    "fbm", "ridged", "billow", "turbulence", "flow", "alligator",
    # SL-4 morphology: the radius resolves via int(.item()) → capture-illegal sync.
    "erode", "dilate",
    # LX-5: debug_print records a thread-local side-effect via an .item() readout —
    # capture-illegal, and it must run on the interpreter anyway so the probe fires.
    "debug_print",
})

# fingerprint-signature -> GraphedProgram (LRU, bytes-aware).
_graph_cache: "OrderedDict[tuple, GraphedProgram]" = OrderedDict()
# Signatures that failed capture — never retried this session.
_blacklist: set[tuple] = set()
# fingerprint -> static capturability (the AST gate is a full walk; memoize it so
# cache-hit replays don't re-walk the program every cook).
_capturable_memo: "dict[str, tuple[bool, int]]" = {}

# PF-1/PF-2 — CUDA-graph crossover gate (originally calibrated on RTX 2080S /
# torch 2.10 from Appendix A + v016_bench). The tier elides a FIXED per-kernel
# launch cost but pays a staging copy that scales with pixel count, so it wins
# only for enough-kernels at low-enough-resolution. Turing measured: wins ≤512²
# broadly, breaks even/loses at 1024² unless kernel-heavy, loses everywhere for
# ~0-kernel programs, loses at 2048² for all program classes tested.
# The px ceilings are per-arch (S-5): initialized from arch_support.gate_profile()
# — a VERIFIED arch (e.g. sm_120, where low-op programs still win 1.66x at 1024²)
# gets its measured ceilings; everything else keeps the Turing values. The module
# attributes stay the contract (HW-1 canary + probes monkeypatch them).
from .arch_support import gate_profile as _gate_profile
_PROFILE = _gate_profile()
_GRAPH_MIN_OPS = 2                    # PF-2: 0/1-op programs capture an ~empty graph → pure loss
_GRAPH_HIGH_OPS = 40                  # kernel-heavy programs keep winning up to the high ceiling
_GRAPH_BASE_PX_CEIL = _PROFILE["graph_base_px_ceil"]   # low-kernel win ceiling (Turing: 512²)
_GRAPH_HIGH_PX_CEIL = _PROFILE["graph_high_px_ceil"]   # hard cap (Turing + sm_120: 1024²)

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
        from . import compiled as _cp
        _keepalive.extend([
            _sl._sampler_cache, _sl._grid_buf, _sl._mip_cache,
            _sl._gauss_mip_cache, _sl._gauss_kernel_cache,
            _ns._worley_offsets_cache, _ns._worley3d_offsets_cache,
            _cp._ENV_TENSOR_CACHE,   # codegen builtin tensors (u/v/ix/... cross-cook cache)
        ])
    except Exception:
        pass


def _iter_tensors(obj):
    """Every torch.Tensor reachable in a cache value (tensor / tuple / list / dict)."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            yield from _iter_tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _iter_tensors(x)


def _snapshot_cache_tensors() -> list:
    """MEM-1: strong refs to every tensor currently in the keepalive'd stdlib/noise
    caches — exactly the storages a just-captured graph may have baked the address
    of. Pinning them on the owning GraphedProgram means a later cache eviction can
    never free the memory that graph replays against (the UC-1 stale-address
    hazard), so eviction no longer has to tear the whole graph cache down."""
    _build_keepalive()
    out: list = []
    for cache in _keepalive:
        try:
            for v in list(cache.values()):
                out.extend(_iter_tensors(v))
        except Exception:
            pass
    return out


def pinned_storages() -> set:
    """MEM-1: `data_ptr()`s of every storage pinned by a live captured graph. The
    budget evictor skips cache entries whose storage is in here — evicting one
    reclaims 0 bytes (a live graph still holds it) and would only force a needless
    recapture. Tolerates non-GraphedProgram sentinels in the cache (tests)."""
    out: set = set()
    for gp in _graph_cache.values():
        for t in getattr(gp, "pinned_entries", ()):
            try:
                out.add(t.untyped_storage().data_ptr())
            except Exception:
                pass
    return out


def free_graphs_only() -> None:
    """MEM-1: reclaim every captured graph (and its pool) WITHOUT resetting the
    capture blacklist or the RNG-poison kill switch. The memory-pressure path uses
    this; `clear_graph_cache()` (which also clears those two) is now test-only."""
    _free_all_graphs()


# ── Static capturability gate ─────────────────────────────────────────

def _capturable(program: Program) -> tuple[bool, int]:
    """(capturable, op_count) from a single AST walk.

    capturable = no statically-detectable sync: excludes while loops, non-static
    -range for loops (param/uniform bounds resolve via .item() at loop entry — a
    capture blocker), and sync-bearing stdlib calls. Anything that slips through
    still fails capture loudly (never silent-wrong). op_count is the static
    tensor-op count (the launch proxy for the PF-1/PF-2 gate), counted in the
    same walk so the graph tier never traverses the AST twice; it is 0 (unused)
    on a non-capturable early-out.

    ENG-7 also bars the host-time builtins, and for a DIFFERENT reason than everything
    else here: they do not sync, they *change*. Every other builtin is derived from the
    binding shapes, so it is constant for a given capture key and baking it into the
    graph is correct. `frame`/`fps`/`time` come from the host's playhead, so a replay
    would keep re-serving whatever the playhead read at capture — an animation frozen on
    one frame, with no error. Unlike a sync, capture would SUCCEED and be wrong, so this
    is the one blocker that has to be caught statically or not at all. (Feeding them as
    static input buffers copied per replay is the real fix; it needs the capture plumbing
    to own the buffer, so it waits for a host that has a playhead at all.)"""
    from .compiled import _OP_TYPES   # lazy: canonical op-type set, avoids import cycle
    from .interpreter import _TIME_BUILTIN_NAMES
    ops = 0
    stack = list(program.statements)
    while stack:
        n = stack.pop()
        cls = n.__class__
        if cls is WhileLoop:
            return (False, 0)
        if cls is ForLoop and try_extract_static_range(n) is None:
            return (False, 0)
        if cls is FunctionCall and n.name in _SYNC_STDLIB:
            return (False, 0)
        if cls is Identifier and n.name in _TIME_BUILTIN_NAMES:
            return (False, 0)
        if isinstance(n, _OP_TYPES):
            ops += 1
        stack.extend(_iter_child_nodes(n))
    return (True, ops)


def _spatial_px(bindings) -> int:
    """Pixels per frame (H*W) of the program's spatial input, or 0 if none."""
    for v in bindings.values():
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            v = v[0]
        if isinstance(v, torch.Tensor) and v.dim() >= 3:
            return v.shape[1] * v.shape[2]
    return 0


def _graph_capture_worthwhile(est_ops: int, px: int) -> bool:
    """PF-1/PF-2 crossover gate: True only inside the measured win region.
    Near-zero-kernel programs never win (empty-graph overhead); low-kernel
    programs win only up to ~512²; kernel-heavy programs up to ~1024²; nothing
    tested wins above 1024². px==0 means no spatial input to size against (a
    pure-scalar program) — tiny either way, and the _GRAPH_MIN_OPS floor already
    rejected the trivial 0/1-op case, so allow rather than forgo a possible win."""
    if est_ops < _GRAPH_MIN_OPS:
        return False
    if px == 0:
        return True
    ceil = _GRAPH_HIGH_PX_CEIL if est_ops >= _GRAPH_HIGH_OPS else _GRAPH_BASE_PX_CEIL
    return px <= ceil


# ── RNG-poison recovery (verified protocol V4) ────────────────────────

def _recover_from_capture_failure(dev_index=None) -> bool:
    """A failed capture leaves the CUDA generator's capture flag stuck.
    set_rng_state does NOT clear it; the working protocol is to run a trivial
    successful 1-op capture whose generator epilogue clears the flag. Returns
    True if torch.rand works afterward. Caps at 6 iterations. HW-2: recover on the
    COOK's device — recovering on the wrong device leaves the real generator poisoned."""
    if dev_index is None:
        dev_index = torch.cuda.current_device()
    dev = f"cuda:{dev_index}"
    for _ in range(6):
        try:
            with torch.cuda.device(dev_index):
                torch.cuda.synchronize()
                _ = (torch.zeros(1, device=dev) + 1)  # swallow a sticky error
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    with torch.cuda.graph(g):
                        _ = torch.zeros(1, device=dev) + 1
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                _ = torch.rand(1, device=dev)  # verify generator is clean
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
        # MEM-1: strong refs to the stdlib/noise cache tensors live at capture time
        # (the addresses this graph may have baked). Keeps them from being freed by a
        # later budget eviction, so eviction stays surgical instead of nuking graphs.
        self.pinned_entries: list = []

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
        """Warm up, then capture. Returns True on success. HW-2: pin capture to the
        COOK's device — a cuda:1 cook must capture on cuda:1's stream, not whatever
        device happens to be current, else capture fails loudly and RNG-recovery runs
        against the wrong generator (spuriously tripping the process-wide kill switch)."""
        global _CAPTURING
        _CAPTURING = True
        idx = _dev_index(device)
        try:
            with torch.cuda.device(idx):
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
        # MEM-1: pin the cache tensors this capture may have baked (see field doc).
        self.pinned_entries = _snapshot_cache_tensors()
        return True

    def replay(self, bindings: dict[str, Any]):
        """Stage inputs, replay, and clone outputs out (ComfyUI caches node
        outputs while the next replay overwrites the static buffer)."""
        self._stage(bindings)
        with torch.cuda.device(self.key[1]):   # S5 (doc 33): replay on the COOK's device
            self.graph.replay()                #   index (key[1]), not the ambient current one
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


def _dev_index(device) -> int:
    """HW-2: the concrete CUDA device index for a device (falling back to the current
    device when unspecified) — the one place that idiom lives."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    return dev.index if dev.index is not None else torch.cuda.current_device()


def _free_all_graphs() -> None:
    global _graph_bytes
    _graph_cache.clear()
    _graph_bytes = 0
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _under_memory_pressure(device=None) -> bool:
    try:
        # HW-2/S4: query the COOK's device, not a bare "cuda" (device 0) — default to the
        # ambient current device index when the caller doesn't pass one.
        dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
        free = get_host_services().get_free_memory(dev)
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
    capturable, est_ops = cap
    if not capturable:
        return None
    # PF-1/PF-2: skip capture/replay outside the measured win region (0-kernel
    # programs at any resolution; low-kernel programs above ~512²) so cuda_graph
    # is never slower than eager. Each resolution is its own capture key, so this
    # decides per (program, resolution); below-region → fall to the interpreter.
    if not _graph_capture_worthwhile(est_ops, _spatial_px(bindings)):
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
            out = gp.replay(bindings)
            tier_trace.record("graph")
            return out
        except Exception as e:
            logger.warning("[TEX] graph replay failed (%s); disabling this key.", e)
            if _graph_cache.pop(key, None) is not None:
                _graph_bytes -= gp.bytes  # keep the byte budget honest on eviction
            _blacklist.add(key)
            return None

    # First sight of this key: check pressure, then attempt capture. HW-2: all device-
    # scoped ops target the COOK's device index, not a bare "cuda".
    dev_index = _dev_index(dev)
    if _under_memory_pressure(dev):
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
        if not _recover_from_capture_failure(dev_index):
            _disable_graph_mode()
    if not ok:
        _blacklist.add(key)
        return None

    _graph_cache[key] = gp
    _graph_bytes += gp.bytes
    _popped = False
    while _graph_bytes > _GRAPH_BYTES_BUDGET and len(_graph_cache) > 1:
        _, old = _graph_cache.popitem(last=False)
        _graph_bytes -= old.bytes
        _popped = True
    if _popped:
        # MEM-1: an evicted graph's per-graph pool blocks return to the allocator
        # only after empty_cache() (measured: 0 bytes reclaimed otherwise, and
        # _graph_bytes undercounts real reserved growth >2x — this keeps the pool
        # from stranding when the LRU rolls over).
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    try:
        out = gp.replay(bindings)
        tier_trace.record("graph")
        return out
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
    """Test-only full reset: graphs + `_blacklist` + the RNG-poison kill switch.
    The memory-pressure/free-memory path uses `free_graphs_only()` instead (MEM-1),
    so reclaiming memory never re-arms a doomed capture."""
    global _graph_mode_disabled
    _free_all_graphs()
    _blacklist.clear()
    _graph_mode_disabled = False
