"""
TEX Noise Library — procedural noise functions for the TEX DSL.

All noise functions operate on PyTorch tensors and return tensors.
Uses arithmetic hash (lowbias32) for TorchInductor-friendly execution.
Supports both 2D and 3D evaluation.

Noise types:
  - Perlin (2D/3D): gradient noise with quintic interpolation
  - Simplex (2D): simplex grid noise
  - FBM: fractional Brownian motion with tiered compilation (eager → jit.trace → torch.compile)
  - Worley/Voronoi (2D/3D): cell-based distance noise (F1, F2)
  - Curl (2D/3D): divergence-free flow field from Perlin potential
  - Ridged (2D/3D): ridged multi-fractal with weight feedback
  - Billow (2D/3D): abs(perlin) FBM, remapped to [-1,1]
  - Turbulence (2D/3D): abs(perlin) FBM, normalized [0,1]
  - Flow (2D/3D): time-varying domain-rotated Perlin
  - Alligator (2D/3D): layered inverted Worley ridges
"""
from __future__ import annotations
import math
import threading
import torch


# Simplex skew/unskew constants
_SKEW_2D = 0.5 * (math.sqrt(3.0) - 1.0)      # ~0.3660254
_UNSKEW_2D = (3.0 - math.sqrt(3.0)) / 6.0     # ~0.2113249


# Worley/Voronoi 9-neighbor offsets, cached per device to avoid per-call allocation
_worley_offsets_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

def _get_worley_offsets(device: torch.device, ndim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached (dx_off, dy_off) tensors for Worley 9-neighbor lookup."""
    key = (device, ndim)
    cached = _worley_offsets_cache.get(key)
    if cached is not None:
        return cached
    offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                            [0, -1],  [0, 0],  [0, 1],
                            [1, -1],  [1, 0],  [1, 1]],
                           dtype=torch.int32, device=device)
    extra_dims = (1,) * ndim
    dx_off = offsets[:, 0].view(9, *extra_dims)
    dy_off = offsets[:, 1].view(9, *extra_dims)
    _worley_offsets_cache[key] = (dx_off, dy_off)
    return dx_off, dy_off


_worley3d_offsets_cache: dict[tuple, tuple] = {}


def _get_worley3d_offsets(device: torch.device, ndim: int) -> tuple:
    """Cached (dx, dy, dz) offsets for the Worley 27-neighbor 3D lookup —
    mirrors _get_worley_offsets so the meshgrid isn't rebuilt (and re-uploaded)
    on every call."""
    key = (device, ndim)
    cached = _worley3d_offsets_cache.get(key)
    if cached is not None:
        return cached
    r = torch.tensor([-1, 0, 1], dtype=torch.int32, device=device)
    gz, gy, gx = torch.meshgrid(r, r, r, indexing="ij")
    shape = (27,) + (1,) * ndim
    offs = (gx.reshape(shape), gy.reshape(shape), gz.reshape(shape))
    _worley3d_offsets_cache[key] = offs
    return offs


# ── Coordinate rank normalization ────────────────────────────────────────────
#
# Nothing obliges a caller to pass coordinates that all share a shape. A constant
# argument — `fbm(u * 8.0, v * 8.0, 0.5, 4)`, `curl(0.5, v * 8.0)` — arrives here
# as a 0-dim tensor sitting beside full [B, H, W] grids, and the pointwise noise
# bodies broadcast it without complaint.
#
# Two kinds of code do NOT broadcast it: the GPU batching paths, which stack N
# frequency- or offset-shifted copies of each coord onto a new leading axis, and
# the accumulator allocations, which size themselves from coords[0] alone. Both
# used to work only because every coord happened to carry the grid's rank — so a
# scalar coord made noise diverge by device (the batching is GPU-only) or fail
# outright. These helpers state the assumption and hold it for any mix of ranks.

def _widest(coords: tuple) -> torch.Tensor:
    """The coord that carries the broadcast result's rank, dtype and device.

    Ties go to the first, so when every coord already shares a shape this is
    coords[0] and the callers below reduce exactly to what they did before.
    """
    return max(coords, key=lambda c: c.dim())


def _stack_coord(parts: list, coord: torch.Tensor, rank: int) -> torch.Tensor:
    """`torch.stack(parts, dim=0)` with the new leading axis kept leading.

    Every entry of `parts` derives from `coord`, so they all carry its shape.
    When that rank is below `rank` — the widest coord in the batch — a plain
    stack is wrong: a 0-dim coord stacks to [N], which broadcasting then RIGHT-
    aligns against a sibling's [N, B, H, W], putting N against W. Padding the gap
    with singleton dims reproduces exactly the alignment the un-stacked coords
    would have had.

    The padding is free: a scalar stays [N, 1, 1, 1] instead of being expanded
    out to the grid, so batching a constant coord costs no extra memory.
    """
    stacked = torch.stack(parts, dim=0)
    pad = rank - coord.dim()
    if pad:
        stacked = stacked.reshape((len(parts),) + (1,) * pad + tuple(coord.shape))
    return stacked


def _align_coords(coords: tuple, ref: torch.Tensor) -> tuple:
    """Coords on ONE device and ONE dtype, ready to be stacked.

    Stacking is what forces this. An un-stacked 0-dim coord is dtype-WEAK — torch
    promotes `fp32_grid + fp16_scalar` to fp32, which is why the per-octave path never
    cared — but `_stack_coord` turns it into a RANKED tensor, and a ranked tensor is
    dtype-STRONG. `torch.lerp` on CUDA then rejects a weight whose dtype differs from its
    endpoints ("expected dtype float for `weight` but got c10::Half") where CPU's lerp
    accepts it: a device divergence with the same cause as the rank one, a constant coord
    that does not look like its siblings. It is reachable — under `precision="fp16"` the
    engine hands the noise layer fp32 `u`/`v` grids and casts the literal to fp16, so
    `fbm(u*8.0, v*8.0, 0.5, 4)` hits it.

    Promoting up front restores exactly what the un-stacked path computes (torch's own
    promotion rule, applied once instead of per-op), and the device hop covers a constant
    that `_to_tensor` built on the CPU. Both are no-ops when the coords already agree, so
    the fp32 same-shape path is untouched.
    """
    dt = coords[0].dtype
    for c in coords[1:]:
        dt = torch.promote_types(dt, c.dtype)
    return tuple(c if (c.dtype == dt and c.device == ref.device)
                 else c.to(device=ref.device, dtype=dt)
                 for c in coords)


def _zeros_broadcast(coords: tuple) -> torch.Tensor:
    """A zero accumulator shaped like the broadcast of every coord.

    `torch.zeros_like(coords[0])` is correct only while the coords share a shape:
    a scalar in the first slot makes a 0-dim accumulator, and the first in-place
    `add_` of a full grid into it fails outright. dtype and device come from the
    widest coord, so a 0-dim constant cannot drag the accumulator off the cook's
    precision or device.
    """
    ref = _widest(coords)
    return torch.zeros(torch.broadcast_shapes(*(c.shape for c in coords)),
                       dtype=ref.dtype, device=ref.device)


# ── Arithmetic hash Perlin noise (table-free, TorchInductor-friendly) ────────
#
# Replaces permutation table lookups with pure integer arithmetic (lowbias32
# hash by Chris Wellons). Gradient selection uses branch-free bit arithmetic
# instead of table gathers. This enables full kernel fusion under torch.compile.
#
# The 8-gradient set matches the classic Perlin 2D set:
#   h&7: 0→(1,0) 1→(-1,0) 2→(0,1) 3→(0,-1)
#        4→(1,1) 5→(-1,1) 6→(1,-1) 7→(-1,-1)
# (diagonal components are NOT normalized to 1/√2 — this matches the original
#  _GRAD2 table which uses 0.7071, but the arithmetic version uses ±1 for
#  diagonals. The visual difference is negligible and the output range is similar.)


def _lowbias32(x: torch.Tensor) -> torch.Tensor:
    """lowbias32 hash (Chris Wellons). Maps int32 → int32 with good avalanche.

    CRITICAL: PyTorch >> on signed int is arithmetic shift (sign-extends).
    We mask after every shift to emulate logical (unsigned) shift right.
    """
    x = x ^ (torch.bitwise_and(x >> 16, 0x0000FFFF))
    x = x * 0x21f0aaad
    x = x ^ (torch.bitwise_and(x >> 15, 0x0001FFFF))
    x = x * 0x735a2d97
    x = x ^ (torch.bitwise_and(x >> 15, 0x0001FFFF))
    return x


def _grad2d_dot(h: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Branch-free gradient dot product for the 8-gradient Perlin set.

    Given hash h (int32) and fractional offsets dx, dy, computes dot(grad, (dx, dy))
    using only arithmetic — no table lookups or torch.where.

    Gradient mapping (h & 7):
      0: ( 1, 0) → dx       1: (-1, 0) → -dx
      2: ( 0, 1) → dy       3: ( 0,-1) → -dy
      4: ( 1, 1) → dx+dy    5: (-1, 1) → -dx+dy
      6: ( 1,-1) → dx-dy    7: (-1,-1) → -dx-dy

    Bit decomposition:
      b0 = h & 1  → sign bit (used by both cardinal and diagonal)
      b1 = (h>>1) & 1 → axis select for cardinal / y-sign for diagonal
      b2 = (h>>2) & 1 → diagonal flag (0=cardinal, 1=diagonal)

    Cardinal (b2=0): b0 controls sign, b1 selects axis (0=x, 1=y)
    Diagonal (b2=1): b0 controls x-sign, b1 controls y-sign
    """
    h7 = h & 7
    b0 = (h7 & 1).float()
    b1 = ((h7 >> 1) & 1).float()
    b2 = ((h7 >> 2) & 1).float()

    sign_b0 = 1.0 - 2.0 * b0  # +1 or -1
    sign_b1 = 1.0 - 2.0 * b1  # +1 or -1

    # Cardinal: gx = sign_b0 * (1-b1), gy = sign_b0 * b1
    # Diagonal: gx = sign_b0,          gy = sign_b1
    # Combined via b2 blend:
    gx = sign_b0 * (1.0 - b1 + b2 * b1)
    gy = (1.0 - b2) * sign_b0 * b1 + b2 * sign_b1

    return gx * dx + gy * dy


def _perlin2d_fast(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Arithmetic hash Perlin noise (no table lookups).

    Pure point-wise arithmetic: no external state access.
    Fully fusible by TorchInductor (torch.compile) and traceable by torch.jit.trace.
    """
    x_floor = torch.floor(x)
    y_floor = torch.floor(y)
    xi = x_floor.to(torch.int32)
    yi = y_floor.to(torch.int32)

    xf = x - x_floor
    yf = y - y_floor

    u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
    v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

    # Arithmetic hash for 4 corners (pre-compute yi products to avoid redundant muls)
    yi_hash = yi * 0x1B873593
    yi1_hash = (yi + 1) * 0x1B873593
    h00 = _lowbias32(xi ^ yi_hash)
    h10 = _lowbias32((xi + 1) ^ yi_hash)
    h01 = _lowbias32(xi ^ yi1_hash)
    h11 = _lowbias32((xi + 1) ^ yi1_hash)

    xf1 = xf - 1.0
    yf1 = yf - 1.0

    # Gradient dot products — fully inlined for clean tracing
    g00 = _grad2d_dot(h00, xf, yf)
    g10 = _grad2d_dot(h10, xf1, yf)
    g01 = _grad2d_dot(h01, xf, yf1)
    g11 = _grad2d_dot(h11, xf1, yf1)

    return torch.lerp(torch.lerp(g00, g10, u), torch.lerp(g01, g11, u), v)


def _simplex_grad_dot(h: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Arithmetic 12-gradient dot product for 2D simplex noise.

    Uses 4 bits of the hash to select from 12 gradient directions via
    branch-free arithmetic. The 12 gradients are:
      (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1),
      (1,0.5), (-1,0.5), (1,-0.5), (-1,-0.5)
    """
    # Use bits 0-3 of hash, mod 12
    gi = (h & 0xF) % 12

    # Compute gradient components arithmetically:
    # gx: 0 for gi in {2,3}, else +1 or -1 based on gi&1
    # gy: 0 for gi in {0,1}, else +1 or -1, halved for gi >= 8
    gx_mask = ((gi != 2) & (gi != 3)).float()
    gx_sign = 1.0 - 2.0 * (gi & 1).float()
    gx = gx_mask * gx_sign

    gy_mask = ((gi != 0) & (gi != 1)).float()
    # Cardinal-y (gi in {2,3}): sign from bit 0 so gi=2→(0,+1), gi=3→(0,-1).
    # Diagonal/half (gi>=4): sign from bit 1, matching the documented set.
    gy_sign_bit = torch.where(gi >= 4, (gi >> 1) & 1, gi & 1).float()
    gy_sign = 1.0 - 2.0 * gy_sign_bit
    gy_mag = torch.where(gi >= 8, 0.5, 1.0)
    gy = gy_mask * gy_sign * gy_mag

    return gx * dx + gy * dy


def _simplex2d_fast(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """2D Simplex noise using arithmetic hash (table-free, Inductor-friendly).

    Uses _lowbias32 hash instead of permutation table for TorchInductor fusion.
    """
    # Skew input space to determine simplex cell
    s = (x + y) * _SKEW_2D
    i = torch.floor(x + s).to(torch.int32)
    j = torch.floor(y + s).to(torch.int32)

    # Unskew back to (x, y) space
    t = (i + j).float() * _UNSKEW_2D
    X0 = i.float() - t
    Y0 = j.float() - t

    # Distances from cell origin
    x0 = x - X0
    y0 = y - Y0

    # Determine which simplex triangle we're in
    gt = (x0 > y0)
    i1 = gt.to(torch.int32)
    j1 = 1 - i1

    # Offsets for middle and last corners
    x1 = x0 - i1.float() + _UNSKEW_2D
    y1 = y0 - j1.float() + _UNSKEW_2D
    x2 = x0 - 1.0 + 2.0 * _UNSKEW_2D
    y2 = y0 - 1.0 + 2.0 * _UNSKEW_2D

    # Arithmetic hash for 3 corners (no permutation table)
    h0 = _lowbias32(i * 0x1B873593 ^ j * 0x27D4EB2D)
    h1 = _lowbias32((i + i1) * 0x1B873593 ^ (j + j1) * 0x27D4EB2D)
    h2 = _lowbias32((i + 1) * 0x1B873593 ^ (j + 1) * 0x27D4EB2D)

    # Corner contributions: radial falloff (0.5 - d²)⁴ × dot(gradient, offset)
    t0 = torch.clamp(0.5 - x0 * x0 - y0 * y0, min=0.0)
    t0 = t0 * t0; t0 = t0 * t0
    n0 = t0 * _simplex_grad_dot(h0, x0, y0)

    t1 = torch.clamp(0.5 - x1 * x1 - y1 * y1, min=0.0)
    t1 = t1 * t1; t1 = t1 * t1
    n1 = t1 * _simplex_grad_dot(h1, x1, y1)

    t2 = torch.clamp(0.5 - x2 * x2 - y2 * y2, min=0.0)
    t2 = t2 * t2; t2 = t2 * t2
    n2 = t2 * _simplex_grad_dot(h2, x2, y2)

    # Scale to ~[-1, 1]
    return 70.0 * (n0 + n1 + n2)


# Number of calls before attempting torch.compile (allows jit.trace to warm up first)
_COMPILE_AFTER_CALLS = 3

# Upper bound on the settle loop below. torch's profiling executor needs
# `_jit_get_num_profiled_runs()` (1 on every torch TEX supports) unoptimized runs, so
# convergence normally lands on the 2nd or 3rd; 8 is slack for a future torch that
# profiles more, and a hard stop so a pathological input cannot spin.
_SETTLE_MAX_RUNS = 8


def _profiled_runs() -> int:
    """How many UNOPTIMIZED passes TorchScript runs per new input signature before it
    installs the fused plan. 1 on every torch TEX currently supports.

    The fallback is deliberately 2, not 1: over-burning costs one extra evaluation once
    per signature, while under-burning silently reopens the reproducibility hole.
    """
    try:
        return max(1, int(torch._C._jit_get_num_profiled_runs()))
    except Exception:
        return 2


def _bitwise_same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-pattern equality — deliberately stricter than `torch.equal` in both directions.

    NaN must compare EQUAL to itself here (torch.equal says no), or a program that legally
    produces NaN could never settle; and -0.0 must compare UNEQUAL to +0.0 (torch.equal
    says yes), because reproducibility is a claim about the bits a user gets back, and a
    sign-flipped zero survives into `1.0/x`. Comparing the byte view gives both.
    """
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    # reshape(-1) BEFORE the byte view: `Tensor.view(dtype)` rejects a 0-dim tensor
    # outright ("self.dim() cannot be 0 to view Float as Byte"), and scalar coordinates
    # are ordinary in TEX — `simplex(2.0, 3.0)` produces exactly that, which broke seven
    # example programs (caustics, simplex_terrain, wood_grain, …) when this compared the
    # 0-dim form directly. Flattening first also normalises any non-contiguous layout.
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                       b.contiguous().reshape(-1).view(torch.uint8))


# ── 3-tier compilation cache helper ─────────────────────────────────────────
#
# Encapsulates the eager → jit.trace → torch.compile upgrade pattern used by
# simplex, FBM, and Worley noise. Each noise type gets its own _TieredCache
# instance, eliminating the per-type cache dict / lock / counter boilerplate.

class _TieredCache:
    """3-tier compilation cache: eager → jit.trace → torch.compile.

    Usage:
        cache = _TieredCache()
        # In the noise function:
        cached = cache.get(key)
        if cached is not None:
            cache.try_upgrade(key, compile_fn)
            return cache.get(key)(*args)
        result = eager_fn(*args)
        cache.store(key, lambda: torch.jit.trace(eager_fn, args))
        return result
    """

    def __init__(self, name: str = "noise"):
        self.name = name          # P6: which noise type, for the compile-visibility event
        self.cache: dict = {}
        self._compile_attempted: set = set()
        self._call_count: dict = {}
        self._settled: set = set()   # (key, shape, dtype) signatures past the profiling window
        self._lock = threading.Lock()

    def get(self, key):
        """Return cached callable, or None. False (trace failed) returns None."""
        val = self.cache.get(key)
        return val if val is not None and val is not False else None

    def try_upgrade(self, key, compile_fn, device=None):
        """Attempt torch.compile upgrade after _COMPILE_AFTER_CALLS calls.

        compile_fn() should return the compiled callable, or raise on failure —
        and it must WARM the compiled callable on the target device, so a
        backend failure (e.g. Triton missing for CUDA) is caught here and the
        traced tier stays in place, instead of caching a callable that raises
        lazily at the real call site.

        KNOWN, UNCLOSED: this promotion swaps the callable mid-process, so if the
        Inductor tier is not bit-identical to the traced tier, cook #4 onward can
        differ from cooks #1-3 — the same shape of defect first_call() closes for
        tier 1 vs tier 2. It is left open deliberately, on two grounds: unlike the
        tier-1 swap (which bought nothing) this one is worth a measured 13-18x, and
        it could not be characterized on the box where this was fixed — neither
        backend compiles there (CUDA has no Triton; the CPU Inductor build fails
        with a CppCompileError), so tier 3 never engages and a bit-equality
        adoption gate could not be tested. Gating adoption on
        `torch.equal(compiled(*probe), incumbent(*probe))` using the dummy
        compile_fn already warms with would close it for ~one 64x64 call on a path
        that already costs ~28s — but on a Triton box that gate would silently
        forfeit the speedup whenever the tiers disagree, so it wants a measurement
        first, not a guess. Nothing here is a *new* regression: this path predates
        the cold-frame fix and is unchanged by it.
        """
        if key in self._compile_attempted or not _can_inductor_compile(device):
            return
        with self._lock:
            if key in self._compile_attempted:
                return
            count = self._call_count.get(key, 0) + 1
            self._call_count[key] = count
            if count < _COMPILE_AFTER_CALLS:
                return
            self._compile_attempted.add(key)
        try:
            import time as _time
            from . import tier_trace as _tt
            _t0 = _time.perf_counter()
            built = compile_fn()
            # Clear the settled marks BEFORE publishing the new callable, never after:
            # the reverse order leaves a window where another thread reads the new tier
            # while the old tier's signatures still read as settled, and serves an
            # unsettled value from it — the exact defect _settle exists to prevent.
            self.forget_settled(key)
            self.cache[key] = built
            _tt.record_noise_compile(self.name, (_time.perf_counter() - _t0) * 1000.0)  # P6
        except Exception:
            pass
        self._call_count.pop(key, None)

    def store(self, key, trace_fn):
        """Store a traced version on first call. trace_fn() → traced callable or raise.

        The trace is built eagerly on the first call: building it runs the eager
        function once more (a one-time double-eval on the very first frame), but
        it makes the traced fast path available from the *second* call onward.
        Deferring the build to the second call instead measurably regressed warm
        CPU throughput (more eager-path runs in steady state), which is the case
        that matters for repeated cooking — so we build on first touch.
        """
        if key in self.cache:
            return
        try:
            self.cache[key] = trace_fn()
        except Exception:
            self.cache[key] = False

    def call(self, key, args, *, device, trace_fn, compile_fn, eager_fn):
        """The one entry point for a tiered noise fn: tier selection + settle discipline.

        Two separate things used to make the value depend on the CALL INDEX rather than on
        the inputs, and both are closed here:

          1. The cold frame returned the EAGER result and cached a trace for every call
             after it, so call #1 of a process ran a different tier than calls #2+.
          2. A traced module is not one numeric object. See _settle().

        Falls back to `eager_fn` whenever no stable compiled tier is available — a failed
        trace, or a signature that refused to settle. Slower and always right, the same
        direction every other TEX tier falls back in.
        """
        fn = self.get(key)
        if fn is not None:
            self.try_upgrade(key, compile_fn, device=device)
            fn = self.get(key)
        elif key not in self.cache:
            # Cold frame: build the trace and answer THIS call from it as well, so tier 1
            # and tier 2 are the same callable by construction. This is free — torch.jit
            # .trace already evaluates the function while recording, so the cold frame was
            # always paying for two evaluations; it only changes which result is returned.
            self.store(key, trace_fn)
            fn = self.get(key)
        if fn is None:
            return eager_fn(*args)
        return self._settle(key, fn, eager_fn, args)

    def _settle(self, key, fn, eager_fn, args):
        """Run `args` through `fn`, first settling a NEW (shape, dtype) signature.

        A traced module is not one numeric object. TorchScript's profiling executor runs
        `torch._C._jit_get_num_profiled_runs()` (= 1) UNOPTIMIZED passes for each new input
        signature and only then installs the fused plan. Measured on CUDA: a module traced
        at 24x32 and then called at 48x64 returned the *eager* value on that shape's first
        call and the fused value forever after (4d40b133 → 952dbe87), and a settled
        signature stays settled even after other shapes run. The same reopening happens per
        dtype — an fp16 cook through the fp32-recorded trace (the cache key is device-only).

        So pinning the cold frame to the traced tier is necessary but NOT sufficient: on its
        own it fixes the process-first cook and silently reopens the identical bug at every
        new resolution — the 512→1024→512 dance is ordinary ComfyUI use, and each new size
        would render one odd frame.

        Each new signature is therefore run until two consecutive results are bit-identical,
        and the settled one is what the caller gets. Because the transition is one-way per
        signature, this is a fixed cost, not a recurring one.

        MEASURED COST (RTX 2080 SUPER, torch 2.5.0+cu118) — settling adds ONE evaluation per
        new signature, and nothing else:
          * steady state: ~1 us/call for the signature build + set lookup, ~0.03% of a warm
            512² fbm(6) call (3.7 ms) and ~0.2% of a warm 512² simplex call (0.56 ms). Warm
            throughput is unchanged inside run-to-run noise.
          * per new (shape, dtype): the big number in a profile here is NOT ours. A raw
            traced module at a new 640² shape already timed 4.74 / 1983.32 / 0.79 ms for its
            first three calls — the NNC fuser compiles the kernel for each new shape, and
            that ~2 s was always paid, just on call #2. Settling makes call #1 pay it
            instead (measured 1993.75 ms vs 1988.85 ms for the same three calls), so the
            delta is a single extra evaluation: ~0.65 ms at 640² CUDA, ~6 ms at 640² CPU.
        """
        sig = (key,) + tuple((tuple(a.shape), a.stride(), a.dtype)
                             for a in args if isinstance(a, torch.Tensor))
        if sig in self._settled:
            return fn(*args)

        # STRIDES are in the signature, not just shape+dtype, because torch's guards are.
        # Measured: after a contiguous 64x64 settled, a TRANSPOSED 64x64 view (same shape,
        # same dtype, stride (1,64)) went ff73847f → b45a711d — the gap reopened on a
        # tensor a shape-only signature calls identical. `is_contiguous()` as a bool is not
        # enough either: transposed (1,64) and expanded (0,1) are both False yet are
        # different guard classes.

        # Inside a real CUDA-graph capture, do NOT settle: the discarded runs would be
        # RECORDED into the graph and re-executed on every replay, forever. This tests the
        # stream rather than graphed.is_capturing(), which is also set during the warm-up
        # that precedes capture — warm-up is exactly when we DO want to settle, so that by
        # capture time the signature is already known and this guard never fires.
        if args and isinstance(args[0], torch.Tensor) and args[0].is_cuda \
                and torch.cuda.is_current_stream_capturing():
            return fn(*args)

        # Burn the profiling passes explicitly rather than inferring them from the first
        # two results agreeing. If a future torch profiles more than once, runs 1..N are
        # ALL unoptimized and would agree with each other, so a bare last-two-agree rule
        # would settle on the unfused value and silently reopen this hole.
        for _ in range(_profiled_runs()):
            fn(*args)

        out = fn(*args)
        for _ in range(_SETTLE_MAX_RUNS):
            nxt = fn(*args)
            if _bitwise_same(out, nxt):
                self._settled.add(sig)
                return nxt
            out = nxt

        # Never converged. Demote the key to eager for the rest of the process: store()
        # short-circuits on the False sentinel and get() maps it to None, so every later
        # call takes the eager path above.
        self.cache[key] = False
        return eager_fn(*args)

    def forget_settled(self, key):
        """Drop the settled signatures for `key` — its callable is about to be replaced.

        A settled signature is a claim about ONE callable's profiling state. Carrying it
        across a tier swap would let the very first call of the new tier be served
        unsettled, which is exactly the defect this class now exists to prevent.
        """
        self._settled = {s for s in self._settled if s[0] != key}


_simplex_cache = _TieredCache("simplex")


def _compile_noise(fn):
    """P2: compile a noise fn with `dynamic=True` so ONE kernel serves every resolution.
    Without it, a resolution dance (512->1024->512) thrashes torch.compile's per-shape
    guards -- the measured 134x / 5.6 s recompile stall. Verified: 1 graph across
    512/1024/2048, within ~1 fp32 ULP of the static-compiled path (invariant-#2 gate),
    full compile speedup kept (~13-18x over eager). The cache key stays shape-UNAWARE
    (one entry), so no per-shape re-trace fragility."""
    return torch.compile(fn, backend='inductor', fullgraph=True, dynamic=True)


def _compile_simplex(device):
    """Compile simplex with Inductor and warm it on the target device."""
    compiled = _compile_noise(_simplex2d_fast)
    dummy = torch.rand(1, 64, 64, device=device)
    compiled(dummy, dummy)
    return compiled


def _simplex2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """2D Simplex noise with 3-tier compilation (eager → jit.trace → torch.compile)."""
    key = x.device
    return _simplex_cache.call(
        key, (x, y), device=key,
        trace_fn=lambda: torch.jit.trace(_simplex2d_fast, (x, y)),
        compile_fn=lambda: _compile_simplex(key),
        eager_fn=_simplex2d_fast)


def _make_fbm_fast_fn(octaves: int):
    """Build a traceable FBM function using arithmetic hash noise.
    No table arguments needed — all hashing is pure arithmetic.
    """
    max_amp = sum(0.5 ** i for i in range(octaves))
    inv_max = 1.0 / max_amp

    def fbm_fn(x, y):
        result = _perlin2d_fast(x, y)
        freq = 2.0
        amp = 0.5
        for _ in range(octaves - 1):
            result = result + _perlin2d_fast(x * freq, y * freq) * amp
            amp = amp * 0.5
            freq = freq * 2.0
        return result * inv_max
    return fbm_fn


_fbm_cache = _TieredCache("fbm")


_inductor_available: dict[str, bool] = {}  # device type -> backend availability

def _can_inductor_compile(device=None) -> bool:
    """Check if TorchInductor can compile for *device* (default: CPU).

    The backend requirement is per-device: CUDA needs Triton, CPU needs a host
    C++ compiler (MSVC on Windows). Gating on the wrong one caches a compiled
    callable that raises at its first real call (bit fbm/ridged/billow/
    turbulence on Triton-less CUDA boxes).
    """
    dev_type = device.type if isinstance(device, torch.device) else ("cuda" if device == "cuda" else "cpu")
    cached = _inductor_available.get(dev_type)
    if cached is not None:
        return cached
    if dev_type == "cuda":
        import importlib.util as _ilu
        ok = _ilu.find_spec("triton") is not None
    else:
        import shutil
        import sys
        if sys.platform != 'win32':
            ok = True  # Linux/macOS have gcc/clang by default
        else:
            # Try the robust MSVC setup from compiled.py
            try:
                from .compiled import _setup_msvc_env as _setup_compiled_msvc
                _setup_compiled_msvc()
            except Exception:
                pass
            ok = shutil.which('cl') is not None
    if ok:
        # Persist compiled kernels across restarts: point inductor's disk cache
        # at TEX's owned dir (shared helper — the TORCHINDUCTOR_CACHE_DIR env var,
        # since torch._inductor.config.cache_dir does not exist on torch 2.10).
        try:
            from .compiled import _ensure_inductor_cache_dir
            _ensure_inductor_cache_dir()
        except Exception:
            pass
    _inductor_available[dev_type] = ok
    return ok


def _fbm2d(x: torch.Tensor, y: torch.Tensor, octaves: int) -> torch.Tensor:
    """Fractional Brownian Motion using Perlin noise.
    Persistence=0.5, lacunarity=2.0. Octaves clamped to 1-10.

    Uses arithmetic hash (table-free) noise for TorchInductor-friendly execution.
    Execution tiers:
      1. First call: eager arithmetic hash (~100ms at 512x512)
      2. Second call: torch.jit.trace (~94ms — modest improvement)
      3. After 3 calls: torch.compile/Inductor (~16ms — 6x speedup, ~28s one-time compile)
    Falls back gracefully if MSVC is unavailable (stays on jit.trace tier).
    """
    octaves = max(1, min(octaves, 10))
    key = (octaves, x.device)

    # ONE function body, used to build the trace, to answer the cold frame, and as the
    # eager fallback — see _TieredCache.call. What this replaces was a hand-mirrored eager
    # copy of fbm_fn, carrying the note "MUST be bit-identical to _make_fbm_fast_fn's
    # fbm_fn so the cold (eager) frame matches every subsequent traced/compiled frame".
    # That goal was right and the mechanism could not deliver it: on CUDA the fuser
    # reassociates the very source the copy was mirroring, so the cold fbm frame measured
    # 2 distinct hashes across 6 cooks. Mirroring is now moot — there is only one body
    # left for the trace to agree with.
    # (The older bug where the no-MSVC branch traced a DIFFERENT, table-based FBM — the
    # first rendered frame and every cached frame showing visibly different noise with
    # no input change — stays fixed: _can_inductor_compile() governs only whether the
    # trace is later upgraded via torch.compile, never which field is computed.)
    fast_fn = _make_fbm_fast_fn(octaves)

    def _compile_fbm():
        compiled = _compile_noise(_make_fbm_fast_fn(octaves))
        dummy = torch.rand(1, 64, 64, device=x.device)
        compiled(dummy, dummy)
        return compiled

    return _fbm_cache.call(
        key, (x, y), device=x.device,
        trace_fn=lambda: torch.jit.trace(fast_fn, (x, y)),
        compile_fn=_compile_fbm,
        eager_fn=fast_fn)


# ── Worley / Voronoi noise (arithmetic hash, table-free) ─────────────────────
#
# Evaluates distance to the nearest (F1) and 2nd-nearest (F2) feature points.
# Each grid cell gets a pseudo-random point via _lowbias32 hash.
# Checks the 3x3 cell neighborhood (9 cells) for closest points.
#
# Uses 3-tier compilation: eager → jit.trace → torch.compile (same as FBM).

def _worley2d_core(x: torch.Tensor, y: torch.Tensor,
                   dx_off: torch.Tensor, dy_off: torch.Tensor) -> torch.Tensor:
    """Core Worley distance computation returning all 9 squared distances."""
    x_floor = torch.floor(x)
    y_floor = torch.floor(y)
    xi = x_floor.to(torch.int32)
    yi = y_floor.to(torch.int32)

    cx = xi.unsqueeze(0) + dx_off
    cy = yi.unsqueeze(0) + dy_off

    base_hash = cx * 0x1B873593 ^ cy * 0x27D4EB2D
    px = cx.float() + (_lowbias32(base_hash) & 0x7FFFFF).float() / 8388607.0
    py = cy.float() + (_lowbias32(base_hash + 0x165667B1) & 0x7FFFFF).float() / 8388607.0

    x_exp = x.unsqueeze(0)
    y_exp = y.unsqueeze(0)
    return (x_exp - px).square() + (y_exp - py).square()


def _worley2d_f1(x: torch.Tensor, y: torch.Tensor,
                 dx_off: torch.Tensor, dy_off: torch.Tensor) -> torch.Tensor:
    """Worley F1 (nearest) — traceable function for jit.trace/torch.compile."""
    dist = _worley2d_core(x, y, dx_off, dy_off)
    return torch.sqrt(dist.min(dim=0).values)


def _worley2d_f2(x: torch.Tensor, y: torch.Tensor,
                 dx_off: torch.Tensor, dy_off: torch.Tensor) -> torch.Tensor:
    """Worley F2 (2nd nearest) — traceable function for jit.trace/torch.compile."""
    dist = _worley2d_core(x, y, dx_off, dy_off)
    sorted_dist, _ = torch.sort(dist, dim=0)
    return torch.sqrt(sorted_dist[1])


_worley_cache = _TieredCache("worley")


def _worley2d(x: torch.Tensor, y: torch.Tensor, return_f2: bool = False) -> torch.Tensor:
    """2D Worley noise with 3-tier compilation.

    Tiers: eager → jit.trace → torch.compile/Inductor.
    """
    # Rank/device come from the widest coord, not from x: a scalar x beside a grid
    # y would otherwise size the 9 neighbour offsets for rank 0 and mis-align them.
    ref = _widest((x, y))
    key = (return_f2, ref.device)
    dx_off, dy_off = _get_worley_offsets(ref.device, ref.dim())

    # Worley's own eager/traced pair happens to agree bitwise on this box — its min/sort
    # over squared distances gives the fuser far less to reassociate than simplex's
    # cancelling `x - X0`. Routing it through the same discipline anyway is the point:
    # it stops parity from depending on which ops the fuser declines to touch.
    fn = _worley2d_f2 if return_f2 else _worley2d_f1

    def _compile_worley():
        compiled = _compile_noise(fn)
        dummy = torch.rand(1, 64, 64, device=ref.device)
        warmup_dx, warmup_dy = _get_worley_offsets(dummy.device, dummy.dim())
        compiled(dummy, dummy, warmup_dx, warmup_dy)
        return compiled

    return _worley_cache.call(
        key, (x, y, dx_off, dy_off), device=ref.device,
        trace_fn=lambda: torch.jit.trace(fn, (x, y, dx_off, dy_off)),
        compile_fn=_compile_worley,
        eager_fn=fn)


# ── Curl noise ───────────────────────────────────────────────────────────────
#
# Curl of a 2D scalar potential field (Perlin noise):
#   curl_x =  dN/dy
#   curl_y = -dN/dx
# Computed via central finite differences. Result is a 2-component vector (vec2)
# representing a divergence-free flow field.

def _curl2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Curl of 2D Perlin noise. Returns [..., 2] tensor (vec2)."""
    eps = 0.001
    inv_2eps = 500.0  # 1.0 / (2.0 * 0.001)
    # Promote BEFORE the branch, not inside it. Curl is a central difference, so the
    # offset has to land in the same dtype on both devices or they answer differently:
    # a 0-dim fp16 constant keeps its dtype under `x + eps`, and at |x| ~ 8 the fp16 ulp
    # (~0.008) swallows an eps of 0.001 outright, while the promoted path resolves it.
    # Measured before this moved out of the branch: curl2 disagreed by 1.1e-01 and curl3
    # by 2.2e+00 across devices under precision="fp16".
    ref = _widest((x, y))
    x, y = _align_coords((x, y), ref)
    if ref.is_cuda:
        # GPU: one batched Perlin call instead of four (see _curl3d). Bit-exact.
        rank = ref.dim()
        xs = _stack_coord([x + eps, x - eps, x, x], x, rank)
        ys = _stack_coord([y, y, y + eps, y - eps], y, rank)
        n = _perlin2d_fast(xs, ys)
        curl_x = (n[2] - n[3]) * inv_2eps    #  dN/dy
        curl_y = -(n[0] - n[1]) * inv_2eps   # -dN/dx
        return torch.stack([curl_x, curl_y], dim=-1)

    n_px = _perlin2d_fast(x + eps, y)
    n_mx = _perlin2d_fast(x - eps, y)
    n_py = _perlin2d_fast(x, y + eps)
    n_my = _perlin2d_fast(x, y - eps)
    curl_x = (n_py - n_my) * inv_2eps   #  dN/dy
    curl_y = -(n_px - n_mx) * inv_2eps   # -dN/dx
    return torch.stack([curl_x, curl_y], dim=-1)


# ── Ridged noise ─────────────────────────────────────────────────────────────
#
# Like FBM but each octave is `1.0 - abs(noise)`, creating sharp ridges.
# The ridge signal is squared for sharper features, and weighted by the
# previous octave's value for self-similar detail concentration.

def _octave_perlins(noise_fn, coords: tuple, octaves: int) -> list:
    """Evaluate `octaves` noise octaves at frequencies 1, 2, 4, ... and return
    them as a list of `octaves` tensors.

    On GPU this batches ALL octaves into ONE noise call (stack the freq-scaled
    coords on a leading dim) — collapsing O(octaves) x ~50 kernel launches down to
    ~50, about 3x faster for fbm/ridged/billow/turbulence. On CPU there's no launch
    overhead to amortize, so it falls back to per-octave calls. Bit-exact either
    way: the i-th result equals noise_fn(coords * 2**i) in both paths (x * 1.0 == x).

    The stacking goes through `_stack_coord` so a constant coord — which reaches
    us 0-dim, e.g. `fbm(u * 8.0, v * 8.0, 0.5, 4)` — keeps its octave axis leading
    instead of right-aligning into a spatial one. Without that the GPU path raised
    where CPU returned a picture.
    """
    freqs = [2.0 ** i for i in range(octaves)]
    # Promote BEFORE the branch so both paths scale the coords in one dtype — the two
    # are only interchangeable if they compute the same expression (see _align_coords).
    ref = _widest(coords)
    coords = _align_coords(coords, ref)
    if octaves > 1 and ref.is_cuda:
        rank = ref.dim()
        stacked = tuple(_stack_coord([c * f for f in freqs], c, rank)
                        for c in coords)
        n = noise_fn(*stacked)
        return [n[i] for i in range(octaves)]
    return [noise_fn(*(c * f for c in coords)) for f in freqs]


def _ridged_nd(noise_fn, coords: tuple, octaves: int) -> torch.Tensor:
    """Ridged multi-fractal noise, parameterized by noise function and coordinates.

    Works for any dimensionality: coords is (x, y) for 2D or (x, y, z) for 3D.
    Weight feedback: each octave's un-scaled signal becomes the next octave's weight,
    concentrating detail in ridge regions. Signal = (1-|n|)^2 * prev_weight is
    already in [0,1] so no clamping is needed.
    """
    octaves = max(1, min(octaves, 10))
    oct_n = _octave_perlins(noise_fn, coords, octaves)
    amp = 1.0
    weight = 1.0
    max_amp = 0.0
    result = _zeros_broadcast(coords)

    for i in range(octaves):
        signal = 1.0 - torch.abs(oct_n[i])
        signal = signal * signal  # sharpen ridges
        signal = signal * weight
        result.add_(signal, alpha=amp)
        weight = signal
        max_amp += amp
        amp *= 0.5

    return result / max_amp


def _ridged2d(x, y, octaves):
    return _ridged_nd(_perlin2d_fast, (x, y), octaves)


# ── Billow / Turbulence shared core ──────────────────────────────────────────

def _abs_fbm_nd_raw(noise_fn, coords: tuple, octaves: int) -> tuple[torch.Tensor, float]:
    """Accumulated abs(noise) across octaves. Returns (result, max_amp).

    Shared core for billow (remaps to [-1,1]) and turbulence (normalized [0,1]).
    """
    octaves = max(1, min(octaves, 10))
    oct_n = _octave_perlins(noise_fn, coords, octaves)
    result = torch.abs(oct_n[0])
    max_amp = 1.0
    amp = 0.5

    for i in range(1, octaves):
        result.add_(torch.abs(oct_n[i]), alpha=amp)
        max_amp += amp
        amp *= 0.5

    return result, max_amp


def _billow2d(x, y, octaves):
    """Billow noise. Returns float in ~[-1, 1]."""
    result, max_amp = _abs_fbm_nd_raw(_perlin2d_fast, (x, y), octaves)
    return result / max_amp * 2.0 - 1.0


def _turbulence2d(x, y, octaves):
    """Turbulence noise. Returns float in ~[0, 1]."""
    result, max_amp = _abs_fbm_nd_raw(_perlin2d_fast, (x, y), octaves)
    return result / max_amp


# ── Flow noise ───────────────────────────────────────────────────────────────
#
# Time-varying Perlin noise with rotating domain offsets per octave.
# The rotation angle is derived from the time parameter, creating smooth
# temporal evolution without the popping artifacts of simple time offset.

def _flow2d(x: torch.Tensor, y: torch.Tensor, time: float) -> torch.Tensor:
    """Flow noise — time-varying domain-warped Perlin. Returns float in ~[-1, 1]."""
    # Rotate input domain based on time (different angle per octave)
    angle = time * 0.5
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rx = x * cos_a - y * sin_a
    ry = x * sin_a + y * cos_a

    # First octave with time-rotated coords
    result = _perlin2d_fast(rx, ry)

    # Additional octaves with increasing rotation
    freq = 2.0
    amp = 0.5
    max_amp = 1.0
    for i in range(1, 4):  # 4 octaves total
        angle_i = time * (0.5 + i * 0.37)
        cos_i = math.cos(angle_i)
        sin_i = math.sin(angle_i)
        xf = x * freq
        yf = y * freq
        rx_i = xf * cos_i - yf * sin_i
        ry_i = xf * sin_i + yf * cos_i
        result.add_(_perlin2d_fast(rx_i, ry_i), alpha=amp)
        max_amp += amp
        amp *= 0.5
        freq *= 2.0

    return result / max_amp


# ── Alligator noise ──────────────────────────────────────────────────────────
#
# Layered cell noise where each octave's Worley F1 distance is combined
# with a smooth-min operator, creating an organic skin-like pattern with
# connected ridges between cells.

def _alligator_nd(worley_fn, coords: tuple, octaves: int) -> torch.Tensor:
    """Alligator noise — layered cell noise with ridge accumulation.

    Each octave inverts and sharpens the Worley F1 distance to create ridges
    at cell boundaries. Works for 2D (worley_fn=_worley2d) or 3D (_worley3d).
    """
    octaves = max(1, min(octaves, 8))
    freq = 1.0
    amp = 1.0
    result = _zeros_broadcast(coords)
    max_amp = 0.0

    for _ in range(octaves):
        d = worley_fn(*(c * freq for c in coords), return_f2=False)
        ridge = 1.0 - torch.clamp(d * 2.0, 0.0, 1.0)
        result.add_(ridge, alpha=amp)
        max_amp += amp
        amp *= 0.5
        freq *= 2.0

    return result / max_amp


def _alligator2d(x, y, octaves):
    return _alligator_nd(_worley2d, (x, y), octaves)


# ── 3D Noise ─────────────────────────────────────────────────────────────────
#
# 3D variants of all noise functions. Use arithmetic hash (_lowbias32) for
# TorchInductor compatibility, same as the 2D implementations.
# 3D Perlin uses the classic 12-gradient set for good isotropy.

def _grad3d_dot(h: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    """3D gradient dot product using Perlin's original 12-gradient set.

    Gradients are the 12 edges of a cube: (±1,±1,0), (±1,0,±1), (0,±1,±1).
    Encoded via h & 15 with Ken Perlin's bit-manipulation trick.
    Uses torch.where for clarity — still fully fusible by TorchInductor.
    """
    h15 = h & 15
    # u = x when h < 8, else y
    u = torch.where(h15 < 8, dx, dy)
    # v = y when h < 4; x when h is 12 or 14; z otherwise
    v = torch.where(h15 < 4, dy, torch.where((h15 == 12) | (h15 == 14), dx, dz))
    # Apply sign bits
    return torch.where((h15 & 1) != 0, -u, u) + torch.where((h15 & 2) != 0, -v, v)


def _perlin3d_fast(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """3D Perlin noise using arithmetic hash (table-free).

    8 corners of a unit cube, 12-gradient set, quintic interpolation.
    Pure pointwise arithmetic — fully fusible by TorchInductor.
    """
    x_floor = torch.floor(x)
    y_floor = torch.floor(y)
    z_floor = torch.floor(z)
    xi = x_floor.to(torch.int32)
    yi = y_floor.to(torch.int32)
    zi = z_floor.to(torch.int32)

    xf = x - x_floor
    yf = y - y_floor
    zf = z - z_floor

    # Quintic interpolation curves
    u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
    v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)
    w = zf * zf * zf * (zf * (zf * 6.0 - 15.0) + 10.0)

    # Arithmetic hash for 8 corners — unique prime multipliers per axis
    xi1 = xi + 1
    yi_hash = yi * 0x1B873593
    yi1_hash = (yi + 1) * 0x1B873593
    zi_hash = zi * 0x27D4EB2D
    zi1_hash = (zi + 1) * 0x27D4EB2D

    h000 = _lowbias32(xi ^ yi_hash ^ zi_hash)
    h100 = _lowbias32(xi1 ^ yi_hash ^ zi_hash)
    h010 = _lowbias32(xi ^ yi1_hash ^ zi_hash)
    h110 = _lowbias32(xi1 ^ yi1_hash ^ zi_hash)
    h001 = _lowbias32(xi ^ yi_hash ^ zi1_hash)
    h101 = _lowbias32(xi1 ^ yi_hash ^ zi1_hash)
    h011 = _lowbias32(xi ^ yi1_hash ^ zi1_hash)
    h111 = _lowbias32(xi1 ^ yi1_hash ^ zi1_hash)

    xf1 = xf - 1.0
    yf1 = yf - 1.0
    zf1 = zf - 1.0

    # Gradient dot products for all 8 corners
    g000 = _grad3d_dot(h000, xf,  yf,  zf)
    g100 = _grad3d_dot(h100, xf1, yf,  zf)
    g010 = _grad3d_dot(h010, xf,  yf1, zf)
    g110 = _grad3d_dot(h110, xf1, yf1, zf)
    g001 = _grad3d_dot(h001, xf,  yf,  zf1)
    g101 = _grad3d_dot(h101, xf1, yf,  zf1)
    g011 = _grad3d_dot(h011, xf,  yf1, zf1)
    g111 = _grad3d_dot(h111, xf1, yf1, zf1)

    # Trilinear interpolation
    return torch.lerp(
        torch.lerp(torch.lerp(g000, g100, u), torch.lerp(g010, g110, u), v),
        torch.lerp(torch.lerp(g001, g101, u), torch.lerp(g011, g111, u), v),
        w
    )


def _worley3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
              return_f2: bool = False) -> torch.Tensor:
    """3D Worley noise. Checks 3x3x3 = 27 cell neighborhood.

    Vectorized: computes all 27 neighbor distances in a single batched pass.
    """
    x_floor = torch.floor(x)
    y_floor = torch.floor(y)
    z_floor = torch.floor(z)
    xi = x_floor.to(torch.int32)
    yi = y_floor.to(torch.int32)
    zi = z_floor.to(torch.int32)

    # 27 neighbor offsets [-1,0,1]^3, cached per (device, rank) — see the 2D path.
    # Rank comes from the widest coord, not from x: a scalar x with grid y/z would
    # otherwise size the offsets for rank 0 and mis-align every neighbour.
    ref = _widest((x, y, z))
    dx_off, dy_off, dz_off = _get_worley3d_offsets(ref.device, ref.dim())

    # Cell coords for all 27 neighbors: [27, *spatial]
    cx = xi.unsqueeze(0) + dx_off
    cy = yi.unsqueeze(0) + dy_off
    cz = zi.unsqueeze(0) + dz_off

    # Hash and random point positions
    base_hash = cx * 0x1B873593 ^ cy * 0x27D4EB2D ^ cz * 0x165667B1
    px = cx.float() + (_lowbias32(base_hash) & 0x7FFFFF).float() / 8388607.0
    py = cy.float() + (_lowbias32(base_hash + 0x165667B1) & 0x7FFFFF).float() / 8388607.0
    pz = cz.float() + (_lowbias32(base_hash + 0x2B873593) & 0x7FFFFF).float() / 8388607.0

    # Squared distances: [27, *spatial]
    x_exp = x.unsqueeze(0)
    y_exp = y.unsqueeze(0)
    z_exp = z.unsqueeze(0)
    dist = (x_exp - px).square() + (y_exp - py).square() + (z_exp - pz).square()

    if return_f2:
        sorted_dist, _ = torch.sort(dist, dim=0)
        return torch.sqrt(sorted_dist[1])
    else:
        return torch.sqrt(dist.min(dim=0).values)


def _curl3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Curl of 3D Perlin noise vector field. Returns [..., 3] tensor (vec3).

    Uses 3 offset copies of Perlin noise as the vector field components:
      F1(p) = perlin(p),  F2(p) = perlin(p + off1),  F3(p) = perlin(p + off2)
    Then curl = (dF3/dy - dF2/dz, dF1/dz - dF3/dx, dF2/dx - dF1/dy).
    """
    eps = 0.001
    inv_2eps = 500.0  # 1.0 / (2.0 * 0.001)

    # Irrational offsets to decorrelate the 3 noise channels
    off1x, off1y, off1z = 31.416, 47.853, 12.679
    off2x, off2y, off2z = 73.156, 19.827, 63.941

    # Promote BEFORE the branch — see _curl2d: an fp16 constant coord makes the central
    # difference resolve differently per device unless both compute the offset in the
    # same dtype.
    ref = _widest((x, y, z))
    x, y, z = _align_coords((x, y, z), ref)
    if ref.is_cuda:
        # GPU: batch all 12 Perlin evaluations into ONE call (stack the coord
        # triples on a leading dim). Collapses ~600 kernel launches to ~50 — about
        # 3x faster at 512^2 — and is bit-exact with the per-call form. On CPU the
        # stacking overhead outweighs the (absent) launch saving, so fall through.
        rank = ref.dim()
        xs = _stack_coord([x + off2x, x + off2x, x + off1x, x + off1x, x, x,
                           x + off2x + eps, x + off2x - eps,
                           x + off1x + eps, x + off1x - eps, x, x], x, rank)
        ys = _stack_coord([y + off2y + eps, y + off2y - eps, y + off1y, y + off1y, y, y,
                           y + off2y, y + off2y, y + off1y, y + off1y,
                           y + eps, y - eps], y, rank)
        zs = _stack_coord([z + off2z, z + off2z, z + off1z + eps, z + off1z - eps,
                           z + eps, z - eps, z + off2z, z + off2z,
                           z + off1z, z + off1z, z, z], z, rank)
        n = _perlin3d_fast(xs, ys, zs)
        curl_x = (n[0] - n[1] - n[2] + n[3]) * inv_2eps   # dF3/dy - dF2/dz
        curl_y = (n[4] - n[5] - n[6] + n[7]) * inv_2eps   # dF1/dz - dF3/dx
        curl_z = (n[8] - n[9] - n[10] + n[11]) * inv_2eps  # dF2/dx - dF1/dy
        return torch.stack([curl_x, curl_y, curl_z], dim=-1)

    # dF3/dy - dF2/dz
    curl_x = (_perlin3d_fast(x + off2x, y + off2y + eps, z + off2z) -
              _perlin3d_fast(x + off2x, y + off2y - eps, z + off2z) -
              _perlin3d_fast(x + off1x, y + off1y, z + off1z + eps) +
              _perlin3d_fast(x + off1x, y + off1y, z + off1z - eps)) * inv_2eps

    # dF1/dz - dF3/dx
    curl_y = (_perlin3d_fast(x, y, z + eps) -
              _perlin3d_fast(x, y, z - eps) -
              _perlin3d_fast(x + off2x + eps, y + off2y, z + off2z) +
              _perlin3d_fast(x + off2x - eps, y + off2y, z + off2z)) * inv_2eps

    # dF2/dx - dF1/dy
    curl_z = (_perlin3d_fast(x + off1x + eps, y + off1y, z + off1z) -
              _perlin3d_fast(x + off1x - eps, y + off1y, z + off1z) -
              _perlin3d_fast(x, y + eps, z) +
              _perlin3d_fast(x, y - eps, z)) * inv_2eps

    return torch.stack([curl_x, curl_y, curl_z], dim=-1)


def _fbm3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, octaves: int) -> torch.Tensor:
    """3D FBM using Perlin noise. Persistence=0.5, lacunarity=2.0."""
    octaves = max(1, min(octaves, 10))
    oct_n = _octave_perlins(_perlin3d_fast, (x, y, z), octaves)
    result = oct_n[0]
    max_amp = 1.0
    amp = 0.5

    for i in range(1, octaves):
        result = result + amp * oct_n[i]
        max_amp += amp
        amp *= 0.5

    return result / max_amp


def _ridged3d(x, y, z, octaves):
    return _ridged_nd(_perlin3d_fast, (x, y, z), octaves)


def _billow3d(x, y, z, octaves):
    """3D billow noise. Returns float in ~[-1, 1]."""
    result, max_amp = _abs_fbm_nd_raw(_perlin3d_fast, (x, y, z), octaves)
    return result / max_amp * 2.0 - 1.0


def _turbulence3d(x, y, z, octaves):
    """3D turbulence noise. Returns float in ~[0, 1]."""
    result, max_amp = _abs_fbm_nd_raw(_perlin3d_fast, (x, y, z), octaves)
    return result / max_amp


def _flow3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, time: float) -> torch.Tensor:
    """3D flow noise — time-varying domain-rotated Perlin."""
    # Rotate around z-axis based on time (different angle per octave)
    angle = time * 0.5
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rx = x * cos_a - y * sin_a
    ry = x * sin_a + y * cos_a

    result = _perlin3d_fast(rx, ry, z)

    freq = 2.0
    amp = 0.5
    max_amp = 1.0
    for i in range(1, 4):
        angle_i = time * (0.5 + i * 0.37)
        cos_i = math.cos(angle_i)
        sin_i = math.sin(angle_i)
        xf = x * freq
        yf = y * freq
        rx_i = xf * cos_i - yf * sin_i
        ry_i = xf * sin_i + yf * cos_i
        result.add_(_perlin3d_fast(rx_i, ry_i, z * freq), alpha=amp)
        max_amp += amp
        amp *= 0.5
        freq *= 2.0

    return result / max_amp


def _alligator3d(x, y, z, octaves):
    return _alligator_nd(_worley3d, (x, y, z), octaves)
