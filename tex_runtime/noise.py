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


# ── Lookup tables ─────────────────────────────────────────────────────────────

# Classic 256-entry Perlin permutation table
_PERM = torch.tensor([
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
    140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
    247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
    57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
    74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
    60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
    65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,
    200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,
    52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
    207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,
    119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
    129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
    218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,
    81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,
    184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,
    222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
], dtype=torch.int64)

# Doubled for overflow-free indexing
_PERM2 = torch.cat([_PERM, _PERM])

# 8 gradient directions for 2D Perlin noise (unit circle at 45-degree intervals).
# Pre-split into separate X/Y component tensors — avoids 3D view creation during
# gather+slice.  grad[h][..., 0] would create a view into a 3D result; flat 1D
# tensors return flat 2D directly.
_GRAD2_X = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.7071, -0.7071, 0.7071, -0.7071], dtype=torch.float32)
_GRAD2_Y = torch.tensor([0.0, 0.0, 1.0, -1.0, 0.7071, 0.7071, -0.7071, -0.7071], dtype=torch.float32)

# Simplex skew/unskew constants
_SKEW_2D = 0.5 * (math.sqrt(3.0) - 1.0)      # ~0.3660254
_UNSKEW_2D = (3.0 - math.sqrt(3.0)) / 6.0     # ~0.2113249


_noise_tables_cache: dict[str, tuple[torch.Tensor, ...]] = {}

# Worley/Voronoi 9-neighbor offsets, cached per device to avoid per-call allocation
_worley_offsets_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

def _get_worley_offsets(device: torch.device, ndim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached (dx_off, dy_off) tensors for Worley 9-neighbor lookup."""
    key = (str(device), ndim)
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


def _get_noise_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get device-local noise lookup tables (cached).

    Returns (perm, grad_x, grad_y).
    Used only by the table-based FBM fallback when MSVC/Inductor is unavailable.
    """
    key = str(device)
    cached = _noise_tables_cache.get(key)
    if cached is not None:
        return cached
    tables = (
        _PERM2.to(device), _GRAD2_X.to(device), _GRAD2_Y.to(device),
    )
    _noise_tables_cache[key] = tables
    return tables


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
    gy_sign_bit = ((gi >> 1) & 1).float()
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

    def __init__(self):
        self.cache: dict = {}
        self._compile_attempted: set = set()
        self._call_count: dict = {}
        self._lock = threading.Lock()

    def get(self, key):
        """Return cached callable, or None. False (trace failed) returns None."""
        val = self.cache.get(key)
        return val if val is not None and val is not False else None

    def try_upgrade(self, key, compile_fn):
        """Attempt torch.compile upgrade after _COMPILE_AFTER_CALLS calls.

        compile_fn() should return the compiled callable, or raise on failure.
        """
        if key in self._compile_attempted or not _can_inductor_compile():
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
            self.cache[key] = compile_fn()
        except Exception:
            pass
        self._call_count.pop(key, None)

    def store(self, key, trace_fn):
        """Store a traced version on first call. trace_fn() → traced callable or raise."""
        if key in self.cache:
            return
        try:
            self.cache[key] = trace_fn()
        except Exception:
            self.cache[key] = False


_simplex_cache = _TieredCache()


def _compile_simplex():
    """Compile simplex with Inductor and warmup."""
    compiled = torch.compile(_simplex2d_fast, backend='inductor', fullgraph=True)
    dummy = torch.rand(1, 64, 64)
    compiled(dummy, dummy)
    return compiled


def _simplex2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """2D Simplex noise with 3-tier compilation (eager → jit.trace → torch.compile)."""
    key = x.device
    cached = _simplex_cache.get(key)
    if cached is not None:
        _simplex_cache.try_upgrade(key, _compile_simplex)
        return _simplex_cache.get(key)(x, y)

    result = _simplex2d_fast(x, y)
    _simplex_cache.store(key, lambda: torch.jit.trace(_simplex2d_fast, (x, y)))
    return result


def _perlin2d_core(x: torch.Tensor, y: torch.Tensor,
                    perm: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """Perlin noise core for use in traced FBM. All tables passed as args (no graph breaks)."""
    x_floor = torch.floor(x)
    y_floor = torch.floor(y)
    xi = x_floor.long()
    yi = y_floor.long()
    xf = x - x_floor
    yf = y - y_floor
    u = xf * 6.0
    u.sub_(15.0).mul_(xf).add_(10.0).mul_(xf).mul_(xf).mul_(xf)
    v = yf * 6.0
    v.sub_(15.0).mul_(yf).add_(10.0).mul_(yf).mul_(yf).mul_(yf)
    xi_mask = xi & 255
    yi_mask = yi & 255
    xi1_mask = (xi_mask + 1) & 255
    yi1_mask = (yi + 1) & 255
    perm_xi = perm[xi_mask]
    perm_xi1 = perm[xi1_mask]
    h00 = perm[(perm_xi + yi_mask) & 511] & 7
    h10 = perm[(perm_xi1 + yi_mask) & 511] & 7
    h01 = perm[(perm_xi + yi1_mask) & 511] & 7
    h11 = perm[(perm_xi1 + yi1_mask) & 511] & 7
    xf1 = xf - 1.0
    yf1 = yf - 1.0
    g00 = gx[h00] * xf
    g00.add_(gy[h00] * yf)
    g10 = gx[h10] * xf1
    g10.add_(gy[h10] * yf)
    g01 = gx[h01] * xf
    g01.add_(gy[h01] * yf1)
    g11 = gx[h11] * xf1
    g11.add_(gy[h11] * yf1)
    return torch.lerp(torch.lerp(g00, g10, u), torch.lerp(g01, g11, u), v)


def _make_fbm_fn(octaves: int):
    """Build a traceable FBM function for a specific octave count.
    The loop is unrolled by torch.jit.trace into a single straight-line graph.
    """
    # Pre-compute amplitude normalization
    max_amp = sum(0.5 ** i for i in range(octaves))
    inv_max = 1.0 / max_amp

    def fbm_fn(x, y, perm, gx, gy):
        result = _perlin2d_core(x, y, perm, gx, gy)
        freq = 2.0
        amp = 0.5
        for _ in range(octaves - 1):
            result = result + _perlin2d_core(x * freq, y * freq, perm, gx, gy) * amp
            amp = amp * 0.5
            freq = freq * 2.0
        return result * inv_max
    return fbm_fn


def _make_fbm_fast_fn(octaves: int):
    """Build a traceable FBM function using arithmetic hash noise.
    No table arguments needed — all hashing is pure arithmetic.

    NOTE: Intentionally duplicates _make_fbm_fn's loop structure.
    Cannot unify because torch.jit.trace requires fixed argument counts —
    the table-based version takes (x, y, perm, gx, gy) while this takes (x, y).
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


_fbm_cache = _TieredCache()


_inductor_available: bool | None = None

def _can_inductor_compile() -> bool:
    """Check if TorchInductor is available, enabling FxGraph cache on first call."""
    global _inductor_available
    if _inductor_available is None:
        import shutil
        import sys
        if sys.platform != 'win32':
            _inductor_available = True  # Linux/macOS have gcc/clang by default
        else:
            # Try the robust MSVC setup from compiled.py
            try:
                from .compiled import _setup_msvc_env as _setup_compiled_msvc
                _setup_compiled_msvc()
            except Exception:
                pass
            _inductor_available = shutil.which('cl') is not None
        if _inductor_available:
            # Enable persistent FxGraph cache so compiled C++ kernels survive
            # process restarts — eliminates the ~28s MSVC compile on cold start.
            try:
                import torch._inductor.config as _ind_cfg
                _ind_cfg.fx_graph_cache = True
            except Exception:
                pass
    return _inductor_available


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

    # Fast path: compiled or traced arithmetic hash FBM
    cached = _fbm_cache.get(key)
    if cached is not None:
        def _compile_fbm():
            fn = _make_fbm_fast_fn(octaves)
            compiled = torch.compile(fn, backend='inductor', fullgraph=True)
            compiled(torch.rand(1, 64, 64), torch.rand(1, 64, 64))
            return compiled
        _fbm_cache.try_upgrade(key, _compile_fbm)
        return _fbm_cache.get(key)(x, y)

    # Eager execution using arithmetic hash (first call)
    result = _perlin2d_fast(x, y)
    max_amp = 1.0
    amplitude = 0.5

    if octaves > 1:
        x_sc = x * 2.0
        y_sc = y * 2.0
        for _ in range(octaves - 1):
            result.add_(_perlin2d_fast(x_sc, y_sc), alpha=amplitude)
            max_amp += amplitude
            amplitude *= 0.5
            x_sc.mul_(2.0)
            y_sc.mul_(2.0)
        result.div_(max_amp)

    # Cache a traced/compiled version for future calls.
    # With MSVC: trace arithmetic hash now, torch.compile later (6x faster).
    # Without MSVC: trace table-based version (slightly faster under jit.trace).
    if _can_inductor_compile():
        _fbm_cache.store(key, lambda: torch.jit.trace(_make_fbm_fast_fn(octaves), (x, y)))
    else:
        def _trace_table_fbm():
            perm, gx, gy = _get_noise_tables(x.device)
            fn = _make_fbm_fn(octaves)
            traced = torch.jit.trace(fn, (x, y, perm, gx, gy))
            def _table_wrapper(x, y, _t=traced, _p=perm, _gx=gx, _gy=gy):
                return _t(x, y, _p, _gx, _gy)
            return _table_wrapper
        _fbm_cache.store(key, _trace_table_fbm)

    return result


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


_worley_cache = _TieredCache()


def _worley2d(x: torch.Tensor, y: torch.Tensor, return_f2: bool = False) -> torch.Tensor:
    """2D Worley noise with 3-tier compilation.

    Tiers: eager → jit.trace → torch.compile/Inductor.
    """
    key = (return_f2, x.device)
    dx_off, dy_off = _get_worley_offsets(x.device, x.dim())

    # Fast path: cached compiled or traced version
    cached = _worley_cache.get(key)
    if cached is not None:
        def _compile_worley():
            fn = _worley2d_f2 if return_f2 else _worley2d_f1
            compiled = torch.compile(fn, backend='inductor', fullgraph=True)
            dummy = torch.rand(1, 64, 64)
            warmup_dx, warmup_dy = _get_worley_offsets(dummy.device, dummy.dim())
            compiled(dummy, dummy, warmup_dx, warmup_dy)
            return compiled
        _worley_cache.try_upgrade(key, _compile_worley)
        return _worley_cache.get(key)(x, y, dx_off, dy_off)

    # Eager execution (first call)
    dist = _worley2d_core(x, y, dx_off, dy_off)
    if return_f2:
        sorted_dist, _ = torch.sort(dist, dim=0)
        result = torch.sqrt(sorted_dist[1])
    else:
        result = torch.sqrt(dist.min(dim=0).values)

    fn = _worley2d_f2 if return_f2 else _worley2d_f1
    _worley_cache.store(key, lambda: torch.jit.trace(fn, (x, y, dx_off, dy_off)))
    return result


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
    n_px = _perlin2d_fast(x + eps, y)
    n_mx = _perlin2d_fast(x - eps, y)
    n_py = _perlin2d_fast(x, y + eps)
    n_my = _perlin2d_fast(x, y - eps)

    inv_2eps = 500.0  # 1.0 / (2.0 * 0.001)
    curl_x = (n_py - n_my) * inv_2eps   #  dN/dy
    curl_y = -(n_px - n_mx) * inv_2eps   # -dN/dx

    return torch.stack([curl_x, curl_y], dim=-1)


# ── Ridged noise ─────────────────────────────────────────────────────────────
#
# Like FBM but each octave is `1.0 - abs(noise)`, creating sharp ridges.
# The ridge signal is squared for sharper features, and weighted by the
# previous octave's value for self-similar detail concentration.

def _ridged_nd(noise_fn, coords: tuple, octaves: int) -> torch.Tensor:
    """Ridged multi-fractal noise, parameterized by noise function and coordinates.

    Works for any dimensionality: coords is (x, y) for 2D or (x, y, z) for 3D.
    Weight feedback: each octave's un-scaled signal becomes the next octave's weight,
    concentrating detail in ridge regions. Signal = (1-|n|)^2 * prev_weight is
    already in [0,1] so no clamping is needed.
    """
    octaves = max(1, min(octaves, 10))
    freq = 1.0
    amp = 1.0
    weight = 1.0
    max_amp = 0.0
    result = torch.zeros_like(coords[0])

    for _ in range(octaves):
        n = noise_fn(*(c * freq for c in coords))
        signal = 1.0 - torch.abs(n)
        signal = signal * signal  # sharpen ridges
        signal = signal * weight
        result.add_(signal, alpha=amp)
        weight = signal
        max_amp += amp
        freq *= 2.0
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
    result = torch.abs(noise_fn(*coords))
    max_amp = 1.0
    freq = 2.0
    amp = 0.5

    for _ in range(octaves - 1):
        result.add_(torch.abs(noise_fn(*(c * freq for c in coords))), alpha=amp)
        max_amp += amp
        amp *= 0.5
        freq *= 2.0

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
    result = torch.zeros_like(coords[0])
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

    # Build 27 neighbor offsets: [-1,0,1]^3
    r = torch.tensor([-1, 0, 1], dtype=torch.int32, device=x.device)
    gz, gy, gx = torch.meshgrid(r, r, r, indexing="ij")
    dx_off = gx.reshape(27, *((1,) * xi.dim()))
    dy_off = gy.reshape(27, *((1,) * xi.dim()))
    dz_off = gz.reshape(27, *((1,) * xi.dim()))

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
    result = _perlin3d_fast(x, y, z)
    max_amp = 1.0
    amp = 0.5
    freq = 2.0

    for _ in range(octaves - 1):
        result.add_(_perlin3d_fast(x * freq, y * freq, z * freq), alpha=amp)
        max_amp += amp
        amp *= 0.5
        freq *= 2.0

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
