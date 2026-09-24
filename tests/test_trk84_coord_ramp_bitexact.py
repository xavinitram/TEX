"""TRK-84 — a moving window's coordinate builtins are now a slice of a cached ramp, and the
slice is bit-exact against the direct formula for every origin, including a tiled cook's.

THE ROW. Every LAT-4 coordinate-builtin LRU miss (a pan, an ROI window, a strip) rebuilt
`ix`/`u`/`iy`/`v` from scratch: `torch.arange(x0, x0+W)` then a divide, per axis — 4 kernels
and 4 allocations that a window move pays on EVERY tick, because the origin never repeats.
The author's ruling: ship a bit-exact fix, or decline.

THE FIX (`Interpreter._coord_ramps`, `tex_runtime/interpreter.py`) caches the FULL-EXTENT
`[0, size)` ramp and its `/max(size-1,1)` normalization, keyed on size alone (never on the
origin), and a window now SLICES it — a view, no kernel, no allocation. Bit-exactness rests on
one fact: `torch.arange(0, size, dtype=fp32)[i] == float32(i)` exactly, for every size and
origin TEX actually cooks (nowhere near fp32's 2**24 integer ceiling), so slicing at
`[x0:x0+w]` reproduces `torch.arange(x0, x0+w, dtype=fp32)` bit-for-bit — there is no
"different rounding order" for IEEE-754 exact arithmetic to introduce. `u`/`v` are the
IDENTICAL division `ramp / max(size-1,1)`, computed once instead of once per window.

THIS FILE proves it two ways: (1) a differential oracle — the OLD formula, kept verbatim below
as the ground truth — against the new interpreter's `u`/`v`/`ix`/`iy`, across many origins,
sizes (including size=1 and a size near a realistic 8K), and BOTH axes independently, all
`torch.equal` (not `allclose`) so the check is bit-exact and not merely close; (2) the tiled
form (`tile=(y0, H_total)`), which the interpreter normalizes into the same `roi` path, gets
the same proof at a non-zero `y0`.

PORTABILITY: CPU, no ComfyUI, no CUDA, no compiler, no numpy, no timing assertion.
"""
import torch

from helpers import *

from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_runtime.interpreter import Interpreter

_CODE = "@OUT = vec4(u, v, ix, iy);"


def _prepare():
    toks = Lexer(_CODE).tokenize()
    prog = Parser(toks, _CODE).parse()
    type_map = TypeChecker(binding_types={"IN": TEXType.VEC4}, source=_CODE).check(prog)
    return prog, type_map


def _new(prog, type_map, W_full, H_full, x0, y0, W, H):
    img = torch.zeros(1, H_full, W_full, 4)
    interp = Interpreter()
    roi = (x0, y0, W, H, W_full, H_full)
    return interp.execute(prog, {"IN": img}, type_map, roi=roi)


def _new_tile(prog, type_map, W_full, H_total, y0, H):
    """The M-4 strip form: `tile=(y0, H_total)` — the 1-D special case the interpreter
    normalizes to `roi=(0, y0, W_full, H, W_full, H_total)` (`interpreter.py::_create_builtins`
    docstring). x0 is always 0 here by construction. The STRIP's extent (H rows, not
    H_total) comes from the bound `IN` tensor's own shape — `tile`/`roi` say where the
    strip sits in the full image, never how tall the strip itself is."""
    img = torch.zeros(1, H, W_full, 4)
    interp = Interpreter()
    return interp.execute(prog, {"IN": img}, type_map, tile=(y0, H_total))


def _old_formula(W_full, H_full, x0, y0, W, H):
    """The PRE-TRK-84 ground truth, kept verbatim: a fresh `torch.arange` + divide per axis,
    exactly what `_create_builtins` did before this fix. This is the oracle — if the new
    sliced-ramp path ever disagrees with this by even one bit, the fix is not bit-exact and
    must be declined per the author's ruling."""
    cdt = torch.float32
    ix = torch.arange(x0, x0 + W, dtype=cdt).view(1, 1, W)
    u = (ix / max(W_full - 1, 1))
    iy = torch.arange(y0, y0 + H, dtype=cdt).view(1, H, 1)
    v = (iy / max(H_full - 1, 1))
    u_e, v_e = u.expand(1, H, W), v.expand(1, H, W)
    ix_e, iy_e = ix.expand(1, H, W), iy.expand(1, H, W)
    return torch.stack([u_e, v_e, ix_e, iy_e], dim=-1)


# (W_full, H_full, x0, y0, W, H)
_ROI_CASES = [
    (64, 64, 0, 0, 32, 32),          # zero origin -- the default whole-frame path
    (64, 64, 5, 0, 32, 32),          # x-only offset
    (64, 64, 0, 7, 32, 32),          # y-only offset
    (64, 64, 13, 19, 32, 17),        # both axes offset, non-square window
    (1, 1, 0, 0, 1, 1),              # degenerate 1x1
    (2049, 2049, 1000, 1500, 512, 300),   # odd full extent, large offset
    (8192, 4096, 7000, 3000, 1000, 900),  # near-8K, non-square full extent
    (96, 96, 24, 24, 48, 48),        # the BENCH-2 gate shape
    (1024, 1024, 256, 256, 512, 512),     # a BENCH-2 CUDA-pin shape (CPU here: bits, not device)
]


def test_trk84_bitexact_against_old_formula_across_origins(r: SubTestResult):
    print("\n--- TRK-84: sliced-ramp u/v/ix/iy are bit-exact against the old formula ---")
    prog, type_map = _prepare()
    bad = []
    for case in _ROI_CASES:
        got = _new(prog, type_map, *case)
        want = _old_formula(*case)
        if not torch.equal(got, want):
            diff = (got - want).abs().max().item()
            bad.append(f"{case}: maxdiff={diff}")
    if bad:
        r.fail("TRK-84 bit-exactness", "; ".join(bad))
    else:
        r.ok(f"{len(_ROI_CASES)} (W_full,H_full,x0,y0,W,H) case(s), all torch.equal "
             f"(bit-exact, not merely close)")


def test_trk84_bitexact_tiled_strip(r: SubTestResult):
    print("\n--- TRK-84: the M-4 tiled strip form is also bit-exact ---")
    prog, type_map = _prepare()
    W_full, H_total = 128, 96
    bad = []
    for y0, H in [(0, 32), (32, 32), (64, 32), (17, 11)]:
        got = _new_tile(prog, type_map, W_full, H_total, y0, H)
        want = _old_formula(W_full, H_total, 0, y0, W_full, H)
        if not torch.equal(got, want):
            diff = (got - want).abs().max().item()
            bad.append(f"tile=({y0},{H_total}) H={H}: maxdiff={diff}")
    if bad:
        r.fail("TRK-84 tile bit-exactness", "; ".join(bad))
    else:
        r.ok("4 strip position(s) (including a non-zero y0), all bit-exact")


def test_trk84_ramp_cache_hits_across_a_pan(r: SubTestResult):
    """The cache actually SHARES the base ramp across a moving window — not just that the
    numbers agree, but that a second, different-origin call reuses the same underlying
    tensor storage rather than allocating a fresh one, which is the whole point of the fix."""
    print("\n--- TRK-84: a pan reuses the SAME cached base ramp across origins ---")
    prog, type_map = _prepare()
    interp = Interpreter()
    # `_device_str` is set by `execute()`'s per-call setup, not `__init__` — a fresh
    # instance has no device identity until its first cook, exactly like the pooled
    # instances `ThreadLocalInterpreterPool` hands out.
    interp.execute(prog, {"IN": torch.zeros(1, 8, 8, 4)}, type_map,
                   roi=(0, 0, 8, 8, 8, 8))
    r1, n1 = interp._coord_ramps(64)
    r2, n2 = interp._coord_ramps(64)
    if r1.data_ptr() != r2.data_ptr() or n1.data_ptr() != n2.data_ptr():
        r.fail("TRK-84 ramp cache", "a second call for the SAME size allocated a fresh ramp "
               "instead of reusing the cached one")
        return
    r3, _ = interp._coord_ramps(48)
    if r3.data_ptr() == r1.data_ptr():
        r.fail("TRK-84 ramp cache", "two DIFFERENT sizes shared one storage — a real bug, "
               "not a cache hit")
        return
    r.ok("same size -> same cached storage; different size -> a distinct entry")


def test_trk84_pooled_instance_across_devices_never_mixes_ramps(r: SubTestResult):
    """The bug this test file's key design caught during development: `_coord_ramps` must
    key on DEVICE too, because `ThreadLocalInterpreterPool` (`tex_runtime/interp_pool.py`)
    reuses one Interpreter across every cook on a thread and `execute()` reassigns
    `self.device` per call. A cache keyed on size alone would serve a CPU-built ramp to a
    later CUDA cook on the SAME pooled instance — caught live by
    `test_bench2_free_memory_queries_per_tick`'s CUDA leg (`distance()` raised "Expected all
    tensors to be on the same device"). Only DEVICE PLACEMENT and the CPU-side bit-exactness
    (proven by the other tests in this file) are checked here — a CPU vs. CUDA VALUE
    comparison is deliberately not, because cross-device agreement is a characterization
    envelope (invariant #9), never a bit-parity contract; that would be a different, and
    weaker, claim than this fix's."""
    print("\n--- TRK-84: one pooled Interpreter, CPU then CUDA, never mixes device ramps ---")
    if not torch.cuda.is_available():
        r.skip("TRK-84 pooled cross-device", "no CUDA device on this box")
        return
    prog, type_map = _prepare()
    interp = Interpreter()      # ONE instance, exactly as the thread-local pool hands out
    cpu_out = interp.execute(prog, {"IN": torch.zeros(1, 32, 32, 4)}, type_map,
                             roi=(4, 4, 16, 16, 32, 32))
    cuda_out = interp.execute(prog, {"IN": torch.zeros(1, 32, 32, 4, device="cuda")}, type_map,
                              roi=(4, 4, 16, 16, 32, 32), device="cuda")
    if cpu_out.device.type != "cpu":
        r.fail("TRK-84 pooled cross-device", f"the CPU cook's own output is not on cpu "
               f"({cpu_out.device})")
    elif cuda_out.device.type != "cuda":
        r.fail("TRK-84 pooled cross-device", f"the CUDA cook's output is not on cuda "
               f"({cuda_out.device}) — a stale CPU ramp leaked into a CUDA cook")
    else:
        r.ok("CPU cook, then a CUDA cook on the SAME pooled instance: no device-mismatch "
             "crash, and correct device placement on both")
