"""PERF-3 — a blur answers the same whatever the process blurred before it.

WHAT WAS WRONG. `_get_gauss_kernels` cached its two 1-D kernels under
`key = (round(sigma, 3), device)` and then built them from the **full-precision** `sigma`.
Key and value therefore described different numbers: two sigmas agreeing to three decimals
shared whichever kernel arrived first, so the SAME program with the SAME bindings on the
SAME device returned different pixels depending on what that process had blurred earlier.
It is not only a last-ulp story — the radius is `ceil(3*sigma)` computed in PYTHON, so
1.9996 and 2.0001 land on a 13-tap and a 15-tap kernel while sharing the key `2.0`.

THE FIX, AND WHY THIS HALF. A cache must key on exactly what its value is built from, and
there were two ways to get there. *Quantise* — round the sigma before the build, so key and
kernel agree — is self-consistent but moves the output of every sigma that is not already a
multiple of 0.001, which is a value change to a shipped builtin. *Key exactly* changes no
output at all: a cold cache always built from the full sigma, so keying on it reproduces the
cold answer every time, which is what a correct cache is for. Exact keying is what shipped.

WHY THIS FILE IS SHAPED THE WAY IT IS. The defect is invisible to any single cook — one
blur in a fresh process is right on both sides. What sees it is ORDER, so every row runs the
same two sigmas in both orders from a cleared cache and requires each sigma's own output to
be the same in both. A row also requires the two sigmas to DISAGREE with each other, so it
cannot be passed by a blur that has stopped depending on sigma, or by a pair chosen so close
that fp32 hides the difference.

CLEARING IS PART OF THE TEST, NOT HOUSEKEEPING. `_gauss_kernel_cache` is a module-level memo
that outlives a cook, so any test that varies sigma by a small amount and compares outputs is
lying unless it clears it first (the PERF-2 findings record this: that lane's first mutation
oracle passed vacuously because the mutated run was served the un-mutated run's kernel).

BOTH DIRECTIONS. `test_perf3_the_rounded_key_is_detected` re-introduces the exact defect —
look the pair up under `round(sigma, 3)`, build it from the full sigma — and requires the
order-independence check to FAIL against it. Without that row, a check that had stopped
measuring anything would look identical to a fixed cache.

PORTABILITY: pure torch on CPU (CUDA rows added when a device exists). No ComfyUI, no
compiler, no numpy, no timing assertion.
"""
import math

from helpers import *

from failure_harness import run_tier
from TEX_Wrangle.tex_runtime import stdlib as _stdlib
# LIB-1: `_get_gauss_kernels` and `_gauss_blur_bchw` both moved onto `stdlib_core.py`, and
# `_gauss_blur_bchw` (the real blur path's caller) reads `_get_gauss_kernels` as ITS OWN
# global — a name bound in `stdlib_core`'s namespace, not a live proxy through the facade's
# re-export. Patching `_stdlib._get_gauss_kernels` (below, pre-LIB-1) only rebinds the
# facade's copy of the name and never reaches that call site, so the mutation guard has to
# patch the module the function is actually defined in.
from TEX_Wrangle.tex_runtime import stdlib_core as _stdlib_core


#: (label, sigma). Each row is blurred at `s` and at `s + 5e-4` — a pair that the old
#: 3-decimal key collapsed onto ONE entry. The two rows are the two ways the collapse
#: shows: the same number of taps with different weights, and a different number of taps.
_SIGMA_ROWS = (
    ("same radius, different weights", 1.5),     # 11 taps either way; weights differ ~9e-5
    ("different radius",               1.9996),  # ceil(3*sigma) is 6 then 7 -> 13 vs 15 taps
)

_DELTA = 5e-4


def _img(device, size=24):
    torch.manual_seed(20260920)
    return torch.rand(1, size, size, 4, dtype=torch.float32).to(device)


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _fresh():
    """Fresh kernel-cache state. Every sigma-varying comparison in this file starts here."""
    _stdlib._gauss_kernel_cache.clear()


def _blur(sigma, device, tier):
    """One cook of `gauss_blur(@A, <sigma>)`, the sigma spelled as a source literal."""
    code = f"@OUT = gauss_blur(@A, {sigma!r});"
    return run_tier(code, {"A": _img(device)}, tier, device=device)["OUT"]


def _differs(a, b):
    """None when two outputs are bit-identical, else the max absolute difference."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf")
    if torch.equal(a, b):
        return None
    return (a.float() - b.float()).abs().max().item()


class _rounded_key_cache:
    """THE MUTATION: re-introduce the defect, without re-implementing the kernel maths.

    The wrapper looks a pair up under `round(sigma, 3)` and, on a miss, delegates to the
    SHIPPED builder at the full sigma and files the result under the rounded key — which is
    exactly what the base sha did in one function. It shares the real cache, so `_fresh()`
    still clears everything this file builds."""

    def __enter__(self):
        real = _stdlib_core._get_gauss_kernels

        def _rounded(sigma, device):
            key = (round(sigma, 3), device)
            hit = _stdlib._gauss_kernel_cache.get(key)
            if hit is not None:
                return hit
            pair = real(sigma, device)            # built from the FULL sigma, as before
            _stdlib._gauss_kernel_cache[key] = pair
            return pair
        self._real = real
        # Patch the DEFINING module (`stdlib_core`), not the facade's re-export — see the
        # LIB-1 note beside the import above; `_gauss_blur_bchw`'s own global lives there.
        _stdlib_core._get_gauss_kernels = _rounded
        return self

    def __exit__(self, *exc):
        _stdlib_core._get_gauss_kernels = self._real
        _fresh()
        return False


def _order_check(sigma, device, tier):
    """Blur at `sigma` and `sigma + 5e-4` in BOTH orders, each from a cleared cache.

    Returns None when each sigma answered the same in both orders (and the two sigmas
    answered differently from each other, so the row measures something), else a message.

    The order comparison is made FIRST and the vacuity guard second, deliberately: a cache
    that serves one kernel for both sigmas makes the two sigmas agree WITHIN an order, so
    testing vacuity first would report the defect as "this row proves nothing".
    """
    hi = sigma + _DELTA
    _fresh()
    a_lo, a_hi = _blur(sigma, device, tier), _blur(hi, device, tier)
    _fresh()
    b_hi, b_lo = _blur(hi, device, tier), _blur(sigma, device, tier)
    _fresh()
    for label, first, second in (("lo", a_lo, b_lo), ("hi", a_hi, b_hi)):
        d = _differs(first, second)
        if d is not None:
            return (f"sigma-{label} depends on blur order: maxdiff {d:g} between "
                    f"'{sigma!r} then {hi!r}' and '{hi!r} then {sigma!r}'")
    if _differs(a_lo, a_hi) is None:
        return (f"sigma {sigma!r} and {hi!r} produced identical pixels, so the row cannot "
                f"detect a shared kernel")
    return None


# ── 1. the pin: the same program answers the same, whatever ran before it ─────

def test_perf3_a_blur_does_not_depend_on_what_was_blurred_first(r: SubTestResult):
    """THE RED-FIRST ROW. On the base sha the two sigmas share one cache entry, so each
    order serves whichever kernel it built first and the two orders disagree."""
    print("\n--- PERF-3: a blur's output does not depend on blur order ---")
    for device in _devices():
        for tier in ("interp", "codegen"):
            for label, sigma in _SIGMA_ROWS:
                name = f"[{device}/{tier}] gauss_blur, {label} (sigma {sigma!r})"
                try:
                    bad = _order_check(sigma, device, tier)
                    if bad:
                        r.fail("PERF-3 order independence", f"{name}: {bad}")
                    else:
                        r.ok(name)
                except Exception as e:
                    r.fail("PERF-3 order independence", f"{name}: {type(e).__name__}: {e}")


def test_perf3_the_cache_is_keyed_on_the_sigma_it_builds_from(r: SubTestResult):
    """The unit-level statement of the same thing, and the check the PERF-2 findings named:
    build a kernel at one sigma, then at a sigma that rounds to the same three decimals, and
    require the second call to hand back its OWN kernel rather than the first's tensor."""
    print("\n--- PERF-3: the kernel cache keys on the sigma it built from ---")
    dev = torch.device("cpu")
    for label, sigma in _SIGMA_ROWS:
        hi = sigma + _DELTA
        name = f"{label}: {sigma!r} vs {hi!r}"
        try:
            assert round(sigma, 3) == round(hi, 3), (
                f"{name}: the pair no longer collides under a 3-decimal key, so the row "
                f"would pass without exercising anything")
            _fresh()
            first_h, _ = _stdlib._get_gauss_kernels(hi, dev)
            second_h, _ = _stdlib._get_gauss_kernels(sigma, dev)
            _fresh()
            assert second_h is not first_h, (
                f"{name}: the second sigma was served the first sigma's tensor")
            assert _differs(first_h, second_h) is not None, (
                f"{name}: the two kernels are bit-identical, so the row proves nothing")
            r.ok(f"{name} (taps {first_h.shape[-1]} vs {second_h.shape[-1]})")
        except Exception as e:
            r.fail("PERF-3 cache key", f"{name}: {type(e).__name__}: {e}")


def test_perf3_a_repeated_sigma_still_hits_the_cache(r: SubTestResult):
    """The other half of a cache key: exact keying must not turn every call into a rebuild.
    The same sigma asked for twice is the same tensor, so the cook path is unchanged."""
    print("\n--- PERF-3: the same sigma still hits ---")
    dev = torch.device("cpu")
    try:
        _fresh()
        a_h, a_v = _stdlib._get_gauss_kernels(1.9996, dev)
        b_h, b_v = _stdlib._get_gauss_kernels(1.9996, dev)
        assert a_h is b_h and a_v is b_v, "a repeated sigma rebuilt its kernel"
        assert len(_stdlib._gauss_kernel_cache) == 1, (
            f"one sigma left {len(_stdlib._gauss_kernel_cache)} entries")
        _fresh()
        r.ok("a repeated sigma is served from the cache")
    except Exception as e:
        r.fail("PERF-3 cache hit", f"{type(e).__name__}: {e}")


# ── 2. the mutation: the pin fails against the defect it was written for ──────

def test_perf3_the_rounded_key_is_detected(r: SubTestResult):
    """THE OTHER DIRECTION. Put `round(sigma, 3)` back — look the pair up rounded, build it
    from the full sigma — and the order-independence check above must FAIL for every row.
    A check that had gone inert would pass here, and would look like a fixed cache."""
    print("\n--- PERF-3: a rounded key is caught by the pin ---")
    for label, sigma in _SIGMA_ROWS:
        name = f"mutation, {label} (sigma {sigma!r})"
        try:
            with _rounded_key_cache():
                bad = _order_check(sigma, "cpu", "interp")
            if bad is None:
                r.fail("PERF-3 mutation",
                       f"{name}: a rounded key produced order-INDEPENDENT output, so the pin "
                       f"above does not measure the defect it was written for")
            else:
                r.ok(f"{name}: caught ({bad})")
        except Exception as e:
            r.fail("PERF-3 mutation", f"{name}: {type(e).__name__}: {e}")


def test_perf3_the_radius_rule_is_the_one_the_key_assumes(r: SubTestResult):
    """The key is `(sigma, device)` and nothing else, which is only right while the kernel
    depends on nothing else. Pin the two things it does depend on: the radius rule
    `ceil(3*sigma)` (also invariant #5's `('halo_arg', 1, 3.0)` reach multiplier) and the
    fp32 build dtype. A future kernel that varied with anything more would need that in the
    key, and this row is where that is noticed."""
    print("\n--- PERF-3: what the kernel depends on ---")
    dev = torch.device("cpu")
    try:
        _fresh()
        for sigma in (0.3, 1.0, 1.5, 1.9996, 2.0001, 7.9):
            kh, kv = _stdlib._get_gauss_kernels(sigma, dev)
            radius = int(math.ceil(3.0 * sigma))
            assert kh.shape == (1, 1, 1, 2 * radius + 1), (
                f"sigma {sigma!r}: {tuple(kh.shape)} is not ceil(3*sigma)={radius}")
            assert kh.dtype is torch.float32 and kv.dtype is torch.float32, (
                f"sigma {sigma!r}: kernels are built fp32, got {kh.dtype}")
        _fresh()
        r.ok("radius is ceil(3*sigma) and the build dtype is fp32, for every sigma tried")
    except Exception as e:
        r.fail("PERF-3 kernel dependencies", f"{type(e).__name__}: {e}")
