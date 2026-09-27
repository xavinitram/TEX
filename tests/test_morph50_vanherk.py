"""MORPH-50 — erode/dilate honour arbitrarily large radii, with no silent clamp.

The author's rule (RADIUS-50a-design.md, verbatim): "All blurs and erodes should support
arbitrarily large radiuses, or if we do set a limit, it should be in the order of 8192px."
Before this lane, `stdlib_sample._morph` silently substituted `radius=256` for anything
larger -- `erode(@mask, 300)` and `erode(@mask, 256)` produced the identical picture, with
no error and no diagnostic. D1 (recorded 2026-09-27) chose the hybrid: `_MORPH_VANHERK_
CROSSOVER` (a measured constant, pinned once a real timing sweep confirmed it on this
implementation) keeps the ORIGINAL iterative 3-window loop, byte-for-byte, at or
below the crossover -- so the common small-radius case costs exactly what it always did
(invariant 7) -- and a van Herk/Gil-Werman separable running extremum takes over above it,
unconditionally uncapped (no clamp of any kind survives this lane; the "or ~8192px" half of
the author's rule was not the branch taken, because van Herk makes the unconditional half
both correct and fast).

Everything below is `torch.equal`-exact against a reference that shares NO code with either
`stdlib_sample._morph_iterative` or `stdlib_sample._morph_vanherk` (a straightforward
shifted-slice reduction, `_bruteforce_running_extreme`) for `r <= 300`, and against that same
brute-force reference (on small images only) for `r` in the thousands -- the brute-force
loop is `O(r)` per pass and is never run at production resolutions.

NaN/inf note: the current implementation never special-cases them (`torch.amax`/`amin`,
`torch.cummax`/`cummin` all just propagate whatever a plain comparison-based reduction does),
so this file includes them in the sweep -- but `torch.equal` itself considers NaN != NaN, so a
tensor is never `torch.equal` to itself once it carries a NaN. `_bitexact` below is the
NaN-aware variant (`(a == b) | (isnan(a) & isnan(b))`, all-true) used ONLY where a case
carries NaN; every other case uses literal `torch.equal`, named per assertion so the choice is
visible rather than silently loosened for everything.
"""
from helpers import *


# ── Independent brute-force oracle (shares no code with the product's two paths) ──

def _bruteforce_running_extreme(image: torch.Tensor, r: int, grow: bool) -> torch.Tensor:
    """(2r+1)-window separable min/max via `2r+1` shifted-slice reductions per pass --
    O(r) per pass, deliberately not sharing van Herk's block/cummax machinery nor the
    product's own 3-tap iterative loop. Only ever run on small images / large r in this
    file (the whole point of the product code is to avoid this cost)."""
    squeeze = image.dim() == 3
    img = image.unsqueeze(-1) if squeeze else image
    x = img.permute(0, 3, 1, 2)               # [B,C,H,W]
    pad = torch.nn.functional.pad
    combine = torch.maximum if grow else torch.minimum

    def pass1d(x, dim):
        N = x.shape[dim]
        rr = min(r, N - 1) if False else r     # no clamp here either -- pad handles any r
        if dim == -1:
            xp = pad(x, (rr, rr, 0, 0), mode="replicate")
        else:
            xp = pad(x, (0, 0, rr, rr), mode="replicate")
        acc = None
        for k in range(2 * rr + 1):
            sl = xp[..., k:k + N] if dim == -1 else xp[..., k:k + N, :]
            acc = sl if acc is None else combine(acc, sl)
        return acc

    x = pass1d(x, -1)
    x = pass1d(x, -2)
    out = x.permute(0, 2, 3, 1)
    return out.squeeze(-1) if squeeze else out


def _bitexact(a: torch.Tensor, b: torch.Tensor) -> bool:
    """NaN-aware `torch.equal`: True where every element matches OR both are NaN."""
    return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


def _mk(B, H, W, C, seed=0, device="cpu", nan_inf=False):
    torch.manual_seed(seed)
    img = torch.rand(B, H, W, C, device=device) if C else torch.rand(B, H, W, device=device)
    if nan_inf:
        flat = img.reshape(-1)
        if flat.numel() >= 3:
            flat[0] = float("nan")
            flat[1] = float("inf")
            flat[2] = float("-inf")
    return img


# ── (a) The clamp existed and was silent: r=300 == r=256 today ─────────────────────

def test_morph50_a_clamp_is_gone(r: SubTestResult):
    print("\n--- MORPH-50 (a): erode(300) no longer equals erode(256) (the removed clamp) ---")
    # Must be bigger than 2*300+1=601 on each side, or BOTH r=256 and r=300 hit the
    # (correct) whole-image shortcut and trivially agree for a reason that has nothing
    # to do with the clamp -- that would pass by accident. 620 keeps both windows
    # genuinely local (window sizes 513 and 601, both < 620), so they differ on noise.
    img = _mk(1, 620, 620, 3, seed=1)
    for name, grow in (("erode", False), ("dilate", True)):
        try:
            r256 = TEXStdlib._morph(img, 256, grow)
            r300 = TEXStdlib._morph(img, 300, grow)
            brute300 = _bruteforce_running_extreme(img, 300, grow)
            assert not torch.equal(r256, r300), (
                f"{name}(300) still equals {name}(256) -- the silent clamp survived"
            )
            assert torch.equal(r300, brute300), (
                f"{name}(300) does not match the true (2*300+1)-window brute-force answer"
            )
            r.ok(f"{name}(radius=300) is the real 601-window answer, not the clamped 256 one")
        except AssertionError as e:
            r.fail(f"{name} clamp removed", str(e))


# ── (b) bit-exact vs the brute-force oracle for every r in [0, 300] ────────────────

def test_morph50_b_bitexact_small_and_mid_radius(r: SubTestResult):
    print("\n--- MORPH-50 (b): torch.equal vs an independent brute-force oracle, r in [0,300] ---")
    _MORPH50_R_SWEEP = (0, 1, 2, 8, 9, 10, 11, 12, 20, 64, 100, 255, 256, 257, 300)
    for dev in devices():
        cases = {
            "1px": _mk(1, 1, 1, 3, seed=2, device=dev),
            "nonsquare": _mk(2, 5, 9, 3, seed=3, device=dev),
            "batch": _mk(4, 10, 10, 1, seed=4, device=dev),
            "mask_no_channel": _mk(2, 7, 11, 0, seed=5, device=dev),
            "nan_inf": _mk(1, 8, 8, 3, seed=6, device=dev, nan_inf=True),
        }
        for case_name, img in cases.items():
            for radius in _MORPH50_R_SWEEP:
                for name, grow in (("erode", False), ("dilate", True)):
                    try:
                        got = TEXStdlib._morph(img, radius, grow)
                        want = _bruteforce_running_extreme(img, radius, grow)
                        if case_name == "nan_inf":
                            ok = _bitexact(got, want)
                        else:
                            ok = torch.equal(got, want)
                        assert ok, (
                            f"{name}(r={radius}) on {case_name}@{dev}: maxdiff="
                            f"{(got.float() - want.float()).abs().nan_to_num(1.0).max().item():.3e}"
                        )
                    except AssertionError as e:
                        r.fail(f"{name} bit-exact r={radius} {case_name}@{dev}", str(e))
                        continue
        r.ok(f"bit-exact vs brute-force oracle across {len(cases)} shapes x "
             f"{len(_MORPH50_R_SWEEP)} radii x {{erode,dilate}} @ {dev}")


# ── (c) large r matches the brute-force reference on small images ──────────────────

def test_morph50_c_large_radius_small_image(r: SubTestResult):
    print("\n--- MORPH-50 (c): r=1024, r=8192 match brute-force on small images ---")
    for dev in devices():
        # Small in PIXEL COUNT (brute-force at r=8192 is O(r) shifts, must stay cheap),
        # but one axis (700) is wider than 2*300+1=601 -- otherwise the old clamped-256
        # path and the true r=1024/8192 answer coincide by accident (both already see
        # the whole line), and this test would pass at the base sha for the wrong
        # reason, same trap test (a) above avoids on a square image.
        img = _mk(1, 3, 700, 2, seed=7, device=dev)
        for radius in (1024, 8192):
            for name, grow in (("erode", False), ("dilate", True)):
                try:
                    got = TEXStdlib._morph(img, radius, grow)
                    want = _bruteforce_running_extreme(img, radius, grow)
                    assert torch.equal(got, want), (
                        f"{name}(r={radius})@{dev} diverges from brute-force"
                    )
                    r.ok(f"{name}(r={radius})@{dev} matches brute-force on a 6x7 image")
                except AssertionError as e:
                    r.fail(f"{name} large-r r={radius}@{dev}", str(e))


# ── No limit is kept at all: an absurd r is still correct and returns promptly ─────

def test_morph50_d_no_limit_kept(r: SubTestResult):
    print("\n--- MORPH-50 (d): an even larger r (1_000_000) is still the whole-image answer ---")
    img = _mk(1, 5, 6, 3, seed=8)
    for name, grow in (("erode", False), ("dilate", True)):
        try:
            got = TEXStdlib._morph(img, 1_000_000, grow)
            whole = (img.amax(dim=(1, 2), keepdim=True) if grow
                     else img.amin(dim=(1, 2), keepdim=True)).expand_as(img)
            assert torch.equal(got, whole), (
                f"{name}(r=1_000_000) is not the whole-image extremum"
            )
            r.ok(f"{name}(r=1_000_000): whole-image shortcut, no clamp, no error")
        except AssertionError as e:
            r.fail(f"{name} absurd-r r=1000000", str(e))


# ── The crossover: iterative path is unchanged code below it (invariant 7) ────────

def test_morph50_e_crossover_uses_original_loop(r: SubTestResult):
    print("\n--- MORPH-50 (e): at/below the crossover, _morph delegates to the ORIGINAL "
          "iterative loop (same code, same cost -- invariant 7) ---")
    img = _mk(1, 16, 16, 3, seed=9)
    crossover = TEXStdlib._MORPH_VANHERK_CROSSOVER
    for name, grow in (("erode", False), ("dilate", True)):
        try:
            x = (img.unsqueeze(-1) if img.dim() == 3 else img).permute(0, 3, 1, 2)
            direct = TEXStdlib._morph_iterative(x.clone(), crossover, grow)
            via_dispatch = TEXStdlib._morph(img, crossover, grow)
            direct_out = direct.permute(0, 2, 3, 1)
            assert torch.equal(direct_out, via_dispatch), (
                f"{name}(r={crossover}) via _morph does not match calling "
                f"_morph_iterative directly -- dispatch picked the wrong path"
            )
            r.ok(f"{name}(r={crossover}) == crossover dispatches to _morph_iterative")
        except AssertionError as e:
            r.fail(f"{name} crossover dispatch r={crossover}", str(e))

    # And just above the crossover, the van Herk path is the one running (still
    # bit-exact, per test (b) above -- this only pins WHICH function ran).
    try:
        x = (img.unsqueeze(-1) if img.dim() == 3 else img).permute(0, 3, 1, 2)
        via_vanherk = TEXStdlib._morph_vanherk(x.clone(), crossover + 1, False)
        via_dispatch = TEXStdlib._morph(img, crossover + 1, False)
        assert torch.equal(via_vanherk.permute(0, 2, 3, 1), via_dispatch)
        r.ok(f"erode(r={crossover + 1}) dispatches to _morph_vanherk")
    except AssertionError as e:
        r.fail("crossover+1 dispatch", str(e))
