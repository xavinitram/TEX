"""ASK-13 — `patch_dist`, a patch-distance primitive.

Design: docs/worklog/ask-13/design.md (decided; this file implements its §4
red-first list, point 6 (edge cases) and the E3011 reserved-name row — points
1/2/3 land as edits to the existing test files their rows name: test_v017_phase2.py
(TST-3 forgotten-tag heuristic), test_v024_phase1.py (ROI reach pin),
test_codegen_optimizer.py (equivalence corpus), stdlib_probe.py (fuzzer/edge-matrix
coverage), test_v017_phase1.py (fuzzer-grammar exclusion pin), test_v019_phase1.py
(precision="auto" decline), test_v023_phase1.py (_NON_LOCAL_SINCE_V022 literal).

design.md §4 point 6 asks for each edge case "both devices, both tiers" — `_DEVICES`
below follows the same convention as test_v024_phase1.py / test_v02[5-8]_phase1.py.
"""
from helpers import *
from failure_harness import run_tier, max_diff

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]


def _raises(code, bindings, tier, needle, device="cpu"):
    """Run `code` under `tier`/`device` and assert it raises with `needle` in the message."""
    try:
        run_tier(code, bindings, tier, device=device)
        return None
    except Exception as e:
        msg = str(e)
        return msg if needle in msg else f"raised but missing {needle!r}: {msg[:150]}"


def test_ask13_t6_radius0_equals_pointwise(r: SubTestResult):
    print("\n--- ASK-13 T6: radius=0 == the pointwise channel-mean squared difference ---")
    img = make_img(1, 6, 6, 3, seed=1)
    dx, dy = 2, -1
    code = "m@OUT = patch_dist(@A.rgb, 2, -1, 0);"
    # Hand-computed reference: shift by (dx, dy) with replicate-clamped borders via
    # plain indexing (torch.roll + edge overwrite would wrap, so build it by hand).
    B, H, W, C = img.shape
    shifted = torch.empty_like(img)
    for y in range(H):
        sy = min(max(y + dy, 0), H - 1)
        for x in range(W):
            sx = min(max(x + dx, 0), W - 1)
            shifted[:, y, x, :] = img[:, sy, sx, :]
    expected = ((img - shifted) ** 2).mean(dim=-1)  # [B,H,W]
    for dev in _DEVICES:
        for tier in ("interp", "codegen"):
            try:
                got = run_tier(code, {"A": img.to(dev)}, tier, device=dev)["OUT"]
                md = (got.cpu() - expected).abs().max().item()
                assert md < 1e-5, f"maxdiff {md}"
                r.ok(f"[{dev}/{tier}] radius=0 matches hand-computed pointwise diff "
                     f"(maxdiff {md:.2e})")
            except Exception as e:
                r.fail(f"ASK-13 T6 radius0 [{dev}/{tier}]", f"{type(e).__name__}: {e}")


def test_ask13_t6_zero_offset_exact_zero(r: SubTestResult):
    print("\n--- ASK-13 T6: dx=dy=0 returns EXACTLY 0.0 everywhere (bit-exact, it is x-x) ---")
    img = make_img(1, 5, 5, 4, seed=2)
    for radius in (0, 1, 3):
        code = f"m@OUT = patch_dist(@A.rgb, 0, 0, {radius});"
        for dev in _DEVICES:
            for tier in ("interp", "codegen"):
                try:
                    got = run_tier(code, {"A": img.to(dev)}, tier, device=dev)["OUT"]
                    assert bool(torch.all(got == 0.0)), \
                        f"nonzero at dx=dy=0 (radius={radius}): max={got.abs().max().item()}"
                    r.ok(f"[{dev}/{tier}] dx=dy=0, radius={radius}: exactly 0.0 everywhere")
                except Exception as e:
                    r.fail(f"ASK-13 T6 zero-offset [{dev}/{tier}] r={radius}",
                           f"{type(e).__name__}: {e}")


def test_ask13_t6_large_offset_finite_clamped(r: SubTestResult):
    print("\n--- ASK-13 T6: dx=W+50 is finite, no raise, matches the clamped-border compare ---")
    img = make_img(1, 4, 4, 3, seed=3)
    B, H, W, C = img.shape
    big_dx = W + 50
    code = f"m@OUT = patch_dist(@A.rgb, {big_dx}, 0, 0);"
    # An offset this large always clamps to the last column (replicate border): the
    # expected field is each pixel compared against ITS OWN ROW's last column.
    last_col = img[:, :, -1:, :].expand(-1, -1, W, -1)
    expected = ((img - last_col) ** 2).mean(dim=-1)  # [B,H,W]
    for dev in _DEVICES:
        for tier in ("interp", "codegen"):
            try:
                got = run_tier(code, {"A": img.to(dev)}, tier, device=dev)["OUT"]
                assert torch.isfinite(got).all(), "non-finite output for a huge (but in-range) offset"
                md = (got.cpu() - expected).abs().max().item()
                assert md < 1e-5, f"maxdiff {md} vs clamped-border reference"
                r.ok(f"[{dev}/{tier}] dx={big_dx} (> W): finite, matches clamped-border "
                     f"compare (maxdiff {md:.2e}), no raise")
            except Exception as e:
                r.fail(f"ASK-13 T6 large-offset [{dev}/{tier}]", f"{type(e).__name__}: {e}")


def test_ask13_t6_nan_propagates(r: SubTestResult):
    print("\n--- ASK-13 T6: a NaN pixel propagates within reach, agrees across tiers (equal_nan) ---")
    img = make_img(1, 9, 9, 3, seed=4)
    cy, cx = 4, 4
    img = img.clone()
    img[:, cy, cx, :] = float("nan")
    dx, dy, radius = 1, 0, 1
    reach = radius + max(abs(dx), abs(dy))  # design.md §4 point 6
    code = f"m@OUT = patch_dist(@A.rgb, {dx}, {dy}, {radius});"
    for dev in _DEVICES:
        results = {}
        for tier in ("interp", "codegen"):
            try:
                results[tier] = run_tier(code, {"A": img.to(dev)}, tier, device=dev)["OUT"]
            except Exception as e:
                r.fail(f"ASK-13 T6 NaN [{dev}/{tier}]", f"{type(e).__name__}: {e}")
        if len(results) != 2:
            continue
        try:
            got_i, got_c = results["interp"].cpu(), results["codegen"].cpu()
            nan_i, nan_c = torch.isnan(got_i), torch.isnan(got_c)
            assert bool(torch.equal(nan_i, nan_c)), "interp/codegen disagree on WHICH pixels are NaN"
            # Containment (an UPPER bound, not saturation): every NaN pixel must lie
            # within Chebyshev distance `reach` of the source — the shift-then-box-mean
            # structure does NOT NaN every pixel in that box (e.g. the two pre-box-mean
            # NaN sites, (cy,cx) itself and its (dx,dy) antipode, sit off-centre within
            # it), only that no NaN escapes it.
            assert bool(nan_i.any()), "no NaN pixel at all — the source vanished"
            ys, xs = torch.where(nan_i[0])
            within = ((ys >= cy - reach) & (ys <= cy + reach)
                      & (xs >= cx - reach) & (xs <= cx + reach))
            assert bool(torch.all(within)), \
                f"a NaN pixel escaped the declared reach={reach}: rows {ys.tolist()} cols {xs.tolist()}"
            non_nan_agree = torch.where(nan_i, torch.zeros_like(got_i), (got_i - got_c).abs())
            md = non_nan_agree.max().item()
            assert md < 1e-5, f"non-NaN maxdiff {md} between tiers"
            r.ok(f"[{dev}] NaN source at ({cy},{cx}): every affected pixel within "
                 f"reach={reach}, interp==codegen under equal_nan (non-NaN maxdiff {md:.2e})")
        except Exception as e:
            r.fail(f"ASK-13 T6 NaN [{dev}]", f"{type(e).__name__}: {e}")


def test_ask13_t6_per_pixel_offset_raises(r: SubTestResult):
    print("\n--- ASK-13 T6: a per-pixel dx raises the structured uniform refusal ---")
    img = make_img(1, 4, 4, 3, seed=5)
    cases = [
        ("per-pixel dx (ix)", "m@OUT = patch_dist(@A.rgb, ix, -1, 1);"),
        ("per-pixel dy (iy)", "m@OUT = patch_dist(@A.rgb, 2, iy, 1);"),
        ("per-pixel radius (ix)", "m@OUT = patch_dist(@A.rgb, 2, -1, ix);"),
    ]
    for dev in _DEVICES:
        for tier in ("interp", "codegen"):
            for label, code in cases:
                err = _raises(code, {"A": img.to(dev)}, tier, "distinct values", device=dev)
                if err is None:
                    r.fail(f"ASK-13 T6 per-pixel [{dev}/{tier}] {label}",
                           "expected a raise, got none")
                elif err.startswith("raised but missing"):
                    r.fail(f"ASK-13 T6 per-pixel [{dev}/{tier}] {label}", err)
                else:
                    r.ok(f"[{dev}/{tier}] {label}: raises the structured uniform refusal "
                         f"({err[:80]})")


def test_ask13_reserved_name_e3011(r: SubTestResult):
    print("\n--- ASK-13: patch_dist is a reserved builtin name (E3011) ---")
    try:
        raised = None
        try:
            check_code("float patch_dist(float x){ return x; }\n@OUT = vec4(0.0);")
        except Exception as e:
            raised = e
        assert raised is not None, "redefining patch_dist as a user function did not raise"
        code = getattr(getattr(raised, "diagnostic", None), "code", None)
        assert code == "E3011", f"wrong error code: {code!r} (raised={raised!r})"
        r.ok("`float patch_dist(...)` user function is refused as E3011 (reserved builtin)")
    except Exception as e:
        r.fail("ASK-13 E3011", f"{type(e).__name__}: {e}")
