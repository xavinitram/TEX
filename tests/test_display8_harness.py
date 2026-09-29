"""GAUSS8-51/BILAT8-51 common brief: `tools/display8.py`, the shared display-8
harness the display-8 approximation work measures against (an approximate path vs. an exact
reference, mapped through the ACES RRT + sRGB ODT and rounded to 8 bits).

Fast rows only: small plates, no blur/host call — this file pins the harness
functions themselves (shapes, determinism, the diff-stats contract), not a
builtin's accuracy band (that belongs to each builtin's own test file).
"""
from __future__ import annotations

import torch

from helpers import *
from helpers import load_display8_harness

_d8 = load_display8_harness()
aces_srgb8, code_diff_stats, plate_day, plate_night = (
    _d8.aces_srgb8, _d8.code_diff_stats, _d8.plate_day, _d8.plate_night,
)


def test_display8_plates_are_deterministic_and_shaped(r: SubTestResult):
    print("\n--- display8: plate_day/plate_night are deterministic given a seed, [1,3,H,W] ---")
    H = W = 32
    for name, fn in (("day", plate_day), ("night", plate_night)):
        a = fn(H, W, torch.device("cpu"))
        b = fn(H, W, torch.device("cpu"))
        if tuple(a.shape) != (1, 3, H, W):
            r.fail(f"display8 {name} shape", f"got {tuple(a.shape)}, expected (1, 3, {H}, {W})")
            continue
        if not torch.equal(a, b):
            r.fail(f"display8 {name} determinism", "same seed produced different plates")
            continue
        if not torch.isfinite(a).all():
            r.fail(f"display8 {name} finite", "non-finite value in the plate")
            continue
        r.ok(f"plate_{name} is deterministic and shaped [1,3,{H},{W}]")
    day_seed1 = plate_day(H, W, torch.device("cpu"), seed=1)
    day_seed2 = plate_day(H, W, torch.device("cpu"), seed=2)
    if torch.equal(day_seed1, day_seed2):
        r.fail("display8 seed sensitivity", "plate_day(seed=1) == plate_day(seed=2)")
    else:
        r.ok("plate_day varies with its seed argument")


def test_display8_aces_srgb8_shape_and_range(r: SubTestResult):
    print("\n--- display8: aces_srgb8 maps [1,3,H,W] scene-linear to [H,W,3] int16 codes in [0,255] ---")
    H = W = 16
    img = plate_night(H, W, torch.device("cpu"))
    codes = aces_srgb8(img)
    if tuple(codes.shape) != (H, W, 3):
        r.fail("display8 aces_srgb8 shape", f"got {tuple(codes.shape)}, expected ({H}, {W}, 3)")
        return
    if codes.dtype != torch.int16:
        r.fail("display8 aces_srgb8 dtype", f"got {codes.dtype}, expected torch.int16")
        return
    lo, hi = int(codes.min()), int(codes.max())
    if lo < 0 or hi > 255:
        r.fail("display8 aces_srgb8 range", f"code range [{lo}, {hi}] outside [0, 255]")
        return
    r.ok(f"aces_srgb8 -> int16 [H,W,3] in [0,255] (observed [{lo}, {hi}])")


def test_display8_code_diff_stats_contract(r: SubTestResult):
    print("\n--- display8: code_diff_stats reports the agreed keys, zero for a==b, nonzero for a!=b ---")
    H = W = 8
    a = torch.zeros(H, W, 3, dtype=torch.int16)
    same = code_diff_stats(a, a.clone())
    expected_keys = {"changed", "ge2", "ge4", "max", "mean_signed", "centre_ge2", "centre_max"}
    if set(same.keys()) != expected_keys:
        r.fail("display8 diff stats keys", f"got {sorted(same.keys())}, expected {sorted(expected_keys)}")
        return
    if same["changed"] != 0.0 or same["max"] != 0 or same["mean_signed"] != 0.0 or same["centre_max"] != 0:
        r.fail("display8 diff stats identity", f"a==b should read all-zero, got {same}")
        return
    r.ok("code_diff_stats(a, a) reads all-zero across every key")

    b = a.clone()
    b[H // 2, W // 2, 0] = 5  # one centre pixel, worst channel diff 5
    diff = code_diff_stats(a, b)
    if diff["max"] != 5 or diff["centre_max"] != 5:
        r.fail("display8 diff stats magnitude", f"expected max/centre_max == 5, got {diff}")
        return
    n = H * W
    if (diff["changed"], diff["ge2"], diff["ge4"]) != (1 / n, 1 / n, 1 / n):
        r.fail("display8 diff stats fractions",
               f"one changed pixel of {n} should give changed = ge2 = ge4 = {1 / n}, got {diff}")
        return
    if diff["centre_ge2"] != 1 / ((H // 2) * (W // 2)):
        r.fail("display8 diff stats centre",
               f"the centre half is {H // 2}x{W // 2} pixels, so centre_ge2 should be "
               f"{1 / ((H // 2) * (W // 2))}, got {diff['centre_ge2']}")
        return
    if abs(diff["mean_signed"] + 5.0 / (n * 3)) > 1e-12:
        r.fail("display8 diff stats mean", f"mean_signed should be -5/{n * 3}, got {diff['mean_signed']}")
        return
    if diff["mean_signed"] >= 0.0:
        r.fail("display8 diff stats sign", f"code_diff_stats(a, b) reports (a - b); b > a at one "
               f"pixel should give a negative mean_signed, got {diff['mean_signed']}")
        return
    r.ok(f"code_diff_stats(a, b) reports a single 5-code centre diff correctly: {diff}")


def test_display8_code_diff_stats_tiny_frame_falls_back_to_full_frame(r: SubTestResult):
    print("\n--- display8: on a frame with H < 4 the centre half is empty, so centre_* read the whole frame ---")
    a = torch.zeros(1, 4, 3, dtype=torch.int16)
    b = a.clone()
    b[0, 3, 0] = 5
    stats = code_diff_stats(a, b)
    if stats["centre_max"] != 5 or stats["centre_ge2"] != 0.25:
        r.fail("display8 diff stats tiny frame",
               f"expected centre_max 5 and centre_ge2 1/4 from the full-frame fallback, got {stats}")
        return
    r.ok(f"a 1x4 frame's centre stats fall back to the full frame: {stats}")
