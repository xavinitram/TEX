"""ASK-1 — native `convolve` builtin.

Design: docs/worklog/ask-1/design.md (decided; this file implements its §4 red-first
list, T8/T11 specifically — T1/T2/T3/T6/T7/T7b/T9/T10 land as edits to the existing
test files their rows name). Each test below cites the design row it satisfies.

T8 note (kernel-wider-than-image): the design predicted this fails loud via CF-6
inflating the cook grid to the kernel's extent. Measured against this head: a BARE
`@OUT = convolve(@A, @K);` does NOT raise — `_exec_assignment`'s plain BindingRef
write path (interpreter.py) never checks a written binding's shape against
`spatial_shape`, so `@OUT` lands shaped like `@A` (the image), silently inconsistent
with the inflated grid. The loud failure design describes DOES happen, but only when
the SAME program also consumes a grid-sized builtin alongside convolve's result (e.g.
`+ vec3(u, v, 0.0)`) — verified below. Both facts are pinned; the discrepancy from the
design doc is called out in the hand-back rather than silently "fixed" by forking CF-6
(which design.md §3 explicitly declines for this ask) or by adding a new guard inside
fn_convolve that the design never asked for.
"""
from helpers import *
from failure_harness import run_tier, max_diff


def _raises(code, bindings, tier, needle):
    """Run `code` under `tier` and assert it raises with `needle` in the message."""
    try:
        run_tier(code, bindings, tier)
        return None
    except Exception as e:
        msg = str(e)
        return msg if needle in msg else f"raised but missing {needle!r}: {msg[:150]}"


def test_ask1_t8_1x1_kernel_is_scale(r: SubTestResult):
    print("\n--- ASK-1 T8: 1x1 kernel == img * w (both tiers, value-pinned) ---")
    img = make_img(1, 4, 4, 3, seed=1)
    w = 0.7
    kernel = torch.full((1, 1, 1, 3), w)
    code = "@OUT = convolve(@A, @K, 0);"   # normalize=0: raw weighted sum, no /w cancel
    expected = img * w
    for tier in ("interp", "codegen"):
        try:
            got = run_tier(code, {"A": img, "K": kernel}, tier)["OUT"]
            md = (got - expected).abs().max().item()
            assert md < 1e-5, f"maxdiff {md} vs img*{w}"
            r.ok(f"[{tier}] 1x1 kernel(w={w}), normalize=0 == img*w (maxdiff {md:.2e})")
        except Exception as e:
            r.fail(f"ASK-1 T8 1x1 [{tier}]", f"{type(e).__name__}: {e}")


def test_ask1_t8_2x2_even_kernel_centering(r: SubTestResult):
    print("\n--- ASK-1 T8: 2x2 even kernel centres at k//2 (both tiers, value-pinned) ---")
    # A 4x4 ramp with distinct per-pixel values makes any centering/flip mistake visible.
    img = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1).expand(1, 4, 4, 3).contiguous()

    # Delta kernel, weight 1 at (row0,col0) pre-flip. Post-flip this lands at (dy=1,dx=1)
    # (bottom-right of the 2x2 tap window); with pad_t=pad_l=k//2=1, pad_b=pad_r=0, that
    # tap always reads padded[y+1,x+1] == the UNPADDED original[y,x] for every (y,x) — no
    # replicate border ever engages, so a correct centering reproduces the image EXACTLY.
    kernel_identity = torch.tensor([[1.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1).expand(1, 2, 2, 3).contiguous()

    # Delta kernel, weight 1 at (row1,col1) pre-flip -> flips to (dy=0,dx=0) (top-left of
    # the window) -> output(y,x) = original(y-1,x-1), replicate-clamped at the top/left
    # border (row 0 / col 0 repeat the border value instead of reading out of bounds).
    kernel_diag = torch.tensor([[0.0, 0.0], [0.0, 1.0]]).reshape(1, 2, 2, 1).expand(1, 2, 2, 3).contiguous()
    expected_diag = torch.tensor([
        [0.,  0.,  1.,  2.],
        [0.,  0.,  1.,  2.],
        [4.,  4.,  5.,  6.],
        [8.,  8.,  9., 10.],
    ]).reshape(1, 4, 4, 1).expand(1, 4, 4, 3)

    cases = [
        ("identity corner (row0,col0)", kernel_identity, img),
        ("diagonal-shift corner (row1,col1)", kernel_diag, expected_diag),
    ]
    for tier in ("interp", "codegen"):
        for label, kernel, expected in cases:
            try:
                got = run_tier("@OUT = convolve(@A, @K, 0);", {"A": img, "K": kernel}, tier)["OUT"]
                md = (got - expected).abs().max().item()
                assert md < 1e-5, f"maxdiff {md}"
                r.ok(f"[{tier}] 2x2 centering, {label}: matches (maxdiff {md:.2e})")
            except Exception as e:
                r.fail(f"ASK-1 T8 2x2 centering [{tier}] {label}", f"{type(e).__name__}: {e}")


def test_ask1_t8_nan_kernel_propagates(r: SubTestResult):
    print("\n--- ASK-1 T8: NaN in the kernel propagates to NaN, never laundered ---")
    img = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1).expand(1, 4, 4, 3).contiguous()
    kernel_nan = torch.tensor([[float("nan"), 0.0], [0.0, 0.0]]).reshape(1, 2, 2, 1).expand(1, 2, 2, 3).contiguous()
    for tier in ("interp", "codegen"):
        try:
            got = run_tier("@OUT = convolve(@A, @K, 0);", {"A": img, "K": kernel_nan}, tier)["OUT"]
            # A convolution kernel is applied UNIFORMLY across the whole image, so a NaN
            # tap poisons every single output pixel, not a subset — the strongest form of
            # "not laundered" (no clamp/nan_to_num anywhere in the path).
            assert torch.isnan(got).all(), "NaN kernel weight did not reach every output pixel"
            r.ok(f"[{tier}] NaN kernel weight -> every output pixel is NaN (not laundered)")
        except Exception as e:
            r.fail(f"ASK-1 T8 NaN [{tier}]", f"{type(e).__name__}: {e}")


def test_ask1_t8_kernel_wider_than_image(r: SubTestResult):
    print("\n--- ASK-1 T8: kernel wider than the image (measured, see module docstring) ---")
    img = make_img(1, 4, 4, 3, seed=1)
    kernel_big = make_img(1, 17, 17, 3, seed=2)
    for tier in ("interp", "codegen"):
        # (a) A bare `@OUT = convolve(...)` does not raise: both tiers agree the output
        # is shaped like the IMAGE (4x4), not the kernel-inflated cook grid (17x17) CF-6
        # computes internally — the two tiers stay bit-exact with each other even though
        # neither matches the nominal grid.
        try:
            got = run_tier("@OUT = convolve(@A, @K);", {"A": img, "K": kernel_big}, tier)["OUT"]
            assert got.shape == img.shape, f"OUT shape {got.shape} != image shape {img.shape}"
            r.ok(f"[{tier}] bare convolve(img, oversized kernel): OUT shaped like the image "
                 f"{tuple(got.shape)}, no exception (cook grid was inflated to 17x17 internally)")
        except Exception as e:
            r.fail(f"ASK-1 T8 wider-kernel bare [{tier}]", f"{type(e).__name__}: {e}")

        # (b) Mixing convolve's result with a grid-sized builtin in the SAME program DOES
        # fail loud — the inflated grid actually gets consumed here, and img-shaped (4x4)
        # can't broadcast against grid-shaped (17x17).
        err = _raises("@OUT = convolve(@A, @K) + vec3(u, v, 0.0);", {"A": img, "K": kernel_big},
                      tier, "must match")
        if err is None:
            r.fail(f"ASK-1 T8 wider-kernel mixed [{tier}]",
                   "expected a loud shape error mixing convolve's result with u/v, got none")
        elif "missing" in str(err) and err.startswith("raised but missing"):
            r.fail(f"ASK-1 T8 wider-kernel mixed [{tier}]", err)
        else:
            r.ok(f"[{tier}] convolve(img, oversized kernel) mixed with u/v: fails loud ({err[:80]})")


def test_ask1_t8_invalid_args_raise(r: SubTestResult):
    print("\n--- ASK-1 T8: invalid convolve() args raise, never clamp (both tiers) ---")
    img = make_img(1, 4, 4, 3, seed=1)
    cases = [
        ("Ck not in {1,C}", make_img(1, 3, 3, 2, seed=3), "channel count"),
        ("kernel batch > 1", make_img(2, 3, 3, 3, seed=4), "batch must be 1"),
        ("kernel too large (258)", make_img(1, 258, 3, 3, seed=5), "out of range"),
    ]
    for tier in ("interp", "codegen"):
        for label, kernel, needle in cases:
            err = _raises("@OUT = convolve(@A, @K);", {"A": img, "K": kernel}, tier, needle)
            if err is None:
                r.fail(f"ASK-1 T8 invalid [{tier}] {label}", "expected a raise, got none")
            elif err.startswith("raised but missing"):
                r.fail(f"ASK-1 T8 invalid [{tier}] {label}", err)
            else:
                r.ok(f"[{tier}] {label}: raises loud ({err[:70]})")


def test_ask1_t11_e3011_reserved_name(r: SubTestResult):
    print("\n--- ASK-1 T11: convolve is a reserved builtin name (E3011) ---")
    try:
        raised = None
        try:
            check_code("float convolve(float x){ return x; }\n@OUT = vec4(0.0);")
        except Exception as e:
            raised = e
        assert raised is not None, "redefining convolve as a user function did not raise"
        code = getattr(getattr(raised, "diagnostic", None), "code", None)
        assert code == "E3011", f"wrong error code: {code!r} (raised={raised!r})"
        r.ok("`float convolve(...)` user function is refused as E3011 (reserved builtin)")
    except Exception as e:
        r.fail("ASK-1 T11 E3011", f"{type(e).__name__}: {e}")
