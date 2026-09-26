"""SCALE-47b phase 1 — the `pixel_args=` registry tag.

`SCALE-47-design.md` §1/§3 (AUTHOR DECISIONS #1): a pixel-unit stdlib argument needs its own
registry tag, separate from `footprint`'s `mult` (a `('halo_arg', i, mult)` reach multiplier
answers "how far does this arg reach in pixels", not "should this arg's VALUE scale with the
cook's resolution" — `bilateral_filter` is the case that forces the split: its footprint is a
FIXED `('halo', 3)` with no `halo_arg`, yet `spatial_sigma` still needs scaling.

This file proves the tag exists, is validated the same loud way `footprint` is (AGENTS.md
invariant #5 — a malformed descriptor must fail at import, never silently mis-tag), and is
attached to exactly the five §1 builtins whose sigma/radius argument is a pixel magnitude:
`gauss_blur`, `erode`, `dilate`, `bilateral_filter` (spatial_sigma only, NOT range_sigma).
"""
from helpers import *
from TEX_Wrangle.tex_runtime import stdlib_registry as R


def test_scale47b_pixel_args_field_exists(r: SubTestResult):
    print("\n--- SCALE-47b: StdlibEntry carries a pixel_args field, default empty ---")
    e = R.StdlibEntry("__test_dummy_no_pixel_args__", lambda *a: None)
    if not hasattr(e, "pixel_args"):
        r.fail("pixel_args field", "StdlibEntry has no `pixel_args` attribute")
        return
    if e.pixel_args != ():
        r.fail("pixel_args default", f"expected () by default, got {e.pixel_args!r}")
        return
    r.ok("StdlibEntry.pixel_args defaults to ()")


def test_scale47b_decorator_accepts_pixel_args(r: SubTestResult):
    print("\n--- SCALE-47b: @stdlib(...) accepts pixel_args= and records it ---")
    saved_registry_snapshot = list(R.REGISTRY)
    name = "__test_scale47b_pixel_args_dummy__"
    try:
        @R.stdlib(name, footprint=("halo_arg", 1, 3.0), pixel_args=(1,))
        def _dummy(image, sigma):
            return None

        entry = next((e for e in R.REGISTRY if e.name == name), None)
        if entry is None:
            r.fail("registration", f"{name!r} did not register")
            return
        if entry.pixel_args != (1,):
            r.fail("pixel_args recorded", f"expected (1,), got {entry.pixel_args!r}")
            return
        r.ok("a decorated fn's pixel_args=(1,) round-trips through the registry")
    finally:
        R.REGISTRY[:] = saved_registry_snapshot


def test_scale47b_pixel_args_by_name(r: SubTestResult):
    print("\n--- SCALE-47b: pixel_args_by_name() derives {name: positions} from the registry ---")
    if not hasattr(R, "pixel_args_by_name"):
        r.fail("pixel_args_by_name", "stdlib_registry has no pixel_args_by_name()")
        return
    m = R.pixel_args_by_name()
    expected = {
        "gauss_blur": (1,),
        "erode": (1,),
        "dilate": (1,),
        "bilateral_filter": (1,),
    }
    for name, want in expected.items():
        got = m.get(name)
        if got != want:
            r.fail(f"pixel_args_by_name[{name}]", f"expected {want!r}, got {got!r}")
            return
    r.ok(f"pixel_args_by_name() reports the five §1 builtins correctly: {sorted(expected)}")


def test_scale47b_bilateral_range_sigma_not_tagged(r: SubTestResult):
    print("\n--- SCALE-47b: bilateral_filter's range_sigma (arg 2) is NOT a pixel_args position ---")
    m = R.pixel_args_by_name()
    got = m.get("bilateral_filter")
    if got is None or 2 in got:
        r.fail("bilateral_filter pixel_args", f"expected an entry excluding index 2, got {got!r}")
        return
    r.ok("range_sigma (a colour-similarity value, not a pixel distance) is excluded")
