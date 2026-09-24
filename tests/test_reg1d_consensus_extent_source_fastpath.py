"""REG-1d — `_consensus_extent`'s non-spatial walk becomes skippable via source text.

BACKGROUND. `_consensus_extent` (`tex_runtime/interpreter.py`) is the single owner of the
cook's (B, H, W) grid, shared by the interpreter and codegen tiers. Before the COLOR-1
non-spatial-argument exclusion (a LUT-like resource bound alongside an image must never
leak its own leading dims into the grid), the walk that finds which wire names a program
READS ran LAZILY -- only inside the branch that fires when two spatial bindings actually
disagree on shape (129 of 129 frozen corpus programs never disagree, so most cooks paid
nothing). COLOR-1 needed the SAME walk's `non_spatial` half available even in the FIRST,
unnarrowed pass (a LUT's own shape must never even be allowed to trigger a false
disagreement), so it moved the walk to the TOP of the function -- unconditional on every
single cook. `interpreter._READS_MEMO` still makes a WARM cook (the same `Program` object
reused across repeat cooks) pay this only once; a COLD cook builds a fresh `Program` every
time (a fresh compile, e.g. `measure_interp(..., cold=True)`, or a user editing and
re-cooking a TEX node in ComfyUI), so that memo always misses and the walk re-pays on every
single cold first cook -- measured as a small but real remainder of the cold-first-cook
regression tracked upstream.

THE FIX. `_consensus_extent` takes an optional `source` (the interpreter's real cook path
always has it via `self._source`; a fused chain and any caller with no source text pass
"", the existing default, so nothing changes for them). When `source` is given and
mentions NONE of the (small, registry-derived) non-spatial-arg function names
(`stdlib_registry.non_spatial_args_by_name()`), the program cannot possibly bind one --
TEX has no string-built calls, so a real call always spells the name verbatim in source --
and `non_spatial` is taken to be the empty set WITHOUT walking. `read` (needed only by the
narrowing branch) stays deferred exactly as it was before COLOR-1, so a program that is
both LUT-free and never disagrees pays for neither.

THIS FILE proves it by COUNT: (1) a non-`apply_lut3d` program pays ZERO non-spatial walks
across many fresh ("cold") compiles when `source` is supplied, where it paid one per compile
before this fix; (2) that same program, forced to disagree in shape, still walks EXACTLY
once (the narrowing branch's lazy fetch) and reaches the SAME grid as the always-walk path,
so the fast path changes nothing about WHAT is computed, only how many times; (3) a program
that DOES call `apply_lut3d` is unaffected -- the walk still runs and the LUT exclusion
still applies, so nothing here reopens the corruption COLOR-1 fixed.
"""
from helpers import *
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_runtime import interpreter as I

_NON_LUT_CODE = "@OUT = vec4(@A.rgb * 2.0, 1.0);"
_LUT_CODE = "@OUT = vec4(apply_lut3d(@A.rgb, @LUT), 1.0);"


def _fresh_compile(code, bindings):
    """Parse + type-check a FRESH `Program` object every call through the SAME front end
    the production seam uses (`tex_cache.parse_and_split`) -- the cold-compile shape: a
    new object each time defeats `interpreter._READS_MEMO`'s `id()`-keyed memo exactly
    like a real cold cook, without pulling in the benchmark harness for it."""
    bt = {n: _infer_binding_type(v) for n, v in bindings.items()}
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    return prog, tm


def _spy_on_reads_walk():
    """Wrap `interpreter._reads_and_non_spatial_cached` (the function `_consensus_extent`
    calls to pay for the walk) with a call counter, monkeypatched onto the module so
    `_consensus_extent`'s own global lookup picks it up. Returns (counter_dict, restore_fn)."""
    orig = I._reads_and_non_spatial_cached
    counter = {"n": 0}

    def wrapper(program):
        counter["n"] += 1
        return orig(program)

    I._reads_and_non_spatial_cached = wrapper

    def restore():
        I._reads_and_non_spatial_cached = orig

    return counter, restore


def test_reg1d_no_walk_for_non_lut_cold_cooks(r: SubTestResult):
    print("\n--- REG-1d: source lets a non-apply_lut3d cold cook skip the non-spatial walk ---")
    img = make_img(1, 8, 8, 4)
    bindings = {"A": img}
    counter, restore = _spy_on_reads_walk()
    try:
        for _ in range(10):
            prog, tm = _fresh_compile(_NON_LUT_CODE, bindings)   # fresh Program == cold
            Interpreter().execute(prog, dict(bindings), tm, source=_NON_LUT_CODE)
        if counter["n"] != 0:
            r.fail("REG-1d no-walk count",
                   f"expected 0 non-spatial walks across 10 fresh cold cooks of a "
                   f"non-apply_lut3d program (source rules it out), got {counter['n']}")
        else:
            r.ok("0 non-spatial walks across 10 fresh cold cooks (was 1 per cook before "
                 "this fast path)")
    finally:
        restore()


def test_reg1d_split_still_walks_once_and_agrees(r: SubTestResult):
    print("\n--- REG-1d: a genuine shape split still walks (once) and the grid is unchanged ---")
    # Two spatial bindings that DISAGREE in H/W -- b_split/hw_split must fire, and with no
    # apply_lut3d call the fast path must still defer to the walk for the narrowing branch,
    # not skip it outright (skipping would be a SILENT-WRONG regression of the pre-existing,
    # pre-COLOR-1 narrowing rule this file must not touch).
    code = "@OUT = vec4(@A.rgb + @B.rgb, 1.0);"
    big = make_img(1, 16, 16, 4)
    small = make_img(1, 8, 8, 4)
    bindings = {"A": big, "B": small}

    prog_fast, tm_fast = _fresh_compile(code, bindings)
    counter, restore = _spy_on_reads_walk()
    try:
        got = I._consensus_extent(dict(bindings), prog_fast, source=code)
        walks_with_source = counter["n"]
    finally:
        restore()

    prog_base, tm_base = _fresh_compile(code, bindings)
    want = I._consensus_extent(dict(bindings), prog_base, source="")   # baseline: always walks

    if walks_with_source != 1:
        r.fail("REG-1d split walk count",
               f"expected exactly 1 walk once a real shape split forces the narrowing "
               f"branch to fetch `read`, got {walks_with_source}")
    else:
        r.ok("exactly 1 walk on a genuine split (the narrowing branch's lazy fetch)")

    if got != want:
        r.fail("REG-1d split grid agreement",
               f"the source-fast-path grid {got} disagrees with the always-walk baseline "
               f"{want} -- the fast path must never change WHAT is computed")
    else:
        r.ok(f"fast-path grid {got} matches the always-walk baseline exactly")


def test_reg1d_apply_lut3d_program_unaffected(r: SubTestResult):
    print("\n--- REG-1d: a real apply_lut3d call still walks -- the LUT exclusion still applies ---")
    img = make_img(1, 16, 16, 4)
    lut = torch.rand(8, 8, 8, 3)             # deliberately LUT-shaped, disagrees with the image
    bindings = {"A": img, "LUT": lut}

    prog, tm = _fresh_compile(_LUT_CODE, bindings)
    counter, restore = _spy_on_reads_walk()
    try:
        got = I._consensus_extent(dict(bindings), prog, source=_LUT_CODE)
        walks = counter["n"]
    finally:
        restore()

    if walks < 1:
        r.fail("REG-1d apply_lut3d walk count",
               "an apply_lut3d program took the fast path and skipped the walk -- the LUT "
               "exclusion this file must not reopen depends on it running")
    else:
        r.ok(f"apply_lut3d program still walked ({walks} time(s)) -- the fast path "
             "correctly declined it")

    # The grid must be the IMAGE's (16,16), never the LUT's own (8,8,8) leaking in --
    # exactly the corruption COLOR-1 fixed (see stdlib_registry.py's own docstring).
    if got != (1, 16, 16):
        r.fail("REG-1d apply_lut3d grid correctness",
               f"expected the image's grid (1, 16, 16), got {got} -- the LUT's own shape "
               f"leaked into the cook grid")
    else:
        r.ok(f"grid {got} is the image's, not the LUT's -- COLOR-1's exclusion still holds")
