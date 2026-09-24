"""REG-1e — `Program.non_spatial_calls` replaces the source-substring fast path.

BACKGROUND. `_consensus_extent` (`tex_runtime/interpreter.py`) is the single owner of the
cook's (B, H, W) grid. COLOR-1 (v0.40) needed its non-spatial-argument exclusion (a LUT-like
resource bound alongside an image must never leak its own leading dims into the grid)
available on every cook, so the walk that finds it became unconditional — and a cold compile
(a fresh `Program` every time) pays that walk on every single cold first cook, because the
per-`Program`-object memo (`_READS_MEMO`) never hits a fresh object.

A first attempt at fixing this (superseded, never released) let `_consensus_extent` take a
`source` string and skip the walk when `source` mentioned no non-spatial-arg builtin name.
Simplify review found the wrong altitude: `source` is a LOSSY proxy for what the executed
`Program` can call — a fused chain's own source is only its TERMINAL stage's, so a caller
had to special-case it (and any future multi-source surface would need the same audit).

THE FIX, this generation: compute the verdict ONCE, where the `Program` is actually BUILT —
`tex_cache.TEXCache.compile_ast`, the ONE shared post-parse pipeline both the normal
(`compile_tex`) and fused (`tex_fusion.compile_fused`) paths go through — and store it AS AN
ATTRIBUTE on the `Program` itself (`non_spatial_calls: bool | None`). A fused chain's flag is
computed over the FULL spliced AST (every stage inlined), so it is sound by construction,
never a partial view. `_consensus_extent` just reads the attribute:
  - `False` -- the program provably cannot bind a non-spatial argument -- skip the walk.
  - `True` or `None` (not computed: a hand-built `Program` that skipped `compile_ast`, or an
    old disk pickle) -- walk, the safe default either way.

The verdict is computed from `_collect_identifiers_and_calls` (`tex_runtime/interpreter.py`)
— the SAME hand-dispatched, single-pass scan `_collect_identifiers` already paid for at
every compile to build `used_builtins`, now extended to also record a `FunctionCall`'s name.
Never a second tree walk, and never the generic `iter_child_nodes` walk
`_collect_binding_reads_and_non_spatial` uses (that one must find EVERY reader and so can
never stop early; this one already has a narrower, hand-maintained dispatch to keep in sync
for identifier-collection and gains nothing by switching).

`Program`'s pickled shape changes (a new slot) — see `tex_cache.py`'s `_AST_EPOCH` fold
(the SAME "extra byte fragment" mechanism already used for `LANGUAGE_VERSION`): the new
slot's file (`tex_compiler/ast_nodes.py`) is already an `_AST_FILES` member, so its OWN
edit already moves the epoch; the extra fragment additionally covers the case where
`stdlib_registry.non_spatial_args_by_name()`'s answer changes WITHOUT touching an
`_AST_FILES` member (a stdlib leaf marking an EXISTING function's arg non-spatial, without
adding a brand new one) — `tests/test_v025_phase1.py::test_cache4_ast_epoch_folds_language_version`
is adapted for the new formula; `test_reg1e_ast_epoch_folds_non_spatial_set` below adds the
matching end-to-end proof (a `.pkl` persisted under the pre-fold epoch is a miss after).

THIS FILE proves, by COUNT and by construction:
  1. zero non-spatial walks across many fresh ("cold") compiles of a non-`apply_lut3d`
     program, compiled through the REAL `TEXCache.compile_ast` (was one walk per compile);
  2. a genuine shape split still walks exactly once and reaches the same grid;
  3. an `apply_lut3d` program is unaffected — still walks, LUT still excluded;
  4. the fused-chain regression, KEPT from the superseded design: `apply_lut3d` in a
     NON-terminal stage + an image whose H equals the LUT's N (the collision shape) under
     `tex_fusion.compile_fused` — sound by construction, no special case, no source needed.
     `.textool` / GraphSpec fused-tool manifests (`tex_tool.py`, the ComfyUI node's
     `_tex_chain` payload) reach the identical mechanism via `tex_fusion.prepare_fused` ->
     `_stages_from_spec` -> this SAME `compile_fused`, so this is their proof too — not a
     separate call site to special-case.
  5. the "`execute()` without compile" regression, KEPT: a `Program` built by
     `parse_and_split` + a raw `TypeChecker` (bypassing `compile_ast` entirely, exactly what
     a hand-built test AST or an external embedder might do) leaves `non_spatial_calls`
     `None` and still walks — never a silent skip on missing data;
  6. the flag survives a `TEXCache._save_to_disk` / `_load_from_disk` round trip intact.
"""
from helpers import *
from TEX_Wrangle.tex_cache import get_cache, parse_and_split
from TEX_Wrangle.tex_runtime import interpreter as I

_NON_LUT_CODE = "@OUT = vec4(@A.rgb * 2.0, 1.0);"
_LUT_CODE = "@OUT = vec4(apply_lut3d(@A.rgb, @LUT), 1.0);"


def _compile_ast(code, bindings):
    """Through the REAL, shared production pipeline (`TEXCache.compile_ast`) -- a FRESH
    `Program` every call, simulating a cold compile exactly like the benchmark harness's
    `compile_program` does, but proving the actual `non_spatial_calls`-setting code path
    rather than a hand-rolled parse+typecheck."""
    bt = {n: _infer_binding_type(v) for n, v in bindings.items()}
    program = parse_and_split(code, bt)
    program, tm, refs, asg, params, used = get_cache().compile_ast(program, bt, source=code)
    return program, tm


def _spy_on_reads_walk():
    orig = I._reads_and_non_spatial_cached
    counter = {"n": 0}

    def wrapper(program):
        counter["n"] += 1
        return orig(program)

    I._reads_and_non_spatial_cached = wrapper
    return counter, (lambda: setattr(I, "_reads_and_non_spatial_cached", orig))


def test_reg1e_flag_set_correctly_by_compile_ast(r: SubTestResult):
    print("\n--- REG-1e: compile_ast sets Program.non_spatial_calls correctly ---")
    img = make_img(1, 8, 8, 4)
    prog_non_lut, _ = _compile_ast(_NON_LUT_CODE, {"A": img})
    if prog_non_lut.non_spatial_calls is not False:
        r.fail("REG-1e non-LUT flag", f"expected False, got {prog_non_lut.non_spatial_calls!r}")
    else:
        r.ok("a non-apply_lut3d program compiles with non_spatial_calls=False")

    lut = torch.rand(8, 8, 8, 3)
    prog_lut, _ = _compile_ast(_LUT_CODE, {"A": img, "LUT": lut})
    if prog_lut.non_spatial_calls is not True:
        r.fail("REG-1e LUT flag", f"expected True, got {prog_lut.non_spatial_calls!r}")
    else:
        r.ok("an apply_lut3d program compiles with non_spatial_calls=True")


def test_reg1e_no_walk_for_non_lut_cold_cooks(r: SubTestResult):
    print("\n--- REG-1e: zero non-spatial walks across 10 fresh cold compiles (non-LUT) ---")
    img = make_img(1, 8, 8, 4)
    bindings = {"A": img}
    counter, restore = _spy_on_reads_walk()
    try:
        for _ in range(10):
            prog, tm = _compile_ast(_NON_LUT_CODE, bindings)     # fresh Program == cold
            Interpreter().execute(prog, dict(bindings), tm)
        if counter["n"] != 0:
            r.fail("REG-1e no-walk count",
                   f"expected 0 non-spatial walks across 10 fresh cold cooks, got "
                   f"{counter['n']} -- Program.non_spatial_calls did not skip the walk")
        else:
            r.ok("0 non-spatial walks across 10 fresh cold cooks (was 1 per cook before "
                 "either fix generation)")
    finally:
        restore()


def test_reg1e_split_still_walks_once_and_agrees(r: SubTestResult):
    print("\n--- REG-1e: a genuine shape split still walks (once) and the grid is unchanged ---")
    code = "@OUT = vec4(@A.rgb + @B.rgb, 1.0);"
    big = make_img(1, 16, 16, 4)
    small = make_img(1, 8, 8, 4)
    bindings = {"A": big, "B": small}

    prog, tm = _compile_ast(code, bindings)
    if prog.non_spatial_calls is not False:
        r.fail("REG-1e split setup", f"expected non_spatial_calls=False, got "
               f"{prog.non_spatial_calls!r}")
        return

    counter, restore = _spy_on_reads_walk()
    try:
        got = I._consensus_extent(dict(bindings), prog)
        walks = counter["n"]
    finally:
        restore()

    if walks != 1:
        r.fail("REG-1e split walk count",
               f"expected exactly 1 walk once a real shape split forces the narrowing "
               f"branch to fetch `read`, got {walks}")
    else:
        r.ok("exactly 1 walk on a genuine split (the narrowing branch's lazy fetch)")

    if got != (1, 16, 16):
        r.fail("REG-1e split grid", f"expected (1, 16, 16), got {got}")
    else:
        r.ok(f"grid {got} matches the image's real extent")


def test_reg1e_apply_lut3d_program_unaffected(r: SubTestResult):
    print("\n--- REG-1e: a real apply_lut3d call still walks -- the LUT exclusion still applies ---")
    img = make_img(1, 16, 16, 4)
    lut = torch.rand(8, 8, 8, 3)             # deliberately LUT-shaped, disagrees with the image
    bindings = {"A": img, "LUT": lut}

    prog, tm = _compile_ast(_LUT_CODE, bindings)
    counter, restore = _spy_on_reads_walk()
    try:
        got = I._consensus_extent(dict(bindings), prog)
        walks = counter["n"]
    finally:
        restore()

    if walks < 1:
        r.fail("REG-1e apply_lut3d walk count",
               "an apply_lut3d program's non_spatial_calls flag skipped the walk -- the "
               "LUT exclusion this file must not reopen depends on it running")
    else:
        r.ok(f"apply_lut3d program still walked ({walks} time(s))")

    if got != (1, 16, 16):
        r.fail("REG-1e apply_lut3d grid correctness",
               f"expected the image's grid (1, 16, 16), got {got} -- the LUT's own shape "
               f"leaked into the cook grid")
    else:
        r.ok(f"grid {got} is the image's, not the LUT's -- COLOR-1's exclusion still holds")


def test_reg1e_fused_chain_lut_nonterminal_end_to_end(r: SubTestResult):
    print("\n--- REG-1e: apply_lut3d in a NON-TERMINAL fused stage -- sound by construction ---")
    from TEX_Wrangle import tex_fusion as FUS

    N = 8
    img = make_img(1, N, N, 4)                     # H equals the LUT's N -- the collision shape
    lut = torch.rand(N, N, N, 3)
    terminal_code = "@OUT = @X;"                    # the terminal's OWN source never mentions it
    stages = [
        {"code": "@OUT = vec4(apply_lut3d(@IMG.rgb, @LUT), 1.0);",
         "chain_input": None, "bindings": {"IMG": img, "LUT": lut}},
        {"code": terminal_code, "chain_input": "X", "bindings": {}},
    ]
    try:
        prog, tm, refs, asg, params, used, merged = FUS.compile_fused(stages, _infer_binding_type)
    except Exception as e:
        r.fail("REG-1e fused chain setup", f"{type(e).__name__}: {e}")
        return

    # Sound by construction: compile_fused hands the FULL spliced AST to compile_ast, so the
    # flag sees the non-terminal stage's call even though the TERMINAL's own source cannot.
    if prog.non_spatial_calls is not True:
        r.fail("REG-1e fused chain flag",
               f"expected non_spatial_calls=True (apply_lut3d is in stage 0, not the "
               f"terminal) -- got {prog.non_spatial_calls!r}; the terminal-only source "
               f"'{terminal_code}' names no such call, so a source-based guess would have "
               f"missed it -- exactly the class of bug this design closes structurally")
    else:
        r.ok("fused Program.non_spatial_calls=True -- the non-terminal stage's call was seen")

    got = I._consensus_extent(dict(merged), prog)
    if got != (1, N, N):
        r.fail("REG-1e fused chain grid",
               f"expected the image's grid (1, {N}, {N}), got {got} -- the LUT's own shape "
               f"leaked into a fused-chain cook")
    else:
        r.ok(f"fused chain grid {got} is the image's, not the LUT's, under the collision "
             f"shape -- this is also the .textool / GraphSpec chain_payload proof: "
             f"tex_fusion.prepare_fused -> _stages_from_spec -> this same compile_fused")


def test_reg1e_execute_without_compile_still_walks(r: SubTestResult):
    print("\n--- REG-1e: a Program built without compile_ast leaves the flag None -- still walks ---")
    img = make_img(1, 16, 16, 4)
    lut = torch.rand(8, 8, 8, 3)
    bindings = {"A": img, "LUT": lut}
    bt = {n: _infer_binding_type(v) for n, v in bindings.items()}

    # parse_and_split + a raw TypeChecker -- NEVER through compile_ast, exactly what a
    # hand-built test AST or an external embedder might hand the interpreter directly.
    program = parse_and_split(_LUT_CODE, bt)
    tm = TypeChecker(binding_types=bt, source=_LUT_CODE).check(program)

    if program.non_spatial_calls is not None:
        r.fail("REG-1e uncompiled flag",
               f"expected non_spatial_calls=None (never computed) for a Program that never "
               f"reached compile_ast, got {program.non_spatial_calls!r}")
        return
    r.ok("a Program built without compile_ast has non_spatial_calls=None")

    counter, restore = _spy_on_reads_walk()
    try:
        got = I._consensus_extent(dict(bindings), program)
        walks = counter["n"]
    finally:
        restore()

    if walks < 1:
        r.fail("REG-1e uncompiled walk count",
               "a Program with non_spatial_calls=None skipped the walk -- None must mean "
               "'unknown', never 'safe to skip'")
    else:
        r.ok(f"None correctly falls back to the walk ({walks} time(s))")
    if got != (1, 16, 16):
        r.fail("REG-1e uncompiled grid", f"expected (1, 16, 16), got {got}")
    else:
        r.ok(f"grid {got} is correct even without ever calling compile_ast")


def test_reg1e_flag_survives_disk_round_trip(r: SubTestResult):
    print("\n--- REG-1e: non_spatial_calls survives a TEXCache disk save/load round trip ---")
    cache = get_cache()
    img = make_img(1, 8, 8, 4)
    lut = torch.rand(8, 8, 8, 3)
    bindings = {"A": img, "LUT": lut}
    bt = {n: _infer_binding_type(v) for n, v in bindings.items()}
    prog, tm = _compile_ast(_LUT_CODE, bindings)
    if prog.non_spatial_calls is not True:
        r.fail("REG-1e disk round-trip setup", f"expected True, got {prog.non_spatial_calls!r}")
        return

    fp = cache.fingerprint(_LUT_CODE, bt)
    (cache._cache_dir / f"{fp}.pkl").unlink(missing_ok=True)
    try:
        cache._save_to_disk(fp, prog, bt)
        loaded = cache._load_from_disk(fp, bt)
        if loaded is None:
            r.fail("REG-1e disk round-trip", "the freshly-saved .pkl was reported as a miss")
            return
        loaded_prog = loaded[0]
        if loaded_prog.non_spatial_calls is not True:
            r.fail("REG-1e disk round-trip",
                   f"expected the loaded Program's non_spatial_calls to survive as True, "
                   f"got {loaded_prog.non_spatial_calls!r}")
        else:
            r.ok("non_spatial_calls survives a disk save/load round trip (True -> True)")
    finally:
        (cache._cache_dir / f"{fp}.pkl").unlink(missing_ok=True)


def test_reg1e_ast_epoch_folds_non_spatial_set(r: SubTestResult):
    print("\n--- CACHE-4/REG-1e: the AST epoch folds the non-spatial builtin name set ---")
    from TEX_Wrangle import tex_cache as C

    h_with = C._hash_files(C._AST_FILES, b"lang:x", b"nonspatial:apply_lut3d")
    h_without = C._hash_files(C._AST_FILES, b"lang:x")
    if h_with == h_without:
        r.fail("REG-1e epoch fragment",
               "the non-spatial-set fragment does not move the hash -- _AST_EPOCH would "
               "not notice a stdlib leaf changing which builtins are non-spatial")
        return
    r.ok("the non-spatial-set fragment changes the hash")

    cache = C.get_cache()
    code = "@OUT = @A * 0.5;"
    bt = {"A": TEXType.VEC4}
    fp = cache.fingerprint(code, bt)
    (cache._cache_dir / f"{fp}.pkl").unlink(missing_ok=True)
    program = parse_and_split(code, bt)
    prog, tm, refs, asg, params, used = cache.compile_ast(program, bt, source=code)

    real_ast = C._AST_EPOCH
    # The epoch AS IT WOULD HAVE HASHED before this fold landed (the fragment simply absent,
    # not merely a different value) -- the shape of a .pkl an installation cooked the day
    # before this fix shipped.
    from TEX_Wrangle.tex_api import LANGUAGE_VERSION as _live
    pre_fold_ast = C._hash_files(C._AST_FILES, b"lang:" + _live.encode())
    if pre_fold_ast == real_ast:
        r.fail("REG-1e epoch fixture", "the fixture's 'pre-fold' epoch coincides with the "
               "real one -- the live non-spatial set must be non-empty for this to test "
               "anything (apply_lut3d's own declaration should guarantee that)")
        return
    try:
        C._AST_EPOCH = pre_fold_ast
        try:
            cache._save_to_disk(fp, prog, bt)
        finally:
            C._AST_EPOCH = real_ast
        loaded = cache._load_from_disk(fp, bt)
        if loaded is not None:
            r.fail("REG-1e epoch end-to-end",
                   "a .pkl persisted under the pre-fold epoch (missing the nonspatial: "
                   "fragment) was served after the fold -- the exact staleness this fold "
                   "closes")
        else:
            r.ok("a .pkl saved under the pre-fold epoch is a miss after the fold -- old "
                 "on-disk programs recompile fresh instead of serving a stale flag")
    finally:
        (cache._cache_dir / f"{fp}.pkl").unlink(missing_ok=True)
