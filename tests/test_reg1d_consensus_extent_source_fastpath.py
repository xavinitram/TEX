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

SOUNDNESS AUDIT (every path that reaches `_consensus_extent` with a non-empty `source`).
The fast path is sound only if `source` names every call the EXECUTED `Program` can make --
a fused chain splices MULTIPLE stages' code into one `Program`, so a caller that passes only
ONE stage's source text (incomplete) could wrongly report `apply_lut3d` absent when an
upstream, non-terminal stage calls it. Audited every `Interpreter.execute` call site in the
product tree:
  - `tex_engine._run_default`'s interpreter call (`source=("" if ctx.fused_chain else
    ctx.code)`) already blanked correctly.
  - `tex_engine._interp_fallback` (the torch_compile/auto/cuda_graph recovery path, reachable
    for a fused chain too -- `select_tier` admits one when `fused_fp_present`) did NOT: it
    passed `source=ctx.code` unconditionally, where `ctx.code` is the TERMINAL stage's own
    source only. FIXED here to the same guard. `test_reg1d_fused_fallback_source_is_blank`
    proves the real function now computes `""`, not the incomplete terminal source.
  - `.textool` fused-tool manifest cooks (`tex_tool.py`) route through `tex_engine.cook(...,
    chain_payload=...)`, i.e. the SAME `ExecContext`/`_interp_fallback` machinery above --
    covered by the same fix, no separate call site.
  - `tex_api.execute()` (a Tier-3 pinned public surface) never passes `source=` at all, so
    `Interpreter.execute`'s own default (`source=""`) applies -- "unknown", never "skip".
    `test_reg1d_tex_api_execute_omits_source` pins this by reading the call.
  - The checkpoint/boundary cook (`tex_checkpoint.py` -> `tex_engine.cook_stage_list` ->
    `tex_chain.cook_stage_list`) and every M-4 tiled/ROI/batch-strip call in `tex_memory.py`
    (`_cook_whole`/`run_tiled`/`run_roi`/`run_batch_strips`) never pass `source=` either --
    same safe-by-omission default. `test_reg1d_boundary_and_tiling_paths_omit_source` pins
    both call sites by source inspection (grep-shaped, not string-literal-brittle).
  - `test_reg1d_fused_chain_lut_nonterminal_end_to_end` builds a REAL 2-stage fused chain
    (`tex_fusion.compile_fused`) with `apply_lut3d` in the NON-terminal stage 0 and an image
    whose H equals the LUT's N (the collision shape) -- proving the fused, spliced Program's
    grid is the image's, not the LUT's, when walked (`source=""`, the value the real fixed
    `_interp_fallback` now computes), matching the unpressured (non-colliding) cook.
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


# ── Soundness audit: every path that can reach `_consensus_extent` with a non-empty
#    `source` must have that source name every call the EXECUTED Program can make. ──

def test_reg1d_fused_fallback_source_is_blank(r: SubTestResult):
    print("\n--- REG-1d: _interp_fallback blanks source for a fused chain (was ctx.code) ---")
    from TEX_Wrangle import tex_engine as E

    class _SpyInterp:
        def __init__(self):
            self.calls = []

        def execute(self, program, bindings, type_map, **kw):
            self.calls.append(kw)
            return {"OUT": torch.zeros(1, 1, 1, 1)}

    spy = _SpyInterp()
    orig_get_interp = E._get_interpreter
    E._get_interpreter = lambda: spy
    try:
        # A fused-chain ExecContext: ctx.code stands for "the terminal stage's OWN source",
        # deliberately NOT the string the fused Program was actually built from -- exactly
        # the shape a real fused cook has (see tex_fusion.py: only the terminal calls
        # compile_fused with its own code; the spec carries the other stages separately).
        ctx = E.ExecContext(program=None, bindings={}, type_map={}, device="cpu",
                            code="@OUT = @X;", latent_channel_count=0,
                            output_names=["OUT"], used_builtins=None,
                            eff_precision="fp32", fused_chain=True)
        E._interp_fallback(ctx, reset_dynamo=False, pass_precision=False)
    finally:
        E._get_interpreter = orig_get_interp

    got = spy.calls[0].get("source") if spy.calls else "<not called>"
    if got != "":
        r.fail("REG-1d fused fallback source",
               f"_interp_fallback passed source={got!r} for a fused chain -- expected '' "
               f"(unknown), never the terminal-only ctx.code, which cannot name a call an "
               f"upstream non-terminal stage makes")
    else:
        r.ok("_interp_fallback blanks source to '' for a fused chain")

    # Control: a NON-fused cook's fallback still forwards the real (complete) source --
    # this fix must not blank it for the ordinary case.
    spy2 = _SpyInterp()
    E._get_interpreter = lambda: spy2
    try:
        ctx2 = E.ExecContext(program=None, bindings={}, type_map={}, device="cpu",
                             code="@OUT = @A * 2.0;", latent_channel_count=0,
                             output_names=["OUT"], used_builtins=None,
                             eff_precision="fp32", fused_chain=False)
        E._interp_fallback(ctx2, reset_dynamo=False, pass_precision=False)
    finally:
        E._get_interpreter = orig_get_interp

    got2 = spy2.calls[0].get("source") if spy2.calls else "<not called>"
    if got2 != "@OUT = @A * 2.0;":
        r.fail("REG-1d non-fused fallback source",
               f"expected the real source forwarded for a non-fused cook, got {got2!r}")
    else:
        r.ok("a non-fused cook's fallback still forwards its real, complete source")


def test_reg1d_tex_api_execute_omits_source(r: SubTestResult):
    print("\n--- REG-1d: tex_api.execute() never passes source -- defaults to '' (walk) ---")
    from TEX_Wrangle import tex_api

    captured = {}
    orig_execute = Interpreter.execute

    def spy_execute(self, program, bindings, type_map, **kw):
        captured.update(kw)
        return orig_execute(self, program, bindings, type_map, **kw)

    Interpreter.execute = spy_execute
    try:
        code = "@OUT = @A * 2.0;"
        img = make_img(1, 4, 4, 4)
        prog = tex_api.compile(code, {"A": TEXType.VEC4})
        tex_api.execute(prog, {"A": img})
    finally:
        Interpreter.execute = orig_execute

    # Either the kwarg is absent (Interpreter.execute's own default "" then applies) or it
    # is explicitly "" -- both mean "unknown, always walk", never a partial source.
    got = captured.get("source", "")
    if got:
        r.fail("REG-1d tex_api.execute source",
               f"tex_api.execute() passed a non-empty source ({got!r}) -- if this ever "
               f"becomes a PARTIAL source (e.g. a future fused surface), the fast path "
               f"would need it blanked the same way _interp_fallback now is")
    else:
        r.ok("tex_api.execute() passes no (or empty) source -- the fast path always "
             "falls back to the walk on this surface")


def test_reg1d_boundary_and_tiling_paths_omit_source(r: SubTestResult):
    print("\n--- REG-1d: the checkpoint/boundary cook and M-4 tiling never pass source ---")
    import inspect
    from TEX_Wrangle import tex_chain, tex_memory

    # `cook_stage_list` (tex_checkpoint.py's real cook path, re-exported as
    # tex_engine.cook_stage_list) and every M-4 tiled/ROI/batch-strip execute() call in
    # tex_memory.py must never spell `source=` -- Interpreter.execute's own default ("")
    # then means "unknown, always walk", the same safe answer as an explicit "".
    sources = {
        "tex_chain.cook_stage_list": inspect.getsource(tex_chain.cook_stage_list),
        "tex_memory._cook_whole": inspect.getsource(tex_memory._cook_whole),
        "tex_memory.run_tiled": inspect.getsource(tex_memory.run_tiled),
        "tex_memory.run_roi": inspect.getsource(tex_memory.run_roi),
        "tex_memory.run_batch_strips": inspect.getsource(tex_memory.run_batch_strips),
    }
    bad = sorted(name for name, src in sources.items() if "source=" in src)
    if bad:
        r.fail("REG-1d boundary/tiling source audit",
               f"function(s) now spell `source=` in an execute() call and must be audited "
               f"for completeness before this fast path can trust it: {', '.join(bad)}")
    else:
        r.ok(f"none of {len(sources)} checkpoint/boundary/tiling call sites pass `source=` "
             f"-- all fall back to the walk")


def test_reg1d_fused_chain_lut_nonterminal_end_to_end(r: SubTestResult):
    print("\n--- REG-1d: LUT in a non-terminal fused stage + a size collision under the fix ---")
    from TEX_Wrangle import tex_fusion as FUS

    N = 8
    img = make_img(1, N, N, 4)                     # H equals the LUT's N -- the collision shape
    lut = torch.rand(N, N, N, 3)
    terminal_code = "@OUT = @X;"                    # never mentions apply_lut3d
    stages = [
        {"code": "@OUT = vec4(apply_lut3d(@IMG.rgb, @LUT), 1.0);",
         "chain_input": None, "bindings": {"IMG": img, "LUT": lut}},
        {"code": terminal_code, "chain_input": "X", "bindings": {}},
    ]
    try:
        prog, tm, refs, asg, params, used, merged = FUS.compile_fused(stages, _infer_binding_type)
    except Exception as e:
        r.fail("REG-1d fused chain setup", f"{type(e).__name__}: {e}")
        return

    # The value the REAL, fixed _interp_fallback now computes for this shape is "" (proved
    # directly above) -- exercise `_consensus_extent` with exactly that, never the
    # incomplete terminal-only source, and confirm the grid is the image's.
    got = I._consensus_extent(dict(merged), prog, source="")
    if got != (1, N, N):
        r.fail("REG-1d fused chain grid",
               f"expected the image's grid (1, {N}, {N}), got {got} -- the LUT's own shape "
               f"leaked into a fused-chain cook")
        return
    r.ok(f"fused chain grid {got} is the image's, not the LUT's, under the collision shape")

    # And the terminal-only source (what the OLD, buggy _interp_fallback passed) really is
    # incapable of proving the exclusion unnecessary -- confirms the bug this fix closes was
    # real, not hypothetical: it names no non_spatial-arg function at all.
    from TEX_Wrangle.tex_runtime.stdlib_registry import non_spatial_args_by_name
    if any(n in terminal_code for n in non_spatial_args_by_name()):
        r.fail("REG-1d fused chain terminal source",
               "the terminal stage's own source unexpectedly mentions a non-spatial-arg "
               "function -- this test's premise (an incomplete source) no longer holds")
    else:
        r.ok("the terminal-only source names no non-spatial-arg function -- confirms the "
             "pre-fix bug was reachable, not merely hypothetical")
