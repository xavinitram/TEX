"""v0.31 — ANIM-1: the animated-parameter contract.

The one item in the release that is pure contract. It writes down, tests and freezes a
guarantee the implementation already provides:

    A `$param` value is a COOK-TIME BINDING. Changing it never recompiles, never
    re-optimizes, never re-emits codegen, never recaptures a CUDA graph, and never changes
    the program's cache identity.

Hosts build keyframing and timelines on that. Today it is true by accident of three separate
decisions — program fingerprints hash code + binding TYPES (`tex_cache.fingerprint`), the
graph-capture key holds param names/shapes/dtypes but not values (`graphed._capture_key`),
and codegen reads params through `_bind` at call time — and nothing tests it as one property.
An innocent change to any of the three would make an animated slider recompile every frame,
and the only symptom would be "the timeline got slow".

WHAT MAKES THIS NON-VACUOUS. Every sweep is measured by spies on the three mechanisms
(`TEXCache.compile_ast`, `compiled._try_codegen`, `GraphedProgram.capture`), and a NEGATIVE
CONTROL sweeps the CODE instead of the param and asserts the same spies DO fire. A test that
can only pass is not a test.

Shapes (roadmap §10.4): CANARY for the guarantee, with the negative control as its
never-vacuous half.
"""
from helpers import *

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_cache import TEXCache, get_cache

_N = 12          # the sweep length: "N values -> N cooks, 0 of everything else"


class _Spies:
    """Count the three things a param change must never cause. Patched at the module/class
    level rather than sampled from cache sizes: a cache that evicts and re-fills would net to
    zero growth while having recompiled every frame, which is precisely the bug."""

    def __init__(self):
        self.compiles = 0
        self.emissions = 0
        self.captures = 0
        self._base = (0, 0, 0)

    def __enter__(self):
        from TEX_Wrangle.tex_runtime import compiled as _C, graphed as _G
        self._C, self._G = _C, _G
        self._orig = (TEXCache.compile_ast, _C._try_codegen, _G.GraphedProgram.capture)
        spy = self

        def compile_ast(self_, *a, **k):
            spy.compiles += 1
            return spy._orig[0](self_, *a, **k)

        def try_codegen(*a, **k):
            spy.emissions += 1
            return spy._orig[1](*a, **k)

        def capture(self_, *a, **k):
            spy.captures += 1
            return spy._orig[2](self_, *a, **k)

        TEXCache.compile_ast = compile_ast
        _C._try_codegen = try_codegen
        _G.GraphedProgram.capture = capture
        return self

    def __exit__(self, *exc):
        TEXCache.compile_ast, self._C._try_codegen, self._G.GraphedProgram.capture = self._orig
        return False

    def mark(self):
        """Everything so far was the COLD cook, which is allowed to do all three."""
        self._base = (self.compiles, self.emissions, self.captures)

    @property
    def since_mark(self) -> tuple:
        return (self.compiles - self._base[0], self.emissions - self._base[1],
                self.captures - self._base[2])

    @property
    def totals(self) -> tuple:
        """Counts over the WHOLE block, cold cook included. `since_mark` is what the contract
        asserts (zero after the cold cook); this is what proves the spies are wired at all."""
        return (self.compiles, self.emissions, self.captures)


def _sweep(r, label, code, binds, pname, values, **cook_kw):
    """Cook `code` once cold, then once per remaining value of `$pname`, and assert the
    sweep caused no compile, no codegen emission, no graph capture and no new fingerprint."""
    cache = get_cache()
    with _Spies() as spy:
        tex_engine.cook(code, {**binds, pname: values[0]}, **cook_kw)
        spy.mark()
        fps_before = set(cache._memory.keys())
        for v in values[1:]:
            tex_engine.cook(code, {**binds, pname: v}, **cook_kw)
        new_fps = set(cache._memory.keys()) - fps_before
    c, e, g = spy.since_mark
    cooks = len(values)
    ok = (c == 0 and e == 0 and g == 0 and not new_fps)
    r.ok(f"{label}: {cooks} cooks, 0 compiles / 0 emissions / 0 captures / 0 new fingerprints") \
        if ok else \
        r.fail(f"ANIM-1 {label}",
               f"cooks={cooks} compiles={c} emissions={e} captures={g} new_fps={len(new_fps)}")


def test_v031_anim_the_spies_are_live(r: SubTestResult):
    """MUTATION GUARD, and it earned its place: zeroing the emission and capture counters used
    to leave all 13 rows passing, because every assertion was of the form "this counter is 0"
    and nothing ever showed the counter could be non-zero. A harness that cannot fail does not
    protect the contract it is pointed at — and the contract it was pointed at turned out to be
    false, undetected, for exactly that reason.

    So: each of the three spies is shown counting, on a cook chosen to trigger it."""
    print("\n--- v0.31 ANIM-1: the spies themselves fire ---")
    A = make_img(1, 32, 32, 4, seed=31)

    # compiles + emissions: a cold, never-before-seen program on the `auto` tier, which measures
    # the codegen baseline before it considers a background compile.
    with cold_engine_state(warm=False), _Spies() as spy:
        tex_engine.cook("@OUT = vec4(@A.rgb * 1.0625 + vec3(0.0078), 1.0);", {"A": A},
                        device_mode="cpu", compile_mode="auto")
        c, e, _g = spy.totals
    r.ok(f"the compile spy counts ({c})") if c >= 1 else \
        r.fail("ANIM-1 spy", "TEXCache.compile_ast was never observed")
    r.ok(f"the codegen-emission spy counts ({e})") if e >= 1 else \
        r.fail("ANIM-1 spy", "compiled._try_codegen was never observed — the counter is inert")

    # captures: needs CUDA and the graph tier.
    if not torch.cuda.is_available():
        r.skip("the capture spy", "no CUDA on this box")
        return
    with cold_engine_state(warm=False), _Spies() as spy:
        tex_engine.cook("@OUT = vec4(@A.rgb * 1.0625 + vec3(0.0078), 1.0);", {"A": A.cuda()},
                        device_mode="cuda", compile_mode="cuda_graph")
        _c, _e, g = spy.totals
    r.ok(f"the graph-capture spy counts ({g})") if g >= 1 else \
        r.fail("ANIM-1 spy", "GraphedProgram.capture was never observed — the counter is inert")


def test_v031_anim_param_sweep_never_recompiles(r: SubTestResult):
    print("\n--- v0.31 ANIM-1: a $param sweep costs N cooks and nothing else ---")
    A = make_img(1, 64, 64, 4, seed=31)
    code = "@OUT = vec4(@A.rgb * $strength + vec3($lift), 1.0);"
    vals = [0.1 + i * 0.07 for i in range(_N)]

    _sweep(r, "interpreter tier (cpu)", code, {"A": A, "lift": 0.02}, "strength", vals,
           device_mode="cpu", precision="fp32")

    if torch.cuda.is_available():
        Ac = A.cuda()
        _sweep(r, "interpreter tier (cuda)", code, {"A": Ac, "lift": 0.02}, "strength", vals,
               device_mode="cuda", precision="fp32")
        _sweep(r, "cuda_graph tier", code, {"A": Ac, "lift": 0.02}, "strength", vals,
               device_mode="cuda", compile_mode="cuda_graph", precision="fp32")
    else:
        # A real skip, not an `ok`. `SubTestResult.skip` is unconditional in helpers, and the
        # `hasattr` fallback this used to carry turned a genuinely-unrun row into a PASS on
        # every CPU-only box — exactly what skip() was added to stop.
        r.skip("cuda tiers", "no CUDA on this box")

    # The codegen/auto tier. `auto` measures the codegen baseline before it ever considers a
    # background compile, so a sweep exercises the emitted-fn-reads-params-at-call-time half.
    _sweep(r, "auto tier (cpu)", code, {"A": A, "lift": 0.02}, "strength", vals,
           device_mode="cpu", compile_mode="auto", precision="fp32")


# The negative control NEEDS a cold cache and the sweeps do not, which is worth saying
# plainly: a cache probe checks memory and then DISK, so on the second run of the suite the
# "different" programs below are already on disk from the first run and never reach the
# compiler. That is correct engine behaviour and a broken test — the assertion is about the
# COMPILER, so it has to start from a cache that has never seen these programs. (It is also
# the only way this test could have rotted silently: it would have passed forever on a
# machine whose cache happened to be cold.) `cold_engine_state` is the shared helpers fixture.


def test_v031_anim_negative_control_code_edit_does_recompile(r: SubTestResult):
    """The half that stops the sweeps above from passing vacuously."""
    print("\n--- v0.31 ANIM-1: negative control — a CODE edit DOES recompile ---")
    A = make_img(1, 64, 64, 4, seed=31)
    with cold_engine_state(warm=False), _Spies() as spy:
        tex_engine.cook("@OUT = vec4(@A.rgb * 1.0, 1.0);", {"A": A}, device_mode="cpu")
        spy.mark()
        for i in range(1, 4):
            # A genuinely different program each time (not a param), so the cache must miss.
            tex_engine.cook(f"@OUT = vec4(@A.rgb * {1.0 + i}, 1.0);", {"A": A}, device_mode="cpu")
        c, _e, _g = spy.since_mark
    r.ok(f"3 code edits -> {c} compiles (the spies are live)") if c == 3 else \
        r.fail("ANIM-1 negative control", f"a code edit produced {c} compiles, expected 3")

    # And the second axis in the fingerprint: a BINDING's type. The same code wired to a VEC3
    # image and to a VEC4 image is two programs.
    #
    # This row is deliberately about `@A` and not about `$k`, and the distinction is the one
    # thing about the contract that is easy to get wrong: a PARAM's type comes from its
    # DECLARATION in the code (`f$k` / `v3$k`), never from the bound value —
    # `infer_binding_type` maps a python float and a 3-float list to FLOAT alike. So "change a
    # param's type" is not a thing a host can do at cook time at all; it is a code edit, and
    # the row above already covers those. LANGUAGE.md §5.1 states it that way for the same
    # reason. (An earlier draft of this test asserted the param axis and failed — which is
    # what a negative control is for.)
    axis = "@OUT = vec4(@A.rgb, 1.0);"
    A3 = make_img(1, 64, 64, 3, seed=7)
    with cold_engine_state(warm=False), _Spies() as spy:
        tex_engine.cook(axis, {"A": A3}, device_mode="cpu")
        spy.mark()
        tex_engine.cook(axis, {"A": A}, device_mode="cpu")        # the same code, VEC4 now
        c, _e, _g = spy.since_mark
    r.ok("a BINDING's type change does recompile (types are in the fingerprint, values are not)") \
        if c >= 1 else \
        r.fail("ANIM-1 type axis", "VEC3 -> VEC4 on the same code did not recompile")


def test_v031_anim_int_crossing_ramp(r: SubTestResult):
    """The row that catches ANIM-1's real failure mode, and the one the original sweeps missed
    because every value they swept was a float.

    A host animating `$k` sends whatever JSON gives it, and JSON gives `2` for a whole number.
    `infer_binding_type` maps 2 → INT and 2.0 → FLOAT, so before the fix a ramp CROSSING a whole
    number minted a second fingerprint — a recompile, a codegen re-emission and a CUDA-graph
    recapture, mid-scrub. Three spellings of "one" (omitted / 1.0 / 1) gave three identities.
    `examples/host_demo.py`, the release's own consumer proof, serves params over HTTP/JSON, so
    this was the shipping configuration and not a corner.

    Asserted at the FINGERPRINT, not through the spies: identity is the guarantee, and a spy
    can only see a consequence of losing it."""
    print("\n--- v0.31 ANIM-1: a ramp through a whole number is ONE program ---")
    A = make_img(1, 32, 32, 4, seed=31)
    code = "@OUT = vec4(@A.rgb * $k, 1.0);"

    fps = {tex_engine.prepare(code, {"A": A.clone(), "k": v}, device_mode="cpu").ctx.fp
           for v in [1, 1.5, 2, 2.5, 3, 0]}
    r.ok("an int-crossing ramp keeps ONE fingerprint") if len(fps) == 1 else \
        r.fail("ANIM-1 int ramp", f"{len(fps)} fingerprints over [1, 1.5, 2, 2.5, 3, 0]")

    declared = "f$k = 1.0;\n@OUT = vec4(@A.rgb * $k, 1.0);"
    trio = {tex_engine.prepare(declared, b, device_mode="cpu").ctx.fp for b in
            ({"A": A.clone()}, {"A": A.clone(), "k": 1.0}, {"A": A.clone(), "k": 1})}
    r.ok("omitted / 1.0 / 1 are the same program") if len(trio) == 1 else \
        r.fail("ANIM-1 int ramp", f"{len(trio)} fingerprints for three spellings of one")

    # A FUSED chain keys on `_fused_fp`, which folds per-stage (code, types) — the same defect
    # lived there, and the original fused test swept only floats.
    import copy
    spec = {"schema": 1,
            "stages": [{"code": "@OUT = vec4(@IN.rgb * 1.15, 1.0);", "image_input": "IN",
                        "params": {}},
                       {"code": "@OUT = vec4(@IN.rgb * $mid, 1.0);", "image_input": "IN",
                        "params": {"mid": 1.0}}],
            "terminal_image_input": "IN"}
    term = "@OUT = vec4(@IN.rgb + vec3(0.01), 1.0);"
    ffps = set()
    for v in [1, 1.5, 2, 2.5]:
        s = copy.deepcopy(spec)
        s["stages"][1]["params"]["mid"] = v
        ffps.add(tex_engine.prepare(term, {"IN": A.clone()}, chain_payload=s,
                                    compile_mode="cuda_graph", device_mode="cpu").ctx.fused_fp)
    r.ok("a fused chain's mid-stage param ramp keeps ONE fused fingerprint") \
        if len(ffps) == 1 else \
        r.fail("ANIM-1 int ramp (fused)", f"{len(ffps)} fused fingerprints")

    # NEVER-SEVER: the fix must not collapse the `@`-WIRE type axis. A scalar wired to `@k`
    # legitimately keys INT apart from FLOAT — a blanket INT→FLOAT collapse would have passed
    # every row above and silently served one compiled program for two different type maps.
    wired = {tex_engine.prepare("@OUT = vec4(@A.rgb * @k, 1.0);", {"A": A.clone(), "k": v},
                                device_mode="cpu").ctx.fp for v in (2, 2.0)}
    r.ok("a scalar-wired @k still keys INT apart from FLOAT") if len(wired) == 2 else \
        r.fail("ANIM-1 int ramp", "the fix collapsed the @-wire type axis")


def test_v031_anim_param_type_matrix(r: SubTestResult):
    """Every param type a host can keyframe, swept. A vec/color param is the interesting one:
    it tensorizes to a [1,1,1,C] static buffer for graph capture, so the guarantee has to hold
    for a value that is not a scalar."""
    print("\n--- v0.31 ANIM-1: the param-type matrix ---")
    A = make_img(1, 48, 48, 4, seed=31)

    _sweep(r, "float param", "@OUT = vec4(@A.rgb * $f, 1.0);", {"A": A}, "f",
           [0.5 + i * 0.1 for i in range(_N)], device_mode="cpu")

    _sweep(r, "int param", "@OUT = vec4(@A.rgb * float($n), 1.0);", {"A": A}, "n",
           list(range(1, 1 + _N)), device_mode="cpu")

    _sweep(r, "vec3 (colour) param", "@OUT = vec4(@A.rgb * $tint, 1.0);", {"A": A}, "tint",
           [[0.2 + i * 0.05, 0.5, 1.0 - i * 0.05] for i in range(_N)], device_mode="cpu")

    # A string param needs its `s$` declaration — without one the checker types `$mode` FLOAT
    # and `len()` rejects it. The declaration is part of the program, so it is inside the
    # fingerprint; the VALUE swept over it is not, which is the thing being pinned.
    _sweep(r, "string param",
           's$mode = "aa";\n@OUT = vec4(vec3(len($mode) * 0.05), 1.0);', {"A": A}, "mode",
           ["a" * (i + 1) for i in range(_N)], device_mode="cpu")


def test_v031_anim_fused_chain_param_sweep(r: SubTestResult):
    """A mid-chain stage's param. Fusion splices the chain into ONE program keyed by
    `_fused_fp`, which folds per-stage (code, binding TYPES) — so the same guarantee has to
    survive the splice, and a host may keyframe any stage of a collapsed chain."""
    print("\n--- v0.31 ANIM-1: a param of a mid-chain stage in a fused chain ---")
    A = make_img(1, 48, 48, 4, seed=31)
    payload = {"schema": 1,
               "stages": [{"code": "@OUT = vec4(@IN.rgb * 1.15, 1.0);", "image_input": "IN",
                           "params": {}},
                          {"code": "@OUT = vec4(@IN.rgb * $mid, 1.0);", "image_input": "IN",
                           "params": {"mid": 1.0}}],
               "terminal_image_input": "IN"}
    term = "@OUT = vec4(@IN.rgb + vec3(0.01), 1.0);"

    cache = get_cache()
    with _Spies() as spy:
        import copy
        for i in range(_N):
            spec = copy.deepcopy(payload)
            spec["stages"][1]["params"]["mid"] = 0.5 + i * 0.05
            tex_engine.cook(term, {"IN": A}, chain_payload=spec, device_mode="cpu")
            if i == 0:
                spy.mark()
                fps_before = set(cache._memory.keys())
        new_fps = set(cache._memory.keys()) - fps_before
    c, e, g = spy.since_mark
    r.ok(f"a fused mid-chain param sweeps in {_N} cooks with 0 recompiles") \
        if c == 0 and e == 0 and g == 0 and not new_fps else \
        r.fail("ANIM-1 fused", f"compiles={c} emissions={e} captures={g} new_fps={len(new_fps)}")


def test_v031_anim_textool_promoted_param_sweep(r: SubTestResult):
    """A `.textool`'s PROMOTED param — the surface a published tool exposes to a host's
    timeline. It routes through `tex_tool.cook_tool`, which resolves defaults and assembles
    the fused stages, so it is a genuinely different call path from the three above."""
    print("\n--- v0.31 ANIM-1: a .textool promoted-param sweep ---")
    from TEX_Wrangle import tex_tool
    A = make_img(1, 48, 48, 3, seed=31)
    # A SHIPPED tool, not a synthetic manifest: the point is to sweep the surface a host's
    # timeline actually binds to, through the real load → resolve-defaults → assemble path.
    root = Path(__file__).resolve().parents[1]
    manifest = tex_tool.load_tool(str(root / "stock" / "grade.textool"))
    cache = get_cache()
    with _Spies() as spy:
        for i in range(_N):
            tex_tool.cook_tool(manifest, {"image": A}, {"gamma": 0.6 + i * 0.08},
                               device_mode="cpu")
            if i == 0:
                spy.mark()
                fps_before = set(cache._memory.keys())
        new_fps = set(cache._memory.keys()) - fps_before
    c, e, g = spy.since_mark
    r.ok(f"the stock Grade tool's promoted $gamma sweeps in {_N} cooks with 0 recompiles") \
        if c == 0 and e == 0 and g == 0 and not new_fps else \
        r.fail("ANIM-1 textool", f"compiles={c} emissions={e} captures={g} new_fps={len(new_fps)}")


def test_v031_anim_contract_is_documented(r: SubTestResult):
    """DERIVATION: the guarantee is only useful if a host can find it. Doc 40 §4.1's
    definition-of-done item (3) is "LANGUAGE.md section merged"; this is that item's test, so
    the doc and the behaviour cannot drift apart silently."""
    print("\n--- v0.31 ANIM-1: the guarantee is written down ---")
    root = Path(__file__).resolve().parents[1]
    txt = (root / "LANGUAGE.md").read_text(encoding="utf-8", errors="ignore")
    need = ["cook-time binding", "never recompiles", "ANIM-1"]
    missing = [n for n in need if n.lower() not in txt.lower()]
    r.ok("LANGUAGE.md states the animated-parameter guarantee") if not missing else \
        r.fail("ANIM-1 docs", f"LANGUAGE.md is missing: {missing}")
