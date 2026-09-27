"""SCALECX-49 — resolution scale threaded through the compiled (torch_compile/auto) and
graphed (cuda_graph) execution tiers.

Per the resolution-scale design (SCALE-COMPILED-48): a scale-active cook (`ctx.scale is
not None`) used to be
bounced, UNCONDITIONALLY, to the plain interpreter the instant `tier_id != "default"`
(`tex_engine_tiers._run_tier`'s old bypass) -- none of `torch_compile`/`auto`/`cuda_graph`
threaded a runtime scale multiplier through their compiled/captured code at all. This ask
makes all three scale-aware:

  - `execute_compiled`/`run_auto` (tex_runtime/compiled.py) key their compiled ARTIFACT
    cache (`_compiled_cache`, and `autotier.make_key`'s own bucket) by an EXPLICIT,
    trailing `scale` component -- appended only when `scale is not None`, so a `scale=None`
    cook keys exactly as before this ask (invariant 7).
  - `run_graphed` (tex_runtime/graphed.py) keys its captured-graph cache (`_capture_key`)
    the identical way, for the identical reason: a `pixel_args=`-tagged builtin's scaled
    radius is a SHAPE baked at capture/compile time (`ceil(3*sigma*scale)` for
    `gauss_blur`, a truncated int for `erode`/`dilate`), so two different scale values on
    the SAME canvas/precision/device genuinely need two different artifacts/captures --
    canvas shape alone does not distinguish them.

Never per call: a repeated request at an already-seen scale value is a cache hit against
the SAME artifact/capture, never a rebuild -- "bounded" means bounded by the number of
DISTINCT scale values a session actually requests, exactly like a distinct canvas shape or
precision already gets its own entry today, never a recompile/recapture per call.

Scope: ROI stays entirely out of the compiled/graphed tiers (`tex_roi.roi_eligibility`
declines ROI outright once `tier_id != "default"`, and separately the instant
`scale is not None` -- both checks run before this ask's new dispatch is ever reached, so
this ask cannot newly arm ROI on a compiled/graphed tier; see test_tierq48_agreement.py).

Hardware honesty: `cuda_graph`'s capture/replay and `torch.compile`'s actual Inductor
lowering both need real CUDA (and, for `torch_compile`, a working backend toolchain this
box does not have -- AGENTS.md's workstation profile: no compiler toolchain). Every test
below that needs either is CPU-runnable by construction (it exercises the KEY/CACHE
construction and the parameter-threading this ask actually changes, with `_try_compile`
mocked out where a real compile would otherwise be required) or explicitly `r.skip()`s
with the reason, never fabricating a result it did not produce.
"""
import torch

from helpers import *
from TEX_Wrangle import tex_engine
from TEX_Wrangle import tex_engine_tiers as _tiers
from TEX_Wrangle import tex_roi as _tex_roi
from TEX_Wrangle.tex_runtime import compiled as _compiled
from TEX_Wrangle.tex_runtime import graphed as _graphed
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_cache import parse_and_split

_CUDA = torch.cuda.is_available()

# A gauss_blur program -- the simplest `pixel_args=`-tagged shape whose resolved kernel
# radius is scale-dependent (docs/resolution-scale.md's own table).
_BLUR_CODE = "@OUT = gauss_blur(@A, 6.0);\n"
_BLUR_BT = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}


def _prog():
    prog = parse_and_split(_BLUR_CODE, _BLUR_BT)
    TypeChecker(binding_types=_BLUR_BT, source=_BLUR_CODE).check(prog)
    return prog


# ── _capture_key (graphed.py): unit-level red-first proof ────────────────────────────
#
# The mechanism the whole cuda_graph half of this ask rests on is pure Python (no CUDA
# needed to construct or compare a key tuple), so the mismatch this ask fixes is provable
# on CPU by comparing the OLD key shape (no scale component at all -- what every call site
# used before this ask, and what `_capture_key` still produces when a caller omits
# `scale=`) against the NEW one.

def test_scalecx49_capture_key_mismatch_before_fix(r: SubTestResult):
    """Red-first: reconstruct the PRE-FIX key (the base-sha `_capture_key`, which took no
    `scale` argument at all) for two cooks of the SAME program/canvas/precision/device but
    DIFFERENT scale values, and show they collide -- the exact bug that would let a capture
    taken at one scale replay for a request at another. Then show the FIXED `_capture_key`
    (this file's own import) distinguishes them."""
    import subprocess
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # TEX_Wrangle
    base_src = subprocess.run(
        ["git", "show", "7477a93:tex_runtime/graphed.py"],
        cwd=repo_root, capture_output=True, text=True, check=True).stdout
    ns = {"__name__": "old_graphed_probe"}
    # Extract and exec ONLY the pre-fix `_capture_key` function body -- not the whole
    # module (which imports `.host`/`.interpreter` by relative package syntax and cannot
    # exec standalone). Isolates exactly the function this test is about.
    start = base_src.index("def _capture_key(")
    end = base_src.index("\n\n\n", start)
    exec(compile("import torch\n" + base_src[start:end], "<old _capture_key>", "exec"), ns)
    old_capture_key = ns["_capture_key"]

    bindings = {"A": make_img(1, 8, 8, 3, seed=1)}
    old_key_half = old_capture_key("fp", torch.device("cuda:0"), "fp32", bindings, None, 0)
    old_key_quarter = old_capture_key("fp", torch.device("cuda:0"), "fp32", bindings, None, 0)
    if old_key_half == old_key_quarter:
        r.ok("pre-fix _capture_key collides across scale values (no scale component at "
             "all) -- confirmed reproducible at base sha 7477a93")
    else:
        r.fail("scalecx49 red-first premise",
               "pre-fix _capture_key did not collide -- this test's premise is stale")

    new_key_half = _graphed._capture_key("fp", torch.device("cuda:0"), "fp32", bindings, None, 0,
                                         scale=0.5)
    new_key_quarter = _graphed._capture_key("fp", torch.device("cuda:0"), "fp32", bindings, None, 0,
                                            scale=0.25)
    if new_key_half != new_key_quarter:
        r.ok("fixed _capture_key distinguishes scale=0.5 from scale=0.25 on the same "
             "program/canvas/precision/device")
    else:
        r.fail("scalecx49 fix", "fixed _capture_key still collides across scale values")


def test_scalecx49_capture_key_scale_none_byte_identical(r: SubTestResult):
    """Invariant 7: a `scale=None` cook's `_capture_key` is shape- and value-identical to
    a call that omits `scale` entirely (every call site before this ask)."""
    bindings = {"A": make_img(1, 8, 8, 3, seed=2)}
    k_omitted = _graphed._capture_key("fp", torch.device("cuda:0"), "fp32", bindings, None, 0)
    k_explicit_none = _graphed._capture_key("fp", torch.device("cuda:0"), "fp32", bindings, None, 0,
                                            scale=None)
    if k_omitted == k_explicit_none and len(k_omitted) == 7:
        r.ok("scale=None keys identically to omitting scale (7-tuple, unchanged shape)")
    else:
        r.fail("scalecx49 invariant 7", f"{k_omitted} != {k_explicit_none}")


def test_scalecx49_capture_key_scale_1_0_normalises_to_none(r: SubTestResult):
    """FIX-SCALECX X2 (B2#3): `scale=1.0` is the documented byte-identical-VALUE case
    (invariant 7's own language) -- unlike a genuinely active value, it must key IDENTICALLY
    to `scale=None`/omitted, not mint its own 8-tuple bucket. Red at 32f6917: `_capture_key`
    treated `scale is not None` (true for 1.0) as sufficient to append a component."""
    bindings = {"A": make_img(1, 8, 8, 3, seed=10)}
    k_none = _graphed._capture_key("fp", torch.device("cuda:0"), "fp32", bindings, None, 0)
    k_one = _graphed._capture_key("fp", torch.device("cuda:0"), "fp32", bindings, None, 0,
                                  scale=1.0)
    if k_none == k_one and len(k_one) == 7:
        r.ok(f"scale=1.0 keys identically to scale=None (7-tuple): {k_one}")
    else:
        r.fail("scalecx49 X2 normalise-1.0", f"scale=None {k_none} != scale=1.0 {k_one}")


def test_scalecx49_blacklist_bounded_across_many_distinct_scale_values(r: SubTestResult):
    """FIX-SCALECX X5 (B4#5): `graphed._blacklist` is a bounded LRU (mirroring compiled.py's
    own `_compile_blacklist`), not a plain unbounded `set` -- SCALECX-49's `scale` component
    widened this key's growth axis (a capturable-but-declining scale-active program now mints
    ONE blacklist entry per DISTINCT scale value seen, not one total), so a long session that
    sweeps many distinct values (a slider drag, a per-frame procedural ramp) must not grow this
    set without limit. Red at 32f6917: `_blacklist` was a plain `set`, `_BLACKLIST_MAX` did not
    exist, and nothing bounded it."""
    real_blacklist = dict(_graphed._blacklist)
    _graphed._blacklist.clear()
    try:
        n = _graphed._BLACKLIST_MAX + 50
        for i in range(n):
            _graphed._blacklist_add(("fp", 0, "fp32", (), (), (), 0, float(i)))
        size = len(_graphed._blacklist)
        oldest_evicted = ("fp", 0, "fp32", (), (), (), 0, 0.0) not in _graphed._blacklist
        newest_kept = ("fp", 0, "fp32", (), (), (), 0, float(n - 1)) in _graphed._blacklist
        if size == _graphed._BLACKLIST_MAX and oldest_evicted and newest_kept:
            r.ok(f"{n} distinct scale-keyed blacklist entries -> bounded at "
                 f"{_graphed._BLACKLIST_MAX} (oldest evicted, newest kept)")
        else:
            r.fail("scalecx49 X5 bounded blacklist",
                  f"size={size} (want {_graphed._BLACKLIST_MAX}), oldest_evicted="
                  f"{oldest_evicted}, newest_kept={newest_kept}")
    finally:
        _graphed._blacklist.clear()
        _graphed._blacklist.update(real_blacklist)


def test_scalecx49_capture_key_bounded_by_distinct_scale_values(r: SubTestResult):
    """Bounded, not per-call: repeated requests at the SAME scale value produce the SAME
    key (a cache hit, not a new capture); only a genuinely different value produces a
    different key."""
    bindings = {"A": make_img(1, 8, 8, 3, seed=3)}
    keys = [_graphed._capture_key("fp", torch.device("cuda:0"), "fp32", bindings, None, 0, scale=s)
           for s in (0.5, 0.25, 0.5, 0.25, 0.5)]
    distinct = set(keys)
    if len(distinct) == 2:
        r.ok(f"5 requests over 2 distinct scale values -> 2 distinct keys (bounded), got "
             f"{len(distinct)}")
    else:
        r.fail("scalecx49 bounded captures", f"expected 2 distinct keys, got {len(distinct)}")


# ── execute_compiled (torch_compile/auto tier) cache-key + threading ─────────────────

def _clear_compiled_state():
    _compiled.clear_compiled_cache()
    _compiled._route_memo.clear()


def test_scalecx49_execute_compiled_cache_key_scale_none_unchanged(r: SubTestResult):
    """Invariant 7: `execute_compiled`'s cache key is the pre-existing 3-tuple
    `(fingerprint, device_type, precision)` when `scale` is `None` -- the exact shape
    every scale=None cook (every ordinary ComfyUI cook) already keyed by."""
    _clear_compiled_state()
    program = _prog()
    calls = {"n": 0}

    def _fake_try_compile(device_type, program, type_map, used_builtins=None,
                          precision="fp32", fingerprint=None):
        calls["n"] += 1
        def _fake_fn(program, bindings, type_map, device, latent_channel_count=0,
                    output_names=None, scale=None):
            return bindings.get("A")
        return _fake_fn, None

    real_try_compile = _compiled._try_compile
    real_count_ops = _compiled._count_tensor_ops
    real_max_depth = _compiled._max_loop_depth
    _compiled._try_compile = _fake_try_compile
    _compiled._count_tensor_ops = lambda p: 100     # force past the trivial-program gate
    _compiled._max_loop_depth = lambda p: 0         # force past the deep-loop gate
    try:
        bindings = {"A": make_img(1, 8, 8, 3, seed=4)}
        _compiled.execute_compiled(program, bindings, {}, "cpu", "fp_scalecx49_none",
                                   scale=None)
        keys = list(_compiled._compiled_cache.keys())
        if len(keys) == 1 and len(keys[0]) == 3:
            r.ok(f"scale=None cache_key is a 3-tuple: {keys[0]}")
        else:
            r.fail("scalecx49 invariant 7", f"expected one 3-tuple key, got {keys}")
    finally:
        _compiled._try_compile = real_try_compile
        _compiled._count_tensor_ops = real_count_ops
        _compiled._max_loop_depth = real_max_depth
        _clear_compiled_state()


def test_scalecx49_execute_compiled_shares_one_artifact_across_a_scale_sweep(r: SubTestResult):
    """FIX-SCALECX X2 (R3#1, R4#4): `scale` is a RUNTIME INPUT, not a cache-key component --
    a sweep of DISTINCT scale values must compile ONCE (one shared artifact), never once per
    distinct value. Red at 32f6917: this same sweep produced 3 compiles/3 cache entries for
    3 distinct values (0.5, 0.25, None) -- R3#1's own measurement showed that costs a real
    `torch.compile()` wrap+trace per value, 2.9x-5.5x slower than sharing one artifact.
    `_try_compile` is mocked (this box has no reliable torch.compile backend for every CI
    config -- AGENTS.md's workstation profile) so this isolates exactly the caching/threading
    change this ask makes, not Inductor's own behaviour; the real-backend proof is the next
    test below, unmocked."""
    _clear_compiled_state()
    program = _prog()
    calls = []
    seen_scales = []

    def _fake_try_compile(device_type, program, type_map, used_builtins=None,
                          precision="fp32", fingerprint=None):
        calls.append(fingerprint)
        def _fake_fn(program, bindings, type_map, device, latent_channel_count=0,
                    output_names=None, scale=None):
            # Proves `scale` actually reaches the compiled callable (never dropped).
            seen_scales.append(scale)
            return bindings.get("A")
        return _fake_fn, None

    real_try_compile = _compiled._try_compile
    real_count_ops = _compiled._count_tensor_ops
    real_max_depth = _compiled._max_loop_depth
    _compiled._try_compile = _fake_try_compile
    _compiled._count_tensor_ops = lambda p: 100
    _compiled._max_loop_depth = lambda p: 0
    try:
        fp = "fp_scalecx49_sweep"
        for s in (0.5, 0.25, 0.5, 0.25, 0.5, None, 1.0):
            bindings = {"A": make_img(1, 8, 8, 3, seed=5)}
            _compiled.execute_compiled(program, bindings, {}, "cpu", fp, scale=s)
        n_compiles = len(calls)
        n_cache_entries = len(_compiled._compiled_cache)
        if n_compiles == 1 and n_cache_entries == 1:
            r.ok(f"7 calls over 4 distinct scale values (0.5, 0.25, None, 1.0) -> "
                 f"{n_compiles} compile, {n_cache_entries} cached artifact (shared, not "
                 f"per-value)")
        else:
            r.fail("scalecx49 X2 shared artifact",
                  f"expected 1 compile/1 cache entry, got {n_compiles}/{n_cache_entries} "
                  f"(calls={calls})")
        if seen_scales == [0.5, 0.25, 0.5, 0.25, 0.5, None, 1.0]:
            r.ok("scale forwarded correctly to the SHARED compiled callable on every call "
                 "(never dropped, never stale from a cached closure)")
        else:
            r.fail("scalecx49 scale threading", f"compiled callable saw {seen_scales}")
    finally:
        _compiled._try_compile = real_try_compile
        _compiled._count_tensor_ops = real_count_ops
        _compiled._max_loop_depth = real_max_depth
        _clear_compiled_state()


def test_scalecx49_execute_compiled_real_backend_compiles_once_across_scale_sweep(r: SubTestResult):
    """FIX-SCALECX X2's own requirement, unmocked: a program ABOVE `_COMPILE_OP_THRESHOLD`
    that has no non-inlined stdlib call (so `_has_fn_calls` is False and real Inductor
    tracing is actually reached, per FIX-SCALECX X1) -- pure arithmetic, no `pixel_args=`
    builtin at all -- must compile through the REAL backend exactly ONCE across a sweep of
    distinct scale values, not once per value. `scale` reaching the callable is still proven
    the same way the mocked test above does; this test additionally proves the REAL
    `torch.compile()` wrap itself is not repeated."""
    if not _compiled.compile_capability().get(
            "cuda_inductor" if torch.cuda.is_available() else "cpu_inductor"):
        r.skip("scalecx49 X2 real-backend sweep", "no working torch.compile backend on this box")
        return
    _clear_compiled_state()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 20 chained additions: comfortably above _COMPILE_OP_THRESHOLD (8), no stdlib call at
    # all (no _has_fn_calls graph-break), so this reaches real Inductor tracing.
    code = "@OUT = @A" + " + 0.001" * 20 + ";\n"
    bt = {"A": TEXType.VEC4}
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    fp = "fp_scalecx49_real_sweep"
    real_try_compile = _compiled._try_compile
    compiles = {"n": 0}

    def _counting_try_compile(*a, **kw):
        compiles["n"] += 1
        return real_try_compile(*a, **kw)

    _compiled._try_compile = _counting_try_compile
    try:
        for s in (0.5, 0.25, 0.5, 0.125, None):
            A = make_img(1, 16, 16, 4, seed=11)
            if device == "cuda":
                A = A.cuda()
            _compiled.execute_compiled(prog, {"A": A}, tm, device, fp, scale=s)
        if compiles["n"] == 1:
            r.ok(f"5 calls over 4 distinct scale values -> {compiles['n']} real "
                 f"torch.compile() attempt (shared artifact, real backend)")
        else:
            r.fail("scalecx49 X2 real backend sweep",
                  f"expected 1 real _try_compile call, got {compiles['n']}")
    finally:
        _compiled._try_compile = real_try_compile
        _clear_compiled_state()


def test_scalecx49_execute_compiled_parity_cpu(r: SubTestResult):
    """Parity requirement (CPU leg): `execute_compiled`'s OUTPUT at scale=1, 1/2, 1/4, 1/8
    matches the plain interpreter's own scale-active output bit-for-bit (both ultimately
    resolve to the SAME interpreter call on this box, which has no torch.compile backend --
    `_select_backend` returns None, so `execute_compiled` self-declines to `_plain_execute`
    exactly like a scale=None cook without a backend already does). This proves `scale`
    survives `execute_compiled`'s own plumbing intact; it does not exercise Inductor's own
    scale handling, which needs a real backend and is reported separately as owed."""
    program = _prog()
    A = make_img(1, 16, 16, 3, seed=6)
    for s in (1.0, 0.5, 0.25, 0.125):
        compiled_out = _compiled.execute_compiled(
            program, {"A": A.clone()}, {}, "cpu", None, scale=s, time_context=None)
        interp_out = _compiled._plain_execute(
            program, {"A": A.clone()}, {}, "cpu", scale=s, time_context=None)
        maxdiff = (compiled_out - interp_out).abs().max().item()
        if maxdiff < 1e-5:
            r.ok(f"execute_compiled(scale={s}) matches plain interpreter, maxdiff={maxdiff:.2e}")
        else:
            r.fail("scalecx49 parity", f"scale={s} maxdiff={maxdiff:.2e} >= 1e-5")


# ── run_auto (autotier) key ───────────────────────────────────────────────────────────

def test_scalecx49_autotier_make_key_scale_none_unchanged(r: SubTestResult):
    from TEX_Wrangle.tex_runtime import autotier
    k_old = autotier.make_key("fp", "cpu", "fp32", (1, 8, 8, 3))
    k_explicit_none = autotier.make_key("fp", "cpu", "fp32", (1, 8, 8, 3), scale=None)
    k_active = autotier.make_key("fp", "cpu", "fp32", (1, 8, 8, 3), scale=0.5)
    if k_old == k_explicit_none and len(k_old) == 4:
        r.ok(f"autotier key unchanged at scale=None: {k_old}")
    else:
        r.fail("scalecx49 autotier invariant 7", f"{k_old} != {k_explicit_none}")
    if k_active != k_old and len(k_active) == 5:
        r.ok(f"autotier key gains a trailing scale component when active: {k_active}")
    else:
        r.fail("scalecx49 autotier bucketing", f"{k_active} did not extend {k_old}")


def test_scalecx49_autotier_make_key_scale_1_0_normalises_to_none(r: SubTestResult):
    """FIX-SCALECX X2 (B2#3): `scale=1.0` must bucket identically to `scale=None` -- the
    documented byte-identical no-op, not its own distinct 5-tuple bucket."""
    from TEX_Wrangle.tex_runtime import autotier
    k_none = autotier.make_key("fp", "cpu", "fp32", (1, 8, 8, 3))
    k_one = autotier.make_key("fp", "cpu", "fp32", (1, 8, 8, 3), scale=1.0)
    if k_none == k_one and len(k_one) == 4:
        r.ok(f"scale=1.0 buckets identically to scale=None (4-tuple): {k_one}")
    else:
        r.fail("scalecx49 X2 autotier normalise-1.0", f"scale=None {k_none} != scale=1.0 {k_one}")


# ── tier_verdict / _run_tier dispatch agreement (CPU-provable: no real GPU/compile
# needed to prove WHICH STRATEGY FUNCTION `_run_tier` invokes) ───────────────────────

def test_scalecx49_run_tier_dispatches_scale_active_to_compiled_graphed_strategies(r: SubTestResult):
    """Before this ask, `_run_tier` bounced a scale-active cook straight to the plain
    interpreter for any `tier_id != "default"`, never calling `_run_torch_compile`/
    `_run_auto`/`_run_cuda_graph` at all. Spy on `_TIER_METHOD` (the exact dict `_run_tier`
    reads) to prove each is now actually invoked for a scale-active cook, matching
    `tier_verdict`'s new `TIER_REASON_SCALE_ACTIVE_COMPILED` claim.

    `torch_compile`/`auto` are exercised through the REAL `tex_engine.prepare()` ->
    `_dispatch_tier()` path on CPU (device resolution is real and stays "cpu" there, so
    `tier_id` genuinely lands on the requested compile_mode). `cuda_graph` additionally
    needs `select_tier` to see a device string starting with `"cuda"` -- `prepare()`'s own
    device resolution falls back to "cpu" with no CUDA hardware present (verified), so that
    row calls `_run_tier` directly against a hand-built `ExecContext` instead, mirroring
    `test_tierq48_agreement.py`'s own `test_tierq48_agrees_a_non_default_tier_never_arms_roi`
    precedent for testing `cuda_graph` routing without real hardware."""
    called = {}
    real_methods = dict(_tiers._TIER_METHOD)

    def _spy(name):
        def _fn(ctx):
            called[name] = called.get(name, 0) + 1
            # Return a trivial, correctly-shaped output without touching torch.compile/
            # CUDA at all -- this test is about DISPATCH, not compilation.
            return {"OUT": ctx.bindings["A"]}
        return _fn

    for name in ("torch_compile", "auto", "cuda_graph"):
        _tiers._TIER_METHOD[name] = _spy(name)
    try:
        for compile_mode in ("torch_compile", "auto"):
            called.clear()
            tier_id = _tiers.select_tier(compile_mode, "cpu", False, False)
            assert tier_id == compile_mode
            plan = tex_engine.prepare(_BLUR_CODE, {"A": make_img(1, 8, 8, 3, seed=7)},
                                     device_mode="cpu", compile_mode=compile_mode, scale=0.5)
            assert plan.tier_id == compile_mode
            tex_engine._dispatch_tier(plan)
            if called.get(compile_mode) == 1:
                r.ok(f"scale-active cook with compile_mode={compile_mode} dispatches to "
                     f"_run_{compile_mode} (was previously forced straight to the "
                     f"interpreter)")
            else:
                r.fail("scalecx49 dispatch",
                      f"compile_mode={compile_mode}: expected 1 call, got {called}")

        # cuda_graph: select_tier only string-checks the device (no real GPU required,
        # same fact test_tierq48_agreement.py's own cuda_graph row leans on).
        called.clear()
        tier_id = _tiers.select_tier("cuda_graph", "cuda:0", False, False)
        assert tier_id == "cuda_graph"
        program = _prog()
        ctx = tex_engine.ExecContext(program, {"A": make_img(1, 8, 8, 3, seed=9)}, {},
                                     "cuda:0", _BLUR_CODE, 0, ["OUT"], None, "fp32",
                                     scale=0.5)
        _tiers._run_tier(ctx, "cuda_graph")
        if called.get("cuda_graph") == 1:
            r.ok("scale-active cook with tier_id=cuda_graph dispatches to _run_cuda_graph "
                 "(was previously forced straight to the interpreter)")
        else:
            r.fail("scalecx49 dispatch", f"cuda_graph: expected 1 call, got {called}")
    finally:
        _tiers._TIER_METHOD.update(real_methods)


def test_scalecx49_run_tier_scale_none_dispatch_unchanged(r: SubTestResult):
    """Invariant 7: a `scale=None` cook's dispatch (which strategy runs) is completely
    unaffected by this ask -- `_run_tier`'s body for that path is the same one dict lookup
    it was before."""
    called = {}
    real_methods = dict(_tiers._TIER_METHOD)

    def _spy(name):
        def _fn(ctx):
            called[name] = called.get(name, 0) + 1
            return {"OUT": ctx.bindings["A"]}
        return _fn

    for name in ("torch_compile",):
        _tiers._TIER_METHOD[name] = _spy(name)
    try:
        plan = tex_engine.prepare(_BLUR_CODE, {"A": make_img(1, 8, 8, 3, seed=8)},
                                 device_mode="cpu", compile_mode="torch_compile")
        assert plan.ctx.scale is None
        tex_engine._dispatch_tier(plan)
        if called.get("torch_compile") == 1:
            r.ok("scale=None cook with compile_mode=torch_compile dispatches to "
                 "_run_torch_compile exactly as before this ask")
        else:
            r.fail("scalecx49 invariant 7 dispatch", f"expected 1 call, got {called}")
    finally:
        _tiers._TIER_METHOD.update(real_methods)


# ── CUDA-gated: real end-to-end capture/replay and torch.compile parity ─────────────

def _tag_pixel_arg(name: str, arg_index: int):
    """Test-only: temporarily register `name`'s argument `arg_index` as a `pixel_args=`
    position, so `Interpreter._eval_call`'s SCALE-47b multiply actually fires for it.
    Returns a restore callable.

    Why this is needed rather than using a REAL `pixel_args=` builtin directly: every
    builtin the registry actually tags this way today (`gauss_blur`/`erode`/`dilate`/
    `bilateral_filter`) is ALSO in `graphed._SYNC_STDLIB` (its pixel-unit argument resolves
    via a capture-illegal `.item()`), so none of them is ever capturable by the cuda_graph
    tier at all -- `_capturable()` declines the program outright before `scale` could matter
    to a capture either way. `mix`/`lerp` (an alias, arg index 2 is `t`) is NOT sync-gated
    and needs no `.item()`, so tagging it here builds a genuinely CAPTURABLE program whose
    OUTPUT still depends on `scale` through the real, unmodified interpreter/`graphed.py`
    scale-multiply path -- this patches DATA (the registry's own lookup table), not the
    mechanism under test."""
    from TEX_Wrangle.tex_runtime import stdlib_registry as _reg
    real_fn = _reg.pixel_args_by_name
    patched = dict(real_fn())
    patched[name] = (arg_index,)
    _reg.pixel_args_by_name = lambda: patched
    def _restore():
        _reg.pixel_args_by_name = real_fn
    return _restore


def _mix_scale_program():
    """`@TMP = mix(@A, @B, 0.5); @OUT = mix(@TMP, @B, 0.5);` -- TWO chained `mix()` calls.

    FIX-SCALECX X3: the ORIGINAL one-call version
    (`@OUT = mix(@A, @B, 0.5);`) has exactly one `FunctionCall` op, so
    `graphed._capturable`'s own static op count is 1 -- BELOW `_GRAPH_MIN_OPS = 2`
    (PF-2, a deliberate, pre-existing, unrelated-to-this-ask floor: "0/1-op programs
    capture an ~empty graph -> pure loss"). `_graph_capture_worthwhile` declines any
    1-op program before `run_graphed` ever computes a `_capture_key` or attempts a
    capture -- confirmed live on real CUDA hardware (sm_75): both of this file's
    CUDA-gated tests FAILED with "run_graphed declined the capturable mix() program",
    never reaching the mismatch/parity they exist to prove. This was a
    bug in THIS TEST's own synthetic program, not in `graphed.py` or in SCALECX-49's
    fix. Chaining a second tagged call raises
    the static op count to 2, clearing `_GRAPH_MIN_OPS` with no other change to the
    mechanism under test.

    With `_tag_pixel_arg("mix", 2)` active, arg 2 (`t`) of EVERY `mix()` call is
    multiplied by `scale`, so `@TMP` becomes `lerp(A, B, 0.5 * scale)` and `@OUT`
    becomes `lerp(TMP, B, 0.5 * scale)`. Still capturable (no sync-gated call, no
    loop) and still exactly reproducible: the tests below compute their expected
    value by running the SAME program through the real, unmodified interpreter
    (`_ref`), never by hand-deriving the two-call arithmetic -- correct regardless of
    how many `mix()` calls the chain has."""
    bt = {"A": TEXType.VEC4, "B": TEXType.VEC4, "OUT": TEXType.VEC4}
    code = "@TMP = mix(@A, @B, 0.5);\n@OUT = mix(@TMP, @B, 0.5);\n"
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    used = _collect_identifiers(prog)
    return prog, tm, used


def test_scalecx49_cuda_graph_capture_mismatch_before_fix_live(r: SubTestResult):
    """Red-first, on real hardware: a CUDA-graph capture taken at one `scale` must not
    replay for a request at another. Reproduces the bug `_capture_key`'s `scale` component
    fixes by reverting to a SCALE-BLIND key (a scratch wrapper dropping the `scale` kwarg --
    exactly what every call site did before this ask) and showing the second, differently-
    scaled request wrongly reuses the first capture; then restores the real key and shows
    both requests resolve correctly."""
    if not _CUDA:
        r.skip("scalecx49 cuda_graph live capture mismatch", "needs a CUDA device")
        return
    from TEX_Wrangle.tex_runtime.interpreter import Interpreter
    restore_tag = _tag_pixel_arg("mix", 2)
    try:
        prog, tm, used = _mix_scale_program()
        A = torch.zeros(1, 16, 16, 4, device="cuda")
        B = torch.ones(1, 16, 16, 4, device="cuda")
        fp = "t_scalecx49_capture_mismatch"

        def _ref(scale):
            return Interpreter().execute(prog, {"A": A.clone(), "B": B.clone()}, tm,
                                         device="cuda", output_names=["OUT"],
                                         used_builtins=used, scale=scale)["OUT"]

        real_key = _graphed._capture_key

        def _scale_blind_key(fingerprint, device, precision, bindings, output_names,
                             latent_channel_count, scale=None):
            return real_key(fingerprint, device, precision, bindings, output_names,
                            latent_channel_count)   # scale dropped -- the pre-fix shape

        _graphed.clear_graph_cache()
        _graphed._capture_key = _scale_blind_key
        try:
            out_first = _graphed.run_graphed(
                prog, {"A": A.clone(), "B": B.clone()}, tm, "cuda", fp,
                output_names=["OUT"], used_builtins=used, scale=0.75)
            out_second_blind = _graphed.run_graphed(
                prog, {"A": A.clone(), "B": B.clone()}, tm, "cuda", fp,
                output_names=["OUT"], used_builtins=used, scale=0.25)
        finally:
            _graphed._capture_key = real_key
        if out_first is None or out_second_blind is None:
            r.fail("scalecx49 capture mismatch premise",
                  "run_graphed declined the capturable mix() program -- cannot exercise "
                  "the mismatch this test proves")
            return
        ref_quarter = _ref(0.25)
        md_blind = (out_second_blind.float() - ref_quarter.float()).abs().max().item()
        if md_blind > 1e-3:
            r.ok(f"pre-fix (scale-blind) key: a capture taken at scale=0.75 wrongly served "
                 f"a scale=0.25 request (maxdiff {md_blind:.3f} vs the correct answer) -- "
                 f"reproduced")
        else:
            r.fail("scalecx49 capture mismatch premise",
                  f"scale-blind key did not reproduce a mismatch (maxdiff {md_blind:.2e}) "
                  f"-- this test's premise is stale")

        _graphed.clear_graph_cache()
        out_half = _graphed.run_graphed(
            prog, {"A": A.clone(), "B": B.clone()}, tm, "cuda", fp,
            output_names=["OUT"], used_builtins=used, scale=0.75)
        out_quarter = _graphed.run_graphed(
            prog, {"A": A.clone(), "B": B.clone()}, tm, "cuda", fp,
            output_names=["OUT"], used_builtins=used, scale=0.25)
        if out_half is None or out_quarter is None:
            r.fail("scalecx49 capture fix", "run_graphed declined after the fix")
            return
        ref_half = _ref(0.75)
        md_half = (out_half.float() - ref_half.float()).abs().max().item()
        md_quarter = (out_quarter.float() - ref_quarter.float()).abs().max().item()
        if md_half < 1e-5 and md_quarter < 1e-5:
            r.ok(f"fixed _capture_key: scale=0.75 (maxdiff {md_half:.2e}) and scale=0.25 "
                 f"(maxdiff {md_quarter:.2e}) each replay correctly from their OWN capture")
        else:
            r.fail("scalecx49 capture fix",
                  f"maxdiff half={md_half:.2e} quarter={md_quarter:.2e} -- expected <1e-5")
    finally:
        restore_tag()
        _graphed.clear_graph_cache()


def test_scalecx49_cuda_parity_live(r: SubTestResult):
    """Parity requirement (CUDA leg): the compiled tiers (`torch_compile`/`auto`) and the
    graphed tier (`cuda_graph`) must match the plain interpreter's own scale-active output
    at scale = 1, 1/2, 1/4, 1/8, within invariant 2's tolerance (1e-5, fp32)."""
    if not _CUDA:
        r.skip("scalecx49 cuda parity (compiled+graphed vs interpreter)", "needs a CUDA device")
        return
    from TEX_Wrangle.tex_runtime.interpreter import Interpreter
    scales = (1.0, 0.5, 0.25, 0.125)

    # -- torch_compile / auto: the codegen-backed gauss_blur program (SCALE-CG-48's own
    #    shape), real hardware, real (or self-declining, per execute_compiled's own
    #    contract) torch.compile. --
    program = _prog()
    A_img = make_img(1, 32, 32, 3, seed=21).cuda()
    for s in scales:
        ref = _compiled._plain_execute(program, {"A": A_img.clone()}, {}, "cuda", scale=s,
                                       time_context=None)
        for fn, name in ((_compiled.execute_compiled, "torch_compile"),
                        (_compiled.run_auto, "auto")):
            out = fn(program, {"A": A_img.clone()}, {}, "cuda",
                    f"t_scalecx49_parity_{name}", scale=s)
            md = (out.float() - ref.float()).abs().max().item()
            if md < 1e-5:
                r.ok(f"[cuda] {name}(scale={s}) matches interpreter, maxdiff={md:.2e}")
            else:
                r.fail("scalecx49 cuda parity", f"{name} scale={s} maxdiff={md:.2e} >= 1e-5")

    # -- cuda_graph: the mix()-tagged capturable program (see the mismatch test above for
    #    why gauss_blur itself cannot reach this tier at all). --
    restore_tag = _tag_pixel_arg("mix", 2)
    try:
        prog, tm, used = _mix_scale_program()
        A = torch.zeros(1, 16, 16, 4, device="cuda")
        B = torch.ones(1, 16, 16, 4, device="cuda")
        _graphed.clear_graph_cache()
        for s in scales:
            ref = Interpreter().execute(prog, {"A": A.clone(), "B": B.clone()}, tm,
                                        device="cuda", output_names=["OUT"],
                                        used_builtins=used, scale=s)["OUT"]
            out = _graphed.run_graphed(prog, {"A": A.clone(), "B": B.clone()}, tm, "cuda",
                                       "t_scalecx49_parity_graph", output_names=["OUT"],
                                       used_builtins=used, scale=s)
            if out is None:
                r.fail("scalecx49 cuda parity", f"cuda_graph scale={s} declined")
                continue
            md = (out.float() - ref.float()).abs().max().item()
            if md < 1e-5:
                r.ok(f"[cuda] cuda_graph(scale={s}) matches interpreter, maxdiff={md:.2e}")
            else:
                r.fail("scalecx49 cuda parity", f"cuda_graph scale={s} maxdiff={md:.2e} >= 1e-5")
    finally:
        restore_tag()
        _graphed.clear_graph_cache()
