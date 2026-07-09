"""
Lazy input cooking (v0.18.0) — tex_lazy analysis + check_lazy_status contract
+ the execute() slot-remap path.

The frontend half (the graphToPrompt prompt-key rename) needs a live-ComfyUI
session; these tests pin every backend contract it relies on. The scheduler
side (ComfyUI actually skipping upstream subgraphs of unrequested lazy slots)
was probe-verified against the installed core's TopologicalSort during the
design phase and is ComfyUI's own tested behaviour, not TEX's.
"""
import json

from helpers import *
from TEX_Wrangle.tex_lazy import lazy_required_bindings, clear_lazy_memo
from TEX_Wrangle.tex_node import TEXWrangleNode


def _res(out):
    """First tensor from an execute() return (tuple or NodeOutput-like)."""
    vals = out if isinstance(out, tuple) else getattr(out, "args", out)
    return vals[0]


def test_lazy_analysis(r: SubTestResult):
    print("\n--- lazy: tex_lazy analysis tiers ---")
    clear_lazy_memo()

    # T1 — wired-but-unreferenced is simply not in the set.
    try:
        s = lazy_required_bindings("@OUT = @A;", {})
        assert "A" in s and "B" not in s
        r.ok("T1: unreferenced input not required")
    except Exception as e:
        r.fail("T1: unreferenced input not required", str(e))

    # T2 — statically dead if-branch is pruned.
    try:
        s = lazy_required_bindings(
            "float k = 0.0; if (k > 0.5) { @OUT = @B; } else { @OUT = vec3(0.1); }", {})
        assert "B" not in s
        r.ok("T2: static-if dead branch pruned")
    except Exception as e:
        r.fail("T2: static-if dead branch pruned", str(e))

    # T3 — the sweet spot: a $param widget value gates the branch.
    try:
        code = "if ($t > 0.5) { @OUT = luma(@B) * vec3(1.0); } else { @OUT = vec3(0.3); }"
        s0 = lazy_required_bindings(code, {"t": 0.0})
        s1 = lazy_required_bindings(code, {"t": 1.0})
        assert "B" not in s0 and "B" in s1
        r.ok("T3: $param value flips the required set")
    except Exception as e:
        r.fail("T3: $param value flips the required set", str(e))

    # Ternary with a foldable condition prunes like if/else (the interpreter
    # short-circuits scalar ternaries, so pruning matches runtime).
    try:
        s = lazy_required_bindings("@OUT = ($t > 0.5) ? luma(@B) : 0.2;", {"t": 0.0})
        assert "B" not in s
        r.ok("T3: literal-cond ternary pruned")
    except Exception as e:
        r.fail("T3: literal-cond ternary pruned", str(e))

    # *0 must NOT sever the dependency: NaN*0=NaN (IEEE) and the optimizer
    # deliberately refuses x*0->0 (shape safety). Only both-literal BinOps fold.
    try:
        s = lazy_required_bindings("@OUT = @A * 0.0;", {})
        assert "A" in s
        r.ok("*0 does not sever the dependency (IEEE honesty)")
    except Exception as e:
        r.fail("*0 does not sever the dependency", str(e))

    # && with one symbolic side never folds -> both branches conservatively kept
    # (matches the interpreter, which evaluates both sides of &&).
    try:
        s = lazy_required_bindings(
            "if ($t > 0.5 && luma(@B) > 0.2) { @OUT = @B; } else { @OUT = vec3(0.1); }",
            {"t": 0.0})
        assert "B" in s
        r.ok("&&-guarded reference conservatively kept")
    except Exception as e:
        r.fail("&&-guarded reference conservatively kept", str(e))

    # String params never fold -> string-gated branches keep both sides.
    try:
        s = lazy_required_bindings(
            'if (contains($m, "x") > 0.5) { @OUT = @B; } else { @OUT = vec3(0.1); }',
            {"m": "yy"})
        assert "B" in s
        r.ok("string-param condition never folds (both kept)")
    except Exception as e:
        r.fail("string-param condition never folds", str(e))

    # fp32 boundary: $t == 0.5 -> `0.5 > 0.5` is false, matching the runtime.
    try:
        code = "if ($t > 0.5) { @OUT = @B; } else { @OUT = vec3(0.1); }"
        s = lazy_required_bindings(code, {"t": 0.5})
        assert "B" not in s
        r.ok("boundary $t=0.5 folds like the runtime (else branch)")
    except Exception as e:
        r.fail("boundary $t=0.5 folds like the runtime", str(e))

    # bool and int params fold; propagate-then-fold chains work (UC-4 reuse).
    try:
        s_b = lazy_required_bindings(
            "if ($on > 0.5) { @OUT = @B; } else { @OUT = vec3(0.1); }", {"on": False})
        s_k = lazy_required_bindings(
            "float k = $n * 2.0; if (k > 1.0) { @OUT = @B; } else { @OUT = vec3(0.1); }",
            {"n": 0})
        assert "B" not in s_b and "B" not in s_k
        r.ok("bool/int params fold; propagate-then-fold chain prunes")
    except Exception as e:
        r.fail("bool/int params fold; propagate-then-fold", str(e))

    # Dead VarDecl (DCE case) is conservatively KEPT by design in v1 — the
    # analysis runs no DCE; the runtime program is DCE'd so this only costs an
    # unnecessary cook, never a wrong one. Pin the documented behaviour.
    try:
        s = lazy_required_bindings("float x = luma(@B); @OUT = @A;", {})
        assert "B" in s
        r.ok("dead VarDecl conservatively kept (documented v1 scope)")
    except Exception as e:
        r.fail("dead VarDecl conservatively kept", str(e))

    # Analysis failure -> None (caller keeps everything); memo is idempotent.
    try:
        assert lazy_required_bindings("this is not tex", {}) is None
        a = lazy_required_bindings("@OUT = @A;", {"t": 0.25})
        b = lazy_required_bindings("@OUT = @A;", {"t": 0.25})
        assert a == b and a is not None
        r.ok("garbage -> None; memo idempotent")
    except Exception as e:
        r.fail("garbage -> None; memo idempotent", str(e))


def test_lazy_check_status(r: SubTestResult):
    print("\n--- lazy: check_lazy_status contract ---")
    N = TEXWrangleNode
    SM2 = json.dumps([{"name": "A", "slot": "in_0", "type": "IMAGE"},
                      {"name": "B", "slot": "in_1", "type": "IMAGE"}])
    SM3 = json.dumps([{"name": "A", "slot": "in_0", "type": "IMAGE"},
                      {"name": "B", "slot": "in_1", "type": "IMAGE"},
                      {"name": "t", "slot": "in_2", "type": "FLOAT"}])
    gate = "if ($t > 0.5) { @OUT = @B; } else { @OUT = @A; }"

    # Widget param decides at once: t=0 -> only A requested.
    try:
        req = N.check_lazy_status(code=gate, t=0.0, _tex_slot_map=SM2,
                                  in_0=None, in_1=None)
        assert req == ["in_0"], req
        r.ok("T3 widget: dead branch's input not requested")
    except Exception as e:
        r.fail("T3 widget: dead branch's input not requested", str(e))

    # T4-lite iteration: wired scalar param cooks FIRST, images deferred.
    try:
        r1 = N.check_lazy_status(code=gate, _tex_slot_map=SM3,
                                 in_0=None, in_1=None, in_2=None)
        assert r1 == ["in_2"], r1
        # round 2 with the cooked value: t=0 -> only A (first spatial, needed).
        r2 = N.check_lazy_status(code=gate, _tex_slot_map=SM3,
                                 in_0=None, in_1=None, in_2=0.0)
        assert r2 == ["in_0"], r2
        r.ok("T4-lite: wired $param cooks first, then folds")
    except Exception as e:
        r.fail("T4-lite: wired $param cooks first, then folds", str(e))

    # R1 fail-safe: when the FIRST spatial input is the dead one, skipping it
    # could change first-wins shape derivation -> everything is requested.
    try:
        req = N.check_lazy_status(code=gate, _tex_slot_map=SM3,
                                  in_0=None, in_1=None, in_2=1.0)
        assert sorted(req) == ["in_0", "in_1"], req
        r.ok("R1: first-spatial-dead fails safe (all requested)")
    except Exception as e:
        r.fail("R1: first-spatial-dead fails safe", str(e))

    # R1 fail-safe: a constant program keeps its wired image (shape anchor).
    try:
        req = N.check_lazy_status(
            code="@OUT = vec3(0.25);",
            _tex_slot_map=json.dumps([{"name": "A", "slot": "in_0", "type": "IMAGE"}]),
            in_0=None)
        assert req == ["in_0"], req
        r.ok("R1: constant program keeps its shape anchor")
    except Exception as e:
        r.fail("R1: constant program keeps its shape anchor", str(e))

    # R2: LATENT wires are always cooked (output typing/fp32 semantics).
    try:
        req = N.check_lazy_status(
            code="@OUT = @A;",
            _tex_slot_map=json.dumps([
                {"name": "A", "slot": "in_0", "type": "IMAGE"},
                {"name": "L", "slot": "in_1", "type": "LATENT"}]),
            in_0=None, in_1=None)
        assert sorted(req) == ["in_0", "in_1"], req
        r.ok("R2: LATENT always requested")
    except Exception as e:
        r.fail("R2: LATENT always requested", str(e))

    # STRING wires are shape-inert: an unreferenced one is skippable.
    try:
        req = N.check_lazy_status(
            code="@OUT = @A;",
            _tex_slot_map=json.dumps([
                {"name": "A", "slot": "in_0", "type": "IMAGE"},
                {"name": "S", "slot": "in_1", "type": "STRING"}]),
            in_0=None, in_1=None)
        assert req == ["in_0"], req
        r.ok("unreferenced STRING wire skipped")
    except Exception as e:
        r.fail("unreferenced STRING wire skipped", str(e))

    # Fused chains request everything (v1 scope).
    try:
        req = N.check_lazy_status(code="@OUT = @A;", _tex_chain="{}",
                                  _tex_slot_map=SM2, in_0=None, in_1=None)
        assert sorted(req) == ["in_0", "in_1"], req
        r.ok("fused chain requests everything (v1)")
    except Exception as e:
        r.fail("fused chain requests everything (v1)", str(e))

    # R3 + robustness: garbage code -> all; no slot map -> []; never raises.
    try:
        req = N.check_lazy_status(code="not tex", _tex_slot_map=SM3,
                                  in_0=None, in_1=None, in_2=None)
        assert sorted(req) == ["in_0", "in_1", "in_2"], req
        assert N.check_lazy_status(code="@OUT=@A;") == []
        assert N.check_lazy_status(code="@OUT=@A;", _tex_slot_map="{broken") == []
        r.ok("R3 garbage -> all; no/broken map -> []; never raises")
    except Exception as e:
        r.fail("R3 garbage/no-map robustness", str(e))


def test_lazy_execute_path(r: SubTestResult):
    print("\n--- lazy: execute() slot remap + E6003 gate ---")
    N = TEXWrangleNode
    img = make_img(1, 8, 8, 3)
    SM = json.dumps([{"name": "A", "slot": "in_0", "type": "IMAGE"},
                     {"name": "B", "slot": "in_1", "type": "IMAGE"}])

    # Remapped cook is bit-exact vs the user-named cook.
    try:
        a = _res(N.execute(code="@OUT = @A * 0.5;", _tex_slot_map=SM,
                           in_0=img, in_1=None))
        b = _res(N.execute(code="@OUT = @A * 0.5;", A=img))
        assert torch.equal(a, b)
        r.ok("in_N remap bit-exact vs user-named cook")
    except Exception as e:
        r.fail("in_N remap bit-exact", str(e))

    # E6003 gate: a dead reference to a lazily-skipped input must not raise...
    try:
        out = N.execute(code="if ($t > 0.5) { @OUT = @B; } else { @OUT = @A * 0.9; }",
                        t=0.0, _tex_slot_map=SM, in_0=img, in_1=None)
        assert out is not None
        r.ok("E6003 gate: skipped dead reference forgiven")
    except Exception as e:
        r.fail("E6003 gate: skipped dead reference forgiven", str(e))

    # ...while the legacy (no slot map) path still raises E6003 exactly as v0.17.
    try:
        try:
            N.execute(code="if ($t > 0.5) { @OUT = @B; } else { @OUT = @A * 0.9; }",
                      t=0.0, A=img)
            r.fail("legacy E6003 preserved", "no error raised")
        except Exception as e:
            assert "E6003" in str(e) or "no input is connected" in str(e)
            r.ok("legacy E6003 preserved without a slot map")
    except Exception as e:
        r.fail("legacy E6003 preserved", str(e))

    # Shape preservation end-to-end: the R1-kept anchor defines the output.
    try:
        out = _res(N.execute(code="@OUT = vec3(0.25);", _tex_slot_map=json.dumps(
            [{"name": "A", "slot": "in_0", "type": "IMAGE"}]), in_0=img))
        assert tuple(out.shape) == (1, 8, 8, 3), out.shape
        r.ok("R1-kept anchor still defines output shape")
    except Exception as e:
        r.fail("R1-kept anchor still defines output shape", str(e))

    # Stray unmapped in_N keys are dropped, never phantom bindings.
    try:
        out = _res(N.execute(code="@OUT = @A * 2.0;", A=img, in_5=img))
        assert torch.equal(out, _res(N.execute(code="@OUT = @A * 2.0;", A=img)))
        r.ok("stray in_N keys dropped (no phantom bindings)")
    except Exception as e:
        r.fail("stray in_N keys dropped", str(e))

    # Fingerprint: deterministic, and the slot map busts the cache.
    try:
        f1 = N.fingerprint_inputs(code="@OUT=@A;", _tex_slot_map=SM, in_0=None, in_1=None)
        f2 = N.fingerprint_inputs(code="@OUT=@A;", _tex_slot_map=SM, in_0=None, in_1=None)
        f3 = N.fingerprint_inputs(code="@OUT=@A;", _tex_slot_map=json.dumps(
            [{"name": "Z", "slot": "in_0", "type": "IMAGE"}]), in_0=None)
        assert f1 == f2 and f1 != f3
        r.ok("fingerprint deterministic; slot-map change busts")
    except Exception as e:
        r.fail("fingerprint slot-map handling", str(e))

    # Schema declares the lazy pool (when the v3 API is importable).
    try:
        from TEX_Wrangle.tex_node import _V3_AVAILABLE
        if _V3_AVAILABLE:
            schema = N.define_schema()
            ids = [inp.id for inp in schema.inputs]
            assert "in_0" in ids and f"in_{N.MAX_LAZY_INPUTS - 1}" in ids
            lazy_flags = [getattr(inp, "lazy", None) for inp in schema.inputs
                          if getattr(inp, "id", "").startswith("in_")]
            assert all(lazy_flags)
            r.ok(f"schema declares {N.MAX_LAZY_INPUTS} lazy pool slots")
        else:
            r.ok("schema pool check skipped (v3 API absent in test venv)")
    except Exception as e:
        r.fail("schema declares lazy pool", str(e))


def test_lazy_schema_pool_ci(r: SubTestResult):
    """C3 (doc 32): the lazy pool (in_0..in_15, lazy=True) must be DECLARED even in the CI
    venv where the real V3 API is absent (IO is None, so the existing schema check no-ops).
    ComfyUI fires the lazy-cook trigger ONLY for schema-declared lazy inputs, so a dropped
    lazy=True would silently turn lazy cooking off (fail-safe, but the feature vanishes). A
    stub IO records the declared inputs so this is verified in CI, not just with ComfyUI."""
    import types
    import TEX_Wrangle.tex_node as tn

    def _mk(*a, **kw):
        ns = types.SimpleNamespace(**kw)
        ns.id = kw.get("id", a[0] if a else None)
        return ns

    class _StubSchema:
        def __init__(self, **kw):
            self.inputs = kw.get("inputs", [])
            self.outputs = kw.get("outputs", [])

    stub = types.SimpleNamespace(
        Schema=_StubSchema,
        String=types.SimpleNamespace(Input=_mk),
        Combo=types.SimpleNamespace(Input=_mk),
        Boolean=types.SimpleNamespace(Input=_mk),
        AnyType=types.SimpleNamespace(Input=_mk, Output=_mk),
    )
    saved = tn.IO
    try:
        tn.IO = stub
        schema = tn.TEXWrangleNode.define_schema()
        n = tn.TEXWrangleNode.MAX_LAZY_INPUTS
        pool = [i for i in schema.inputs if str(getattr(i, "id", "")).startswith("in_")]
        ids = {getattr(i, "id", None) for i in pool}
        lazy_flags = [getattr(i, "lazy", None) for i in pool]
        assert len(pool) == n, f"expected {n} lazy pool slots, got {len(pool)}"
        assert "in_0" in ids and f"in_{n - 1}" in ids, f"pool ids wrong: {sorted(ids)}"
        assert all(f is True for f in lazy_flags), f"pool not all lazy=True: {lazy_flags}"
        r.ok(f"lazy pool declared in CI: {n} in_N slots, all lazy=True (V3 API stubbed)")
    except Exception as e:
        r.fail("lazy schema pool (CI stub)", str(e))
    finally:
        tn.IO = saved
