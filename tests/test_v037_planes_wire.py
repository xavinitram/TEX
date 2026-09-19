"""DATA-6, the wire half of "Planes": `PlanesValue`, demand-driven expansion, and the arms that
make `@beauty.diffuse` cook.

THE SHAPE. A host wires ONE input carrying many named planes and a program reads them by name.
The wire value is `tex_marshalling.PlanesValue` — a `{name: tensor [B,H,W,C<=4]}` plus an
optional per-plane `descs` map, deliberately NOT a tensor subclass. `expand_plane_bindings`
turns it into ordinary per-plane tensor bindings under the VERBATIM dotted name, for the planes
the SOURCE mentions and no other, at the one seam the Promise precedent already uses
(`tex_engine.prepare`, beside the IO-1 resolution; `cook_stage_list` for a chain). No tier, no
emitter and no cache learns what a plane is.

WHAT IS PINNED HERE, and why each row is the shape it is:
  * expansion is DEMAND-DRIVEN and REMOVES the base row — the PM-10 laziness proof is
    structural (an unmentioned plane never enters the dict), and the spy on the ingest path
    confirms a property the design already guarantees rather than establishing one;
  * the CACHE-6 boundary key sees a PlanesValue as a TENSOR binding whose identity is every
    declared plane's name and shape — never its address (the P0-H lesson, plan A2/F20);
  * the interpreter and codegen tiers agree on a plane cook (invariant #2);
  * a typo'd plane is W7009 with a did-you-mean and then an honest E6003; a declared plane the
    program does not read is SILENT and never trips W7002; a plane named after a swizzle is
    E3304 at expansion and `Z` (uppercase) is not one; a raw dict stays E7005 with a hint;
  * a host's DATA-1 tag reaches every expanded plane (plan A3/F19); a fused chain refuses a
    dotted export (plan A1); the E6003 message names the WIRE as the slot (plan A18);
  * PLANES is gated on the engine egress profile exactly as ARRAY is, so under the ComfyUI
    profile no PlanesValue can be constructed or expanded and nothing changes (invariant #7);
  * plane WRITES are deferred: `@OUT.diffuse = ...` is a compile error whose hint says so.

Red-first: `test_a_planes_wire_expands_only_the_mentioned_planes` and
`test_boundary_key_moves_when_an_unread_plane_changes` fail on the base sha (no `PlanesValue`,
no expansion, the boundary key address-folds an unknown object).
"""
import os
from unittest import mock

from helpers import *

import failure_harness as fh
import test_integration as ti
from TEX_Wrangle import tex_api, tex_engine, tex_fusion, tex_marshalling
from TEX_Wrangle.tex_cache import get_cache, parse_and_split
from TEX_Wrangle.tex_compiler.types import (
    TEXType, CHANNEL_MAP, VALID_SWIZZLES, array_wires_enabled, set_array_wires,
    planes_wires_enabled)
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_marshalling import BufferMeta, infer_binding_type
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, InterpreterError


def _PlanesValue(*a, **kw):
    """Imported at call time so the red-first rows fail on their own assertion at base."""
    from TEX_Wrangle.tex_marshalling import PlanesValue
    return PlanesValue(*a, **kw)


def expand_plane_bindings(*a, **kw):
    from TEX_Wrangle.tex_marshalling import expand_plane_bindings as f
    return f(*a, **kw)


def _expand_plane_meta(*a, **kw):
    from TEX_Wrangle.tex_marshalling import _expand_plane_meta as f
    return f(*a, **kw)


class _planes_enabled:
    """Flip the engine-profile switch for one block and restore it — the same flag
    `tex_marshalling.set_egress_profile("engine")` flips, read directly."""
    def __init__(self, on=True):
        self._on = on

    def __enter__(self):
        self._prev = array_wires_enabled()
        set_array_wires(self._on)

    def __exit__(self, *a):
        set_array_wires(self._prev)


def _pv(B=1, H=8, W=8, seed=3, **extra):
    """A deterministic three-plane wire: `diffuse` (vec3), `specular` (vec3), `Z` (float, and
    uppercase on purpose — the conventional EXR depth name, which collides with nothing)."""
    torch.manual_seed(seed)
    planes = {"diffuse": torch.rand(B, H, W, 3), "specular": torch.rand(B, H, W, 3),
              "Z": torch.rand(B, H, W, 1)}
    planes.update(extra)
    return _PlanesValue(planes)


LIT = ("vec3 lit = @beauty.diffuse + @beauty.specular;\n"
       "float depth = @beauty.Z;\n"
       "@OUT = vec4(lit * depth, 1.0);\n")
DIFFUSE_ONLY = "@OUT = vec4(@beauty.diffuse, 1.0);\n"


def _code_of(e) -> str:
    return getattr(e, "_code", "") or getattr(getattr(e, "diagnostic", None), "code", "")


def _hint_of(e) -> str:
    return getattr(e, "_hint", "") or getattr(getattr(e, "diagnostic", None), "hint", "")


# ── red-first: the expansion and the boundary key ───────────────────────────────

def test_a_planes_wire_expands_only_the_mentioned_planes(r: SubTestResult):
    print("\n--- DATA-6 L-C2: expansion is demand-driven and removes the base row ---")
    try:
        with _planes_enabled(True):
            pv = _pv()
            out = expand_plane_bindings({"beauty": pv, "k": 0.5}, DIFFUSE_ONLY)
            assert set(out) == {"beauty.diffuse", "k"}, sorted(out)
            assert out["beauty.diffuse"] is pv.planes["diffuse"], "the plane tensor itself, no copy"
            assert out["k"] == 0.5
            r.ok("`@beauty.diffuse` alone: {beauty.diffuse, k} — no `beauty`, no `Z`, no `specular`")
    except Exception as e:
        r.fail("demand-driven expansion", str(e))
    try:
        with _planes_enabled(True):
            pv = _pv()
            out = expand_plane_bindings({"beauty": pv}, LIT)
            assert set(out) == {"beauty.diffuse", "beauty.specular", "beauty.Z"}, sorted(out)
            types = {n: infer_binding_type(v) for n, v in out.items()}
            assert types == {"beauty.diffuse": TEXType.VEC3, "beauty.specular": TEXType.VEC3,
                             "beauty.Z": TEXType.FLOAT}, types
            r.ok("three mentioned planes: three rows under the verbatim dotted names, typed per plane")
    except Exception as e:
        r.fail("three-plane expansion", str(e))
    try:
        # INVARIANT #7: no PlanesValue -> the SAME dict object back, nothing allocated.
        plain = {"A": make_img(1, 4, 4, 3), "k": 1.0}
        assert expand_plane_bindings(plain, "@OUT = @A * $k;") is plain
        with _planes_enabled(True):
            assert expand_plane_bindings(plain, "@OUT = @A.rgb * $k;") is plain
        r.ok("no PlanesValue present: `bindings` is returned unchanged (identity), both profiles")
    except Exception as e:
        r.fail("identity without planes", str(e))
    try:
        # a swizzled plane read is a plane read followed by an ordinary ChannelAccess (design §1):
        # the demand set carries `diffuse`, never `diffuse.rgb`.
        with _planes_enabled(True):
            out = expand_plane_bindings({"beauty": _pv()},
                                        "@OUT = vec4(@beauty.diffuse.rgb * @beauty.Z, 1.0);")
            assert set(out) == {"beauty.diffuse", "beauty.Z"}, sorted(out)
        r.ok("`@beauty.diffuse.rgb`: the demand set is the one-segment plane name")
    except Exception as e:
        r.fail("swizzled plane demand", str(e))


def test_boundary_key_moves_when_an_unread_plane_changes(r: SubTestResult):
    print("\n--- DATA-6 L-C2: the CACHE-6 boundary key is content-derived over the plane SET (A2/F20) ---")

    def key(pv):
        stages = [{"code": DIFFUSE_ONLY, "bindings": {"beauty": pv}},
                  {"code": "@OUT = @A * 0.5;", "chain_input": "A", "bindings": {}}]
        return tex_engine.boundary_lineage_key(stages, 1, "cpu", "fp32", upstream=("src#1",))

    try:
        with _planes_enabled(True):
            k1, k2 = key(_pv(seed=1)), key(_pv(seed=2))
            assert k1 == k2, "two wires with the same plane set and shapes must key the same " \
                             "(content-derived, never the object's address)"
            r.ok("same declared set + shapes, different objects/pixels: SAME key (not address-keyed)")
    except Exception as e:
        r.fail("boundary key content-derived", str(e))
    try:
        with _planes_enabled(True):
            base = key(_pv())
            # `Z` is never read by the prefix program; its shape is still part of the wire's identity
            assert key(_pv(Z=torch.rand(1, 8, 8, 2))) != base, "an unread plane's shape must move the key"
            # ...and so is the declared SET: an extra unread plane is a different wire
            assert key(_pv(N=torch.rand(1, 8, 8, 3))) != base, "an extra unread plane must move the key"
            assert key(_pv(B=2)) != base, "the wire's batch is part of its identity"
            r.ok("an unread plane's shape, an extra plane, a different batch: each MOVES the key")
    except Exception as e:
        r.fail("boundary key moves", str(e))
    try:
        with _planes_enabled(True):
            pv = _pv()
            assert tex_engine._is_tensor_binding(pv), "a PlanesValue carries pixels: tensor side"
            shape = tex_engine._binding_shape(pv)
            assert shape == ("Z", 1, 8, 8, 1, "diffuse", 1, 8, 8, 3, "specular", 1, 8, 8, 3), shape
            r.ok("_is_tensor_binding / _binding_shape: every plane's name and shape, sorted")
    except Exception as e:
        r.fail("tensor-binding arms", str(e))


# ── the cook: both tiers, both devices ────────────────────────────────────────────

def test_plane_cook_is_bit_exact_across_tiers(r: SubTestResult):
    print("\n--- DATA-6 L-C2: `@beauty.diffuse + @beauty.specular` cooks on both tiers (invariant #2) ---")
    for dev in devices():
        try:
            with _planes_enabled(True):
                pv = _pv(B=2, H=16, W=16)
                pv.planes = {n: t.to(dev) for n, t in pv.planes.items()}
                res = tex_engine.cook(LIT, {"beauty": pv}, device_mode=dev)
                interp = res.outputs["OUT"]
                ref = ((pv.planes["diffuse"] + pv.planes["specular"]) * pv.planes["Z"])
                assert interp.shape == (2, 16, 16, 4), interp.shape
                assert (interp[..., :3] - ref).abs().max().item() < 1e-5
                assert torch.all(interp[..., 3] == 1.0)
                assert sorted(res.binding_names) == ["beauty.Z", "beauty.diffuse", "beauty.specular"], \
                    res.binding_names
                # the codegen tier, on the SAME expanded bindings the engine cooks (the seam
                # expands before any tier; the harness drives the tier one layer below routing)
                expanded = expand_plane_bindings({"beauty": pv}, LIT)
                cg = fh.run_tier(LIT, expanded, "codegen", device=dev)["OUT"]
                assert fh.max_diff({"OUT": interp}, {"OUT": cg}) < 1e-5, \
                    f"interp vs codegen maxdiff {fh.max_diff({'OUT': interp}, {'OUT': cg})}"
            r.ok(f"[{dev}] engine cook == reference, and codegen == interpreter within 1e-5")
        except Exception as e:
            r.fail(f"[{dev}] plane cook", str(e))
    try:
        with _planes_enabled(True):
            pv = _pv()
            res = tex_engine.cook("@OUT = vec4(@beauty.diffuse.rgb * 2.0, @beauty.Z);",
                                  {"beauty": pv}, device_mode="cpu")
            out = res.outputs["OUT"]
            assert (out[..., :3] - pv.planes["diffuse"] * 2.0).abs().max().item() < 1e-6
            assert (out[..., 3:] - pv.planes["Z"]).abs().max().item() < 1e-6
        r.ok("`@beauty.diffuse.rgb` — a plane read followed by an ordinary swizzle — cooks")
    except Exception as e:
        r.fail("swizzled plane read cooks", str(e))
    try:
        # the same program through the stage-list family (cook_stage_list expands per stage)
        with _planes_enabled(True):
            pv = _pv()
            one = tex_engine.cook_stage_list([{"code": LIT, "bindings": {"beauty": pv}}])["OUT"]
            two = tex_engine.cook_stage_list(
                [{"code": DIFFUSE_ONLY, "bindings": {"beauty": pv}},
                 {"code": "@OUT = @A * 0.5;", "chain_input": "A", "bindings": {}}])["OUT"]
            assert one.shape == (1, 8, 8, 4) and two.shape == (1, 8, 8, 4)
            assert (two[..., :3] - pv.planes["diffuse"] * 0.5).abs().max().item() < 1e-6
        r.ok("cook_stage_list: a one-stage and a two-stage chain both expand the wire")
    except Exception as e:
        r.fail("stage-list expansion", str(e))


def test_pm10_laziness_an_unread_plane_is_never_marshalled(r: SubTestResult):
    print("\n--- DATA-6 L-C2: PM-10 — a plane the program does not read never reaches ingest ---")
    try:
        with _planes_enabled(True):
            pv = _pv()
            seen = []
            real = tex_marshalling.to_fp32_if_int_image

            def spy(t, device=None):
                seen.append(t.data_ptr())
                return real(t, device=device)

            with mock.patch.object(tex_marshalling, "to_fp32_if_int_image", spy):
                res = tex_engine.cook(DIFFUSE_ONLY, {"beauty": pv}, device_mode="cpu")
            assert res.outputs["OUT"].shape == (1, 8, 8, 4)
            assert pv.planes["diffuse"].data_ptr() in seen, "the read plane is ingested"
            for unread in ("specular", "Z"):
                assert pv.planes[unread].data_ptr() not in seen, f"`{unread}` was marshalled"
            assert res.binding_names == ["beauty.diffuse"], res.binding_names
        r.ok("only `diffuse` passed through the ingest cast; `specular` and `Z` never did")
    except Exception as e:
        r.fail("laziness spy", str(e))


# ── the refusals ──────────────────────────────────────────────────────────────

def test_undeclared_plane_is_w7009_with_a_did_you_mean(r: SubTestResult):
    print("\n--- DATA-6 L-C2: `@beauty.diffues` -> W7009 (did you mean diffuse?) then E6003 ---")
    try:
        with _planes_enabled(True):
            warnings = []
            raised = None
            try:
                expand_plane_bindings({"beauty": _pv()}, "@OUT = vec4(@beauty.diffues, 1.0);",
                                      on_warning=warnings.append)
            except InterpreterError as e:
                raised = e
            assert len(warnings) == 1, warnings
            w = warnings[0]
            assert w.code == "W7009" and w.severity == "warning", (w.code, w.severity)
            assert "diffues" in w.message and "beauty" in w.message, w.message
            assert w.suggestions and w.suggestions[0] == "diffuse", w.suggestions
            assert "diffuse" in w.hint, w.hint
            assert raised is not None and _code_of(raised) == "E6003", raised
            assert "slot 'beauty'" in str(raised) and "diffues" in str(raised), str(raised)
            assert "diffuse" in _hint_of(raised), _hint_of(raised)
        r.ok("W7009 carries the did-you-mean; the cook is then refused E6003 naming the slot")
    except Exception as e:
        r.fail("W7009", str(e))
    try:
        # through the engine, with no collector: the advisory is LOGGED, the refusal raised
        with _planes_enabled(True):
            import logging
            records = []
            handler = logging.Handler()
            handler.emit = lambda rec: records.append(rec.getMessage())
            log = logging.getLogger("TEX")
            log.addHandler(handler)
            try:
                try:
                    tex_engine.cook("@OUT = vec4(@beauty.diffues, 1.0);", {"beauty": _pv()},
                                    device_mode="cpu")
                    raise AssertionError("cooked an undeclared plane")
                except InterpreterError as e:
                    assert _code_of(e) == "E6003", _code_of(e)
            finally:
                log.removeHandler(handler)
            assert any("W7009" in m and "diffues" in m for m in records), records
        r.ok("engine path: W7009 logged, E6003 raised")
    except Exception as e:
        r.fail("W7009 via engine", str(e))


def test_collision_is_e3304_and_Z_does_not_collide(r: SubTestResult):
    print("\n--- DATA-6 L-C2: a plane named after a swizzle is E3304; `Z` is not one ---")
    try:
        with _planes_enabled(True):
            collision = set(CHANNEL_MAP) | set(VALID_SWIZZLES)
            assert len(collision) == 38
            for bad in ("rgb", "z", "a", "xyzw", "bgr"):
                pv = _pv(**{bad: torch.rand(1, 8, 8, 1)})
                try:
                    # never read by the program: the ambiguity is a property of the WIRE
                    expand_plane_bindings({"beauty": pv}, DIFFUSE_ONLY)
                    raise AssertionError(f"plane `{bad}` was accepted")
                except InterpreterError as e:
                    assert _code_of(e) == "E3304", (bad, _code_of(e))
                    assert f"plane `{bad}` on `@beauty` collides with the swizzle `.{bad}`" in str(e), str(e)
                    assert "rename the plane" in str(e), str(e)
        r.ok("rgb / z / a / xyzw / bgr: E3304 at expansion, even when unread")
    except Exception as e:
        r.fail("E3304", str(e))
    try:
        with _planes_enabled(True):
            # the finding that matters most: the conventional EXR names are uppercase and free
            pv = _pv(N=torch.rand(1, 8, 8, 3), RGBA=torch.rand(1, 8, 8, 4), R=torch.rand(1, 8, 8, 1))
            out = expand_plane_bindings({"beauty": pv},
                                        "@OUT = vec4(@beauty.N * @beauty.Z, @beauty.R);")
            assert set(out) == {"beauty.N", "beauty.Z", "beauty.R"}, sorted(out)
            res = tex_engine.cook("@OUT = vec4(@beauty.diffuse * @beauty.Z, 1.0);",
                                  {"beauty": _pv()}, device_mode="cpu")
            assert res.outputs["OUT"].shape == (1, 8, 8, 4)
        r.ok("Z / N / RGBA / R do not collide: `@beauty.Z` is a plane read with no rename")
    except Exception as e:
        r.fail("Z does not collide", str(e))


def test_unread_declared_plane_is_silent_and_w7002_stays_quiet(r: SubTestResult):
    print("\n--- DATA-6 L-C2: a declared plane the program does not read draws NO warning ---")
    try:
        with _planes_enabled(True):
            pv = _pv()
            warnings = []
            out = expand_plane_bindings({"beauty": pv}, DIFFUSE_ONLY, on_warning=warnings.append)
            assert warnings == [], [w.code for w in warnings]
            # the map the engine derives after expansion names only the mentioned plane, so the
            # W7002 lint cannot see `specular` or `Z` — by construction, not by exemption
            bt = {n: infer_binding_type(v) for n, v in out.items()}
            diags = tex_api.check(DIFFUSE_ONLY, bt)
            assert not any(d.code == "W7002" for d in diags), [d.message for d in diags]
            assert not any(d.severity == "error" for d in diags), [d.message for d in diags]
        r.ok("expansion emits nothing for `specular`/`Z`; the lint over the expanded map has no W7002")
    except Exception as e:
        r.fail("unread plane silent", str(e))
    try:
        # a host lint that types the BASE PLANES (HOOK-2 shape) must not hear "beauty is unused"
        # for a program that reads one of its planes — but must for one that reads none.
        with _planes_enabled(True):
            diags = tex_api.check("@OUT = vec4(@beauty.diffuse, 1.0); // lint\n",
                                  {"beauty": TEXType.PLANES})
            assert not any(d.code == "W7002" for d in diags), [d.message for d in diags]
            diags = tex_api.check("@OUT = vec4(1.0); // lint\n",
                                  {"beauty": TEXType.PLANES, "img": TEXType.VEC4})
            w = sorted(d.message for d in diags if d.code == "W7002")
            assert len(w) == 2 and "beauty" in w[0] and "img" in w[1], w
        r.ok("host-typed PLANES base: no W7002 when a plane is read; W7002 when nothing is")
    except Exception as e:
        r.fail("W7002 arm", str(e))


def test_raw_dict_stays_e7005_with_a_planesvalue_hint(r: SubTestResult):
    print("\n--- DATA-6 L-C2: a raw {name: tensor} dict is still E7005, now with a hint ---")
    try:
        try:
            infer_binding_type({"diffuse": torch.rand(1, 4, 4, 3), "Z": torch.rand(1, 4, 4, 1)})
            raise AssertionError("a raw dict was typed")
        except InterpreterError as e:
            assert _code_of(e) == "E7005", _code_of(e)
            assert "cannot be typed by TEX" in str(e), str(e)
            assert "PlanesValue" in _hint_of(e), _hint_of(e)
        # ...on both profiles: the refusal is not a profile question
        with _planes_enabled(True):
            try:
                infer_binding_type({"diffuse": torch.rand(1, 4, 4, 3)})
                raise AssertionError("a raw dict was typed (engine)")
            except InterpreterError as e:
                assert _code_of(e) == "E7005" and "PlanesValue" in _hint_of(e)
        # a dict that is NOT plane-shaped gets the terminal's hint unchanged
        try:
            infer_binding_type({"x": 1})
            raise AssertionError("typed")
        except InterpreterError as e:
            assert _code_of(e) == "E7005" and "PlanesValue" not in _hint_of(e), _hint_of(e)
        r.ok("E7005 unchanged; the hint says `did you mean PlanesValue` for a tensor dict only")
    except Exception as e:
        r.fail("E7005 hint", str(e))


# ── DATA-1 meta, E6003, fusion ────────────────────────────────────────────────

def test_meta_fans_out_to_expanded_planes(r: SubTestResult):
    print("\n--- DATA-6 L-C2: a host tag filed under `beauty` reaches `beauty.diffuse` (A3/F19) ---")
    try:
        with _planes_enabled(True):
            pv = _pv()
            res = tex_engine.cook(LIT, {"beauty": pv}, device_mode="cpu",
                                  binding_meta={"beauty": BufferMeta("linear", "opaque", 7)})
            assert res.out_meta is not None, "every colour tag was dropped (F19)"
            m = res.out_meta["OUT"]
            assert (m.colorspace, m.premult, m.frame) == ("linear", "opaque", 7), m
        r.ok("cook: out_meta['OUT'] carries the wire's tag through the expansion")
    except Exception as e:
        r.fail("meta fan-out via cook", str(e))
    try:
        with _planes_enabled(True):
            from TEX_Wrangle.tex_io import BufferDesc
            pv = _PlanesValue(_pv().planes, descs={"diffuse": BufferDesc("float16", "srgb")})
            meta = {"beauty": BufferMeta(premult="opaque"), "other": BufferMeta("linear")}
            out = _expand_plane_meta(meta, {"beauty": pv, "other": make_img(1, 4, 4, 3)})
            assert out["other"] is meta["other"] and out["beauty"] is meta["beauty"]
            assert out["beauty.diffuse"].colorspace == "srgb", "the plane's own desc wins"
            assert out["beauty.diffuse"].premult == "opaque", "the wire's premult is kept"
            assert out["beauty.Z"] is meta["beauty"], "no desc: the wire's tag verbatim"
            # a tag the host filed under the plane itself is never overwritten
            out = _expand_plane_meta({"beauty.diffuse": BufferMeta("oklab")}, {"beauty": pv})
            assert out["beauty.diffuse"].colorspace == "oklab"
            # no PlanesValue: the same dict object back
            assert _expand_plane_meta(meta, {"other": make_img(1, 4, 4, 3)}) is meta
        r.ok("_expand_plane_meta: descs win, wire tag fans out, host per-plane tag kept, identity without planes")
    except Exception as e:
        r.fail("_expand_plane_meta", str(e))


def test_e6003_names_the_slot_for_a_plane_read(r: SubTestResult):
    print("\n--- DATA-6 L-C2: E6003 for `p@beauty.diffuse` names slot 'beauty' (A18) ---")
    try:
        with _planes_enabled(True):
            try:
                tex_engine.cook("@OUT = vec4(p@beauty.diffuse.rgb, 1.0);", {"img": make_img(1, 4, 4, 3)},
                                device_mode="cpu")
                raise AssertionError("cooked with nothing on `beauty`")
            except InterpreterError as e:
                assert _code_of(e) == "E6003", _code_of(e)
                msg = str(e)
                assert "no input is connected to slot 'beauty'" in msg, msg
                assert "(read as @beauty.diffuse)" in msg, msg
                assert "slot 'beauty.diffuse'" not in msg, msg
        r.ok("the slot is the wire, the read is shown beside it")
    except Exception as e:
        r.fail("E6003 A18", str(e))
    try:
        # an ordinary unconnected wire keeps the exact message it always had
        try:
            tex_engine.cook("@OUT = vec4(@A.rgb, 1.0);", {}, device_mode="cpu")
            raise AssertionError("cooked")
        except InterpreterError as e:
            assert str(e).endswith("no input is connected to slot 'A'."), str(e)
        r.ok("a bare wire's E6003 message is byte-identical to before")
    except Exception as e:
        r.fail("E6003 unchanged", str(e))


def test_fusion_refuses_a_dotted_export(r: SubTestResult):
    print("\n--- DATA-6 L-C2: a fused chain refuses a dotted export (A1) ---")
    try:
        img = make_img(1, 4, 4, 3)
        stages = [{"code": "@OUT = @A * 0.5;", "exports": ["beauty.diffuse"], "bindings": {"A": img}},
                  {"code": "@OUT = @P + @Q;", "chain_inputs": {"P": [0, "OUT"], "Q": [0, "beauty.diffuse"]},
                   "bindings": {}}]
        for on in (False, True):
            with _planes_enabled(on):
                try:
                    tex_fusion.compile_fused(stages, infer_binding_type)
                    raise AssertionError("fused a dotted export")
                except tex_fusion.FusionError as e:
                    assert "@beauty.diffuse" in str(e) and "plane" in str(e), str(e)
        r.ok("FusionError names the export and calls it a plane write, both profiles")
    except Exception as e:
        r.fail("fusion refusal", str(e))
    try:
        # a whole-wire export still fuses exactly as before
        img = make_img(1, 4, 4, 3)
        stages = [{"code": "@OUT = @A * 0.5; @extra = @A * 2.0;", "exports": ["extra"], "bindings": {"A": img}},
                  {"code": "@OUT = @P + @Q * 0.25;", "chain_inputs": {"P": [0, "OUT"], "Q": [0, "extra"]},
                   "bindings": {}}]
        out = tex_engine.cook_stage_list(stages)["OUT"]
        assert (out - (img * 0.5 + img * 2.0 * 0.25)).abs().max().item() < 1e-6
        r.ok("an identifier export fuses unchanged")
    except Exception as e:
        r.fail("identifier export", str(e))


# ── invisibility, and the deferred write ──────────────────────────────────────

def test_planes_are_invisible_under_the_comfy_profile(r: SubTestResult):
    print("\n--- DATA-6 L-C2: under the ComfyUI profile nothing about planes exists (Q2, invariant #7) ---")
    try:
        assert not planes_wires_enabled() and not array_wires_enabled(), "profile leaked ON"
        try:
            _PlanesValue({"diffuse": torch.rand(1, 4, 4, 3)})
            raise AssertionError("constructed a PlanesValue under comfy")
        except ValueError as e:
            assert "engine egress profile" in str(e), str(e)
        r.ok("comfy: a PlanesValue cannot be constructed")
    except Exception as e:
        r.fail("comfy construct", str(e))
    try:
        with _planes_enabled(True):
            pv = _pv()
        # built under engine, then the profile drops: no arm types it, and the terminal names the switch
        try:
            infer_binding_type(pv)
            raise AssertionError("typed a PlanesValue under comfy")
        except InterpreterError as e:
            assert _code_of(e) == "E7005" and "engine egress profile" in _hint_of(e), _hint_of(e)
        bindings = {"beauty": pv}
        assert expand_plane_bindings(bindings, DIFFUSE_ONLY) is bindings, \
            "comfy: expansion must not touch the dict"
        try:
            tex_engine.cook(DIFFUSE_ONLY, {"beauty": pv}, device_mode="cpu")
            raise AssertionError("cooked a plane wire under comfy")
        except InterpreterError as e:
            assert _code_of(e) == "E7005", _code_of(e)
        r.ok("comfy: infer -> E7005 (hint names the profile); expand -> identity; cook -> E7005")
    except Exception as e:
        r.fail("comfy typing", str(e))
    try:
        # the checker's PLANES arms under comfy: a clear error, never a crash (L-B escalation 3)
        for src, code in (("@OUT = @beauty;", "E3203"),
                          ("@OUT = vec4(@beauty.rgb, 1.0);", "E3300"),
                          ("@OUT = vec4(@beauty.diffuse, 1.0);", "E3300")):
            diags = tex_api.check(src, {"beauty": TEXType.PLANES})
            codes = [d.code for d in diags]
            assert code in codes and "E0000" not in codes, (src, codes)
        with _planes_enabled(True):
            codes = [d.code for d in tex_api.check("@OUT = @beauty; // on\n", {"beauty": TEXType.PLANES})]
            assert "E3203" in codes and "E0000" not in codes, codes
            codes = [d.code for d in tex_api.check("@OUT = vec4((@beauty).r, 1.0); // on\n",
                                                   {"beauty": TEXType.PLANES})]
            assert "E3300" in codes and "E0000" not in codes, codes
        r.ok("E3203 on `@OUT = @beauty`, E3300 on a channel of the wire — both profiles, no E0000")
    except Exception as e:
        r.fail("checker arms", str(e))
    try:
        # the ARRAY-parity gate test's idiom: the egress profile flips both, and restores both
        tex_marshalling.set_egress_profile("engine")
        try:
            assert planes_wires_enabled()
            pv = _pv()
            assert infer_binding_type(pv) is TEXType.PLANES
        finally:
            tex_marshalling.set_egress_profile("comfy")
        assert not planes_wires_enabled()
        r.ok("set_egress_profile('engine') is the one switch; 'comfy' restores it")
    except Exception as e:
        r.fail("egress switch", str(e))


def test_plane_write_is_a_compile_error_naming_the_deferral(r: SubTestResult):
    print("\n--- DATA-6 L-C2: `@OUT.diffuse = ...` is refused and the hint names the deferral ---")
    try:
        for on in (False, True):
            with _planes_enabled(on):
                diags = tex_api.check("@OUT.diffuse = vec3(1.0);" + ("// on\n" if on else ""), {})
                errs = [d for d in diags if d.severity == "error"]
                assert errs and errs[0].code == "E3302", [(d.code, d.message) for d in diags]
                assert "plane write" in errs[0].hint and "not supported" in errs[0].hint, errs[0].hint
                try:
                    get_cache().compile_tex("@OUT.diffuse = vec3(1.0);" + ("// on2\n" if on else "//2\n"), {})
                    raise AssertionError("compiled a plane write")
                except Exception as e:
                    assert "E3302" in str(e) or "swizzle" in str(e), str(e)
        r.ok("E3302 whose hint says plane wires are read-only in this version, both profiles")
    except Exception as e:
        r.fail("plane write deferred", str(e))


# ── the harness, and the mutations ────────────────────────────────────────────

def test_harness_prepares_a_plane_program(r: SubTestResult):
    print("\n--- DATA-6 L-C2: `_prepare_example` builds a PlanesValue and lets the seam expand it ---")
    code = ("vec3 lit = p@beauty.diffuse + @beauty.specular;\n"
            "float k = luma(@img);\n"
            "@OUT = vec4(lit * @beauty.Z * k, 1.0);\n")
    try:
        assert not planes_wires_enabled()
        program, bindings, type_map, outs = ti._prepare_example(code, 1, 8, 8)
        assert not planes_wires_enabled(), "the harness must restore the profile"
        assert set(bindings) == {"beauty.diffuse", "beauty.specular", "beauty.Z", "img"}, sorted(bindings)
        # a 1-channel plane arrives in the [B,H,W] mask shape, the FLOAT-binding convention
        assert bindings["beauty.Z"].shape == (1, 8, 8), bindings["beauty.Z"].shape
        assert bindings["beauty.diffuse"].shape == (1, 8, 8, 3), bindings["beauty.diffuse"].shape
        assert tuple(bindings["img"].shape[:3]) == (1, 8, 8), bindings["img"].shape
        assert outs == ["OUT"]
        res = Interpreter().execute(program, dict(bindings), type_map, device="cpu", output_names=outs)
        assert res["OUT"].shape == (1, 8, 8, 4)
        r.ok("expanded rows only (no `beauty`, no unread plane), cooks on the CPU interpreter, profile restored")
    except Exception as e:
        r.fail("harness plane program", str(e))
    try:
        hints, _ = ti._collect_binding_hints(parse_and_split("@OUT = vec4(p@beauty.diffuse, 1.0);", {}))
        assert hints == {"beauty": TEXType.PLANES}, hints
        with _planes_enabled(True):
            hints, _ = ti._collect_binding_hints(
                parse_and_split("@OUT = vec4(p@beauty.diffuse, 1.0); // on\n", {"beauty": TEXType.PLANES}))
            assert hints == {"beauty": TEXType.PLANES}, hints
            pv = ti._make_dummy_binding(TEXType.PLANES, B=2, H=4, W=4)
            assert sorted(pv.planes) == ["Z", "diffuse", "specular"], sorted(pv.planes)
            assert pv.planes["Z"].shape == (2, 4, 4, 1) and pv.planes["diffuse"].shape == (2, 4, 4, 3)
        r.ok("hint filed under the BASE on both profiles; the dummy is diffuse/specular/Z")
    except Exception as e:
        r.fail("harness pieces", str(e))
    try:
        # a swizzle program takes exactly the path it always took (the twelve dotted examples)
        program, bindings, type_map, outs = ti._prepare_example(
            "float g = @image.g; @OUT = vec4(@image.rgb * g, 1.0);", 1, 8, 8)
        assert set(bindings) == {"image"}, sorted(bindings)
        r.ok("a swizzle program still discovers its wire under its own name")
    except Exception as e:
        r.fail("harness swizzle path", str(e))


def test_expansion_mutations(r: SubTestResult):
    print("\n--- DATA-6 L-C2: mutations, both directions (demand set, base removal, collision set) ---")
    with _planes_enabled(True):
        # demand set: the rows above assert `Z` is ABSENT when unread; a mutant that expands
        # every declared plane makes it PRESENT — and the assertion the rows make must catch it.
        try:
            real = tex_marshalling._plane_demand
            try:
                tex_marshalling._plane_demand = lambda code, base: sorted(("diffuse", "specular", "Z"))
                out = expand_plane_bindings({"beauty": _pv()}, DIFFUSE_ONLY)
                assert "beauty.Z" in out, "mutant did not fire"
            finally:
                tex_marshalling._plane_demand = real
            out = expand_plane_bindings({"beauty": _pv()}, DIFFUSE_ONLY)
            assert "beauty.Z" not in out
            r.ok("demand set: the all-planes mutant is caught by the laziness assertion; restored")
        except Exception as e:
            r.fail("demand mutant", str(e))
        # base removal: a mutant expansion that KEEPS the base row (applied at the engine's own
        # bound name, since prepare() re-expands anything left in the dict) leaves a PlanesValue
        # for the interpreter to ingest — refused loudly, never a wrong picture. The rows above
        # assert `beauty` is absent, so the mutant is caught there too.
        try:
            out = expand_plane_bindings({"beauty": _pv(), "img": make_img(1, 4, 4, 3)}, LIT)
            assert "beauty" not in out and "img" in out
            real = tex_engine._expand_plane_bindings

            def keep_base(b, code, **kw):
                from TEX_Wrangle.tex_marshalling import PlanesValue
                return {**real(b, code, **kw),
                        **{n: v for n, v in b.items() if v.__class__ is PlanesValue}}

            tex_engine._expand_plane_bindings = keep_base
            try:
                try:
                    tex_engine.cook(LIT, {"beauty": _pv()}, device_mode="cpu")
                    r.fail("base mutant", "a kept base row was cooked")
                except (InterpreterError, TypeError, RuntimeError) as e:
                    r.ok(f"base removal: a kept base row is refused loudly ({type(e).__name__})")
            finally:
                tex_engine._expand_plane_bindings = real
            assert tex_engine.cook(LIT, {"beauty": _pv()}, device_mode="cpu").outputs["OUT"].shape \
                == (1, 8, 8, 4), "restored"
        except Exception as e:
            r.fail("base removal", str(e))
        # collision set, both ways: the real set refuses `rgb` and admits `RGB`; an emptied
        # set admits `rgb` (so the E3304 row is load-bearing on the set), and a set that
        # wrongly included `Z` would refuse the commonest plane name.
        try:
            real = tex_marshalling._PLANE_COLLISION
            try:
                tex_marshalling._PLANE_COLLISION = frozenset()
                out = expand_plane_bindings({"beauty": _pv(rgb=torch.rand(1, 8, 8, 3))},
                                            "@OUT = vec4(@beauty.rgb, 1.0);")
                assert "beauty.rgb" in out, "emptied-set mutant did not fire"
                tex_marshalling._PLANE_COLLISION = real | {"Z"}
                try:
                    expand_plane_bindings({"beauty": _pv()}, DIFFUSE_ONLY)
                    raise AssertionError("a set containing `Z` did not refuse")
                except InterpreterError as e:
                    assert _code_of(e) == "E3304"
            finally:
                tex_marshalling._PLANE_COLLISION = real
            out = expand_plane_bindings({"beauty": _pv(RGB=torch.rand(1, 8, 8, 3))},
                                        "@OUT = vec4(@beauty.RGB, 1.0);")
            assert "beauty.RGB" in out
            assert tex_marshalling._PLANE_COLLISION == frozenset(CHANNEL_MAP) | frozenset(VALID_SWIZZLES)
            r.ok("collision set: emptied -> `rgb` admitted; +Z -> `Z` refused; real -> `RGB` admitted")
        except Exception as e:
            r.fail("collision mutant", str(e))
    assert not planes_wires_enabled()
