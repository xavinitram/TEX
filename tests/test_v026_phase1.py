"""
v0.26 Phase 1 — "Tools" (the bundling promise).

TOOL-1  the .textool manifest + loader: schema validation BEFORE any compile, promoted-param
        mapping (external widget -> a $param in one stage), single-stage + fused cook paths.
TOOL-3  tool = compilation unit: warm keys re-fingerprinted at install from the inline code
        (never carried in the file — fingerprints are unstable across TEX versions, ENG-5).
TOOL-4  the `tex build` CLI (validate + type-check + report; validate-only by default).
TOOL-5  a downloaded .textool is untrusted input to a code generator — the emitter injection
        audit, pinned by an adversarial-AST fuzz lane; schema-first validation; resource limits.
STOCK   Grade / Blur / Merge / Vignette as shipped .textool exemplars, + a fused composite.
LANG-7  the tex_lsp.py stdio LSP over check() + the registry (diagnostics / completion / hover).

Release exit gate (roadmap §9): a .textool round-trips author -> publish -> fresh-install ->
cook, BIT-IDENTICAL to the unfused graph (test_tool_roundtrip_unfused). CPU-pinned for
determinism; CUDA looped when present.
"""
import ast
import hashlib
import json
import os
import re
import tempfile

from helpers import *  # noqa: F401,F403  (SubTestResult, torch, make_img)
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib  # noqa: F401  (populates REGISTRY)
from TEX_Wrangle import tex_engine, tex_tool, tex_lsp
from TEX_Wrangle.tex_tool import TEXToolError
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_runtime.codegen import _CodeGen
from TEX_Wrangle.tex_marshalling import infer_binding_type
from TEX_Wrangle import tex_api

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]
_STOCK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock")


def _stock(name):
    return tex_tool.load_tool(os.path.join(_STOCK, name + ".textool"))


# ── TOOL-1 + STOCK: the release exit gate ──────────────────────────────────────
def test_tool_roundtrip_unfused(r: SubTestResult):
    print("\n--- TOOL-1: a fused tool cooks BIT-IDENTICAL to the unfused graph (exit gate) ---")
    for dev in _DEVICES:
        try:
            m = _stock("grade_vignette")
            img = torch.rand(1, 24, 24, 3)
            pv = {"gain": 1.3, "gamma": 2.2, "strength": 1.4}
            fused = tex_tool.cook_tool(m, {"image": img}, pv, device_mode=dev).outputs["OUT"]
            # unfused: cook stage 0 (grade), feed its @OUT into the terminal (vignette).
            s0 = m.graphspec["stages"][0]["code"]
            out0 = tex_engine.cook(s0, {"image": img, "gain": 1.3, "gamma": 2.2},
                                   device_mode=dev).outputs["OUT"]
            out1 = tex_engine.cook(m.terminal_code, {"image": out0, "strength": 1.4},
                                   device_mode=dev).outputs["OUT"]
            maxdiff = (fused.float() - out1.float()).abs().max().item()
            assert maxdiff < 1e-5, f"fused vs unfused maxdiff={maxdiff}"
            r.ok(f"[{dev}] fused tool == unfused graph (maxdiff={maxdiff:.2e})")
        except Exception as e:
            r.fail(f"tool roundtrip [{dev}]", str(e))


def test_tool_stock_exemplars(r: SubTestResult):
    print("\n--- STOCK: every shipped .textool loads, preflights clean, and cooks ---")
    img = torch.rand(1, 16, 16, 3)
    inputs = {"image": img, "A": img, "B": torch.rand(1, 16, 16, 3)}
    try:
        stems = sorted(f[:-8] for f in os.listdir(_STOCK) if f.endswith(".textool"))
        assert {"grade", "blur", "merge", "vignette", "grade_vignette"} <= set(stems), stems
        for stem in stems:
            m = _stock(stem)
            pf = tex_tool.preflight_tool(m)
            assert pf["ok"], f"{stem} preflight failed: {pf['diagnostics']}"
            need = {i["name"]: inputs[i["name"]] for i in m.inputs}
            res = tex_tool.cook_tool(m, need, {}, device_mode="cpu")
            assert res.outputs, f"{stem} produced no outputs"
        r.ok(f"all {len(stems)} stock tools load+preflight+cook: {', '.join(stems)}")
    except Exception as e:
        r.fail("stock exemplars", str(e))


def test_tool_manifest_keys(r: SubTestResult):
    print("\n--- TOOL-1 canary: the manifest + promoted-param key set (a host contract) ---")
    try:
        m = _stock("grade")
        d = m.to_dict()
        need = {"manifest_schema", "name", "tool_version", "tex_language", "min_engine",
                "category", "context", "doc", "author", "inputs", "outputs",
                "promoted_params", "code"}
        assert need <= set(d), f"missing manifest keys: {need - set(d)}"
        pk = set(d["promoted_params"][0])
        assert pk == {"name", "internal", "stage", "type", "default", "metadata"}, pk
        assert d["outputs"] == [{"name": "OUT", "type": "IMAGE"}], d["outputs"]
        # a multi-output tool declares its ports (unblocks instancing)
        vo = _stock("vignette").to_dict()["outputs"]
        assert {o["name"] for o in vo} == {"darkened", "vignette_mask"}, vo
        # fused manifest carries the graphspec form instead of code
        fd = _stock("grade_vignette").to_dict()
        assert {"graphspec", "terminal_code", "terminal_image_input"} <= set(fd), set(fd)
        assert "code" not in fd
        r.ok("manifest + promoted-param key sets are stable")
    except Exception as e:
        r.fail("manifest keys", str(e))


def test_tool_metadata_tooltip_options(r: SubTestResult):
    print("\n--- TOOL-1: metadata 'tooltip' + labelled-choice 'options' (host UI hints) ---")
    try:
        base = {"manifest_schema": 1, "name": "X", "tex_language": "0.23",
                "code": "@OUT = @image;", "inputs": [{"name": "image", "type": "IMAGE"}]}
        # tooltip: a plain string, round-trips through to_dict()/tool_summary().
        m = tex_tool.load_tool({**base, "promoted_params": [
            {"name": "gamma", "type": "f", "default": 1.0,
             "metadata": {"min": 0.0, "max": 4.0, "tooltip": "Power curve."}}]})
        assert m.promoted_params[0].metadata["tooltip"] == "Power curve."
        d = m.to_dict()["promoted_params"][0]["metadata"]
        assert d["tooltip"] == "Power curve.", d
        s = tex_tool.tool_summary(m)["widgets"][0]["metadata"]
        assert s["tooltip"] == "Power curve.", s

        # options: a labelled-choice list, only on an 'i' param, round-trips the same way.
        m2 = tex_tool.load_tool({**base, "promoted_params": [
            {"name": "channel", "type": "i", "default": 0,
             "metadata": {"min": 0, "max": 2, "step": 1, "options": ["Red", "Green", "Blue"]}}]})
        d2 = m2.to_dict()["promoted_params"][0]["metadata"]
        assert d2["options"] == ["Red", "Green", "Blue"], d2
        s2 = tex_tool.tool_summary(m2)["widgets"][0]["metadata"]
        assert s2["options"] == ["Red", "Green", "Blue"], s2

        # a real-world-shaped combo widget: default 0, min 0, max n-1, step 1, n labelled
        # options -- the common case a labelled-choice widget takes -- is accepted.
        n = 11
        tex_tool.load_tool({**base, "promoted_params": [
            {"name": "mode", "type": "i", "default": 0,
             "metadata": {"min": 0, "max": n - 1, "step": 1,
                          "options": [f"opt{k}" for k in range(n)]}}]})
        r.ok("tooltip + options accepted, round-trip through to_dict()/tool_summary()")
    except Exception as e:
        r.fail("metadata tooltip/options", str(e))


def test_tool_input_optional(r: SubTestResult):
    print("\n--- TOOL-1: inputs[*].optional (host UI hint; TEX binds nothing either way) ---")
    try:
        base = {"manifest_schema": 1, "name": "X", "tex_language": "0.23",
                "code": "@OUT = @image;", "promoted_params": []}
        m = tex_tool.load_tool({**base, "inputs": [
            {"name": "image", "type": "IMAGE"},
            {"name": "mask", "type": "MASK", "optional": True}]})
        assert m.inputs[1]["optional"] is True and "optional" not in m.inputs[0], m.inputs
        d = m.to_dict()["inputs"]
        assert d[1]["optional"] is True and "optional" not in d[0], d
        s = tex_tool.tool_summary(m)["inputs"]
        assert s[1]["optional"] is True and "optional" not in s[0], s
        path = tex_tool.write_tool(m, tempfile.mkdtemp())
        reloaded = tex_tool.load_tool(path)
        assert reloaded.inputs[1]["optional"] is True, reloaded.inputs

        # optional: false round-trips as WRITTEN (not dropped for being falsy).
        m2 = tex_tool.load_tool({**base, "inputs": [
            {"name": "image", "type": "IMAGE"},
            {"name": "mask", "type": "MASK", "optional": False}]})
        assert m2.inputs[1]["optional"] is False, m2.inputs

        # an absent key stays absent (byte-identity with an old manifest).
        m3 = tex_tool.load_tool({**base, "inputs": [{"name": "image", "type": "IMAGE"}]})
        assert "optional" not in m3.to_dict()["inputs"][0], m3.to_dict()["inputs"]

        # a declined key ('extent') on an input keeps silently dropping -- nothing here
        # wires it to anything; it stays unrecognised, same as any other unknown inputs[*] key.
        m4 = tex_tool.load_tool({**base, "inputs": [
            {"name": "image", "type": "IMAGE", "extent": "own"}]})
        assert "extent" not in m4.to_dict()["inputs"][0], m4.to_dict()["inputs"]

        r.ok("inputs[*].optional round-trips true/false; absent key and 'extent' stay unrecognised")
    except Exception as e:
        r.fail("input optional", str(e))


def test_tool_promoted_params(r: SubTestResult):
    print("\n--- TOOL-1 derivation: promoted values land in the right stage; omitted -> default ---")
    try:
        m = _stock("grade")
        img = torch.rand(1, 8, 8, 3)
        # explicit gamma vs its default (1.0): a change must move pixels
        d_default = tex_tool.cook_tool(m, {"image": img}, {}, device_mode="cpu").outputs["OUT"]
        d_default2 = tex_tool.cook_tool(m, {"image": img}, {"gamma": 1.0}, device_mode="cpu").outputs["OUT"]
        d_changed = tex_tool.cook_tool(m, {"image": img}, {"gamma": 0.4}, device_mode="cpu").outputs["OUT"]
        assert torch.equal(d_default, d_default2), "omitted param did not fall back to its default"
        assert not torch.equal(d_default, d_changed), "a promoted-param change did not move pixels"
        # fused: a promoted value must reach the correct stage's bindings
        mgv = _stock("grade_vignette")
        gs, tb = tex_tool._fused_cook_inputs(mgv, img, {"gain": 2.0, "strength": 0.3})
        assert gs["stages"][0]["params"]["gain"] == 2.0, "stage-0 promoted value not applied"
        assert tb["strength"] == 0.3, "terminal promoted value not applied"
        r.ok("promoted params resolve to the right stage, defaults included")
    except Exception as e:
        r.fail("promoted params", str(e))


# ── TOOL-3: warm keys (re-derived at install, never stored) ─────────────────────
def test_tool_warm_keys(r: SubTestResult):
    print("\n--- TOOL-3: warm keys re-derive from inline code; install is validate-only ---")
    try:
        single = tex_tool.tool_warm_keys(_stock("grade"))
        fused = tex_tool.tool_warm_keys(_stock("grade_vignette"))
        assert single and single[0], "single-stage warm key empty"
        assert fused and fused[0].startswith("fused_"), f"fused warm key wrong: {fused}"
        # no fingerprint is ever stored in the manifest (ENG-5)
        raw = _stock("grade").to_dict()
        blob = str(raw)
        assert single[0] not in blob, "a fingerprint leaked into the manifest"
        # install (validate-only default) writes without compiling
        import tempfile
        dest = tempfile.mkdtemp()
        info = tex_tool.install_tool(_stock("grade"), dest, warm=False)
        assert info["ok"] and os.path.exists(info["path"]), info
        assert info["warm_keys"] == [], "validate-only install should not derive warm keys"

        # The warm key must equal what a DEFAULT-path cook actually FINGERPRINTS — the engine
        # keys on infer_binding_type(RAW value) (tex_engine.prepare, before _convert_param_value),
        # so a param's warm type must be inferred from its raw default, NOT its semantic hint. A
        # param whose JSON default serialized as an int (an `f` slider saved as `1`) is the trap:
        # the cook keys it INT; a semantic-typed warm key (FLOAT) would silently never be hit.
        from TEX_Wrangle.tex_cache import TEXCache
        for default in (1, 1.0):                 # int-serialized AND float default
            mt = tex_tool.load_tool({"manifest_schema": 1, "name": "WK", "tex_language": "0.23",
                  "code": "f$s = 1.0;\n@OUT = @image * $s;",
                  "inputs": [{"name": "image", "type": "IMAGE"}],
                  "promoted_params": [{"name": "s", "internal": "s", "type": "f", "default": default}]})
            wk = tex_tool.tool_warm_keys(mt)
            bt = {"image": infer_binding_type(torch.zeros(1, 8, 8, 3)),
                  "s": infer_binding_type(default)}     # exactly what a default-path cook infers
            cook_fp = TEXCache.fingerprint(mt.code, bt)
            assert cook_fp in wk, f"default={default!r}: cook fingerprint not in warm keys {wk}"
        r.ok("warm keys re-derived (not stored), validate-only install, and match the cook fingerprint")
    except Exception as e:
        r.fail("warm keys", str(e))


# ── audit#5 regression pins (warm / preflight / exemplar / validation seams) ────
def test_tool_audit5_fixes(r: SubTestResult):
    print("\n--- audit#5: warm actually compiles, preflight is typed, Merge composites ---")
    try:
        # (1) install_tool(warm=True) actually materializes a codegen fn (was a dead prewarm call).
        # >= 1: an IMAGE tool now warms both the RGB and RGBA channel variant (audit#6 #8), so a
        # tool with an image input materializes 2 codegen fns; a channel-independent tool, 1.
        info = tex_tool.install_tool(_stock("grade"), tempfile.mkdtemp(), warm=True, device="cpu")
        assert info.get("warmed", {}).get("codegen", 0) >= 1, f"warm did not compile: {info}"
        assert not any("warm-compile skipped" in w for w in info["warnings"]), info["warnings"]
        # (2) single-stage preflight is TYPE-AWARE: @image.a on a VEC3 IMAGE input must fail
        bad = tex_tool.load_tool({"manifest_schema": 1, "name": "B", "tex_language": "0.23",
              "code": "@OUT = vec4(@image.a);", "inputs": [{"name": "image", "type": "IMAGE"}],
              "promoted_params": []})
        assert not tex_tool.preflight_tool(bad)["ok"], "type-blind preflight false-passed @image.a"
        # (3) Merge ops 4 (overlay) and 5 (soft-light) actually composite (were pass-through)
        m = _stock("merge")
        A = torch.rand(1, 8, 8, 3); B = torch.rand(1, 8, 8, 3)
        for op in (4, 5):
            out = tex_tool.cook_tool(m, {"A": A, "B": B}, {"operation": op}, device_mode="cpu").outputs["OUT"]
            assert not torch.allclose(out, B, atol=1e-4), f"Merge op {op} is a no-op"
        # (4) fused cook with a missing source raises TEXToolError, not a bare KeyError
        try:
            tex_tool.cook_tool(_stock("grade_vignette"), {}, {}, device_mode="cpu")
            assert False, "missing fused source did not raise"
        except tex_tool.TEXToolError:
            pass
        r.ok("warm compiles, preflight is typed, Merge composites, fused-source error is clean")
    except Exception as e:
        r.fail("audit5 fixes", str(e))


# ── audit#6: the post-release audit fixes (warm typing, fused source, robustness caps) ──
def test_tool_audit6_fixes(r: SubTestResult):
    print("\n--- audit#6: warm-key typing, fused terminal_image_input, robustness caps ---")
    try:
        from TEX_Wrangle.tex_cache import TEXCache
        from TEX_Wrangle.tex_compiler.type_checker import BINDING_HINT_TYPES

        # (#1/#7) _hint_value infers to the canonical hint→type map (bool→INT, vectors→VEC*),
        # so a promoted param keys as the type its widget produces (was: b/c/v* all → FLOAT).
        for h in ("f", "i", "s", "b", "c", "v2", "v3", "v4"):
            got = infer_binding_type(tex_tool._hint_value(h, 0.0))
            assert got == BINDING_HINT_TYPES[h], f"_hint_value[{h}] → {got.name}, want {BINDING_HINT_TYPES[h].name}"

        # (#1) a bool-param tool's warm key MATCHES the fingerprint the cook computes (else the
        # warmed artifact is never found and TOOL-3 is a silent no-op for the whole param class).
        bt = tex_tool.load_tool({"manifest_schema": 1, "name": "BT", "tex_language": "0.23",
             "code": "b$inv=0;\n@OUT = $inv > 0 ? vec4(1.0-@image.rgb, 1.0) : vec4(@image.rgb, 1.0);",
             "inputs": [{"name": "image", "type": "IMAGE"}],
             "promoted_params": [{"name": "inv", "internal": "inv", "type": "b", "default": False}]})
        wk = tex_tool.tool_warm_keys(bt)
        cook3 = TEXCache.fingerprint(bt.code, {"image": infer_binding_type(torch.zeros(1, 4, 4, 3)),
                                               "inv": infer_binding_type(False)})
        cook4 = TEXCache.fingerprint(bt.code, {"image": infer_binding_type(torch.zeros(1, 4, 4, 4)),
                                               "inv": infer_binding_type(False)})
        assert cook3 in wk, "bool-param warm key misses the RGB cook fingerprint"
        assert cook4 in wk, "warm keys don't cover the RGBA (VEC4) variant"   # (#8)

        # (#3) a fused graphspec that omits terminal_image_input (region_to_collapse_plan does)
        # still cooks: _fused_cook_inputs makes the manifest field authoritative on the copy.
        f = tex_tool.load_tool({"manifest_schema": 1, "name": "F2", "tex_language": "0.23",
            "graphspec": {"schema": 1, "stages": [{"code": "@OUT=@image;", "image_input": "image",
                          "params": {}}]},   # NO terminal_image_input inside the graphspec
            "terminal_code": "@OUT=@image;", "terminal_image_input": "image",
            "inputs": [{"name": "image", "type": "IMAGE"}], "promoted_params": []})
        gs, _term = tex_tool._fused_cook_inputs(f, torch.zeros(1, 4, 4, 3), None)
        assert gs.get("terminal_image_input") == "image", "fused gs missing authoritative source key"
        assert len(tex_tool._assemble_fused_stages(f, torch.zeros(1, 4, 4, 3), {})) >= 2

        # (#2) a linear fused stage without image_input is rejected at validate (was a raw KeyError
        # at cook, via a publish path that never preflights).
        try:
            tex_tool.validate_manifest({"name": "B", "tex_language": "0.23",
                "graphspec": {"schema": 1, "stages": [{"code": "@OUT=@image;", "params": {}}],
                              "terminal_image_input": "image"},
                "terminal_code": "@OUT=@image;", "terminal_image_input": "image",
                "inputs": [{"name": "image", "type": "IMAGE"}]})
            assert False, "linear stage missing image_input was not rejected"
        except TEXToolError as e:
            assert "image_input" in str(e)

        # (#4) an oversized manifest is rejected on the dict/write path, not only on file-load.
        try:
            tex_tool.validate_manifest({"name": "X", "tex_language": "0.23", "code": "@OUT=@image;",
                "inputs": [{"name": "image", "type": "IMAGE"}], "promoted_params": [],
                "junk": "z" * (tex_tool.MAX_TOOL_BYTES + 8)})
            assert False, "oversized manifest was not rejected"
        except TEXToolError:
            pass

        # (#21) validate_manifest returns its parsed lists (consumed by load_tool; validated once).
        parsed = tex_tool.validate_manifest({"name": "G", "tex_language": "0.23",
            "code": "f$s=1.0;\n@OUT=@image*$s;", "inputs": [{"name": "image", "type": "IMAGE"}],
            "promoted_params": [{"name": "s", "internal": "s", "type": "f", "default": 1.0}]})
        assert isinstance(parsed, dict) and len(parsed["promoted"]) == 1 and len(parsed["inputs"]) == 1

        r.ok("warm keys match the cook (bool/vector/RGBA), fused source is authoritative, "
             "linear/oversized manifests rejected, validate parses once")
    except Exception as e:
        r.fail("audit6 fixes", str(e))


# ── TOOL-5: schema rejects malformed manifests BEFORE any compile ───────────────
def test_tool_schema_rejects(r: SubTestResult):
    print("\n--- TOOL-5 canary: malformed / unsafe manifests are rejected pre-compile ---")
    base = {"manifest_schema": 1, "name": "X", "tex_language": "0.23",
            "code": "@OUT = @image;", "inputs": [{"name": "image", "type": "IMAGE"}],
            "promoted_params": []}
    fbase = {"manifest_schema": 1, "name": "F", "tex_language": "0.23",
             "graphspec": {"schema": 1, "stages": [{"code": "@OUT=@image;", "image_input": "image",
                           "params": {}}], "terminal_image_input": "image"},
             "terminal_code": "@OUT=@image;", "terminal_image_input": "image",
             "inputs": [{"name": "image", "type": "IMAGE"}], "promoted_params": []}
    cases = {
        "newer manifest_schema": {**base, "manifest_schema": 99},
        "both code+graphspec": {**base, "graphspec": {"stages": [{"code": "x"}]}},
        "code + non-dict graphspec": {**base, "graphspec": [1, 2, 3]},       # Sec3: is_fused agreement
        "neither code nor graphspec": {k: v for k, v in base.items() if k != "code"},
        "bad promoted type": {**base, "promoted_params": [{"name": "p", "type": "zzz"}]},
        "non-scalar metadata": {**base, "promoted_params":
                                [{"name": "p", "type": "f", "metadata": {"min": [1, 2]}}]},
        "unknown metadata key": {**base, "promoted_params":
                                 [{"name": "p", "type": "f", "metadata": {"evil": 1}}]},
        "bad context": {**base, "context": "malware"},
        "newer min_engine": {**base, "min_engine": "999.0.0"},
        # F5: single-stage promoted internal collides with an input / duplicates
        "internal collides with input": {**base, "promoted_params":
                                         [{"name": "p", "internal": "image", "type": "f"}]},
        "duplicate internal single-stage": {**base, "promoted_params":
            [{"name": "a", "internal": "g", "type": "f"}, {"name": "b", "internal": "g", "type": "f"}]},
        # F1/F3: fused-form validation gaps that used to CRASH cook/preflight
        "promoted stage out of range": {**fbase, "promoted_params":
                                        [{"name": "p", "internal": "p", "stage": 5, "type": "f"}]},
        "negative promoted stage": {**fbase, "promoted_params":
                                    [{"name": "p", "internal": "p", "stage": -1, "type": "f"}]},
        "bool promoted stage": {**fbase, "promoted_params":
                                [{"name": "p", "internal": "p", "stage": True, "type": "f"}]},
        "non-dict terminal_params": {**fbase, "terminal_params": "oops"},
        "fused >1 input": {**fbase, "inputs": [{"name": "image"}, {"name": "extra"}]},
        "terminal_image_input not an input": {**fbase, "terminal_image_input": "nope"},
        # new metadata keys: 'tooltip' (string, capped) and 'options' (labelled choices on an
        # 'i' param only, capped, cross-checked against default/min/max/step)
        "tooltip not a string": {**base, "promoted_params":
            [{"name": "p", "type": "f", "metadata": {"tooltip": 123}}]},
        "tooltip too long": {**base, "promoted_params":
            [{"name": "p", "type": "f",
              "metadata": {"tooltip": "x" * (tex_tool.MAX_TOOLTIP_CHARS + 1)}}]},
        "options on a float param": {**base, "promoted_params":
            [{"name": "p", "type": "f", "metadata": {"options": ["a", "b"]}}]},
        "options not a list": {**base, "promoted_params":
            [{"name": "p", "type": "i", "metadata": {"options": "a,b"}}]},
        "options empty": {**base, "promoted_params":
            [{"name": "p", "type": "i", "metadata": {"options": []}}]},
        "options non-str entry": {**base, "promoted_params":
            [{"name": "p", "type": "i", "metadata": {"options": ["a", 2]}}]},
        "options empty-str entry": {**base, "promoted_params":
            [{"name": "p", "type": "i", "metadata": {"options": ["a", ""]}}]},
        "options over MAX_OPTIONS": {**base, "promoted_params":
            [{"name": "p", "type": "i",
              "metadata": {"options": [str(k) for k in range(tex_tool.MAX_OPTIONS + 1)]}}]},
        "options entry over MAX_OPTION_CHARS": {**base, "promoted_params":
            [{"name": "p", "type": "i",
              "metadata": {"options": ["x" * (tex_tool.MAX_OPTION_CHARS + 1)]}}]},
        "options default out of range": {**base, "promoted_params":
            [{"name": "p", "type": "i", "default": 5, "metadata": {"options": ["a", "b"]}}]},
        "options default bool": {**base, "promoted_params":
            [{"name": "p", "type": "i", "default": True, "metadata": {"options": ["a", "b"]}}]},
        "options default float": {**base, "promoted_params":
            [{"name": "p", "type": "i", "default": 0.5, "metadata": {"options": ["a", "b"]}}]},
        "options contradicts min": {**base, "promoted_params":
            [{"name": "p", "type": "i", "metadata": {"min": 1, "options": ["a", "b", "c"]}}]},
        "options contradicts max": {**base, "promoted_params":
            [{"name": "p", "type": "i", "metadata": {"max": 5, "options": ["a", "b", "c"]}}]},
        "options contradicts step": {**base, "promoted_params":
            [{"name": "p", "type": "i", "metadata": {"step": 2, "options": ["a", "b", "c"]}}]},
        # new inputs[*] key: 'optional' must be a bool; a fused tool's sole source may not be one
        "input optional not a bool": {**base, "inputs":
            [{"name": "image", "type": "IMAGE", "optional": "yes"}]},
        "fused sole input optional": {**fbase, "inputs":
            [{"name": "image", "type": "IMAGE", "optional": True}]},
    }
    failed = []
    for label, raw in cases.items():
        try:
            tex_tool.load_tool(raw)
            failed.append(label)          # should have raised
        except TEXToolError:
            pass
        except Exception as e:
            failed.append(f"{label} (wrong exc {type(e).__name__})")
    if failed:
        r.fail("schema rejects", f"these were NOT rejected: {failed}")
    else:
        r.ok(f"all {len(cases)} malformed/unsafe manifests rejected with TEXToolError")


# ── byte-identity guard: a pre-existing manifest loads/serialises unchanged ─────────
# Every hash below was captured against tex_tool.py BEFORE 'tooltip'/'options'/'optional'
# existed, then re-checked (git stash) to read identically after -- none of the manifests
# below uses any of the three, so a change to to_dict()/tool_summary()/write_tool's shape
# for an OLD manifest, or a new key leaking into one, turns one of these red.
#
# `written_bytes` is LINE-ENDING-NORMALISED (CRLF -> LF before hashing), and so is every
# `written_bytes` pin in this file. write_tool writes its JSON through a TEXT-mode handle, so
# it emits the platform's newline: CRLF on Windows, LF everywhere else. Hashing the raw bytes
# pinned one platform's newline convention rather than the manifest, and the same unchanged
# tool read as drifted on Linux. Normalising loses nothing the pin is for: json.dump escapes
# any CR or LF inside a string value, so the only CRLFs in the file are the ones the handle
# wrote between lines, and every other byte (key order, indentation, separators, UTF-8
# content, the trailing newline or its absence) is still pinned exactly.
_MANIFEST_BASELINE_SHA256 = {
    "blur.textool": {
        "to_dict": "cc8d30cbf3d31a33dd7bb684dcae94561b4da9827fd459b4fd91c5f812135da4",
        "tool_summary": "37497076f938e2b7086c86f5f807d4658a8388594fbf06bd18bf8c999c22ca1b",
        "written_bytes": "e7ffcbc8017bcb6c409d4958ffe85ac995df8e16d6c7f90999cc800be20d5507",
    },
    "grade.textool": {
        "to_dict": "108748f0c2c3ceb926551a9188a4cc50096756f000ea098900a78f4a1733b07e",
        "tool_summary": "040db66ad91572cfe1d2557a02f7260fb0a45af714864fbb9599a45987573802",
        "written_bytes": "0ea64dc0227e4489a6eb5f67f811f452ee664750ffed3b616b0aa0158b2f48bd",
    },
    "grade_vignette.textool": {
        "to_dict": "cfd407ab512b35d0fdca53b1cd0fb29f31a487699210910c182e44d756a4f9db",
        "tool_summary": "783d1a2d1afa0cca7ac2937c4ded691beff331174ce95b155327e72616c3dc5e",
        "written_bytes": "d23dd4d7065b5ede02fb441faeaee46d3dcbd2144d1ad6a29bd7318e34b89ea6",
    },
    "merge.textool": {
        "to_dict": "52ebde0e7eda07b52b4c9fcec17820a525d24923208ab7666f795aff1879331d",
        "tool_summary": "f7e68d2ca334ac03e60d46b3f7096c562f7dfc2ded7f9a369219b1b1d5b261b0",
        "written_bytes": "a86aba76b00e2acc9efc148aa6cdba87ae8c5d27bf60b2894e6019a61e07f3d4",
    },
    "vignette.textool": {
        "to_dict": "74a8b0b0b8660dbee20989f3614a7294fb46e2fd9762517ee1a43a2c53430f34",
        "tool_summary": "11f9fb5e822646fd0e37d6352b2e7ffe4f871e9d300318864b26ea9694bab3f7",
        "written_bytes": "c69ab8e7746fdbcc3fe4ed8b8e5b1d09f653473d82d326d0ed00d65e4a6e4692",
    },
}
# Two representative pre-existing manifest shapes (single-stage / fused) -- neither uses any
# of the new keys -- pinned the same way as the stock exemplars above.
_OLD_SINGLE_MANIFEST = {"manifest_schema": 1, "name": "OldSingle", "tex_language": "0.23",
    "code": "f$s = 1.0;\n@OUT = @image * $s;",
    "inputs": [{"name": "image", "type": "IMAGE"}],
    "promoted_params": [{"name": "s", "internal": "s", "type": "f", "default": 1.0,
                          "metadata": {"min": 0.0, "max": 4.0, "label": "Scale"}}]}
_OLD_SINGLE_SHA256 = {"to_dict": "5b45f25b610e7d2721aafeb64de0845a2ce5c685fa0da07a4c100756c137a6aa",
                      "tool_summary": "87b30e1509b63f22b4706dfe3fa628d5ff6b5ce2c1933b37a50b3f4a879a8df6",
                      "written_bytes": "452eafa6871f6391d5cde7374e3bf972e142be586a648c2be3fa041d3df294e6"}
_OLD_FUSED_MANIFEST = {"manifest_schema": 1, "name": "OldFused", "tex_language": "0.23",
    "graphspec": {"schema": 1, "stages": [{"code": "@OUT=@image;", "image_input": "image",
                  "params": {}}], "terminal_image_input": "image"},
    "terminal_code": "@OUT=@image;", "terminal_image_input": "image",
    "inputs": [{"name": "image", "type": "IMAGE"}], "promoted_params": []}
_OLD_FUSED_SHA256 = {"to_dict": "664492abe15a48683a08ac5fa44d107eec568ab5aacff8a81a02b0396935ebc9",
                     "tool_summary": "492d0a579d93a0e332c4b949085fe53dac96db0e67855d0f39c3705eecb21653",
                     "written_bytes": "4c64f7ec2e5eb8ec83c39da2a42a7b625c4cbf1388ecf7f160aa24dec8c1353f"}


def _hash_manifest(m) -> dict:
    """sha256 of to_dict()/tool_summary() (canonical JSON) + the bytes write_tool writes, with
    its platform newlines normalised to LF -- one comparable fingerprint per serialisation path
    a host might rely on, identical on every OS (see the note above _MANIFEST_BASELINE_SHA256)."""
    d = m.to_dict()
    s = tex_tool.tool_summary(m)
    path = tex_tool.write_tool(m, tempfile.mkdtemp())
    with open(path, "rb") as fh:
        written = fh.read()
    return {
        "to_dict": hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest(),
        "tool_summary": hashlib.sha256(json.dumps(s, sort_keys=True).encode("utf-8")).hexdigest(),
        "written_bytes": hashlib.sha256(written.replace(b"\r\n", b"\n")).hexdigest(),
    }


def test_tool_manifest_byte_identity(r: SubTestResult):
    print("\n--- byte-identity guard: a pre-existing manifest loads/serialises unchanged ---")
    try:
        mismatches = []
        for fn, expect in _MANIFEST_BASELINE_SHA256.items():
            got = _hash_manifest(_stock(fn[:-8]))
            if got != expect:
                mismatches.append((fn, expect, got))
        for label, raw, expect in (("OLD_SINGLE", _OLD_SINGLE_MANIFEST, _OLD_SINGLE_SHA256),
                                    ("OLD_FUSED", _OLD_FUSED_MANIFEST, _OLD_FUSED_SHA256)):
            got = _hash_manifest(tex_tool.load_tool(dict(raw)))
            if got != expect:
                mismatches.append((label, expect, got))
        assert not mismatches, f"manifest serialisation drifted: {mismatches}"
        # no stock tool's summary carries the new 'optional' key (none declares it).
        for fn in _MANIFEST_BASELINE_SHA256:
            summ = tex_tool.tool_summary(_stock(fn[:-8]))
            assert all("optional" not in i for i in summ["inputs"]), summ["inputs"]
        r.ok(f"{len(_MANIFEST_BASELINE_SHA256) + 2} pre-existing manifests' "
             f"to_dict()/tool_summary()/written-bytes are unchanged; no stock summary "
             f"carries 'optional'")
    except Exception as e:
        r.fail("manifest byte-identity", str(e))


def test_tool_js_publish_filter_pin(r: SubTestResult):
    print("\n--- ComfyUI-path pin: the JS publish filter still forwards only the five "
          "original metadata keys ---")
    try:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        js_path = os.path.join(here, "js", "tex_extension.js")
        with open(js_path, encoding="utf-8") as fh:
            src = fh.read()
        m = re.search(r"const META_KEYS = new Set\(\[(.*?)\]\)", src)
        assert m, "META_KEYS literal not found in js/tex_extension.js"
        keys = set(re.findall(r'"(\w+)"', m.group(1)))
        assert keys == {"min", "max", "step", "precision", "label"}, (
            f"js/tex_extension.js META_KEYS changed to {sorted(keys)} -- forwarding "
            f"tooltip/options to ComfyUI is a deliberate, separate frontend change, not this one")
        r.ok("js/tex_extension.js still forwards exactly the five original metadata keys")
    except Exception as e:
        r.fail("JS publish filter pin", str(e))


# ── fused tools with fed inputs: `inputs[*].feeds` + the co-extent refusal ──────────────
# A fused tool may route inputs beyond its one source into named stage bindings. The oracle is the
# exit gate's: cook_tool equals cooking the stages one by one (invariant #2's spirit for tools).
def _merge_feed_manifest():
    """Merge-shaped: source -> Blur -> a Merge terminal reading the chain as @A and the tool's
    second external input as @B (fed into the terminal). The stage code is the stock tools' own."""
    return {"manifest_schema": 1, "name": "BlurMerge", "tex_language": "0.23",
            "graphspec": {"schema": 1, "stages": [{"code": _stock("blur").code,
                                                   "image_input": "image", "params": {}}]},
            "terminal_code": _stock("merge").code, "terminal_image_input": "A",
            "inputs": [{"name": "A", "type": "IMAGE"},
                       {"name": "B", "type": "IMAGE", "feeds": [["terminal", "B"]]}],
            "outputs": [{"name": "OUT", "type": "IMAGE"}],
            "promoted_params": [
                {"name": "sigma", "internal": "sigma", "stage": 0, "type": "f", "default": 2.0},
                {"name": "operation", "internal": "operation", "stage": "terminal", "type": "i",
                 "default": 0},
                {"name": "mix", "internal": "mix", "stage": "terminal", "type": "f", "default": 1.0}]}


_FEED_GRADE = "f$gain = 1.0;\n@OUT = @image * $gain;"
_FEED_PLATE = "@OUT = lerp(@image, @plate, 0.25);"
_FEED_TAIL = "@OUT = @image * 0.9;"


def _upstream_feed_manifest():
    """Linear, three stages; the second input feeds the MIDDLE (upstream) stage as @plate."""
    return {"manifest_schema": 1, "name": "PlateOver", "tex_language": "0.23",
            "graphspec": {"schema": 1, "stages": [
                {"code": _FEED_GRADE, "image_input": "image", "params": {}},
                {"code": _FEED_PLATE, "image_input": "image", "params": {}}]},
            "terminal_code": _FEED_TAIL, "terminal_image_input": "image",
            "inputs": [{"name": "image", "type": "IMAGE"},
                       {"name": "plate", "type": "IMAGE", "feeds": [[1, "plate"]]}],
            "promoted_params": [{"name": "gain", "internal": "gain", "stage": 0, "type": "f",
                                 "default": 1.2}]}


_FEED_DAG_S0 = "f$sigma = 1.5;\n@OUT = gauss_blur(@image, $sigma);"
_FEED_DAG_S1 = "@OUT = @a * @matte;"
_FEED_DAG_T = "@OUT = lerp(@x, @y, 0.5);"


def _dag_feed_manifest():
    """DAG: stage 0 blurs the source, stage 1 multiplies stage 0 by a fed MASK, the terminal mixes
    both. The fed MASK is declared BEFORE the source on purpose: the source is found by name."""
    return {"manifest_schema": 1, "name": "MatteMix", "tex_language": "0.23",
            "graphspec": {"schema": 1, "dag": True, "source_stage": 0, "source_binding": "image",
                          "stages": [{"code": _FEED_DAG_S0, "params": {}},
                                     {"code": _FEED_DAG_S1, "params": {},
                                      "chain_inputs": {"a": [0, "OUT"]}}],
                          "terminal_chain_inputs": {"x": [0, "OUT"], "y": [1, "OUT"]}},
            "terminal_code": _FEED_DAG_T, "terminal_image_input": "src",
            "inputs": [{"name": "matte", "type": "MASK", "feeds": [[1, "matte"]]},
                       {"name": "src", "type": "IMAGE"}],
            "promoted_params": []}


def _feed_oracle_cases(g):
    """(label, manifest, cook inputs, params, stage-by-stage @OUT as fn(device)) per routing."""
    A = torch.rand(1, 20, 20, 3, generator=g)
    B = torch.rand(1, 20, 20, 3, generator=g)
    P = torch.rand(1, 12, 16, 3, generator=g)
    S = torch.rand(1, 12, 16, 3, generator=g)
    M = torch.rand(1, 10, 14, generator=g)
    D = torch.rand(1, 10, 14, 3, generator=g)
    mm = tex_tool.load_tool(_merge_feed_manifest())

    def merge_ref(dev):
        o0 = tex_engine.cook(mm.graphspec["stages"][0]["code"], {"image": A, "sigma": 1.7},
                             device_mode=dev).outputs["OUT"]
        return tex_engine.cook(mm.terminal_code, {"A": o0, "B": B, "operation": 4, "mix": 0.8},
                               device_mode=dev).outputs["OUT"]

    def upstream_ref(dev):
        o0 = tex_engine.cook(_FEED_GRADE, {"image": S, "gain": 1.2}, device_mode=dev).outputs["OUT"]
        o1 = tex_engine.cook(_FEED_PLATE, {"image": o0, "plate": P}, device_mode=dev).outputs["OUT"]
        return tex_engine.cook(_FEED_TAIL, {"image": o1}, device_mode=dev).outputs["OUT"]

    def dag_ref(dev):
        o0 = tex_engine.cook(_FEED_DAG_S0, {"image": D}, device_mode=dev).outputs["OUT"]
        o1 = tex_engine.cook(_FEED_DAG_S1, {"a": o0, "matte": M}, device_mode=dev).outputs["OUT"]
        return tex_engine.cook(_FEED_DAG_T, {"x": o0, "y": o1}, device_mode=dev).outputs["OUT"]

    return [
        ("Merge shape: second input feeds the terminal", mm, {"A": A, "B": B},
         {"sigma": 1.7, "operation": 4, "mix": 0.8}, merge_ref),
        ("second input feeds an upstream linear stage", tex_tool.load_tool(_upstream_feed_manifest()),
         {"image": S, "plate": P}, {}, upstream_ref),
        ("a MASK input feeds a DAG stage", tex_tool.load_tool(_dag_feed_manifest()),
         {"src": D, "matte": M}, {}, dag_ref),
    ]


def test_tool_fused_feeds_roundtrip_unfused(r: SubTestResult):
    print("\n--- fed inputs: a fused tool cooks equal to its stages cooked one by one ---")
    for dev in _DEVICES:
        try:
            cases = _feed_oracle_cases(torch.Generator().manual_seed(21))
        except Exception as e:
            r.fail(f"feeds oracle setup [{dev}]", f"{type(e).__name__}: {e}")
            continue
        for label, m, inputs, pv, ref_fn in cases:
            try:
                before = json.dumps(m.to_dict(), sort_keys=True)
                fused = tex_tool.cook_tool(m, dict(inputs), pv, device_mode=dev).outputs["OUT"]
                # a fed value is written into a COPY of the stage params, never into the manifest
                assert json.dumps(m.to_dict(), sort_keys=True) == before, "the cook mutated the manifest"
                ref = ref_fn(dev)
                assert fused.shape == ref.shape, f"shape {tuple(fused.shape)} != {tuple(ref.shape)}"
                maxdiff = (fused.float() - ref.float()).abs().max().item()
                assert maxdiff < 1e-5, f"fused vs stage-by-stage maxdiff={maxdiff}"
                r.ok(f"[{dev}] {label}: == stage-by-stage (maxdiff={maxdiff:.2e})")
            except Exception as e:
                r.fail(f"feeds oracle [{dev}] {label}", f"{type(e).__name__}: {e}")


def _param_free_feed_manifests():
    """The three routings with every stage free of `$params`, for the CUDA codegen leg below."""
    term = _sum_feed_manifest("@OUT = gauss_blur(@image, 2.0);")
    term["terminal_code"] = "vec3 a = @A;\nvec3 b = @B;\n@OUT = vec4(lerp(a, b, 0.35) + a * b * 0.2, 1.0);"
    up = _upstream_feed_manifest()
    up["graphspec"]["stages"][0]["code"] = "@OUT = @image * 1.2;"
    up["promoted_params"] = []
    dag = _dag_feed_manifest()
    dag["graphspec"]["stages"][0]["code"] = "@OUT = gauss_blur(@image, 1.5);"
    return {"feed into the terminal": term, "feed into an upstream stage": up, "feed into a DAG stage": dag}


def test_tool_fused_feeds_codegen_parity(r: SubTestResult):
    print("\n--- fed inputs, invariant #2: the tool's fused program, interpreter == codegen ---")
    from TEX_Wrangle import tex_fusion
    from TEX_Wrangle.tex_runtime import tier_trace
    from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute
    for dev in _DEVICES:
        g = torch.Generator().manual_seed(29)
        try:
            pf = _param_free_feed_manifests()
            cases = [
                (label, tex_tool.load_tool(raw), torch.rand(1, 16, 16, 3, generator=g),
                 ({"matte": torch.rand(1, 16, 16, generator=g)} if "DAG" in label else
                  {("plate" if "upstream" in label else "B"): torch.rand(1, 16, 16, 3, generator=g)}), {})
                for label, raw in pf.items()]
            if dev == "cpu":
                # The stock-code Merge tool with promoted params, CPU only: on CUDA a program that reads
                # a scalar `$param` is served by the interpreter today (codegen declines on the device
                # mix), with or without fed inputs, so a CUDA leg there would compare the interpreter
                # with itself.
                cases.append(("stock Blur -> Merge with promoted params", tex_tool.load_tool(_merge_feed_manifest()),
                              torch.rand(1, 16, 16, 3, generator=g), {"B": torch.rand(1, 16, 16, 3, generator=g)},
                              {"sigma": 1.3, "operation": 5, "mix": 0.7}))
        except Exception as e:
            r.fail(f"feeds codegen setup [{dev}]", f"{type(e).__name__}: {e}")
            continue
        for label, m, src, extras, pv in cases:
            try:
                src = src.to(dev)
                extras = {k: v.to(dev) for k, v in extras.items()}
                gs, tb = tex_tool._fused_cook_inputs(m, src, pv, extras)
                prog, tm, _refs, asg, _pinfo, used, merged = tex_fusion.prepare_fused(
                    gs, m.terminal_code, tb, infer_binding_type)
                on = sorted(asg)

                def fresh():
                    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in merged.items()}

                interp = Interpreter().execute(prog, fresh(), tm, device=dev, output_names=on)["OUT"]
                # Key the codegen fn by the program's REAL fused fingerprint: the fn is persisted to
                # disk under this key, so a made-up name reused for an edited program would load a
                # stale fn in a later process.
                fp = tex_fusion.fused_fingerprint(gs, m.terminal_code, tb, infer_binding_type)
                assert fp, "the fused program has no fingerprint"
                tier_trace.reset()
                cg = _codegen_only_execute(prog, fresh(), tm, dev, output_names=on, used_builtins=used,
                                           fingerprint=fp, time_context=None)
                served = tier_trace.last()
                # _codegen_only_execute falls back to the interpreter silently; a parity check that
                # compared the interpreter with itself would pass vacuously.
                assert served is not None and served.tier == "codegen", f"codegen did not serve: {served}"
                cg = cg["OUT"] if isinstance(cg, dict) else cg
                md = (interp.float() - cg.float()).abs().max().item()
                assert md < 1e-5, f"interp vs codegen maxdiff={md}"
                r.ok(f"[{dev}] {label}: interp == codegen (maxdiff={md:.1e})")
            except Exception as e:
                r.fail(f"feeds codegen parity [{dev}] {label}", f"{type(e).__name__}: {e}")


_FEED_SUM = "@OUT = @A + @B;"


def _sum_feed_manifest(upstream_code):
    return {"manifest_schema": 1, "name": "Sum", "tex_language": "0.23",
            "graphspec": {"schema": 1, "stages": [{"code": upstream_code, "image_input": "image",
                                                   "params": {}}]},
            "terminal_code": _FEED_SUM, "terminal_image_input": "A",
            "inputs": [{"name": "A", "type": "IMAGE"},
                       {"name": "B", "type": "IMAGE", "feeds": [["terminal", "B"]]}],
            "promoted_params": []}


def test_tool_fused_feeds_extent_refusal(r: SubTestResult):
    print("\n--- fed inputs: non-co-extent inputs are REFUSED before the engine (silently wrong otherwise) ---")
    from TEX_Wrangle.tex_marshalling import Promise
    from TEX_Wrangle.tex_runtime.interpreter import InterpreterError
    real_cook = tex_engine.cook
    calls = []

    def spy(*a, **k):
        calls.append(a[0])
        return real_cook(*a, **k)

    def refused(m, inputs, code, input_name):
        calls.clear()
        tex_engine.cook = spy
        try:
            tex_tool.cook_tool(m, inputs, {}, device_mode="cpu")
        except TEXToolError as e:
            assert e.code == code, f"code {e.code!r} != {code!r} ({e})"
            assert e.input == input_name, f"input {e.input!r} != {input_name!r}"
            assert not calls, "tex_engine.cook was called before the refusal"
            return str(e)
        finally:
            tex_engine.cook = real_cook
        raise AssertionError("cooked instead of refusing")

    g = torch.Generator().manual_seed(23)
    rows = [  # label, upstream code, source, fed input, silently wrong through the engine directly?
        ("a B=4 input beside a B=1 source", "@OUT = @image * (fi + 1.0);",
         torch.rand(1, 8, 8, 4, generator=g), torch.rand(4, 8, 8, 4, generator=g), True),
        ("a [1,1,W,C] strip source beside a full frame", "@OUT = @image * v;",
         torch.rand(1, 1, 8, 4, generator=g), torch.rand(1, 8, 8, 4, generator=g), True),
        ("an H mismatch", "@OUT = @image * v;",
         torch.rand(1, 8, 8, 4, generator=g), torch.rand(1, 9, 8, 4, generator=g), False),
    ]
    for label, upstream, A, B, silent in rows:
        try:
            m = tex_tool.load_tool(_sum_feed_manifest(upstream))
            msg = refused(m, {"A": A, "B": B}, "fused-input-extent", "B")
            assert "'B'" in msg and "'A'" in msg, f"the refusal names neither input: {msg}"
            detail = ""
            if silent:
                # Negative control: the SAME inputs, handed to the engine directly (bypassing the
                # refusal), cook without error, at the stage-by-stage output shape, to wrong pixels.
                gs, tb = tex_tool._fused_cook_inputs(m, A, {}, {"B": B})
                fused = real_cook(m.terminal_code, tb, chain_payload=gs, device_mode="cpu").outputs["OUT"]
                o0 = real_cook(upstream, {"image": A}, device_mode="cpu").outputs["OUT"]
                ref = real_cook(_FEED_SUM, {"A": o0, "B": B}, device_mode="cpu").outputs["OUT"]
                assert fused.shape == ref.shape, f"control shapes differ {fused.shape} {ref.shape}"
                md = (fused.float() - ref.float()).abs().max().item()
                assert md > 0.5, f"negative control no longer diverges (maxdiff={md}) -- re-measure"
                detail = f"; unrefused it is silently wrong (maxdiff={md:.3f}, same shape)"
            r.ok(f"{label}: refused fused-input-extent, engine never called{detail}")
        except Exception as e:
            r.fail(f"extent refusal: {label}", f"{type(e).__name__}: {e}")

    try:
        m = tex_tool.load_tool(_sum_feed_manifest("@OUT = @image * v;"))
        A = torch.rand(1, 8, 8, 4, generator=g)
        refused(m, {"A": A}, "fused-input-missing", "B")
        refused(m, {"A": A, "B": 0.5}, "fused-input-extent", "B")
        refused(m, {"A": 0.5, "B": A}, "fused-input-extent", "A")
        # [B,H,W] is compared, never channels: a MASK input beside an RGB source cooks.
        dm = tex_tool.load_tool(_dag_feed_manifest())
        out = tex_tool.cook_tool(dm, {"src": torch.rand(2, 6, 5, 3, generator=g),
                                      "matte": torch.rand(2, 6, 5, generator=g)}, {}, device_mode="cpu")
        assert tuple(out.outputs["OUT"].shape) == (2, 6, 5, 3), out.outputs["OUT"].shape
        # a promised input is resolved first: landed -> cooks equal to the tensor; unlanded -> E7007
        B = torch.rand(1, 8, 8, 4, generator=g)
        p = Promise("B", type=infer_binding_type(B), shape=tuple(B.shape))
        try:
            tex_engine.cook = spy
            calls.clear()
            tex_tool.cook_tool(m, {"A": A, "B": p}, {}, device_mode="cpu")
            raise AssertionError("an unlanded promised input cooked")
        except InterpreterError as e:
            assert getattr(e, "_code", "") == "E7007", f"{getattr(e, '_code', '')}: {e}"
            assert not calls, "tex_engine.cook was called for an unlanded promise"
        finally:
            tex_engine.cook = real_cook
        p.land(B)
        got = tex_tool.cook_tool(m, {"A": A, "B": p}, {}, device_mode="cpu").outputs["OUT"]
        want = tex_tool.cook_tool(m, {"A": A, "B": B}, {}, device_mode="cpu").outputs["OUT"]
        assert torch.equal(got, want), "a landed promise cooked differently from its tensor"
        r.ok("missing -> fused-input-missing; non-tensor -> fused-input-extent; MASK beside RGB cooks; "
             "promised input resolved first (unlanded E7007, landed == tensor)")
    except Exception as e:
        r.fail("extent refusal: presence / tensor / promise rows", f"{type(e).__name__}: {e}")


def _feeds_dag_generator_manifest():
    """A DAG whose stage 1 reads nothing: no chain, no source, no feed (a generator)."""
    d = _dag_feed_manifest()
    d["graphspec"]["stages"].append({"code": "@OUT = vec3(u, v, 0.5);", "params": {}})
    d["graphspec"]["terminal_chain_inputs"] = {"x": [0, "OUT"], "y": [1, "OUT"], "z": [2, "OUT"]}
    d["terminal_code"] = "@OUT = (@x + @y + @z) / 3.0;"
    return d


def test_tool_fused_feeds_rejects(r: SubTestResult):
    print("\n--- fed inputs: malformed / colliding / unanchored manifests are refused with a stable code ---")

    def lin(**over):
        d = _merge_feed_manifest()
        d.update(over)
        return d

    def with_inputs(base, *extra):
        d = base()
        d["inputs"] = list(extra)
        return d

    A = {"name": "A", "type": "IMAGE"}

    def B(feeds, **kw):
        return {"name": "B", "type": "IMAGE", "feeds": feeds, **kw}

    dag = _dag_feed_manifest
    INVALID, COLLISION = "fused-feed-invalid", "fused-feed-collision"
    single = {"manifest_schema": 1, "name": "M", "tex_language": "0.23", "code": _stock("merge").code,
              "inputs": [A, B([["terminal", "B"]])], "promoted_params": []}
    staged = lin()
    staged["graphspec"]["stages"][0]["params"] = {"k": 1.0}
    termp = lin(terminal_params={"k": 1.0})
    cases = {
        # shape of `feeds`
        "feeds not a list": (lin(inputs=[A, B("terminal")]), INVALID, "B"),
        "feeds empty": (lin(inputs=[A, B([])]), INVALID, "B"),
        "feeds over MAX_STAGES pairs": (lin(inputs=[A, B([["terminal", f"b{k}"]
                                                           for k in range(tex_tool.MAX_STAGES + 1)])]),
                                        INVALID, "B"),
        "feeds pair not a pair": (lin(inputs=[A, B([["terminal"]])]), INVALID, "B"),
        "feeds bool stage": (lin(inputs=[A, B([[True, "b"]])]), INVALID, "B"),
        "feeds null stage": (lin(inputs=[A, B([[None, "b"]])]), INVALID, "B"),
        "feeds binding not an identifier": (lin(inputs=[A, B([["terminal", "1b"]])]), INVALID, "B"),
        # routing
        "feeds on the source": (lin(inputs=[{**A, "feeds": [[0, "extra"]]}, B([["terminal", "B"]])]),
                                INVALID, "A"),
        "an unrouted extra input": (lin(inputs=[A, B([["terminal", "B"]]), {"name": "C", "type": "IMAGE"}]),
                                    "fused-input-unrouted", "C"),
        "feeds stage out of range": (lin(inputs=[A, B([[1, "b"]])]), INVALID, "B"),
        "feeds negative stage": (lin(inputs=[A, B([[-1, "b"]])]), INVALID, "B"),
        "a LATENT feed": (lin(inputs=[A, {**B([["terminal", "B"]]), "type": "LATENT"}]), INVALID, "B"),
        "a fed input marked optional": (lin(inputs=[A, B([["terminal", "B"]], optional=True)]), INVALID, "B"),
        "feeds on a single-stage (code) tool": (single, INVALID, "B"),
        # collisions -- a feed may not take a name its stage already binds
        "collides with the source injection (linear stage 0 image_input)":
            (lin(inputs=[A, B([[0, "image"]])]), COLLISION, "B"),
        "collides with the terminal's chain input (terminal_image_input)":
            (lin(inputs=[A, B([["terminal", "A"]])]), COLLISION, "B"),
        "collides with a linear stage's chain input":
            (with_inputs(_upstream_feed_manifest, {"name": "image", "type": "IMAGE"},
                         {"name": "plate", "type": "IMAGE", "feeds": [[1, "image"]]}), COLLISION, "plate"),
        "collides with a baked stage param": (lin(graphspec=staged["graphspec"], inputs=[A, B([[0, "k"]])]),
                                              COLLISION, "B"),
        "collides with a baked terminal param": ({**termp, "inputs": [A, B([["terminal", "k"]])]},
                                                 COLLISION, "B"),
        "collides with a promoted internal on its stage": (lin(inputs=[A, B([[0, "sigma"]])]), COLLISION, "B"),
        "collides with a promoted internal on the terminal": (lin(inputs=[A, B([["terminal", "mix"]])]),
                                                              COLLISION, "B"),
        "collides with another feed": (lin(inputs=[A, B([["terminal", "B"]]),
                                                   {"name": "C", "type": "IMAGE", "feeds": [["terminal", "B"]]}]),
                                       COLLISION, "C"),
        "collides with a DAG stage's chain_inputs key":
            (with_inputs(dag, {"name": "matte", "type": "MASK", "feeds": [[1, "a"]]},
                         {"name": "src", "type": "IMAGE"}), COLLISION, "matte"),
        "collides with terminal_chain_inputs":
            (with_inputs(dag, {"name": "matte", "type": "MASK", "feeds": [["terminal", "x"]]},
                         {"name": "src", "type": "IMAGE"}), COLLISION, "matte"),
        "collides with a DAG source injection point":
            (with_inputs(dag, {"name": "matte", "type": "MASK", "feeds": [[0, "image"]]},
                         {"name": "src", "type": "IMAGE"}), COLLISION, "matte"),
        "collides with the DAG source socket on the terminal":
            (with_inputs(dag, {"name": "matte", "type": "MASK", "feeds": [["terminal", "src"]]},
                         {"name": "src", "type": "IMAGE"}), COLLISION, "matte"),
        # a DAG stage that reads nothing would adopt the fused extent instead of its own
        "an unanchored DAG stage (a generator)": (_feeds_dag_generator_manifest(), "fused-stage-unanchored", None),
    }
    failed = []
    for label, (raw, code, input_name) in cases.items():
        try:
            tex_tool.load_tool(raw)
            failed.append(f"{label}: NOT refused")
        except TEXToolError as e:
            if e.code != code or e.input != input_name:
                failed.append(f"{label}: code={e.code!r} input={e.input!r} ({e})")
        except Exception as e:
            failed.append(f"{label}: wrong exception {type(e).__name__}: {e}")
    # the bases themselves load (so every refusal above is the row's own doing), and a fed DAG
    # stage that reads ONLY a feed is anchored by it
    anchored = _feeds_dag_generator_manifest()
    anchored["graphspec"]["stages"][2]["code"] = "@OUT = vec3(@m, @m, 0.5);"
    anchored["inputs"][0]["feeds"] = [[1, "matte"], [2, "m"]]
    for label, raw in (("Merge-shaped", _merge_feed_manifest()), ("upstream", _upstream_feed_manifest()),
                       ("DAG", _dag_feed_manifest()), ("DAG stage anchored by a feed", anchored)):
        try:
            tex_tool.load_tool(raw)
        except Exception as e:
            failed.append(f"base '{label}' did not load: {type(e).__name__}: {e}")
    if failed:
        r.fail("feeds rejects", "; ".join(failed))
    else:
        r.ok(f"all {len(cases)} malformed/colliding/unanchored manifests refused with the stable code "
             f"and offending input; the four valid bases load")


def test_tool_fused_input_refusals_unchanged(r: SubTestResult):
    print("\n--- a fused tool that declares no feeds: the one-input refusals keep text and type, gain a code ---")
    fbase = {"manifest_schema": 1, "name": "F", "tex_language": "0.23",
             "graphspec": {"schema": 1, "stages": [{"code": "@OUT=@image;", "image_input": "image",
                           "params": {}}], "terminal_image_input": "image"},
             "terminal_code": "@OUT=@image;", "terminal_image_input": "image",
             "inputs": [{"name": "image", "type": "IMAGE"}], "promoted_params": []}
    rows = [
        ("two inputs, no feeds", {**fbase, "inputs": [{"name": "image"}, {"name": "extra"}]},
         "a fused tool must declare exactly one external input (the single fusion source)",
         "fused-input-count", None),
        ("sole input optional", {**fbase, "inputs": [{"name": "image", "type": "IMAGE", "optional": True}]},
         "a fused tool's sole external input may not be marked 'optional' (the engine requires it to "
         "splice the chain)", None, None),
        ("terminal_image_input not an input", {**fbase, "terminal_image_input": "nope"},
         "terminal_image_input 'nope' is not a declared input", None, None),
    ]
    for label, raw, text, code, input_name in rows:
        try:
            try:
                tex_tool.load_tool(raw)
                raise AssertionError("not refused")
            except TEXToolError as e:
                assert type(e) is TEXToolError, type(e)
                assert str(e) == text and e.args == (text,), f"text drifted: {e.args!r}"
                assert e.code == code and e.input == input_name, (e.code, e.input)
            r.ok(f"{label}: text + type unchanged, code={code!r}")
        except Exception as e:
            r.fail(f"one-input refusal: {label}", f"{type(e).__name__}: {e}")
    try:
        text = "fused tool 'GradeVignette' needs its source input 'image'"
        try:
            tex_tool.cook_tool(_stock("grade_vignette"), {}, {}, device_mode="cpu")
            raise AssertionError("missing source not refused")
        except TEXToolError as e:
            assert type(e) is TEXToolError and str(e) == text and e.args == (text,), e.args
            assert e.code == "fused-input-missing" and e.input == "image", (e.code, e.input)
        r.ok("cook-time missing source: text + type unchanged, code='fused-input-missing'")
    except Exception as e:
        r.fail("one-input refusal: missing source at cook", f"{type(e).__name__}: {e}")


def test_tool_fused_feeds_manifest_roundtrip(r: SubTestResult):
    print("\n--- fed inputs: load / to_dict / tool_summary / write_tool round-trip; preflight; warm keys ---")
    try:
        m = tex_tool.load_tool(_merge_feed_manifest())
        assert m.inputs[1]["feeds"] == [["terminal", "B"]] and "feeds" not in m.inputs[0], m.inputs
        d = m.to_dict()
        assert d["inputs"] == _merge_feed_manifest()["inputs"], d["inputs"]
        assert tex_tool.tool_summary(m)["inputs"] == d["inputs"], tex_tool.tool_summary(m)["inputs"]
        dest = tempfile.mkdtemp()
        again = tex_tool.load_tool(tex_tool.write_tool(m, dest))
        assert again.to_dict() == d, "write_tool -> load_tool changed the manifest"
        assert tex_tool.install_tool(m, tempfile.mkdtemp())["ok"]
        # a tuple-spelled feeds (a Python host's dict) normalizes to the JSON shape
        raw = _merge_feed_manifest()
        raw["inputs"][1]["feeds"] = (("terminal", "B"),)
        assert tex_tool.load_tool(raw).inputs[1]["feeds"] == [["terminal", "B"]]
        r.ok("feeds round-trips load_tool / to_dict / tool_summary / write_tool / install_tool")
    except Exception as e:
        r.fail("feeds manifest round-trip", f"{type(e).__name__}: {e}")

    for label, build, inputs_fn in (
            ("Merge-shaped", _merge_feed_manifest,
             lambda C: {"A": torch.rand(1, 8, 8, C), "B": torch.rand(1, 8, 8, C)}),
            ("upstream", _upstream_feed_manifest,
             lambda C: {"image": torch.rand(1, 8, 8, C), "plate": torch.rand(1, 8, 8, C)}),
            ("DAG, MASK declared first", _dag_feed_manifest,
             lambda C: {"src": torch.rand(1, 8, 8, C), "matte": torch.rand(1, 8, 8)})):
        try:
            m = tex_tool.load_tool(build())
            pf = tex_tool.preflight_tool(m)
            assert pf["ok"], f"preflight failed: {pf['diagnostics']}"
            seen = []
            real_fp = tex_engine._fused_fingerprint

            def spy(*a, **k):
                fp = real_fp(*a, **k)
                seen.append(fp)
                return fp

            tex_engine._fused_fingerprint = spy
            try:
                for C in (3, 4):          # an RGB and an RGBA cook, every IMAGE input aligned
                    tex_tool.cook_tool(m, inputs_fn(C), {}, device_mode="cpu")
            finally:
                tex_engine._fused_fingerprint = real_fp
            wk = tex_tool.tool_warm_keys(m)
            assert len(seen) == 2 and all(fp in wk for fp in seen), f"cook keys {seen} not in warm keys {wk}"
            r.ok(f"{label}: preflight ok; warm keys {len(wk)} include both cooks' keys")
        except Exception as e:
            r.fail(f"feeds preflight/warm keys: {label}", f"{type(e).__name__}: {e}")


def test_tool_fused_feeds_rekey(r: SubTestResult):
    print("\n--- fed inputs: an RGB then an RGBA fed input re-keys the fused program (no stale program) ---")
    from TEX_Wrangle import tex_fusion
    try:
        d = _sum_feed_manifest("@OUT = @image * 1.1;")
        d["terminal_code"] = "@OUT = @B * luma(@A);"
        m = tex_tool.load_tool(d)
        # The distinct-key assertion is the one a mis-keying fails: on this terminal a program keyed
        # for the other channel count still cooks the right pixels (the interpreter follows the
        # tensor it is handed), so the pixel check alone would not see the memo serve it.
        g = torch.Generator().manual_seed(31)
        A = torch.rand(1, 6, 6, 3, generator=g)
        keys = []
        for C in (3, 4):
            B = torch.rand(1, 6, 6, C, generator=g)
            out = tex_tool.cook_tool(m, {"A": A, "B": B}, {}, device_mode="cpu").outputs["OUT"]
            key = tex_fusion._fused_memo_key(
                tex_tool._assemble_fused_stages(m, A, {}, {"B": B}), infer_binding_type)
            assert key in tex_fusion._FUSED_MEMO, "the cooked program is not the one keyed for this input"
            keys.append(key)
            o0 = tex_engine.cook("@OUT = @image * 1.1;", {"image": A}, device_mode="cpu").outputs["OUT"]
            ref = tex_engine.cook(d["terminal_code"], {"A": o0, "B": B}, device_mode="cpu").outputs["OUT"]
            assert out.shape[-1] == C and out.shape == ref.shape, (tuple(out.shape), tuple(ref.shape))
            assert (out - ref).abs().max().item() < 1e-5, f"C={C}: stale or wrong program"
        assert keys[0] != keys[1], "an RGB and an RGBA fed input share one fused program key"
        r.ok("RGB -> RGBA fed input: distinct memo keys, each cook == stage-by-stage at its channel count")
    except Exception as e:
        r.fail("feeds re-key", f"{type(e).__name__}: {e}")


# A tool that declares no `feeds` must validate, load and cook exactly as before `feeds` existed.
# The serialisation hashes below were captured against tex_tool.py BEFORE `feeds` existed (the
# tooltip/options/optional manifests of the two tests above; the stock and OLD_* ones are already
# pinned by test_tool_manifest_byte_identity). The cook half is pinned structurally, which holds on
# any box and torch build: cook_tool must hand tex_engine.cook exactly the call rebuilt below from the
# manifest alone, and the warm keys must be the ones that call derives. Their `written_bytes` pins are
# line-ending-normalised like every other in this file, because write_tool writes platform newlines
# (the note above _MANIFEST_BASELINE_SHA256).
_UI_HINT_MANIFESTS = {
    "tooltip": {"manifest_schema": 1, "name": "X", "tex_language": "0.23", "code": "@OUT = @image;",
                "inputs": [{"name": "image", "type": "IMAGE"}],
                "promoted_params": [{"name": "gamma", "type": "f", "default": 1.0,
                                     "metadata": {"min": 0.0, "max": 4.0, "tooltip": "Power curve."}}]},
    "options": {"manifest_schema": 1, "name": "X", "tex_language": "0.23", "code": "@OUT = @image;",
                "inputs": [{"name": "image", "type": "IMAGE"}],
                "promoted_params": [{"name": "channel", "type": "i", "default": 0,
                                     "metadata": {"min": 0, "max": 2, "step": 1,
                                                  "options": ["Red", "Green", "Blue"]}}]},
    "combo11": {"manifest_schema": 1, "name": "X", "tex_language": "0.23", "code": "@OUT = @image;",
                "inputs": [{"name": "image", "type": "IMAGE"}],
                "promoted_params": [{"name": "mode", "type": "i", "default": 0,
                                     "metadata": {"min": 0, "max": 10, "step": 1,
                                                  "options": [f"opt{k}" for k in range(11)]}}]},
    "optional_true": {"manifest_schema": 1, "name": "X", "tex_language": "0.23", "code": "@OUT = @image;",
                      "promoted_params": [], "inputs": [{"name": "image", "type": "IMAGE"},
                                                        {"name": "mask", "type": "MASK", "optional": True}]},
    "optional_false": {"manifest_schema": 1, "name": "X", "tex_language": "0.23", "code": "@OUT = @image;",
                       "promoted_params": [], "inputs": [{"name": "image", "type": "IMAGE"},
                                                         {"name": "mask", "type": "MASK", "optional": False}]},
    "optional_absent": {"manifest_schema": 1, "name": "X", "tex_language": "0.23", "code": "@OUT = @image;",
                        "promoted_params": [], "inputs": [{"name": "image", "type": "IMAGE"}]},
    "extent_dropped": {"manifest_schema": 1, "name": "X", "tex_language": "0.23", "code": "@OUT = @image;",
                       "promoted_params": [], "inputs": [{"name": "image", "type": "IMAGE", "extent": "own"}]},
}
_UI_HINT_SHA256 = {
    "tooltip": {"to_dict": "bc570315648b166935b1034274446ce353feca89f5f100fddb35f8ef813da8c8",
                "tool_summary": "fbae9d162ccfe05e69af0dd859d4cefdf173551877483fa72e92683d37b1a82d",
                "written_bytes": "8bbb7bbc75dd36045363ddc489d2b292ad2df9bbfc1cbf0257ada4946252caee"},
    "options": {"to_dict": "d041ce56b282871a57f0cf459eb3687c58f6f9a5effc98b4d7e676dd9891391f",
                "tool_summary": "6c59de31d3b8ad9698686d1a31ba138ae289de1793e9ee62cc9db2c601079856",
                "written_bytes": "9331d4a1262bbae43bc24a569ee5a9b9a9f22f188e4fa029951c7f0b676dfefc"},
    "combo11": {"to_dict": "49b635301b1dac97006d106a4fc7cb119c19c987e158f067ce8f649ada5d421a",
                "tool_summary": "4084060ebd9d7cd17727136806ece595d1c29fad2dcd957b848ebd73e2bdbb03",
                "written_bytes": "1642c9c917019f1b5129c14afe9cf3733964a67e645f5275e4b97210c8167954"},
    "optional_true": {"to_dict": "63326c39f4ede45973f8fcb87ed221333e4fba63de458ef4ea87284819a14ac6",
                      "tool_summary": "022f80701d34331bcd2a964d3c04bc21cbd37288faebfd8b6ae53f224d52af7c",
                      "written_bytes": "d644060b022edfd7abcd345a8774c0a807917616c61d2735d7df5a18880f3e37"},
    "optional_false": {"to_dict": "195accf2b5051013b68fa2e9ec7f6ecd8a8e74ef95d2be20f38cac680fa0e368",
                       "tool_summary": "e6c3a352aeaa9d0c296c35190fe46d2585a556f4259280f3658452f6145611da",
                       "written_bytes": "d9f96cb0b51ab2a8bca666627d71d74deca04d29064ba78ca9af2690e40ae646"},
    "optional_absent": {"to_dict": "dc9560cc17e09179f1e9b74b19153aefbecc81f4cb4d7436fd08ad3148d8e305",
                        "tool_summary": "3afc2bc05a5b4bba5a0e4a67a3e7c204c924171a6c525c2891e134c08b065d41",
                        "written_bytes": "6c5581dc54737ea4185414f5ff0ad76d0bc5b1b03ba09ace6cedc5a7b6cfc1b3"},
    "extent_dropped": {"to_dict": "dc9560cc17e09179f1e9b74b19153aefbecc81f4cb4d7436fd08ad3148d8e305",
                       "tool_summary": "3afc2bc05a5b4bba5a0e4a67a3e7c204c924171a6c525c2891e134c08b065d41",
                       "written_bytes": "6c5581dc54737ea4185414f5ff0ad76d0bc5b1b03ba09ace6cedc5a7b6cfc1b3"},
}


def _canon_call(v):
    """A comparable spelling of an engine call's arguments: tensors by identity, scalars by type
    AND value (an int default and a float default key different programs)."""
    if isinstance(v, torch.Tensor):
        return ("tensor", id(v))
    if isinstance(v, dict):
        return {k: _canon_call(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon_call(x) for x in v]
    return (type(v).__name__, v)


def _pre_feeds_engine_call(m, inputs, params):
    """(args, kwargs) cook_tool handed tex_engine.cook for a tool before `feeds` existed, rebuilt
    from the manifest alone rather than through tex_tool's helpers."""
    if not m.is_fused:
        return (m.code, {**inputs, **{pp.internal: params.get(pp.name, pp.default)
                                       for pp in m.promoted_params}}), {}
    gs = dict(m.graphspec)
    gs["stages"] = [dict(s) for s in m.graphspec.get("stages", [])]
    gs["terminal_image_input"] = m.terminal_image_input
    term = dict(m.terminal_params or {})
    for pp in m.promoted_params:
        val = params.get(pp.name, pp.default)
        if pp.stage in (None, "terminal"):
            term[pp.internal] = val
        else:
            st = gs["stages"][pp.stage]
            st["params"] = {**(st.get("params") or {}), pp.internal: val}
    return (m.terminal_code, {**term, m.terminal_image_input: inputs[m.terminal_image_input]}), \
        {"chain_payload": gs}


def test_tool_no_feeds_is_pre_feeds_identical(r: SubTestResult):
    print("\n--- a tool without feeds: serialises, cooks and warm-keys exactly as before feeds existed ---")
    from TEX_Wrangle.tex_fusion import fused_fingerprint
    try:
        drift = [k for k, raw in _UI_HINT_MANIFESTS.items()
                 if _hash_manifest(tex_tool.load_tool(json.loads(json.dumps(raw)))) != _UI_HINT_SHA256[k]]
        assert not drift, f"serialisation drifted for: {drift}"
        r.ok(f"{len(_UI_HINT_MANIFESTS)} tooltip/options/optional manifests serialise byte-identically")
    except Exception as e:
        r.fail("no-feeds identity: serialisation", f"{type(e).__name__}: {e}")

    manifests = {f"stock {fn[:-8]}": _stock(fn[:-8]).to_dict() for fn in _MANIFEST_BASELINE_SHA256}
    manifests.update({f"ui {k}": v for k, v in _UI_HINT_MANIFESTS.items()})
    manifests.update({"OLD_SINGLE": _OLD_SINGLE_MANIFEST, "OLD_FUSED": _OLD_FUSED_MANIFEST})
    real_cook = tex_engine.cook
    g = torch.Generator().manual_seed(37)
    pool = {"image": torch.rand(1, 12, 12, 3, generator=g), "A": torch.rand(1, 12, 12, 3, generator=g),
            "B": torch.rand(1, 12, 12, 3, generator=g), "mask": torch.rand(1, 12, 12, generator=g)}
    for label, raw in manifests.items():
        try:
            m = tex_tool.load_tool(json.loads(json.dumps(raw)))
            need = {i["name"]: pool[i["name"]] for i in m.inputs}
            nudged = {pp.name: pp.default + 1 for pp in m.promoted_params
                      if pp.type in ("f", "i") and not isinstance(pp.default, bool)
                      and not pp.metadata.get("options")}
            for pset in ({}, nudged):
                seen = []

                def spy(*a, **k):
                    seen.append((_canon_call(a), _canon_call(k)))
                    return real_cook(*a, **k)

                tex_engine.cook = spy
                try:
                    got = tex_tool.cook_tool(m, dict(need), pset, device_mode="cpu").outputs
                finally:
                    tex_engine.cook = real_cook
                ref_args, ref_kwargs = _pre_feeds_engine_call(m, dict(need), pset)
                want_call = (_canon_call(ref_args), _canon_call({**ref_kwargs, "device_mode": "cpu"}))
                assert seen == [want_call], f"engine call drifted:\n got  {seen}\n want {want_call}"
                want = real_cook(*ref_args, **ref_kwargs, device_mode="cpu").outputs
                assert sorted(got) == sorted(want) and all(torch.equal(got[k], want[k]) for k in want), \
                    "pixels differ from the pre-feeds call"
            if m.is_fused:
                keys = []
                for ch in tex_tool._image_channel_variants(m):
                    (code, tb), kw = _pre_feeds_engine_call(
                        m, {m.terminal_image_input: tex_tool._placeholder_tensor(m.inputs[0].get("type"), ch)},
                        tex_tool._repr_params(m, warm=True))
                    fp = fused_fingerprint(kw["chain_payload"], code, tb, infer_binding_type)
                    if fp and fp not in keys:
                        keys.append(fp)
                assert tex_tool.tool_warm_keys(m) == keys, "fused warm keys drifted"
            r.ok(f"{label}: engine call + pixels identical (defaults and nudged params)"
                 + ("; warm keys identical" if m.is_fused else ""))
        except Exception as e:
            r.fail(f"no-feeds identity: {label}", f"{type(e).__name__}: {e}")


# ── TOOL-5: the adversarial-AST emitter fuzz lane ───────────────────────────────
def _gen_body(code: str, bindings: dict) -> str:
    program = Parser(Lexer(code).tokenize(), source=code).parse()
    binding_types = {n: infer_binding_type(v) for n, v in bindings.items()}
    type_map = TypeChecker(binding_types=binding_types, source=code).check(program)
    gen = _CodeGen(type_map)
    gen.emit_program(program)
    return "\n".join(gen._preamble + gen._lines)


# The dangerous-name blocklist. The emitter emits many sanctioned generated helpers
# (`_bp`, `_t*`, `_lv_*`, `_uf_*`) and a few builtins (`RuntimeError`, `int`, `float`) which
# an allowlist would have to enumerate exhaustively; a blocklist of the escape vectors is the
# robust regression tripwire, backstopped by the REAL gate (the type checker rejects unknown
# functions BEFORE codegen, so a bare `eval()`/`__import__()` never reaches the emitter).
_DANGER_CALLS = {"eval", "exec", "__import__", "compile", "open", "globals", "locals",
                 "vars", "getattr", "setattr", "delattr", "input", "system", "popen", "execfile"}
_DANGER_ATTRS = {"__globals__", "__builtins__", "__class__", "__subclasses__", "__bases__",
                 "__code__", "__dict__", "system", "popen", "mro",
                 # torch/pickle vectors that would smuggle arbitrary code through an attr call
                 "load", "save", "jit", "hub", "loads", "dump", "dumps", "mmap"}


def _scan_for_injection(src: str) -> list:
    """Parse GENERATED Python and walk it: string literals are ast.Constant (safe DATA), so
    only real CODE trips a finding — an import, a Call to a blocklisted dangerous name, or an
    Attribute to a dunder/system/pickle vector. This is the precise 'code vs data' distinction
    repr() buys (a repr'd string containing `import os` is a Constant, not an Import node)."""
    findings = []
    tree = ast.parse("def _tex_fuzz(_bind, _fns, _env, _torch, _math, _dev, _sp):\n" + src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            findings.append("import statement in generated code")
        elif isinstance(node, ast.Attribute) and node.attr in _DANGER_ATTRS:
            findings.append(f"attribute .{node.attr}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _DANGER_CALLS:
            findings.append(f"call to {node.func.id}()")
    return findings


def test_tool_emitter_fuzz(r: SubTestResult):
    print("\n--- TOOL-5: adversarial-AST fuzz — no hostile .textool escapes the emitter ---")
    img = torch.rand(1, 8, 8, 3)
    # ARM 1 — programs that reference dangerous/unknown names must be REJECTED before codegen
    # (the type checker gates unknown functions; the lexer is ASCII-only so confusables can't
    # even tokenise as one identifier).
    reject = [
        '@OUT = __import__("os");',
        '@OUT = eval("1+1");',
        '@OUT = exec("x");',
        '@OUT = system("rm -rf /");',
        'float а = 1.0; @OUT = vec4(а, 0, 0, 1);',   # Cyrillic 'а' identifier
        '@OUT = getattr(@image, "x");',
    ]
    # ARM 2 — programs that are VALID TEX but carry hostile-looking identifiers / strings; they
    # compile, and the generated code must contain NO real dangerous call/attr/import (only
    # prefixed locals + repr'd string data).
    benign = [
        'float __globals__ = 0.5; @OUT = @image * __globals__;',
        'float eval = 0.3; float exec = 0.2; @OUT = @image * (eval + exec);',
        'float import = 1.0; @OUT = @image * import;',       # 'import' is a valid TEX identifier
        'string s = "__import__(\'os\').system(\'x\')"; @OUT = @image;',
        'string s = "\\"; import os #"; @OUT = @image;',
        'string s = "\\n\\t\\"quotes\\" and #{fmt}"; @OUT = @image;',
        # user FUNCTIONS with hostile names — exercises the `_uf_{name}` emission and the
        # `raise RuntimeError('… in {name}()')` string-interpolation site (codegen.py:1177-1179).
        'float __globals__(float x) { return x * 2.0; } @OUT = @image * __globals__(0.5);',
        'float system(float x) { return x + 0.1; } @OUT = @image * system(0.3);',
    ]
    problems = []
    for code in reject:
        diags = tex_api.check(code, {"image": infer_binding_type(img)})
        if not any(getattr(d, "severity", "error") == "error" for d in diags):
            problems.append(f"NOT rejected: {code!r}")
    for code in benign:
        try:
            body = _gen_body(code, {"image": img})
        except Exception as e:
            problems.append(f"benign program failed to compile ({e}): {code!r}")
            continue
        found = _scan_for_injection(body)
        if found:
            problems.append(f"INJECTION {found} from: {code!r}")
    # ARM 3 — the FUSED path: hostile identifiers spliced across stages go through the SAME
    # emitter (with extra `_s{i}_u_` prefixing). Scan the generated source of a fused program.
    try:
        from TEX_Wrangle.tex_fusion import compile_fused
        stages = [
            {"code": 'float eval = 0.2; @OUT = @image * eval;', "chain_input": None,
             "bindings": {"image": img}},
            {"code": 'float __class__(float x){return x;} @OUT = @image * __class__(0.7);',
             "chain_input": "image", "bindings": {}},
        ]
        prog, type_map, *_ = compile_fused(stages, infer_binding_type)
        gen = _CodeGen(type_map)
        gen.emit_program(prog)
        fused_src = "\n".join(gen._preamble + gen._lines)
        found = _scan_for_injection(fused_src)
        if found:
            problems.append(f"FUSED INJECTION {found}")
    except Exception as e:
        problems.append(f"fused fuzz path errored: {e}")
    if problems:
        r.fail("emitter fuzz", "; ".join(problems[:6]))
    else:
        r.ok(f"{len(reject)} hostile programs rejected, {len(benign)} benign-hostile + a fused "
             f"chain emit no dangerous code/attr/import")


# ── LANG-7: the LSP over check() + the registry ─────────────────────────────────
def test_lsp_smoke(r: SubTestResult):
    print("\n--- LANG-7: tex_lsp diagnostics / completion / hover ---")
    try:
        s = tex_lsp.LSPServer()
        init, _ = s.handle("initialize", {})
        assert "hoverProvider" in init["capabilities"], init
        # broken program -> an error diagnostic
        _, notes = s.handle("textDocument/didOpen",
                            {"textDocument": {"uri": "u", "text": "@OUT = nosuchfn(@image);"}})
        diags = notes[0]["params"]["diagnostics"]
        assert diags and any(d["severity"] == 1 for d in diags), diags
        # clean program -> no diagnostics
        _, notes2 = s.handle("textDocument/didChange",
                             {"textDocument": {"uri": "u"},
                              "contentChanges": [{"text": "@OUT = @image * 0.5;"}]})
        assert notes2[0]["params"]["diagnostics"] == [], "clean program flagged"
        # completion contains real stdlib functions
        comp, _ = s.handle("textDocument/completion", {})
        labels = {i["label"] for i in comp["items"]}
        assert "gauss_blur" in labels and "lerp" in labels, "completion missing stdlib fns"
        # hover over a function name returns markdown
        s.handle("textDocument/didOpen",
                 {"textDocument": {"uri": "h", "text": "@OUT = gauss_blur(@image, 2.0);"}})
        hv, _ = s.handle("textDocument/hover",
                        {"textDocument": {"uri": "h"}, "position": {"line": 0, "character": 10}})
        assert hv and "gauss_blur" in hv["contents"]["value"], hv
        r.ok("LSP: diagnostics, clearing, completion, and hover all work")
    except Exception as e:
        r.fail("lsp smoke", str(e))


def test_lsp_bad_frames(r: SubTestResult):
    print("\n--- LANG-7: the stdio LSP loop survives malformed JSON-RPC frames ---")
    # A valid-JSON but non-OBJECT body (int / array (a JSON-RPC batch) / string / bool / null)
    # must be SKIPPED, not crash main() (msg.get(...) is read outside the handler try) — and null
    # must not masquerade as EOF. Drive main() over fake stdio and assert a valid request after
    # five bad frames is still answered.
    import io as _io, json as _json, sys as _sys

    def frame(obj_or_raw):
        body = obj_or_raw if isinstance(obj_or_raw, bytes) else _json.dumps(obj_or_raw).encode()
        return b"Content-Length: %d\r\n\r\n" % len(body) + body

    stream = (frame(b"42") + frame([1, 2, 3]) + frame(b'"x"') + frame(True) + frame(b"null")
              + frame({"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": {}})
              + frame({"jsonrpc": "2.0", "method": "exit"}))

    class _In:
        def __init__(self, d): self.b = _io.BytesIO(d)
        def readline(self): return self.b.readline()
        def read(self, n): return self.b.read(n)

    class _Out:
        def __init__(self): self.b = _io.BytesIO()
        def write(self, d): self.b.write(d)
        def flush(self): pass

    class _S:
        pass

    si, so = _S(), _S()
    si.buffer, so.buffer = _In(stream), _Out()
    real_in, real_out = _sys.stdin, _sys.stdout
    _sys.stdin, _sys.stdout = si, so
    try:
        tex_lsp.main()          # must not raise despite the five malformed frames
        emitted = so.buffer.b.getvalue()
        answered = b'"id": 7' in emitted or b'"id":7' in emitted
    except Exception as e:
        _sys.stdin, _sys.stdout = real_in, real_out
        r.fail("lsp bad frames", f"main() crashed on a malformed frame: {e}")
        return
    finally:
        _sys.stdin, _sys.stdout = real_in, real_out
    if answered:
        r.ok("LSP main() skips non-object frames (int/array/str/bool/null) and answers the next request")
    else:
        r.fail("lsp bad frames", "valid request after bad frames was not answered")


# ── TOOL-4: the `tex build` CLI ─────────────────────────────────────────────────
def test_cli_build(r: SubTestResult):
    print("\n--- TOOL-4: `tex build` validates a good tool and rejects a broken one ---")
    from TEX_Wrangle import tex_cli

    class _Args:
        def __init__(self, tool):
            self.tool, self.warm, self.emit, self.as_json = tool, False, None, False

    try:
        # a valid stock tool: build_fn returns without exiting
        tex_cli.build_fn(_Args(os.path.join(_STOCK, "grade.textool")))
        # a manifest with a TEX type error: preflight fails -> SystemExit(1)
        import json as _json
        import tempfile
        bad = {"manifest_schema": 1, "name": "Bad", "tex_language": "0.23",
               "code": "@OUT = nosuchfn(@image);",
               "inputs": [{"name": "image", "type": "IMAGE"}], "promoted_params": []}
        fd, p = tempfile.mkstemp(suffix=".textool")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            _json.dump(bad, fh)
        raised = False
        try:
            tex_cli.build_fn(_Args(p))
        except SystemExit as se:
            raised = bool(se.code)
        assert raised, "tex build did not fail on a type-erroring tool"
        r.ok("tex build passes a valid tool and exits non-zero on a broken one")
    except Exception as e:
        r.fail("cli build", str(e))
