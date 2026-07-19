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
import os
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
