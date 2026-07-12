"""v0.19.0 Phase 3 — show the truth & open the loop (backend-verifiable half).

C6-ux (snippet discoverability), S-3 (LLM cheatsheet), S-2 (workflow JSONs) and C4-ux
(near-singularity cyan diagnostic) are all testable headlessly. The frontend items
(C1-ux HUD, C2-ux doctor modal) ship as JS and are signed off in the LIVE session
(docs/live-session-checklist.md) — they can't be render-verified without a ComfyUI canvas.
"""
import json
import os
import sys
from pathlib import Path
from helpers import *

_PKG = Path(__file__).resolve().parent.parent
_TOOLS = str(_PKG / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)


# ── C6-ux: snippet discoverability ───────────────────────────────────────────

def test_c6ux_default_code_snippet_hint(r: SubTestResult):
    print("\n--- C6-ux: default node code advertises the snippet browser ---")
    from TEX_Wrangle.tex_node import TEXWrangleNode as N
    schema = N.INPUT_TYPES() if hasattr(N, "INPUT_TYPES") else None
    # Pull the code widget's default out of the schema (v3 IO.Schema or classic dict).
    default = _find_code_default(schema)
    n_examples = len(list((_PKG / "examples").glob("*.tex")))
    if default is None:
        r.fail("C6-ux", "could not locate the code widget default")
    elif "TEX Snippets" not in default:
        r.fail("C6-ux hint", f"default code lacks the snippet hint: {default!r}")
    elif str(n_examples) not in default:
        r.fail("C6-ux count", f"hint says a count that isn't the real {n_examples} examples")
    else:
        r.ok(f"default code points to the snippet browser ({n_examples} examples, count matches)")


def _find_code_default(schema):
    """Best-effort extraction of the 'code' widget default across schema representations."""
    try:
        # classic dict: {"required"/"optional": {"code": ("STRING", {"default": ...})}}
        for grp in ("required", "optional"):
            spec = (schema or {}).get(grp, {}).get("code")
            if spec and len(spec) > 1 and isinstance(spec[1], dict) and "default" in spec[1]:
                return spec[1]["default"]
    except Exception:
        pass
    # v3 IO.Schema: read the source default string directly (stable literal).
    src = (_PKG / "tex_node.py").read_text(encoding="utf-8")
    i = src.find('"// TEX Wrangle\\n"')
    if i != -1:
        return src[i:i + 400]
    return None


# ── S-3: LLM authoring cheatsheet ────────────────────────────────────────────

def test_s3_cheatsheet_drift(r: SubTestResult):
    print("\n--- S-3: LLM cheatsheet regenerates identically (drift guard) ---")
    import gen_llm_cheatsheet as G
    out = _PKG / "wiki" / "LLM-Cheatsheet.md"
    if not out.exists():
        # wiki/ is a separate (gitignored) repo checkout — absent on a fresh CI
        # clone. Skip the drift guard when the page isn't present rather than
        # failing CI for a file this repo intentionally doesn't track.
        r.skip("S-3 cheatsheet drift", "wiki/ checkout absent (separate wiki repo); run tools/gen_llm_cheatsheet.py in a wiki checkout")
        return
    if out.read_text(encoding="utf-8") != G.render():
        r.fail("S-3 drift", "LLM-Cheatsheet.md is stale — run tools/gen_llm_cheatsheet.py")
    else:
        r.ok("LLM-Cheatsheet.md is current (regenerates byte-identical from the registry)")


def test_s3_worked_examples_compile(r: SubTestResult):
    print("\n--- S-3: every worked example in the cheatsheet compiles (spot-check) ---")
    import gen_llm_cheatsheet as G
    from TEX_Wrangle import tex_api
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    bad = []
    for intent, code in G.WORKED:
        try:
            tex_api.compile(code, bt)
        except Exception as e:
            bad.append(f"{intent[:40]}: {type(e).__name__}")
    if bad:
        r.fail("S-3 examples", "worked examples must compile: " + "; ".join(bad))
    else:
        r.ok(f"all {len(G.WORKED)} cheatsheet worked examples compile (grade/vignette/mix/blur)")


# ── S-2: workflow JSONs ──────────────────────────────────────────────────────

def test_s2_workflows_smoke(r: SubTestResult):
    print("\n--- S-2: example workflows parse, embed a real snippet, compile, sink to Preview ---")
    from TEX_Wrangle import tex_api
    wf_dir = _PKG / "examples" / "workflows"
    files = sorted(wf_dir.glob("*.json")) if wf_dir.exists() else []
    if not files:
        r.fail("S-2 missing", "no workflows — run tools/gen_workflows.py")
        return
    examples = {p.read_text(encoding="utf-8") for p in (_PKG / "examples").glob("*.tex")}
    bad = []
    for p in files:
        try:
            wf = json.loads(p.read_text(encoding="utf-8"))
            tex = [n for n in wf["nodes"] if n["type"] == "TEX_Wrangle"][0]
            code = tex["widgets_values"][0]
            assert code in examples, "embedded code is not a shipped examples/*.tex snippet"
            assert any(n["type"] == "PreviewImage" for n in wf["nodes"]), "no PreviewImage sink"
            tex_api.compile(code, {"OUT": TEXType.VEC4})
        except Exception as e:
            bad.append(f"{p.name}: {type(e).__name__}: {e}")
    if bad:
        r.fail("S-2 smoke", "; ".join(bad))
    else:
        r.ok(f"all {len(files)} workflows: valid JSON, real snippet, compiles, PreviewImage sink")


def test_s2_workflows_drift(r: SubTestResult):
    print("\n--- S-2: workflows regenerate identically (drift guard) ---")
    import gen_workflows as G
    wf_dir = _PKG / "examples" / "workflows"
    want = G.generate()
    stale = [f for f, c in want.items()
             if not (wf_dir / f).exists() or (wf_dir / f).read_text(encoding="utf-8") != c]
    if stale:
        r.fail("S-2 drift", f"stale/missing workflows (run tools/gen_workflows.py): {stale}")
    else:
        r.ok(f"all {len(want)} workflows current (regenerate byte-identical from snippets)")


# ── C4-ux: near-singularity cyan diagnostic ──────────────────────────────────

_SING = "float z = u - u;\n@OUT = vec4(vec3(sdiv(1.0, z)), 1.0);"
_CLEAN = "@OUT = vec4(@A.rgb, 1.0);"


def _cyan_count(t):
    t = t.float()
    return int(((t[..., 0] == 0.0) & (t[..., 1] == 1.0) & (t[..., 2] == 1.0)).sum())


def test_c4ux_cyan_on_singularity(r: SubTestResult):
    print("\n--- C4-ux: guarded-division epsilon branch paints cyan (toggle on) ---")
    from TEX_Wrangle.tex_node import TEXWrangleNode as N
    from TEX_Wrangle.tex_runtime import guard_trace
    img = make_img(1, 8, 8, 3)
    out = N.execute(code=_SING, A=img, device="cpu", debug_nan_highlight=True)
    t = out[0] if isinstance(out, tuple) else out
    if _cyan_count(t) != 64:
        r.fail("C4-ux cyan", f"1/(u-u) should be all cyan, got {_cyan_count(t)}/64")
    elif guard_trace.count() <= 0:
        r.fail("C4-ux count", "near-singularity count should be > 0")
    elif guard_trace.armed():
        r.fail("C4-ux disarm", "guard_trace left armed after the cook")
    else:
        r.ok(f"sdiv(1,u-u) → 64/64 cyan pixels, count={guard_trace.count()}, disarmed after")


def test_c4ux_clean_no_cyan(r: SubTestResult):
    print("\n--- C4-ux: a clean program shows no cyan (no false positives) ---")
    from TEX_Wrangle.tex_node import TEXWrangleNode as N
    img = make_img(1, 8, 8, 3)
    out = N.execute(code=_CLEAN, A=img, device="cpu", debug_nan_highlight=True)
    t = out[0] if isinstance(out, tuple) else out
    if _cyan_count(t) != 0:
        r.fail("C4-ux clean", f"clean program should have no cyan, got {_cyan_count(t)}")
    else:
        r.ok("clean program with the toggle on paints zero cyan pixels")


def test_c4ux_additive_and_zero_cost_off(r: SubTestResult):
    print("\n--- C4-ux: diagnostic is additive; guards don't alter output when off ---")
    from TEX_Wrangle.tex_node import TEXWrangleNode as N
    from TEX_Wrangle.tex_runtime import guard_trace
    img = make_img(1, 16, 16, 3)
    # A program that USES sdiv but never hits the guard (denominator never ~0): output must
    # be bit-identical with the toggle on vs off — the guard hook changed nothing.
    code = "@OUT = vec4(vec3(sdiv(@A.r, 2.0)), 1.0);"
    off = N.execute(code=code, A=img, device="cpu", debug_nan_highlight=False)
    on = N.execute(code=code, A=img, device="cpu", debug_nan_highlight=True)
    off_t = (off[0] if isinstance(off, tuple) else off).float()
    on_t = (on[0] if isinstance(on, tuple) else on).float()
    if not torch.equal(off_t, on_t):
        r.fail("C4-ux additive", f"toggle changed a non-singular output (maxdiff "
               f"{float((off_t - on_t).abs().max()):.2e})")
    elif guard_trace.armed():
        r.fail("C4-ux off-state", "guard_trace armed after a toggle-off cook (must be free)")
    else:
        r.ok("non-singular sdiv output bit-identical on/off; disarmed when off (zero-cost)")


# ── C1-ux / C2-ux: frontend presence + payload contract (render-verified in LIVE) ─────

def test_c1ux_c2ux_frontend_present(r: SubTestResult):
    print("\n--- C1-ux/C2-ux: HUD badge + doctor modal wired in the JS (render check = LIVE) ---")
    js = (_PKG / "js" / "tex_extension.js").read_text(encoding="utf-8")
    need = [
        ("_texEnsurePerfBadge", "C1-ux DOM perf badge factory"),
        ("tex-floating-perf-badge", "C1-ux badge CSS/class"),
        ("_texUpdatePerfBadge", "C1-ux badge update from tex_perf"),
        ("this._texProbes", "C1-ux probe capture (debug_print)"),
        ("near_singularities", "C4-ux count surfaced in the badge"),
        ("_showDoctorDialog", "C2-ux doctor modal"),
        ('"TEX Doctor"', "C2-ux menu entry"),
        ("tex-doctor-caveat", "C2-ux arch-caveat styling"),
    ]
    missing = [f"{s} ({why})" for s, why in need if s not in js]
    if missing:
        r.fail("C1/C2-ux JS", "frontend hook removed: " + "; ".join(missing))
    else:
        r.ok("HUD badge + probes + doctor modal wired in the JS (8 hooks present)")


def test_c2ux_doctor_payload_shape(r: SubTestResult):
    print("\n--- C2-ux: the doctor payload the modal renders has the expected shape ---")
    from TEX_Wrangle.tex_doctor import collect_doctor_facts
    facts = collect_doctor_facts()
    # _renderDoctorFacts walks these top-level keys; arch must carry verified/note (S-5).
    top = {"torch", "triton", "msvc", "cache", "tiers", "arch"}
    arch = facts.get("arch", {})
    if not top.issubset(facts):
        r.fail("C2-ux shape", f"doctor payload missing keys: {top - set(facts)}")
    elif not ({"arch", "verified", "note"} <= set(arch)):
        r.fail("C2-ux arch", f"arch fact missing verified/note the modal reads: {arch}")
    else:
        r.ok("doctor payload has all keys the modal walks; arch carries verified+note")
