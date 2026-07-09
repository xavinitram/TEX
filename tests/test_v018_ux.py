"""
v0.18.0 UX/debugging (DBG-1 perf HUD payload; UX-1 diagnostics; DBG-3 NaN overlay; ...).
The frontend *rendering* halves (badge, hover, doctor panel) need a live-ComfyUI session;
these tests pin the BACKEND contracts that feed them.
"""
from helpers import *


class _StubNodeOutput:
    """Stand-in for comfy_api's IO.NodeOutput so the v3 `ui=` path is testable without
    ComfyUI on the path (the standalone test venv has _V3_AVAILABLE=False)."""
    def __init__(self, *args, ui=None, **kw):
        self.args = args
        self.ui = ui

    @property
    def result(self):
        return self.args


class _StubIO:
    NodeOutput = _StubNodeOutput


def test_dbg1_perf_hud_payload(r: SubTestResult):
    print("\n--- DBG-1: tier/timing HUD payload on the v3 ui= channel ---")
    import TEX_Wrangle.tex_node as TN
    from TEX_Wrangle.tex_runtime import tier_trace
    img = make_img(1, 16, 16, 3, seed=1)
    code = "@OUT = vec4(@A.rgb * 1.1, 1.0);"

    # (a) tuple path unchanged when v3 is absent (additive-only: no ui leaks in)
    tup = TN.TEXWrangleNode.execute(code=code, A=img, device="cpu")
    if not isinstance(tup, tuple):
        r.fail("DBG-1 tuple path", f"expected tuple with v3 absent, got {type(tup).__name__}")
        return

    # (b) v3 path carries the HUD facts (stub NodeOutput so we can read .ui headlessly)
    saved_v3, saved_io = TN._V3_AVAILABLE, TN.IO
    TN._V3_AVAILABLE, TN.IO = True, _StubIO
    try:
        out = TN.TEXWrangleNode.execute(code=code, A=img, device="cpu")
        ui = getattr(out, "ui", None)
        if not (isinstance(ui, dict) and "tex_perf" in ui and ui["tex_perf"]):
            r.fail("DBG-1 ui payload", f"missing tex_perf in ui={ui}")
            return
        perf = ui["tex_perf"][0]
        missing = [k for k in ("tier", "fallback_from", "reason", "elapsed_ms",
                               "device", "precision") if k not in perf]
        if missing:
            r.fail("DBG-1 ui fields", f"payload missing {missing}")
            return
        # the payload's tier matches the trace (same facts, not invented)
        tr = tier_trace.last()
        expect_tier = tr.tier if tr is not None else "interpreter"
        if perf["tier"] != expect_tier:
            r.fail("DBG-1 tier consistency",
                   f"ui tier {perf['tier']!r} != tier_trace {expect_tier!r}")
            return
        if not (isinstance(perf["elapsed_ms"], (int, float)) and perf["elapsed_ms"] >= 0):
            r.fail("DBG-1 elapsed_ms", f"bad elapsed_ms {perf['elapsed_ms']!r}")
            return
        # additive-only: the result tuple equals the v3-absent tuple, bit-for-bit
        a = tup[0]; b = out.result[0]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if (a.float() - b.float()).abs().max().item() != 0.0:
                r.fail("DBG-1 additive-only", "ui= path changed the result tensor")
                return
    finally:
        TN._V3_AVAILABLE, TN.IO = saved_v3, saved_io

    # (c) auto-precision decision is surfaced (CUDA gate; on CPU it reports fp32)
    TN._V3_AVAILABLE, TN.IO = True, _StubIO
    try:
        out = TN.TEXWrangleNode.execute(code=code, A=img, device="cpu", precision="auto")
        perf = out.ui["tex_perf"][0]
        if perf["precision"] != "fp32":  # CPU always resolves auto->fp32
            r.fail("DBG-1 auto precision", f"CPU auto should be fp32, got {perf['precision']}")
            return
        if not perf["precision_reason"]:
            r.fail("DBG-1 auto reason", "auto decision has no recorded reason")
            return
    finally:
        TN._V3_AVAILABLE, TN.IO = saved_v3, saved_io

    r.ok("ui= carries tier/fallback/reason/elapsed_ms/device/precision; matches trace; "
         "additive-only; auto decision surfaced")


def _diag_text(code):
    """The full rendered diagnostic (message + hint) a user would see — the REAL path,
    not `str(exc)` (which drops the hint). Asserts on this, not internal hint tables."""
    from TEX_Wrangle.tex_compiler.parser import ParseError
    try:
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        TypeChecker(binding_types={"A": TEXType.VEC3, "B": TEXType.VEC3}, source=code).check(prog)
        return "NO ERROR"
    except TEXMultiError as e:
        return " ".join((d.message or "") + " " + (d.hint or "") for d in e.diagnostics)
    except (ParseError, TypeCheckError) as e:
        d = getattr(e, "diagnostic", None)
        return (d.message + " " + (d.hint or "")) if d else str(e)
    except Exception as e:  # noqa
        return str(e)


# 10 common GLSL/HLSL-brain mistakes; each must render a diagnostic whose message+hint
# contains a helpful pointer (UX-1: was 8/10 — foreign types + the reserved-var gotcha).
_MISTAKES = {
    "texture2D":   ("@OUT=vec4(texture2D(@A,u,v).rgb,1.0);", "sample"),
    "float3 type": ("float3 p=vec3(0.0); @OUT=vec4(p,1.0);", "vec3"),
    "float4 type": ("float4 c=@A; @OUT=c;", "vec4"),
    "half type":   ("half h=0.5; @OUT=@A;", "float"),
    "mat2 type":   ("mat2 m; @OUT=@A;", "mat3"),
    "reserved v":  ("float v=0.5; @OUT=vec4(v,v,v,1.0);", "built-in"),
    "reserved u":  ("float u=0.1; @OUT=@A;", "built-in"),
    "reserved ix": ("float ix=0.0; @OUT=@A;", "built-in"),
    "reserved PI": ("float PI=3.0; @OUT=@A;", "built-in"),
    "unknown fn":  ("@OUT=vec4(mix3(@A,@B,0.5),1.0);", "function"),
}


def test_ux1_diagnostics_reachability(r: SubTestResult):
    print("\n--- UX-1: diagnostics reachability (foreign types + reserved-var hints) ---")
    gaps = []
    for name, (code, expect) in _MISTAKES.items():
        text = _diag_text(code)
        if expect.lower() not in text.lower():
            gaps.append(f"{name}: no '{expect}' in rendered diagnostic — got {text[:80]!r}")
    if gaps:
        r.fail("UX-1 diagnostics", f"{len(gaps)}/10 unhelpful:\n  " + "\n  ".join(gaps))
    else:
        r.ok(f"all {len(_MISTAKES)}/10 common mistakes render a helpful message (rendered "
             "message+hint, the real frontend path)")


def test_dbg3_nan_overlay(r: SubTestResult):
    print("\n--- DBG-3: NaN/Inf magenta overlay ---")
    from TEX_Wrangle.tex_node import _nan_highlight, TEXWrangleNode as N
    fails = []
    # (a) the helper: a non-finite CHANNEL magenta-flags the whole PIXEL; finite untouched
    img = torch.rand(1, 4, 4, 3)
    img[0, 0, 0, 0] = float("nan")
    img[0, 1, 1, 1] = float("inf")
    out = _nan_highlight(img.clone())
    mag = torch.tensor([1.0, 0.0, 1.0])
    if not torch.allclose(out[0, 0, 0], mag):
        fails.append("NaN pixel not magenta")
    if not torch.allclose(out[0, 1, 1], mag):
        fails.append("Inf pixel not magenta")
    if not torch.allclose(out[0, 2, 2], img[0, 2, 2]):
        fails.append("finite pixel was altered")
    # (b) zero-cost / no-op guarantees: a finite tensor and a non-tensor pass straight through
    fin = torch.rand(1, 4, 4, 3)
    if _nan_highlight(fin) is not fin:
        fails.append("finite tensor was copied (should be the same object)")
    if _nan_highlight("str") != "str":
        fails.append("non-tensor not passed through")
    # (c) full node path: on a NaN-producing program the toggle paints magenta; off leaves
    #     the raw (non-magenta) output — proving the overlay only runs when enabled
    nan_prog = "@OUT = vec4(vec3(pow(@A.r - 2.0, 0.5)), 1.0);"  # pow(neg, 0.5) -> NaN
    A = make_img(1, 8, 8, 3, seed=2)
    on = N.execute(code=nan_prog, A=A, device="cpu", debug_nan_highlight=True)[0]
    off = N.execute(code=nan_prog, A=A, device="cpu", debug_nan_highlight=False)[0]
    if isinstance(on, torch.Tensor):
        # at least one magenta pixel present when on
        is_mag = (on[..., 0] == 1.0) & (on[..., 1] == 0.0) & (on[..., 2] == 1.0)
        if not bool(is_mag.any()):
            fails.append("toggle ON produced no magenta pixels on a NaN program")
        if isinstance(off, torch.Tensor):
            off_mag = (off[..., 0] == 1.0) & (off[..., 1] == 0.0) & (off[..., 2] == 1.0)
            if bool(off_mag.all()) and not bool(is_mag.all()):
                fails.append("toggle OFF magenta-painted anyway")
    if fails:
        r.fail("DBG-3 NaN overlay", "; ".join(fails))
    else:
        r.ok("non-finite pixels -> magenta (whole pixel); finite/non-tensor no-op; "
             "node toggle paints on-only")


def test_lx5_debug_print(r: SubTestResult):
    print("\n--- LX-5: debug_print value-at-pixel probe (interpreter-only) ---")
    from TEX_Wrangle.tex_runtime import tier_trace
    from failure_harness import run_tier, max_diff
    binds = {"A": make_img(1, 8, 8, 3, seed=7)}
    probed = '@OUT = vec4(vec3(debug_print("luma", luma(@A.rgb), 2, 3)), 1.0);'
    plain = "@OUT = vec4(vec3(luma(@A.rgb)), 1.0);"
    fails = []

    # (a) @OUT is bit-identical with/without the probe (it returns value unchanged)
    tier_trace.reset()
    a = run_tier(probed, binds, "interp")
    b = run_tier(plain, binds, "interp")
    if max_diff(a, b) != 0.0:
        fails.append(f"@OUT changed by the probe (maxdiff {max_diff(a, b):.2e})")

    # (b) the probe recorded the value at pixel (x=2, y=3) — equals OUT there
    probes = tier_trace.get_probes()
    if not probes:
        fails.append("no probe recorded under the interpreter")
    else:
        pv = probes[0]["value"]
        pv = pv[0] if isinstance(pv, list) else pv
        out_at = a["OUT"][0, 3, 2, 0].item()  # value[0, y=3, x=2]; OUT=vec3(luma)
        if abs(float(pv) - out_at) > 1e-4:
            fails.append(f"probe value {pv} != luma@(2,3) {out_at}")
        if probes[0]["label"] != "luma" or probes[0]["x"] != 2 or probes[0]["y"] != 3:
            fails.append(f"probe label/coords wrong: {probes[0]}")

    # (c) codegen must NOT silently drop it — it falls back to the interpreter, so the
    #     result still matches AND the probe still fires (never a silent no-op).
    tier_trace.reset()
    c = run_tier(probed, binds, "codegen")
    if max_diff(a, c) > 1e-5:
        fails.append(f"codegen path result diverged (maxdiff {max_diff(a, c):.2e})")
    if not tier_trace.get_probes():
        fails.append("codegen tier SILENTLY dropped the probe (LX-5 requires a hard reject)")

    if fails:
        r.fail("LX-5 debug_print", "; ".join(fails))
    else:
        r.ok("probe records value-at-pixel + returns value unchanged (@OUT bit-exact); "
             "codegen falls back so the probe never silently no-ops")


def test_dbg4_doctor(r: SubTestResult):
    print("\n--- DBG-4: tex doctor environment report (never raises) ---")
    from TEX_Wrangle.tex_doctor import collect_doctor_facts
    KEYS = {"torch", "triton", "msvc", "cache", "tiers", "recent_tiers"}
    fails = []
    facts = collect_doctor_facts()
    if not KEYS <= set(facts):
        fails.append(f"missing keys: {KEYS - set(facts)}")
    # tier routing is reported for every (mode, device)
    tiers = facts.get("tiers", {})
    if isinstance(tiers, dict) and "cuda_graph@cuda" not in tiers:
        fails.append("tier routing not reported")

    # a probe that throws must NOT take the report down — still 200-shaped, all keys,
    # the failing fact captured as an {error} stub, the others intact.
    saved = torch.cuda.is_available
    torch.cuda.is_available = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        facts2 = collect_doctor_facts()
    except Exception as e:
        fails.append(f"collect raised when a probe threw: {e}")
        facts2 = {}
    finally:
        torch.cuda.is_available = saved
    if facts2:
        if not KEYS <= set(facts2):
            fails.append("keys dropped when a probe threw")
        if "error" not in facts2.get("torch", {}):
            fails.append("torch fact didn't capture the probe error")
        if not isinstance(facts2.get("tiers"), dict) or "error" in facts2["tiers"]:
            fails.append("an unrelated fact was collateral-damaged")

    # doc 33: the /tex_wrangle/doctor route serialises this payload — it must be strict JSON
    # (no NaN/Inf/tensor/set leaking a non-conforming token to the browser).
    import json
    for label, payload in (("facts", facts), ("facts-under-probe-error", facts2)):
        if payload:
            try:
                json.dumps(payload, allow_nan=False)
            except (ValueError, TypeError) as e:
                fails.append(f"{label} not strict-JSON serialisable: {type(e).__name__}: {e}")

    if fails:
        r.fail("DBG-4 doctor", "; ".join(fails))
    else:
        r.ok("doctor reports torch/triton/msvc/cache/tiers/recent_tiers; a throwing probe "
             "is isolated (all keys stay, others intact); payload is strict JSON — never 500s")


def test_lx5_json_nan_safe(r: SubTestResult):
    print("\n--- LX-5: debug_print of NaN/Inf serializes as valid JSON (audit) ---")
    import json
    from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib as S
    from TEX_Wrangle.tex_runtime import tier_trace
    tier_trace.reset()
    S.fn_debug_print("scalar_nan", float("nan"))
    S.fn_debug_print("scalar_inf", float("inf"))
    S.fn_debug_print("tensor_nan", torch.full((1, 4, 4, 3), float("nan")), 0, 0)
    payload = {"tex_probes": tier_trace.get_probes()}
    try:
        json.dumps(payload, allow_nan=False)  # strict JSON: raises on NaN/Infinity tokens
        r.ok("non-finite probe values sanitized to null; payload is strict-JSON valid")
    except ValueError as e:
        r.fail("LX-5 JSON NaN", f"probe payload is invalid JSON (NaN/Inf leaked): {e}")


def test_dbg1_nan_fingerprint(r: SubTestResult):
    print("\n--- DBG-1: debug_nan_highlight busts the cache fingerprint (audit) ---")
    from TEX_Wrangle.tex_node import TEXWrangleNode as N
    base = dict(code="@OUT = vec4(@A.rgb, 1.0);", A=make_img(1, 8, 8, 3, seed=1), device="cpu")
    fp_off = N.fingerprint_inputs(**base, debug_nan_highlight=False)
    fp_on = N.fingerprint_inputs(**base, debug_nan_highlight=True)
    if fp_off == fp_on:
        r.fail("DBG-1 fingerprint", "toggling debug_nan_highlight does not bust the cache "
               "(the overlay silently does nothing on first toggle)")
    else:
        r.ok("debug_nan_highlight toggle changes the fingerprint (fresh cook on toggle)")


def test_ux2_tooltip_honesty(r: SubTestResult):
    print("\n--- UX-2: compile_mode tooltip states the Triton reality ---")
    import re
    import TEX_Wrangle.tex_node as TN
    src = Path(TN.__file__).read_text(encoding="utf-8")
    m = re.search(r'IO\.Combo\.Input\(\s*"compile_mode".*?tooltip="([^"]*)"', src, re.S)
    tip = m.group(1) if m else ""
    if "Triton" not in tip:
        r.fail("UX-2 tooltip", "compile_mode tooltip does not mention Triton (dishonest "
               "about auto/torch_compile falling back to the interpreter without it)")
    elif "fall back" not in tip.lower():
        r.fail("UX-2 tooltip", "tooltip mentions Triton but not the fallback consequence")
    else:
        r.ok("compile_mode tooltip is honest: names the Triton dependency + the "
             "interpreter fallback (points to `tex doctor`)")
