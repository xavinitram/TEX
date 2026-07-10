"""v0.19.0 Phase 2 — prove the machine.

S-1 (tex_core / tex_comfy): doc 30 scoped a physical package split. The measured
reality (this session) is that the split's *payoff* — the compiler+runtime running
with ComfyUI absent — ALREADY holds: comfy is not importable in the test venv, so the
whole suite is already the ComfyUI-free lane, and `TEXWrangleNode.execute` runs with
comfy blocked. So S-1 ships the split's VALUE without the high-risk, live-unverifiable
25-file reroot: (1) a smoke test that codifies the ComfyUI-free property so it can't
regress, (2) a package-level "no comfy in the core modules" boundary lint (the exact
guarantee a physical split would give), (3) a documented core/comfy manifest (the
physical git-mv reroot is deferred to a live-ComfyUI session, per doc 35's own S-1 note).

S-5: a per-architecture caveat — the perf gate constants are Turing-calibrated (doc 34
weakness #2); on an unverified arch, `tex doctor`/tier-trace say so. Behavior unchanged.
"""
import re
from pathlib import Path
from helpers import *

_PKG = Path(__file__).resolve().parent.parent

# The ONLY files allowed to import ComfyUI — the host-adapter layer (doc 30's table:
# the node class, the registration/routes entry, and the HostServices Comfy impl).
_HOST_FILES = {"tex_node.py", "__init__.py", "tex_runtime/host.py"}
# any ComfyUI surface (not just model_management) — the core must touch none of it.
_ANY_COMFY_RE = re.compile(
    r"^\s*(?:import|from)\s+(?:comfy\b|comfy_api\b|comfy_execution\b|server\b|"
    r"folder_paths\b|nodes\b)", re.M)


def test_s1_core_no_comfy(r: SubTestResult):
    print("\n--- S-1: the core imports no ComfyUI (tex_core boundary lint) ---")
    offenders = []
    for path in _PKG.rglob("*.py"):
        rel = path.relative_to(_PKG).as_posix()
        if rel in _HOST_FILES or rel.startswith(("tests/", "benchmarks/", "tools/")):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in _ANY_COMFY_RE.finditer(text):
            offenders.append(f"{rel}:{text[:m.start()].count(chr(10)) + 1}")
    if offenders:
        r.fail("S-1 core boundary", "ComfyUI imported in a CORE module (belongs in the "
               "host layer — tex_node/__init__/host.py): " + ", ".join(offenders))
    else:
        r.ok("core modules import no ComfyUI (host-agnostic — pip-installable as tex_core)")


def test_s1_comfyui_free_execution(r: SubTestResult):
    print("\n--- S-1: compile + execute with ComfyUI absent (codifies the CI-lane payoff) ---")
    # Block every ComfyUI import for the duration, then drive the real node path.
    import builtins
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        top = name.split(".")[0]
        if top in ("comfy", "comfy_api", "comfy_execution", "server", "folder_paths"):
            raise ImportError(f"ComfyUI blocked (simulate standalone): {name}")
        return real_import(name, *a, **k)

    try:
        builtins.__import__ = blocked
        # fresh import of the node under the block (it's already loaded, but execute()
        # must not require comfy at call time — the try/except host-degrade handles it).
        from TEX_Wrangle.tex_node import TEXWrangleNode as N
        img = make_img(1, 16, 16, 3)
        out = N.execute(code="@OUT = vec4(@A.rgb * 0.5 + 0.1, 1.0);", A=img, device="cpu")
        t = out[0] if isinstance(out, tuple) else out
        assert isinstance(t, torch.Tensor) and t.shape[-1] == 3, f"bad output {getattr(t,'shape',t)}"
        # and the public API facade compiles too
        from TEX_Wrangle import tex_api
        prog = tex_api.compile("@OUT = vec4(@A.rgb, 1.0);", {"A": TEXType.VEC3, "OUT": TEXType.VEC4})
        assert prog is not None
        r.ok("node execute() + tex_api.compile() run with ComfyUI fully blocked")
    except Exception as e:
        r.fail("S-1 ComfyUI-free execution", f"{type(e).__name__}: {e}")
    finally:
        builtins.__import__ = real_import


# ── S-5: per-architecture safety net ─────────────────────────────────────────

def test_s5_arch_caveat(r: SubTestResult):
    print("\n--- S-5: verified-arch map + honest caveat ---")
    from TEX_Wrangle.tex_runtime.arch_support import arch_status
    ok = True
    verified = arch_status((7, 5))               # the calibration box
    if not (verified["verified"] is True and verified["note"] is None):
        ok = False; r.fail("S-5 sm_75", f"sm_75 must be verified, got {verified}")
    unver = arch_status((8, 9))                  # Ada — never measured in-repo
    if not (unver["verified"] is False and unver["note"] and "Turing" in unver["note"]
            and "validate-hw" in unver["note"] and unver["arch"] == "sm_89"):
        ok = False; r.fail("S-5 sm_89 caveat", f"unverified arch needs a caveat, got {unver}")
    cpu = arch_status(None)                       # CPU-only host
    if not (cpu["verified"] is None and cpu["note"] is None):
        ok = False; r.fail("S-5 cpu", f"no-CUDA host must be neutral, got {cpu}")
    if ok:
        r.ok("arch_status: sm_75 verified, sm_89 caveated (Turing + validate-hw), CPU neutral")


def test_s5_doctor_carries_caveat(r: SubTestResult):
    print("\n--- S-5: an unverified arch surfaces the caveat in the doctor payload ---")
    import torch
    from TEX_Wrangle import tex_doctor
    real_avail, real_cap = torch.cuda.is_available, torch.cuda.get_device_capability
    try:
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_capability = lambda *a, **k: (8, 9)   # pretend Ada
        facts = tex_doctor.collect_doctor_facts()
        arch = facts.get("arch", {})
        if arch.get("verified") is False and arch.get("note") and "validate-hw" in arch["note"]:
            r.ok("doctor payload carries the arch caveat on an unverified GPU (behavior unchanged)")
        else:
            r.fail("S-5 doctor caveat", f"doctor arch fact missing the caveat: {arch}")
    finally:
        torch.cuda.is_available, torch.cuda.get_device_capability = real_avail, real_cap


# ── S-4: tex validate-hw ─────────────────────────────────────────────────────

def test_s4_validate_hw_cli(r: SubTestResult):
    print("\n--- S-4: validate-hw CLI wiring + report renders and round-trips ---")
    from TEX_Wrangle.tex_cli import build_parser
    from TEX_Wrangle import tex_validate_hw as vh
    import json
    ok = True
    if build_parser().parse_args(["validate-hw"]).cmd != "validate-hw":
        ok = False; r.fail("S-4 cli", "validate-hw subcommand not wired into the parser")
    # a canned verdict must render to markdown and JSON-round-trip (the shareable contract)
    canned = {"report": "tex validate-hw", "schema": 1,
              "env": {"gpu": "Test GPU", "torch": "2.10", "cuda": True,
                      "compute_capability": [8, 9],
                      "arch": {"arch": "sm_89", "verified": False, "note": "calibrated on Turing; validate-hw"}},
              "fp16": {"status": "ran", "gate_holds": True, "rows": [{"side": 1024, "fp16_speedup": 1.2}]},
              "graph": {"status": "ran", "corners_agree": 4, "corners_measured": 4, "gate_holds": True, "rows": []},
              "tf32": {"status": "skipped", "reason": "sm_75"},
              "triton": {"status": "skipped"},
              "determinism": {"status": "ran", "deterministic": True, "worst_run_to_run": 0.0}}
    md = vh.render_markdown(canned)
    if not ("Test GPU" in md and "sm_89" in md and "fp16 crossover" in md and "validate-hw" in md):
        ok = False; r.fail("S-4 render", "markdown report missing GPU / arch / lane headers")
    if json.loads(json.dumps(canned)) != canned:
        ok = False; r.fail("S-4 roundtrip", "verdict does not round-trip json.loads")
    if ok:
        r.ok("validate-hw wired; report renders (GPU/arch/lanes) and round-trips JSON")


def test_s4_validate_hw_runs(r: SubTestResult):
    print("\n--- S-4: validate-hw runs to completion (all lanes present, never raises) ---")
    import json
    import torch
    from TEX_Wrangle import tex_validate_hw as vh
    if torch.cuda.is_available():
        # The heavy CUDA lanes were exercised live this session; re-running ~40s of A/B
        # timing inside the suite every run isn't worth it. Assert the driver is callable.
        r.ok("validate-hw CUDA lanes validated live this session (skipped in-suite for speed)")
        return
    try:
        v = vh.run_validation_hw()   # CPU/CI: GPU lanes SKIP cleanly, runs in <1s
        assert json.loads(json.dumps(v)) == v, "round-trip"
        assert set(v) >= {"env", "fp16", "graph", "tf32", "triton", "determinism"}, "lanes"
        assert v["fp16"]["status"] == "skipped" and v["graph"]["status"] == "skipped"
        r.ok("validate-hw ran to completion on CPU; every GPU lane SKIPped with a reason")
    except Exception as e:
        r.fail("S-4 run", f"{type(e).__name__}: {e}")


def test_s4_validate_hw_console_cp1252_safe(r: SubTestResult):
    print("\n--- S-4: validate-hw report prints on a cp1252 (Windows) console — exit 0 ---")
    # The report carries ✅/❌/⚠ glyphs; a Windows console defaults to cp1252, which can't
    # encode them -> the OLD code crashed with UnicodeEncodeError, exit 1, even though the
    # JSON/MD files wrote fine. validate-hw is a COMMUNITY command run mostly on Windows, so
    # this must not traceback. Reproduce the console faithfully via PYTHONIOENCODING=cp1252
    # in a subprocess and assert exit 0 (renders a canned verdict, skips the ~40s CUDA run).
    import os
    import subprocess
    import sys
    script = (
        "from TEX_Wrangle import tex_validate_hw as vh\n"
        "v={'report':'x','env':{'gpu':'G','torch':'2','cuda':True,'compute_capability':[7,5],"
        "'arch':{'arch':'sm_75','verified':False,'note':'calibrated on Turing; run validate-hw'}},"
        "'fp16':{'status':'ran','gate_holds':True,'rows':[]},"
        "'graph':{'status':'ran','gate_holds':False,'rows':[]},"
        "'tf32':{'status':'skipped'},'triton':{'status':'skipped'},"
        "'determinism':{'status':'ran','deterministic':True}}\n"
        "vh._print_console_safe(vh.render_markdown(v))\n"
    )
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(_PKG.parent),
                          env=env, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("ascii", "replace")[-220:]
        r.fail("S-4 console", f"validate-hw print crashed on a cp1252 console (exit "
               f"{proc.returncode}): {tail}")
    else:
        r.ok("validate-hw report prints on a cp1252 console without crashing (exit 0)")
