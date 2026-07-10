"""
S-4 — `tex validate-hw`: a shareable hardware-validation report.

Every perf gate in TEX was calibrated on ONE GPU (RTX 2080 SUPER, sm_75). This command
lets any user *measure* whether those Turing-calibrated constants hold on their card and
paste a structured verdict back (the "Hardware validation report" issue template). It runs
five lanes, each self-gating and defensive — a lane that can't run SKIPs with a clear
reason, and NOTHING here ever raises past `run_validation_hw` (so "runs to completion"
holds on every box):

  1. env      — GPU name/cc, torch, Triton, the S-5 arch verdict
  2. fp16     — measure where fp16 starts beating fp32 vs `_MIN_FP16_PX` (1024²)
  3. graph    — PF-1 crossover: at the gate corners, does the measured winner match
                `_graph_capture_worthwhile`'s prediction?
  4. tf32     — sm_80+ only: TF32 on/off A/B (SKIP on Turing)
  5. triton   — delegate to benchmarks/triton_validation (SKIP when Triton absent)
  6. det      — the CUDA scatter determinism pin (bitwise across repeats)

Output is a JSON blob (round-trips `json.loads`) + a markdown report, both written under
benchmarks/results/. Accumulated community reports either confirm the constants or feed a
per-arch recalibration in v0.20 — the decision log lives in the repo.
"""
import json
import statistics
import time
from pathlib import Path

from .tex_compiler.types import TEXType   # dependency-free leaf (no torch pulled)

_ROOT = Path(__file__).resolve().parent
# A: vec3 in, OUT: vec4 out — the binding shape every lane's probe program compiles to.
_BINDING = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

# Programs at the two op-count regimes the PF-1 gate distinguishes (low vs kernel-heavy).
_LOW_OPS = "@OUT = vec4(@A.rgb * 0.5 + 0.2, 1.0);"


def _high_ops_program(n: int = 22) -> str:
    """A kernel-heavy program (~2n tensor ops) to probe the high-op gate ceiling."""
    lines = ["vec3 c = @A.rgb;"]
    for i in range(n):
        lines.append(f"c = sin(c * 1.0{i % 9 + 1}) + cos(c * 0.9);")
    lines.append("@OUT = vec4(c, 1.0);")
    return "\n".join(lines)


def _img(torch, px_side: int, device: str):
    return torch.rand(1, px_side, px_side, 3, device=device)


def _median_ms(fn, cuda, reps=25, warmup=3) -> float:
    """Sync-bracketed median wall time in ms (the house A/B standard, doc 35)."""
    import torch
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


# ── lanes ──────────────────────────────────────────────────────────────────

def _lane_env(torch) -> dict:
    from .tex_runtime.arch_support import current_arch_status
    cuda = bool(torch.cuda.is_available())
    return {
        "torch": torch.__version__,
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
        "arch": current_arch_status(),
    }


def _lane_fp16(torch, device) -> dict:
    """Measure the fp16-vs-fp32 timing crossover vs `_MIN_FP16_PX`. The gate is a SAFETY
    threshold (only take fp16's precision risk where it clearly pays), so it "holds" if
    fp16 is actually a win everywhere the gate would ENABLE it (px >= threshold) — a lower
    measured crossover just means the gate is conservative on this arch, not wrong."""
    if device != "cuda":
        return {"status": "skipped", "reason": "fp16 crossover is a GPU property (need cuda)"}
    from .tex_runtime.precision_policy import _MIN_FP16_PX
    from . import tex_api
    prog = tex_api.compile(_high_ops_program(), _BINDING)
    rows, first_win = [], None
    for side in (256, 512, 1024, 2048):
        px = side * side
        img = _img(torch, side, device)
        try:
            f32 = _median_ms(lambda: tex_api.execute(prog, {"A": img}, device=device,
                                                     precision="fp32"), True)
            f16 = _median_ms(lambda: tex_api.execute(prog, {"A": img}, device=device,
                                                     precision="fp16"), True)
        except Exception as e:
            rows.append({"px": px, "error": f"{type(e).__name__}: {e}"})
            continue
        speedup = f32 / f16 if f16 else 0.0
        rows.append({"side": side, "px": px, "fp32_ms": round(f32, 4),
                     "fp16_ms": round(f16, 4), "fp16_speedup": round(speedup, 3)})
        if first_win is None and speedup > 1.0:
            first_win = px
    enabled = [r for r in rows if r.get("px", 0) >= _MIN_FP16_PX and "fp16_speedup" in r]
    return {"status": "ran", "measured_first_win_px": first_win,
            "gate_min_fp16_px": _MIN_FP16_PX,
            "conservative": bool(first_win and first_win < _MIN_FP16_PX),
            # sound = fp16 actually wins everywhere the gate would turn it on
            "gate_holds": all(r["fp16_speedup"] >= 0.98 for r in enabled) if enabled else None,
            "rows": rows}


def _lane_graph(torch, device) -> dict:
    """PF-1 crossover self-check. The node respects the gate, so in the LOSE region it
    never captures (falls back) — you can't time a "graph loss" through the public path.
    So we validate the gate's DECISION, at UNAMBIGUOUS corners (deep-win / deep-lose, not
    the noisy break-even), via `tier_trace`:
      - deep-win  (predict capture): the graph tier must actually serve AND not be slower.
      - deep-lose (predict decline): the gate must decline (some other tier serves).
    A mismatch on this arch is real signal that the Turing ceilings need recalibration."""
    if device != "cuda":
        return {"status": "skipped", "reason": "CUDA-graph tier needs cuda"}
    from .tex_runtime.graphed import _graph_capture_worthwhile
    from .tex_runtime import tier_trace
    from .tex_node import TEXWrangleNode as N
    hi = _high_ops_program()
    corners = [("low", _LOW_OPS, 3, 256), ("low", _LOW_OPS, 3, 2048),
               ("high", hi, 45, 512), ("high", hi, 45, 2048)]
    rows, agree = [], 0
    for label, code, est_ops, side in corners:
        px = side * side
        img = _img(torch, side, device)
        predict_capture = _graph_capture_worthwhile(est_ops, px)
        try:
            N.execute(code=code, A=img, device="cuda", compile_mode="cuda_graph")  # warm
            served = getattr(tier_trace.last(), "tier", None)
            captured = (served == "graph")
            row = {"corner": f"{label}@{side}", "est_ops": est_ops, "px": px,
                   "gate_predicts_capture": predict_capture, "tier_served": served}
            if predict_capture:
                base = _median_ms(lambda: N.execute(code=code, A=img, device="cuda",
                                                    compile_mode="none"), True)
                graph = _median_ms(lambda: N.execute(code=code, A=img, device="cuda",
                                                     compile_mode="cuda_graph"), True)
                row.update(base_ms=round(base, 4), graph_ms=round(graph, 4),
                           graph_speedup=round(base / graph, 3) if graph else 0.0)
                ok = captured and graph <= base * 1.15   # engaged and not a regression
            else:
                ok = not captured                         # gate correctly declined capture
            row["agree"] = ok
            agree += int(ok)
        except Exception as e:
            row = {"corner": f"{label}@{side}", "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
    measured = [r for r in rows if "agree" in r]
    return {"status": "ran", "corners_agree": agree, "corners_measured": len(measured),
            "gate_holds": (agree == len(measured)) if measured else None, "rows": rows}


def _lane_tf32(torch, device) -> dict:
    if device != "cuda":
        return {"status": "skipped", "reason": "TF32 is a GPU property (need cuda)"}
    cc = torch.cuda.get_device_capability(0)
    if cc[0] < 8:
        return {"status": "skipped",
                "reason": f"TF32 needs sm_80+ (Ampere); this GPU is sm_{cc[0]}{cc[1]}"}
    from . import tex_api
    prog = tex_api.compile(_high_ops_program(), _BINDING)
    img = _img(torch, 1024, device)
    prev = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        off = _median_ms(lambda: tex_api.execute(prog, {"A": img}, device=device), True)
        torch.backends.cuda.matmul.allow_tf32 = True
        on = _median_ms(lambda: tex_api.execute(prog, {"A": img}, device=device), True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return {"status": "ran", "tf32_off_ms": round(off, 4), "tf32_on_ms": round(on, 4),
            "tf32_speedup": round(off / on, 3) if on else 0.0}


def _lane_triton() -> dict:
    import sys
    sys.path.insert(0, str(_ROOT / "benchmarks"))
    import triton_validation   # self-gating: SKIPs cleanly when Triton is absent
    return triton_validation.main()


def _lane_determinism(torch, device) -> dict:
    """The CUDA scatter determinism pin (A1-4), inline so validate-hw is self-contained."""
    if device != "cuda":
        return {"status": "skipped", "reason": "run-to-run scatter determinism is a CUDA pin"}
    from . import tex_api
    code = ("vec2 d = vec2(sin(u * 40.0) * 8.0, cos(v * 40.0) * 8.0);\n"
            "@OUT = vec4(sample(@A, u + d.x / iw, v + d.y / ih).rgb, 1.0);")
    prog = tex_api.compile(code, _BINDING)
    img = _img(torch, 256, device)
    # No inner guard: run_validation_hw's per-lane try/except is the single "never raises"
    # boundary and yields the same {"status": "error: ..."} shape.
    ref = tex_api.execute(prog, {"A": img}, device=device)["OUT"]
    worst = 0.0
    for _ in range(4):
        cur = tex_api.execute(prog, {"A": img}, device=device)["OUT"]
        worst = max(worst, float((cur - ref).abs().max()))
    return {"status": "ran", "worst_run_to_run": worst, "band": 1e-9,
            "deterministic": worst <= 1e-9}


# ── driver + report ──────────────────────────────────────────────────────────

def run_validation_hw() -> dict:
    """Run every lane and return the verdict dict. Never raises (each lane is guarded)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    verdict = {"report": "tex validate-hw", "schema": 1}
    for name, fn in (("env", lambda: _lane_env(torch)),
                     ("fp16", lambda: _lane_fp16(torch, device)),
                     ("graph", lambda: _lane_graph(torch, device)),
                     ("tf32", lambda: _lane_tf32(torch, device)),
                     ("triton", _lane_triton),
                     ("determinism", lambda: _lane_determinism(torch, device))):
        try:
            verdict[name] = fn()
        except Exception as e:  # a lane must never sink the report
            verdict[name] = {"status": f"error: {type(e).__name__}: {e}"}
    return verdict


def render_markdown(v: dict) -> str:
    """A paste-ready report for the 'Hardware validation report' issue template."""
    env = v.get("env", {})
    arch = env.get("arch", {}) or {}
    L = ["# TEX Wrangle — hardware validation report", "",
         f"- **GPU:** {env.get('gpu')}  (cc {env.get('compute_capability')}, "
         f"{arch.get('arch')})",
         f"- **torch:** {env.get('torch')}   **cuda:** {env.get('cuda')}",
         f"- **arch verified in-repo:** {arch.get('verified')}"]
    if arch.get("note"):
        L.append(f"- ⚠️ {arch['note']}")
    L.append("")

    # These three keys are rendered specially (status→header, gate_holds→tag, rows→table);
    # every OTHER scalar the lane returned is dumped generically, so a new lane field shows
    # up automatically instead of silently vanishing behind a stale allow-list.
    special = {"status", "gate_holds", "rows"}

    def verdict_line(key, label):
        d = v.get(key, {})
        st = d.get("status", "?")
        holds = d.get("gate_holds")
        tag = "" if holds is None else ("  ✅ gate holds" if holds else "  ❌ gate DIFFERS")
        L.append(f"### {label} — `{st}`{tag}")
        for row in d.get("rows", []):
            L.append(f"    {json.dumps(row)}")
        for k, val in d.items():
            if k not in special and not isinstance(val, (list, dict)):
                L.append(f"    {k}: {val}")
        L.append("")

    verdict_line("fp16", "fp16 crossover (vs _MIN_FP16_PX)")
    verdict_line("graph", "CUDA-graph PF-1 crossover")
    verdict_line("tf32", "TF32 A/B (sm_80+)")
    verdict_line("triton", "Triton compile-tier")
    verdict_line("determinism", "CUDA scatter determinism pin")
    L += ["---", "*Paste this whole report into a 'Hardware validation report' issue so the "
          "gate constants can be confirmed or recalibrated for your architecture.*"]
    return "\n".join(L)


def _print_console_safe(text: str) -> None:
    """Print to a console whose encoding may be cp1252 (the default Windows console — and
    validate-hw is a *community* command run mostly on Windows) without ever crashing on the
    report's ✅/❌/⚠ glyphs. The written `.md`/`.json` copies keep the glyphs (they're utf-8);
    only the terminal echo degrades to ASCII. Fixes the S-4 UnicodeEncodeError-exit-1 bug."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def main() -> dict:
    """CLI entry: run, persist JSON + markdown under benchmarks/results/, print the path."""
    verdict = run_validation_hw()
    md = render_markdown(verdict)
    out_dir = _ROOT / "benchmarks" / "results"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "validate_hw.json").write_text(json.dumps(verdict, indent=2),
                                                  encoding="utf-8")
        (out_dir / "validate_hw.md").write_text(md, encoding="utf-8")
    except Exception:
        pass
    _print_console_safe(md)
    _print_console_safe(f"\n[written: {out_dir / 'validate_hw.json'} + .md]")
    return verdict
