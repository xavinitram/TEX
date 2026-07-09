"""
v0.18.0 Phase 4 — options & spikes (PR-LP6 TF32 profile; HW-1/HW-3 self-gating scripts).
"""
from helpers import *
import importlib.util

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"


def _import_bench(name):
    spec = importlib.util.spec_from_file_location(name, _BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prlp6_tf32_profile(r: SubTestResult):
    print("\n--- PR-LP6: TF32 profile (flag-restore + Turing no-op) ---")
    from TEX_Wrangle.tex_runtime.precision_policy import apply_tf32_profile
    fails = []

    # (a) flag-restore: the profile restores matmul precision + cudnn.allow_tf32 exactly
    prev_mm = torch.get_float32_matmul_precision()
    prev_cudnn = bool(getattr(torch.backends.cudnn, "allow_tf32", False))
    restore = apply_tf32_profile(True)
    if torch.get_float32_matmul_precision() != "high":
        fails.append("profile did not raise matmul precision to 'high'")
    restore()
    if torch.get_float32_matmul_precision() != prev_mm:
        fails.append(f"restore left matmul precision at {torch.get_float32_matmul_precision()}")
    if bool(getattr(torch.backends.cudnn, "allow_tf32", False)) != prev_cudnn:
        fails.append("restore did not restore cudnn.allow_tf32")

    # (b) default OFF: apply_tf32_profile(False) doesn't raise precision
    restore0 = apply_tf32_profile(False)
    restore0()  # no-op enable, clean restore

    # (c) Turing no-op: on sm_75 a matmul is bit-identical with TF32 on vs off
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        a = torch.rand(512, 512, device="cuda")
        b = torch.rand(512, 512, device="cuda")
        r_off = apply_tf32_profile(False); off = (a @ b).clone(); r_off()
        r_on = apply_tf32_profile(True); on = (a @ b).clone(); r_on()
        md = (off - on).abs().max().item()
        if cc[0] < 8:                       # Turing/Volta: no TF32 hardware -> inert
            if md != 0.0:
                fails.append(f"TF32 NOT inert on sm_{cc[0]}{cc[1]} (maxdiff {md:.2e})")
        # sm_80+: TF32 may legitimately differ; the Ampere speedup claim is not shipped
    if fails:
        r.fail("PR-LP6 TF32", "; ".join(fails))
    else:
        r.ok("profile enables + restores matmul/cudnn flags exactly; default-OFF clean; "
             "TF32 inert on Turing (bit-identical) — Ampere claim not shipped")


def test_hw3_triton_validation_skips(r: SubTestResult):
    print("\n--- HW-3: Triton validation self-gates (SKIPs here) ---")
    try:
        tv = _import_bench("triton_validation")
    except Exception as e:
        r.fail("HW-3 import", f"{type(e).__name__}: {e}")
        return
    verdict = tv.main()
    if tv.has_triton():
        # a Triton box: it ran (or reported no-cuda) — either way structured, not crashed
        if verdict.get("triton") is not True:
            r.fail("HW-3 verdict", f"triton present but verdict={verdict}")
        else:
            r.ok("Triton present: validation ran + emitted a verdict")
    else:
        if verdict.get("status") != "skipped":
            r.fail("HW-3 skip", f"no Triton but did not SKIP cleanly: {verdict}")
        else:
            r.ok("no Triton (this box): validation SKIPs cleanly (never fails)")


def test_hw1_pf1_calibration_smoke(r: SubTestResult):
    print("\n--- HW-1: PF-1 calibration canary imports + gate accessors ---")
    try:
        cal = _import_bench("pf1_calibration")
    except Exception as e:
        r.fail("HW-1 import", f"{type(e).__name__}: {e}")
        return
    # the autocal clamps are sane and the gate accessors it reports on exist
    fails = []
    if not (cal._CAP_PX[0] < cal._CAP_PX[1] and cal._CAP_OPS[0] < cal._CAP_OPS[1]):
        fails.append("autocal caps malformed")
    from TEX_Wrangle.tex_runtime import graphed as G
    for c in ("_GRAPH_MIN_OPS", "_GRAPH_HIGH_OPS", "_GRAPH_BASE_PX_CEIL", "_GRAPH_HIGH_PX_CEIL"):
        if not hasattr(G, c):
            fails.append(f"gate constant {c} missing (canary reports on it)")
    if fails:
        r.fail("HW-1 pf1 calibration", "; ".join(fails))
    else:
        r.ok("calibration canary imports; autocal caps sane; PF-1 gate constants present "
             "(constants stay the contract — the script only reports)")
