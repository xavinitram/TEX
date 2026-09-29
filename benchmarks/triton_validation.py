#!/usr/bin/env python3
"""
HW-3 — Triton-present validation (self-gating).

The torch.compile / auto tiers only give a GPU speedup with Triton, which most Windows
installs lack. Without Triton this script SKIPs cleanly and must never fail for lack of it.
With Triton it currently emits a "not-implemented" verdict: the parity and timing checks
are not written yet, so it must not be read as a measurement.

    python benchmarks/triton_validation.py            # runs where Triton exists, else SKIPs
"""
import importlib.util
import json
import sys
from pathlib import Path

_b = Path(__file__).resolve().parent
sys.path.insert(0, str(_b.parent.parent))
sys.path.insert(0, str(_b))


def has_triton() -> bool:
    try:
        return importlib.util.find_spec("triton") is not None
    except Exception:
        return False


def run_validation() -> dict:
    """Triton-present path. No check is executed yet: the verdict says
    "not-implemented" (never "ran") so no consumer reads a stub as a green result."""
    import torch
    verdict = {"triton": True, "status": "not-implemented",
               "cuda": bool(torch.cuda.is_available())}
    if not torch.cuda.is_available():
        verdict["status"] = "no-cuda"
        return verdict
    try:
        # To implement: (1) force compile_mode=torch_compile, (2) assert codegen parity
        # vs interpreter (tol 1e-5), (3) time compile vs interpreter, (4) A/B
        # max-autotune-no-cudagraphs (adopt only on >=1.2x).
        verdict["compile_parity"] = "unmeasured"
        verdict["max_autotune_speedup"] = None
    except Exception as e:
        verdict["status"] = f"error: {type(e).__name__}: {e}"
    return verdict


def main() -> dict:
    if not has_triton():
        verdict = {"triton": False, "status": "skipped",
                   "reason": "Triton absent (expected on Windows / no-Triton boxes)"}
    else:
        verdict = run_validation()
    out = _b / "results" / "triton_validation.json"
    try:
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    except Exception:
        pass
    print(json.dumps(verdict, indent=2))
    return verdict


if __name__ == "__main__":
    main()
