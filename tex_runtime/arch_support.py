"""
S-5 — per-architecture safety net.

Every perf gate in this repo (the PF-1 graph-tier crossover, `precision_policy`'s
`_MIN_FP16_PX`, the autotier thresholds) was calibrated on ONE GPU: an RTX 2080 SUPER
(Turing, sm_75). That's a disclosed single-card hypothesis (doc 34 weakness #2). The
honest response — chosen over silently auto-recalibrating per box, which would make user
reports incomparable — is a *visible caveat*: on any compute capability we have NOT
measured, `tex doctor` says so and points at `tex validate-hw` (S-4). The DOM HUD surfaces
the same caveat once C1-ux (Phase 3) wires it into the tooltip.

On **unverified** arches behavior is unchanged — the Turing gate constants apply and the
caveat string points at `tex validate-hw`. On a **verified** arch (one with a measured
entry in `_GATE_PROFILES`), `gate_profile()` supplies that arch's measured constants to
the gate owners (graphed's PF-1 ceilings, precision_policy's fp16 floor) at import. This
is still not silent per-box auto-tuning: profiles are repo-committed measurements with
their decision log in benchmarks/results/, so user reports stay comparable per arch.
This module remains the single source of the *calibration identity*: `CALIB_ARCH`/
`_CALIB_GPU` name the GPU the original gates were tuned on, so the doctor caveat,
`tex validate-hw`, and (with C1-ux) the HUD all speak from one fact instead of re-typing
"sm_75 / Turing" inline.
"""

# The ONE GPU the original perf gates were tuned on — the calibration identity.
CALIB_ARCH = (7, 5)                    # sm_75 Turing (RTX 2080 SUPER)
_CALIB_GPU = "RTX 2080 SUPER, Turing sm_75"

# Measured gate profiles per verified architecture. Every value is backed by a
# same-session A/B measurement on that arch (`tex validate-hw` + the PF-1 corner
# probes); the measurements are recorded inline below and in CHANGELOG v0.20.0
# (benchmarks/results/ output is machine-local, gitignored). Unverified arches
# fall back to the Turing calibration — behavior unchanged there (the S-5 caveat
# below still applies and `tex doctor` says so).
#   sm_75  — original calibration (RTX 2080 SUPER, torch 2.10).
#   sm_120 — RTX 5070 Ti Laptop (Blackwell, 12GB), torch 2.12+cu130:
#            * CUDA-graph low-op ceiling raised 512²→1024²: measured 1.66x win at
#              1024² (Turing gate declined it), 0.94x at 2048² (still correctly
#              declined above). High-op ceiling 1024² re-confirmed (1.87x at
#              1024², 0.99x at 2048²).
#            * fp16 floor raised 1024²→2048²: node-path A/B measured fp16-auto at
#              0.80x at 1024² (a real loss end-to-end; kernel-level was only
#              1.35x, eaten by cast-in/cast-out), vs a clear 2.0x at 2048².
_GATE_PROFILES = {
    (7, 5): {
        "graph_base_px_ceil": 512 * 512,
        "graph_high_px_ceil": 1024 * 1024,
        "min_fp16_px": 1024 * 1024,
    },
    (12, 0): {
        "graph_base_px_ceil": 1024 * 1024,
        "graph_high_px_ceil": 1024 * 1024,
        "min_fp16_px": 2048 * 2048,
    },
}

# Compute capabilities whose gate constants are backed by same-session A/B measurement.
# Grows as `tex validate-hw` reports come in (S-4) — the decision log lives in the repo.
VERIFIED_ARCHS = frozenset(_GATE_PROFILES)


def gate_profile() -> dict:
    """The measured gate profile for the live CUDA device, falling back to the
    Turing calibration for unverified arches (S-5: never silently auto-tune a box
    we haven't measured — profiles are repo-committed measurements, so user
    reports stay comparable). Never raises: a wedged driver or CPU-only host
    reads as unverified → Turing defaults."""
    try:
        import torch
        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    except Exception:
        cap = None
    if cap is not None:
        prof = _GATE_PROFILES.get((int(cap[0]), int(cap[1])))
        if prof is not None:
            return dict(prof)
    return dict(_GATE_PROFILES[CALIB_ARCH])


def arch_status(capability) -> dict:
    """Classify a CUDA compute capability against the verified-arch map.

    `capability` is a `(major, minor)` pair (e.g. `torch.cuda.get_device_capability()`)
    or None on a CPU-only host. Returns a flat, JSON-safe dict:
      - `arch`:     "sm_86" | None
      - `verified`: True (measured) | False (caveat applies) | None (no CUDA)
      - `note`:     the caveat string when unverified, else None
    """
    if capability is None:
        return {"arch": None, "verified": None, "note": None}
    cc = (int(capability[0]), int(capability[1]))
    arch = f"sm_{cc[0]}{cc[1]}"
    if cc in VERIFIED_ARCHS:
        return {"arch": arch, "verified": True, "note": None}
    return {"arch": arch, "verified": False,
            "note": (f"perf gates calibrated on Turing ({_CALIB_GPU}); "
                     f"run `tex validate-hw` to check {arch}")}


def current_arch_status() -> dict:
    """`arch_status` for the live CUDA device (or the CPU stub). Never raises — a wedged
    driver reads as no-CUDA, same as the doctor probes."""
    try:
        import torch
        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    except Exception:
        cap = None
    return arch_status(cap)
