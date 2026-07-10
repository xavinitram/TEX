"""
S-5 — per-architecture safety net.

Every perf gate in this repo (the PF-1 graph-tier crossover, `precision_policy`'s
`_MIN_FP16_PX`, the autotier thresholds) was calibrated on ONE GPU: an RTX 2080 SUPER
(Turing, sm_75). That's a disclosed single-card hypothesis (doc 34 weakness #2). The
honest response — chosen over silently auto-recalibrating per box, which would make user
reports incomparable — is a *visible caveat*: on any compute capability we have NOT
measured, `tex doctor` says so and points at `tex validate-hw` (S-4). The DOM HUD surfaces
the same caveat once C1-ux (Phase 3) wires it into the tooltip.

**Behavior is unchanged on every path** — this module produces strings only. The gates
keep working exactly as before; we just stop pretending they're arch-universal. This is
the single source of the *calibration identity*: `CALIB_ARCH`/`_CALIB_GPU` name the one
GPU every gate was tuned on, so the doctor caveat, `tex validate-hw`, and (with C1-ux) the
HUD all speak from one fact instead of re-typing "sm_75 / Turing" inline.
"""

# The ONE GPU every perf gate in this repo was tuned on — the calibration identity.
CALIB_ARCH = (7, 5)                    # sm_75 Turing (RTX 2080 SUPER)
_CALIB_GPU = "RTX 2080 SUPER, Turing sm_75"
# Compute capabilities whose gate constants are backed by same-session A/B measurement.
# Grows as `tex validate-hw` reports come in (S-4) — the decision log lives in the repo.
VERIFIED_ARCHS = frozenset({CALIB_ARCH})


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
