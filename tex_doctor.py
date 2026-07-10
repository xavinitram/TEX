"""
DBG-4 — `tex doctor`: environment + tier-availability facts for user troubleshooting.

`collect_doctor_facts()` returns a flat dict of what's installed and which acceleration
tiers are reachable, so a user can answer "why is cuda_graph/torch_compile not kicking
in?" without reading source. EVERY probe is isolated in its own try/except: one broken
fact (a wedged CUDA driver, a missing cache dir) never takes the whole report — or the
/tex_wrangle/doctor route — down. The route contract is: always 200, always all keys.
"""
import os
import shutil


def _fact(fn):
    """Run one probe; on any failure return an {error} stub instead of raising."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — the whole point is to never propagate
        return {"error": f"{type(e).__name__}: {e}"}


def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _torch_facts():
    import torch
    cuda = bool(torch.cuda.is_available())
    return {
        "version": torch.__version__,
        "cuda_available": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
    }


def _cache_facts():
    from .tex_cache import get_cache
    c = get_cache()
    d = getattr(c, "torch_compile_cache_dir", None)
    entries, size = 0, 0
    if d and os.path.isdir(d):
        for root, _, files in os.walk(d):
            for f in files:
                entries += 1
                try:
                    size += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return {"dir": str(d) if d else None, "entries": entries,
            "size_mb": round(size / (1024 * 1024), 2)}


def _tier_facts():
    """Which tier `select_tier` picks for each (compile_mode, device) — the routing a
    user actually gets, so 'auto on a no-Triton box → interpreter' is visible."""
    from .tex_node import TEXWrangleNode as N
    out = {}
    for dev in ("cpu", "cuda"):
        for mode in ("none", "torch_compile", "auto", "cuda_graph"):
            out[f"{mode}@{dev}"] = N.select_tier(mode, dev, False, False)
    return out


def collect_doctor_facts() -> dict:
    """Flat, never-raising environment report (see module docstring)."""
    from .tex_runtime import tier_trace
    from .tex_runtime.arch_support import current_arch_status
    return {
        "torch": _fact(_torch_facts),
        "triton": _fact(lambda: {"present": _has_module("triton")}),
        "msvc": _fact(lambda: {"cl_on_path": shutil.which("cl") is not None}),
        "cache": _fact(_cache_facts),
        "tiers": _fact(_tier_facts),
        "recent_tiers": _fact(tier_trace.recent),
        "arch": _fact(current_arch_status),  # S-5: verified-arch caveat
        "noise_compiles": _fact(tier_trace.noise_compiles),  # P6: noise compile visibility
    }
