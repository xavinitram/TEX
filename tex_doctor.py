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
    """Which tier `select_tier` picks for each (compile_mode, device) — SELECTION, not
    availability. `select_tier` is pure routing and consults no toolchain, so this reads
    back `"auto"` for `auto@cuda` even on a no-Triton box (the fallback to the
    interpreter happens later, inside that tier's own trial) — the routing decision is
    visible here, whether the chosen tier can actually engage is `capabilities()`,
    below. ENG-1: asks the engine directly; the doctor never needed the ComfyUI node for this."""
    from .tex_engine import select_tier as _select_tier
    out = {}
    for dev in ("cpu", "cuda"):
        for mode in ("none", "torch_compile", "auto", "cuda_graph"):
            out[f"{mode}@{dev}"] = _select_tier(mode, dev, False, False)
    return out


def _xfer_facts():
    """ENG-8: the cached host<->device transfer-cost model, if measured. Uses the
    non-probing peek() so the report never triggers a bandwidth probe."""
    from .tex_runtime import xfer
    lanes = xfer.peek()
    # b is ms_per_byte; GB/s = 1e9 bytes / (b ms * 1e-3 s/ms) / 1e9 = 1e-6 / b.
    return {"measured": bool(lanes),
            "lanes": {k: {"latency_ms": round(a, 4), "gb_per_s": (round(1e-6 / b, 2)
                          if b > 0 else None)} for k, (a, b) in lanes.items()}}


def _memory_profile_facts():
    """GOV-1: which memory/effort preset is in force, and the knobs it carries.

    Reported because S-5's rule is that a profile must be NAMEABLE, not just effective — the
    same reason `arch_support.gate_profile` is a committed table rather than a heuristic. Two
    users comparing numbers have to be able to see they were on different presets."""
    from .tex_memory import active_profile, profile_knobs, profiles
    return {"active": active_profile(), "available": list(profiles()),
            "knobs": profile_knobs()}


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
        "xfer": _fact(_xfer_facts),  # ENG-8: measured PCIe transfer-cost model
        "memory_profile": _fact(_memory_profile_facts),  # GOV-1: the named preset in force
    }


# ── BRIEF-4 — capabilities(): a per-tier capability REPORT ──────────────────────────
#
# `collect_doctor_facts()` above answers "what does this box look like"; `capabilities()`
# answers "did tier T actually engage" — a different question `_tier_facts()` cannot answer,
# because `select_tier` is pure routing (see its docstring). A row is a REPORT, never a
# contract: it names what THIS process has shown on THIS box so far, never a promise about
# any other box or any later cook. Every row is isolated by `_row()` below, the `_fact()`
# discipline above specialised to capabilities()'s fixed 4-key shape, so one raising probe
# can never take another row — or the call — down. Read-only, always: no row here ever
# calls `_setup_msvc_env` (a <=30s subprocess) or `_can_inductor_compile` (which compiles and
# mutates `TORCHINDUCTOR_CACHE_DIR`) — each reads only what a REAL cook already left behind.

_NOISE_PROMOTION_NOTE = ("engages on the default cook path on a key's 4th call, "
                         "whatever compile_mode says")


def _row(fn):
    """Run one row probe; on any failure return the closed-shape 'unknown' stub instead
    of raising — never widen this to the free-form `{"error": ...}` shape `_fact` uses,
    because every capabilities() row is pinned to exactly four keys (C1)."""
    try:
        return fn()
    except Exception as e:
        return {"status": "unknown", "evidence": "static", "why_not": None,
                "note": f"probe raised: {type(e).__name__}: {e}"}


def _cuda_available() -> bool:
    import torch
    return bool(torch.cuda.is_available())


def _cuda_unavailable() -> dict:
    return {"status": "unavailable", "evidence": "static",
            "why_not": "CUDA is not available (torch.cuda.is_available() is False)",
            "note": None}


def _inductor_prereq(dev_type: str):
    """Static, side-effect-free: does the inductor backend's PREREQUISITE hold for
    `dev_type`? Mirrors `noise._can_inductor_compile`'s own gate exactly, without ever
    calling it (that function compiles a probe kernel and mutates env). Returns
    `(ok, why_not)`: `ok` is `True` (holds), `False` (fails — `why_not` names what's
    missing), or `None` (Windows CPU, before any inductor-CPU attempt this process —
    not knowable without running the vcvarsall search this call must not perform)."""
    import importlib.util
    if dev_type == "cuda":
        if not _cuda_available():
            return False, "CUDA is not available (torch.cuda.is_available() is False)"
        if importlib.util.find_spec("triton") is None:
            return False, ("Triton is not installed (torch.compile's inductor backend "
                           "needs it on CUDA)")
        return True, None
    # cpu
    import sys
    if sys.platform != "win32":
        return True, None
    if shutil.which("cl") is not None or os.environ.get("INCLUDE"):
        return True, None
    from .tex_runtime import compiled as _compiled
    if _compiled._msvc_env_initialized:
        return False, ("no MSVC (cl.exe) found on PATH and no INCLUDE set; the vcvarsall "
                       "search already ran this process and found nothing")
    return None, None


def _row_none(dev_type: str) -> dict:
    if dev_type == "cuda" and not _cuda_available():
        return _cuda_unavailable()
    return {"status": "works", "evidence": "static", "why_not": None, "note": None}


def _row_torch_compile(backend: str, dev_type: str) -> dict:
    if dev_type == "cuda" and not _cuda_available():
        return _cuda_unavailable()
    if backend == "inductor":
        ok, why_not = _inductor_prereq(dev_type)
        if ok is False:
            return {"status": "unavailable", "evidence": "static", "why_not": why_not,
                    "note": None}
    from .tex_runtime import compiled as _compiled
    measured = _compiled._backend_status.get((backend, dev_type))
    if measured is True:
        return {"status": "works", "evidence": "measured", "why_not": None, "note": None}
    if measured is False:
        return {"status": "unavailable", "evidence": "measured",
                "why_not": f"torch.compile's '{backend}' backend failed at least once on "
                           f"{dev_type} this process", "note": None}
    return {"status": "unknown", "evidence": "static", "why_not": None, "note": None}


def _row_cuda_graph() -> dict:
    if not _cuda_available():
        return _cuda_unavailable()
    from .tex_runtime import graphed as _graphed
    if _graphed._graph_mode_disabled:
        err = _graphed._last_capture_error[0]
        why = "CUDA-graph capture was disabled after repeated capture failures"
        if err:
            why += f": {err}"
        return {"status": "unavailable", "evidence": "measured", "why_not": why, "note": None}
    if len(_graphed._graph_cache) > 0:
        return {"status": "works", "evidence": "measured", "why_not": None, "note": None}
    return {"status": "unknown", "evidence": "static", "why_not": None, "note": None}


def _key_device(key):
    """The `torch.device` embedded in a `_TieredCache` key: bare for simplex
    (`x.device`), the last element of a tuple for fbm (`(octaves, device)`) and worley
    (`(return_f2, device)`)."""
    import torch
    if isinstance(key, torch.device):
        return key
    if isinstance(key, tuple):
        for part in reversed(key):
            if isinstance(part, torch.device):
                return part
    return None


def _noise_has_promoted(dev_type: str) -> bool:
    """A live torch.compile'd callable under ANY `_TieredCache` key on `dev_type` — read
    straight off the three caches, using the same notion of 'promoted' the noise-tier
    tests use: not cold (`None`), not the eager sentinel (`False`), not a jit.trace
    `ScriptFunction`."""
    import torch
    from .tex_runtime import noise as _noise
    for cache in (_noise._simplex_cache, _noise._fbm_cache, _noise._worley_cache):
        for key, held in list(cache.cache.items()):
            if held is None or held is False or isinstance(held, torch.jit.ScriptFunction):
                continue
            dev = _key_device(key)
            if dev is not None and dev.type == dev_type:
                return True
    return False


def _noise_promotion_failure(dev_type: str):
    """The most recent recorded noise-promotion failure on `dev_type`, or `None`. Reads
    the BRIEF-4 failure ring (`tier_trace.noise_compile_failures`) plus the inductor
    probe's own cached verdict — never triggers either."""
    from .tex_runtime import noise as _noise
    from .tex_runtime import tier_trace as _tier_trace
    if _noise._inductor_available.get(dev_type) is False:
        return f"no torch.compile toolchain available for {dev_type} this process"
    for e in reversed(_tier_trace.noise_compile_failures()):
        if e.get("device") == dev_type:
            return f"{e['noise']} failed to promote ({e['error']})"
    return None


def _row_noise_promotion(dev_type: str) -> dict:
    if dev_type == "cuda" and not _cuda_available():
        d = _cuda_unavailable()
        d["note"] = _NOISE_PROMOTION_NOTE
        return d
    ok, why_not = _inductor_prereq(dev_type)
    if ok is False:
        return {"status": "unavailable", "evidence": "static", "why_not": why_not,
                "note": _NOISE_PROMOTION_NOTE}
    # Success outranks failure: a key that promoted reads as engaging even beside a
    # different key that failed on the same device (each noise type owns its own cache,
    # so — unlike the bool-per-backend rows above — both can be true at once).
    if _noise_has_promoted(dev_type):
        return {"status": "works", "evidence": "measured", "why_not": None,
                "note": _NOISE_PROMOTION_NOTE}
    failure = _noise_promotion_failure(dev_type)
    if failure is not None:
        return {"status": "unavailable", "evidence": "measured", "why_not": failure,
                "note": _NOISE_PROMOTION_NOTE}
    return {"status": "unknown", "evidence": "static", "why_not": None,
            "note": _NOISE_PROMOTION_NOTE}


def capabilities() -> dict:
    """BRIEF-4 — a read-only, per-tier capability REPORT: for each execution tier, did it
    (or the toolchain it needs) actually WORK in this process, is it known to be
    UNAVAILABLE (and why), or is that simply UNKNOWN (prerequisites hold, nothing has
    exercised it yet)? Never a fixed ladder — a box reports what IT has (see the module-
    level comment above for the read-only guarantee).

    Schema (DEVELOPMENT.md ENG-5 Tier 2 — additive-only; a row/key/vocabulary change
    bumps `schema`)::

        {"schema": 1, "rows": {"<mode>[:<backend>]@<device>": {
            "status": "works" | "unknown" | "unavailable",
            "evidence": "static" | "measured",
            "why_not": str | None,   # non-empty iff status == "unavailable"
            "note": str | None}}}
    """
    rows = {
        "none@cpu": _row(lambda: _row_none("cpu")),
        "none@cuda": _row(lambda: _row_none("cuda")),
        "torch_compile:inductor@cuda": _row(lambda: _row_torch_compile("inductor", "cuda")),
        "torch_compile:cudagraphs@cuda": _row(lambda: _row_torch_compile("cudagraphs", "cuda")),
        "torch_compile:inductor@cpu": _row(lambda: _row_torch_compile("inductor", "cpu")),
        "cuda_graph@cuda": _row(_row_cuda_graph),
        "noise_promotion@cpu": _row(lambda: _row_noise_promotion("cpu")),
        "noise_promotion@cuda": _row(lambda: _row_noise_promotion("cuda")),
    }
    return {"schema": 1, "rows": rows}


# ── SEC-2 — mac_key_path(): where the cache's signing key lives, queryable from Python ──
#
# An embedding host that rewrites directory ACLs on the roots it owns (for roaming user
# profiles) asked for one thing: a way to find out where BRIEF-10's per-user cache-signing
# key lives, so its ACL rewrite does not break or expose it. Deliberately placed HERE, next
# to `capabilities()`, rather than added AS a row inside it: that report's row names, key set
# and vocabularies are pinned (DEVELOPMENT.md ENG-5 Tier 2 — a row/key/vocabulary change bumps
# `schema`, which a vendoring host must be told about before a tag). A separate, additive
# function costs nothing and forces no re-pin, so that is the one built.


def mac_key_path() -> str | None:
    """Where BRIEF-10's per-user cache-signing key lives on disk, or would live once minted —
    NEVER the key, its bytes, or any derivative of it. Pure path arithmetic: reads only the
    environment TEX's own key-homing logic already reads (`tex_recovery._mac_key_home`, which
    does no filesystem I/O itself) and never touches disk — so asking this question can never
    create the key, its directory, or the file as a side effect.

    Returns `None` when no per-user writable home resolves on this box/OS (`_mac_key_home()`
    returns `None`): the process then falls back to a per-process EPHEMERAL key that is never
    written to disk at all, so there IS no path to report. That is the "no key" case this
    function can represent, and `None` is how it represents it.

    A non-`None` return is a *location*, not a claim that a file sits there: the key is minted
    lazily, on this process's first `sign_pickle`/`load_verified` call (see
    `tex_recovery._mac_key`/`_resolve_or_create_key`), so the path this returns may not exist
    on disk yet. Calling this function never advances that moment — it may be called before
    TEX has cooked anything at all, and it never mints, reads, or repairs the key file."""
    from .tex_recovery import _MAC_KEY_FILE, _mac_key_home
    home = _mac_key_home()
    if home is None:
        return None
    return os.path.join(home, _MAC_KEY_FILE)
