"""tex_results_keys.py — CACHE-1 lineage-key minting (split out of tex_results.py, NEG-6).

The value-independent half of a cooked output's identity: env_epoch (execution-environment
identity) and lineage_key (the content-addressable key a cook's result is stored/served
under). tex_results.py re-exports every name here, so every existing caller — which reads
through the tex_results module attribute, never this module directly — keeps working
unchanged. See tex_results.py's own module docstring for the two-halves picture (CACHE-1
here, CACHE-2 — ResultCache, the engine frame cache — still in tex_results.py).
"""

import hashlib
import json


# ── CACHE-1: lineage keys ─────────────────────────────────────────────────────

# env_epoch is a pure function of (active CUDA device, torch, code epoch), so memoize it per
# device — it is folded into every result key. Keyed by torch.cuda.current_device() (-1 for CPU)
# so a multi-GPU host that switches devices between cooks gets each GPU's real identity.
_ENV_EPOCH_CACHE: dict = {}


def _code_epoch() -> str:
    """The compiler/codegen code identity a cached RESULT is only reproducible under: the CACHE-4
    CODEGEN_EPOCH (which nests AST_EPOCH, so ANY parse/typecheck/optimize OR codegen change bumps
    it). A code change can re-dispatch conv/bilateral kernels (~1 ulp) and move any pixel, so a
    spilled frame from a prior codegen epoch must not be served — folding this epoch into the
    result key mints a fresh key on every such change."""
    try:
        from .tex_cache import codegen_epoch
        return codegen_epoch()
    except Exception:
        return "0"


def env_epoch() -> str:
    """The execution-environment identity a cached result is only valid within: torch
    version + GPU identity (device name + compute capability) + the code epoch. Folding all
    three into every result key means a frame minted under one environment is never served
    under another — the silent cross-environment hit a result cache must not have. Mirrors
    and extends xfer._version_tag (device name + torch); adds compute capability + code epoch.
    Memoized PER active CUDA device (torch/GPU identity is fixed per device, but a heterogeneous
    multi-GPU host switches current_device between cooks — a single process-wide memo would freeze
    the epoch to whichever GPU was active at the first call and stamp a cuda:1 frame with cuda:0's
    identity)."""
    parts = []
    dev = -1
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
    except Exception:
        pass
    cached = _ENV_EPOCH_CACHE.get(dev)
    if cached is not None:
        return cached
    try:
        import torch
        parts.append(torch.__version__.split("+")[0])
        if dev >= 0:
            parts.append(torch.cuda.get_device_name(dev))
            cc = torch.cuda.get_device_capability(dev)
            parts.append(f"sm{cc[0]}{cc[1]}")
    except Exception:
        pass
    parts.append(_code_epoch())
    tag = "|".join(parts)
    _ENV_EPOCH_CACHE[dev] = tag
    return tag


def _canon_params(params) -> str:
    """Deterministic, collision-free encoding of a program's scalar/vector param values.
    Tensor values must NOT be here — a tensor input enters a lineage key by its upstream
    key, never its pixels. `default=repr` keeps a stray unexpected type from raising (it
    just keys conservatively); sort_keys makes name order irrelevant."""
    return json.dumps(params or {}, sort_keys=True, default=repr)


def _canon_viewer(vc) -> str:
    """PM-11: deterministic encoding of the host's viewer values, same shape as
    `_canon_time` — sorted, `repr(float(...))` so a sub-ULP difference in an exposure
    slider mints a distinct key rather than colliding onto a stale rendered frame."""
    if not vc:
        return "n"
    return json.dumps({k: repr(float(v)) for k, v in vc.items()}, sort_keys=True)


def _canon_time(tc) -> str:
    """Deterministic encoding of the ENG-7 host playhead. ALL playhead builtins move output
    pixels while being kept out of the program fingerprint (interpreter `_TIME_BUILTIN_NAMES` =
    frame/fps/time), so a result key must carry every one of them, by EXACT value — folding the
    whole normalized dict (not just `frame`) future-proofs a fourth builtin, and `repr(float)`
    keeps fractional/sub-frame playheads (motion blur, retime) distinct where `int(frame)` would
    collide them onto a stale frame."""
    if not tc:
        return "n"
    return json.dumps({k: repr(float(v)) for k, v in tc.items()}, sort_keys=True)


def lineage_key(*, program_fp, device, precision, params=None, upstream=(),
                frame=None, time_context=None, quality=None, flags=(), canvas=None,
                viewer_context=None) -> str:
    """CACHE-1: the content-addressable identity of a cooked RESULT (a hex SHA-256).

    Composes H(program_fp × params × upstream × frame × device × precision/quality ×
    env_epoch × flags × canvas). Structured, length-prefixed encoding (mirrors
    TEXCache.fingerprint) so no component can bleed into an adjacent one.

    program_fp   the value-independent program fingerprint (fp or fused_fp).
    device       MANDATORY. str(device); a cook on another device is a different result.
    precision    MANDATORY. the EFFECTIVE precision the cook ran at.
    params       the non-tensor binding values (widget $params); enter by value.
    upstream     the lineage keys of this cook's tensor inputs (empty under ComfyUI, where
                 there is no TEX-internal upstream edge yet — a GRAPH-1 host threads them).
    frame        a single host playhead frame, or None (a still). Keyed by exact value.
    time_context the FULL ENG-7 playhead dict {frame,fps,time,...}, or None — every builtin in
                 it moves pixels, so every one must key (a `time`- or `fps`-only animation is a
                 distinct result even at the same frame). The engine passes this; a caller with
                 only a frame number may pass `frame=` instead.
    quality      a preview/final quality tag (PREC-1), or None.
    flags        any extra keying flags (e.g. an output name for a per-output key).
    canvas       a canvas / ROI descriptor (W,H[,x0,y0,w,h]); two cooks at different canvas
                 sizes or ROIs are distinct results (keys carry it from day one).
    viewer_context  PM-11: the host's viewer values, or None. UNLIKE every component above,
                 this one is OMITTED from the hash entirely when None — a program that never
                 calls `viewer_exposure()`/`viewer_gamma()` must key IDENTICALLY to a build
                 that predates PM-11 (invariant #7: this ask cannot invalidate every frame any
                 other program ever cached). The caller decides: pass the real dict only when
                 `interpreter._reads_viewer_builtin(program)` is True, `None` otherwise — never
                 pass it unconditionally the way the engine passes `time_context` (every
                 program can read `frame`/`fps`/`time` as bare identifiers with no call, so
                 there was never a "before" key shape to preserve for that one).
    """
    if program_fp is None:
        raise ValueError("lineage_key needs a program fingerprint (fp or fused_fp)")
    if device is None or precision is None:
        raise ValueError("lineage_key: device and precision are MANDATORY key components "
                         "(invariant #9 — a cross-device/precision hit is never served)")
    h = hashlib.sha256()

    def feed(tag: str, s: str) -> None:
        b = f"{tag}={s}".encode()
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)

    feed("fp", str(program_fp))
    feed("dev", str(device))
    feed("prec", str(precision))
    feed("env", env_epoch())
    feed("par", _canon_params(params))
    feed("up", json.dumps([str(u) for u in upstream]))
    feed("frm", "n" if frame is None else repr(float(frame)))   # exact value, no int() collide
    feed("tc", _canon_time(time_context))                        # every playhead builtin keys
    feed("q", "n" if quality is None else str(quality))
    feed("flg", json.dumps(sorted(str(f) for f in flags)))
    # canvas is any JSON-able shape/ROI descriptor (a dict {"shape":[B,H,W,C],"roi":[...]}, or a
    # legacy (W,H) tuple) — the engine keys each output by its produced-frame shape, so a
    # different batch/canvas/ROI mints a distinct key.
    feed("cnv", "n" if canvas is None else json.dumps(canvas, sort_keys=True, default=list))
    # PM-11: conditional, unlike every feed above it — see the docstring. Omitting the call
    # entirely (not merely feeding "n") is load-bearing: inserting ANY new `feed` unconditionally
    # would shift the byte stream for every existing key, viewer-using or not.
    if viewer_context is not None:
        feed("view", _canon_viewer(viewer_context))
    return h.hexdigest()
