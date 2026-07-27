"""
host_demo.py — a second host for TEX (PORT-5 / PM-2, v0.28.0).

This is the proof that TEX's cook engine runs with **no ComfyUI**: a standalone image viewer
that builds a 3-stage pipeline (grade -> blur -> vignette) as a GraphSpec, compiles it fused
through `tex_engine`, and scrubs one slider (vignette strength) live. No node, no JS extension,
no comfy import — just torch + the TEX package + the Python standard library.

What it demonstrates (the roadmap's PM-2 acceptance):
  * ENG-1 / SCHED-1 — the engine cooks a fused GraphSpec directly (`tex_engine.cook(chain_payload=)`).
  * ENG-3       — the 'engine' egress profile hands back raw fp32 (scene values survive).
  * CACHE-2     — a `ResultCache` armed BY THIS HOST (not the node): scrub back to a strength you
                  already visited and the frame is a cache hit, no recook.
  * SCHED-3     — every cook takes a `CancelToken`, so a newer slider drag aborts the stale cook.
  * DATA-4      — the whole thing is held through one `EngineSession`.
  * PM-2        — engine-side cook < 50 ms/frame warm at 1024^2 on the sm_120 box (measured at
                  startup; display transport is excluded, per the acceptance).

Run it:  python examples/host_demo.py            (then open http://127.0.0.1:8760)
         python examples/host_demo.py --bench     (just the PM-2 benchmark, no server)

An http.server viewer is used because the Windows embedded CPython ships no tkinter; the browser
is the display surface, and its transport (a raw-RGBA blit to a <canvas>) is deliberately kept
out of the cook budget.
"""
import argparse
import hashlib
import os
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# --- import the TEX package standalone (the tests/CLI path: add custom_nodes, import the pkg) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # .../custom_nodes

import torch
from TEX_Wrangle import (tex_engine, tex_fusion, tex_results, tex_marshalling, tex_api, tex_roi,
                         tex_cookqueue)
from TEX_Wrangle.tex_runtime import profile as tex_profile
from TEX_Wrangle.tex_runtime.host import NullHostServices, CookCancelled


class _Cancel:
    """A concrete SCHED-3 CancelToken (the exported `CancelToken` is a Protocol, not a class).
    Trips once a NEWER frame request supersedes this one — so a fast slider drag abandons the
    stale cook at its next yield point instead of computing frames nobody will see."""
    __slots__ = ("alive",)

    def __init__(self):
        self.alive = True

    def check(self):
        if not self.alive:
            raise CookCancelled("superseded by a newer frame")


class _Chain:
    """Two CancelTokens as one: the SCHED-4 queue's per-attempt token AND this host's own
    supersede latch. A cook takes exactly one `cancel=`, and the queue's token is not optional
    — it is the only channel preemption, shedding and `close()` travel down — so a host with a
    second reason to abort chains rather than substitutes."""
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b

    def check(self):
        self.a.check()
        self.b.check()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BENCH_RES = 1024          # the PM-2 measurement resolution
DISPLAY_RES = 512         # the interactive viewer (kept small so the RGBA blit stays snappy)

# The 3-stage pipeline. Upstream stages are source-first and EXCLUDE the terminal; each names the
# binding (@IN) that carries the chain. The terminal (vignette) owns the promoted slider $strength.
_GRADE = "@OUT = vec4(@IN.rgb * 1.15 + vec3(0.02), 1.0);"
_BLUR = "@OUT = gauss_blur(@IN, 2.0);"
_VIGNETTE = ("float d = distance(vec2(u, v), vec2(0.5, 0.5));"
             "float vig = 1.0 - $strength * smoothstep(0.15, 0.75, d);"
             "@OUT = vec4(@IN.rgb * vig, 1.0);")
_SPEC = {"schema": 1,
         "stages": [{"code": _GRADE, "image_input": "IN", "params": {}},
                    {"code": _BLUR, "image_input": "IN", "params": {}}],
         "terminal_image_input": "IN"}


# ── v0.30 rung-2: the ROI viewport comp ──────────────────────────────────────────────────
# PM-6 asks for "a 10-node comp at proxy resolution with ROI cooks + cache hits at interactive
# rate". These ten stages are a real grade/filter chain, not padding — each is a node a
# compositor would actually ship, and each is ROI-EXECUTABLE (pointwise, or a direct-tensor
# halo op). They are cooked UNFUSED, one engine cook per stage, for two reasons the roadmap
# makes structural: `roi=` is refused on a fused chain (tex_engine's gate), and fusion splices
# stages behind local variables, which blocks the reach analysis outright. Unfused per-stage
# cooking is also what gives SCHED-2 a per-region graph and CACHE-2 a per-stage cache point.
_WHOLE_FRAME = 1 << 30   # "unbounded reach" — clamps to the whole frame in _needed_windows


def _code_fp(code: str) -> str:
    """A process-STABLE fingerprint for a stage's source."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]

_COMP_STAGES = [
    ("exposure",   "@OUT = vec4(@IN.rgb * $exposure, 1.0);",                      {"exposure": 1.05}),
    ("blackpoint", "@OUT = vec4(max(@IN.rgb - vec3($black), vec3(0.0)), 1.0);",    {"black": 0.02}),
    ("gamma",      "@OUT = vec4(spow(@IN.rgb, vec3($gamma)), 1.0);",               {"gamma": 0.95}),
    ("saturation", "float y = luma(@IN);\n"
                   "@OUT = vec4(mix(vec3(y), @IN.rgb, $sat), 1.0);",               {"sat": 1.10}),
    ("blur",       "@OUT = gauss_blur(@IN, $sigma);",                              {"sigma": 1.5}),
    ("sharpen",    "@OUT = vec4(clamp(@IN.rgb + ($amount) * (@IN.rgb - gauss_blur(@IN, 2.0).rgb), "
                   "vec3(0.0), vec3(1.0)), 1.0);",                                 {"amount": 0.6}),
    ("tint",       "@OUT = vec4(@IN.rgb * vec3(1.0 + $tint, 1.0, 1.0 - $tint), 1.0);", {"tint": 0.03}),
    ("contrast",   "@OUT = vec4((@IN.rgb - vec3(0.5)) * $contrast + vec3(0.5), 1.0);", {"contrast": 1.08}),
    ("glow",       "@OUT = vec4(@IN.rgb + $glow * gauss_blur(@IN, 4.0).rgb, 1.0);", {"glow": 0.12}),
    ("vignette",   "float d = distance(vec2(u, v), vec2(0.5, 0.5));\n"
                   "@OUT = vec4(@IN.rgb * (1.0 - $strength * smoothstep(0.15, 0.75, d)), 1.0);",
                                                                                   {"strength": 0.5}),
]


class RoiComp:
    """The rung-2 viewport: a 10-stage comp where each stage keeps a persistent FULL-FRAME
    canvas and an edit re-cooks only the viewport window through the dirty suffix.

    Why canvases: a stage's ROI cook needs its input over `window ⊕ halo`, which the upstream
    canvas already holds from the last full cook — so a window cook never has to widen back up
    the chain. `run_roi` narrows the upstream canvas internally (we hand it the whole tensor and
    the window), and the returned patch is pasted into this stage's canvas. Every stage result
    is also keyed into a CACHE-2 `ResultCache` by a CACHE-1 lineage key that includes the window,
    so revisiting a value the user already scrubbed to is a cache hit, not a cook."""

    def __init__(self, res: int, device: str = DEVICE):
        self.res, self.device = res, device
        self.source = Host._make_source(res).to(device)
        self.params = {n: dict(p) for n, _c, p in _COMP_STAGES}
        self.cache = tex_results.ResultCache()          # CACHE-2, armed by THIS host
        self.canvas: list = [None] * len(_COMP_STAGES)  # per-stage full-frame result
        # Whether canvas[i] is a buffer WE own and may write in place. A cook output arrives
        # frozen under ENG-12, so the first window write has to clone; after that it is ours.
        self._owned: list = [False] * len(_COMP_STAGES)
        self._halo_memo: dict = {}                      # (stage, params) -> reach; see _halo_of

    def _key(self, i: int, roi, up_key):
        """CACHE-1 lineage key for stage i, INCLUDING its upstream chain.

        The upstream link is load-bearing, not decoration: without it a stage's identity is
        blind to every edit above it, so changing `exposure` and re-cooking serves the stale
        downstream frame from cache and reports a perfect match against the pre-edit image.
        CACHE-1 exists to key a tensor input by its producer's key rather than its pixels —
        this is that contract. The window is in the key too, so a window cook and a
        whole-frame cook of the same params can never be served for each other.

        `up_key` is stage i-1's whole-frame key, and it is REQUIRED (None only for i == 0).
        Deriving it here by recursion instead makes the chain O(n^2) — 55 SHA-256 keys per
        all-dirty frame at 10 stages, 210 at 20 — which `cook()` measured at 1.20 ms/frame,
        more than the terminal scrub it is supposed to be timing. Keeping a recursive fallback
        would leave that quadratic path alive as a silent possibility; a caller that forgets
        now gets a TypeError instead."""
        up = [] if i == 0 else [up_key]
        return tex_results.lineage_key(
            # sha256, not hash(): str hashing is PYTHONHASHSEED-salted, so a `hash()` key is
            # not stable across processes — a trap this repo has hit before.
            program_fp=f"{_COMP_STAGES[i][0]}:{_code_fp(_COMP_STAGES[i][1])}",
            device=self.device, precision="fp32",
            params={**{k: round(float(v), 4) for k, v in self.params[_COMP_STAGES[i][0]].items()},
                    "_stage": i, "_roi": tuple(roi) if roi else None},
            upstream=up,
            canvas={"shape": [1, self.res, self.res, 4]})

    def _cache_hit(self, key, win):
        """A CACHE-2 entry for `key` as `(patch, served_window)`, or None meaning "cook it".

        The window an entry covers is NOT necessarily `win`. A stage can DECLINE the window (a
        gather, an unbounded reach, a refused rect) and hand back a WHOLE FRAME, which is then
        stored under a key that names the window — so assuming `win` on a hit pastes a full
        frame into a w×h slice and raises ("expanded size (120) must match 192").

        It has to be re-derived here rather than stored beside the tensor: CACHE-2 holds
        TENSORS, and `ResultCache.put` silently ignores anything else, so caching a
        `(patch, served)` pair does not fail — it just never caches at all (measured: every
        revisit re-cooked, and the hit-rate probe went to zero). Shape answers it exactly,
        because the engine serves one of those two extents and nothing else. An entry matching
        neither is not what this stage asked for, so distrust it and cook."""
        patch = self.cache.get(key)
        if patch is None:
            return None
        ph, pw = int(patch.shape[1]), int(patch.shape[2])
        if (pw, ph) == (self.res, self.res):
            return patch, None                             # a whole-frame result (or a decline)
        if win is not None and (pw, ph) == (win[2], win[3]):
            return patch, win
        return None

    def _halo_of(self, i: int) -> int:
        """The neighbour reach stage `i` reads, from the same ROI-1 descriptors the engine uses.

        A NON-EXECUTABLE stage reports `halo=0`, which is the opposite of what it means: it has
        unbounded reach (a gather), so a consumer must widen to the whole frame, not to nothing.
        Answering 0 there would under-grow the upstream window and leave exactly the stale ring
        `_needed_windows` exists to prevent — the whitelist posture ("unknown → whole image")
        inverted. Memoized per (stage, params) because a scrub changes params every frame and
        `roi_plan`'s own memo keys on them, so an un-memoized call re-parses inside the loop
        (measured 0.431 ms/frame across a 10-stage comp)."""
        key = (i, tuple(sorted(self.params[_COMP_STAGES[i][0]].items())))
        hit = self._halo_memo.get(key)
        if hit is None:
            plan = tex_roi.roi_plan(_COMP_STAGES[i][1], self.params[_COMP_STAGES[i][0]])
            hit = self._halo_memo[key] = (int(plan.halo) if plan.executable else _WHOLE_FRAME)
        return hit

    def _needed_windows(self, roi, dirty_from: int) -> list:
        """The window each dirty stage must cook so the FINAL window is correct.

        Patching only the requested rect is wrong the moment a downstream stage has a halo: it
        reads a ring of neighbours just outside the patch, and that ring still holds pre-edit
        pixels. So walk the suffix BACKWARDS, growing the window by each consumer's halo and
        clamping to the frame — the same `ROI ⊕ halo` composition `run_roi` does within one
        stage, lifted to the chain. Measured before this: stage-5 sharpen was wrong over
        2157 px and stage-9 vignette over 3987 px on any upstream edit."""
        last = len(_COMP_STAGES) - 1
        need = [None] * len(_COMP_STAGES)
        need[last] = roi
        for i in range(last - 1, dirty_from - 1, -1):
            x0, y0, w, h, W, H = need[i + 1]
            pad = self._halo_of(i + 1)
            nx0, ny0 = max(0, x0 - pad), max(0, y0 - pad)
            nx1, ny1 = min(W, x0 + w + pad), min(H, y0 + h + pad)
            need[i] = (nx0, ny0, nx1 - nx0, ny1 - ny0, W, H)
        return need

    def cook(self, roi=None, dirty_from: int = 0, cancel=None, use_cache: bool = True):
        """Cook stages `dirty_from..end`, over `roi` when given. Returns (frame, ms, hits).

        `use_cache=False` forces every dirty stage to actually cook — the honest setting for a
        benchmark row that claims to measure an all-dirty recook."""
        t0 = time.perf_counter()
        hits = 0
        # A window cook writes INTO an existing canvas, so every stage must already hold one.
        # Priming here (rather than seeding mid-loop) keeps the invariant in one place: a
        # mid-loop seed could not work anyway, since stage i's input IS canvas[i-1].
        if roi is not None and any(c is None for c in self.canvas):
            self.cook(None, 0, cancel=cancel, use_cache=use_cache)
        need = self._needed_windows(roi, dirty_from) if roi is not None else None
        up_key = None                                      # stage i-1's whole-frame key
        for i, (name, code, _defaults) in enumerate(_COMP_STAGES):
            win = need[i] if need is not None else None
            key = None
            if use_cache:
                # The whole-frame key is what stage i+1 links to, so it is carried forward even
                # for a clean-prefix stage that never cooks — hence before any `continue`. With
                # `use_cache=False` nothing reads a key, and hashing one anyway put 1.20 ms of
                # SHA-256 per frame inside the all-dirty rows that claim to time cooking.
                prev_key, up_key = up_key, self._key(i, None, up_key)
                key = up_key if win is None else self._key(i, win, prev_key)
            if i < dirty_from and self.canvas[i] is not None:
                hits += 1
                continue                                   # clean prefix: the canvas stands
            hit = self._cache_hit(key, win) if use_cache else None
            src = self.source if i == 0 else self.canvas[i - 1]
            if hit is not None:
                patch, served = hit
                hits += 1
            else:
                res = tex_engine.cook(code, {"IN": src, **self.params[name]},
                                      device_mode=self.device, precision="fp32",
                                      roi=win, roi_exec=win is not None, cancel=cancel)
                patch = res.outputs["OUT"]
                # A stage can DECLINE the window (a gather, an unbounded reach, a refused
                # rect) and hand back a whole frame. `cooked_roi` is how the engine says so —
                # ignoring it and pasting a full-frame tensor into a window is a crash, which
                # is exactly what this demo did before.
                served = res.cooked_roi
                if use_cache:
                    self.cache.put(key, patch, canvas={"shape": list(patch.shape)})
            if served is None:
                # A whole-frame result REPLACES the canvas, and it arrives frozen (ENG-12) —
                # a cook output, or a cache entry other stages may still be holding. Take it
                # as-is and remember we do not own it; copying here would charge every
                # whole-frame cook for a write that may never come (measured: 10 full-frame
                # clones per frame, +2.8 ms at 1920^2, on a row that never opens a window).
                self.canvas[i], self._owned[i] = patch, False
            else:
                if not self._owned[i]:
                    # First in-place write to this canvas: buy our own buffer now. Once, not
                    # per frame — the next window cook writes straight into it.
                    self.canvas[i], self._owned[i] = self.canvas[i].clone(), True
                x0, y0, w, h, _W, _H = served
                self.canvas[i][:, y0:y0 + h, x0:x0 + w] = patch   # our buffer: write in place
        if self.device == "cuda":
            torch.cuda.synchronize()
        return self.canvas[-1], (time.perf_counter() - t0) * 1000.0, hits


def bench_pm6(res: int = 1024, roi_side: int = 512, frames: int = 12) -> dict:
    """PM-6: scrub a 10-node comp at proxy resolution with ROI cooks + cache hits.

    Three rows, which together are the claim: an all-dirty WHOLE-FRAME recook (what a host
    without ROI pays), an all-dirty ROI recook (the window win), and the interactive case — the
    user drags the LAST node's slider, so the nine upstream canvases stand and only the terminal
    stage cooks its window. HONEST SCOPE: PM-6 names the sm_120 box; this is measured on
    whatever card is present and must be reported as such."""
    comp = RoiComp(res)
    comp.cook(None, 0)                                     # prime every canvas (cold)
    span = max(1, res - roi_side)

    def _med(fn):
        fn(-1)          # warm on a value the measured range never repeats
        return round(statistics.median([_timed(fn, i) for i in range(frames)]), 3)

    def _timed(fn, i):
        t = time.perf_counter(); fn(i); return (time.perf_counter() - t) * 1000.0

    # The two "all-dirty" rows pass use_cache=False so they genuinely cook all ten stages.
    # With the cache on they were one engine cook plus nine CACHE-2 hits — about a tenth of
    # the work their label claimed, which made the speedups a comparison between two
    # equally-hollow rows.
    def _whole_all(i):
        comp.params["vignette"]["strength"] = 0.30 + i * 0.01
        comp.cook(None, 0, use_cache=False)

    def _roi_all(i):
        comp.params["vignette"]["strength"] = 0.50 + i * 0.01
        comp.cook((span // 2, span // 2, roi_side, roi_side, res, res), 0, use_cache=False)

    def _roi_terminal(i):
        # The interactive case: the user drags the LAST node's slider, so the nine upstream
        # canvases stand and only the terminal stage cooks its window. The cache stays ON —
        # serving a revisited value is part of what makes a scrub cheap.
        comp.params["vignette"]["strength"] = 0.70 + i * 0.01
        comp.cook((span // 2, span // 2, roi_side, roi_side, res, res), len(_COMP_STAGES) - 1)

    out = {"stages": len(_COMP_STAGES), "res": res, "roi": roi_side,
           "device": DEVICE,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "capability": ("sm_" + "".join(map(str, torch.cuda.get_device_capability(0))))
                         if torch.cuda.is_available() else None,
           "whole_all_ms": _med(_whole_all),
           "roi_all_ms": _med(_roi_all),
           "roi_terminal_ms": _med(_roi_terminal)}
    out["roi_speedup"] = round(out["whole_all_ms"] / max(out["roi_all_ms"], 1e-9), 2)
    out["terminal_speedup"] = round(out["whole_all_ms"] / max(out["roi_terminal_ms"], 1e-9), 2)
    return out


class Host:
    """The standalone host: owns the session, the source image, and the frame cache, and cooks the
    fused chain for a given slider value — serving a cached frame when the value repeats (CACHE-2)."""

    def __init__(self, res: int):
        self.res = res
        self.session = tex_api.default_session()
        self.session.set_host(NullHostServices())          # DATA-4 / ENG-2: no comfy, but VRAM-aware
        tex_marshalling.set_egress_profile("engine")       # ENG-3: raw fp32 out, values preserved
        self.source = self._make_source(res)
        # The fused fingerprint keys the frame cache. Value-independent, so every strength shares it.
        self.fp = tex_fusion.fused_fingerprint(
            _SPEC, _VIGNETTE, {"IN": self.source, "strength": 0.5}, tex_marshalling.infer_binding_type)
        self.cache = tex_results.ResultCache()             # CACHE-2, armed by THIS host
        # ENG-9: one cook at a time (per-thread interpreters, but the MUT tier state is single-cook-
        # thread). `_lock` serializes the cooks; `_req_lock` (a SEPARATE short-held lock) guards the
        # supersede bookkeeping so new_request doesn't block on an in-flight cook AND so two
        # concurrent /frame requests can't race the check-then-set on `_current` (a lost supersede).
        self._lock = threading.Lock()
        self._req_lock = threading.Lock()
        self._current = None

    def new_request(self) -> "_Cancel":
        """Mint the cancel token for a fresh frame request, superseding any in-flight cook. The
        supersede (read `_current`, cancel it, store the new one) is atomic under `_req_lock`, so a
        newer request never fails to cancel an older one. Honest limit: 'newer' is call order, which
        under a threading server can differ from slider-arrival order for near-simultaneous requests
        — a benign display nit given ~1 ms cooks, not a lost/stale-forever frame."""
        tok = _Cancel()
        with self._req_lock:
            if self._current is not None:
                self._current.alive = False
            self._current = tok
        return tok

    @staticmethod
    def _make_source(res: int) -> torch.Tensor:
        """A procedural test image (radial gradient + a grid), so the demo needs no image file."""
        y, x = torch.meshgrid(torch.linspace(0, 1, res), torch.linspace(0, 1, res), indexing="ij")
        r = ((x - 0.5) ** 2 + (y - 0.5) ** 2).sqrt()
        grid = ((x * 16).sin().abs() * (y * 16).sin().abs())
        rgb = torch.stack([0.6 - r + 0.3 * grid, 0.5 * x + 0.2 * grid, 0.5 * y + 0.2 * grid], dim=-1)
        return rgb.clamp(0, 1).unsqueeze(0).to(DEVICE)     # [1, H, W, 3]

    def _key(self, strength: float) -> str:
        # CACHE-1 lineage key at a FIXED resolution, so revisiting a strength recomputes the same key.
        return tex_results.lineage_key(
            program_fp=self.fp, device=DEVICE, precision="fp32",
            params={"strength": round(strength, 4)}, canvas={"shape": [1, self.res, self.res, 4]})

    def cook(self, strength: float, cancel=None):
        """Return (frame [1,H,W,4] fp32, cook_ms, was_cache_hit) for this slider value. Serialized
        (one cook at a time); a cancel token that has been superseded aborts with CookCancelled."""
        strength = round(strength, 4)              # cook the SAME value the cache key rounds to,
        with self._lock:                           # so a cached frame always matches its key
            key = self._key(strength)
            hit = self.cache.get(key)
            if hit is not None:
                return hit, 0.0, True
            t0 = time.perf_counter()
            res = tex_engine.cook(_VIGNETTE, {"IN": self.source, "strength": strength},
                                  chain_payload=_SPEC, device_mode=DEVICE, precision="fp32", cancel=cancel)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0   # the ENGINE-side cook (PM-2), before the put
            out = res.outputs["OUT"]
            self.cache.put(key, out, canvas={"shape": list(out.shape)})
            return out, ms, False

    def rgba_bytes(self, frame: torch.Tensor) -> bytes:
        """[1,H,W,4] fp32 -> packed RGBA8 bytes for a <canvas> blit (display, off the cook budget)."""
        u8 = (frame[0].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()
        return bytes(u8.reshape(-1).tolist())


class QueuedHost:
    """v0.31 — the SCHED-4 / PRED-1 consumer proof (doc 39 §0: every engine feature needs a
    second consumer, or app logic smuggles itself into the engine).

    A viewer's slider is exactly the workload the cook queue exists for. Every drag submits an
    INTERACTIVE frame; every settled value ALSO submits its two neighbours as SPECULATIVE
    "play-hover" work, on the bet that the user is still dragging. When the next drag arrives
    it preempts whichever prefetch is mid-cook, and the abandoned prefetch goes back on the
    queue rather than being thrown away.

    The host supplies the psychology and TEX supplies the economics, which is the PRED-1 split:
    this file decides that a neighbouring slider value is worth `_NEIGHBOUR_CONFIDENCE`, and
    `SpeculativePolicy` decides whether that bet clears the floor once PROF-1 has priced the
    program. Nothing here reaches inside the engine — it is `submit()`, `cancel()`, `result()`."""

    _NEIGHBOUR_CONFIDENCE = 0.45      # "still dragging" — a guess this host owns, not TEX
    _NEIGHBOUR_STEPS = (-0.05, 0.05)

    def __init__(self, host: "Host"):
        self.host = host
        self.queue = tex_cookqueue.CookQueue(name="tex-demo-cook")
        # A low floor and a shallow backlog: a viewer wants a couple of frames of lookahead,
        # not a queue that outlives the drag it was predicting.
        self.queue.install_policy(tex_cookqueue.SpeculativePolicy(
            min_value_ms=0.2, max_pending=4, unknown_min_confidence=0.4))
        self.pkey = tex_profile.make_key(host.fp, DEVICE, "fp32")
        self.px = host.res * host.res

    def _submit(self, strength: float, klass: int, reason: str, confidence: float):
        # The cook MUST receive the queue's own token — that is the only thing `_JobToken.check`
        # is ever called through, so a job that swaps it for another can never be preempted,
        # shed, or stopped by `close()`. An interactive frame ALSO wants the host's supersede
        # latch (a newer slider drag beats an older cook of the same class, which the queue
        # will not do for it: same-class work is FIFO), so those two are chained rather than
        # one replacing the other.
        if klass == tex_cookqueue.INTERACTIVE:
            supersede = self.host.new_request()
            fn = (lambda cancel: self.host.cook(strength, cancel=_Chain(cancel, supersede)))
        else:
            fn = (lambda cancel: self.host.cook(strength, cancel=cancel))
        return self.queue.submit(fn, klass=klass, reason=reason, confidence=confidence,
                                 profile_key=self.pkey, px=self.px)

    def frame(self, strength: float, *, timeout: float = 30.0):
        """The viewer's frame request. Returns (frame, cook_ms, was_cache_hit)."""
        out = self._submit(strength, tex_cookqueue.INTERACTIVE, "slider", 1.0).result(timeout)
        # Prefetch the neighbours AFTER the frame is in hand, so the speculation can never
        # be what the interactive request is waiting behind.
        for d in self._NEIGHBOUR_STEPS:
            nxt = round(min(1.0, max(0.0, strength + d)), 4)
            if nxt != strength:
                self._submit(nxt, tex_cookqueue.SPECULATIVE, tex_cookqueue.PLAY_HOVER,
                             self._NEIGHBOUR_CONFIDENCE)
        return out

    def stats(self) -> dict:
        return self.queue.snapshot()

    def close(self) -> None:
        self.queue.close()


def bench_pm7(res: int = DISPLAY_RES, frames: int = 12) -> dict:
    """PM-7 from the DEMO's side: does a slider drag actually preempt the prefetches it left
    behind, and does the abandoned work survive?

    `benchmarks/cookqueue_bench.py` measures the queue in isolation; this measures it inside a
    real host, which is the part doc 39 §4 asks for. The scrub walks a slider the way a user
    does — small steps, never revisiting — so every frame lands on speculation that is either
    already cooked (a win) or still cooking (a preemption)."""
    host = Host(res)
    qh = QueuedHost(host)
    try:
        qh.frame(0.5)                                  # prime: compile, warm the tiers
        qh.queue.drain(timeout=60)

        # A COMMITTED render started while the user keeps scrubbing — the workload the
        # starvation brake exists for, and the one the demo has to be able to finish. Long
        # enough (many yield points) that the slider genuinely lands on it mid-flight.
        heavy = ("vec4 b0 = gauss_blur(@IN, 5.0);\n"
                 + "\n".join(f"vec4 b{i} = gauss_blur(b{i-1}, 5.0);" for i in range(1, 30))
                 + "\n@OUT = b29;")
        render = qh.queue.submit(
            lambda cancel: tex_engine.cook(heavy, {"IN": host.source}, device_mode=DEVICE,
                                           precision="fp32", cancel=cancel),
            klass=tex_cookqueue.COMMITTED, reason="user render")

        lat, hits = [], 0
        for i in range(frames):
            s = round(0.10 + i * 0.05, 4)
            t0 = time.perf_counter()
            _frame, _ms, hit = qh.frame(s)
            lat.append((time.perf_counter() - t0) * 1000.0)
            hits += 1 if hit else 0
        render_done = render.wait(120.0)
        qh.queue.drain(timeout=120)
        st = qh.stats()["stats"]
        return {"res": res, "device": DEVICE, "frames": frames,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "capability": ("sm_" + "".join(map(str, torch.cuda.get_device_capability(0))))
                              if torch.cuda.is_available() else None,
                "interactive_median_ms": round(statistics.median(lat), 3),
                "prefetch_hits": hits,
                "committed_render_completed": bool(render_done),
                "committed_render_attempts": render.attempts,
                "submitted": st["submitted"], "completed": st["completed"],
                "preempted": st["preempted"], "requeued": st["requeued"],
                "preempt_denied": st["preempt_denied"],
                "refused": st["refused"], "shed": st["shed"]}
    finally:
        qh.close()


def run_benchmark() -> bool:
    """PM-2: cook the fused chain warm at 1024^2 and report the per-frame median. Returns pass/fail."""
    host = Host(BENCH_RES)
    for i in range(6):                                     # warm the tiers with DISTINCT strengths so
        host.cook(0.30 + 0.01 * i)                         # each is a real cook (compile/codegen/tier settle)
    host.cache.clear()                                     # measure real cooks, not cache hits
    samples = []
    for i in range(30):
        _, ms, _ = host.cook(0.5 + 0.001 * i)              # each strength distinct -> a real cook (no per-iter clear)
        samples.append(ms)
    med = statistics.median(samples)
    ok = med < 50.0
    print(f"PM-2 benchmark: {med:.2f} ms/frame warm at {BENCH_RES}^2 on {DEVICE} "
          f"(<50 ms target: {'PASS' if ok else 'FAIL'})")
    return ok


# ── the viewer ────────────────────────────────────────────────────────────────

_PAGE = """<!doctype html><meta charset=utf-8><title>TEX host demo</title>
<style>body{font:14px system-ui;background:#111;color:#ddd;text-align:center}
canvas{border:1px solid #333;margin:12px;image-rendering:pixelated}
#s{width:400px}#stat{font-family:monospace;color:#8c8}</style>
<h3>TEX standalone host — grade &rarr; blur &rarr; vignette (no ComfyUI)</h3>
<canvas id=c width=%(res)d height=%(res)d></canvas><br>
vignette strength <input id=s type=range min=0 max=1 step=0.01 value=0.5>
<div id=stat></div>
<script>
const c=document.getElementById('c'),ctx=c.getContext('2d'),s=document.getElementById('s'),st=document.getElementById('stat');
const R=%(res)d, img=ctx.createImageData(R,R);
async function draw(){
  const r=await fetch('/frame?s='+s.value);
  if(r.status!==200) return;                       // 204: a newer frame superseded this cook (SCHED-3)
  const buf=new Uint8ClampedArray(await r.arrayBuffer());
  img.data.set(buf); ctx.putImageData(img,0,0);
  st.textContent='strength '+(+s.value).toFixed(2)+'  |  cook '+r.headers.get('X-Cook-Ms')+' ms  |  '+r.headers.get('X-Cache');
}
s.addEventListener('input',draw); draw();
</script>"""


def serve(host: Host, port: int) -> None:
    page = (_PAGE % {"res": host.res}).encode()
    # v0.31: the viewer cooks THROUGH the SCHED-4 queue, so the demo genuinely submits,
    # preempts and cancels rather than merely being able to. `/stats` exposes the counters.
    qh = QueuedHost(host)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass                                          # quiet

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(200, "text/html; charset=utf-8", page)
            elif u.path == "/frame":
                strength = float(parse_qs(u.query).get("s", ["0.5"])[0])
                try:
                    frame, ms, hitq = qh.frame(strength)
                except CookCancelled:
                    self._send(204, "text/plain", b"")    # a newer frame won; the browser skips this
                    return
                body = host.rgba_bytes(frame)
                self._send(200, "application/octet-stream", body,
                           extra={"X-Cook-Ms": f"{ms:.1f}", "X-Cache": "HIT" if hitq else "cooked"})
            elif u.path == "/stats":
                import json as _json
                self._send(200, "application/json", _json.dumps(qh.stats()).encode())
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code, ctype, body, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"TEX host demo on http://127.0.0.1:{port}  (Ctrl-C to stop)")
    print(f"  cooking through the SCHED-4 queue; counters at http://127.0.0.1:{port}/stats")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    finally:
        qh.close()


def main():
    ap = argparse.ArgumentParser(description="TEX standalone host demo (PORT-5 / PM-6)")
    ap.add_argument("--bench", action="store_true", help="run only the PM-2 benchmark, no server")
    ap.add_argument("--bench-pm6", action="store_true",
                    help="run only the PM-6 ROI-viewport benchmark (10-node comp), no server")
    ap.add_argument("--bench-pm7", action="store_true",
                    help="run only the PM-7 cook-queue scrub (SCHED-4 + PRED-1), no server")
    ap.add_argument("--resolution", type=int, default=1024, help="PM-6 frame size (default 1024)")
    ap.add_argument("--roi", type=int, default=512, help="PM-6 viewport window (default 512)")
    ap.add_argument("--port", type=int, default=8760)
    args = ap.parse_args()

    if args.bench_pm7:
        r = bench_pm7(res=DISPLAY_RES)
        print(f"\n=== PM-7: a slider scrub through the SCHED-4 queue @ {r['res']}^2, "
              f"{r['capability'] or r['device']} ===")
        print(f"  interactive frame (median) : {r['interactive_median_ms']:8.2f} ms")
        print(f"  served from a prefetch     : {r['prefetch_hits']}/{r['frames']}")
        print(f"  COMMITTED render finished  : {r['committed_render_completed']} "
              f"(attempts={r['committed_render_attempts']})")
        print(f"  queue: {r['submitted']} submitted, {r['completed']} completed, "
              f"{r['preempted']} preempted, {r['requeued']} re-queued, "
              f"{r['preempt_denied']} preempts denied by the brake, "
              f"{r['refused']} refused, {r['shed']} shed")
        if r["capability"]:
            print(f"  NOTE: measured on {r['capability']} — PM-7 names the sm_120 box.")
        sys.exit(0)

    if args.bench_pm6:
        r = bench_pm6(res=args.resolution, roi_side=args.roi)
        print(f"\n=== PM-6: {r['stages']}-node comp @ {r['res']}^2, ROI {r['roi']}^2, "
              f"{r['capability'] or r['device']} ===")
        print(f"  all-dirty WHOLE-FRAME : {r['whole_all_ms']:8.2f} ms/frame")
        print(f"  all-dirty ROI         : {r['roi_all_ms']:8.2f} ms/frame   ({r['roi_speedup']}x)")
        print(f"  terminal-knob scrub   : {r['roi_terminal_ms']:8.2f} ms/frame   "
              f"({r['terminal_speedup']}x)")
        print(f"  (the two all-dirty rows cook all {r['stages']} stages; the scrub row reuses "
              f"the {r['stages'] - 1} clean upstream canvases)")
        if r["capability"]:
            print(f"  NOTE: measured on {r['capability']} — PM-6 names the sm_120 box.")
        sys.exit(0)

    ok = run_benchmark()
    if args.bench:
        sys.exit(0 if ok else 1)
    serve(Host(DISPLAY_RES), args.port)


if __name__ == "__main__":
    main()
