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
from TEX_Wrangle import tex_engine, tex_fusion, tex_results, tex_marshalling, tex_api
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

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass                                          # quiet

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(200, "text/html; charset=utf-8", page)
            elif u.path == "/frame":
                strength = float(parse_qs(u.query).get("s", ["0.5"])[0])
                tok = host.new_request()                  # SCHED-3: supersedes any in-flight cook
                try:
                    frame, ms, hitq = host.cook(strength, cancel=tok)
                except CookCancelled:
                    self._send(204, "text/plain", b"")    # a newer frame won; the browser skips this
                    return
                body = host.rgba_bytes(frame)
                self._send(200, "application/octet-stream", body,
                           extra={"X-Cook-Ms": f"{ms:.1f}", "X-Cache": "HIT" if hitq else "cooked"})
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
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


def main():
    ap = argparse.ArgumentParser(description="TEX standalone host demo (PORT-5)")
    ap.add_argument("--bench", action="store_true", help="run only the PM-2 benchmark, no server")
    ap.add_argument("--port", type=int, default=8760)
    args = ap.parse_args()
    ok = run_benchmark()
    if args.bench:
        sys.exit(0 if ok else 1)
    serve(Host(DISPLAY_RES), args.port)


if __name__ == "__main__":
    main()
