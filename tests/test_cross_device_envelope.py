"""
PR-LP1 — the cross-device parity *envelope* (doc 28 §2.1).

The only exactness contract in TEX is interp<->codegen on the SAME device
(AGENTS.md invariant #2). CPU<->CUDA bit-parity does NOT exist and nothing pays
for it — measured fp32 cross-device divergence is already 1.8e-7 (pointwise) up to
6.1e-2 (scatter, where a coordinate-rounding ULP legally moves a whole deposit to a
neighbouring pixel: structurally benign, numerically large).

This test does not sell parity; it *pins the envelope*. Each program class must stay
inside its measured divergence band, so a torch/driver upgrade that blows a band is a
loud, recorded decision instead of silent drift. Bands are ~10x the measured maxdiff
(Appendix A) and globally scalable via TEX_ENVELOPE_SCALE for a deliberate re-band.

CUDA-gated: skips clean on a CPU-only box.
"""
from helpers import *


# (label, code, band, structural) — structural=True compares total energy (img_sum
# rel-tol) instead of pointwise, because scatter coordinate-rounding legally relocates
# quanta between neighbouring pixels (a large pointwise diff, conserved in the sum).
_PROBES = [
    ("grade_pointwise",
     "vec3 c = @A.rgb; c = pow(c, vec3(1.0/2.2));"
     "float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));"
     "c = lerp(vec3(lum), c, 1.2); @OUT = vec4(c, 1.0);",
     1e-5, False),
    ("mat3xform",
     "mat3 m = mat3(0.393, 0.769, 0.189, 0.349, 0.686, 0.168, 0.272, 0.534, 0.131);"
     "@OUT = vec4(m * @A.rgb, 1.0);",
     1e-5, False),
    ("gauss_conv",
     "@OUT = vec4(gauss_blur(@A, 2.0), 1.0);",
     1e-5, False),
    ("fbm_noise",
     "float freq = 4.0; float n = fbm(u * freq, v * freq, 6); @OUT = vec4(n, n, n, 1.0);",
     1e-5, False),
    ("lens_warp",
     "float su = u + sin(v * 6.28) * 0.02; float sv = v + cos(u * 6.28) * 0.02;"
     "@OUT = sample(@A, su, sv);",
     5e-4, False),
    ("scatter",
     "@OUT[ix, iy] = vec3(0.0);"
     "float dx = simplex(u * 3.0, v * 3.0) * 8.0;"
     "float dy = simplex(u * 3.0 + 5.3, v * 3.0 + 7.1) * 8.0;"
     "int tx = int(ix + dx); int ty = int(iy + dy);"
     "@OUT[tx, ty] += vec3(0.1);",
     1e-3, True),
]


def _run_on(code, device, img):
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    binds = {}
    bt = {}
    if "@A" in code:
        binds["A"] = img.to(device)
        bt["A"] = TEXType.VEC3
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    out = Interpreter().execute(prog, binds, tm, device=device,
                                output_names=["OUT"], precision="fp32")
    return out["OUT"]


def test_prlp1_cross_device_envelope(r: SubTestResult):
    print("\n--- PR-LP1: cross-device parity envelope (same-device is the real contract) ---")
    if not torch.cuda.is_available():
        r.ok("PR-LP1 cross-device envelope (no GPU, SKIPPED)")
        return
    scale = float(os.environ.get("TEX_ENVELOPE_SCALE", "1.0"))
    img = make_img(1, 128, 128, 3, seed=7)  # identical bits on both devices via .to()
    for label, code, band, structural in _PROBES:
        try:
            cpu = _run_on(code, "cpu", img).float()
            gpu = _run_on(code, "cuda", img).float().cpu()
            limit = band * scale
            if structural:
                s_cpu, s_gpu = cpu.sum().item(), gpu.sum().item()
                rel = abs(s_cpu - s_gpu) / (abs(s_cpu) + 1e-8)
                if rel > limit:
                    r.fail(f"PR-LP1 {label}",
                           f"total-energy rel {rel:.2e} > band {limit:.1e} "
                           f"(sum cpu {s_cpu:.4f} vs gpu {s_gpu:.4f})")
                else:
                    r.ok(f"{label}: energy-conserving across devices (rel {rel:.1e} <= {limit:.1e})")
            else:
                md = (cpu - gpu).abs().max().item()
                if md > limit:
                    r.fail(f"PR-LP1 {label}",
                           f"cross-device maxdiff {md:.2e} > band {limit:.1e} "
                           f"— a driver/torch change blew the envelope; re-band deliberately")
                else:
                    r.ok(f"{label}: within envelope (maxdiff {md:.1e} <= band {limit:.1e})")
        except Exception as e:
            r.fail(f"PR-LP1 {label}", f"{type(e).__name__}: {e}")
