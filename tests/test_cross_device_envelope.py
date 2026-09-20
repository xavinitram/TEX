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
from TEX_Wrangle.tex_cache import parse_and_split


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
    # ASK-5: worley_id — structural=True like scatter, because a cross-device boundary
    # flip RELOCATES which cell wins (a whole id moves, not an fp32 quantum), the same
    # relocation shape as scatter's coordinate rounding, not a magnitude drift a
    # pointwise band would suit. Unlike worley_f1/f2, worley_id never enters
    # `_TieredCache` (no jit.trace/torch.compile fusion — the reassociation source
    # behind worley_f1/f2's OWN same-device tier-promotion envelope), so measured
    # cross-device divergence here was exactly 0.0 on this box (grid and random
    # coordinates, 64^2 to 2048^2, coordinate scales 8 to 4080) — no boundary flip
    # was found, not that none can occur. Banded at a small headroom above that
    # measurement rather than at literal zero (AGENTS.md invariant 9: CPU<->GPU is
    # a characterization envelope, never a bit-parity claim), following
    # `_ENVELOPE_SIMPLEX`'s cpu=0.0 precedent in test_v031_noise_tiers.py.
    ("worley_id",
     "float freq = 8.0; float id = worley_id(u * freq, v * freq); @OUT = vec4(id, id, id, 1.0);",
     1e-6, True),
]


def _run_on(code, device, img):
    binds = {}
    bt = {}
    if "@A" in code:
        binds["A"] = img.to(device)
        bt["A"] = TEXType.VEC3
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    out = Interpreter().execute(prog, binds, tm, device=device,
                                output_names=["OUT"], precision="fp32")
    return out["OUT"]


def envelope_verdict(cpu, gpu, band, structural, scale=1.0):
    """`(within, metric, limit, note)` for one probe's fp32 device pair.

    The comparator, lifted out of the loop so something that is not a GPU can drive it. Both
    tensors are already fp32 and on the CPU; nothing here is device-aware, which is the point —
    the decision was never about CUDA, only its inputs were.

    `structural` picks total-energy relative error over pointwise maxdiff, because scatter's
    coordinate rounding legally RELOCATES a quantum to a neighbouring pixel: a large pointwise
    difference that the sum conserves. The two arms therefore disagree on exactly that shape,
    and `test_prlp1_the_envelope_comparator_is_not_inert` drives the disagreement.
    """
    limit = band * scale
    if structural:
        s_cpu, s_gpu = cpu.sum().item(), gpu.sum().item()
        rel = abs(s_cpu - s_gpu) / (abs(s_cpu) + 1e-8)
        return rel <= limit, rel, limit, f"sum cpu {s_cpu:.4f} vs gpu {s_gpu:.4f}"
    md = (cpu - gpu).abs().max().item()
    return md <= limit, md, limit, ""


def test_prlp1_cross_device_envelope(r: SubTestResult):
    print("\n--- PR-LP1: cross-device parity envelope (same-device is the real contract) ---")
    if not torch.cuda.is_available():
        r.skip("PR-LP1 cross-device envelope",
               "no CUDA on this box - the envelope needs both devices")
        return
    scale = float(os.environ.get("TEX_ENVELOPE_SCALE", "1.0"))
    img = make_img(1, 128, 128, 3, seed=7)  # identical bits on both devices via .to()
    for label, code, band, structural in _PROBES:
        try:
            cpu = _run_on(code, "cpu", img).float()
            gpu = _run_on(code, "cuda", img).float().cpu()
            within, metric, limit, note = envelope_verdict(cpu, gpu, band, structural, scale)
            if structural:
                if not within:
                    r.fail(f"PR-LP1 {label}",
                           f"total-energy rel {metric:.2e} > band {limit:.1e} ({note})")
                else:
                    r.ok(f"{label}: energy-conserving across devices "
                         f"(rel {metric:.1e} <= {limit:.1e})")
            else:
                if not within:
                    r.fail(f"PR-LP1 {label}",
                           f"cross-device maxdiff {metric:.2e} > band {limit:.1e} "
                           f"— a driver/torch change blew the envelope; re-band deliberately")
                else:
                    r.ok(f"{label}: within envelope (maxdiff {metric:.1e} <= band {limit:.1e})")
        except Exception as e:
            r.fail(f"PR-LP1 {label}", f"{type(e).__name__}: {e}")


def test_prlp1_the_envelope_comparator_is_not_inert(r: SubTestResult):
    """Invariant 9's enforcer runs only on a box with a GPU, so on the one automated lane the
    row above is a skip and this file asserts nothing about the comparator at all. These
    witnesses are fabricated fp32 pairs, so they run everywhere and prove the mutation shape:
    a pair outside its band must RED, a pair inside it must not, the two arms must disagree on
    the relocation shape `structural` exists for, and `TEX_ENVELOPE_SCALE` must actually widen
    the band rather than be read and dropped.
    """
    print("\n--- PR-LP1: the envelope comparator fires (no GPU needed) ---")
    base = torch.zeros(1, 8, 8, 3)
    base[0, 2, 2] = 1.0

    drifted = base.clone()
    drifted[0, 2, 2] += 1e-3                       # a magnitude drift: both arms should see it
    moved = base.clone()
    moved[0, 2, 2] = 0.0
    moved[0, 2, 3] = 1.0                           # a relocation: pointwise-huge, sum-conserved

    checks = [
        ("pointwise, inside the band", base, base, 1e-5, False, 1.0, True),
        ("pointwise, outside the band", base, drifted, 1e-5, False, 1.0, False),
        ("structural, inside the band", base, base, 1e-6, True, 1.0, True),
        ("structural, outside the band", base, drifted, 1e-6, True, 1.0, False),
        # The arms must disagree here, or `structural` is decoration: a relocated quantum is a
        # 1.0 pointwise diff and a perfectly conserved sum.
        ("pointwise sees a relocation", base, moved, 1e-3, False, 1.0, False),
        ("structural forgives a relocation", base, moved, 1e-6, True, 1.0, True),
        # TEX_ENVELOPE_SCALE is the deliberate re-band lever; a scale that does not widen the
        # limit is a knob that reads as working and is not.
        ("a scale of 1000 widens the band", base, drifted, 1e-5, False, 1000.0, True),
    ]
    bad = []
    for label, a, b, band, structural, scale, want in checks:
        within, metric, limit, _note = envelope_verdict(a, b, band, structural, scale)
        if within is not want:
            bad.append(f"{label}: within={within}, expected {want} "
                       f"(metric {metric:.2e}, limit {limit:.1e})")
    if bad:
        r.fail("PR-LP1 comparator witness", "\n  ".join(bad))
    else:
        r.ok(f"the envelope comparator decides all {len(checks)} fabricated cases correctly, "
             f"with no CUDA in the room")
