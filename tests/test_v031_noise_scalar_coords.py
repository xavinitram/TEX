"""v0.31 — NOISE-SCALAR: a constant coordinate must render the same picture on every device.

    `fbm(u*8.0, v*8.0, 0.5, 4)` returned an image on CPU and RAISED on CUDA.

A constant argument reaches the noise layer as a 0-dim tensor sitting beside full
[B, H, W] coordinate grids. The pointwise noise bodies broadcast it without complaint,
which is why 3D Perlin, flow and (z-slot) Worley were always fine. Four other places
assumed every coordinate carried the grid's rank:

  1. `_octave_perlins` batches all octaves into ONE noise call on GPU by stacking the
     frequency-scaled coords on a new leading axis. A 0-dim coord stacks to [N], which
     broadcasting then RIGHT-aligns against a sibling's [N, B, H, W] — putting the
     octave count against W. Because the batching is GPU-only, so was the failure:
     "The size of tensor a (32) must match the size of tensor b (4) at ... dimension 3".
     Hit fbm / ridged / billow / turbulence in 3D, and ridged in 2D.
  2. `_curl3d` / `_curl2d` stack 12 (resp. 4) offset samples the same way, same result.
  3. `_ridged_nd` / `_alligator_nd` allocated their accumulator with
     `torch.zeros_like(coords[0])`. A constant in the FIRST slot made that 0-dim, and the
     first in-place `add_` of a grid into it failed — on BOTH devices, so this half was a
     plain crash rather than a divergence, and it hid behind the louder CUDA one.
  4. `_worley3d` / `_worley2d` took the neighbour-offset rank from x alone, so a constant
     x sized the 27- (resp. 9-) neighbour offsets for rank 0 and mis-aligned every one.

Fixed by `_widest` / `_stack_coord` / `_zeros_broadcast` in tex_runtime/noise.py: the
stacked axis is kept LEADING with the coord's own dims right-aligned under it, and the
accumulator is sized from the broadcast of every coord rather than from the first.
The padding is free — a constant stays [N,1,1,1] instead of being expanded to the grid.

The fix is a no-op wherever the coords already share a shape, which is every call that
worked before: measured bit-identical against the pre-fix module on 248 (device, dtype,
shape, function) combinations spanning cpu/cuda, fp32/fp16, ranks 2-4, and all 23 noise
entry points.

What each row pins:
  * device_parity        — NEVER-SEVER. The reported bug, end-to-end through `tex_engine
                           .cook`, for every 3D form with a constant in the z slot AND in
                           the x slot. Needs CUDA; the divergence cannot exist without it.
  * equals_grid          — THE correctness row, and the one a crash-only test cannot
                           replace. Getting the right SHAPE is not the claim — a constant
                           coord must produce exactly what a full grid of that constant
                           produces. Bit-exact, per device, so it has teeth on CPU-only CI.
  * cooks_in_any_slot    — the half that failed on BOTH devices (root cause 3 and 4), which
                           a CPU-vs-CUDA comparison would score as "agreeing" while both
                           sides raised. Runs anywhere.
  * two_d_family         — the 2D siblings share `_octave_perlins`, `_ridged_nd`,
                           `_alligator_nd`, `_curl2d` and `_worley2d`, so they carried the
                           same defect: `curl(0.5, v*8.0)` diverged by device too.
  * helpers_are_noops    — CANARY on the mechanism. `_stack_coord` at equal rank must be
                           exactly `torch.stack(..., dim=0)`, and `_zeros_broadcast` at
                           equal shapes exactly `torch.zeros_like(coords[0])`. If either
                           drifts, the fix stops being free on the hot path and the
                           bit-exactness measured above is silently void.
"""
from helpers import *

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import noise as _noise
from TEX_Wrangle.tex_runtime.noise import _stack_coord, _widest, _zeros_broadcast

_CUDA = torch.cuda.is_available()

_K = 0.5        # the constant coordinate under test
_H, _W = 24, 32


# name -> the TEX expression, with {z} the slot the constant goes in.
# vec3-valued forms (curl) are wrapped separately so @OUT stays a vec4.
_FORMS_3D = [
    ("perlin",     "vec3(perlin({x}, {y}, {z}))"),
    ("simplex",    "vec3(simplex({x}, {y}, {z}))"),
    ("fbm",        "vec3(fbm({x}, {y}, {z}, 4))"),
    ("ridged",     "vec3(ridged({x}, {y}, {z}, 4))"),
    ("billow",     "vec3(billow({x}, {y}, {z}, 4))"),
    ("turbulence", "vec3(turbulence({x}, {y}, {z}, 4))"),
    ("flow",       "vec3(flow({x}, {y}, {z}, 0.3))"),
    ("worley_f1",  "vec3(worley_f1({x}, {y}, {z}))"),
    ("worley_f2",  "vec3(worley_f2({x}, {y}, {z}))"),
    ("voronoi",    "vec3(voronoi({x}, {y}, {z}))"),
    ("alligator",  "vec3(alligator({x}, {y}, {z}, 3))"),
    ("curl",       "curl({x}, {y}, {z})"),
]

# Curl is a CENTRAL DIFFERENCE over a 0.001 step, so it multiplies whatever the two
# devices' fp32 units disagree about by 1/(2*eps) = 500. Measured on an RTX 2080 SUPER
# (torch 2.5.0+cu118): every other 3D form lands under 2.3e-06 CPU-vs-CUDA, curl at
# 5.7e-04 — and it reads 5.7e-04 for all-grid coords too, i.e. that is the honest
# hardware floor for curl and not something a constant coord introduces. The budgets
# below sit ~40x over each measured value: far under the O(0.1) a genuine value bug
# would show, far over ordinary hardware noise.
_TOL = 1e-4
_TOL_CURL = 5e-3


def _cook(prog, dev):
    """Cook `prog` on `dev` and hand back @OUT. Mirrors the reported repro exactly."""
    img = torch.rand(1, _H, _W, 4, device=dev)
    res = tex_engine.cook(prog, {"A": img}, device_mode=dev)
    return res.outputs[res.output_names[0]]


def _prog(expr):
    return "@OUT = vec4(%s, 1.0);" % expr


def _slots():
    """(label, x, y, z) with the constant in each coordinate slot in turn.

    The z slot is the reported repro. x and y are here because root causes 3 and 4 only
    fire when the constant lands FIRST — `zeros_like(coords[0])` and the offset rank both
    read coords[0] — so a z-only test would leave half the fix unpinned.
    """
    g = "u*8.0"
    h = "v*8.0"
    k = repr(_K)
    return [
        ("z=const", g, h, k),
        ("x=const", k, h, "v*3.0"),
        ("y=const", g, k, "v*3.0"),
        ("all const", k, k, k),
    ]


def test_v031_noise_scalar_coord_device_parity(r: SubTestResult):
    """NEVER-SEVER: a constant coordinate renders the same on CPU and CUDA."""
    print("\n--- v0.31 NOISE-SCALAR: constant coord, CPU vs CUDA (end-to-end) ---")
    if not _CUDA:
        r.skip("scalar-coord device parity", "no CUDA device — the divergence is GPU-only")
        return

    bad = []
    checked = 0
    for slot, x, y, z in _slots():
        for name, tmpl in _FORMS_3D:
            prog = _prog(tmpl.format(x=x, y=y, z=z))
            try:
                a = _cook(prog, "cpu").float().cpu()
            except Exception as e:
                bad.append("%s %s: CPU raised %s: %s" % (name, slot, type(e).__name__, e))
                continue
            try:
                b = _cook(prog, "cuda").float().cpu()
            except Exception as e:
                bad.append("%s %s: CUDA raised %s: %s" % (name, slot, type(e).__name__, e))
                continue
            checked += 1
            if a.shape != b.shape:
                bad.append("%s %s: shape %s vs %s" % (name, slot, tuple(a.shape), tuple(b.shape)))
                continue
            tol = _TOL_CURL if name == "curl" else _TOL
            d = float((a - b).abs().max())
            if not (d <= tol):
                bad.append("%s %s: maxdiff=%.3e > %.0e" % (name, slot, d, tol))

    if bad:
        r.fail("scalar-coord device parity",
               "%d of %d diverged by device:\n  " % (len(bad), len(bad) + checked)
               + "\n  ".join(bad[:12]))
    else:
        r.ok("all %d (3D form x constant slot) pairs agree CPU vs CUDA" % checked)


def test_v031_noise_scalar_coord_equals_grid(r: SubTestResult):
    """A constant coord must equal a full grid of that constant — bit-exact, per device.

    This is the claim; "it stopped raising" is not. A fix that got the rank right but the
    ALIGNMENT wrong would still cook a picture on both devices and still agree between
    them — it would just be the wrong picture. Comparing against the materialized grid is
    what distinguishes the two.
    """
    print("\n--- v0.31 NOISE-SCALAR: constant coord == grid of that constant ---")
    devices = ["cpu"] + (["cuda"] if _CUDA else [])
    cases = [
        ("fbm o1",      lambda x, y, z: _noise._fbm3d(x, y, z, 1)),
        ("fbm o4",      lambda x, y, z: _noise._fbm3d(x, y, z, 4)),
        ("fbm o8",      lambda x, y, z: _noise._fbm3d(x, y, z, 8)),
        ("ridged",      lambda x, y, z: _noise._ridged3d(x, y, z, 4)),
        ("billow",      lambda x, y, z: _noise._billow3d(x, y, z, 4)),
        ("turbulence",  lambda x, y, z: _noise._turbulence3d(x, y, z, 4)),
        ("flow",        lambda x, y, z: _noise._flow3d(x, y, z, 0.3)),
        ("alligator",   lambda x, y, z: _noise._alligator3d(x, y, z, 3)),
        ("curl",        lambda x, y, z: _noise._curl3d(x, y, z)),
        ("worley_f1",   lambda x, y, z: _noise._worley3d(x, y, z, return_f2=False)),
        ("worley_f2",   lambda x, y, z: _noise._worley3d(x, y, z, return_f2=True)),
        ("perlin",      lambda x, y, z: _noise._perlin3d_fast(x, y, z)),
    ]
    bad = []
    checked = 0
    for dev in devices:
        gen = torch.Generator().manual_seed(7)
        base_x = (torch.rand((1, _H, _W), generator=gen) * 8.0).to(dev)
        base_y = (torch.rand((1, _H, _W), generator=gen) * 8.0).to(dev)
        base_z = (torch.rand((1, _H, _W), generator=gen) * 3.0).to(dev)
        konst = torch.scalar_tensor(_K, dtype=torch.float32, device=dev)
        grid = torch.full_like(base_x, _K)
        # The constant in each slot, against the identical call with it materialized.
        trios = [
            ("z", (base_x, base_y, konst), (base_x, base_y, grid)),
            ("x", (konst, base_y, base_z), (grid, base_y, base_z)),
            ("y", (base_x, konst, base_z), (base_x, grid, base_z)),
        ]
        for name, fn in cases:
            for slot, scalar_args, grid_args in trios:
                try:
                    a = fn(*scalar_args)
                    b = fn(*grid_args)
                except Exception as e:
                    bad.append("%s %s@%s raised %s: %s"
                               % (name, slot, dev, type(e).__name__, e))
                    continue
                checked += 1
                if a.shape != b.shape:
                    bad.append("%s %s@%s: shape %s vs %s"
                               % (name, slot, dev, tuple(a.shape), tuple(b.shape)))
                elif not torch.equal(a, b):
                    bad.append("%s %s@%s: not bit-exact, maxdiff=%.3e"
                               % (name, slot, dev, float((a - b).abs().max())))
    if bad:
        r.fail("constant coord == grid",
               "%d mismatched:\n  " % len(bad) + "\n  ".join(bad[:12]))
    else:
        r.ok("all %d (3D form x slot x device) constants equal their grid bit-exactly"
             % checked)


def test_v031_noise_scalar_coord_cooks_in_any_slot(r: SubTestResult):
    """The half that failed on BOTH devices: a constant in the FIRST coordinate slot.

    `_ridged_nd`/`_alligator_nd` sized their accumulator from coords[0] and `_worley*`
    took the neighbour-offset rank from x, so a leading constant raised everywhere —
    "output with shape [] doesn't match the broadcast shape" on CPU and a stack-alignment
    error on CUDA. Two failures, no divergence, which is exactly what a parity-only test
    scores as a pass. Runs on every device present.
    """
    print("\n--- v0.31 NOISE-SCALAR: leading constant cooks on every device ---")
    devices = ["cpu"] + (["cuda"] if _CUDA else [])
    progs = [
        ("ridged3 x",    "vec3(ridged(0.5, v*8.0, u*3.0, 4))"),
        ("alligator3 x", "vec3(alligator(0.5, v*8.0, u*3.0, 3))"),
        ("worley3 x",    "vec3(worley_f1(0.5, v*8.0, u*3.0))"),
        ("ridged2 x",    "vec3(ridged(0.5, v*8.0, 4))"),
        ("alligator2 x", "vec3(alligator(0.5, v*8.0, 3))"),
        ("worley2 x",    "vec3(worley_f1(0.5, v*8.0))"),
        ("curl3 x",      "curl(0.5, v*8.0, u*3.0)"),
        ("curl2 x",      "vec3(curl(0.5, v*8.0), 0.0)"),
    ]
    bad = []
    checked = 0
    for dev in devices:
        for name, expr in progs:
            try:
                out = _cook(_prog(expr), dev)
            except Exception as e:
                bad.append("%s@%s raised %s: %s" % (name, dev, type(e).__name__, e))
                continue
            checked += 1
            if tuple(out.shape)[1:3] != (_H, _W):
                bad.append("%s@%s: cooked %s, expected a %dx%d frame"
                           % (name, dev, tuple(out.shape), _H, _W))
            elif not bool(torch.isfinite(out).all()):
                bad.append("%s@%s: produced non-finite values" % (name, dev))
    if bad:
        r.fail("leading constant cooks", "\n  " + "\n  ".join(bad[:12]))
    else:
        r.ok("all %d (form x device) leading-constant programs cook a finite frame"
             % checked)


def test_v031_noise_scalar_coord_2d_family(r: SubTestResult):
    """The 2D siblings share the same helpers and carried the same defect.

    Tolerance rather than bit-equality here, and only because `_worley2d` runs through
    `_TieredCache`: it settles a traced module PER input signature, so the constant call
    and the grid call legitimately profile as two signatures and the fuser may associate
    one of them differently. Measured spread on CPU: 5.96e-08 for alligator2 — one ULP —
    and exactly 0 for every other 2D form and for all of CUDA. That is pre-existing tier
    behaviour (the un-cached 3D Worley is bit-exact on both devices), not the broadcast.
    """
    print("\n--- v0.31 NOISE-SCALAR: 2D family with a constant coord ---")
    devices = ["cpu"] + (["cuda"] if _CUDA else [])
    cases = [
        ("curl2",       lambda x, y: _noise._curl2d(x, y)),
        ("worley2_f1",  lambda x, y: _noise._worley2d(x, y, return_f2=False)),
        ("worley2_f2",  lambda x, y: _noise._worley2d(x, y, return_f2=True)),
        ("ridged2",     lambda x, y: _noise._ridged2d(x, y, 4)),
        ("billow2",     lambda x, y: _noise._billow2d(x, y, 4)),
        ("turbulence2", lambda x, y: _noise._turbulence2d(x, y, 4)),
        ("alligator2",  lambda x, y: _noise._alligator2d(x, y, 3)),
        ("perlin2",     lambda x, y: _noise._perlin2d_fast(x, y)),
        ("flow2",       lambda x, y: _noise._flow2d(x, y, 0.3)),
    ]
    tol = 1e-6
    bad = []
    checked = 0
    for dev in devices:
        gen = torch.Generator().manual_seed(11)
        base = (torch.rand((1, _H, _W), generator=gen) * 8.0).to(dev)
        konst = torch.scalar_tensor(_K, dtype=torch.float32, device=dev)
        grid = torch.full_like(base, _K)
        for name, fn in cases:
            for slot, s_args, g_args in (("x", (konst, base), (grid, base)),
                                         ("y", (base, konst), (base, grid))):
                try:
                    a = fn(*s_args)
                    b = fn(*g_args)
                except Exception as e:
                    bad.append("%s %s@%s raised %s: %s"
                               % (name, slot, dev, type(e).__name__, e))
                    continue
                checked += 1
                if a.shape != b.shape:
                    bad.append("%s %s@%s: shape %s vs %s"
                               % (name, slot, dev, tuple(a.shape), tuple(b.shape)))
                    continue
                d = float((a - b).abs().max())
                if not (d <= tol):
                    bad.append("%s %s@%s: maxdiff=%.3e > %.0e" % (name, slot, dev, d, tol))
    if bad:
        r.fail("2D constant coord", "\n  " + "\n  ".join(bad[:12]))
    else:
        r.ok("all %d (2D form x slot x device) constants match their grid" % checked)


def test_v031_noise_scalar_coord_reduced_precision(r: SubTestResult):
    """The same claim under precision="fp16" / "bf16" — a SECOND, dtype-shaped divergence.

    Rank was not the only way a constant coord failed to look like its siblings. Under a
    reduced-precision cook the engine hands the noise layer fp32 `u`/`v` grids but casts
    the literal to the cook dtype, so `fbm(u*8.0, v*8.0, 0.5, 4)` arrives as two fp32
    grids and one fp16 scalar. Un-stacked that is harmless — a 0-dim tensor is dtype-WEAK
    and torch promotes it — but `_stack_coord` makes it RANKED, and a ranked tensor is
    dtype-STRONG. CUDA's `torch.lerp` then rejects a weight whose dtype differs from its
    endpoints ("expected dtype float for `weight` but got c10::Half"); CPU's lerp accepts
    it. Twelve (form x slot) pairs cooked on CPU and raised on CUDA.

    Curl needs the promotion on BOTH paths, not just the batched one. It is a central
    difference, and at |x| ~ 8 the fp16 ulp (~0.008) swallows an eps of 0.001 whole, so a
    device that keeps the constant in fp16 and a device that promotes it disagree about
    the derivative rather than about the dtype: measured 1.1e-01 (curl2) and 2.2e+00
    (curl3) between devices when only the CUDA branch promoted.
    """
    print("\n--- v0.31 NOISE-SCALAR: constant coord under fp16 / bf16 ---")
    if not _CUDA:
        r.skip("reduced-precision scalar coord",
               "no CUDA — reduced precision is a CUDA cook mode and the strict lerp "
               "dtype check is the CUDA kernel's")
        return

    bad = []
    checked = 0
    for prec in ("fp16", "bf16"):
        for slot, x, y, z in _slots():
            for name, tmpl in _FORMS_3D:
                prog = _prog(tmpl.format(x=x, y=y, z=z))
                got = {}
                for dev in ("cpu", "cuda"):
                    try:
                        img = torch.rand(1, _H, _W, 4, device=dev)
                        res = tex_engine.cook(prog, {"A": img}, device_mode=dev,
                                              precision=prec)
                        got[dev] = res.outputs[res.output_names[0]].float().cpu()
                    except Exception as e:
                        got[dev] = "%s: %s" % (type(e).__name__, e)
                cpu_bad = isinstance(got["cpu"], str)
                cuda_bad = isinstance(got["cuda"], str)
                if cpu_bad or cuda_bad:
                    bad.append("%s %s %s: cpu=%s cuda=%s"
                               % (prec, name, slot,
                                  "RAISED" if cpu_bad else "ok",
                                  got["cuda"] if cuda_bad else "ok"))
                    continue
                checked += 1
                # Looser than the fp32 row for the obvious reason — the cook itself is
                # reduced precision, so the two devices' rounding has more room to differ.
                tol = 5e-2 if name == "curl" else 1e-2
                d = float((got["cpu"] - got["cuda"]).abs().max())
                if not (d <= tol):
                    bad.append("%s %s %s: maxdiff=%.3e > %.0e" % (prec, name, slot, d, tol))
    if bad:
        r.fail("reduced-precision scalar coord",
               "%d diverged:\n  " % len(bad) + "\n  ".join(bad[:12]))
    else:
        r.ok("all %d (precision x 3D form x slot) pairs agree CPU vs CUDA" % checked)


def test_v031_noise_scalar_coord_batched_equals_per_octave(r: SubTestResult):
    """`_octave_perlins` promises its two paths are bit-exact. Hold it for any rank mix.

    The GPU batch and the CPU per-octave loop are the same function by contract — the
    docstring's "Bit-exact either way" is what lets the batching exist at all. A constant
    coord broke that silently (the batched path raised where the loop returned), and the
    fix restores it by padding rather than by expanding.

    This is also the only row covering INTERMEDIATE ranks — a coord of rank 1 or 2 beside
    rank-3 grids. Those do not arise from today's DSL, where coordinates are either full
    grids or 0-dim constants, but `_stack_coord`'s padding rule claims to reproduce
    right-alignment for every rank, and an untested claim is a guess.
    """
    print("\n--- v0.31 NOISE-SCALAR: batched octaves == per-octave, any rank mix ---")
    devices = ["cpu"] + (["cuda"] if _CUDA else [])
    freqs = (1.0, 2.0, 4.0, 8.0)
    bad = []
    checked = 0
    for dev in devices:
        gen = torch.Generator().manual_seed(3)
        x = (torch.rand((1, _H, _W), generator=gen) * 8.0).to(dev)
        y = (torch.rand((1, _H, _W), generator=gen) * 8.0).to(dev)
        for zshape in ((), (_W,), (_H, _W), (1, _H, _W)):
            z = (torch.scalar_tensor(0.7, dtype=torch.float32, device=dev) if not zshape
                 else (torch.rand(zshape, generator=gen) * 3.0).to(dev))
            batched = _noise._octave_perlins(_noise._perlin3d_fast, (x, y, z), len(freqs))
            ref = [_noise._perlin3d_fast(x * f, y * f, z * f) for f in freqs]
            checked += 1
            for i, (a, b) in enumerate(zip(batched, ref)):
                if a.shape != b.shape:
                    bad.append("%s z%s octave %d: shape %s vs %s"
                               % (dev, zshape or "()", i, tuple(a.shape), tuple(b.shape)))
                elif not torch.equal(a, b):
                    bad.append("%s z%s octave %d: not bit-exact, maxdiff=%.3e"
                               % (dev, zshape or "()", i, float((a - b).abs().max())))
    if bad:
        r.fail("batched == per-octave", "\n  " + "\n  ".join(bad[:12]))
    else:
        r.ok("batched octaves == per-octave for %d rank mixes on %s" % (checked, devices))


def test_v031_noise_scalar_coord_shipped_example(r: SubTestResult):
    """REGRESSION WITNESS: `examples/turbulent_displace.tex` shipped hitting this bug.

    It writes `float nz = $evolution;` — a scalar parameter — and feeds it to
    `fbm(nx, ny, nz, $complexity)`, which is the reported repro in shipped example code.
    Verified: on the pre-fix module this example cooked on CPU and raised on CUDA with
    "The size of tensor a (32) must match the size of tensor b (3)".

    It shipped because the example corpus is only ever cooked on ONE device:
    `test_example_files` in test_integration.py runs all 114 examples with `device="cpu"`,
    so a GPU-only divergence is structurally invisible to it. This row does not close that
    hole — it pins the one example known to have fallen through it.
    """
    print("\n--- v0.31 NOISE-SCALAR: the shipped example that hit this ---")
    path = (Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            / "examples" / "turbulent_displace.tex")
    if not path.exists():
        r.skip("shipped example cooks", "examples/turbulent_displace.tex not present")
        return

    with open(path, encoding="utf-8") as f:
        prog = f.read()
    if "fbm(" not in prog or "$evolution" not in prog:
        r.skip("shipped example cooks",
               "turbulent_displace.tex no longer feeds a scalar param to fbm — "
               "this row's premise moved, retarget it rather than deleting it")
        return

    devices = ["cpu"] + (["cuda"] if _CUDA else [])
    bad = []
    for dev in devices:
        img = torch.rand(1, _H, _W, 4, device=dev)
        try:
            res = tex_engine.cook(prog, {"image": img}, device_mode=dev)
            out = res.outputs[res.output_names[0]]
        except Exception as e:
            bad.append("%s raised %s: %s" % (dev, type(e).__name__, e))
            continue
        if tuple(out.shape)[1:3] != (_H, _W):
            bad.append("%s cooked %s, expected a %dx%d frame"
                       % (dev, tuple(out.shape), _H, _W))
        elif not bool(torch.isfinite(out).all()):
            bad.append("%s produced non-finite values" % dev)
    if bad:
        r.fail("shipped example cooks", "\n  " + "\n  ".join(bad))
    else:
        r.ok("turbulent_displace.tex cooks a finite frame on %s" % devices)


def test_v031_noise_scalar_coord_helpers_are_noops(r: SubTestResult):
    """CANARY: the fix must stay free on the path that already worked.

    Every currently-working call passes coords that already share a shape. On that path
    `_stack_coord` must reduce to the bare `torch.stack(..., dim=0)` it replaced and
    `_zeros_broadcast` to the bare `torch.zeros_like(coords[0])` — same values, same
    dtype, same device. If either grows a copy, a cast or a device hop, the batching
    paths stop being the optimisation they are documented to be, and the 248-combination
    bit-exactness this fix was measured against no longer holds.
    """
    print("\n--- v0.31 NOISE-SCALAR: helpers are no-ops at equal rank ---")
    devices = ["cpu"] + (["cuda"] if _CUDA else [])
    bad = []
    for dev in devices:
        for dtype in (torch.float32, torch.float16):
            if dtype is torch.float16 and dev == "cpu":
                continue    # fp16 on CPU is not a cook mode TEX offers
            for shape in ((1, _H, _W), (_H, _W), (1, 4, 6, 2)):
                a = torch.rand(shape, device=dev, dtype=dtype)
                b = torch.rand(shape, device=dev, dtype=dtype)
                c = torch.rand(shape, device=dev, dtype=dtype)
                coords = (a, b, c)
                tag = "%s/%s/%s" % (dev, str(dtype).replace("torch.", ""), tuple(shape))

                # _widest must pick coords[0] when nothing is wider — that identity is
                # what makes every `ref.device`/`ref.dim()` below unchanged from the
                # `x.device`/`x.dim()` they replaced.
                if _widest(coords) is not a:
                    bad.append("%s: _widest did not return coords[0] at equal rank" % tag)

                parts = [a * 1.0, a * 2.0, a * 4.0]
                got = _stack_coord(parts, a, a.dim())
                want = torch.stack(parts, dim=0)
                if got.shape != want.shape or not torch.equal(got, want):
                    bad.append("%s: _stack_coord != torch.stack at equal rank" % tag)

                zb = _zeros_broadcast(coords)
                zl = torch.zeros_like(a)
                if (zb.shape != zl.shape or zb.dtype != zl.dtype
                        or zb.device != zl.device or not torch.equal(zb, zl)):
                    bad.append("%s: _zeros_broadcast != zeros_like(coords[0]) "
                               "(%s/%s/%s vs %s/%s/%s)"
                               % (tag, tuple(zb.shape), zb.dtype, zb.device,
                                  tuple(zl.shape), zl.dtype, zl.device))

                # ...and it must still pad correctly when a coord IS narrower.
                k = torch.scalar_tensor(_K, dtype=dtype, device=dev)
                padded = _stack_coord([k * 1.0, k * 2.0, k * 4.0], k, a.dim())
                if tuple(padded.shape) != (3,) + (1,) * a.dim():
                    bad.append("%s: _stack_coord padded a 0-dim coord to %s, expected %s"
                               % (tag, tuple(padded.shape), (3,) + (1,) * a.dim()))
                if _zeros_broadcast((k, b, c)).shape != zl.shape:
                    bad.append("%s: _zeros_broadcast did not broadcast past a 0-dim coord"
                               % tag)
    if bad:
        r.fail("helpers are no-ops", "\n  " + "\n  ".join(bad[:12]))
    else:
        r.ok("_widest/_stack_coord/_zeros_broadcast reduce to the originals on %s" % devices)
