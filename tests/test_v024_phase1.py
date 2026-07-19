"""
v0.24 Phase 1 — "See less, cook less" (spatial laziness).

ROI-2  tex_roi.py — per-binding spatial footprint analysis on the lattice
       point ⊑ halo(u,d,l,r) ⊑ image, composing $param folding + the ROI-1 reach model
       (gauss_blur's radius is 3·sigma, carried as the ('halo_arg',1,3.0) multiplier) +
       affine offset extraction. `roi_plan` derives the ROI-3 execution plan.
ROI-3  the interpreter's tile=(y0,H_total) generalized to roi=(x0,y0,w,h,W,H) (2-D
       seam-exact coords) + tex_memory.run_roi (narrow inputs to ROI ⊕ halo, cook, crop).
ROI-4  THE SHIP GATE (this file): the reach-pinning derivation test (impulse spread ≤
       declared reach), spatial never-sever rows (unknown → whole-frame, never a shrunk
       ROI), and the differential ROI oracle (random programs × random ROIs, ROI-assembled
       == whole-frame cook within the FUS-3 1e-5 tolerance). ROI ships nothing until green.

CPU-pinned for determinism; CUDA looped when present.
"""
import math

from helpers import *  # noqa: F401,F403  (SubTestResult, Lexer/Parser/TypeChecker, torch, TEXType)
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib  # noqa: F401  (populates REGISTRY)
from TEX_Wrangle import tex_roi
from TEX_Wrangle.tex_memory import run_roi
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_compiler.ast_nodes import NumberLiteral

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]
_TOL = 1e-5   # the FUS-3 convention: pointwise/morphology bit-exact, conv/bilateral ~1 ulp


def _compile(code, binding_types):
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    tc = TypeChecker(binding_types=binding_types, source=code)
    tm = tc.check(prog)
    outs = sorted(tc.assigned_bindings.keys())
    return prog, tm, outs


def _declared_reach(name, arg_vals):
    """The reach tex_roi's model declares for `name(img, arg_vals…)` — the SAME resolver
    ROI-3 narrows by, so a wrong multiplier is caught here, not at ship. The image is arg 0;
    the radius/sigma args follow (so a ('halo_arg', 1, …) reads arg_vals[0])."""
    args = [None] + [NumberLiteral(value=float(v)) for v in arg_vals]
    return tex_roi._call_reach(name, args)


# ── ROI-2: footprint analysis ─────────────────────────────────────────────────

def test_roi2_footprints(r: SubTestResult):
    print("\n--- ROI-2: per-binding footprints on the lattice ---")
    cases = [
        ("@OUT = @A * 0.5;", {}, "A", "point", 0),
        ("@OUT = gauss_blur(@A, 2.0);", {}, "A", "halo", 6),          # 3·sigma
        ("@OUT = erode(@A, 3.0);", {}, "A", "halo", 3),               # radius=pixels
        ("@OUT = gauss_blur(gauss_blur(@A, 2.0), 2.0);", {}, "A", "halo", 12),  # additive
        ("@OUT = @A + gauss_blur(@A, 2.0);", {}, "A", "halo", 6),     # LUB over reads
        ("@OUT = bilateral_filter(@A, 1.5, 0.2);", {}, "A", "halo", 3),
        ("@OUT = sample(@A, u * 0.5, v);", {}, "A", "image", 0),      # gather
        ("@OUT = @A / img_max(@A);", {}, "A", "image", 0),            # reduction
    ]
    for code, params, name, want_kind, want_reach in cases:
        try:
            fp = tex_roi.binding_footprints(code, params)
            got = fp[name]
            assert got.kind == want_kind, f"{name}: kind {got.kind} != {want_kind}"
            if want_kind == "halo":
                assert got.reach == want_reach, f"{name}: reach {got.reach} != {want_reach}"
            r.ok(f"footprint {code!r} -> {name}:{want_kind}"
                 + (f"({want_reach})" if want_kind == "halo" else ""))
        except Exception as e:
            r.fail(f"ROI-2 footprint {code!r}", f"{type(e).__name__}: {e}")

    # $param folding: a widget sigma resolves the halo; a symbolic (wired) one → not narrow.
    try:
        assert tex_roi.binding_footprints("@OUT = gauss_blur(@A, $s);", {"s": 2.0})["A"].reach == 6
        assert tex_roi.binding_footprints("@OUT = gauss_blur(@A, $s);", {})["A"].kind == "image"
        r.ok("ROI-2 folds a widget radius; a symbolic radius conservatively → image")
    except Exception as e:
        r.fail("ROI-2 param folding", f"{type(e).__name__}: {e}")


def test_roi2_plan_executability(r: SubTestResult):
    print("\n--- ROI-2: roi_plan executability (whitelist posture) ---")
    #  code, params, want_executable, want_halo
    cases = [
        ("@OUT = @A * 0.5 + vec4(u, v, 0.0, 0.0);", {}, True, 0),
        ("@OUT = gauss_blur(@A, 2.0);", {}, True, 6),
        ("@OUT = erode(@A, 3.0) + dilate(@A, 1.0);", {}, True, 3),   # max reach
        ("@OUT = gauss_blur(@A, $s);", {"s": 1.0}, True, 3),
        # SAFE multi-statement forms that must NOT be over-blocked by the halo-through-name
        # guard: a name holding an INPUT (no halo) fed to a blur, a multi-output program, and a
        # pointwise setup line before a blur all stay executable with the correct halo.
        ("vec4 x = @A;\n@OUT = gauss_blur(x, 2.0);", {}, True, 6),
        ("@OUT = gauss_blur(@A, 2.0);\n@MASK = @A * 0.5;", {}, True, 6),
        ("@A2 = @A * 0.5;\n@OUT = gauss_blur(@A2, 2.0);", {}, True, 6),
        # a SINGLE grounded blur inside an `if` composes fine (halo 6 covers both branches) —
        # the third-audit recursion must not over-block it while catching the nested CHAIN.
        ("if (u > 0.5) { @OUT = gauss_blur(@A, 2.0); } else { @OUT = @A; }", {}, True, 6),
        ("@OUT = gauss_blur(@A, $s);", {}, False, 0),                # symbolic radius blocks
        ("@OUT = sample(@A, u, v);", {}, False, 0),                  # gather → whole-frame
        ("@OUT = @A * img_mean(@A);", {}, False, 0),                 # reduction → whole-frame
        ("@OUT = fetch(@A, ix + 2, iy);", {}, False, 0),             # gather → whole-frame
        ("@OUT[ix, iy] = vec4(u, v, 0.0, 1.0);", {}, False, 0),      # scatter → whole-frame
    ]
    for code, params, want_exec, want_halo in cases:
        try:
            plan = tex_roi.roi_plan(code, params)
            assert plan.executable == want_exec, f"executable {plan.executable} != {want_exec}"
            if want_exec:
                assert plan.halo == want_halo, f"halo {plan.halo} != {want_halo}"
            r.ok(f"plan {code!r} -> exec={want_exec}" + (f" halo={want_halo}" if want_exec else ""))
        except Exception as e:
            r.fail(f"ROI-2 plan {code!r}", f"{type(e).__name__}: {e}")


# ── ROI-4 part 1: the reach-pinning derivation test ───────────────────────────

def test_roi4_reach_pinning(r: SubTestResult):
    print("\n--- ROI-4: declared halo reach >= the impl's actual neighbour reach ---")
    # Perturb ONE pixel of an input; a halo op's output can change only within its kernel
    # radius of that pixel. The measured spread must be <= the reach the descriptor declares
    # (the reach ROI-3 narrows by). A wrong multiplier (gauss's 3x) makes measured > declared.
    B, H, W, C = 1, 41, 41, 4
    cy, cx = H // 2, W // 2
    #  fn,           extra-args,   builds a program on @A of shape [B,H,W,C]
    fns = [
        ("gauss_blur", [1.5]),
        ("gauss_blur", [2.5]),
        ("erode", [3.0]),
        ("dilate", [2.0]),
        ("bilateral_filter", [1.5, 0.2]),
    ]
    for dev in _DEVICES:
        for name, extra in fns:
            try:
                torch.manual_seed(3)
                base = torch.rand(B, H, W, C, device=dev)
                pert = base.clone()
                pert[:, cy, cx, :] += 0.5   # perturb the centre pixel
                args = ", ".join(str(float(a)) for a in extra)
                code = f"@OUT = {name}(@A, {args});"
                prog, tm, outs = _compile(code, {"A": TEXType.VEC4})
                interp = Interpreter()
                ob = interp.execute(prog, {"A": base}, tm, device=dev, output_names=outs)["OUT"]
                op = interp.execute(prog, {"A": pert}, tm, device=dev, output_names=outs)["OUT"]
                diff = (ob - op).abs().sum(dim=-1)[0]        # [H,W] where the output changed
                nz = (diff > 1e-6).nonzero(as_tuple=False)
                if nz.numel() == 0:
                    measured = 0
                else:
                    dy = (nz[:, 0] - cy).abs().max().item()
                    dx = (nz[:, 1] - cx).abs().max().item()
                    measured = max(dy, dx)
                declared = _declared_reach(name, extra)
                assert isinstance(declared, int), f"{name} declared reach not an int: {declared}"
                assert measured <= declared, (
                    f"{name}{extra}: measured reach {measured} > declared {declared} "
                    f"(narrowing would under-pad → seams)")
                r.ok(f"reach[{dev}] {name}{extra}: measured {measured} <= declared {declared}")
            except Exception as e:
                r.fail(f"ROI-4 reach {name}{extra} [{dev}]", f"{type(e).__name__}: {e}")


# ── ROI-4 part 2: spatial never-sever rows ────────────────────────────────────

def test_roi4_never_sever(r: SubTestResult):
    print("\n--- ROI-4: whitelist posture — unknown never shrinks to an ROI ---")
    # Mirrors test_lazy_cooking's never-sever rows: a program the analysis can't prove
    # sub-region-safe must fall to whole-frame (executable=False), never cook a shrunk ROI.
    must_fallback = [
        ("@OUT = sample(@A, u, v);", {}, "a data-dependent gather"),
        ("@OUT = sample(@A, u + 0.01, v);", {}, "an affine gather (absolute coords)"),
        ("@OUT = fetch(@A, ix + 3, iy);", {}, "an affine fetch"),
        ("@OUT = vec4(vec3(img_mean(@A)), 1.0);", {}, "a whole-image reduction"),
        ("@OUT = @A * img_max(@A);", {}, "pointwise × reduction"),
        ("@OUT = gauss_blur(@A, $wired);", {}, "a symbolic (wired) blur radius"),
        ("@OUT[ix, iy] = @A;", {}, "a scatter write"),
        ("@OUT[ix, iy].r = @A.r;", {}, "a channel-wrapped scatter target"),
        ("@OUT = @A;\n@OUT[ix, iy].rgb = vec3(0.0);", {}, "a swizzle-wrapped scatter target"),
        ("@OUT = img_mean();", {}, "a zero-arg reduction (degenerate)"),
        ("@OUT = sample_mip(@A, u, v, 2.0);", {}, "a mip gather"),
        ("@OUT = @A[ix, iy];", {}, "an index-access gather"),
        # bug-hunt classes: a gather/reduction the analysis must catch through indirection —
        # a local-variable alias, a user-function parameter, or a bindless (generated) image.
        ("vec4 x = @A;\n@OUT = sample(x, u + 0.2, v);", {}, "a gather through a local alias"),
        ("vec4 f(vec4 im){return sample(im, u + 0.2, v);}\n@OUT = f(@A);", {},
         "a gather through a user-function parameter"),
        ("vec4 x = @A;\n@OUT = x / (img_max(x) + 0.01);", {}, "a reduction through a local alias"),
        ("@OUT = sample(vec4(u, v, 0.0, 1.0), u, v);", {}, "a gather over a bindless image"),
        ("@OUT = @A * img_mean(vec4(u, v, 0.0, 1.0));", {}, "a reduction over a bindless image"),
        # a halo-op chain split across a local variable — the reach can't compose across the
        # statement boundary, so the cook halo would be under-sized (ROI-edge contamination).
        ("vec4 b = gauss_blur(@A, 2.0);\n@OUT = gauss_blur(b, 2.0);", {}, "a split blur chain (VarDecl init)"),
        ("vec4 e = erode(@A, 3.0);\n@OUT = dilate(e, 3.0);", {}, "a split morphology chain (VarDecl init)"),
        # second-audit: a halo result flowing through a name via the OTHER spellings —
        # a bare-decl+assignment, a reassignment, and an intermediate @binding output.
        ("vec4 t;\nt = gauss_blur(@A, 2.0);\n@OUT = gauss_blur(t, 2.0);", {}, "a blur chain through a bare assignment"),
        ("vec4 x = @A;\nx = gauss_blur(x, 2.0);\n@OUT = gauss_blur(x, 2.0);", {}, "a blur chain through a reassigned local"),
        ("@T = gauss_blur(@A, 2.0);\n@OUT = gauss_blur(@T, 2.0);", {}, "a blur chain through an intermediate @binding"),
        ("vec4 b = gauss_blur(@A, 2.0);\n@OUT = b * 0.5;", {}, "a blur result stored in a local, then read"),
        # third-audit: a halo chain through a name NESTED in an `if` body — the halo-named
        # collection must recurse into blocks (case-1 does not treat `if` as a boundary, since a
        # single grounded blur in a branch composes fine), or the ±12 chain is under-cooked at ±6.
        ("if (u > 0.5) { @T = gauss_blur(@A, 2.0); @OUT = gauss_blur(@T, 2.0); } else { @OUT = @A; }",
         {}, "a blur chain through a name nested in an if-block"),
    ]
    for code, params, why in must_fallback:
        try:
            assert not tex_roi.roi_plan(code, params).executable, \
                f"analysis wrongly ROI-executed {why}"
            r.ok(f"never-sever: {why} → whole-frame")
        except Exception as e:
            r.fail(f"ROI-4 never-sever ({why})", f"{type(e).__name__}: {e}")

    # A LARGER-than-needed footprint is fine (a missed optimisation); a SHRUNK one never.
    # gauss_blur analysed with a conservative reach is safe; we assert reach is never < true.
    try:
        # reach declared for sigma=0.4 is ceil(3*0.4)=2; the impl radius is also ceil(1.2)=2.
        assert _declared_reach("gauss_blur", [0.4]) >= 2
        r.ok("never-sever: a conservative (>=) reach is the only allowed error direction")
    except Exception as e:
        r.fail("ROI-4 conservative reach", f"{type(e).__name__}: {e}")


# ── ROI-4 part 3: tile ≡ roi, and the differential oracle ─────────────────────

def test_roi3_tile_is_roi_special_case(r: SubTestResult):
    print("\n--- ROI-3: tile=(y0,H_total) ≡ roi=(0,y0,W,H,W,H_total) (seam-exact) ---")
    # The strip path must be byte-identical after normalizing tile→roi (invariant #7): the
    # M-4 machinery is heavily used and must not have moved.
    code = "@OUT = vec4(u, v, u * v, 1.0);"   # coordinate-only, so any drift shows immediately
    for dev in _DEVICES:
        try:
            B, H, W = 1, 20, 16
            prog, tm, outs = _compile(code, {})
            interp = Interpreter()
            # spatial context comes from a dummy binding of the strip's shape.
            y0, Hstrip, Htot = 6, 8, 20
            strip = torch.zeros(B, Hstrip, W, 4, device=dev)
            via_tile = interp.execute(prog, {"A": strip}, tm, device=dev, output_names=outs,
                                      tile=(y0, Htot))["OUT"]
            via_roi = interp.execute(prog, {"A": strip}, tm, device=dev, output_names=outs,
                                     roi=(0, y0, W, Hstrip, W, Htot))["OUT"]
            assert torch.equal(via_tile, via_roi), "tile and its roi form diverged"
            # Not tautological: also pin the coordinates against an INDEPENDENT literal grid
            # (u=ix/(W-1), v=iy/(H_total-1) with iy offset by y0), so a coordinate regression
            # vs v0.23 — not just tile/roi agreeing with each other — reds here.
            iy = (torch.arange(y0, y0 + Hstrip, dtype=torch.float32, device=dev).view(1, Hstrip, 1))
            ix = torch.arange(W, dtype=torch.float32, device=dev).view(1, 1, W)
            u = (ix / (W - 1)).expand(1, Hstrip, W)
            v = (iy / (Htot - 1)).expand(1, Hstrip, W)
            want = torch.stack([u, v, u * v, torch.ones_like(u)], dim=-1)
            assert torch.equal(via_roi, want), "roi coordinates diverged from the literal grid"
            r.ok(f"tile≡roi[{dev}]: coordinates byte-identical AND match the literal grid")
        except Exception as e:
            r.fail(f"ROI-3 tile≡roi [{dev}]", f"{type(e).__name__}: {e}")


def _gen_expr(rng, depth):
    """A random vec4 expression in the ROI-executable class (pointwise + blur/morphology),
    type-safe by construction (every leaf and node is vec4)."""
    if depth <= 0:
        return rng.choice(["@A", "@B", "vec4(u, v, 0.5, 1.0)", "vec4(0.3, 0.6, 0.2, 1.0)"])
    a = _gen_expr(rng, depth - 1)
    pick = rng.random()
    if pick < 0.30:
        b = _gen_expr(rng, depth - 1)
        c1 = round(rng.uniform(0.1, 0.9), 3)
        return f"({a} * {c1} + {b} * {round(1.0 - c1, 3)})"
    if pick < 0.48:
        return f"clamp({a}, 0.0, 1.0)"
    if pick < 0.64:
        b = _gen_expr(rng, depth - 1)
        return f"mix({a}, {b}, {round(rng.uniform(0.0, 1.0), 3)})"
    if pick < 0.78:
        return f"gauss_blur({a}, {round(rng.uniform(0.4, 1.8), 3)})"   # small halos
    if pick < 0.86:
        return f"erode({a}, {rng.randint(1, 3)}.0)"
    if pick < 0.94:
        return f"dilate({a}, {rng.randint(1, 3)}.0)"
    # bilateral_filter: the 4th whitelist op, footprint ('halo', 3). Named in the oracle's
    # 'conv/bilateral ~1 ulp' tolerance calibration but previously never generated.
    return f"bilateral_filter({a}, {round(rng.uniform(0.4, 1.4), 3)}, {round(rng.uniform(0.1, 0.4), 3)})"


def _rand_rois(rng, W, H, n):
    out = [(0, 0, W, H, W, H)]   # always include the whole frame
    for _ in range(n):
        w = rng.randint(2, W); h = rng.randint(2, H)
        x0 = rng.randint(0, W - w); y0 = rng.randint(0, H - h)
        out.append((x0, y0, w, h, W, H))
    return out


def test_roi4_differential_oracle(r: SubTestResult):
    print("\n--- ROI-4: differential oracle — ROI cook == whole-frame cook ---")
    import os, random
    seed = int(os.environ.get("TEX_ROI_FUZZ_SEED", "20260718"))
    N = int(os.environ.get("TEX_ROI_FUZZ_N", "40"))
    rng = random.Random(seed)
    B, H, W = 1, 28, 36
    btypes = {"A": TEXType.VEC4, "B": TEXType.VEC4}
    worst = 0.0
    fails = 0
    checked = 0
    for dev in _DEVICES:
        for i in range(N):
            # Every program carries a partial-broadcast COMPANION output (@ROW=iy/ih, a
            # [B,H,1]) and, on some, a pointwise SETUP local — so the fuzz lane exercises
            # multi-output per-dim crop AND multi-statement dataflow, not just @OUT=<expr>
            # (the coverage gap that let the round-2 bugs through).
            expr = _gen_expr(rng, rng.randint(1, 3))
            if rng.random() < 0.5:
                code = (f"vec4 s = @A * {round(rng.uniform(0.2, 0.9), 2)};\n"
                        f"@OUT = {expr.replace('@A', 's')};\n@ROW = iy / ih;")
            else:
                code = f"@OUT = {expr};\n@ROW = iy / ih;"
            try:
                plan = tex_roi.roi_plan(code, {})
                if not plan.executable:
                    continue   # generator can emit whole-frame programs; skip (oracle covers exec only)
                prog, tm, outs = _compile(code, btypes)
                torch.manual_seed(1000 + i)
                binds = {"A": torch.rand(B, H, W, 4, device=dev),
                         "B": torch.rand(B, H, W, 4, device=dev)}
                interp = Interpreter()
                full = interp.execute(prog, dict(binds), tm, device=dev, output_names=outs)
                for roi in _rand_rois(rng, W, H, 4):
                    got = run_roi(interp, prog, dict(binds), tm, dev, 0, outs, None, "fp32",
                                  roi, plan.narrow, plan.halo)
                    checked += 1
                    for n in outs:                       # per-dim crop over EVERY output
                        ref = _crop_ref(full[n], roi)
                        if got[n].shape != ref.shape:
                            fails += 1
                            if fails <= 5:
                                r.fail("ROI-4 oracle shape",
                                       f"[{dev}] {code!r} out={n} roi={roi}: "
                                       f"{tuple(got[n].shape)} != {tuple(ref.shape)}")
                            continue
                        md = (got[n].float() - ref.float()).abs().max().item()
                        worst = max(worst, md)
                        if md >= _TOL:
                            fails += 1
                            if fails <= 5:
                                r.fail("ROI-4 oracle maxdiff",
                                       f"[{dev}] {code!r} out={n} roi={roi}: maxdiff {md:.3e}")
            except Exception as e:
                fails += 1
                if fails <= 5:
                    r.fail("ROI-4 oracle exception", f"[{dev}] {code!r}: {type(e).__name__}: {e}")
    if fails == 0:
        r.ok(f"differential oracle: {checked} (program × ROI) cooks match whole-frame "
             f"(worst maxdiff {worst:.2e} < {_TOL:.0e}) over seed {seed}")
    else:
        r.fail("ROI-4 oracle summary", f"{fails} (program × ROI × output) mismatches (seed {seed})")


def test_roi4_partition_assembly(r: SubTestResult):
    print("\n--- ROI-4: assembling a partition of ROIs reproduces the whole frame ---")
    # The real use: tile the viewport into ROIs, cook each, stitch — no seams, no gaps.
    B, H, W = 1, 24, 30
    programs = [
        "@OUT = @A * 0.7 + vec4(u, v, 0.0, 0.0);",
        "@OUT = gauss_blur(@A, 1.5);",
        "@OUT = clamp(dilate(@A, 2.0) * 0.5 + erode(@A, 1.0) * 0.5, 0.0, 1.0);",
    ]
    for dev in _DEVICES:
        for code in programs:
            try:
                plan = tex_roi.roi_plan(code, {})
                prog, tm, outs = _compile(code, {"A": TEXType.VEC4})
                torch.manual_seed(7)
                A = torch.rand(B, H, W, 4, device=dev)
                interp = Interpreter()
                full = interp.execute(prog, {"A": A}, tm, device=dev, output_names=outs)["OUT"]
                buf = torch.zeros_like(full)
                # a 3×2 grid of ROIs partitioning the frame (irregular splits)
                xs = [0, 11, 20, W]
                ys = [0, 13, H]
                for yi in range(len(ys) - 1):
                    for xi in range(len(xs) - 1):
                        x0, y0 = xs[xi], ys[yi]
                        w, h = xs[xi + 1] - x0, ys[yi + 1] - y0
                        got = run_roi(interp, prog, {"A": A}, tm, dev, 0, outs, None, "fp32",
                                      (x0, y0, w, h, W, H), plan.narrow, plan.halo)["OUT"]
                        buf[:, y0:y0 + h, x0:x0 + w] = got
                md = (buf.float() - full.float()).abs().max().item()
                assert md < _TOL, f"assembled partition maxdiff {md:.3e} >= {_TOL:.0e}"
                r.ok(f"partition[{dev}]: {code[:34]}… assembles to whole (maxdiff {md:.1e})")
            except Exception as e:
                r.fail(f"ROI-4 partition [{dev}] {code[:30]}", f"{type(e).__name__}: {e}")


def _crop_ref(full, roi):
    """Crop a whole-frame output to the ROI PER SPATIAL DIM (only a full-size dim is cropped;
    a broadcast size-1 dim passes through) — the reference run_roi must match."""
    x0, y0, w, h, W, H = roi
    if isinstance(full, torch.Tensor) and full.dim() >= 3:
        if full.shape[1] == H:
            full = full[:, y0:y0 + h]
        if full.shape[2] == W:
            full = full[:, :, x0:x0 + w]
    return full


def test_roi4_partial_broadcast_crop(r: SubTestResult):
    print("\n--- ROI-4: run_roi crops each spatial dim independently (multi-output) ---")
    # BUG the oracle was blind to: an output full in one spatial dim and broadcast (size 1) in
    # the other (a gradient row/col companion, a broadcast strip passthrough) must be cropped
    # on its full dim and left intact on its broadcast dim — not returned at cook-region extent.
    B, H, W = 1, 24, 32
    Bcast = torch.rand(B, 1, W, 4)   # height-broadcast input
    progs = [
        ("@OUT = gauss_blur(@A, 2.0);\n@ROW = iy / ih;", {"A": TEXType.VEC4}),
        ("@OUT = gauss_blur(@A, 2.0);\n@COL = ix / iw;", {"A": TEXType.VEC4}),
        ("@OUT = erode(@A, 2.0);\n@ROW = iy / ih;", {"A": TEXType.VEC4}),
        ("@OUT = gauss_blur(@A, 1.0) + @B * 0.0;\n@P = @B;", {"A": TEXType.VEC4, "B": TEXType.VEC4}),
    ]
    rois = [(10, 8, 6, 5, W, H), (0, 0, W, H, W, H), (W - 4, H - 3, 4, 3, W, H), (5, 6, 20, 10, W, H)]
    for dev in _DEVICES:
        for code, bt in progs:
            try:
                plan = tex_roi.roi_plan(code, {})
                assert plan.executable, f"expected executable: {code}"
                prog, tm, outs = _compile(code, bt)
                torch.manual_seed(9)
                binds = {"A": torch.rand(B, H, W, 4, device=dev)}
                if "B" in bt:
                    binds["B"] = Bcast.to(dev)
                interp = Interpreter()
                full = interp.execute(prog, dict(binds), tm, device=dev, output_names=outs)
                worst = 0.0
                for roi in rois:
                    got = run_roi(interp, prog, dict(binds), tm, dev, 0, outs, None, "fp32",
                                  roi, plan.narrow, plan.halo)
                    for n in outs:
                        ref = _crop_ref(full[n], roi)
                        assert got[n].shape == ref.shape, \
                            f"{n} roi={roi}: {tuple(got[n].shape)} != {tuple(ref.shape)}"
                        worst = max(worst, (got[n].float() - ref.float()).abs().max().item())
                assert worst < _TOL, f"maxdiff {worst:.3e}"
                r.ok(f"per-dim crop[{dev}]: {code.splitlines()[1].strip()} (maxdiff {worst:.1e})")
            except Exception as e:
                r.fail(f"ROI-4 partial-broadcast [{dev}] {code[:30]}", f"{type(e).__name__}: {e}")


def test_roi3_engine_integration(r: SubTestResult):
    print("\n--- ROI-3: the engine cook(roi=) path (flag gate + fp32 clamp + fallback) ---")
    # The ship gate must certify the path production would actually run: cook() -> prepare()
    # -> _run_default, the four-part gate, scalar_params extraction, ExecContext.roi threading,
    # and the fp32 clamp. Everything else drives run_roi/roi_plan directly.
    import os as _os
    from TEX_Wrangle import tex_engine, tex_roi as _R
    W, H = 32, 24
    roi = (8, 6, 10, 9, W, H)
    x0, y0, w, h, _W, _H = roi
    prev = _os.environ.get("TEX_ROI_EXEC")

    def _full(code, binds, **kw):
        return tex_engine.cook(code, dict(binds), device_mode="cpu", **kw).outputs["OUT"]

    def _roi(code, binds, **kw):
        return tex_engine.cook(code, dict(binds), device_mode="cpu", roi=roi, **kw).outputs["OUT"]

    try:
        torch.manual_seed(11)
        A = torch.rand(1, H, W, 4)
        # (1) flag OFF (default): roi= is ignored → whole-frame output shape.
        _os.environ["TEX_ROI_EXEC"] = "0"; _R.clear_roi_memo()
        off = _roi("@OUT = gauss_blur(@A, 2.0);", {"A": A})
        r.ok("engine: flag OFF → roi ignored (whole frame)") if tuple(off.shape) == (1, H, W, 4) \
            else r.fail("engine flag OFF", f"shape {tuple(off.shape)} != whole")

        # (2) flag ON + executable → ROI-sized output matching the whole-frame crop.
        _os.environ["TEX_ROI_EXEC"] = "1"; _R.clear_roi_memo()
        for code in ("@OUT = gauss_blur(@A, 2.0);", "@OUT = @A * 0.5 + vec4(u, v, 0.0, 0.0);"):
            full = _full(code, {"A": A}); got = _roi(code, {"A": A})
            md = (full[:, y0:y0 + h, x0:x0 + w].float() - got.float()).abs().max().item() \
                if tuple(got.shape) == (1, h, w, 4) else 9.9
            r.ok(f"engine: flag ON executable → ROI cook ({code[:22]}… maxdiff {md:.1e})") \
                if md < _TOL else r.fail("engine ON executable", f"{code}: shape {tuple(got.shape)} maxdiff {md:.2e}")

        # (3) flag ON + NOT executable (a gather) → whole-frame (roi ignored, no crash).
        ng = _roi("@OUT = sample(@A, u, v);", {"A": A})
        r.ok("engine: flag ON non-executable (gather) → whole frame") \
            if tuple(ng.shape) == (1, H, W, 4) else r.fail("engine ON non-exec", f"shape {tuple(ng.shape)}")

        # (4) the ROI path is clamped to fp32 even when precision=fp16 (the ~1-ulp conv slack
        #     would scale at fp16, and the oracle only validates fp32).
        code = "@OUT = gauss_blur(@A, 2.0);"
        full = _full(code, {"A": A})
        got16 = _roi(code, {"A": A}, precision="fp16")
        md = (full[:, y0:y0 + h, x0:x0 + w].float() - got16.float()).abs().max().item() \
            if tuple(got16.shape) == (1, h, w, 4) else 9.9
        r.ok(f"engine: precision=fp16 ROI cook clamped to fp32 (maxdiff {md:.1e})") \
            if md < _TOL else r.fail("engine fp16 clamp", f"shape {tuple(got16.shape)} maxdiff {md:.2e}")
    except Exception as e:
        r.fail("ROI-3 engine integration", f"{type(e).__name__}: {e}")
    finally:
        if prev is None:
            _os.environ.pop("TEX_ROI_EXEC", None)
        else:
            _os.environ["TEX_ROI_EXEC"] = prev
        _R.clear_roi_memo()


# ── ROI-6: temporal laziness groundwork ───────────────────────────────────────

def test_roi6_frame_window(r: SubTestResult):
    print("\n--- ROI-6: frame-window analysis + batch sliceability ---")
    #  code, want_window, want_sliceable.  A frame op is an ABSOLUTE-index gather into the
    #  batch, so ANY frame op — even one reading the current frame (offset 0) — is unsafe under
    #  a dim-0 strip (global fi clamps against the strip, not B_total): NOT sliceable in v1.
    cases = [
        ("@OUT = @A * 0.5;", (0, 0), True),                              # pure per-frame
        ("@OUT = @A * (fi / (fn - 1.0));", (0, 0), True),                # fi builtin — per-frame
        ("@OUT = gauss_blur(@A, 1.5);", (0, 0), True),                   # spatial, still per-frame
        ("@OUT = @A[ix, iy];", (0, 0), True),                           # 2-arg spatial fetch, per-frame
        ("@OUT = fetch_frame(@A, fi, ix, iy);", (0, 0), False),          # frame op, own frame — NOT sliceable
        ("@OUT = sample_frame(@A, fi, u, v);", (0, 0), False),          # frame op, own frame — NOT sliceable
        ("@OUT = fetch_frame(@A, fi - 1, ix, iy);", (-1, 0), False),     # reads the previous frame
        ("@OUT = sample_frame(@A, fi + 1, u, v);", (0, 1), False),       # reads the next frame
        ("@OUT = @A[ix, iy, fi - 1];", (-1, 0), False),                 # 3-arg index SUGAR (frame last)
        ("@OUT = @A(u, v, fi + 1);", (0, 1), False),                    # 3-arg sample SUGAR
        ("@OUT = @A * 0.5 + fetch_frame(@A, fi - 2, ix, iy) * 0.5;", (-2, 0), False),
        ("@OUT = fetch_frame(@A, 0.0, ix, iy);", None, False),           # a FIXED frame → whole batch
    ]
    for code, want_win, want_slice in cases:
        try:
            win = tex_roi.frame_window(code, {})
            slic = tex_roi.batch_sliceable(code, {})
            assert win == want_win, f"window {win} != {want_win}"
            assert slic == want_slice, f"sliceable {slic} != {want_slice}"
            r.ok(f"frame_window {code!r} -> {want_win}, sliceable={want_slice}")
        except Exception as e:
            r.fail(f"ROI-6 frame_window {code!r}", f"{type(e).__name__}: {e}")


def test_roi6_batch_strip_equivalence(r: SubTestResult):
    print("\n--- ROI-6: batch-strip cook == whole-batch cook (seam-exact fi/fn) ---")
    from TEX_Wrangle.tex_memory import run_batch_strips
    Bn, H, W = 6, 12, 10
    programs = [
        "@OUT = @A * 0.5 + vec4(u, v, 0.0, 0.0);",          # per-frame, no fi
        "@OUT = @A * (fi / (fn - 1.0));",                   # temporal fade — fi/fn must be seam-exact
        "@OUT = vec4(fi / (fn - 1.0), v, 0.0, 1.0);",       # fi-only generative
        "@OUT = gauss_blur(@A, 1.5) * (fi / fn);",          # spatial × temporal
    ]
    for dev in _DEVICES:
        for code in programs:
            try:
                assert tex_roi.batch_sliceable(code, {}), f"expected sliceable: {code}"
                prog, tm, outs = _compile(code, {"A": TEXType.VEC4})
                torch.manual_seed(5)
                A = torch.rand(Bn, H, W, 4, device=dev)
                interp = Interpreter()
                whole = interp.execute(prog, {"A": A}, tm, device=dev, output_names=outs)["OUT"]
                for n_strips in (2, 3, 4):
                    got = run_batch_strips(interp, prog, {"A": A}, tm, dev, 0, outs, None,
                                           "fp32", n_strips)["OUT"]
                    md = (got.float() - whole.float()).abs().max().item()
                    assert got.shape == whole.shape, f"shape {tuple(got.shape)} != {tuple(whole.shape)}"
                    assert md < _TOL, f"n_strips={n_strips} maxdiff {md:.3e} >= {_TOL:.0e}"
                r.ok(f"batch-strip[{dev}]: {code[:38]}… == whole batch (2/3/4 strips)")
            except Exception as e:
                r.fail(f"ROI-6 batch-strip [{dev}] {code[:30]}", f"{type(e).__name__}: {e}")

        # Multi-output with a batch-BROADCAST companion. Regression: the old stitch keyed on
        # shape[0]==(f1-f0), so a [1,...] passthrough output was left uninitialized on the
        # multi-frame strips (mixed strip sizes) and wrongly replicated to [B_total,...] when
        # every strip was size 1. The companion must come back [1,...] == the whole-batch cook.
        mo = "@OUT = @A;\n@AUX = @B;"
        try:
            assert tex_roi.batch_sliceable(mo, {}), "expected sliceable (no frame op)"
            prog, tm, outs = _compile(mo, {"A": TEXType.VEC4, "B": TEXType.VEC4})
            torch.manual_seed(7)
            A = torch.rand(Bn, H, W, 4, device=dev)
            Bbc = torch.rand(1, H, W, 4, device=dev)          # batch-broadcast companion input
            interp = Interpreter()
            whole = interp.execute(prog, {"A": A, "B": Bbc}, tm, device=dev, output_names=outs)
            for n_strips in (2, 3, 4, Bn):                     # incl. mixed sizes (4) and all-size-1 (Bn)
                got = run_batch_strips(interp, prog, {"A": A, "B": Bbc}, tm, dev, 0, outs,
                                       None, "fp32", n_strips)
                for nm in outs:
                    assert got[nm].shape == whole[nm].shape, \
                        f"{nm} n_strips={n_strips} shape {tuple(got[nm].shape)} != {tuple(whole[nm].shape)}"
                    md = (got[nm].float() - whole[nm].float()).abs().max().item()
                    assert md < _TOL, f"{nm} n_strips={n_strips} maxdiff {md:.3e}"
            r.ok(f"batch-strip[{dev}]: multi-output + batch-broadcast companion == whole")
        except Exception as e:
            r.fail(f"ROI-6 batch-strip multi-output [{dev}]", f"{type(e).__name__}: {e}")


def test_roi6_fi_seam_exact(r: SubTestResult):
    print("\n--- ROI-6: fi starts at f0, fn reports B_total under a batch strip ---")
    # The seam-exactness pin: a single strip's fi values must equal the corresponding slice of
    # the whole-batch fi (the temporal twin of tile≡roi for iy).
    for dev in _DEVICES:
        try:
            Bn, H, W = 5, 4, 4
            code = "@OUT = vec4(fi, fn, 0.0, 1.0);"   # expose fi/fn directly
            prog, tm, outs = _compile(code, {"A": TEXType.VEC4})
            A = torch.zeros(Bn, H, W, 4, device=dev)
            interp = Interpreter()
            whole = interp.execute(prog, {"A": A}, tm, device=dev, output_names=outs)["OUT"]
            # cook frames [2,5) as a strip of a batch of 5
            strip = interp.execute(prog, {"A": A.narrow(0, 2, 3)}, tm, device=dev,
                                   output_names=outs, batch_slice=(2, 5))["OUT"]
            assert torch.equal(strip, whole[2:5]), "strip fi/fn diverged from the whole-batch slice"
            # fi is the absolute frame index; fn is the full batch count in every strip
            assert float(strip[0, 0, 0, 0]) == 2.0 and float(strip[-1, 0, 0, 0]) == 4.0
            assert float(strip[0, 0, 0, 1]) == 5.0
            r.ok(f"fi/fn seam-exact[{dev}]: strip fi=[2,4], fn=5 matches whole batch")
        except Exception as e:
            r.fail(f"ROI-6 fi seam-exact [{dev}]", f"{type(e).__name__}: {e}")
