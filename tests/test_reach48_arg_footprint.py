"""
REACH-48 — per-argument reach declarations for a multi-image-argument stdlib builtin.

TIERS-48-design.md SS B.2 point 2 names the gap directly: "a registry entry whose reach
differs per-input-argument and where MULTIPLE arguments are themselves images... is a
genuinely new registry shape this inventory did not find an existing precedent for." The
registry's `footprint=` field (ROI-1) describes only arg 0's (the image's) reach; every
OTHER argument was walked by `tex_roi._accumulate` at the CALLER'S outer context
(pointwise/narrow) regardless of what it actually was — silently correct for a scalar
radius/flag, silently WRONG the moment a second argument is itself a full image/array
binding with its own reach.

The census this lane ran (AGENTS.md's stdlib registry, every `@stdlib(...)` entry):
`convolve(image, kernel[, normalize])` is the ONLY registered builtin with a genuine
second image/array argument today (`kernel`, a bound [Bk,kH,kW,Ck] tensor read whole,
per its own docstring in stdlib_sample.py) — `apply_lut3d`'s `lut` is explicitly a
`non_spatial_args` RESOURCE, not a per-pixel-addressed image, and every other non-'point'
builtin (`erode`/`dilate`/`gauss_blur`/`bilateral_filter`/`patch_dist`) has exactly one
image argument plus plain scalars. Per-arg reach declared this lane:
  * `convolve` arg 1 (`kernel`) = `'image'` (read whole, independent of arg 0's own
    reach and of ROI narrowing — matches its documented true semantics; this is an
    EXPRESSIVENESS fix, not a behaviour change to `roi_plan`'s executability, since
    `convolve` already blocks ROI via arg 0's `footprint='image'`).

This closes the other half of the `DEVELOPMENT.md` §"Rejected design decisions" entry
for `convolve`'s footprint (ASK-1, v0.35), which named exactly this reopening condition:
"a 'this argument is read whole' descriptor in tex_roi (a fourth footprint field) ...
widens invariant #5's vocabulary."
"""
from helpers import *
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib  # populates REGISTRY
from TEX_Wrangle.tex_runtime.stdlib_core import _to_tensor
from TEX_Wrangle.tex_runtime import stdlib_registry as R
from TEX_Wrangle import tex_roi as _R


def test_reach48_convolve_kernel_arg_declared(r: SubTestResult):
    print("\n--- REACH-48: convolve's kernel argument carries its own declared reach ---")
    try:
        by_name = {n: e for e in R.REGISTRY for n in e.names}
        e = by_name["convolve"]
        assert dict(e.arg_footprint) == {1: "image"}, (
            f"convolve.arg_footprint = {dict(e.arg_footprint)!r}, want {{1: 'image'}}")
        r.ok("convolve declares arg_footprint={1: 'image'} for its kernel argument")
    except Exception as ex:
        r.fail("REACH-48 convolve declaration", f"{type(ex).__name__}: {ex}")

    # The single-source cache both `arg_footprint_by_name()` and `tex_roi._arg_footmap()`
    # read agrees with the registry entry (REG-1c discipline, same as pixel_args/
    # non_spatial_args' own caches).
    try:
        cache = R.arg_footprint_by_name()
        assert cache.get("convolve") == {1: "image"}, (
            f"arg_footprint_by_name()['convolve'] = {cache.get('convolve')!r}")
        r.ok("arg_footprint_by_name() single-sources the same declaration")
    except Exception as ex:
        r.fail("REACH-48 cache", f"{type(ex).__name__}: {ex}")


def test_reach48_loud_guard_clean(r: SubTestResult):
    print("\n--- REACH-48: no unclassified multi-image-argument candidate in the registry ---")
    try:
        cand = R.unclassified_image_arg_candidates()
        assert not cand, (f"unclassified multi-image-argument-looking fns: {cand} — declare "
                          f"arg_footprint in stdlib_registry or confirm it's a scalar")
        r.ok("unclassified_image_arg_candidates() is empty")
    except Exception as ex:
        r.fail("REACH-48 loud guard clean", f"{type(ex).__name__}: {ex}")

    # The guard is not vacuous: it DOES fire for a probe fn shaped like convolve's kernel
    # idiom (`_to_tensor(name)` feeding a `.dim(` rank check on a non-first, undeclared
    # argument), and stops firing the moment that argument gets an arg_footprint
    # declaration — proven by direct REGISTRY manipulation (append/pop), never left
    # registered either way.
    def _probe_second_is_tensor(image, second):
        # Mirrors convolve's own idiom literally, so the source-scanning guard matches it.
        ker = second if second.__class__ is torch.Tensor else _to_tensor(second)
        if ker.dim() != 4:
            raise ValueError("bad")
        return image

    try:
        before = R.unclassified_image_arg_candidates()
        assert "_reach48_probe" not in before
        entry_undeclared = R.StdlibEntry(
            "_reach48_probe", _probe_second_is_tensor, footprint="point")
        R.REGISTRY.append(entry_undeclared)
        try:
            fired = R.unclassified_image_arg_candidates()
            assert "_reach48_probe" in fired, (
                f"loud guard did not flag an undeclared probe shaped like convolve's kernel "
                f"idiom: {fired}")
        finally:
            R.REGISTRY.remove(entry_undeclared)
        r.ok("loud guard fires for an undeclared second-image-argument probe")

        entry_declared = R.StdlibEntry(
            "_reach48_probe", _probe_second_is_tensor, footprint="point",
            arg_footprint=((1, "image"),))
        R.REGISTRY.append(entry_declared)
        try:
            clean = R.unclassified_image_arg_candidates()
            assert "_reach48_probe" not in clean, (
                f"loud guard still flags a probe with arg_footprint declared: {clean}")
        finally:
            R.REGISTRY.remove(entry_declared)
        r.ok("loud guard stops firing once arg_footprint declares the second argument")
    except Exception as ex:
        r.fail("REACH-48 loud guard mutation", f"{type(ex).__name__}: {ex}")


def test_reach48_arg_footprint_malformed_fails_loud(r: SubTestResult):
    print("\n--- REACH-48: a malformed arg_footprint raises at decoration (fail-loud) ---")
    bad = [
        ((0, "image"),),            # index 0 — arg 0's reach is footprint's job, not this one's
        ((1, "image"), (1, "point")),  # repeated index
        ((1, ("halo", -1)),),       # malformed descriptor (negative radius)
        ((-1, "image"),),           # negative index
        ((1.0, "image"),),          # float index
        ((True, "image"),),         # bool masquerading as an index
        ("image",),                 # not a tuple of pairs
        (("image",),),              # a 1-tuple, not an (index, descriptor) pair
    ]
    try:
        leaked = [af for af in bad if R._valid_arg_footprint(af)]
        assert not leaked, f"validator accepted malformed arg_footprint: {leaked}"
        r.ok(f"_valid_arg_footprint rejects all {len(bad)} malformed descriptors")
    except Exception as ex:
        r.fail("REACH-48 validator rejects", f"{type(ex).__name__}: {ex}")

    # The decorator itself must raise (fail-loud at import, same discipline as footprint).
    try:
        raised = 0
        for af in bad:
            try:
                @R.stdlib("_reach48_bad_probe", footprint="point", arg_footprint=af)
                def _f(image, second):
                    return image
            except ValueError:
                raised += 1
            else:
                # Clean up if it somehow succeeded, so a bug here doesn't pollute REGISTRY.
                R.REGISTRY[:] = [e for e in R.REGISTRY if e.name != "_reach48_bad_probe"]
        assert raised == len(bad), f"only {raised}/{len(bad)} malformed decorations raised"
        r.ok("stdlib() raises ValueError for every malformed arg_footprint")
    except Exception as ex:
        r.fail("REACH-48 decorator fail-loud", f"{type(ex).__name__}: {ex}")

    # And a WELL-FORMED one is accepted (the other direction of the mutation check).
    try:
        @R.stdlib("_reach48_good_probe", footprint="point", arg_footprint=((1, "image"),))
        def _g(image, second):
            return image
        entry = next(e for e in R.REGISTRY if e.name == "_reach48_good_probe")
        assert dict(entry.arg_footprint) == {1: "image"}
        r.ok("a well-formed arg_footprint is accepted and stored")
    except Exception as ex:
        r.fail("REACH-48 decorator accepts well-formed", f"{type(ex).__name__}: {ex}")
    finally:
        R.REGISTRY[:] = [e for e in R.REGISTRY if e.name != "_reach48_good_probe"]


def test_reach48_binding_footprints_reports_kernel_whole(r: SubTestResult):
    print("\n--- REACH-48: binding_footprints reports convolve's @kernel as 'image' ---")
    try:
        _R.clear_roi_memo()
        fps = _R.binding_footprints("@OUT = convolve(@A, @K);", {})
        assert fps is not None, "binding_footprints returned None"
        kfp = fps.get("K")
        assert kfp is not None and kfp.kind == "image", (
            f"@K footprint = {kfp} — convolve's kernel argument must report 'image' "
            f"(TIERS-48-design.md SS B.2 point 2's per-arg reach)")
        afp = fps.get("A")
        assert afp is not None and afp.kind == "image", f"@A footprint regressed to {afp}"
        r.ok("binding_footprints: @A image (unchanged), @K image (REACH-48 fix)")
    except Exception as ex:
        r.fail("REACH-48 binding_footprints", f"{type(ex).__name__}: {ex}")
    finally:
        _R.clear_roi_memo()


def test_reach48_windowed_kernel_silently_corrupts_convolve(r: SubTestResult):
    """RED-FIRST pixel proof (TIERS-48-design.md SS B.2 point 2's "silent-wrong window"):
    before this lane, `binding_footprints` reported convolve's `kernel` argument as
    'point' (zero reach) — exactly the footprint a narrow-cook-crop consumer (a future
    `chain_windows`-shaped per-argument DAG walk, v0.49) would use to justify serving
    `@kernel` a WINDOW instead of the whole binding. This proves, with real pixels, what
    that would cost: crop the kernel to the single-pixel window 'point' licenses (zero
    halo around one pixel) and show the convolve output diverges from the correct
    whole-kernel result. The registry/tex_roi fix above closes the gap structurally (the
    substrate now reports 'image'), so no live consumer can make this mistake — this test
    is the evidence for WHY that fix is not merely cosmetic."""
    print("\n--- REACH-48 red-first: cropping @kernel to its old (wrong) footprint corrupts pixels ---")
    try:
        torch.manual_seed(48)
        A = torch.rand(1, 12, 12, 3)
        K = torch.rand(1, 5, 5, 1)          # a real kernel with information at every tap
        whole = TEXStdlib.fn_convolve(A, K, 1)

        # What a consumer trusting the OLD 'point' (zero-halo) footprint would serve:
        # a single-pixel window at the kernel's centre, everywhere else zeroed.
        cy, cx = K.shape[1] // 2, K.shape[2] // 2
        K_windowed = torch.zeros_like(K)
        K_windowed[:, cy:cy + 1, cx:cx + 1, :] = K[:, cy:cy + 1, cx:cx + 1, :]
        windowed = TEXStdlib.fn_convolve(A, K_windowed, 1)

        diff = (whole - windowed).abs().max().item()
        assert diff > 1e-3, (
            f"expected the windowed kernel to diverge from the whole one (proves the "
            f"danger a wrong 'point' footprint would license); got maxdiff {diff:.3e} — "
            f"the probe itself needs a kernel with more spread")
        r.ok(f"proved: serving @kernel the window 'point' would license corrupts convolve's "
             f"output (maxdiff {diff:.3e}) — this is the silent-wrong window per-arg reach "
             f"declarations close before any consumer trusts the substrate to crop it")
    except AssertionError as ex:
        r.fail("REACH-48 red-first pixel proof", str(ex))
    except Exception as ex:
        r.fail("REACH-48 red-first pixel proof", f"{type(ex).__name__}: {ex}")


def test_reach48_roi_plan_unaffected_default_path(r: SubTestResult):
    """Invariant #7 on the roi=None path: `roi_plan`/`binding_footprints` for every
    program that does NOT call a multi-image-argument builtin must be byte-identical
    before/after this lane — the per-arg lookup is an additive `_call_arg_reach(...)`
    that returns None (no declaration) for every other registered function, so
    `_accumulate`'s existing `_accumulate(a, ctx_halo, ...)` behaviour for `rest` args is
    unchanged whenever no arg_footprint applies."""
    print("\n--- REACH-48: roi_plan/binding_footprints unchanged for non-multi-image programs ---")
    cases = [
        ("@OUT = gauss_blur(@A, 2.0);", {}),
        ("@OUT = erode(@A, 3);", {}),
        ("@OUT = bilateral_filter(@A, 1.5, 0.2);", {}),
        ("@OUT = over(@A, @B);", {}),
        ("@OUT = mix(@A, @B, 0.5);", {}),
        ("@OUT = @A * 0.5;", {}),
    ]
    try:
        for code, ps in cases:
            _R.clear_roi_memo()
            plan = _R.roi_plan(code, ps)
            _R.clear_roi_memo()
            fps = _R.binding_footprints(code, ps)
            # Just exercising them end to end without raising is the invariant-7-shaped
            # check here; the specific expected shapes for these programs are already
            # pinned by test_v023_phase1.py / test_v030_phase1.py and unchanged by this
            # lane (no case above touches convolve or any arg_footprint-bearing fn).
            assert plan is not None and fps is not None
        r.ok(f"{len(cases)} non-multi-image programs: roi_plan/binding_footprints run clean")
    except Exception as ex:
        r.fail("REACH-48 default-path invariant 7", f"{type(ex).__name__}: {ex}")
    finally:
        _R.clear_roi_memo()
