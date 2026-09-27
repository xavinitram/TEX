"""
REG-1 — the single-source stdlib registry.

One `@stdlib(...)` decorator co-located with each `fn_*` impl replaces the
hand-maintained 143-entry `get_functions()` dict and (via TST-3) the parallel
taxonomy sets. The rule that keeps this a *readability* win, not a clever-registry
loss (all four audit agents flagged it): the decorator is **pure data attachment** —

  * the name is **explicit** (no dynamic `fn_*` discovery),
  * it attaches metadata only (no signature inference, no return-type magic),
  * `get_functions()` becomes `{name: fn for e in REGISTRY}` — one readable line
    replacing 143 hand-listed rows that could drift from the impls.

Layering note (Option 1): the *type contract* stays in the compiler —
`FUNCTION_SIGNATURES` (tex_compiler) is NOT derived from this runtime registry, so
the "compiler has zero edges into runtime" invariant holds. Instead TST-3 machine-
checks registry↔signatures parity, so the two representations cannot silently
drift. The `spatial`/`sync`/`non_local` tags carried here let TST-3 *derive* the
codegen/graphed/tiling taxonomy sets.
"""
import re as _re
from dataclasses import dataclass


@dataclass(frozen=True)
class StdlibEntry:
    """One registered stdlib function. `fn` is the raw callable (the same object
    `TEXStdlib.<attr>` resolves to). `doc`/`ex` are populated by DOC-4."""
    name: str
    fn: object
    aliases: tuple = ()
    spatial: bool = False      # codegen routes specially (stencil/sample)
    sync: bool = False         # graph tier must synchronise around it
    # ROI-1: access footprint — which input pixels one output pixel reads; the
    # substrate ROI-2/5/6 build on. One of: 'point' (per-pixel, default), 'image'
    # (whole-image reduction or data-dependent gather), ('halo', r) (fixed radius r),
    # ('halo_arg', i) (radius from arg i), ('halo_arg', i, mult) (pixel reach = mult·arg i —
    # gauss_blur's radius is 3·sigma, so mult=3.0), ('frame', i) (temporal window from arg i).
    footprint: "str | tuple" = "point"
    doc: str = ""
    ex: str = ""
    # LANG-4: help data migrated OUT of the hand-kept JS `TEX_HELP_DATA` into the registry
    # so it is single-sourced. `sig` is the human signature ("sin(x) → float"); `category`
    # is the help-panel grouping ("Math"). Empty on entries with no help (none, today).
    sig: str = ""
    category: str = ""
    # COLOR-1 (v0.40 simplify): 0-based positions of arguments that are a non-spatial
    # RESOURCE (a plain bound tensor read by the function but never a per-pixel driving
    # wire) rather than an ordinary spatial/image argument — `apply_lut3d`'s LUT
    # (`non_spatial_args=(1,)`) is the first case. `_consensus_extent`'s (B,H,W) shape
    # scan and `graphed._spatial_px`'s capture-worthiness estimate both exclude a wire
    # bound at one of these positions, generically, from `_collect_binding_reads`'s walk
    # (interpreter.py) — a function need only declare the field, no engine-side edit.
    # Empty (the default) for every function whose arguments are all ordinary.
    non_spatial_args: tuple = ()
    # SCALE-47b: 0-based positions of arguments whose CONTRACT is "a distance in pixels"
    # (`gauss_blur`'s sigma, `erode`/`dilate`'s radius, `bilateral_filter`'s spatial_sigma) —
    # a NEW, independent tag rather than reusing `footprint`'s `mult` (SCALE-47-design.md §3):
    # `mult` answers "how far does this arg reach for ROI halo purposes", not "should this
    # arg's VALUE scale with the cook's resolution" — two different questions about the same
    # argument that happen to share one answer for every builtin registered today (each of
    # the four `pixel_args=` builtins is also `halo_arg`-tagged on that same index; BILAT-50
    # removed the one case, `bilateral_filter`'s old fixed `('halo', 3)`, that used to force
    # the two tags apart — see `_valid_pixel_args` below for the check that still holds
    # either way). Empty (the default) for every function with no pixel-unit argument. A
    # cook's `scale=` multiplies the RESOLVED value at each of these positions before the
    # call (interpreter) / references a runtime scalar at the same positions in the emitted
    # call (codegen) — never an AST-level fold, so the emitted source stays identical across
    # scale values (one compiled artifact per program, not one per scale).
    pixel_args: tuple = ()
    # FIX-PACE P4 (Phase C): a builtin whose device cost is genuinely expensive but whose
    # ACCESS footprint is still 'point' (it reads only its own coordinate args, so ROI/tiling
    # need not know about it) -- footprint answers "which pixels does this read", not "how
    # expensive is this to run", and a runtime-variable-cost builtin (octave count) or a
    # per-pixel search (cellular/Worley) can be both 'point'-footprint AND heavy. A SEPARATE
    # tag from `footprint` (never folded into it): tagging a 'point' builtin `heavy=True`
    # must never change its ROI/tiling treatment, only `pacing_heavy.heavy_builtin_names()`'s
    # per-statement cost classification (`tex_runtime/pacing.py:paced_check`'s own `heavy=`
    # bypass of stride economization). Default `False` for every function with no runtime-
    # variable or otherwise underestimated cost — this tag is additive, never a downgrade.
    heavy: bool = False
    # REACH-48 (TIERS-48-design.md SS B.2 point 2): per-argument reach for a builtin with
    # SEVERAL image arguments. `footprint` describes arg 0's (the image's) reach; a function
    # like `convolve(image, kernel, normalize)` reads a SECOND argument (`kernel`) as its own
    # full image/array binding, whose reach is independent of arg 0's and of `pixel_args`/
    # `non_spatial_args` (which both describe a NON-image scalar or resource, the opposite
    # question). `arg_footprint` is a tuple of `(index, descriptor)` pairs — a tuple, not a
    # dict, for the same reason `pixel_args`/`non_spatial_args` are tuples: `StdlibEntry` is
    # frozen and hashable-by-value, so every field must stay hashable. `index` is always >= 1
    # (arg 0's reach is `footprint`'s job — declaring it here too would be a contradiction,
    # not a description); `descriptor` is validated exactly like a whole-function `footprint`
    # (`_valid_footprint`), so a malformed per-arg descriptor fails loud at import, same as a
    # malformed `footprint` does. Empty (the default) for every function with only one image
    # argument — this tag is additive, and every existing registration is unaffected.
    arg_footprint: tuple = ()

    @property
    def names(self) -> tuple:
        """Primary name plus any aliases."""
        return (self.name, *self.aliases)

    @property
    def non_local(self) -> bool:
        """Derived (ROI-1, invariant #5): a non-'point' footprint reads beyond the
        current pixel, so a program calling it can't be split into strips (M-4). This
        is the single source `tex_memory._NON_LOCAL_FNS` and `gen_function_reference`
        read — it replaces the old hand-set boolean field, no consumer changed."""
        return self.footprint != "point"


# Registered in decoration (source) order as the class body of TEXStdlib executes.
REGISTRY: list[StdlibEntry] = []


def _valid_footprint(fp) -> bool:
    """ROI-1 footprint well-formedness. A malformed descriptor (a typo like
    ('halo', 'x') or a bare ('frame',)) must fail LOUD at import, not silently
    mis-tag a function — the exact silent-wrong class the taxonomy exists to close.
    'halo' takes a positive number; 'halo_arg'/'frame' a non-negative arg index.
    'halo_arg' takes an OPTIONAL third element `mult` (>0) — the ROI-2 reach multiplier
    that turns the argument into a pixel reach (`gauss_blur`'s kernel radius is 3·sigma,
    not sigma; the descriptor carries `mult=3.0`). Default multiplier is 1.0 (erode/dilate,
    whose radius argument already is the pixel reach). A1 (v0.50 Phase C): an OPTIONAL
    fourth element `approx_above` (>0) — the RAW (pre-`mult`) argument value past which the
    builtin itself switches to a downscale/resample approximation whose grid is anchored to
    the crop, not the frame (`gauss_blur`'s pyramid, `bilateral_filter`'s detail-transfer);
    `tex_roi._reach_of` declines to narrow past it (see its own comment). bool is rejected
    explicitly (it is an int subclass, and True as a radius/threshold is a bug, not one)."""
    if fp == "point" or fp == "image":
        return True
    if not isinstance(fp, tuple) or len(fp) not in (2, 3, 4):
        return False
    kind, val = fp[0], fp[1]
    if isinstance(val, bool):
        return False
    if kind == "halo":
        return len(fp) == 2 and isinstance(val, (int, float)) and val > 0
    if kind == "frame":
        return len(fp) == 2 and isinstance(val, int) and val >= 0
    if kind == "halo_arg":
        if not (isinstance(val, int) and val >= 0):
            return False
        if len(fp) == 2:
            return True
        mult = fp[2]
        if not (isinstance(mult, (int, float)) and not isinstance(mult, bool) and mult > 0):
            return False
        if len(fp) == 3:
            return True
        approx_above = fp[3]
        return (isinstance(approx_above, (int, float)) and not isinstance(approx_above, bool)
                and approx_above > 0)
    return False


def _valid_pixel_args(pixel_args, footprint) -> bool:
    """SCALE-47b: each position must be a non-negative int, arg 0 (the image) is never a
    pixel-magnitude argument, and — the TST-3-style derivation check AGENTS.md invariant #5
    asks for — a `halo_arg` footprint's OWN index is always itself a scalable magnitude (the
    two tags describe the same argument for different questions), so declaring `pixel_args`
    without it would silently under-scale that footprint's own halo derivation. The rule only
    FIRES when a `halo_arg` footprint is actually present — a function could in principle
    still declare `pixel_args` against a fixed (non-`halo_arg`) footprint, or no footprint at
    all (`bilateral_filter` used to be exactly that case, before BILAT-50 gave its window its
    own `halo_arg`; nothing registered today exercises the non-`halo_arg` branch any more, but
    the check stays permissive for whatever the next one is)."""
    if any((not isinstance(i, int)) or isinstance(i, bool) or i < 1 for i in pixel_args):
        return False
    if isinstance(footprint, tuple) and footprint and footprint[0] == "halo_arg":
        if footprint[1] not in pixel_args:
            return False
    return True


def _valid_arg_footprint(arg_footprint) -> bool:
    """REACH-48: `arg_footprint` well-formedness, the same discipline `_valid_footprint`
    already applies to the whole-function descriptor. Each item must be an `(index,
    descriptor)` pair, `index` a non-negative-excluding-zero int (arg 0's reach is
    `footprint`'s job, not this one's — index 0 here is a contradiction, not a typo, so it
    is rejected the same way a bool radius is), no index repeated (one declaration per
    argument), and `descriptor` a valid `_valid_footprint` value."""
    seen = set()
    for item in arg_footprint:
        if not (isinstance(item, tuple) and len(item) == 2):
            return False
        idx, fp = item
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 1:
            return False
        if idx in seen:
            return False
        seen.add(idx)
        if not _valid_footprint(fp):
            return False
    return True


def stdlib(name, *, aliases=(), spatial=False, sync=False, footprint="point",
           doc="", ex="", sig="", category="", non_spatial_args=(), pixel_args=(),
           heavy=False, arg_footprint=()):
    """Record one StdlibEntry and return the decorated object UNCHANGED (so an
    inner `@staticmethod` still applies). Pure data attachment — the name is
    explicit; nothing is inferred or discovered. `footprint` (ROI-1) is validated
    here so a malformed descriptor can never reach the registry. `sig`/`category`
    (LANG-4) carry the help data that used to live only in the JS. `non_spatial_args`
    (COLOR-1) names which 0-based argument positions are a non-spatial resource, not
    an ordinary image/coordinate argument — see `StdlibEntry.non_spatial_args`.
    `pixel_args` (SCALE-47b) names which 0-based argument positions are a pixel-unit
    magnitude a cook's `scale=` must multiply — see `StdlibEntry.pixel_args`.
    `heavy` (FIX-PACE P4) marks a function device-expensive independent of its
    footprint — see `StdlibEntry.heavy`. `arg_footprint` (REACH-48) names the
    independent reach of any OTHER (index >= 1) argument that is itself a full image/
    array binding — see `StdlibEntry.arg_footprint`."""
    if not _valid_footprint(footprint):
        raise ValueError(
            f"stdlib({name!r}): invalid footprint {footprint!r}. Expected 'point', "
            f"'image', ('halo', r>0), ('halo_arg', i>=0), or ('frame', i>=0).")
    if not _valid_pixel_args(pixel_args, footprint):
        raise ValueError(
            f"stdlib({name!r}): invalid pixel_args {pixel_args!r}. Expected a tuple of "
            f"positive ints (arg 0, the image, is never a pixel magnitude), and it must "
            f"include a 'halo_arg' footprint's own index when one is declared.")
    if not isinstance(heavy, bool):
        raise ValueError(f"stdlib({name!r}): heavy must be a bool, got {heavy!r}.")
    if not _valid_arg_footprint(arg_footprint):
        raise ValueError(
            f"stdlib({name!r}): invalid arg_footprint {arg_footprint!r}. Expected a tuple "
            f"of (index>=1, descriptor) pairs, each descriptor valid per `_valid_footprint`, "
            f"no index repeated.")

    def deco(obj):
        fn = obj.__func__ if isinstance(obj, staticmethod) else obj
        REGISTRY.append(StdlibEntry(name, fn, tuple(aliases), spatial, sync,
                                    footprint, doc, ex, sig, category,
                                    tuple(non_spatial_args), tuple(pixel_args), heavy,
                                    tuple(arg_footprint)))
        # REG-1c: a registration changes what `non_spatial_args_by_name()`/`pixel_args_by_name()`/
        # `arg_footprint_by_name()` must answer, so their caches (below) are invalidated here —
        # the ONLY place `REGISTRY` grows. This also covers late registration (a decorator
        # running after the first lookup): the next call rebuilds from the now-longer
        # `REGISTRY` instead of answering from a stale snapshot.
        global _NON_SPATIAL_CACHE_READY, _PIXEL_ARGS_CACHE_READY, _ARG_FOOTPRINT_CACHE_READY
        _NON_SPATIAL_CACHE_READY = False
        _PIXEL_ARGS_CACHE_READY = False
        _ARG_FOOTPRINT_CACHE_READY = False
        return obj
    return deco


# REG-1c: built lazily by `non_spatial_args_by_name()` and invalidated by `stdlib()`'s
# `deco` above on every new registration — the pair keeps this a correct
# O(1)-after-first-use cache rather than a stale snapshot. A plain dict (not a `None`
# sentinel) so the cache-store census in `tests/test_v018_docs.py` sees it: see its
# ARCHITECTURE.md row.
_NON_SPATIAL_CACHE: dict = {}
_NON_SPATIAL_CACHE_READY = False


def non_spatial_args_by_name() -> dict:
    """{name: non_spatial_args} for every registered name (aliases expanded) whose
    `non_spatial_args` is non-empty — the single source `_collect_binding_reads`
    (interpreter.py) and `graphed._spatial_px` derive their LUT-class exclusion from.

    REG-1c: cached here (not just at `interpreter._READS_MEMO`, which is keyed per
    PROGRAM object): a cold compile builds a fresh `Program` every call, so that
    per-program memo never hits and this whole-registry scan used to re-pay on every
    single cold compile — the ONE-TIME cost `_collect_binding_reads_and_non_spatial`'s
    docstring assumed. Rebuilt once per process (or once per new registration, via
    `stdlib()`'s `deco`) instead."""
    global _NON_SPATIAL_CACHE_READY
    if not _NON_SPATIAL_CACHE_READY:
        _NON_SPATIAL_CACHE.clear()
        _NON_SPATIAL_CACHE.update(
            (n, e.non_spatial_args) for e in REGISTRY if e.non_spatial_args for n in e.names)
        _NON_SPATIAL_CACHE_READY = True
    return _NON_SPATIAL_CACHE


# SCALE-47b: the mirror of REG-1c's non-spatial cache, same build-once-invalidate-on-register
# discipline (see `stdlib()`'s `deco`, which flips both readiness flags on every new
# registration — one `REGISTRY` growth site, two derived caches).
_PIXEL_ARGS_CACHE: dict = {}
_PIXEL_ARGS_CACHE_READY = False


def pixel_args_by_name() -> dict:
    """{name: pixel_args} for every registered name (aliases expanded) whose `pixel_args` is
    non-empty — the single source the interpreter's scale-multiply dispatch and codegen's
    emission both read. Same cache shape and invalidation rule as `non_spatial_args_by_name()`."""
    global _PIXEL_ARGS_CACHE_READY
    if not _PIXEL_ARGS_CACHE_READY:
        _PIXEL_ARGS_CACHE.clear()
        _PIXEL_ARGS_CACHE.update(
            (n, e.pixel_args) for e in REGISTRY if e.pixel_args for n in e.names)
        _PIXEL_ARGS_CACHE_READY = True
    return _PIXEL_ARGS_CACHE


# REACH-48: the mirror of REG-1c's caches, same build-once-invalidate-on-register discipline.
_ARG_FOOTPRINT_CACHE: dict = {}
_ARG_FOOTPRINT_CACHE_READY = False


def arg_footprint_by_name() -> dict:
    """{name: {index: descriptor}} for every registered name (aliases expanded) whose
    `arg_footprint` is non-empty — the single source `tex_roi._call_reach`/`_accumulate`
    read to resolve a multi-image-argument builtin's OTHER argument(s) to their own,
    independently-declared reach (TIERS-48-design.md SS B.2 point 2). Same cache shape and
    invalidation rule as `non_spatial_args_by_name()`/`pixel_args_by_name()`."""
    global _ARG_FOOTPRINT_CACHE_READY
    if not _ARG_FOOTPRINT_CACHE_READY:
        _ARG_FOOTPRINT_CACHE.clear()
        _ARG_FOOTPRINT_CACHE.update(
            (n, dict(e.arg_footprint)) for e in REGISTRY if e.arg_footprint for n in e.names)
        _ARG_FOOTPRINT_CACHE_READY = True
    return _ARG_FOOTPRINT_CACHE


def unclassified_image_arg_candidates() -> list:
    """REACH-48 loud guard (TIERS-48-design.md SS B.2 point 2, mirroring
    `unclassified_fragile_candidates()`'s own structural-marker shape): a registered fn
    whose impl resolves a non-first, non-`pixel_args`, non-`non_spatial_args`, non-own-
    `halo_arg`-index argument through the SAME idiom `convolve`'s `kernel` uses —
    `<local> = <name> if <name>.__class__ is torch.Tensor else _to_tensor(<name>)` (or a
    bare `_to_tensor(<name>)`) followed by a rank check on the resulting local (`.dim(`) —
    but where that argument position is not already covered by an `arg_footprint`
    declaration. This is the signal that distinguishes a SECOND IMAGE/ARRAY binding (read
    as a whole tensor with its own shape contract) from a plain scalar resolved via
    `_host_scalar`/`_uniform_scalar_or_raise`/`.item()` (neither of which is rank-checked).
    Structural, not exhaustive — like its fp16 sibling, it reduces the drift risk for a NEW
    multi-image-argument builtin added without a per-argument reach declaration; it does
    not replace reviewing a new registration by hand."""
    import inspect
    out = []
    for e in REGISTRY:
        try:
            params = list(inspect.signature(e.fn).parameters.keys())
        except (TypeError, ValueError):
            continue
        accounted = {0, *e.pixel_args, *e.non_spatial_args, *(i for i, _ in e.arg_footprint)}
        fp = e.footprint
        if isinstance(fp, tuple) and fp and fp[0] == "halo_arg":
            accounted.add(fp[1])
        body = _fn_body_src(e.fn)
        for i, pname in enumerate(params):
            if i in accounted:
                continue
            esc = _re.escape(pname)
            m = _re.search(rf"(\w+)\s*=\s*{esc}\b[^\n]*?_to_tensor\({esc}\)", body)
            local = m.group(1) if m else pname
            if _re.search(rf"_to_tensor\({esc}\)", body) and (
                    _re.search(rf"\b{_re.escape(local)}\.dim\(", body)
                    or _re.search(rf"\b{esc}\.dim\(", body)):
                out.append(e.name)
                break
    return sorted(set(out))


def functions() -> dict:
    """`{name: fn}` for every registered name (aliases expanded) — the view that
    backs `TEXStdlib.get_functions()`."""
    out = {}
    for e in REGISTRY:
        for n in e.names:
            out[n] = e.fn
    return out


def spatial_names() -> frozenset:
    """The registry-derived set of stencil/spatial function names (STR-7): single
    source for codegen's `_SPATIAL_STDLIB`, eliminating the hand-maintained literal.
    Function form (not a module-level constant) so it's evaluated AFTER `TEXStdlib`'s
    class body has populated `REGISTRY` — TST-3 already proves this derivation equals
    the old literal exactly."""
    return frozenset(n for e in REGISTRY for n in e.names if e.spatial)


def non_local_names() -> frozenset:
    """The registry-derived set of names whose footprint != 'point' (ROI-1): the
    single source for `tex_memory._NON_LOCAL_FNS`, replacing that hand-kept literal.
    Function form (evaluated AFTER `TEXStdlib`'s class body has populated `REGISTRY`),
    mirroring `spatial_names()` — TST-3 proves it equals the old 18-name literal."""
    return frozenset(n for e in REGISTRY for n in e.names if e.non_local)


def _decode_sig(s: str) -> str:
    """Decode a stored JS-escaped `sig` (\\uXXXX, \\n, \\", \\\\) into display text. The
    registry stores sigs in their JS-escaped form so they compare byte-for-byte against
    the editor's TEX_HELP_DATA (the LANG-4 drift test); callers decode for display."""
    s = _re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)
    return s.replace("\\n", " ").replace('\\"', '"').replace("\\\\", "\\")


def _help_entry(e, decode: bool) -> dict:
    """One help dict from a registry entry: name, aliases, sig, desc (`doc`), example
    (`ex`), category, tags. The single entry-builder for `help_entries`/`help_lookup`."""
    return {
        "name": e.name,
        "aliases": list(e.aliases),
        "sig": _decode_sig(e.sig) if decode else e.sig,
        "desc": e.doc,
        "example": e.ex,
        "category": e.category,
        "tags": [t for t, on in (("spatial", e.spatial), ("sync", e.sync),
                                 ("non-local", e.non_local)) if on],
    }


def help_entries(decode: bool = False) -> list:
    """LANG-4: the function help data, single-sourced from the registry (the JS
    `TEX_HELP_DATA` function entries are now a drift-pinned MIRROR of this). `decode=True`
    turns the stored JS-escaped sig into display text."""
    return [_help_entry(e, decode) for e in REGISTRY]


def help_lookup(name: str, decode: bool = True):
    """The help entry for one function name (primary or alias), or None — builds only the
    matched entry, not the whole list. Used by the CLI `tex help <fn>`."""
    for e in REGISTRY:
        if name == e.name or name in e.aliases:
            return _help_entry(e, decode)
    return None


# ── C2-st: fp16 precision taxonomy (single source) ────────────────────────────
# precision_policy's fp16 gate had a SECOND, un-federated taxonomy (doc 34 weakness
# #8): hand-coded `_FP16_FRAGILE_FNS`/`_BOUNDED_FNS` with zero link to the registry, so
# a new fp16-fragile stdlib fn silently defaulted to fp16-ELIGIBLE (the unsafe
# direction — and the A1-1 fuzzer proved it, finding `degrees` amplifying 57x). This is
# now the single home. `FP16_FRAGILE` = a fp16 half-ULP wrecks it (discontinuous /
# domain-restricted / exp-growth / near-singular / unbounded reduction). `FP16_BOUNDED`
# = range-capped smooth (sin/cos/tanh/atan) — accepted, their fp16 error is caught by
# the gain/magnitude analysis instead. Everything else is pointwise-safe by omission
# (abs/min/max/mix/clamp/lerp/...); the gate's amplifier arithmetic (pow/dot/fit/degrees
# scaling) lives in precision_policy._gm, not here.
FP16_FRAGILE = frozenset({
    "floor", "round", "ceil", "fract", "trunc", "mod", "sign",
    "step", "smoothstep",
    "acos", "asin", "sqrt", "log", "log2", "log10",   # log10 added — C2-st found the gap
    "exp", "pow2", "pow10",                            # F3: 2^x / 10^x exp-growth (like exp)
    "smin", "smax",
    "tan", "atan2", "normalize", "hypot", "sdiv", "sinh", "cosh",
    "arr_sum",
    # F4: compositing fns that divide by a value that can approach 0 (unbounded fp16 gain
    # near a vanishing alpha / (1-b) / b). `under` delegates to `over`; `atop` does NOT
    # divide (out_a = bg.a), so it is correctly omitted.
    "over", "under", "unpremultiply", "color_dodge", "color_burn", "vivid_light",
    # ASK-1: convolve is an unbounded weighted reduction over up to 66049 taps (the
    # arr_sum class above) AND normalize divides by a kernel sum that can approach zero
    # (the F4 class above) -- two independent fp16-fragile reasons, either one enough.
    "convolve",
    # ASK-13: patch_dist's box mean is an unbounded reduction over up to 65^2 taps
    # (radius clamps to 32) of a SQUARED difference -- the arr_sum class above, and
    # squaring is itself amplifying near zero. Neither `_FRAGILE_NAME_STEMS` (prefix
    # match) nor `_IMPL_FRAGILE_MARKERS` (looks only for `_safe_div(`/`sdiv(`) catches
    # this name, so it is classified here by hand.
    "patch_dist",
    # ASK-4: img_width/img_height are fp32-only builtins (never fp16 themselves,
    # invariant #4's reason -- a large dimension isn't an fp16 value), but the
    # PRODUCT of image lineage with their runtime magnitude amplifies fp16 error the
    # same way `sin(@A.r * iw)` does for the `iw` builtin (`_BUILTIN_MAG` in
    # precision_policy.py closes that for the coordinate IDENTIFIERS; a FunctionCall's
    # gain is scored from its args, not its own magnitude, so an unknown call like
    # `img_width(@K)` reads magnitude 1 by default and would launder the hazard).
    # Neither the name-prefix stems nor the impl markers above catch "img_" or a
    # shape read, so -- like patch_dist -- this is classified here by hand rather than
    # relying on the loud guard to notice.
    "img_width", "img_height",
    # ASK-5: worley_id is the `floor`/`step` class by name (a discontinuous, per-cell
    # hash of an argmin winner — a half-ULP coordinate nudge can flip which cell wins
    # and relocate the whole id, not perturb it), but "worley_id" matches neither
    # `_FRAGILE_NAME_STEMS` (no stem there reads "worley") nor `_IMPL_FRAGILE_MARKERS`
    # (its body has no `_safe_div`/`sdiv`), so neither loud guard below catches it.
    # Classified here by hand, same as `patch_dist` above.
    "worley_id",
    # ASK-6c: select(cond, a, b) IS a per-pixel branch (torch.where), just spelled as a call
    # instead of `if` / `?:` — an fp16 cond that rounds across the 0.5 threshold flips the
    # entire pick, arm values included, not a scalar quantum. It is declined unconditionally
    # on sight, exactly like the IfElse/TernaryOp/WhileLoop branch in `precision_policy`, and
    # this set is how a fn is declined on sight; neither guard below reads "select".
    "select",
})
FP16_BOUNDED = frozenset({"sin", "cos", "tanh", "atan"})

# Name-PREFIX stems whose fp16 behaviour a maintainer MUST classify (the "loud" guard):
# a registered fn whose name STARTS WITH one of these but is NOT classified is almost
# certainly an un-triaged fp16 hazard (the degrees/exp/log10 class the federation
# already caught). Prefix (not substring) match so `distance` doesn't false-match `tan`.
_FRAGILE_NAME_STEMS = ("sqrt", "exp", "log", "tan", "sinh", "cosh", "acos", "asin",
                       "floor", "round", "ceil", "fract", "trunc", "normalize",
                       "hypot", "sdiv", "pow2", "pow10")

# F2/F4 root fix: a STRUCTURAL fragility signal the name guard misses. A fn whose impl
# divides by a data-dependent value (`_safe_div`/`sdiv` — used only for VARIABLE
# denominators; constant division uses `/`) amplifies fp16 error near the zero, no matter
# what it's named. `torch.exp(` is deliberately NOT a marker: exp(-d²) (bilateral weights)
# is bounded, so it would false-positive — exp-growth is covered by the name stems + the
# explicit exp/pow2/pow10 entries instead. Coverage caveat (G4): this catches a fn's own
# body plus ONE level of `TEXStdlib.fn_*` delegation (so `under`→`over` is caught). A
# multi-hop wrapper chain still needs hand-classification — the guard reduces, not
# eliminates, the drift risk.
_IMPL_FRAGILE_MARKERS = ("_safe_div(", "sdiv(")


def _fn_body_src(fn) -> str:
    """A registered fn's source WITHOUT its decorator lines — so `ex=`/`doc=` example text
    (which can contain `sdiv(...)`) isn't scanned as if it were the body (a false-positive
    source, G4)."""
    try:
        import inspect
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return ""
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("def "):
            return "\n".join(lines[i:])
    return src


def _impl_looks_fragile(fn, _depth: int = 0) -> bool:
    body = _fn_body_src(fn)
    if any(m in body for m in _IMPL_FRAGILE_MARKERS):
        return True
    # Resolve ONE level of same-module delegation: `under` is `return TEXStdlib.fn_over(...)`
    # — its own body has no marker, but it inherits `over`'s fragility. Match the callee's
    # fn-name against the registry (no TEXStdlib import → no cycle).
    if _depth == 0:
        import re
        callees = set(re.findall(r"TEXStdlib\.(fn_\w+)\s*\(", body))
        if callees:
            for e in REGISTRY:
                if getattr(e.fn, "__name__", "") in callees and _impl_looks_fragile(e.fn, 1):
                    return True
    return False


def unclassified_fragile_candidates() -> list:
    """Registered fn names that LOOK fp16-fragile — by name-prefix (the degrees/exp/log10
    class) OR by implementation (divides by a data-dependent value — the compositing class
    F4) — but aren't classified. The loud guard for a new fn added without an fp16 triage.
    FP16_FRAGILE / FP16_BOUNDED are the public constants both consumers read directly."""
    out = set()
    for e in REGISTRY:
        impl_frag = _impl_looks_fragile(e.fn)
        for n in e.names:
            if n in FP16_FRAGILE or n in FP16_BOUNDED:
                continue
            if impl_frag or any(n.startswith(stem) for stem in _FRAGILE_NAME_STEMS):
                out.add(n)
    return sorted(out)
