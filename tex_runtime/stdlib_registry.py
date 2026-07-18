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
    # ('halo_arg', i) (radius from arg i), ('frame', i) (temporal window from arg i).
    footprint: "str | tuple" = "point"
    doc: str = ""
    ex: str = ""
    # LANG-4: help data migrated OUT of the hand-kept JS `TEX_HELP_DATA` into the registry
    # so it is single-sourced. `sig` is the human signature ("sin(x) → float"); `category`
    # is the help-panel grouping ("Math"). Empty on entries with no help (none, today).
    sig: str = ""
    category: str = ""

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
    bool is rejected explicitly (it is an int subclass, and True as a radius is a
    bug, not a radius)."""
    if fp == "point" or fp == "image":
        return True
    if isinstance(fp, tuple) and len(fp) == 2:
        kind, val = fp
        if isinstance(val, bool):
            return False
        if kind == "halo":
            return isinstance(val, (int, float)) and val > 0
        if kind in ("halo_arg", "frame"):
            return isinstance(val, int) and val >= 0
    return False


def stdlib(name, *, aliases=(), spatial=False, sync=False, footprint="point",
           doc="", ex="", sig="", category=""):
    """Record one StdlibEntry and return the decorated object UNCHANGED (so an
    inner `@staticmethod` still applies). Pure data attachment — the name is
    explicit; nothing is inferred or discovered. `footprint` (ROI-1) is validated
    here so a malformed descriptor can never reach the registry. `sig`/`category`
    (LANG-4) carry the help data that used to live only in the JS."""
    if not _valid_footprint(footprint):
        raise ValueError(
            f"stdlib({name!r}): invalid footprint {footprint!r}. Expected 'point', "
            f"'image', ('halo', r>0), ('halo_arg', i>=0), or ('frame', i>=0).")

    def deco(obj):
        fn = obj.__func__ if isinstance(obj, staticmethod) else obj
        REGISTRY.append(StdlibEntry(name, fn, tuple(aliases), spatial, sync,
                                    footprint, doc, ex, sig, category))
        return obj
    return deco


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
