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
    non_local: bool = False    # reads neighbouring pixels — wrong if tiled
    doc: str = ""
    ex: str = ""

    @property
    def names(self) -> tuple:
        """Primary name plus any aliases."""
        return (self.name, *self.aliases)


# Registered in decoration (source) order as the class body of TEXStdlib executes.
REGISTRY: list[StdlibEntry] = []


def stdlib(name, *, aliases=(), spatial=False, sync=False, non_local=False,
           doc="", ex=""):
    """Record one StdlibEntry and return the decorated object UNCHANGED (so an
    inner `@staticmethod` still applies). Pure data attachment — the name is
    explicit; nothing is inferred or discovered."""
    def deco(obj):
        fn = obj.__func__ if isinstance(obj, staticmethod) else obj
        REGISTRY.append(StdlibEntry(name, fn, tuple(aliases), spatial, sync,
                                    non_local, doc, ex))
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
