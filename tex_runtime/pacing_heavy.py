"""PACE-47d — registry-derived per-statement heavy/cheap classification.

A small, self-contained helper (deliberately its own file rather than an addition to
`interpreter_analysis.py` or `codegen.py`, per this ask's own brief: "keep your edits
confined to the poll sites and a small helper", while SCALE-47b edits interpreter/codegen
emission in parallel — this stays out of that surface's way entirely).

**What "heavy" means here, and why.** PACE-47c closed the completed-tail blind spot
(`tex_runtime/pacing.py:paced_check`'s `heavy=` parameter: bypass the stride economization
entirely, always record) for exactly two builtins (`gauss_blur`, the mip pyramid) by hand,
because those happened to already have an internal multi-pass poll point to hang it on.
PACE-47d generalizes the CLASSIFICATION half — "is the statement about to run heavy" —
so every paced poll point can ask the same question, DERIVED FROM THE REGISTRY rather than
a hand-maintained name list: a builtin is heavy here iff its OWN `@stdlib(...)` footprint
(ROI-1) is `('halo', r)` or `('halo_arg', i[, mult])` — a fixed or argument-derived pixel
reach, the coordinator's own criterion. This covers `gauss_blur`/`erode`/`dilate`/
`bilateral_filter` by construction, with zero per-name special-casing.

**What this does NOT cover, on purpose.** `sample_mip`/`sample_mip_gauss` are multi-pass
but their own footprint is `'image'` (COLOR-1: a `('halo_arg', kernel)` shape cannot express
"reads whichever mip level the LOD argument resolves to" — see `stdlib_sample.py`'s own
comment at that declaration). They are NOT in `heavy_builtin_names()`'s output. This is
safe, not merely convenient: `_build_mip_pyramid`'s own internal poll (PACE-47c, already
`heavy=True`) sits at the TOP of its per-level loop, before ANY of that level's compute —
functionally an entry poll, reached before this module's classification could ever have
helped. A per-pixel-footprint criterion could not express "multi-pass" as a registry tag
without inventing one; reusing an EXISTING tag (footprint) for a question it partly answers,
rather than adding a new one this ask did not ask for, is the smaller, safer surface.
"""
from __future__ import annotations

from collections import OrderedDict

from ..tex_compiler.ast_nodes import FunctionCall, iter_child_nodes

_HEAVY_NAMES_CACHE: "frozenset[str] | None" = None


def _compute_heavy_builtin_names() -> frozenset[str]:
    from .stdlib_registry import REGISTRY
    names: set[str] = set()
    for entry in REGISTRY:
        fp = entry.footprint
        # FIX-PACE P4: heaviness is now TWO independent registry signals, OR'd together --
        # a halo-shaped footprint (device-expensive because it reads beyond one pixel), OR
        # the entry's own `heavy` tag (device-expensive for a reason footprint cannot
        # express: a runtime-variable octave count, a per-pixel cellular search — still
        # 'point'-footprint, so ROI/tiling need not change). Neither implies the other.
        if (isinstance(fp, tuple) and fp[0] in ("halo", "halo_arg")) or entry.heavy:
            names.update(entry.names)
    return frozenset(names)


def heavy_builtin_names() -> frozenset[str]:
    """Every registered name (aliases expanded) whose footprint is halo-shaped OR whose
    own `heavy` registry tag is set (FIX-PACE P4) — cached process-wide (the registry does
    not change after import; `stdlib()`'s own `deco` only ever APPENDS during module load,
    never after)."""
    global _HEAVY_NAMES_CACHE
    if _HEAVY_NAMES_CACHE is None:
        _HEAVY_NAMES_CACHE = _compute_heavy_builtin_names()
    return _HEAVY_NAMES_CACHE


def _stmt_calls_heavy_builtin(stmt) -> bool:
    """Generic child-node walk (the same shape `interpreter_analysis`'s own
    `_collect_binding_reads_and_non_spatial` uses) looking for any `FunctionCall` whose
    `.name` is registry-heavy. Descends into every nested expression/block (an `if`/`for`
    body, a nested call's own arguments), so a heavy call anywhere inside a compound
    top-level statement marks the whole statement."""
    heavy_names = heavy_builtin_names()
    stack = [stmt]
    while stack:
        node = stack.pop()
        if type(node) is FunctionCall and node.name in heavy_names:
            return True
        stack.extend(iter_child_nodes(node))
    return False


#: Per-STATEMENT-LIST memo, id-keyed with an `is` re-check (the exact shape and safety
#: argument as `interpreter_analysis._READS_MEMO`: a `Program` is a slotted dataclass with
#: no `__dict__`/`__weakref__`, so `id()` is the only cheap key, and a recycled id belongs
#: to a different object the `is` check catches). Keyed by the STATEMENTS LIST
#: (`program.statements`) rather than the `Program` object itself: `stmts = program.
#: statements` is a plain attribute read of the SAME list object every time (a dataclass
#: field, not a property that copies), so every call site already holds the right memo
#: key in hand without threading `program` itself through call sites that today only ever
#: receive `stmts` (`_exec_stmts_profiled` is one — PACE-45 never needed `program` there
#: and this ask's own brief asks for the smallest surface, not a new parameter everywhere).
#: Classifying is a pure function of the AST, so it belongs once per program, not once per
#: cook.
_HEAVY_STMT_MEMO: "OrderedDict[int, tuple[object, frozenset[int]]]" = OrderedDict()
_HEAVY_STMT_MEMO_MAX = 128


def heavy_stmt_ids(stmts) -> frozenset:
    """`{id(stmt) for stmt in stmts if it calls a heavy builtin}` — the set a poll site
    checks its own upcoming (or just-finished) statement's `id()` against. `stmts` is a
    program's own top-level statement list (`Program.statements`). Memoized per list
    object; a cache miss pays one full-program walk, the same cost class
    `_collect_binding_reads` already pays on a cold program."""
    key = id(stmts)
    cached = _HEAVY_STMT_MEMO.get(key)
    if cached is not None and cached[0] is stmts:
        _HEAVY_STMT_MEMO.move_to_end(key)
        return cached[1]
    heavy_ids = frozenset(id(stmt) for stmt in stmts if _stmt_calls_heavy_builtin(stmt))
    _HEAVY_STMT_MEMO[key] = (stmts, heavy_ids)
    if len(_HEAVY_STMT_MEMO) > _HEAVY_STMT_MEMO_MAX:
        _HEAVY_STMT_MEMO.popitem(last=False)
    return heavy_ids


def program_has_any_heavy_stmt(program) -> bool:
    """For a single, PER-COOK (not per-statement) poll point — the codegen/stencil
    route's entry poll, reached once before any of the program's statements run — asking
    "is there any heavy statement in this program at all" is the coarse-but-safe
    equivalent of per-statement classification: it costs one extra `bool(...)` over an
    already-memoized set, and erring toward `True` here only ever costs a poll this route
    pays once per cook, never per statement."""
    return bool(heavy_stmt_ids(program.statements))
