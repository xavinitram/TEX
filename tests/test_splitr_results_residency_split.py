"""
SPLIT-R — the CACHE-8 residency ladder split out of `tex_results.py`, and the properties
that keep it split.

`tex_results.py` came out of NEG-6 at 1887 lines, sitting AT that floor, with the cut
pre-specified in the same comment: the CACHE-8 residency ladder (`set_vram_budget`,
`_enforce_residency`, `_queue_demotions`, `_drain_demotes`, `_promote`) moves into
`tex_results_residency.py` as a MIXIN (`_ResultCacheResidency`) that `ResultCache` inherits,
so every existing caller — which reads these as `ResultCache` methods, never as module-level
names — keeps working unchanged. `_dev_bucket`, the one plain helper the ladder shares with
the rest of `ResultCache` (`_admit`/`_remove`/`governed_bytes`/`evict_bytes`, all still in
tex_results.py), moves with it and is re-exported the same way NEG-6 re-exported
`env_epoch`/`lineage_key`.

**On red-first, honestly.** Same note NEG-6 made about NEG-2's own split: a pure move has a
weak red-first story. The one assertion genuinely red on the base sha for the right reason is
`test_reg2_loc_budget`'s lowered `_HEADROOM_FLOOR` (`test_v017_phase2.py`). The tests below are
red on the base sha only because `tex_results_residency.py` does not exist there; their real
job is to red on a FUTURE change that undoes the split, swaps the mixin for a copy, or lets the
two modules grow into a cycle.

Locking is checked ELSEWHERE, not here: RACE-43, TRK-178 and the `test_v033_cache8` threaded
races exercise the moved methods' actual lock discipline (this split changes no lock scope —
every `with self._lock:` in the moved bodies is byte-for-byte what it was before the move).
This file's job is narrower: prove the move itself is total and cycle-free.
"""
from helpers import *
import ast

_PKG = Path(__file__).resolve().parent.parent

# The move, name by name: every one of these must now be a METHOD OF `_ResultCacheResidency`
# in tex_results_residency.py, and `ResultCache` must resolve it to that exact function object
# (inherited, not copied) via the mixin.
_MOVED_METHODS = ("set_vram_budget", "_enforce_residency", "_queue_demotions",
                  "_drain_demotes", "_promote")
# Moved too, but a plain module-level function rather than a method — re-exported onto
# tex_results the same way NEG-6 re-exported env_epoch/lineage_key.
_MOVED_FUNCS = ("_dev_bucket",)


def _runtime_module_level_imports(tree: ast.Module):
    """The dotted name of every import statement that RUNS at module level. Deliberately
    excludes function-local imports (the lazy edges ARCHITECTURE.md refuses to let anyone
    hoist) and anything inside an `if TYPE_CHECKING:` block, which never executes."""
    out = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            if node.module:
                out.append(prefix + node.module)
            else:                       # `from . import x`
                out.extend(prefix + a.name for a in node.names)
    return out


def _loc(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def test_splitr_tex_results_residency_exists_and_carries_the_move(r: SubTestResult):
    print("\n--- SPLIT-R: tex_results_residency.py exists and carries the CACHE-8 ladder ---")
    f = _PKG / "tex_results_residency.py"
    if not f.exists():
        r.fail("SPLIT-R split module",
               "tex_results_residency.py missing — the split has been reverted")
        return
    n = _loc(f)
    if n <= 80:
        r.fail("SPLIT-R split module", f"tex_results_residency.py: {n} lines (<= 80) — emptied out")
        return
    r.ok(f"tex_results_residency.py ({n} lines) carries the move; "
         f"tex_results.py is {_loc(_PKG / 'tex_results.py')} lines")


def test_splitr_resultcache_inherits_the_mixin(r: SubTestResult):
    print("\n--- SPLIT-R: ResultCache inherits _ResultCacheResidency, methods resolve to it ---")
    from TEX_Wrangle import tex_results, tex_results_residency
    fails = []
    if not issubclass(tex_results.ResultCache, tex_results_residency._ResultCacheResidency):
        fails.append("ResultCache no longer inherits _ResultCacheResidency")
    for name in _MOVED_METHODS:
        mixin_fn = tex_results_residency._ResultCacheResidency.__dict__.get(name)
        if mixin_fn is None:
            fails.append(f"{name}: not defined in _ResultCacheResidency")
            continue
        resolved = getattr(tex_results.ResultCache, name, None)
        # unwrap: an instance method accessed off the CLASS in py3 is the plain function
        resolved_fn = getattr(resolved, "__func__", resolved)
        if resolved_fn is not mixin_fn:
            fails.append(f"{name}: ResultCache.{name} is not _ResultCacheResidency's function "
                         f"— it has been shadowed/copied rather than inherited")
        if name in tex_results.ResultCache.__dict__:
            fails.append(f"{name}: ALSO defined directly on ResultCache — the move is "
                         f"half-done (a copy now shadows the mixin)")
    if fails:
        r.fail("SPLIT-R mixin", "; ".join(fails))
    else:
        r.ok(f"{len(_MOVED_METHODS)} residency methods resolve on ResultCache to the SAME "
             f"functions _ResultCacheResidency defines — inherited, not copied")


def test_splitr_dev_bucket_is_the_same_object(r: SubTestResult):
    print("\n--- SPLIT-R: _dev_bucket defines in tex_results_residency, re-exports as the SAME object ---")
    from TEX_Wrangle import tex_results, tex_results_residency
    fails = []
    for name in _MOVED_FUNCS:
        if not hasattr(tex_results_residency, name):
            fails.append(f"{name}: not defined in tex_results_residency")
            continue
        if not hasattr(tex_results, name):
            fails.append(f"{name}: not re-exported onto tex_results")
            continue
        if getattr(tex_results, name) is not getattr(tex_results_residency, name):
            fails.append(f"{name}: tex_results's object is NOT tex_results_residency's object")
        if name not in vars(tex_results):
            fails.append(f"{name}: reachable but not in vars(tex_results) — the re-export "
                         f"has been replaced by a module __getattr__ shim")
    if fails:
        r.fail("SPLIT-R _dev_bucket", "; ".join(fails))
    else:
        r.ok("_dev_bucket defines in tex_results_residency and re-exports as the SAME real "
             "global on tex_results")


def test_splitr_host_seam_resolves(r: SubTestResult):
    print("\n--- SPLIT-R: the moved methods actually run through ResultCache, CPU-safe ---")
    # CPU-safe exercise of the real moved code: constructing a cache, arming and disarming the
    # residency tier (set_vram_budget -> _enforce_residency -> _drain_demotes, all reachable
    # with zero CUDA-resident entries) and confirming _dev_bucket classifies both device
    # spellings correctly — the same "actually call it, not just resolve it" bar NEG-6 held
    # lineage_key to.
    from TEX_Wrangle import tex_results
    import tempfile
    fails = []
    try:
        c = tex_results.ResultCache(cache_dir=tempfile.mkdtemp())
        c.set_vram_budget(64)           # arms it; nothing resident, so no demotion is queued
        c.set_vram_budget(None)         # disarms it again
    except Exception as e:
        fails.append(f"set_vram_budget(...) raised: {type(e).__name__}: {e}")
    if tex_results._dev_bucket("cuda:0") != "cuda":
        fails.append("_dev_bucket('cuda:0') != 'cuda'")
    if tex_results._dev_bucket("cpu") != "cpu":
        fails.append("_dev_bucket('cpu') != 'cpu'")
    if fails:
        r.fail("SPLIT-R host seam", "; ".join(fails))
    else:
        r.ok("ResultCache.set_vram_budget arms/disarms through the mixin without error, and "
             "_dev_bucket classifies both device spellings")


def test_splitr_tex_results_residency_never_imports_tex_results_at_runtime(r: SubTestResult):
    print("\n--- SPLIT-R: tex_results_residency -> tex_results is not an edge (direction of travel) ---")
    # The whole basis of the zero-cost claim: the moment tex_results_residency could reach
    # tex_results at import time, tex_results could no longer import it at load and every
    # re-export/mixin base would have to become a function-local import at each surviving
    # call site.
    f = _PKG / "tex_results_residency.py"
    if not f.exists():
        r.fail("SPLIT-R direction of travel", "tex_results_residency.py missing")
        return
    fails = []
    for name in _runtime_module_level_imports(ast.parse(f.read_text(encoding="utf-8"))):
        if "tex_results" in name or "tex_chain" in name or "tex_engine" in name:
            fails.append(f"tex_results_residency.py imports {name!r} at module level — a cycle")
    # And the edge that MUST exist, in the other direction: tex_results imports
    # tex_results_residency at load, which is what makes the mixin base available when the
    # class statement executes.
    res_imports = _runtime_module_level_imports(
        ast.parse((_PKG / "tex_results.py").read_text(encoding="utf-8")))
    if not any(n.endswith("tex_results_residency") for n in res_imports):
        fails.append("tex_results.py does not import tex_results_residency at module level — "
                     "the mixin base has gone function-local")
    if fails:
        r.fail("SPLIT-R direction of travel", "; ".join(fails))
    else:
        r.ok("tex_results_residency imports nothing that reaches tex_results, tex_chain or "
             "tex_engine at load, and tex_results imports tex_results_residency top-level — "
             "the mixin costs 0 us/cook")
