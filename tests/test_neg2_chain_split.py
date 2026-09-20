"""
NEG-2 — the chain cook split out of `tex_engine.py`, and the properties that keep it split.

`tex_engine.py` came out of ENG-14 at 1628 lines under a `_HEADROOM_FLOOR` of 1700, and four
perf lanes then spent 45 of the 72 remaining lines, leaving 27 against a feature budgeted at
+38. NEG-2 is the split ENG-14's design note pre-specified for exactly this moment: the
CACHE-6 stage-list family and CACHE-1's lineage keys move into `tex_chain.py`, which
`tex_engine` imports at load and re-exports, so no caller anywhere changed.

**On red-first, honestly.** A pure move has a weak red-first story and inventing a failing
test for one would be dishonest. Exactly one assertion in this release is genuinely red on
the base sha for the right reason — `test_reg2_loc_budget`'s lowered `_HEADROOM_FLOOR`
(`test_v017_phase2.py`), which asserts the property the release exists to create. The tests
below are red on the base sha only because `tex_chain.py` does not exist there; their real
job is to red on a FUTURE change that undoes the split, swaps the re-export for a shim, or
lets the two modules grow into a cycle. That asymmetry is stated here rather than hidden.

The re-export surface itself is pinned elsewhere, deliberately: by
`test_v022_phase1.test_eng1_node_is_a_marshaller`, which is the single tuple naming
everything `tex_engine` still promises to expose.
"""
from helpers import *
import ast

_PKG = Path(__file__).resolve().parent.parent

# The move, name by name: every one of these must now be DEFINED in tex_chain and
# re-exported onto tex_engine as the same object.
_MOVED_FUNCS = (
    # CACHE-6, the chain cook itself
    "cook_stage_list", "boundary_lineage_key", "cook_fused_cached",
    "_is_tensor_binding", "_binding_shape",
    # CACHE-1, the per-output lineage keys
    "_compute_lineage",
    # The two engine primitives that travelled so tex_chain could stay a leaf
    "_compile_or_raise", "_get_interpreter", "_clear_all_interpreter_caches",
)
# Moved too, but an object rather than a function, so `__module__` cannot answer for it.
# It is the ENG-9 pool: there must be exactly ONE per process, so it travels with the
# accessor that creates it and the sweep that empties it.
_MOVED_OBJECTS = ("_interp_pool",)


def _loc(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def _runtime_module_level_imports(tree: ast.Module):
    """The dotted name of every import statement that RUNS at module level.

    Deliberately excludes two classes. Function-local imports are the lazy edges
    ARCHITECTURE.md refuses to let anyone hoist. And an `if TYPE_CHECKING:` block never
    executes, so an import inside one creates no edge — `tex_chain` annotates two
    parameters with `tex_engine`'s value bundles, which are strings under PEP 563 and are
    never evaluated.
    """
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


def test_neg2_tex_chain_exists_and_carries_the_move(r: SubTestResult):
    print("\n--- NEG-2: tex_chain.py exists and carries the chain cook ---")
    # Same shape as DOC-7b's STR-7 check and ENG-14's: a split that can be silently
    # reverted is not a split, and a file that exists but is a stub would pass exists().
    f = _PKG / "tex_chain.py"
    if not f.exists():
        r.fail("NEG-2 split module", "tex_chain.py missing — the split has been reverted")
        return
    n = _loc(f)
    if n <= 300:
        r.fail("NEG-2 split module", f"tex_chain.py: {n} lines (<= 300) — emptied out")
        return
    r.ok(f"tex_chain.py ({n} lines) carries the move; "
         f"tex_engine.py is {_loc(_PKG / 'tex_engine.py')} lines")


def test_neg2_the_moved_names_are_the_same_objects(r: SubTestResult):
    print("\n--- NEG-2: tex_chain defines them, tex_engine re-exports the SAME object ---")
    # IDENTITY, not equality: a re-export that rebuilt an equal-looking wrapper would
    # break every caller that compares or patches by `is`, and would quietly give the
    # process a SECOND ENG-9 interpreter pool, which is a correctness bug, not a style one.
    from TEX_Wrangle import tex_engine, tex_chain
    fails = []
    for name in _MOVED_FUNCS + _MOVED_OBJECTS:
        if not hasattr(tex_chain, name):
            fails.append(f"{name}: not defined in tex_chain")
            continue
        if not hasattr(tex_engine, name):
            fails.append(f"{name}: not re-exported onto tex_engine")
            continue
        if getattr(tex_engine, name) is not getattr(tex_chain, name):
            fails.append(f"{name}: tex_engine's object is NOT tex_chain's object")
        # A REAL global, not a PEP 562 `__getattr__` shim. The mechanism is load-bearing:
        # a module `__getattr__` never services a bare global-name lookup inside the
        # module, so a "tidy-up" to one would force a function-local import back into
        # every surviving caller — measured at 0.286 us per site per cook (ENG-14 F13).
        if name not in vars(tex_engine):
            fails.append(f"{name}: reachable but not in vars(tex_engine) — the re-export "
                         f"has been replaced by a module __getattr__ shim")
    for name in _MOVED_FUNCS:
        mod = getattr(getattr(tex_engine, name, None), "__module__", None)
        if mod != "TEX_Wrangle.tex_chain":
            fails.append(f"{name}: __module__ is {mod!r}, expected TEX_Wrangle.tex_chain")
    # The half-done move: a name left assigned in tex_engine.py would shadow the import.
    eng_src = (_PKG / "tex_engine.py").read_text(encoding="utf-8")
    for name in _MOVED_OBJECTS:
        if f"\n{name} = " in eng_src:
            fails.append(f"{name}: still assigned in tex_engine.py — the move is half-done")
    if fails:
        r.fail("NEG-2 homes", "; ".join(fails))
    else:
        r.ok(f"{len(_MOVED_FUNCS)} functions + {len(_MOVED_OBJECTS)} object define in "
             f"tex_chain and re-export as the SAME real globals on tex_engine")


def test_neg2_tex_chain_never_imports_tex_engine_at_runtime(r: SubTestResult):
    print("\n--- NEG-2: tex_chain -> tex_engine is not an edge (the direction of travel) ---")
    # This is the whole basis of the zero-cost claim, and it is what ENG-14's design note
    # meant by "the chain / lineage group becomes a leaf once run()'s call into
    # _compute_lineage is inverted". The moment tex_chain can reach tex_engine at import
    # time, tex_engine can no longer import it at load and every re-export has to become a
    # function-local import at each surviving call site.
    f = _PKG / "tex_chain.py"
    if not f.exists():
        r.fail("NEG-2 direction of travel", "tex_chain.py missing")
        return
    fails = []
    for name in _runtime_module_level_imports(ast.parse(f.read_text(encoding="utf-8"))):
        if "tex_engine" in name:
            fails.append(f"tex_chain.py imports {name!r} at module level — that is a cycle")
        # The same prohibition ENG-14 machine-checked for the planners: this edge is
        # function-local by standing order (ARCHITECTURE.md, AGENTS.md).
        if "tex_memory" in name:
            fails.append(f"tex_chain.py hoisted {name!r} to module level")
    # And the edge that MUST exist, in the other direction: tex_engine imports tex_chain at
    # load, which is what binds the moved names into the global slots its callers read.
    eng_imports = _runtime_module_level_imports(
        ast.parse((_PKG / "tex_engine.py").read_text(encoding="utf-8")))
    if not any(n.endswith("tex_chain") for n in eng_imports):
        fails.append("tex_engine.py does not import tex_chain at module level — the "
                     "re-export has gone function-local")
    if fails:
        r.fail("NEG-2 direction of travel", "; ".join(fails))
    else:
        r.ok("tex_chain imports nothing that reaches tex_engine or tex_memory at load, and "
             "tex_engine imports tex_chain top-level — every re-export costs 0 us/cook")
