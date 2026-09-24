"""
NEG-6 — the CACHE-1 key-minting leaf split out of `tex_results.py`, and the properties
that keep it split.

`tex_results.py` came out of NEG-1 at 1903 lines under a `_HEADROOM_FLOOR` of 1958, with the
cut pre-specified: the CACHE-1 lineage-key family (`_code_epoch`, `env_epoch`,
`_ENV_EPOCH_CACHE`, `_canon_params`, `_canon_time`, `lineage_key`) moves into
`tex_results_keys.py`, which `tex_results` imports at load and re-exports, so no caller
anywhere changed — including the one caller an embedding host has stated as its sole seam
into this machinery, `tex_results.lineage_key`.

**On red-first, honestly.** Same shape as NEG-2's own note: a pure move has a weak red-first
story, and inventing a failing test for one would be dishonest. The one assertion genuinely
red on the base sha for the right reason is `test_reg2_loc_budget`'s lowered
`_HEADROOM_FLOOR` (`test_v017_phase2.py`), which asserts the property this release exists to
create. The tests below are red on the base sha only because `tex_results_keys.py` does not
exist there; their real job is to red on a FUTURE change that undoes the split, swaps the
re-export for a shim, or lets the two modules grow into a cycle.

The G1 bytecode-identity proof for the five moved functions (NEG-2's own technique, reused
verbatim) lives in the lane's own evidence, kept local and never shipped — this file checks
identity and import direction, not bytecode; the two are complementary evidence, not the same
claim.
"""
from helpers import *
import ast

_PKG = Path(__file__).resolve().parent.parent

# The move, name by name: every one of these must now be DEFINED in tex_results_keys and
# re-exported onto tex_results as the same object.
_MOVED_FUNCS = ("_code_epoch", "env_epoch", "_canon_params", "_canon_time", "lineage_key")
# Moved too, but data rather than a function, so `__module__` cannot answer for it — a plain
# module-level dict, checked by identity instead.
_MOVED_OBJECTS = ("_ENV_EPOCH_CACHE",)


def _runtime_module_level_imports(tree: ast.Module):
    """The dotted name of every import statement that RUNS at module level.

    Deliberately excludes function-local imports (the lazy edges ARCHITECTURE.md refuses to
    let anyone hoist) and anything inside an `if TYPE_CHECKING:` block, which never executes.
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


def _loc(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def test_neg6_tex_results_keys_exists_and_carries_the_move(r: SubTestResult):
    print("\n--- NEG-6: tex_results_keys.py exists and carries the CACHE-1 key leaf ---")
    f = _PKG / "tex_results_keys.py"
    if not f.exists():
        r.fail("NEG-6 split module", "tex_results_keys.py missing — the split has been reverted")
        return
    n = _loc(f)
    if n <= 80:
        r.fail("NEG-6 split module", f"tex_results_keys.py: {n} lines (<= 80) — emptied out")
        return
    r.ok(f"tex_results_keys.py ({n} lines) carries the move; "
         f"tex_results.py is {_loc(_PKG / 'tex_results.py')} lines")


def test_neg6_the_moved_names_are_the_same_objects(r: SubTestResult):
    print("\n--- NEG-6: tex_results_keys defines them, tex_results re-exports the SAME object ---")
    # IDENTITY, not equality: this is what makes `tex_results.lineage_key` — the one name an
    # embedding host has said it depends on — the exact function a host-style import reaches,
    # not a rebuilt wrapper around it, and what keeps _ENV_EPOCH_CACHE a SINGLE memo dict
    # rather than two caches quietly diverging.
    from TEX_Wrangle import tex_results, tex_results_keys
    fails = []
    for name in _MOVED_FUNCS + _MOVED_OBJECTS:
        if not hasattr(tex_results_keys, name):
            fails.append(f"{name}: not defined in tex_results_keys")
            continue
        if not hasattr(tex_results, name):
            fails.append(f"{name}: not re-exported onto tex_results")
            continue
        if getattr(tex_results, name) is not getattr(tex_results_keys, name):
            fails.append(f"{name}: tex_results's object is NOT tex_results_keys's object")
        # A REAL global, not a PEP 562 `__getattr__` shim — a module `__getattr__` never
        # services a bare global-name lookup inside the module itself, so a "tidy-up" to one
        # would force a function-local import back into every surviving caller.
        if name not in vars(tex_results):
            fails.append(f"{name}: reachable but not in vars(tex_results) — the re-export "
                         f"has been replaced by a module __getattr__ shim")
    for name in _MOVED_FUNCS:
        mod = getattr(getattr(tex_results, name, None), "__module__", None)
        if mod != "TEX_Wrangle.tex_results_keys":
            fails.append(f"{name}: __module__ is {mod!r}, expected TEX_Wrangle.tex_results_keys")
    # The half-done move: a name left assigned in tex_results.py would shadow the import.
    res_src = (_PKG / "tex_results.py").read_text(encoding="utf-8")
    for name in _MOVED_OBJECTS:
        if f"\n{name} = " in res_src or f"\n{name}:" in res_src:
            fails.append(f"{name}: still assigned in tex_results.py — the move is half-done")
    if fails:
        r.fail("NEG-6 homes", "; ".join(fails))
    else:
        r.ok(f"{len(_MOVED_FUNCS)} functions + {len(_MOVED_OBJECTS)} object define in "
             f"tex_results_keys and re-export as the SAME real globals on tex_results")


def test_neg6_host_seam_resolves(r: SubTestResult):
    print("\n--- NEG-6: the one name a host has said it depends on still resolves ---")
    # Two import shapes, both host-realistic: the module-attribute read every real caller in
    # this tree uses (tex_chain, examples/host_demo.py), and the direct `from ... import` a
    # handful of tests use. Both must reach the SAME callable and it must actually mint a key.
    from TEX_Wrangle import tex_results
    from TEX_Wrangle.tex_results import lineage_key as direct_lineage_key
    fails = []
    if tex_results.lineage_key is not direct_lineage_key:
        fails.append("tex_results.lineage_key is not the same object `from ... import "
                     "lineage_key` resolves")
    try:
        k = tex_results.lineage_key(program_fp="p", device="cpu", precision="fp32")
        if not (isinstance(k, str) and len(k) == 64):
            fails.append(f"lineage_key did not return a hex-64 key: {k!r}")
    except Exception as e:
        fails.append(f"tex_results.lineage_key(...) raised: {type(e).__name__}: {e}")
    if fails:
        r.fail("NEG-6 host seam", "; ".join(fails))
    else:
        r.ok("tex_results.lineage_key resolves both by module attribute and by direct "
             "import, and mints a real key")


def test_neg6_tex_results_keys_never_imports_tex_results_at_runtime(r: SubTestResult):
    print("\n--- NEG-6: tex_results_keys -> tex_results is not an edge (direction of travel) ---")
    # The whole basis of the zero-cost claim: the moment tex_results_keys could reach
    # tex_results at import time, tex_results could no longer import it at load and every
    # re-export would have to become a function-local import at each surviving call site.
    f = _PKG / "tex_results_keys.py"
    if not f.exists():
        r.fail("NEG-6 direction of travel", "tex_results_keys.py missing")
        return
    fails = []
    for name in _runtime_module_level_imports(ast.parse(f.read_text(encoding="utf-8"))):
        if "tex_results" in name or "tex_chain" in name or "tex_engine" in name:
            fails.append(f"tex_results_keys.py imports {name!r} at module level — a cycle")
    # And the edge that MUST exist, in the other direction: tex_results imports
    # tex_results_keys at load, which is what binds the moved names into the global slots
    # its callers already read.
    res_imports = _runtime_module_level_imports(
        ast.parse((_PKG / "tex_results.py").read_text(encoding="utf-8")))
    if not any(n.endswith("tex_results_keys") for n in res_imports):
        fails.append("tex_results.py does not import tex_results_keys at module level — "
                     "the re-export has gone function-local")
    if fails:
        r.fail("NEG-6 direction of travel", "; ".join(fails))
    else:
        r.ok("tex_results_keys imports nothing that reaches tex_results, tex_chain or "
             "tex_engine at load, and tex_results imports tex_results_keys top-level — "
             "every re-export costs 0 us/cook")
