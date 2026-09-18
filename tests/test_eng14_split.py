"""
ENG-14 — the two leaves split out of `tex_engine.py`, and the properties that keep
them split.

`tex_engine.py` sat at exactly 2000/2000 against REG-2's hard budget, so the next line
added to it was a red test. ENG-14 bought the headroom by MOVING code to a better home:
`tex_buffers.py` (the ENG-6/ENG-12 frame-handoff and buffer-ownership contract) and
`tex_tiling.py` (the cook-fit planners), both imported by `tex_engine` at load and
re-exported, so no caller anywhere changed.

**On red-first, honestly.** A pure move has a weak red-first story and inventing a
failing test for one would be dishonest. Exactly one assertion in this release is
genuinely red on the base sha for the right reason — `test_reg2_loc_budget`'s new
`_HEADROOM_FLOOR` (`test_v017_phase2.py`), which asserts the property the release exists
to create. The tests below are RED on the base sha only because the two files do not
exist there; their real job is to red on a FUTURE change that undoes the split, hoists
an import that must stay lazy, or quietly swaps the re-export for a `__getattr__` shim.
That asymmetry is stated here rather than hidden.

The re-export surface itself is pinned elsewhere, deliberately: by
`test_v022_phase1.test_eng1_node_is_a_marshaller`, which already owned that tuple
before this release and now names all 28 promised attributes.
"""
from helpers import *
import ast

_PKG = Path(__file__).resolve().parent.parent

# The move, name by name: symbol -> the module that must now define it.
_HOMES = {
    "_owned_copy": "tex_buffers", "to_dlpack": "tex_buffers",
    "from_dlpack": "tex_buffers", "is_frozen": "tex_buffers",
    "frame_version": "tex_buffers", "verify_unmutated": "tex_buffers",
    "frozen_copy": "tex_buffers", "freeze": "tex_buffers",
    "_disown_inputs": "tex_buffers",
    "_tile_plan": "tex_tiling", "_halo_tile_plan": "tex_tiling",
    "_preflight_memory": "tex_tiling", "_tdr_strip_floor": "tex_tiling",
    "_scalar_params": "tex_tiling",
}
# Moved too, but a float rather than a function, so `__module__` cannot answer for it.
_MOVED_CONSTANTS = {"_TDR_BUDGET_MS": "tex_tiling"}

# What a LEAF is allowed to import at module level. `tex_runtime.host` is the PORT-1
# host seam, which sits below both modules in the layer table.
_LEAF_IMPORTS_ALLOWED = {"__future__", "math", "typing", "torch", ".tex_runtime.host"}


def _loc(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def _module_level_imports(tree: ast.Module):
    """(module_name, is_module_level) for every import STATEMENT at module level only.
    Function-local imports are deliberately not reported: the `tex_memory` / `tex_roi` /
    `autotier` imports inside the moved planner bodies are exactly the deliberate lazy
    edges ARCHITECTURE.md refuses to let anyone hoist."""
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


def test_eng14_the_two_leaves_exist(r: SubTestResult):
    print("\n--- ENG-14: tex_buffers.py and tex_tiling.py exist and carry the move ---")
    # Same shape as DOC-7b's STR-7 check: a split that can be silently reverted is not
    # a split. A file that exists but is a stub would pass `exists()` alone.
    fails = []
    for mod, floor in (("tex_buffers.py", 100), ("tex_tiling.py", 100)):
        f = _PKG / mod
        if not f.exists():
            fails.append(f"{mod}: missing — the ENG-14 split has been reverted")
            continue
        n = _loc(f)
        if n <= floor:
            fails.append(f"{mod}: {n} lines (<= {floor}) — the move has been emptied out")
    if fails:
        r.fail("ENG-14 split modules", "; ".join(fails))
    else:
        r.ok(f"tex_buffers.py ({_loc(_PKG / 'tex_buffers.py')} lines) and "
             f"tex_tiling.py ({_loc(_PKG / 'tex_tiling.py')} lines) carry the move")


def test_eng14_the_moved_names_live_in_their_new_homes(r: SubTestResult):
    print("\n--- ENG-14: every moved name is defined in its new module ---")
    from TEX_Wrangle import tex_engine
    fails = []
    for name, home in sorted(_HOMES.items()):
        obj = getattr(tex_engine, name, None)
        if obj is None:
            fails.append(f"{name}: not reachable on tex_engine at all")
            continue
        mod = getattr(obj, "__module__", None)
        if mod != f"TEX_Wrangle.{home}":
            fails.append(f"{name}: __module__ is {mod!r}, expected TEX_Wrangle.{home}")
        # A REAL global, not a PEP 562 `__getattr__` shim. The mechanism is load-bearing:
        # a module `__getattr__` never services a bare global-name lookup inside the
        # module, so a "tidy-up" to one would force a function-local import back into
        # every surviving caller — measured at 0.286 us per site per cook.
        if name not in vars(tex_engine):
            fails.append(f"{name}: reachable but not in vars(tex_engine) — the re-export "
                         f"has been replaced by a module __getattr__ shim")
    for name, home in sorted(_MOVED_CONSTANTS.items()):
        if name not in vars(tex_engine):
            fails.append(f"{name}: not a real global on tex_engine")
        src = (_PKG / f"{home}.py").read_text(encoding="utf-8")
        if f"\n{name} = " not in src:
            fails.append(f"{name}: not defined at module level in {home}.py")
        eng = (_PKG / "tex_engine.py").read_text(encoding="utf-8")
        if f"\n{name} = " in eng:
            fails.append(f"{name}: still assigned in tex_engine.py — the move is half-done")
    if fails:
        r.fail("ENG-14 homes", "; ".join(fails))
    else:
        r.ok(f"{len(_HOMES)} functions + {len(_MOVED_CONSTANTS)} constant define in their "
             f"new module and re-export as real tex_engine globals")


def test_eng14_the_new_modules_stay_leaves(r: SubTestResult):
    print("\n--- ENG-14: tex_buffers and tex_tiling import nothing that reaches back ---")
    # This is the whole basis of the zero-cost claim. The moment either module can reach
    # `tex_engine`, `tex_engine` can no longer import it at load, and the re-export has
    # to become a function-local import in every surviving caller.
    fails = []
    for mod in ("tex_buffers.py", "tex_tiling.py"):
        f = _PKG / mod
        if not f.exists():
            fails.append(f"{mod}: missing")
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for name in _module_level_imports(tree):
            if name not in _LEAF_IMPORTS_ALLOWED:
                fails.append(f"{mod}: module-level import {name!r} is not a leaf import "
                             f"(allowed: {sorted(_LEAF_IMPORTS_ALLOWED)})")
            if "tex_engine" in name:
                fails.append(f"{mod}: imports tex_engine at module level — this is a cycle")
    if fails:
        r.fail("ENG-14 leafness", "; ".join(fails))
    else:
        r.ok("both new modules import only __future__/math/typing/torch/the host seam "
             "at load — tex_engine can keep importing them top-level")


def test_eng14_tex_engine_still_imports_tex_memory_lazily(r: SubTestResult):
    print("\n--- ENG-14: the tex_engine <-> tex_memory edge is still function-local ---")
    # ARCHITECTURE.md and AGENTS.md both forbid hoisting this exact edge, in prose. It is
    # also the reason the planners went to a NEW leaf instead of into tex_memory.py, so
    # this release is the right moment to make the prohibition machine-checked.
    tree = ast.parse((_PKG / "tex_engine.py").read_text(encoding="utf-8"))
    hoisted = [n for n in _module_level_imports(tree) if "tex_memory" in n]
    if hoisted:
        r.fail("ENG-14 lazy memory edge",
               f"tex_engine.py imports tex_memory at module level ({hoisted}) — "
               f"forcing load-time reintroduces an ordering crash")
    else:
        r.ok("tex_engine.py has zero top-level tex_memory imports; the cycle stays broken "
             "by hand, exactly as it was")
