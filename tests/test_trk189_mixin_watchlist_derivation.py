"""TRK-189 — DERIVE, from the filesystem, that every mixin a split moved code INTO rides
every cache-invalidation watch-list its parent module is on.

THE DEFECT THIS GUARDS: SPLIT-I (v0.44 Phase A1) moved interpreter code out of
`tex_runtime/interpreter.py` into three sibling files — `interpreter_spatial.py`,
`interpreter_control_flow.py`, `interpreter_binding.py` — that `Interpreter` composes as
mixins. `interpreter.py` sits on `tex_cache._CODEGEN_FILES` (CACHE-4's watch-list for the
`.cg` codegen/interpreter tier), but the three new files were never added beside it. A future
edit to any of them would change interpreter semantics exactly as an `interpreter.py` edit
does, yet would bump no epoch and invalidate no `.cg` sidecar — a silently stale cache, the
same failure class CACHE-4's own docstring (`tex_cache.py`) exists to prevent for the AST/
codegen split itself.

THE RULE, kept general rather than interpreter-specific so it also covers SPLIT-R's
`tex_results_*.py` family against `tex_results.py` (and any future split of the same shape,
without a second hand-written test): for a parent module `X.py` and its mixin siblings
`X_*.py` living beside it — discovered by globbing the parent's own directory, never typed
out as a literal list, or the exact defect above recurs one level up — every watch-list that
carries the parent must also carry every sibling. `tex_cache.epoch_partitions()` is the one
function that exposes the real watch-lists (`ast`, `codegen`, `verdict`); this test reads
lists through it rather than re-importing the module's private `_AST_FILES` etc., so it is
pinned to the same public seam `test_v025_phase1.py`'s own CACHE-4 tripwire uses.

MUTATION (both directions, checked by inspection of the assertion below — no live git edit
needed to see it): dropping any one of the three `interpreter_*.py` paths from
`tex_cache._CODEGEN_FILES` reds `test_trk189_interpreter_mixins_...` (the sibling reappears
in `missing`); adding a fourth mixin file on disk with no source change reds it identically,
which is the point — the test derives its expectation from the tree, not from a count typed
here. `tex_results.py` is not on any watch-list today (confirmed by
`test_trk189_results_mixins_...` printing the vacuous-pass line), so that row is currently a
guard against a FUTURE regression: the day `tex_results.py` (or a file the SPLIT-R family
grows) joins a watch-list without its siblings, this test is what reds.

PORTABILITY: filesystem + `tex_cache.epoch_partitions()` only. No ComfyUI, no CUDA, no
numpy, no timing.
"""
import glob
import os

from helpers import *
from TEX_Wrangle import tex_cache as C


def _siblings(directory: str, parent_name: str) -> set:
    """Every `<stem>_*.py` file beside `parent_name` in `directory` — the mixin family,
    derived from disk. The glob's underscore requirement is what keeps `parent_name` itself
    out of its own sibling set (`interpreter.py` does not match `interpreter_*.py`)."""
    stem = parent_name[:-3]
    return {os.path.basename(p) for p in glob.glob(os.path.join(directory, f"{stem}_*.py"))}


def _check_family(r: SubTestResult, parent_name: str, directory: str, label: str):
    siblings = _siblings(directory, parent_name)
    if not siblings:
        r.ok(f"[{label}] no `{parent_name[:-3]}_*.py` siblings on disk — nothing to derive")
        return
    parts = C.epoch_partitions()
    carried_by = []
    for part_name, files in parts.items():
        names = {p.name for p in files}
        if parent_name not in names:
            continue
        carried_by.append(part_name)
        missing = siblings - names
        assert not missing, (
            f"[{label}] tex_cache's `{part_name}` watch-list carries `{parent_name}` but "
            f"not its sibling(s) {sorted(missing)} — an edit to them would not invalidate "
            f"the cache tier `{part_name}` gates")
    if carried_by:
        r.ok(f"[{label}] every watch-list carrying `{parent_name}` ({sorted(carried_by)}) "
             f"also carries its sibling(s) {sorted(siblings)}")
    else:
        r.ok(f"[{label}] `{parent_name}` is not on any watch-list today, so its "
             f"sibling(s) {sorted(siblings)} owe none either (regression guard for when "
             f"it joins one)")


def test_trk189_interpreter_mixins_ride_every_watchlist_interpreter_is_on(r: SubTestResult):
    print("\n--- TRK-189: interpreter_*.py mixins ride every watch-list interpreter.py "
          "is on ---")
    try:
        runtime_dir = os.path.join(os.path.dirname(os.path.abspath(C.__file__)), "tex_runtime")
        _check_family(r, "interpreter.py", runtime_dir, "SPLIT-I")
    except Exception as e:
        r.fail("TRK-189 interpreter mixins", f"{type(e).__name__}: {e}")


def test_trk189_results_mixins_ride_every_watchlist_results_is_on(r: SubTestResult):
    print("\n--- TRK-189: tex_results_*.py mixins ride every watch-list tex_results.py "
          "is on ---")
    try:
        pkg_dir = os.path.dirname(os.path.abspath(C.__file__))
        _check_family(r, "tex_results.py", pkg_dir, "SPLIT-R")
    except Exception as e:
        r.fail("TRK-189 results mixins", f"{type(e).__name__}: {e}")
